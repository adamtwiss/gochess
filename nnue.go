package chess

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// NNUE architecture: HalfKP (40960 inputs) -> 2x256 -> 32 -> 1
// Input features: king_square (64) * piece_type (10: 5 piece types x 2 colors) * piece_square (64) = 40960
// Two accumulators: White perspective, Black perspective
// Concatenated (512) -> 32 hidden neurons -> 1 output
// ClippedReLU activations, int16 weights for inference

const (
	// Network dimensions
	NNUEInputSize   = 40960 // 64 * 10 * 64
	NNUEHiddenSize  = 256   // per perspective
	NNUEHidden2Size = 32    // second hidden layer
	NNUEOutputSize  = 1

	// HalfKP piece indexing: 10 piece types (excluding kings)
	// 0=WhitePawn, 1=WhiteKnight, 2=WhiteBishop, 3=WhiteRook, 4=WhiteQueen
	// 5=BlackPawn, 6=BlackKnight, 7=BlackBishop, 8=BlackRook, 9=BlackQueen
	nnueNumPieceTypes = 10

	// Quantization scale factors
	nnueInputScale  = 127 // input layer weight scale
	nnueHiddenScale = 64  // hidden layer weight scale
	nnueOutputScale = nnueInputScale * nnueHiddenScale

	// ClippedReLU bounds (after quantization)
	nnueClipMin = 0
	nnueClipMax = nnueInputScale // 127

	// File magic and version
	nnueMagic   = uint32(0x4E4E5545) // "NNUE"
	nnueVersion = uint32(1)
)

// NNUENet holds all network weights (shared read-only across threads).
type NNUENet struct {
	// Input layer: [feature_index][hidden_neuron]
	InputWeights [NNUEInputSize][NNUEHiddenSize]int16
	InputBiases  [NNUEHiddenSize]int16

	// Hidden layer: [concat_input][hidden2_neuron]
	HiddenWeights [NNUEHiddenSize * 2][NNUEHidden2Size]int16
	HiddenBiases  [NNUEHidden2Size]int32

	// Output layer: [hidden2_neuron]
	OutputWeights [NNUEHidden2Size]int16
	OutputBias    int32

	// Transposed hidden weights for SIMD forward pass: [output][input]
	// Layout: HWT[j*(HiddenSize*2)+i] = HiddenWeights[i][j]
	// Populated by PrepareWeights(), nil when SIMD not available.
	HWT []int16
}

// NNUEAccumulator holds per-position accumulator state.
type NNUEAccumulator struct {
	White    [NNUEHiddenSize]int16 // accumulator from White perspective
	Black    [NNUEHiddenSize]int16 // accumulator from Black perspective
	Computed bool
}

// NNUEAccumulatorStack provides push/pop for MakeMove/UnmakeMove.
type NNUEAccumulatorStack struct {
	stack []NNUEAccumulator
	top   int
}

// NewNNUEAccumulatorStack creates a new accumulator stack with the given capacity.
func NewNNUEAccumulatorStack(capacity int) *NNUEAccumulatorStack {
	if capacity < 1 {
		capacity = 256
	}
	return &NNUEAccumulatorStack{
		stack: make([]NNUEAccumulator, capacity),
		top:   0,
	}
}

// Current returns a pointer to the current accumulator.
func (s *NNUEAccumulatorStack) Current() *NNUEAccumulator {
	return &s.stack[s.top]
}

// Push copies the current accumulator and advances the stack pointer.
func (s *NNUEAccumulatorStack) Push() {
	if s.top+1 >= len(s.stack) {
		// Grow the stack
		s.stack = append(s.stack, NNUEAccumulator{})
	}
	s.stack[s.top+1] = s.stack[s.top]
	s.top++
}

// PushEmpty advances the stack pointer without copying (for king moves
// where RecomputeAccumulator will overwrite everything).
func (s *NNUEAccumulatorStack) PushEmpty() {
	if s.top+1 >= len(s.stack) {
		s.stack = append(s.stack, NNUEAccumulator{})
	}
	s.top++
	s.stack[s.top].Computed = false
}

// Pop restores the previous accumulator state.
func (s *NNUEAccumulatorStack) Pop() {
	if s.top > 0 {
		s.top--
	}
}

// DeepCopy creates a deep copy of the accumulator stack (for Lazy SMP thread copies).
func (s *NNUEAccumulatorStack) DeepCopy() *NNUEAccumulatorStack {
	newStack := &NNUEAccumulatorStack{
		stack: make([]NNUEAccumulator, len(s.stack)),
		top:   s.top,
	}
	copy(newStack.stack, s.stack)
	return newStack
}

// Reset resets the stack to position 0 with Computed=false.
func (s *NNUEAccumulatorStack) Reset() {
	s.top = 0
	s.stack[0].Computed = false
}

// nnuePieceIndex maps a Piece (1-12, excluding kings) to a HalfKP piece type index (0-9).
// Returns -1 for Empty or kings.
func nnuePieceIndex(p Piece) int {
	switch p {
	case WhitePawn:
		return 0
	case WhiteKnight:
		return 1
	case WhiteBishop:
		return 2
	case WhiteRook:
		return 3
	case WhiteQueen:
		return 4
	case BlackPawn:
		return 5
	case BlackKnight:
		return 6
	case BlackBishop:
		return 7
	case BlackRook:
		return 8
	case BlackQueen:
		return 9
	default:
		return -1 // Empty or King
	}
}

// HalfKPIndex computes the feature index for a given perspective.
// perspective: White (0) or Black (1)
// kingSq: king square from that perspective
// piece: the piece (1-12, not king)
// pieceSq: the square the piece is on
//
// For Black perspective, squares are mirrored (^56) and piece colors are swapped.
// Feature = kingSq * 640 + pieceType * 64 + pieceSq
func HalfKPIndex(perspective Color, kingSq Square, piece Piece, pieceSq Square) int {
	pi := nnuePieceIndex(piece)
	if pi < 0 {
		return -1 // king or empty — not a feature
	}

	ks := int(kingSq)
	ps := int(pieceSq)

	if perspective == Black {
		// Mirror squares vertically
		ks ^= 56
		ps ^= 56
		// Swap piece colors: WhitePawn(0) <-> BlackPawn(5), etc.
		if pi < 5 {
			pi += 5
		} else {
			pi -= 5
		}
	}

	return ks*nnueNumPieceTypes*64 + pi*64 + ps
}

// AddFeature adds a feature (piece at square) to the accumulator for both perspectives.
func (net *NNUENet) AddFeature(acc *NNUEAccumulator, piece Piece, sq Square, wKingSq, bKingSq Square) {
	wIdx := HalfKPIndex(White, wKingSq, piece, sq)
	bIdx := HalfKPIndex(Black, bKingSq, piece, sq)

	if wIdx >= 0 {
		if nnueUseAVX2 {
			nnueAccAdd256(&acc.White[0], &net.InputWeights[wIdx][0])
		} else {
			for i := 0; i < NNUEHiddenSize; i++ {
				acc.White[i] += net.InputWeights[wIdx][i]
			}
		}
	}
	if bIdx >= 0 {
		if nnueUseAVX2 {
			nnueAccAdd256(&acc.Black[0], &net.InputWeights[bIdx][0])
		} else {
			for i := 0; i < NNUEHiddenSize; i++ {
				acc.Black[i] += net.InputWeights[bIdx][i]
			}
		}
	}
}

// RemoveFeature removes a feature (piece at square) from the accumulator for both perspectives.
func (net *NNUENet) RemoveFeature(acc *NNUEAccumulator, piece Piece, sq Square, wKingSq, bKingSq Square) {
	wIdx := HalfKPIndex(White, wKingSq, piece, sq)
	bIdx := HalfKPIndex(Black, bKingSq, piece, sq)

	if wIdx >= 0 {
		for i := 0; i < NNUEHiddenSize; i++ {
			acc.White[i] -= net.InputWeights[wIdx][i]
		}
	}
	if bIdx >= 0 {
		for i := 0; i < NNUEHiddenSize; i++ {
			acc.Black[i] -= net.InputWeights[bIdx][i]
		}
	}
}

// SubAddFeature combines RemoveFeature(oldSq) + AddFeature(newSq) for the same piece
// in a single pass over the accumulator, halving the number of accumulator writes.
func (net *NNUENet) SubAddFeature(acc *NNUEAccumulator, piece Piece, fromSq, toSq Square, wKingSq, bKingSq Square) {
	wIdxOld := HalfKPIndex(White, wKingSq, piece, fromSq)
	wIdxNew := HalfKPIndex(White, wKingSq, piece, toSq)
	if wIdxOld >= 0 && wIdxNew >= 0 {
		if nnueUseAVX2 {
			nnueAccSubAdd256(&acc.White[0], &net.InputWeights[wIdxOld][0], &net.InputWeights[wIdxNew][0])
		} else {
			for i := 0; i < NNUEHiddenSize; i++ {
				acc.White[i] += net.InputWeights[wIdxNew][i] - net.InputWeights[wIdxOld][i]
			}
		}
	}
	bIdxOld := HalfKPIndex(Black, bKingSq, piece, fromSq)
	bIdxNew := HalfKPIndex(Black, bKingSq, piece, toSq)
	if bIdxOld >= 0 && bIdxNew >= 0 {
		if nnueUseAVX2 {
			nnueAccSubAdd256(&acc.Black[0], &net.InputWeights[bIdxOld][0], &net.InputWeights[bIdxNew][0])
		} else {
			for i := 0; i < NNUEHiddenSize; i++ {
				acc.Black[i] += net.InputWeights[bIdxNew][i] - net.InputWeights[bIdxOld][i]
			}
		}
	}
}

// SubSubAddFeature combines two removes and one add in a single pass per perspective.
// Used for captures: remove captured piece at capSq, remove moving piece at fromSq, add moving piece at toSq.
func (net *NNUENet) SubSubAddFeature(acc *NNUEAccumulator, movePiece Piece, fromSq, toSq Square, capPiece Piece, capSq Square, wKingSq, bKingSq Square) {
	// White perspective
	wIdxMoveOld := HalfKPIndex(White, wKingSq, movePiece, fromSq)
	wIdxMoveNew := HalfKPIndex(White, wKingSq, movePiece, toSq)
	wIdxCap := HalfKPIndex(White, wKingSq, capPiece, capSq)
	if wIdxMoveOld >= 0 && wIdxMoveNew >= 0 && wIdxCap >= 0 {
		if nnueUseAVX2 {
			nnueAccSubSubAdd256(&acc.White[0], &net.InputWeights[wIdxMoveOld][0], &net.InputWeights[wIdxMoveNew][0], &net.InputWeights[wIdxCap][0])
		} else {
			for i := 0; i < NNUEHiddenSize; i++ {
				acc.White[i] += net.InputWeights[wIdxMoveNew][i] - net.InputWeights[wIdxMoveOld][i] - net.InputWeights[wIdxCap][i]
			}
		}
	}
	// Black perspective
	bIdxMoveOld := HalfKPIndex(Black, bKingSq, movePiece, fromSq)
	bIdxMoveNew := HalfKPIndex(Black, bKingSq, movePiece, toSq)
	bIdxCap := HalfKPIndex(Black, bKingSq, capPiece, capSq)
	if bIdxMoveOld >= 0 && bIdxMoveNew >= 0 && bIdxCap >= 0 {
		if nnueUseAVX2 {
			nnueAccSubSubAdd256(&acc.Black[0], &net.InputWeights[bIdxMoveOld][0], &net.InputWeights[bIdxMoveNew][0], &net.InputWeights[bIdxCap][0])
		} else {
			for i := 0; i < NNUEHiddenSize; i++ {
				acc.Black[i] += net.InputWeights[bIdxMoveNew][i] - net.InputWeights[bIdxMoveOld][i] - net.InputWeights[bIdxCap][i]
			}
		}
	}
}

// RecomputeAccumulator performs a full recompute of the accumulator from the board state.
// Used after king moves, SetFEN, and Reset.
func (net *NNUENet) RecomputeAccumulator(acc *NNUEAccumulator, b *Board) {
	// Start with biases
	acc.White = net.InputBiases
	acc.Black = net.InputBiases

	wKingSq := b.Pieces[WhiteKing].LSB()
	bKingSq := b.Pieces[BlackKing].LSB()

	// Add all non-king pieces
	for piece := WhitePawn; piece <= BlackQueen; piece++ {
		if piece == WhiteKing || piece == BlackKing {
			continue
		}
		bb := b.Pieces[piece]
		for bb != 0 {
			sq := bb.PopLSB()
			wIdx := HalfKPIndex(White, wKingSq, piece, sq)
			bIdx := HalfKPIndex(Black, bKingSq, piece, sq)
			if wIdx >= 0 {
				for i := 0; i < NNUEHiddenSize; i++ {
					acc.White[i] += net.InputWeights[wIdx][i]
				}
			}
			if bIdx >= 0 {
				for i := 0; i < NNUEHiddenSize; i++ {
					acc.Black[i] += net.InputWeights[bIdx][i]
				}
			}
		}
	}

	acc.Computed = true
}

// clippedReLU applies ClippedReLU: clamp to [0, nnueClipMax].
func clippedReLU(x int16) int16 {
	if x < nnueClipMin {
		return nnueClipMin
	}
	if x > nnueClipMax {
		return nnueClipMax
	}
	return x
}

// Evaluate runs the NNUE forward pass and returns a score in centipawns
// relative to the side to move.
func (net *NNUENet) Evaluate(acc *NNUEAccumulator, sideToMove Color) int {
	// SIMD fast path
	if nnueUseAVX2 && net.HWT != nil {
		return net.evaluateSIMD(acc, sideToMove)
	}

	// Concatenate accumulators: [stm_perspective | opponent_perspective]
	// This way the network always sees "my pieces" first
	var stm, opp *[NNUEHiddenSize]int16
	if sideToMove == White {
		stm = &acc.White
		opp = &acc.Black
	} else {
		stm = &acc.Black
		opp = &acc.White
	}

	// Hidden layer: ClippedReLU(accumulator) -> hidden2
	// hidden2[j] = bias[j] + sum_i(weight[i][j] * clipped_relu(acc[i]))
	var hidden2 [NNUEHidden2Size]int32
	hidden2 = net.HiddenBiases

	// Process both perspectives together to improve cache locality on hidden2.
	// For each i, we process stm[i] with weights[i] and opp[i] with weights[256+i].
	for i := 0; i < NNUEHiddenSize; i++ {
		// Inline ClippedReLU: clamp to [0, 127]
		sv := stm[i]
		ov := opp[i]

		// Fast path: both zero or negative
		if sv <= 0 && ov <= 0 {
			continue
		}

		cs := int32(sv)
		if cs < 0 {
			cs = 0
		} else if cs > 127 {
			cs = 127
		}

		co := int32(ov)
		if co < 0 {
			co = 0
		} else if co > 127 {
			co = 127
		}

		ws := &net.HiddenWeights[i]
		wo := &net.HiddenWeights[NNUEHiddenSize+i]

		if cs != 0 && co != 0 {
			// Both non-zero: combine to halve loop iterations
			for j := 0; j < NNUEHidden2Size; j++ {
				hidden2[j] += cs*int32(ws[j]) + co*int32(wo[j])
			}
		} else if cs != 0 {
			for j := 0; j < NNUEHidden2Size; j++ {
				hidden2[j] += cs * int32(ws[j])
			}
		} else {
			for j := 0; j < NNUEHidden2Size; j++ {
				hidden2[j] += co * int32(wo[j])
			}
		}
	}

	// Output layer: ClippedReLU(hidden2) -> output
	// output = bias + sum_j(weight[j] * clipped_relu(hidden2[j] / hiddenScale))
	output := int32(net.OutputBias)
	for j := 0; j < NNUEHidden2Size; j++ {
		scaled := hidden2[j] >> 6 // / 64 (nnueHiddenScale)
		if scaled <= 0 {
			continue
		}
		if scaled > 127 {
			scaled = 127
		}
		output += scaled * int32(net.OutputWeights[j])
	}

	// Scale output to centipawns
	return int(output) / nnueOutputScale
}

// PrepareWeights transposes hidden weights for SIMD forward pass.
// Must be called after loading or creating a network.
func (net *NNUENet) PrepareWeights() {
	if !nnueUseAVX2 {
		return
	}
	inputDim := NNUEHiddenSize * 2 // 512
	outputDim := NNUEHidden2Size   // 32
	net.HWT = make([]int16, outputDim*inputDim)
	for i := 0; i < inputDim; i++ {
		for j := 0; j < outputDim; j++ {
			net.HWT[j*inputDim+i] = net.HiddenWeights[i][j]
		}
	}
}

// evaluateSIMD runs the NNUE forward pass using AVX2 SIMD instructions.
func (net *NNUENet) evaluateSIMD(acc *NNUEAccumulator, sideToMove Color) int {
	var stm, opp *[NNUEHiddenSize]int16
	if sideToMove == White {
		stm = &acc.White
		opp = &acc.Black
	} else {
		stm = &acc.Black
		opp = &acc.White
	}

	// Apply ClippedReLU and concatenate into input buffer
	var input [NNUEHiddenSize * 2]int16
	nnueCReLU256(&stm[0], &input[0])
	nnueCReLU256(&opp[0], &input[NNUEHiddenSize])

	// Hidden layer matrix multiply (the bottleneck)
	var hidden2 [NNUEHidden2Size]int32
	nnueMatMul32x512(&input[0], &net.HWT[0], &net.HiddenBiases[0], &hidden2[0])

	// Output layer (scalar — only 32 elements, not worth SIMD)
	output := int32(net.OutputBias)
	for j := 0; j < NNUEHidden2Size; j++ {
		scaled := hidden2[j] >> 6
		if scaled <= 0 {
			continue
		}
		if scaled > 127 {
			scaled = 127
		}
		output += scaled * int32(net.OutputWeights[j])
	}

	return int(output) / nnueOutputScale
}

// SaveNNUE writes the network weights to a binary file.
func SaveNNUE(path string, net *NNUENet) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return writeNNUE(f, net)
}

func writeNNUE(w io.Writer, net *NNUENet) error {
	// Header
	if err := binary.Write(w, binary.LittleEndian, nnueMagic); err != nil {
		return fmt.Errorf("writing magic: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, nnueVersion); err != nil {
		return fmt.Errorf("writing version: %w", err)
	}

	// Input weights and biases
	if err := binary.Write(w, binary.LittleEndian, &net.InputWeights); err != nil {
		return fmt.Errorf("writing input weights: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, &net.InputBiases); err != nil {
		return fmt.Errorf("writing input biases: %w", err)
	}

	// Hidden weights and biases
	if err := binary.Write(w, binary.LittleEndian, &net.HiddenWeights); err != nil {
		return fmt.Errorf("writing hidden weights: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, &net.HiddenBiases); err != nil {
		return fmt.Errorf("writing hidden biases: %w", err)
	}

	// Output weights and bias
	if err := binary.Write(w, binary.LittleEndian, &net.OutputWeights); err != nil {
		return fmt.Errorf("writing output weights: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, &net.OutputBias); err != nil {
		return fmt.Errorf("writing output bias: %w", err)
	}

	return nil
}

// LoadNNUE reads network weights from a binary file.
func LoadNNUE(path string) (*NNUENet, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	net, err := readNNUE(f)
	if err != nil {
		return nil, err
	}
	net.PrepareWeights()
	return net, nil
}

func readNNUE(r io.Reader) (*NNUENet, error) {
	// Header
	var magic, version uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("reading magic: %w", err)
	}
	if magic != nnueMagic {
		return nil, fmt.Errorf("invalid NNUE file magic: 0x%X (expected 0x%X)", magic, nnueMagic)
	}
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("reading version: %w", err)
	}
	if version != nnueVersion {
		return nil, fmt.Errorf("unsupported NNUE version: %d (expected %d)", version, nnueVersion)
	}

	net := &NNUENet{}

	// Input weights and biases
	if err := binary.Read(r, binary.LittleEndian, &net.InputWeights); err != nil {
		return nil, fmt.Errorf("reading input weights: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &net.InputBiases); err != nil {
		return nil, fmt.Errorf("reading input biases: %w", err)
	}

	// Hidden weights and biases
	if err := binary.Read(r, binary.LittleEndian, &net.HiddenWeights); err != nil {
		return nil, fmt.Errorf("reading hidden weights: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &net.HiddenBiases); err != nil {
		return nil, fmt.Errorf("reading hidden biases: %w", err)
	}

	// Output weights and bias
	if err := binary.Read(r, binary.LittleEndian, &net.OutputWeights); err != nil {
		return nil, fmt.Errorf("reading output weights: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &net.OutputBias); err != nil {
		return nil, fmt.Errorf("reading output bias: %w", err)
	}

	return net, nil
}

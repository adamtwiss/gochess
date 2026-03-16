package chess

// NNUE v5 architecture: shallow wide net for Bullet GPU training.
// (768×16 → 1024)×2 → 1×8
//
// Single hidden layer with CReLU activation, 8 output buckets by material count.
// Designed for Bullet's sigmoid-based training which requires shallow architectures
// (gradient vanishing prevents training deeper nets effectively).
//
// Quantization: QA=255 (input/accumulator scale), QB=64 (output weight scale).
// CReLU clips accumulator to [0, QA=255].
// Output = sum(crelu(acc) * outputWeight) / QA + outputBias, then / QB.

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const (
	// V5 network dimensions
	NNUEv5HiddenSize = 1024 // per perspective (wider than v4's 256)

	// V5 quantization scale factors
	nnueV5InputScale  = 255 // CReLU clips to [0, 255] — Bullet standard for CReLU/SCReLU
	nnueV5OutputScale = 64  // output weight scale
	nnueV5BiasScale   = nnueV5InputScale * nnueV5OutputScale // 16320

	// V5 CReLU bounds
	nnueV5ClipMin = 0
	nnueV5ClipMax = nnueV5InputScale // 255

	// V5 eval scale: Bullet trains in sigmoid space where output/400 → win probability.
	// The network output needs to be multiplied by this to get centipawns.
	nnueV5EvalScale = 400

	// File version
	nnueVersionV5 = uint32(5)
)

// NNUENetV5 holds v5 network weights (shared read-only across threads).
type NNUENetV5 struct {
	// Input layer: [feature_index][hidden_neuron] — 12288 × 1024
	InputWeights [NNUEInputSize][NNUEv5HiddenSize]int16
	InputBiases  [NNUEv5HiddenSize]int16

	// Output layer: [bucket][concat_hidden] — 8 × 2048
	// concat_hidden = STM accumulator (1024) + NTM accumulator (1024)
	OutputWeights [NNUEOutputBuckets][NNUEv5HiddenSize * 2]int16
	OutputBias    [NNUEOutputBuckets]int32
}

// NNUEAccumulatorV5 holds per-position accumulator state for v5 nets.
type NNUEAccumulatorV5 struct {
	White    [NNUEv5HiddenSize]int16
	Black    [NNUEv5HiddenSize]int16
	Computed bool
}

// NNUEAccumulatorStackV5 provides push/pop for MakeMove/UnmakeMove.
type NNUEAccumulatorStackV5 struct {
	stack []NNUEAccumulatorV5
	top   int
}

// NewNNUEAccumulatorStackV5 creates a new v5 accumulator stack.
func NewNNUEAccumulatorStackV5(capacity int) *NNUEAccumulatorStackV5 {
	if capacity < 1 {
		capacity = 256
	}
	return &NNUEAccumulatorStackV5{
		stack: make([]NNUEAccumulatorV5, capacity),
		top:   0,
	}
}

// Current returns the current accumulator.
func (s *NNUEAccumulatorStackV5) Current() *NNUEAccumulatorV5 {
	return &s.stack[s.top]
}

// Push advances the stack for MakeMove (no copy — needs recompute).
func (s *NNUEAccumulatorStackV5) Push() {
	s.top++
	if s.top >= len(s.stack) {
		s.stack = append(s.stack, NNUEAccumulatorV5{})
	}
	s.stack[s.top].Computed = false
}

// PushCopy advances the stack and copies the current accumulator for incremental update.
func (s *NNUEAccumulatorStackV5) PushCopy() {
	prev := s.stack[s.top]
	s.top++
	if s.top >= len(s.stack) {
		s.stack = append(s.stack, NNUEAccumulatorV5{})
	}
	s.stack[s.top] = prev // copy White, Black, Computed
}

// Pop restores the stack for UnmakeMove.
func (s *NNUEAccumulatorStackV5) Pop() {
	s.top--
}

// RecomputeAccumulator rebuilds the accumulator from scratch for a given perspective.
func (net *NNUENetV5) RecomputeAccumulator(b *Board, acc *NNUEAccumulatorV5, perspective Color) {
	var dst *[NNUEv5HiddenSize]int16
	if perspective == White {
		dst = &acc.White
	} else {
		dst = &acc.Black
	}

	// Start with biases
	copy(dst[:], net.InputBiases[:])

	kingSq := b.Pieces[pieceOf(WhiteKing, perspective)].LSB()

	// Add all pieces on the board
	occ := b.AllPieces
	for occ != 0 {
		sq := Square(occ.PopLSB())
		piece := b.Squares[sq]
		if piece == Empty {
			continue
		}

		idx := HalfKAIndex(perspective, kingSq, piece, sq)
		if idx < 0 {
			continue
		}

		for i := 0; i < NNUEv5HiddenSize; i++ {
			dst[i] += net.InputWeights[idx][i]
		}
	}
}

// AddFeatureV5 adds a feature to the v5 accumulator for both perspectives.
func (net *NNUENetV5) AddFeature(acc *NNUEAccumulatorV5, piece Piece, sq Square, wKingSq, bKingSq Square) {
	wIdx := HalfKAIndex(White, wKingSq, piece, sq)
	bIdx := HalfKAIndex(Black, bKingSq, piece, sq)

	if wIdx >= 0 {
		for i := 0; i < NNUEv5HiddenSize; i++ {
			acc.White[i] += net.InputWeights[wIdx][i]
		}
	}
	if bIdx >= 0 {
		for i := 0; i < NNUEv5HiddenSize; i++ {
			acc.Black[i] += net.InputWeights[bIdx][i]
		}
	}
}

// RemoveFeatureV5 removes a feature from the v5 accumulator for both perspectives.
func (net *NNUENetV5) RemoveFeature(acc *NNUEAccumulatorV5, piece Piece, sq Square, wKingSq, bKingSq Square) {
	wIdx := HalfKAIndex(White, wKingSq, piece, sq)
	bIdx := HalfKAIndex(Black, bKingSq, piece, sq)

	if wIdx >= 0 {
		for i := 0; i < NNUEv5HiddenSize; i++ {
			acc.White[i] -= net.InputWeights[wIdx][i]
		}
	}
	if bIdx >= 0 {
		for i := 0; i < NNUEv5HiddenSize; i++ {
			acc.Black[i] -= net.InputWeights[bIdx][i]
		}
	}
}

// ForwardV5 computes the v5 NNUE evaluation. Returns centipawns from side-to-move perspective.
func (net *NNUENetV5) Forward(acc *NNUEAccumulatorV5, stm Color, pieceCount int) int {
	bucket := OutputBucket(pieceCount)

	var stmAcc, ntmAcc *[NNUEv5HiddenSize]int16
	if stm == White {
		stmAcc = &acc.White
		ntmAcc = &acc.Black
	} else {
		stmAcc = &acc.Black
		ntmAcc = &acc.White
	}

	// Compute output: dot product of CReLU(accumulators) with output weights + bias
	output := int32(net.OutputBias[bucket])

	// STM perspective (first half of output weights)
	for i := 0; i < NNUEv5HiddenSize; i++ {
		// CReLU: clamp to [0, QA=255]
		v := int32(stmAcc[i])
		if v < nnueV5ClipMin {
			v = nnueV5ClipMin
		}
		if v > nnueV5ClipMax {
			v = nnueV5ClipMax
		}
		output += v * int32(net.OutputWeights[bucket][i])
	}

	// NTM perspective (second half of output weights)
	for i := 0; i < NNUEv5HiddenSize; i++ {
		v := int32(ntmAcc[i])
		if v < nnueV5ClipMin {
			v = nnueV5ClipMin
		}
		if v > nnueV5ClipMax {
			v = nnueV5ClipMax
		}
		output += v * int32(net.OutputWeights[bucket][NNUEv5HiddenSize+i])
	}

	// Scale: divide by QA*QB to get the raw network output, then multiply by eval_scale
	// output is at scale QA * QB = 255 * 64 = 16320
	// Final centipawns = output / 16320 * 400 = output * 400 / 16320
	// Simplify: output * 25 / 1020 (close enough for integer arithmetic)
	// Or more precisely: output / 16320 * 400
	result := int(output) * nnueV5EvalScale / nnueV5BiasScale

	return result
}

// Fingerprint returns a checksum string for the v5 network.
func (net *NNUENetV5) Fingerprint() string {
	var h uint64
	for i := 0; i < NNUEv5HiddenSize && i < len(net.InputWeights[0]); i++ {
		h = h*31 + uint64(uint16(net.InputWeights[0][i]))
	}
	h = h*31 + uint64(uint32(net.OutputBias[0]))
	return fmt.Sprintf("%016x", h)
}

// SaveNNUEV5 writes a v5 network to a binary file.
func SaveNNUEV5(path string, net *NNUENetV5) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return writeNNUEV5(f, net)
}

func writeNNUEV5(w io.Writer, net *NNUENetV5) error {
	if err := binary.Write(w, binary.LittleEndian, nnueMagic); err != nil {
		return fmt.Errorf("writing magic: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, nnueVersionV5); err != nil {
		return fmt.Errorf("writing version: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, &net.InputWeights); err != nil {
		return fmt.Errorf("writing input weights: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, &net.InputBiases); err != nil {
		return fmt.Errorf("writing input biases: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, &net.OutputWeights); err != nil {
		return fmt.Errorf("writing output weights: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, &net.OutputBias); err != nil {
		return fmt.Errorf("writing output bias: %w", err)
	}
	return nil
}

// LoadNNUEV5 reads a v5 network from a binary file.
func LoadNNUEV5(path string) (*NNUENetV5, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic, version uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("reading magic: %w", err)
	}
	if magic != nnueMagic {
		return nil, fmt.Errorf("invalid NNUE magic: 0x%X", magic)
	}
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("reading version: %w", err)
	}
	if version != uint32(5) {
		return nil, fmt.Errorf("expected v5, got v%d", version)
	}

	net := &NNUENetV5{}
	if err := binary.Read(f, binary.LittleEndian, &net.InputWeights); err != nil {
		return nil, fmt.Errorf("reading input weights: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &net.InputBiases); err != nil {
		return nil, fmt.Errorf("reading input biases: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &net.OutputWeights); err != nil {
		return nil, fmt.Errorf("reading output weights: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &net.OutputBias); err != nil {
		return nil, fmt.Errorf("reading output bias: %w", err)
	}

	return net, nil
}

// NNUEEvaluateRelativeV5 returns the v5 NNUE evaluation from the side to move's perspective.
func (b *Board) NNUEEvaluateRelativeV5() int {
	acc := b.NNUEAccV5.Current()

	// Recompute if needed
	if !acc.Computed {
		b.NNUENetV5.RecomputeAccumulator(b, acc, White)
		b.NNUENetV5.RecomputeAccumulator(b, acc, Black)
		acc.Computed = true
	}

	// Count pieces for output bucket selection
	pieceCount := 0
	occ := b.AllPieces
	for occ != 0 {
		pieceCount++
		occ &= occ - 1
	}

	return b.NNUENetV5.Forward(acc, b.SideToMove, pieceCount)
}

// DetectNNUEVersion reads the header of an NNUE file and returns the version number.
func DetectNNUEVersion(path string) (uint32, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	var magic, version uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return 0, fmt.Errorf("reading magic: %w", err)
	}
	if magic != nnueMagic {
		return 0, fmt.Errorf("invalid NNUE magic: 0x%X", magic)
	}
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return 0, fmt.Errorf("reading version: %w", err)
	}
	return version, nil
}

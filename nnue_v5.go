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
	// V5 default network dimensions (overridden by file contents at load time)
	NNUEv5DefaultHidden = 1024 // default if not detected

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
// Hidden size is dynamic — detected from the file at load time.
type NNUENetV5 struct {
	HiddenSize int // per perspective (e.g. 1024, 1536)

	// Input layer: [feature_index * HiddenSize] — flattened 12288 × HiddenSize
	InputWeights []int16
	InputBiases  []int16

	// Output layer: [bucket * HiddenSize*2] — flattened 8 × (2*HiddenSize)
	OutputWeights []int16
	OutputBias    [NNUEOutputBuckets]int32

	// Activation: false = CReLU (clamp to [0,QA]), true = SCReLU (clamp then square)
	UseSCReLU bool

	// Pairwise: consecutive accumulator pairs are multiplied before output layer.
	// Halves the effective width: HiddenSize → HiddenSize/2 per perspective.
	UsePairwise bool
}

// inputWeightRow returns the input weights for a given feature index.
func (net *NNUENetV5) inputWeightRow(featureIdx int) []int16 {
	off := featureIdx * net.HiddenSize
	return net.InputWeights[off : off+net.HiddenSize]
}

// outputWeightRow returns the output weights for a given bucket.
func (net *NNUENetV5) outputWeightRow(bucket int) []int16 {
	outWidth := net.HiddenSize * 2
	if net.UsePairwise {
		outWidth = net.HiddenSize // pairwise halves it
	}
	off := bucket * outWidth
	return net.OutputWeights[off : off+outWidth]
}

// NNUEAccumulatorV5 holds per-position accumulator state for v5 nets.
type NNUEAccumulatorV5 struct {
	White    []int16
	Black    []int16
	Computed bool
	Dirty    DirtyPiece // lazy materialization: pending update from MakeMove
}

// FinnyEntryV5 caches an accumulator and the board state that produced it.
// Used to avoid full recomputes on king bucket changes.
type FinnyEntryV5 struct {
	Acc     []int16      // cached accumulator values (len = hiddenSize)
	Pieces  [13]Bitboard // piece occupancy when this entry was last written
	Valid   bool         // whether this entry has been populated
}

// NNUEAccumulatorStackV5 provides push/pop for MakeMove/UnmakeMove.
type NNUEAccumulatorStackV5 struct {
	stack      []NNUEAccumulatorV5
	top        int
	hiddenSize int
	// Finny table: [perspective][kingBucket][mirror] accumulator cache
	// mirror=0 for king on files a-d, mirror=1 for king on files e-h
	finny [2][NNUEKingBuckets][2]FinnyEntryV5
}

// NewNNUEAccumulatorStackV5 creates a new v5 accumulator stack.
func NewNNUEAccumulatorStackV5(capacity int) *NNUEAccumulatorStackV5 {
	return NewNNUEAccumulatorStackV5WithSize(capacity, NNUEv5DefaultHidden)
}

// NewNNUEAccumulatorStackV5WithSize creates a stack with a specific hidden size.
func NewNNUEAccumulatorStackV5WithSize(capacity, hiddenSize int) *NNUEAccumulatorStackV5 {
	if capacity < 1 {
		capacity = 256
	}
	s := &NNUEAccumulatorStackV5{
		stack:      make([]NNUEAccumulatorV5, capacity),
		top:        0,
		hiddenSize: hiddenSize,
	}
	for i := range s.stack {
		s.stack[i].White = make([]int16, hiddenSize)
		s.stack[i].Black = make([]int16, hiddenSize)
	}
	// Allocate finny table entries
	for p := 0; p < 2; p++ {
		for bk := 0; bk < NNUEKingBuckets; bk++ {
			for m := 0; m < 2; m++ {
				s.finny[p][bk][m].Acc = make([]int16, hiddenSize)
			}
		}
	}
	return s
}

// DeepCopy creates a deep copy of the v5 accumulator stack (for Lazy SMP thread copies).
func (s *NNUEAccumulatorStackV5) DeepCopy() *NNUEAccumulatorStackV5 {
	newStack := &NNUEAccumulatorStackV5{
		stack:      make([]NNUEAccumulatorV5, len(s.stack)),
		top:        s.top,
		hiddenSize: s.hiddenSize,
	}
	for i := range s.stack {
		newStack.stack[i].White = make([]int16, s.hiddenSize)
		newStack.stack[i].Black = make([]int16, s.hiddenSize)
		copy(newStack.stack[i].White, s.stack[i].White)
		copy(newStack.stack[i].Black, s.stack[i].Black)
		newStack.stack[i].Computed = s.stack[i].Computed
		newStack.stack[i].Dirty = s.stack[i].Dirty
	}
	// Deep copy finny table
	for p := 0; p < 2; p++ {
		for bk := 0; bk < NNUEKingBuckets; bk++ {
			for m := 0; m < 2; m++ {
				newStack.finny[p][bk][m].Acc = make([]int16, s.hiddenSize)
				copy(newStack.finny[p][bk][m].Acc, s.finny[p][bk][m].Acc)
				newStack.finny[p][bk][m].Pieces = s.finny[p][bk][m].Pieces
				newStack.finny[p][bk][m].Valid = s.finny[p][bk][m].Valid
			}
		}
	}
	return newStack
}

// Current returns the current accumulator.
func (s *NNUEAccumulatorStackV5) Current() *NNUEAccumulatorV5 {
	return &s.stack[s.top]
}

// Push advances the stack for MakeMove. Dirty.Type is set by the caller.
// No accumulator copy — MaterializeV5 copies from parent on demand.
func (s *NNUEAccumulatorStackV5) Push() {
	s.top++
	if s.top >= len(s.stack) {
		s.stack = append(s.stack, NNUEAccumulatorV5{
			White: make([]int16, s.hiddenSize),
			Black: make([]int16, s.hiddenSize),
		})
	}
	s.stack[s.top].Computed = false
	s.stack[s.top].Dirty.Type = 0 // default: full recompute
}

// Parent returns the parent accumulator (one level up in the stack).
func (s *NNUEAccumulatorStackV5) Parent() *NNUEAccumulatorV5 {
	if s.top <= 0 {
		return nil
	}
	return &s.stack[s.top-1]
}

// Pop restores the stack for UnmakeMove.
func (s *NNUEAccumulatorStackV5) Pop() {
	s.top--
}

// InvalidateFinny clears all finny table entries (e.g. after SetFEN/Reset).
func (s *NNUEAccumulatorStackV5) InvalidateFinny() {
	for p := 0; p < 2; p++ {
		for bk := 0; bk < NNUEKingBuckets; bk++ {
			for m := 0; m < 2; m++ {
				s.finny[p][bk][m].Valid = false
			}
		}
	}
}

// RefreshAccumulator uses the finny table to avoid full recomputes.
// Diffs cached piece bitboards against current board state and applies only
// the changed features. Falls back to full recompute if no cache entry exists.
func (s *NNUEAccumulatorStackV5) RefreshAccumulator(net *NNUENetV5, b *Board, acc *NNUEAccumulatorV5, perspective Color) {
	kingSq := b.Pieces[pieceOf(WhiteKing, perspective)].LSB()
	// Perspective-adjusted bucket: Black mirrors the king square vertically
	ks := int(kingSq)
	if perspective == Black {
		ks ^= 56
	}
	bucket := kingBucketTable[ks]
	mirrorIdx := 0
	if kingBucketMirrorFile[ks] {
		mirrorIdx = 1
	}
	entry := &s.finny[perspective][bucket][mirrorIdx]

	var dst []int16
	if perspective == White {
		dst = acc.White
	} else {
		dst = acc.Black
	}

	if !entry.Valid {
		// No cache — full recompute and populate
		net.RecomputeAccumulator(b, acc, perspective)
		copy(entry.Acc, dst)
		entry.Pieces = b.Pieces
		entry.Valid = true
		return
	}

	// Apply deltas directly to the cached accumulator, then copy to dst.
	// This avoids one full accumulator copy vs the old copy-modify-writeback pattern.
	cachedAcc := entry.Acc

	// Diff each piece type's bitboard: cached vs current
	for pc := Piece(1); pc <= 12; pc++ {
		prev := entry.Pieces[pc]
		curr := b.Pieces[pc]
		if prev == curr {
			continue
		}
		// Removed squares: in prev but not in curr
		removed := prev &^ curr
		for removed != 0 {
			sq := Square(removed.PopLSB())
			idx := HalfKAIndex(perspective, kingSq, pc, sq)
			if idx >= 0 {
				subWeightsV5Slice(cachedAcc, net.inputWeightRow(idx))
			}
		}
		// Added squares: in curr but not in prev
		added := curr &^ prev
		for added != 0 {
			sq := Square(added.PopLSB())
			idx := HalfKAIndex(perspective, kingSq, pc, sq)
			if idx >= 0 {
				addWeightsV5Slice(cachedAcc, net.inputWeightRow(idx))
			}
		}
	}

	// Copy updated cache to accumulator
	copy(dst, cachedAcc)
	entry.Pieces = b.Pieces
}

// RecomputeAccumulator rebuilds the accumulator from scratch for a given perspective.
func (net *NNUENetV5) RecomputeAccumulator(b *Board, acc *NNUEAccumulatorV5, perspective Color) {
	var dst []int16
	if perspective == White {
		dst = acc.White
	} else {
		dst = acc.Black
	}

	// Start with biases
	copy(dst, net.InputBiases)

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

		addWeightsV5Slice(dst, net.inputWeightRow(idx))
	}
}

// addWeightsV5Slice adds int16 weights to an accumulator slice (dynamic width).
// Uses a single SIMD call for the full width.
func addWeightsV5Slice(acc []int16, weights []int16) {
	n := len(acc)
	if nnueUseSIMD {
		nnueAccAddN(&acc[0], &weights[0], n)
	} else {
		for i := 0; i < n; i++ {
			acc[i] += weights[i]
		}
	}
}

// subWeightsV5Slice subtracts int16 weights from an accumulator slice (dynamic width).
func subWeightsV5Slice(acc []int16, weights []int16) {
	n := len(acc)
	if nnueUseSIMD {
		nnueAccSubN(&acc[0], &weights[0], n)
	} else {
		for i := 0; i < n; i++ {
			acc[i] -= weights[i]
		}
	}
}

// AddFeature adds a feature to the v5 accumulator for both perspectives.
func (net *NNUENetV5) AddFeature(acc *NNUEAccumulatorV5, piece Piece, sq Square, wKingSq, bKingSq Square) {
	wIdx := HalfKAIndex(White, wKingSq, piece, sq)
	bIdx := HalfKAIndex(Black, bKingSq, piece, sq)

	if wIdx >= 0 {
		addWeightsV5Slice(acc.White, net.inputWeightRow(wIdx))
	}
	if bIdx >= 0 {
		addWeightsV5Slice(acc.Black, net.inputWeightRow(bIdx))
	}
}

// RemoveFeature removes a feature from the v5 accumulator for both perspectives.
func (net *NNUENetV5) RemoveFeature(acc *NNUEAccumulatorV5, piece Piece, sq Square, wKingSq, bKingSq Square) {
	wIdx := HalfKAIndex(White, wKingSq, piece, sq)
	bIdx := HalfKAIndex(Black, bKingSq, piece, sq)

	if wIdx >= 0 {
		subWeightsV5Slice(acc.White, net.inputWeightRow(wIdx))
	}
	if bIdx >= 0 {
		subWeightsV5Slice(acc.Black, net.inputWeightRow(bIdx))
	}
}

// ForwardV5 computes the v5 NNUE evaluation. Returns centipawns from side-to-move perspective.
func (net *NNUENetV5) Forward(acc *NNUEAccumulatorV5, stm Color, pieceCount int) int {
	bucket := OutputBucket(pieceCount)
	H := net.HiddenSize
	PW := H / 2 // pairwise size

	var stmAcc, ntmAcc []int16
	if stm == White {
		stmAcc = acc.White
		ntmAcc = acc.Black
	} else {
		stmAcc = acc.Black
		ntmAcc = acc.White
	}

	outW := net.outputWeightRow(bucket)

	var output int64
	if net.UsePairwise && nnueUseSIMDV5 {
		sum := nnueV5PairwiseDotN(&stmAcc[0], &stmAcc[PW], &outW[0], PW)
		sum += nnueV5PairwiseDotN(&ntmAcc[0], &ntmAcc[PW], &outW[PW], PW)
		output = sum/int64(nnueV5InputScale) + int64(net.OutputBias[bucket])
	} else if net.UsePairwise {
		output = int64(net.OutputBias[bucket])
		for i := 0; i < PW; i++ {
			a := int32(stmAcc[i])
			b := int32(stmAcc[i+PW])
			if a < 0 { a = 0 }
			if a > nnueV5ClipMax { a = nnueV5ClipMax }
			if b < 0 { b = 0 }
			if b > nnueV5ClipMax { b = nnueV5ClipMax }
			output += int64((a * b) / nnueV5InputScale) * int64(outW[i])
		}
		for i := 0; i < PW; i++ {
			a := int32(ntmAcc[i])
			b := int32(ntmAcc[i+PW])
			if a < 0 { a = 0 }
			if a > nnueV5ClipMax { a = nnueV5ClipMax }
			if b < 0 { b = 0 }
			if b > nnueV5ClipMax { b = nnueV5ClipMax }
			output += int64((a * b) / nnueV5InputScale) * int64(outW[PW+i])
		}
	} else if net.UseSCReLU && nnueUseSIMDV5 {
		sum := nnueV5SCReLUDotN(&stmAcc[0], &outW[0], H)
		sum += nnueV5SCReLUDotN(&ntmAcc[0], &outW[H], H)
		output = sum/int64(nnueV5InputScale) + int64(net.OutputBias[bucket])
	} else if net.UseSCReLU {
		var sum int64
		for i := 0; i < H; i++ {
			v := int32(stmAcc[i])
			if v < 0 { v = 0 }
			if v > nnueV5ClipMax { v = nnueV5ClipMax }
			sum += int64(v*v) * int64(outW[i])
		}
		for i := 0; i < H; i++ {
			v := int32(ntmAcc[i])
			if v < 0 { v = 0 }
			if v > nnueV5ClipMax { v = nnueV5ClipMax }
			sum += int64(v*v) * int64(outW[H+i])
		}
		output = sum/int64(nnueV5InputScale) + int64(net.OutputBias[bucket])
	} else if nnueUseSIMDV5 {
		output = int64(net.OutputBias[bucket])
		output += int64(nnueV5CReLUDotN(&stmAcc[0], &outW[0], H))
		output += int64(nnueV5CReLUDotN(&ntmAcc[0], &outW[H], H))
	} else {
		output = int64(net.OutputBias[bucket])
		for i := 0; i < H; i++ {
			v := int32(stmAcc[i])
			if v < 0 { v = 0 }
			if v > nnueV5ClipMax { v = nnueV5ClipMax }
			output += int64(v) * int64(outW[i])
		}
		for i := 0; i < H; i++ {
			v := int32(ntmAcc[i])
			if v < 0 { v = 0 }
			if v > nnueV5ClipMax { v = nnueV5ClipMax }
			output += int64(v) * int64(outW[H+i])
		}
	}

	// Scale: divide by QA*QB to get the raw network output, then multiply by eval_scale
	// output is at scale QA * QB = 255 * 64 = 16320
	// Final centipawns = output / 16320 * 400 = output * 400 / 16320
	result := int(output) * nnueV5EvalScale / nnueV5BiasScale

	// SCReLU eval scale correction: squared activation produces wider dynamic range
	// than CReLU, making search thresholds (tuned for CReLU) effectively tighter.
	// Bracketed cross-engine: 0.75 (+28) < 0.80 (+35) > 0.85 (+7). Peak at 0.80.
	// TODO: apply this correction in the Bullet converter output weights instead.
	if net.UseSCReLU {
		result = result * 4 / 5
	}

	return result
}

// Fingerprint returns a checksum string for the v5 network.
func (net *NNUENetV5) Fingerprint() string {
	var h uint64
	row := net.inputWeightRow(0)
	for i := 0; i < len(row); i++ {
		h = h*31 + uint64(uint16(row[i]))
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
	version := nnueVersionV5 // v5 for plain CReLU
	if net.UseSCReLU || net.UsePairwise {
		version = uint32(6) // v6 has flags byte
	}
	if err := binary.Write(w, binary.LittleEndian, version); err != nil {
		return fmt.Errorf("writing version: %w", err)
	}
	if version == 6 {
		var flags uint8
		if net.UseSCReLU {
			flags |= 1
		}
		if net.UsePairwise {
			flags |= 2
		}
		if err := binary.Write(w, binary.LittleEndian, flags); err != nil {
			return fmt.Errorf("writing flags: %w", err)
		}
	}
	if err := binary.Write(w, binary.LittleEndian, net.InputWeights); err != nil {
		return fmt.Errorf("writing input weights: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, net.InputBiases); err != nil {
		return fmt.Errorf("writing input biases: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, net.OutputWeights); err != nil {
		return fmt.Errorf("writing output weights: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, &net.OutputBias); err != nil {
		return fmt.Errorf("writing output bias: %w", err)
	}
	return nil
}

// inferV5HiddenSize computes the hidden size from file size and header offset.
// Layout: InputWeights(NNUEInputSize*H*2) + InputBiases(H*2) + OutputWeights(8*outW*2) + OutputBias(8*4)
// where outW = H*2 (plain) or H (pairwise).
func inferV5HiddenSize(fileSize int64, headerSize int64, pairwise bool) (int, error) {
	dataSize := fileSize - headerSize
	biasBytes := int64(NNUEOutputBuckets * 4) // OutputBias: 8 * int32
	dataSize -= biasBytes

	// dataSize = H * (NNUEInputSize*2 + 2 + outMul*2)
	// outMul = 8*2 (plain) or 8 (pairwise) — output weights per hidden neuron
	var outMul int64
	if pairwise {
		outMul = int64(NNUEOutputBuckets) // 8 weights per neuron (pairwise halves)
	} else {
		outMul = int64(NNUEOutputBuckets) * 2 // 16 weights per neuron (both perspectives)
	}
	bytesPerNeuron := int64(NNUEInputSize)*2 + 2 + outMul*2
	if dataSize <= 0 || dataSize%bytesPerNeuron != 0 {
		return 0, fmt.Errorf("cannot infer hidden size: data=%d bytes, per_neuron=%d", dataSize, bytesPerNeuron)
	}
	return int(dataSize / bytesPerNeuron), nil
}

// LoadNNUEV5 reads a v5 network from a binary file. Hidden size is auto-detected.
func LoadNNUEV5(path string) (*NNUENetV5, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Get file size for hidden size inference
	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	fileSize := fi.Size()

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

	net := &NNUENetV5{}
	var headerSize int64

	if version == 5 {
		headerSize = 8 // magic + version
	} else if version == 6 {
		var flags uint8
		if err := binary.Read(f, binary.LittleEndian, &flags); err != nil {
			return nil, fmt.Errorf("reading flags: %w", err)
		}
		net.UseSCReLU = flags&1 != 0
		net.UsePairwise = flags&2 != 0
		headerSize = 9 // magic + version + flags
	} else {
		return nil, fmt.Errorf("expected v5 or v6, got v%d", version)
	}

	H, err := inferV5HiddenSize(fileSize, headerSize, net.UsePairwise)
	if err != nil {
		return nil, fmt.Errorf("v%d: %w", version, err)
	}
	if nnueUseSIMD && H%256 != 0 {
		return nil, fmt.Errorf("v%d: hidden size %d is not a multiple of 256 (required for SIMD accumulator ops)", version, H)
	}
	net.HiddenSize = H

	return readNNUEV5Body(f, net)
}

func readNNUEV5Body(f io.Reader, net *NNUENetV5) (*NNUENetV5, error) {
	H := net.HiddenSize

	// Allocate and read input weights: NNUEInputSize × H
	net.InputWeights = make([]int16, NNUEInputSize*H)
	if err := binary.Read(f, binary.LittleEndian, net.InputWeights); err != nil {
		return nil, fmt.Errorf("reading input weights: %w", err)
	}

	// Input biases: H
	net.InputBiases = make([]int16, H)
	if err := binary.Read(f, binary.LittleEndian, net.InputBiases); err != nil {
		return nil, fmt.Errorf("reading input biases: %w", err)
	}

	// Output weights: 8 × outWidth
	outWidth := H * 2
	if net.UsePairwise {
		outWidth = H
	}
	net.OutputWeights = make([]int16, NNUEOutputBuckets*outWidth)
	if err := binary.Read(f, binary.LittleEndian, net.OutputWeights); err != nil {
		return nil, fmt.Errorf("reading output weights: %w", err)
	}

	// Output bias: 8 × int32
	if err := binary.Read(f, binary.LittleEndian, &net.OutputBias); err != nil {
		return nil, fmt.Errorf("reading output bias: %w", err)
	}

	return net, nil
}

// NNUEEvaluateRelativeV5 returns the v5 NNUE evaluation from the side to move's perspective.
func (b *Board) NNUEEvaluateRelativeV5() int {
	acc := b.NNUEAccV5.Current()

	// Lazy materialization: copy from parent + apply delta, or full recompute
	if !acc.Computed {
		b.NNUEAccV5.MaterializeV5(b.NNUENetV5, b)
	}

	// Count pieces for output bucket selection
	pieceCount := b.AllPieces.Count()

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

// copySubAddV5Slice computes dst[i] = src[i] + newW[i] - oldW[i] for dynamic width.
// Fused single-pass: avoids separate copy + sub + add.
func copySubAddV5Slice(dst, src, oldW, newW []int16) {
	n := len(dst)
	if nnueUseSIMD {
		nnueAccCopySubAddN(&dst[0], &src[0], &oldW[0], &newW[0], n)
	} else {
		for i := 0; i < n; i++ {
			dst[i] = src[i] + newW[i] - oldW[i]
		}
	}
}

// copySubSubAddV5Slice computes dst[i] = src[i] + newW[i] - oldW[i] - capW[i] for dynamic width.
// Fused single-pass for captures.
func copySubSubAddV5Slice(dst, src, oldW, newW, capW []int16) {
	n := len(dst)
	if nnueUseSIMD {
		nnueAccCopySubSubAddN(&dst[0], &src[0], &oldW[0], &newW[0], &capW[0], n)
	} else {
		for i := 0; i < n; i++ {
			dst[i] = src[i] + newW[i] - oldW[i] - capW[i]
		}
	}
}

// MaterializeV5 applies the pending lazy update for the current accumulator.
// Copies from the parent accumulator and applies the dirty piece delta.
// Falls back to full recompute if: dirty type is 0 (king bucket change),
// at root (no parent), or parent is not yet materialized.
func (s *NNUEAccumulatorStackV5) MaterializeV5(net *NNUENetV5, b *Board) {
	acc := &s.stack[s.top]
	d := &acc.Dirty

	// Full recompute needed?
	if d.Type == 0 || s.top == 0 || !s.stack[s.top-1].Computed {
		s.RefreshAccumulator(net, b, acc, White)
		s.RefreshAccumulator(net, b, acc, Black)
		acc.Computed = true
		return
	}

	parent := &s.stack[s.top-1]
	wKingSq := b.Pieces[WhiteKing].LSB()
	bKingSq := b.Pieces[BlackKing].LSB()

	switch d.Type {
	case 1, 6: // quiet move or king move same bucket — fused copy+SubAdd
		wIdxOld := HalfKAIndex(White, wKingSq, d.Piece, d.From)
		wIdxNew := HalfKAIndex(White, wKingSq, d.Piece, d.To)
		if wIdxOld >= 0 && wIdxNew >= 0 {
			copySubAddV5Slice(acc.White, parent.White,
				net.inputWeightRow(wIdxOld), net.inputWeightRow(wIdxNew))
		} else {
			copy(acc.White, parent.White)
		}
		bIdxOld := HalfKAIndex(Black, bKingSq, d.Piece, d.From)
		bIdxNew := HalfKAIndex(Black, bKingSq, d.Piece, d.To)
		if bIdxOld >= 0 && bIdxNew >= 0 {
			copySubAddV5Slice(acc.Black, parent.Black,
				net.inputWeightRow(bIdxOld), net.inputWeightRow(bIdxNew))
		} else {
			copy(acc.Black, parent.Black)
		}
	case 2, 3: // capture or en passant — fused copy+SubSubAdd
		wIdxMoveOld := HalfKAIndex(White, wKingSq, d.Piece, d.From)
		wIdxMoveNew := HalfKAIndex(White, wKingSq, d.Piece, d.To)
		wIdxCap := HalfKAIndex(White, wKingSq, d.CapPiece, d.CapSq)
		if wIdxMoveOld >= 0 && wIdxMoveNew >= 0 && wIdxCap >= 0 {
			copySubSubAddV5Slice(acc.White, parent.White,
				net.inputWeightRow(wIdxMoveOld), net.inputWeightRow(wIdxMoveNew),
				net.inputWeightRow(wIdxCap))
		} else {
			copy(acc.White, parent.White)
		}
		bIdxMoveOld := HalfKAIndex(Black, bKingSq, d.Piece, d.From)
		bIdxMoveNew := HalfKAIndex(Black, bKingSq, d.Piece, d.To)
		bIdxCap := HalfKAIndex(Black, bKingSq, d.CapPiece, d.CapSq)
		if bIdxMoveOld >= 0 && bIdxMoveNew >= 0 && bIdxCap >= 0 {
			copySubSubAddV5Slice(acc.Black, parent.Black,
				net.inputWeightRow(bIdxMoveOld), net.inputWeightRow(bIdxMoveNew),
				net.inputWeightRow(bIdxCap))
		} else {
			copy(acc.Black, parent.Black)
		}
	case 4: // promotion (rare) — copy + update
		copy(acc.White, parent.White)
		copy(acc.Black, parent.Black)
		net.RemoveFeature(acc, d.Piece, d.From, wKingSq, bKingSq)
		net.AddFeature(acc, d.PromoPc, d.To, wKingSq, bKingSq)
	case 5: // capture-promotion (rare) — copy + update
		copy(acc.White, parent.White)
		copy(acc.Black, parent.Black)
		net.RemoveFeature(acc, d.CapPiece, d.To, wKingSq, bKingSq)
		net.RemoveFeature(acc, d.Piece, d.From, wKingSq, bKingSq)
		net.AddFeature(acc, d.PromoPc, d.To, wKingSq, bKingSq)
	case 7: // castling same bucket (rare) — copy + update
		copy(acc.White, parent.White)
		copy(acc.Black, parent.Black)
		net.RemoveFeature(acc, d.Piece, d.From, wKingSq, bKingSq)
		net.AddFeature(acc, d.Piece, d.To, wKingSq, bKingSq)
		rookPiece := Piece(WhiteRook)
		if d.Piece == BlackKing {
			rookPiece = BlackRook
		}
		net.RemoveFeature(acc, rookPiece, d.RookFrom, wKingSq, bKingSq)
		net.AddFeature(acc, rookPiece, d.RookTo, wKingSq, bKingSq)
	}

	acc.Computed = true
}

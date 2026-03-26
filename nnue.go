package chess

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// NNUE architecture: HalfKA with king buckets
// Input features: king_bucket (16) * piece_type (12: 6 piece types x 2 colors) * piece_square (64) = 12288
// Two accumulators: White perspective, Black perspective
// Concatenated (512) -> 32 hidden neurons -> 32 hidden neurons -> 1 output
// ClippedReLU activations, int16 weights for inference

const (
	// Network dimensions
	NNUEInputSize   = 12288 // 16 buckets * 12 piece types * 64 squares
	NNUEHiddenSize  = 256   // per perspective
	NNUEHidden2Size = 32    // second hidden layer
	NNUEHidden3Size = 32    // third hidden layer
	NNUEOutputSize  = 1     // legacy (unused, kept for reference)
	NNUEOutputBuckets = 8   // material-based output buckets

	// HalfKA piece indexing: 12 piece types (including kings)
	// 0=WhitePawn, 1=WhiteKnight, 2=WhiteBishop, 3=WhiteRook, 4=WhiteQueen, 5=WhiteKing
	// 6=BlackPawn, 7=BlackKnight, 8=BlackBishop, 9=BlackRook, 10=BlackQueen, 11=BlackKing
	nnueNumPieceTypes = 12

	// King buckets: 16 (4 mirrored files × 4 rank groups)
	NNUEKingBuckets = 16

	// Quantization scale factors
	nnueInputScale  = 127 // input layer weight scale
	nnueHiddenScale = 64  // hidden layer weight scale
	nnueOutputScale = nnueInputScale * nnueHiddenScale
	NNUEEvalScale   = 1 // post-inference scale factor (1 = no scaling)

	// ClippedReLU bounds (after quantization)
	nnueClipMin = 0
	nnueClipMax = nnueInputScale // 127

	// File magic and version
	nnueMagic   = uint32(0x4E4E5545) // "NNUE"
	nnueVersion = uint32(4)          // v4: output buckets (v3 was HalfKA without buckets)
)

// kingBucketTable maps each of the 64 squares to one of 16 king buckets.
// Uses horizontal mirroring (files a-d mirror to e-h) and 4 rank groups.
// Bucket = mirroredFile * 4 + rankGroup
//
//	rankGroup: ranks 1-2 = 0, ranks 3-4 = 1, ranks 5-6 = 2, ranks 7-8 = 3
//	mirroredFile: files a-d = 0-3, files e-h mirrored to 3-0
var kingBucketTable [64]int

// kingBucketMirrorFile indicates whether a king square requires file mirroring
// for piece square indexing (king on files e-h → mirror piece squares too).
var kingBucketMirrorFile [64]bool

func init() {
	for sq := 0; sq < 64; sq++ {
		file := sq % 8
		rank := sq / 8

		// Mirror file: map to 0-3 range
		mirroredFile := file
		mirror := false
		if file >= 4 {
			mirroredFile = 7 - file
			mirror = true
		}

		// Rank group: 0-1 → 0, 2-3 → 1, 4-5 → 2, 6-7 → 3
		rankGroup := rank / 2

		kingBucketTable[sq] = mirroredFile*4 + rankGroup
		kingBucketMirrorFile[sq] = mirror
	}
}

// KingBucket returns the king bucket index (0-15) for a king on the given square.
func KingBucket(sq Square) int {
	return kingBucketTable[sq]
}

// OutputBucket maps piece count (2-32) to an output bucket index (0-7).
// Bucket boundaries: pieces are divided into 8 roughly equal ranges.
// Minimum 2 pieces (both kings). Maximum 32 (all pieces).
func OutputBucket(pieceCount int) int {
	// (pieceCount - 2) maps to 0-30, divide by 4 to get 0-7, clamp
	bucket := (pieceCount - 2) / 4
	if bucket < 0 {
		bucket = 0
	}
	if bucket >= NNUEOutputBuckets {
		bucket = NNUEOutputBuckets - 1
	}
	return bucket
}

// NNUENet holds all network weights (shared read-only across threads).
type NNUENet struct {
	// Input layer: [feature_index][hidden_neuron]
	InputWeights [NNUEInputSize][NNUEHiddenSize]int16
	InputBiases  [NNUEHiddenSize]int16

	// Hidden layer 1: [concat_input][hidden2_neuron]
	HiddenWeights [NNUEHiddenSize * 2][NNUEHidden2Size]int16
	HiddenBiases  [NNUEHidden2Size]int32

	// Hidden layer 2: [hidden2_neuron][hidden3_neuron]
	Hidden2Weights [NNUEHidden2Size][NNUEHidden3Size]int16
	Hidden2Biases  [NNUEHidden3Size]int32

	// Output layer: [bucket][hidden3_neuron] — material-based output buckets
	OutputWeights [NNUEOutputBuckets][NNUEHidden3Size]int16
	OutputBias    [NNUEOutputBuckets]int32

	// Transposed hidden weights for SIMD forward pass: [output][input]
	// Layout: HWT[j*(HiddenSize*2)+i] = HiddenWeights[i][j]
	// Populated by PrepareWeights(), nil when SIMD not available.
	HWT []int16

	// Int8 transposed hidden weights for SIMD int8 matmul: same layout as HWT.
	// Values clamped from int16 to [-128, 127] at load time.
	// Used by nnueMatMul32x512_i8 (VPMADDUBSW on AVX2, SMULL on NEON).
	HWT8 []int8

	// Transposed hidden2 weights for SIMD: [output_k][input_j]
	// Layout: Hidden2WT[k*32+j] = Hidden2Weights[j][k]
	Hidden2WT []int16
}

// DirtyPiece describes a pending accumulator update (deferred from MakeMove).
type DirtyPiece struct {
	// Type encodes the move kind for lazy materialization.
	//   0 = no pending update (bucket-changing king move → full recompute)
	//   1 = quiet move (SubAdd: remove piece@From, add piece@To)
	//   2 = capture (SubSubAdd: remove piece@From, add piece@To, remove capPiece@CapSq)
	//   3 = en passant (SubSubAdd with different capSq)
	//   4 = promotion (Remove pawn@From, Add promoted@To)
	//   5 = capture-promotion (Remove captured@To, Remove pawn@From, Add promoted@To)
	//   6 = king move same bucket (SubAdd for king feature only, one perspective)
	//   7 = castling same bucket (king SubAdd + rook SubAdd, one perspective)
	Type     uint8
	Piece    Piece  // moving piece (before promotion)
	From     Square // origin square
	To       Square // destination square
	CapPiece Piece  // captured piece (for captures/EP)
	CapSq    Square // capture square (same as To for normal captures, different for EP)
	PromoPc  Piece  // promoted piece (for promotions)
	// For castling (type 7): rook from/to
	RookFrom Square
	RookTo   Square
}

// NNUEAccumulator holds per-position accumulator state.
type NNUEAccumulator struct {
	White    [NNUEHiddenSize]int16 // accumulator from White perspective
	Black    [NNUEHiddenSize]int16 // accumulator from Black perspective
	Computed bool
	Dirty    DirtyPiece // pending update (lazy materialization)
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

// Push advances the stack pointer for a lazy update.
// The accumulator is NOT copied — it will be materialized on demand
// when Evaluate() needs it. This saves the 1KB copy for nodes that
// get pruned before evaluation.
func (s *NNUEAccumulatorStack) Push() {
	if s.top+1 >= len(s.stack) {
		s.stack = append(s.stack, NNUEAccumulator{})
	}
	s.top++
	s.stack[s.top].Computed = false
	s.stack[s.top].Dirty.Type = 0
}

// PushEmpty advances the stack pointer without copying (for king moves
// that change bucket, where RecomputeAccumulator will overwrite everything).
func (s *NNUEAccumulatorStack) PushEmpty() {
	if s.top+1 >= len(s.stack) {
		s.stack = append(s.stack, NNUEAccumulator{})
	}
	s.top++
	s.stack[s.top].Computed = false
	s.stack[s.top].Dirty.Type = 0 // bucket change → full recompute on Materialize
}

// Pop restores the previous accumulator state.
func (s *NNUEAccumulatorStack) Pop() {
	if s.top > 0 {
		s.top--
	}
}

// Materialize ensures the current accumulator is computed by copying from the
// parent and applying the pending delta. This is the lazy evaluation core —
// nodes that are pruned before eval never pay this cost.
//
// If the parent is not computed (multiple lazy levels stacked), falls back to
// a full recompute from the board state. This keeps the implementation simple
// while still capturing the main benefit: skipping work for pruned nodes.
func (s *NNUEAccumulatorStack) Materialize(net *NNUENet, b *Board) {
	acc := &s.stack[s.top]
	if acc.Computed {
		return
	}

	d := &acc.Dirty
	if d.Type == 0 || s.top == 0 || !s.stack[s.top-1].Computed {
		// Bucket-changing king move, root position, or parent not materialized — full recompute
		net.RecomputeAccumulator(acc, b)
		return
	}

	parent := &s.stack[s.top-1]
	wKingSq := b.Pieces[WhiteKing].LSB()
	bKingSq := b.Pieces[BlackKing].LSB()

	switch d.Type {
	case 1, 6: // quiet move or king move same bucket — fused copy+SubAdd
		if nnueUseSIMD {
			wIdxOld := HalfKAIndex(White, wKingSq, d.Piece, d.From)
			wIdxNew := HalfKAIndex(White, wKingSq, d.Piece, d.To)
			if wIdxOld >= 0 && wIdxNew >= 0 {
				nnueAccCopySubAdd256(&acc.White[0], &parent.White[0],
					&net.InputWeights[wIdxOld][0], &net.InputWeights[wIdxNew][0])
			} else {
				acc.White = parent.White
			}
			bIdxOld := HalfKAIndex(Black, bKingSq, d.Piece, d.From)
			bIdxNew := HalfKAIndex(Black, bKingSq, d.Piece, d.To)
			if bIdxOld >= 0 && bIdxNew >= 0 {
				nnueAccCopySubAdd256(&acc.Black[0], &parent.Black[0],
					&net.InputWeights[bIdxOld][0], &net.InputWeights[bIdxNew][0])
			} else {
				acc.Black = parent.Black
			}
		} else {
			acc.White = parent.White
			acc.Black = parent.Black
			net.SubAddFeature(acc, d.Piece, d.From, d.To, wKingSq, bKingSq)
		}
	case 2, 3: // capture or en passant — fused copy+SubSubAdd
		if nnueUseSIMD {
			wIdxMoveOld := HalfKAIndex(White, wKingSq, d.Piece, d.From)
			wIdxMoveNew := HalfKAIndex(White, wKingSq, d.Piece, d.To)
			wIdxCap := HalfKAIndex(White, wKingSq, d.CapPiece, d.CapSq)
			if wIdxMoveOld >= 0 && wIdxMoveNew >= 0 && wIdxCap >= 0 {
				nnueAccCopySubSubAdd256(&acc.White[0], &parent.White[0],
					&net.InputWeights[wIdxMoveOld][0], &net.InputWeights[wIdxMoveNew][0],
					&net.InputWeights[wIdxCap][0])
			} else {
				acc.White = parent.White
			}
			bIdxMoveOld := HalfKAIndex(Black, bKingSq, d.Piece, d.From)
			bIdxMoveNew := HalfKAIndex(Black, bKingSq, d.Piece, d.To)
			bIdxCap := HalfKAIndex(Black, bKingSq, d.CapPiece, d.CapSq)
			if bIdxMoveOld >= 0 && bIdxMoveNew >= 0 && bIdxCap >= 0 {
				nnueAccCopySubSubAdd256(&acc.Black[0], &parent.Black[0],
					&net.InputWeights[bIdxMoveOld][0], &net.InputWeights[bIdxMoveNew][0],
					&net.InputWeights[bIdxCap][0])
			} else {
				acc.Black = parent.Black
			}
		} else {
			acc.White = parent.White
			acc.Black = parent.Black
			net.SubSubAddFeature(acc, d.Piece, d.From, d.To, d.CapPiece, d.CapSq, wKingSq, bKingSq)
		}
	case 4: // promotion (no capture) — rare, use copy+update
		acc.White = parent.White
		acc.Black = parent.Black
		net.RemoveFeature(acc, d.Piece, d.From, wKingSq, bKingSq)
		net.AddFeature(acc, d.PromoPc, d.To, wKingSq, bKingSq)
	case 5: // capture-promotion — rare, use copy+update
		acc.White = parent.White
		acc.Black = parent.Black
		net.RemoveFeature(acc, d.CapPiece, d.To, wKingSq, bKingSq)
		net.RemoveFeature(acc, d.Piece, d.From, wKingSq, bKingSq)
		net.AddFeature(acc, d.PromoPc, d.To, wKingSq, bKingSq)
	case 7: // castling, same bucket — rare, use copy+update
		acc.White = parent.White
		acc.Black = parent.Black
		net.SubAddFeature(acc, d.Piece, d.From, d.To, wKingSq, bKingSq)
		rookPiece := Piece(WhiteRook)
		if d.Piece == BlackKing {
			rookPiece = BlackRook
		}
		net.SubAddFeature(acc, rookPiece, d.RookFrom, d.RookTo, wKingSq, bKingSq)
	}

	acc.Computed = true
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

// nnuePieceIndexTable maps Piece values (0-12) to HalfKA piece type indices.
// -1 for Empty (0). Kings ARE included (indices 5 and 11).
var nnuePieceIndexTable = [13]int{
	-1, // Empty
	0,  // WhitePawn
	1,  // WhiteKnight
	2,  // WhiteBishop
	3,  // WhiteRook
	4,  // WhiteQueen
	5,  // WhiteKing
	6,  // BlackPawn
	7,  // BlackKnight
	8,  // BlackBishop
	9,  // BlackRook
	10, // BlackQueen
	11, // BlackKing
}

// nnuePieceIndex maps a Piece (1-12) to a HalfKA piece type index (0-11).
// Returns -1 for Empty.
func nnuePieceIndex(p Piece) int {
	return nnuePieceIndexTable[p]
}

// HalfKAIndex computes the feature index for a given perspective.
// perspective: White (0) or Black (1)
// kingSq: king square from that perspective (used for bucket + mirroring)
// piece: the piece (1-12, including kings)
// pieceSq: the square the piece is on
//
// For Black perspective, squares are mirrored vertically (^56) and piece colors are swapped.
// When the king is on files e-h, piece squares are mirrored horizontally for symmetry.
// Feature = bucket * 768 + pieceType * 64 + pieceSq
func HalfKAIndex(perspective Color, kingSq Square, piece Piece, pieceSq Square) int {
	pi := nnuePieceIndexTable[piece]
	if pi < 0 {
		return -1 // empty — not a feature
	}

	ks := int(kingSq)
	ps := int(pieceSq)

	if perspective == Black {
		// Mirror squares vertically
		ks ^= 56
		ps ^= 56
		// Swap piece colors: 0-5 <-> 6-11
		if pi >= 6 {
			pi -= 6
		} else {
			pi += 6
		}
	}

	// Get bucket and check if file mirroring is needed
	bucket := kingBucketTable[ks]
	if kingBucketMirrorFile[ks] {
		// Mirror piece square horizontally (file a↔h, b↔g, c↔f, d↔e)
		ps = (ps & ^7) | (7 - (ps & 7))
	}

	return bucket*(nnueNumPieceTypes*64) + pi*64 + ps
}

// AddFeature adds a feature (piece at square) to the accumulator for both perspectives.
func (net *NNUENet) AddFeature(acc *NNUEAccumulator, piece Piece, sq Square, wKingSq, bKingSq Square) {
	wIdx := HalfKAIndex(White, wKingSq, piece, sq)
	bIdx := HalfKAIndex(Black, bKingSq, piece, sq)

	if wIdx >= 0 {
		if nnueUseSIMD {
			nnueAccAdd256(&acc.White[0], &net.InputWeights[wIdx][0])
		} else {
			for i := 0; i < NNUEHiddenSize; i++ {
				acc.White[i] += net.InputWeights[wIdx][i]
			}
		}
	}
	if bIdx >= 0 {
		if nnueUseSIMD {
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
	wIdx := HalfKAIndex(White, wKingSq, piece, sq)
	bIdx := HalfKAIndex(Black, bKingSq, piece, sq)

	if wIdx >= 0 {
		if nnueUseSIMD {
			nnueAccSub256(&acc.White[0], &net.InputWeights[wIdx][0])
		} else {
			for i := 0; i < NNUEHiddenSize; i++ {
				acc.White[i] -= net.InputWeights[wIdx][i]
			}
		}
	}
	if bIdx >= 0 {
		if nnueUseSIMD {
			nnueAccSub256(&acc.Black[0], &net.InputWeights[bIdx][0])
		} else {
			for i := 0; i < NNUEHiddenSize; i++ {
				acc.Black[i] -= net.InputWeights[bIdx][i]
			}
		}
	}
}

// SubAddFeature combines RemoveFeature(oldSq) + AddFeature(newSq) for the same piece
// in a single pass over the accumulator, halving the number of accumulator writes.
func (net *NNUENet) SubAddFeature(acc *NNUEAccumulator, piece Piece, fromSq, toSq Square, wKingSq, bKingSq Square) {
	wIdxOld := HalfKAIndex(White, wKingSq, piece, fromSq)
	wIdxNew := HalfKAIndex(White, wKingSq, piece, toSq)
	if wIdxOld >= 0 && wIdxNew >= 0 {
		if nnueUseSIMD {
			nnueAccSubAdd256(&acc.White[0], &net.InputWeights[wIdxOld][0], &net.InputWeights[wIdxNew][0])
		} else {
			for i := 0; i < NNUEHiddenSize; i++ {
				acc.White[i] += net.InputWeights[wIdxNew][i] - net.InputWeights[wIdxOld][i]
			}
		}
	}
	bIdxOld := HalfKAIndex(Black, bKingSq, piece, fromSq)
	bIdxNew := HalfKAIndex(Black, bKingSq, piece, toSq)
	if bIdxOld >= 0 && bIdxNew >= 0 {
		if nnueUseSIMD {
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
	wIdxMoveOld := HalfKAIndex(White, wKingSq, movePiece, fromSq)
	wIdxMoveNew := HalfKAIndex(White, wKingSq, movePiece, toSq)
	wIdxCap := HalfKAIndex(White, wKingSq, capPiece, capSq)
	if wIdxMoveOld >= 0 && wIdxMoveNew >= 0 && wIdxCap >= 0 {
		if nnueUseSIMD {
			nnueAccSubSubAdd256(&acc.White[0], &net.InputWeights[wIdxMoveOld][0], &net.InputWeights[wIdxMoveNew][0], &net.InputWeights[wIdxCap][0])
		} else {
			for i := 0; i < NNUEHiddenSize; i++ {
				acc.White[i] += net.InputWeights[wIdxMoveNew][i] - net.InputWeights[wIdxMoveOld][i] - net.InputWeights[wIdxCap][i]
			}
		}
	}
	// Black perspective
	bIdxMoveOld := HalfKAIndex(Black, bKingSq, movePiece, fromSq)
	bIdxMoveNew := HalfKAIndex(Black, bKingSq, movePiece, toSq)
	bIdxCap := HalfKAIndex(Black, bKingSq, capPiece, capSq)
	if bIdxMoveOld >= 0 && bIdxMoveNew >= 0 && bIdxCap >= 0 {
		if nnueUseSIMD {
			nnueAccSubSubAdd256(&acc.Black[0], &net.InputWeights[bIdxMoveOld][0], &net.InputWeights[bIdxMoveNew][0], &net.InputWeights[bIdxCap][0])
		} else {
			for i := 0; i < NNUEHiddenSize; i++ {
				acc.Black[i] += net.InputWeights[bIdxMoveNew][i] - net.InputWeights[bIdxMoveOld][i] - net.InputWeights[bIdxCap][i]
			}
		}
	}
}

// RecomputeAccumulator performs a full recompute of the accumulator from the board state.
// Used after bucket-changing king moves, SetFEN, and Reset.
func (net *NNUENet) RecomputeAccumulator(acc *NNUEAccumulator, b *Board) {
	// Start with biases
	acc.White = net.InputBiases
	acc.Black = net.InputBiases

	wKingSq := b.Pieces[WhiteKing].LSB()
	bKingSq := b.Pieces[BlackKing].LSB()

	// Pre-gather all feature indices, then do a tight SIMD loop.
	// Separates index computation from memory-heavy SIMD work for better ILP.
	// Buffer is 64 to handle illegal FENs with >32 pieces without crashing.
	var wIndices [64]int
	var bIndices [64]int
	count := 0

	for piece := WhitePawn; piece <= BlackKing; piece++ {
		bb := b.Pieces[piece]
		for bb != 0 {
			sq := bb.PopLSB()
			wIndices[count] = HalfKAIndex(White, wKingSq, piece, sq)
			bIndices[count] = HalfKAIndex(Black, bKingSq, piece, sq)
			count++
		}
	}

	if nnueUseSIMD {
		for i := 0; i < count; i++ {
			if wIndices[i] >= 0 {
				nnueAccAdd256(&acc.White[0], &net.InputWeights[wIndices[i]][0])
			}
			if bIndices[i] >= 0 {
				nnueAccAdd256(&acc.Black[0], &net.InputWeights[bIndices[i]][0])
			}
		}
	} else {
		for i := 0; i < count; i++ {
			if wIdx := wIndices[i]; wIdx >= 0 {
				for j := 0; j < NNUEHiddenSize; j++ {
					acc.White[j] += net.InputWeights[wIdx][j]
				}
			}
			if bIdx := bIndices[i]; bIdx >= 0 {
				for j := 0; j < NNUEHiddenSize; j++ {
					acc.Black[j] += net.InputWeights[bIdx][j]
				}
			}
		}
	}

	acc.Computed = true
}


// Evaluate runs the NNUE forward pass and returns a score in centipawns
// relative to the side to move.
func (net *NNUENet) Evaluate(acc *NNUEAccumulator, sideToMove Color, pieceCount int) int {
	bucket := OutputBucket(pieceCount)
	// SIMD fast path
	if nnueUseSIMD && net.HWT != nil {
		return net.evaluateSIMD(acc, sideToMove, bucket)
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

	// Hidden layer 2: ReLU(hidden2) -> hidden3 (no upper clamp)
	var hidden3 [NNUEHidden3Size]int32
	hidden3 = net.Hidden2Biases
	for j := 0; j < NNUEHidden2Size; j++ {
		scaled := hidden2[j] >> 6 // / 64 (nnueHiddenScale)
		if scaled <= 0 {
			continue
		}
		for k := 0; k < NNUEHidden3Size; k++ {
			hidden3[k] += scaled * int32(net.Hidden2Weights[j][k])
		}
	}

	// Output layer: ReLU(hidden3) -> output (no upper clamp), using material bucket
	output := net.OutputBias[bucket]
	for k := 0; k < NNUEHidden3Size; k++ {
		scaled := hidden3[k] >> 6 // / 64 (nnueHiddenScale)
		if scaled <= 0 {
			continue
		}
		output += scaled * int32(net.OutputWeights[bucket][k])
	}

	// Scale output to centipawns
	return int(output) / nnueOutputScale * NNUEEvalScale
}

// PrepareWeights transposes hidden weights for SIMD forward pass.
// Must be called after loading or creating a network.
func (net *NNUENet) PrepareWeights() {
	if !nnueUseSIMD {
		return
	}
	// Transpose hidden1 weights: HWT[j][i] = HiddenWeights[i][j]
	inputDim := NNUEHiddenSize * 2 // 512
	outputDim := NNUEHidden2Size   // 32
	net.HWT = make([]int16, outputDim*inputDim)
	for i := 0; i < inputDim; i++ {
		for j := 0; j < outputDim; j++ {
			net.HWT[j*inputDim+i] = net.HiddenWeights[i][j]
		}
	}
	// Create int8 version of transposed hidden weights for SIMD int8 matmul
	net.HWT8 = make([]int8, outputDim*inputDim)
	clamped := 0
	for i := 0; i < len(net.HWT); i++ {
		v := net.HWT[i]
		if v < -128 {
			v = -128
			clamped++
		} else if v > 127 {
			v = 127
			clamped++
		}
		net.HWT8[i] = int8(v)
	}
	if clamped > 0 {
		fmt.Printf("info string warning: %d/%d hidden weights clamped to int8 range\n", clamped, len(net.HWT))
	}

	// Transpose hidden2 weights: Hidden2WT[k][j] = Hidden2Weights[j][k]
	net.Hidden2WT = make([]int16, NNUEHidden2Size*NNUEHidden3Size)
	for j := 0; j < NNUEHidden2Size; j++ {
		for k := 0; k < NNUEHidden3Size; k++ {
			net.Hidden2WT[k*NNUEHidden2Size+j] = net.Hidden2Weights[j][k]
		}
	}
}

// evaluateSIMD runs the NNUE forward pass using SIMD instructions (AVX2 or NEON).
func (net *NNUENet) evaluateSIMD(acc *NNUEAccumulator, sideToMove Color, bucket int) int {
	var stm, opp *[NNUEHiddenSize]int16
	if sideToMove == White {
		stm = &acc.White
		opp = &acc.Black
	} else {
		stm = &acc.Black
		opp = &acc.White
	}

	// Apply ClippedReLU and pack into uint8 buffer (half the memory of int16)
	var input8 [NNUEHiddenSize * 2]byte
	nnueCReLU256to8(&stm[0], &input8[0])
	nnueCReLU256to8(&opp[0], &input8[NNUEHiddenSize])

	// Hidden layer matrix multiply using int8 weights (2× throughput via VPMADDUBSW/SMULL)
	var hidden2 [NNUEHidden2Size]int32
	nnueMatMul32x512_i8(&input8[0], &net.HWT8[0], &net.HiddenBiases[0], &hidden2[0])

	// Hidden layer 2: SIMD 32×32 matmul with ReLU activation
	var hidden3 [NNUEHidden3Size]int32
	nnueMatMul32x32ReLU(&hidden2[0], &net.Hidden2WT[0], &net.Hidden2Biases[0], &hidden3[0])

	// Output layer: SIMD dot product with ReLU activation, using material bucket
	output := net.OutputBias[bucket] + nnueDotReLU32(&hidden3[0], &net.OutputWeights[bucket][0])

	return int(output) / nnueOutputScale * NNUEEvalScale
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

	// Hidden layer 1 weights and biases
	if err := binary.Write(w, binary.LittleEndian, &net.HiddenWeights); err != nil {
		return fmt.Errorf("writing hidden weights: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, &net.HiddenBiases); err != nil {
		return fmt.Errorf("writing hidden biases: %w", err)
	}

	// Hidden layer 2 weights and biases
	if err := binary.Write(w, binary.LittleEndian, &net.Hidden2Weights); err != nil {
		return fmt.Errorf("writing hidden2 weights: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, &net.Hidden2Biases); err != nil {
		return fmt.Errorf("writing hidden2 biases: %w", err)
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
// Fingerprint returns a simple checksum of the network weights for debugging.
// It hashes the first 256 input weights + output bias to produce a short hex string.
func (net *NNUENet) Fingerprint() string {
	var h uint64
	for i := 0; i < 256 && i < len(net.InputWeights[0]); i++ {
		h = h*31 + uint64(uint16(net.InputWeights[0][i]))
	}
	h = h*31 + uint64(uint32(net.OutputBias[0]))
	return fmt.Sprintf("%016x", h)
}

func LoadNNUE(path string) (*NNUENet, error) {
	return loadNNUEImpl(path, false)
}

// LoadNNUEAnyVersion loads an NNUE net from either v3 or v4 format.
// v3 nets have their single output replicated into all output buckets.
func LoadNNUEAnyVersion(path string) (*NNUENet, error) {
	return loadNNUEImpl(path, true)
}

func loadNNUEImpl(path string, allowV3 bool) (*NNUENet, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	net, err := readNNUEVersioned(f, allowV3)
	if err != nil {
		return nil, err
	}
	net.PrepareWeights()
	fmt.Fprintf(os.Stderr, "info string NNUE fingerprint %s from %s\n", net.Fingerprint(), path)
	return net, nil
}

func readNNUE(r io.Reader) (*NNUENet, error) {
	return readNNUEVersioned(r, false)
}

// readNNUEVersioned reads an NNUE net. If allowV3 is true, it accepts v3 (single output)
// nets and replicates the output weights into all output buckets.
func readNNUEVersioned(r io.Reader, allowV3 bool) (*NNUENet, error) {
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
	if version != nnueVersion && !(allowV3 && version == 3) {
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

	// Hidden layer 1 weights and biases
	if err := binary.Read(r, binary.LittleEndian, &net.HiddenWeights); err != nil {
		return nil, fmt.Errorf("reading hidden weights: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &net.HiddenBiases); err != nil {
		return nil, fmt.Errorf("reading hidden biases: %w", err)
	}

	// Hidden layer 2 weights and biases
	if err := binary.Read(r, binary.LittleEndian, &net.Hidden2Weights); err != nil {
		return nil, fmt.Errorf("reading hidden2 weights: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &net.Hidden2Biases); err != nil {
		return nil, fmt.Errorf("reading hidden2 biases: %w", err)
	}

	// Output weights and bias
	if version == 3 {
		// v3: single output layer — read then replicate into all buckets
		var singleWeights [NNUEHidden3Size]int16
		var singleBias int32
		if err := binary.Read(r, binary.LittleEndian, &singleWeights); err != nil {
			return nil, fmt.Errorf("reading v3 output weights: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, &singleBias); err != nil {
			return nil, fmt.Errorf("reading v3 output bias: %w", err)
		}
		for b := 0; b < NNUEOutputBuckets; b++ {
			net.OutputWeights[b] = singleWeights
			net.OutputBias[b] = singleBias
		}
	} else {
		if err := binary.Read(r, binary.LittleEndian, &net.OutputWeights); err != nil {
			return nil, fmt.Errorf("reading output weights: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, &net.OutputBias); err != nil {
			return nil, fmt.Errorf("reading output bias: %w", err)
		}
	}

	return net, nil
}

package chess

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestKingBuckets(t *testing.T) {
	// Verify all 64 squares map to valid buckets
	for sq := 0; sq < 64; sq++ {
		bucket := kingBucketTable[sq]
		if bucket < 0 || bucket >= NNUEKingBuckets {
			t.Errorf("square %d: bucket %d out of range [0, %d)", sq, bucket, NNUEKingBuckets)
		}
	}

	// Verify horizontal mirror symmetry: e1 and d1 should have the same bucket
	// because files a-d (0-3) and e-h (4-7) mirror
	for rank := 0; rank < 8; rank++ {
		for file := 0; file < 4; file++ {
			sqLeft := rank*8 + file
			sqRight := rank*8 + (7 - file)
			if kingBucketTable[sqLeft] != kingBucketTable[sqRight] {
				t.Errorf("mirror mismatch: sq %d (bucket %d) vs sq %d (bucket %d)",
					sqLeft, kingBucketTable[sqLeft], sqRight, kingBucketTable[sqRight])
			}
		}
	}

	// Verify 16 distinct buckets exist
	seen := make(map[int]bool)
	for sq := 0; sq < 64; sq++ {
		seen[kingBucketTable[sq]] = true
	}
	if len(seen) != NNUEKingBuckets {
		t.Errorf("expected %d distinct buckets, got %d", NNUEKingBuckets, len(seen))
	}
}

func TestHalfKAIndex(t *testing.T) {
	// Basic index range test — all pieces including kings
	for ks := Square(0); ks < 64; ks++ {
		for piece := WhitePawn; piece <= BlackKing; piece++ {
			for ps := Square(0); ps < 64; ps++ {
				idx := HalfKAIndex(White, ks, piece, ps)
				if idx < 0 || idx >= NNUEInputSize {
					t.Errorf("HalfKAIndex(White, %d, %d, %d) = %d out of range [0, %d)", ks, piece, ps, idx, NNUEInputSize)
				}
				idx = HalfKAIndex(Black, ks, piece, ps)
				if idx < 0 || idx >= NNUEInputSize {
					t.Errorf("HalfKAIndex(Black, %d, %d, %d) = %d out of range [0, %d)", ks, piece, ps, idx, NNUEInputSize)
				}
			}
		}
	}
}

func TestHalfKAIndexEmptyReturnsNegative(t *testing.T) {
	idx := HalfKAIndex(White, 4, Empty, 4)
	if idx != -1 {
		t.Errorf("expected -1 for empty, got %d", idx)
	}
}

func TestHalfKAIndexKingsIncluded(t *testing.T) {
	// Kings should return valid indices (unlike HalfKP)
	idx := HalfKAIndex(White, 4, WhiteKing, 4)
	if idx < 0 {
		t.Errorf("expected valid index for WhiteKing, got %d", idx)
	}
	idx = HalfKAIndex(Black, 60, BlackKing, 60)
	if idx < 0 {
		t.Errorf("expected valid index for BlackKing, got %d", idx)
	}
}

func TestHalfKAIndexUniqueness(t *testing.T) {
	// All indices for a given perspective+kingSq should be unique
	seen := make(map[int]bool)
	kingSq := Square(4) // e1
	for piece := WhitePawn; piece <= BlackKing; piece++ {
		for ps := Square(0); ps < 64; ps++ {
			idx := HalfKAIndex(White, kingSq, piece, ps)
			if idx >= 0 {
				if seen[idx] {
					t.Errorf("duplicate index %d for piece=%d sq=%d", idx, piece, ps)
				}
				seen[idx] = true
			}
		}
	}
}

func TestHalfKASymmetry(t *testing.T) {
	// A White pawn on e2 from White's perspective with king on e1
	// should map to the same index as a Black pawn on e7 from Black's perspective with king on e8
	// (after mirroring: squares ^56, colors swapped)

	wIdx := HalfKAIndex(White, NewSquare(4, 0), WhitePawn, NewSquare(4, 1))
	bIdx := HalfKAIndex(Black, NewSquare(4, 7), BlackPawn, NewSquare(4, 6))

	if wIdx != bIdx {
		t.Errorf("symmetry broken: White index=%d, Black index=%d", wIdx, bIdx)
	}

	// Also test king feature symmetry
	wIdx = HalfKAIndex(White, NewSquare(4, 0), WhiteKing, NewSquare(4, 0))
	bIdx = HalfKAIndex(Black, NewSquare(4, 7), BlackKing, NewSquare(4, 7))
	if wIdx != bIdx {
		t.Errorf("king symmetry broken: White index=%d, Black index=%d", wIdx, bIdx)
	}
}

func TestHalfKAFileMirroring(t *testing.T) {
	// When king is on e1 (file 4, mirrored file 3), piece squares should be mirrored
	// vs when king is on d1 (file 3, no mirror)
	// King on d1 bucket == king on e1 bucket (both file 3 after mirror)
	d1 := NewSquare(3, 0) // file 3
	e1 := NewSquare(4, 0) // file 4, mirrored to 3

	if KingBucket(d1) != KingBucket(e1) {
		t.Fatalf("d1 and e1 should have same bucket, got %d and %d", KingBucket(d1), KingBucket(e1))
	}

	// A pawn on a2 with king on d1 should map to same index as pawn on h2 with king on e1
	// because both mirror to the same effective position
	idx1 := HalfKAIndex(White, d1, WhitePawn, NewSquare(0, 1)) // a2
	idx2 := HalfKAIndex(White, e1, WhitePawn, NewSquare(7, 1)) // h2 (mirrored to a2)
	if idx1 != idx2 {
		t.Errorf("file mirror not working: d1+a2=%d, e1+h2=%d", idx1, idx2)
	}
}

func TestNNUEAccumulatorStack(t *testing.T) {
	stack := NewNNUEAccumulatorStack(8)

	// Initial state
	if stack.Current().Computed {
		t.Error("initial accumulator should not be computed")
	}

	// Mark as computed and set a value
	stack.Current().Computed = true
	stack.Current().White[0] = 42

	// Push is now lazy — does NOT copy, marks as not computed
	stack.Push()
	if stack.Current().Computed {
		t.Error("lazy-pushed accumulator should not be computed")
	}

	// Pop should restore parent
	stack.Pop()
	if stack.Current().White[0] != 42 {
		t.Errorf("after pop expected 42, got %d", stack.Current().White[0])
	}
	if !stack.Current().Computed {
		t.Error("parent should still be computed after pop")
	}
}

func TestNNUEAccumulatorStackDeepCopy(t *testing.T) {
	stack := NewNNUEAccumulatorStack(8)
	stack.Current().White[0] = 100
	stack.Current().Computed = true
	stack.Push()
	stack.Current().White[0] = 200
	stack.Current().Computed = true // manually set for this test

	cp := stack.DeepCopy()
	if cp.Current().White[0] != 200 {
		t.Errorf("deep copy should have 200, got %d", cp.Current().White[0])
	}

	// Modifying original should not affect copy
	stack.Current().White[0] = 999
	if cp.Current().White[0] != 200 {
		t.Errorf("deep copy should be independent, got %d", cp.Current().White[0])
	}
}

func makeTestNet() *NNUENet {
	net := &NNUENet{}
	// Set small known weights for testing
	for i := range net.InputBiases {
		net.InputBiases[i] = 1
	}
	// Set a few input weights to known values
	// Feature for White pawn on e2, king on e1
	idx := HalfKAIndex(White, NewSquare(4, 0), WhitePawn, NewSquare(4, 1))
	if idx >= 0 {
		for i := 0; i < NNUEHiddenSize; i++ {
			net.InputWeights[idx][i] = int16(i % 10)
		}
	}
	// Set hidden and output layers to simple values
	for i := 0; i < NNUEHiddenSize*2; i++ {
		for j := 0; j < NNUEHidden2Size; j++ {
			net.HiddenWeights[i][j] = 1
		}
	}
	for j := 0; j < NNUEHidden2Size; j++ {
		net.HiddenBiases[j] = 0
	}
	for b := 0; b < NNUEOutputBuckets; b++ {
		for j := 0; j < NNUEHidden3Size; j++ {
			net.OutputWeights[b][j] = 1
		}
	}
	return net
}

func TestNNUEForwardPass(t *testing.T) {
	net := makeTestNet()
	acc := &NNUEAccumulator{}

	// Start with biases
	acc.White = net.InputBiases
	acc.Black = net.InputBiases
	acc.Computed = true

	// Evaluate — should produce a non-zero score from the biases alone
	score := net.Evaluate(acc, White, 32)

	// With all biases = 1, ClippedReLU(1) = 1
	// hidden2[j] = 0 + 256*1*1 + 256*1*1 = 512 for each j
	// scaled = 512/64 = 8, clamped to [0,127] -> 8
	// output = 0 + 32*8*1 = 256
	// score = 256 / (127*64) = 256/8128 = 0
	// (integer division truncates)
	expected := 256 / nnueOutputScale
	if score != expected {
		t.Errorf("expected score %d from bias-only accumulator, got %d", expected, score)
	}
}

func TestNNUEAddRemoveFeature(t *testing.T) {
	net := makeTestNet()
	acc := &NNUEAccumulator{}
	acc.White = net.InputBiases
	acc.Black = net.InputBiases
	acc.Computed = true

	// Save the initial accumulator
	var origWhite [NNUEHiddenSize]int16
	copy(origWhite[:], acc.White[:])

	wKingSq := NewSquare(4, 0)
	bKingSq := NewSquare(4, 7)

	// Add a feature
	net.AddFeature(acc, WhitePawn, NewSquare(4, 1), wKingSq, bKingSq)

	// Accumulator should have changed
	changed := false
	for i := 0; i < NNUEHiddenSize; i++ {
		if acc.White[i] != origWhite[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Error("AddFeature did not modify accumulator")
	}

	// Remove the same feature
	net.RemoveFeature(acc, WhitePawn, NewSquare(4, 1), wKingSq, bKingSq)

	// Should be back to original
	for i := 0; i < NNUEHiddenSize; i++ {
		if acc.White[i] != origWhite[i] {
			t.Errorf("after remove, White[%d] = %d, expected %d", i, acc.White[i], origWhite[i])
			break
		}
	}
}

func TestNNUERecomputeAccumulator(t *testing.T) {
	net := makeTestNet()

	var b Board
	b.Reset()

	acc1 := &NNUEAccumulator{}
	net.RecomputeAccumulator(acc1, &b)

	if !acc1.Computed {
		t.Error("recompute should set Computed=true")
	}

	// Do it again — should get the same result
	acc2 := &NNUEAccumulator{}
	net.RecomputeAccumulator(acc2, &b)

	for i := 0; i < NNUEHiddenSize; i++ {
		if acc1.White[i] != acc2.White[i] {
			t.Errorf("White[%d] mismatch: %d vs %d", i, acc1.White[i], acc2.White[i])
			break
		}
		if acc1.Black[i] != acc2.Black[i] {
			t.Errorf("Black[%d] mismatch: %d vs %d", i, acc1.Black[i], acc2.Black[i])
			break
		}
	}
}

func TestNNUEIncrementalVsFullRecompute(t *testing.T) {
	net := makeTestNet()

	// Set up a more interesting net with some non-trivial weights
	for i := 0; i < NNUEInputSize; i++ {
		for j := 0; j < NNUEHiddenSize; j++ {
			net.InputWeights[i][j] = int16((i*7 + j*13) % 127)
		}
	}

	var b Board
	b.Reset()

	// Full recompute
	accFull := &NNUEAccumulator{}
	net.RecomputeAccumulator(accFull, &b)

	// Incremental: start with biases, add each piece (including kings for HalfKA)
	accInc := &NNUEAccumulator{}
	accInc.White = net.InputBiases
	accInc.Black = net.InputBiases
	accInc.Computed = true

	wKingSq := b.Pieces[WhiteKing].LSB()
	bKingSq := b.Pieces[BlackKing].LSB()

	for piece := WhitePawn; piece <= BlackKing; piece++ {
		bb := b.Pieces[piece]
		for bb != 0 {
			sq := bb.PopLSB()
			net.AddFeature(accInc, piece, sq, wKingSq, bKingSq)
		}
	}

	// Compare
	for i := 0; i < NNUEHiddenSize; i++ {
		if accFull.White[i] != accInc.White[i] {
			t.Errorf("White[%d] full=%d inc=%d", i, accFull.White[i], accInc.White[i])
			break
		}
		if accFull.Black[i] != accInc.Black[i] {
			t.Errorf("Black[%d] full=%d inc=%d", i, accFull.Black[i], accInc.Black[i])
			break
		}
	}
}

func TestNNUESaveLoad(t *testing.T) {
	net := makeTestNet()

	dir := t.TempDir()
	path := filepath.Join(dir, "test.nnue")

	if err := SaveNNUE(path, net); err != nil {
		t.Fatalf("SaveNNUE failed: %v", err)
	}

	// Check file was created
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("file is empty")
	}

	loaded, err := LoadNNUE(path)
	if err != nil {
		t.Fatalf("LoadNNUE failed: %v", err)
	}

	// Compare biases
	for i := 0; i < NNUEHiddenSize; i++ {
		if net.InputBiases[i] != loaded.InputBiases[i] {
			t.Errorf("InputBiases[%d] mismatch: %d vs %d", i, net.InputBiases[i], loaded.InputBiases[i])
			break
		}
	}

	// Compare a sample of input weights
	idx := HalfKAIndex(White, NewSquare(4, 0), WhitePawn, NewSquare(4, 1))
	if idx >= 0 {
		for i := 0; i < NNUEHiddenSize; i++ {
			if net.InputWeights[idx][i] != loaded.InputWeights[idx][i] {
				t.Errorf("InputWeights[%d][%d] mismatch: %d vs %d", idx, i, net.InputWeights[idx][i], loaded.InputWeights[idx][i])
				break
			}
		}
	}

	// Compare output
	for b := 0; b < NNUEOutputBuckets; b++ {
		if net.OutputBias[b] != loaded.OutputBias[b] {
			t.Errorf("OutputBias[%d] mismatch: %d vs %d", b, net.OutputBias[b], loaded.OutputBias[b])
		}
		for j := 0; j < NNUEHidden3Size; j++ {
			if net.OutputWeights[b][j] != loaded.OutputWeights[b][j] {
				t.Errorf("OutputWeights[%d][%d] mismatch: %d vs %d", b, j, net.OutputWeights[b][j], loaded.OutputWeights[b][j])
				break
			}
		}
	}
}

func TestNNUEEvaluateSymmetry(t *testing.T) {
	// Create a symmetric net — same weights for both perspectives
	net := &NNUENet{}
	for i := range net.InputBiases {
		net.InputBiases[i] = 10
	}
	for i := 0; i < NNUEInputSize; i++ {
		for j := 0; j < NNUEHiddenSize; j++ {
			net.InputWeights[i][j] = int16((i*3 + j*5) % 50)
		}
	}
	for i := 0; i < NNUEHiddenSize*2; i++ {
		for j := 0; j < NNUEHidden2Size; j++ {
			net.HiddenWeights[i][j] = 1
		}
	}
	for bk := 0; bk < NNUEOutputBuckets; bk++ {
		for j := 0; j < NNUEHidden3Size; j++ {
			net.OutputWeights[bk][j] = 1
		}
	}

	var b Board
	b.Reset()

	acc := &NNUEAccumulator{}
	net.RecomputeAccumulator(acc, &b)

	// In the starting position, White's evaluation and Black's evaluation
	// should be symmetric (same magnitude, side-to-move adjusted)
	scoreW := net.Evaluate(acc, White, 32)
	scoreB := net.Evaluate(acc, Black, 32)

	// Starting position is perfectly symmetric, so both perspectives
	// should give the same score
	if scoreW != scoreB {
		t.Errorf("symmetric position: White score=%d, Black score=%d (should be equal)", scoreW, scoreB)
	}
}

func TestNNUEIncrementalMakeUnmake(t *testing.T) {
	// Create a net with varied weights
	net := &NNUENet{}
	for i := range net.InputBiases {
		net.InputBiases[i] = 10
	}
	for i := 0; i < NNUEInputSize; i++ {
		for j := 0; j < NNUEHiddenSize; j++ {
			net.InputWeights[i][j] = int16((i*7 + j*13) % 100)
		}
	}

	var b Board
	b.NNUENet = net
	b.NNUEAcc = NewNNUEAccumulatorStack(256)
	b.Reset()

	// Get reference accumulator from full recompute
	refAcc := &NNUEAccumulator{}
	net.RecomputeAccumulator(refAcc, &b)

	// Verify the board's accumulator matches the reference after Reset
	acc := b.NNUEAcc.Current()
	for i := 0; i < NNUEHiddenSize; i++ {
		if acc.White[i] != refAcc.White[i] {
			t.Fatalf("after Reset: White[%d] inc=%d full=%d", i, acc.White[i], refAcc.White[i])
		}
		if acc.Black[i] != refAcc.Black[i] {
			t.Fatalf("after Reset: Black[%d] inc=%d full=%d", i, acc.Black[i], refAcc.Black[i])
		}
	}

	// Make a few moves and verify incremental == full recompute at each step
	moves := b.GenerateLegalMoves()
	for mi, m := range moves {
		if mi >= 5 {
			break
		}
		b.MakeMove(m)

		// Materialize the lazy accumulator before comparing
		b.NNUEAcc.Materialize(net, &b)

		// Full recompute for comparison
		fullAcc := &NNUEAccumulator{}
		net.RecomputeAccumulator(fullAcc, &b)

		incAcc := b.NNUEAcc.Current()
		for i := 0; i < NNUEHiddenSize; i++ {
			if incAcc.White[i] != fullAcc.White[i] {
				t.Fatalf("move %d (%s): White[%d] inc=%d full=%d", mi, m.String(), i, incAcc.White[i], fullAcc.White[i])
			}
			if incAcc.Black[i] != fullAcc.Black[i] {
				t.Fatalf("move %d (%s): Black[%d] inc=%d full=%d", mi, m.String(), i, incAcc.Black[i], fullAcc.Black[i])
			}
		}

		b.UnmakeMove(m)
	}

	// After unmake, should be back to original
	acc = b.NNUEAcc.Current()
	for i := 0; i < NNUEHiddenSize; i++ {
		if acc.White[i] != refAcc.White[i] {
			t.Fatalf("after unmake: White[%d] inc=%d full=%d", i, acc.White[i], refAcc.White[i])
		}
		if acc.Black[i] != refAcc.Black[i] {
			t.Fatalf("after unmake: Black[%d] inc=%d full=%d", i, acc.Black[i], refAcc.Black[i])
		}
	}
}

func TestNNUEIncrementalKingMove(t *testing.T) {
	// Position where a king can move: after 1.e4 e5 2.Ke2
	net := &NNUENet{}
	for i := range net.InputBiases {
		net.InputBiases[i] = 5
	}
	for i := 0; i < NNUEInputSize; i++ {
		for j := 0; j < NNUEHiddenSize; j++ {
			net.InputWeights[i][j] = int16((i*3 + j*7) % 80)
		}
	}

	var b Board
	b.NNUENet = net
	b.NNUEAcc = NewNNUEAccumulatorStack(256)
	b.SetFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")

	// Make e7e5
	m1, _ := b.ParseUCIMove("e7e5")
	b.MakeMove(m1)

	// Make king move Ke2
	m2, _ := b.ParseUCIMove("e1e2")
	b.MakeMove(m2)

	// King moved — materialize and verify against full recompute
	b.NNUEAcc.Materialize(net, &b)
	fullAcc := &NNUEAccumulator{}
	net.RecomputeAccumulator(fullAcc, &b)

	incAcc := b.NNUEAcc.Current()
	for i := 0; i < NNUEHiddenSize; i++ {
		if incAcc.White[i] != fullAcc.White[i] {
			t.Fatalf("after king move: White[%d] inc=%d full=%d", i, incAcc.White[i], fullAcc.White[i])
		}
		if incAcc.Black[i] != fullAcc.Black[i] {
			t.Fatalf("after king move: Black[%d] inc=%d full=%d", i, incAcc.Black[i], fullAcc.Black[i])
		}
	}

	// Unmake both moves
	b.UnmakeMove(m2)
	b.UnmakeMove(m1)

	// Should match original position
	origAcc := &NNUEAccumulator{}
	net.RecomputeAccumulator(origAcc, &b)
	incAcc = b.NNUEAcc.Current()
	for i := 0; i < NNUEHiddenSize; i++ {
		if incAcc.White[i] != origAcc.White[i] {
			t.Fatalf("after unmake king: White[%d] inc=%d full=%d", i, incAcc.White[i], origAcc.White[i])
		}
	}
}

func TestNNUEPieceIndex(t *testing.T) {
	tests := []struct {
		piece    Piece
		expected int
	}{
		{WhitePawn, 0}, {WhiteKnight, 1}, {WhiteBishop, 2},
		{WhiteRook, 3}, {WhiteQueen, 4}, {WhiteKing, 5},
		{BlackPawn, 6}, {BlackKnight, 7}, {BlackBishop, 8},
		{BlackRook, 9}, {BlackQueen, 10}, {BlackKing, 11},
		{Empty, -1},
	}
	for _, tt := range tests {
		got := nnuePieceIndex(tt.piece)
		if got != tt.expected {
			t.Errorf("nnuePieceIndex(%d) = %d, expected %d", tt.piece, got, tt.expected)
		}
	}
}

func TestNNUESIMDvsGeneric(t *testing.T) {
	if !nnueUseSIMD {
		t.Skip("SIMD not available")
	}

	// Create a net with varied weights
	net := &NNUENet{}
	for i := range net.InputBiases {
		net.InputBiases[i] = int16(i%50 - 25)
	}
	for i := 0; i < NNUEInputSize; i++ {
		for j := 0; j < NNUEHiddenSize; j++ {
			net.InputWeights[i][j] = int16((i*7 + j*13) % 200 - 100)
		}
	}
	for i := 0; i < NNUEHiddenSize*2; i++ {
		for j := 0; j < NNUEHidden2Size; j++ {
			net.HiddenWeights[i][j] = int16((i*3 + j*11) % 120 - 60)
		}
	}
	for j := 0; j < NNUEHidden2Size; j++ {
		net.HiddenBiases[j] = int32(j*100 - 1600)
	}
	for b := 0; b < NNUEOutputBuckets; b++ {
		for j := 0; j < NNUEHidden3Size; j++ {
			net.OutputWeights[b][j] = int16(j*5 - 80)
		}
		net.OutputBias[b] = 42
	}
	net.PrepareWeights()

	var board Board
	board.Reset()
	acc := &NNUEAccumulator{}
	net.RecomputeAccumulator(acc, &board)

	// Force generic path
	origFlag := nnueUseSIMD
	nnueUseSIMD = false
	genericScore := net.Evaluate(acc, White, 32)

	// Force SIMD path
	nnueUseSIMD = true
	simdScore := net.Evaluate(acc, White, 32)

	nnueUseSIMD = origFlag

	if genericScore != simdScore {
		t.Errorf("SIMD vs Generic mismatch: simd=%d generic=%d", simdScore, genericScore)
	}
	t.Logf("SIMD=%d Generic=%d (match)", simdScore, genericScore)

	// Also test Black perspective
	nnueUseSIMD = false
	genericB := net.Evaluate(acc, Black, 32)
	nnueUseSIMD = true
	simdB := net.Evaluate(acc, Black, 32)
	nnueUseSIMD = origFlag

	if genericB != simdB {
		t.Errorf("Black: SIMD vs Generic mismatch: simd=%d generic=%d", simdB, genericB)
	}
}

func BenchmarkNNUEForwardGeneric(b *testing.B) {
	net := &NNUENet{}
	for i := range net.InputBiases {
		net.InputBiases[i] = int16(i%50 - 25)
	}
	for i := 0; i < NNUEInputSize; i++ {
		for j := 0; j < NNUEHiddenSize; j++ {
			net.InputWeights[i][j] = int16((i*7 + j*13) % 200 - 100)
		}
	}
	for i := 0; i < NNUEHiddenSize*2; i++ {
		for j := 0; j < NNUEHidden2Size; j++ {
			net.HiddenWeights[i][j] = int16((i*3 + j*11) % 120 - 60)
		}
	}
	for j := 0; j < NNUEHidden2Size; j++ {
		net.HiddenBiases[j] = int32(j*100 - 1600)
	}
	for bk := 0; bk < NNUEOutputBuckets; bk++ {
		for j := 0; j < NNUEHidden3Size; j++ {
			net.OutputWeights[bk][j] = int16(j*5 - 80)
		}
		net.OutputBias[bk] = 42
	}

	var board Board
	board.Reset()
	acc := &NNUEAccumulator{}
	net.RecomputeAccumulator(acc, &board)

	origFlag := nnueUseSIMD
	nnueUseSIMD = false
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		net.Evaluate(acc, White, 32)
	}
	nnueUseSIMD = origFlag
}

func BenchmarkNNUEForwardSIMD(b *testing.B) {
	if !nnueUseSIMD {
		b.Skip("SIMD not available")
	}

	net := &NNUENet{}
	for i := range net.InputBiases {
		net.InputBiases[i] = int16(i%50 - 25)
	}
	for i := 0; i < NNUEInputSize; i++ {
		for j := 0; j < NNUEHiddenSize; j++ {
			net.InputWeights[i][j] = int16((i*7 + j*13) % 200 - 100)
		}
	}
	for i := 0; i < NNUEHiddenSize*2; i++ {
		for j := 0; j < NNUEHidden2Size; j++ {
			net.HiddenWeights[i][j] = int16((i*3 + j*11) % 120 - 60)
		}
	}
	for j := 0; j < NNUEHidden2Size; j++ {
		net.HiddenBiases[j] = int32(j*100 - 1600)
	}
	for bk := 0; bk < NNUEOutputBuckets; bk++ {
		for j := 0; j < NNUEHidden3Size; j++ {
			net.OutputWeights[bk][j] = int16(j*5 - 80)
		}
		net.OutputBias[bk] = 42
	}
	net.PrepareWeights()

	var board Board
	board.Reset()
	acc := &NNUEAccumulator{}
	net.RecomputeAccumulator(acc, &board)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		net.Evaluate(acc, White, 32)
	}
}

// compareAccumulators checks that two accumulators match and reports
// mismatches via t.Fatalf with the given context message.
func compareAccumulators(t *testing.T, inc, full *NNUEAccumulator, context string) {
	t.Helper()
	for i := 0; i < NNUEHiddenSize; i++ {
		if inc.White[i] != full.White[i] {
			t.Fatalf("%s: White[%d] inc=%d full=%d", context, i, inc.White[i], full.White[i])
		}
		if inc.Black[i] != full.Black[i] {
			t.Fatalf("%s: Black[%d] inc=%d full=%d", context, i, inc.Black[i], full.Black[i])
		}
	}
}

// TestNNUEIncrementalEdgeCases tests NNUE accumulator consistency across
// castling, en passant, promotions, captures, null moves, and long
// move sequences. Uses synthetic weights (real net.nnue is HalfKP v2).
func TestNNUEIncrementalEdgeCases(t *testing.T) {
	// Use synthetic weights for HalfKA
	net := &NNUENet{}
	for i := range net.InputBiases {
		net.InputBiases[i] = 5
	}
	for i := 0; i < NNUEInputSize; i++ {
		for j := 0; j < NNUEHiddenSize; j++ {
			net.InputWeights[i][j] = int16((i*7 + j*13) % 100)
		}
	}

	// Helper: set up board with NNUE from a FEN, play a sequence of UCI
	// moves, checking incremental vs full recompute after each move, then
	// unmake all and verify restoration.
	testSequence := func(t *testing.T, fen string, uciMoves []string) {
		t.Helper()
		var b Board
		b.NNUENet = net
		b.NNUEAcc = NewNNUEAccumulatorStack(256)
		b.SetFEN(fen)

		// Verify initial state
		initAcc := &NNUEAccumulator{}
		net.RecomputeAccumulator(initAcc, &b)
		compareAccumulators(t, b.NNUEAcc.Current(), initAcc, "initial position")

		// Play each move
		played := make([]Move, 0, len(uciMoves))
		for i, uci := range uciMoves {
			m, err := b.ParseUCIMove(uci)
			if err != nil {
				t.Fatalf("move %d (%s): parse error: %v", i, uci, err)
			}
			b.MakeMove(m)
			played = append(played, m)

			b.NNUEAcc.Materialize(net, &b)
			fullAcc := &NNUEAccumulator{}
			net.RecomputeAccumulator(fullAcc, &b)
			compareAccumulators(t, b.NNUEAcc.Current(), fullAcc,
				fmt.Sprintf("after move %d (%s)", i, uci))
		}

		// Unmake all moves in reverse and verify at each step
		for i := len(played) - 1; i >= 0; i-- {
			b.UnmakeMove(played[i])
			fullAcc := &NNUEAccumulator{}
			net.RecomputeAccumulator(fullAcc, &b)
			compareAccumulators(t, b.NNUEAcc.Current(), fullAcc,
				fmt.Sprintf("after unmake move %d (%s)", i, uciMoves[i]))
		}

		// Final state should match initial
		compareAccumulators(t, b.NNUEAcc.Current(), initAcc, "back to initial position")
	}

	t.Run("WhiteKingsideCastle", func(t *testing.T) {
		// 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.O-O
		testSequence(t,
			"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			[]string{"e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "e1g1"},
		)
	})

	t.Run("WhiteQueensideCastle", func(t *testing.T) {
		// 1.d4 d5 2.Nc3 Nc6 3.Bf4 Bf5 4.Qd2 Qd7 5.O-O-O
		testSequence(t,
			"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			[]string{"d2d4", "d7d5", "b1c3", "b8c6", "c1f4", "c8f5", "d1d2", "d8d7", "e1c1"},
		)
	})

	t.Run("BlackKingsideCastle", func(t *testing.T) {
		testSequence(t,
			"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			[]string{"e2e4", "e7e5", "g1f3", "g8f6", "f1c4", "f8c5", "d2d3", "e8g8"},
		)
	})

	t.Run("BlackQueensideCastle", func(t *testing.T) {
		testSequence(t,
			"r3kbnr/pppqpppp/2n1b3/3p4/3P4/2N1B3/PPPQPPPP/R3KBNR b KQkq - 6 4",
			[]string{"e8c8"},
		)
	})

	t.Run("EnPassantWhite", func(t *testing.T) {
		testSequence(t,
			"rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
			[]string{"e5d6"},
		)
	})

	t.Run("EnPassantBlack", func(t *testing.T) {
		testSequence(t,
			"rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
			[]string{"d4e3"},
		)
	})

	t.Run("EnPassantSequence", func(t *testing.T) {
		testSequence(t,
			"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			[]string{"e2e4", "d7d5", "e4e5", "f7f5", "e5f6"}, // en passant on f6
		)
	})

	t.Run("PromotionQueen", func(t *testing.T) {
		testSequence(t,
			"3k4/4P3/8/8/8/8/8/4K3 w - - 0 1",
			[]string{"e7e8q"},
		)
	})

	t.Run("PromotionKnight", func(t *testing.T) {
		testSequence(t,
			"3k4/4P3/8/8/8/8/8/4K3 w - - 0 1",
			[]string{"e7e8n"},
		)
	})

	t.Run("PromotionRook", func(t *testing.T) {
		testSequence(t,
			"3k4/4P3/8/8/8/8/8/4K3 w - - 0 1",
			[]string{"e7e8r"},
		)
	})

	t.Run("PromotionBishop", func(t *testing.T) {
		testSequence(t,
			"3k4/4P3/8/8/8/8/8/4K3 w - - 0 1",
			[]string{"e7e8b"},
		)
	})

	t.Run("CapturePromotion", func(t *testing.T) {
		testSequence(t,
			"3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1",
			[]string{"e7d8q"},
		)
	})

	t.Run("CapturePromotionKnight", func(t *testing.T) {
		testSequence(t,
			"3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1",
			[]string{"e7d8n"},
		)
	})

	t.Run("BlackPromotion", func(t *testing.T) {
		testSequence(t,
			"4k3/8/8/8/8/8/4p3/3K4 b - - 0 1",
			[]string{"e2e1q"},
		)
	})

	t.Run("BlackCapturePromotion", func(t *testing.T) {
		testSequence(t,
			"4k3/8/8/8/8/8/4p3/3RK3 b - - 0 1",
			[]string{"e2d1q"},
		)
	})

	t.Run("SimpleCaptures", func(t *testing.T) {
		testSequence(t,
			"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			[]string{
				"e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6",
				"f3g5", "d7d5", "e4d5", "c6d4",
			},
		)
	})

	t.Run("ManyCaptures", func(t *testing.T) {
		testSequence(t,
			"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
			[]string{"f3e5", "c6e5", "d2d4", "e5c4"},
		)
	})

	t.Run("NullMoveDoesNotCorrupt", func(t *testing.T) {
		var b Board
		b.NNUENet = net
		b.NNUEAcc = NewNNUEAccumulatorStack(256)
		b.SetFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")

		beforeAcc := &NNUEAccumulator{}
		net.RecomputeAccumulator(beforeAcc, &b)
		compareAccumulators(t, b.NNUEAcc.Current(), beforeAcc, "before null move")

		b.MakeNullMove()

		nullAcc := &NNUEAccumulator{}
		net.RecomputeAccumulator(nullAcc, &b)
		compareAccumulators(t, b.NNUEAcc.Current(), nullAcc, "during null move")

		m, _ := b.ParseUCIMove("g1f3")
		b.MakeMove(m)
		b.NNUEAcc.Materialize(net, &b)
		fullAcc := &NNUEAccumulator{}
		net.RecomputeAccumulator(fullAcc, &b)
		compareAccumulators(t, b.NNUEAcc.Current(), fullAcc, "move after null move")

		b.UnmakeMove(m)
		compareAccumulators(t, b.NNUEAcc.Current(), nullAcc, "unmake after null move")

		b.UnmakeNullMove()
		compareAccumulators(t, b.NNUEAcc.Current(), beforeAcc, "after unmake null move")
	})

	t.Run("LongGameSequence", func(t *testing.T) {
		testSequence(t,
			"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			[]string{
				"e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7",
			},
		)
	})

	t.Run("KingMovesThenCapture", func(t *testing.T) {
		testSequence(t,
			"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
			[]string{
				"e7e5", "e1e2", "d7d5", "e2e3", "d5e4", "e3e4",
			},
		)
	})

	t.Run("CastleThenCaptures", func(t *testing.T) {
		testSequence(t,
			"r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
			[]string{"e1g1", "c6d4", "f3d4", "e5d4"},
		)
	})

	t.Run("RookCapturedOnHomeSquare", func(t *testing.T) {
		testSequence(t,
			"r3k2r/1ppppppp/8/8/8/8/1PPPPPPP/R3K2R w KQkq - 0 1",
			[]string{"a1a8"},
		)
	})

	t.Run("DoubleKingMovesAlternating", func(t *testing.T) {
		testSequence(t,
			"4k3/8/8/8/8/8/8/4K3 w - - 0 1",
			[]string{
				"e1d1", "e8d8", "d1c1", "d8c8", "c1b1", "c8b8", "b1a1", "b8a8",
			},
		)
	})

	t.Run("PromotionSequenceMultiple", func(t *testing.T) {
		testSequence(t,
			"8/P3k3/8/8/8/8/p3K3/8 w - - 0 1",
			[]string{"a7a8q", "a2a1q"},
		)
	})

	t.Run("KingMoveSameBucket", func(t *testing.T) {
		// King moves within the same bucket should use incremental update
		testSequence(t,
			"4k3/8/8/8/8/8/8/4K3 w - - 0 1",
			[]string{"e1d1", "e8d8"}, // e1→d1: both file 3-4 region, same bucket
		)
	})

	t.Run("KingMoveBucketChange", func(t *testing.T) {
		// King moves that cross bucket boundaries
		testSequence(t,
			"4k3/8/8/8/8/8/8/4K3 w - - 0 1",
			[]string{
				"e1d2", // might or might not change bucket depending on rank group
				"e8d7",
				"d2c3",
				"d7c6",
			},
		)
	})
}

// TestNNUEIncrementalRandomGames plays random games with NNUE attached,
// checking incremental vs full recompute at every ply. This is a
// fuzz-style coverage test.
func TestNNUEIncrementalRandomGames(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping random game test in short mode")
	}

	// Use synthetic net
	net := &NNUENet{}
	for i := range net.InputBiases {
		net.InputBiases[i] = int16(i % 50)
	}
	for i := 0; i < NNUEInputSize; i++ {
		for j := 0; j < NNUEHiddenSize; j++ {
			net.InputWeights[i][j] = int16((i*7 + j*13 + 3) % 127)
		}
	}

	rng := rand.New(rand.NewSource(42))
	numGames := 20
	maxPly := 200

	for g := 0; g < numGames; g++ {
		var b Board
		b.NNUENet = net
		b.NNUEAcc = NewNNUEAccumulatorStack(512)
		b.Reset()

		played := make([]Move, 0, maxPly)

		for ply := 0; ply < maxPly; ply++ {
			moves := b.GenerateLegalMoves()
			if len(moves) == 0 {
				break // checkmate or stalemate
			}

			m := moves[rng.Intn(len(moves))]
			b.MakeMove(m)
			played = append(played, m)

			b.NNUEAcc.Materialize(net, &b)
			fullAcc := &NNUEAccumulator{}
			net.RecomputeAccumulator(fullAcc, &b)
			incAcc := b.NNUEAcc.Current()
			for i := 0; i < NNUEHiddenSize; i++ {
				if incAcc.White[i] != fullAcc.White[i] || incAcc.Black[i] != fullAcc.Black[i] {
					t.Fatalf("game %d, ply %d, move %s: accumulator mismatch at [%d] "+
						"inc W=%d/B=%d full W=%d/B=%d",
						g, ply, m.String(), i,
						incAcc.White[i], incAcc.Black[i],
						fullAcc.White[i], fullAcc.Black[i])
				}
			}

			// 50-move rule or threefold would make games too long
			if b.HalfmoveClock >= 100 {
				break
			}
		}

		// Unmake all moves and verify final state
		for i := len(played) - 1; i >= 0; i-- {
			b.UnmakeMove(played[i])
		}
		fullAcc := &NNUEAccumulator{}
		net.RecomputeAccumulator(fullAcc, &b)
		incAcc := b.NNUEAcc.Current()
		for i := 0; i < NNUEHiddenSize; i++ {
			if incAcc.White[i] != fullAcc.White[i] || incAcc.Black[i] != fullAcc.Black[i] {
				t.Fatalf("game %d, after full unmake: accumulator mismatch at [%d] "+
					"inc W=%d/B=%d full W=%d/B=%d",
					g, i,
					incAcc.White[i], incAcc.Black[i],
					fullAcc.White[i], fullAcc.Black[i])
			}
		}
	}
}

// TestNNUEV7SaveLoadRoundTrip tests v7 (with L1 hidden layer) save/load and forward pass.
func TestNNUEV7SaveLoadRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	H := 256 // small for testing (must be multiple of 256 for SIMD)
	L1 := 16

	net := &NNUENetV5{
		HiddenSize: H,
		L1Size:     L1,
		UseSCReLU:  false,
		UsePairwise: false,
	}
	net.InputWeights = make([]int16, NNUEInputSize*H)
	net.InputBiases = make([]int16, H)
	net.L1Weights = make([]int16, 2*H*L1)
	net.L1Biases = make([]int16, L1)
	net.OutputWeights = make([]int16, NNUEOutputBuckets*L1)

	// Fill with small random values
	for i := range net.InputWeights {
		net.InputWeights[i] = int16(rng.Intn(201) - 100)
	}
	for i := range net.InputBiases {
		net.InputBiases[i] = int16(rng.Intn(201) - 100)
	}
	for i := range net.L1Weights {
		net.L1Weights[i] = int16(rng.Intn(201) - 100)
	}
	for i := range net.L1Biases {
		net.L1Biases[i] = int16(rng.Intn(201) - 100)
	}
	for i := range net.OutputWeights {
		net.OutputWeights[i] = int16(rng.Intn(201) - 100)
	}
	for b := 0; b < NNUEOutputBuckets; b++ {
		net.OutputBias[b] = int32(rng.Intn(2001) - 1000)
	}

	// Save
	dir := t.TempDir()
	path := filepath.Join(dir, "test-v7.nnue")
	if err := SaveNNUEV5(path, net); err != nil {
		t.Fatalf("SaveNNUEV5 failed: %v", err)
	}

	// Verify version
	ver, err := DetectNNUEVersion(path)
	if err != nil {
		t.Fatalf("DetectNNUEVersion failed: %v", err)
	}
	if ver != 7 {
		t.Fatalf("expected version 7, got %d", ver)
	}

	// Load
	loaded, err := LoadNNUEV5(path)
	if err != nil {
		t.Fatalf("LoadNNUEV5 failed: %v", err)
	}

	// Verify fields
	if loaded.HiddenSize != H {
		t.Errorf("HiddenSize: got %d, want %d", loaded.HiddenSize, H)
	}
	if loaded.L1Size != L1 {
		t.Errorf("L1Size: got %d, want %d", loaded.L1Size, L1)
	}
	if len(loaded.L1Weights) != len(net.L1Weights) {
		t.Fatalf("L1Weights length: got %d, want %d", len(loaded.L1Weights), len(net.L1Weights))
	}
	for i := range net.L1Weights {
		if net.L1Weights[i] != loaded.L1Weights[i] {
			t.Errorf("L1Weights[%d] mismatch: %d vs %d", i, net.L1Weights[i], loaded.L1Weights[i])
			break
		}
	}
	for i := range net.L1Biases {
		if net.L1Biases[i] != loaded.L1Biases[i] {
			t.Errorf("L1Biases[%d] mismatch: %d vs %d", i, net.L1Biases[i], loaded.L1Biases[i])
			break
		}
	}
	for i := range net.OutputWeights {
		if net.OutputWeights[i] != loaded.OutputWeights[i] {
			t.Errorf("OutputWeights[%d] mismatch: %d vs %d", i, net.OutputWeights[i], loaded.OutputWeights[i])
			break
		}
	}
	for b := 0; b < NNUEOutputBuckets; b++ {
		if net.OutputBias[b] != loaded.OutputBias[b] {
			t.Errorf("OutputBias[%d] mismatch: %d vs %d", b, net.OutputBias[b], loaded.OutputBias[b])
		}
	}

	// Test forward pass produces non-zero output
	var b Board
	b.Reset()
	b.SetFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

	acc := NNUEAccumulatorV5{
		White: make([]int16, H),
		Black: make([]int16, H),
	}
	loaded.RecomputeAccumulator(&b, &acc, White)
	loaded.RecomputeAccumulator(&b, &acc, Black)

	score := loaded.Forward(&acc, White, b.AllPieces.Count())
	t.Logf("v7 startpos eval: %d cp", score)
	// With random weights, the score should be non-trivial
	if score == 0 {
		t.Error("forward pass returned exactly 0 with random weights — likely a bug")
	}

	// Test SCReLU variant produces different (but non-zero) output
	loaded.UseSCReLU = true
	scoreSCReLU := loaded.Forward(&acc, White, b.AllPieces.Count())
	t.Logf("v7 startpos eval (SCReLU): %d cp", scoreSCReLU)
	if scoreSCReLU == 0 {
		t.Error("SCReLU forward pass returned exactly 0 with random weights — likely a bug")
	}
	if scoreSCReLU == score {
		t.Error("SCReLU forward pass returned same value as CReLU — activation not applied")
	}
	loaded.UseSCReLU = false // restore
}

// TestNNUEV5SIMDvsGeneric validates that SIMD v5 forward pass matches the Go fallback
// across multiple net architectures and positions.
func TestNNUEV5SIMDvsGeneric(t *testing.T) {
	if !nnueUseSIMDV5 {
		t.Skip("v5 SIMD not available")
	}

	nets := []string{
		"net-v5-120sb-sb120.nnue",
		"net-v5-1536-wdl00-sb800.nnue",
		"net-v5-768pw-wdl0-sb400.nnue",
		"net-v5-screlu-sb120.nnue",
	}

	positions := []string{
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
		"r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
		"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
		"rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
	}

	for _, netName := range nets {
		net, err := LoadNNUEV5(netName)
		if err != nil {
			t.Logf("Skipping %s: %v", netName, err)
			continue
		}
		t.Run(filepath.Base(netName), func(t *testing.T) {
			for _, fen := range positions {
				var board Board
				board.SetFEN(fen)
				board.AttachNNUEV5(net)
				board.NNUEAccV5.MaterializeV5(net, &board)
				acc := board.NNUEAccV5.Current()
				pieceCount := board.AllPieces.Count()

				// SIMD path
				simdScore := net.Forward(acc, board.SideToMove, pieceCount)

				// Go fallback path
				origV5 := nnueUseSIMDV5
				nnueUseSIMDV5 = false
				genericScore := net.Forward(acc, board.SideToMove, pieceCount)
				nnueUseSIMDV5 = origV5

				// Pairwise nets have expected rounding differences: SIMD divides
			// by QA once at the end while the Go fallback divides per-element,
			// causing truncation differences of a few centipawns.
			if net.UsePairwise {
				diff := simdScore - genericScore
				if diff < 0 { diff = -diff }
				if diff > 20 {
					t.Errorf("FEN %q: SIMD=%d Generic=%d (diff %d too large for pairwise rounding)",
						fen, simdScore, genericScore, diff)
				} else {
					t.Logf("FEN %q: SIMD=%d Generic=%d (diff %d, pairwise rounding OK)",
						fen, simdScore, genericScore, diff)
				}
			} else if simdScore != genericScore {
				t.Errorf("FEN %q: SIMD=%d Generic=%d (mismatch)",
					fen, simdScore, genericScore)
			} else {
				t.Logf("FEN %q: SIMD=%d Generic=%d (match)", fen, simdScore, genericScore)
			}
			}
		})
	}
}

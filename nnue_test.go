package chess

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestHalfKPIndex(t *testing.T) {
	// Basic index range test
	for ks := Square(0); ks < 64; ks++ {
		for piece := WhitePawn; piece <= BlackQueen; piece++ {
			if piece == WhiteKing || piece == BlackKing {
				continue
			}
			for ps := Square(0); ps < 64; ps++ {
				idx := HalfKPIndex(White, ks, piece, ps)
				if idx < 0 || idx >= NNUEInputSize {
					t.Errorf("HalfKPIndex(White, %d, %d, %d) = %d out of range [0, %d)", ks, piece, ps, idx, NNUEInputSize)
				}
				idx = HalfKPIndex(Black, ks, piece, ps)
				if idx < 0 || idx >= NNUEInputSize {
					t.Errorf("HalfKPIndex(Black, %d, %d, %d) = %d out of range [0, %d)", ks, piece, ps, idx, NNUEInputSize)
				}
			}
		}
	}
}

func TestHalfKPIndexKingReturnsNegative(t *testing.T) {
	idx := HalfKPIndex(White, 4, WhiteKing, 4)
	if idx != -1 {
		t.Errorf("expected -1 for king, got %d", idx)
	}
	idx = HalfKPIndex(Black, 60, BlackKing, 60)
	if idx != -1 {
		t.Errorf("expected -1 for king, got %d", idx)
	}
}

func TestHalfKPIndexUniqueness(t *testing.T) {
	// All indices for a given perspective+kingSq should be unique
	seen := make(map[int]bool)
	kingSq := Square(4) // e1
	for piece := WhitePawn; piece <= BlackQueen; piece++ {
		if piece == WhiteKing || piece == BlackKing {
			continue
		}
		for ps := Square(0); ps < 64; ps++ {
			idx := HalfKPIndex(White, kingSq, piece, ps)
			if idx >= 0 {
				if seen[idx] {
					t.Errorf("duplicate index %d for piece=%d sq=%d", idx, piece, ps)
				}
				seen[idx] = true
			}
		}
	}
}

func TestHalfKPSymmetry(t *testing.T) {
	// A White pawn on e2 from White's perspective with king on e1
	// should map to the same index as a Black pawn on e7 from Black's perspective with king on e8
	// (after mirroring: squares ^56, colors swapped)

	wIdx := HalfKPIndex(White, NewSquare(4, 0), WhitePawn, NewSquare(4, 1))
	bIdx := HalfKPIndex(Black, NewSquare(4, 7), BlackPawn, NewSquare(4, 6))

	if wIdx != bIdx {
		t.Errorf("symmetry broken: White index=%d, Black index=%d", wIdx, bIdx)
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

	// Push and verify copy
	stack.Push()
	if !stack.Current().Computed {
		t.Error("pushed accumulator should be computed")
	}
	if stack.Current().White[0] != 42 {
		t.Errorf("expected 42, got %d", stack.Current().White[0])
	}

	// Modify top
	stack.Current().White[0] = 99

	// Pop should restore
	stack.Pop()
	if stack.Current().White[0] != 42 {
		t.Errorf("after pop expected 42, got %d", stack.Current().White[0])
	}
}

func TestNNUEAccumulatorStackDeepCopy(t *testing.T) {
	stack := NewNNUEAccumulatorStack(8)
	stack.Current().White[0] = 100
	stack.Current().Computed = true
	stack.Push()
	stack.Current().White[0] = 200

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
	idx := HalfKPIndex(White, NewSquare(4, 0), WhitePawn, NewSquare(4, 1))
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
		net.OutputWeights[j] = 1
	}
	net.OutputBias = 0
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
	score := net.Evaluate(acc, White)

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

	// Incremental: start with biases, add each piece
	accInc := &NNUEAccumulator{}
	accInc.White = net.InputBiases
	accInc.Black = net.InputBiases
	accInc.Computed = true

	wKingSq := b.Pieces[WhiteKing].LSB()
	bKingSq := b.Pieces[BlackKing].LSB()

	for piece := WhitePawn; piece <= BlackQueen; piece++ {
		if piece == WhiteKing || piece == BlackKing {
			continue
		}
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
	idx := HalfKPIndex(White, NewSquare(4, 0), WhitePawn, NewSquare(4, 1))
	if idx >= 0 {
		for i := 0; i < NNUEHiddenSize; i++ {
			if net.InputWeights[idx][i] != loaded.InputWeights[idx][i] {
				t.Errorf("InputWeights[%d][%d] mismatch: %d vs %d", idx, i, net.InputWeights[idx][i], loaded.InputWeights[idx][i])
				break
			}
		}
	}

	// Compare output
	if net.OutputBias != loaded.OutputBias {
		t.Errorf("OutputBias mismatch: %d vs %d", net.OutputBias, loaded.OutputBias)
	}
	for j := 0; j < NNUEHidden2Size; j++ {
		if net.OutputWeights[j] != loaded.OutputWeights[j] {
			t.Errorf("OutputWeights[%d] mismatch: %d vs %d", j, net.OutputWeights[j], loaded.OutputWeights[j])
			break
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
	for j := 0; j < NNUEHidden2Size; j++ {
		net.OutputWeights[j] = 1
	}

	var b Board
	b.Reset()

	acc := &NNUEAccumulator{}
	net.RecomputeAccumulator(acc, &b)

	// In the starting position, White's evaluation and Black's evaluation
	// should be symmetric (same magnitude, side-to-move adjusted)
	scoreW := net.Evaluate(acc, White)
	scoreB := net.Evaluate(acc, Black)

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

	// King moved — accumulator should have been fully recomputed
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
		{WhiteRook, 3}, {WhiteQueen, 4},
		{BlackPawn, 5}, {BlackKnight, 6}, {BlackBishop, 7},
		{BlackRook, 8}, {BlackQueen, 9},
		{WhiteKing, -1}, {BlackKing, -1}, {Empty, -1},
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
		net.OutputWeights[j] = int16(j*5 - 80)
	}
	net.OutputBias = 42
	net.PrepareWeights()

	var b Board
	b.Reset()
	acc := &NNUEAccumulator{}
	net.RecomputeAccumulator(acc, &b)

	// Force generic path
	origFlag := nnueUseSIMD
	nnueUseSIMD = false
	genericScore := net.Evaluate(acc, White)

	// Force SIMD path
	nnueUseSIMD = true
	simdScore := net.Evaluate(acc, White)

	nnueUseSIMD = origFlag

	if genericScore != simdScore {
		t.Errorf("SIMD vs Generic mismatch: simd=%d generic=%d", simdScore, genericScore)
	}
	t.Logf("SIMD=%d Generic=%d (match)", simdScore, genericScore)

	// Also test Black perspective
	nnueUseSIMD = false
	genericB := net.Evaluate(acc, Black)
	nnueUseSIMD = true
	simdB := net.Evaluate(acc, Black)
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
		net.OutputWeights[j] = int16(j*5 - 80)
	}
	net.OutputBias = 42

	var board Board
	board.Reset()
	acc := &NNUEAccumulator{}
	net.RecomputeAccumulator(acc, &board)

	origFlag := nnueUseSIMD
	nnueUseSIMD = false
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		net.Evaluate(acc, White)
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
		net.OutputWeights[j] = int16(j*5 - 80)
	}
	net.OutputBias = 42
	net.PrepareWeights()

	var board Board
	board.Reset()
	acc := &NNUEAccumulator{}
	net.RecomputeAccumulator(acc, &board)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		net.Evaluate(acc, White)
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
// move sequences. Uses the real net.nnue for maximum fidelity.
func TestNNUEIncrementalEdgeCases(t *testing.T) {
	// Try to load the real network; fall back to synthetic if unavailable.
	net, err := LoadNNUE("/home/adam/code/gochess/net.nnue")
	if err != nil {
		t.Logf("net.nnue not available (%v), using synthetic weights", err)
		net = &NNUENet{}
		for i := range net.InputBiases {
			net.InputBiases[i] = 5
		}
		for i := 0; i < NNUEInputSize; i++ {
			for j := 0; j < NNUEHiddenSize; j++ {
				net.InputWeights[i][j] = int16((i*7 + j*13) % 100)
			}
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
		// White pawn on e5, Black plays d7d5, White captures en passant e5d6
		testSequence(t,
			"rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
			[]string{"e5d6"},
		)
	})

	t.Run("EnPassantBlack", func(t *testing.T) {
		// Black pawn on d4, White plays e2e4, Black captures en passant d4e3
		testSequence(t,
			"rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
			[]string{"d4e3"},
		)
	})

	t.Run("EnPassantSequence", func(t *testing.T) {
		// Full game leading to en passant
		testSequence(t,
			"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			[]string{"e2e4", "d7d5", "e4e5", "f7f5", "e5f6"}, // en passant on f6
		)
	})

	t.Run("PromotionQueen", func(t *testing.T) {
		// White pawn on e7, promote to queen
		testSequence(t,
			"4k3/4P3/8/8/8/8/8/4K3 w - - 0 1",
			[]string{"e7e8q"},
		)
	})

	t.Run("PromotionKnight", func(t *testing.T) {
		testSequence(t,
			"4k3/4P3/8/8/8/8/8/4K3 w - - 0 1",
			[]string{"e7e8n"},
		)
	})

	t.Run("PromotionRook", func(t *testing.T) {
		testSequence(t,
			"4k3/4P3/8/8/8/8/8/4K3 w - - 0 1",
			[]string{"e7e8r"},
		)
	})

	t.Run("PromotionBishop", func(t *testing.T) {
		testSequence(t,
			"4k3/4P3/8/8/8/8/8/4K3 w - - 0 1",
			[]string{"e7e8b"},
		)
	})

	t.Run("CapturePromotion", func(t *testing.T) {
		// White pawn on e7 captures on d8 and promotes
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
			"4k3/8/8/8/8/8/4p3/4K3 b - - 0 1",
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
		// Italian Game with captures
		testSequence(t,
			"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			[]string{
				"e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6",
				"f3g5", // Ng5 threatening f7
				"d7d5", // d5 counter
				"e4d5", // exd5 capture
				"c6d4", // Nxd4 capture
			},
		)
	})

	t.Run("ManyCaptures", func(t *testing.T) {
		// Position with lots of captures available
		testSequence(t,
			"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
			[]string{
				"f3e5", // Nxe5 capture
				"c6e5", // Nxe5 capture back
				"d2d4", // d4
				"e5c4", // Nxc4 capture bishop
			},
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

		// After null move, accumulator should still be valid
		// (same position, just side flipped)
		nullAcc := &NNUEAccumulator{}
		net.RecomputeAccumulator(nullAcc, &b)
		compareAccumulators(t, b.NNUEAcc.Current(), nullAcc, "during null move")

		// Make a move inside null move, verify
		m, _ := b.ParseUCIMove("g1f3")
		b.MakeMove(m)
		fullAcc := &NNUEAccumulator{}
		net.RecomputeAccumulator(fullAcc, &b)
		compareAccumulators(t, b.NNUEAcc.Current(), fullAcc, "move after null move")

		b.UnmakeMove(m)
		compareAccumulators(t, b.NNUEAcc.Current(), nullAcc, "unmake after null move")

		b.UnmakeNullMove()
		compareAccumulators(t, b.NNUEAcc.Current(), beforeAcc, "after unmake null move")
	})

	t.Run("LongGameSequence", func(t *testing.T) {
		// Scholar's mate + extra moves to stress test
		testSequence(t,
			"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			[]string{
				"e2e4", "e7e5",
				"d1h5", "b8c6",
				"f1c4", "g8f6",
				"h5f7", // Qxf7 capture, check
			},
		)
	})

	t.Run("KingMovesThenCapture", func(t *testing.T) {
		// Test that king moves (full recompute) followed by non-king
		// incremental updates work correctly in sequence.
		testSequence(t,
			"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
			[]string{
				"e7e5",
				"e1e2", // White king move (full recompute)
				"d7d5", // Black pawn push
				"e2e3", // White king move again
				"d5e4", // Black captures pawn
				"e3e4", // White king captures pawn (king move + capture)
			},
		)
	})

	t.Run("CastleThenCaptures", func(t *testing.T) {
		// Castle early then play captures — tests that castle's
		// king-move recompute properly sets up for subsequent incremental.
		testSequence(t,
			"r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
			[]string{
				"e1g1",  // White castles kingside (king move -> recompute)
				"c6d4",  // Nxd4
				"f3d4",  // Nxd4 capture
				"e5d4",  // exd4 capture
			},
		)
	})

	t.Run("RookCapturedOnHomeSquare", func(t *testing.T) {
		// Test capturing a rook on its home square (castling rights change)
		testSequence(t,
			"r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
			[]string{
				"a1a8", // Rook captures rook on a8 (removes both castling rights)
			},
		)
	})

	t.Run("DoubleKingMovesAlternating", func(t *testing.T) {
		// Both kings moving alternately — each triggers full recompute
		testSequence(t,
			"4k3/8/8/8/8/8/8/4K3 w - - 0 1",
			[]string{
				"e1d2", "e8d7",
				"d2c3", "d7c6",
				"c3b4", "c6b5",
				"b4a5", "b5a6",
			},
		)
	})

	t.Run("PromotionSequenceMultiple", func(t *testing.T) {
		// Both sides promote
		testSequence(t,
			"4k3/1P6/8/8/8/8/1p6/4K3 w - - 0 1",
			[]string{
				"b7b8q", // White promotes queen
				"b2b1q", // Black promotes queen
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

	// Use synthetic net (real net not required for correctness check)
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

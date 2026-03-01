package chess

import (
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

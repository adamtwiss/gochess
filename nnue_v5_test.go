package chess

import "testing"

// loadAnyV5Net loads any available v5 net for testing.
func loadAnyV5Net(t *testing.T) *NNUENetV5 {
	t.Helper()
	for _, name := range []string{
		"net-v5-120sb-sb120.nnue",
		"net-v5-1024s-w0-e400s400.nnue",
		"net-v5-1536-w5-e800s800.nnue",
	} {
		net, err := LoadNNUEV5(name)
		if err == nil {
			return net
		}
	}
	t.Skip("no v5 net available")
	return nil
}

// TestMaterializeV5AllDirtyTypes verifies that MaterializeV5 produces the same
// eval as a full recompute for every dirty type: quiet moves, captures,
// en passant, promotions, capture-promotions, king moves (same bucket), and castling.
func TestMaterializeV5AllDirtyTypes(t *testing.T) {
	net := loadAnyV5Net(t)

	cases := []struct {
		name string
		fen  string
		move Move // will be constructed from algebraic
		from Square
		to   Square
		flag int
	}{
		{
			name: "QuietPawnPush",
			fen:  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			from: NewSquare(4, 1), to: NewSquare(4, 3), // e2e4
		},
		{
			name: "QuietKnightMove",
			fen:  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			from: NewSquare(6, 0), to: NewSquare(5, 2), // Nf3
		},
		{
			name: "Capture",
			fen:  "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
			from: NewSquare(4, 3), to: NewSquare(3, 4), // exd5
		},
		{
			name: "EnPassant",
			fen:  "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
			from: NewSquare(4, 4), to: NewSquare(3, 5), flag: FlagEnPassant, // exd6 e.p.
		},
		{
			name: "PromotionQueen",
			fen:  "8/4P3/8/8/8/8/4k3/4K3 w - - 0 1",
			from: NewSquare(4, 6), to: NewSquare(4, 7), flag: FlagPromoteQ, // e8=Q
		},
		{
			name: "CapturePromotion",
			fen:  "3r4/4P3/8/8/8/8/4k3/4K3 w - - 0 1",
			from: NewSquare(4, 6), to: NewSquare(3, 7), flag: FlagPromoteQ, // exd8=Q
		},
		{
			name: "KingsideCastle",
			fen:  "r1bqkbnr/pppppppp/2n5/8/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
			from: NewSquare(4, 0), to: NewSquare(6, 0), flag: FlagCastle, // O-O
		},
		{
			name: "QueensideCastle",
			fen:  "r3kbnr/pppqpppp/2n5/3p1b2/3P1B2/2N2N2/PPPQPPPP/R3KB1R w KQkq - 6 5",
			from: NewSquare(4, 0), to: NewSquare(2, 0), flag: FlagCastle, // O-O-O
		},
		{
			name: "BlackCapture",
			fen:  "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2",
			from: NewSquare(4, 4), to: NewSquare(3, 3), // exd4
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			move := NewMoveFlags(tc.from, tc.to, tc.flag)

			// Path A: incremental (MakeMove + MaterializeV5)
			var boardInc Board
			boardInc.SetFEN(tc.fen)
			boardInc.AttachNNUEV5(net)
			boardInc.NNUEAccV5.MaterializeV5(net, &boardInc) // materialize root
			boardInc.MakeMove(move)
			boardInc.NNUEAccV5.MaterializeV5(net, &boardInc) // incremental update
			accInc := boardInc.NNUEAccV5.Current()

			// Path B: full recompute from the post-move position
			var boardFull Board
			boardFull.SetFEN(tc.fen)
			boardFull.AttachNNUEV5(net)
			boardFull.MakeMove(move)
			// Force full recompute by resetting dirty type
			accFull := boardFull.NNUEAccV5.Current()
			accFull.Dirty.Type = 0
			boardFull.NNUEAccV5.MaterializeV5(net, &boardFull)

			// Compare accumulators
			for i := 0; i < net.HiddenSize; i++ {
				if accInc.White[i] != accFull.White[i] {
					t.Fatalf("White[%d]: incremental=%d recompute=%d", i, accInc.White[i], accFull.White[i])
				}
				if accInc.Black[i] != accFull.Black[i] {
					t.Fatalf("Black[%d]: incremental=%d recompute=%d", i, accInc.Black[i], accFull.Black[i])
				}
			}

			// Also verify Forward produces same eval
			pieceCount := boardInc.AllPieces.Count()
			evalInc := net.Forward(accInc, boardInc.SideToMove, pieceCount)
			evalFull := net.Forward(accFull, boardFull.SideToMove, pieceCount)
			if evalInc != evalFull {
				t.Errorf("Forward mismatch: incremental=%d recompute=%d", evalInc, evalFull)
			}
		})
	}
}

// TestNNUEV5DeepCopy verifies that DeepCopy produces an independent copy —
// modifying the copy does not affect the original (critical for Lazy SMP).
func TestNNUEV5DeepCopy(t *testing.T) {
	net := loadAnyV5Net(t)

	var board Board
	board.Reset()
	board.AttachNNUEV5(net)
	board.NNUEAccV5.MaterializeV5(net, &board)

	// Save original accumulator values
	orig := board.NNUEAccV5.Current()
	origWhite0 := orig.White[0]
	origBlack0 := orig.Black[0]

	// Deep copy
	copyStack := board.NNUEAccV5.DeepCopy()

	// Modify the copy
	copyAcc := copyStack.Current()
	copyAcc.White[0] = 9999
	copyAcc.Black[0] = -9999

	// Verify original is unaffected
	if orig.White[0] != origWhite0 {
		t.Errorf("original White[0] changed: was %d, now %d", origWhite0, orig.White[0])
	}
	if orig.Black[0] != origBlack0 {
		t.Errorf("original Black[0] changed: was %d, now %d", origBlack0, orig.Black[0])
	}

	// Verify finny table independence
	copyStack.InvalidateFinny()
	// Original finny should still be valid (if it was populated)
	// Just check it doesn't panic
	board.NNUEAccV5.MaterializeV5(net, &board)
}

// TestNNUEV5EvaluateRelative verifies the full eval pipeline:
// NNUEEvaluateRelativeV5 → MaterializeV5 → Forward.
func TestNNUEV5EvaluateRelative(t *testing.T) {
	net := loadAnyV5Net(t)

	positions := []struct {
		fen string
	}{
		{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
		{"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"},
		{"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"},
		{"8/8/4kpp1/3p1b2/p6P/2B5/6P1/6K1 w - - 0 1"},
	}

	for _, pos := range positions {
		var board Board
		board.SetFEN(pos.fen)
		board.AttachNNUEV5(net)

		eval := board.NNUEEvaluateRelativeV5()

		// Sanity: eval should be in a reasonable range
		if eval < -5000 || eval > 5000 {
			t.Errorf("FEN %q: eval %d out of reasonable range", pos.fen, eval)
		}

		// Verify consistency: calling twice gives same result
		eval2 := board.NNUEEvaluateRelativeV5()
		if eval != eval2 {
			t.Errorf("FEN %q: eval not deterministic: %d vs %d", pos.fen, eval, eval2)
		}
	}
}

// TestMaterializeV5MultipleMovesSequence verifies accumulator correctness
// across a sequence of moves, comparing incremental vs full recompute at each step.
func TestMaterializeV5MultipleMovesSequence(t *testing.T) {
	net := loadAnyV5Net(t)

	// Play a short game from startpos
	moves := []struct {
		from, to Square
		flag     int
	}{
		{NewSquare(4, 1), NewSquare(4, 3), FlagNone},  // e4
		{NewSquare(4, 6), NewSquare(4, 4), FlagNone},  // e5
		{NewSquare(6, 0), NewSquare(5, 2), FlagNone},  // Nf3
		{NewSquare(1, 7), NewSquare(2, 5), FlagNone},  // Nc6
		{NewSquare(5, 0), NewSquare(1, 4), FlagNone},  // Bb5
		{NewSquare(0, 6), NewSquare(0, 5), FlagNone},  // a6
		{NewSquare(1, 4), NewSquare(2, 5), FlagNone},  // Bxc6 (capture)
		{NewSquare(3, 6), NewSquare(2, 5), FlagNone},  // dxc6 (capture)
		{NewSquare(4, 0), NewSquare(6, 0), FlagCastle}, // O-O
	}

	var board Board
	board.Reset()
	board.AttachNNUEV5(net)
	board.NNUEAccV5.MaterializeV5(net, &board)

	for i, m := range moves {
		move := NewMoveFlags(m.from, m.to, m.flag)
		board.MakeMove(move)
		board.NNUEAccV5.MaterializeV5(net, &board)

		// Full recompute for comparison
		var refBoard Board
		refBoard.SetFEN(board.ToFEN())
		refBoard.AttachNNUEV5(net)
		refBoard.NNUEAccV5.MaterializeV5(net, &refBoard)

		accInc := board.NNUEAccV5.Current()
		accRef := refBoard.NNUEAccV5.Current()

		for j := 0; j < net.HiddenSize; j++ {
			if accInc.White[j] != accRef.White[j] {
				t.Fatalf("move %d: White[%d] mismatch: inc=%d ref=%d", i, j, accInc.White[j], accRef.White[j])
			}
			if accInc.Black[j] != accRef.Black[j] {
				t.Fatalf("move %d: Black[%d] mismatch: inc=%d ref=%d", i, j, accInc.Black[j], accRef.Black[j])
			}
		}
	}
}

// TestCopySubAddV5SliceSIMDvsGeneric verifies the fused copy+sub+add SIMD
// kernel matches the generic Go implementation.
func TestCopySubAddV5SliceSIMDvsGeneric(t *testing.T) {
	if !nnueUseSIMD {
		t.Skip("SIMD not available")
	}

	const N = 1024
	src := make([]int16, N)
	oldW := make([]int16, N)
	newW := make([]int16, N)

	// Fill with varied data
	for i := 0; i < N; i++ {
		src[i] = int16(i*7 - 3000)
		oldW[i] = int16(i*3 - 1500)
		newW[i] = int16(i*11 - 5000)
	}

	// SIMD path
	dstSIMD := make([]int16, N)
	copySubAddV5Slice(dstSIMD, src, oldW, newW)

	// Generic path
	dstGeneric := make([]int16, N)
	origSIMD := nnueUseSIMD
	nnueUseSIMD = false
	copySubAddV5Slice(dstGeneric, src, oldW, newW)
	nnueUseSIMD = origSIMD

	for i := 0; i < N; i++ {
		if dstSIMD[i] != dstGeneric[i] {
			t.Fatalf("copySubAdd[%d]: SIMD=%d Generic=%d", i, dstSIMD[i], dstGeneric[i])
		}
	}
}

// TestCopySubSubAddV5SliceSIMDvsGeneric verifies the fused copy+sub+sub+add
// SIMD kernel matches the generic Go implementation.
func TestCopySubSubAddV5SliceSIMDvsGeneric(t *testing.T) {
	if !nnueUseSIMD {
		t.Skip("SIMD not available")
	}

	const N = 1024
	src := make([]int16, N)
	oldW := make([]int16, N)
	newW := make([]int16, N)
	capW := make([]int16, N)

	for i := 0; i < N; i++ {
		src[i] = int16(i*7 - 3000)
		oldW[i] = int16(i*3 - 1500)
		newW[i] = int16(i*11 - 5000)
		capW[i] = int16(i*5 - 2000)
	}

	// SIMD path
	dstSIMD := make([]int16, N)
	copySubSubAddV5Slice(dstSIMD, src, oldW, newW, capW)

	// Generic path
	dstGeneric := make([]int16, N)
	origSIMD := nnueUseSIMD
	nnueUseSIMD = false
	copySubSubAddV5Slice(dstGeneric, src, oldW, newW, capW)
	nnueUseSIMD = origSIMD

	for i := 0; i < N; i++ {
		if dstSIMD[i] != dstGeneric[i] {
			t.Fatalf("copySubSubAdd[%d]: SIMD=%d Generic=%d", i, dstSIMD[i], dstGeneric[i])
		}
	}
}

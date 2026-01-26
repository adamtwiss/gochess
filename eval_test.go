package chess

import "testing"

func TestEvalStartingPosition(t *testing.T) {
	var b Board
	b.Reset()

	score := b.Evaluate()

	// Starting position should be roughly equal (close to 0)
	// Small differences possible due to mobility
	if score < -50 || score > 50 {
		t.Errorf("Starting position eval = %d, expected close to 0", score)
	}
}

func TestEvalMaterialAdvantage(t *testing.T) {
	var b Board

	// White up a queen
	b.SetFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
	baseScore := b.Evaluate()

	// Remove black queen
	b.SetFEN("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
	scoreWithoutBQ := b.Evaluate()

	diff := scoreWithoutBQ - baseScore
	// Should reflect queen value from PST (varies by square, ~900 range)
	if diff < 700 || diff > 1200 {
		t.Errorf("Queen removal changed score by %d, expected ~900", diff)
	}
}

func TestEvalMobility(t *testing.T) {
	var b Board

	// Open position - pieces have more mobility
	b.SetFEN("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
	openScore := b.Evaluate()

	// Closed position - pieces have less mobility
	b.SetFEN("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
	closedScore := b.Evaluate()

	// The open position (Italian Game) should generally score higher for white
	// due to the bishop on c4 having good mobility
	t.Logf("Open position score: %d", openScore)
	t.Logf("Closed position score: %d", closedScore)
}

func TestEvalRelative(t *testing.T) {
	var b Board
	b.SetFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")

	absScore := b.Evaluate()
	relScore := b.EvaluateRelative()

	// Black to move, so relative should be negated
	if relScore != -absScore {
		t.Errorf("EvaluateRelative() = %d, expected %d", relScore, -absScore)
	}
}

func TestEvalSymmetry(t *testing.T) {
	var b Board
	b.Reset()

	// After e4 e5, the position should still be roughly symmetrical
	b.SetFEN("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
	score := b.Evaluate()

	// Should be close to 0 due to symmetry (PST values are near-symmetric)
	if score < -50 || score > 50 {
		t.Errorf("Symmetric position eval = %d, expected close to 0", score)
	}
}

func TestEvalPSTKnightCentralization(t *testing.T) {
	// Knight on e4 (central) should score higher than knight on a1 (corner)
	var b Board

	// Minimal position: just kings and a white knight on e4
	b.SetFEN("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1")
	centralScore := b.Evaluate()

	// Knight on a1
	b.SetFEN("4k3/8/8/8/8/8/8/N3K3 w - - 0 1")
	cornerScore := b.Evaluate()

	if centralScore <= cornerScore {
		t.Errorf("Central knight (%d) should score higher than corner knight (%d)", centralScore, cornerScore)
	}
	t.Logf("Central knight: %d, Corner knight: %d, diff: %d", centralScore, cornerScore, centralScore-cornerScore)
}

func TestEvalTaperedPhase(t *testing.T) {
	var b Board

	// Starting position should have phase = 0 (all pieces present)
	b.Reset()
	phase := b.computePhase()
	if phase != 0 {
		t.Errorf("Starting position phase = %d, expected 0", phase)
	}

	// K+P vs K endgame should have max phase (TotalPhase)
	b.SetFEN("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
	phase = b.computePhase()
	if phase != TotalPhase {
		t.Errorf("K+P vs K phase = %d, expected %d", phase, TotalPhase)
	}

	// K+Q vs K should have reduced phase
	b.SetFEN("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
	phase = b.computePhase()
	expectedPhase := TotalPhase - QueenPhase
	if phase != expectedPhase {
		t.Errorf("K+Q vs K phase = %d, expected %d", phase, expectedPhase)
	}
}

func TestPawnHashIncremental(t *testing.T) {
	var b Board
	b.Reset()

	// Play some moves and verify pawn hash matches full recompute
	moves := []string{"e2e4", "e7e5", "d2d4", "d7d5"}
	for _, ms := range moves {
		from := ParseSquare(ms[:2])
		to := ParseSquare(ms[2:4])
		legalMoves := b.GenerateLegalMoves()
		var m Move
		for _, lm := range legalMoves {
			if lm.From() == from && lm.To() == to {
				m = lm
				break
			}
		}
		if m == NoMove {
			t.Fatalf("Could not find legal move %s", ms)
		}
		b.MakeMove(m)

		expected := b.PawnHash()
		if b.PawnHashKey != expected {
			t.Errorf("After %s: incremental pawn hash %x != recomputed %x", ms, b.PawnHashKey, expected)
		}
	}

	// Unmake all moves and verify hash is restored
	for i := len(moves) - 1; i >= 0; i-- {
		n := len(b.UndoStack)
		undo := b.UndoStack[n-1]
		b.UnmakeMove(undo.Move)
	}

	expected := b.PawnHash()
	if b.PawnHashKey != expected {
		t.Errorf("After unmake all: incremental pawn hash %x != recomputed %x", b.PawnHashKey, expected)
	}
}

func TestPawnHashCapture(t *testing.T) {
	// Position where a pawn can capture another pawn
	var b Board
	b.SetFEN("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")

	// e4xd5
	legalMoves := b.GenerateLegalMoves()
	var captureMove Move
	for _, m := range legalMoves {
		from := m.From()
		to := m.To()
		if from == ParseSquare("e4") && to == ParseSquare("d5") {
			captureMove = m
			break
		}
	}
	if captureMove == NoMove {
		t.Fatal("Could not find e4xd5")
	}

	b.MakeMove(captureMove)
	expected := b.PawnHash()
	if b.PawnHashKey != expected {
		t.Errorf("After pawn capture: incremental %x != recomputed %x", b.PawnHashKey, expected)
	}

	b.UnmakeMove(captureMove)
	expected = b.PawnHash()
	if b.PawnHashKey != expected {
		t.Errorf("After unmake capture: incremental %x != recomputed %x", b.PawnHashKey, expected)
	}
}

func TestPawnHashPromotion(t *testing.T) {
	// Pawn about to promote
	var b Board
	b.SetFEN("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")

	legalMoves := b.GenerateLegalMoves()
	var promoMove Move
	for _, m := range legalMoves {
		if m.From() == ParseSquare("a7") && m.IsPromotion() {
			promoMove = m
			break
		}
	}
	if promoMove == NoMove {
		t.Fatal("Could not find promotion move")
	}

	b.MakeMove(promoMove)
	expected := b.PawnHash()
	if b.PawnHashKey != expected {
		t.Errorf("After promotion: incremental %x != recomputed %x", b.PawnHashKey, expected)
	}

	b.UnmakeMove(promoMove)
	expected = b.PawnHash()
	if b.PawnHashKey != expected {
		t.Errorf("After unmake promotion: incremental %x != recomputed %x", b.PawnHashKey, expected)
	}
}

func TestPawnHashEnPassant(t *testing.T) {
	// En passant position
	var b Board
	b.SetFEN("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")

	legalMoves := b.GenerateLegalMoves()
	var epMove Move
	for _, m := range legalMoves {
		if m.From() == ParseSquare("e5") && m.To() == ParseSquare("d6") {
			epMove = m
			break
		}
	}
	if epMove == NoMove {
		t.Fatal("Could not find en passant move")
	}

	b.MakeMove(epMove)
	expected := b.PawnHash()
	if b.PawnHashKey != expected {
		t.Errorf("After en passant: incremental %x != recomputed %x", b.PawnHashKey, expected)
	}

	b.UnmakeMove(epMove)
	expected = b.PawnHash()
	if b.PawnHashKey != expected {
		t.Errorf("After unmake en passant: incremental %x != recomputed %x", b.PawnHashKey, expected)
	}
}

func TestPawnDoubled(t *testing.T) {
	var b Board

	// Doubled white pawns on e-file vs single pawn
	b.SetFEN("4k3/8/8/8/4P3/4P3/8/4K3 w - - 0 1")
	doubledScore := b.Evaluate()

	b.SetFEN("4k3/8/8/8/4P3/3P4/8/4K3 w - - 0 1")
	normalScore := b.Evaluate()

	if doubledScore >= normalScore {
		t.Errorf("Doubled pawns (%d) should score lower than normal (%d)", doubledScore, normalScore)
	}
	t.Logf("Doubled: %d, Normal: %d, diff: %d", doubledScore, normalScore, doubledScore-normalScore)
}

func TestPawnIsolated(t *testing.T) {
	var b Board

	// d4 pawn isolated (support pawn far away on g3)
	b.SetFEN("4k3/8/8/8/3P4/6P1/8/4K3 w - - 0 1")
	isolatedScore := b.Evaluate()

	// d4 pawn not isolated (support pawn on c3, adjacent file)
	b.SetFEN("4k3/8/8/8/3P4/2P5/8/4K3 w - - 0 1")
	supportedScore := b.Evaluate()

	if isolatedScore >= supportedScore {
		t.Errorf("Isolated pawn (%d) should score lower than supported (%d)", isolatedScore, supportedScore)
	}
	t.Logf("Isolated: %d, Supported: %d, diff: %d", isolatedScore, supportedScore, isolatedScore-supportedScore)
}

func TestPawnPassed(t *testing.T) {
	var b Board

	// White passed pawn on e5 (no black pawns blocking)
	b.SetFEN("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1")
	passedScore := b.Evaluate()

	// Same pawn but blocked by enemy pawn on e6
	b.SetFEN("4k3/8/4p3/4P3/8/8/8/4K3 w - - 0 1")
	blockedScore := b.Evaluate()

	// Passed pawn should be valued higher (white gets bonus, not penalized by enemy pawn presence)
	if passedScore <= blockedScore {
		t.Errorf("Passed pawn (%d) should score higher than blocked (%d)", passedScore, blockedScore)
	}
	t.Logf("Passed: %d, Blocked: %d, diff: %d", passedScore, blockedScore, passedScore-blockedScore)
}

func TestPawnPassedRankBonus(t *testing.T) {
	var b Board

	// Passed pawn on rank 6 (more advanced)
	b.SetFEN("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1")
	advancedScore := b.Evaluate()

	// Passed pawn on rank 3 (less advanced)
	b.SetFEN("4k3/8/8/8/8/4P3/8/4K3 w - - 0 1")
	earlyScore := b.Evaluate()

	if advancedScore <= earlyScore {
		t.Errorf("Advanced passed pawn (%d) should score higher than early (%d)", advancedScore, earlyScore)
	}
	t.Logf("Rank 6: %d, Rank 3: %d, diff: %d", advancedScore, earlyScore, advancedScore-earlyScore)
}

func TestPawnConnected(t *testing.T) {
	var b Board

	// Connected pawns (d4+e3: e3 defends d4... actually d4 attacks e5/c5, e3 attacks d4/f4)
	// Use d4+c3 where c3 pawn attacks d4
	b.SetFEN("4k3/8/8/8/3P4/2P5/8/4K3 w - - 0 1")
	connectedScore := b.Evaluate()

	// Disconnected pawns on a4 and h4
	b.SetFEN("4k3/8/8/8/P6P/8/8/4K3 w - - 0 1")
	disconnectedScore := b.Evaluate()

	if connectedScore <= disconnectedScore {
		t.Errorf("Connected pawns (%d) should score higher than disconnected (%d)", connectedScore, disconnectedScore)
	}
	t.Logf("Connected: %d, Disconnected: %d, diff: %d", connectedScore, disconnectedScore, connectedScore-disconnectedScore)
}

func TestKingSafety(t *testing.T) {
	var b Board

	// Castled king with intact pawn shield
	b.SetFEN("r1bq1rk1/ppppbppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 6 5")
	shieldedScore := b.Evaluate()

	// King on e1 with pawns pushed away (exposed)
	b.SetFEN("r1bq1rk1/ppppbppp/2n2n2/4p3/2BPP3/8/PPP2PPP/RNBQK2R w KQ - 0 5")
	exposedScore := b.Evaluate()

	t.Logf("Shielded king: %d, Exposed king: %d, diff: %d", shieldedScore, exposedScore, shieldedScore-exposedScore)
}

func TestPawnHashCaching(t *testing.T) {
	var b Board
	pt := NewPawnTable(1)
	b.PawnTable = pt

	b.SetFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")

	// First call should miss
	entry1 := b.probePawnEval()

	// Second call with same pawn hash should hit
	entry2, ok := pt.Probe(b.PawnHashKey)
	if !ok {
		t.Error("Expected pawn hash table hit on second probe")
	}

	if entry1.WhiteMG != entry2.WhiteMG || entry1.WhiteEG != entry2.WhiteEG {
		t.Errorf("Cached entry mismatch: first=(%d,%d), second=(%d,%d)",
			entry1.WhiteMG, entry1.WhiteEG, entry2.WhiteMG, entry2.WhiteEG)
	}
}

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
	// Include pawns so endgame scale factor doesn't zero out the eval
	var b Board

	// Knight on e4 (central) with a pawn to avoid insufficient material scaling
	b.SetFEN("4k3/8/8/8/4N3/8/4P3/4K3 w - - 0 1")
	centralScore := b.Evaluate()

	// Knight on a1 with same pawn
	b.SetFEN("4k3/8/8/8/8/8/4P3/N3K3 w - - 0 1")
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
	// Friendly king near passer, enemy king far, so king proximity helps
	b.SetFEN("k7/8/4P3/8/4K3/8/8/8 w - - 0 1")
	advancedScore := b.Evaluate()

	// Passed pawn on rank 3 (less advanced)
	b.SetFEN("k7/8/8/8/4K3/4P3/8/8 w - - 0 1")
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

func TestBishopPair(t *testing.T) {
	var b Board

	// Two bishops vs one bishop + knight (similar material value)
	b.SetFEN("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1")
	pairScore := b.Evaluate()

	b.SetFEN("4k3/8/8/8/8/8/8/2B1KN2 w - - 0 1")
	singleScore := b.Evaluate()

	// Bishop pair should give a bonus
	if pairScore <= singleScore {
		t.Errorf("Bishop pair (%d) should score higher than bishop+knight (%d)", pairScore, singleScore)
	}
	t.Logf("Bishop pair: %d, Bishop+Knight: %d, diff: %d", pairScore, singleScore, pairScore-singleScore)
}

func TestKnightOutpost(t *testing.T) {
	var b Board

	// Knight on d5 outpost: no black pawns on c or e files to attack it
	b.SetFEN("4k3/pp4pp/8/3N4/8/8/PP4PP/4K3 w - - 0 1")
	outpostScore := b.Evaluate()

	// Knight on d5 but enemy pawns on c6 and e6 can attack it
	b.SetFEN("4k3/pp4pp/2p1p3/3N4/8/8/PP4PP/4K3 w - - 0 1")
	noOutpostScore := b.Evaluate()

	if outpostScore <= noOutpostScore {
		t.Errorf("Knight outpost (%d) should score higher than non-outpost (%d)", outpostScore, noOutpostScore)
	}
	t.Logf("Outpost: %d, Non-outpost: %d, diff: %d", outpostScore, noOutpostScore, outpostScore-noOutpostScore)
}

func TestKnightOutpostSupported(t *testing.T) {
	var b Board

	// Knight on d5 outpost supported by pawn on c4 or e4
	b.SetFEN("4k3/pp4pp/8/3N4/4P3/8/PP4PP/4K3 w - - 0 1")
	supportedScore := b.Evaluate()

	// Knight on d5 outpost unsupported (pawns on a2, b2, not supporting)
	b.SetFEN("4k3/pp4pp/8/3N4/8/8/PP4PP/4K3 w - - 0 1")
	unsupportedScore := b.Evaluate()

	if supportedScore <= unsupportedScore {
		t.Errorf("Supported outpost (%d) should score higher than unsupported (%d)", supportedScore, unsupportedScore)
	}
	t.Logf("Supported: %d, Unsupported: %d, diff: %d", supportedScore, unsupportedScore, supportedScore-unsupportedScore)
}

func TestRookOpenFile(t *testing.T) {
	var b Board

	// Rook on open e-file (no pawns on e-file for either side)
	b.SetFEN("4k3/pppp1ppp/8/8/8/8/PPPP1PPP/4R1K1 w - - 0 1")
	openScore := b.Evaluate()

	// Rook on closed e-file (both sides have e-pawns)
	b.SetFEN("4k3/pppppppp/8/8/8/8/PPPPPPPP/4R1K1 w - - 0 1")
	closedScore := b.Evaluate()

	if openScore <= closedScore {
		t.Errorf("Rook on open file (%d) should score higher than closed (%d)", openScore, closedScore)
	}
	t.Logf("Open: %d, Closed: %d, diff: %d", openScore, closedScore, openScore-closedScore)
}

func TestRookSemiOpenFile(t *testing.T) {
	var b Board

	// Semi-open: Rook on e1, white pawn on d2 (not e-file), black pawn on e7
	// e-file is semi-open for white rook (no white pawn, enemy pawn present)
	b.SetFEN("4k3/4p3/8/8/8/8/3P4/4R1K1 w - - 0 1")
	semiOpenScore := b.Evaluate()

	// Closed: Rook on e1, white pawn on e2, black pawn on d7
	// e-file is closed for white rook (white pawn on it)
	b.SetFEN("4k3/3p4/8/8/8/8/4P3/4R1K1 w - - 0 1")
	closedScore := b.Evaluate()

	if semiOpenScore <= closedScore {
		t.Errorf("Rook on semi-open file (%d) should score higher than closed (%d)", semiOpenScore, closedScore)
	}
	t.Logf("Semi-open: %d, Closed: %d, diff: %d", semiOpenScore, closedScore, semiOpenScore-closedScore)
}

func TestRookOn7thRank(t *testing.T) {
	var b Board

	// White rook on 7th rank (b7)
	b.SetFEN("4k3/1R6/8/8/8/8/8/4K3 w - - 0 1")
	seventhScore := b.Evaluate()

	// White rook on 5th rank (b5) — same file, different rank
	b.SetFEN("4k3/8/8/1R6/8/8/8/4K3 w - - 0 1")
	fifthScore := b.Evaluate()

	if seventhScore <= fifthScore {
		t.Errorf("Rook on 7th rank (%d) should score higher than 5th (%d)", seventhScore, fifthScore)
	}
	t.Logf("7th rank: %d, 5th rank: %d, diff: %d", seventhScore, fifthScore, seventhScore-fifthScore)
}

func TestBishopOpenPosition(t *testing.T) {
	var b Board

	// Bishop with few pawns (open position)
	b.SetFEN("4k3/8/8/8/8/8/4P3/4KB2 w - - 0 1")
	openScore := b.Evaluate()

	// Bishop with many pawns (closed position)
	b.SetFEN("4k3/pppppppp/8/8/8/8/PPPPPPPP/4KB2 w - - 0 1")
	closedScore := b.Evaluate()

	// Remove pawn material influence: open position has fewer pawns
	// so we focus on the bishop bonus. The open position should
	// give a higher per-bishop bonus even though it has less material.
	t.Logf("Open (few pawns): %d, Closed (many pawns): %d", openScore, closedScore)
	// We can't directly compare due to material differences, so just log.
	// The BishopOpenPosition bonus adds 3 * (16 - totalPawns) per bishop.
}

func TestPassedPawnNotBlocked(t *testing.T) {
	var b Board

	// White passed pawn on e5, square ahead (e6) is empty
	b.SetFEN("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1")
	freeScore := b.Evaluate()

	// White passed pawn on e5, blocked by a piece on e6
	b.SetFEN("4k3/8/4n3/4P3/8/8/8/4K3 w - - 0 1")
	blockedScore := b.Evaluate()

	if freeScore <= blockedScore {
		t.Errorf("Free passed pawn (%d) should score higher than blocked (%d)", freeScore, blockedScore)
	}
	t.Logf("Free: %d, Blocked: %d, diff: %d", freeScore, blockedScore, freeScore-blockedScore)
}

func TestRookBehindPassedPawn(t *testing.T) {
	var b Board

	// White rook on e1 behind white passed pawn on e5
	b.SetFEN("4k3/8/8/4P3/8/8/8/4KR2 w - - 0 1")
	behindScore := b.Evaluate()

	// White rook on e7 in front of white passed pawn on e5 (not behind)
	b.SetFEN("4k3/4R3/8/4P3/8/8/8/4K3 w - - 0 1")
	frontScore := b.Evaluate()

	t.Logf("Rook behind passer: %d, Rook in front: %d, diff: %d", behindScore, frontScore, behindScore-frontScore)
}

func TestSafeMobility(t *testing.T) {
	var b Board

	// Knight with lots of safe squares
	b.SetFEN("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1")
	freeScore := b.Evaluate()

	// Knight surrounded by friendly pieces (fewer safe squares)
	b.SetFEN("4k3/8/8/3PBP2/3PNP2/3PBP2/8/4K3 w - - 0 1")
	crammedScore := b.Evaluate()

	// The free knight should have higher mobility (though crammed has more material)
	// We'll check that eval doesn't count friendly squares as mobility
	t.Logf("Free knight: %d, Crammed knight: %d", freeScore, crammedScore)
}

func TestOutpostMask(t *testing.T) {
	// Verify OutpostMask for a few key squares

	// White perspective: e4 (square 28) should include d4-d7 and f4-f7 on adjacent files
	sq := NewSquare(4, 3) // e4
	mask := OutpostMask[White][sq]

	// Should include d4, d5, d6, d7 (adjacent file d, ranks 3-7)
	for r := 3; r < 8; r++ {
		dSq := NewSquare(3, r)
		if !mask.IsSet(dSq) {
			t.Errorf("OutpostMask[White][e4] should include %s", dSq)
		}
		fSq := NewSquare(5, r)
		if !mask.IsSet(fSq) {
			t.Errorf("OutpostMask[White][e4] should include %s", fSq)
		}
	}

	// Should NOT include ranks below (d3, d2, d1, etc.)
	for r := 0; r < 3; r++ {
		dSq := NewSquare(3, r)
		if mask.IsSet(dSq) {
			t.Errorf("OutpostMask[White][e4] should NOT include %s", dSq)
		}
	}

	// Should NOT include e-file itself
	for r := 0; r < 8; r++ {
		eSq := NewSquare(4, r)
		if mask.IsSet(eSq) {
			t.Errorf("OutpostMask[White][e4] should NOT include %s (same file)", eSq)
		}
	}

	// Black perspective: e5 (square 36) should include d5-d0 and f5-f0
	sq = NewSquare(4, 4) // e5
	mask = OutpostMask[Black][sq]
	for r := 4; r >= 0; r-- {
		dSq := NewSquare(3, r)
		if !mask.IsSet(dSq) {
			t.Errorf("OutpostMask[Black][e5] should include %s", dSq)
		}
		fSq := NewSquare(5, r)
		if !mask.IsSet(fSq) {
			t.Errorf("OutpostMask[Black][e5] should include %s", fSq)
		}
	}

	// Edge case: a-file square should only have b-file in mask
	sq = NewSquare(0, 3) // a4
	mask = OutpostMask[White][sq]
	for r := 3; r < 8; r++ {
		bSq := NewSquare(1, r)
		if !mask.IsSet(bSq) {
			t.Errorf("OutpostMask[White][a4] should include %s", bSq)
		}
	}
	// No file to the left of a-file
	if mask.Count() != 5 {
		t.Errorf("OutpostMask[White][a4] should have 5 squares (b4-b8), got %d", mask.Count())
	}
}

func TestChebyshevDistance(t *testing.T) {
	tests := []struct {
		sq1, sq2 Square
		want     int
	}{
		{NewSquare(0, 0), NewSquare(0, 0), 0}, // same square
		{NewSquare(0, 0), NewSquare(1, 0), 1}, // adjacent horizontal
		{NewSquare(0, 0), NewSquare(0, 1), 1}, // adjacent vertical
		{NewSquare(0, 0), NewSquare(1, 1), 1}, // adjacent diagonal
		{NewSquare(0, 0), NewSquare(7, 7), 7}, // a1-h8
		{NewSquare(3, 3), NewSquare(6, 5), 3}, // d4-g6
	}
	for _, tt := range tests {
		got := chebyshevDistance(tt.sq1, tt.sq2)
		if got != tt.want {
			t.Errorf("chebyshevDistance(%s, %s) = %d, want %d", tt.sq1, tt.sq2, got, tt.want)
		}
	}
}

func TestPassedPawnKingProximity(t *testing.T) {
	var b Board

	// White king close to passed pawn on e5
	b.SetFEN("7k/8/8/4P3/8/4K3/8/8 w - - 0 1")
	closeScore := b.Evaluate()

	// White king far from passed pawn on e5
	b.SetFEN("7k/8/8/4P3/8/8/8/K7 w - - 0 1")
	farScore := b.Evaluate()

	if closeScore <= farScore {
		t.Errorf("Friendly king close to passer (%d) should score higher than far (%d)", closeScore, farScore)
	}
	t.Logf("King close: %d, King far: %d, diff: %d", closeScore, farScore, closeScore-farScore)
}

func TestPassedPawnEnemyKingFar(t *testing.T) {
	var b Board

	// Enemy king far from white passed pawn on e5
	b.SetFEN("k7/8/8/4P3/8/4K3/8/8 w - - 0 1")
	farScore := b.Evaluate()

	// Enemy king close to white passed pawn on e5
	b.SetFEN("4k3/8/8/4P3/8/4K3/8/8 w - - 0 1")
	closeScore := b.Evaluate()

	if farScore <= closeScore {
		t.Errorf("Enemy king far from passer (%d) should score higher than close (%d)", farScore, closeScore)
	}
	t.Logf("Enemy far: %d, Enemy close: %d, diff: %d", farScore, closeScore, farScore-closeScore)
}

func TestPassedPawnFreePath(t *testing.T) {
	var b Board

	// Passed pawn on e5, entire path (e6, e7, e8) clear
	b.SetFEN("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1")
	freeScore := b.Evaluate()

	// Passed pawn on e5, path blocked by piece on e7 (not directly ahead)
	b.SetFEN("4k3/4n3/8/4P3/8/8/8/4K3 w - - 0 1")
	blockedPathScore := b.Evaluate()

	if freeScore <= blockedPathScore {
		t.Errorf("Free path passer (%d) should score higher than blocked path (%d)", freeScore, blockedPathScore)
	}
	t.Logf("Free path: %d, Blocked path: %d, diff: %d", freeScore, blockedPathScore, freeScore-blockedPathScore)
}

func TestPassedPawnProtected(t *testing.T) {
	var b Board

	// Test evaluatePassedPawns directly to isolate the protection feature
	// e5 passer protected by f4 pawn (f4 NorthWest = e5)
	b.SetFEN("k7/8/8/4P3/5P2/8/8/K7 w - - 0 1")
	if b.PawnTable == nil {
		b.PawnTable = NewPawnTable(1)
	}
	entry := b.probePawnEval()
	protMG, protEG := b.evaluatePassedPawns(White, &entry)

	// e5 passer, f3 pawn does NOT defend e5 (f3 attacks e4/g4)
	b.SetFEN("k7/8/8/4P3/8/5P2/8/K7 w - - 0 1")
	entry = b.probePawnEval()
	unprotMG, unprotEG := b.evaluatePassedPawns(White, &entry)

	if protMG+protEG <= unprotMG+unprotEG {
		t.Errorf("Protected passer (MG=%d, EG=%d) should score higher than unprotected (MG=%d, EG=%d)",
			protMG, protEG, unprotMG, unprotEG)
	}
	t.Logf("Protected: MG=%d EG=%d, Unprotected: MG=%d EG=%d", protMG, protEG, unprotMG, unprotEG)
}

func TestPassedPawnConnected(t *testing.T) {
	var b Board

	// Connected passers on d5 and e5 (adjacent files, both passed)
	b.SetFEN("4k3/8/8/3PP3/8/8/8/4K3 w - - 0 1")
	connectedScore := b.Evaluate()

	// Separated passers on a5 and h5 (far apart, not connected)
	b.SetFEN("4k3/8/8/P6P/8/8/8/4K3 w - - 0 1")
	separatedScore := b.Evaluate()

	if connectedScore <= separatedScore {
		t.Errorf("Connected passers (%d) should score higher than separated (%d)", connectedScore, separatedScore)
	}
	t.Logf("Connected: %d, Separated: %d, diff: %d", connectedScore, separatedScore, connectedScore-separatedScore)
}

func TestEvalCalibration(t *testing.T) {
	positions := []struct {
		name    string
		fen     string
		humanCP int // approximate White-relative human eval
	}{
		{"Move19_BishopPairPlusPawn", "r1bqr1k1/ppnp1ppp/2p1p3/2Pn4/P1NP4/2P2BP1/2Q1PP1P/1RB2RK1 b - - 4 19", 125},
		{"Move24_EqualMaterial", "r3r1k1/pp3ppp/1np2n2/4b2b/1P2P2N/P1NRBPP1/6BP/R5K1 w - - 3 24", 0},
		{"Move34_BNvsB", "5k2/1p3ppp/1pp5/4b3/1P2P3/P6P/5KB1/3N4 w - - 1 34", 75},
	}

	for _, pos := range positions {
		var b Board
		b.SetFEN(pos.fen)
		score := b.Evaluate()
		t.Logf("%s: eval=%d, human~%d, error=%d", pos.name, score, pos.humanCP, score-pos.humanCP)
	}

	// Starting position should be near zero
	var b Board
	b.Reset()
	startScore := b.Evaluate()
	t.Logf("Starting position: %d", startScore)
	if startScore > 50 || startScore < -50 {
		t.Errorf("Starting position eval %d too far from 0", startScore)
	}
}

func TestEvalCache(t *testing.T) {
	// Verify cached eval matches fresh eval
	var b Board
	b.Reset()
	b.EvalTable = NewEvalTable(1)

	score1 := b.Evaluate()
	score2 := b.Evaluate() // should hit cache
	if score1 != score2 {
		t.Errorf("Cached eval %d != fresh eval %d", score2, score1)
	}

	// After a move, eval should change
	moves := b.GenerateLegalMoves()
	b.MakeMove(moves[0])
	score3 := b.Evaluate()
	_ = score3 // just verify no panic

	b.UnmakeMove(moves[0])
	score4 := b.Evaluate() // should hit original cache entry
	if score4 != score1 {
		t.Errorf("Eval after unmake %d != original %d", score4, score1)
	}
}

func TestPawnBackward(t *testing.T) {
	var b Board
	// e3 pawn is backward: d-file empty, f-file has pawn on f5 (ahead),
	// stop square e4 controlled by Black d5 pawn
	b.SetFEN("4k3/8/8/3p1P2/8/4P3/8/4K3 w - - 0 1")
	backwardScore := b.Evaluate()

	// Same pawns but d5 removed — e3 is no longer backward
	b.SetFEN("4k3/8/8/5P2/8/4P3/8/4K3 w - - 0 1")
	freeScore := b.Evaluate()

	if backwardScore >= freeScore {
		t.Errorf("Backward pawn (%d) should score lower than free (%d)", backwardScore, freeScore)
	}
	t.Logf("Backward: %d, Free: %d, diff: %d", backwardScore, freeScore, backwardScore-freeScore)
}

func TestTrappedRook(t *testing.T) {
	var b Board

	// Black king on f8, rook on h8 — rook is trapped
	b.SetFEN("r2q1k1r/pppb1ppp/2nbpn2/8/2QP4/2N2NP1/PP2PPBP/R1B2RK1 w - - 3 10")
	trapped := b.Evaluate()

	// Same but Black has castled (king g8, rook f8) — rook is NOT trapped
	b.SetFEN("r2q1rk1/pppb1ppp/2nbpn2/8/2QP4/2N2NP1/PP2PPBP/R1B2RK1 w - - 3 10")
	castled := b.Evaluate()

	// White should prefer facing the trapped-rook position (higher eval)
	if trapped <= castled {
		t.Errorf("Trapped rook position (%d) should score higher for White than castled (%d)", trapped, castled)
	}
	t.Logf("Trapped rook: %d, Castled: %d, diff: %d", trapped, castled, trapped-castled)
}

func TestCastlingPreferred(t *testing.T) {
	var b Board
	b.SetFEN("r2qk2r/pppb1ppp/2nbpn2/8/2QP4/2N2NP1/PP2PPBP/R1B2RK1 b kq - 2 9")
	tt := NewTranspositionTable(16)
	move, _ := b.SearchWithTT(10, 0, tt)
	// Engine should NOT play Kf8 (e8f8) which traps the h-rook
	if move.From() == NewSquare(4, 7) && move.To() == NewSquare(5, 7) && move.Flags() == FlagNone {
		t.Errorf("Engine played Kf8 which traps the rook — expected O-O or other developing move, got %s", move.String())
	}
	t.Logf("Best move: %s (flags=%d)", move.String(), move.Flags())
}

// --- Endgame improvement tests ---

func TestInsufficientMaterialKvK(t *testing.T) {
	var b Board
	b.SetFEN("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
	score := b.Evaluate()
	if score != 0 {
		t.Errorf("KvK eval = %d, expected 0", score)
	}
}

func TestInsufficientMaterialKNvK(t *testing.T) {
	var b Board
	// White has a knight, black has nothing — should be ~0
	b.SetFEN("4k3/8/8/8/8/8/8/4KN2 w - - 0 1")
	score := b.Evaluate()
	if score < -5 || score > 5 {
		t.Errorf("KNvK eval = %d, expected ~0", score)
	}
}

func TestInsufficientMaterialKBvK(t *testing.T) {
	var b Board
	b.SetFEN("4k3/8/8/8/8/8/8/4KB2 w - - 0 1")
	score := b.Evaluate()
	if score < -5 || score > 5 {
		t.Errorf("KBvK eval = %d, expected ~0", score)
	}
}

func TestInsufficientMaterialKNNvK(t *testing.T) {
	var b Board
	b.SetFEN("4k3/8/8/8/8/8/8/3NKN2 w - - 0 1")
	score := b.Evaluate()
	if score < -5 || score > 5 {
		t.Errorf("KNNvK eval = %d, expected ~0", score)
	}
}

func TestKBvKBSameColor(t *testing.T) {
	var b Board
	// White bishop on c1 (dark), black bishop on f8 (dark) — same color
	b.SetFEN("5b1k/8/8/8/8/8/8/2B1K3 w - - 0 1")
	// Verify setup
	wOnLight := b.Pieces[WhiteBishop]&LightSquares != 0
	bOnLight := b.Pieces[BlackBishop]&LightSquares != 0
	if wOnLight != bOnLight {
		t.Fatalf("Setup error: bishops on different colors (wLight=%v, bLight=%v)", wOnLight, bOnLight)
	}
	score := b.Evaluate()
	if score < -5 || score > 5 {
		t.Errorf("KBvKB same-color eval = %d, expected ~0", score)
	}
}

func TestKBvKBOppositeColor(t *testing.T) {
	// In bare KBvKB, both sides get scale=0 from the per-side "<=1 minor" check
	// regardless of bishop color. The same-color bishop check is an additional
	// path. Here we just verify that opposite-color bishops are detected correctly.
	var b Board

	// White bishop on f1 (light), black bishop on f8 (dark) — opposite colors
	b.SetFEN("4kb2/8/8/8/8/8/8/4KB2 w - - 0 1")
	wOnLight := b.Pieces[WhiteBishop]&LightSquares != 0
	bOnLight := b.Pieces[BlackBishop]&LightSquares != 0
	if wOnLight == bOnLight {
		t.Fatalf("Setup error: bishops on same color (wLight=%v, bLight=%v)", wOnLight, bOnLight)
	}

	// Both sides still get scale=0 from the per-side check (<=1 minor, no pawns)
	// The same-color bishop path doesn't activate for opposite-color bishops
	score := b.Evaluate()
	t.Logf("KBvKB opposite-color: score=%d (both sides scale=0 from per-side check)", score)
}

func TestKQvKEval(t *testing.T) {
	var b Board
	b.SetFEN("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
	score := b.Evaluate()
	if score < 500 {
		t.Errorf("KQvK eval = %d, expected strongly favoring White (>500)", score)
	}
	t.Logf("KQvK eval = %d", score)
}

func TestKRvKNScaling(t *testing.T) {
	var b Board
	// KR vs KN — should be heavily scaled (wScale=16 since rook side has no pawns and opponent has a minor)
	b.SetFEN("4k3/8/8/8/8/8/8/3nKR2 w - - 0 1")
	score := b.Evaluate()
	wScale, _ := b.endgameScale()
	if wScale != 16 {
		t.Errorf("KRvKN wScale = %d, expected 16", wScale)
	}
	// Score should be small due to heavy scaling
	if score > 100 {
		t.Errorf("KRvKN eval = %d, expected small (<100) due to scaling", score)
	}
	t.Logf("KRvKN eval = %d, wScale = %d", score, wScale)
}

func TestHalfmoveClockScaling(t *testing.T) {
	// Use separate Board objects to avoid eval cache interference
	// (halfmove clock is not part of Zobrist hash)
	var b1 Board
	b1.SetFEN("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
	baseScore := b1.Evaluate()

	var b2 Board
	b2.SetFEN("4k3/8/8/8/8/8/8/3QK3 w - - 80 1")
	scaledScore := b2.Evaluate()

	// At clock=80, score should be 20% of base
	if scaledScore >= baseScore/2 {
		t.Errorf("Halfmove 80 score (%d) should be much less than half of base (%d)", scaledScore, baseScore)
	}
	if scaledScore <= 0 {
		t.Errorf("Halfmove 80 score (%d) should still be positive", scaledScore)
	}
	t.Logf("Base score: %d, Clock=80 score: %d (%.0f%%)", baseScore, scaledScore, float64(scaledScore)/float64(baseScore)*100)
}

func TestBadBishop(t *testing.T) {
	var b Board

	// Light-squared bishop with pawns on light squares (bad bishop)
	// Bishop on f1 (light), pawns on e2, g2 (both light squares)
	b.SetFEN("4k3/8/8/8/8/8/4P1P1/4KB2 w - - 0 1")
	badScore := b.Evaluate()

	// Light-squared bishop with pawns on dark squares (good bishop)
	// Bishop on f1 (light), pawns on d2, f2 (both dark squares)
	b.SetFEN("4k3/8/8/8/8/8/3P1P2/4KB2 w - - 0 1")
	goodScore := b.Evaluate()

	if badScore >= goodScore {
		t.Errorf("Bad bishop (%d) should score lower than good bishop (%d)", badScore, goodScore)
	}
	t.Logf("Bad bishop: %d, Good bishop: %d, diff: %d", badScore, goodScore, badScore-goodScore)
}

func TestDoubledRooks(t *testing.T) {
	var b Board

	// Verify doubled rooks bonus is applied by checking endgameScale indirectly.
	// Use positions with pawns to keep files closed and minimize other eval differences.
	// Two rooks on the d-file (d1 and d3), pawns blocking open-file bonuses.
	b.SetFEN("4k3/pppppppp/8/8/8/3R4/PPPPPPPP/3R1K2 w - - 0 1")
	if b.PawnTable == nil {
		b.PawnTable = NewPawnTable(1)
	}
	entry := b.probePawnEval()
	doubledMG, doubledEG := b.evaluatePieces(White, &entry)

	// Two rooks on different files (d1 and e3), same pawn structure.
	b.SetFEN("4k3/pppppppp/8/8/8/4R3/PPPPPPPP/3R1K2 w - - 0 1")
	entry = b.probePawnEval()
	separateMG, separateEG := b.evaluatePieces(White, &entry)

	// The doubled rooks bonus (12 MG + 18 EG = 30) should be present
	doubledTotal := doubledMG + doubledEG
	separateTotal := separateMG + separateEG
	diff := doubledTotal - separateTotal
	t.Logf("Doubled: MG=%d EG=%d total=%d, Separate: MG=%d EG=%d total=%d, diff=%d",
		doubledMG, doubledEG, doubledTotal, separateMG, separateEG, separateTotal, diff)

	// The doubled bonus is 12+18=30. Other factors may vary, but the bonus should
	// contribute measurably. Check that the difference is within expected range.
	if diff < DoubledRooksMG+DoubledRooksEG-15 {
		t.Errorf("Doubled rook bonus not detected: diff=%d, expected at least %d",
			diff, DoubledRooksMG+DoubledRooksEG-15)
	}
}

func TestOCBDrawishness(t *testing.T) {
	var b Board

	// Opposite-colored bishops with pawns — should get OCBScale (64)
	// White bishop on f1 (light), black bishop on c8 (light) — same color, NOT OCB
	// White bishop on f1 (light), black bishop on f8 (dark) — opposite color, IS OCB
	b.SetFEN("5b1k/4p3/8/8/8/8/4P3/4KB2 w - - 0 1")
	wScale, bScale := b.endgameScale()
	wBishopLight := b.Pieces[WhiteBishop]&LightSquares != 0
	bBishopLight := b.Pieces[BlackBishop]&LightSquares != 0
	if wBishopLight == bBishopLight {
		t.Fatalf("Setup error: bishops on same color (wLight=%v, bLight=%v)", wBishopLight, bBishopLight)
	}
	if wScale != OCBScale || bScale != OCBScale {
		t.Errorf("OCB with pawns: wScale=%d, bScale=%d, expected both %d", wScale, bScale, OCBScale)
	}

	// Same-colored bishops with pawns — should stay at 128
	// White bishop on c1 (dark), black bishop on f8 (dark) — same color
	b.SetFEN("5b1k/4p3/8/8/8/8/4P3/2B1K3 w - - 0 1")
	wScale, bScale = b.endgameScale()
	wBishopLight = b.Pieces[WhiteBishop]&LightSquares != 0
	bBishopLight = b.Pieces[BlackBishop]&LightSquares != 0
	if wBishopLight != bBishopLight {
		t.Fatalf("Setup error: bishops on different colors (wLight=%v, bLight=%v)", wBishopLight, bBishopLight)
	}
	if wScale != 128 || bScale != 128 {
		t.Errorf("Same-color bishops with pawns: wScale=%d, bScale=%d, expected both 128", wScale, bScale)
	}
}

func TestOCBPureBishopOnly(t *testing.T) {
	var b Board

	// OCB with rooks present — should NOT trigger OCB scaling (not pure bishop endgame)
	b.SetFEN("r4b1k/4p3/8/8/8/8/4P3/R3KB2 w - - 0 1")
	wScale, bScale := b.endgameScale()
	if wScale == OCBScale || bScale == OCBScale {
		t.Errorf("OCB with rooks: wScale=%d, bScale=%d, should NOT be %d (rooks present)", wScale, bScale, OCBScale)
	}

	// OCB with knights present — should NOT trigger OCB scaling
	b.SetFEN("n4b1k/4p3/8/8/8/8/4P3/N3KB2 w - - 0 1")
	wScale, bScale = b.endgameScale()
	if wScale == OCBScale || bScale == OCBScale {
		t.Errorf("OCB with knights: wScale=%d, bScale=%d, should NOT be %d (knights present)", wScale, bScale, OCBScale)
	}
}

func TestEndgameScale(t *testing.T) {
	tests := []struct {
		name       string
		fen        string
		wantWScale int
		wantBScale int
	}{
		// Bare king vs bare king
		{"KvK", "4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0, 0},
		// KN vs K - insufficient material
		{"KNvK", "4k3/8/8/8/8/8/8/3NK3 w - - 0 1", 0, 0},
		// KB vs K - insufficient material
		{"KBvK", "4k3/8/8/8/8/8/8/3BK3 w - - 0 1", 0, 0},
		// K vs KN - insufficient for both
		{"KvKN", "3nk3/8/8/8/8/8/8/4K3 w - - 0 1", 0, 0},
		// KNN vs K - can't force mate
		{"KNNvK", "4k3/8/8/8/8/8/8/2NNK3 w - - 0 1", 0, 0},
		// KR vs KN - drawish (wScale=16)
		{"KRvKN", "3nk3/8/8/8/8/8/8/3RK3 w - - 0 1", 16, 0},
		// KR vs KB - drawish (wScale=16)
		{"KRvKB", "3bk3/8/8/8/8/8/8/3RK3 w - - 0 1", 16, 0},
		// K vs KR - drawish for black (bScale=16 since black has rook, white has minor=0 so bScale=16 doesn't trigger)
		{"KNvKR", "3rk3/8/8/8/8/8/8/3NK3 w - - 0 1", 0, 16},
		// KQ vs K - can win (queen is a major, white has pawns=0 but majors>0)
		{"KQvK", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 128, 0},
		// KR vs K - can win
		{"KRvK", "4k3/8/8/8/8/8/8/3RK3 w - - 0 1", 128, 0},
		// KBB vs K - can win (2 minors, not KNN)
		{"KBBvK", "4k3/8/8/8/8/8/8/2BBK3 w - - 0 1", 128, 0},
		// KBN vs K - can win (2 minors, not KNN)
		{"KBNvK", "4k3/8/8/8/8/8/8/2BNK3 w - - 0 1", 128, 0},
		// With pawns - both sides normal
		{"KPvKP", "4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1", 128, 128},
		// KN+P vs K - white has pawn so not insufficient
		{"KNPvK", "4k3/8/8/8/8/8/4P3/3NK3 w - - 0 1", 128, 0},
		// OCB with pawns - drawish
		{"OCB+pawns", "5b1k/4p3/8/8/8/8/4P3/4KB2 w - - 0 1", 64, 64},
		// Same-color bishops with pawns - normal
		{"SCB+pawns", "5b1k/4p3/8/8/8/8/4P3/2B1K3 w - - 0 1", 128, 128},
	}

	for _, tt := range tests {
		var b Board
		b.SetFEN(tt.fen)
		wScale, bScale := b.endgameScale()
		if wScale != tt.wantWScale || bScale != tt.wantBScale {
			t.Errorf("%s: endgameScale() = (%d, %d), want (%d, %d)",
				tt.name, wScale, bScale, tt.wantWScale, tt.wantBScale)
		}
	}
}

func TestEndgameKingDistance(t *testing.T) {
	var b Board
	// KR vs K, enemy king on edge — uses rook to test endgame king distance
	// without queen safe check asymmetry distorting the comparison
	b.SetFEN("k7/8/8/8/8/8/8/3RK3 w - - 0 1")
	edgeScore := b.Evaluate()

	// KR vs K, enemy king in center
	b.SetFEN("4k3/8/8/8/8/8/8/3RK3 w - - 0 1")
	centerScore := b.Evaluate()

	// Enemy on edge should score higher (easier to mate)
	if edgeScore <= centerScore {
		t.Errorf("Enemy king on edge (%d) should score higher than center (%d)", edgeScore, centerScore)
	}
	t.Logf("Edge: %d, Center: %d, diff: %d", edgeScore, centerScore, edgeScore-centerScore)
}

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

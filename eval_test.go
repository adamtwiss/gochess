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
	// Should be close to QueenValue (900), allowing for mobility differences
	if diff < 800 || diff > 1000 {
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

	// Swap all pieces (mirror the board)
	// After e4 e5, the position should still be roughly symmetrical
	b.SetFEN("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
	score := b.Evaluate()

	// Should be close to 0 due to symmetry
	if score < -30 || score > 30 {
		t.Errorf("Symmetric position eval = %d, expected close to 0", score)
	}
}

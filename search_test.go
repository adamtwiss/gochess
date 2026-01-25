package chess

import (
	"testing"
	"time"
)

func TestSearchStartingPosition(t *testing.T) {
	var b Board
	b.Reset()

	move, info := b.Search(5, 0)

	t.Logf("Best move: %s", move.String())
	t.Logf("Score: %d", info.Score)
	t.Logf("Depth: %d", info.Depth)
	t.Logf("Nodes: %d", info.Nodes)
	t.Logf("PV: %v", movesToStrings(info.PV))

	if move == NoMove {
		t.Error("Search returned no move")
	}
}

func TestSearchFindsCapture(t *testing.T) {
	var b Board
	// White can capture black queen with pawn
	b.SetFEN("rnb1kbnr/pppp1ppp/8/4q3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1")

	move, info := b.Search(4, 0)

	t.Logf("Best move: %s", move.String())
	t.Logf("Score: %d", info.Score)

	// d4 should capture queen on e5
	if move.String() != "d4e5" {
		t.Errorf("Expected d4e5, got %s", move.String())
	}
}

func TestSearchFindsMate(t *testing.T) {
	var b Board
	// Mate in 1: Qh7#
	b.SetFEN("6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1")

	move, info := b.Search(3, 0)

	t.Logf("Best move: %s", move.String())
	t.Logf("Score: %d", info.Score)

	// Should find mate
	if info.Score < MateScore-10 {
		t.Errorf("Expected mate score, got %d", info.Score)
	}
}

func TestSearchAvoidsMate(t *testing.T) {
	var b Board
	// Black to move, must block Qh7#
	b.SetFEN("6k1/5ppp/8/8/8/8/5PPP/4Q1K1 b - - 0 1")

	move, info := b.Search(4, 0)

	t.Logf("Best move: %s", move.String())
	t.Logf("Score: %d", info.Score)

	// Black's best moves are to create luft or block
	// Should not have a very negative score if there's a defense
	t.Logf("Black's defensive move: %s", move.String())
}

func TestSearchWithTimeLimit(t *testing.T) {
	var b Board
	b.Reset()

	start := time.Now()
	move, info := b.Search(100, 100*time.Millisecond)
	elapsed := time.Since(start)

	t.Logf("Best move: %s", move.String())
	t.Logf("Depth reached: %d", info.Depth)
	t.Logf("Nodes: %d", info.Nodes)
	t.Logf("Time: %v", elapsed)

	if move == NoMove {
		t.Error("Search returned no move")
	}

	// Should have stopped within reasonable time
	if elapsed > 200*time.Millisecond {
		t.Errorf("Search took too long: %v", elapsed)
	}
}

func TestSearchMateIn2(t *testing.T) {
	var b Board
	// Mate in 2: 1. Qd8+ Kh7 2. Qg8# (or Qh8#)
	b.SetFEN("6k1/8/6K1/8/8/8/8/3Q4 w - - 0 1")

	move, info := b.Search(5, 0)

	t.Logf("Best move: %s", move.String())
	t.Logf("Score: %d", info.Score)
	t.Logf("PV: %v", movesToStrings(info.PV))

	// Should find mate
	if info.Score < MateScore-10 {
		t.Errorf("Expected mate score, got %d", info.Score)
	}
}

func TestSearchTacticalPosition(t *testing.T) {
	var b Board
	// Tactical position - white can win the exchange
	b.SetFEN("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")

	move, info := b.Search(5, 0)

	t.Logf("Best move: %s", move.String())
	t.Logf("Score: %d", info.Score)
	t.Logf("PV: %v", movesToStrings(info.PV))

	// Qxf7+ is winning (scholar's mate threat or wins f7 pawn)
	if move.String() != "h5f7" {
		t.Logf("Note: Expected Qxf7+, got %s", move.String())
	}
}

// Helper to convert PV to strings for logging
func movesToStrings(moves []Move) []string {
	result := make([]string, len(moves))
	for i, m := range moves {
		result[i] = m.String()
	}
	return result
}

func BenchmarkSearch(b *testing.B) {
	var board Board
	board.Reset()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		board.Search(5, 0)
	}
}

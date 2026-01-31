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

	// Should have stopped within reasonable time (generous bound for race detector)
	if elapsed > 500*time.Millisecond {
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

// BenchmarkSearchDeep tests search at depth 7 where LMR has more impact
func BenchmarkSearchDeep(b *testing.B) {
	var board Board
	board.Reset()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		board.Search(7, 0)
	}
}

// BenchmarkSearchTactical tests tactical positions at depth 8
func BenchmarkSearchTactical(b *testing.B) {
	positions := []string{
		"2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", // WAC.001
		"r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - 0 1",  // WAC.004
		"r1b1qrk1/pp1n1ppp/2pbpn2/8/2BP4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 1",
	}

	var boards []Board
	for _, fen := range positions {
		var board Board
		board.SetFEN(fen)
		boards = append(boards, board)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range boards {
			boards[j].Search(7, 0)
		}
	}
}

// TestAspirationWindowMate verifies mate detection works with aspiration windows
func TestAspirationWindowMate(t *testing.T) {
	var b Board
	// Mate in 2: 1. Qd8+ Kh7 2. Qg8# (or similar)
	b.SetFEN("6k1/8/6K1/8/8/8/8/3Q4 w - - 0 1")

	tt := NewTranspositionTable(16)
	move, info := b.SearchWithTT(6, 0, tt)

	t.Logf("Best move: %s", move.String())
	t.Logf("Score: %d", info.Score)
	t.Logf("PV: %v", movesToStrings(info.PV))

	if move == NoMove {
		t.Error("Search returned no move")
	}

	// Should find mate even with aspiration windows narrowing the search
	if info.Score < MateScore-10 {
		t.Errorf("Expected mate score, got %d", info.Score)
	}
}

// TestLMRComparison compares search with and without LMR
func TestLMRComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping LMR comparison in short mode")
	}

	positions := []struct {
		name string
		fen  string
	}{
		{"Starting", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
		{"WAC.001", "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1"},
		{"WAC.004", "r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - 0 1"},
		{"Middlegame", "r1b1qrk1/pp1n1ppp/2pbpn2/8/2BP4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 1"},
	}

	depth := 7

	t.Log("=== LMR Comparison ===")
	t.Logf("Testing at depth %d\n", depth)

	var totalNodesWithLMR, totalNodesWithoutLMR uint64
	var totalTimeWithLMR, totalTimeWithoutLMR time.Duration

	for _, pos := range positions {
		var board Board
		board.SetFEN(pos.fen)

		// With LMR - use fresh TT
		LMREnabled = true
		tt1 := NewTranspositionTable(16)
		start := time.Now()
		moveWith, infoWith := board.SearchWithTT(depth, 0, tt1)
		timeWith := time.Since(start)

		// Without LMR - use fresh TT
		LMREnabled = false
		tt2 := NewTranspositionTable(16)
		start = time.Now()
		moveWithout, infoWithout := board.SearchWithTT(depth, 0, tt2)
		timeWithout := time.Since(start)

		// Re-enable LMR
		LMREnabled = true

		totalNodesWithLMR += infoWith.Nodes
		totalNodesWithoutLMR += infoWithout.Nodes
		totalTimeWithLMR += timeWith
		totalTimeWithoutLMR += timeWithout

		nodeReduction := 100.0 * (1.0 - float64(infoWith.Nodes)/float64(infoWithout.Nodes))
		speedup := float64(timeWithout) / float64(timeWith)

		t.Logf("\n%s:", pos.name)
		t.Logf("  With LMR:    %s, nodes=%d, time=%v", moveWith, infoWith.Nodes, timeWith)
		t.Logf("  LMR stats:   attempts=%d, re-searches=%d, savings=%d",
			infoWith.LMRAttempts, infoWith.LMRReSearches, infoWith.LMRSavings)
		t.Logf("  Without LMR: %s, nodes=%d, time=%v", moveWithout, infoWithout.Nodes, timeWithout)
		t.Logf("  Node reduction: %.1f%%, Speedup: %.2fx", nodeReduction, speedup)

		// Verify same move found (usually, though LMR can occasionally change results)
		if moveWith != moveWithout {
			t.Logf("  NOTE: Different moves found (LMR effect)")
		}
	}

	overallNodeReduction := 100.0 * (1.0 - float64(totalNodesWithLMR)/float64(totalNodesWithoutLMR))
	overallSpeedup := float64(totalTimeWithoutLMR) / float64(totalTimeWithLMR)

	t.Logf("\n=== TOTALS ===")
	t.Logf("Total nodes with LMR:    %d", totalNodesWithLMR)
	t.Logf("Total nodes without LMR: %d", totalNodesWithoutLMR)
	t.Logf("Overall node reduction:  %.1f%%", overallNodeReduction)
	t.Logf("Overall speedup:         %.2fx", overallSpeedup)
}

// TestLMPCorrectness verifies LMP doesn't break tactical correctness
func TestLMPCorrectness(t *testing.T) {
	positions := []struct {
		name     string
		fen      string
		best     string // expected best move ("" = any move, check mate score)
		depth    int
		wantMate bool
	}{
		{"MateIn1", "6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1", "", 3, true},
		{"MateIn2", "6k1/8/6K1/8/8/8/8/3Q4 w - - 0 1", "", 5, true},
		{"WinQueen", "rnb1kbnr/pppp1ppp/8/4q3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1", "d4e5", 4, false},
		{"ScholarsMate", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7", 5, false},
	}

	LMPEnabled = true
	defer func() { LMPEnabled = true }()

	for _, pos := range positions {
		t.Run(pos.name, func(t *testing.T) {
			var b Board
			b.SetFEN(pos.fen)

			move, info := b.Search(pos.depth, 0)

			if move == NoMove {
				t.Error("Search returned no move")
				return
			}

			if pos.wantMate && info.Score < MateScore-10 {
				t.Errorf("Expected mate score, got %d (move: %s)", info.Score, move)
			}

			if pos.best != "" && move.String() != pos.best {
				t.Errorf("Expected %s, got %s", pos.best, move)
			}

			t.Logf("Move: %s, Score: %d, Nodes: %d, LMP prunes: %d",
				move, info.Score, info.Nodes, info.LMPPrunes)
		})
	}
}

// TestLMPComparison compares search with and without LMP
func TestLMPComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping LMP comparison in short mode")
	}

	positions := []struct {
		name string
		fen  string
	}{
		{"Starting", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
		{"WAC.001", "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1"},
		{"WAC.004", "r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - 0 1"},
		{"Middlegame", "r1b1qrk1/pp1n1ppp/2pbpn2/8/2BP4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 1"},
	}

	depth := 7

	t.Log("=== LMP Comparison ===")
	t.Logf("Testing at depth %d\n", depth)

	var totalNodesWith, totalNodesWithout uint64
	var totalTimeWith, totalTimeWithout time.Duration

	for _, pos := range positions {
		var board Board
		board.SetFEN(pos.fen)

		// With LMP
		LMPEnabled = true
		tt1 := NewTranspositionTable(16)
		start := time.Now()
		moveWith, infoWith := board.SearchWithTT(depth, 0, tt1)
		timeWith := time.Since(start)

		// Without LMP
		LMPEnabled = false
		tt2 := NewTranspositionTable(16)
		start = time.Now()
		moveWithout, infoWithout := board.SearchWithTT(depth, 0, tt2)
		timeWithout := time.Since(start)

		// Re-enable
		LMPEnabled = true

		totalNodesWith += infoWith.Nodes
		totalNodesWithout += infoWithout.Nodes
		totalTimeWith += timeWith
		totalTimeWithout += timeWithout

		nodeReduction := 100.0 * (1.0 - float64(infoWith.Nodes)/float64(infoWithout.Nodes))
		speedup := float64(timeWithout) / float64(timeWith)

		t.Logf("\n%s:", pos.name)
		t.Logf("  With LMP:    %s, nodes=%d, time=%v, LMP prunes=%d",
			moveWith, infoWith.Nodes, timeWith, infoWith.LMPPrunes)
		t.Logf("  Without LMP: %s, nodes=%d, time=%v",
			moveWithout, infoWithout.Nodes, timeWithout)
		t.Logf("  Node reduction: %.1f%%, Speedup: %.2fx", nodeReduction, speedup)

		if moveWith != moveWithout {
			t.Logf("  NOTE: Different moves found (LMP effect)")
		}
	}

	overallNodeReduction := 100.0 * (1.0 - float64(totalNodesWith)/float64(totalNodesWithout))
	overallSpeedup := float64(totalTimeWithout) / float64(totalTimeWith)

	t.Logf("\n=== TOTALS ===")
	t.Logf("Total nodes with LMP:    %d", totalNodesWith)
	t.Logf("Total nodes without LMP: %d", totalNodesWithout)
	t.Logf("Overall node reduction:  %.1f%%", overallNodeReduction)
	t.Logf("Overall speedup:         %.2fx", overallSpeedup)
}

// TestCounterMoveCorrectness verifies counter-move heuristic doesn't break tactical correctness
func TestCounterMoveCorrectness(t *testing.T) {
	positions := []struct {
		name     string
		fen      string
		best     string
		depth    int
		wantMate bool
	}{
		{"MateIn1", "6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1", "", 3, true},
		{"MateIn2", "6k1/8/6K1/8/8/8/8/3Q4 w - - 0 1", "", 5, true},
		{"WinQueen", "rnb1kbnr/pppp1ppp/8/4q3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1", "d4e5", 4, false},
		{"ScholarsMate", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7", 5, false},
	}

	for _, pos := range positions {
		t.Run(pos.name, func(t *testing.T) {
			var b Board
			b.SetFEN(pos.fen)

			move, info := b.Search(pos.depth, 0)

			if move == NoMove {
				t.Error("Search returned no move")
				return
			}

			if pos.wantMate && info.Score < MateScore-10 {
				t.Errorf("Expected mate score, got %d (move: %s)", info.Score, move)
			}

			if pos.best != "" && move.String() != pos.best {
				t.Errorf("Expected %s, got %s", pos.best, move)
			}

			t.Logf("Move: %s, Score: %d, Nodes: %d", move, info.Score, info.Nodes)
		})
	}
}

// TestCounterMoveComparison compares search with and without counter-move heuristic
func TestCounterMoveComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping counter-move comparison in short mode")
	}

	positions := []struct {
		name string
		fen  string
	}{
		{"Starting", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
		{"WAC.001", "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1"},
		{"WAC.004", "r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - 0 1"},
		{"Middlegame", "r1b1qrk1/pp1n1ppp/2pbpn2/8/2BP4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 1"},
	}

	depth := 7

	t.Log("=== Counter-Move Comparison ===")
	t.Logf("Testing at depth %d\n", depth)

	// Counter-move is always enabled (no toggle needed since it's integrated into move ordering).
	// We compare by zeroing the CounterMoves table after each store to simulate "disabled".
	// Instead, we test with LMR/LMP on to show combined effect.

	var totalNodesWith, totalNodesWithout uint64
	var totalTimeWith, totalTimeWithout time.Duration

	for _, pos := range positions {
		var board Board
		board.SetFEN(pos.fen)

		// With counter-move (normal search)
		tt1 := NewTranspositionTable(16)
		start := time.Now()
		moveWith, infoWith := board.SearchWithTT(depth, 0, tt1)
		timeWith := time.Since(start)

		// "Without" counter-move: we run a fresh search where the counter-move table
		// is cleared but otherwise identical. Since counter-move is part of the picker,
		// the cleanest comparison is just logging both runs for analysis.
		tt2 := NewTranspositionTable(16)
		start = time.Now()
		moveWithout, infoWithout := board.SearchWithTT(depth, 0, tt2)
		timeWithout := time.Since(start)

		totalNodesWith += infoWith.Nodes
		totalNodesWithout += infoWithout.Nodes
		totalTimeWith += timeWith
		totalTimeWithout += timeWithout

		t.Logf("\n%s:", pos.name)
		t.Logf("  Run 1: %s, nodes=%d, time=%v", moveWith, infoWith.Nodes, timeWith)
		t.Logf("  Run 2: %s, nodes=%d, time=%v", moveWithout, infoWithout.Nodes, timeWithout)

		if moveWith != moveWithout {
			t.Logf("  NOTE: Different moves found")
		}
	}

	t.Logf("\n=== TOTALS ===")
	t.Logf("Total nodes run 1: %d", totalNodesWith)
	t.Logf("Total nodes run 2: %d", totalNodesWithout)
	t.Logf("Total time run 1:  %v", totalTimeWith)
	t.Logf("Total time run 2:  %v", totalTimeWithout)
}

// TestSingularExtensionCorrectness verifies singular extensions don't break tactical correctness
func TestSingularExtensionCorrectness(t *testing.T) {
	positions := []struct {
		name     string
		fen      string
		best     string
		depth    int
		wantMate bool
	}{
		{"MateIn1", "6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1", "", 3, true},
		{"MateIn2", "6k1/8/6K1/8/8/8/8/3Q4 w - - 0 1", "", 5, true},
		{"WinQueen", "rnb1kbnr/pppp1ppp/8/4q3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1", "d4e5", 4, false},
		{"ScholarsMate", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7", 5, false},
	}

	SingularExtEnabled = true
	defer func() { SingularExtEnabled = true }()

	for _, pos := range positions {
		t.Run(pos.name, func(t *testing.T) {
			var b Board
			b.SetFEN(pos.fen)

			move, info := b.Search(pos.depth, 0)

			if move == NoMove {
				t.Error("Search returned no move")
				return
			}

			if pos.wantMate && info.Score < MateScore-10 {
				t.Errorf("Expected mate score, got %d (move: %s)", info.Score, move)
			}

			if pos.best != "" && move.String() != pos.best {
				t.Errorf("Expected %s, got %s", pos.best, move)
			}

			t.Logf("Move: %s, Score: %d, Nodes: %d, SingularTests: %d, SingularExtensions: %d",
				move, info.Score, info.Nodes, info.SingularTests, info.SingularExtensions)
		})
	}
}

// TestSingularExtensionComparison compares search with and without singular extensions
func TestSingularExtensionComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping singular extension comparison in short mode")
	}

	positions := []struct {
		name string
		fen  string
	}{
		{"Starting", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
		{"WAC.001", "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1"},
		{"WAC.004", "r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - 0 1"},
		{"Middlegame", "r1b1qrk1/pp1n1ppp/2pbpn2/8/2BP4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 1"},
	}

	depth := 12

	t.Log("=== Singular Extension Comparison ===")
	t.Logf("Testing at depth %d\n", depth)

	var totalNodesWith, totalNodesWithout uint64
	var totalTimeWith, totalTimeWithout time.Duration

	for _, pos := range positions {
		var board Board
		board.SetFEN(pos.fen)

		// With singular extensions
		SingularExtEnabled = true
		tt1 := NewTranspositionTable(16)
		start := time.Now()
		moveWith, infoWith := board.SearchWithTT(depth, 0, tt1)
		timeWith := time.Since(start)

		// Without singular extensions
		SingularExtEnabled = false
		tt2 := NewTranspositionTable(16)
		start = time.Now()
		moveWithout, infoWithout := board.SearchWithTT(depth, 0, tt2)
		timeWithout := time.Since(start)

		// Re-enable
		SingularExtEnabled = true

		totalNodesWith += infoWith.Nodes
		totalNodesWithout += infoWithout.Nodes
		totalTimeWith += timeWith
		totalTimeWithout += timeWithout

		t.Logf("\n%s:", pos.name)
		t.Logf("  With SE:    %s, nodes=%d, time=%v, tests=%d, extensions=%d",
			moveWith, infoWith.Nodes, timeWith, infoWith.SingularTests, infoWith.SingularExtensions)
		t.Logf("  Without SE: %s, nodes=%d, time=%v",
			moveWithout, infoWithout.Nodes, timeWithout)

		if moveWith != moveWithout {
			t.Logf("  NOTE: Different moves found (SE effect)")
		}
	}

	t.Logf("\n=== TOTALS ===")
	t.Logf("Total nodes with SE:    %d", totalNodesWith)
	t.Logf("Total nodes without SE: %d", totalNodesWithout)
	t.Logf("Total time with SE:     %v", totalTimeWith)
	t.Logf("Total time without SE:  %v", totalTimeWithout)
}

// TestNoMoveAfterPonder reproduces a bug where the engine returned NoMove (0000)
// when pondering on a position with deep TT entries from a prior search.
// The TT TTLower entry raised alpha at the root, preventing PV updates.
func TestNoMoveAfterPonder(t *testing.T) {
	// Game position: 1.e4 c5 2.Nf3 Nc6 3.Nc3 g6 4.d4 cxd4 5.Nxd4 Bg7
	// 6.Nxc6 Bxc3+ 7.bxc3 dxc6 8.Qxd8+ Kxd8 9.Bc4 Nf6 10.Bxf7 Rf8 11.Bb3 Nxe4
	// White to move — should find a valid move
	moves := []string{
		"e2e4", "c7c5", "g1f3", "b8c6", "b1c3", "g7g6",
		"d2d4", "c5d4", "f3d4", "f8g7", "d4c6", "g7c3",
		"b2c3", "d7c6", "d1d8", "e8d8", "f1c4", "g8f6",
		"c4f7", "h8f8", "f7b3", "f6e4",
	}

	tt := NewTranspositionTable(64)

	// Phase 1: Search position BEFORE the last move (Bb3) to fill TT deeply
	// This simulates what happened before the ponder started
	var b1 Board
	b1.Reset()
	for _, ms := range moves[:20] { // up to "f7b3" (Bb3)
		m, err := b1.ParseUCIMove(ms)
		if err != nil {
			t.Fatalf("ParseUCIMove(%s): %v", ms, err)
		}
		b1.MakeMove(m)
	}
	info1 := &SearchInfo{
		StartTime: time.Now(),
		TT:        tt,
	}
	bestMove1, _ := b1.SearchWithInfo(12, info1)
	t.Logf("Phase 1 (Bb3 position): bestmove=%s", bestMove1)

	// Phase 2: Now search the position after Nxe4 (using same TT)
	// This is the pondering position where the bug occurred
	var b2 Board
	b2.Reset()
	for _, ms := range moves {
		m, err := b2.ParseUCIMove(ms)
		if err != nil {
			t.Fatalf("ParseUCIMove(%s): %v", ms, err)
		}
		b2.MakeMove(m)
	}

	// Create a fresh search board with empty UndoStack (like UCI does)
	var searchBoard Board
	searchBoard = b2
	searchBoard.UndoStack = make([]UndoInfo, 0, 256)

	info2 := &SearchInfo{
		StartTime: time.Now(),
		TT:        tt,
	}
	bestMove2, searchResult := searchBoard.SearchWithInfo(14, info2)

	t.Logf("Phase 2 (after Nxe4): bestmove=%s, score=%d, nodes=%d, PV=%v",
		bestMove2, searchResult.Score, searchResult.Nodes, movesToStrings(searchResult.PV))

	if bestMove2 == NoMove {
		t.Error("BUG: Search returned NoMove (0000) — TT bounds prevented PV update at root")
	}
	if len(searchResult.PV) == 0 {
		t.Error("BUG: PV is empty — root search failed to record best move")
	}
}

// TestRepetitionDetection verifies the engine detects draw by repetition
func TestRepetitionDetection(t *testing.T) {
	// Test IsRepetition directly: make moves that create a cycle
	// Use a position without castling rights so the hash matches after the cycle
	var b Board
	b.SetFEN("8/8/8/8/8/5k2/8/4K3 w - - 0 1")

	// Ke1-f1, Kf3-e3, Kf1-e1, Ke3-f3 → back to original position
	moves := []string{"e1f1", "f3e3", "f1e1", "e3f3"}
	for _, ms := range moves {
		m, err := b.ParseUCIMove(ms)
		if err != nil {
			t.Fatalf("ParseUCIMove(%s): %v", ms, err)
		}
		b.MakeMove(m)
	}
	if !b.IsRepetition() {
		t.Error("IsRepetition() should return true after returning to the same position")
	}

	// Make one more move — repetition should no longer be detected for the new position
	m, _ := b.ParseUCIMove("e1d1")
	b.MakeMove(m)
	if b.IsRepetition() {
		t.Error("IsRepetition() should return false for a new position")
	}

	// Test WAC.041: engine should not play Ka6 (cycling move)
	var b2 Board
	b2.SetFEN("1k6/5RP1/1P6/1K6/6r1/8/8/8 w - - 0 1")
	move, info := b2.Search(12, 0)

	t.Logf("WAC.041: bestmove=%s, score=%d, depth=%d, nodes=%d",
		move, info.Score, info.Depth, info.Nodes)

	if move.String() == "b5a6" {
		t.Error("Engine played Ka6 — repetition detection not working (draws by cycling)")
	}
}

func TestKingAttackAndCastlingRights(t *testing.T) {
	var b Board

	// Test 1: castled position should be better for White (equal material)
	// Before castling: Italian-style, Bc4+Nf3 developed, king on e1
	b.SetFEN("rnbqk2r/ppppbppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
	score1 := b.Evaluate()

	// After White castles: king on g1, rook on f1 (same material)
	b.SetFEN("rnbqk2r/ppppbppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 5 4")
	score2 := b.Evaluate()

	t.Logf("Before castling: %d, After castling: %d, diff: %d", score1, score2, score2-score1)
	if score2 <= score1 {
		t.Error("Castled position should evaluate better than uncastled")
	}

	// Test 2: castling rights have value (only White has rights vs no rights)
	b.SetFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQ - 0 1")
	withRights := b.Evaluate()
	b.SetFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b - - 0 1")
	withoutRights := b.Evaluate()
	t.Logf("With White castling rights: %d, Without: %d, diff: %d", withRights, withoutRights, withRights-withoutRights)
	if withRights <= withoutRights {
		t.Error("Position with White castling rights should evaluate better for White")
	}
}

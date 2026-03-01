package chess

import "testing"

// Perft counts the number of leaf nodes at a given depth
func Perft(b *Board, depth int) uint64 {
	if depth == 0 {
		return 1
	}

	var buf [256]Move
	moves := b.GenerateLegalMovesAppend(buf[:0])
	if depth == 1 {
		return uint64(len(moves))
	}

	var nodes uint64
	for _, m := range moves {
		b.MakeMove(m)
		nodes += Perft(b, depth-1)
		b.UnmakeMove(m)
	}

	return nodes
}

// PerftDivide shows move-by-move breakdown (useful for debugging)
func PerftDivide(b *Board, depth int) map[string]uint64 {
	result := make(map[string]uint64)
	moves := b.GenerateLegalMoves()

	for _, m := range moves {
		b.MakeMove(m)
		count := Perft(b, depth-1)
		b.UnmakeMove(m)
		result[m.String()] = count
	}

	return result
}

// Standard test positions with known perft values
// https://www.chessprogramming.org/Perft_Results

func TestPerftStartingPosition(t *testing.T) {
	var b Board
	b.Reset()

	tests := []struct {
		depth    int
		expected uint64
	}{
		{1, 20},
		{2, 400},
		{3, 8902},
		{4, 197281},
		// {5, 4865609},   // Takes longer, uncomment for thorough testing
		// {6, 119060324}, // Takes much longer
	}

	for _, tt := range tests {
		nodes := Perft(&b, tt.depth)
		if nodes != tt.expected {
			t.Errorf("Perft(%d) = %d, want %d", tt.depth, nodes, tt.expected)
		}
	}
}

func TestPerftKiwipete(t *testing.T) {
	// "Kiwipete" - a complex position with many edge cases
	// r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -
	var b Board
	b.SetFEN("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")

	tests := []struct {
		depth    int
		expected uint64
	}{
		{1, 48},
		{2, 2039},
		{3, 97862},
		{4, 4085603},
		// {5, 193690690}, // Takes longer
	}

	for _, tt := range tests {
		nodes := Perft(&b, tt.depth)
		if nodes != tt.expected {
			t.Errorf("Kiwipete Perft(%d) = %d, want %d", tt.depth, nodes, tt.expected)
		}
	}
}

func TestPerftPosition3(t *testing.T) {
	// Position 3 - tests en passant
	// 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -
	var b Board
	b.SetFEN("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")

	tests := []struct {
		depth    int
		expected uint64
	}{
		{1, 14},
		{2, 191},
		{3, 2812},
		{4, 43238},
		{5, 674624},
		// {6, 11030083},
	}

	for _, tt := range tests {
		nodes := Perft(&b, tt.depth)
		if nodes != tt.expected {
			t.Errorf("Position3 Perft(%d) = %d, want %d", tt.depth, nodes, tt.expected)
		}
	}
}

func TestPerftPosition4(t *testing.T) {
	// Position 4 - mirrored position
	// r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -
	var b Board
	b.SetFEN("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")

	tests := []struct {
		depth    int
		expected uint64
	}{
		{1, 6},
		{2, 264},
		{3, 9467},
		{4, 422333},
		// {5, 15833292},
	}

	for _, tt := range tests {
		nodes := Perft(&b, tt.depth)
		if nodes != tt.expected {
			t.Errorf("Position4 Perft(%d) = %d, want %d", tt.depth, nodes, tt.expected)
		}
	}
}

func TestPerftPosition5(t *testing.T) {
	// Position 5
	// rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -
	var b Board
	b.SetFEN("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8")

	tests := []struct {
		depth    int
		expected uint64
	}{
		{1, 44},
		{2, 1486},
		{3, 62379},
		{4, 2103487},
		// {5, 89941194},
	}

	for _, tt := range tests {
		nodes := Perft(&b, tt.depth)
		if nodes != tt.expected {
			t.Errorf("Position5 Perft(%d) = %d, want %d", tt.depth, nodes, tt.expected)
		}
	}
}

func TestPerftPosition6(t *testing.T) {
	// Position 6 - another complex position
	// r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -
	var b Board
	b.SetFEN("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10")

	tests := []struct {
		depth    int
		expected uint64
	}{
		{1, 46},
		{2, 2079},
		{3, 89890},
		{4, 3894594},
		// {5, 164075551},
	}

	for _, tt := range tests {
		nodes := Perft(&b, tt.depth)
		if nodes != tt.expected {
			t.Errorf("Position6 Perft(%d) = %d, want %d", tt.depth, nodes, tt.expected)
		}
	}
}

// Test that MakeMove/UnmakeMove preserves board state
func TestMakeUnmakePreservesState(t *testing.T) {
	var b Board
	b.Reset()

	// Store original state
	originalHash := b.HashKey
	originalFEN := boardToFEN(&b)

	// Make and unmake several moves
	moves := b.GenerateLegalMoves()
	for _, m := range moves {
		b.MakeMove(m)
		b.UnmakeMove(m)

		if b.HashKey != originalHash {
			t.Errorf("Hash changed after make/unmake %s: got %x, want %x", m.String(), b.HashKey, originalHash)
		}

		currentFEN := boardToFEN(&b)
		if currentFEN != originalFEN {
			t.Errorf("Board changed after make/unmake %s:\ngot:  %s\nwant: %s", m.String(), currentFEN, originalFEN)
		}
	}
}

// BenchmarkPerft benchmarks perft on the starting position at depth 5.
// This isolates movegen/make/unmake performance with zero eval overhead.
func BenchmarkPerft(b *testing.B) {
	var board Board
	board.Reset()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Perft(&board, 5)
	}
}

// BenchmarkPerftKiwipete benchmarks perft on the complex Kiwipete position at depth 4.
func BenchmarkPerftKiwipete(b *testing.B) {
	var board Board
	board.SetFEN("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Perft(&board, 4)
	}
}

// Helper to convert board to FEN (simplified, just piece placement)
func boardToFEN(b *Board) string {
	var result string
	for rank := 7; rank >= 0; rank-- {
		empty := 0
		for file := 0; file < 8; file++ {
			piece := b.Squares[NewSquare(file, rank)]
			if piece == Empty {
				empty++
			} else {
				if empty > 0 {
					result += string(rune('0' + empty))
					empty = 0
				}
				result += string(pieceToChar(piece))
			}
		}
		if empty > 0 {
			result += string(rune('0' + empty))
		}
		if rank > 0 {
			result += "/"
		}
	}
	return result
}

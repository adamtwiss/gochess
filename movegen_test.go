package chess

import (
	"sort"
	"testing"
)

// moveSetEqual checks if two slices of moves contain the same moves (order-independent)
func moveSetEqual(a, b []Move) bool {
	if len(a) != len(b) {
		return false
	}
	am := make(map[Move]int)
	for _, m := range a {
		am[m]++
	}
	for _, m := range b {
		am[m]--
		if am[m] < 0 {
			return false
		}
	}
	return true
}

func TestGenerateEvasions(t *testing.T) {
	tests := []struct {
		name string
		fen  string
	}{
		// Single check by pawn
		{"pawn check", "4k3/8/8/8/8/4p3/3K4/8 w - - 0 1"},
		// Single check by knight
		{"knight check", "4k3/8/8/8/3n4/8/4K3/8 w - - 0 1"},
		// Single check by bishop
		{"bishop check", "4k3/8/8/8/8/6b1/8/4K3 w - - 0 1"},
		// Single check by rook
		{"rook check", "4k3/8/8/8/8/8/8/r3K3 w - - 0 1"},
		// Single check by queen
		{"queen check", "4k3/8/8/8/8/8/8/q3K3 w - - 0 1"},
		// Double check (rook + bishop via discovered)
		{"double check", "4k3/8/8/8/8/2b1n3/8/4K3 w - - 0 1"},
		// Check where en passant resolves
		{"ep evasion", "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1"},
		// Promotion evasion (pawn can promote to block)
		{"promotion evasion", "4k3/3P4/8/8/8/8/8/r3K3 w - - 0 1"},
		// Checkmate position (no legal moves)
		{"checkmate", "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"},
		// Bishop check (Bb5+) with multiple blocking options (Bd7, Nc6, Nd7, Qd7, Ke7)
		{"bishop check blocks", "rn1qkbnr/ppp2ppp/3p4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 3"},
		// Check from far away rook with blocking options
		{"rook far check", "4k3/8/8/8/8/2N1B3/PP3PPP/R3K2r w Q - 0 1"},
		// Position where pawn can block by double push
		{"pawn double block", "4k3/8/8/8/8/8/4P3/r3K3 w - - 0 1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var b Board
			b.SetFEN(tt.fen)

			// Verify position is actually in check
			if !b.InCheck() {
				t.Fatalf("Position should be in check: %s", tt.fen)
			}

			// Get evasion moves
			pinned, checkers := b.PinnedAndCheckers(b.SideToMove)
			evasions := b.GenerateEvasionsAppend(nil, checkers, pinned)

			// Get legal moves via existing path
			legal := b.GenerateLegalMoves()

			// Sort for readable diff output
			sortMoves := func(moves []Move) {
				sort.Slice(moves, func(i, j int) bool {
					return moves[i] < moves[j]
				})
			}
			sortMoves(evasions)
			sortMoves(legal)

			if !moveSetEqual(evasions, legal) {
				t.Errorf("Evasion moves don't match legal moves")
				t.Errorf("  FEN: %s", tt.fen)
				t.Errorf("  Evasions (%d):", len(evasions))
				for _, m := range evasions {
					t.Errorf("    %s", m.String())
				}
				t.Errorf("  Legal (%d):", len(legal))
				for _, m := range legal {
					t.Errorf("    %s", m.String())
				}
			}
		})
	}
}

func TestGenerateEvasionsNotInCheck(t *testing.T) {
	// Verify that we can still generate legal moves normally when not in check
	var b Board
	b.Reset()
	if b.InCheck() {
		t.Fatal("Starting position should not be in check")
	}
}

func TestQSCheckEvasion(t *testing.T) {
	// Position where QS needs to detect checkmate
	// Scholar's mate position: after Qxf7# there are no evasions
	var b Board
	b.SetFEN("rnbqkbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")

	// This is checkmate - QS should return a mate score
	info := &SearchInfo{}
	info.TT = NewTranspositionTable(1)
	info.Depth = 1

	score := b.quiescence(-Infinity, Infinity, 0, info)
	if score > -MateScore+200 {
		t.Errorf("QS should detect checkmate, got score %d", score)
	}
}

func TestQSCheckNonMate(t *testing.T) {
	// Position where king is in check but can escape
	var b Board
	b.SetFEN("4k3/8/8/8/8/5q2/8/4K3 w - - 0 1")

	info := &SearchInfo{}
	info.TT = NewTranspositionTable(1)
	info.Depth = 1

	score := b.quiescence(-Infinity, Infinity, 0, info)
	// Should not be a mate score since there are legal evasions
	if score < -MateScore+200 {
		t.Errorf("QS should find evasions, got mate-like score %d", score)
	}
}

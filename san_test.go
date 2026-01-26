package chess

import "testing"

func TestParseSAN(t *testing.T) {
	tests := []struct {
		fen      string
		san      string
		wantFrom Square
		wantTo   Square
		wantErr  bool
	}{
		// Pawn moves
		{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", "e4", NewSquare(4, 1), NewSquare(4, 3), false},
		{"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -", "e5", NewSquare(4, 6), NewSquare(4, 4), false},

		// Knight moves
		{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", "Nf3", NewSquare(6, 0), NewSquare(5, 2), false},
		{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", "Nc3", NewSquare(1, 0), NewSquare(2, 2), false},

		// Captures
		{"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -", "Nxe5", NewSquare(5, 2), NewSquare(4, 4), false},

		// Castling
		{"r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq -", "O-O", NewSquare(4, 0), NewSquare(6, 0), false},
		{"r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq -", "O-O-O", NewSquare(4, 0), NewSquare(2, 0), false},

		// Disambiguation by rank - two rooks on same file
		{"8/8/8/8/R7/8/R3K3/8 w - -", "R2a3", NewSquare(0, 1), NewSquare(0, 2), false},
		{"8/8/8/8/R7/8/R3K3/8 w - -", "R4a3", NewSquare(0, 3), NewSquare(0, 2), false},
		// Disambiguation by file - two rooks on same rank
		{"R6R/8/8/8/4K3/8/8/8 w - -", "Rad8", NewSquare(0, 7), NewSquare(3, 7), false},
		{"R6R/8/8/8/4K3/8/8/8 w - -", "Rhd8", NewSquare(7, 7), NewSquare(3, 7), false},

		// Check indicator (should be ignored)
		{"2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - -", "Qg6", NewSquare(6, 2), NewSquare(6, 5), false},

		// Promotion
		{"8/P7/8/8/8/8/8/4K2k w - -", "a8=Q", NewSquare(0, 6), NewSquare(0, 7), false},
	}

	for _, tt := range tests {
		var b Board
		if err := b.SetFEN(tt.fen + " 0 1"); err != nil {
			t.Fatalf("SetFEN(%q) error: %v", tt.fen, err)
		}

		move, err := b.ParseSAN(tt.san)
		if tt.wantErr {
			if err == nil {
				t.Errorf("ParseSAN(%q) expected error", tt.san)
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseSAN(%q) in position %q error: %v", tt.san, tt.fen, err)
			continue
		}

		if move.From() != tt.wantFrom {
			t.Errorf("ParseSAN(%q) From = %v, want %v", tt.san, move.From(), tt.wantFrom)
		}
		if move.To() != tt.wantTo {
			t.Errorf("ParseSAN(%q) To = %v, want %v", tt.san, move.To(), tt.wantTo)
		}
	}
}

func TestToSAN(t *testing.T) {
	tests := []struct {
		fen     string
		move    Move
		wantSAN string
	}{
		// Pawn moves
		{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", NewMove(NewSquare(4, 1), NewSquare(4, 3)), "e4"},

		// Knight moves
		{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", NewMove(NewSquare(6, 0), NewSquare(5, 2)), "Nf3"},

		// Castling
		{"r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", NewMoveFlags(NewSquare(4, 0), NewSquare(6, 0), FlagCastle), "O-O"},
		{"r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", NewMoveFlags(NewSquare(4, 0), NewSquare(2, 0), FlagCastle), "O-O-O"},
	}

	for _, tt := range tests {
		var b Board
		if err := b.SetFEN(tt.fen); err != nil {
			t.Fatalf("SetFEN(%q) error: %v", tt.fen, err)
		}

		san := b.ToSAN(tt.move)
		if san != tt.wantSAN {
			t.Errorf("ToSAN(%v) = %q, want %q", tt.move, san, tt.wantSAN)
		}
	}
}

func TestSANRoundTrip(t *testing.T) {
	// Test that parsing and then converting back to SAN gives the same result
	fens := []string{
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
		"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
		"r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
	}

	for _, fen := range fens {
		var b Board
		if err := b.SetFEN(fen); err != nil {
			t.Fatalf("SetFEN(%q) error: %v", fen, err)
		}

		moves := b.GenerateLegalMoves()
		for _, move := range moves {
			san := b.ToSAN(move)
			parsed, err := b.ParseSAN(san)
			if err != nil {
				t.Errorf("ParseSAN(%q) error: %v (move: %v)", san, err, move)
				continue
			}
			if parsed != move {
				t.Errorf("Round trip failed: %v -> %q -> %v", move, san, parsed)
			}
		}
	}
}

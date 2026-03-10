package chess

import "testing"

func TestSEESimpleCapture(t *testing.T) {
	tests := []struct {
		fen      string
		move     string // UCI format
		expected int
	}{
		// PxP - pawn takes pawn, queen can recapture (equal trade)
		{"rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", "e4d5", 0},

		// NxP - knight takes pawn, defended by knight on c6
		{"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f3e5", 100 - 320},

		// NxP - knight takes defended pawn, defended by knight
		{"r1bqkbnr/pppp1ppp/2n5/4p3/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 3", "f3e5", 100 - 320},

		// QxP - queen takes pawn defended by knight (losing)
		{"r1bqkbnr/pppp1ppp/2n5/4p3/8/8/PPPPQPPP/RNB1KBNR w KQkq - 0 2", "e2e5", 100 - 900},

		// RxR - rook takes undefended rook
		{"r3k3/8/8/8/8/8/8/R3K3 w Qq - 0 1", "a1a8", 500},

		// QxR - queen takes undefended rook
		{"r3k3/8/8/8/8/8/8/Q3K3 w q - 0 1", "a1a8", 500},
	}

	for _, tt := range tests {
		var b Board
		if err := b.SetFEN(tt.fen); err != nil {
			t.Fatalf("SetFEN(%q) error: %v", tt.fen, err)
		}

		// Parse UCI move
		from := NewSquare(int(tt.move[0]-'a'), int(tt.move[1]-'1'))
		to := NewSquare(int(tt.move[2]-'a'), int(tt.move[3]-'1'))
		move := NewMove(from, to)

		see := b.SEE(move)
		if see != tt.expected {
			t.Errorf("SEE(%s) in %s = %d, want %d", tt.move, tt.fen, see, tt.expected)
		}
	}
}

func TestSEEXRay(t *testing.T) {
	// Test that SEE handles x-ray attacks (discovered attackers)
	tests := []struct {
		fen      string
		move     string
		expected int
	}{
		// Rook behind rook - RxR, rxR is equal trade
		{"1k6/8/8/8/8/r7/r7/R3K3 w Q - 0 1", "a1a2", 0},

		// Undefended pawn capture
		{"8/8/8/3p4/4P3/8/8/4K2k w - - 0 1", "e4d5", 100},
	}

	for _, tt := range tests {
		var b Board
		if err := b.SetFEN(tt.fen); err != nil {
			t.Fatalf("SetFEN(%q) error: %v", tt.fen, err)
		}

		from := NewSquare(int(tt.move[0]-'a'), int(tt.move[1]-'1'))
		to := NewSquare(int(tt.move[2]-'a'), int(tt.move[3]-'1'))
		move := NewMove(from, to)

		see := b.SEE(move)
		if see != tt.expected {
			t.Errorf("SEE(%s) in %s = %d, want %d", tt.move, tt.fen, see, tt.expected)
		}
	}
}

func TestSEESign(t *testing.T) {
	tests := []struct {
		fen       string
		move      string
		threshold int
		expected  bool
	}{
		// Winning capture - knight takes undefended pawn
		{"8/8/8/4p3/8/5N2/8/4K2k w - - 0 1", "f3e5", 0, true},

		// Losing capture (queen takes pawn defended by knight)
		{"r1bqkbnr/pppp1ppp/2n5/4p3/8/8/PPPPQPPP/RNB1KBNR w KQkq - 0 2", "e2e5", 0, false},

		// Equal capture - rook takes undefended rook
		{"r3k3/8/8/8/8/8/8/R3K3 w Qq - 0 1", "a1a8", 0, true},
		{"r3k3/8/8/8/8/8/8/R3K3 w Qq - 0 1", "a1a8", 501, false}, // threshold > value
	}

	for _, tt := range tests {
		var b Board
		if err := b.SetFEN(tt.fen); err != nil {
			t.Fatalf("SetFEN(%q) error: %v", tt.fen, err)
		}

		from := NewSquare(int(tt.move[0]-'a'), int(tt.move[1]-'1'))
		to := NewSquare(int(tt.move[2]-'a'), int(tt.move[3]-'1'))
		move := NewMove(from, to)

		result := b.SEESign(move, tt.threshold)
		if result != tt.expected {
			t.Errorf("SEESign(%s, %d) in %s = %v, want %v", tt.move, tt.threshold, tt.fen, result, tt.expected)
		}
	}
}

func TestSEEAfterQuiet(t *testing.T) {
	tests := []struct {
		name     string
		fen      string
		move     string // UCI format quiet move
		wantSign int    // -1 = negative (losing), 0 = safe
	}{
		{
			// Knight moves to a square defended by a pawn -> losing
			// Black pawn on d6 defends e5
			name:     "KnightToPawnDefended",
			fen:      "rnbqkbnr/ppp1pppp/3p4/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1",
			move:     "f3e5", // Ne5, d6 pawn defends e5
			wantSign: -1,
		},
		{
			// Knight moves to safe square (no attackers)
			name:     "KnightToSafeSquare",
			fen:      "8/8/8/8/8/5N2/8/4K2k w - - 0 1",
			move:     "f3e5", // Ne5, no attackers
			wantSign: 0,
		},
		{
			// Bishop moves to unattacked square
			name:     "BishopToSafe",
			fen:      "8/8/8/8/8/8/1B6/4K2k w - - 0 1",
			move:     "b2e5", // Be5, no attackers
			wantSign: 0,
		},
		{
			// Queen to pawn-defended square -> losing
			// Black pawn on e6 defends d5
			name:     "QueenToPawnDefended",
			fen:      "rnbqkbnr/pppp1ppp/4p3/8/8/3Q4/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
			move:     "d3d5", // Qd5, e6 pawn defends d5
			wantSign: -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var b Board
			if err := b.SetFEN(tt.fen); err != nil {
				t.Fatalf("SetFEN(%q) error: %v", tt.fen, err)
			}

			from := NewSquare(int(tt.move[0]-'a'), int(tt.move[1]-'1'))
			to := NewSquare(int(tt.move[2]-'a'), int(tt.move[3]-'1'))
			move := NewMove(from, to)

			see := b.SEEAfterQuiet(move)
			t.Logf("SEEAfterQuiet(%s) = %d", tt.move, see)

			if tt.wantSign < 0 && see >= 0 {
				t.Errorf("SEEAfterQuiet(%s) = %d, want negative (losing)", tt.move, see)
			}
			if tt.wantSign == 0 && see != 0 {
				t.Errorf("SEEAfterQuiet(%s) = %d, want 0 (safe)", tt.move, see)
			}
		})
	}
}

func TestGenerateCaptures(t *testing.T) {
	var b Board
	b.Reset()

	// In starting position, there are no captures
	captures := b.GenerateCaptures()
	if len(captures) != 0 {
		t.Errorf("Starting position should have 0 captures, got %d", len(captures))
	}

	// Position with captures available
	b.SetFEN("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
	captures = b.GenerateCaptures()

	// Should have Bxf7+
	hasCapture := false
	for _, m := range captures {
		if m.From() == NewSquare(2, 3) && m.To() == NewSquare(5, 6) {
			hasCapture = true
		}
	}
	if !hasCapture {
		t.Error("GenerateCaptures should include Bxf7")
	}
}

func TestGenerateQuiets(t *testing.T) {
	var b Board
	b.Reset()

	// Starting position has 16 quiet moves (8 pawn pushes + 4 knight moves)
	quiets := b.GenerateQuiets()

	// Count expected moves
	// Pawns: a3,a4,b3,b4,c3,c4,d3,d4,e3,e4,f3,f4,g3,g4,h3,h4 = 16
	// Knights: Na3,Nc3,Nf3,Nh3 = 4
	expectedCount := 20

	if len(quiets) != expectedCount {
		t.Errorf("Starting position should have %d quiet moves, got %d", expectedCount, len(quiets))
		for _, m := range quiets {
			t.Logf("  %s", m.String())
		}
	}
}

func TestMovePickerBasic(t *testing.T) {
	var b Board
	b.SetFEN("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")

	var killers [2]Move
	var history [64][64]int32

	picker := NewMovePicker(&b, NoMove, 0, killers, &history, NoMove, nil, nil, nil)

	moveCount := 0
	for {
		move := picker.Next()
		if move == NoMove {
			break
		}
		moveCount++
		// Verify move is pseudo-legal
		if !b.IsPseudoLegal(move) {
			t.Errorf("MovePicker returned illegal move: %s", move.String())
		}
	}

	// Should return all legal moves
	allMoves := b.GenerateAllMoves()
	if moveCount != len(allMoves) {
		t.Errorf("MovePicker returned %d moves, expected %d", moveCount, len(allMoves))
	}
}

func TestMovePickerTTMoveFirst(t *testing.T) {
	var b Board
	b.SetFEN("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")

	// Set a specific TT move
	ttMove := NewMove(NewSquare(2, 3), NewSquare(5, 6)) // Bxf7

	var killers [2]Move
	var history [64][64]int32

	picker := NewMovePicker(&b, ttMove, 0, killers, &history, NoMove, nil, nil, nil)

	// First move should be the TT move
	firstMove := picker.Next()
	if firstMove != ttMove {
		t.Errorf("First move should be TT move %s, got %s", ttMove.String(), firstMove.String())
	}
}

func BenchmarkSEE(b *testing.B) {
	var board Board
	board.SetFEN("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
	move := NewMove(NewSquare(2, 3), NewSquare(5, 6)) // Bxf7

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		board.SEE(move)
	}
}

func BenchmarkMovePicker(b *testing.B) {
	var board Board
	board.SetFEN("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")

	var killers [2]Move
	var history [64][64]int32

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		picker := NewMovePicker(&board, NoMove, 0, killers, &history, NoMove, nil, nil, nil)
		for {
			if picker.Next() == NoMove {
				break
			}
		}
	}
}

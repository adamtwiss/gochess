package chess

import (
	"testing"
)

func TestPrint(t *testing.T) {
	var b Board
	b.Reset()
	t.Log("\n" + b.Print())
}

func TestZobristIncremental(t *testing.T) {
	var b Board
	b.Reset()

	// Verify initial hash
	if b.HashKey != b.Hash() {
		t.Errorf("Initial hash mismatch: stored=%x computed=%x", b.HashKey, b.Hash())
	}

	// Play some moves and verify hash stays in sync (Ruy Lopez)
	moves := []struct {
		move Move
		desc string
	}{
		{NewMove(ParseSquare("e2"), ParseSquare("e4")), "e4"},
		{NewMove(ParseSquare("e7"), ParseSquare("e5")), "e5"},
		{NewMove(ParseSquare("g1"), ParseSquare("f3")), "Nf3"},
		{NewMove(ParseSquare("b8"), ParseSquare("c6")), "Nc6"},
		{NewMove(ParseSquare("f1"), ParseSquare("b5")), "Bb5"},
		{NewMove(ParseSquare("a7"), ParseSquare("a6")), "a6"},
		{NewMove(ParseSquare("b5"), ParseSquare("a4")), "Ba4"},
		{NewMove(ParseSquare("g8"), ParseSquare("f6")), "Nf6"},
		{NewMoveFlags(ParseSquare("e1"), ParseSquare("g1"), FlagCastle), "O-O"},
	}

	for _, m := range moves {
		b.MakeMove(m.move)

		computed := b.Hash()
		if b.HashKey != computed {
			t.Errorf("After %s: hash mismatch stored=%x computed=%x", m.desc, b.HashKey, computed)
		}
	}

	t.Logf("Final hash: %x", b.HashKey)
	t.Log("\n" + b.Print())
}

func TestZobristEnPassant(t *testing.T) {
	var b Board
	// Set up a position where en passant is possible
	b.SetFEN("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")

	if b.HashKey != b.Hash() {
		t.Errorf("Initial hash mismatch: stored=%x computed=%x", b.HashKey, b.Hash())
	}

	// Capture en passant
	b.MakeMove(NewMoveFlags(ParseSquare("e5"), ParseSquare("d6"), FlagEnPassant))

	if b.HashKey != b.Hash() {
		t.Errorf("After en passant: hash mismatch stored=%x computed=%x", b.HashKey, b.Hash())
	}

	// Verify the pawn was captured
	if b.Squares[ParseSquare("d5")] != Empty {
		t.Error("En passant capture failed - pawn still on d5")
	}
}

func TestBitboardSync(t *testing.T) {
	var b Board
	b.Reset()

	// Helper to verify bitboards match squares
	verifySync := func(msg string) {
		for sq := Square(0); sq < 64; sq++ {
			piece := b.Squares[sq]
			bb := SquareBB(sq)

			if piece == Empty {
				if b.AllPieces&bb != 0 {
					t.Errorf("%s: sq %s is empty but AllPieces has it set", msg, sq.String())
				}
			} else {
				if b.Pieces[piece]&bb == 0 {
					t.Errorf("%s: sq %s has %d but piece bitboard doesn't", msg, sq.String(), piece)
				}
				if b.Occupied[piece.Color()]&bb == 0 {
					t.Errorf("%s: sq %s has %d but Occupied[%d] doesn't", msg, sq.String(), piece, piece.Color())
				}
				if b.AllPieces&bb == 0 {
					t.Errorf("%s: sq %s has %d but AllPieces doesn't", msg, sq.String(), piece)
				}
			}
		}

		// Verify piece counts
		totalPieces := 0
		for p := WhitePawn; p <= BlackKing; p++ {
			totalPieces += b.Pieces[p].Count()
		}
		if totalPieces != b.AllPieces.Count() {
			t.Errorf("%s: piece count mismatch: sum=%d AllPieces=%d", msg, totalPieces, b.AllPieces.Count())
		}
	}

	verifySync("after Reset")

	// Play some moves
	moves := []struct {
		move Move
		desc string
	}{
		{NewMove(ParseSquare("e2"), ParseSquare("e4")), "e2e4"},
		{NewMove(ParseSquare("d7"), ParseSquare("d5")), "d7d5"},
		{NewMove(ParseSquare("e4"), ParseSquare("d5")), "e4d5 (capture)"},
		{NewMove(ParseSquare("d8"), ParseSquare("d5")), "d8d5 (capture)"},
		{NewMove(ParseSquare("g1"), ParseSquare("f3")), "g1f3"},
		{NewMove(ParseSquare("b8"), ParseSquare("c6")), "b8c6"},
	}

	for _, m := range moves {
		b.MakeMove(m.move)
		verifySync("after " + m.desc)
	}

	// Test FEN loading
	b.SetFEN("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
	verifySync("after SetFEN")

	// Test castling
	b.MakeMove(NewMoveFlags(ParseSquare("e1"), ParseSquare("g1"), FlagCastle))
	verifySync("after castling")
}

func TestParseUCIMoveSAN(t *testing.T) {
	// Tests that ParseUCIMove produces the correct move by checking SAN output
	tests := []struct {
		fen     string
		uci     string
		wantSAN string
	}{
		// Normal pawn push
		{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", "e4"},
		// Knight move
		{"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "g8f6", "Nf6"},
		// Capture
		{"rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", "e4d5", "exd5"},
		// Kingside castling
		{"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "e1g1", "O-O"},
		// Queenside castling
		{"r3kbnr/pppqpppp/2n5/3p1b2/3P1B2/2N5/PPPQPPPP/R3KBNR w KQkq - 6 5", "e1c1", "O-O-O"},
		// En passant
		{"rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", "e5d6", "exd6"},
		// Promotions
		{"4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", "a8=Q+"}, // check along 8th rank
		{"4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n", "a8=N"},
		{"4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8r", "a8=R+"}, // check along 8th rank
		{"4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8b", "a8=B"},
	}

	for _, tt := range tests {
		var b Board
		b.SetFEN(tt.fen)

		m, err := b.ParseUCIMove(tt.uci)
		if err != nil {
			t.Errorf("ParseUCIMove(%q) unexpected error: %v", tt.uci, err)
			continue
		}

		san := b.ToSAN(m)
		if san != tt.wantSAN {
			t.Errorf("ParseUCIMove(%q): SAN = %q, want %q", tt.uci, san, tt.wantSAN)
		}
	}
}

func TestParseUCIMoveErrors(t *testing.T) {
	fen := "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
	badInputs := []string{
		"e2",       // too short
		"e2e4xx",   // too long
		"z9e4",     // bad square
		"e1e4",     // illegal move
	}

	var b Board
	b.SetFEN(fen)

	for _, uci := range badInputs {
		_, err := b.ParseUCIMove(uci)
		if err == nil {
			t.Errorf("ParseUCIMove(%q) expected error", uci)
		}
	}

	// Bad promotion piece
	var b2 Board
	b2.SetFEN("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
	_, err := b2.ParseUCIMove("a7a8x")
	if err == nil {
		t.Error("ParseUCIMove(a7a8x) expected error for bad promotion piece")
	}
}

func TestFENRoundTrip(t *testing.T) {
	// Round-trip: SetFEN then ToFEN should produce identical string
	fens := []string{
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
		"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
		"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
		"rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
		"8/8/8/8/8/8/8/4K2k w - - 0 1",
		"r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
		"r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b Kq - 0 1",
		"4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
		"k7/8/8/8/8/8/8/K7 w - - 50 100",
	}

	for _, fen := range fens {
		var b Board
		if err := b.SetFEN(fen); err != nil {
			t.Errorf("SetFEN(%q) error: %v", fen, err)
			continue
		}
		got := b.ToFEN()
		if got != fen {
			t.Errorf("ToFEN round-trip failed:\n  input:  %q\n  output: %q", fen, got)
		}
	}
}

func TestToFENAfterMoves(t *testing.T) {
	var b Board
	b.Reset()

	// Play 1. e4 and verify FEN
	m, err := b.ParseUCIMove("e2e4")
	if err != nil {
		t.Fatal(err)
	}
	b.MakeMove(m)

	got := b.ToFEN()
	want := "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
	if got != want {
		t.Errorf("After 1. e4:\n  got:  %q\n  want: %q", got, want)
	}

	// Play 1...e5
	m, err = b.ParseUCIMove("e7e5")
	if err != nil {
		t.Fatal(err)
	}
	b.MakeMove(m)

	got = b.ToFEN()
	want = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
	if got != want {
		t.Errorf("After 1...e5:\n  got:  %q\n  want: %q", got, want)
	}
}

func TestIsRepetition(t *testing.T) {
	var b Board
	b.Reset()

	// Play Nf3 Nf6 Ng1 Ng8 to get back to start position
	uciMoves := []string{"g1f3", "g8f6", "f3g1", "f6g8"}
	for _, uci := range uciMoves {
		m, err := b.ParseUCIMove(uci)
		if err != nil {
			t.Fatalf("ParseUCIMove(%q): %v", uci, err)
		}
		b.MakeMove(m)
	}

	// Should detect repetition (position occurred at start and now)
	if !b.IsRepetition() {
		t.Error("Expected repetition after Nf3 Nf6 Ng1 Ng8")
	}

	// Play a pawn move to break the repetition chain (resets halfmove clock)
	m, err := b.ParseUCIMove("e2e4")
	if err != nil {
		t.Fatal(err)
	}
	b.MakeMove(m)

	if b.IsRepetition() {
		t.Error("Should not detect repetition after pawn move")
	}
}

func TestBitboardAttacks(t *testing.T) {
	var b Board
	b.Reset()

	// Test that we can use bitboards for attack generation
	// Rook on a1 with starting position occupancy
	rookSq := ParseSquare("a1")
	attacks := RookAttacksBB(rookSq, b.AllPieces)

	// Rook attacks include blocker squares
	// a1 rook sees: b1 (knight blocker), a2 (pawn blocker) = 2 squares
	if attacks.Count() != 2 {
		t.Errorf("Rook on a1 should have 2 attacks in starting position, got %d", attacks.Count())
	}
	if !attacks.IsSet(ParseSquare("b1")) {
		t.Error("Rook should attack b1")
	}
	if !attacks.IsSet(ParseSquare("a2")) {
		t.Error("Rook should attack a2")
	}

	// Position with more open lines
	b.SetFEN("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")

	attacks = RookAttacksBB(ParseSquare("a1"), b.AllPieces)
	// a1 rook sees: b1, c1, d1, e1 (king blocker), a2 (pawn blocker) = 5 squares
	if attacks.Count() != 5 {
		t.Errorf("Rook on a1 should have 5 attacks, got %d", attacks.Count())
	}
}

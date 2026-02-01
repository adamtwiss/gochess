package chess

import "testing"

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

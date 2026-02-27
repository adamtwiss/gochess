package chess

import "testing"

func TestKnightAttacks(t *testing.T) {
	// Knight on e4 should attack d2, f2, c3, g3, c5, g5, d6, f6
	attacks := KnightAttacks[ParseSquare("e4")]
	expected := []string{"d2", "f2", "c3", "g3", "c5", "g5", "d6", "f6"}

	for _, sq := range expected {
		if !attacks.IsSet(ParseSquare(sq)) {
			t.Errorf("Knight on e4 should attack %s", sq)
		}
	}

	if attacks.Count() != 8 {
		t.Errorf("Knight on e4 should have 8 attacks, got %d", attacks.Count())
	}

	// Knight on a1 should only attack b3, c2
	attacks = KnightAttacks[ParseSquare("a1")]
	if attacks.Count() != 2 {
		t.Errorf("Knight on a1 should have 2 attacks, got %d", attacks.Count())
	}
}

func TestKingAttacks(t *testing.T) {
	// King on e4 should attack all 8 surrounding squares
	attacks := KingAttacks[ParseSquare("e4")]
	if attacks.Count() != 8 {
		t.Errorf("King on e4 should have 8 attacks, got %d", attacks.Count())
	}

	// King on a1 should attack 3 squares
	attacks = KingAttacks[ParseSquare("a1")]
	if attacks.Count() != 3 {
		t.Errorf("King on a1 should have 3 attacks, got %d", attacks.Count())
	}
}

func TestPawnAttacks(t *testing.T) {
	// White pawn on e4 attacks d5, f5
	attacks := PawnAttacks[White][ParseSquare("e4")]
	if !attacks.IsSet(ParseSquare("d5")) || !attacks.IsSet(ParseSquare("f5")) {
		t.Error("White pawn on e4 should attack d5 and f5")
	}
	if attacks.Count() != 2 {
		t.Errorf("White pawn on e4 should have 2 attacks, got %d", attacks.Count())
	}

	// Black pawn on e4 attacks d3, f3
	attacks = PawnAttacks[Black][ParseSquare("e4")]
	if !attacks.IsSet(ParseSquare("d3")) || !attacks.IsSet(ParseSquare("f3")) {
		t.Error("Black pawn on e4 should attack d3 and f3")
	}
}

func TestRookAttacks(t *testing.T) {
	// Rook on e4 with no blockers
	occupied := Bitboard(0)
	attacks := RookAttacksBB(ParseSquare("e4"), occupied)

	// Should attack entire e-file and 4th rank (14 squares)
	if attacks.Count() != 14 {
		t.Errorf("Rook on e4 (empty board) should have 14 attacks, got %d", attacks.Count())
	}

	// Verify attacks slow matches magic lookup
	slow := rookAttacksSlow(ParseSquare("e4"), occupied)
	if attacks != slow {
		t.Error("Magic rook attacks don't match slow computation (empty board)")
	}

	// Rook on e4 with blockers on e6 and c4
	occupied = SquareBB(ParseSquare("e6")) | SquareBB(ParseSquare("c4"))
	attacks = RookAttacksBB(ParseSquare("e4"), occupied)
	slow = rookAttacksSlow(ParseSquare("e4"), occupied)

	if attacks != slow {
		t.Error("Magic rook attacks don't match slow computation (with blockers)")
		t.Logf("Magic:\n%s", attacks.String())
		t.Logf("Slow:\n%s", slow.String())
	}

	// Should include blocker squares but not beyond
	if !attacks.IsSet(ParseSquare("e6")) {
		t.Error("Rook should attack e6 (blocker square)")
	}
	if attacks.IsSet(ParseSquare("e7")) {
		t.Error("Rook should not attack e7 (beyond blocker)")
	}
}

func TestBishopAttacks(t *testing.T) {
	// Bishop on e4 with no blockers
	occupied := Bitboard(0)
	attacks := BishopAttacksBB(ParseSquare("e4"), occupied)

	// Should attack diagonals (13 squares)
	if attacks.Count() != 13 {
		t.Errorf("Bishop on e4 (empty board) should have 13 attacks, got %d", attacks.Count())
	}

	// Verify attacks slow matches magic lookup
	slow := bishopAttacksSlow(ParseSquare("e4"), occupied)
	if attacks != slow {
		t.Error("Magic bishop attacks don't match slow computation (empty board)")
	}

	// Bishop with blockers
	occupied = SquareBB(ParseSquare("g6")) | SquareBB(ParseSquare("c2"))
	attacks = BishopAttacksBB(ParseSquare("e4"), occupied)
	slow = bishopAttacksSlow(ParseSquare("e4"), occupied)

	if attacks != slow {
		t.Error("Magic bishop attacks don't match slow computation (with blockers)")
	}
}

func TestQueenAttacks(t *testing.T) {
	// Queen attacks = rook attacks + bishop attacks
	occupied := SquareBB(ParseSquare("e6")) | SquareBB(ParseSquare("g6"))
	sq := ParseSquare("e4")

	queen := QueenAttacksBB(sq, occupied)
	rook := RookAttacksBB(sq, occupied)
	bishop := BishopAttacksBB(sq, occupied)

	if queen != (rook | bishop) {
		t.Error("Queen attacks should equal rook | bishop attacks")
	}
}

func TestLineBB(t *testing.T) {
	// Same rank: e4-h4
	e4, h4 := ParseSquare("e4"), ParseSquare("h4")
	line := LineBB[e4][h4]
	// Should contain all of rank 4
	for f := 0; f < 8; f++ {
		sq := NewSquare(f, 3)
		if !line.IsSet(sq) {
			t.Errorf("LineBB[e4][h4] should contain %s", sq)
		}
	}

	// Same file: e2-e7
	e2, e7 := ParseSquare("e2"), ParseSquare("e7")
	line = LineBB[e2][e7]
	for r := 0; r < 8; r++ {
		sq := NewSquare(4, r)
		if !line.IsSet(sq) {
			t.Errorf("LineBB[e2][e7] should contain %s", sq)
		}
	}

	// Diagonal: a1-h8
	a1, h8 := ParseSquare("a1"), ParseSquare("h8")
	line = LineBB[a1][h8]
	for i := 0; i < 8; i++ {
		sq := NewSquare(i, i)
		if !line.IsSet(sq) {
			t.Errorf("LineBB[a1][h8] should contain %s", sq)
		}
	}

	// Not on same line: a1-b3
	a1 = ParseSquare("a1")
	b3 := ParseSquare("b3")
	if LineBB[a1][b3] != 0 {
		t.Error("LineBB[a1][b3] should be 0 (not aligned)")
	}
}

func TestBetweenBB(t *testing.T) {
	// Between e2 and e6: e3, e4, e5
	e2, e6 := ParseSquare("e2"), ParseSquare("e6")
	between := BetweenBB[e2][e6]
	if !between.IsSet(ParseSquare("e3")) || !between.IsSet(ParseSquare("e4")) || !between.IsSet(ParseSquare("e5")) {
		t.Error("BetweenBB[e2][e6] should contain e3, e4, e5")
	}
	if between.Count() != 3 {
		t.Errorf("BetweenBB[e2][e6] should have 3 squares, got %d", between.Count())
	}

	// Adjacent squares: e4-e5 should have empty between
	e4, e5 := ParseSquare("e4"), ParseSquare("e5")
	if BetweenBB[e4][e5] != 0 {
		t.Error("BetweenBB[e4][e5] should be 0 (adjacent)")
	}

	// Diagonal: a1-d4 → b2, c3
	a1, d4 := ParseSquare("a1"), ParseSquare("d4")
	between = BetweenBB[a1][d4]
	if between.Count() != 2 || !between.IsSet(ParseSquare("b2")) || !between.IsSet(ParseSquare("c3")) {
		t.Error("BetweenBB[a1][d4] should contain b2, c3")
	}

	// Not aligned: a1-b3
	a1 = ParseSquare("a1")
	b3 := ParseSquare("b3")
	if BetweenBB[a1][b3] != 0 {
		t.Error("BetweenBB[a1][b3] should be 0 (not aligned)")
	}
}

func TestPinnedPieces(t *testing.T) {
	tests := []struct {
		name    string
		fen     string
		us      Color
		pinned  []string // squares of pinned pieces
	}{
		{
			name:   "rook pin on file",
			fen:    "4k3/8/8/8/4r3/8/4N3/4K3 w - - 0 1",
			us:     White,
			pinned: []string{"e2"},
		},
		{
			name:   "bishop pin on diagonal",
			fen:    "4k3/8/8/6b1/8/4N3/8/2K5 w - - 0 1",
			us:     White,
			pinned: []string{"e3"},
		},
		{
			name:   "no pins",
			fen:    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			us:     White,
			pinned: nil,
		},
		{
			name:   "double pin (two snipers)",
			fen:    "4k3/4r3/8/b7/8/8/3NR3/4K3 w - - 0 1",
			us:     White,
			pinned: []string{"e2", "d2"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var b Board
			b.SetFEN(tc.fen)
			pinned := b.PinnedPieces(tc.us)

			if len(tc.pinned) == 0 {
				if pinned != 0 {
					t.Errorf("expected no pinned pieces, got %d", pinned.Count())
				}
				return
			}

			if pinned.Count() != len(tc.pinned) {
				t.Errorf("expected %d pinned pieces, got %d", len(tc.pinned), pinned.Count())
			}

			for _, sq := range tc.pinned {
				if !pinned.IsSet(ParseSquare(sq)) {
					t.Errorf("expected %s to be pinned", sq)
				}
			}
		})
	}
}

func TestIsLegalMatchesMakeUnmake(t *testing.T) {
	// Verify IsLegal matches the old make/unmake approach for complex positions
	positions := []string{
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // Kiwipete
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                               // Position 3
		"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",       // Position 4
		"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",              // Position 5
		"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/3P1N1P/PPP1NPP1/R2Q1RK1 w - - 0 1", // Position 6
		"r1bqk2r/ppppbppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",     // Ruy Lopez-ish
		"rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 1",           // EP available
	}

	for _, fen := range positions {
		var b Board
		b.SetFEN(fen)
		moves := b.GenerateAllMoves()
		pinned, checkers := b.PinnedAndCheckers(b.SideToMove)
		inCheck := checkers != 0

		for _, m := range moves {
			// Old approach: make/unmake
			us := b.SideToMove
			b.MakeMove(m)
			kingSq := b.Pieces[pieceOf(WhiteKing, us)].LSB()
			oldResult := !b.IsAttacked(kingSq, 1-us)
			b.UnmakeMove(m)

			// New approach
			newResult := b.IsLegal(m, pinned, inCheck)

			if oldResult != newResult {
				t.Errorf("FEN=%s move=%s: old=%v new=%v", fen, m, oldResult, newResult)
			}
		}
	}
}

func TestMagicConsistency(t *testing.T) {
	// Test that magic lookups match slow computation for all squares
	// with various occupancy patterns
	for sq := Square(0); sq < 64; sq++ {
		// Test empty board
		if RookAttacksBB(sq, 0) != rookAttacksSlow(sq, 0) {
			t.Errorf("Rook magic mismatch at %s (empty)", sq.String())
		}
		if BishopAttacksBB(sq, 0) != bishopAttacksSlow(sq, 0) {
			t.Errorf("Bishop magic mismatch at %s (empty)", sq.String())
		}

		// Test with some random occupancy patterns
		patterns := []Bitboard{
			0x0000001818000000, // Center squares
			0xFF000000000000FF, // Ranks 1 and 8
			0x8181818181818181, // Files a and h
		}

		for _, occ := range patterns {
			if RookAttacksBB(sq, occ) != rookAttacksSlow(sq, occ) {
				t.Errorf("Rook magic mismatch at %s with occupancy %x", sq.String(), occ)
			}
			if BishopAttacksBB(sq, occ) != bishopAttacksSlow(sq, occ) {
				t.Errorf("Bishop magic mismatch at %s with occupancy %x", sq.String(), occ)
			}
		}
	}
}

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

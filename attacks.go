package chess

// Attack tables for non-sliding pieces
var (
	KnightAttacks [64]Bitboard
	KingAttacks   [64]Bitboard
	PawnAttacks   [2][64]Bitboard // [color][square]
)

// Magic bitboard structures for sliding pieces
type Magic struct {
	Mask   Bitboard   // Relevant occupancy mask
	Magic  uint64     // Magic number
	Shift  int        // Right shift amount (64 - bits in mask)
	Offset int        // Offset into attack table
}

var (
	RookMagics   [64]Magic
	BishopMagics [64]Magic

	// Attack tables - indexed by magic index
	RookAttacks   []Bitboard
	BishopAttacks []Bitboard
)

// Precomputed magic numbers for rooks (from well-known sources)
var rookMagicNumbers = [64]uint64{
	0x0080001020400080, 0x0040001000200040, 0x0080081000200080, 0x0080040800100080,
	0x0080020400080080, 0x0080010200040080, 0x0080008001000200, 0x0080002040800100,
	0x0000800020400080, 0x0000400020005000, 0x0000801000200080, 0x0000800800100080,
	0x0000800400080080, 0x0000800200040080, 0x0000800100020080, 0x0000800040800100,
	0x0000208000400080, 0x0000404000201000, 0x0000808010002000, 0x0000808008001000,
	0x0000808004000800, 0x0000808002000400, 0x0000010100020004, 0x0000020000408104,
	0x0000208080004000, 0x0000200040005000, 0x0000100080200080, 0x0000080080100080,
	0x0000040080080080, 0x0000020080040080, 0x0000010080800200, 0x0000800080004100,
	0x0000204000800080, 0x0000200040401000, 0x0000100080802000, 0x0000080080801000,
	0x0000040080800800, 0x0000020080800400, 0x0000020001010004, 0x0000800040800100,
	0x0000204000808000, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
	0x0000040008008080, 0x0000020004008080, 0x0000010002008080, 0x0000004081020004,
	0x0000204000800080, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
	0x0000040008008080, 0x0000020004008080, 0x0000800100020080, 0x0000800041000080,
	0x00FFFCDDFCED714A, 0x007FFCDDFCED714A, 0x003FFFCDFFD88096, 0x0000040810002101,
	0x0001000204080011, 0x0001000204000801, 0x0001000082000401, 0x0001FFFAABFAD1A2,
}

// Precomputed magic numbers for bishops
var bishopMagicNumbers = [64]uint64{
	0x0002020202020200, 0x0002020202020000, 0x0004010202000000, 0x0004040080000000,
	0x0001104000000000, 0x0000821040000000, 0x0000410410400000, 0x0000104104104000,
	0x0000040404040400, 0x0000020202020200, 0x0000040102020000, 0x0000040400800000,
	0x0000011040000000, 0x0000008210400000, 0x0000004104104000, 0x0000002082082000,
	0x0004000808080800, 0x0002000404040400, 0x0001000202020200, 0x0000800802004000,
	0x0000800400A00000, 0x0000200100884000, 0x0000400082082000, 0x0000200041041000,
	0x0002080010101000, 0x0001040008080800, 0x0000208004010400, 0x0000404004010200,
	0x0000840000802000, 0x0000404002011000, 0x0000808001041000, 0x0000404000820800,
	0x0001041000202000, 0x0000820800101000, 0x0000104400080800, 0x0000020080080080,
	0x0000404040040100, 0x0000808100020100, 0x0001010100020800, 0x0000808080010400,
	0x0000820820004000, 0x0000410410002000, 0x0000082088001000, 0x0000002011000800,
	0x0000080100400400, 0x0001010101000200, 0x0002020202000400, 0x0001010101000200,
	0x0000410410400000, 0x0000208208200000, 0x0000002084100000, 0x0000000020880000,
	0x0000001002020000, 0x0000040408020000, 0x0004040404040000, 0x0002020202020000,
	0x0000104104104000, 0x0000002082082000, 0x0000000020841000, 0x0000000000208800,
	0x0000000010020200, 0x0000000404080200, 0x0000040404040400, 0x0002020202020200,
}

// Number of relevant bits for each square
var rookRelevantBits = [64]int{
	12, 11, 11, 11, 11, 11, 11, 12,
	11, 10, 10, 10, 10, 10, 10, 11,
	11, 10, 10, 10, 10, 10, 10, 11,
	11, 10, 10, 10, 10, 10, 10, 11,
	11, 10, 10, 10, 10, 10, 10, 11,
	11, 10, 10, 10, 10, 10, 10, 11,
	11, 10, 10, 10, 10, 10, 10, 11,
	12, 11, 11, 11, 11, 11, 11, 12,
}

var bishopRelevantBits = [64]int{
	6, 5, 5, 5, 5, 5, 5, 6,
	5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 7, 7, 7, 7, 5, 5,
	5, 5, 7, 9, 9, 7, 5, 5,
	5, 5, 7, 9, 9, 7, 5, 5,
	5, 5, 7, 7, 7, 7, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5,
	6, 5, 5, 5, 5, 5, 5, 6,
}

func init() {
	initKnightAttacks()
	initKingAttacks()
	initPawnAttacks()
	initMagics()
}

func initKnightAttacks() {
	for sq := Square(0); sq < 64; sq++ {
		bb := SquareBB(sq)
		attacks := Bitboard(0)

		// All 8 knight moves
		// Moving right needs NotFileH/NotFileGH, moving left needs NotFileA/NotFileAB
		attacks |= (bb & NotFileGH) >> 6  // 1 down, 2 right
		attacks |= (bb & NotFileH) >> 15  // 2 down, 1 right
		attacks |= (bb & NotFileA) >> 17  // 2 down, 1 left
		attacks |= (bb & NotFileAB) >> 10 // 1 down, 2 left

		attacks |= (bb & NotFileAB) << 6  // 1 up, 2 left
		attacks |= (bb & NotFileA) << 15  // 2 up, 1 left
		attacks |= (bb & NotFileH) << 17  // 2 up, 1 right
		attacks |= (bb & NotFileGH) << 10 // 1 up, 2 right

		KnightAttacks[sq] = attacks
	}
}

func initKingAttacks() {
	for sq := Square(0); sq < 64; sq++ {
		bb := SquareBB(sq)
		attacks := Bitboard(0)

		attacks |= bb.North()
		attacks |= bb.South()
		attacks |= bb.East()
		attacks |= bb.West()
		attacks |= bb.NorthEast()
		attacks |= bb.NorthWest()
		attacks |= bb.SouthEast()
		attacks |= bb.SouthWest()

		KingAttacks[sq] = attacks
	}
}

func initPawnAttacks() {
	for sq := Square(0); sq < 64; sq++ {
		bb := SquareBB(sq)

		// White pawn attacks (moving north)
		PawnAttacks[White][sq] = bb.NorthEast() | bb.NorthWest()

		// Black pawn attacks (moving south)
		PawnAttacks[Black][sq] = bb.SouthEast() | bb.SouthWest()
	}
}

// Generate rook relevant occupancy mask (excludes edge squares)
func rookMask(sq Square) Bitboard {
	var mask Bitboard
	rank := sq.Rank()
	file := sq.File()

	// North (exclude rank 8)
	for r := rank + 1; r < 7; r++ {
		mask.Set(NewSquare(file, r))
	}
	// South (exclude rank 1)
	for r := rank - 1; r > 0; r-- {
		mask.Set(NewSquare(file, r))
	}
	// East (exclude file h)
	for f := file + 1; f < 7; f++ {
		mask.Set(NewSquare(f, rank))
	}
	// West (exclude file a)
	for f := file - 1; f > 0; f-- {
		mask.Set(NewSquare(f, rank))
	}

	return mask
}

// Generate bishop relevant occupancy mask (excludes edge squares)
func bishopMask(sq Square) Bitboard {
	var mask Bitboard
	rank := sq.Rank()
	file := sq.File()

	// NE diagonal
	for r, f := rank+1, file+1; r < 7 && f < 7; r, f = r+1, f+1 {
		mask.Set(NewSquare(f, r))
	}
	// NW diagonal
	for r, f := rank+1, file-1; r < 7 && f > 0; r, f = r+1, f-1 {
		mask.Set(NewSquare(f, r))
	}
	// SE diagonal
	for r, f := rank-1, file+1; r > 0 && f < 7; r, f = r-1, f+1 {
		mask.Set(NewSquare(f, r))
	}
	// SW diagonal
	for r, f := rank-1, file-1; r > 0 && f > 0; r, f = r-1, f-1 {
		mask.Set(NewSquare(f, r))
	}

	return mask
}

// Generate rook attacks for a given square and occupancy
func rookAttacksSlow(sq Square, occupied Bitboard) Bitboard {
	var attacks Bitboard
	rank := sq.Rank()
	file := sq.File()

	// North
	for r := rank + 1; r <= 7; r++ {
		s := NewSquare(file, r)
		attacks.Set(s)
		if occupied.IsSet(s) {
			break
		}
	}
	// South
	for r := rank - 1; r >= 0; r-- {
		s := NewSquare(file, r)
		attacks.Set(s)
		if occupied.IsSet(s) {
			break
		}
	}
	// East
	for f := file + 1; f <= 7; f++ {
		s := NewSquare(f, rank)
		attacks.Set(s)
		if occupied.IsSet(s) {
			break
		}
	}
	// West
	for f := file - 1; f >= 0; f-- {
		s := NewSquare(f, rank)
		attacks.Set(s)
		if occupied.IsSet(s) {
			break
		}
	}

	return attacks
}

// Generate bishop attacks for a given square and occupancy
func bishopAttacksSlow(sq Square, occupied Bitboard) Bitboard {
	var attacks Bitboard
	rank := sq.Rank()
	file := sq.File()

	// NE
	for r, f := rank+1, file+1; r <= 7 && f <= 7; r, f = r+1, f+1 {
		s := NewSquare(f, r)
		attacks.Set(s)
		if occupied.IsSet(s) {
			break
		}
	}
	// NW
	for r, f := rank+1, file-1; r <= 7 && f >= 0; r, f = r+1, f-1 {
		s := NewSquare(f, r)
		attacks.Set(s)
		if occupied.IsSet(s) {
			break
		}
	}
	// SE
	for r, f := rank-1, file+1; r >= 0 && f <= 7; r, f = r-1, f+1 {
		s := NewSquare(f, r)
		attacks.Set(s)
		if occupied.IsSet(s) {
			break
		}
	}
	// SW
	for r, f := rank-1, file-1; r >= 0 && f >= 0; r, f = r-1, f-1 {
		s := NewSquare(f, r)
		attacks.Set(s)
		if occupied.IsSet(s) {
			break
		}
	}

	return attacks
}

// Generate all occupancy variations for a given mask
func generateOccupancies(mask Bitboard) []Bitboard {
	bits := mask.Count()
	n := 1 << bits
	occupancies := make([]Bitboard, n)

	// Get indices of set bits in mask
	indices := make([]Square, bits)
	temp := mask
	for i := 0; i < bits; i++ {
		indices[i] = temp.PopLSB()
	}

	// Generate all 2^bits combinations
	for i := 0; i < n; i++ {
		var occ Bitboard
		for j := 0; j < bits; j++ {
			if i&(1<<j) != 0 {
				occ.Set(indices[j])
			}
		}
		occupancies[i] = occ
	}

	return occupancies
}

func initMagics() {
	// Calculate total size needed for attack tables
	rookOffset := 0
	bishopOffset := 0

	for sq := Square(0); sq < 64; sq++ {
		rookOffset += 1 << rookRelevantBits[sq]
		bishopOffset += 1 << bishopRelevantBits[sq]
	}

	RookAttacks = make([]Bitboard, rookOffset)
	BishopAttacks = make([]Bitboard, bishopOffset)

	// Initialize rook magics
	offset := 0
	for sq := Square(0); sq < 64; sq++ {
		mask := rookMask(sq)
		bits := rookRelevantBits[sq]
		magic := rookMagicNumbers[sq]

		RookMagics[sq] = Magic{
			Mask:   mask,
			Magic:  magic,
			Shift:  64 - bits,
			Offset: offset,
		}

		// Fill attack table for all occupancy variations
		occupancies := generateOccupancies(mask)
		for _, occ := range occupancies {
			index := int((uint64(occ) * magic) >> (64 - bits))
			RookAttacks[offset+index] = rookAttacksSlow(sq, occ)
		}

		offset += 1 << bits
	}

	// Initialize bishop magics
	offset = 0
	for sq := Square(0); sq < 64; sq++ {
		mask := bishopMask(sq)
		bits := bishopRelevantBits[sq]
		magic := bishopMagicNumbers[sq]

		BishopMagics[sq] = Magic{
			Mask:   mask,
			Magic:  magic,
			Shift:  64 - bits,
			Offset: offset,
		}

		// Fill attack table for all occupancy variations
		occupancies := generateOccupancies(mask)
		for _, occ := range occupancies {
			index := int((uint64(occ) * magic) >> (64 - bits))
			BishopAttacks[offset+index] = bishopAttacksSlow(sq, occ)
		}

		offset += 1 << bits
	}
}

// RookAttacksBB returns the rook attack bitboard for a square given occupancy
func RookAttacksBB(sq Square, occupied Bitboard) Bitboard {
	m := &RookMagics[sq]
	index := int((uint64(occupied&m.Mask) * m.Magic) >> m.Shift)
	return RookAttacks[m.Offset+index]
}

// BishopAttacksBB returns the bishop attack bitboard for a square given occupancy
func BishopAttacksBB(sq Square, occupied Bitboard) Bitboard {
	m := &BishopMagics[sq]
	index := int((uint64(occupied&m.Mask) * m.Magic) >> m.Shift)
	return BishopAttacks[m.Offset+index]
}

// QueenAttacksBB returns the queen attack bitboard (rook + bishop attacks)
func QueenAttacksBB(sq Square, occupied Bitboard) Bitboard {
	return RookAttacksBB(sq, occupied) | BishopAttacksBB(sq, occupied)
}

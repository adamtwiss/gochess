package chess

import "math/bits"

// Bitboard represents a 64-bit board with one bit per square
// Bit 0 = a1, Bit 7 = h1, Bit 8 = a2, ..., Bit 63 = h8
type Bitboard uint64

// File and rank masks
const (
	FileA Bitboard = 0x0101010101010101
	FileB Bitboard = 0x0202020202020202
	FileC Bitboard = 0x0404040404040404
	FileD Bitboard = 0x0808080808080808
	FileE Bitboard = 0x1010101010101010
	FileF Bitboard = 0x2020202020202020
	FileG Bitboard = 0x4040404040404040
	FileH Bitboard = 0x8080808080808080

	Rank1 Bitboard = 0x00000000000000FF
	Rank2 Bitboard = 0x000000000000FF00
	Rank3 Bitboard = 0x0000000000FF0000
	Rank4 Bitboard = 0x00000000FF000000
	Rank5 Bitboard = 0x000000FF00000000
	Rank6 Bitboard = 0x0000FF0000000000
	Rank7 Bitboard = 0x00FF000000000000
	Rank8 Bitboard = 0xFF00000000000000

	// Square color masks (for bishop color detection)
	LightSquares Bitboard = 0x55AA55AA55AA55AA
	DarkSquares  Bitboard = 0xAA55AA55AA55AA55

	// Useful combinations
	NotFileA Bitboard = ^FileA
	NotFileH Bitboard = ^FileH
	NotFileAB Bitboard = ^(FileA | FileB)
	NotFileGH Bitboard = ^(FileG | FileH)
)

// SquareBB returns a bitboard with only the given square set
func SquareBB(sq Square) Bitboard {
	return Bitboard(1) << sq
}

// Set sets the bit at the given square
func (b *Bitboard) Set(sq Square) {
	*b |= SquareBB(sq)
}

// Clear clears the bit at the given square
func (b *Bitboard) Clear(sq Square) {
	*b &^= SquareBB(sq)
}

// IsSet returns true if the bit at the given square is set
func (b Bitboard) IsSet(sq Square) bool {
	return b&SquareBB(sq) != 0
}

// Count returns the number of set bits (population count)
func (b Bitboard) Count() int {
	return bits.OnesCount64(uint64(b))
}

// LSB returns the least significant bit (lowest square index)
func (b Bitboard) LSB() Square {
	if b == 0 {
		return NoSquare
	}
	return Square(bits.TrailingZeros64(uint64(b)))
}

// PopLSB removes and returns the least significant bit
func (b *Bitboard) PopLSB() Square {
	sq := b.LSB()
	*b &= *b - 1 // Clear the LSB
	return sq
}

// MSB returns the most significant bit (highest square index)
func (b Bitboard) MSB() Square {
	if b == 0 {
		return NoSquare
	}
	return Square(63 - bits.LeadingZeros64(uint64(b)))
}

// String returns a visual representation of the bitboard
func (b Bitboard) String() string {
	var result string
	result += "  +---+---+---+---+---+---+---+---+\n"
	for rank := 7; rank >= 0; rank-- {
		result += string(rune('1'+rank)) + " |"
		for file := 0; file < 8; file++ {
			sq := Square(rank*8 + file)
			if b.IsSet(sq) {
				result += " X |"
			} else {
				result += "   |"
			}
		}
		result += "\n  +---+---+---+---+---+---+---+---+\n"
	}
	result += "    a   b   c   d   e   f   g   h\n"
	return result
}

// Shift operations for move generation

// North shifts all bits up one rank
func (b Bitboard) North() Bitboard {
	return b << 8
}

// South shifts all bits down one rank
func (b Bitboard) South() Bitboard {
	return b >> 8
}

// East shifts all bits right one file (with wrap protection)
func (b Bitboard) East() Bitboard {
	return (b & NotFileH) << 1
}

// West shifts all bits left one file (with wrap protection)
func (b Bitboard) West() Bitboard {
	return (b & NotFileA) >> 1
}

// NorthEast shifts all bits up-right
func (b Bitboard) NorthEast() Bitboard {
	return (b & NotFileH) << 9
}

// NorthWest shifts all bits up-left
func (b Bitboard) NorthWest() Bitboard {
	return (b & NotFileA) << 7
}

// SouthEast shifts all bits down-right
func (b Bitboard) SouthEast() Bitboard {
	return (b & NotFileH) >> 7
}

// SouthWest shifts all bits down-left
func (b Bitboard) SouthWest() Bitboard {
	return (b & NotFileA) >> 9
}

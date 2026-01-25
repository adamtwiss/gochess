package chess

// Move represents a chess move encoded in 16 bits:
// bits 0-5:   from square (0-63)
// bits 6-11:  to square (0-63)
// bits 12-15: flags (promotion piece, special moves)
type Move uint16

// Move flags (4 bits: bits 12-15)
const (
	FlagNone      = 0
	FlagEnPassant = 1
	FlagCastle    = 2
	FlagPromoteN  = 4  // Knight promotion
	FlagPromoteB  = 5  // Bishop promotion
	FlagPromoteR  = 6  // Rook promotion
	FlagPromoteQ  = 7  // Queen promotion
	FlagPromotion = 4  // Any promotion has bit 2 set
)

// NoMove represents an invalid/null move
const NoMove Move = 0

// NewMove creates a move from source and destination squares
func NewMove(from, to Square) Move {
	return Move(from) | Move(to)<<6
}

// NewMoveFlags creates a move with flags
func NewMoveFlags(from, to Square, flags int) Move {
	return Move(from) | Move(to)<<6 | Move(flags)<<12
}

// From returns the source square
func (m Move) From() Square {
	return Square(m & 0x3F)
}

// To returns the destination square
func (m Move) To() Square {
	return Square((m >> 6) & 0x3F)
}

// Flags returns the move flags
func (m Move) Flags() int {
	return int(m >> 12)
}

// IsEnPassant returns true if this is an en passant capture
func (m Move) IsEnPassant() bool {
	return m.Flags()&FlagEnPassant != 0
}

// IsCastle returns true if this is a castling move
func (m Move) IsCastle() bool {
	return m.Flags()&FlagCastle != 0
}

// IsPromotion returns true if this is a pawn promotion
func (m Move) IsPromotion() bool {
	return m.Flags()&FlagPromotion != 0
}

// PromotionPiece returns the piece to promote to (for white)
// Caller should adjust for color if needed
func (m Move) PromotionPiece() Piece {
	if !m.IsPromotion() {
		return Empty
	}
	switch m.Flags() {
	case FlagPromoteN:
		return WhiteKnight
	case FlagPromoteB:
		return WhiteBishop
	case FlagPromoteR:
		return WhiteRook
	case FlagPromoteQ:
		return WhiteQueen
	}
	return Empty
}

// String returns the move in UCI notation (e.g., "e2e4", "e7e8q")
func (m Move) String() string {
	if m == NoMove {
		return "0000"
	}
	s := m.From().String() + m.To().String()
	if m.IsPromotion() {
		switch m.Flags() {
		case FlagPromoteN:
			s += "n"
		case FlagPromoteB:
			s += "b"
		case FlagPromoteR:
			s += "r"
		case FlagPromoteQ:
			s += "q"
		}
	}
	return s
}

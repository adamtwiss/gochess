package chess

// Piece represents a chess piece
type Piece int8

const (
	Empty Piece = iota
	WhitePawn
	WhiteKnight
	WhiteBishop
	WhiteRook
	WhiteQueen
	WhiteKing
	BlackPawn
	BlackKnight
	BlackBishop
	BlackRook
	BlackQueen
	BlackKing
)

// PieceColor returns the color of a piece (undefined for Empty)
func (p Piece) Color() Color {
	if p >= BlackPawn {
		return Black
	}
	return White
}

// Color represents a player color
type Color int8

const (
	White Color = iota
	Black
)

// Square represents a position on the board (0-63)
type Square int8

const NoSquare Square = -1

// File returns the file (column) of the square (0-7, a-h)
func (s Square) File() int {
	return int(s) % 8
}

// Rank returns the rank (row) of the square (0-7, 1-8)
func (s Square) Rank() int {
	return int(s) / 8
}

// NewSquare creates a square from file and rank (0-indexed)
func NewSquare(file, rank int) Square {
	return Square(rank*8 + file)
}

// String returns algebraic notation for the square (e.g., "e4")
func (s Square) String() string {
	if s == NoSquare {
		return "-"
	}
	return string(rune('a'+s.File())) + string(rune('1'+s.Rank()))
}

// ParseSquare converts algebraic notation to a Square
func ParseSquare(s string) Square {
	if s == "-" || len(s) != 2 {
		return NoSquare
	}
	file := int(s[0] - 'a')
	rank := int(s[1] - '1')
	if file < 0 || file > 7 || rank < 0 || rank > 7 {
		return NoSquare
	}
	return NewSquare(file, rank)
}

// CastlingRights represents the castling availability
type CastlingRights uint8

const (
	WhiteKingside  CastlingRights = 1 << iota // K
	WhiteQueenside                            // Q
	BlackKingside                             // k
	BlackQueenside                            // q
	NoCastling     CastlingRights = 0
	AllCastling    CastlingRights = WhiteKingside | WhiteQueenside | BlackKingside | BlackQueenside
)

// Board represents a chess board position
type Board struct {
	Squares       [64]Piece
	SideToMove    Color
	Castling      CastlingRights
	EnPassant     Square
	HalfmoveClock int
	FullmoveNum   int
	HashKey       uint64

	// Bitboards for fast move generation
	Pieces    [13]Bitboard // One bitboard per piece type (index 0 unused)
	Occupied  [2]Bitboard  // All pieces by color [White], [Black]
	AllPieces Bitboard     // All pieces on board

	// Undo stack for MakeMove/UnmakeMove (per-board to avoid sharing issues)
	UndoStack []UndoInfo
}

// Clear removes all pieces from the board and resets state
func (b *Board) Clear() {
	for i := range b.Squares {
		b.Squares[i] = Empty
	}
	for i := range b.Pieces {
		b.Pieces[i] = 0
	}
	b.Occupied[White] = 0
	b.Occupied[Black] = 0
	b.AllPieces = 0
	b.SideToMove = White
	b.Castling = NoCastling
	b.EnPassant = NoSquare
	b.HalfmoveClock = 0
	b.FullmoveNum = 1
	b.HashKey = 0
	// Reset undo stack (keep capacity if already allocated)
	if b.UndoStack == nil {
		b.UndoStack = make([]UndoInfo, 0, 256)
	} else {
		b.UndoStack = b.UndoStack[:0]
	}
}

// putPiece places a piece on a square, updating all board representations
func (b *Board) putPiece(piece Piece, sq Square) {
	bb := SquareBB(sq)
	b.Squares[sq] = piece
	b.Pieces[piece] |= bb
	b.Occupied[piece.Color()] |= bb
	b.AllPieces |= bb
}

// removePiece removes a piece from a square, updating all board representations
func (b *Board) removePiece(sq Square) Piece {
	piece := b.Squares[sq]
	if piece == Empty {
		return Empty
	}
	bb := SquareBB(sq)
	b.Squares[sq] = Empty
	b.Pieces[piece] &^= bb
	b.Occupied[piece.Color()] &^= bb
	b.AllPieces &^= bb
	return piece
}

// movePiece moves a piece from one square to another (no capture handling)
func (b *Board) movePiece(from, to Square) {
	piece := b.Squares[from]
	fromBB := SquareBB(from)
	toBB := SquareBB(to)
	moveBB := fromBB | toBB

	b.Squares[from] = Empty
	b.Squares[to] = piece
	b.Pieces[piece] ^= moveBB
	b.Occupied[piece.Color()] ^= moveBB
	b.AllPieces ^= moveBB
}

// Reset sets the board to the standard starting position
func (b *Board) Reset() {
	b.Clear()

	// White pieces
	b.putPiece(WhiteRook, NewSquare(0, 0))
	b.putPiece(WhiteKnight, NewSquare(1, 0))
	b.putPiece(WhiteBishop, NewSquare(2, 0))
	b.putPiece(WhiteQueen, NewSquare(3, 0))
	b.putPiece(WhiteKing, NewSquare(4, 0))
	b.putPiece(WhiteBishop, NewSquare(5, 0))
	b.putPiece(WhiteKnight, NewSquare(6, 0))
	b.putPiece(WhiteRook, NewSquare(7, 0))
	for f := 0; f < 8; f++ {
		b.putPiece(WhitePawn, NewSquare(f, 1))
	}

	// Black pieces
	b.putPiece(BlackRook, NewSquare(0, 7))
	b.putPiece(BlackKnight, NewSquare(1, 7))
	b.putPiece(BlackBishop, NewSquare(2, 7))
	b.putPiece(BlackQueen, NewSquare(3, 7))
	b.putPiece(BlackKing, NewSquare(4, 7))
	b.putPiece(BlackBishop, NewSquare(5, 7))
	b.putPiece(BlackKnight, NewSquare(6, 7))
	b.putPiece(BlackRook, NewSquare(7, 7))
	for f := 0; f < 8; f++ {
		b.putPiece(BlackPawn, NewSquare(f, 6))
	}

	b.SideToMove = White
	b.Castling = AllCastling
	b.EnPassant = NoSquare
	b.HalfmoveClock = 0
	b.FullmoveNum = 1
	b.HashKey = b.Hash()
}

// SetFEN parses a FEN string and sets the board position
func (b *Board) SetFEN(fen string) error {
	b.Clear()

	parts := splitFEN(fen)
	if len(parts) < 1 {
		return &FENError{"empty FEN string"}
	}

	// Parse piece placement
	rank := 7
	file := 0
	for _, c := range parts[0] {
		switch {
		case c == '/':
			rank--
			file = 0
		case c >= '1' && c <= '8':
			file += int(c - '0')
		default:
			piece := fenToPiece(c)
			if piece == Empty {
				return &FENError{"invalid piece character: " + string(c)}
			}
			b.putPiece(piece, NewSquare(file, rank))
			file++
		}
	}

	// Parse side to move
	if len(parts) >= 2 {
		switch parts[1] {
		case "w":
			b.SideToMove = White
		case "b":
			b.SideToMove = Black
		default:
			return &FENError{"invalid side to move: " + parts[1]}
		}
	}

	// Parse castling rights
	if len(parts) >= 3 {
		b.Castling = NoCastling
		if parts[2] != "-" {
			for _, c := range parts[2] {
				switch c {
				case 'K':
					b.Castling |= WhiteKingside
				case 'Q':
					b.Castling |= WhiteQueenside
				case 'k':
					b.Castling |= BlackKingside
				case 'q':
					b.Castling |= BlackQueenside
				}
			}
		}
	}

	// Parse en passant square
	if len(parts) >= 4 {
		b.EnPassant = ParseSquare(parts[3])
	}

	// Parse halfmove clock
	if len(parts) >= 5 {
		b.HalfmoveClock = atoi(parts[4])
	}

	// Parse fullmove number
	if len(parts) >= 6 {
		b.FullmoveNum = atoi(parts[5])
		if b.FullmoveNum < 1 {
			b.FullmoveNum = 1
		}
	}

	b.HashKey = b.Hash()
	return nil
}

// Move moves a piece from one square to another
// This is a basic implementation that doesn't validate legality
func (b *Board) Move(from, to Square) {
	piece := b.Squares[from]
	captured := b.Squares[to]
	oldCastling := b.Castling
	oldEnPassant := b.EnPassant

	// Remove captured piece from bitboards (if any)
	if captured != Empty {
		b.removePiece(to)
		b.HashKey ^= Zobrist.Pieces[captured][to]
	}

	// Move the piece (updates Squares and bitboards)
	b.movePiece(from, to)

	// Update hash for the move
	b.HashKey ^= Zobrist.Pieces[piece][from]
	b.HashKey ^= Zobrist.Pieces[piece][to]

	// Handle en passant capture
	if (piece == WhitePawn || piece == BlackPawn) && to == oldEnPassant {
		if piece == WhitePawn {
			capturedSq := to - 8
			b.HashKey ^= Zobrist.Pieces[BlackPawn][capturedSq]
			b.removePiece(capturedSq)
		} else {
			capturedSq := to + 8
			b.HashKey ^= Zobrist.Pieces[WhitePawn][capturedSq]
			b.removePiece(capturedSq)
		}
	}

	// Update en passant square
	b.EnPassant = NoSquare
	if piece == WhitePawn && to-from == 16 {
		b.EnPassant = from + 8
	} else if piece == BlackPawn && from-to == 16 {
		b.EnPassant = from - 8
	}

	// Update castling rights when pieces move
	switch from {
	case NewSquare(4, 0): // White king moves
		b.Castling &^= WhiteKingside | WhiteQueenside
	case NewSquare(4, 7): // Black king moves
		b.Castling &^= BlackKingside | BlackQueenside
	case NewSquare(0, 0): // White queenside rook
		b.Castling &^= WhiteQueenside
	case NewSquare(7, 0): // White kingside rook
		b.Castling &^= WhiteKingside
	case NewSquare(0, 7): // Black queenside rook
		b.Castling &^= BlackQueenside
	case NewSquare(7, 7): // Black kingside rook
		b.Castling &^= BlackKingside
	}

	// Update castling rights when rooks are captured
	switch to {
	case NewSquare(0, 0):
		b.Castling &^= WhiteQueenside
	case NewSquare(7, 0):
		b.Castling &^= WhiteKingside
	case NewSquare(0, 7):
		b.Castling &^= BlackQueenside
	case NewSquare(7, 7):
		b.Castling &^= BlackKingside
	}

	// Handle castling move (move the rook)
	if piece == WhiteKing && from == NewSquare(4, 0) {
		if to == NewSquare(6, 0) { // Kingside
			b.movePiece(NewSquare(7, 0), NewSquare(5, 0))
			b.HashKey ^= Zobrist.Pieces[WhiteRook][NewSquare(7, 0)]
			b.HashKey ^= Zobrist.Pieces[WhiteRook][NewSquare(5, 0)]
		} else if to == NewSquare(2, 0) { // Queenside
			b.movePiece(NewSquare(0, 0), NewSquare(3, 0))
			b.HashKey ^= Zobrist.Pieces[WhiteRook][NewSquare(0, 0)]
			b.HashKey ^= Zobrist.Pieces[WhiteRook][NewSquare(3, 0)]
		}
	} else if piece == BlackKing && from == NewSquare(4, 7) {
		if to == NewSquare(6, 7) { // Kingside
			b.movePiece(NewSquare(7, 7), NewSquare(5, 7))
			b.HashKey ^= Zobrist.Pieces[BlackRook][NewSquare(7, 7)]
			b.HashKey ^= Zobrist.Pieces[BlackRook][NewSquare(5, 7)]
		} else if to == NewSquare(2, 7) { // Queenside
			b.movePiece(NewSquare(0, 7), NewSquare(3, 7))
			b.HashKey ^= Zobrist.Pieces[BlackRook][NewSquare(0, 7)]
			b.HashKey ^= Zobrist.Pieces[BlackRook][NewSquare(3, 7)]
		}
	}

	// Update hash for castling rights changes
	if oldCastling != b.Castling {
		// XOR out old castling rights
		if oldCastling&WhiteKingside != 0 {
			b.HashKey ^= Zobrist.Castling[0]
		}
		if oldCastling&WhiteQueenside != 0 {
			b.HashKey ^= Zobrist.Castling[1]
		}
		if oldCastling&BlackKingside != 0 {
			b.HashKey ^= Zobrist.Castling[2]
		}
		if oldCastling&BlackQueenside != 0 {
			b.HashKey ^= Zobrist.Castling[3]
		}
		// XOR in new castling rights
		if b.Castling&WhiteKingside != 0 {
			b.HashKey ^= Zobrist.Castling[0]
		}
		if b.Castling&WhiteQueenside != 0 {
			b.HashKey ^= Zobrist.Castling[1]
		}
		if b.Castling&BlackKingside != 0 {
			b.HashKey ^= Zobrist.Castling[2]
		}
		if b.Castling&BlackQueenside != 0 {
			b.HashKey ^= Zobrist.Castling[3]
		}
	}

	// Update hash for en passant changes
	if oldEnPassant != NoSquare {
		b.HashKey ^= Zobrist.EnPassant[oldEnPassant.File()]
	}
	if b.EnPassant != NoSquare {
		b.HashKey ^= Zobrist.EnPassant[b.EnPassant.File()]
	}

	// Update halfmove clock
	if piece == WhitePawn || piece == BlackPawn || captured != Empty {
		b.HalfmoveClock = 0
	} else {
		b.HalfmoveClock++
	}

	// Update fullmove number and side to move
	if b.SideToMove == Black {
		b.FullmoveNum++
	}
	b.SideToMove = 1 - b.SideToMove
	b.HashKey ^= Zobrist.SideToMove
}

// Print outputs the board in a human-readable format
func (b *Board) Print() string {
	var result string
	result += "  +---+---+---+---+---+---+---+---+\n"
	for rank := 7; rank >= 0; rank-- {
		result += string(rune('1'+rank)) + " |"
		for file := 0; file < 8; file++ {
			piece := b.Squares[NewSquare(file, rank)]
			result += " " + pieceToChar(piece) + " |"
		}
		result += "\n  +---+---+---+---+---+---+---+---+\n"
	}
	result += "    a   b   c   d   e   f   g   h\n"
	return result
}

// FENError represents an error parsing FEN notation
type FENError struct {
	msg string
}

func (e *FENError) Error() string {
	return "FEN error: " + e.msg
}

// Helper functions

func splitFEN(fen string) []string {
	var parts []string
	start := 0
	for i := 0; i <= len(fen); i++ {
		if i == len(fen) || fen[i] == ' ' {
			if i > start {
				parts = append(parts, fen[start:i])
			}
			start = i + 1
		}
	}
	return parts
}

func fenToPiece(c rune) Piece {
	switch c {
	case 'P':
		return WhitePawn
	case 'N':
		return WhiteKnight
	case 'B':
		return WhiteBishop
	case 'R':
		return WhiteRook
	case 'Q':
		return WhiteQueen
	case 'K':
		return WhiteKing
	case 'p':
		return BlackPawn
	case 'n':
		return BlackKnight
	case 'b':
		return BlackBishop
	case 'r':
		return BlackRook
	case 'q':
		return BlackQueen
	case 'k':
		return BlackKing
	default:
		return Empty
	}
}

func atoi(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			break
		}
		n = n*10 + int(c-'0')
	}
	return n
}

func pieceToChar(p Piece) string {
	switch p {
	case WhitePawn:
		return "P"
	case WhiteKnight:
		return "N"
	case WhiteBishop:
		return "B"
	case WhiteRook:
		return "R"
	case WhiteQueen:
		return "Q"
	case WhiteKing:
		return "K"
	case BlackPawn:
		return "p"
	case BlackKnight:
		return "n"
	case BlackBishop:
		return "b"
	case BlackRook:
		return "r"
	case BlackQueen:
		return "q"
	case BlackKing:
		return "k"
	default:
		return " "
	}
}

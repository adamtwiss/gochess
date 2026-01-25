package chess

// UndoInfo stores information needed to undo a move
type UndoInfo struct {
	Move          Move
	Captured      Piece
	Castling      CastlingRights
	EnPassant     Square
	HalfmoveClock int
	HashKey       uint64
}

// UndoStack stores undo information for move history
type UndoStack struct {
	stack []UndoInfo
}

// Global undo stack (or could be per-board)
var undoStack UndoStack

func init() {
	undoStack.stack = make([]UndoInfo, 0, 256)
}

// MakeMove makes a move on the board and stores undo information
func (b *Board) MakeMove(m Move) {
	from := m.From()
	to := m.To()
	flags := m.Flags()
	piece := b.Squares[from]
	captured := b.Squares[to]

	// Store undo info
	undo := UndoInfo{
		Move:          m,
		Captured:      captured,
		Castling:      b.Castling,
		EnPassant:     b.EnPassant,
		HalfmoveClock: b.HalfmoveClock,
		HashKey:       b.HashKey,
	}
	undoStack.stack = append(undoStack.stack, undo)

	// Handle en passant capture specially
	if flags == FlagEnPassant {
		captured = Empty // Actual capture is on a different square
		capturedSq := to
		if b.SideToMove == White {
			capturedSq = to - 8
		} else {
			capturedSq = to + 8
		}
		// Update hash for captured pawn
		capturedPawn := b.Squares[capturedSq]
		b.HashKey ^= Zobrist.Pieces[capturedPawn][capturedSq]
		b.removePiece(capturedSq)
	}

	// Remove captured piece (if any, and not en passant)
	if captured != Empty {
		b.removePiece(to)
		b.HashKey ^= Zobrist.Pieces[captured][to]
	}

	// Move the piece
	b.movePiece(from, to)
	b.HashKey ^= Zobrist.Pieces[piece][from]
	b.HashKey ^= Zobrist.Pieces[piece][to]

	// Handle promotion
	if flags&FlagPromotion != 0 {
		promoPiece := m.PromotionPiece()
		if b.SideToMove == Black {
			promoPiece += 6 // Convert to black piece
		}
		// Remove the pawn and place the promoted piece
		b.HashKey ^= Zobrist.Pieces[piece][to]    // Remove pawn from hash
		b.HashKey ^= Zobrist.Pieces[promoPiece][to] // Add promoted piece to hash
		b.removePiece(to)
		b.putPiece(promoPiece, to)
	}

	// Handle castling
	if flags == FlagCastle {
		if b.SideToMove == White {
			if to == NewSquare(6, 0) { // Kingside
				b.movePiece(NewSquare(7, 0), NewSquare(5, 0))
				b.HashKey ^= Zobrist.Pieces[WhiteRook][NewSquare(7, 0)]
				b.HashKey ^= Zobrist.Pieces[WhiteRook][NewSquare(5, 0)]
			} else { // Queenside
				b.movePiece(NewSquare(0, 0), NewSquare(3, 0))
				b.HashKey ^= Zobrist.Pieces[WhiteRook][NewSquare(0, 0)]
				b.HashKey ^= Zobrist.Pieces[WhiteRook][NewSquare(3, 0)]
			}
		} else {
			if to == NewSquare(6, 7) { // Kingside
				b.movePiece(NewSquare(7, 7), NewSquare(5, 7))
				b.HashKey ^= Zobrist.Pieces[BlackRook][NewSquare(7, 7)]
				b.HashKey ^= Zobrist.Pieces[BlackRook][NewSquare(5, 7)]
			} else { // Queenside
				b.movePiece(NewSquare(0, 7), NewSquare(3, 7))
				b.HashKey ^= Zobrist.Pieces[BlackRook][NewSquare(0, 7)]
				b.HashKey ^= Zobrist.Pieces[BlackRook][NewSquare(3, 7)]
			}
		}
	}

	// Update en passant square in hash
	if b.EnPassant != NoSquare {
		b.HashKey ^= Zobrist.EnPassant[b.EnPassant.File()]
	}

	// Set new en passant square
	b.EnPassant = NoSquare
	if piece == WhitePawn && to-from == 16 {
		b.EnPassant = from + 8
		b.HashKey ^= Zobrist.EnPassant[b.EnPassant.File()]
	} else if piece == BlackPawn && from-to == 16 {
		b.EnPassant = from - 8
		b.HashKey ^= Zobrist.EnPassant[b.EnPassant.File()]
	}

	// Update castling rights in hash (XOR out old rights)
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

	// Update castling rights
	switch from {
	case NewSquare(4, 0):
		b.Castling &^= WhiteKingside | WhiteQueenside
	case NewSquare(4, 7):
		b.Castling &^= BlackKingside | BlackQueenside
	case NewSquare(0, 0):
		b.Castling &^= WhiteQueenside
	case NewSquare(7, 0):
		b.Castling &^= WhiteKingside
	case NewSquare(0, 7):
		b.Castling &^= BlackQueenside
	case NewSquare(7, 7):
		b.Castling &^= BlackKingside
	}
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

	// Update castling rights in hash (XOR in new rights)
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

	// Update halfmove clock
	if piece == WhitePawn || piece == BlackPawn || captured != Empty || undo.Captured != Empty {
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

// UnmakeMove undoes the last move
func (b *Board) UnmakeMove(m Move) {
	// Pop undo info
	n := len(undoStack.stack)
	if n == 0 {
		return
	}
	undo := undoStack.stack[n-1]
	undoStack.stack = undoStack.stack[:n-1]

	from := m.From()
	to := m.To()
	flags := m.Flags()

	// Switch side to move back
	b.SideToMove = 1 - b.SideToMove

	// Restore state
	b.Castling = undo.Castling
	b.EnPassant = undo.EnPassant
	b.HalfmoveClock = undo.HalfmoveClock
	b.HashKey = undo.HashKey
	if b.SideToMove == White {
		b.FullmoveNum--
	}

	// Handle promotion - restore original pawn
	if flags&FlagPromotion != 0 {
		b.removePiece(to)
		if b.SideToMove == White {
			b.putPiece(WhitePawn, to)
		} else {
			b.putPiece(BlackPawn, to)
		}
	}

	// Move piece back
	b.movePiece(to, from)

	// Handle castling - move rook back
	if flags == FlagCastle {
		if b.SideToMove == White {
			if to == NewSquare(6, 0) { // Kingside
				b.movePiece(NewSquare(5, 0), NewSquare(7, 0))
			} else { // Queenside
				b.movePiece(NewSquare(3, 0), NewSquare(0, 0))
			}
		} else {
			if to == NewSquare(6, 7) { // Kingside
				b.movePiece(NewSquare(5, 7), NewSquare(7, 7))
			} else { // Queenside
				b.movePiece(NewSquare(3, 7), NewSquare(0, 7))
			}
		}
	}

	// Restore captured piece
	if undo.Captured != Empty {
		b.putPiece(undo.Captured, to)
	}

	// Handle en passant - restore captured pawn
	if flags == FlagEnPassant {
		capturedSq := to
		if b.SideToMove == White {
			capturedSq = to - 8
			b.putPiece(BlackPawn, capturedSq)
		} else {
			capturedSq = to + 8
			b.putPiece(WhitePawn, capturedSq)
		}
	}
}

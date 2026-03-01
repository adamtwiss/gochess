package chess

// UndoInfo stores information needed to undo a move.
// Fields ordered largest-first for optimal packing (24 bytes vs 32 with int).
type UndoInfo struct {
	HashKey       uint64
	PawnHashKey   uint64
	Move          Move
	HalfmoveClock int16
	Captured      Piece
	Castling      CastlingRights
	EnPassant     Square
}

// MakeMove makes a move on the board and stores undo information
func (b *Board) MakeMove(m Move) {
	from := m.From()
	to := m.To()
	flags := m.Flags()
	piece := b.Squares[from]
	captured := b.Squares[to]

	// Push NNUE accumulator before any piece modifications.
	// King moves use PushEmpty (no copy — RecomputeAccumulator overwrites everything).
	isKingMove := piece == WhiteKing || piece == BlackKing
	if b.NNUEAcc != nil {
		if isKingMove {
			b.NNUEAcc.PushEmpty()
		} else {
			b.NNUEAcc.Push()
		}
	}

	// Store undo info
	undo := UndoInfo{
		Move:          m,
		Captured:      captured,
		Castling:      b.Castling,
		EnPassant:     b.EnPassant,
		HalfmoveClock: b.HalfmoveClock,
		HashKey:       b.HashKey,
		PawnHashKey:   b.PawnHashKey,
	}
	b.UndoStack = append(b.UndoStack, undo)

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
		b.PawnHashKey ^= Zobrist.Pieces[capturedPawn][capturedSq]
		b.removePiece(capturedSq)
	}

	// Remove captured piece (if any, and not en passant)
	if captured != Empty {
		b.removePiece(to)
		b.HashKey ^= Zobrist.Pieces[captured][to]
		// Update pawn hash if a pawn was captured
		if captured == WhitePawn || captured == BlackPawn {
			b.PawnHashKey ^= Zobrist.Pieces[captured][to]
		}
	}

	// Move the piece
	b.movePiece(from, to)
	b.HashKey ^= Zobrist.Pieces[piece][from]
	b.HashKey ^= Zobrist.Pieces[piece][to]

	// Update pawn hash for pawn moves
	if piece == WhitePawn || piece == BlackPawn {
		b.PawnHashKey ^= Zobrist.Pieces[piece][from]
		b.PawnHashKey ^= Zobrist.Pieces[piece][to]
	}

	// Handle promotion
	if flags&FlagPromotion != 0 {
		promoPiece := m.PromotionPiece()
		if b.SideToMove == Black {
			promoPiece += 6 // Convert to black piece
		}
		// Remove the pawn and place the promoted piece
		b.HashKey ^= Zobrist.Pieces[piece][to]    // Remove pawn from hash
		b.HashKey ^= Zobrist.Pieces[promoPiece][to] // Add promoted piece to hash
		// Pawn disappears from pawn hash on promotion
		b.PawnHashKey ^= Zobrist.Pieces[piece][to]
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

	// Update castling rights
	oldCastling := b.Castling
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

	// Update castling rights in hash (only when changed)
	if oldCastling != b.Castling {
		changed := oldCastling ^ b.Castling
		if changed&WhiteKingside != 0 {
			b.HashKey ^= Zobrist.Castling[0]
		}
		if changed&WhiteQueenside != 0 {
			b.HashKey ^= Zobrist.Castling[1]
		}
		if changed&BlackKingside != 0 {
			b.HashKey ^= Zobrist.Castling[2]
		}
		if changed&BlackQueenside != 0 {
			b.HashKey ^= Zobrist.Castling[3]
		}
	}

	// Update halfmove clock (reset on pawn move or capture)
	if piece == WhitePawn || piece == BlackPawn || undo.Captured != Empty {
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

	// NNUE accumulator update — done directly here instead of through
	// putPiece/removePiece/movePiece hooks to avoid per-operation overhead.
	if b.NNUENet != nil && b.NNUEAcc != nil {
		if isKingMove {
			// King moves: full recompute (king square changed, all features shift)
			b.NNUENet.RecomputeAccumulator(b.NNUEAcc.Current(), b)
		} else {
			// Non-king moves: incremental update
			acc := b.NNUEAcc.Current()
			wKingSq := b.Pieces[WhiteKing].LSB()
			bKingSq := b.Pieces[BlackKing].LSB()
			net := b.NNUENet

			if flags == FlagEnPassant {
				// En passant: remove captured pawn + move our pawn
				capSq := to - 8
				capPiece := BlackPawn
				if b.SideToMove == White { // note: side already flipped
					capSq = to + 8
					capPiece = WhitePawn
				}
				net.SubSubAddFeature(acc, piece, from, to, capPiece, capSq, wKingSq, bKingSq)
			} else if flags&FlagPromotion != 0 {
				// Promotion: remove pawn from 'from', add promoted piece at 'to'
				// (captured piece at 'to' already handled)
				promoPiece := m.PromotionPiece()
				if b.SideToMove == White { // side already flipped
					promoPiece += 6
				}
				if undo.Captured != Empty {
					// Capture-promotion: remove captured + remove pawn + add promoted
					net.RemoveFeature(acc, undo.Captured, to, wKingSq, bKingSq)
				}
				net.RemoveFeature(acc, piece, from, wKingSq, bKingSq)
				net.AddFeature(acc, promoPiece, to, wKingSq, bKingSq)
			} else if undo.Captured != Empty {
				// Capture: remove captured piece + move our piece
				net.SubSubAddFeature(acc, piece, from, to, undo.Captured, to, wKingSq, bKingSq)
			} else {
				// Quiet move: just move our piece
				net.SubAddFeature(acc, piece, from, to, wKingSq, bKingSq)
			}
		}
	}
}

// MakeNullMove makes a null move (pass turn without moving)
func (b *Board) MakeNullMove() {
	// No NNUE push needed for null moves: no pieces change, so the accumulator
	// is still valid. Child moves Push/Pop their own copies, leaving this slot
	// untouched. UnmakeNullMove skips Pop to match.

	// Store undo info
	undo := UndoInfo{
		Move:      NoMove,
		EnPassant: b.EnPassant,
		HashKey:   b.HashKey,
	}
	b.UndoStack = append(b.UndoStack, undo)

	// Clear en passant
	if b.EnPassant != NoSquare {
		b.HashKey ^= Zobrist.EnPassant[b.EnPassant.File()]
		b.EnPassant = NoSquare
	}

	// Switch side to move
	b.SideToMove = 1 - b.SideToMove
	b.HashKey ^= Zobrist.SideToMove
}

// UnmakeNullMove undoes a null move
func (b *Board) UnmakeNullMove() {
	n := len(b.UndoStack)
	if n == 0 {
		return
	}
	undo := b.UndoStack[n-1]
	b.UndoStack = b.UndoStack[:n-1]

	b.SideToMove = 1 - b.SideToMove
	b.EnPassant = undo.EnPassant
	b.HashKey = undo.HashKey

	// No NNUE pop needed: MakeNullMove skips Push (no pieces changed).
}

// UnmakeMove undoes the last move
func (b *Board) UnmakeMove(m Move) {
	// Pop undo info
	n := len(b.UndoStack)
	if n == 0 {
		panic("UnmakeMove: empty undo stack!")
	}
	undo := b.UndoStack[n-1]
	b.UndoStack = b.UndoStack[:n-1]

	// Verify we're undoing the right move
	if undo.Move != m {
		panic("UnmakeMove: undo stack mismatch - expected " + undo.Move.String() + ", got " + m.String())
	}

	// Pop NNUE accumulator to restore pre-move state.
	// No NNUE hooks in putPiece/removePiece/movePiece, so piece restorations
	// below have zero NNUE overhead.
	if b.NNUEAcc != nil {
		b.NNUEAcc.Pop()
	}

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
	b.PawnHashKey = undo.PawnHashKey
	if b.SideToMove == Black {
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

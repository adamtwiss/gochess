package chess

// MoveGen is a stateful move generator
type MoveGen struct {
	board   *Board
	moves   []Move
	index   int
	phase   int
	genDone bool
}

// NewMoveGen creates a new move generator for the given board
func NewMoveGen(b *Board) *MoveGen {
	return &MoveGen{
		board: b,
		moves: make([]Move, 0, 64), // pre-allocate reasonable capacity
	}
}

// Reset resets the generator for a new position
func (mg *MoveGen) Reset(b *Board) {
	mg.board = b
	mg.moves = mg.moves[:0]
	mg.index = 0
	mg.phase = 0
	mg.genDone = false
}

// Next returns the next move, or NoMove if no more moves
// For now this is a simple implementation; we'll add staged generation later
func (mg *MoveGen) Next() Move {
	if !mg.genDone {
		mg.generateAllMoves()
		mg.genDone = true
	}

	if mg.index >= len(mg.moves) {
		return NoMove
	}

	m := mg.moves[mg.index]
	mg.index++
	return m
}

// GenerateAllMoves returns all pseudo-legal moves for the current position
func (b *Board) GenerateAllMoves() []Move {
	mg := NewMoveGen(b)
	mg.generateAllMoves()
	return mg.moves
}

// generateAllMoves generates all pseudo-legal moves
func (mg *MoveGen) generateAllMoves() {
	b := mg.board
	us := b.SideToMove
	them := 1 - us

	ourPieces := b.Occupied[us]
	theirPieces := b.Occupied[them]
	empty := ^b.AllPieces

	// Pawns
	if us == White {
		mg.generateWhitePawnMoves(empty, theirPieces)
	} else {
		mg.generateBlackPawnMoves(empty, theirPieces)
	}

	// Knights
	knights := b.Pieces[pieceOf(WhiteKnight, us)]
	for knights != 0 {
		from := knights.PopLSB()
		attacks := KnightAttacks[from] & ^ourPieces
		mg.addMoves(from, attacks)
	}

	// Bishops
	bishops := b.Pieces[pieceOf(WhiteBishop, us)]
	for bishops != 0 {
		from := bishops.PopLSB()
		attacks := BishopAttacksBB(from, b.AllPieces) & ^ourPieces
		mg.addMoves(from, attacks)
	}

	// Rooks
	rooks := b.Pieces[pieceOf(WhiteRook, us)]
	for rooks != 0 {
		from := rooks.PopLSB()
		attacks := RookAttacksBB(from, b.AllPieces) & ^ourPieces
		mg.addMoves(from, attacks)
	}

	// Queens
	queens := b.Pieces[pieceOf(WhiteQueen, us)]
	for queens != 0 {
		from := queens.PopLSB()
		attacks := QueenAttacksBB(from, b.AllPieces) & ^ourPieces
		mg.addMoves(from, attacks)
	}

	// King
	king := b.Pieces[pieceOf(WhiteKing, us)]
	if king != 0 {
		from := king.LSB()
		attacks := KingAttacks[from] & ^ourPieces
		mg.addMoves(from, attacks)

		// Castling
		mg.generateCastlingMoves(from)
	}
}

func (mg *MoveGen) generateWhitePawnMoves(empty, theirPieces Bitboard) {
	b := mg.board
	pawns := b.Pieces[WhitePawn]

	// Single push
	push1 := (pawns << 8) & empty
	// Double push (only from rank 2)
	push2 := ((push1 & Rank3) << 8) & empty

	// Promotions (push to rank 8)
	promo := push1 & Rank8
	push1 &^= Rank8

	// Add single pushes
	for push1 != 0 {
		to := push1.PopLSB()
		mg.moves = append(mg.moves, NewMove(to-8, to))
	}

	// Add double pushes
	for push2 != 0 {
		to := push2.PopLSB()
		mg.moves = append(mg.moves, NewMove(to-16, to))
	}

	// Add promotions
	for promo != 0 {
		to := promo.PopLSB()
		from := to - 8
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteQ))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteR))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteB))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteN))
	}

	// Captures
	captureL := ((pawns & NotFileA) << 7) & theirPieces
	captureR := ((pawns & NotFileH) << 9) & theirPieces

	// Capture promotions
	promoL := captureL & Rank8
	promoR := captureR & Rank8
	captureL &^= Rank8
	captureR &^= Rank8

	for captureL != 0 {
		to := captureL.PopLSB()
		mg.moves = append(mg.moves, NewMove(to-7, to))
	}
	for captureR != 0 {
		to := captureR.PopLSB()
		mg.moves = append(mg.moves, NewMove(to-9, to))
	}

	// Capture promotions
	for promoL != 0 {
		to := promoL.PopLSB()
		from := to - 7
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteQ))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteR))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteB))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteN))
	}
	for promoR != 0 {
		to := promoR.PopLSB()
		from := to - 9
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteQ))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteR))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteB))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteN))
	}

	// En passant
	if b.EnPassant != NoSquare {
		epBB := SquareBB(b.EnPassant)
		epL := ((pawns & NotFileA) << 7) & epBB
		epR := ((pawns & NotFileH) << 9) & epBB

		if epL != 0 {
			to := b.EnPassant
			mg.moves = append(mg.moves, NewMoveFlags(to-7, to, FlagEnPassant))
		}
		if epR != 0 {
			to := b.EnPassant
			mg.moves = append(mg.moves, NewMoveFlags(to-9, to, FlagEnPassant))
		}
	}
}

func (mg *MoveGen) generateBlackPawnMoves(empty, theirPieces Bitboard) {
	b := mg.board
	pawns := b.Pieces[BlackPawn]

	// Single push
	push1 := (pawns >> 8) & empty
	// Double push (only from rank 7)
	push2 := ((push1 & Rank6) >> 8) & empty

	// Promotions (push to rank 1)
	promo := push1 & Rank1
	push1 &^= Rank1

	// Add single pushes
	for push1 != 0 {
		to := push1.PopLSB()
		mg.moves = append(mg.moves, NewMove(to+8, to))
	}

	// Add double pushes
	for push2 != 0 {
		to := push2.PopLSB()
		mg.moves = append(mg.moves, NewMove(to+16, to))
	}

	// Add promotions
	for promo != 0 {
		to := promo.PopLSB()
		from := to + 8
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteQ))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteR))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteB))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteN))
	}

	// Captures
	captureL := ((pawns & NotFileA) >> 9) & theirPieces
	captureR := ((pawns & NotFileH) >> 7) & theirPieces

	// Capture promotions
	promoL := captureL & Rank1
	promoR := captureR & Rank1
	captureL &^= Rank1
	captureR &^= Rank1

	for captureL != 0 {
		to := captureL.PopLSB()
		mg.moves = append(mg.moves, NewMove(to+9, to))
	}
	for captureR != 0 {
		to := captureR.PopLSB()
		mg.moves = append(mg.moves, NewMove(to+7, to))
	}

	// Capture promotions
	for promoL != 0 {
		to := promoL.PopLSB()
		from := to + 9
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteQ))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteR))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteB))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteN))
	}
	for promoR != 0 {
		to := promoR.PopLSB()
		from := to + 7
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteQ))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteR))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteB))
		mg.moves = append(mg.moves, NewMoveFlags(from, to, FlagPromoteN))
	}

	// En passant
	if b.EnPassant != NoSquare {
		epBB := SquareBB(b.EnPassant)
		epL := ((pawns & NotFileA) >> 9) & epBB
		epR := ((pawns & NotFileH) >> 7) & epBB

		if epL != 0 {
			to := b.EnPassant
			mg.moves = append(mg.moves, NewMoveFlags(to+9, to, FlagEnPassant))
		}
		if epR != 0 {
			to := b.EnPassant
			mg.moves = append(mg.moves, NewMoveFlags(to+7, to, FlagEnPassant))
		}
	}
}

func (mg *MoveGen) generateCastlingMoves(kingSq Square) {
	b := mg.board
	us := b.SideToMove

	if us == White {
		// Kingside: e1-g1, need f1 and g1 empty, e1/f1/g1 not attacked
		if b.Castling&WhiteKingside != 0 {
			if b.Squares[NewSquare(5, 0)] == Empty && b.Squares[NewSquare(6, 0)] == Empty {
				if !b.IsAttacked(NewSquare(4, 0), Black) &&
					!b.IsAttacked(NewSquare(5, 0), Black) &&
					!b.IsAttacked(NewSquare(6, 0), Black) {
					mg.moves = append(mg.moves, NewMoveFlags(kingSq, NewSquare(6, 0), FlagCastle))
				}
			}
		}
		// Queenside: e1-c1, need b1/c1/d1 empty, e1/d1/c1 not attacked
		if b.Castling&WhiteQueenside != 0 {
			if b.Squares[NewSquare(1, 0)] == Empty &&
				b.Squares[NewSquare(2, 0)] == Empty &&
				b.Squares[NewSquare(3, 0)] == Empty {
				if !b.IsAttacked(NewSquare(4, 0), Black) &&
					!b.IsAttacked(NewSquare(3, 0), Black) &&
					!b.IsAttacked(NewSquare(2, 0), Black) {
					mg.moves = append(mg.moves, NewMoveFlags(kingSq, NewSquare(2, 0), FlagCastle))
				}
			}
		}
	} else {
		// Black kingside
		if b.Castling&BlackKingside != 0 {
			if b.Squares[NewSquare(5, 7)] == Empty && b.Squares[NewSquare(6, 7)] == Empty {
				if !b.IsAttacked(NewSquare(4, 7), White) &&
					!b.IsAttacked(NewSquare(5, 7), White) &&
					!b.IsAttacked(NewSquare(6, 7), White) {
					mg.moves = append(mg.moves, NewMoveFlags(kingSq, NewSquare(6, 7), FlagCastle))
				}
			}
		}
		// Black queenside
		if b.Castling&BlackQueenside != 0 {
			if b.Squares[NewSquare(1, 7)] == Empty &&
				b.Squares[NewSquare(2, 7)] == Empty &&
				b.Squares[NewSquare(3, 7)] == Empty {
				if !b.IsAttacked(NewSquare(4, 7), White) &&
					!b.IsAttacked(NewSquare(3, 7), White) &&
					!b.IsAttacked(NewSquare(2, 7), White) {
					mg.moves = append(mg.moves, NewMoveFlags(kingSq, NewSquare(2, 7), FlagCastle))
				}
			}
		}
	}
}

func (mg *MoveGen) addMoves(from Square, targets Bitboard) {
	for targets != 0 {
		to := targets.PopLSB()
		mg.moves = append(mg.moves, NewMove(from, to))
	}
}

// IsAttacked returns true if the square is attacked by the given color
func (b *Board) IsAttacked(sq Square, by Color) bool {
	// Pawn attacks
	if PawnAttacks[1-by][sq]&b.Pieces[pieceOf(WhitePawn, by)] != 0 {
		return true
	}

	// Knight attacks
	if KnightAttacks[sq]&b.Pieces[pieceOf(WhiteKnight, by)] != 0 {
		return true
	}

	// King attacks
	if KingAttacks[sq]&b.Pieces[pieceOf(WhiteKing, by)] != 0 {
		return true
	}

	// Bishop/Queen attacks (diagonals)
	bishopsQueens := b.Pieces[pieceOf(WhiteBishop, by)] | b.Pieces[pieceOf(WhiteQueen, by)]
	if BishopAttacksBB(sq, b.AllPieces)&bishopsQueens != 0 {
		return true
	}

	// Rook/Queen attacks (ranks/files)
	rooksQueens := b.Pieces[pieceOf(WhiteRook, by)] | b.Pieces[pieceOf(WhiteQueen, by)]
	if RookAttacksBB(sq, b.AllPieces)&rooksQueens != 0 {
		return true
	}

	return false
}

// isAttackedWithOcc checks if a square is attacked by a color using a custom
// occupancy bitboard for sliding piece lookups. Used for king-move legality
// (king removed from occupancy) and evasion generation.
func (b *Board) isAttackedWithOcc(sq Square, by Color, occ Bitboard) bool {
	if PawnAttacks[1-by][sq]&b.Pieces[pieceOf(WhitePawn, by)] != 0 {
		return true
	}
	if KnightAttacks[sq]&b.Pieces[pieceOf(WhiteKnight, by)] != 0 {
		return true
	}
	if KingAttacks[sq]&b.Pieces[pieceOf(WhiteKing, by)] != 0 {
		return true
	}
	bq := b.Pieces[pieceOf(WhiteBishop, by)] | b.Pieces[pieceOf(WhiteQueen, by)]
	if BishopAttacksBB(sq, occ)&bq != 0 {
		return true
	}
	rq := b.Pieces[pieceOf(WhiteRook, by)] | b.Pieces[pieceOf(WhiteQueen, by)]
	if RookAttacksBB(sq, occ)&rq != 0 {
		return true
	}
	return false
}

// InCheck returns true if the side to move is in check
func (b *Board) InCheck() bool {
	us := b.SideToMove
	kingSq := b.Pieces[pieceOf(WhiteKing, us)].LSB()
	return b.IsAttacked(kingSq, 1-us)
}

// PinnedPieces returns a bitboard of friendly pieces pinned to the king.
func (b *Board) PinnedPieces(us Color) Bitboard {
	pinned, _ := b.PinnedAndCheckers(us)
	return pinned
}

// PinnedAndCheckers returns both pinned pieces and the set of enemy pieces
// giving check. Computing both together shares the sliding-piece ray work
// and avoids a separate InCheck() call.
func (b *Board) PinnedAndCheckers(us Color) (pinned, checkers Bitboard) {
	them := 1 - us
	kingSq := b.Pieces[pieceOf(WhiteKing, us)].LSB()
	ourPieces := b.Occupied[us]

	// Non-sliding checkers
	checkers = PawnAttacks[us][kingSq] & b.Pieces[pieceOf(WhitePawn, them)]
	checkers |= KnightAttacks[kingSq] & b.Pieces[pieceOf(WhiteKnight, them)]

	// Sliding pieces: snipers are enemy sliders that could see the king on an empty board
	enemyRQ := b.Pieces[pieceOf(WhiteRook, them)] | b.Pieces[pieceOf(WhiteQueen, them)]
	enemyBQ := b.Pieces[pieceOf(WhiteBishop, them)] | b.Pieces[pieceOf(WhiteQueen, them)]
	snipers := (RookAttacksBB(kingSq, 0) & enemyRQ) | (BishopAttacksBB(kingSq, 0) & enemyBQ)

	for snipers != 0 {
		sniperSq := snipers.PopLSB()
		between := BetweenBB[kingSq][sniperSq] & b.AllPieces
		if between == 0 {
			// No pieces between → checker
			checkers |= SquareBB(sniperSq)
		} else if between&(between-1) == 0 && between&ourPieces != 0 {
			// Exactly one of our pieces between → pinned
			pinned |= between
		}
	}

	return
}

// IsLegal returns true if the move is legal (doesn't leave king in check).
// pinned should be precomputed via PinnedPieces(). inCheck indicates whether
// the side to move is currently in check (avoids recomputing).
func (b *Board) IsLegal(m Move, pinned Bitboard, inCheck bool) bool {
	us := b.SideToMove
	from := m.From()
	to := m.To()
	flags := m.Flags()

	// Castling: fully validated during generation (king doesn't pass through attack).
	if flags == FlagCastle {
		return true
	}

	kingSq := b.Pieces[pieceOf(WhiteKing, us)].LSB()
	them := 1 - us

	// En passant: always use full make/unmake (two pieces removed from same rank
	// can create discovered checks in unusual ways).
	if flags == FlagEnPassant {
		b.MakeMove(m)
		check := b.IsAttacked(kingSq, them)
		b.UnmakeMove(m)
		return !check
	}

	// King moves: check destination with king removed from occupancy
	// (so sliding pieces aren't blocked by the king itself).
	if from == kingSq {
		occ := b.AllPieces &^ SquareBB(from)
		return !b.isAttackedWithOcc(to, them, occ)
	}

	// Non-king moves when in check: must block or capture the checker.
	// Fall back to full make/unmake since this is rare (~5% of nodes).
	if inCheck {
		b.MakeMove(m)
		check := b.IsAttacked(kingSq, them)
		b.UnmakeMove(m)
		return !check
	}

	// Not in check, non-king move: if piece is not pinned, always legal.
	if pinned&SquareBB(from) == 0 {
		return true
	}

	// Pinned piece: legal only if it moves along the pin line.
	return LineBB[kingSq][from]&SquareBB(to) != 0
}

// IsLegalSlow is a convenience wrapper that checks both pseudo-legality and
// legality. Use in non-performance-critical paths (PV extraction).
func (b *Board) IsLegalSlow(m Move) bool {
	if !b.IsPseudoLegal(m) {
		return false
	}
	pinned, checkers := b.PinnedAndCheckers(b.SideToMove)
	return b.IsLegal(m, pinned, checkers != 0)
}

// GenerateLegalMoves returns all legal moves
func (b *Board) GenerateLegalMoves() []Move {
	pseudoLegal := b.GenerateAllMoves()
	legal := make([]Move, 0, len(pseudoLegal))
	pinned, checkers := b.PinnedAndCheckers(b.SideToMove)
	inCheck := checkers != 0

	for _, m := range pseudoLegal {
		if b.IsLegal(m, pinned, inCheck) {
			legal = append(legal, m)
		}
	}

	return legal
}

// GenerateCaptures returns all pseudo-legal capture moves (including promotions)
func (b *Board) GenerateCaptures() []Move {
	return b.GenerateCapturesAppend(make([]Move, 0, 32))
}

// GenerateCapturesAppend appends pseudo-legal captures to the provided slice
func (b *Board) GenerateCapturesAppend(moves []Move) []Move {
	us := b.SideToMove
	them := 1 - us
	theirPieces := b.Occupied[them]
	ourPieces := b.Occupied[us]

	// Pawn captures and promotions
	if us == White {
		pawns := b.Pieces[WhitePawn]

		// Captures
		captureL := ((pawns & NotFileA) << 7) & theirPieces
		captureR := ((pawns & NotFileH) << 9) & theirPieces

		for captureL != 0 {
			to := captureL.PopLSB()
			from := to - 7
			if to.Rank() == 7 {
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
			} else {
				moves = append(moves, NewMove(from, to))
			}
		}
		for captureR != 0 {
			to := captureR.PopLSB()
			from := to - 9
			if to.Rank() == 7 {
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
			} else {
				moves = append(moves, NewMove(from, to))
			}
		}

		// Non-capture promotions (still tactical)
		empty := ^b.AllPieces
		promo := ((pawns << 8) & empty) & Rank8
		for promo != 0 {
			to := promo.PopLSB()
			from := to - 8
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
		}

		// En passant
		if b.EnPassant != NoSquare {
			epBB := SquareBB(b.EnPassant)
			epL := ((pawns & NotFileA) << 7) & epBB
			epR := ((pawns & NotFileH) << 9) & epBB
			if epL != 0 {
				moves = append(moves, NewMoveFlags(b.EnPassant-7, b.EnPassant, FlagEnPassant))
			}
			if epR != 0 {
				moves = append(moves, NewMoveFlags(b.EnPassant-9, b.EnPassant, FlagEnPassant))
			}
		}
	} else {
		pawns := b.Pieces[BlackPawn]

		// Captures
		captureL := ((pawns & NotFileA) >> 9) & theirPieces
		captureR := ((pawns & NotFileH) >> 7) & theirPieces

		for captureL != 0 {
			to := captureL.PopLSB()
			from := to + 9
			if to.Rank() == 0 {
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
			} else {
				moves = append(moves, NewMove(from, to))
			}
		}
		for captureR != 0 {
			to := captureR.PopLSB()
			from := to + 7
			if to.Rank() == 0 {
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
			} else {
				moves = append(moves, NewMove(from, to))
			}
		}

		// Non-capture promotions (still tactical)
		empty := ^b.AllPieces
		promo := ((pawns >> 8) & empty) & Rank1
		for promo != 0 {
			to := promo.PopLSB()
			from := to + 8
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
		}

		// En passant
		if b.EnPassant != NoSquare {
			epBB := SquareBB(b.EnPassant)
			epL := ((pawns & NotFileA) >> 9) & epBB
			epR := ((pawns & NotFileH) >> 7) & epBB
			if epL != 0 {
				moves = append(moves, NewMoveFlags(b.EnPassant+9, b.EnPassant, FlagEnPassant))
			}
			if epR != 0 {
				moves = append(moves, NewMoveFlags(b.EnPassant+7, b.EnPassant, FlagEnPassant))
			}
		}
	}

	// Knight captures
	knights := b.Pieces[pieceOf(WhiteKnight, us)]
	for knights != 0 {
		from := knights.PopLSB()
		attacks := KnightAttacks[from] & theirPieces
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// Bishop captures
	bishops := b.Pieces[pieceOf(WhiteBishop, us)]
	for bishops != 0 {
		from := bishops.PopLSB()
		attacks := BishopAttacksBB(from, b.AllPieces) & theirPieces
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// Rook captures
	rooks := b.Pieces[pieceOf(WhiteRook, us)]
	for rooks != 0 {
		from := rooks.PopLSB()
		attacks := RookAttacksBB(from, b.AllPieces) & theirPieces
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// Queen captures
	queens := b.Pieces[pieceOf(WhiteQueen, us)]
	for queens != 0 {
		from := queens.PopLSB()
		attacks := QueenAttacksBB(from, b.AllPieces) & theirPieces
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// King captures
	king := b.Pieces[pieceOf(WhiteKing, us)]
	if king != 0 {
		from := king.LSB()
		attacks := KingAttacks[from] & theirPieces & ^ourPieces
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	return moves
}

// GenerateQuiets returns all pseudo-legal quiet moves (non-captures, non-promotions)
func (b *Board) GenerateQuiets() []Move {
	return b.GenerateQuietsAppend(make([]Move, 0, 32))
}

// GenerateQuietsAppend appends pseudo-legal quiet moves to the provided slice
func (b *Board) GenerateQuietsAppend(moves []Move) []Move {
	us := b.SideToMove
	empty := ^b.AllPieces

	// Pawn quiet moves (non-promotions)
	if us == White {
		pawns := b.Pieces[WhitePawn]

		// Single push (not to promotion rank)
		push1 := ((pawns << 8) & empty) &^ Rank8
		for push1 != 0 {
			to := push1.PopLSB()
			moves = append(moves, NewMove(to-8, to))
		}

		// Double push
		push1ForDouble := (pawns << 8) & empty
		push2 := ((push1ForDouble & Rank3) << 8) & empty
		for push2 != 0 {
			to := push2.PopLSB()
			moves = append(moves, NewMove(to-16, to))
		}
	} else {
		pawns := b.Pieces[BlackPawn]

		// Single push (not to promotion rank)
		push1 := ((pawns >> 8) & empty) &^ Rank1
		for push1 != 0 {
			to := push1.PopLSB()
			moves = append(moves, NewMove(to+8, to))
		}

		// Double push
		push1ForDouble := (pawns >> 8) & empty
		push2 := ((push1ForDouble & Rank6) >> 8) & empty
		for push2 != 0 {
			to := push2.PopLSB()
			moves = append(moves, NewMove(to+16, to))
		}
	}

	// Knight quiets
	knights := b.Pieces[pieceOf(WhiteKnight, us)]
	for knights != 0 {
		from := knights.PopLSB()
		attacks := KnightAttacks[from] & empty
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// Bishop quiets
	bishops := b.Pieces[pieceOf(WhiteBishop, us)]
	for bishops != 0 {
		from := bishops.PopLSB()
		attacks := BishopAttacksBB(from, b.AllPieces) & empty
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// Rook quiets
	rooks := b.Pieces[pieceOf(WhiteRook, us)]
	for rooks != 0 {
		from := rooks.PopLSB()
		attacks := RookAttacksBB(from, b.AllPieces) & empty
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// Queen quiets
	queens := b.Pieces[pieceOf(WhiteQueen, us)]
	for queens != 0 {
		from := queens.PopLSB()
		attacks := QueenAttacksBB(from, b.AllPieces) & empty
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// King quiets and castling
	king := b.Pieces[pieceOf(WhiteKing, us)]
	if king != 0 {
		from := king.LSB()
		attacks := KingAttacks[from] & empty
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}

		// Castling
		mg := &MoveGen{board: b, moves: moves}
		mg.generateCastlingMoves(from)
		moves = mg.moves
	}

	return moves
}

// GenerateEvasionsAppend generates all legal evasion moves when in check.
// The returned moves are fully legal — no IsLegal filtering needed.
// checkers and pinned should be precomputed via PinnedAndCheckers().
func (b *Board) GenerateEvasionsAppend(moves []Move, checkers, pinned Bitboard) []Move {
	us := b.SideToMove
	them := 1 - us
	kingSq := b.Pieces[pieceOf(WhiteKing, us)].LSB()
	ourPieces := b.Occupied[us]

	// King evasions: always generated (both single and double check)
	occ := b.AllPieces ^ SquareBB(kingSq) // remove king so sliders see through
	targets := KingAttacks[kingSq] &^ ourPieces
	for targets != 0 {
		to := targets.PopLSB()
		if !b.isAttackedWithOcc(to, them, occ) {
			moves = append(moves, NewMove(kingSq, to))
		}
	}

	// Double check: only king moves are legal
	if checkers&(checkers-1) != 0 {
		return moves
	}

	// Single check: can also block or capture the checker
	checkerSq := checkers.LSB()
	// target = capture the checker OR block the ray between king and checker
	// BetweenBB is empty for non-sliding checkers (knight, pawn)
	target := SquareBB(checkerSq) | BetweenBB[kingSq][checkerSq]
	blockTarget := BetweenBB[kingSq][checkerSq] // blocking squares only (no capture)

	// Only non-pinned pieces can resolve check (pinned pieces can never
	// interpose or capture a checker that's on a different line than the pin)
	nonPinned := ourPieces &^ pinned &^ SquareBB(kingSq)

	// Knights
	knights := b.Pieces[pieceOf(WhiteKnight, us)] & nonPinned
	for knights != 0 {
		from := knights.PopLSB()
		attacks := KnightAttacks[from] & target
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// Bishops
	bishops := b.Pieces[pieceOf(WhiteBishop, us)] & nonPinned
	for bishops != 0 {
		from := bishops.PopLSB()
		attacks := BishopAttacksBB(from, b.AllPieces) & target
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// Rooks
	rooks := b.Pieces[pieceOf(WhiteRook, us)] & nonPinned
	for rooks != 0 {
		from := rooks.PopLSB()
		attacks := RookAttacksBB(from, b.AllPieces) & target
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// Queens
	queens := b.Pieces[pieceOf(WhiteQueen, us)] & nonPinned
	for queens != 0 {
		from := queens.PopLSB()
		attacks := QueenAttacksBB(from, b.AllPieces) & target
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// Pawns
	pawns := b.Pieces[pieceOf(WhitePawn, us)] & nonPinned
	checkerBB := SquareBB(checkerSq)
	empty := ^b.AllPieces

	if us == White {
		promoRank := Rank8

		// Pawn captures onto checker square
		captureL := ((pawns & NotFileA) << 7) & checkerBB
		captureR := ((pawns & NotFileH) << 9) & checkerBB

		for captureL != 0 {
			to := captureL.PopLSB()
			from := to - 7
			if SquareBB(to)&promoRank != 0 {
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
			} else {
				moves = append(moves, NewMove(from, to))
			}
		}
		for captureR != 0 {
			to := captureR.PopLSB()
			from := to - 9
			if SquareBB(to)&promoRank != 0 {
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
			} else {
				moves = append(moves, NewMove(from, to))
			}
		}

		// Pawn pushes onto blocking squares (or capture square for push-promotions)
		push1 := ((pawns << 8) & empty) & target
		push2 := ((((pawns << 8) & empty & Rank3) << 8) & empty) & blockTarget

		for push1 != 0 {
			to := push1.PopLSB()
			from := to - 8
			if SquareBB(to)&promoRank != 0 {
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
			} else {
				moves = append(moves, NewMove(from, to))
			}
		}
		for push2 != 0 {
			to := push2.PopLSB()
			moves = append(moves, NewMove(to-16, to))
		}

		// En passant: only if the captured pawn IS the checking piece
		if b.EnPassant != NoSquare {
			// The captured pawn is one rank below the EP square for white
			capturedPawnSq := b.EnPassant - 8
			if Square(capturedPawnSq) == checkerSq {
				epBB := SquareBB(b.EnPassant)
				epL := ((pawns & NotFileA) << 7) & epBB
				epR := ((pawns & NotFileH) << 9) & epBB
				if epL != 0 {
					m := NewMoveFlags(b.EnPassant-7, b.EnPassant, FlagEnPassant)
					// EP always validated via make/unmake (rare discovered check edge cases)
					b.MakeMove(m)
					if !b.IsAttacked(kingSq, them) {
						moves = append(moves, m)
					}
					b.UnmakeMove(m)
				}
				if epR != 0 {
					m := NewMoveFlags(b.EnPassant-9, b.EnPassant, FlagEnPassant)
					b.MakeMove(m)
					if !b.IsAttacked(kingSq, them) {
						moves = append(moves, m)
					}
					b.UnmakeMove(m)
				}
			}
		}
	} else {
		// Black pawns
		promoRank := Rank1

		captureL := ((pawns & NotFileA) >> 9) & checkerBB
		captureR := ((pawns & NotFileH) >> 7) & checkerBB

		for captureL != 0 {
			to := captureL.PopLSB()
			from := to + 9
			if SquareBB(to)&promoRank != 0 {
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
			} else {
				moves = append(moves, NewMove(from, to))
			}
		}
		for captureR != 0 {
			to := captureR.PopLSB()
			from := to + 7
			if SquareBB(to)&promoRank != 0 {
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
			} else {
				moves = append(moves, NewMove(from, to))
			}
		}

		push1 := ((pawns >> 8) & empty) & target
		push2 := ((((pawns >> 8) & empty & Rank6) >> 8) & empty) & blockTarget

		for push1 != 0 {
			to := push1.PopLSB()
			from := to + 8
			if SquareBB(to)&promoRank != 0 {
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
				moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
			} else {
				moves = append(moves, NewMove(from, to))
			}
		}
		for push2 != 0 {
			to := push2.PopLSB()
			moves = append(moves, NewMove(to+16, to))
		}

		// En passant
		if b.EnPassant != NoSquare {
			capturedPawnSq := b.EnPassant + 8
			if Square(capturedPawnSq) == checkerSq {
				epBB := SquareBB(b.EnPassant)
				epL := ((pawns & NotFileA) >> 9) & epBB
				epR := ((pawns & NotFileH) >> 7) & epBB
				if epL != 0 {
					m := NewMoveFlags(b.EnPassant+9, b.EnPassant, FlagEnPassant)
					b.MakeMove(m)
					if !b.IsAttacked(kingSq, them) {
						moves = append(moves, m)
					}
					b.UnmakeMove(m)
				}
				if epR != 0 {
					m := NewMoveFlags(b.EnPassant+7, b.EnPassant, FlagEnPassant)
					b.MakeMove(m)
					if !b.IsAttacked(kingSq, them) {
						moves = append(moves, m)
					}
					b.UnmakeMove(m)
				}
			}
		}
	}

	return moves
}

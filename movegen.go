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

// InCheck returns true if the side to move is in check
func (b *Board) InCheck() bool {
	us := b.SideToMove
	kingSq := b.Pieces[pieceOf(WhiteKing, us)].LSB()
	return b.IsAttacked(kingSq, 1-us)
}

// IsLegal returns true if the move is legal (doesn't leave king in check)
func (b *Board) IsLegal(m Move) bool {
	us := b.SideToMove

	// Make the move temporarily
	b.MakeMove(m)

	// Check if our king is in check
	kingSq := b.Pieces[pieceOf(WhiteKing, us)].LSB()
	inCheck := b.IsAttacked(kingSq, 1-us)

	// Unmake the move
	b.UnmakeMove(m)

	return !inCheck
}

// GenerateLegalMoves returns all legal moves
func (b *Board) GenerateLegalMoves() []Move {
	pseudoLegal := b.GenerateAllMoves()
	legal := make([]Move, 0, len(pseudoLegal))

	for _, m := range pseudoLegal {
		if b.IsLegal(m) {
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

package chess

// MovePicker provides staged move generation for search.
// Moves are generated and returned in order of likely quality:
// 1. TT/PV move
// 2. Good captures (SEE >= 0) ordered by MVV-LVA
// 3. Killer moves
// 4. Quiet moves ordered by history
// 5. Bad captures (SEE < 0)
type MovePicker struct {
	board       *Board
	ttMove      Move
	killers     [2]Move
	counterMove Move
	history     *[64][64]int
	stage       int
	moves       []Move
	scores      []int
	index       int
	ply         int
	skipQuiet   bool // For quiescence search
}

// MovePicker stages
const (
	stageTTMove = iota
	stageGenerateCaptures
	stageGoodCaptures
	stageKiller1
	stageKiller2
	stageCounterMove
	stageGenerateQuiets
	stageQuiets
	stageBadCaptures
	stageDone
)

// NewMovePicker creates a new move picker for the main search
func NewMovePicker(b *Board, ttMove Move, ply int, killers [2]Move, history *[64][64]int, counterMove Move) *MovePicker {
	return &MovePicker{
		board:       b,
		ttMove:      ttMove,
		killers:     killers,
		counterMove: counterMove,
		history:     history,
		ply:         ply,
		stage:       stageTTMove,
		moves:       make([]Move, 0, 64),
		scores:      make([]int, 0, 64),
	}
}

// NewMovePickerQuiescence creates a move picker for quiescence search (captures only)
func NewMovePickerQuiescence(b *Board) *MovePicker {
	return &MovePicker{
		board:     b,
		stage:     stageGenerateCaptures,
		moves:     make([]Move, 0, 32),
		scores:    make([]int, 0, 32),
		skipQuiet: true,
	}
}

// Next returns the next move to try, or NoMove if no more moves
func (mp *MovePicker) Next() Move {
	for {
		switch mp.stage {
		case stageTTMove:
			mp.stage = stageGenerateCaptures
			if mp.ttMove != NoMove && mp.board.IsPseudoLegal(mp.ttMove) {
				return mp.ttMove
			}

		case stageGenerateCaptures:
			mp.generateAndScoreCaptures()
			mp.stage = stageGoodCaptures
			mp.index = 0

		case stageGoodCaptures:
			for mp.index < len(mp.moves) {
				move := mp.pickBest()
				if move == NoMove {
					break
				}
				if move == mp.ttMove {
					continue // Already tried
				}
				// Only return good captures here (SEE >= 0)
				// Score is at mp.index-1 since pickBest incremented it
				scoreIdx := mp.index - 1
				if scoreIdx >= 0 && mp.scores[scoreIdx] >= 0 {
					return move
				}
				// Put this back for bad captures stage
				mp.index--
				break
			}
			if mp.skipQuiet {
				mp.stage = stageBadCaptures
			} else {
				mp.stage = stageKiller1
			}

		case stageKiller1:
			mp.stage = stageKiller2
			if mp.killers[0] != NoMove && mp.killers[0] != mp.ttMove {
				if mp.board.IsPseudoLegal(mp.killers[0]) && !mp.isCapture(mp.killers[0]) {
					return mp.killers[0]
				}
			}

		case stageKiller2:
			mp.stage = stageCounterMove
			if mp.killers[1] != NoMove && mp.killers[1] != mp.ttMove && mp.killers[1] != mp.killers[0] {
				if mp.board.IsPseudoLegal(mp.killers[1]) && !mp.isCapture(mp.killers[1]) {
					return mp.killers[1]
				}
			}

		case stageCounterMove:
			mp.stage = stageGenerateQuiets
			if mp.counterMove != NoMove &&
				mp.counterMove != mp.ttMove &&
				mp.counterMove != mp.killers[0] &&
				mp.counterMove != mp.killers[1] {
				if mp.board.IsPseudoLegal(mp.counterMove) && !mp.isCapture(mp.counterMove) {
					return mp.counterMove
				}
			}

		case stageGenerateQuiets:
			mp.generateAndScoreQuiets()
			mp.stage = stageQuiets

		case stageQuiets:
			for mp.index < len(mp.moves) {
				move := mp.pickBest()
				if move == mp.ttMove || move == mp.killers[0] || move == mp.killers[1] || move == mp.counterMove {
					continue // Already tried
				}
				return move
			}
			mp.stage = stageBadCaptures
			// Restore capture list for bad captures
			mp.restoreBadCaptures()

		case stageBadCaptures:
			for mp.index < len(mp.moves) {
				move := mp.pickBest()
				if move == mp.ttMove {
					continue
				}
				return move
			}
			mp.stage = stageDone

		case stageDone:
			return NoMove
		}
	}
}

// generateAndScoreCaptures generates all captures and scores them
func (mp *MovePicker) generateAndScoreCaptures() {
	captures := mp.board.GenerateCaptures()
	mp.moves = mp.moves[:0]
	mp.scores = mp.scores[:0]

	for _, m := range captures {
		mp.moves = append(mp.moves, m)
		// Score by MVV-LVA for initial ordering, SEE for good/bad classification
		score := mp.mvvLva(m)
		// Adjust by SEE: good captures get positive boost, bad get negative
		see := mp.board.SEE(m)
		if see < 0 {
			score = see - 10000 // Bad captures go to end
		}
		mp.scores = append(mp.scores, score)
	}

	// Store count of captures for bad capture restoration
	mp.index = 0
}

// generateAndScoreQuiets generates quiet moves and scores by history
func (mp *MovePicker) generateAndScoreQuiets() {
	quiets := mp.board.GenerateQuiets()

	// Reset for quiet moves
	mp.moves = mp.moves[:0]
	mp.scores = mp.scores[:0]

	for _, m := range quiets {
		mp.moves = append(mp.moves, m)
		// Score by history heuristic
		score := 0
		if mp.history != nil {
			score = mp.history[m.From()][m.To()]
		}
		mp.scores = append(mp.scores, score)
	}
	mp.index = 0
}

// restoreBadCaptures reloads bad captures for final stage
func (mp *MovePicker) restoreBadCaptures() {
	captures := mp.board.GenerateCaptures()
	mp.moves = mp.moves[:0]
	mp.scores = mp.scores[:0]

	for _, m := range captures {
		see := mp.board.SEE(m)
		if see < 0 {
			mp.moves = append(mp.moves, m)
			mp.scores = append(mp.scores, see)
		}
	}
	mp.index = 0
}

// pickBest finds the best move from current index onwards and returns it
// Uses partial selection sort - only finds the maximum, doesn't fully sort
func (mp *MovePicker) pickBest() Move {
	if mp.index >= len(mp.moves) {
		return NoMove
	}

	bestIdx := mp.index
	bestScore := mp.scores[bestIdx]

	for i := mp.index + 1; i < len(mp.moves); i++ {
		if mp.scores[i] > bestScore {
			bestScore = mp.scores[i]
			bestIdx = i
		}
	}

	// Swap best to current position
	if bestIdx != mp.index {
		mp.moves[mp.index], mp.moves[bestIdx] = mp.moves[bestIdx], mp.moves[mp.index]
		mp.scores[mp.index], mp.scores[bestIdx] = mp.scores[bestIdx], mp.scores[mp.index]
	}

	move := mp.moves[mp.index]
	mp.index++
	return move
}

// mvvLva returns MVV-LVA score for a capture
func (mp *MovePicker) mvvLva(m Move) int {
	to := m.To()
	from := m.From()

	victim := mp.board.Squares[to]
	if victim == Empty {
		// En passant
		if m.Flags() == FlagEnPassant {
			return 100*10 - 100 // Pawn x Pawn
		}
		return 0
	}

	attacker := mp.board.Squares[from]

	// Normalize to piece type value (1-6 range)
	victimValue := SEEPieceValues[victim]
	attackerValue := SEEPieceValues[attacker]

	// MVV-LVA: maximize victim value, minimize attacker value
	return victimValue*10 - attackerValue
}

// isCapture returns true if the move is a capture
func (mp *MovePicker) isCapture(m Move) bool {
	return mp.board.Squares[m.To()] != Empty || m.Flags() == FlagEnPassant
}

// IsPseudoLegal checks if a move is pseudo-legal (valid piece movement, may leave king in check)
func (b *Board) IsPseudoLegal(m Move) bool {
	if m == NoMove {
		return false
	}

	from := m.From()
	to := m.To()
	piece := b.Squares[from]

	// Must have a piece on the from square
	if piece == Empty {
		return false
	}

	// Must be our piece
	us := b.SideToMove
	pieceColor := White
	if piece >= BlackPawn {
		pieceColor = Black
	}
	if pieceColor != us {
		return false
	}

	// Can't capture our own piece
	target := b.Squares[to]
	if target != Empty {
		targetColor := White
		if target >= BlackPawn {
			targetColor = Black
		}
		if targetColor == us {
			return false
		}
	}

	// Check move is valid for this piece type
	flags := m.Flags()

	// Handle special moves
	if flags == FlagCastle {
		return b.isCastleLegal(m)
	}

	if flags == FlagEnPassant {
		return to == b.EnPassant && (piece == WhitePawn || piece == BlackPawn)
	}

	// Promotion moves are only valid for pawns
	if flags&FlagPromotion != 0 {
		if piece != WhitePawn && piece != BlackPawn {
			return false
		}
	}

	// Check the piece can actually make this move
	pieceType := piece
	if piece >= BlackPawn {
		pieceType -= 6
	}

	switch pieceType {
	case WhitePawn:
		return b.isPawnMoveLegal(from, to, us, flags)
	case WhiteKnight:
		return KnightAttacks[from]&SquareBB(to) != 0
	case WhiteBishop:
		return BishopAttacksBB(from, b.AllPieces)&SquareBB(to) != 0
	case WhiteRook:
		return RookAttacksBB(from, b.AllPieces)&SquareBB(to) != 0
	case WhiteQueen:
		return QueenAttacksBB(from, b.AllPieces)&SquareBB(to) != 0
	case WhiteKing:
		return KingAttacks[from]&SquareBB(to) != 0
	}

	return false
}

// isPawnMoveLegal checks if a pawn move is valid
func (b *Board) isPawnMoveLegal(from, to Square, us Color, flags int) bool {
	fromRank := from.Rank()
	toRank := to.Rank()
	fromFile := from.File()
	toFile := to.File()

	target := b.Squares[to]

	if us == White {
		// Must be moving forward
		if toRank <= fromRank {
			return false
		}

		// Capture (diagonal)
		if fromFile != toFile {
			if abs(fromFile-toFile) != 1 || toRank != fromRank+1 {
				return false
			}
			return target != Empty // Must capture something
		}

		// Push
		if target != Empty {
			return false // Can't push onto a piece
		}

		if toRank == fromRank+1 {
			return true // Single push
		}

		if toRank == fromRank+2 && fromRank == 1 {
			// Double push - check intermediate square is empty
			return b.Squares[NewSquare(fromFile, 2)] == Empty
		}

		return false
	} else {
		// Black - must be moving backward (in rank terms)
		if toRank >= fromRank {
			return false
		}

		// Capture
		if fromFile != toFile {
			if abs(fromFile-toFile) != 1 || toRank != fromRank-1 {
				return false
			}
			return target != Empty
		}

		// Push
		if target != Empty {
			return false
		}

		if toRank == fromRank-1 {
			return true
		}

		if toRank == fromRank-2 && fromRank == 6 {
			return b.Squares[NewSquare(fromFile, 5)] == Empty
		}

		return false
	}
}

// isCastleLegal checks if a castling move is legal
func (b *Board) isCastleLegal(m Move) bool {
	from := m.From()
	to := m.To()
	us := b.SideToMove

	if us == White {
		if from != NewSquare(4, 0) {
			return false
		}
		if to == NewSquare(6, 0) {
			// Kingside
			if b.Castling&WhiteKingside == 0 {
				return false
			}
			if b.Squares[NewSquare(5, 0)] != Empty || b.Squares[NewSquare(6, 0)] != Empty {
				return false
			}
			if b.IsAttacked(NewSquare(4, 0), Black) || b.IsAttacked(NewSquare(5, 0), Black) || b.IsAttacked(NewSquare(6, 0), Black) {
				return false
			}
			return true
		}
		if to == NewSquare(2, 0) {
			// Queenside
			if b.Castling&WhiteQueenside == 0 {
				return false
			}
			if b.Squares[NewSquare(1, 0)] != Empty || b.Squares[NewSquare(2, 0)] != Empty || b.Squares[NewSquare(3, 0)] != Empty {
				return false
			}
			if b.IsAttacked(NewSquare(4, 0), Black) || b.IsAttacked(NewSquare(3, 0), Black) || b.IsAttacked(NewSquare(2, 0), Black) {
				return false
			}
			return true
		}
	} else {
		if from != NewSquare(4, 7) {
			return false
		}
		if to == NewSquare(6, 7) {
			if b.Castling&BlackKingside == 0 {
				return false
			}
			if b.Squares[NewSquare(5, 7)] != Empty || b.Squares[NewSquare(6, 7)] != Empty {
				return false
			}
			if b.IsAttacked(NewSquare(4, 7), White) || b.IsAttacked(NewSquare(5, 7), White) || b.IsAttacked(NewSquare(6, 7), White) {
				return false
			}
			return true
		}
		if to == NewSquare(2, 7) {
			if b.Castling&BlackQueenside == 0 {
				return false
			}
			if b.Squares[NewSquare(1, 7)] != Empty || b.Squares[NewSquare(2, 7)] != Empty || b.Squares[NewSquare(3, 7)] != Empty {
				return false
			}
			if b.IsAttacked(NewSquare(4, 7), White) || b.IsAttacked(NewSquare(3, 7), White) || b.IsAttacked(NewSquare(2, 7), White) {
				return false
			}
			return true
		}
	}

	return false
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

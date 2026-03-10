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
	history     *[64][64]int32
	contHist    *[13][64]int16 // continuation history sub-table for prev move's piece+square
	captHist    *[13][64][7]int16
	stage       int
	moves       []Move
	scores      []int
	badMoves    []Move // saved bad captures from first pass
	badScores   []int
	index       int
	ply         int
	skipQuiet   bool // For quiescence search
	checkers    Bitboard
	pinned      Bitboard
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

	// Evasion stages (used when in check)
	stageEvasionTTMove
	stageGenerateEvasions
	stageEvasions
)

// NewMovePicker creates a new move picker for the main search
func NewMovePicker(b *Board, ttMove Move, ply int, killers [2]Move, history *[64][64]int32, counterMove Move, contHist *[13][64]int16, captHist *[13][64][7]int16) *MovePicker {
	mp := &MovePicker{}
	mp.Init(b, ttMove, ply, killers, history, counterMove, contHist, captHist)
	return mp
}

// NewMovePickerQuiescence creates a move picker for quiescence search (captures only)
func NewMovePickerQuiescence(b *Board, captHist *[13][64][7]int16) *MovePicker {
	mp := &MovePicker{}
	mp.InitQuiescence(b, captHist)
	return mp
}

// Init resets a MovePicker for reuse, avoiding heap allocations on subsequent calls
func (mp *MovePicker) Init(b *Board, ttMove Move, ply int, killers [2]Move, history *[64][64]int32, counterMove Move, contHist *[13][64]int16, captHist *[13][64][7]int16) {
	mp.board = b
	mp.ttMove = ttMove
	mp.killers = killers
	mp.counterMove = counterMove
	mp.history = history
	mp.contHist = contHist
	mp.captHist = captHist
	mp.ply = ply
	mp.stage = stageTTMove
	mp.index = 0
	mp.skipQuiet = false
	if mp.moves == nil {
		mp.moves = make([]Move, 0, 64)
		mp.scores = make([]int, 0, 64)
		mp.badMoves = make([]Move, 0, 16)
		mp.badScores = make([]int, 0, 16)
	}
}

// InitQuiescence resets a MovePicker for quiescence search reuse
func (mp *MovePicker) InitQuiescence(b *Board, captHist *[13][64][7]int16) {
	mp.board = b
	mp.ttMove = NoMove
	mp.captHist = captHist
	mp.stage = stageGenerateCaptures
	mp.index = 0
	mp.skipQuiet = true
	if mp.moves == nil {
		mp.moves = make([]Move, 0, 32)
		mp.scores = make([]int, 0, 32)
	}
}

// InitEvasion resets a MovePicker for evasion mode (when in check).
// Evasion moves are fully legal — no IsLegal filtering needed by the caller.
func (mp *MovePicker) InitEvasion(b *Board, ttMove Move, ply int, checkers, pinned Bitboard,
	history *[64][64]int32, contHist *[13][64]int16, captHist *[13][64][7]int16) {
	mp.board = b
	mp.ttMove = ttMove
	mp.checkers = checkers
	mp.pinned = pinned
	mp.history = history
	mp.contHist = contHist
	mp.captHist = captHist
	mp.ply = ply
	mp.stage = stageEvasionTTMove
	mp.index = 0
	mp.skipQuiet = false
	if mp.moves == nil {
		mp.moves = make([]Move, 0, 32)
		mp.scores = make([]int, 0, 32)
		mp.badMoves = make([]Move, 0, 16)
		mp.badScores = make([]int, 0, 16)
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
			// TT move already filtered during scoring
			for mp.index < len(mp.moves) {
				return mp.pickBest()
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
			// TT/killers/counter already filtered during scoring
			for mp.index < len(mp.moves) {
				return mp.pickBest()
			}
			mp.stage = stageBadCaptures
			// Restore capture list for bad captures
			mp.restoreBadCaptures()

		case stageBadCaptures:
			// TT move already filtered during capture scoring
			for mp.index < len(mp.moves) {
				return mp.pickBest()
			}
			mp.stage = stageDone

		case stageDone:
			return NoMove

		// Evasion stages
		case stageEvasionTTMove:
			mp.stage = stageGenerateEvasions
			if mp.ttMove != NoMove && mp.board.IsPseudoLegal(mp.ttMove) {
				if mp.board.IsLegal(mp.ttMove, mp.pinned, mp.checkers != 0) {
					return mp.ttMove
				}
			}

		case stageGenerateEvasions:
			mp.generateAndScoreEvasions()
			mp.stage = stageEvasions

		case stageEvasions:
			// TT move already filtered during evasion scoring
			for mp.index < len(mp.moves) {
				return mp.pickBest()
			}
			mp.stage = stageDone
		}
	}
}

// generateAndScoreCaptures generates all captures, partitions into good and bad.
// The TT move is filtered out so pickBest never returns it (avoids redundant scan).
func (mp *MovePicker) generateAndScoreCaptures() {
	mp.moves = mp.board.GenerateCapturesAppend(mp.moves[:0])
	mp.scores = mp.scores[:0]
	mp.badMoves = mp.badMoves[:0]
	mp.badScores = mp.badScores[:0]

	// Partition: good captures (SEE >= 0) stay in mp.moves, bad go to mp.badMoves.
	// Uses SEESign for fast early exits on obvious cases (e.g. PxN, equal trades).
	// TT move is excluded from both (already tried in stageTTMove).
	j := 0
	for i := 0; i < len(mp.moves); i++ {
		m := mp.moves[i]
		if m == mp.ttMove {
			continue
		}
		if !mp.board.SEESign(m, 0) {
			mp.badMoves = append(mp.badMoves, m)
			mp.badScores = append(mp.badScores, mp.captHistScore(m))
		} else {
			mp.moves[j] = m
			mp.scores = append(mp.scores, mp.mvvLva(m)+mp.captHistScore(m))
			j++
		}
	}
	mp.moves = mp.moves[:j]
	mp.index = 0
}

// generateAndScoreQuiets generates quiet moves and scores by history.
// TT move, killers, and counter-move are filtered out to avoid redundant pickBest scans.
func (mp *MovePicker) generateAndScoreQuiets() {
	mp.moves = mp.board.GenerateQuietsAppend(mp.moves[:0])
	mp.scores = mp.scores[:0]

	j := 0
	for i := 0; i < len(mp.moves); i++ {
		m := mp.moves[i]
		if m == mp.ttMove || m == mp.killers[0] || m == mp.killers[1] || m == mp.counterMove {
			continue
		}
		mp.moves[j] = m
		score := 0
		if mp.history != nil {
			score = int(mp.history[m.From()][m.To()])
		}
		if mp.contHist != nil {
			piece := mp.board.Squares[m.From()]
			score += 2 * int(mp.contHist[piece][m.To()])
		}
		mp.scores = append(mp.scores, score)
		j++
	}
	mp.moves = mp.moves[:j]
	mp.index = 0
}

// generateAndScoreEvasions generates evasion moves and scores them.
// Captures scored above quiets for natural ordering.
// TT move is filtered out (already tried in stageEvasionTTMove).
func (mp *MovePicker) generateAndScoreEvasions() {
	mp.moves = mp.board.GenerateEvasionsAppend(mp.moves[:0], mp.checkers, mp.pinned)
	mp.scores = mp.scores[:0]

	j := 0
	for i := 0; i < len(mp.moves); i++ {
		m := mp.moves[i]
		if m == mp.ttMove {
			continue
		}
		mp.moves[j] = m
		score := 0

		if m.IsPromotion() {
			if m.Flags() == FlagPromoteQ {
				score = 9000
			} else {
				score = -1000 // underpromotions
			}
		} else if mp.board.Squares[m.To()] != Empty || m.Flags() == FlagEnPassant {
			// Capture: MVV-LVA + capture history
			score = 10000 + mp.mvvLva(m) + mp.captHistScore(m)
		} else {
			// Quiet: history + continuation history
			if mp.history != nil {
				score = int(mp.history[m.From()][m.To()])
			}
			if mp.contHist != nil {
				piece := mp.board.Squares[m.From()]
				score += 2 * int(mp.contHist[piece][m.To()])
			}
		}

		mp.scores = append(mp.scores, score)
		j++
	}
	mp.moves = mp.moves[:j]
	mp.index = 0
}

// restoreBadCaptures swaps in the saved bad captures from the first pass
func (mp *MovePicker) restoreBadCaptures() {
	mp.moves = append(mp.moves[:0], mp.badMoves...)
	mp.scores = append(mp.scores[:0], mp.badScores...)
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

// captHistScore returns the capture history score for a capture move
func (mp *MovePicker) captHistScore(m Move) int {
	if mp.captHist == nil {
		return 0
	}
	piece := mp.board.Squares[m.From()]
	victim := mp.board.Squares[m.To()]
	ct := capturedType(victim)
	if ct == 0 && m.Flags() == FlagEnPassant {
		ct = 1
	}
	return int(mp.captHist[piece][m.To()][ct])
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
	isPromo := flags&FlagPromotion != 0

	target := b.Squares[to]

	if us == White {
		// Promotion flag must match reaching rank 8 (index 7)
		if (toRank == 7) != isPromo {
			return false
		}

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
		// Promotion flag must match reaching rank 1 (index 0)
		if (toRank == 0) != isPromo {
			return false
		}

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

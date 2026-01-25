package chess

import (
	"time"
)

const (
	Infinity  = 30000
	MateScore = 29000
)

// SearchInfo holds information about the search
type SearchInfo struct {
	Nodes     uint64
	Depth     int
	Score     int
	PV        []Move
	StartTime time.Time
	MaxTime   time.Duration
	Stopped   bool
	TT        *TranspositionTable
}

// Search performs iterative deepening search and returns the best move
func (b *Board) Search(maxDepth int, maxTime time.Duration) (Move, SearchInfo) {
	return b.SearchWithTT(maxDepth, maxTime, nil)
}

// SearchWithTT performs search with an optional transposition table
func (b *Board) SearchWithTT(maxDepth int, maxTime time.Duration, tt *TranspositionTable) (Move, SearchInfo) {
	info := SearchInfo{
		StartTime: time.Now(),
		MaxTime:   maxTime,
		TT:        tt,
	}

	// Create a default TT if none provided
	if info.TT == nil {
		info.TT = NewTranspositionTable(16) // 16 MB default
	}

	var bestMove Move
	var pv []Move

	// Iterative deepening
	for depth := 1; depth <= maxDepth; depth++ {
		info.Depth = depth

		score := b.negamax(depth, 0, -Infinity, Infinity, &info, &pv)

		// Check if we ran out of time mid-search
		if info.Stopped {
			break
		}

		// Save results from this iteration
		info.Score = score
		if len(pv) > 0 {
			bestMove = pv[0]
			info.PV = make([]Move, len(pv))
			copy(info.PV, pv)
		}

		// Check time after completing a depth
		if maxTime > 0 && time.Since(info.StartTime) > maxTime/2 {
			// If we've used more than half our time, unlikely to finish next depth
			break
		}
	}

	return bestMove, info
}

// negamax performs alpha-beta search from the current position
// ply is the distance from root (for mate score adjustment)
func (b *Board) negamax(depth, ply int, alpha, beta int, info *SearchInfo, pv *[]Move) int {
	// Check time periodically
	if info.Nodes&4095 == 0 && info.MaxTime > 0 {
		if time.Since(info.StartTime) >= info.MaxTime {
			info.Stopped = true
			return 0
		}
	}

	if info.Stopped {
		return 0
	}

	info.Nodes++

	// Probe transposition table
	ttMove := NoMove
	alphaOrig := alpha

	if entry, found := info.TT.Probe(b.HashKey); found {
		ttMove = entry.Move

		if int(entry.Depth) >= depth {
			score := int(entry.Score)
			// Adjust mate scores for distance from root
			if score > MateScore-100 {
				score -= ply
			} else if score < -MateScore+100 {
				score += ply
			}

			switch entry.Flag {
			case TTExact:
				*pv = []Move{ttMove}
				return score
			case TTLower:
				if score > alpha {
					alpha = score
				}
			case TTUpper:
				if score < beta {
					beta = score
				}
			}

			if alpha >= beta {
				*pv = []Move{ttMove}
				return score
			}
		}
	}

	// Leaf node - go to quiescence search
	if depth <= 0 {
		return b.quiescence(alpha, beta, info)
	}

	moves := b.GenerateLegalMoves()

	// Check for checkmate or stalemate
	if len(moves) == 0 {
		if b.InCheck() {
			// Checkmate - return negative mate score adjusted for ply
			return -MateScore + ply
		}
		// Stalemate
		return 0
	}

	// Order moves for better pruning
	pvMove := ttMove // Prefer TT move
	if pvMove == NoMove && len(*pv) > 0 {
		pvMove = (*pv)[0]
	}
	orderMoves(moves, pvMove, b)

	// Clear PV for this node
	localPV := make([]Move, 0, depth)
	*pv = (*pv)[:0]

	bestMove := moves[0]
	bestScore := -Infinity

	for _, move := range moves {
		b.MakeMove(move)

		childPV := make([]Move, 0, depth-1)
		score := -b.negamax(depth-1, ply+1, -beta, -alpha, info, &childPV)

		b.UnmakeMove(move)

		if info.Stopped {
			return 0
		}

		if score > bestScore {
			bestScore = score
			bestMove = move

			if score > alpha {
				alpha = score

				// Update PV
				localPV = localPV[:0]
				localPV = append(localPV, move)
				localPV = append(localPV, childPV...)
				*pv = localPV

				if alpha >= beta {
					// Beta cutoff
					break
				}
			}
		}
	}

	// Store in transposition table
	var flag TTFlag
	if bestScore <= alphaOrig {
		flag = TTUpper
	} else if bestScore >= beta {
		flag = TTLower
	} else {
		flag = TTExact
	}

	// Adjust mate score for storage (relative to this position)
	storeScore := bestScore
	if storeScore > MateScore-100 {
		storeScore += ply
	} else if storeScore < -MateScore+100 {
		storeScore -= ply
	}

	info.TT.Store(b.HashKey, depth, storeScore, flag, bestMove)

	return bestScore
}

// quiescence searches captures until the position is quiet
func (b *Board) quiescence(alpha, beta int, info *SearchInfo) int {
	info.Nodes++

	// Stand pat - evaluate the current position
	standPat := b.EvaluateRelative()

	if standPat >= beta {
		return beta
	}

	if standPat > alpha {
		alpha = standPat
	}

	// Generate and search captures only
	moves := b.GenerateLegalMoves()
	captures := filterCaptures(moves, b)

	// Order captures by MVV-LVA
	orderCaptures(captures, b)

	for _, move := range captures {
		b.MakeMove(move)
		score := -b.quiescence(-beta, -alpha, info)
		b.UnmakeMove(move)

		if score >= beta {
			return beta
		}
		if score > alpha {
			alpha = score
		}
	}

	return alpha
}

// filterCaptures returns only capture moves
func filterCaptures(moves []Move, b *Board) []Move {
	captures := make([]Move, 0, len(moves)/4)
	for _, m := range moves {
		to := m.To()
		if b.Squares[to] != Empty || m.Flags() == FlagEnPassant {
			captures = append(captures, m)
		}
	}
	return captures
}

// orderMoves sorts moves for better alpha-beta pruning
// PV move first, then captures by MVV-LVA, then other moves
func orderMoves(moves []Move, pvMove Move, b *Board) {
	// Simple insertion sort with scoring
	scores := make([]int, len(moves))

	for i, m := range moves {
		score := 0

		// PV move gets highest priority
		if m == pvMove {
			score = 100000
		} else {
			to := m.To()
			captured := b.Squares[to]

			if captured != Empty {
				// MVV-LVA: prioritize capturing valuable pieces with less valuable pieces
				score = 10000 + mvvLva(b.Squares[m.From()], captured)
			} else if m.Flags() == FlagEnPassant {
				score = 10000 + mvvLva(WhitePawn, BlackPawn)
			} else if m.Flags()&FlagPromotion != 0 {
				// Promotions are usually good
				score = 9000
			}
		}

		scores[i] = score
	}

	// Sort by score descending (simple insertion sort, fine for ~30-40 moves)
	for i := 1; i < len(moves); i++ {
		j := i
		for j > 0 && scores[j] > scores[j-1] {
			moves[j], moves[j-1] = moves[j-1], moves[j]
			scores[j], scores[j-1] = scores[j-1], scores[j]
			j--
		}
	}
}

// orderCaptures sorts captures by MVV-LVA
func orderCaptures(moves []Move, b *Board) {
	scores := make([]int, len(moves))

	for i, m := range moves {
		to := m.To()
		captured := b.Squares[to]
		if captured != Empty {
			scores[i] = mvvLva(b.Squares[m.From()], captured)
		} else if m.Flags() == FlagEnPassant {
			scores[i] = mvvLva(WhitePawn, BlackPawn)
		}
	}

	for i := 1; i < len(moves); i++ {
		j := i
		for j > 0 && scores[j] > scores[j-1] {
			moves[j], moves[j-1] = moves[j-1], moves[j]
			scores[j], scores[j-1] = scores[j-1], scores[j]
			j--
		}
	}
}

// mvvLva returns a score for Most Valuable Victim - Least Valuable Attacker
func mvvLva(attacker, victim Piece) int {
	// Normalize to white pieces for comparison
	attackerType := attacker
	if attacker >= BlackPawn {
		attackerType -= 6
	}
	victimType := victim
	if victim >= BlackPawn {
		victimType -= 6
	}

	// Victim value * 10 - attacker value
	// This prioritizes capturing queens with pawns over capturing pawns with queens
	victimValue := pieceValue(victimType)
	attackerValue := pieceValue(attackerType)

	return victimValue*10 - attackerValue
}

// pieceValue returns the value of a piece type (white pieces only)
func pieceValue(p Piece) int {
	switch p {
	case WhitePawn:
		return 1
	case WhiteKnight:
		return 3
	case WhiteBishop:
		return 3
	case WhiteRook:
		return 5
	case WhiteQueen:
		return 9
	case WhiteKing:
		return 100
	}
	return 0
}

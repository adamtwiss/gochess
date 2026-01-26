package chess

import (
	"time"
)

const (
	Infinity  = 30000
	MateScore = 29000
)

// LMREnabled controls whether Late Move Reductions are used
// Set to false for benchmarking comparisons
var LMREnabled = true

// LMR reduction table - indexed by [depth][moveNumber]
// Precomputed for efficiency
var lmrTable [64][64]int

func init() {
	// Initialize LMR table using logarithmic formula
	// Conservative approach - cap at reasonable reductions
	for depth := 1; depth < 64; depth++ {
		for moveNum := 1; moveNum < 64; moveNum++ {
			if depth >= 3 && moveNum >= 3 {
				// Base reduction of 1 for late moves
				reduction := 1

				// Increase reduction for very late moves at higher depths
				if depth >= 6 && moveNum >= 10 {
					reduction = 2
				}

				// Cap reduction to leave at least depth 1
				if reduction > depth-2 {
					reduction = depth - 2
				}
				if reduction < 0 {
					reduction = 0
				}
				lmrTable[depth][moveNum] = reduction
			}
		}
	}
}

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

	// Killer moves: 2 slots per ply, max 64 ply
	Killers [64][2]Move

	// History table: indexed by [from][to], stores cutoff counts
	History [64][64]int

	// LMR statistics (for debugging/analysis)
	LMRAttempts   uint64 // Times LMR was attempted
	LMRReSearches uint64 // Times we had to re-search at full depth
	LMRSavings    uint64 // Successful LMR prunings (no re-search needed)

	// OnDepth is called after each completed iteration of iterative deepening.
	// Parameters: depth, score, cumulative nodes, PV for this depth.
	OnDepth func(depth, score int, nodes uint64, pv []Move)
}

// storeKiller stores a killer move at the given ply
func (info *SearchInfo) storeKiller(ply int, move Move) {
	if ply >= 64 {
		return
	}
	if move != info.Killers[ply][0] {
		info.Killers[ply][1] = info.Killers[ply][0]
		info.Killers[ply][0] = move
	}
}

// Search performs iterative deepening search and returns the best move
func (b *Board) Search(maxDepth int, maxTime time.Duration) (Move, SearchInfo) {
	return b.SearchWithTT(maxDepth, maxTime, nil)
}

// SearchWithTT performs search with an optional transposition table
func (b *Board) SearchWithTT(maxDepth int, maxTime time.Duration, tt *TranspositionTable) (Move, SearchInfo) {
	info := &SearchInfo{
		StartTime: time.Now(),
		MaxTime:   maxTime,
		TT:        tt,
	}
	return b.SearchWithInfo(maxDepth, info)
}

// SearchWithInfo performs search using a caller-provided SearchInfo.
// The caller may pre-configure fields like TT and OnDepth.
// StartTime and MaxTime should be set before calling.
func (b *Board) SearchWithInfo(maxDepth int, info *SearchInfo) (Move, SearchInfo) {
	// Create a default TT if none provided
	if info.TT == nil {
		info.TT = NewTranspositionTable(16) // 16 MB default
	}

	// Clear history table for new search
	for i := range info.History {
		for j := range info.History[i] {
			info.History[i][j] = 0
		}
	}

	// Clear killer moves
	for i := range info.Killers {
		info.Killers[i][0] = NoMove
		info.Killers[i][1] = NoMove
	}

	var bestMove Move
	var pv []Move
	prevScore := 0

	// Iterative deepening with aspiration windows
	for depth := 1; depth <= maxDepth; depth++ {
		info.Depth = depth

		var score int

		if depth >= 4 && prevScore > -MateScore+100 && prevScore < MateScore-100 {
			// Aspiration window search
			delta := 25
			alpha, beta := prevScore-delta, prevScore+delta
			for {
				pv = pv[:0]
				score = b.negamax(depth, 0, alpha, beta, info, &pv)
				if info.Stopped {
					break
				}
				if score <= alpha {
					delta = widenDelta(delta)
					alpha = prevScore - delta
					continue
				}
				if score >= beta {
					delta = widenDelta(delta)
					beta = prevScore + delta
					continue
				}
				break // score within window
			}
		} else {
			pv = pv[:0]
			score = b.negamax(depth, 0, -Infinity, Infinity, info, &pv)
		}

		// Check if we ran out of time mid-search
		if info.Stopped {
			break
		}

		// Save results from this iteration
		prevScore = score
		info.Score = score
		if len(pv) > 0 {
			bestMove = pv[0]
			info.PV = make([]Move, len(pv))
			copy(info.PV, pv)
		}

		if info.OnDepth != nil {
			info.OnDepth(depth, score, info.Nodes, info.PV)
		}

		// Check time after completing a depth
		if info.MaxTime > 0 && time.Since(info.StartTime) > info.MaxTime/2 {
			// If we've used more than half our time, unlikely to finish next depth
			break
		}
	}

	return bestMove, *info
}

// widenDelta returns the next wider aspiration window delta
func widenDelta(delta int) int {
	switch {
	case delta <= 25:
		return 100
	case delta <= 100:
		return 500
	default:
		return Infinity
	}
}

// negamax performs alpha-beta search from the current position
// ply is the distance from root (for mate score adjustment)
func (b *Board) negamax(depth, ply int, alpha, beta int, info *SearchInfo, pv *[]Move) int {
	// Guard against stack overflow (Go has limited goroutine stack)
	if ply >= 64 {
		return b.EvaluateRelative()
	}

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

	inCheck := b.InCheck()

	// Null-move pruning
	// Skip if: in check, at root, or depth too shallow
	if depth >= 3 && !inCheck && ply > 0 {
		R := 2 // Reduction factor
		if depth > 6 {
			R = 3
		}

		b.MakeNullMove()
		var nullPV []Move
		score := -b.negamax(depth-1-R, ply+1, -beta, -beta+1, info, &nullPV)
		b.UnmakeNullMove()

		if info.Stopped {
			return 0
		}

		if score >= beta {
			return beta // Null-move cutoff
		}
	}

	// Reverse Futility Pruning (Static Null Move Pruning)
	// At shallow depths, if static eval is far above beta, prune
	if depth <= 3 && !inCheck && ply > 0 {
		staticEval := b.EvaluateRelative()
		margin := depth * 120
		if staticEval-margin >= beta {
			return staticEval - margin
		}
	}

	// Get killers for this ply
	var killers [2]Move
	if ply < 64 {
		killers = info.Killers[ply]
	}

	// Prefer TT move, fall back to PV move
	pvMove := ttMove
	if pvMove == NoMove && len(*pv) > 0 {
		pvMove = (*pv)[0]
	}

	// Use MovePicker for staged move generation
	picker := NewMovePicker(b, pvMove, ply, killers, &info.History)

	// Clear PV for this node
	localPV := make([]Move, 0, depth)
	*pv = (*pv)[:0]

	bestMove := NoMove
	bestScore := -Infinity
	moveCount := 0

	for {
		move := picker.Next()
		if move == NoMove {
			break
		}

		// Check legality (MovePicker returns pseudo-legal moves)
		if !b.IsLegal(move) {
			continue
		}

		moveCount++

		// Check if capture BEFORE making the move
		isCap := isCapture(move, b)

		b.MakeMove(move)

		// Check extension: extend search by 1 ply when move gives check
		givesCheck := b.InCheck()
		extension := 0
		if givesCheck {
			extension = 1
		}
		newDepth := depth - 1 + extension

		childPV := make([]Move, 0, depth-1)
		var score int

		// Late Move Reductions (LMR) + Principal Variation Search (PVS)
		isKiller := move == killers[0] || move == killers[1]
		hasHighHistory := info.History[move.From()][move.To()] > 1000

		reduction := 0
		if LMREnabled && !inCheck && !isCap && !move.IsPromotion() && !isKiller && !hasHighHistory && !givesCheck {
			d, m := depth, moveCount
			if d >= 64 {
				d = 63
			}
			if m >= 64 {
				m = 63
			}
			reduction = lmrTable[d][m]
		}

		if reduction > 0 {
			info.LMRAttempts++

			// LMR: reduced depth, zero window
			score = -b.negamax(newDepth-reduction, ply+1, -alpha-1, -alpha, info, &childPV)

			if score > alpha && !info.Stopped {
				// LMR failed high → re-search full depth, zero window (PVS)
				info.LMRReSearches++
				childPV = childPV[:0]
				score = -b.negamax(newDepth, ply+1, -alpha-1, -alpha, info, &childPV)
			} else {
				info.LMRSavings++
			}

			if score > alpha && score < beta && !info.Stopped {
				// PVS failed high → full window re-search
				childPV = childPV[:0]
				score = -b.negamax(newDepth, ply+1, -beta, -alpha, info, &childPV)
			}
		} else if moveCount > 1 {
			// PVS: zero-window for non-first moves
			score = -b.negamax(newDepth, ply+1, -alpha-1, -alpha, info, &childPV)
			if score > alpha && score < beta && !info.Stopped {
				// Failed high → full window re-search
				childPV = childPV[:0]
				score = -b.negamax(newDepth, ply+1, -beta, -alpha, info, &childPV)
			}
		} else {
			// First move: always full window
			score = -b.negamax(newDepth, ply+1, -beta, -alpha, info, &childPV)
		}

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
					// Beta cutoff - update killer moves and history for quiet moves
					if !isCap {
						info.storeKiller(ply, move)
						info.History[move.From()][move.To()] += depth * depth
					}
					break
				}
			}
		}
	}

	// Check for checkmate or stalemate
	if moveCount == 0 {
		if inCheck {
			// Checkmate - return negative mate score adjusted for ply
			return -MateScore + ply
		}
		// Stalemate
		return 0
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
	return b.quiescenceWithDepth(alpha, beta, info, 0)
}

// quiescenceWithDepth is the internal quiescence search with depth tracking
func (b *Board) quiescenceWithDepth(alpha, beta int, info *SearchInfo, qsDepth int) int {
	// Limit quiescence depth to prevent stack overflow
	if qsDepth >= 32 {
		return b.EvaluateRelative()
	}

	info.Nodes++

	// Stand pat - evaluate the current position
	standPat := b.EvaluateRelative()

	if standPat >= beta {
		return beta
	}

	if standPat > alpha {
		alpha = standPat
	}

	// Use MovePicker for captures only
	picker := NewMovePickerQuiescence(b)

	for {
		move := picker.Next()
		if move == NoMove {
			break
		}

		// Skip bad captures (SEE < 0) - delta pruning
		if !b.SEESign(move, 0) {
			continue
		}

		// Check legality
		if !b.IsLegal(move) {
			continue
		}

		b.MakeMove(move)
		score := -b.quiescenceWithDepth(-beta, -alpha, info, qsDepth+1)
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

// isCapture returns true if the move is a capture
func isCapture(m Move, b *Board) bool {
	return b.Squares[m.To()] != Empty || m.Flags() == FlagEnPassant
}


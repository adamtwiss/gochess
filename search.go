package chess

import (
	"math"
	"sync"
	"sync/atomic"
	"time"
)

const (
	Infinity  = 30000
	MateScore = 29000

	// Contempt: small penalty for accepting draws (repetition, 50-move rule).
	// Positive value means the engine prefers to play on rather than draw.
	Contempt = 10

	// MaxPly is the maximum search depth (for pre-allocated arrays)
	MaxPly = 64
	// MaxQSDepth is the maximum quiescence search depth
	MaxQSDepth = 32
)

// LMREnabled controls whether Late Move Reductions are used
// Set to false for benchmarking comparisons
var LMREnabled = true

// LMPEnabled controls whether Late Move Pruning is used
var LMPEnabled = true

// SingularExtEnabled controls whether Singular Extensions are used
var SingularExtEnabled = true

// SEEQuietPruneEnabled controls whether SEE-based quiet move pruning is used
var SEEQuietPruneEnabled = true

// Late Move Pruning: at shallow depths, skip quiet moves past this move count.
// Indexed by depth (0 unused). Roughly 3 + depth*depth.
var lmpThreshold = [9]int{0, 5, 8, 12, 18, 25, 34, 44, 56}

// LMR reduction table - indexed by [depth][moveNumber]
// Precomputed for efficiency
var lmrTable [64][64]int

func init() {
	// Initialize LMR table using logarithmic formula:
	//   reduction = ln(depth) * ln(moveNum) / C
	// C controls aggressiveness: lower = more reduction.
	// 2.0 is a moderate value (Stockfish uses ~2.25 with additional adjustments).
	const C = 2.0
	for depth := 1; depth < 64; depth++ {
		for moveNum := 1; moveNum < 64; moveNum++ {
			if depth >= 3 && moveNum >= 3 {
				reduction := int(math.Log(float64(depth)) * math.Log(float64(moveNum)) / C)
				// Cap reduction to leave at least depth 1
				if reduction > depth-2 {
					reduction = depth - 2
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
	// Stopped indicates the search should abort. Accessed atomically (0=running, 1=stopped).
	Stopped int32
	TT      *TranspositionTable

	// Deadline is the absolute time (UnixNano) after which the search must stop.
	// Accessed atomically. 0 means no deadline (use MaxTime/StartTime fallback).
	Deadline int64

	// Killer moves: 2 slots per ply, max 64 ply
	Killers [64][2]Move

	// History table: indexed by [from][to], stores cutoff counts.
	// int32 keeps the table at 16KB (fits in L1 cache) vs 32KB with int.
	History [64][64]int32

	// Counter-move heuristic: indexed by [piece][toSquare] of the previous move
	CounterMoves [13][64]Move

	// Continuation history: indexed by [prevPiece][prevTo][curPiece][curTo].
	// Captures the pattern "after piece X moved to square Y, quiet move Z tends
	// to cause beta cutoffs". Updated alongside History on quiet beta cutoffs.
	// ~1.3MB per thread (int16).
	ContHistory [13][64][13][64]int16

	// LMR statistics (for debugging/analysis)
	LMRAttempts   uint64 // Times LMR was attempted
	LMRReSearches uint64 // Times we had to re-search at full depth
	LMRSavings    uint64 // Successful LMR prunings (no re-search needed)

	// LMP statistics
	LMPPrunes uint64 // Moves pruned by late move pruning

	// SEE quiet pruning statistics
	SEEQuietPrunes uint64 // Moves pruned by SEE quiet pruning

	// Singular extension: excluded move per ply for verification search
	ExcludedMove [64]Move

	// Singular extension statistics
	SingularTests      uint64
	SingularExtensions uint64

	// OnDepth is called after each completed iteration of iterative deepening.
	// Parameters: depth, score, cumulative nodes, PV for this depth.
	OnDepth func(depth, score int, nodes uint64, pv []Move)

	// Static eval at each ply, for "improving" detection (LMR/LMP adjustment)
	StaticEvals [MaxPly + 1]int

	// ThreadIndex identifies this thread (0 = main thread)
	ThreadIndex int

	// HelperInfos holds pointers to helper thread SearchInfos.
	// Only used by the main thread (ThreadIndex 0) for node aggregation in OnDepth.
	HelperInfos []*SearchInfo

	// Pre-allocated search structures (avoid per-node heap allocations)
	pickers [MaxPly + MaxQSDepth]MovePicker
	pvTable [MaxPly + 1][MaxPly + 1]Move
	pvLen   [MaxPly + 1]int
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

	// Clear counter-moves
	for i := range info.CounterMoves {
		for j := range info.CounterMoves[i] {
			info.CounterMoves[i][j] = NoMove
		}
	}

	// Clear continuation history
	info.ContHistory = [13][64][13][64]int16{}

	// Clear excluded moves
	for i := range info.ExcludedMove {
		info.ExcludedMove[i] = NoMove
	}

	// Initialize Deadline from MaxTime if not already set (backward compat for Search/SearchWithTT/EPD callers)
	if info.MaxTime > 0 && atomic.LoadInt64(&info.Deadline) == 0 {
		atomic.StoreInt64(&info.Deadline, info.StartTime.Add(info.MaxTime).UnixNano())
	}

	// Advance TT generation so stale entries from previous searches are cheap to evict.
	// Only called here (main thread); helpers share the TT and see the same generation.
	info.TT.NewSearch()

	var bestMove Move
	prevScore := 0

	// Iterative deepening with aspiration windows
	for depth := 1; depth <= maxDepth; depth++ {
		info.Depth = depth
		iterStart := time.Now()

		var score int

		if depth >= 4 && prevScore > -MateScore+100 && prevScore < MateScore-100 {
			// Aspiration window search
			delta := 25
			alpha, beta := prevScore-delta, prevScore+delta
			for {
				score = b.negamax(depth, 0, alpha, beta, info)
				if atomic.LoadInt32(&info.Stopped) != 0 {
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
			score = b.negamax(depth, 0, -Infinity, Infinity, info)
		}

		// Check if we ran out of time mid-search
		if atomic.LoadInt32(&info.Stopped) != 0 {
			break
		}

		// Save results from this iteration
		prevScore = score
		info.Score = score
		if info.pvLen[0] > 0 {
			bestMove = info.pvTable[0][0]
			info.PV = make([]Move, info.pvLen[0])
			copy(info.PV, info.pvTable[0][:info.pvLen[0]])
		}

		// If the PV is short (likely due to TT cutoffs), extend it by
		// walking the transposition table from the end of the known PV.
		if info.TT != nil && len(info.PV) < depth {
			info.PV = b.extendPVFromTT(info.PV, depth, info.TT)
		}

		if info.OnDepth != nil {
			info.OnDepth(depth, score, atomic.LoadUint64(&info.Nodes), info.PV)
		}

		// Check if remaining time is less than the last iteration took
		if d := atomic.LoadInt64(&info.Deadline); d > 0 {
			now := time.Now().UnixNano()
			remaining := d - now
			iterElapsed := now - iterStart.UnixNano()
			if remaining > 0 && remaining < iterElapsed {
				break
			}
		}
	}

	return bestMove, *info
}

// smpSkipDepths controls depth diversification for Lazy SMP helper threads.
// Each row is indexed by (depth % len(row)). A true value means the helper
// thread at that index should skip searching at that depth during iterative
// deepening. This ensures threads are at different depths at any given time,
// improving TT diversity. Rows are assigned to threads round-robin.
var smpSkipDepths = [20][]bool{
	{false, true},                   // thread 1
	{false, false, true},            // thread 2
	{false, false, false, true},     // thread 3
	{false, true, false, true},      // thread 4
	{false, false, true, false, true},                    // thread 5
	{false, false, false, true, false, true},             // thread 6
	{false, false, false, false, true, false, true},      // thread 7
	{false, true, false, false, false, true, false, true}, // thread 8
	{false, false, true, false, false, false, true, false, true},             // thread 9
	{false, false, false, true, false, false, false, true, false, true},      // thread 10
	{false, false, false, false, true, false, false, false, true, false, true}, // thread 11
	{false, true, false, false, false, false, true, false, false, false, true}, // thread 12
	{false, false, true, false, false, false, false, true, false, false, false, true},      // thread 13
	{false, false, false, true, false, false, false, false, true, false, false, false, true}, // thread 14
	{false, false, false, false, true, false, false, false, false, true, false, false, false}, // thread 15
	{false, true, false, false, false, false, false, true, false, false, false, false, true},  // thread 16
	{false, false, true, false, false, false, false, false, true, false, false, false, false},  // thread 17
	{false, false, false, true, false, false, false, false, false, true, false, false, false},  // thread 18
	{false, false, false, false, true, false, false, false, false, false, true, false, false},  // thread 19
	{false, true, false, false, false, false, false, false, true, false, false, false, false},  // thread 20
}

// SearchParallel performs search using Lazy SMP with numThreads threads.
// All threads share the transposition table; each has its own Board, SearchInfo,
// eval cache, and pawn hash table. The main thread (thread 0) runs with the
// provided info (including OnDepth callback). Helper threads use depth
// diversification to improve TT coverage.
//
// If numThreads <= 1, delegates directly to SearchWithInfo.
func (b *Board) SearchParallel(maxDepth int, info *SearchInfo, numThreads int) (Move, SearchInfo) {
	if numThreads <= 1 {
		return b.SearchWithInfo(maxDepth, info)
	}

	// Create helper thread infos
	helpers := make([]*SearchInfo, numThreads-1)
	helperBoards := make([]Board, numThreads-1)

	for i := 0; i < numThreads-1; i++ {
		// Deep copy the board (including undo stack for repetition detection)
		helperBoards[i] = *b
		helperBoards[i].UndoStack = make([]UndoInfo, len(b.UndoStack), len(b.UndoStack)+256)
		copy(helperBoards[i].UndoStack, b.UndoStack)
		// Per-thread eval and pawn tables
		helperBoards[i].PawnTable = NewPawnTable(1)
		helperBoards[i].EvalTable = NewEvalTable(1)

		helpers[i] = &SearchInfo{
			StartTime:   info.StartTime,
			MaxTime:     info.MaxTime,
			TT:          info.TT, // shared TT
			ThreadIndex: i + 1,
		}
		// Copy deadline
		if d := atomic.LoadInt64(&info.Deadline); d > 0 {
			atomic.StoreInt64(&helpers[i].Deadline, d)
		}
	}

	// Store helper infos on the main thread for node aggregation
	info.HelperInfos = helpers
	info.ThreadIndex = 0

	// Wrap the main thread's OnDepth to aggregate nodes from all threads
	callerOnDepth := info.OnDepth
	if callerOnDepth != nil {
		info.OnDepth = func(d, score int, nodes uint64, pv []Move) {
			// Aggregate nodes from all helpers
			totalNodes := nodes
			for _, h := range helpers {
				totalNodes += atomic.LoadUint64(&h.Nodes)
			}
			callerOnDepth(d, score, totalNodes, pv)
		}
	}

	// Launch helper goroutines
	var wg sync.WaitGroup
	for i := 0; i < numThreads-1; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			helperSearch(&helperBoards[idx], maxDepth, helpers[idx])
		}(i)
	}

	// Main thread searches normally
	bestMove, result := b.SearchWithInfo(maxDepth, info)

	// Stop all helpers
	for _, h := range helpers {
		atomic.StoreInt32(&h.Stopped, 1)
	}
	wg.Wait()

	// Aggregate final node count
	for _, h := range helpers {
		result.Nodes += atomic.LoadUint64(&h.Nodes)
	}

	return bestMove, result
}

// helperSearch runs iterative deepening with depth diversification for a helper thread.
// It uses the same logic as SearchWithInfo but skips some depths based on the
// thread's skip pattern to ensure threads are at different depths.
func helperSearch(b *Board, maxDepth int, info *SearchInfo) {
	if info.TT == nil {
		info.TT = NewTranspositionTable(16)
	}

	// Clear per-search tables
	for i := range info.History {
		for j := range info.History[i] {
			info.History[i][j] = 0
		}
	}
	for i := range info.Killers {
		info.Killers[i][0] = NoMove
		info.Killers[i][1] = NoMove
	}
	for i := range info.CounterMoves {
		for j := range info.CounterMoves[i] {
			info.CounterMoves[i][j] = NoMove
		}
	}
	info.ContHistory = [13][64][13][64]int16{}
	for i := range info.ExcludedMove {
		info.ExcludedMove[i] = NoMove
	}

	if info.MaxTime > 0 && atomic.LoadInt64(&info.Deadline) == 0 {
		atomic.StoreInt64(&info.Deadline, info.StartTime.Add(info.MaxTime).UnixNano())
	}

	// Select skip pattern for this thread
	skipIdx := (info.ThreadIndex - 1) % len(smpSkipDepths)
	skipPattern := smpSkipDepths[skipIdx]

	prevScore := 0

	for depth := 1; depth <= maxDepth; depth++ {
		// Depth diversification: skip some depths based on thread's pattern
		if len(skipPattern) > 0 && skipPattern[depth%len(skipPattern)] {
			continue
		}

		info.Depth = depth

		var score int
		if depth >= 4 && prevScore > -MateScore+100 && prevScore < MateScore-100 {
			delta := 25
			alpha, beta := prevScore-delta, prevScore+delta
			for {
				score = b.negamax(depth, 0, alpha, beta, info)
				if atomic.LoadInt32(&info.Stopped) != 0 {
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
				break
			}
		} else {
			score = b.negamax(depth, 0, -Infinity, Infinity, info)
		}

		if atomic.LoadInt32(&info.Stopped) != 0 {
			break
		}

		prevScore = score

		// Check time: stop if remaining time is less than last iteration
		if d := atomic.LoadInt64(&info.Deadline); d > 0 {
			now := time.Now().UnixNano()
			remaining := d - now
			if remaining <= 0 {
				break
			}
		}
	}
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

// extendPVFromTT extends a PV by walking the transposition table from the
// position after the last PV move. This recovers full-length PVs when the
// search returned early due to TT cutoffs. The board state is preserved.
func (b *Board) extendPVFromTT(pv []Move, maxLen int, tt *TranspositionTable) []Move {
	if len(pv) >= maxLen {
		return pv
	}

	// Replay the known PV moves
	var madeStack []Move
	for _, m := range pv {
		b.MakeMove(m)
		madeStack = append(madeStack, m)
	}

	// Walk the TT to extend
	seen := make(map[uint64]bool)
	for len(pv) < maxLen {
		if seen[b.HashKey] {
			break // cycle detection
		}
		seen[b.HashKey] = true

		entry, found := tt.Probe(b.HashKey)
		if !found || entry.Move == NoMove {
			break
		}
		m := entry.Move
		if !b.IsPseudoLegal(m) || !b.IsLegal(m) {
			break
		}
		pv = append(pv, m)
		b.MakeMove(m)
		madeStack = append(madeStack, m)
	}

	// Undo all moves to restore original position
	for i := len(madeStack) - 1; i >= 0; i-- {
		b.UnmakeMove(madeStack[i])
	}

	return pv
}

// negamax performs alpha-beta search from the current position
// ply is the distance from root (for mate score adjustment)
func (b *Board) negamax(depth, ply int, alpha, beta int, info *SearchInfo) int {
	// Guard against stack overflow (Go has limited goroutine stack)
	if ply >= MaxPly {
		return b.EvaluateRelative()
	}

	// Clear PV for this node (must happen before any early return so parent
	// doesn't copy stale PV data from a previous search at this ply)
	info.pvLen[ply] = 0

	// Check time periodically
	if atomic.LoadUint64(&info.Nodes)&4095 == 0 {
		if d := atomic.LoadInt64(&info.Deadline); d > 0 && time.Now().UnixNano() >= d {
			atomic.StoreInt32(&info.Stopped, 1)
			return 0
		}
	}

	if atomic.LoadInt32(&info.Stopped) != 0 {
		return 0
	}

	atomic.AddUint64(&info.Nodes, 1)

	// Draw detection: repetition and 50-move rule
	if ply > 0 {
		if b.HalfmoveClock >= 100 {
			return -Contempt
		}
		if b.IsRepetition() {
			return -Contempt
		}
	}

	// Probe transposition table
	ttMove := NoMove
	alphaOrig := alpha
	var ttHit bool
	var ttEntry TTEntry

	if entry, found := info.TT.Probe(b.HashKey); found {
		ttHit = true
		ttEntry = entry
		ttMove = entry.Move

		if info.ExcludedMove[ply] == NoMove && int(entry.Depth) >= depth && ply > 0 {
			score := int(entry.Score)
			// Adjust mate scores for distance from root
			if score > MateScore-100 {
				score -= ply
			} else if score < -MateScore+100 {
				score += ply
			}

			switch entry.Flag {
			case TTExact:
				info.pvTable[ply][0] = ttMove
				info.pvLen[ply] = 1
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
				info.pvTable[ply][0] = ttMove
				info.pvLen[ply] = 1
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
	// Skip if: in check, at root, depth too shallow, or no non-pawn material (zugzwang risk)
	stmNonPawn := b.Occupied[b.SideToMove] &^ b.Pieces[pieceOf(WhitePawn, b.SideToMove)] &^ b.Pieces[pieceOf(WhiteKing, b.SideToMove)]
	if depth >= 3 && !inCheck && ply > 0 && stmNonPawn != 0 {
		R := 2 // Reduction factor
		if depth > 6 {
			R = 3
		}

		b.MakeNullMove()
		score := -b.negamax(depth-1-R, ply+1, -beta, -beta+1, info)
		b.UnmakeNullMove()

		if atomic.LoadInt32(&info.Stopped) != 0 {
			return 0
		}

		if score >= beta {
			return beta // Null-move cutoff
		}
	}

	// Compute static eval for pruning and LMR improving detection.
	// Stored per-ply so we can compare to 2 plies ago.
	staticEval := -Infinity
	improving := false
	if !inCheck {
		staticEval = b.EvaluateRelative()
		if ply <= MaxPly {
			info.StaticEvals[ply] = staticEval
		}
		// Position is "improving" if our eval is better than 2 plies ago.
		// When ply-2 was in check we stored -Infinity, so improving=true
		// (conservative: don't reduce extra when uncertain).
		if ply >= 2 {
			improving = staticEval > info.StaticEvals[ply-2]
		}

		// Reverse Futility Pruning (Static Null Move Pruning)
		// If static eval is far above beta, prune the whole node
		if depth <= 3 && ply > 0 {
			margin := depth * 120
			if staticEval-margin >= beta {
				return staticEval - margin
			}
		}
	} else {
		if ply <= MaxPly {
			info.StaticEvals[ply] = -Infinity
		}
	}

	// Get killers for this ply
	var killers [2]Move
	if ply < 64 {
		killers = info.Killers[ply]
	}

	// Counter-move and continuation history lookup from opponent's last move
	var counterMove Move
	var contHistPtr *[13][64]int16
	if len(b.UndoStack) > 0 {
		undo := b.UndoStack[len(b.UndoStack)-1]
		pm := undo.Move
		if pm != NoMove {
			prevPiece := b.Squares[pm.To()]
			if prevPiece != Empty {
				counterMove = info.CounterMoves[prevPiece][pm.To()]
				contHistPtr = &info.ContHistory[prevPiece][pm.To()]
			}
		}
	}

	// Use MovePicker for staged move generation (reuse pre-allocated picker)
	info.pickers[ply].Init(b, ttMove, ply, killers, &info.History, counterMove, contHistPtr)
	picker := &info.pickers[ply]

	bestMove := NoMove
	bestScore := -Infinity
	moveCount := 0

	// Track quiet moves searched before beta cutoff for history penalty
	var quietsTried [64]Move
	quietsCount := 0

	for {
		move := picker.Next()
		if move == NoMove {
			break
		}

		// Skip excluded move (singular extension verification search)
		if move == info.ExcludedMove[ply] {
			continue
		}

		// Check legality (MovePicker returns pseudo-legal moves)
		if !b.IsLegal(move) {
			continue
		}

		moveCount++

		// Check if capture BEFORE making the move
		isCap := isCapture(move, b)

		// SEE quiet pruning: compute SEE before MakeMove (doesn't modify board)
		var seeQuietScore int
		checkSEEQuiet := false
		if SEEQuietPruneEnabled && ply > 0 && !inCheck && depth <= 8 &&
			!isCap && !move.IsPromotion() &&
			move != killers[0] && move != killers[1] &&
			move != counterMove && move != ttMove &&
			bestScore > -MateScore+100 {
			seeQuietScore = b.SEEAfterQuiet(move)
			checkSEEQuiet = true
		}

		// Singular extension: if TT move is significantly better than alternatives, extend it
		singularExtension := 0
		if SingularExtEnabled &&
			move == ttMove &&
			ttMove != NoMove &&
			ply > 0 &&
			depth >= 10 &&
			!inCheck &&
			info.ExcludedMove[ply] == NoMove &&
			ttHit &&
			ttEntry.Flag != TTUpper &&
			int(ttEntry.Depth) >= depth-3 {

			ttScore := int(ttEntry.Score)
			if ttScore > MateScore-100 {
				ttScore -= ply
			} else if ttScore < -MateScore+100 {
				ttScore += ply
			}

			// Skip singular extension for mate scores — margin comparison is meaningless
			if ttScore <= MateScore-100 && ttScore >= -MateScore+100 {
				singularBeta := ttScore - depth*3
				singularDepth := (depth - 1) / 2

				info.ExcludedMove[ply] = ttMove
				singularScore := b.negamax(singularDepth, ply, singularBeta-1, singularBeta, info)
				info.ExcludedMove[ply] = NoMove

				if atomic.LoadInt32(&info.Stopped) != 0 {
					return 0
				}

				info.SingularTests++

				if singularScore < singularBeta {
					singularExtension = 1
					info.SingularExtensions++
				}
			}
		}

		b.MakeMove(move)

		// Check extension: extend search by 1 ply when move gives check
		givesCheck := b.InCheck()

		// Futility pruning: at shallow depths, skip quiet moves that can't raise alpha
		if staticEval > -Infinity && depth <= 2 && !inCheck && !givesCheck &&
			!isCap && !move.IsPromotion() &&
			bestScore > -MateScore+100 {
			futilityMargin := [3]int{0, 200, 400}
			if staticEval+futilityMargin[depth] <= alpha {
				b.UnmakeMove(move)
				continue
			}
		}

		// Late Move Pruning: at shallow depths, skip late quiet moves
		// Placed after MakeMove so we can exempt check-giving moves
		if LMPEnabled && ply > 0 && !inCheck && depth >= 1 && depth <= 8 &&
			!isCap && !move.IsPromotion() && !givesCheck &&
			moveCount > lmpThreshold[depth] &&
			bestScore > -MateScore+100 {
			info.LMPPrunes++
			b.UnmakeMove(move)
			continue
		}

		// SEE quiet pruning: prune quiet moves where piece lands on a losing square
		if checkSEEQuiet && !givesCheck && seeQuietScore < -depth*80 {
			info.SEEQuietPrunes++
			b.UnmakeMove(move)
			continue
		}

		extension := 0
		if givesCheck {
			extension = 1
		}
		if singularExtension > 0 && extension == 0 {
			extension = singularExtension
		}
		newDepth := depth - 1 + extension

		var score int

		// Track quiet moves for history penalty on beta cutoff
		if !isCap && !move.IsPromotion() && quietsCount < len(quietsTried) {
			quietsTried[quietsCount] = move
			quietsCount++
		}

		// Late Move Reductions (LMR) + Principal Variation Search (PVS)
		isKiller := move == killers[0] || move == killers[1]

		reduction := 0
		if LMREnabled && !inCheck && !isCap && !move.IsPromotion() && !isKiller && !givesCheck {
			d, m := depth, moveCount
			if d >= 64 {
				d = 63
			}
			if m >= 64 {
				m = 63
			}
			reduction = lmrTable[d][m]

			if reduction > 0 {
				// Reduce less at PV nodes where accuracy matters most
				if beta-alpha > 1 {
					reduction--
				}

				// Reduce less when the position is improving (eval > eval 2 plies ago)
				if improving {
					reduction--
				}

				// Continuous history adjustment: good history reduces less, bad more
				histScore := info.History[move.From()][move.To()]
				if contHistPtr != nil {
					histScore += int32(contHistPtr[b.Squares[move.To()]][move.To()])
				}
				reduction -= int(histScore / 5000)

				// Clamp: never extend (negative), never reduce past depth 1
				if reduction < 0 {
					reduction = 0
				}
				if reduction > newDepth-1 {
					reduction = newDepth - 1
				}
			}
		}

		if reduction > 0 {
			info.LMRAttempts++

			// LMR: reduced depth, zero window
			score = -b.negamax(newDepth-reduction, ply+1, -alpha-1, -alpha, info)

			if score > alpha && atomic.LoadInt32(&info.Stopped) == 0 {
				// LMR failed high → re-search full depth, zero window (PVS)
				info.LMRReSearches++
				score = -b.negamax(newDepth, ply+1, -alpha-1, -alpha, info)
			} else {
				info.LMRSavings++
			}

			if score > alpha && score < beta && atomic.LoadInt32(&info.Stopped) == 0 {
				// PVS failed high → full window re-search
				score = -b.negamax(newDepth, ply+1, -beta, -alpha, info)
			}
		} else if moveCount > 1 {
			// PVS: zero-window for non-first moves
			score = -b.negamax(newDepth, ply+1, -alpha-1, -alpha, info)
			if score > alpha && score < beta && atomic.LoadInt32(&info.Stopped) == 0 {
				// Failed high → full window re-search
				score = -b.negamax(newDepth, ply+1, -beta, -alpha, info)
			}
		} else {
			// First move: always full window
			score = -b.negamax(newDepth, ply+1, -beta, -alpha, info)
		}

		b.UnmakeMove(move)

		if atomic.LoadInt32(&info.Stopped) != 0 {
			return 0
		}

		if score > bestScore {
			bestScore = score
			bestMove = move

			if score > alpha {
				alpha = score

				// Update PV using triangular table
				info.pvTable[ply][0] = move
				copy(info.pvTable[ply][1:], info.pvTable[ply+1][:info.pvLen[ply+1]])
				info.pvLen[ply] = 1 + info.pvLen[ply+1]

				if alpha >= beta {
					// Beta cutoff - update killer moves, history, and counter-move for quiet moves
					if !isCap {
						bonus := int32(depth * depth)
						info.storeKiller(ply, move)
						info.History[move.From()][move.To()] += bonus

						// Update continuation history
						if contHistPtr != nil {
							curPiece := b.Squares[move.From()]
							ch := int32(contHistPtr[curPiece][move.To()]) + bonus
							if ch > 32000 {
								ch = 32000
							}
							contHistPtr[curPiece][move.To()] = int16(ch)
						}

						// Penalize all quiet moves tried before the cutoff move
						for i := 0; i < quietsCount-1; i++ {
							q := quietsTried[i]
							info.History[q.From()][q.To()] -= bonus

							// Penalize continuation history
							if contHistPtr != nil {
								qPiece := b.Squares[q.From()]
								ch := int32(contHistPtr[qPiece][q.To()]) - bonus
								if ch < -32000 {
									ch = -32000
								}
								contHistPtr[qPiece][q.To()] = int16(ch)
							}
						}

						// Store counter-move
						if len(b.UndoStack) > 0 {
							undo := b.UndoStack[len(b.UndoStack)-1]
							pm := undo.Move
							if pm != NoMove {
								prevPiece := b.Squares[pm.To()]
								if prevPiece != Empty {
									info.CounterMoves[prevPiece][pm.To()] = move
								}
							}
						}
					}
					break
				}
			}
		}
	}

	// Check for checkmate or stalemate
	if moveCount == 0 {
		if info.ExcludedMove[ply] != NoMove {
			// Singular verification: no alternative found, return alpha
			return alpha
		}
		if inCheck {
			// Checkmate - return negative mate score adjusted for ply
			return -MateScore + ply
		}
		// Stalemate
		return 0
	}

	// Store in transposition table (skip during singular verification)
	if info.ExcludedMove[ply] == NoMove {
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
	}

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

	atomic.AddUint64(&info.Nodes, 1)

	// Stand pat - evaluate the current position
	standPat := b.EvaluateRelative()

	if standPat >= beta {
		return beta
	}

	if standPat > alpha {
		alpha = standPat
	}

	// Use MovePicker for captures only (reuse pre-allocated picker)
	qsIdx := MaxPly + qsDepth
	info.pickers[qsIdx].InitQuiescence(b)
	picker := &info.pickers[qsIdx]

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

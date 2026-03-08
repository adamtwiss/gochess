package chess

import (
	"math"
	"sort"
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

// DoubleSingularExtEnabled controls whether double singular extensions are used
// (extend by 2 when TT move is overwhelmingly better than alternatives)
var DoubleSingularExtEnabled = true

// NegativeSingularExtEnabled controls whether negative singular extensions are used
// (reduce by 1 when alternatives are just as good as TT move)
var NegativeSingularExtEnabled = true

// SEEQuietPruneEnabled controls whether SEE-based quiet move pruning is used
var SEEQuietPruneEnabled = true

// IIREnabled controls whether Internal Iterative Reductions are used
var IIREnabled = true

// RazoringEnabled controls whether razoring is used
var RazoringEnabled = true

// DeltaPruningEnabled controls whether delta pruning in quiescence is used
var DeltaPruningEnabled = true

// RecaptureExtEnabled controls whether recapture extensions are used
var RecaptureExtEnabled = true

// PassedPawnExtEnabled controls whether passed pawn push extensions are used
var PassedPawnExtEnabled = true

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

	// SoftDeadline is the soft time limit (UnixNano). The search may stop here
	// if the position is stable (best move not changing, score not fluctuating).
	// 0 means fall back to Deadline-only logic (used by EPD/benchmark/movetime).
	// Accessed atomically.
	SoftDeadline int64

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

	// Capture history: indexed by [movingPiece][toSquare][capturedPieceType].
	// Tracks which captures caused beta cutoffs, improving ordering among
	// equal-MVV-LVA captures. ~11.4KB per thread (int16).
	CaptHistory [13][64][7]int16

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
	SingularTests              uint64
	SingularExtensions         uint64
	DoubleSingularExtensions   uint64
	NegativeSingularExtensions uint64

	// Recapture and passed pawn extension statistics
	RecaptureExtensions  uint64
	PassedPawnExtensions uint64

	// Tablebase statistics
	TBHits uint64

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

	// Dynamic time management: best-move stability tracking (main thread only, no atomics needed)
	tmPrevBestMove   Move // best move from previous iteration
	tmPrevScore      int  // score from previous iteration
	tmBestMoveStable int  // consecutive iterations with unchanged best move
	tmHasData        bool // true after first completed iteration

	// RootMoves holds root moves sorted by score from the previous iteration.
	// Initialized before iterative deepening, re-sorted between iterations.
	RootMoves []RootMove

	// Pre-allocated search structures (avoid per-node heap allocations)
	pickers [MaxPly + MaxQSDepth]MovePicker
	pvTable [MaxPly + 1][MaxPly + 1]Move
	pvLen   [MaxPly + 1]int
}

// RootMove holds a root move and its score from the previous iteration,
// used to order root moves by quality across iterative deepening.
type RootMove struct {
	Move  Move
	Score int
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

// historyBonus computes a depth-based bonus for history updates, capped to avoid
// over-weighting very deep searches.
func historyBonus(depth int) int32 {
	b := int32(depth * depth)
	if b > 1200 {
		b = 1200
	}
	return b
}

// updateHistory applies the gravity formula: new = old + bonus - old*|bonus|/D.
// This naturally bounds values to approximately [-D, +D] and prevents saturation.
// D=16384 for the int32 history table.
func (info *SearchInfo) updateHistory(from, to Square, bonus int32) {
	v := info.History[from][to]
	abs := bonus
	if abs < 0 {
		abs = -abs
	}
	info.History[from][to] = v + bonus - v*abs/16384
}

// updateContHistory applies gravity to a continuation history entry.
// D=16384 for int16 (values stay in [-16384, 16384] which fits int16 range).
func updateContHistory(table *[13][64]int16, piece Piece, to Square, bonus int32) {
	v := int32(table[piece][to])
	abs := bonus
	if abs < 0 {
		abs = -abs
	}
	table[piece][to] = int16(v + bonus - v*abs/16384)
}

// updateCaptHistory applies gravity to a capture history entry.
func (info *SearchInfo) updateCaptHistory(piece Piece, to Square, cpt int, bonus int32) {
	v := int32(info.CaptHistory[piece][to][cpt])
	abs := bonus
	if abs < 0 {
		abs = -abs
	}
	info.CaptHistory[piece][to][cpt] = int16(v + bonus - v*abs/16384)
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

	// Clear capture history
	info.CaptHistory = [13][64][7]int16{}

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

	// Probe Syzygy tablebases at root (DTZ)
	if b.tbCanProbeRoot() {
		if tbMove, tbWDL, _, ok := b.TBProbeRoot(); ok {
			info.TBHits++
			tbScore := tbWDLToScore(tbWDL)
			info.Score = tbScore
			info.PV = []Move{tbMove}
			if info.OnDepth != nil {
				info.OnDepth(1, tbScore, 1, info.PV)
			}
			return tbMove, *info
		}
	}

	// Initialize root moves: generate all legal moves, sorted by TT move first
	legalMoves := b.GenerateLegalMoves()
	info.RootMoves = make([]RootMove, len(legalMoves))
	for i, m := range legalMoves {
		info.RootMoves[i] = RootMove{Move: m, Score: -Infinity}
	}

	var bestMove Move
	prevScore := 0

	// Iterative deepening with aspiration windows
	for depth := 1; depth <= maxDepth; depth++ {
		info.Depth = depth
		iterStart := time.Now()

		var score int

		if depth >= 4 && prevScore > -MateScore+100 && prevScore < MateScore-100 {
			// Aspiration window search with Stockfish-style fail handling:
			// On fail-low: narrow beta down (we know score <= old alpha),
			//   re-center alpha on actual score. On fail-high: narrow alpha
			//   up, re-center beta on actual score.
			delta := 15
			alpha := prevScore - delta
			beta := prevScore + delta
			if alpha < -Infinity {
				alpha = -Infinity
			}
			if beta > Infinity {
				beta = Infinity
			}
			for {
				score = b.negamax(depth, 0, alpha, beta, info)
				if atomic.LoadInt32(&info.Stopped) != 0 {
					break
				}
				if score <= alpha {
					// Fail low: true score is below window.
					// Narrow beta down to old alpha, widen alpha based on score.
					beta = (alpha + beta) / 2
					alpha = score - delta
					if alpha < -Infinity {
						alpha = -Infinity
					}
					delta += delta / 2
					continue
				}
				if score >= beta {
					// Fail high: true score is above window.
					// Narrow alpha up to old beta, widen beta based on score.
					alpha = beta
					beta = score + delta
					if beta > Infinity {
						beta = Infinity
					}
					delta += delta / 2
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

		// Sort root moves by score from this iteration (best first).
		// The best move goes to index 0 so it's searched first with full window
		// on the next iteration — moves 2-N get PVS zero-window searches.
		sort.Slice(info.RootMoves, func(i, j int) bool {
			return info.RootMoves[i].Score > info.RootMoves[j].Score
		})

		// If the PV is short (likely due to TT cutoffs), extend it by
		// walking the transposition table from the end of the known PV.
		if info.TT != nil && len(info.PV) < depth {
			info.PV = b.extendPVFromTT(info.PV, depth, info.TT)
		}

		if info.OnDepth != nil {
			info.OnDepth(depth, score, info.Nodes, info.PV)
		}

		// Dynamic time management: soft/hard deadline check
		softDL := atomic.LoadInt64(&info.SoftDeadline)
		hardDL := atomic.LoadInt64(&info.Deadline)
		now := time.Now().UnixNano()

		if softDL > 0 && depth >= 4 {
			// Update stability tracking
			if info.tmHasData {
				if bestMove != info.tmPrevBestMove {
					info.tmBestMoveStable = 0
				} else {
					info.tmBestMoveStable++
				}
			}

			// Score delta (ignore mate-range swings)
			scoreDelta := 0
			if info.tmHasData {
				scoreDelta = abs(score - info.tmPrevScore)
				isMateRange := score > MateScore-200 || score < -MateScore+200 ||
					info.tmPrevScore > MateScore-200 || info.tmPrevScore < -MateScore+200
				if isMateRange {
					scoreDelta = 0
				}
			}

			// Save for next iteration
			info.tmPrevBestMove = bestMove
			info.tmPrevScore = score
			info.tmHasData = true

			// Time scaling factor (1.0 = use soft limit as-is)
			scale := 1.0

			// Stable best move → stop early (more aggressive with longer stability)
			if info.tmBestMoveStable >= 8 {
				scale *= 0.35
			} else if info.tmBestMoveStable >= 5 {
				scale *= 0.5
			} else if info.tmBestMoveStable >= 3 {
				scale *= 0.7
			} else if info.tmBestMoveStable >= 1 {
				scale *= 0.85
			}

			// Effort concentration: very stable move with stable score → extra reduction
			if info.tmBestMoveStable >= 5 && scoreDelta <= 10 {
				scale *= 0.8
			}

			// Unstable score → extend time
			if scoreDelta > 50 {
				scale *= 1.4
			} else if scoreDelta > 25 {
				scale *= 1.2
			}

			// Compute adjusted soft deadline
			softDuration := softDL - info.StartTime.UnixNano()
			adjustedSoft := info.StartTime.UnixNano() + int64(float64(softDuration)*scale)

			// Clamp to hard deadline
			if hardDL > 0 && adjustedSoft > hardDL {
				adjustedSoft = hardDL
			}

			if now >= adjustedSoft {
				break // soft stop
			}

			// Also check if next iteration would exceed hard deadline.
			// Use 2x last iteration time as estimate (exponential branching).
			if hardDL > 0 {
				remaining := hardDL - now
				iterElapsed := now - iterStart.UnixNano()
				if remaining > 0 && remaining < 2*iterElapsed {
					break
				}
			}
		} else if hardDL > 0 {
			// No soft deadline: existing behavior (EPD/benchmark/movetime/depth<4)
			remaining := hardDL - now
			iterElapsed := now - iterStart.UnixNano()
			if remaining > 0 && remaining < 2*iterElapsed {
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
		// Per-thread pawn table
		helperBoards[i].PawnTable = NewPawnTable(1)
		// Per-thread NNUE accumulator (net is shared read-only)
		if b.NNUEAcc != nil {
			helperBoards[i].NNUEAcc = b.NNUEAcc.DeepCopy()
		}

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
				totalNodes += h.Nodes
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
		result.Nodes += h.Nodes
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
	info.CaptHistory = [13][64][7]int16{}
	for i := range info.ExcludedMove {
		info.ExcludedMove[i] = NoMove
	}

	if info.MaxTime > 0 && atomic.LoadInt64(&info.Deadline) == 0 {
		atomic.StoreInt64(&info.Deadline, info.StartTime.Add(info.MaxTime).UnixNano())
	}

	// Initialize root moves for this helper thread
	legalMoves := b.GenerateLegalMoves()
	info.RootMoves = make([]RootMove, len(legalMoves))
	for i, m := range legalMoves {
		info.RootMoves[i] = RootMove{Move: m, Score: -Infinity}
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
			delta := 15
			alpha := prevScore - delta
			beta := prevScore + delta
			if alpha < -Infinity {
				alpha = -Infinity
			}
			if beta > Infinity {
				beta = Infinity
			}
			for {
				score = b.negamax(depth, 0, alpha, beta, info)
				if atomic.LoadInt32(&info.Stopped) != 0 {
					break
				}
				if score <= alpha {
					beta = (alpha + beta) / 2
					alpha = score - delta
					if alpha < -Infinity {
						alpha = -Infinity
					}
					delta += delta / 2
					continue
				}
				if score >= beta {
					alpha = beta
					beta = score + delta
					if beta > Infinity {
						beta = Infinity
					}
					delta += delta / 2
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

		// Sort root moves by score from this iteration
		sort.Slice(info.RootMoves, func(i, j int) bool {
			return info.RootMoves[i].Score > info.RootMoves[j].Score
		})

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
		if !b.IsPseudoLegal(m) || !b.IsLegalSlow(m) {
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
	if info.Nodes&1023 == 0 {
		if d := atomic.LoadInt64(&info.Deadline); d > 0 && time.Now().UnixNano() >= d {
			atomic.StoreInt32(&info.Stopped, 1)
			return 0
		}
	}

	if atomic.LoadInt32(&info.Stopped) != 0 {
		return 0
	}

	info.Nodes++

	// Draw detection: repetition and 50-move rule
	if ply > 0 {
		if b.HalfmoveClock >= 100 {
			return -Contempt
		}
		if b.IsRepetition() {
			return -Contempt
		}
	}

	// Syzygy WDL probe (after draw detection, before TT)
	if ply > 0 && b.tbCanProbeWDL(depth) && b.HalfmoveClock == 0 {
		if tbScore, ok := b.TBProbeWDL(); ok {
			info.TBHits++
			// Store in TT for future cutoffs
			var ttFlag TTFlag
			var score int
			if tbScore >= TBWinScore {
				score = tbScore - ply // Adjust for distance like mate scores
				ttFlag = TTLower
			} else if tbScore <= TBLossScore {
				score = tbScore + ply
				ttFlag = TTUpper
			} else {
				score = tbScore
				ttFlag = TTExact
			}

			if ttFlag == TTExact ||
				(ttFlag == TTLower && score >= beta) ||
				(ttFlag == TTUpper && score <= alpha) {
				info.TT.Store(b.HashKey, MaxPly, score, ttFlag, NoMove, 0)
				return score
			}

			// Use TB score as bound even if we can't cut
			if ttFlag == TTLower && score > alpha {
				alpha = score
			}
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
				if ttMove != NoMove {
					info.pvTable[ply][0] = ttMove
					info.pvLen[ply] = 1
				} else {
					info.pvLen[ply] = 0
				}
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
				if ttMove != NoMove {
					info.pvTable[ply][0] = ttMove
					info.pvLen[ply] = 1
				} else {
					info.pvLen[ply] = 0
				}
				return score
			}
		}
	}

	// Leaf node - go to quiescence search
	if depth <= 0 {
		return b.quiescence(alpha, beta, ply, info)
	}

	// Compute pinned pieces and checkers together (shares slider ray work).
	// This replaces separate InCheck() + PinnedPieces() calls.
	pinned, checkers := b.PinnedAndCheckers(b.SideToMove)
	inCheck := checkers != 0
	checkSq, discoverers := b.CheckData(b.SideToMove)

	// Compute static eval for pruning and LMR improving detection.
	// Stored per-ply so we can compare to 2 plies ago.
	// Computed early because NMP, RFP, razoring, and futility all need it.
	// Use TT staticEval when available to avoid recomputing.
	staticEval := -Infinity
	improving := false
	if !inCheck {
		if ttHit && ttEntry.StaticEval > -MateScore+100 {
			staticEval = int(ttEntry.StaticEval)
		} else {
			staticEval = b.EvaluateRelative()
		}
		if ply <= MaxPly {
			info.StaticEvals[ply] = staticEval
		}
		// Position is "improving" if our eval is better than 2 plies ago.
		// When ply-2 was in check we stored -Infinity, so improving=true
		// (conservative: don't reduce extra when uncertain).
		if ply >= 2 {
			improving = staticEval > info.StaticEvals[ply-2]
		}
	} else {
		if ply <= MaxPly {
			info.StaticEvals[ply] = -Infinity
		}
	}

	// Internal Iterative Reduction: reduce depth when no TT move exists.
	// Searching without a good move to try first is less efficient.
	if IIREnabled && depth >= 6 && ttMove == NoMove && !inCheck {
		depth--
	}

	// Null-move pruning
	// Skip if: in check, at root, depth too shallow, or no non-pawn material (zugzwang risk)
	stmNonPawn := b.Occupied[b.SideToMove] &^ b.Pieces[pieceOf(WhitePawn, b.SideToMove)] &^ b.Pieces[pieceOf(WhiteKing, b.SideToMove)]
	if depth >= 3 && !inCheck && ply > 0 && stmNonPawn != 0 && beta-alpha == 1 {
		// Adaptive reduction: scales with depth and eval margin above beta
		R := 3 + depth/3
		if staticEval > beta {
			evalR := int((staticEval - beta) / 200)
			if evalR > 3 {
				evalR = 3
			}
			R += evalR
		}
		// Clamp so null-move search is at least depth 1 (not pure quiescence)
		if depth-1-R < 1 {
			R = depth - 2
		}

		b.MakeNullMove()
		score := -b.negamax(depth-1-R, ply+1, -beta, -beta+1, info)
		b.UnmakeNullMove()

		if atomic.LoadInt32(&info.Stopped) != 0 {
			return 0
		}

		if score >= beta {
			// Verification search at high depths to guard against zugzwang.
			// Re-search at reduced depth without null move to confirm the cutoff.
			if depth >= 12 {
				vScore := b.negamax(depth-1-R, ply, beta-1, beta, info)
				if vScore >= beta {
					return beta
				}
			} else {
				return beta
			}
		}
	}

	if !inCheck {
		// Reverse Futility Pruning (Static Null Move Pruning)
		// If static eval is far above beta, prune the whole node.
		// Improving-aware margin: tighter when eval is trending up
		// improving = true -> depth * 85 (trust rising eval, prune tighter)
		// improving = false -> depth * 120 (conservative, same as baseline)
		if depth <= 6 && ply > 0 {
			margin := depth * 120
			if improving {
				margin = depth * 85
			}
			if staticEval-margin >= beta {
				return staticEval - margin
			}
		}

		// Razoring: at shallow depths, if eval is far below alpha, drop to quiescence
		if RazoringEnabled && depth <= 2 && ply > 0 {
			razoringMargin := 400 + depth*100
			if staticEval+razoringMargin < alpha {
				score := b.quiescence(alpha, beta, ply, info)
				if score < alpha {
					return score
				}
			}
		}
	}

	// ProbCut: at moderate+ depths, if a shallow search of captures with
	// raised beta confirms the position is winning, prune the node.
	probCutBeta := beta + 200
	if !inCheck && ply > 0 && depth >= 5 && staticEval+100 >= probCutBeta {
		pcDepth := depth - 4
		var pcMoves [64]Move
		pcCount := 0
		caps := b.GenerateCaptures()
		for _, m := range caps {
			if b.SEESign(m, 0) && pcCount < len(pcMoves) {
				pcMoves[pcCount] = m
				pcCount++
			}
		}
		for i := 0; i < pcCount; i++ {
			m := pcMoves[i]
			if !b.IsLegalSlow(m) {
				continue
			}
			b.MakeMove(m)
			// Zero-window search at reduced depth
			score := -b.negamax(pcDepth, ply+1, -probCutBeta, -probCutBeta+1, info)
			b.UnmakeMove(m)
			if atomic.LoadInt32(&info.Stopped) != 0 {
				return 0
			}
			if score >= probCutBeta {
				return score
			}
		}
	}

	// Root move ordering: at ply 0, use pre-sorted RootMoves list instead of MovePicker.
	// Moves are sorted by score from the previous iteration (best first).
	if ply == 0 && len(info.RootMoves) > 0 {
		return b.searchRoot(depth, alpha, beta, info, pinned, inCheck, checkers, checkSq, discoverers)
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
	if inCheck {
		info.pickers[ply].InitEvasion(b, ttMove, ply, checkers, pinned, &info.History, contHistPtr, &info.CaptHistory)
	} else {
		info.pickers[ply].Init(b, ttMove, ply, killers, &info.History, counterMove, contHistPtr, &info.CaptHistory)
	}
	picker := &info.pickers[ply]

	bestMove := NoMove
	bestScore := -Infinity
	moveCount := 0

	// Track quiet moves searched before beta cutoff for history penalty
	var quietsTried [64]Move
	quietsCount := 0

	// Track captures searched before beta cutoff for capture history penalty
	var capturesTried [32]Move
	capturesCount := 0

	for {
		move := picker.Next()
		if move == NoMove {
			break
		}

		// Skip excluded move (singular extension verification search)
		if move == info.ExcludedMove[ply] {
			continue
		}

		// Check legality (MovePicker returns pseudo-legal moves; evasion moves are already legal)
		if !inCheck && !b.IsLegal(move, pinned, false) {
			continue
		}

		moveCount++

		// Check if capture BEFORE making the move
		isCap := isCapture(move, b)

		// SEE capture pruning: at shallow depths, prune captures that lose material
		if isCap && ply > 0 && !inCheck && depth <= 6 &&
			move != ttMove && bestScore > -MateScore+100 &&
			!b.SEESign(move, -depth*100) {
			continue
		}

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
					// Double extension: if TT move is overwhelmingly better
					if DoubleSingularExtEnabled && singularScore < singularBeta-depth*3 {
						singularExtension = 2
						info.DoubleSingularExtensions++
					}
				} else if NegativeSingularExtEnabled && singularScore >= ttScore+depth*3 {
					// Negative extension: alternatives are just as good, reduce
					singularExtension = -1
					info.NegativeSingularExtensions++
				}
			}
		}

		// Save moved piece before MakeMove for consistent history indexing
		movedPiece := b.Squares[move.From()]

		// History-based pruning: prune quiet moves with deeply negative history at shallow depths
		if ply > 0 && !inCheck && !improving && depth <= 3 &&
			!isCap && !move.IsPromotion() &&
			move != ttMove &&
			move != killers[0] && move != killers[1] &&
			move != counterMove &&
			bestScore > -MateScore+100 {
			histPruneScore := info.History[move.From()][move.To()]
			if contHistPtr != nil {
				histPruneScore += int32(contHistPtr[movedPiece][move.To()])
			}
			if histPruneScore < -2000*int32(depth) {
				continue
			}
		}

		b.MakeMove(move)

		// Check extension: extend search by 1 ply when move gives check
		var givesCheck bool
		flags := move.Flags()
		if flags == FlagEnPassant || flags == FlagCastle {
			// Rare special cases: fall back to full check detection
			givesCheck = b.InCheck()
		} else {
			// Piece type index: 1-6 (Pawn..King)
			pt := int(movedPiece)
			if pt > 6 {
				pt -= 6 // Black piece → white piece type index
			}
			if move.IsPromotion() {
				pt = flags - FlagPromoteN + 2 // FlagPromoteN=4→Knight(2), ..., FlagPromoteQ=7→Queen(5)
			}
			givesCheck = checkSq[pt]&SquareBB(move.To()) != 0 ||
				discoverers&SquareBB(move.From()) != 0
		}

		// Futility pruning: use estimated post-LMR depth for tighter margin
		if staticEval > -Infinity && depth <= 8 && !inCheck && !givesCheck &&
			!isCap && !move.IsPromotion() &&
			bestScore > -MateScore+100 {
			// Estimate LMR reduction for this move
			lmrDepth := depth
			if moveCount > 1 && depth >= 2 {
				d, m := depth, moveCount
				if d >= 64 {
					d = 63
				}
				if m >= 64 {
					m = 63
				}
				r := lmrTable[d][m]
				if r > 0 {
					lmrDepth = depth - r
					if lmrDepth < 1 {
						lmrDepth = 1
					}
				}
			}
			if staticEval+100+lmrDepth*100 <= alpha {
				b.UnmakeMove(move)
				continue
			}
		}

		// Late Move Pruning: at shallow depths, skip late quiet moves
		// Placed after MakeMove so we can exempt check-giving moves
	// Formula: (3 + depth*depth), +50% when improving
		if LMPEnabled && ply > 0 && !inCheck && depth >= 1 && depth <= 8 &&
			!isCap && !move.IsPromotion() && !givesCheck &&
			bestScore > -MateScore+100 {
			lmpLimit := 3 + depth*depth
			if improving && depth >= 3 {
				lmpLimit += lmpLimit / 2
			}
			if moveCount > lmpLimit {
				info.LMPPrunes++
				b.UnmakeMove(move)
				continue
			}
		}

		// SEE quiet pruning: prune quiet moves where piece lands on a losing square
		if checkSEEQuiet && !givesCheck && seeQuietScore < -20*depth*depth {
			info.SEEQuietPrunes++
			b.UnmakeMove(move)
			continue
		}

		extension := 0
		if givesCheck {
			extension = 1
		}
		if singularExtension != 0 && extension == 0 {
			extension = singularExtension
		}

		// Recapture extension: extend when recapturing on the same square
		// the opponent just captured on, to resolve tactical exchanges fully.
		if RecaptureExtEnabled && extension == 0 && isCap && len(b.UndoStack) >= 2 {
			prevUndo := b.UndoStack[len(b.UndoStack)-2] // opponent's move (current move is at top)
			if prevUndo.Captured != Empty && move.To() == prevUndo.Move.To() {
				extension = 1
				info.RecaptureExtensions++
			}
		}

		// Passed pawn push extension: extend pawn pushes to 6th or 7th rank
		// to help resolve critical promotion races and endgame tactics.
		if PassedPawnExtEnabled && extension == 0 && !isCap {
			movedPiece := b.Squares[move.To()]
			moverColor := b.SideToMove ^ 1 // side that just moved (MakeMove flips SideToMove)
			if movedPiece == pieceOf(WhitePawn, moverColor) {
				rank := move.To().Rank()
				relRank := rank              // White: rank 0-7
				if moverColor == Black {
					relRank = 7 - rank // Black: flip so 7th = rank 1
				}
				if relRank >= 5 { // 6th rank (index 5) or 7th rank (index 6)
					// Check if it's actually a passed pawn
					enemyPawns := b.Pieces[pieceOf(WhitePawn, moverColor^1)]
					if PassedPawnMask[moverColor][move.To()]&enemyPawns == 0 {
						extension = 1
						info.PassedPawnExtensions++
					}
				}
			}
		}

		newDepth := depth - 1 + extension

		var score int

		// Track quiet moves for history penalty on beta cutoff
		if !isCap && !move.IsPromotion() && quietsCount < len(quietsTried) {
			quietsTried[quietsCount] = move
			quietsCount++
		}

		// Track captures for capture history penalty on beta cutoff
		if isCap && capturesCount < len(capturesTried) {
			capturesTried[capturesCount] = move
			capturesCount++
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

				// Reduce more at expected cut nodes (zero window, not first move)
				if beta-alpha == 1 && moveCount > 1 {
					reduction++
				}

				// Reduce less when the position is improving (eval > eval 2 plies ago)
				if improving {
					reduction--
				}

				// Continuous history adjustment: good history reduces less, bad more
				histScore := info.History[move.From()][move.To()]
				if contHistPtr != nil {
					histScore += int32(contHistPtr[movedPiece][move.To()])
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

		// LMR for captures: reduce captures with bad capture history
		if LMREnabled && !inCheck && isCap && !move.IsPromotion() && !givesCheck && moveCount > 1 && move != ttMove {
			// Only reduce at non-PV nodes (zero window search)
			if beta-alpha == 1 {
				piece := b.Squares[move.From()]
				cpt := capturedType(b.Squares[move.To()])
				if move.Flags() == FlagEnPassant {
					cpt = 1 // pawn
				}
				captHistVal := info.CaptHistory[piece][move.To()][cpt]

				// Only reduce captures with negative capture history
				if captHistVal < 0 {
					reduction = 1
					// Increase reduction for very bad capture history
					if captHistVal < -2000 {
						reduction = 2
					}
					// Never reduce past depth 1
					if reduction > newDepth-1 {
						reduction = newDepth - 1
					}
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
						bonus := historyBonus(depth)
						info.storeKiller(ply, move)
						info.updateHistory(move.From(), move.To(), bonus)

						// Update continuation history
						if contHistPtr != nil {
							curPiece := b.Squares[move.From()]
							updateContHistory(contHistPtr, curPiece, move.To(), bonus)
						}

						// Penalize all quiet moves tried before the cutoff move
						for i := 0; i < quietsCount-1; i++ {
							q := quietsTried[i]
							info.updateHistory(q.From(), q.To(), -bonus)

							// Penalize continuation history
							if contHistPtr != nil {
								qPiece := b.Squares[q.From()]
								updateContHistory(contHistPtr, qPiece, q.To(), -bonus)
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
					} else {
						// Capture caused beta cutoff — update capture history
						bonus := historyBonus(depth)
						piece := b.Squares[move.From()]
						cpt := capturedType(b.Squares[move.To()])
						if move.Flags() == FlagEnPassant {
							cpt = 1 // pawn
						}
						info.updateCaptHistory(piece, move.To(), cpt, bonus)

						// Penalize captures tried before cutoff
						for i := 0; i < capturesCount-1; i++ {
							c := capturesTried[i]
							cp := b.Squares[c.From()]
							ct := capturedType(b.Squares[c.To()])
							if c.Flags() == FlagEnPassant {
								ct = 1
							}
							info.updateCaptHistory(cp, c.To(), ct, -bonus)
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

		info.TT.Store(b.HashKey, depth, storeScore, flag, bestMove, staticEval)
	}

	return bestScore
}

// searchRoot handles the root node (ply=0) using the pre-sorted RootMoves list.
// Moves are ordered by score from the previous iteration rather than MovePicker heuristics.
func (b *Board) searchRoot(depth, alpha, beta int, info *SearchInfo, pinned Bitboard, inCheck bool, checkers Bitboard, checkSq [7]Bitboard, discoverers Bitboard) int {
	ply := 0
	alphaOrig := alpha

	// TT move for singular extension
	ttMove := NoMove
	var ttHit bool
	var ttEntry TTEntry
	if entry, found := info.TT.Probe(b.HashKey); found {
		ttHit = true
		ttEntry = entry
		ttMove = entry.Move
	}

	bestMove := NoMove
	bestScore := -Infinity
	moveCount := 0

	// Track quiet and capture moves for history penalty on beta cutoff
	var quietsTried [64]Move
	quietsCount := 0
	var capturesTried [32]Move
	capturesCount := 0

	for ri := range info.RootMoves {
		move := info.RootMoves[ri].Move

		if atomic.LoadInt32(&info.Stopped) != 0 {
			return 0
		}

		moveCount++
		isCap := isCapture(move, b)
		movedPiece := b.Squares[move.From()]

		// Singular extension at root
		singularExtension := 0
		if SingularExtEnabled &&
			move == ttMove &&
			ttMove != NoMove &&
			depth >= 10 &&
			!inCheck &&
			info.ExcludedMove[ply] == NoMove &&
			ttHit &&
			ttEntry.Flag != TTUpper &&
			int(ttEntry.Depth) >= depth-3 {

			ttScore := int(ttEntry.Score)
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
					if DoubleSingularExtEnabled && singularScore < singularBeta-depth*3 {
						singularExtension = 2
						info.DoubleSingularExtensions++
					}
				} else if NegativeSingularExtEnabled && singularScore >= ttScore+depth*3 {
					singularExtension = -1
					info.NegativeSingularExtensions++
				}
			}
		}

		b.MakeMove(move)

		// Check extension
		var givesCheck bool
		flags := move.Flags()
		if flags == FlagEnPassant || flags == FlagCastle {
			givesCheck = b.InCheck()
		} else {
			pt := int(movedPiece)
			if pt > 6 {
				pt -= 6
			}
			if move.IsPromotion() {
				pt = flags - FlagPromoteN + 2
			}
			givesCheck = checkSq[pt]&SquareBB(move.To()) != 0 ||
				discoverers&SquareBB(move.From()) != 0
		}

		extension := 0
		if givesCheck {
			extension = 1
		}
		if singularExtension != 0 && extension == 0 {
			extension = singularExtension
		}

		// Recapture extension
		if RecaptureExtEnabled && extension == 0 && isCap && len(b.UndoStack) >= 2 {
			prevUndo := b.UndoStack[len(b.UndoStack)-2]
			if prevUndo.Captured != Empty && move.To() == prevUndo.Move.To() {
				extension = 1
				info.RecaptureExtensions++
			}
		}

		// Passed pawn push extension
		if PassedPawnExtEnabled && extension == 0 && !isCap {
			mp := b.Squares[move.To()]
			moverColor := b.SideToMove ^ 1
			if mp == pieceOf(WhitePawn, moverColor) {
				rank := move.To().Rank()
				relRank := rank
				if moverColor == Black {
					relRank = 7 - rank
				}
				if relRank >= 5 {
					enemyPawns := b.Pieces[pieceOf(WhitePawn, moverColor^1)]
					if PassedPawnMask[moverColor][move.To()]&enemyPawns == 0 {
						extension = 1
						info.PassedPawnExtensions++
					}
				}
			}
		}

		newDepth := depth - 1 + extension

		var score int

		// Track quiet and capture moves for history penalty
		if !isCap && !move.IsPromotion() && quietsCount < len(quietsTried) {
			quietsTried[quietsCount] = move
			quietsCount++
		}
		if isCap && capturesCount < len(capturesTried) {
			capturesTried[capturesCount] = move
			capturesCount++
		}

		// LMR + PVS at root
		isKiller := false // no killers at root
		reduction := 0
		if LMREnabled && !inCheck && !isCap && !move.IsPromotion() && !isKiller && !givesCheck && moveCount > 1 {
			d, m := depth, moveCount
			if d >= 64 {
				d = 63
			}
			if m >= 64 {
				m = 63
			}
			reduction = lmrTable[d][m]
			if reduction > 0 {
				// Reduce less at PV nodes
				if beta-alpha > 1 {
					reduction--
				}
				// History adjustment
				histScore := info.History[move.From()][move.To()]
				reduction -= int(histScore / 5000)
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
			score = -b.negamax(newDepth-reduction, ply+1, -alpha-1, -alpha, info)
			if score > alpha && atomic.LoadInt32(&info.Stopped) == 0 {
				info.LMRReSearches++
				score = -b.negamax(newDepth, ply+1, -alpha-1, -alpha, info)
			} else {
				info.LMRSavings++
			}
			if score > alpha && score < beta && atomic.LoadInt32(&info.Stopped) == 0 {
				score = -b.negamax(newDepth, ply+1, -beta, -alpha, info)
			}
		} else if moveCount > 1 {
			// PVS: zero-window for non-first moves
			score = -b.negamax(newDepth, ply+1, -alpha-1, -alpha, info)
			if score > alpha && score < beta && atomic.LoadInt32(&info.Stopped) == 0 {
				score = -b.negamax(newDepth, ply+1, -beta, -alpha, info)
			}
		} else {
			// First move: full window
			score = -b.negamax(newDepth, ply+1, -beta, -alpha, info)
		}

		b.UnmakeMove(move)

		if atomic.LoadInt32(&info.Stopped) != 0 {
			return 0
		}

		// Update root move score for sorting on next iteration
		info.RootMoves[ri].Score = score

		if score > bestScore {
			bestScore = score
			bestMove = move

			if score > alpha {
				alpha = score

				// Update PV
				info.pvTable[ply][0] = move
				copy(info.pvTable[ply][1:], info.pvTable[ply+1][:info.pvLen[ply+1]])
				info.pvLen[ply] = 1 + info.pvLen[ply+1]

				if alpha >= beta {
					// Beta cutoff at root — full history updates
					if !isCap {
						bonus := historyBonus(depth)
						info.updateHistory(move.From(), move.To(), bonus)

						// Penalize quiet moves tried before the cutoff move
						for i := 0; i < quietsCount-1; i++ {
							q := quietsTried[i]
							info.updateHistory(q.From(), q.To(), -bonus)
						}
					} else {
						// Capture caused beta cutoff
						bonus := historyBonus(depth)
						piece := b.Squares[move.From()]
						cpt := capturedType(b.Squares[move.To()])
						if move.Flags() == FlagEnPassant {
							cpt = 1
						}
						info.updateCaptHistory(piece, move.To(), cpt, bonus)

						// Penalize captures tried before cutoff
						for i := 0; i < capturesCount-1; i++ {
							c := capturesTried[i]
							cp := b.Squares[c.From()]
							ct := capturedType(b.Squares[c.To()])
							if c.Flags() == FlagEnPassant {
								ct = 1
							}
							info.updateCaptHistory(cp, c.To(), ct, -bonus)
						}
					}
					break
				}
			}
		}
	}

	// Store in TT
	if info.ExcludedMove[ply] == NoMove {
		var flag TTFlag
		if bestScore <= alphaOrig {
			flag = TTUpper
		} else if bestScore >= beta {
			flag = TTLower
		} else {
			flag = TTExact
		}
		info.TT.Store(b.HashKey, depth, bestScore, flag, bestMove, 0)
	}

	return bestScore
}

// quiescence searches captures until the position is quiet
func (b *Board) quiescence(alpha, beta, ply int, info *SearchInfo) int {
	return b.quiescenceWithDepth(alpha, beta, ply, info, 0)
}

// quiescenceWithDepth is the internal quiescence search with depth tracking.
// Uses fail-soft: returns the actual best score (not clamped to [alpha, beta]),
// which is required for correct TT interaction.
func (b *Board) quiescenceWithDepth(alpha, beta, ply int, info *SearchInfo, qsDepth int) int {
	// Limit quiescence depth to prevent stack overflow
	if qsDepth >= 32 {
		return b.EvaluateRelative()
	}

	info.Nodes++

	// Check time periodically
	if info.Nodes&1023 == 0 {
		if d := atomic.LoadInt64(&info.Deadline); d > 0 && time.Now().UnixNano() >= d {
			atomic.StoreInt32(&info.Stopped, 1)
			return 0
		}
	}

	if atomic.LoadInt32(&info.Stopped) != 0 {
		return 0
	}

	// Probe transposition table
	ttMove := NoMove
	alphaOrig := alpha
	var ttHit bool
	var ttStaticEval int16

	if entry, found := info.TT.Probe(b.HashKey); found {
		ttHit = true
		ttMove = entry.Move
		ttStaticEval = entry.StaticEval

		if int(entry.Depth) >= -1 {
			score := int(entry.Score)
			// Adjust mate scores for distance from root
			if score > MateScore-100 {
				score -= ply
			} else if score < -MateScore+100 {
				score += ply
			}

			switch entry.Flag {
			case TTExact:
				return score
			case TTLower:
				if score >= beta {
					return score
				}
			case TTUpper:
				if score <= alpha {
					return score
				}
			}
		}
	}

	// Precompute pinned pieces and checkers
	pinned, checkers := b.PinnedAndCheckers(b.SideToMove)
	qsInCheck := checkers != 0

	qsIdx := MaxPly + qsDepth

	// When in check, generate all evasion moves (captures + blocks + king moves)
	if qsInCheck {
		info.pickers[qsIdx].InitEvasion(b, ttMove, 0, checkers, pinned, nil, nil, &info.CaptHistory)
		picker := &info.pickers[qsIdx]
		bestScore := -Infinity
		bestMove := NoMove
		moveCount := 0

		for {
			move := picker.Next()
			if move == NoMove {
				break
			}
			moveCount++

			b.MakeMove(move)
			score := -b.quiescenceWithDepth(-beta, -alpha, ply+1, info, qsDepth+1)
			b.UnmakeMove(move)

			if score > bestScore {
				bestScore = score
				bestMove = move
			}
			if score > alpha {
				alpha = score
				if score >= beta {
					break
				}
			}
		}

		// Checkmate detection
		if moveCount == 0 {
			return -MateScore + ply
		}

		// Store in TT
		storeScore := bestScore
		if storeScore > MateScore-100 {
			storeScore += ply
		} else if storeScore < -MateScore+100 {
			storeScore -= ply
		}
		var flag TTFlag
		if bestScore >= beta {
			flag = TTLower
		} else if bestScore <= alphaOrig {
			flag = TTUpper
		} else {
			flag = TTExact
		}
		info.TT.Store(b.HashKey, -1, storeScore, flag, bestMove, -Infinity)
		return bestScore
	}

	// Stand pat - evaluate the current position (only when not in check).
	// Use TT staticEval when available to avoid recomputing.
	var standPat int
	if ttHit && ttStaticEval > int16(-MateScore+100) {
		standPat = int(ttStaticEval)
	} else {
		standPat = b.EvaluateRelative()
	}
	bestScore := standPat

	if bestScore >= beta {
		return bestScore
	}

	if bestScore > alpha {
		alpha = bestScore
	}

	// Use MovePicker for captures only (reuse pre-allocated picker)
	info.pickers[qsIdx].InitQuiescence(b, &info.CaptHistory)
	picker := &info.pickers[qsIdx]
	bestMove := NoMove

	for {
		move := picker.Next()
		if move == NoMove {
			break
		}

		// Delta pruning: skip captures that can't possibly raise alpha
		// even with the maximum material gain
		if DeltaPruningEnabled && !move.IsPromotion() {
			capturedPiece := b.Squares[move.To()]
			if move.Flags() == FlagEnPassant {
				capturedPiece = pieceOf(WhitePawn, 1-b.SideToMove)
			}
			if capturedPiece != Empty && standPat+SEEPieceValues[capturedPiece]+200 <= alpha {
				continue
			}
		}

		// Skip bad captures (SEE < 0)
		if !b.SEESign(move, 0) {
			continue
		}

		// Check legality
		if !b.IsLegal(move, pinned, false) {
			continue
		}

		b.MakeMove(move)
		score := -b.quiescenceWithDepth(-beta, -alpha, ply+1, info, qsDepth+1)
		b.UnmakeMove(move)

		if score > bestScore {
			bestScore = score
			bestMove = move
		}
		if score > alpha {
			alpha = score
			if score >= beta {
				break
			}
		}
	}

	// Store in TT
	storeScore := bestScore
	if storeScore > MateScore-100 {
		storeScore += ply
	} else if storeScore < -MateScore+100 {
		storeScore -= ply
	}
	var flag TTFlag
	if bestScore >= beta {
		flag = TTLower
	} else if bestScore <= alphaOrig {
		flag = TTUpper
	} else {
		flag = TTExact
	}
	info.TT.Store(b.HashKey, -1, storeScore, flag, bestMove, standPat)
	return bestScore
}

// isCapture returns true if the move is a capture
func isCapture(m Move, b *Board) bool {
	return b.Squares[m.To()] != Empty || m.Flags() == FlagEnPassant
}

// capturedType returns the piece type (1-6) of a captured piece, color-normalized.
// For white pieces (1-6), returns as-is. For black pieces (7-12), subtracts 6.
func capturedType(p Piece) int {
	if p >= BlackPawn {
		return int(p - 6)
	}
	return int(p)
}

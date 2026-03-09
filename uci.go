package chess

import (
	"bufio"
	"chess/syzygy"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// goParams holds parsed "go" command parameters.
type goParams struct {
	depth, movetime, wtime, btime, winc, binc, movestogo int
	infinite, ponder                                     bool
}

// UCIEngine implements the UCI protocol for the chess engine.
type UCIEngine struct {
	board        Board
	tt           *TranspositionTable
	hashSizeMB   int
	threads      int // number of search threads (Lazy SMP)
	moveOverhead int // ms safety margin for UCI communication + OS scheduling
	searchInfo   *SearchInfo
	searchMu     sync.Mutex
	searchWg     sync.WaitGroup
	input        *bufio.Scanner
	output       io.Writer
	book         *OpeningBook
	pondering      bool
	ponderDone     chan struct{}
	ponderParams   goParams
	ponderSide     Color
	ponderOpt      bool
	ponderBookMove string // set by cmdPonderhit when book has a move
	nnueNet        *NNUENet // loaded NNUE network (nil when not loaded)
}

// NewUCIEngine creates a UCI engine reading from stdin and writing to stdout.
func NewUCIEngine() *UCIEngine {
	return NewUCIEngineWithIO(os.Stdin, os.Stdout)
}

// NewUCIEngineWithIO creates a UCI engine with custom I/O (useful for testing).
func NewUCIEngineWithIO(in io.Reader, out io.Writer) *UCIEngine {
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 64*1024), 64*1024)
	e := &UCIEngine{
		hashSizeMB:   64,
		threads:      1,
		moveOverhead: 50,
		input:        scanner,
		output:       out,
		ponderOpt:    true,
	}
	e.tt = NewTranspositionTable(e.hashSizeMB)
	e.board.Reset()
	return e
}

// Run enters the UCI command loop, reading and dispatching commands until "quit".
func (e *UCIEngine) Run() {
	defer e.cmdStop()
	for e.input.Scan() {
		line := strings.TrimSpace(e.input.Text())
		if line == "" {
			continue
		}
		tokens := strings.Fields(line)
		switch tokens[0] {
		case "uci":
			e.cmdUCI()
		case "isready":
			e.cmdIsReady()
		case "ucinewgame":
			e.cmdNewGame()
		case "position":
			e.cmdPosition(tokens[1:])
		case "go":
			e.cmdGo(tokens[1:])
		case "stop":
			e.cmdStop()
		case "ponderhit":
			e.cmdPonderhit()
		case "setoption":
			e.cmdSetOption(tokens[1:])
		case "quit":
			e.cmdStop()
			return
		case "d":
			e.cmdDebug()
		}
	}
}

func (e *UCIEngine) send(format string, args ...interface{}) {
	fmt.Fprintf(e.output, format+"\n", args...)
}

// SetBook sets the opening book for the engine.
func (e *UCIEngine) SetBook(book *OpeningBook) {
	e.book = book
}

// SetNNUE loads an NNUE network and wires it into the engine's board.
func (e *UCIEngine) SetNNUE(net *NNUENet) {
	e.nnueNet = net
	e.board.NNUENet = net
	if net != nil {
		if e.board.NNUEAcc == nil {
			e.board.NNUEAcc = NewNNUEAccumulatorStack(512)
		}
		e.board.NNUENet.RecomputeAccumulator(e.board.NNUEAcc.Current(), &e.board)
		e.send("info string SetNNUE fingerprint %s", net.Fingerprint())
	} else {
		e.board.NNUEAcc = nil
	}
}

func (e *UCIEngine) cmdUCI() {
	e.send("id name GoChess")
	e.send("id author Adam")
	e.send("option name Hash type spin default 64 min 1 max 4096")
	e.send("option name Threads type spin default 1 min 1 max 256")
	e.send("option name Ponder type check default true")
	e.send("option name MoveOverhead type spin default 50 min 0 max 1000")
	e.send("option name OwnBook type check default true")
	e.send("option name BookFile type string default <empty>")
	e.send("option name UseNNUE type check default true")
	e.send("option name NNUEFile type string default <empty>")
	e.send("option name SyzygyPath type string default <empty>")
	e.send("option name SyzygyProbeDepth type spin default 1 min 1 max 100")
	e.send("uciok")
}

func (e *UCIEngine) cmdIsReady() {
	e.send("readyok")
}

func (e *UCIEngine) cmdNewGame() {
	e.cmdStop()
	e.tt.Clear()
	e.board.Reset()
	// Recompute NNUE accumulator after board reset
	if e.nnueNet != nil && e.board.NNUENet != nil && e.board.NNUEAcc != nil {
		e.board.NNUENet.RecomputeAccumulator(e.board.NNUEAcc.Current(), &e.board)
	}
}

func (e *UCIEngine) cmdPosition(tokens []string) {
	if len(tokens) == 0 {
		return
	}

	idx := 0
	if tokens[0] == "startpos" {
		e.board.Reset()
		idx = 1
	} else if tokens[0] == "fen" {
		// Collect FEN fields (up to 6 parts, or until "moves" keyword)
		fenParts := []string{}
		idx = 1
		for idx < len(tokens) && tokens[idx] != "moves" && len(fenParts) < 6 {
			fenParts = append(fenParts, tokens[idx])
			idx++
		}
		fen := strings.Join(fenParts, " ")
		if err := e.board.SetFEN(fen); err != nil {
			return
		}
	} else {
		return
	}

	// Apply moves
	if idx < len(tokens) && tokens[idx] == "moves" {
		idx++
		for idx < len(tokens) {
			move, err := e.board.ParseUCIMove(tokens[idx])
			if err != nil {
				return
			}
			e.board.MakeMove(move)
			idx++
		}
	}
}

// parseGoParams parses the tokens following a "go" command into goParams.
func parseGoParams(tokens []string) goParams {
	p := goParams{depth: 64}
	for i := 0; i < len(tokens); i++ {
		switch tokens[i] {
		case "depth":
			if i+1 < len(tokens) {
				i++
				p.depth, _ = strconv.Atoi(tokens[i])
			}
		case "movetime":
			if i+1 < len(tokens) {
				i++
				p.movetime, _ = strconv.Atoi(tokens[i])
			}
		case "wtime":
			if i+1 < len(tokens) {
				i++
				p.wtime, _ = strconv.Atoi(tokens[i])
			}
		case "btime":
			if i+1 < len(tokens) {
				i++
				p.btime, _ = strconv.Atoi(tokens[i])
			}
		case "winc":
			if i+1 < len(tokens) {
				i++
				p.winc, _ = strconv.Atoi(tokens[i])
			}
		case "binc":
			if i+1 < len(tokens) {
				i++
				p.binc, _ = strconv.Atoi(tokens[i])
			}
		case "movestogo":
			if i+1 < len(tokens) {
				i++
				p.movestogo, _ = strconv.Atoi(tokens[i])
			}
		case "infinite":
			p.infinite = true
		case "ponder":
			p.ponder = true
		}
	}
	return p
}

func (e *UCIEngine) cmdGo(tokens []string) {
	e.cmdStop()

	params := parseGoParams(tokens)

	// Probe opening book before search (skip when pondering)
	if !params.ponder && e.book != nil {
		if move, ok := e.book.PickMove(&e.board); ok {
			e.send("info string book move")
			result := move.String()
			if e.ponderOpt {
				var tmpBoard Board
				tmpBoard = e.board
				tmpBoard.UndoStack = make([]UndoInfo, 0, 8)
				tmpBoard.MakeMove(move)
				if pm, pok := e.book.PickMove(&tmpBoard); pok {
					result += " ponder " + pm.String()
				}
			}
			e.send("bestmove %s", result)
			return
		}
	}

	// Compute time allocation
	var softMS, hardMS int
	if params.ponder {
		softMS, hardMS = 0, 0 // infinite during ponder
	} else {
		softMS, hardMS = computeSearchTime(params.movetime, params.wtime, params.btime,
			params.winc, params.binc, params.movestogo, params.infinite, e.board.SideToMove, e.moveOverhead)
	}

	// Copy board for search goroutine (preserve game history for repetition detection)
	var searchBoard Board
	searchBoard = e.board
	gameHistory := len(e.board.UndoStack)
	searchBoard.UndoStack = make([]UndoInfo, gameHistory, gameHistory+256)
	copy(searchBoard.UndoStack, e.board.UndoStack)
	searchBoard.PawnTable = NewPawnTable(1)
	// Deep copy NNUE accumulator for search thread (net is shared read-only)
	if e.board.NNUEAcc != nil {
		searchBoard.NNUEAcc = e.board.NNUEAcc.DeepCopy()
	}

	e.searchMu.Lock()
	now := time.Now()
	info := &SearchInfo{
		StartTime: now,
		MaxTime:   time.Duration(hardMS) * time.Millisecond,
		TT:        e.tt,
	}
	if hardMS > 0 {
		atomic.StoreInt64(&info.Deadline, now.Add(time.Duration(hardMS)*time.Millisecond).UnixNano())
	}
	if softMS > 0 && softMS != hardMS {
		atomic.StoreInt64(&info.SoftDeadline, now.Add(time.Duration(softMS)*time.Millisecond).UnixNano())
	}
	info.OnDepth = func(d, score int, nodes uint64, pv []Move) {
		elapsed := time.Since(info.StartTime)
		ms := elapsed.Milliseconds()
		if ms == 0 {
			ms = 1
		}
		nps := nodes * 1000 / uint64(ms)
		hashfull := e.tt.Hashfull()

		scoreStr := formatScore(score)

		pvStrs := make([]string, len(pv))
		for i, m := range pv {
			pvStrs[i] = m.String()
		}

		tbhitsStr := ""
		if SyzygyEnabled && info.TBHits > 0 {
			tbhitsStr = fmt.Sprintf(" tbhits %d", info.TBHits)
		}
		e.send("info depth %d score %s nodes %d nps %d time %d hashfull %d%s pv %s",
			d, scoreStr, nodes, nps, ms, hashfull, tbhitsStr, strings.Join(pvStrs, " "))
	}
	e.searchInfo = info
	numThreads := e.threads

	if params.ponder {
		e.pondering = true
		e.ponderDone = make(chan struct{})
		e.ponderParams = params
		e.ponderSide = e.board.SideToMove
	}
	e.searchMu.Unlock()

	e.searchWg.Add(1)
	go func() {
		defer e.searchWg.Done()
		bestMove, searchResult := searchBoard.SearchParallel(params.depth, info, numThreads)

		// If pondering and search finished before ponderhit/stop, wait
		e.searchMu.Lock()
		if e.pondering {
			ponderDone := e.ponderDone
			e.searchMu.Unlock()
			if ponderDone != nil {
				<-ponderDone
			}
		} else {
			e.searchMu.Unlock()
		}

		// Check if ponderhit found a book move
		e.searchMu.Lock()
		bookResult := e.ponderBookMove
		e.ponderBookMove = ""
		e.searchMu.Unlock()

		// Build bestmove string
		var result string
		if bookResult != "" {
			result = bookResult
		} else {
			result = bestMove.String()
			if e.ponderOpt && len(searchResult.PV) >= 2 {
				result += " ponder " + searchResult.PV[1].String()
			}
		}
		e.send("bestmove %s", result)

		e.searchMu.Lock()
		e.searchInfo = nil
		e.searchMu.Unlock()
	}()
}

func (e *UCIEngine) cmdStop() {
	e.searchMu.Lock()
	if e.searchInfo != nil {
		atomic.StoreInt32(&e.searchInfo.Stopped, 1)
		// Stop helper threads (Lazy SMP) — SearchParallel also does this
		// when the main thread returns, but stopping them here too ensures
		// faster shutdown when cmdStop is called externally.
		for _, h := range e.searchInfo.HelperInfos {
			atomic.StoreInt32(&h.Stopped, 1)
		}
	}
	if e.pondering {
		e.pondering = false
		if e.ponderDone != nil {
			close(e.ponderDone)
			e.ponderDone = nil
		}
	}
	e.searchMu.Unlock()
	e.searchWg.Wait()
}

func (e *UCIEngine) cmdPonderhit() {
	e.searchMu.Lock()
	defer e.searchMu.Unlock()
	if !e.pondering || e.searchInfo == nil {
		return
	}

	// Check book now that the ponder position is confirmed real
	if e.book != nil {
		if move, ok := e.book.PickMove(&e.board); ok {
			e.send("info string book move")
			result := move.String()
			if e.ponderOpt {
				var tmpBoard Board
				tmpBoard = e.board
				tmpBoard.UndoStack = make([]UndoInfo, 0, 8)
				tmpBoard.MakeMove(move)
				if pm, pok := e.book.PickMove(&tmpBoard); pok {
					result += " ponder " + pm.String()
				}
			}
			e.ponderBookMove = result
			// Stop the running search; goroutine will use ponderBookMove
			atomic.StoreInt32(&e.searchInfo.Stopped, 1)
			e.pondering = false
			if e.ponderDone != nil {
				close(e.ponderDone)
				e.ponderDone = nil
			}
			return
		}
	}

	p := e.ponderParams
	softMS, hardMS := computeSearchTime(p.movetime, p.wtime, p.btime,
		p.winc, p.binc, p.movestogo, p.infinite, e.ponderSide, e.moveOverhead)
	phNow := time.Now()
	if hardMS > 0 {
		atomic.StoreInt64(&e.searchInfo.Deadline, phNow.Add(time.Duration(hardMS)*time.Millisecond).UnixNano())
	}
	if softMS > 0 && softMS != hardMS {
		atomic.StoreInt64(&e.searchInfo.SoftDeadline, phNow.Add(time.Duration(softMS)*time.Millisecond).UnixNano())
	}
	e.pondering = false
	if e.ponderDone != nil {
		close(e.ponderDone)
		e.ponderDone = nil
	}
}

func (e *UCIEngine) cmdSetOption(tokens []string) {
	// Parse "name <name> value <value>"
	nameIdx := -1
	valueIdx := -1
	for i, t := range tokens {
		if t == "name" {
			nameIdx = i + 1
		}
		if t == "value" {
			valueIdx = i + 1
		}
	}
	if nameIdx < 0 || valueIdx < 0 || nameIdx >= len(tokens) || valueIdx >= len(tokens) {
		return
	}

	// Collect name tokens (everything between "name" and "value")
	var nameParts []string
	for i := nameIdx; i < len(tokens); i++ {
		if tokens[i] == "value" {
			break
		}
		nameParts = append(nameParts, tokens[i])
	}
	name := strings.Join(nameParts, " ")

	if strings.EqualFold(name, "Hash") {
		mb, err := strconv.Atoi(tokens[valueIdx])
		if err != nil || mb < 1 {
			return
		}
		if mb > 4096 {
			mb = 4096
		}
		e.hashSizeMB = mb
		e.tt = NewTranspositionTable(mb)
	} else if strings.EqualFold(name, "Threads") {
		n, err := strconv.Atoi(tokens[valueIdx])
		if err != nil || n < 1 {
			return
		}
		if n > 256 {
			n = 256
		}
		e.threads = n
	} else if strings.EqualFold(name, "MoveOverhead") {
		n, err := strconv.Atoi(tokens[valueIdx])
		if err != nil || n < 0 {
			return
		}
		if n > 1000 {
			n = 1000
		}
		e.moveOverhead = n
	} else if strings.EqualFold(name, "Ponder") {
		e.ponderOpt = strings.EqualFold(tokens[valueIdx], "true")
	} else if strings.EqualFold(name, "OwnBook") {
		if e.book != nil {
			e.book.SetUseBook(strings.EqualFold(tokens[valueIdx], "true"))
		}
	} else if strings.EqualFold(name, "BookFile") {
		value := strings.Join(tokens[valueIdx:], " ")
		if value == "" || value == "<empty>" {
			e.book = nil
			return
		}
		book, err := LoadOpeningBook(value)
		if err != nil {
			e.send("info string failed to load book: %v", err)
			return
		}
		e.book = book
		e.send("info string book loaded: %d positions", book.Size())
	} else if strings.EqualFold(name, "UseNNUE") {
		UseNNUE = strings.EqualFold(tokens[valueIdx], "true")
		e.send("info string UseNNUE = %v", UseNNUE)
	} else if strings.EqualFold(name, "NNUEFile") {
		value := strings.Join(tokens[valueIdx:], " ")
		if value == "" || value == "<empty>" {
			e.nnueNet = nil
			e.board.NNUENet = nil
			e.board.NNUEAcc = nil
			return
		}
		net, err := LoadNNUE(value)
		if err != nil {
			e.send("info string failed to load NNUE: %v", err)
			return
		}
		e.nnueNet = net
		e.board.NNUENet = net
		if e.board.NNUEAcc == nil {
			e.board.NNUEAcc = NewNNUEAccumulatorStack(512)
		}
		e.board.NNUENet.RecomputeAccumulator(e.board.NNUEAcc.Current(), &e.board)
		e.send("info string NNUE loaded from %s (fingerprint %s)", value, net.Fingerprint())
	} else if strings.EqualFold(name, "SyzygyPath") {
		value := strings.Join(tokens[valueIdx:], " ")
		if value == "" || value == "<empty>" {
			SyzygyFree()
			e.send("info string Syzygy tablebases disabled")
			return
		}
		if SyzygyInit(value) {
			e.send("info string Syzygy tablebases loaded: up to %d-piece", syzygy.MaxPieceCount)
		} else {
			if !SyzygyCGOAvailable() {
				e.send("info string binary built without CGO, Syzygy tablebases unavailable (install gcc and rebuild)")
			} else {
				e.send("info string failed to load Syzygy tablebases from %s", value)
			}
		}
	} else if strings.EqualFold(name, "SyzygyProbeDepth") {
		n, err := strconv.Atoi(tokens[valueIdx])
		if err != nil || n < 1 {
			return
		}
		SyzygyProbeDepth = n
	}
}

func (e *UCIEngine) cmdDebug() {
	e.send("%s", e.board.Print())
	e.send("FEN: %s", e.board.ToFEN())
}

// computeSearchTime calculates soft and hard time allocations in milliseconds.
// Soft limit is the base allocation where dynamic time management may stop early.
// Hard limit is never exceeded. For movetime mode, soft == hard (no dynamic scaling).
func computeSearchTime(movetime, wtime, btime, winc, binc, movestogo int, infinite bool, side Color, moveOverhead int) (int, int) {
	if infinite {
		return 0, 0
	}
	if movetime > 0 {
		return movetime, movetime // fixed: soft == hard
	}

	var timeLeft, inc int
	if side == White {
		timeLeft = wtime
		inc = winc
	} else {
		timeLeft = btime
		inc = binc
	}

	if timeLeft <= 0 {
		return 0, 0 // no clock info, rely on depth limit
	}

	// Subtract overhead from available time
	timeLeft -= moveOverhead
	if timeLeft < 1 {
		timeLeft = 1
	}

	movesLeft := movestogo
	if movesLeft <= 0 {
		movesLeft = 25 // sudden death default: slightly more aggressive than 30
	}

	// Base allocation: time / movesLeft + most of the increment
	// The increment is guaranteed time back, so use 80% of it
	softAlloc := timeLeft/movesLeft + inc*4/5

	// Cap at half remaining time (before increment consideration)
	maxAlloc := timeLeft / 2
	if softAlloc > maxAlloc {
		softAlloc = maxAlloc
	}

	// Emergency time: when very low on time, be conservative
	// Below 1 second (minus overhead), limit soft to min(timeLeft/10, inc)
	if timeLeft < 1000 {
		emergency := timeLeft / 10
		if inc > 0 && inc < emergency {
			emergency = inc
		}
		if emergency < 10 {
			emergency = 10
		}
		if softAlloc > emergency {
			softAlloc = emergency
		}
	}

	// Floor at 10ms
	if softAlloc < 10 {
		softAlloc = 10
	}

	// Hard limit: allows extending on unstable positions
	var hardAlloc int
	if movestogo > 0 {
		// Tournament TC: tighter hard limits
		hardAlloc = softAlloc * 2
		// Cap by moves remaining: generous early, tight late
		capPct := 20 + movestogo/2
		if capPct > 40 {
			capPct = 40
		}
		mtgCap := timeLeft * capPct / 100
		if hardAlloc > mtgCap {
			hardAlloc = mtgCap
		}
	} else {
		// Sudden death: allow up to 3x soft
		hardAlloc = softAlloc * 3
	}

	// Absolute hard cap: never use more than timeLeft/5 + inc
	// This ensures we always keep a buffer of ~80% remaining time
	maxHard := timeLeft/5 + inc
	if maxHard > timeLeft*3/4 {
		maxHard = timeLeft * 3 / 4
	}
	if hardAlloc > maxHard {
		hardAlloc = maxHard
	}

	// Hard must be >= soft
	if hardAlloc < softAlloc {
		hardAlloc = softAlloc
	}

	return softAlloc, hardAlloc
}

// formatScore formats a score for UCI info output.
func formatScore(score int) string {
	if score > MateScore-100 {
		plies := MateScore - score
		moves := (plies + 1) / 2
		return fmt.Sprintf("mate %d", moves)
	}
	if score < -MateScore+100 {
		plies := MateScore + score
		moves := (plies + 1) / 2
		return fmt.Sprintf("mate -%d", moves)
	}
	return fmt.Sprintf("cp %d", score)
}

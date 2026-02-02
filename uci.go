package chess

import (
	"bufio"
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
		hashSizeMB: 64,
		threads:    1,
		input:      scanner,
		output:     out,
		ponderOpt:  true,
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

func (e *UCIEngine) cmdUCI() {
	e.send("id name GoChess")
	e.send("id author Adam")
	e.send("option name Hash type spin default 64 min 1 max 4096")
	e.send("option name Threads type spin default 1 min 1 max 256")
	e.send("option name Ponder type check default true")
	e.send("option name OwnBook type check default true")
	e.send("option name BookFile type string default <empty>")
	e.send("uciok")
}

func (e *UCIEngine) cmdIsReady() {
	e.send("readyok")
}

func (e *UCIEngine) cmdNewGame() {
	e.cmdStop()
	e.tt.Clear()
	e.board.Reset()
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
		if move, name, ok := e.book.PickMove(e.board.HashKey); ok {
			if name != "" {
				e.send("info string book: %s", name)
			}
			result := move.String()
			if e.ponderOpt {
				var tmpBoard Board
				tmpBoard = e.board
				tmpBoard.UndoStack = make([]UndoInfo, 0, 8)
				tmpBoard.MakeMove(move)
				if pm, _, pok := e.book.PickMove(tmpBoard.HashKey); pok {
					result += " ponder " + pm.String()
				}
			}
			e.send("bestmove %s", result)
			return
		}
	}

	// Compute time allocation
	var allocMS int
	if params.ponder {
		allocMS = 0 // infinite during ponder
	} else {
		allocMS = computeSearchTime(params.movetime, params.wtime, params.btime,
			params.winc, params.binc, params.movestogo, params.infinite, e.board.SideToMove)
	}

	// Copy board for search goroutine (preserve game history for repetition detection)
	var searchBoard Board
	searchBoard = e.board
	gameHistory := len(e.board.UndoStack)
	searchBoard.UndoStack = make([]UndoInfo, gameHistory, gameHistory+256)
	copy(searchBoard.UndoStack, e.board.UndoStack)
	searchBoard.PawnTable = NewPawnTable(1)

	e.searchMu.Lock()
	now := time.Now()
	info := &SearchInfo{
		StartTime: now,
		MaxTime:   time.Duration(allocMS) * time.Millisecond,
		TT:        e.tt,
	}
	if allocMS > 0 {
		atomic.StoreInt64(&info.Deadline, now.Add(info.MaxTime).UnixNano())
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

		e.send("info depth %d score %s nodes %d nps %d time %d hashfull %d pv %s",
			d, scoreStr, nodes, nps, ms, hashfull, strings.Join(pvStrs, " "))
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
		if move, name, ok := e.book.PickMove(e.board.HashKey); ok {
			if name != "" {
				e.send("info string book: %s", name)
			}
			result := move.String()
			if e.ponderOpt {
				var tmpBoard Board
				tmpBoard = e.board
				tmpBoard.UndoStack = make([]UndoInfo, 0, 8)
				tmpBoard.MakeMove(move)
				if pm, _, pok := e.book.PickMove(tmpBoard.HashKey); pok {
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
	allocMS := computeSearchTime(p.movetime, p.wtime, p.btime,
		p.winc, p.binc, p.movestogo, p.infinite, e.ponderSide)
	if allocMS > 0 {
		atomic.StoreInt64(&e.searchInfo.Deadline, time.Now().Add(time.Duration(allocMS)*time.Millisecond).UnixNano())
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
	}
}

func (e *UCIEngine) cmdDebug() {
	e.send("%s", e.board.Print())
	e.send("FEN: %s", e.board.ToFEN())
}

// computeSearchTime calculates the time allocation in milliseconds for a search.
func computeSearchTime(movetime, wtime, btime, winc, binc, movestogo int, infinite bool, side Color) int {
	if infinite {
		return 0
	}
	if movetime > 0 {
		return movetime
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
		return 0 // no clock info, rely on depth limit
	}

	movesLeft := movestogo
	if movesLeft <= 0 {
		movesLeft = 30 // sudden death default
	}

	alloc := timeLeft/movesLeft + inc*3/4

	// Cap at half remaining time
	maxAlloc := timeLeft / 2
	if alloc > maxAlloc {
		alloc = maxAlloc
	}

	// Floor at 10ms
	if alloc < 10 {
		alloc = 10
	}

	return alloc
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

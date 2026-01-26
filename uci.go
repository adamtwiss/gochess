package chess

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// UCIEngine implements the UCI protocol for the chess engine.
type UCIEngine struct {
	board      Board
	tt         *TranspositionTable
	hashSizeMB int
	searchInfo *SearchInfo
	searchMu   sync.Mutex
	searchWg   sync.WaitGroup
	input      *bufio.Scanner
	output     io.Writer
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
		input:      scanner,
		output:     out,
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

func (e *UCIEngine) cmdUCI() {
	e.send("id name GoChess")
	e.send("id author Adam")
	e.send("option name Hash type spin default 64 min 1 max 4096")
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

func (e *UCIEngine) cmdGo(tokens []string) {
	e.cmdStop()

	// Parse go parameters
	depth := 64
	var movetime, wtime, btime, winc, binc, movestogo int
	infinite := false

	for i := 0; i < len(tokens); i++ {
		switch tokens[i] {
		case "depth":
			if i+1 < len(tokens) {
				i++
				depth, _ = strconv.Atoi(tokens[i])
			}
		case "movetime":
			if i+1 < len(tokens) {
				i++
				movetime, _ = strconv.Atoi(tokens[i])
			}
		case "wtime":
			if i+1 < len(tokens) {
				i++
				wtime, _ = strconv.Atoi(tokens[i])
			}
		case "btime":
			if i+1 < len(tokens) {
				i++
				btime, _ = strconv.Atoi(tokens[i])
			}
		case "winc":
			if i+1 < len(tokens) {
				i++
				winc, _ = strconv.Atoi(tokens[i])
			}
		case "binc":
			if i+1 < len(tokens) {
				i++
				binc, _ = strconv.Atoi(tokens[i])
			}
		case "movestogo":
			if i+1 < len(tokens) {
				i++
				movestogo, _ = strconv.Atoi(tokens[i])
			}
		case "infinite":
			infinite = true
		}
	}

	// Compute time allocation
	allocMS := computeSearchTime(movetime, wtime, btime, winc, binc, movestogo, infinite, e.board.SideToMove)

	// Copy board for search goroutine (fresh UndoStack)
	var searchBoard Board
	searchBoard = e.board
	searchBoard.UndoStack = make([]UndoInfo, 0, 256)
	searchBoard.PawnTable = NewPawnTable(1)

	e.searchMu.Lock()
	info := &SearchInfo{
		StartTime: time.Now(),
		MaxTime:   time.Duration(allocMS) * time.Millisecond,
		TT:        e.tt,
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
	e.searchMu.Unlock()

	e.searchWg.Add(1)
	go func() {
		defer e.searchWg.Done()
		bestMove, _ := searchBoard.SearchWithInfo(depth, info)
		e.send("bestmove %s", bestMove.String())
		e.searchMu.Lock()
		e.searchInfo = nil
		e.searchMu.Unlock()
	}()
}

func (e *UCIEngine) cmdStop() {
	e.searchMu.Lock()
	if e.searchInfo != nil {
		e.searchInfo.Stopped = true
	}
	e.searchMu.Unlock()
	e.searchWg.Wait()
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

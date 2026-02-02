package chess

import (
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/peterh/liner"
)

// CLIEngine provides an interactive command-line interface for debugging
// positions, running searches, and evaluating the board state.
type CLIEngine struct {
	board      Board
	tt         *TranspositionTable
	hashSizeMB int
	line       *liner.State
	histFile   string
}

// NewCLIEngine creates a new interactive CLI engine.
func NewCLIEngine() *CLIEngine {
	home, _ := os.UserHomeDir()
	return &CLIEngine{
		hashSizeMB: 64,
		histFile:   filepath.Join(home, ".chess_history"),
	}
}

// Run starts the interactive CLI loop.
func (c *CLIEngine) Run() {
	c.board.Reset()
	c.tt = NewTranspositionTable(c.hashSizeMB)

	c.line = liner.NewLiner()
	defer c.line.Close()
	c.line.SetCtrlCAborts(true)

	// Load history
	if f, err := os.Open(c.histFile); err == nil {
		c.line.ReadHistory(f)
		f.Close()
	}

	fmt.Println("Chess engine interactive mode. Type 'help' for commands.")
	fmt.Println()

	for {
		input, err := c.line.Prompt("chess> ")
		if err != nil {
			// Ctrl+C or EOF
			break
		}

		input = strings.TrimSpace(input)
		if input == "" {
			continue
		}
		c.line.AppendHistory(input)

		tokens := strings.Fields(input)
		cmd := strings.ToLower(tokens[0])

		switch cmd {
		case "help":
			c.cmdHelp()
		case "quit", "exit":
			c.saveHistory()
			return
		case "set":
			c.cmdSet(tokens[1:])
		case "fen":
			c.cmdFEN()
		case "board", "d":
			c.cmdBoard()
		case "reset":
			c.cmdReset()
		case "eval":
			c.cmdEval()
		case "moves":
			c.cmdMoves()
		case "search":
			c.cmdSearch(tokens[1:])
		case "epd":
			c.cmdEPD(input[len(tokens[0]):])
		case "perft":
			c.cmdPerft(tokens[1:])
		case "uci":
			c.saveHistory()
			c.switchToUCI()
			return
		default:
			fmt.Printf("Unknown command: %s (type 'help' for commands)\n", cmd)
		}
	}

	c.saveHistory()
}

func (c *CLIEngine) saveHistory() {
	if f, err := os.Create(c.histFile); err == nil {
		c.line.WriteHistory(f)
		f.Close()
	}
}

func (c *CLIEngine) cmdHelp() {
	fmt.Println("Commands:")
	fmt.Println("  set <FEN>         Set position from FEN string")
	fmt.Println("  reset             Reset to starting position")
	fmt.Println("  board, d          Display the board and FEN")
	fmt.Println("  fen               Print current FEN")
	fmt.Println("  eval              Print evaluation breakdown")
	fmt.Println("  moves             List legal moves")
	fmt.Println("  search <depth>    Search to fixed depth (Ctrl+C to stop)")
	fmt.Println("  epd <EPD>         Solve an EPD position (Ctrl+C to stop)")
	fmt.Println("  perft <depth>     Run perft to given depth")
	fmt.Println("  uci               Switch to UCI protocol mode")
	fmt.Println("  help              Show this help")
	fmt.Println("  quit, exit        Exit")
}

func (c *CLIEngine) cmdSet(args []string) {
	if len(args) == 0 {
		fmt.Println("Usage: set <FEN>")
		return
	}
	fen := strings.Join(args, " ")
	if err := c.board.SetFEN(fen); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Print(c.board.Print())
	fmt.Printf("FEN: %s\n", c.board.ToFEN())
}

func (c *CLIEngine) cmdFEN() {
	fmt.Println(c.board.ToFEN())
}

func (c *CLIEngine) cmdBoard() {
	fmt.Print(c.board.Print())
	fmt.Printf("FEN: %s\n", c.board.ToFEN())
}

func (c *CLIEngine) cmdReset() {
	c.board.Reset()
	c.tt.Clear()
	fmt.Println("Board reset to starting position.")
	fmt.Print(c.board.Print())
}

func (c *CLIEngine) cmdMoves() {
	moves := c.board.GenerateLegalMoves()
	if len(moves) == 0 {
		if c.board.InCheck() {
			fmt.Println("Checkmate!")
		} else {
			fmt.Println("Stalemate!")
		}
		return
	}
	for i, m := range moves {
		san := c.board.ToSAN(m)
		if i > 0 && i%10 == 0 {
			fmt.Println()
		} else if i > 0 {
			fmt.Print("  ")
		}
		fmt.Print(san)
	}
	fmt.Printf("\n(%d legal moves)\n", len(moves))
}

func (c *CLIEngine) cmdSearch(args []string) {
	maxDepth := 20
	if len(args) > 0 {
		if d, err := strconv.Atoi(args[0]); err == nil && d > 0 {
			maxDepth = d
		}
	}

	// Copy board for search (preserves undo stack for repetition detection)
	var searchBoard Board = c.board
	searchBoard.UndoStack = append([]UndoInfo(nil), c.board.UndoStack...)
	searchBoard.EvalTable = c.board.EvalTable
	searchBoard.PawnTable = c.board.PawnTable

	info := &SearchInfo{
		StartTime: time.Now(),
		TT:        c.tt,
	}
	info.OnDepth = func(depth, score int, nodes uint64, pv []Move) {
		elapsed := time.Since(info.StartTime)
		nps := uint64(0)
		if elapsed > 0 {
			nps = nodes * 1000000000 / uint64(elapsed.Nanoseconds())
		}
		pvStr := pvToSANCopy(&searchBoard, pv)
		fmt.Printf("  depth %2d  score %6d  nodes %10d  nps %8d  pv %s\n",
			depth, score, nodes, nps, pvStr)
	}

	// Set up Ctrl+C handler to interrupt search
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	done := make(chan struct{})
	go func() {
		select {
		case <-sigCh:
			atomic.StoreInt32(&info.Stopped, 1)
		case <-done:
		}
	}()

	fmt.Printf("Searching to depth %d... (Ctrl+C to stop)\n", maxDepth)
	bestMove, _ := searchBoard.SearchWithInfo(maxDepth, info)

	close(done)
	signal.Stop(sigCh)

	elapsed := time.Since(info.StartTime)
	fmt.Printf("  ---\n")
	fmt.Printf("  bestmove: %s  nodes: %d  %s  time: %v\n",
		c.board.ToSAN(bestMove), info.Nodes,
		FormatKNPS(info.Nodes, elapsed),
		elapsed.Round(time.Millisecond))

	// Hash table hit rates
	ttProbes, ttHits, _ := c.tt.Stats()
	var evalProbes, evalHits, pawnProbes, pawnHits uint64
	if searchBoard.EvalTable != nil {
		evalProbes, evalHits = searchBoard.EvalTable.Stats()
	}
	if searchBoard.PawnTable != nil {
		pawnProbes, pawnHits = searchBoard.PawnTable.Stats()
	}
	fmt.Printf("  TT: %s  eval: %s  pawn: %s\n",
		formatHitrate(ttProbes, ttHits),
		formatHitrate(evalProbes, evalHits),
		formatHitrate(pawnProbes, pawnHits))
}

func (c *CLIEngine) cmdEPD(rawArgs string) {
	epdStr := strings.TrimSpace(rawArgs)
	if epdStr == "" {
		fmt.Println("Usage: epd <EPD string>")
		fmt.Println("  Example: epd 2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - bm Qg6")
		return
	}

	epd, err := ParseEPD(epdStr)
	if err != nil {
		fmt.Printf("Error parsing EPD: %v\n", err)
		return
	}

	// Set up the board from EPD FEN to display it
	var displayBoard Board
	fullFEN := epd.FEN + " 0 1"
	if err := displayBoard.SetFEN(fullFEN); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	id := epd.ID
	if id == "" {
		id = "position"
	}

	fmt.Printf("--- %s ---\n", id)
	fmt.Print(displayBoard.Print())
	fmt.Printf("FEN: %s\n", epd.FEN)
	if len(epd.BestMoves) > 0 {
		fmt.Printf("Best moves: %s\n", strings.Join(epd.BestMoves, ", "))
	}
	if len(epd.AvoidMoves) > 0 {
		fmt.Printf("Avoid moves: %s\n", strings.Join(epd.AvoidMoves, ", "))
	}
	fmt.Println()

	c.tt.Clear()

	info := &SearchInfo{
		StartTime: time.Now(),
		TT:        c.tt,
		OnDepth: func(depth, score int, nodes uint64, pv []Move) {
			pvStr := pvToSANCopy(&displayBoard, pv)
			fmt.Printf("  depth %2d  score %6d  nodes %10d  pv %s\n",
				depth, score, nodes, pvStr)
		},
	}

	// Set up Ctrl+C handler
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	done := make(chan struct{})
	go func() {
		select {
		case <-sigCh:
			atomic.StoreInt32(&info.Stopped, 1)
		case <-done:
		}
	}()

	fmt.Println("Searching... (Ctrl+C to stop)")

	maxDepth := MaxPly
	maxTime := 5 * time.Minute
	info.MaxTime = maxTime

	result, err := RunEPDTestWithInfo(epd, maxDepth, maxTime, c.tt, info, 1)

	close(done)
	signal.Stop(sigCh)

	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	elapsed := time.Since(info.StartTime)
	status := "PASS"
	if !result.Passed {
		status = "FAIL"
	}
	expected := strings.Join(epd.BestMoves, "/")
	if len(epd.AvoidMoves) > 0 {
		expected = "avoid " + strings.Join(epd.AvoidMoves, "/")
	}

	solveStr := "-"
	if result.SolveDepth > 0 {
		solveStr = fmt.Sprintf("d%d/%v", result.SolveDepth, result.SolveTime.Round(time.Millisecond))
	}

	fmt.Printf("  ---\n")
	fmt.Printf("  [%s] found: %s  expected: %s  solve: %s\n",
		status, result.BestMoveSAN, expected, solveStr)
	fmt.Printf("  nodes: %d  %s  depth: %d  time: %v\n",
		result.SearchInfo.Nodes, FormatKNPS(result.SearchInfo.Nodes, elapsed),
		result.SearchInfo.Depth, elapsed.Round(time.Millisecond))
}

func (c *CLIEngine) cmdEval() {
	// Ensure pawn table and eval table are initialized
	if c.board.PawnTable == nil {
		c.board.PawnTable = NewPawnTable(1)
	}
	if c.board.EvalTable == nil {
		c.board.EvalTable = NewEvalTable(1)
	}

	fmt.Print(c.board.Print())
	fmt.Printf("FEN: %s\n", c.board.ToFEN())

	phase := c.board.computePhase()
	phaseLabel := "opening"
	if phase > 16 {
		phaseLabel = "endgame"
	} else if phase > 8 {
		phaseLabel = "middlegame"
	}
	stm := "White"
	if c.board.SideToMove == Black {
		stm = "Black"
	}
	fmt.Printf("Phase: %d/%d (%s)  Side to move: %s\n\n", phase, TotalPhase, phaseLabel, stm)

	// PST + Material
	wPstMG, wPstEG := c.board.evaluatePST(White)
	bPstMG, bPstEG := c.board.evaluatePST(Black)

	// Pawn structure
	pawnEntry := c.board.probePawnEval()
	wPawnMG, wPawnEG := int(pawnEntry.WhiteMG), int(pawnEntry.WhiteEG)
	bPawnMG, bPawnEG := int(pawnEntry.BlackMG), int(pawnEntry.BlackEG)

	// Pieces (mobility + positional)
	wPcMG, wPcEG := c.board.evaluatePieces(White, &pawnEntry)
	bPcMG, bPcEG := c.board.evaluatePieces(Black, &pawnEntry)

	// Passed pawns
	wPpMG, wPpEG := c.board.evaluatePassedPawns(White, &pawnEntry)
	bPpMG, bPpEG := c.board.evaluatePassedPawns(Black, &pawnEntry)

	// King safety
	wKsMG, wKsEG := c.board.evaluateKingSafety(White)
	bKsMG, bKsEG := c.board.evaluateKingSafety(Black)

	// Space
	wSpMG, wSpEG := c.board.evaluateSpace(White)
	bSpMG, bSpEG := c.board.evaluateSpace(Black)

	// Threats
	wThMG, wThEG := c.board.evaluateThreats(White)
	bThMG, bThEG := c.board.evaluateThreats(Black)

	// Castling rights (MG only)
	wCastleMG := 0
	bCastleMG := 0
	if c.board.Castling&WhiteKingside != 0 {
		wCastleMG += CastlingRightsMG
	}
	if c.board.Castling&WhiteQueenside != 0 {
		wCastleMG += CastlingRightsMG
	}
	if c.board.Castling&BlackKingside != 0 {
		bCastleMG += CastlingRightsMG
	}
	if c.board.Castling&BlackQueenside != 0 {
		bCastleMG += CastlingRightsMG
	}

	// Print table
	fmt.Printf("%-20s %8s %8s %8s %8s %8s %8s\n",
		"Component", "W MG", "W EG", "B MG", "B EG", "Net MG", "Net EG")
	fmt.Println(strings.Repeat("-", 76))

	printRow := func(name string, wMG, wEG, bMG, bEG int) {
		fmt.Printf("%-20s %8d %8d %8d %8d %8d %8d\n",
			name, wMG, wEG, bMG, bEG, wMG-bMG, wEG-bEG)
	}

	printRow("PST + Material", wPstMG, wPstEG, bPstMG, bPstEG)
	printRow("Pawn Structure", wPawnMG, wPawnEG, bPawnMG, bPawnEG)
	printRow("Pieces/Mobility", wPcMG, wPcEG, bPcMG, bPcEG)
	printRow("Passed Pawns", wPpMG, wPpEG, bPpMG, bPpEG)
	printRow("King Safety", wKsMG, wKsEG, bKsMG, bKsEG)
	printRow("Space", wSpMG, wSpEG, bSpMG, bSpEG)
	printRow("Threats", wThMG, wThEG, bThMG, bThEG)
	printRow("Castling", wCastleMG, 0, bCastleMG, 0)

	fmt.Println(strings.Repeat("-", 76))

	totalMG := (wPstMG - bPstMG) + (wPawnMG - bPawnMG) + (wPcMG - bPcMG) +
		(wPpMG - bPpMG) + (wKsMG - bKsMG) + (wSpMG - bSpMG) +
		(wThMG - bThMG) + (wCastleMG - bCastleMG)
	totalEG := (wPstEG - bPstEG) + (wPawnEG - bPawnEG) + (wPcEG - bPcEG) +
		(wPpEG - bPpEG) + (wKsEG - bKsEG) + (wSpEG - bSpEG) +
		(wThEG - bThEG)

	fmt.Printf("%-20s %8s %8s %8s %8s %8d %8d\n",
		"Total", "", "", "", "", totalMG, totalEG)

	tapered := (totalMG*(TotalPhase-phase) + totalEG*phase) / TotalPhase
	fmt.Printf("\nTapered score: %d cp (White-relative)\n", tapered)

	relative := tapered
	if c.board.SideToMove == Black {
		relative = -relative
	}
	fmt.Printf("Side-to-move relative: %d cp\n", relative)
}

func (c *CLIEngine) cmdPerft(args []string) {
	maxDepth := 5
	if len(args) > 0 {
		if d, err := strconv.Atoi(args[0]); err == nil && d > 0 {
			maxDepth = d
		}
	}

	fmt.Printf("FEN: %s\n", c.board.ToFEN())
	start := time.Now()
	for depth := 1; depth <= maxDepth; depth++ {
		nodes := perft(&c.board, depth)
		elapsed := time.Since(start)
		fmt.Printf("  depth %d: %d nodes (%v)\n", depth, nodes, elapsed.Round(time.Millisecond))
	}
}

func (c *CLIEngine) switchToUCI() {
	// Close liner to restore terminal state
	c.line.Close()

	fmt.Println("Switching to UCI mode...")

	engine := NewUCIEngine()
	// Share the TT
	engine.tt = c.tt
	engine.board = c.board

	// Send UCI identification (the uci command expects this)
	engine.cmdUCI()

	// Enter UCI loop
	engine.Run()
}

// formatHitrate formats a probes/hits pair as "hits/probes (pct%)"
func formatHitrate(probes, hits uint64) string {
	if probes == 0 {
		return "0/0"
	}
	pct := float64(hits) / float64(probes) * 100
	return fmt.Sprintf("%d/%d (%.1f%%)", hits, probes, pct)
}

// pvToSANCopy converts a PV line to SAN notation on a copy of the board.
func pvToSANCopy(b *Board, pv []Move) string {
	var bc Board = *b
	bc.UndoStack = nil
	parts := make([]string, 0, len(pv))
	for _, m := range pv {
		parts = append(parts, bc.ToSAN(m))
		bc.MakeMove(m)
	}
	return strings.Join(parts, " ")
}

// perft counts the number of leaf nodes at a given depth (for verification).
func perft(b *Board, depth int) uint64 {
	if depth == 0 {
		return 1
	}
	moves := b.GenerateLegalMoves()
	if depth == 1 {
		return uint64(len(moves))
	}
	var nodes uint64
	for _, m := range moves {
		b.MakeMove(m)
		nodes += perft(b, depth-1)
		b.UnmakeMove(m)
	}
	return nodes
}

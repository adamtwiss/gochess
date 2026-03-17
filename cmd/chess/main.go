package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"chess"
	"golang.org/x/term"
)

func main() {
	epdFile := flag.String("e", "", "EPD test suite file to run")
	maxTimeMS := flag.Int("t", 5000, "max time per position in milliseconds")
	maxPositions := flag.Int("n", 0, "number of positions to run (0 = all)")
	depth := flag.Int("d", 64, "max search depth")
	hashMB := flag.Int("hash", 64, "transposition table size in MB")
	verbose := flag.Bool("v", false, "verbose output: show board, per-depth search info, and stats")

	// SMP flag
	threads := flag.Int("threads", 1, "number of search threads (Lazy SMP)")

	// Book building flags
	buildBook := flag.Bool("buildbook", false, "build opening book from PGN files")
	bookPGN := flag.String("pgn", "", "PGN file with GM games for book building")
	bookOut := flag.String("bookout", "book.bin", "output file for built book")
	bookDepth := flag.Int("bookdepth", 30, "max full moves to include in book")
	bookMinFreq := flag.Int("bookminfreq", 3, "min frequency to include a move")
	bookTopN := flag.Int("booktop", 8, "max moves per position")

	// Book loading flag
	bookFile := flag.String("book", "", "opening book file for UCI mode")

	// NNUE flags
	nnueFile := flag.String("nnue", "", "NNUE network file (default: net.nnue in current directory)")
	classical := flag.Bool("classical", false, "disable NNUE, use classical eval only")

	// Syzygy tablebase flag
	syzygyPath := flag.String("syzygy", "", "path to Syzygy tablebase files")

	// Benchmark flags
	benchmark := flag.Bool("benchmark", false, "run multi-suite benchmark")
	benchSave := flag.String("save", "", "save benchmark results to JSON file")
	benchCompare := flag.String("compare", "", "compare against saved benchmark JSON file")

	// Mode flags
	forceUCI := flag.Bool("uci", false, "force UCI protocol mode (default when stdin is not a terminal)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: chess [options]\n\nOptions:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  chess                                          # interactive mode\n")
		fmt.Fprintf(os.Stderr, "  chess -uci                                     # UCI mode\n")
		fmt.Fprintf(os.Stderr, "  chess -e testdata/wac.epd -t 5000 -n 20 -threads 4\n")
		fmt.Fprintf(os.Stderr, "  chess -benchmark -t 200                            # run benchmark\n")
		fmt.Fprintf(os.Stderr, "  chess -benchmark -t 200 -save base.json            # save results\n")
		fmt.Fprintf(os.Stderr, "  chess -benchmark -t 200 -compare base.json         # compare\n")
		fmt.Fprintf(os.Stderr, "  chess -buildbook -pgn games.pgn -bookout book.bin\n")
		fmt.Fprintf(os.Stderr, "  chess -book book.bin\n")
		fmt.Fprintf(os.Stderr, "  chess -classical                                   # disable NNUE\n")
		fmt.Fprintf(os.Stderr, "  chess -nnue custom.nnue                            # use specific net\n")
	}
	flag.Parse()

	if *buildBook {
		if *bookPGN == "" {
			fmt.Fprintf(os.Stderr, "Error: -pgn is required for -buildbook\n")
			os.Exit(1)
		}
		opts := chess.BookBuildOptions{
			MaxPly:  *bookDepth * 2,
			MinFreq: *bookMinFreq,
			TopN:    *bookTopN,
		}
		if err := chess.BuildOpeningBook(*bookPGN, *bookOut, opts); err != nil {
			fmt.Fprintf(os.Stderr, "Error building book: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Book written to %s\n", *bookOut)
		return
	}

	// Resolve binary directory for auto-loading net.nnue and book.bin.
	// CWD is unreliable (cutechess-cli sets it to /tmp), so look next to the binary.
	exePath, _ := os.Executable()
	exeDir := filepath.Dir(exePath)

	// Load NNUE network (before any mode branches)
	var nnueNet *chess.NNUENet
	var nnueNetV5 *chess.NNUENetV5
	if *classical {
		// Explicitly disable NNUE
		chess.UseNNUE = false
	} else if *nnueFile != "" {
		// Explicit net path — detect version and load
		version, err := chess.DetectNNUEVersion(*nnueFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading NNUE: %v\n", err)
			os.Exit(1)
		}
		if version == 5 || version == 6 {
			nnueNetV5, err = chess.LoadNNUEV5(*nnueFile)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error loading NNUE v5: %v\n", err)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "NNUE v5 loaded from %s (fingerprint %s)\n", *nnueFile, nnueNetV5.Fingerprint())
		} else {
			nnueNet, err = chess.LoadNNUEAnyVersion(*nnueFile)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error loading NNUE: %v\n", err)
				os.Exit(1)
			}
			chess.GlobalNNUENet = nnueNet
			fmt.Fprintf(os.Stderr, "NNUE loaded from %s\n", *nnueFile)
		}
	} else {
		// Try net.nnue next to the binary, then in CWD
		defaultNet := filepath.Join(exeDir, "net.nnue")
		var err error
		nnueNet, err = chess.LoadNNUEAnyVersion(defaultNet)
		if err != nil {
			// Fall back to CWD (for development convenience)
			nnueNet, err = chess.LoadNNUEAnyVersion("net.nnue")
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: net.nnue not found (tried %s and CWD), NNUE available via UCI NNUEFile option\n", exeDir)
			// Don't set UseNNUE=false — the eval checks b.NNUENet != nil,
			// and a GUI may load a net later via UCI NNUEFile option.
		} else {
			chess.GlobalNNUENet = nnueNet
			fmt.Fprintf(os.Stderr, "NNUE loaded from %s\n", defaultNet)
		}
	}

	// Initialize Syzygy tablebases if specified
	if *syzygyPath != "" {
		if chess.SyzygyInit(*syzygyPath) {
			fmt.Fprintf(os.Stderr, "Syzygy tablebases loaded from %s (up to %d-piece)\n", *syzygyPath, chess.SyzygyMaxPieceCount())
		} else {
			if !chess.SyzygyCGOAvailable() {
				fmt.Fprintf(os.Stderr, "Warning: binary built without CGO, Syzygy tablebases unavailable (install gcc and rebuild)\n")
			} else {
				fmt.Fprintf(os.Stderr, "Warning: failed to load Syzygy tablebases from %s\n", *syzygyPath)
			}
		}
	}

	// Load opening book (before any mode branches)
	var book *chess.OpeningBook
	if *bookFile != "" {
		// Explicit path — must exist
		var err error
		book, err = chess.LoadOpeningBook(*bookFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading book: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "Opening book loaded from %s\n", *bookFile)
	} else {
		// Try book.bin next to the binary, then in CWD
		defaultBook := filepath.Join(exeDir, "book.bin")
		if b, err := chess.LoadOpeningBook(defaultBook); err == nil {
			book = b
			fmt.Fprintf(os.Stderr, "Opening book loaded from %s\n", defaultBook)
		} else if b, err := chess.LoadOpeningBook("book.bin"); err == nil {
			book = b
			fmt.Fprintf(os.Stderr, "Opening book loaded from book.bin\n")
		}
	}

	if *benchmark {
		runBenchmark(*maxTimeMS, *hashMB, *depth, *threads, *benchSave, *benchCompare)
		return
	}

	if *epdFile != "" {
		runEPD(*epdFile, *depth, time.Duration(*maxTimeMS)*time.Millisecond, *maxPositions, *hashMB, *verbose, *threads)
		return
	}

	// If forced UCI or stdin is not a terminal, use UCI mode
	if *forceUCI || !term.IsTerminal(int(os.Stdin.Fd())) {
		engine := chess.NewUCIEngine()
		if book != nil {
			engine.SetBook(book)
		}
		if nnueNetV5 != nil {
			engine.SetNNUEV5(nnueNetV5)
		} else if nnueNet != nil {
			engine.SetNNUE(nnueNet)
		}
		engine.Run()
		return
	}

	// Interactive CLI mode
	cli := chess.NewCLIEngine()
	if nnueNetV5 != nil {
		cli.SetNNUEV5(nnueNetV5)
	} else if nnueNet != nil {
		cli.SetNNUE(nnueNet)
	}
	cli.Run()
}

// formatHitrate formats a probes/hits pair as "hits/probes (pct%)"
func formatHitrate(probes, hits uint64) string {
	if probes == 0 {
		return "0/0"
	}
	pct := float64(hits) / float64(probes) * 100
	return fmt.Sprintf("%d/%d (%.1f%%)", hits, probes, pct)
}

// pvToSAN converts a PV line to SAN notation by replaying moves on a board copy.
func pvToSAN(b *chess.Board, pv []chess.Move) string {
	var bc chess.Board = *b
	bc.UndoStack = nil
	parts := make([]string, 0, len(pv))
	for _, m := range pv {
		parts = append(parts, bc.ToSAN(m))
		bc.MakeMove(m)
	}
	return strings.Join(parts, " ")
}

func runEPD(filename string, depth int, maxTime time.Duration, maxPositions int, hashMB int, verbose bool, threads int) {
	positions, err := chess.LoadEPDFile(filename)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading EPD file: %v\n", err)
		os.Exit(1)
	}

	if maxPositions > 0 && maxPositions < len(positions) {
		positions = positions[:maxPositions]
	}

	tt := chess.NewTranspositionTable(hashMB)

	passed := 0
	failed := 0
	totalNodes := uint64(0)
	nodeScore := 0.0
	timeScore := 0.0
	maxTimeMs := float64(maxTime.Milliseconds())
	suiteStart := time.Now()

	// STS weighted scoring accumulators
	type themeStats struct {
		score    int
		count    int
	}
	themes := make(map[string]*themeStats)
	var themeOrder []string // preserve first-seen order
	hasWeighted := false

	fmt.Printf("EPD Test Suite: %s\n", filename)
	fmt.Printf("Positions: %d, Depth: %d, Time: %v, Hash: %dMB, Threads: %d\n\n",
		len(positions), depth, maxTime, hashMB, threads)

	for i, pos := range positions {
		tt.Clear()

		id := pos.ID
		if id == "" {
			id = fmt.Sprintf("#%d", i+1)
		}

		var result *chess.EPDTestResult
		if verbose {
			result, err = runVerbose(pos, id, depth, maxTime, tt, threads)
		} else {
			result, err = chess.RunEPDTestWithInfo(pos, depth, maxTime, tt, nil, threads)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error on position %d: %v\n", i+1, err)
			continue
		}

		totalNodes += result.SearchInfo.Nodes
		if result.Passed {
			passed++
			if result.SolveNodes > 0 {
				nodeScore += chess.EPDLogScore(float64(result.SearchInfo.Nodes), float64(result.SolveNodes))
			}
			if result.SolveTime > 0 {
				timeScore += chess.EPDLogScore(maxTimeMs, float64(result.SolveTime.Milliseconds()))
			}
		} else {
			failed++
		}

		// Track STS weighted scoring
		if result.HasWeighted {
			hasWeighted = true
			theme := chess.ExtractSTSTheme(pos.ID)
			if theme == "" {
				theme = "(unknown)"
			}
			ts, ok := themes[theme]
			if !ok {
				ts = &themeStats{}
				themes[theme] = ts
				themeOrder = append(themeOrder, theme)
			}
			ts.score += result.WeightedScore
			ts.count++
		}

		expected := strings.Join(pos.BestMoves, "/")
		status := "PASS"
		if !result.Passed {
			status = "FAIL"
		}

		if verbose {
			if result.HasWeighted {
				fmt.Printf("[%s] %s: found %s, expected %s, score=%d\n\n", status, id, result.BestMoveSAN, expected, result.WeightedScore)
			} else {
				fmt.Printf("[%s] %s: found %s, expected %s\n\n", status, id, result.BestMoveSAN, expected)
			}
		} else {
			solveStr := "-"
			if result.SolveDepth > 0 {
				solveStr = fmt.Sprintf("d%d/%v", result.SolveDepth, result.SolveTime.Round(time.Millisecond))
			}
			nps := chess.FormatKNPS(result.SearchInfo.Nodes, result.TimeTaken)
			if result.HasWeighted {
				fmt.Printf("[%s] %-12s found %-8s expected %-12s score=%-4d depth=%-3d solve=%-14s %-14s time=%v\n",
					status, id, result.BestMoveSAN, expected,
					result.WeightedScore, result.SearchInfo.Depth, solveStr, nps, result.TimeTaken.Round(time.Millisecond))
			} else {
				fmt.Printf("[%s] %-12s found %-8s expected %-12s depth=%-3d solve=%-14s %-14s time=%v\n",
					status, id, result.BestMoveSAN, expected,
					result.SearchInfo.Depth, solveStr, nps, result.TimeTaken.Round(time.Millisecond))
			}
		}
	}

	total := passed + failed
	elapsed := time.Since(suiteStart)
	pct := 0.0
	if total > 0 {
		pct = float64(passed) / float64(total) * 100
	}

	// Compute maximum possible log-scores (if every position solved at depth 1)
	maxTimeScore := float64(total) * math.Log2(maxTimeMs)

	fmt.Printf("\n=== SUMMARY ===\n")
	fmt.Printf("Passed: %d/%d (%.1f%%)\n", passed, total, pct)
	fmt.Printf("Failed: %d\n", failed)
	fmt.Printf("Node score: %.1f  (log2 nodes saved by early solve, higher=better)\n", nodeScore)
	fmt.Printf("Time score: %.1f / %.1f  (log2 time saved vs limit, higher=better)\n", timeScore, maxTimeScore)
	fmt.Printf("Total nodes: %d\n", totalNodes)
	fmt.Printf("Average: %s\n", chess.FormatKNPS(totalNodes, elapsed))
	fmt.Printf("Total time: %v\n", elapsed.Round(time.Millisecond))

	// STS per-theme summary
	if hasWeighted {
		fmt.Printf("\n=== THEME SCORES ===\n")
		fmt.Printf("%-50s %6s %6s %5s\n", "Theme", "Score", "Max", "Pct")
		fmt.Println(strings.Repeat("-", 71))

		totalScore := 0
		totalMax := 0
		for _, theme := range themeOrder {
			ts := themes[theme]
			maxScore := ts.count * 100
			pct := 0.0
			if maxScore > 0 {
				pct = float64(ts.score) / float64(maxScore) * 100
			}
			fmt.Printf("%-50s %6d %6d %4.1f%%\n", theme, ts.score, maxScore, pct)
			totalScore += ts.score
			totalMax += maxScore
		}

		fmt.Println(strings.Repeat("-", 71))
		totalPct := 0.0
		if totalMax > 0 {
			totalPct = float64(totalScore) / float64(totalMax) * 100
		}
		fmt.Printf("%-50s %6d %6d %4.1f%%\n", "TOTAL", totalScore, totalMax, totalPct)
	}
}

func runVerbose(pos *chess.EPDPosition, id string, depth int, maxTime time.Duration, tt *chess.TranspositionTable, threads int) (*chess.EPDTestResult, error) {
	var b chess.Board
	fullFEN := pos.FEN + " 0 1"
	if err := b.SetFEN(fullFEN); err != nil {
		return nil, fmt.Errorf("invalid FEN: %w", err)
	}

	fmt.Printf("--- %s ---\n", id)
	fmt.Print(b.Print())
	fmt.Printf("FEN: %s\n", pos.FEN)
	fmt.Println()

	info := &chess.SearchInfo{
		OnDepth: func(d, score int, nodes uint64, pv []chess.Move) {
			pvStr := pvToSAN(&b, pv)
			fmt.Printf("  depth %2d  score %6d  nodes %10d  pv %s\n", d, score, nodes, pvStr)
		},
	}

	result, err := chess.RunEPDTestWithInfo(pos, depth, maxTime, tt, info, threads)
	if err != nil {
		return nil, err
	}

	// Print summary for this position
	nps := chess.FormatKNPS(result.SearchInfo.Nodes, result.TimeTaken)
	solveStr := "-"
	if result.SolveDepth > 0 {
		solveStr = fmt.Sprintf("d%d/%v", result.SolveDepth, result.SolveTime.Round(time.Millisecond))
	}
	fmt.Printf("  ---\n")
	fmt.Printf("  nodes: %d  %s  max depth: %d  solve: %s  time: %v\n",
		result.SearchInfo.Nodes, nps, result.SearchInfo.Depth, solveStr, result.TimeTaken.Round(time.Millisecond))
	fmt.Printf("  TT: %s  pawn: %s\n",
		formatHitrate(result.TTProbes, result.TTHits),
		formatHitrate(result.PawnProbes, result.PawnHits))

	return result, nil
}

func runBenchmark(timeLimitMs, hashMB, depth, threads int, saveFile, compareFile string) {
	suites := chess.DefaultBenchmarkSuites()

	fmt.Printf("=== Chess Engine Benchmark ===\n")
	fmt.Printf("Time: %dms/pos  Hash: %dMB  Threads: %d\n\n", timeLimitMs, hashMB, threads)

	result, err := chess.RunBenchmark(suites, timeLimitMs, hashMB, depth, threads, func(p chess.BenchmarkProgress) {
		status := "FAIL"
		if p.Solved {
			status = "PASS"
		}
		fmt.Printf("  [%s] %s %s %d/%d  score=%.1f\n",
			status, p.Suite, p.ID, p.PositionNum, p.TotalInSuite, p.Score)
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Benchmark error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()

	if compareFile != "" {
		baseline, err := chess.LoadBenchmarkResult(compareFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading baseline: %v\n", err)
			os.Exit(1)
		}
		printBenchmarkComparison(result, baseline)
	} else {
		printBenchmarkResults(result)
	}

	if saveFile != "" {
		if err := chess.SaveBenchmarkResult(saveFile, result); err != nil {
			fmt.Fprintf(os.Stderr, "Error saving results: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("\nResults saved to %s\n", saveFile)
	}
}

func printBenchmarkResults(r *chess.BenchmarkResult) {
	fmt.Printf("%-12s %10s %10s %9s %10s\n", "Suite", "Solved", "Score", "Avg/pos", "kNPS")
	fmt.Println(strings.Repeat("-", 55))

	for _, s := range r.Suites {
		avg := 0.0
		if s.Total > 0 {
			avg = s.Score / float64(s.Total)
		}
		knps := formatKNPSNum(s.TotalNodes, s.DurationMs)
		fmt.Printf("%-12s %5d/%-4d %10.1f %9.2f %10s\n",
			s.Name, s.Solved, s.Total, s.Score, avg, knps)
	}

	fmt.Println(strings.Repeat("-", 55))
	total := r.TotalPositions()
	avg := 0.0
	if total > 0 {
		avg = r.TotalScore() / float64(total)
	}
	knps := formatKNPSNum(r.TotalNodes(), r.TotalDuration())
	fmt.Printf("%-12s %5d/%-4d %10.1f %9.2f %10s\n",
		"TOTAL", r.TotalSolved(), total, r.TotalScore(), avg, knps)
}

func printBenchmarkComparison(current, baseline *chess.BenchmarkResult) {
	fmt.Printf("%-12s %18s %22s %16s %18s\n", "Suite", "Solved", "Score", "Avg/pos", "kNPS")
	fmt.Println(strings.Repeat("-", 90))

	// Build baseline lookup by suite name
	baseMap := make(map[string]*chess.BenchmarkSuiteResult)
	for i := range baseline.Suites {
		baseMap[baseline.Suites[i].Name] = &baseline.Suites[i]
	}

	for _, s := range current.Suites {
		b, ok := baseMap[s.Name]
		if !ok {
			// No baseline for this suite, just print current
			avg := 0.0
			if s.Total > 0 {
				avg = s.Score / float64(s.Total)
			}
			knps := formatKNPSNum(s.TotalNodes, s.DurationMs)
			fmt.Printf("%-12s %5d/%-4d         %10.1f              %6.2f %18s\n",
				s.Name, s.Solved, s.Total, s.Score, avg, knps)
			continue
		}

		curAvg := 0.0
		baseAvg := 0.0
		if s.Total > 0 {
			curAvg = s.Score / float64(s.Total)
		}
		if b.Total > 0 {
			baseAvg = b.Score / float64(b.Total)
		}
		avgDelta := curAvg - baseAvg

		sign := "+"
		if avgDelta < 0 {
			sign = ""
		}

		npsStr := formatKNPSComparison(b.TotalNodes, b.DurationMs, s.TotalNodes, s.DurationMs)

		fmt.Printf("%-12s %3d->%3d/%-4d %8.1f->%7.1f %6.2f->%5.2f (%s%.2f) %s\n",
			s.Name, b.Solved, s.Solved, s.Total,
			b.Score, s.Score,
			baseAvg, curAvg, sign, avgDelta, npsStr)
	}

	fmt.Println(strings.Repeat("-", 90))

	curTotal := current.TotalPositions()
	baseTotal := baseline.TotalPositions()
	curAvg := 0.0
	baseAvg := 0.0
	if curTotal > 0 {
		curAvg = current.TotalScore() / float64(curTotal)
	}
	if baseTotal > 0 {
		baseAvg = baseline.TotalScore() / float64(baseTotal)
	}
	avgDelta := curAvg - baseAvg
	sign := "+"
	if avgDelta < 0 {
		sign = ""
	}

	npsStr := formatKNPSComparison(
		baseline.TotalNodes(), baseline.TotalDuration(),
		current.TotalNodes(), current.TotalDuration())

	fmt.Printf("%-12s %3d->%3d/%-4d %8.1f->%7.1f %6.2f->%5.2f (%s%.2f) %s\n",
		"TOTAL", baseline.TotalSolved(), current.TotalSolved(), curTotal,
		baseline.TotalScore(), current.TotalScore(),
		baseAvg, curAvg, sign, avgDelta, npsStr)
}

func formatKNPSNum(nodes uint64, durationMs float64) string {
	if durationMs <= 0 {
		return "-"
	}
	knps := float64(nodes) / durationMs // nodes/ms = knps
	whole := int(knps + 0.5)
	if whole >= 1000 {
		return fmt.Sprintf("%d,%03d", whole/1000, whole%1000)
	}
	return fmt.Sprintf("%d", whole)
}

func formatKNPSComparison(baseNodes uint64, baseMs float64, curNodes uint64, curMs float64) string {
	if baseMs <= 0 || curMs <= 0 {
		return "-"
	}
	baseKNPS := float64(baseNodes) / baseMs
	curKNPS := float64(curNodes) / curMs
	pctDelta := (curKNPS - baseKNPS) / baseKNPS * 100
	sign := "+"
	if pctDelta < 0 {
		sign = ""
	}
	return fmt.Sprintf("%s->%s (%s%.1f%%)",
		formatKNPSNum(baseNodes, baseMs),
		formatKNPSNum(curNodes, curMs),
		sign, pctDelta)
}

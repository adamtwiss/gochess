package main

import (
	"flag"
	"fmt"
	"math"
	"os"
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
	bookECO := flag.String("eco", "", "ECO PGN file for opening names")
	bookOut := flag.String("bookout", "book.bin", "output file for built book")
	bookDepth := flag.Int("bookdepth", 30, "max full moves to include in book")
	bookMinFreq := flag.Int("bookminfreq", 3, "min frequency to include a move")
	bookTopN := flag.Int("booktop", 8, "max moves per position")

	// Book loading flag
	bookFile := flag.String("book", "", "opening book file for UCI mode")

	// Mode flags
	forceUCI := flag.Bool("uci", false, "force UCI protocol mode (default when stdin is not a terminal)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: chess [options]\n\nOptions:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  chess                                          # interactive mode\n")
		fmt.Fprintf(os.Stderr, "  chess -uci                                     # UCI mode\n")
		fmt.Fprintf(os.Stderr, "  chess -e testdata/wac.epd -t 5000 -n 20 -threads 4\n")
		fmt.Fprintf(os.Stderr, "  chess -buildbook -pgn games.pgn -eco eco.pgn -bookout book.bin\n")
		fmt.Fprintf(os.Stderr, "  chess -book book.bin\n")
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
		if err := chess.BuildOpeningBook(*bookPGN, *bookECO, *bookOut, opts); err != nil {
			fmt.Fprintf(os.Stderr, "Error building book: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Book written to %s\n", *bookOut)
		return
	}

	if *epdFile != "" {
		runEPD(*epdFile, *depth, time.Duration(*maxTimeMS)*time.Millisecond, *maxPositions, *hashMB, *verbose, *threads)
		return
	}

	// If forced UCI or stdin is not a terminal, use UCI mode
	if *forceUCI || !term.IsTerminal(int(os.Stdin.Fd())) {
		engine := chess.NewUCIEngine()
		if *bookFile != "" {
			book, err := chess.LoadOpeningBook(*bookFile)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error loading book: %v\n", err)
				os.Exit(1)
			}
			engine.SetBook(book)
		}
		engine.Run()
		return
	}

	// Interactive CLI mode
	cli := chess.NewCLIEngine()
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

		expected := strings.Join(pos.BestMoves, "/")
		status := "PASS"
		if !result.Passed {
			status = "FAIL"
		}

		if verbose {
			fmt.Printf("[%s] %s: found %s, expected %s\n\n", status, id, result.BestMoveSAN, expected)
		} else {
			solveStr := "-"
			if result.SolveDepth > 0 {
				solveStr = fmt.Sprintf("d%d/%v", result.SolveDepth, result.SolveTime.Round(time.Millisecond))
			}
			nps := chess.FormatKNPS(result.SearchInfo.Nodes, result.TimeTaken)
			fmt.Printf("[%s] %-12s found %-8s expected %-12s depth=%-3d solve=%-14s %-14s time=%v\n",
				status, id, result.BestMoveSAN, expected,
				result.SearchInfo.Depth, solveStr, nps, result.TimeTaken.Round(time.Millisecond))
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
	fmt.Printf("  TT: %s  eval: %s  pawn: %s\n",
		formatHitrate(result.TTProbes, result.TTHits),
		formatHitrate(result.EvalProbes, result.EvalHits),
		formatHitrate(result.PawnProbes, result.PawnHits))

	return result, nil
}

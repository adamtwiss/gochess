package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"chess"
)

func main() {
	epdFile := flag.String("e", "", "EPD test suite file to run")
	maxTimeMS := flag.Int("t", 5000, "max time per position in milliseconds")
	maxPositions := flag.Int("n", 0, "number of positions to run (0 = all)")
	depth := flag.Int("d", 64, "max search depth")
	hashMB := flag.Int("hash", 64, "transposition table size in MB")
	verbose := flag.Bool("v", false, "verbose output: show board, per-depth search info, and stats")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: chess [options]\n\nOptions:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExample:\n  chess -e testdata/wac.epd -t 5000 -n 20\n")
	}
	flag.Parse()

	if *epdFile != "" {
		runEPD(*epdFile, *depth, time.Duration(*maxTimeMS)*time.Millisecond, *maxPositions, *hashMB, *verbose)
		return
	}

	flag.Usage()
	os.Exit(1)
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

func runEPD(filename string, depth int, maxTime time.Duration, maxPositions int, hashMB int, verbose bool) {
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
	suiteStart := time.Now()

	fmt.Printf("EPD Test Suite: %s\n", filename)
	fmt.Printf("Positions: %d, Depth: %d, Time: %v, Hash: %dMB\n\n",
		len(positions), depth, maxTime, hashMB)

	for i, pos := range positions {
		tt.Clear()

		id := pos.ID
		if id == "" {
			id = fmt.Sprintf("#%d", i+1)
		}

		if verbose {
			result, err := runVerbose(pos, id, depth, maxTime, tt)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error on position %d: %v\n", i+1, err)
				continue
			}
			totalNodes += result.SearchInfo.Nodes
			if result.Passed {
				passed++
			} else {
				failed++
			}
			expected := strings.Join(pos.BestMoves, "/")
			status := "PASS"
			if !result.Passed {
				status = "FAIL"
			}
			fmt.Printf("[%s] %s: found %s, expected %s\n\n", status, id, result.BestMoveSAN, expected)
		} else {
			result, err := chess.RunEPDTest(pos, depth, maxTime, tt)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error on position %d: %v\n", i+1, err)
				continue
			}
			totalNodes += result.SearchInfo.Nodes
			if result.Passed {
				passed++
			} else {
				failed++
			}
			expected := strings.Join(pos.BestMoves, "/")
			status := "PASS"
			if !result.Passed {
				status = "FAIL"
			}
			fmt.Printf("[%s] %-12s found %-8s expected %-12s depth=%d nodes=%-10d time=%v\n",
				status, id, result.BestMoveSAN, expected,
				result.SearchInfo.Depth, result.SearchInfo.Nodes, result.TimeTaken.Round(time.Millisecond))
		}
	}

	total := passed + failed
	elapsed := time.Since(suiteStart)
	pct := 0.0
	if total > 0 {
		pct = float64(passed) / float64(total) * 100
	}

	fmt.Printf("\n=== SUMMARY ===\n")
	fmt.Printf("Passed: %d/%d (%.1f%%)\n", passed, total, pct)
	fmt.Printf("Failed: %d\n", failed)
	fmt.Printf("Total nodes: %d\n", totalNodes)
	fmt.Printf("Total time: %v\n", elapsed.Round(time.Millisecond))
}

func runVerbose(pos *chess.EPDPosition, id string, depth int, maxTime time.Duration, tt *chess.TranspositionTable) (*chess.EPDTestResult, error) {
	var b chess.Board
	fullFEN := pos.FEN + " 0 1"
	if err := b.SetFEN(fullFEN); err != nil {
		return nil, fmt.Errorf("invalid FEN: %w", err)
	}

	fmt.Printf("--- %s ---\n", id)
	fmt.Print(b.Print())
	fmt.Printf("FEN: %s\n", pos.FEN)
	fmt.Println()

	start := time.Now()

	info := &chess.SearchInfo{
		StartTime: time.Now(),
		MaxTime:   maxTime,
		TT:        tt,
		OnDepth: func(d, score int, nodes uint64, pv []chess.Move) {
			pvStr := pvToSAN(&b, pv)
			fmt.Printf("  depth %2d  score %6d  nodes %10d  pv %s\n", d, score, nodes, pvStr)
		},
	}

	bestMove, searchInfo := b.SearchWithInfo(depth, info)
	elapsed := time.Since(start)

	// Print summary for this position
	nps := uint64(0)
	if elapsed > 0 {
		nps = searchInfo.Nodes * uint64(time.Second) / uint64(elapsed)
	}
	fmt.Printf("  ---\n")
	fmt.Printf("  nodes: %d  NPS: %d  max depth: %d  time: %v\n",
		searchInfo.Nodes, nps, searchInfo.Depth, elapsed.Round(time.Millisecond))

	// Build result (same logic as RunEPDTest)
	result := &chess.EPDTestResult{
		Position:    pos,
		BestMove:    bestMove,
		BestMoveSAN: b.ToSAN(bestMove),
		SearchInfo:  searchInfo,
		TimeTaken:   elapsed,
	}

	for _, bm := range pos.BestMoves {
		expectedMove, err := b.ParseSAN(bm)
		if err != nil {
			continue
		}
		if bestMove == expectedMove {
			result.Passed = true
			break
		}
	}

	if result.Passed && len(pos.AvoidMoves) > 0 {
		for _, am := range pos.AvoidMoves {
			avoidMove, err := b.ParseSAN(am)
			if err != nil {
				continue
			}
			if bestMove == avoidMove {
				result.Passed = false
				break
			}
		}
	}

	return result, nil
}

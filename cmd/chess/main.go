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

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: chess [options]\n\nOptions:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExample:\n  chess -e testdata/wac.epd -t 5000 -n 20\n")
	}
	flag.Parse()

	if *epdFile != "" {
		runEPD(*epdFile, *depth, time.Duration(*maxTimeMS)*time.Millisecond, *maxPositions, *hashMB)
		return
	}

	flag.Usage()
	os.Exit(1)
}

func runEPD(filename string, depth int, maxTime time.Duration, maxPositions int, hashMB int) {
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

		result, err := chess.RunEPDTest(pos, depth, maxTime, tt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error on position %d: %v\n", i+1, err)
			continue
		}

		totalNodes += result.SearchInfo.Nodes

		status := "PASS"
		if !result.Passed {
			status = "FAIL"
			failed++
		} else {
			passed++
		}

		id := pos.ID
		if id == "" {
			id = fmt.Sprintf("#%d", i+1)
		}

		expected := strings.Join(pos.BestMoves, "/")
		fmt.Printf("[%s] %-12s found %-8s expected %-12s depth=%d nodes=%-10d time=%v\n",
			status, id, result.BestMoveSAN, expected,
			result.SearchInfo.Depth, result.SearchInfo.Nodes, result.TimeTaken.Round(time.Millisecond))
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

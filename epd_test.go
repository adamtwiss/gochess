package chess

import (
	"math"
	"strings"
	"testing"
	"time"
)

func TestParseEPD(t *testing.T) {
	tests := []struct {
		input     string
		wantFEN   string
		wantBM    []string
		wantID    string
		wantError bool
	}{
		{
			input:   `2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - bm Qg6; id "WAC.001";`,
			wantFEN: "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - -",
			wantBM:  []string{"Qg6"},
			wantID:  "WAC.001",
		},
		{
			input:   `r4q1k/p2bR1rp/2p2Q1N/5p2/5p2/2P5/PP3PPP/R5K1 w - - bm Rf7 Nf7+; id "WAC.008";`,
			wantFEN: "r4q1k/p2bR1rp/2p2Q1N/5p2/5p2/2P5/PP3PPP/R5K1 w - -",
			wantBM:  []string{"Rf7", "Nf7+"},
			wantID:  "WAC.008",
		},
		{
			input:     "",
			wantError: false, // Empty lines return nil, nil
		},
		{
			input:     "# comment",
			wantError: false, // Comments return nil, nil
		},
	}

	for _, tt := range tests {
		epd, err := ParseEPD(tt.input)
		if tt.wantError {
			if err == nil {
				t.Errorf("ParseEPD(%q) expected error", tt.input)
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseEPD(%q) error: %v", tt.input, err)
			continue
		}
		if epd == nil {
			continue // Empty/comment line
		}

		if epd.FEN != tt.wantFEN {
			t.Errorf("FEN = %q, want %q", epd.FEN, tt.wantFEN)
		}
		if len(epd.BestMoves) != len(tt.wantBM) {
			t.Errorf("BestMoves = %v, want %v", epd.BestMoves, tt.wantBM)
		} else {
			for i, bm := range epd.BestMoves {
				if bm != tt.wantBM[i] {
					t.Errorf("BestMoves[%d] = %q, want %q", i, bm, tt.wantBM[i])
				}
			}
		}
		if epd.ID != tt.wantID {
			t.Errorf("ID = %q, want %q", epd.ID, tt.wantID)
		}
	}
}

func TestParseEPDAvoidMoves(t *testing.T) {
	line := `r2qkb1r/pp1b1ppp/2n1pn2/2ppN3/3P4/1BN1P3/PPP2PPP/R1BQK2R w KQkq - bm Nxd7; am exd5; id "test.am";`
	epd, err := ParseEPD(line)
	if err != nil {
		t.Fatalf("ParseEPD error: %v", err)
	}
	if len(epd.BestMoves) != 1 || epd.BestMoves[0] != "Nxd7" {
		t.Errorf("BestMoves = %v, want [Nxd7]", epd.BestMoves)
	}
	if len(epd.AvoidMoves) != 1 || epd.AvoidMoves[0] != "exd5" {
		t.Errorf("AvoidMoves = %v, want [exd5]", epd.AvoidMoves)
	}
	if epd.ID != "test.am" {
		t.Errorf("ID = %q, want %q", epd.ID, "test.am")
	}
}

func TestParseEPDComments(t *testing.T) {
	line := `2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - bm Qg6; c0 "mate in 3"; c1 "famous position"; id "WAC.001";`
	epd, err := ParseEPD(line)
	if err != nil {
		t.Fatalf("ParseEPD error: %v", err)
	}
	if len(epd.Comments) != 2 {
		t.Errorf("Comments count = %d, want 2", len(epd.Comments))
	}
	if len(epd.Comments) >= 1 && epd.Comments[0] != "mate in 3" {
		t.Errorf("Comments[0] = %q, want %q", epd.Comments[0], "mate in 3")
	}
	if epd.RawOperands["c0"] != "mate in 3" {
		t.Errorf("RawOperands[c0] = %q, want %q", epd.RawOperands["c0"], "mate in 3")
	}
}

func TestParseEPDMalformed(t *testing.T) {
	// Too few fields should return error
	_, err := ParseEPD("rnbqkbnr/pppppppp/8/8")
	if err == nil {
		t.Error("Expected error for EPD with too few fields")
	}
}

func TestLoadEPDFileWAC(t *testing.T) {
	positions, err := LoadEPDFile("testdata/wac.epd")
	if err != nil {
		t.Fatalf("LoadEPDFile error: %v", err)
	}
	if len(positions) < 100 {
		t.Errorf("Expected at least 100 WAC positions, got %d", len(positions))
	}
	// Every position should have at least one best move and an ID
	for i, pos := range positions {
		if len(pos.BestMoves) == 0 {
			t.Errorf("Position %d has no best moves", i)
			break
		}
		if pos.ID == "" {
			t.Errorf("Position %d has no ID", i)
			break
		}
		// FEN should be parseable
		var b Board
		if err := b.SetFEN(pos.FEN + " 0 1"); err != nil {
			t.Errorf("Position %d (%s): invalid FEN %q: %v", i, pos.ID, pos.FEN, err)
			break
		}
	}
}

func TestFormatKNPS(t *testing.T) {
	tests := []struct {
		nodes   uint64
		elapsed time.Duration
		want    string
	}{
		{0, 0, "- kNPS"},
		{1000000, time.Second, "1,000 kNPS"},
		{500000, time.Second, "500 kNPS"},
		{50000, time.Second, "50 kNPS"},
	}
	for _, tt := range tests {
		got := FormatKNPS(tt.nodes, tt.elapsed)
		if got != tt.want {
			t.Errorf("FormatKNPS(%d, %v) = %q, want %q", tt.nodes, tt.elapsed, got, tt.want)
		}
	}
}

func TestRunEPDTest(t *testing.T) {
	// Test a simple tactical position
	epd := &EPDPosition{
		FEN:       "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - -",
		BestMoves: []string{"Qg6"},
		ID:        "WAC.001",
	}

	result, err := RunEPDTest(epd, 8, 5*time.Second, nil)
	if err != nil {
		t.Fatalf("RunEPDTest error: %v", err)
	}

	t.Logf("WAC.001: found %s, expected Qg6, passed=%v, nodes=%d, time=%v",
		result.BestMoveSAN, result.Passed, result.SearchInfo.Nodes, result.TimeTaken)

	if !result.Passed {
		t.Errorf("WAC.001: expected to find Qg6, got %s", result.BestMoveSAN)
	}
}

func TestWACSample(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping WAC sample test in short mode")
	}

	// Test first 5 WAC positions
	wacSample := []string{
		`2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - bm Qg6; id "WAC.001";`,
		`8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - bm Rxb2; id "WAC.002";`,
		`5rk1/1ppb3p/p1pb4/6q1/3P1p1r/2P1R2P/PP1BQ1P1/5RKN w - - bm Rg3; id "WAC.003";`,
		`r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - bm Qxh7+; id "WAC.004";`,
		`5k2/6pp/p1qN4/1p1p4/3P4/2PKP2Q/PP3r2/3R4 b - - bm Qc4+; id "WAC.005";`,
	}

	tt := NewTranspositionTable(32)
	passed := 0

	for _, line := range wacSample {
		epd, err := ParseEPD(line)
		if err != nil {
			t.Fatalf("ParseEPD error: %v", err)
		}

		result, err := RunEPDTest(epd, 8, 2*time.Second, tt)
		if err != nil {
			t.Fatalf("RunEPDTest error: %v", err)
		}

		status := "FAIL"
		if result.Passed {
			passed++
			status = "PASS"
		}

		t.Logf("[%s] %s: found %s, expected %s, depth=%d, nodes=%d, time=%v",
			status, epd.ID, result.BestMoveSAN, strings.Join(epd.BestMoves, "/"),
			result.SearchInfo.Depth, result.SearchInfo.Nodes, result.TimeTaken)
	}

	t.Logf("Passed %d/%d positions", passed, len(wacSample))
}

// TestWACFile runs all positions from testdata/wac.epd file
// Use: go test -v -run TestWACFile -timeout 30m
func TestWACFile(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping full WAC suite in short mode")
	}

	positions, err := LoadEPDFile("testdata/wac.epd")
	if err != nil {
		t.Fatalf("LoadEPDFile error: %v", err)
	}

	if len(positions) == 0 {
		t.Skip("No positions in testdata/wac.epd")
	}

	depth := 8
	maxTime := 2 * time.Second

	tt := NewTranspositionTable(64) // 64MB TT
	passed := 0
	failed := 0
	nodeScore := 0.0
	timeScore := 0.0
	maxTimeMs := float64(maxTime.Milliseconds())

	for _, epd := range positions {
		tt.Clear() // Clear TT between positions to avoid hash collision issues
		result, err := RunEPDTest(epd, depth, maxTime, tt)
		if err != nil {
			t.Errorf("%s: error: %v", epd.ID, err)
			continue
		}

		status := "FAIL"
		if result.Passed {
			passed++
			status = "PASS"
			if result.SolveNodes > 0 {
				nodeScore += EPDLogScore(float64(result.SearchInfo.Nodes), float64(result.SolveNodes))
			}
			if result.SolveTime > 0 {
				timeScore += EPDLogScore(maxTimeMs, float64(result.SolveTime.Milliseconds()))
			}
		} else {
			failed++
		}

		t.Logf("[%s] %s: found %s, expected %s, depth=%d, nodes=%d, time=%v",
			status, epd.ID, result.BestMoveSAN, strings.Join(epd.BestMoves, "/"),
			result.SearchInfo.Depth, result.SearchInfo.Nodes, result.TimeTaken)
	}

	pct := float64(passed) / float64(len(positions)) * 100
	maxTimeScore := float64(len(positions)) * math.Log2(maxTimeMs)
	t.Logf("\n=== SUMMARY ===")
	t.Logf("Passed: %d/%d (%.1f%%)", passed, len(positions), pct)
	t.Logf("Failed: %d", failed)
	t.Logf("Node score: %.1f  (higher=better)", nodeScore)
	t.Logf("Time score: %.1f / %.1f  (higher=better)", timeScore, maxTimeScore)
}

// TestECMFile runs all positions from testdata/ecm.epd file
// Use: go test -v -run TestECMFile -timeout 30m
func TestECMFile(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping full ECM suite in short mode")
	}

	positions, err := LoadEPDFile("testdata/ecm.epd")
	if err != nil {
		t.Fatalf("LoadEPDFile error: %v", err)
	}

	if len(positions) == 0 {
		t.Skip("No positions in testdata/ecm.epd")
	}

	tt := NewTranspositionTable(64) // 64MB TT
	passed := 0
	failed := 0
	nodeScore := 0.0
	timeScore := 0.0

	depth := 8
	maxTime := 2 * time.Second
	maxTimeMs := float64(maxTime.Milliseconds())

	for _, epd := range positions {
		tt.Clear() // Clear TT between positions to avoid hash collision issues
		result, err := RunEPDTest(epd, depth, maxTime, tt)
		if err != nil {
			t.Errorf("%s: error: %v", epd.ID, err)
			continue
		}

		status := "FAIL"
		if result.Passed {
			passed++
			status = "PASS"
			if result.SolveNodes > 0 {
				nodeScore += EPDLogScore(float64(result.SearchInfo.Nodes), float64(result.SolveNodes))
			}
			if result.SolveTime > 0 {
				timeScore += EPDLogScore(maxTimeMs, float64(result.SolveTime.Milliseconds()))
			}
		} else {
			failed++
		}

		t.Logf("[%s] %s: found %s, expected %s, depth=%d, nodes=%d, time=%v",
			status, epd.ID, result.BestMoveSAN, strings.Join(epd.BestMoves, "/"),
			result.SearchInfo.Depth, result.SearchInfo.Nodes, result.TimeTaken)
	}

	pct := float64(passed) / float64(len(positions)) * 100
	maxTimeScore := float64(len(positions)) * math.Log2(maxTimeMs)
	t.Logf("\n=== SUMMARY ===")
	t.Logf("Passed: %d/%d (%.1f%%)", passed, len(positions), pct)
	t.Logf("Failed: %d", failed)
	t.Logf("Node score: %.1f  (higher=better)", nodeScore)
	t.Logf("Time score: %.1f / %.1f  (higher=better)", timeScore, maxTimeScore)
}

// BenchmarkWAC runs the full WAC suite for benchmarking
func BenchmarkWAC(b *testing.B) {
	// Load first 5 positions
	wacPositions := []string{
		`2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - bm Qg6; id "WAC.001";`,
		`8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - bm Rxb2; id "WAC.002";`,
		`5rk1/1ppb3p/p1pb4/6q1/3P1p1r/2P1R2P/PP1BQ1P1/5RKN w - - bm Rg3; id "WAC.003";`,
		`r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - bm Qxh7+; id "WAC.004";`,
		`5k2/6pp/p1qN4/1p1p4/3P4/2PKP2Q/PP3r2/3R4 b - - bm Qc4+; id "WAC.005";`,
	}

	var positions []*EPDPosition
	for _, line := range wacPositions {
		epd, _ := ParseEPD(line)
		if epd != nil {
			positions = append(positions, epd)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tt := NewTranspositionTable(16)
		for _, pos := range positions {
			RunEPDTest(pos, 6, 1*time.Second, tt)
		}
	}
}

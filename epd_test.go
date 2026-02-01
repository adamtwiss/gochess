package chess

import (
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

	tt := NewTranspositionTable(64) // 64MB TT
	passed := 0
	failed := 0

	depth := 8
	maxTime := 2 * time.Second

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
		} else {
			failed++
		}

		t.Logf("[%s] %s: found %s, expected %s, depth=%d, nodes=%d, time=%v",
			status, epd.ID, result.BestMoveSAN, strings.Join(epd.BestMoves, "/"),
			result.SearchInfo.Depth, result.SearchInfo.Nodes, result.TimeTaken)
	}

	pct := float64(passed) / float64(len(positions)) * 100
	t.Logf("\n=== SUMMARY ===")
	t.Logf("Passed: %d/%d (%.1f%%)", passed, len(positions), pct)
	t.Logf("Failed: %d", failed)
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

	depth := 8
	maxTime := 2 * time.Second

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
		} else {
			failed++
		}

		t.Logf("[%s] %s: found %s, expected %s, depth=%d, nodes=%d, time=%v",
			status, epd.ID, result.BestMoveSAN, strings.Join(epd.BestMoves, "/"),
			result.SearchInfo.Depth, result.SearchInfo.Nodes, result.TimeTaken)
	}

	pct := float64(passed) / float64(len(positions)) * 100
	t.Logf("\n=== SUMMARY ===")
	t.Logf("Passed: %d/%d (%.1f%%)", passed, len(positions), pct)
	t.Logf("Failed: %d", failed)
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

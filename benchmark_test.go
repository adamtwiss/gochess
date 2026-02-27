package chess

import (
	"math"
	"os"
	"testing"
)

func TestBenchmarkScore(t *testing.T) {
	tests := []struct {
		name        string
		timeLimitMs float64
		solveTimeMs float64
		want        float64
	}{
		{"unsolved (zero solve time)", 200, 0, 0},
		{"unsolved (negative solve time)", 200, -1, 0},
		{"at deadline", 200, 200, 1.0},
		{"halfway", 200, 100, 2.0},
		{"instant (1ms floor)", 200, 0.5, 1.0 + math.Log2(200)},
		{"1ms solve", 200, 1, 1.0 + math.Log2(200)},
		{"50ms solve", 200, 50, 1.0 + math.Log2(4)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BenchmarkScore(tt.timeLimitMs, tt.solveTimeMs)
			if math.Abs(got-tt.want) > 0.001 {
				t.Errorf("BenchmarkScore(%v, %v) = %v, want %v", tt.timeLimitMs, tt.solveTimeMs, got, tt.want)
			}
		})
	}
}

func TestRunBenchmarkSmall(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark integration test in short mode")
	}

	// Load WAC and take just the first 5 positions
	positions, err := LoadEPDFile("testdata/wac.epd")
	if err != nil {
		t.Fatalf("failed to load EPD: %v", err)
	}
	if len(positions) < 5 {
		t.Fatalf("expected at least 5 positions, got %d", len(positions))
	}

	// Write a small temp EPD file
	tmpFile := t.TempDir() + "/test.epd"
	var lines []byte
	for _, p := range positions[:5] {
		line := p.FEN
		for k, v := range p.RawOperands {
			line += " " + k + " " + v + ";"
		}
		lines = append(lines, []byte(line+"\n")...)
	}
	if err := os.WriteFile(tmpFile, lines, 0644); err != nil {
		t.Fatalf("failed to write temp EPD: %v", err)
	}

	smallSuites := []BenchmarkSuiteConfig{
		{Name: "WAC-5", Filename: tmpFile},
	}

	callbackCount := 0
	result, err := RunBenchmark(smallSuites, 100, 16, 64, 1, func(p BenchmarkProgress) {
		callbackCount++
	})
	if err != nil {
		t.Fatalf("RunBenchmark failed: %v", err)
	}

	if len(result.Suites) != 1 {
		t.Fatalf("expected 1 suite, got %d", len(result.Suites))
	}

	suite := result.Suites[0]
	if suite.Total != 5 {
		t.Errorf("expected 5 positions, got %d", suite.Total)
	}
	if suite.Name != "WAC-5" {
		t.Errorf("expected suite name WAC-5, got %s", suite.Name)
	}
	if len(suite.Positions) != 5 {
		t.Errorf("expected 5 position results, got %d", len(suite.Positions))
	}
	if callbackCount != 5 {
		t.Errorf("expected 5 callbacks, got %d", callbackCount)
	}
	if result.TimeLimitMs != 100 {
		t.Errorf("expected TimeLimitMs=100, got %d", result.TimeLimitMs)
	}

	// Verify aggregate helpers
	if result.TotalPositions() != 5 {
		t.Errorf("TotalPositions() = %d, want 5", result.TotalPositions())
	}
	if result.TotalSolved() > 5 {
		t.Errorf("TotalSolved() = %d, want <= 5", result.TotalSolved())
	}
	if result.TotalScore() < 0 {
		t.Errorf("TotalScore() = %f, want >= 0", result.TotalScore())
	}

	// Verify each position has reasonable fields
	for i, pos := range suite.Positions {
		if pos.FEN == "" {
			t.Errorf("position %d: empty FEN", i)
		}
		if pos.Nodes == 0 {
			t.Errorf("position %d: zero nodes", i)
		}
		if pos.Solved && pos.Score <= 0 {
			t.Errorf("position %d: solved but score=%f", i, pos.Score)
		}
		if !pos.Solved && pos.Score != 0 {
			t.Errorf("position %d: unsolved but score=%f", i, pos.Score)
		}
	}
}

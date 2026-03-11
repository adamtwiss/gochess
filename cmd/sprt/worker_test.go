package main

import (
	"testing"
)

func TestParseCutechessOutput(t *testing.T) {
	// Real cutechess-cli output
	output := `Started game 1 of 50 (New vs Base)
Finished game 1 (New vs Base): 1-0 {White wins by adjudication}
Score of New vs Base: 1 - 0 - 0  [1.000] 1
Started game 2 of 50 (Base vs New)
Finished game 50 (New vs Base): 1/2-1/2 {Draw by adjudication}
Score of New vs Base: 20 - 15 - 15  [0.550] 50
Elo difference: 34.9 +/- 57.3, LOS: 84.9 %, DrawRatio: 30.0 %
`
	wins, draws, losses, err := parseCutechessOutput(output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// cutechess format: W - L - D, so we get W=20, L=15, D=15
	// parseCutechessOutput returns (wins, draws, losses) mapped from (W, D, L)
	if wins != 20 || draws != 15 || losses != 15 {
		t.Errorf("expected 20/15/15, got %d/%d/%d", wins, draws, losses)
	}
}

func TestParseCutechessOutputNoScore(t *testing.T) {
	_, _, _, err := parseCutechessOutput("some random output\nno score line here\n")
	if err == nil {
		t.Error("expected error for missing score line")
	}
}

func TestLastLines(t *testing.T) {
	s := "line1\nline2\nline3\nline4\nline5"
	result := lastLines(s, 3)
	if result != "line3\nline4\nline5" {
		t.Errorf("unexpected result: %q", result)
	}

	// Fewer lines than requested
	result = lastLines("a\nb", 5)
	if result != "a\nb" {
		t.Errorf("unexpected result: %q", result)
	}
}

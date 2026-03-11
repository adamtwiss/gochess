package main

import (
	"math"
	"testing"
)

func TestComputeLLR(t *testing.T) {
	// Test case: clear winner (many wins, few losses)
	// With elo0=0, elo1=10, a strong positive result should give positive LLR
	llr := ComputeLLR(150, 600, 100, 0, 10)
	if llr <= 0 {
		t.Errorf("expected positive LLR for winning result, got %f", llr)
	}

	// Test case: clear loser
	llr = ComputeLLR(100, 600, 150, 0, 10)
	if llr >= 0 {
		t.Errorf("expected negative LLR for losing result, got %f", llr)
	}

	// Test case: even result should be near zero or slightly negative
	// (since H0 is 0 Elo, even result supports H0)
	llr = ComputeLLR(200, 600, 200, 0, 10)
	if llr > 0.5 { // should be close to 0 or slightly negative
		t.Errorf("expected near-zero LLR for even result, got %f", llr)
	}

	// Test case: zero games
	llr = ComputeLLR(0, 0, 0, 0, 10)
	if llr != 0 {
		t.Errorf("expected 0 LLR for zero games, got %f", llr)
	}
}

func TestEstimateElo(t *testing.T) {
	// Even result → ~0 Elo
	elo, _ := EstimateElo(200, 600, 200)
	if math.Abs(elo) > 1 {
		t.Errorf("expected ~0 Elo for even result, got %f", elo)
	}

	// Winning result → positive Elo
	elo, eloErr := EstimateElo(150, 600, 100)
	if elo <= 0 {
		t.Errorf("expected positive Elo, got %f", elo)
	}
	if eloErr <= 0 {
		t.Errorf("expected positive error bar, got %f", eloErr)
	}

	// Zero games
	elo, eloErr = EstimateElo(0, 0, 0)
	if elo != 0 || eloErr != 0 {
		t.Errorf("expected 0,0 for zero games, got %f,%f", elo, eloErr)
	}
}

func TestSPRTBounds(t *testing.T) {
	cfg := DefaultSPRT() // alpha=0.05, beta=0.05
	lower := cfg.LowerBound()
	upper := cfg.UpperBound()

	// log(0.05/0.95) ≈ -2.944
	if math.Abs(lower-math.Log(0.05/0.95)) > 0.001 {
		t.Errorf("unexpected lower bound: %f", lower)
	}
	// log(0.95/0.05) ≈ 2.944
	if math.Abs(upper-math.Log(0.95/0.05)) > 0.001 {
		t.Errorf("unexpected upper bound: %f", upper)
	}
}

func TestUpdateSPRT(t *testing.T) {
	cfg := DefaultSPRT()

	// Strong winner should be accepted
	r := SPRTResult{Wins: 300, Draws: 600, Losses: 100}
	UpdateSPRT(&r, cfg)
	if r.Status != "accepted" {
		t.Errorf("expected accepted, got %s (LLR=%f)", r.Status, r.LLR)
	}

	// Strong loser should be rejected
	r = SPRTResult{Wins: 100, Draws: 600, Losses: 300}
	UpdateSPRT(&r, cfg)
	if r.Status != "rejected" {
		t.Errorf("expected rejected, got %s (LLR=%f)", r.Status, r.LLR)
	}

	// Few games should still be running
	r = SPRTResult{Wins: 5, Draws: 10, Losses: 5}
	UpdateSPRT(&r, cfg)
	if r.Status != "running" {
		t.Errorf("expected running, got %s (LLR=%f)", r.Status, r.LLR)
	}
}

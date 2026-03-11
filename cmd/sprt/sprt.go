package main

import (
	"math"
)

// SPRTConfig holds the hypothesis test parameters.
type SPRTConfig struct {
	Elo0  float64 `json:"elo0"`  // null hypothesis (e.g. 0)
	Elo1  float64 `json:"elo1"`  // alternative hypothesis (e.g. 10)
	Alpha float64 `json:"alpha"` // type I error rate (e.g. 0.05)
	Beta  float64 `json:"beta"`  // type II error rate (e.g. 0.05)
}

// SPRTResult holds accumulated game results and derived statistics.
type SPRTResult struct {
	Wins   int     `json:"wins"`
	Draws  int     `json:"draws"`
	Losses int     `json:"losses"`
	LLR    float64 `json:"llr"`
	Elo    float64 `json:"elo"`
	EloErr float64 `json:"elo_err"` // 95% CI half-width
	Status string  `json:"status"`  // "running", "accepted", "rejected"
}

// DefaultSPRT returns a standard SPRT config for small Elo gains.
func DefaultSPRT() SPRTConfig {
	return SPRTConfig{Elo0: 0, Elo1: 10, Alpha: 0.05, Beta: 0.05}
}

// LowerBound returns the H0 acceptance threshold: log(beta / (1 - alpha)).
func (c SPRTConfig) LowerBound() float64 {
	return math.Log(c.Beta / (1 - c.Alpha))
}

// UpperBound returns the H1 acceptance threshold: log((1 - beta) / alpha).
func (c SPRTConfig) UpperBound() float64 {
	return math.Log((1 - c.Beta) / c.Alpha)
}

// eloToScore converts an Elo difference to expected score.
func eloToScore(elo float64) float64 {
	return 1.0 / (1.0 + math.Pow(10, -elo/400.0))
}

// scoreToElo converts an expected score to Elo difference.
func scoreToElo(s float64) float64 {
	if s <= 0 || s >= 1 {
		return 0
	}
	return -400.0 * math.Log10(1.0/s-1.0)
}

// ComputeLLR computes the log-likelihood ratio for the trinomial SPRT.
// Uses the Bernoulli approximation (score-based, not pentanomial).
func ComputeLLR(w, d, l int, elo0, elo1 float64) float64 {
	n := float64(w + d + l)
	if n == 0 {
		return 0
	}
	s := (float64(w) + float64(d)/2.0) / n // observed score
	s0 := eloToScore(elo0)
	s1 := eloToScore(elo1)

	// Clamp observed score to avoid log(0)
	if s <= 0 || s >= 1 {
		return 0
	}

	return n * (s*math.Log(s1/s0) + (1-s)*math.Log((1-s1)/(1-s0)))
}

// EstimateElo returns the Elo estimate and 95% CI half-width from W/D/L.
func EstimateElo(w, d, l int) (elo, eloErr float64) {
	n := float64(w + d + l)
	if n == 0 {
		return 0, 0
	}
	wf := float64(w) / n
	df := float64(d) / n
	_ = float64(l) / n

	score := wf + df/2.0
	elo = scoreToElo(score)

	// Score variance: E[X^2] - E[X]^2 where X in {0, 0.5, 1}
	variance := wf*1.0 + df*0.25 - score*score
	if variance < 0 {
		variance = 0
	}

	// 95% CI using logistic derivative for Elo conversion
	if score > 0 && score < 1 {
		// d(Elo)/d(score) = -400 / (ln(10) * score * (1-score))
		deriv := 400.0 / (math.Ln10 * score * (1 - score))
		eloErr = 1.96 * deriv * math.Sqrt(variance/n)
	}

	return elo, eloErr
}

// UpdateSPRT recomputes the SPRT result from total W/D/L.
func UpdateSPRT(r *SPRTResult, cfg SPRTConfig) {
	r.LLR = ComputeLLR(r.Wins, r.Draws, r.Losses, cfg.Elo0, cfg.Elo1)
	r.Elo, r.EloErr = EstimateElo(r.Wins, r.Draws, r.Losses)

	lower := cfg.LowerBound()
	upper := cfg.UpperBound()

	if r.LLR >= upper {
		r.Status = "accepted"
	} else if r.LLR <= lower {
		r.Status = "rejected"
	} else {
		r.Status = "running"
	}
}

package chess

import (
	"fmt"
	"math"
	"testing"
)

// TestNNUEActivationHistogram prints activation distribution histograms for each layer
// across many diverse positions. This reveals whether neurons have healthy activation
// spreads or are clustered near zero (barely alive).
func TestNNUEActivationHistogram(t *testing.T) {
	net, err := LoadNNUE("net.nnue")
	if err != nil {
		t.Skipf("no net.nnue found: %v", err)
	}
	net.PrepareWeights()
	trainNet := DequantizeNetwork(net)

	positions := []string{
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
		"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
		"rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
		"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
		"r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 5",
		"r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 6 8",
		"r1bq1rk1/pp2bppp/2n1pn2/3p4/3P1B2/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
		"rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1",
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1",
		"4k3/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/4K3 w kq - 0 1",
		"4k3/8/8/8/8/8/4PPPP/4K2R w K - 0 1",
		"4k3/pppp4/8/8/8/8/PPPP4/4K3 w - - 0 1",
		"8/5k2/8/8/3Q4/8/8/4K3 w - - 0 1",
		"8/5k2/8/8/3r4/8/5PP1/4K3 w - - 0 1",
		"8/8/3k4/8/8/3K4/3P4/8 w - - 0 1",
		"8/3k4/8/8/8/8/3KR3/8 w - - 0 1",
		"r1bqk2r/ppppbppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
		"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
		"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
		"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
		"4k3/pp3ppp/4p3/3p4/3P4/4P3/PP3PPP/4K3 w - - 0 1",
		"4k3/1p4pp/p2p4/P1pPp3/2P1P3/6PP/1P6/4K3 w - - 0 1",
		"8/8/8/4k3/8/4K3/8/8 w - - 0 1",
		"r1bq1rk1/ppp2ppp/2np4/8/3NP1n1/2N5/PPP2PPP/R1BQKB1R w KQ - 0 1",
		"r3kb1r/pp1n1ppp/1qp1pn2/3p1b2/3P1B2/2NBPN2/PPP2PPP/R2QK2R w KQkq - 0 8",
		"2kr3r/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP2PPP/R1B1K2R w KQ - 0 8",
		"r1b1k1nr/ppppqppp/2n5/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
	}

	// Collect all activations across all positions
	// Pre-activation (before CReLU) for hidden1, post-activation for hidden2/3
	allPreAct1 := make([]float32, 0, len(positions)*NNUEHiddenSize*2)
	allPostAct1 := make([]float32, 0, len(positions)*NNUEHiddenSize*2)
	allPostAct2 := make([]float32, 0, len(positions)*NNUEHidden2Size)
	allPostAct3 := make([]float32, 0, len(positions)*NNUEHidden3Size)

	for _, fen := range positions {
		var b Board
		b.Reset()
		if err := b.SetFEN(fen); err != nil {
			t.Fatalf("bad FEN: %v", err)
		}

		sample := extractFeatures(&b)

		// Compute pre-activation accumulator values manually
		preAct := make([]float32, NNUEHiddenSize*2)
		// White perspective (first 256)
		for i := 0; i < NNUEHiddenSize; i++ {
			acc := trainNet.InputBiases[i]
			for _, idx := range sample.WhiteFeatures {
				acc += trainNet.InputWeights[idx][i]
			}
			preAct[i] = acc
		}
		// Black perspective (second 256)
		for i := 0; i < NNUEHiddenSize; i++ {
			acc := trainNet.InputBiases[i]
			for _, idx := range sample.BlackFeatures {
				acc += trainNet.InputWeights[idx][i]
			}
			preAct[NNUEHiddenSize+i] = acc
		}
		allPreAct1 = append(allPreAct1, preAct...)

		// Full forward pass gives post-activation values
		_, hidden1, hidden2, hidden3 := trainNet.Forward(sample)
		allPostAct1 = append(allPostAct1, hidden1[:]...)
		allPostAct2 = append(allPostAct2, hidden2[:]...)
		allPostAct3 = append(allPostAct3, hidden3[:]...)
	}

	// Print histograms
	fmt.Printf("=== NNUE Activation Histograms (%d positions) ===\n\n", len(positions))

	fmt.Println("--- Hidden1 PRE-activation (before CReLU) ---")
	fmt.Println("Shows raw accumulator values. Negative = dead for this input.")
	printHistogram(allPreAct1, []float64{
		-2.0, -1.5, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0,
	})

	fmt.Println("\n--- Hidden1 POST-activation (after CReLU, clipped to [0,1]) ---")
	printHistogram(allPostAct1, []float64{
		0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
	})

	fmt.Println("\n--- Hidden2 POST-activation (after ReLU) ---")
	printHistogram(allPostAct2, []float64{
		0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
	})

	fmt.Println("\n--- Hidden3 POST-activation (after ReLU) ---")
	printHistogram(allPostAct3, []float64{
		0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
	})

	// Summary statistics
	fmt.Println("\n=== Summary Statistics ===")
	printStats("Hidden1 pre-act", allPreAct1)
	printStats("Hidden1 post-act", allPostAct1)
	printStats("Hidden2 post-act", allPostAct2)
	printStats("Hidden3 post-act", allPostAct3)
}

func printHistogram(values []float32, edges []float64) {
	bins := make([]int, len(edges)+1)
	for _, v := range values {
		placed := false
		for i, edge := range edges {
			if float64(v) <= edge {
				bins[i]++
				placed = true
				break
			}
		}
		if !placed {
			bins[len(edges)]++
		}
	}

	total := len(values)
	maxCount := 0
	for _, c := range bins {
		if c > maxCount {
			maxCount = c
		}
	}

	barWidth := 50
	for i, count := range bins {
		var label string
		if i == 0 {
			label = fmt.Sprintf("      ≤%6.2f", edges[0])
		} else if i == len(edges) {
			label = fmt.Sprintf("      >%6.2f", edges[len(edges)-1])
		} else {
			label = fmt.Sprintf("%6.2f–%6.2f", edges[i-1], edges[i])
		}

		barLen := 0
		if maxCount > 0 {
			barLen = count * barWidth / maxCount
		}
		bar := ""
		for j := 0; j < barLen; j++ {
			bar += "█"
		}

		pct := 100.0 * float64(count) / float64(total)
		fmt.Printf("  %s │ %s %d (%.1f%%)\n", label, bar, count, pct)
	}
}

func printStats(name string, values []float32) {
	if len(values) == 0 {
		return
	}

	var sum, sumSq float64
	var min, max float32
	min = values[0]
	max = values[0]
	var zeroCount, nearZeroCount int

	for _, v := range values {
		sum += float64(v)
		sumSq += float64(v) * float64(v)
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		if v == 0 {
			zeroCount++
		}
		if v >= 0 && v < 0.01 {
			nearZeroCount++
		}
	}

	n := float64(len(values))
	mean := sum / n
	variance := sumSq/n - mean*mean
	stddev := math.Sqrt(math.Max(0, variance))

	fmt.Printf("  %-20s  min=%7.4f  max=%7.4f  mean=%7.4f  std=%7.4f  zero=%d/%d (%.1f%%)  near-zero=%d (%.1f%%)\n",
		name, min, max, mean, stddev, zeroCount, len(values), 100*float64(zeroCount)/n,
		nearZeroCount, 100*float64(nearZeroCount)/n)
}

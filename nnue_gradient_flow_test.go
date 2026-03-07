package chess

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestNNUEGradientFlow checks whether gradients flow properly through all layers
// during NNUE training. With CReLU (clips to [0, 1.0]) on every hidden layer,
// gradients may die — especially at the input layer.
func TestNNUEGradientFlow(t *testing.T) {
	// Try to load and dequantize existing network; fall back to fresh random init
	var trainNet *NNUETrainNet
	net, err := LoadNNUE("net.nnue")
	if err != nil {
		t.Logf("no net.nnue found, using fresh random init: %v", err)
		rng := rand.New(rand.NewSource(42))
		trainNet = NewNNUETrainNet(rng)
	} else {
		net.PrepareWeights()
		trainNet = DequantizeNetwork(net)
		t.Log("loaded and dequantized net.nnue")
	}

	cfg := NNUETrainConfig{
		K:      400.0,
		Lambda: 0.5,
	}

	// Build test positions
	positions := []struct {
		name   string
		fen    string
		result float32
		score  float32
	}{
		{"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.5, 30},
		{"italian", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", 0.4, -20},
		{"endgame", "4k3/8/8/8/8/8/4PPPP/4K2R w K - 0 1", 1.0, 500},
		{"queens gambit", "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2", 0.5, 10},
	}

	// Accumulate gradients across all positions
	grads := &NNUETrainGradients{}
	grads.dirtyInputs = make([]int, 0, 4096)
	grads.dirtySet = make([]bool, NNUEInputSize)

	var allSamples []*NNUETrainSample

	for _, pos := range positions {
		var b Board
		b.Reset()
		if err := b.SetFEN(pos.fen); err != nil {
			t.Fatalf("bad FEN %s: %v", pos.name, err)
		}

		sample := extractFeatures(&b)
		sample.Result = pos.result
		sample.Score = pos.score
		sample.HasScore = true
		allSamples = append(allSamples, sample)

		output, hidden1, hidden2, hidden3 := trainNet.Forward(sample)
		trainNet.Backward(sample, grads, output, hidden1, hidden2, hidden3, cfg)
	}

	// Analyze gradient magnitudes at each layer
	fmt.Println("=== NNUE Gradient Flow Diagnostic ===")
	fmt.Println()

	// Output layer
	{
		fmt.Println("--- Output Layer ---")
		fmt.Printf("  OutputBias gradient: %.8e\n", grads.OutputBias)
		analyzeGradArray1D("OutputWeights", grads.OutputWeights[:], NNUEHidden3Size)
	}

	// Hidden2 layer (Hidden2Weights, Hidden2Biases)
	{
		fmt.Println("--- Hidden Layer 2 (Hidden2) ---")
		analyzeGradArray1D("Hidden2Biases", grads.Hidden2Biases[:], NNUEHidden3Size)
		var flat []float32
		for j := 0; j < NNUEHidden2Size; j++ {
			flat = append(flat, grads.Hidden2Weights[j][:]...)
		}
		analyzeGradArray1D("Hidden2Weights", flat, len(flat))
	}

	// Hidden layer 1 (HiddenWeights, HiddenBiases)
	{
		fmt.Println("--- Hidden Layer 1 ---")
		analyzeGradArray1D("HiddenBiases", grads.HiddenBiases[:], NNUEHidden2Size)
		var flat []float32
		for i := 0; i < NNUEHiddenSize*2; i++ {
			flat = append(flat, grads.HiddenWeights[i][:]...)
		}
		analyzeGradArray1D("HiddenWeights", flat, len(flat))
	}

	// Input layer — only check sparse entries that were actually touched
	{
		fmt.Println("--- Input Layer (sparse, only active features) ---")
		analyzeGradArray1D("InputBiases", grads.InputBiases[:], NNUEHiddenSize)

		// Collect all input weight gradients for touched features
		var sparseFlat []float32
		for _, idx := range grads.dirtyInputs {
			sparseFlat = append(sparseFlat, grads.InputWeights[idx][:]...)
		}
		numTouchedFeatures := len(grads.dirtyInputs)
		fmt.Printf("  Input features touched: %d / %d\n", numTouchedFeatures, NNUEInputSize)
		if len(sparseFlat) > 0 {
			analyzeGradArray1D("InputWeights (touched)", sparseFlat, len(sparseFlat))
		} else {
			fmt.Println("  WARNING: No input weight gradients at all!")
		}
	}

	// Activation statistics: show how many neurons are in the gradient-active zone
	fmt.Println()
	fmt.Println("=== Activation Statistics (per position) ===")
	for i, pos := range positions {
		sample := allSamples[i]
		_, hidden1, hidden2, hidden3 := trainNet.Forward(sample)

		fmt.Printf("\n  Position: %s\n", pos.name)

		// Hidden1 (after CReLU)
		var dead1, active1, saturated1 int
		for j := 0; j < NNUEHiddenSize*2; j++ {
			if hidden1[j] <= 0 {
				dead1++
			} else if hidden1[j] >= nnueFloatClipMax {
				saturated1++
			} else {
				active1++
			}
		}
		total1 := NNUEHiddenSize * 2
		fmt.Printf("    Hidden1 (%d neurons): dead=%d (%.1f%%) active=%d (%.1f%%) saturated=%d (%.1f%%)\n",
			total1, dead1, pct(dead1, total1), active1, pct(active1, total1), saturated1, pct(saturated1, total1))

		// Hidden2 (pre-CReLU values, since CReLU is applied inside Hidden2->Hidden3 matmul)
		var dead2, active2, saturated2 int
		for j := 0; j < NNUEHidden2Size; j++ {
			if hidden2[j] <= 0 {
				dead2++
			} else if hidden2[j] >= nnueFloatClipMax {
				saturated2++
			} else {
				active2++
			}
		}
		fmt.Printf("    Hidden2 (%d neurons): dead=%d (%.1f%%) active=%d (%.1f%%) saturated=%d (%.1f%%)\n",
			NNUEHidden2Size, dead2, pct(dead2, NNUEHidden2Size), active2, pct(active2, NNUEHidden2Size), saturated2, pct(saturated2, NNUEHidden2Size))

		// Hidden3 (pre-CReLU values)
		var dead3, active3, saturated3 int
		for j := 0; j < NNUEHidden3Size; j++ {
			if hidden3[j] <= 0 {
				dead3++
			} else if hidden3[j] >= nnueFloatClipMax {
				saturated3++
			} else {
				active3++
			}
		}
		fmt.Printf("    Hidden3 (%d neurons): dead=%d (%.1f%%) active=%d (%.1f%%) saturated=%d (%.1f%%)\n",
			NNUEHidden3Size, dead3, pct(dead3, NNUEHidden3Size), active3, pct(active3, NNUEHidden3Size), saturated3, pct(saturated3, NNUEHidden3Size))
	}

	// Gradient magnitude ratios between layers (check for vanishing)
	fmt.Println()
	fmt.Println("=== Layer-to-Layer Gradient Ratio ===")

	outputMean := meanAbsGrad(grads.OutputWeights[:])
	var h2flat []float32
	for j := 0; j < NNUEHidden2Size; j++ {
		h2flat = append(h2flat, grads.Hidden2Weights[j][:]...)
	}
	hidden2Mean := meanAbsGrad(h2flat)

	var h1flat []float32
	for i := 0; i < NNUEHiddenSize*2; i++ {
		h1flat = append(h1flat, grads.HiddenWeights[i][:]...)
	}
	hidden1Mean := meanAbsGrad(h1flat)

	var inputFlat []float32
	for _, idx := range grads.dirtyInputs {
		inputFlat = append(inputFlat, grads.InputWeights[idx][:]...)
	}
	inputMean := meanAbsGrad(inputFlat)

	fmt.Printf("  Output    mean|grad|: %.8e\n", outputMean)
	fmt.Printf("  Hidden2   mean|grad|: %.8e  (ratio to output: %.4f)\n", hidden2Mean, hidden2Mean/outputMean)
	fmt.Printf("  Hidden1   mean|grad|: %.8e  (ratio to output: %.4f)\n", hidden1Mean, hidden1Mean/outputMean)
	fmt.Printf("  Input     mean|grad|: %.8e  (ratio to output: %.4f)\n", inputMean, inputMean/outputMean)

	if inputMean == 0 {
		t.Error("CRITICAL: Input layer gradients are completely zero — no learning signal reaching input weights")
	} else if inputMean/outputMean < 1e-6 {
		t.Errorf("WARNING: Input gradients are %.2e times smaller than output — severe vanishing gradient", inputMean/outputMean)
	}
}

func analyzeGradArray1D(name string, arr []float32, total int) {
	if len(arr) == 0 {
		fmt.Printf("  %s: empty\n", name)
		return
	}
	var sumAbs float64
	var maxAbs float64
	var zeroCount int
	for _, v := range arr {
		abs := math.Abs(float64(v))
		sumAbs += abs
		if abs > maxAbs {
			maxAbs = abs
		}
		if v == 0 {
			zeroCount++
		}
	}
	meanAbs := sumAbs / float64(len(arr))
	fmt.Printf("  %s (%d params): mean|grad|=%.8e  max|grad|=%.8e  zero=%.1f%%\n",
		name, total, meanAbs, maxAbs, pct(zeroCount, len(arr)))
}

func meanAbsGrad(arr []float32) float64 {
	if len(arr) == 0 {
		return 0
	}
	var sum float64
	for _, v := range arr {
		sum += math.Abs(float64(v))
	}
	return sum / float64(len(arr))
}

func pct(n, total int) float64 {
	if total == 0 {
		return 0
	}
	return 100.0 * float64(n) / float64(total)
}

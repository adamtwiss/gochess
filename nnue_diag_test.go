package chess

import (
	"fmt"
	"math"
	"testing"
)

// TestNNUEDiagnostic compares float32 training forward pass vs quantized inference
// to determine whether small NNUE scores come from training or quantization.
func TestNNUEDiagnostic(t *testing.T) {
	// Load the quantized network
	net, err := LoadNNUE("net.nnue")
	if err != nil {
		t.Skipf("no net.nnue found: %v", err)
	}
	net.PrepareWeights()

	// Dequantize back to float32
	trainNet := DequantizeNetwork(net)

	positions := []struct {
		name string
		fen  string
	}{
		{"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
		{"after e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"},
		{"missing pawn", "rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1"},
		{"missing knight", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1"},
		{"missing bishop", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1"},
		{"missing rook", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1"},
		{"missing queen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"},
		{"italian", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"},
		{"queens gambit", "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2"},
		{"endgame rook up", "4k3/8/8/8/8/8/4PPPP/4K2R w K - 0 1"},
		{"endgame queen up", "4k3/8/8/8/8/8/4PPPP/3QK3 w - - 0 1"},
	}

	for _, pos := range positions {
		var b Board
		b.Reset()
		if err := b.SetFEN(pos.fen); err != nil {
			t.Fatalf("bad FEN %s: %v", pos.name, err)
		}

		// Build features for the training forward pass
		sample := extractFeatures(&b)

		// Float32 forward pass
		floatOutput, _, _, _ := trainNet.Forward(sample)

		// Quantized inference
		accStack := NewNNUEAccumulatorStack(8)
		b.NNUEAcc = accStack
		b.NNUENet = net
		net.RecomputeAccumulator(accStack.Current(), &b)
		quantOutput := net.Evaluate(accStack.Current(), b.SideToMove)

		// Classical eval for reference
		classicalOutput := b.EvaluateRelative()

		fmt.Printf("%-20s  float32=%.1f  quantized=%d  classical=%d  ratio=%.2fx\n",
			pos.name, floatOutput, quantOutput, classicalOutput,
			float64(quantOutput)/float64(classicalOutput+1))
	}
}

// extractFeatures builds an NNUETrainSample from a Board position
func extractFeatures(b *Board) *NNUETrainSample {
	sample := &NNUETrainSample{
		SideToMove: b.SideToMove,
	}

	wKingSq := b.Pieces[WhiteKing].LSB()
	bKingSq := b.Pieces[BlackKing].LSB()

	// Iterate all pieces including kings
	for piece := WhitePawn; piece <= BlackKing; piece++ {
		bb := b.Pieces[piece]
		for bb != 0 {
			sq := bb.PopLSB()
			wIdx := HalfKAIndex(White, wKingSq, piece, sq)
			bIdx := HalfKAIndex(Black, bKingSq, piece, sq)
			sample.WhiteFeatures = append(sample.WhiteFeatures, wIdx)
			sample.BlackFeatures = append(sample.BlackFeatures, bIdx)
		}
	}

	return sample
}

// TestNNUEQuantizationAccuracy checks how much quantization error there is
func TestNNUEQuantizationAccuracy(t *testing.T) {
	net, err := LoadNNUE("net.nnue")
	if err != nil {
		t.Skipf("no net.nnue found: %v", err)
	}
	net.PrepareWeights()

	trainNet := DequantizeNetwork(net)

	// Check weight statistics
	var inputMin, inputMax float32
	var inputNonZero int
	for i := range trainNet.InputWeights {
		for j := range trainNet.InputWeights[i] {
			v := trainNet.InputWeights[i][j]
			if v != 0 {
				inputNonZero++
			}
			if v < inputMin {
				inputMin = v
			}
			if v > inputMax {
				inputMax = v
			}
		}
	}

	var hiddenMin, hiddenMax float32
	for i := range trainNet.HiddenWeights {
		for j := range trainNet.HiddenWeights[i] {
			v := trainNet.HiddenWeights[i][j]
			if v < hiddenMin {
				hiddenMin = v
			}
			if v > hiddenMax {
				hiddenMax = v
			}
		}
	}

	var outMin, outMax float32
	for j := range trainNet.OutputWeights {
		v := trainNet.OutputWeights[j]
		if v < outMin {
			outMin = v
		}
		if v > outMax {
			outMax = v
		}
	}

	total := NNUEInputSize * NNUEHiddenSize
	fmt.Printf("Input weights:  min=%.4f max=%.4f nonzero=%d/%d (%.1f%%)\n",
		inputMin, inputMax, inputNonZero, total, 100*float64(inputNonZero)/float64(total))
	fmt.Printf("Hidden weights: min=%.4f max=%.4f\n", hiddenMin, hiddenMax)
	fmt.Printf("Output weights: min=%.4f max=%.4f\n", outMin, outMax)
	fmt.Printf("Output bias:    %.4f\n", trainNet.OutputBias)

	// Check accumulator values for startpos
	var b Board
	b.Reset()
	sample := extractFeatures(&b)
	_, hidden1, hidden2, hidden3 := trainNet.Forward(sample)

	// Count active hidden1 neurons (after CReLU)
	var active1, saturated1 int
	var h1Min, h1Max float32
	for i := 0; i < NNUEHiddenSize*2; i++ {
		if hidden1[i] > 0 {
			active1++
		}
		if hidden1[i] >= 1.0 {
			saturated1++
		}
		if hidden1[i] < h1Min {
			h1Min = hidden1[i]
		}
		if hidden1[i] > h1Max {
			h1Max = hidden1[i]
		}
	}

	var active2, saturated2 int
	var h2Min, h2Max float32
	for j := 0; j < NNUEHidden2Size; j++ {
		if hidden2[j] > 0 {
			active2++
		}
		if hidden2[j] >= 1.0 {
			saturated2++
		}
		if hidden2[j] < h2Min {
			h2Min = hidden2[j]
		}
		if hidden2[j] > h2Max {
			h2Max = hidden2[j]
		}
	}

	fmt.Printf("\nStartpos hidden1: active=%d/%d saturated=%d min=%.4f max=%.4f\n",
		active1, NNUEHiddenSize*2, saturated1, h1Min, h1Max)
	fmt.Printf("Startpos hidden2: active=%d/%d saturated=%d min=%.4f max=%.4f\n",
		active2, NNUEHidden2Size, saturated2, h2Min, h2Max)

	// Hidden3 stats
	var active3, saturated3 int
	var h3Min, h3Max float32
	for j := 0; j < NNUEHidden3Size; j++ {
		if hidden3[j] > 0 {
			active3++
		}
		if hidden3[j] >= 1.0 {
			saturated3++
		}
		if hidden3[j] < h3Min {
			h3Min = hidden3[j]
		}
		if hidden3[j] > h3Max {
			h3Max = hidden3[j]
		}
	}
	fmt.Printf("Startpos hidden3: active=%d/%d saturated=%d min=%.4f max=%.4f\n",
		active3, NNUEHidden3Size, saturated3, h3Min, h3Max)

	// Check if quantization introduces error
	// For the accumulator values at startpos
	accStack := NewNNUEAccumulatorStack(8)
	b.NNUEAcc = accStack
	b.NNUENet = net
	net.RecomputeAccumulator(accStack.Current(), &b)

	// Compare float accumulator values vs quantized
	var maxAccErr float64
	for i := 0; i < NNUEHiddenSize; i++ {
		floatVal := float64(0)
		for j := 0; j < NNUEHiddenSize; j++ {
			_ = j
		}
		// The quantized accumulator / 127 should ≈ the pre-CReLU accumulator value
		quantVal := float64(accStack.Current().White[i]) / float64(nnueInputScale)
		// Recompute float accumulator manually
		floatAcc := float64(trainNet.InputBiases[i])
		for _, idx := range sample.WhiteFeatures {
			floatAcc += float64(trainNet.InputWeights[idx][i])
		}
		err := math.Abs(floatAcc - quantVal)
		if err > maxAccErr {
			maxAccErr = err
		}
		_ = floatVal
	}
	fmt.Printf("\nMax accumulator quantization error: %.6f\n", maxAccErr)
}

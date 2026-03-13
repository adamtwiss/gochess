package chess

import (
	"math"
	"math/rand"
	"testing"
)

func TestParseNNUETrainData(t *testing.T) {
	// FEN;result format
	line := "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1;0.5"
	sample, err := ParseNNUETrainData(line)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if sample.SideToMove != Black {
		t.Error("expected Black to move")
	}
	if sample.Result != 0.5 {
		t.Errorf("expected result 0.5, got %f", sample.Result)
	}
	if sample.HasScore {
		t.Error("should not have score")
	}
	if len(sample.WhiteFeatures) == 0 {
		t.Error("expected non-empty white features")
	}
	if len(sample.BlackFeatures) == 0 {
		t.Error("expected non-empty black features")
	}
}

func TestParseNNUETrainDataWithScore(t *testing.T) {
	// FEN;score;result format
	line := "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1;35;0.5"
	sample, err := ParseNNUETrainData(line)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if !sample.HasScore {
		t.Error("should have score")
	}
	if sample.Score != 35 {
		t.Errorf("expected score 35, got %f", sample.Score)
	}
	if sample.Result != 0.5 {
		t.Errorf("expected result 0.5, got %f", sample.Result)
	}
}

func TestNNUETrainNetForward(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	net := NewNNUETrainNet(rng)

	sample := &NNUETrainSample{
		WhiteFeatures: []int{0, 100, 200, 300},
		BlackFeatures: []int{50, 150, 250, 350},
		SideToMove:    White,
		Result:        0.5,
	}

	output, _, _, _ := net.Forward(sample)

	// Should produce a finite value
	if math.IsNaN(float64(output)) || math.IsInf(float64(output), 0) {
		t.Errorf("forward pass produced invalid output: %f", output)
	}
}

func TestNNUEQuantization(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	trainNet := NewNNUETrainNet(rng)

	// Make weights larger so quantization error is relatively small
	for i := range trainNet.InputBiases {
		trainNet.InputBiases[i] = float32(rng.Float64()*2 - 1)
	}

	infNet := QuantizeNetwork(trainNet)

	// Verify biases are roughly correct
	for i := range trainNet.InputBiases {
		expected := int16(math.Round(float64(trainNet.InputBiases[i]) * float64(nnueInputScale)))
		if infNet.InputBiases[i] != expected {
			t.Errorf("InputBiases[%d]: expected %d, got %d", i, expected, infNet.InputBiases[i])
			break
		}
	}
}

func TestQuantizeRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	trainNet := NewNNUETrainNet(rng)

	// Set up a position and evaluate with both nets
	var b Board
	b.Reset()

	// Quantize
	infNet := QuantizeNetwork(trainNet)

	// Create a sample from the board
	sample := makeSampleFromBoard(&b, White)

	// Evaluate with training net
	trainOutput, _, _, _ := trainNet.Forward(sample)

	// Evaluate with inference net
	acc := &NNUEAccumulator{}
	infNet.RecomputeAccumulator(acc, &b)
	infOutput := infNet.Evaluate(acc, White, b.AllPieces.Count())

	// They should be roughly similar (quantization error)
	diff := math.Abs(float64(trainOutput) - float64(infOutput))
	// Allow generous tolerance due to quantization
	if diff > 100 {
		t.Errorf("quantization error too large: train=%.1f inf=%d diff=%.1f",
			trainOutput, infOutput, diff)
	}
	t.Logf("Train output: %.1f, Inference output: %d, Diff: %.1f", trainOutput, infOutput, diff)
}

func TestQuantizeRoundTripLargeWeights(t *testing.T) {
	// Test with larger weights that exercise the CReLU boundaries.
	// This catches the 127x scaling bug where training clips at 127.0
	// but inference clips at 1.0 (represented as 127 in int16).
	rng := rand.New(rand.NewSource(99))
	trainNet := NewNNUETrainNet(rng)

	// Scale up weights so accumulator outputs are in the [0, 1] CReLU active range
	for i := range trainNet.InputWeights {
		for j := range trainNet.InputWeights[i] {
			trainNet.InputWeights[i][j] *= 100
		}
	}
	for i := range trainNet.InputBiases {
		trainNet.InputBiases[i] = 0.5
	}
	// Scale hidden and output weights to produce meaningful outputs
	for i := range trainNet.HiddenWeights {
		for j := range trainNet.HiddenWeights[i] {
			trainNet.HiddenWeights[i][j] *= 10
		}
	}
	for bk := 0; bk < NNUEOutputBuckets; bk++ {
		for j := 0; j < NNUEHidden3Size; j++ {
			trainNet.OutputWeights[bk][j] *= 10
		}
	}

	var b Board
	b.Reset()

	sample := makeSampleFromBoard(&b, White)

	trainOutput, _, _, _ := trainNet.Forward(sample)
	t.Logf("Training forward output: %.1f", trainOutput)

	infNet := QuantizeNetwork(trainNet)
	acc := &NNUEAccumulator{}
	infNet.RecomputeAccumulator(acc, &b)
	infOutput := infNet.Evaluate(acc, White, b.AllPieces.Count())
	t.Logf("Inference output: %d", infOutput)

	diff := math.Abs(float64(trainOutput) - float64(infOutput))
	t.Logf("Diff: %.1f", diff)

	// With correct scaling, the difference should be small (quantization error only).
	// Before the fix, this would be off by ~127x.
	if diff > 50 {
		t.Errorf("quantization error too large: train=%.1f inf=%d diff=%.1f",
			trainOutput, infOutput, diff)
	}
}

func TestDequantizeRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	original := NewNNUETrainNet(rng)

	// Quantize then dequantize
	quantized := QuantizeNetwork(original)
	restored := DequantizeNetwork(quantized)

	// Check input weights round-trip
	maxInputDiff := float32(0)
	for i := 0; i < 100; i++ { // spot check first 100 features
		for j := 0; j < NNUEHiddenSize; j++ {
			diff := original.InputWeights[i][j] - restored.InputWeights[i][j]
			if diff < 0 {
				diff = -diff
			}
			if diff > maxInputDiff {
				maxInputDiff = diff
			}
		}
	}
	// Quantization error should be < 1/127 ≈ 0.008
	if maxInputDiff > 0.01 {
		t.Errorf("input weight round-trip error too large: %.6f", maxInputDiff)
	}
	t.Logf("Max input weight round-trip error: %.6f", maxInputDiff)

	// Check output weights round-trip
	maxOutputDiff := float32(0)
	for bk := 0; bk < NNUEOutputBuckets; bk++ {
		for j := 0; j < NNUEHidden3Size; j++ {
			diff := original.OutputWeights[bk][j] - restored.OutputWeights[bk][j]
			if diff < 0 {
				diff = -diff
			}
			if diff > maxOutputDiff {
				maxOutputDiff = diff
			}
		}
	}
	// Quantization error should be < 1/64 ≈ 0.016
	if maxOutputDiff > 0.02 {
		t.Errorf("output weight round-trip error too large: %.6f", maxOutputDiff)
	}
	t.Logf("Max output weight round-trip error: %.6f", maxOutputDiff)
}

// makeSampleFromBoard creates a training sample from a board position.
func makeSampleFromBoard(b *Board, stm Color) *NNUETrainSample {
	wKingSq := b.Pieces[WhiteKing].LSB()
	bKingSq := b.Pieces[BlackKing].LSB()
	sample := &NNUETrainSample{
		SideToMove: stm,
		Result:     0.5,
		PieceCount: b.AllPieces.Count(),
	}
	for piece := WhitePawn; piece <= BlackKing; piece++ {
		bb := b.Pieces[piece]
		for bb != 0 {
			sq := bb.PopLSB()
			wIdx := HalfKAIndex(White, wKingSq, piece, sq)
			bIdx := HalfKAIndex(Black, bKingSq, piece, sq)
			if wIdx >= 0 {
				sample.WhiteFeatures = append(sample.WhiteFeatures, wIdx)
			}
			if bIdx >= 0 {
				sample.BlackFeatures = append(sample.BlackFeatures, bIdx)
			}
		}
	}
	return sample
}

package chess

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
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

func TestNNUETrainSmallConvergence(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	// Create a small training dataset with known patterns
	dir := t.TempDir()
	dataFile := filepath.Join(dir, "train.dat")
	binFile := filepath.Join(dir, "train.nnbin")

	// Write some training data (startpos with different results)
	f, err := os.Create(dataFile)
	if err != nil {
		t.Fatal(err)
	}
	// Generate diverse positions
	positions := []string{
		"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1;50;1.0",
		"rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1;30;1.0",
		"rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1;20;0.5",
		"rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2;10;0.5",
		"r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2;25;1.0",
		"rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2;40;1.0",
		"rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2;15;0.5",
		"rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1;5;0.0",
		"rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1;10;0.0",
		"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3;0;0.5",
	}
	for _, pos := range positions {
		f.WriteString(pos + "\n")
	}
	f.Close()

	// Preprocess
	numTrain, numVal, err := PreprocessNNUEToFile(dataFile, binFile)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Preprocessed: %d train, %d val", numTrain, numVal)

	// Open binary file
	bf, err := OpenNNBinFile(binFile)
	if err != nil {
		t.Fatal(err)
	}
	defer bf.Close()

	// Train for a few epochs
	trainer := NewNNUETrainer(42)
	cfg := NNUETrainConfig{
		Epochs:    10,
		LR:        0.001,
		BatchSize: 5,
		Lambda:    0.5,
		K:         400.0,
	}

	var losses []float64
	trainer.Train(bf, cfg, func(epoch int, trainLoss, valLoss float64) {
		losses = append(losses, trainLoss)
		t.Logf("Epoch %d: train=%.6f val=%.6f", epoch, trainLoss, valLoss)
	})

	if len(losses) < 2 {
		t.Fatal("expected at least 2 epochs")
	}

	// Loss should generally decrease (or at least not explode)
	lastLoss := losses[len(losses)-1]
	if math.IsNaN(lastLoss) || math.IsInf(lastLoss, 0) {
		t.Errorf("training produced invalid loss: %f", lastLoss)
	}
}

func TestNNUEPreprocessAndRead(t *testing.T) {
	dir := t.TempDir()
	dataFile := filepath.Join(dir, "test.dat")
	binFile := filepath.Join(dir, "test.nnbin")

	// Write test data
	f, err := os.Create(dataFile)
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1;0.5\n")
	f.WriteString("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1;1.0\n")
	f.WriteString("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1;0.0\n")
	f.Close()

	numTrain, numVal, err := PreprocessNNUEToFile(dataFile, binFile)
	if err != nil {
		t.Fatal(err)
	}

	total := numTrain + numVal
	if total != 3 {
		t.Errorf("expected 3 total records, got %d", total)
	}

	bf, err := OpenNNBinFile(binFile)
	if err != nil {
		t.Fatal(err)
	}
	defer bf.Close()

	// Read all records
	for i := 0; i < int(total); i++ {
		s, err := bf.ReadRecord(i)
		if err != nil {
			t.Fatalf("error reading record %d: %v", i, err)
		}
		if len(s.WhiteFeatures) == 0 {
			t.Errorf("record %d has no white features", i)
		}
		if s.Result != 0.0 && s.Result != 0.5 && s.Result != 1.0 {
			t.Errorf("record %d has unexpected result: %f", i, s.Result)
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

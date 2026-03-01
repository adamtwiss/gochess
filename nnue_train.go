package chess

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// NNUETrainNet holds float32 weights for training (higher precision than int16 inference).
type NNUETrainNet struct {
	InputWeights  [NNUEInputSize][NNUEHiddenSize]float32
	InputBiases   [NNUEHiddenSize]float32
	HiddenWeights [NNUEHiddenSize * 2][NNUEHidden2Size]float32
	HiddenBiases  [NNUEHidden2Size]float32
	OutputWeights [NNUEHidden2Size]float32
	OutputBias    float32
}

// NNUETrainConfig holds training hyperparameters.
type NNUETrainConfig struct {
	Epochs       int
	LR           float64
	BatchSize    int
	Lambda       float64 // weight for result vs score: loss = lambda*MSE(result) + (1-lambda)*MSE(score)
	K            float64 // sigmoid scaling constant
	MaxPositions int     // limit training positions per epoch (0=use all)
}

// DefaultNNUETrainConfig returns sensible defaults.
func DefaultNNUETrainConfig() NNUETrainConfig {
	return NNUETrainConfig{
		Epochs:    100,
		LR:        0.01,
		BatchSize: 16384,
		Lambda:    0.5,
		K:         400.0,
	}
}

// NNUETrainSample represents a parsed training position.
type NNUETrainSample struct {
	WhiteFeatures []int     // active HalfKP feature indices for White perspective
	BlackFeatures []int     // active HalfKP feature indices for Black perspective
	SideToMove    Color     // side to move
	Result        float32   // game result from White's perspective (1.0, 0.5, 0.0)
	Score         float32   // search score in centipawns (White-relative)
	HasScore      bool      // whether Score is valid
}

// nnbin file format constants
const (
	nnbinMagic   = uint32(0x4E4E424E) // "NNBN"
	nnbinVersion = uint16(1)
)

// NNBinHeader is the header for the binary training data cache.
type NNBinHeader struct {
	Magic         uint32
	Version       uint16
	NumTrain      uint32
	NumValidation uint32
}

// NNBinFile represents an opened binary training data file.
type NNBinFile struct {
	file          *os.File
	NumTrain      uint32
	NumValidation uint32
	headerSize    int64
	trainOffset   int64
	valOffset     int64
	recordOffsets []int64 // byte offset of each record
}

// NewNNUETrainNet creates a randomly initialized training network.
func NewNNUETrainNet(rng *rand.Rand) *NNUETrainNet {
	net := &NNUETrainNet{}

	// Input layer initialization scaled for CReLU [0, 1.0] with ~30 active features.
	// Target accumulator std ~0.4 so most positive values stay in (0, 1) without
	// saturating at the CReLU upper bound, giving ~49% of neurons active gradient
	// flow (vs ~26% with full He init where std ~1.4 causes heavy saturation).
	inputScale := float32(0.4 / math.Sqrt(30.0))
	for i := range net.InputWeights {
		for j := range net.InputWeights[i] {
			net.InputWeights[i][j] = float32(rng.NormFloat64()) * inputScale
		}
	}
	// Biases start at zero
	for i := range net.InputBiases {
		net.InputBiases[i] = 0
	}

	// Hidden layer: fan_in = 512 (concatenated accumulators)
	hiddenScale := float32(math.Sqrt(2.0 / 512.0))
	for i := range net.HiddenWeights {
		for j := range net.HiddenWeights[i] {
			net.HiddenWeights[i][j] = float32(rng.NormFloat64()) * hiddenScale
		}
	}
	for i := range net.HiddenBiases {
		net.HiddenBiases[i] = 0
	}

	// Output layer: fan_in = 32
	outputScale := float32(math.Sqrt(2.0 / 32.0))
	for i := range net.OutputWeights {
		net.OutputWeights[i] = float32(rng.NormFloat64()) * outputScale
	}
	net.OutputBias = 0

	return net
}

// Forward computes the network output and stores intermediates for backprop.
// Returns the raw output (before sigmoid) in centipawns, side-to-move relative.
func (net *NNUETrainNet) Forward(sample *NNUETrainSample) (output float32, hidden1 [NNUEHiddenSize * 2]float32, hidden2 [NNUEHidden2Size]float32) {
	// Accumulator: bias + sum of active feature weights
	var stmAcc, oppAcc [NNUEHiddenSize]float32

	var stmFeatures, oppFeatures []int
	if sample.SideToMove == White {
		stmFeatures = sample.WhiteFeatures
		oppFeatures = sample.BlackFeatures
	} else {
		stmFeatures = sample.BlackFeatures
		oppFeatures = sample.WhiteFeatures
	}

	// Initialize with biases
	for i := 0; i < NNUEHiddenSize; i++ {
		stmAcc[i] = net.InputBiases[i]
		oppAcc[i] = net.InputBiases[i]
	}

	// Add active features
	for _, idx := range stmFeatures {
		for i := 0; i < NNUEHiddenSize; i++ {
			stmAcc[i] += net.InputWeights[idx][i]
		}
	}
	for _, idx := range oppFeatures {
		for i := 0; i < NNUEHiddenSize; i++ {
			oppAcc[i] += net.InputWeights[idx][i]
		}
	}

	// ClippedReLU and store in hidden1
	for i := 0; i < NNUEHiddenSize; i++ {
		hidden1[i] = clippedReLUf(stmAcc[i])
		hidden1[NNUEHiddenSize+i] = clippedReLUf(oppAcc[i])
	}

	// Hidden layer
	for j := 0; j < NNUEHidden2Size; j++ {
		sum := net.HiddenBiases[j]
		for i := 0; i < NNUEHiddenSize*2; i++ {
			sum += hidden1[i] * net.HiddenWeights[i][j]
		}
		hidden2[j] = sum
	}

	// Output layer with ClippedReLU on hidden2
	output = net.OutputBias
	for j := 0; j < NNUEHidden2Size; j++ {
		output += clippedReLUf(hidden2[j]) * net.OutputWeights[j]
	}

	return output, hidden1, hidden2
}

// nnueFloatClipMax is the CReLU upper bound in float training space.
// The quantized inference clips at nnueClipMax (127) in int16 space,
// which represents 1.0 in float space. Training must clip at 1.0 to match.
const nnueFloatClipMax = float32(1.0)

func clippedReLUf(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > nnueFloatClipMax {
		return nnueFloatClipMax
	}
	return x
}

func clippedReLUGrad(x float32) float32 {
	if x > 0 && x < nnueFloatClipMax {
		return 1
	}
	return 0
}

func nnueSigmoid(x float64, K float64) float64 {
	return 1.0 / (1.0 + math.Pow(10.0, -x/K))
}

// NNUETrainGradients accumulates gradients for all network weights.
type NNUETrainGradients struct {
	InputWeights  [NNUEInputSize][NNUEHiddenSize]float32
	InputBiases   [NNUEHiddenSize]float32
	HiddenWeights [NNUEHiddenSize * 2][NNUEHidden2Size]float32
	HiddenBiases  [NNUEHidden2Size]float32
	OutputWeights [NNUEHidden2Size]float32
	OutputBias    float32
}

// Backward computes gradients for a single sample and accumulates into grads.
func (net *NNUETrainNet) Backward(sample *NNUETrainSample, grads *NNUETrainGradients,
	output float32, hidden1 [NNUEHiddenSize * 2]float32, hidden2 [NNUEHidden2Size]float32,
	cfg NNUETrainConfig) {

	// Compute loss gradient: d(loss)/d(output)
	pred := nnueSigmoid(float64(output), cfg.K)

	var target float64
	if sample.HasScore {
		// Blended target: lambda * result + (1-lambda) * nnueSigmoid(score/K)
		scoreTarget := nnueSigmoid(float64(sample.Score), cfg.K)
		// Convert result to side-to-move relative
		result := float64(sample.Result)
		if sample.SideToMove == Black {
			result = 1.0 - result
		}
		target = cfg.Lambda*result + (1.0-cfg.Lambda)*scoreTarget
	} else {
		// Result only
		target = float64(sample.Result)
		if sample.SideToMove == Black {
			target = 1.0 - target
		}
	}

	// d(loss)/d(output) = 2 * (pred - target) * pred * (1 - pred) * ln(10) / K
	dOutput := float32(2.0 * (pred - target) * pred * (1.0 - pred) * math.Ln10 / cfg.K)

	// Output layer gradients
	grads.OutputBias += dOutput
	var dHidden2 [NNUEHidden2Size]float32
	for j := 0; j < NNUEHidden2Size; j++ {
		h2 := clippedReLUf(hidden2[j])
		grads.OutputWeights[j] += dOutput * h2
		dHidden2[j] = dOutput * net.OutputWeights[j] * clippedReLUGrad(hidden2[j])
	}

	// Hidden layer gradients
	for j := 0; j < NNUEHidden2Size; j++ {
		grads.HiddenBiases[j] += dHidden2[j]
		for i := 0; i < NNUEHiddenSize*2; i++ {
			grads.HiddenWeights[i][j] += dHidden2[j] * hidden1[i]
		}
	}

	// Input layer gradients (sparse — only active features)
	var dHidden1 [NNUEHiddenSize * 2]float32
	for i := 0; i < NNUEHiddenSize*2; i++ {
		var sum float32
		for j := 0; j < NNUEHidden2Size; j++ {
			sum += dHidden2[j] * net.HiddenWeights[i][j]
		}
		dHidden1[i] = sum
	}

	var stmFeatures, oppFeatures []int
	if sample.SideToMove == White {
		stmFeatures = sample.WhiteFeatures
		oppFeatures = sample.BlackFeatures
	} else {
		stmFeatures = sample.BlackFeatures
		oppFeatures = sample.WhiteFeatures
	}

	// STM perspective accumulator gradients
	for _, idx := range stmFeatures {
		for i := 0; i < NNUEHiddenSize; i++ {
			// dL/dAcc[i] * dAcc[i]/dW = dHidden1[i] * clippedReLUGrad(acc[i])
			// Since hidden1[i] = clippedReLU(acc[i]), we need the pre-activation
			// But we stored the post-activation in hidden1. Use the grad of the clipped version.
			grad := dHidden1[i] * clippedReLUGrad(hidden1[i])
			grads.InputWeights[idx][i] += grad
			grads.InputBiases[i] += grad
		}
	}

	// Opponent perspective accumulator gradients
	for _, idx := range oppFeatures {
		for i := 0; i < NNUEHiddenSize; i++ {
			grad := dHidden1[NNUEHiddenSize+i] * clippedReLUGrad(hidden1[NNUEHiddenSize+i])
			grads.InputWeights[idx][i] += grad
			grads.InputBiases[i] += grad
		}
	}
}

// ParseNNUETrainData parses a FEN;result or FEN;score;result line into a training sample.
func ParseNNUETrainData(line string) (*NNUETrainSample, error) {
	parts := strings.Split(line, ";")
	if len(parts) < 2 {
		return nil, fmt.Errorf("invalid format: %q", line)
	}

	var fen string
	var result float32
	var score float32
	var hasScore bool

	if len(parts) >= 3 {
		// FEN;score;result format
		fen = strings.TrimSpace(parts[0])
		s, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 32)
		if err != nil {
			return nil, fmt.Errorf("invalid score: %q", parts[1])
		}
		score = float32(s)
		hasScore = true
		r, err := strconv.ParseFloat(strings.TrimSpace(parts[2]), 32)
		if err != nil {
			return nil, fmt.Errorf("invalid result: %q", parts[2])
		}
		result = float32(r)
	} else {
		// FEN;result format
		fen = strings.TrimSpace(parts[0])
		r, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 32)
		if err != nil {
			return nil, fmt.Errorf("invalid result: %q", parts[1])
		}
		result = float32(r)
	}

	var b Board
	if err := b.SetFEN(fen); err != nil {
		return nil, fmt.Errorf("invalid FEN: %w", err)
	}

	sample := &NNUETrainSample{
		SideToMove: b.SideToMove,
		Result:     result,
		Score:      score,
		HasScore:   hasScore,
	}

	// Extract active features
	wKingSq := b.Pieces[WhiteKing].LSB()
	bKingSq := b.Pieces[BlackKing].LSB()

	for piece := WhitePawn; piece <= BlackQueen; piece++ {
		if piece == WhiteKing || piece == BlackKing {
			continue
		}
		bb := b.Pieces[piece]
		for bb != 0 {
			sq := bb.PopLSB()
			wIdx := HalfKPIndex(White, wKingSq, piece, sq)
			bIdx := HalfKPIndex(Black, bKingSq, piece, sq)
			if wIdx >= 0 {
				sample.WhiteFeatures = append(sample.WhiteFeatures, wIdx)
			}
			if bIdx >= 0 {
				sample.BlackFeatures = append(sample.BlackFeatures, bIdx)
			}
		}
	}

	return sample, nil
}

// NNBinRecord is the on-disk representation of one training sample.
type NNBinRecord struct {
	SideToMove byte    // 0=White, 1=Black
	Result     float32 // game result (White-relative)
	Score      float32 // search score (White-relative, 0 if not available)
	HasScore   byte    // 1 if Score is valid
	NumWhite   uint16  // number of White perspective features
	NumBlack   uint16  // number of Black perspective features
	// Followed by NumWhite + NumBlack uint16 feature indices
}

// PreprocessNNUEToFile reads FEN training data and writes a binary cache.
func PreprocessNNUEToFile(dataFile, binFile string) (numTrain, numVal uint32, err error) {
	// Read all lines
	f, err := os.Open(dataFile)
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()

	var lines []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		lines = append(lines, line)
	}
	if err := scanner.Err(); err != nil {
		return 0, 0, err
	}

	// Shuffle deterministically
	rng := rand.New(rand.NewSource(42))
	rng.Shuffle(len(lines), func(i, j int) {
		lines[i], lines[j] = lines[j], lines[i]
	})

	// 90/10 split
	total := len(lines)
	numTrain = uint32(total * 9 / 10)
	numVal = uint32(total) - numTrain

	// Write binary file
	out, err := os.Create(binFile)
	if err != nil {
		return 0, 0, err
	}
	defer out.Close()

	w := bufio.NewWriter(out)

	// Header
	header := NNBinHeader{
		Magic:         nnbinMagic,
		Version:       nnbinVersion,
		NumTrain:      numTrain,
		NumValidation: numVal,
	}
	if err := binary.Write(w, binary.LittleEndian, &header); err != nil {
		return 0, 0, fmt.Errorf("writing header: %w", err)
	}

	// Write records
	for _, line := range lines {
		sample, err := ParseNNUETrainData(line)
		if err != nil {
			continue // skip bad lines
		}
		if err := writeNNBinRecord(w, sample); err != nil {
			return 0, 0, err
		}
	}

	if err := w.Flush(); err != nil {
		return 0, 0, err
	}

	return numTrain, numVal, nil
}

func writeNNBinRecord(w io.Writer, sample *NNUETrainSample) error {
	var stm byte
	if sample.SideToMove == Black {
		stm = 1
	}
	var hs byte
	if sample.HasScore {
		hs = 1
	}

	rec := NNBinRecord{
		SideToMove: stm,
		Result:     sample.Result,
		Score:      sample.Score,
		HasScore:   hs,
		NumWhite:   uint16(len(sample.WhiteFeatures)),
		NumBlack:   uint16(len(sample.BlackFeatures)),
	}
	if err := binary.Write(w, binary.LittleEndian, &rec); err != nil {
		return err
	}

	// Write feature indices
	for _, idx := range sample.WhiteFeatures {
		if err := binary.Write(w, binary.LittleEndian, uint16(idx)); err != nil {
			return err
		}
	}
	for _, idx := range sample.BlackFeatures {
		if err := binary.Write(w, binary.LittleEndian, uint16(idx)); err != nil {
			return err
		}
	}
	return nil
}

// OpenNNBinFile opens a preprocessed binary training data file.
func OpenNNBinFile(path string) (*NNBinFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	var header NNBinHeader
	if err := binary.Read(f, binary.LittleEndian, &header); err != nil {
		f.Close()
		return nil, fmt.Errorf("reading header: %w", err)
	}
	if header.Magic != nnbinMagic {
		f.Close()
		return nil, fmt.Errorf("invalid magic: 0x%X", header.Magic)
	}
	if header.Version != nnbinVersion {
		f.Close()
		return nil, fmt.Errorf("unsupported version: %d", header.Version)
	}

	headerSize := int64(binary.Size(header))

	// Build record offset index by scanning through the file
	total := int(header.NumTrain + header.NumValidation)
	offsets := make([]int64, 0, total)

	pos := headerSize
	for i := 0; i < total; i++ {
		offsets = append(offsets, pos)

		var rec NNBinRecord
		if err := binary.Read(f, binary.LittleEndian, &rec); err != nil {
			f.Close()
			return nil, fmt.Errorf("scanning record %d: %w", i, err)
		}
		featureBytes := int64(rec.NumWhite+rec.NumBlack) * 2
		pos += int64(binary.Size(rec)) + featureBytes

		// Skip feature data
		if _, err := f.Seek(featureBytes, io.SeekCurrent); err != nil {
			f.Close()
			return nil, fmt.Errorf("seeking past features at record %d: %w", i, err)
		}
	}

	return &NNBinFile{
		file:          f,
		NumTrain:      header.NumTrain,
		NumValidation: header.NumValidation,
		headerSize:    headerSize,
		trainOffset:   headerSize,
		recordOffsets: offsets,
	}, nil
}

// Close closes the binary file.
func (bf *NNBinFile) Close() error {
	return bf.file.Close()
}

// ReadRecord reads a single training sample at the given record index.
func (bf *NNBinFile) ReadRecord(index int) (*NNUETrainSample, error) {
	if index < 0 || index >= len(bf.recordOffsets) {
		return nil, fmt.Errorf("record index %d out of range [0, %d)", index, len(bf.recordOffsets))
	}

	if _, err := bf.file.Seek(bf.recordOffsets[index], io.SeekStart); err != nil {
		return nil, err
	}

	var rec NNBinRecord
	if err := binary.Read(bf.file, binary.LittleEndian, &rec); err != nil {
		return nil, err
	}

	sample := &NNUETrainSample{
		SideToMove: Color(rec.SideToMove),
		Result:     rec.Result,
		Score:      rec.Score,
		HasScore:   rec.HasScore != 0,
	}

	// Read feature indices
	sample.WhiteFeatures = make([]int, rec.NumWhite)
	for i := 0; i < int(rec.NumWhite); i++ {
		var idx uint16
		if err := binary.Read(bf.file, binary.LittleEndian, &idx); err != nil {
			return nil, err
		}
		sample.WhiteFeatures[i] = int(idx)
	}

	sample.BlackFeatures = make([]int, rec.NumBlack)
	for i := 0; i < int(rec.NumBlack); i++ {
		var idx uint16
		if err := binary.Read(bf.file, binary.LittleEndian, &idx); err != nil {
			return nil, err
		}
		sample.BlackFeatures[i] = int(idx)
	}

	return sample, nil
}

// StreamBatch reads a batch of samples from the given index range.
func (bf *NNBinFile) StreamBatch(startIdx, count int) ([]*NNUETrainSample, error) {
	end := startIdx + count
	if end > len(bf.recordOffsets) {
		end = len(bf.recordOffsets)
	}

	samples := make([]*NNUETrainSample, 0, end-startIdx)
	for i := startIdx; i < end; i++ {
		s, err := bf.ReadRecord(i)
		if err != nil {
			return nil, err
		}
		samples = append(samples, s)
	}
	return samples, nil
}

// Adam optimizer state for a single parameter.
type nnueAdamState struct {
	m float64 // first moment
	v float64 // second moment
}

// NNUETrainer holds the training network and optimizer state.
type NNUETrainer struct {
	Net  *NNUETrainNet
	adam struct {
		inputWeights  [NNUEInputSize][NNUEHiddenSize]nnueAdamState
		inputBiases   [NNUEHiddenSize]nnueAdamState
		hiddenWeights [NNUEHiddenSize * 2][NNUEHidden2Size]nnueAdamState
		hiddenBiases  [NNUEHidden2Size]nnueAdamState
		outputWeights [NNUEHidden2Size]nnueAdamState
		outputBias    nnueAdamState
	}
	step int
}

// NewNNUETrainer creates a new trainer with a randomly initialized network.
func NewNNUETrainer(seed int64) *NNUETrainer {
	rng := rand.New(rand.NewSource(seed))
	return &NNUETrainer{
		Net: NewNNUETrainNet(rng),
	}
}

// adamUpdate applies one Adam step to a parameter.
func adamUpdate(param *float32, grad float64, state *nnueAdamState, lr float64, step int) {
	const (
		beta1 = 0.9
		beta2 = 0.999
		eps   = 1e-8
	)

	state.m = beta1*state.m + (1-beta1)*grad
	state.v = beta2*state.v + (1-beta2)*grad*grad

	mHat := state.m / (1 - math.Pow(beta1, float64(step)))
	vHat := state.v / (1 - math.Pow(beta2, float64(step)))

	*param -= float32(lr * mHat / (math.Sqrt(vHat) + eps))
}

// Train runs the NNUE training loop.
func (trainer *NNUETrainer) Train(bf *NNBinFile, cfg NNUETrainConfig,
	onEpoch func(epoch int, trainLoss, valLoss float64)) {

	numTrain := int(bf.NumTrain)
	if cfg.MaxPositions > 0 && cfg.MaxPositions < numTrain {
		numTrain = cfg.MaxPositions
	}
	numWorkers := runtime.NumCPU()
	if numWorkers > 8 {
		numWorkers = 8
	}

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		// Shuffle training indices
		indices := make([]int, numTrain)
		for i := range indices {
			indices[i] = i
		}
		rand.Shuffle(numTrain, func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})

		totalLoss := 0.0
		numSamples := 0

		for batchStart := 0; batchStart < numTrain; batchStart += cfg.BatchSize {
			batchEnd := batchStart + cfg.BatchSize
			if batchEnd > numTrain {
				batchEnd = numTrain
			}
			batchIndices := indices[batchStart:batchEnd]
			batchSize := len(batchIndices)

			// Read batch
			samples := make([]*NNUETrainSample, batchSize)
			for i, idx := range batchIndices {
				s, err := bf.ReadRecord(idx)
				if err != nil {
					continue
				}
				samples[i] = s
			}

			// Parallel gradient computation
			perWorker := (batchSize + numWorkers - 1) / numWorkers
			workerGrads := make([]NNUETrainGradients, numWorkers)
			workerLoss := make([]float64, numWorkers)
			workerCount := make([]int, numWorkers)

			var wg sync.WaitGroup
			for w := 0; w < numWorkers; w++ {
				wg.Add(1)
				go func(workerIdx int) {
					defer wg.Done()
					start := workerIdx * perWorker
					end := start + perWorker
					if end > batchSize {
						end = batchSize
					}

					for i := start; i < end; i++ {
						if samples[i] == nil {
							continue
						}
						output, hidden1, hidden2 := trainer.Net.Forward(samples[i])
						trainer.Net.Backward(samples[i], &workerGrads[workerIdx],
							output, hidden1, hidden2, cfg)

						// Compute loss
						pred := nnueSigmoid(float64(output), cfg.K)
						var target float64
						if samples[i].HasScore {
							result := float64(samples[i].Result)
							if samples[i].SideToMove == Black {
								result = 1.0 - result
							}
							scoreTarget := nnueSigmoid(float64(samples[i].Score), cfg.K)
							target = cfg.Lambda*result + (1.0-cfg.Lambda)*scoreTarget
						} else {
							target = float64(samples[i].Result)
							if samples[i].SideToMove == Black {
								target = 1.0 - target
							}
						}
						diff := pred - target
						workerLoss[workerIdx] += diff * diff
						workerCount[workerIdx]++
					}
				}(w)
			}
			wg.Wait()

			// Aggregate gradients and loss
			var totalGrads NNUETrainGradients
			for w := 0; w < numWorkers; w++ {
				totalLoss += workerLoss[w]
				numSamples += workerCount[w]
				addGradients(&totalGrads, &workerGrads[w])
			}

			// Scale gradients by batch size
			scale := 1.0 / float64(batchSize)
			trainer.step++

			// Apply Adam updates
			trainer.applyAdamUpdates(&totalGrads, scale, cfg.LR)
		}

		// Compute losses
		trainLoss := totalLoss / float64(numSamples)

		valLoss := 0.0
		if bf.NumValidation > 0 {
			valLoss = trainer.computeValidationLoss(bf, cfg)
		}

		if onEpoch != nil {
			onEpoch(epoch, trainLoss, valLoss)
		}
	}
}

func addGradients(dst, src *NNUETrainGradients) {
	dst.OutputBias += src.OutputBias
	for j := 0; j < NNUEHidden2Size; j++ {
		dst.OutputWeights[j] += src.OutputWeights[j]
		dst.HiddenBiases[j] += src.HiddenBiases[j]
	}
	for i := 0; i < NNUEHiddenSize*2; i++ {
		for j := 0; j < NNUEHidden2Size; j++ {
			dst.HiddenWeights[i][j] += src.HiddenWeights[i][j]
		}
	}
	// Sparse input gradients — only non-zero entries
	for i := 0; i < NNUEInputSize; i++ {
		for j := 0; j < NNUEHiddenSize; j++ {
			if src.InputWeights[i][j] != 0 {
				dst.InputWeights[i][j] += src.InputWeights[i][j]
			}
		}
	}
	for j := 0; j < NNUEHiddenSize; j++ {
		dst.InputBiases[j] += src.InputBiases[j]
	}
}

func (trainer *NNUETrainer) applyAdamUpdates(grads *NNUETrainGradients, scale, lr float64) {
	step := trainer.step

	// Output layer
	adamUpdate(&trainer.Net.OutputBias, float64(grads.OutputBias)*scale,
		&trainer.adam.outputBias, lr, step)
	for j := 0; j < NNUEHidden2Size; j++ {
		adamUpdate(&trainer.Net.OutputWeights[j], float64(grads.OutputWeights[j])*scale,
			&trainer.adam.outputWeights[j], lr, step)
		adamUpdate(&trainer.Net.HiddenBiases[j], float64(grads.HiddenBiases[j])*scale,
			&trainer.adam.hiddenBiases[j], lr, step)
	}

	// Hidden layer
	for i := 0; i < NNUEHiddenSize*2; i++ {
		for j := 0; j < NNUEHidden2Size; j++ {
			adamUpdate(&trainer.Net.HiddenWeights[i][j], float64(grads.HiddenWeights[i][j])*scale,
				&trainer.adam.hiddenWeights[i][j], lr, step)
		}
	}

	// Input layer (sparse)
	for i := 0; i < NNUEInputSize; i++ {
		hasNonZero := false
		for j := 0; j < NNUEHiddenSize; j++ {
			if grads.InputWeights[i][j] != 0 {
				hasNonZero = true
				break
			}
		}
		if !hasNonZero {
			continue
		}
		for j := 0; j < NNUEHiddenSize; j++ {
			if grads.InputWeights[i][j] != 0 {
				adamUpdate(&trainer.Net.InputWeights[i][j], float64(grads.InputWeights[i][j])*scale,
					&trainer.adam.inputWeights[i][j], lr, step)
			}
		}
	}

	// Input biases
	for j := 0; j < NNUEHiddenSize; j++ {
		adamUpdate(&trainer.Net.InputBiases[j], float64(grads.InputBiases[j])*scale,
			&trainer.adam.inputBiases[j], lr, step)
	}
}

func (trainer *NNUETrainer) computeValidationLoss(bf *NNBinFile, cfg NNUETrainConfig) float64 {
	valStart := int(bf.NumTrain)
	valEnd := valStart + int(bf.NumValidation)

	totalLoss := 0.0
	count := 0

	for i := valStart; i < valEnd; i++ {
		s, err := bf.ReadRecord(i)
		if err != nil {
			continue
		}
		output, _, _ := trainer.Net.Forward(s)
		pred := nnueSigmoid(float64(output), cfg.K)

		var target float64
		if s.HasScore {
			result := float64(s.Result)
			if s.SideToMove == Black {
				result = 1.0 - result
			}
			scoreTarget := nnueSigmoid(float64(s.Score), cfg.K)
			target = cfg.Lambda*result + (1.0-cfg.Lambda)*scoreTarget
		} else {
			target = float64(s.Result)
			if s.SideToMove == Black {
				target = 1.0 - target
			}
		}

		diff := pred - target
		totalLoss += diff * diff
		count++
	}

	if count == 0 {
		return 0
	}
	return totalLoss / float64(count)
}

// QuantizeNetwork converts a float32 training network to int16 for inference.
func QuantizeNetwork(train *NNUETrainNet) *NNUENet {
	net := &NNUENet{}

	// Input layer: scale by nnueInputScale (127)
	for i := range train.InputWeights {
		for j := range train.InputWeights[i] {
			net.InputWeights[i][j] = int16(math.Round(float64(train.InputWeights[i][j]) * float64(nnueInputScale)))
		}
	}
	for j := range train.InputBiases {
		net.InputBiases[j] = int16(math.Round(float64(train.InputBiases[j]) * float64(nnueInputScale)))
	}

	// Hidden layer: scale by nnueHiddenScale (64)
	for i := range train.HiddenWeights {
		for j := range train.HiddenWeights[i] {
			net.HiddenWeights[i][j] = int16(math.Round(float64(train.HiddenWeights[i][j]) * float64(nnueHiddenScale)))
		}
	}
	for j := range train.HiddenBiases {
		// Biases are int32 and include the input scale already
		net.HiddenBiases[j] = int32(math.Round(float64(train.HiddenBiases[j]) * float64(nnueInputScale) * float64(nnueHiddenScale)))
	}

	// Output layer: bias is int32 at scale nnueOutputScale (8128).
	// Weights are int16 at scale nnueHiddenScale (64), NOT nnueOutputScale,
	// because the second CReLU activation (after >>6) is at scale nnueInputScale (127).
	// Product: activation(127) * weight(64) = scale 8128 = nnueOutputScale, matching bias.
	for j := range train.OutputWeights {
		net.OutputWeights[j] = int16(math.Round(float64(train.OutputWeights[j]) * float64(nnueHiddenScale)))
	}
	net.OutputBias = int32(math.Round(float64(train.OutputBias) * float64(nnueOutputScale)))

	net.PrepareWeights()
	return net
}

// DequantizeNetwork converts an int16 inference network back to float32 for training.
// This reverses QuantizeNetwork, enabling resumed training from a saved .nnue file.
func DequantizeNetwork(net *NNUENet) *NNUETrainNet {
	train := &NNUETrainNet{}

	// Input layer: divide by nnueInputScale (127)
	for i := range net.InputWeights {
		for j := range net.InputWeights[i] {
			train.InputWeights[i][j] = float32(net.InputWeights[i][j]) / nnueInputScale
		}
	}
	for j := range net.InputBiases {
		train.InputBiases[j] = float32(net.InputBiases[j]) / nnueInputScale
	}

	// Hidden weights: divide by nnueHiddenScale (64)
	for i := range net.HiddenWeights {
		for j := range net.HiddenWeights[i] {
			train.HiddenWeights[i][j] = float32(net.HiddenWeights[i][j]) / nnueHiddenScale
		}
	}

	// Hidden biases: divide by nnueInputScale * nnueHiddenScale (127*64 = 8128)
	for j := range net.HiddenBiases {
		train.HiddenBiases[j] = float32(net.HiddenBiases[j]) / nnueOutputScale
	}

	// Output weights: divide by nnueHiddenScale (64), matching QuantizeNetwork
	for j := range net.OutputWeights {
		train.OutputWeights[j] = float32(net.OutputWeights[j]) / nnueHiddenScale
	}
	// Output bias: divide by nnueOutputScale (8128)
	train.OutputBias = float32(net.OutputBias) / nnueOutputScale

	return train
}

// LoadWeights loads weights from a quantized inference network into the trainer.
// This enables resuming training from a previously saved .nnue file.
func (trainer *NNUETrainer) LoadWeights(net *NNUENet) {
	trainer.Net = DequantizeNetwork(net)
}

// TuneK finds the optimal sigmoid scaling constant K using golden section search.
func (trainer *NNUETrainer) TuneK(bf *NNBinFile, lambda float64) float64 {
	// Sample up to 50K positions from the start (data is pre-shuffled in .nnbin)
	numTrain := int(bf.NumTrain)
	sampleSize := numTrain
	if sampleSize > 50000 {
		sampleSize = 50000
	}

	computeError := func(K float64) float64 {
		totalLoss := 0.0
		count := 0
		for i := 0; i < sampleSize; i++ {
			s, err := bf.ReadRecord(i)
			if err != nil {
				continue
			}
			output, _, _ := trainer.Net.Forward(s)
			pred := nnueSigmoid(float64(output), K)

			var target float64
			if s.HasScore {
				result := float64(s.Result)
				if s.SideToMove == Black {
					result = 1.0 - result
				}
				scoreTarget := nnueSigmoid(float64(s.Score), K)
				target = lambda*result + (1.0-lambda)*scoreTarget
			} else {
				target = float64(s.Result)
				if s.SideToMove == Black {
					target = 1.0 - target
				}
			}

			diff := pred - target
			totalLoss += diff * diff
			count++
		}
		if count == 0 {
			return 0
		}
		return totalLoss / float64(count)
	}

	fmt.Printf("  Sampling %d/%d positions\n", sampleSize, numTrain)

	// Golden section search
	phi := (math.Sqrt(5) + 1) / 2
	a, b := 100.0, 800.0
	iter := 0
	for b-a > 1.0 {
		c := b - (b-a)/phi
		d := a + (b-a)/phi
		ec := computeError(c)
		ed := computeError(d)
		iter++
		fmt.Printf("  iter %2d: K in [%.1f, %.1f]  err=%.8f\n", iter, a, b, math.Min(ec, ed))
		if ec < ed {
			b = d
		} else {
			a = c
		}
	}
	return (a + b) / 2
}

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
	InputWeights   [NNUEInputSize][NNUEHiddenSize]float32
	InputBiases    [NNUEHiddenSize]float32
	HiddenWeights  [NNUEHiddenSize * 2][NNUEHidden2Size]float32
	HiddenBiases   [NNUEHidden2Size]float32
	Hidden2Weights [NNUEHidden2Size][NNUEHidden3Size]float32
	Hidden2Biases  [NNUEHidden3Size]float32
	OutputWeights  [NNUEOutputBuckets][NNUEHidden3Size]float32
	OutputBias     [NNUEOutputBuckets]float32
}

// NNUETrainConfig holds training hyperparameters.
type NNUETrainConfig struct {
	Epochs       int
	LR           float64
	BatchSize    int
	Lambda       float64 // weight for result vs score: loss = lambda*MSE(result) + (1-lambda)*MSE(score)
	K            float64 // sigmoid scaling constant
	MaxPositions int             // limit training positions per epoch (0=use all)
	Stop         <-chan struct{} // if non-nil, checked each epoch; close to stop early
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
	WhiteFeatures []int     // active HalfKA feature indices for White perspective
	BlackFeatures []int     // active HalfKA feature indices for Black perspective
	SideToMove    Color     // side to move
	Result        float32   // game result from White's perspective (1.0, 0.5, 0.0)
	Score         float32   // search score in centipawns (White-relative)
	HasScore      bool      // whether Score is valid
	PieceCount    int       // total pieces on board (for output bucket selection)
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

	// Input layer initialization scaled for CReLU [0, 1.0] with ~32 active features.
	// (HalfKA includes kings, so ~30 non-king pieces + 2 kings = ~32 features)
	// Target accumulator std ~0.4 so most positive values stay in (0, 1) without
	// saturating at the CReLU upper bound, giving ~49% of neurons active gradient
	// flow (vs ~26% with full He init where std ~1.4 causes heavy saturation).
	inputScale := float32(0.4 / math.Sqrt(32.0))
	for i := range net.InputWeights {
		for j := range net.InputWeights[i] {
			net.InputWeights[i][j] = float32(rng.NormFloat64()) * inputScale
		}
	}
	// Input biases start slightly positive so accumulator neurons begin in the
	// active CReLU region and must be pushed out, preventing early neuron death.
	for i := range net.InputBiases {
		net.InputBiases[i] = 0.1
	}

	// Hidden layer 1: fan_in = 512 (concatenated accumulators)
	hiddenScale := float32(math.Sqrt(2.0 / 512.0))
	for i := range net.HiddenWeights {
		for j := range net.HiddenWeights[i] {
			net.HiddenWeights[i][j] = float32(rng.NormFloat64()) * hiddenScale
		}
	}
	for i := range net.HiddenBiases {
		net.HiddenBiases[i] = 0.1
	}

	// Hidden layer 2: fan_in = 32
	hidden2Scale := float32(math.Sqrt(2.0 / 32.0))
	for i := range net.Hidden2Weights {
		for j := range net.Hidden2Weights[i] {
			net.Hidden2Weights[i][j] = float32(rng.NormFloat64()) * hidden2Scale
		}
	}
	for i := range net.Hidden2Biases {
		net.Hidden2Biases[i] = 0.1
	}

	// Output layer: fan_in = 32, one set per output bucket
	outputScale := float32(math.Sqrt(2.0 / 32.0))
	for b := 0; b < NNUEOutputBuckets; b++ {
		for i := range net.OutputWeights[b] {
			net.OutputWeights[b][i] = float32(rng.NormFloat64()) * outputScale
		}
		net.OutputBias[b] = 0
	}

	return net
}

// Forward computes the network output and stores intermediates for backprop.
// Returns the raw output (before sigmoid) in centipawns, side-to-move relative.
func (net *NNUETrainNet) Forward(sample *NNUETrainSample) (output float32, hidden1 [NNUEHiddenSize * 2]float32, hidden2 [NNUEHidden2Size]float32, hidden3 [NNUEHidden3Size]float32) {
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

	// Hidden layer 2 with ReLU on hidden2 (no upper clamp)
	for k := 0; k < NNUEHidden3Size; k++ {
		sum := net.Hidden2Biases[k]
		for j := 0; j < NNUEHidden2Size; j++ {
			sum += reluF(hidden2[j]) * net.Hidden2Weights[j][k]
		}
		hidden3[k] = sum
	}

	// Output layer with ReLU on hidden3 (no upper clamp), using material bucket
	bucket := OutputBucket(sample.PieceCount)
	output = net.OutputBias[bucket]
	for k := 0; k < NNUEHidden3Size; k++ {
		output += reluF(hidden3[k]) * net.OutputWeights[bucket][k]
	}

	return output, hidden1, hidden2, hidden3
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

// reluF is plain ReLU without upper clamp — used for hidden2/hidden3 layers
// which need unconstrained positive range to express large eval magnitudes.
func reluF(x float32) float32 {
	if x < 0 {
		return 0
	}
	return x
}

func reluGrad(x float32) float32 {
	if x > 0 {
		return 1
	}
	return 0
}

func nnueSigmoid(x float64, K float64) float64 {
	return 1.0 / (1.0 + math.Pow(10.0, -x/K))
}

// NNUETrainGradients accumulates gradients for all network weights.
type NNUETrainGradients struct {
	InputWeights   [NNUEInputSize][NNUEHiddenSize]float32
	InputBiases    [NNUEHiddenSize]float32
	HiddenWeights  [NNUEHiddenSize * 2][NNUEHidden2Size]float32
	HiddenBiases   [NNUEHidden2Size]float32
	Hidden2Weights [NNUEHidden2Size][NNUEHidden3Size]float32
	Hidden2Biases  [NNUEHidden3Size]float32
	OutputWeights  [NNUEOutputBuckets][NNUEHidden3Size]float32
	OutputBias     [NNUEOutputBuckets]float32

	// Sparse tracking for InputWeights — only zero/aggregate touched rows
	dirtyInputs []int  // which InputWeights rows were modified
	dirtySet    []bool // fast membership test (indexed by input feature)
}

// Backward computes gradients for a single sample and accumulates into grads.
func (net *NNUETrainNet) Backward(sample *NNUETrainSample, grads *NNUETrainGradients,
	output float32, hidden1 [NNUEHiddenSize * 2]float32, hidden2 [NNUEHidden2Size]float32, hidden3 [NNUEHidden3Size]float32,
	cfg NNUETrainConfig) {

	// Compute loss gradient: d(loss)/d(output)
	pred := nnueSigmoid(float64(output), cfg.K)

	var target float64
	if sample.HasScore {
		score := float64(sample.Score)
		if sample.SideToMove == Black {
			score = -score
		}
		scoreTarget := nnueSigmoid(score, cfg.K)
		result := float64(sample.Result)
		if sample.SideToMove == Black {
			result = 1.0 - result
		}
		target = cfg.Lambda*result + (1.0-cfg.Lambda)*scoreTarget
	} else {
		target = float64(sample.Result)
		if sample.SideToMove == Black {
			target = 1.0 - target
		}
	}

	// d(loss)/d(output) = 2 * (pred - target) * pred * (1 - pred) * ln(10) / K
	dOutput := float32(2.0 * (pred - target) * pred * (1.0 - pred) * math.Ln10 / cfg.K)

	// Output layer gradients: output = bias + sum(ReLU(hidden3) * weights), using material bucket
	bucket := OutputBucket(sample.PieceCount)
	grads.OutputBias[bucket] += dOutput
	var dHidden3 [NNUEHidden3Size]float32
	for k := 0; k < NNUEHidden3Size; k++ {
		h3 := reluF(hidden3[k])
		grads.OutputWeights[bucket][k] += dOutput * h3
		dHidden3[k] = dOutput * net.OutputWeights[bucket][k] * reluGrad(hidden3[k])
	}

	// Hidden layer 2 gradients: hidden3 = bias + sum(ReLU(hidden2) * weights)
	var dHidden2 [NNUEHidden2Size]float32
	for k := 0; k < NNUEHidden3Size; k++ {
		grads.Hidden2Biases[k] += dHidden3[k]
	}
	for j := 0; j < NNUEHidden2Size; j++ {
		h2 := reluF(hidden2[j])
		for k := 0; k < NNUEHidden3Size; k++ {
			grads.Hidden2Weights[j][k] += dHidden3[k] * h2
		}
		// Backprop through ReLU to hidden2
		var sum float32
		for k := 0; k < NNUEHidden3Size; k++ {
			sum += dHidden3[k] * net.Hidden2Weights[j][k]
		}
		dHidden2[j] = sum * reluGrad(hidden2[j])
	}

	// Hidden layer 1 gradients: hidden2 = bias + sum(hidden1 * weights)
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

	// Input bias gradients — accumulate once per perspective, NOT per feature.
	// STM perspective bias gradient
	for i := 0; i < NNUEHiddenSize; i++ {
		grads.InputBiases[i] += dHidden1[i] * clippedReLUGrad(hidden1[i])
	}
	// Opponent perspective bias gradient
	for i := 0; i < NNUEHiddenSize; i++ {
		grads.InputBiases[i] += dHidden1[NNUEHiddenSize+i] * clippedReLUGrad(hidden1[NNUEHiddenSize+i])
	}

	// STM perspective accumulator weight gradients
	for _, idx := range stmFeatures {
		if !grads.dirtySet[idx] {
			grads.dirtySet[idx] = true
			grads.dirtyInputs = append(grads.dirtyInputs, idx)
		}
		for i := 0; i < NNUEHiddenSize; i++ {
			grads.InputWeights[idx][i] += dHidden1[i] * clippedReLUGrad(hidden1[i])
		}
	}

	// Opponent perspective accumulator weight gradients
	for _, idx := range oppFeatures {
		if !grads.dirtySet[idx] {
			grads.dirtySet[idx] = true
			grads.dirtyInputs = append(grads.dirtyInputs, idx)
		}
		for i := 0; i < NNUEHiddenSize; i++ {
			grads.InputWeights[idx][i] += dHidden1[NNUEHiddenSize+i] * clippedReLUGrad(hidden1[NNUEHiddenSize+i])
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
		PieceCount: b.AllPieces.Count(),
	}

	// Extract active features (HalfKA: all pieces including kings)
	wKingSq := b.Pieces[WhiteKing].LSB()
	bKingSq := b.Pieces[BlackKing].LSB()

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

	// Write to a temp file first, then rename atomically.
	// This prevents a truncated .nnbin if preprocessing is interrupted.
	tmpFile := binFile + ".tmp"
	out, err := os.Create(tmpFile)
	if err != nil {
		return 0, 0, err
	}

	w := bufio.NewWriter(out)

	// Header
	header := NNBinHeader{
		Magic:         nnbinMagic,
		Version:       nnbinVersion,
		NumTrain:      numTrain,
		NumValidation: numVal,
	}
	if err := binary.Write(w, binary.LittleEndian, &header); err != nil {
		out.Close()
		os.Remove(tmpFile)
		return 0, 0, fmt.Errorf("writing header: %w", err)
	}

	// Write records
	for _, line := range lines {
		sample, err := ParseNNUETrainData(line)
		if err != nil {
			continue // skip bad lines
		}
		if err := writeNNBinRecord(w, sample); err != nil {
			out.Close()
			os.Remove(tmpFile)
			return 0, 0, err
		}
	}

	if err := w.Flush(); err != nil {
		out.Close()
		os.Remove(tmpFile)
		return 0, 0, err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmpFile)
		return 0, 0, err
	}

	// Atomic rename — only creates the final file if everything succeeded.
	if err := os.Rename(tmpFile, binFile); err != nil {
		os.Remove(tmpFile)
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

// nnbinRecordSize is the fixed-size header of each record (before variable-length features).
const nnbinRecordSize = 1 + 4 + 4 + 1 + 2 + 2 // SideToMove + Result + Score + HasScore + NumWhite + NumBlack = 14 bytes

// ReadRecord reads a single training sample at the given record index.
func (bf *NNBinFile) ReadRecord(index int) (*NNUETrainSample, error) {
	if index < 0 || index >= len(bf.recordOffsets) {
		return nil, fmt.Errorf("record index %d out of range [0, %d)", index, len(bf.recordOffsets))
	}
	return bf.readRecordAt(index, nil)
}

// readRecordAt reads a record using ReadAt (concurrent-safe, no seek).
// If sample is non-nil, it reuses the struct and feature slices.
func (bf *NNBinFile) readRecordAt(index int, sample *NNUETrainSample) (*NNUETrainSample, error) {
	offset := bf.recordOffsets[index]

	// Read fixed header
	var hdr [nnbinRecordSize]byte
	if _, err := bf.file.ReadAt(hdr[:], offset); err != nil {
		return nil, err
	}

	// Manual decode (little-endian)
	stm := hdr[0]
	result := math.Float32frombits(binary.LittleEndian.Uint32(hdr[1:5]))
	score := math.Float32frombits(binary.LittleEndian.Uint32(hdr[5:9]))
	hasScore := hdr[9]
	numWhite := int(binary.LittleEndian.Uint16(hdr[10:12]))
	numBlack := int(binary.LittleEndian.Uint16(hdr[12:14]))

	if sample == nil {
		sample = &NNUETrainSample{}
	}
	sample.SideToMove = Color(stm)
	sample.Result = result
	sample.Score = score
	sample.HasScore = hasScore != 0
	sample.PieceCount = numWhite // each piece generates one feature per perspective

	// Read feature indices in one ReadAt call
	numFeatures := numWhite + numBlack
	featureBytes := numFeatures * 2
	buf := make([]byte, featureBytes)
	if _, err := bf.file.ReadAt(buf, offset+nnbinRecordSize); err != nil {
		return nil, err
	}

	// Reuse or allocate feature slices
	if cap(sample.WhiteFeatures) >= numWhite {
		sample.WhiteFeatures = sample.WhiteFeatures[:numWhite]
	} else {
		sample.WhiteFeatures = make([]int, numWhite)
	}
	if cap(sample.BlackFeatures) >= numBlack {
		sample.BlackFeatures = sample.BlackFeatures[:numBlack]
	} else {
		sample.BlackFeatures = make([]int, numBlack)
	}

	for i := 0; i < numWhite; i++ {
		sample.WhiteFeatures[i] = int(binary.LittleEndian.Uint16(buf[i*2 : i*2+2]))
	}
	base := numWhite * 2
	for i := 0; i < numBlack; i++ {
		sample.BlackFeatures[i] = int(binary.LittleEndian.Uint16(buf[base+i*2 : base+i*2+2]))
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
		inputWeights   [NNUEInputSize][NNUEHiddenSize]nnueAdamState
		inputBiases    [NNUEHiddenSize]nnueAdamState
		hiddenWeights  [NNUEHiddenSize * 2][NNUEHidden2Size]nnueAdamState
		hiddenBiases   [NNUEHidden2Size]nnueAdamState
		hidden2Weights [NNUEHidden2Size][NNUEHidden3Size]nnueAdamState
		hidden2Biases  [NNUEHidden3Size]nnueAdamState
		outputWeights  [NNUEOutputBuckets][NNUEHidden3Size]nnueAdamState
		outputBias     [NNUEOutputBuckets]nnueAdamState
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

	// Pre-allocate worker gradient structs (~40MB each) once, reuse across all batches.
	workerGrads := make([]NNUETrainGradients, numWorkers)
	for w := range workerGrads {
		initGradientTracking(&workerGrads[w])
	}
	workerLoss := make([]float64, numWorkers)
	workerCount := make([]int, numWorkers)

	// Pre-allocate sample storage reused across batches.
	// Each worker gets its own slice of samples to read into (parallel ReadAt).
	maxBatch := cfg.BatchSize
	samples := make([]*NNUETrainSample, maxBatch)
	for i := range samples {
		samples[i] = &NNUETrainSample{}
	}

	var totalGrads NNUETrainGradients
	initGradientTracking(&totalGrads)

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		// Check for early stop signal
		if cfg.Stop != nil {
			select {
			case <-cfg.Stop:
				return
			default:
			}
		}

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

			// Parallel read + forward + backward.
			// Each worker reads its own samples using ReadAt (no shared seek pointer),
			// zeros its own gradients, and computes in one pass.
			perWorker := (batchSize + numWorkers - 1) / numWorkers

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

					// Zero this worker's accumulators (parallel — each worker zeros its own)
					workerLoss[workerIdx] = 0
					workerCount[workerIdx] = 0
					zeroGradients(&workerGrads[workerIdx])

					for i := start; i < end; i++ {
						s, err := bf.readRecordAt(batchIndices[i], samples[i])
						if err != nil {
							continue
						}
						samples[i] = s

						output, hidden1, hidden2, hidden3 := trainer.Net.Forward(s)
						trainer.Net.Backward(s, &workerGrads[workerIdx],
							output, hidden1, hidden2, hidden3, cfg)

						// Compute loss
						pred := nnueSigmoid(float64(output), cfg.K)
						var target float64
						if s.HasScore {
							result := float64(s.Result)
							if s.SideToMove == Black {
								result = 1.0 - result
							}
							score := float64(s.Score)
							if s.SideToMove == Black {
								score = -score
							}
							scoreTarget := nnueSigmoid(score, cfg.K)
							target = cfg.Lambda*result + (1.0-cfg.Lambda)*scoreTarget
						} else {
							target = float64(s.Result)
							if s.SideToMove == Black {
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
			zeroGradients(&totalGrads)
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

// initGradientTracking initializes the sparse tracking arrays (call once at allocation).
func initGradientTracking(g *NNUETrainGradients) {
	g.dirtyInputs = make([]int, 0, 4096)
	g.dirtySet = make([]bool, NNUEInputSize)
}

// zeroGradients resets all gradient accumulators to zero.
// InputWeights uses sparse zeroing — only rows touched since last zero are cleared.
func zeroGradients(g *NNUETrainGradients) {
	for b := range g.OutputBias {
		g.OutputBias[b] = 0
	}
	for b := range g.OutputWeights {
		for k := range g.OutputWeights[b] {
			g.OutputWeights[b][k] = 0
		}
	}
	for k := range g.Hidden2Biases {
		g.Hidden2Biases[k] = 0
	}
	for j := range g.Hidden2Weights {
		for k := range g.Hidden2Weights[j] {
			g.Hidden2Weights[j][k] = 0
		}
	}
	for j := range g.HiddenBiases {
		g.HiddenBiases[j] = 0
	}
	for i := range g.HiddenWeights {
		for j := range g.HiddenWeights[i] {
			g.HiddenWeights[i][j] = 0
		}
	}
	for j := range g.InputBiases {
		g.InputBiases[j] = 0
	}
	// Sparse zero: only clear rows that were actually written
	for _, idx := range g.dirtyInputs {
		for j := range g.InputWeights[idx] {
			g.InputWeights[idx][j] = 0
		}
		g.dirtySet[idx] = false
	}
	g.dirtyInputs = g.dirtyInputs[:0]
}

func addGradients(dst, src *NNUETrainGradients) {
	for b := 0; b < NNUEOutputBuckets; b++ {
		dst.OutputBias[b] += src.OutputBias[b]
		for k := 0; k < NNUEHidden3Size; k++ {
			dst.OutputWeights[b][k] += src.OutputWeights[b][k]
		}
	}
	for k := 0; k < NNUEHidden3Size; k++ {
		dst.Hidden2Biases[k] += src.Hidden2Biases[k]
	}
	for j := 0; j < NNUEHidden2Size; j++ {
		dst.HiddenBiases[j] += src.HiddenBiases[j]
		for k := 0; k < NNUEHidden3Size; k++ {
			dst.Hidden2Weights[j][k] += src.Hidden2Weights[j][k]
		}
	}
	for i := 0; i < NNUEHiddenSize*2; i++ {
		for j := 0; j < NNUEHidden2Size; j++ {
			dst.HiddenWeights[i][j] += src.HiddenWeights[i][j]
		}
	}
	// Sparse input gradients — only dirty rows
	for _, idx := range src.dirtyInputs {
		if !dst.dirtySet[idx] {
			dst.dirtySet[idx] = true
			dst.dirtyInputs = append(dst.dirtyInputs, idx)
		}
		for j := 0; j < NNUEHiddenSize; j++ {
			dst.InputWeights[idx][j] += src.InputWeights[idx][j]
		}
	}
	for j := 0; j < NNUEHiddenSize; j++ {
		dst.InputBiases[j] += src.InputBiases[j]
	}
}

func (trainer *NNUETrainer) applyAdamUpdates(grads *NNUETrainGradients, scale, lr float64) {
	step := trainer.step

	// Output layer (hidden3 -> output), per bucket
	for b := 0; b < NNUEOutputBuckets; b++ {
		adamUpdate(&trainer.Net.OutputBias[b], float64(grads.OutputBias[b])*scale,
			&trainer.adam.outputBias[b], lr, step)
		for j := 0; j < NNUEHidden3Size; j++ {
			adamUpdate(&trainer.Net.OutputWeights[b][j], float64(grads.OutputWeights[b][j])*scale,
				&trainer.adam.outputWeights[b][j], lr, step)
		}
	}

	// Hidden2 layer (hidden2 -> hidden3)
	for j := 0; j < NNUEHidden3Size; j++ {
		adamUpdate(&trainer.Net.Hidden2Biases[j], float64(grads.Hidden2Biases[j])*scale,
			&trainer.adam.hidden2Biases[j], lr, step)
	}
	for i := 0; i < NNUEHidden2Size; i++ {
		for j := 0; j < NNUEHidden3Size; j++ {
			adamUpdate(&trainer.Net.Hidden2Weights[i][j], float64(grads.Hidden2Weights[i][j])*scale,
				&trainer.adam.hidden2Weights[i][j], lr, step)
		}
	}

	// Hidden layer (concat accum -> hidden2)
	for j := 0; j < NNUEHidden2Size; j++ {
		adamUpdate(&trainer.Net.HiddenBiases[j], float64(grads.HiddenBiases[j])*scale,
			&trainer.adam.hiddenBiases[j], lr, step)
	}
	for i := 0; i < NNUEHiddenSize*2; i++ {
		for j := 0; j < NNUEHidden2Size; j++ {
			adamUpdate(&trainer.Net.HiddenWeights[i][j], float64(grads.HiddenWeights[i][j])*scale,
				&trainer.adam.hiddenWeights[i][j], lr, step)
		}
	}

	// Input layer (sparse — only dirty rows)
	for _, idx := range grads.dirtyInputs {
		for j := 0; j < NNUEHiddenSize; j++ {
			if grads.InputWeights[idx][j] != 0 {
				adamUpdate(&trainer.Net.InputWeights[idx][j], float64(grads.InputWeights[idx][j])*scale,
					&trainer.adam.inputWeights[idx][j], lr, step)
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
	numVal := valEnd - valStart
	if numVal == 0 {
		return 0
	}

	numWorkers := runtime.NumCPU()
	perWorker := (numVal + numWorkers - 1) / numWorkers
	wLoss := make([]float64, numWorkers)
	wCount := make([]int, numWorkers)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerIdx int) {
			defer wg.Done()
			start := valStart + workerIdx*perWorker
			end := start + perWorker
			if end > valEnd {
				end = valEnd
			}

			var sample NNUETrainSample
			for i := start; i < end; i++ {
				s, err := bf.readRecordAt(i, &sample)
				if err != nil {
					continue
				}
				output, _, _, _ := trainer.Net.Forward(s)
				pred := nnueSigmoid(float64(output), cfg.K)

				var target float64
				if s.HasScore {
					result := float64(s.Result)
					if s.SideToMove == Black {
						result = 1.0 - result
					}
					score := float64(s.Score)
					if s.SideToMove == Black {
						score = -score
					}
					scoreTarget := nnueSigmoid(score, cfg.K)
					target = cfg.Lambda*result + (1.0-cfg.Lambda)*scoreTarget
				} else {
					target = float64(s.Result)
					if s.SideToMove == Black {
						target = 1.0 - target
					}
				}

				diff := pred - target
				wLoss[workerIdx] += diff * diff
				wCount[workerIdx]++
			}
		}(w)
	}
	wg.Wait()

	totalLoss := 0.0
	count := 0
	for w := 0; w < numWorkers; w++ {
		totalLoss += wLoss[w]
		count += wCount[w]
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

	// Hidden2 layer: same quantization as hidden layer.
	// Input activation is at scale nnueInputScale (127) after CReLU >> 6.
	// Weights scaled by nnueHiddenScale (64), biases by nnueInputScale * nnueHiddenScale (8128).
	for i := range train.Hidden2Weights {
		for j := range train.Hidden2Weights[i] {
			net.Hidden2Weights[i][j] = int16(math.Round(float64(train.Hidden2Weights[i][j]) * float64(nnueHiddenScale)))
		}
	}
	for j := range train.Hidden2Biases {
		net.Hidden2Biases[j] = int32(math.Round(float64(train.Hidden2Biases[j]) * float64(nnueInputScale) * float64(nnueHiddenScale)))
	}

	// Output layer: bias is int32 at scale nnueOutputScale (8128).
	// Weights are int16 at scale nnueHiddenScale (64), NOT nnueOutputScale,
	// because the CReLU activation (after >>6) is at scale nnueInputScale (127).
	// Product: activation(127) * weight(64) = scale 8128 = nnueOutputScale, matching bias.
	for b := 0; b < NNUEOutputBuckets; b++ {
		for j := range train.OutputWeights[b] {
			net.OutputWeights[b][j] = int16(math.Round(float64(train.OutputWeights[b][j]) * float64(nnueHiddenScale)))
		}
		net.OutputBias[b] = int32(math.Round(float64(train.OutputBias[b]) * float64(nnueOutputScale)))
	}

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

	// Hidden2 weights: divide by nnueHiddenScale (64)
	for i := range net.Hidden2Weights {
		for j := range net.Hidden2Weights[i] {
			train.Hidden2Weights[i][j] = float32(net.Hidden2Weights[i][j]) / nnueHiddenScale
		}
	}
	// Hidden2 biases: divide by nnueInputScale * nnueHiddenScale (8128)
	for j := range net.Hidden2Biases {
		train.Hidden2Biases[j] = float32(net.Hidden2Biases[j]) / nnueOutputScale
	}

	// Output weights: divide by nnueHiddenScale (64), matching QuantizeNetwork
	for b := 0; b < NNUEOutputBuckets; b++ {
		for j := range net.OutputWeights[b] {
			train.OutputWeights[b][j] = float32(net.OutputWeights[b][j]) / nnueHiddenScale
		}
		// Output bias: divide by nnueOutputScale (8128)
		train.OutputBias[b] = float32(net.OutputBias[b]) / nnueOutputScale
	}

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
			output, _, _, _ := trainer.Net.Forward(s)
			pred := nnueSigmoid(float64(output), K)

			var target float64
			if s.HasScore {
				result := float64(s.Result)
				if s.SideToMove == Black {
					result = 1.0 - result
				}
				score := float64(s.Score)
				if s.SideToMove == Black {
					score = -score
				}
				scoreTarget := nnueSigmoid(score, K)
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

// TrainBinpack runs the NNUE training loop using binpack files directly.
// This eliminates the .nnbin preprocessing step.
func (trainer *NNUETrainer) TrainBinpack(bf *BinpackFile, cfg NNUETrainConfig,
	onEpoch func(epoch int, trainLoss, valLoss float64)) {

	trainFraction := 0.9
	numWorkers := runtime.NumCPU()

	// Pre-allocate worker gradient structs
	workerGrads := make([]NNUETrainGradients, numWorkers)
	for w := range workerGrads {
		initGradientTracking(&workerGrads[w])
	}
	workerLoss := make([]float64, numWorkers)
	workerCount := make([]int, numWorkers)

	var totalGrads NNUETrainGradients
	initGradientTracking(&totalGrads)

	rng := rand.New(rand.NewSource(42))

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		// Check for early stop signal
		if cfg.Stop != nil {
			select {
			case <-cfg.Stop:
				return
			default:
			}
		}

		reader := bf.NewEpochReader(rng, trainFraction)
		numTrain := reader.NumTrainRecords()
		if cfg.MaxPositions > 0 && cfg.MaxPositions < numTrain {
			numTrain = cfg.MaxPositions
		}

		totalLoss := 0.0
		numSamples := 0
		processed := 0

		// Pre-allocate sample buffer for batches
		sampleBuf := make([]*NNUETrainSample, 0, cfg.BatchSize)

		for processed < numTrain {
			batchSize := cfg.BatchSize
			if processed+batchSize > numTrain {
				batchSize = numTrain - processed
			}

			// Read batch via block-shuffled reader
			batch, err := reader.NextBatch(batchSize, sampleBuf)
			if err != nil || batch == nil {
				break
			}
			actualBatch := len(batch)
			processed += actualBatch

			// Parallel forward + backward
			perWorker := (actualBatch + numWorkers - 1) / numWorkers

			var wg sync.WaitGroup
			for w := 0; w < numWorkers; w++ {
				wg.Add(1)
				go func(workerIdx int) {
					defer wg.Done()
					start := workerIdx * perWorker
					end := start + perWorker
					if end > actualBatch {
						end = actualBatch
					}

					workerLoss[workerIdx] = 0
					workerCount[workerIdx] = 0
					zeroGradients(&workerGrads[workerIdx])

					for i := start; i < end; i++ {
						s := batch[i]
						output, hidden1, hidden2, hidden3 := trainer.Net.Forward(s)
						trainer.Net.Backward(s, &workerGrads[workerIdx],
							output, hidden1, hidden2, hidden3, cfg)

						pred := nnueSigmoid(float64(output), cfg.K)
						var target float64
						if s.HasScore {
							result := float64(s.Result)
							if s.SideToMove == Black {
								result = 1.0 - result
							}
							score := float64(s.Score)
							if s.SideToMove == Black {
								score = -score
							}
							scoreTarget := nnueSigmoid(score, cfg.K)
							target = cfg.Lambda*result + (1.0-cfg.Lambda)*scoreTarget
						} else {
							target = float64(s.Result)
							if s.SideToMove == Black {
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
			zeroGradients(&totalGrads)
			for w := 0; w < numWorkers; w++ {
				totalLoss += workerLoss[w]
				numSamples += workerCount[w]
				addGradients(&totalGrads, &workerGrads[w])
			}

			// Scale gradients and apply Adam updates
			scale := 1.0 / float64(actualBatch)
			trainer.step++
			trainer.applyAdamUpdates(&totalGrads, scale, cfg.LR)
		}

		// Compute losses
		trainLoss := 0.0
		if numSamples > 0 {
			trainLoss = totalLoss / float64(numSamples)
		}

		valLoss := trainer.computeValidationLossBinpack(bf, cfg, trainFraction)

		if onEpoch != nil {
			onEpoch(epoch, trainLoss, valLoss)
		}
	}
}

// computeValidationLossBinpack computes validation loss from binpack files.
func (trainer *NNUETrainer) computeValidationLossBinpack(bf *BinpackFile, cfg NNUETrainConfig, trainFraction float64) float64 {
	// Read validation samples
	valSamples, err := bf.ValidationSamples(trainFraction)
	if err != nil || len(valSamples) == 0 {
		return 0
	}

	numWorkers := runtime.NumCPU()
	perWorker := (len(valSamples) + numWorkers - 1) / numWorkers
	wLoss := make([]float64, numWorkers)
	wCount := make([]int, numWorkers)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerIdx int) {
			defer wg.Done()
			start := workerIdx * perWorker
			end := start + perWorker
			if end > len(valSamples) {
				end = len(valSamples)
			}

			for i := start; i < end; i++ {
				s := valSamples[i]
				output, _, _, _ := trainer.Net.Forward(s)
				pred := nnueSigmoid(float64(output), cfg.K)

				var target float64
				if s.HasScore {
					result := float64(s.Result)
					if s.SideToMove == Black {
						result = 1.0 - result
					}
					score := float64(s.Score)
					if s.SideToMove == Black {
						score = -score
					}
					scoreTarget := nnueSigmoid(score, cfg.K)
					target = cfg.Lambda*result + (1.0-cfg.Lambda)*scoreTarget
				} else {
					target = float64(s.Result)
					if s.SideToMove == Black {
						target = 1.0 - target
					}
				}

				diff := pred - target
				wLoss[workerIdx] += diff * diff
				wCount[workerIdx]++
			}
		}(w)
	}
	wg.Wait()

	totalLoss := 0.0
	count := 0
	for w := 0; w < numWorkers; w++ {
		totalLoss += wLoss[w]
		count += wCount[w]
	}
	if count == 0 {
		return 0
	}
	return totalLoss / float64(count)
}

// TuneKBinpack finds the optimal sigmoid scaling constant K using binpack files.
func (trainer *NNUETrainer) TuneKBinpack(bf *BinpackFile, lambda float64) float64 {
	// Sample up to 50K positions from the first file
	sampleSize := bf.NumRecords()
	if sampleSize > 50000 {
		sampleSize = 50000
	}

	// Read samples
	samples := make([]*NNUETrainSample, 0, sampleSize)
	for i := 0; i < sampleSize; i++ {
		rec, err := bf.ReadRecord(i)
		if err != nil {
			continue
		}
		samples = append(samples, ExtractFeaturesFromBinpack(rec))
	}

	computeError := func(K float64) float64 {
		totalLoss := 0.0
		for _, s := range samples {
			output, _, _, _ := trainer.Net.Forward(s)
			pred := nnueSigmoid(float64(output), K)

			var target float64
			if s.HasScore {
				result := float64(s.Result)
				if s.SideToMove == Black {
					result = 1.0 - result
				}
				score := float64(s.Score)
				if s.SideToMove == Black {
					score = -score
				}
				scoreTarget := nnueSigmoid(score, K)
				target = lambda*result + (1.0-lambda)*scoreTarget
			} else {
				target = float64(s.Result)
				if s.SideToMove == Black {
					target = 1.0 - target
				}
			}

			diff := pred - target
			totalLoss += diff * diff
		}
		if len(samples) == 0 {
			return 0
		}
		return totalLoss / float64(len(samples))
	}

	fmt.Printf("  Sampling %d/%d positions\n", len(samples), bf.NumRecords())

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

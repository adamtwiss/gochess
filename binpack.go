package chess

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/bits"
	"math/rand"
	"os"
)

// Binary training data format (.bin) for NNUE.
//
// Each record is exactly 32 bytes with no file header, so files can be
// concatenated with cat and record count is inferred from file size.
//
// Record layout (32 bytes, little-endian):
//
//	Offset  Size  Field
//	------  ----  -----
//	0       8     Occupancy bitmap (uint64) — bit i set = piece on square i
//	8       16    Piece nibbles — 4 bits per occupied square, LSB-first
//	24      1     Side to move (0=White, 1=Black)
//	25      1     Castling rights (4-bit mask: K=1, Q=2, k=4, q=8)
//	26      1     En passant file (0-7, or 8 = none)
//	27      1     Halfmove clock (uint8, capped at 255)
//	28      2     Score (int16, White-relative centipawns, clamped ±32000)
//	30      1     Result (0=Black wins, 1=Draw, 2=White wins)
//	31      1     Reserved (0)

const (
	BinpackRecordSize = 32
	BinpackBlockSize  = 2048 // records per block (2048 * 32 = 64KB)

	binpackNoEP      = 8
	binpackMaxScore  = 32000
	binpackMinScore  = -32000
)

// PackPosition encodes a Board position with score and result into a 32-byte binpack record.
func PackPosition(b *Board, score int16, result uint8) [BinpackRecordSize]byte {
	var rec [BinpackRecordSize]byte

	// Occupancy bitmap
	occ := uint64(b.AllPieces)
	binary.LittleEndian.PutUint64(rec[0:8], occ)

	// Pack piece nibbles: iterate set bits in LSB order
	tmp := occ
	nibbleIdx := 0
	for tmp != 0 {
		sq := bits.TrailingZeros64(tmp)
		tmp &= tmp - 1 // clear LSB

		piece := byte(b.Squares[sq])
		bytePos := 8 + nibbleIdx/2
		if nibbleIdx%2 == 0 {
			rec[bytePos] = piece // low nibble
		} else {
			rec[bytePos] |= piece << 4 // high nibble
		}
		nibbleIdx++
	}

	// Side to move
	if b.SideToMove == Black {
		rec[24] = 1
	}

	// Castling
	rec[25] = byte(b.Castling)

	// En passant file
	if b.EnPassant == NoSquare {
		rec[26] = binpackNoEP
	} else {
		rec[26] = byte(b.EnPassant.File())
	}

	// Halfmove clock
	hmc := b.HalfmoveClock
	if hmc > 255 {
		hmc = 255
	}
	if hmc < 0 {
		hmc = 0
	}
	rec[27] = byte(hmc)

	// Score (clamped)
	s := score
	if s > binpackMaxScore {
		s = binpackMaxScore
	}
	if s < binpackMinScore {
		s = binpackMinScore
	}
	binary.LittleEndian.PutUint16(rec[28:30], uint16(s))

	// Result
	rec[30] = result

	return rec
}

// UnpackPosition decodes a 32-byte binpack record into a Board, score, and result.
// The returned Board has correct Squares, Pieces bitboards, Occupied, AllPieces,
// SideToMove, Castling, EnPassant, HalfmoveClock, and Zobrist hash.
func UnpackPosition(rec [BinpackRecordSize]byte) (*Board, int16, uint8, error) {
	b := &Board{}
	b.Clear()

	occ := binary.LittleEndian.Uint64(rec[0:8])

	// Unpack pieces from nibbles
	tmp := occ
	nibbleIdx := 0
	for tmp != 0 {
		sq := Square(bits.TrailingZeros64(tmp))
		tmp &= tmp - 1

		bytePos := 8 + nibbleIdx/2
		var piece Piece
		if nibbleIdx%2 == 0 {
			piece = Piece(rec[bytePos] & 0x0F)
		} else {
			piece = Piece(rec[bytePos] >> 4)
		}
		nibbleIdx++

		if piece < WhitePawn || piece > BlackKing {
			return nil, 0, 0, fmt.Errorf("invalid piece %d at square %d", piece, sq)
		}

		b.Squares[sq] = piece
		b.Pieces[piece] |= 1 << Bitboard(sq)
		if piece.Color() == White {
			b.Occupied[White] |= 1 << Bitboard(sq)
		} else {
			b.Occupied[Black] |= 1 << Bitboard(sq)
		}
		b.AllPieces |= 1 << Bitboard(sq)
	}

	// Side to move
	if rec[24] == 1 {
		b.SideToMove = Black
	}

	// Castling
	b.Castling = CastlingRights(rec[25])

	// En passant
	epFile := rec[26]
	if epFile < 8 {
		if b.SideToMove == White {
			b.EnPassant = NewSquare(int(epFile), 5) // rank 6 (0-indexed: 5)
		} else {
			b.EnPassant = NewSquare(int(epFile), 2) // rank 3 (0-indexed: 2)
		}
	} else {
		b.EnPassant = NoSquare
	}

	// Halfmove clock
	b.HalfmoveClock = int16(rec[27])

	// Fullmove number (not stored — default to 1)
	b.FullmoveNum = 1

	// Rebuild Zobrist hash and PST scores
	b.HashKey = b.Hash()
	b.PawnHashKey = b.PawnHash()
	for sq := Square(0); sq < 64; sq++ {
		p := b.Squares[sq]
		if p != Empty {
			c := p.Color()
			b.PSTScoreMG[c] += pstCombinedMG[p][sq]
			b.PSTScoreEG[c] += pstCombinedEG[p][sq]
		}
	}

	score := int16(binary.LittleEndian.Uint16(rec[28:30]))
	result := rec[30]

	return b, score, result, nil
}

// ResultToUint8 converts a float result (0.0, 0.5, 1.0 from White's perspective)
// to the binpack encoding (0=Black wins, 1=Draw, 2=White wins).
func ResultToUint8(result float64) uint8 {
	return uint8(math.Round(result * 2))
}

// ResultToFloat converts a binpack result (0, 1, 2) to a float (0.0, 0.5, 1.0).
func ResultToFloat(result uint8) float32 {
	return float32(result) / 2.0
}

// ExtractFeaturesFromBinpack extracts HalfKA features from a binpack record
// directly, without building a full Board. Returns an NNUETrainSample.
func ExtractFeaturesFromBinpack(rec [BinpackRecordSize]byte) *NNUETrainSample {
	occ := binary.LittleEndian.Uint64(rec[0:8])

	// First pass: find king squares and all pieces
	type pieceEntry struct {
		piece Piece
		sq    Square
	}
	var pieces [32]pieceEntry
	var wKingSq, bKingSq Square
	nPieces := 0

	tmp := occ
	nibbleIdx := 0
	for tmp != 0 {
		sq := Square(bits.TrailingZeros64(tmp))
		tmp &= tmp - 1

		bytePos := 8 + nibbleIdx/2
		var piece Piece
		if nibbleIdx%2 == 0 {
			piece = Piece(rec[bytePos] & 0x0F)
		} else {
			piece = Piece(rec[bytePos] >> 4)
		}
		nibbleIdx++

		if piece == WhiteKing {
			wKingSq = sq
		} else if piece == BlackKing {
			bKingSq = sq
		}

		pieces[nPieces] = pieceEntry{piece, sq}
		nPieces++
	}

	// Side to move
	stm := White
	if rec[24] == 1 {
		stm = Black
	}

	// Score and result
	score := int16(binary.LittleEndian.Uint16(rec[28:30]))
	result := rec[30]

	sample := &NNUETrainSample{
		SideToMove:    stm,
		Result:        ResultToFloat(result),
		Score:         float32(score),
		HasScore:      true,
		PieceCount:    nPieces,
		WhiteFeatures: make([]int, 0, nPieces),
		BlackFeatures: make([]int, 0, nPieces),
	}

	// Extract HalfKA features for all pieces (including kings)
	for i := 0; i < nPieces; i++ {
		p := pieces[i].piece
		sq := pieces[i].sq

		wIdx := HalfKAIndex(White, wKingSq, p, sq)
		bIdx := HalfKAIndex(Black, bKingSq, p, sq)
		if wIdx >= 0 {
			sample.WhiteFeatures = append(sample.WhiteFeatures, wIdx)
		}
		if bIdx >= 0 {
			sample.BlackFeatures = append(sample.BlackFeatures, bIdx)
		}
	}

	return sample
}

// WriteBinpackRecord writes a single 32-byte record to a writer.
func WriteBinpackRecord(w io.Writer, rec [BinpackRecordSize]byte) error {
	_, err := w.Write(rec[:])
	return err
}

// BinpackFile represents one or more opened binpack data files for training.
// Multiple files are treated as a virtual concatenation.
type BinpackFile struct {
	files     []*os.File
	fileSizes []int64   // number of records in each file
	cumCounts []int     // cumulative record counts for indexing
	total     int       // total records across all files
}

// OpenBinpackFiles opens one or more binpack files for reading.
func OpenBinpackFiles(paths ...string) (*BinpackFile, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("no files specified")
	}

	bf := &BinpackFile{}

	cumulative := 0
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			bf.Close()
			return nil, fmt.Errorf("opening %s: %w", path, err)
		}

		stat, err := f.Stat()
		if err != nil {
			f.Close()
			bf.Close()
			return nil, fmt.Errorf("stat %s: %w", path, err)
		}

		size := stat.Size()
		if size%BinpackRecordSize != 0 {
			f.Close()
			bf.Close()
			return nil, fmt.Errorf("%s: file size %d is not a multiple of record size %d", path, size, BinpackRecordSize)
		}

		numRecords := int(size / BinpackRecordSize)
		bf.files = append(bf.files, f)
		bf.fileSizes = append(bf.fileSizes, int64(numRecords))
		cumulative += numRecords
		bf.cumCounts = append(bf.cumCounts, cumulative)
	}

	bf.total = cumulative
	return bf, nil
}

// NumRecords returns the total number of records across all files.
func (bf *BinpackFile) NumRecords() int {
	return bf.total
}

// Close closes all underlying files.
func (bf *BinpackFile) Close() error {
	var firstErr error
	for _, f := range bf.files {
		if err := f.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// ReadRecord reads a single record by global index.
func (bf *BinpackFile) ReadRecord(index int) ([BinpackRecordSize]byte, error) {
	var rec [BinpackRecordSize]byte
	if index < 0 || index >= bf.total {
		return rec, fmt.Errorf("index %d out of range [0, %d)", index, bf.total)
	}

	// Find which file this index belongs to
	fileIdx := 0
	localIdx := index
	for fileIdx < len(bf.cumCounts)-1 && index >= bf.cumCounts[fileIdx] {
		fileIdx++
	}
	if fileIdx > 0 {
		localIdx = index - bf.cumCounts[fileIdx-1]
	}

	offset := int64(localIdx) * BinpackRecordSize
	_, err := bf.files[fileIdx].ReadAt(rec[:], offset)
	return rec, err
}

// readBlock reads a contiguous block of records from a specific file.
// Returns the records as a byte slice (multiple of BinpackRecordSize).
func (bf *BinpackFile) readBlock(fileIdx int, startRecord, count int) ([]byte, error) {
	buf := make([]byte, count*BinpackRecordSize)
	offset := int64(startRecord) * BinpackRecordSize
	n, err := bf.files[fileIdx].ReadAt(buf, offset)
	if err != nil && err != io.EOF {
		return nil, err
	}
	// Truncate to complete records
	n = (n / BinpackRecordSize) * BinpackRecordSize
	return buf[:n], nil
}

// BinpackEpochReader provides block-shuffled streaming over binpack data.
// It divides the data into 64KB blocks, shuffles block order, reads each block
// in a single I/O, then shuffles records within the block.
type BinpackEpochReader struct {
	bf          *BinpackFile
	rng         *rand.Rand
	blocks      []blockRef   // shuffled block references
	blockIdx    int          // current block in the epoch
	buf         []byte       // current block data
	bufRecords  int          // number of records in current block
	bufPos      int          // current position within block
	order       []int        // shuffled indices within current block
}

// blockRef identifies a contiguous block of records within a specific file.
type blockRef struct {
	fileIdx    int
	startRec   int
	numRecords int
}

// NumTrainRecords returns the total number of training records this reader will yield.
func (r *BinpackEpochReader) NumTrainRecords() int {
	total := 0
	for _, b := range r.blocks {
		total += b.numRecords
	}
	return total
}

// NextBatch reads the next batch of samples, extracting HalfKA features on the fly.
// Returns nil when the epoch is exhausted.
func (r *BinpackEpochReader) NextBatch(size int, samples []*NNUETrainSample) ([]*NNUETrainSample, error) {
	samples = samples[:0]

	for len(samples) < size {
		// Need to load a new block?
		if r.bufPos >= r.bufRecords {
			if r.blockIdx >= len(r.blocks) {
				break // epoch exhausted
			}
			block := r.blocks[r.blockIdx]
			r.blockIdx++

			var err error
			r.buf, err = r.bf.readBlock(block.fileIdx, block.startRec, block.numRecords)
			if err != nil {
				return nil, fmt.Errorf("reading block: %w", err)
			}
			r.bufRecords = len(r.buf) / BinpackRecordSize
			r.bufPos = 0

			// Shuffle within block
			if cap(r.order) >= r.bufRecords {
				r.order = r.order[:r.bufRecords]
			} else {
				r.order = make([]int, r.bufRecords)
			}
			for i := range r.order {
				r.order[i] = i
			}
			r.rng.Shuffle(r.bufRecords, func(i, j int) {
				r.order[i], r.order[j] = r.order[j], r.order[i]
			})
		}

		// Extract records from current block
		for r.bufPos < r.bufRecords && len(samples) < size {
			idx := r.order[r.bufPos]
			r.bufPos++

			var rec [BinpackRecordSize]byte
			copy(rec[:], r.buf[idx*BinpackRecordSize:(idx+1)*BinpackRecordSize])
			samples = append(samples, ExtractFeaturesFromBinpack(rec))
		}
	}

	if len(samples) == 0 {
		return nil, nil
	}
	return samples, nil
}

// ValidationSamples reads all validation samples (last 1-trainFraction of each file).
func (bf *BinpackFile) ValidationSamples(trainFraction float64) ([]*NNUETrainSample, error) {
	var samples []*NNUETrainSample

	for fi, numRecs := range bf.fileSizes {
		trainRecs := int(float64(numRecs) * trainFraction)
		valRecs := int(numRecs) - trainRecs
		if valRecs <= 0 {
			continue
		}

		buf, err := bf.readBlock(fi, trainRecs, valRecs)
		if err != nil {
			return nil, fmt.Errorf("reading validation from file %d: %w", fi, err)
		}

		for off := 0; off+BinpackRecordSize <= len(buf); off += BinpackRecordSize {
			var rec [BinpackRecordSize]byte
			copy(rec[:], buf[off:off+BinpackRecordSize])
			samples = append(samples, ExtractFeaturesFromBinpack(rec))
		}
	}

	return samples, nil
}

// BinpackFile implements TrainingDataSource so it can be used interchangeably
// with SFBinpackSource in the training pipeline.

// NewEpochReader wraps the existing block-shuffled reader to satisfy TrainingDataSource.
func (bf *BinpackFile) NewEpochReader(rng *rand.Rand, trainFraction float64) TrainingEpochReader {
	return bf.NewBlockEpochReader(rng, trainFraction)
}

// NewBlockEpochReader creates a block-shuffled reader (original NewEpochReader logic).
func (bf *BinpackFile) NewBlockEpochReader(rng *rand.Rand, trainFraction float64) *BinpackEpochReader {
	reader := &BinpackEpochReader{
		bf:  bf,
		rng: rng,
	}

	for fi, numRecs := range bf.fileSizes {
		trainRecs := int(float64(numRecs) * trainFraction)
		for start := 0; start < trainRecs; start += BinpackBlockSize {
			count := BinpackBlockSize
			if start+count > trainRecs {
				count = trainRecs - start
			}
			reader.blocks = append(reader.blocks, blockRef{
				fileIdx:    fi,
				startRec:   start,
				numRecords: count,
			})
		}
	}

	rng.Shuffle(len(reader.blocks), func(i, j int) {
		reader.blocks[i], reader.blocks[j] = reader.blocks[j], reader.blocks[i]
	})

	return reader
}


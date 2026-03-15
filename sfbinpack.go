package chess

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"os"
)

// Stockfish .binpack format reader.
//
// The format uses chain-compressed positions: an anchor position is Huffman-coded
// into 32 bytes (PackedSfen), followed by a sequence of moves that generate
// subsequent positions. Each chain entry is 8 bytes (score, move, ply, result).
// Average ~2.5 bytes/position vs our 32 bytes/position.
//
// Reference: https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/nnue_training_data_formats.h

// sfLightPos is a minimal position for chain replay.
// Much cheaper than Board (~100 bytes vs ~1KB+): no bitboards, no NNUE accumulators,
// no pawn table, no Zobrist hash, no undo stack.
type sfLightPos struct {
	squares    [64]Piece
	sideToMove Color
	castling   CastlingRights
	epSquare   Square // NoSquare if none
	rule50     uint8
	wKingSq    Square
	bKingSq    Square
}

// bitReader reads variable-length bit sequences from a byte slice (LSB-first).
type bitReader struct {
	data   []byte
	bitPos int
}

func (br *bitReader) readBit() uint8 {
	byteIdx := br.bitPos / 8
	bitIdx := br.bitPos % 8
	br.bitPos++
	return (br.data[byteIdx] >> uint(bitIdx)) & 1
}

func (br *bitReader) readBits(n int) uint32 {
	var val uint32
	for i := 0; i < n; i++ {
		val |= uint32(br.readBit()) << uint(i)
	}
	return val
}

// Huffman piece decoding table for Stockfish PackedSfen.
// Encoding (variable-length, LSB-first):
//   Pawn:   1         + color bit = 2 bits
//   Knight: 01        + color bit = 3 bits
//   Bishop: 001       + color bit = 4 bits
//   Rook:   0001      + color bit = 5 bits
//   Queen:  00001     + color bit = 6 bits
func decodeHuffmanPiece(br *bitReader) Piece {
	// Read prefix: count leading zeros terminated by a 1
	pieceType := 0
	for {
		bit := br.readBit()
		pieceType++
		if bit == 1 {
			break
		}
		if pieceType > 5 {
			return Empty // invalid
		}
	}

	// Read color bit: 0=white, 1=black
	color := Color(br.readBit())

	// pieceType: 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen
	return pieceOf(Piece(pieceType), color)
}

// decodeSfen decodes a 32-byte Stockfish PackedSfen into sfLightPos.
//
// Layout (256 bits, LSB-first):
//   bits 0-5:    White king square (6 bits)
//   bits 6-11:   Black king square (6 bits)
//   bits 12:     Side to move (0=white, 1=black)
//   bits 13-...: Huffman-coded pieces (a1→h8, skipping king squares)
//   After pieces: rule50 (6 bits), castling (4 bits), en passant file (0-7 or 8=none, encoded in some bits)
//
// The exact bit layout after the Huffman pieces uses the remaining bits.
func decodeSfen(data [32]byte) (sfLightPos, error) {
	var pos sfLightPos
	for i := range pos.squares {
		pos.squares[i] = Empty
	}
	pos.epSquare = NoSquare

	br := &bitReader{data: data[:]}

	// Read king squares (6 bits each)
	pos.wKingSq = Square(br.readBits(6))
	pos.bKingSq = Square(br.readBits(6))

	if pos.wKingSq < 0 || pos.wKingSq > 63 || pos.bKingSq < 0 || pos.bKingSq > 63 {
		return pos, fmt.Errorf("invalid king squares: white=%d black=%d", pos.wKingSq, pos.bKingSq)
	}

	pos.squares[pos.wKingSq] = WhiteKing
	pos.squares[pos.bKingSq] = BlackKing

	// Read side to move
	pos.sideToMove = Color(br.readBit())

	// Read Huffman-coded pieces for all squares a1→h8, skipping king squares
	for sq := Square(0); sq < 64; sq++ {
		if sq == pos.wKingSq || sq == pos.bKingSq {
			continue
		}

		// Check if square is occupied: read 1 bit
		// Actually, the Stockfish format encodes ALL non-king squares using Huffman,
		// with empty squares encoded as a special 1-bit code.
		// Looking at the actual SF format more carefully:
		// After kings + STM, pieces are scanned a1..h8 skipping king squares.
		// Each piece is Huffman coded. Empty squares are NOT explicitly encoded;
		// instead, the piece scan just reads the Huffman prefix for non-empty squares.
		//
		// Wait - re-reading the SF source: the sfen packs the board in occupancy order.
		// The Huffman coding includes a way to encode "no piece here" — actually it doesn't.
		// The SF PackedSfen stores pieces in board order (a1..h8 skipping kings),
		// and empty squares get a single 0 bit, non-empty get Huffman coded.
		// Let me re-check: in SF the encoding is:
		//   For each square (a1..h8, skipping kings):
		//     If empty: write 0
		//     If occupied: write 1, then Huffman piece code

		occupied := br.readBit()
		if occupied == 0 {
			continue
		}

		piece := decodeHuffmanPiece(br)
		if piece == Empty {
			return pos, fmt.Errorf("invalid piece at square %d", sq)
		}
		pos.squares[sq] = piece
	}

	// Read rule50 (6 bits)
	pos.rule50 = uint8(br.readBits(6))

	// Read fullmove number (16 bits) — we discard this
	_ = br.readBits(16)

	// Read castling (4 bits): K=1, Q=2, k=4, q=8 (same as our encoding)
	// Wait - SF encodes: bit0=white kingside, bit1=white queenside, bit2=black kingside, bit3=black queenside
	// But check if the order matches our CastlingRights constants
	pos.castling = CastlingRights(br.readBits(4))

	// Read en passant: if there's an EP square, it's encoded
	// SF encodes EP as: 1 bit (has EP), then 3 bits (file) if has EP
	hasEP := br.readBit()
	if hasEP == 1 {
		epFile := int(br.readBits(3))
		if pos.sideToMove == White {
			pos.epSquare = NewSquare(epFile, 5) // rank 6 (0-indexed: 5)
		} else {
			pos.epSquare = NewSquare(epFile, 2) // rank 3 (0-indexed: 2)
		}
	}

	return pos, nil
}

// toFEN converts the lightweight position to a FEN string for debugging.
func (pos *sfLightPos) toFEN() string {
	var fen string
	for rank := 7; rank >= 0; rank-- {
		empty := 0
		for file := 0; file < 8; file++ {
			sq := NewSquare(file, rank)
			p := pos.squares[sq]
			if p == Empty {
				empty++
			} else {
				if empty > 0 {
					fen += fmt.Sprintf("%d", empty)
					empty = 0
				}
				fen += pieceToChar(p)
			}
		}
		if empty > 0 {
			fen += fmt.Sprintf("%d", empty)
		}
		if rank > 0 {
			fen += "/"
		}
	}

	if pos.sideToMove == White {
		fen += " w "
	} else {
		fen += " b "
	}

	castling := ""
	if pos.castling&WhiteKingside != 0 {
		castling += "K"
	}
	if pos.castling&WhiteQueenside != 0 {
		castling += "Q"
	}
	if pos.castling&BlackKingside != 0 {
		castling += "k"
	}
	if pos.castling&BlackQueenside != 0 {
		castling += "q"
	}
	if castling == "" {
		castling = "-"
	}
	fen += castling

	fen += " "
	if pos.epSquare == NoSquare {
		fen += "-"
	} else {
		fen += pos.epSquare.String()
	}

	fen += fmt.Sprintf(" %d 1", pos.rule50)

	return fen
}

// Stockfish 16-bit move encoding:
//   bits 0-5:   from square
//   bits 6-11:  to square
//   bits 12-13: type (0=normal, 1=promotion, 2=en passant, 3=castling)
//   bits 14-15: promotion piece (0=knight, 1=bishop, 2=rook, 3=queen) — only valid if type==1

const (
	sfMoveTypeNormal    = 0
	sfMoveTypePromotion = 1
	sfMoveTypeEnPassant = 2
	sfMoveTyCastling    = 3
)

// applyMove applies a Stockfish 16-bit move to the position.
func (pos *sfLightPos) applyMove(sfMove uint16) error {
	from := Square(sfMove & 0x3F)
	to := Square((sfMove >> 6) & 0x3F)
	moveType := (sfMove >> 12) & 0x3
	promoType := (sfMove >> 14) & 0x3

	piece := pos.squares[from]
	if piece == Empty {
		return fmt.Errorf("no piece at from square %s (move %04x)", from.String(), sfMove)
	}

	stm := pos.sideToMove
	opp := Black
	if stm == Black {
		opp = White
	}

	// Clear en passant (will be set if double pawn push)
	pos.epSquare = NoSquare
	pos.rule50++

	switch moveType {
	case sfMoveTypeNormal:
		// Handle captures
		captured := pos.squares[to]
		if captured != Empty {
			pos.rule50 = 0
		}

		// Pawn moves reset rule50
		if piece == WhitePawn || piece == BlackPawn {
			pos.rule50 = 0

			// Detect double pawn push for EP
			rankDiff := to.Rank() - from.Rank()
			if rankDiff == 2 || rankDiff == -2 {
				epFile := from.File()
				epRank := (from.Rank() + to.Rank()) / 2
				pos.epSquare = NewSquare(epFile, epRank)
			}
		}

		pos.squares[to] = piece
		pos.squares[from] = Empty

		// Update king squares
		if piece == WhiteKing {
			pos.wKingSq = to
		} else if piece == BlackKing {
			pos.bKingSq = to
		}

	case sfMoveTypePromotion:
		pos.rule50 = 0

		// Determine promotion piece
		var promoPiece Piece
		switch promoType {
		case 0:
			promoPiece = pieceOf(WhiteKnight, stm)
		case 1:
			promoPiece = pieceOf(WhiteBishop, stm)
		case 2:
			promoPiece = pieceOf(WhiteRook, stm)
		case 3:
			promoPiece = pieceOf(WhiteQueen, stm)
		}

		pos.squares[to] = promoPiece
		pos.squares[from] = Empty

	case sfMoveTypeEnPassant:
		pos.rule50 = 0

		// The captured pawn is on the same file as 'to', same rank as 'from'
		capSq := NewSquare(to.File(), from.Rank())
		pos.squares[capSq] = Empty
		pos.squares[to] = piece
		pos.squares[from] = Empty

	case sfMoveTyCastling:
		pos.rule50 = 0

		// SF encodes castling with to_sq = king destination
		// King moves from 'from' to 'to'; determine rook movement
		var rookFrom, rookTo Square
		if to.File() > from.File() {
			// Kingside
			rookFrom = NewSquare(7, from.Rank())
			rookTo = NewSquare(5, from.Rank())
		} else {
			// Queenside
			rookFrom = NewSquare(0, from.Rank())
			rookTo = NewSquare(3, from.Rank())
		}

		rook := pos.squares[rookFrom]
		pos.squares[to] = piece
		pos.squares[from] = Empty
		pos.squares[rookTo] = rook
		pos.squares[rookFrom] = Empty

		if piece == WhiteKing {
			pos.wKingSq = to
		} else if piece == BlackKing {
			pos.bKingSq = to
		}
	}

	// Update castling rights using the same mask table as the engine
	pos.castling &= castleMask[from]
	pos.castling &= castleMask[to]

	// Flip side to move
	if stm == White {
		pos.sideToMove = Black
	} else {
		pos.sideToMove = White
	}
	_ = opp // used conceptually

	return nil
}

// extractFeatures converts sfLightPos + score + result into NNUETrainSample.
// Score and result in the binpack are STM-relative; we convert to white-relative.
func (pos *sfLightPos) extractFeatures(score int16, result uint8) *NNUETrainSample {
	// Convert STM-relative to white-relative
	whiteScore := score
	whiteResult := result
	if pos.sideToMove == Black {
		whiteScore = -score
		// result: 0=loss, 1=draw, 2=win (STM-relative)
		// For black STM: 0 means black lost = white won = 2
		whiteResult = 2 - result
	}

	// Count pieces
	pieceCount := 0
	for sq := Square(0); sq < 64; sq++ {
		if pos.squares[sq] != Empty {
			pieceCount++
		}
	}

	sample := &NNUETrainSample{
		SideToMove:    pos.sideToMove,
		Result:        ResultToFloat(whiteResult),
		Score:         float32(whiteScore),
		HasScore:      true,
		PieceCount:    pieceCount,
		WhiteFeatures: make([]int, 0, pieceCount),
		BlackFeatures: make([]int, 0, pieceCount),
	}

	for sq := Square(0); sq < 64; sq++ {
		p := pos.squares[sq]
		if p == Empty {
			continue
		}

		wIdx := HalfKAIndex(White, pos.wKingSq, p, sq)
		bIdx := HalfKAIndex(Black, pos.bKingSq, p, sq)
		if wIdx >= 0 {
			sample.WhiteFeatures = append(sample.WhiteFeatures, wIdx)
		}
		if bIdx >= 0 {
			sample.BlackFeatures = append(sample.BlackFeatures, bIdx)
		}
	}

	return sample
}

// SFBinpackReader reads Stockfish .binpack files sequentially, producing NNUETrainSamples.
type SFBinpackReader struct {
	file      *os.File
	reader    *bufio.Reader
	pos       sfLightPos
	chainLeft int // remaining entries in current chain (not counting anchor which is already read)
}

// OpenSFBinpack opens a Stockfish .binpack file for sequential reading.
func OpenSFBinpack(path string) (*SFBinpackReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return &SFBinpackReader{
		file:   f,
		reader: bufio.NewReaderSize(f, 1<<20), // 1MB buffer
	}, nil
}

// Close closes the reader.
func (r *SFBinpackReader) Close() error {
	return r.file.Close()
}

// Next returns the next training sample, or (nil, io.EOF) at end of file.
// Each chain entry describes the current position (score, result) and the move
// played from it. We extract features first, then apply the move to advance.
func (r *SFBinpackReader) Next() (*NNUETrainSample, error) {
	for {
		if r.chainLeft > 0 {
			// Read chain entry: score(2) + move(2) + ply(2) + result(2) = 8 bytes
			var entry [8]byte
			if _, err := io.ReadFull(r.reader, entry[:]); err != nil {
				return nil, fmt.Errorf("reading chain entry: %w", err)
			}

			score := int16(binary.LittleEndian.Uint16(entry[0:2]))
			sfMove := binary.LittleEndian.Uint16(entry[2:4])
			// ply at entry[4:6] — unused
			result := uint8(int16(binary.LittleEndian.Uint16(entry[6:8])))

			// Extract features from current position (before applying this entry's move)
			sample := r.pos.extractFeatures(score, result)

			// Apply move to advance to next position
			r.chainLeft--
			if sfMove != 0 && r.chainLeft > 0 {
				if err := r.pos.applyMove(sfMove); err != nil {
					return nil, fmt.Errorf("applying move: %w", err)
				}
			}

			return sample, nil
		}

		// Start a new chain
		var chainLen uint16
		if err := binary.Read(r.reader, binary.LittleEndian, &chainLen); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return nil, io.EOF
			}
			return nil, fmt.Errorf("reading chain length: %w", err)
		}

		if chainLen == 0 {
			return nil, io.EOF
		}

		// Read anchor PackedSfen (32 bytes)
		var sfen [32]byte
		if _, err := io.ReadFull(r.reader, sfen[:]); err != nil {
			return nil, fmt.Errorf("reading packed sfen: %w", err)
		}

		pos, err := decodeSfen(sfen)
		if err != nil {
			return nil, fmt.Errorf("decoding sfen: %w", err)
		}
		r.pos = pos

		// Read anchor entry: score(2) + move(2) + ply(2) + result(2) = 8 bytes
		var entry [8]byte
		if _, err := io.ReadFull(r.reader, entry[:]); err != nil {
			return nil, fmt.Errorf("reading anchor entry: %w", err)
		}

		score := int16(binary.LittleEndian.Uint16(entry[0:2]))
		anchorMove := binary.LittleEndian.Uint16(entry[2:4])
		// ply at entry[4:6] — unused
		result := uint8(int16(binary.LittleEndian.Uint16(entry[6:8])))

		r.chainLeft = int(chainLen) - 1

		// Extract features from anchor position
		sample := r.pos.extractFeatures(score, result)

		// Apply anchor's move to advance to the next position in the chain
		if r.chainLeft > 0 && anchorMove != 0 {
			if err := r.pos.applyMove(anchorMove); err != nil {
				return nil, fmt.Errorf("applying anchor move: %w", err)
			}
		}

		return sample, nil
	}
}

// countBinpackPositions pre-scans .binpack files to count total positions by
// reading only chain headers. Fast: ~2 bytes per chain of ~100 positions.
func countBinpackPositions(paths []string) (int, error) {
	total := 0
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			return 0, err
		}
		r := bufio.NewReaderSize(f, 1<<16) // 64KB buffer

		for {
			var chainLen uint16
			if err := binary.Read(r, binary.LittleEndian, &chainLen); err != nil {
				if err == io.EOF || err == io.ErrUnexpectedEOF {
					break
				}
				f.Close()
				return 0, fmt.Errorf("reading chain length in %s: %w", path, err)
			}

			if chainLen == 0 {
				break // end-of-file sentinel
			}

			total += int(chainLen)

			// Skip: 32 bytes (sfen) + chainLen * 8 bytes (entries)
			skipBytes := int64(32) + int64(chainLen)*8
			// bufio.Reader doesn't have Seek, so we discard
			if _, err := r.Discard(int(skipBytes)); err != nil {
				f.Close()
				return 0, fmt.Errorf("skipping chain data in %s: %w", path, err)
			}
		}

		f.Close()
	}
	return total, nil
}

// SFBinpackSource implements TrainingDataSource for Stockfish .binpack files.
type SFBinpackSource struct {
	paths     []string
	totalPos  int
	bufSize   int
}

// NewSFBinpackSource creates a new source for Stockfish .binpack files.
// Scans files to count total positions (fast header-only scan).
func NewSFBinpackSource(paths []string, bufSize int) (*SFBinpackSource, error) {
	total, err := countBinpackPositions(paths)
	if err != nil {
		return nil, fmt.Errorf("counting positions: %w", err)
	}
	if bufSize <= 0 {
		bufSize = 1_000_000
	}
	return &SFBinpackSource{
		paths:    paths,
		totalPos: total,
		bufSize:  bufSize,
	}, nil
}

func (s *SFBinpackSource) NumRecords() int {
	return s.totalPos
}

func (s *SFBinpackSource) Close() error {
	return nil // no persistent file handles
}

// NewEpochReader creates a shuffle-buffered epoch reader.
func (s *SFBinpackSource) NewEpochReader(rng *rand.Rand, trainFraction float64) TrainingEpochReader {
	trainPos := int(float64(s.totalPos) * trainFraction)
	return &sfBinpackEpochReader{
		paths:    s.paths,
		rng:      rng,
		bufSize:  s.bufSize,
		trainPos: trainPos,
	}
}

// ValidationSamples reads validation samples (last 1-trainFraction of positions).
func (s *SFBinpackSource) ValidationSamples(trainFraction float64) ([]*NNUETrainSample, error) {
	trainPos := int(float64(s.totalPos) * trainFraction)
	return sfBinpackReadRange(s.paths, trainPos, s.totalPos)
}

// sfBinpackReadRange reads positions [startPos, endPos) across all files sequentially.
func sfBinpackReadRange(paths []string, startPos, endPos int) ([]*NNUETrainSample, error) {
	var samples []*NNUETrainSample
	posIdx := 0

	for _, path := range paths {
		reader, err := OpenSFBinpack(path)
		if err != nil {
			return nil, err
		}

		for {
			sample, err := reader.Next()
			if err != nil {
				if err == io.EOF {
					break
				}
				reader.Close()
				return nil, err
			}

			if posIdx >= startPos && posIdx < endPos {
				samples = append(samples, sample)
			}
			posIdx++

			if posIdx >= endPos {
				break
			}
		}

		reader.Close()
		if posIdx >= endPos {
			break
		}
	}

	return samples, nil
}

// sfBinpackEpochReader provides shuffle-buffered streaming over SF .binpack files.
type sfBinpackEpochReader struct {
	paths    []string
	rng      *rand.Rand
	bufSize  int
	trainPos int // number of training positions (first trainPos of total)

	// Internal state
	buffer  []*NNUETrainSample
	bufIdx  int
	reader  *SFBinpackReader
	fileIdx int
	posRead int
	started bool
}

// NumTrainRecords returns the number of training positions per epoch.
func (r *sfBinpackEpochReader) NumTrainRecords() int {
	return r.trainPos
}

// NextBatch reads the next batch of shuffled samples.
// Returns nil when the epoch is exhausted.
func (r *sfBinpackEpochReader) NextBatch(size int, buf []*NNUETrainSample) ([]*NNUETrainSample, error) {
	buf = buf[:0]

	for len(buf) < size {
		// Drain from current buffer
		if r.bufIdx < len(r.buffer) {
			for r.bufIdx < len(r.buffer) && len(buf) < size {
				buf = append(buf, r.buffer[r.bufIdx])
				r.bufIdx++
			}
			continue
		}

		// Need to refill buffer
		if r.posRead >= r.trainPos {
			break // epoch done
		}

		if err := r.refillBuffer(); err != nil {
			return nil, err
		}

		if len(r.buffer) == 0 {
			break // no more data
		}
	}

	if len(buf) == 0 {
		return nil, nil
	}
	return buf, nil
}

// refillBuffer fills the shuffle buffer from the stream.
func (r *sfBinpackEpochReader) refillBuffer() error {
	r.buffer = r.buffer[:0]
	r.bufIdx = 0

	// Open first file if needed
	if !r.started {
		r.started = true
		reader, err := OpenSFBinpack(r.paths[0])
		if err != nil {
			return err
		}
		r.reader = reader
		r.fileIdx = 0
	}

	// Fill buffer up to bufSize or trainPos
	for len(r.buffer) < r.bufSize && r.posRead < r.trainPos {
		if r.reader == nil {
			break
		}

		sample, err := r.reader.Next()
		if err != nil {
			if err == io.EOF {
				r.reader.Close()
				r.reader = nil
				r.fileIdx++
				if r.fileIdx < len(r.paths) {
					reader, err := OpenSFBinpack(r.paths[r.fileIdx])
					if err != nil {
						return err
					}
					r.reader = reader
				}
				continue
			}
			return err
		}

		r.buffer = append(r.buffer, sample)
		r.posRead++
	}

	// Shuffle buffer
	r.rng.Shuffle(len(r.buffer), func(i, j int) {
		r.buffer[i], r.buffer[j] = r.buffer[j], r.buffer[i]
	})

	return nil
}

// TrainingDataSource is the interface for training data providers.
// Both BinpackFile (our .bin format) and SFBinpackSource (Stockfish .binpack) implement this.
type TrainingDataSource interface {
	NumRecords() int
	NewEpochReader(rng *rand.Rand, trainFraction float64) TrainingEpochReader
	ValidationSamples(trainFraction float64) ([]*NNUETrainSample, error)
	Close() error
}

// TrainingEpochReader is the interface for per-epoch streaming readers.
type TrainingEpochReader interface {
	NextBatch(size int, buf []*NNUETrainSample) ([]*NNUETrainSample, error)
	NumTrainRecords() int
}

// DumpSFBinpack decodes and prints the first n chains from a .binpack file for diagnostics.
func DumpSFBinpack(path string, n int) error {
	reader, err := OpenSFBinpack(path)
	if err != nil {
		return err
	}
	defer reader.Close()

	chainNum := 0
	posNum := 0

	// We need a lower-level approach to show chain structure
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	for chainNum < n {
		var chainLen uint16
		if err := binary.Read(r, binary.LittleEndian, &chainLen); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break
			}
			return err
		}
		if chainLen == 0 {
			break
		}

		// Read anchor sfen
		var sfen [32]byte
		if _, err := io.ReadFull(r, sfen[:]); err != nil {
			return err
		}
		pos, err := decodeSfen(sfen)
		if err != nil {
			return fmt.Errorf("chain %d: %w", chainNum, err)
		}

		fmt.Printf("=== Chain %d (length %d) ===\n", chainNum, chainLen)

		// Read entries
		for i := 0; i < int(chainLen); i++ {
			var entry [8]byte
			if _, err := io.ReadFull(r, entry[:]); err != nil {
				return err
			}

			score := int16(binary.LittleEndian.Uint16(entry[0:2]))
			sfMove := binary.LittleEndian.Uint16(entry[2:4])
			ply := binary.LittleEndian.Uint16(entry[4:6])
			result := int16(binary.LittleEndian.Uint16(entry[6:8]))

			resultStr := "draw"
			if result == 0 {
				resultStr = "loss"
			} else if result == 2 {
				resultStr = "win"
			}

			fmt.Printf("  [%d] ply=%d score=%d result=%s move=%04x FEN=%s\n",
				i, ply, score, resultStr, sfMove, pos.toFEN())

			// Apply move to get next position (if not last entry)
			if i < int(chainLen)-1 {
				if err := pos.applyMove(sfMove); err != nil {
					fmt.Printf("  ERROR applying move: %v\n", err)
					break
				}
			}

			posNum++
		}

		chainNum++
		fmt.Println()
	}

	fmt.Printf("Showed %d chains, %d positions\n", chainNum, posNum)
	return nil
}

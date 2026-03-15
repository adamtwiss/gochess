package chess

import (
	"encoding/binary"
	"io"
	"os"
	"testing"
)

// TestDecodeSfenStartPos verifies that a hand-crafted PackedSfen for the starting
// position decodes correctly.
func TestDecodeSfenStartPos(t *testing.T) {
	// Build a PackedSfen for the starting position by encoding it manually.
	sfen := encodeSfenForTest(t, startPosPieces(), White, AllCastling, NoSquare, 0, 1)

	pos, err := decodeSfen(sfen)
	if err != nil {
		t.Fatalf("decodeSfen: %v", err)
	}

	if pos.sideToMove != White {
		t.Errorf("STM: got %d, want White", pos.sideToMove)
	}
	if pos.wKingSq != NewSquare(4, 0) {
		t.Errorf("wKingSq: got %s, want e1", pos.wKingSq.String())
	}
	if pos.bKingSq != NewSquare(4, 7) {
		t.Errorf("bKingSq: got %s, want e8", pos.bKingSq.String())
	}
	if pos.castling != AllCastling {
		t.Errorf("castling: got %d, want %d", pos.castling, AllCastling)
	}
	if pos.epSquare != NoSquare {
		t.Errorf("epSquare: got %s, want NoSquare", pos.epSquare.String())
	}

	// Verify all pieces
	var b Board
	b.Reset()
	b.SetFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
	for sq := Square(0); sq < 64; sq++ {
		if pos.squares[sq] != b.Squares[sq] {
			t.Errorf("square %s: got %d, want %d", sq.String(), pos.squares[sq], b.Squares[sq])
		}
	}
}

// TestSfLightPosApplyMove tests move application on the lightweight position.
func TestSfLightPosApplyMove(t *testing.T) {
	// Start from a known position and apply some moves, verifying against Board.MakeMove
	var b Board
	b.Reset()
	b.SetFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

	pos := boardToSfLightPos(&b)

	// 1. e2e4 (double pawn push)
	sfMove := encodeSfMove(NewSquare(4, 1), NewSquare(4, 3), sfMoveTypeNormal, 0)
	if err := pos.applyMove(sfMove); err != nil {
		t.Fatalf("e2e4: %v", err)
	}
	if pos.sideToMove != Black {
		t.Errorf("after e4: STM should be Black")
	}
	if pos.epSquare != NewSquare(4, 2) {
		t.Errorf("after e4: EP square should be e3, got %s", pos.epSquare.String())
	}
	if pos.squares[NewSquare(4, 3)] != WhitePawn {
		t.Errorf("after e4: e4 should have white pawn")
	}
	if pos.squares[NewSquare(4, 1)] != Empty {
		t.Errorf("after e4: e2 should be empty")
	}

	// 2. e7e5 (double pawn push)
	sfMove = encodeSfMove(NewSquare(4, 6), NewSquare(4, 4), sfMoveTypeNormal, 0)
	if err := pos.applyMove(sfMove); err != nil {
		t.Fatalf("e7e5: %v", err)
	}
	if pos.epSquare != NewSquare(4, 5) {
		t.Errorf("after e5: EP square should be e6, got %s", pos.epSquare.String())
	}

	// 3. Nf3
	sfMove = encodeSfMove(NewSquare(6, 0), NewSquare(5, 2), sfMoveTypeNormal, 0)
	if err := pos.applyMove(sfMove); err != nil {
		t.Fatalf("Nf3: %v", err)
	}
	if pos.squares[NewSquare(5, 2)] != WhiteKnight {
		t.Errorf("after Nf3: f3 should have white knight")
	}
	if pos.epSquare != NoSquare {
		t.Errorf("after Nf3: EP square should be NoSquare")
	}
}

// TestSfLightPosCastling tests castling move application.
func TestSfLightPosCastling(t *testing.T) {
	var b Board
	b.Reset()
	b.SetFEN("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
	pos := boardToSfLightPos(&b)

	// Kingside castling: O-O
	sfMove := encodeSfMove(NewSquare(4, 0), NewSquare(6, 0), sfMoveTyCastling, 0)
	if err := pos.applyMove(sfMove); err != nil {
		t.Fatalf("O-O: %v", err)
	}
	if pos.squares[NewSquare(6, 0)] != WhiteKing {
		t.Errorf("after O-O: g1 should have white king")
	}
	if pos.squares[NewSquare(5, 0)] != WhiteRook {
		t.Errorf("after O-O: f1 should have white rook")
	}
	if pos.squares[NewSquare(4, 0)] != Empty {
		t.Errorf("after O-O: e1 should be empty")
	}
	if pos.squares[NewSquare(7, 0)] != Empty {
		t.Errorf("after O-O: h1 should be empty")
	}
	if pos.wKingSq != NewSquare(6, 0) {
		t.Errorf("after O-O: wKingSq should be g1")
	}
	// White castling rights should be cleared
	if pos.castling&(WhiteKingside|WhiteQueenside) != 0 {
		t.Errorf("after O-O: white castling rights should be gone")
	}

	// Black queenside castling: O-O-O
	sfMove = encodeSfMove(NewSquare(4, 7), NewSquare(2, 7), sfMoveTyCastling, 0)
	if err := pos.applyMove(sfMove); err != nil {
		t.Fatalf("O-O-O: %v", err)
	}
	if pos.squares[NewSquare(2, 7)] != BlackKing {
		t.Errorf("after O-O-O: c8 should have black king")
	}
	if pos.squares[NewSquare(3, 7)] != BlackRook {
		t.Errorf("after O-O-O: d8 should have black rook")
	}
	if pos.bKingSq != NewSquare(2, 7) {
		t.Errorf("after O-O-O: bKingSq should be c8")
	}
}

// TestSfLightPosEnPassant tests en passant capture.
func TestSfLightPosEnPassant(t *testing.T) {
	var b Board
	b.Reset()
	b.SetFEN("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3")
	pos := boardToSfLightPos(&b)

	// f5xe6 en passant
	sfMove := encodeSfMove(NewSquare(5, 4), NewSquare(4, 5), sfMoveTypeEnPassant, 0)
	if err := pos.applyMove(sfMove); err != nil {
		t.Fatalf("fxe6 EP: %v", err)
	}
	if pos.squares[NewSquare(4, 5)] != WhitePawn {
		t.Errorf("after fxe6: e6 should have white pawn")
	}
	if pos.squares[NewSquare(4, 4)] != Empty {
		t.Errorf("after fxe6: e5 should be empty (captured)")
	}
	if pos.squares[NewSquare(5, 4)] != Empty {
		t.Errorf("after fxe6: f5 should be empty")
	}
}

// TestSfLightPosPromotion tests pawn promotion.
func TestSfLightPosPromotion(t *testing.T) {
	var b Board
	b.Reset()
	b.SetFEN("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
	pos := boardToSfLightPos(&b)

	// e7e8=Q
	sfMove := encodeSfMove(NewSquare(4, 6), NewSquare(4, 7), sfMoveTypePromotion, 3) // 3=queen
	if err := pos.applyMove(sfMove); err != nil {
		t.Fatalf("e8=Q: %v", err)
	}
	if pos.squares[NewSquare(4, 7)] != WhiteQueen {
		t.Errorf("after e8=Q: e8 should have white queen, got %d", pos.squares[NewSquare(4, 7)])
	}
}

// TestExtractFeaturesSTMConversion verifies that STM-relative scores/results
// are correctly converted to white-relative.
func TestExtractFeaturesSTMConversion(t *testing.T) {
	var b Board
	b.Reset()
	b.SetFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
	pos := boardToSfLightPos(&b)

	// STM is Black. Score=+50 means +50 for Black = -50 for White
	// Result=2 (win) means Black wins = 0 for White
	sample := pos.extractFeatures(50, 2)

	if sample.Score != -50 {
		t.Errorf("score: got %.0f, want -50 (white-relative)", sample.Score)
	}
	if sample.Result != 0.0 {
		t.Errorf("result: got %.1f, want 0.0 (black win = white loss)", sample.Result)
	}

	// STM is Black. Score=-100 = -100 for Black = +100 for White
	// Result=0 (loss) means Black lost = 2 for White (white wins)
	sample2 := pos.extractFeatures(-100, 0)
	if sample2.Score != 100 {
		t.Errorf("score2: got %.0f, want 100", sample2.Score)
	}
	if sample2.Result != 1.0 {
		t.Errorf("result2: got %.1f, want 1.0 (white win)", sample2.Result)
	}
}

// TestSfBinpackWriteReadRoundtrip creates a small .binpack file, reads it back,
// and verifies the decoded positions match.
func TestSfBinpackWriteReadRoundtrip(t *testing.T) {
	// Create a minimal .binpack file with a 2-position chain
	tmpFile := t.TempDir() + "/test.binpack"
	f, err := os.Create(tmpFile)
	if err != nil {
		t.Fatal(err)
	}

	// Chain length = 2
	binary.Write(f, binary.LittleEndian, uint16(2))

	// Anchor sfen: starting position
	sfen := encodeSfenForTest(t, startPosPieces(), White, AllCastling, NoSquare, 0, 1)
	f.Write(sfen[:])

	// Anchor entry: score=15, move=e2e4, ply=0, result=1 (draw)
	e2e4 := encodeSfMove(NewSquare(4, 1), NewSquare(4, 3), sfMoveTypeNormal, 0)
	writeChainEntry(f, 15, e2e4, 0, 1)

	// Second entry: score=-10, move=e7e5, ply=1, result=1 (draw)
	e7e5 := encodeSfMove(NewSquare(4, 6), NewSquare(4, 4), sfMoveTypeNormal, 0)
	writeChainEntry(f, -10, e7e5, 1, 1)

	// EOF sentinel
	binary.Write(f, binary.LittleEndian, uint16(0))
	f.Close()

	// Read it back
	reader, err := OpenSFBinpack(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()

	// First sample: anchor (starting position, White to move)
	s1, err := reader.Next()
	if err != nil {
		t.Fatalf("reading sample 1: %v", err)
	}
	if s1.SideToMove != White {
		t.Errorf("s1 STM: got %d, want White", s1.SideToMove)
	}
	// Score is STM-relative, White to move, so white-relative score = 15
	if s1.Score != 15 {
		t.Errorf("s1 score: got %.0f, want 15", s1.Score)
	}

	// Second sample: after e2e4, Black to move
	s2, err := reader.Next()
	if err != nil {
		t.Fatalf("reading sample 2: %v", err)
	}
	if s2.SideToMove != Black {
		t.Errorf("s2 STM: got %d, want Black", s2.SideToMove)
	}
	// Score is -10 STM-relative (Black), so white-relative = +10
	if s2.Score != 10 {
		t.Errorf("s2 score: got %.0f, want 10 (STM-relative -10 for Black = white +10)", s2.Score)
	}

	// Should be EOF
	s3, err := reader.Next()
	if err != io.EOF {
		t.Errorf("expected EOF, got sample=%v err=%v", s3, err)
	}
}

// TestCountBinpackPositions verifies the pre-scan position counter.
func TestCountBinpackPositions(t *testing.T) {
	tmpFile := t.TempDir() + "/test.binpack"
	f, err := os.Create(tmpFile)
	if err != nil {
		t.Fatal(err)
	}

	// Chain 1: length 3
	binary.Write(f, binary.LittleEndian, uint16(3))
	sfen := encodeSfenForTest(t, startPosPieces(), White, AllCastling, NoSquare, 0, 1)
	f.Write(sfen[:])
	e2e4 := encodeSfMove(NewSquare(4, 1), NewSquare(4, 3), sfMoveTypeNormal, 0)
	writeChainEntry(f, 10, e2e4, 0, 1)
	e7e5 := encodeSfMove(NewSquare(4, 6), NewSquare(4, 4), sfMoveTypeNormal, 0)
	writeChainEntry(f, -5, e7e5, 1, 1)
	nf3 := encodeSfMove(NewSquare(6, 0), NewSquare(5, 2), sfMoveTypeNormal, 0)
	writeChainEntry(f, 8, nf3, 2, 1)

	// Chain 2: length 1
	binary.Write(f, binary.LittleEndian, uint16(1))
	f.Write(sfen[:])
	writeChainEntry(f, 0, 0, 0, 1)

	// EOF sentinel
	binary.Write(f, binary.LittleEndian, uint16(0))
	f.Close()

	count, err := countBinpackPositions([]string{tmpFile})
	if err != nil {
		t.Fatal(err)
	}
	if count != 4 {
		t.Errorf("count: got %d, want 4 (3 + 1)", count)
	}
}

// TestSFBinpackSourceInterface verifies that SFBinpackSource satisfies TrainingDataSource.
func TestSFBinpackSourceInterface(t *testing.T) {
	tmpFile := t.TempDir() + "/test.binpack"
	f, err := os.Create(tmpFile)
	if err != nil {
		t.Fatal(err)
	}

	// Single chain with 2 entries
	binary.Write(f, binary.LittleEndian, uint16(2))
	sfen := encodeSfenForTest(t, startPosPieces(), White, AllCastling, NoSquare, 0, 1)
	f.Write(sfen[:])
	e2e4 := encodeSfMove(NewSquare(4, 1), NewSquare(4, 3), sfMoveTypeNormal, 0)
	writeChainEntry(f, 10, e2e4, 0, 1)
	d7d5 := encodeSfMove(NewSquare(3, 6), NewSquare(3, 4), sfMoveTypeNormal, 0)
	writeChainEntry(f, -15, d7d5, 1, 2)
	binary.Write(f, binary.LittleEndian, uint16(0))
	f.Close()

	src, err := NewSFBinpackSource([]string{tmpFile}, 100)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()

	if src.NumRecords() != 2 {
		t.Errorf("NumRecords: got %d, want 2", src.NumRecords())
	}

	// Verify it satisfies the interface
	var _ TrainingDataSource = src
}

// TestBinpackFileTrainingDataSource verifies that BinpackFile satisfies TrainingDataSource.
func TestBinpackFileTrainingDataSource(t *testing.T) {
	// Create a small .bin file
	tmpFile := t.TempDir() + "/test.bin"
	f, err := os.Create(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	var b Board
	b.Reset()
	b.SetFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
	rec := PackPosition(&b, 10, 1)
	f.Write(rec[:])
	f.Close()

	bf, err := OpenBinpackFiles(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	defer bf.Close()

	// Verify it satisfies the interface
	var _ TrainingDataSource = bf
}

// --- Test helpers ---

// encodeSfMove encodes a move in Stockfish's 16-bit format.
func encodeSfMove(from, to Square, moveType, promoType int) uint16 {
	return uint16(from) | uint16(to)<<6 | uint16(moveType)<<12 | uint16(promoType)<<14
}

// writeChainEntry writes an 8-byte chain entry.
func writeChainEntry(f *os.File, score int16, move uint16, ply uint16, result int16) {
	binary.Write(f, binary.LittleEndian, score)
	binary.Write(f, binary.LittleEndian, move)
	binary.Write(f, binary.LittleEndian, ply)
	binary.Write(f, binary.LittleEndian, result)
}

// boardToSfLightPos creates an sfLightPos from a Board.
func boardToSfLightPos(b *Board) sfLightPos {
	var pos sfLightPos
	for i := range pos.squares {
		pos.squares[i] = b.Squares[i]
	}
	pos.sideToMove = b.SideToMove
	pos.castling = b.Castling
	pos.epSquare = b.EnPassant
	if b.HalfmoveClock > 255 {
		pos.rule50 = 255
	} else {
		pos.rule50 = uint8(b.HalfmoveClock)
	}
	// Find kings
	for sq := Square(0); sq < 64; sq++ {
		if b.Squares[sq] == WhiteKing {
			pos.wKingSq = sq
		} else if b.Squares[sq] == BlackKing {
			pos.bKingSq = sq
		}
	}
	return pos
}

// startPosPieces returns the piece layout of the starting position.
func startPosPieces() [64]Piece {
	var sq [64]Piece
	// Rank 1
	sq[0] = WhiteRook
	sq[1] = WhiteKnight
	sq[2] = WhiteBishop
	sq[3] = WhiteQueen
	sq[4] = WhiteKing
	sq[5] = WhiteBishop
	sq[6] = WhiteKnight
	sq[7] = WhiteRook
	// Rank 2
	for f := 0; f < 8; f++ {
		sq[8+f] = WhitePawn
	}
	// Ranks 3-6: empty (default)
	// Rank 7
	for f := 0; f < 8; f++ {
		sq[48+f] = BlackPawn
	}
	// Rank 8
	sq[56] = BlackRook
	sq[57] = BlackKnight
	sq[58] = BlackBishop
	sq[59] = BlackQueen
	sq[60] = BlackKing
	sq[61] = BlackBishop
	sq[62] = BlackKnight
	sq[63] = BlackRook
	return sq
}

// encodeSfenForTest manually encodes a PackedSfen for testing the decoder.
func encodeSfenForTest(t *testing.T, pieces [64]Piece, stm Color, castling CastlingRights, ep Square, rule50 uint8, fullmove uint16) [32]byte {
	t.Helper()

	var data [32]byte
	bw := &bitWriter{data: data[:]}

	// Find king squares
	var wKingSq, bKingSq Square
	for sq := Square(0); sq < 64; sq++ {
		if pieces[sq] == WhiteKing {
			wKingSq = sq
		} else if pieces[sq] == BlackKing {
			bKingSq = sq
		}
	}

	// Write king squares (6 bits each)
	bw.writeBits(uint32(wKingSq), 6)
	bw.writeBits(uint32(bKingSq), 6)

	// Side to move
	bw.writeBits(uint32(stm), 1)

	// Pieces a1..h8, skipping kings
	for sq := Square(0); sq < 64; sq++ {
		if sq == wKingSq || sq == bKingSq {
			continue
		}
		p := pieces[sq]
		if p == Empty {
			bw.writeBit(0)
		} else {
			bw.writeBit(1)
			encodeHuffmanPieceForTest(bw, p)
		}
	}

	// rule50 (6 bits)
	bw.writeBits(uint32(rule50), 6)

	// fullmove (16 bits)
	bw.writeBits(uint32(fullmove), 16)

	// castling (4 bits)
	bw.writeBits(uint32(castling), 4)

	// en passant
	if ep == NoSquare {
		bw.writeBit(0)
	} else {
		bw.writeBit(1)
		bw.writeBits(uint32(ep.File()), 3)
	}

	copy(data[:], bw.data)
	return data
}

// bitWriter writes variable-length bit sequences to a byte slice (LSB-first).
type bitWriter struct {
	data   []byte
	bitPos int
}

func (bw *bitWriter) writeBit(bit uint8) {
	byteIdx := bw.bitPos / 8
	bitIdx := bw.bitPos % 8
	if bit != 0 {
		bw.data[byteIdx] |= 1 << uint(bitIdx)
	}
	bw.bitPos++
}

func (bw *bitWriter) writeBits(val uint32, n int) {
	for i := 0; i < n; i++ {
		bw.writeBit(uint8((val >> uint(i)) & 1))
	}
}

// encodeHuffmanPieceForTest encodes a piece using the SF Huffman scheme.
func encodeHuffmanPieceForTest(bw *bitWriter, p Piece) {
	// Determine piece type (1-5) and color
	var pieceType int
	var color Color
	switch p {
	case WhitePawn:
		pieceType, color = 1, White
	case BlackPawn:
		pieceType, color = 1, Black
	case WhiteKnight:
		pieceType, color = 2, White
	case BlackKnight:
		pieceType, color = 2, Black
	case WhiteBishop:
		pieceType, color = 3, White
	case BlackBishop:
		pieceType, color = 3, Black
	case WhiteRook:
		pieceType, color = 4, White
	case BlackRook:
		pieceType, color = 4, Black
	case WhiteQueen:
		pieceType, color = 5, White
	case BlackQueen:
		pieceType, color = 5, Black
	}

	// Write (pieceType-1) zeros, then a 1
	for i := 1; i < pieceType; i++ {
		bw.writeBit(0)
	}
	bw.writeBit(1)

	// Color bit
	bw.writeBit(uint8(color))
}

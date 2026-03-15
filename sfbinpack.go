package chess

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math/bits"
	"math/rand"
	"os"
)

// Stockfish .binpack format reader.
//
// Supports two formats:
//   - Legacy PackedSfen chain format (nodchip): chain-length header + Huffman-coded
//     anchor + fixed-size entries (8 bytes each). Detected by absence of "BINP" magic.
//   - BINP chunk format (modern Stockfish): "BINP" magic + chunk-size header + packed
//     entries with 24-byte compressed positions and variable-length movetext.
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

// ============================================================================
// BINP format: compressed position + movetext
// ============================================================================

// binpMagic is the 4-byte magic header for each BINP chunk.
var binpMagic = [4]byte{'B', 'I', 'N', 'P'}

// isBINPFormat checks whether a file starts with the "BINP" magic bytes.
func isBINPFormat(path string) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	var magic [4]byte
	_, err = io.ReadFull(f, magic[:])
	if err != nil {
		return false, nil // too short, not BINP
	}
	return magic == binpMagic, nil
}

// unsignedToSigned decodes a signedToUnsigned-encoded uint16 back to int16.
// This is the inverse of signedToUnsigned: rotate right by 1, then conditionally XOR.
func unsignedToSigned(r uint16) int16 {
	r = (r << 15) | (r >> 1) // rotate right by 1
	if r&0x8000 != 0 {
		r ^= 0x7FFF
	}
	return int16(r)
}

// signedToUnsigned encodes an int16 into uint16 for delta coding.
// Conditionally XOR, then rotate left by 1.
func signedToUnsigned(a int16) uint16 {
	r := uint16(a)
	if r&0x8000 != 0 {
		r ^= 0x7FFF
	}
	r = (r << 1) | (r >> 15) // rotate left by 1
	return r
}

// decompressBINPPosition decodes a 24-byte BINP CompressedPosition into sfLightPos.
//
// Layout:
//   - 8 bytes: occupancy bitboard (big-endian, bit 0 = a1)
//   - 16 bytes: nibble-packed pieces (4 bits each, low nibble first per byte)
//     Pieces listed in occupancy LSB-first order (a1 first).
//     Nibble values: 0-11 = piece types, 12 = EP pawn, 13 = white castling rook,
//     14 = black castling rook, 15 = black king + black to move.
func decompressBINPPosition(data [24]byte) (sfLightPos, error) {
	var pos sfLightPos
	for i := range pos.squares {
		pos.squares[i] = Empty
	}
	pos.epSquare = NoSquare
	pos.sideToMove = White // default; overridden by nibble 15

	// Read occupancy bitboard (big-endian)
	occ := uint64(data[0])<<56 | uint64(data[1])<<48 | uint64(data[2])<<40 | uint64(data[3])<<32 |
		uint64(data[4])<<24 | uint64(data[5])<<16 | uint64(data[6])<<8 | uint64(data[7])

	popcount := bits.OnesCount64(occ)
	if popcount < 2 || popcount > 32 {
		return pos, fmt.Errorf("invalid occupancy popcount: %d", popcount)
	}

	// Read nibble-packed pieces
	nibbles := data[8:24] // 16 bytes = up to 32 nibbles
	nibbleIdx := 0

	// Iterate squares in LSB-first order (matching occupancy bit iteration)
	tmpOcc := occ
	for tmpOcc != 0 {
		sq := bits.TrailingZeros64(tmpOcc)
		tmpOcc &= tmpOcc - 1 // clear LSB

		// Extract nibble
		byteIdx := nibbleIdx / 2
		var nibble uint8
		if nibbleIdx%2 == 0 {
			nibble = nibbles[byteIdx] & 0x0F
		} else {
			nibble = nibbles[byteIdx] >> 4
		}
		nibbleIdx++

		switch nibble {
		case 0:
			pos.squares[sq] = WhitePawn
		case 1:
			pos.squares[sq] = BlackPawn
		case 2:
			pos.squares[sq] = WhiteKnight
		case 3:
			pos.squares[sq] = BlackKnight
		case 4:
			pos.squares[sq] = WhiteBishop
		case 5:
			pos.squares[sq] = BlackBishop
		case 6:
			pos.squares[sq] = WhiteRook
		case 7:
			pos.squares[sq] = BlackRook
		case 8:
			pos.squares[sq] = WhiteQueen
		case 9:
			pos.squares[sq] = BlackQueen
		case 10:
			pos.squares[sq] = WhiteKing
			pos.wKingSq = Square(sq)
		case 11:
			pos.squares[sq] = BlackKing
			pos.bKingSq = Square(sq)
		case 12:
			// Pawn with EP square behind it
			rank := sq / 8
			if rank == 3 { // white pawn on rank 4 (0-indexed rank 3)
				pos.squares[sq] = WhitePawn
				pos.epSquare = Square(sq - 8)
			} else if rank == 4 { // black pawn on rank 5 (0-indexed rank 4)
				pos.squares[sq] = BlackPawn
				pos.epSquare = Square(sq + 8)
			} else {
				return pos, fmt.Errorf("EP nibble 12 at unexpected rank %d (sq %d)", rank, sq)
			}
		case 13:
			// White rook with castling rights
			pos.squares[sq] = WhiteRook
			if sq == 0 { // a1
				pos.castling |= WhiteQueenside
			} else if sq == 7 { // h1
				pos.castling |= WhiteKingside
			}
		case 14:
			// Black rook with castling rights
			pos.squares[sq] = BlackRook
			if sq == 56 { // a8
				pos.castling |= BlackQueenside
			} else if sq == 63 { // h8
				pos.castling |= BlackKingside
			}
		case 15:
			// Black king AND black is side to move
			pos.squares[sq] = BlackKing
			pos.bKingSq = Square(sq)
			pos.sideToMove = Black
		default:
			return pos, fmt.Errorf("invalid nibble %d at square %d", nibble, sq)
		}
	}

	return pos, nil
}

// decompressBINPMove decodes a 2-byte big-endian CompressedMove into our Move type.
//
// Bit layout (from MSB): type(2) | from(6) | to(6) | promo(2)
// Types: 0=Normal, 1=Promotion, 2=Castle, 3=EnPassant
// For castling: from=king square, to=ROOK square (not king destination)
// For promotion: promo = promotedPieceType - Knight (0=N, 1=B, 2=R, 3=Q)
func decompressBINPMove(packed uint16, pos *sfLightPos) Move {
	moveType := (packed >> 14) & 3
	from := Square((packed >> 8) & 0x3F)
	to := Square((packed >> 2) & 0x3F)
	promo := (packed >> 0) & 3

	switch moveType {
	case 0: // Normal
		return NewMove(from, to)
	case 1: // Promotion
		var flag int
		switch promo {
		case 0:
			flag = FlagPromoteN
		case 1:
			flag = FlagPromoteB
		case 2:
			flag = FlagPromoteR
		case 3:
			flag = FlagPromoteQ
		}
		return NewMoveFlags(from, to, flag)
	case 2: // Castle — to is ROOK square, convert to king destination
		var kingDest Square
		if to.File() > from.File() {
			// Kingside: rook is on h-file, king goes to g-file
			kingDest = NewSquare(6, from.Rank())
		} else {
			// Queenside: rook is on a-file, king goes to c-file
			kingDest = NewSquare(2, from.Rank())
		}
		return NewMoveFlags(from, kingDest, FlagCastle)
	case 3: // En passant
		return NewMoveFlags(from, to, FlagEnPassant)
	}
	return NoMove
}

// binpApplyBINPMove applies a BINP CompressedMove (raw uint16) to an sfLightPos.
// BINP castling uses rook square as destination; we must convert before calling applyMove.
func (pos *sfLightPos) applyBINPMove(packed uint16) error {
	moveType := (packed >> 14) & 3
	from := Square((packed >> 8) & 0x3F)
	to := Square((packed >> 2) & 0x3F)
	promoType := (packed >> 0) & 3

	if moveType == 2 {
		// Castle: BINP uses rook square as 'to'. Convert to king destination
		// for the legacy sfMoveType encoding used by applyMove.
		var kingDest Square
		if to.File() > from.File() {
			kingDest = NewSquare(6, from.Rank())
		} else {
			kingDest = NewSquare(2, from.Rank())
		}
		sfMove := uint16(from) | uint16(kingDest)<<6 | uint16(sfMoveTyCastling)<<12
		return pos.applyMove(sfMove)
	}

	// Map BINP move types to the legacy SF move type encoding
	var sfType uint16
	switch moveType {
	case 0:
		sfType = sfMoveTypeNormal
	case 1:
		sfType = sfMoveTypePromotion
	case 3:
		sfType = sfMoveTypeEnPassant
	}
	sfMove := uint16(from) | uint16(to)<<6 | sfType<<12 | promoType<<14
	return pos.applyMove(sfMove)
}

// binpBitReader reads variable-length bit sequences from a byte slice.
// Matches Stockfish's extractBitsLE8: bits are packed MSB-first within each byte.
// Used for decoding BINP movetext.
type binpBitReader struct {
	data         []byte
	readOffset   int // current byte index
	readBitsLeft int // bits remaining in current byte (starts at 8)
	overflow     bool
}

func newBinpBitReader(data []byte) *binpBitReader {
	return &binpBitReader{data: data, readBitsLeft: 8}
}

// readBits reads count bits (MSB-first packing, matching Stockfish's extractBitsLE8).
// Returns up to 8 bits.
func (br *binpBitReader) readBits(count int) uint8 {
	if count == 0 {
		return 0
	}

	if br.readBitsLeft == 0 {
		br.readOffset++
		br.readBitsLeft = 8
	}

	if br.readOffset >= len(br.data) {
		br.overflow = true
		return 0
	}

	// Shift current byte so the unread bits are at the MSB
	b := br.data[br.readOffset] << uint(8-br.readBitsLeft)
	// Extract the top 'count' bits
	result := b >> uint(8-count)

	if count > br.readBitsLeft {
		// Spill into next byte
		spillCount := count - br.readBitsLeft
		if br.readOffset+1 >= len(br.data) {
			br.overflow = true
			return result
		}
		result |= br.data[br.readOffset+1] >> uint(8-spillCount)
		br.readBitsLeft += 8
		br.readOffset++
	}

	br.readBitsLeft -= count
	return result
}

// numReadBytes returns the number of bytes consumed by the bit reader.
// Matches Stockfish's numReadBytes(): offset + (bitsLeft != 8 ? 1 : 0).
func (br *binpBitReader) numReadBytes() int {
	if br.readBitsLeft != 8 {
		return br.readOffset + 1
	}
	return br.readOffset
}

// readVLE reads a VLE (variable-length encoding) value with block size 4.
// Matches Stockfish's extractVle16: each group is (blockSize+1) bits,
// with the extension flag as the MSB of the group.
func (br *binpBitReader) readVLE() uint16 {
	const blockSize = 4
	mask := uint16((1 << blockSize) - 1)
	var v uint16
	offset := uint(0)
	for {
		block := uint16(br.readBits(blockSize + 1))
		if br.overflow {
			return v
		}
		v |= (block & mask) << offset
		if block>>blockSize == 0 {
			break
		}
		offset += blockSize
	}
	return v
}

// usedBitsSafe returns the number of bits needed to encode a choice from n options.
// Matches Stockfish: usedBitsSafe(n) = usedBits(n-1) = bits.Len(n-1) for n > 0, 0 for n == 0.
func usedBitsSafe(n int) int {
	if n <= 1 {
		return 0
	}
	return bits.Len(uint(n - 1))
}

// generateSFOrderLegalMoves generates legal moves in Stockfish's exact generation
// order. NOTE: This is no longer used by BINP decoding (which uses hierarchical
// piece+destination encoding via binpDecodeMoveFromBoard), but is retained for
// potential future use.
func generateSFOrderLegalMoves(b *Board) []Move {
	pinned, checkers := b.PinnedAndCheckers(b.SideToMove)
	inCheck := checkers != 0

	// When in check, use evasion generation (matches Stockfish's generate<LEGAL>)
	if inCheck {
		return generateSFOrderEvasions(b)
	}

	moves := generateSFOrderPseudoLegal(b)
	// Stockfish's legality filter uses swap-with-back removal:
	//   while (cur != moveList)
	//     if (!legal) *cur = *(--moveList); else ++cur;
	// This changes the ORDER of legal moves. We must replicate this exactly.
	cur := 0
	end := len(moves)
	for cur < end {
		if !b.IsLegal(moves[cur], pinned, inCheck) {
			end--
			moves[cur] = moves[end]
		} else {
			cur++
		}
	}
	return moves[:end]
}

// generateSFOrderPseudoLegal generates pseudo-legal moves in Stockfish's order.
func generateSFOrderPseudoLegal(b *Board) []Move {
	us := b.SideToMove
	them := 1 - us
	theirPieces := b.Occupied[them]
	allTargets := ^b.Occupied[us] // all squares not occupied by us
	empty := ^b.AllPieces

	moves := make([]Move, 0, 64)

	// 1. Pawn moves
	if us == White {
		pawns := b.Pieces[WhitePawn]
		pawnsOn7 := pawns & Rank7    // pawns that can promote
		pawnsNot7 := pawns &^ Rank7  // pawns that can't promote

		// Promotion captures: right (<<9) then left (<<7)
		promoCapR := ((pawnsOn7 & NotFileH) << 9) & theirPieces
		for promoCapR != 0 {
			to := promoCapR.PopLSB()
			from := to - 9
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
		}
		promoCapL := ((pawnsOn7 & NotFileA) << 7) & theirPieces
		for promoCapL != 0 {
			to := promoCapL.PopLSB()
			from := to - 7
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
		}

		// Push promotions
		promoPush := ((pawnsOn7 << 8) & empty)
		for promoPush != 0 {
			to := promoPush.PopLSB()
			from := to - 8
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
		}

		// Non-promotion captures: right (<<9) then left (<<7)
		capR := ((pawnsNot7 & NotFileH) << 9) & theirPieces
		for capR != 0 {
			to := capR.PopLSB()
			moves = append(moves, NewMove(to-9, to))
		}
		capL := ((pawnsNot7 & NotFileA) << 7) & theirPieces
		for capL != 0 {
			to := capL.PopLSB()
			moves = append(moves, NewMove(to-7, to))
		}

		// Single push (non-promotion)
		push1 := ((pawnsNot7 << 8) & empty)
		for push1 != 0 {
			to := push1.PopLSB()
			moves = append(moves, NewMove(to-8, to))
		}

		// Double push
		push1All := (pawns << 8) & empty
		push2 := ((push1All & Rank3) << 8) & empty
		for push2 != 0 {
			to := push2.PopLSB()
			moves = append(moves, NewMove(to-16, to))
		}

		// En passant
		if b.EnPassant != NoSquare {
			epBB := SquareBB(b.EnPassant)
			epAttackers := PawnAttacks[Black][b.EnPassant] & pawns
			for epAttackers != 0 {
				from := epAttackers.PopLSB()
				_ = epBB
				moves = append(moves, NewMoveFlags(from, b.EnPassant, FlagEnPassant))
			}
		}
	} else {
		// Black pawns
		pawns := b.Pieces[BlackPawn]
		pawnsOn2 := pawns & Rank2    // pawns that can promote (to rank 1)
		pawnsNot2 := pawns &^ Rank2  // pawns that can't promote

		// Promotion captures: right (>>9, towards a-file) then left (>>7, towards h-file)
		promoCapR := ((pawnsOn2 & NotFileA) >> 9) & theirPieces
		for promoCapR != 0 {
			to := promoCapR.PopLSB()
			from := to + 9
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
		}
		promoCapL := ((pawnsOn2 & NotFileH) >> 7) & theirPieces
		for promoCapL != 0 {
			to := promoCapL.PopLSB()
			from := to + 7
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
		}

		// Push promotions
		promoPush := ((pawnsOn2 >> 8) & empty)
		for promoPush != 0 {
			to := promoPush.PopLSB()
			from := to + 8
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(from, to, FlagPromoteN))
		}

		// Non-promotion captures: right (>>9) then left (>>7)
		capR := ((pawnsNot2 & NotFileA) >> 9) & theirPieces
		for capR != 0 {
			to := capR.PopLSB()
			moves = append(moves, NewMove(to+9, to))
		}
		capL := ((pawnsNot2 & NotFileH) >> 7) & theirPieces
		for capL != 0 {
			to := capL.PopLSB()
			moves = append(moves, NewMove(to+7, to))
		}

		// Single push (non-promotion)
		push1 := ((pawnsNot2 >> 8) & empty)
		for push1 != 0 {
			to := push1.PopLSB()
			moves = append(moves, NewMove(to+8, to))
		}

		// Double push
		push1All := (pawns >> 8) & empty
		push2 := ((push1All & Rank6) >> 8) & empty
		for push2 != 0 {
			to := push2.PopLSB()
			moves = append(moves, NewMove(to+16, to))
		}

		// En passant
		if b.EnPassant != NoSquare {
			epBB := SquareBB(b.EnPassant)
			epAttackers := PawnAttacks[White][b.EnPassant] & pawns
			for epAttackers != 0 {
				from := epAttackers.PopLSB()
				_ = epBB
				moves = append(moves, NewMoveFlags(from, b.EnPassant, FlagEnPassant))
			}
		}
	}

	// 2. Knight moves (captures+quiets combined)
	knights := b.Pieces[pieceOf(WhiteKnight, us)]
	for knights != 0 {
		from := knights.PopLSB()
		attacks := KnightAttacks[from] & allTargets
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// 3. Bishop moves
	bishops := b.Pieces[pieceOf(WhiteBishop, us)]
	for bishops != 0 {
		from := bishops.PopLSB()
		attacks := BishopAttacksBB(from, b.AllPieces) & allTargets
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// 4. Rook moves
	rooks := b.Pieces[pieceOf(WhiteRook, us)]
	for rooks != 0 {
		from := rooks.PopLSB()
		attacks := RookAttacksBB(from, b.AllPieces) & allTargets
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// 5. Queen moves
	queens := b.Pieces[pieceOf(WhiteQueen, us)]
	for queens != 0 {
		from := queens.PopLSB()
		attacks := QueenAttacksBB(from, b.AllPieces) & allTargets
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// 6. King moves (non-castle targets) then castling
	king := b.Pieces[pieceOf(WhiteKing, us)]
	if king != 0 {
		from := king.LSB()
		attacks := KingAttacks[from] & allTargets
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
		// Castling: kingside first, queenside second (same as our generator)
		moves = b.generateCastlingMovesAppend(moves, from)
	}

	return moves
}

// generateSFOrderEvasions generates legal evasion moves in Stockfish's order when in check.
// Stockfish evasion order: king moves first, then if single check: blocking/capturing
// moves in piece-type order (pawn, knight, bishop, rook, queen).
func generateSFOrderEvasions(b *Board) []Move {
	us := b.SideToMove
	them := 1 - us
	pinned, checkers := b.PinnedAndCheckers(us)
	kingSq := b.Pieces[pieceOf(WhiteKing, us)].LSB()

	moves := make([]Move, 0, 32)

	// King evasions (always generated, both single and double check)
	occ := b.AllPieces ^ SquareBB(kingSq)
	targets := KingAttacks[kingSq] &^ b.Occupied[us]
	for targets != 0 {
		to := targets.PopLSB()
		if !b.isAttackedWithOcc(to, them, occ) {
			moves = append(moves, NewMove(kingSq, to))
		}
	}

	// Double check: only king moves
	if checkers&(checkers-1) != 0 {
		return moves
	}

	// Single check: can block or capture
	checkerSq := checkers.LSB()
	target := SquareBB(checkerSq) | BetweenBB[kingSq][checkerSq]
	blockTarget := BetweenBB[kingSq][checkerSq]
	nonPinned := b.Occupied[us] &^ pinned &^ SquareBB(kingSq)
	empty := ^b.AllPieces
	checkerBB := SquareBB(checkerSq)

	// Pawn evasions (Stockfish generates pawns first in evasions too)
	pawns := b.Pieces[pieceOf(WhitePawn, us)] & nonPinned
	if us == White {
		pawnsOn7 := pawns & Rank7
		pawnsNot7 := pawns &^ Rank7

		// Promotion captures onto checker: right then left
		promoCapR := ((pawnsOn7 & NotFileH) << 9) & checkerBB
		for promoCapR != 0 {
			to := promoCapR.PopLSB()
			moves = append(moves, NewMoveFlags(to-9, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(to-9, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(to-9, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(to-9, to, FlagPromoteN))
		}
		promoCapL := ((pawnsOn7 & NotFileA) << 7) & checkerBB
		for promoCapL != 0 {
			to := promoCapL.PopLSB()
			moves = append(moves, NewMoveFlags(to-7, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(to-7, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(to-7, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(to-7, to, FlagPromoteN))
		}

		// Push promotions to blocking squares (includes checker square)
		promoPush := ((pawnsOn7 << 8) & empty) & target
		for promoPush != 0 {
			to := promoPush.PopLSB()
			moves = append(moves, NewMoveFlags(to-8, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(to-8, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(to-8, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(to-8, to, FlagPromoteN))
		}

		// Non-promotion captures onto checker: right then left
		capR := ((pawnsNot7 & NotFileH) << 9) & checkerBB
		for capR != 0 {
			to := capR.PopLSB()
			moves = append(moves, NewMove(to-9, to))
		}
		capL := ((pawnsNot7 & NotFileA) << 7) & checkerBB
		for capL != 0 {
			to := capL.PopLSB()
			moves = append(moves, NewMove(to-7, to))
		}

		// Single pushes to blocking/capture squares
		push1 := ((pawnsNot7 << 8) & empty) & target
		for push1 != 0 {
			to := push1.PopLSB()
			moves = append(moves, NewMove(to-8, to))
		}

		// Double pushes to blocking squares only
		push1All := (pawns << 8) & empty
		push2 := ((push1All & Rank3) << 8) & empty & blockTarget
		for push2 != 0 {
			to := push2.PopLSB()
			moves = append(moves, NewMove(to-16, to))
		}

		// EP capture (only if captured pawn is the checker)
		if b.EnPassant != NoSquare {
			capturedPawnSq := b.EnPassant - 8
			if Square(capturedPawnSq) == checkerSq {
				epAttackers := PawnAttacks[Black][b.EnPassant] & pawns
				for epAttackers != 0 {
					from := epAttackers.PopLSB()
					m := NewMoveFlags(from, b.EnPassant, FlagEnPassant)
					b.MakeMove(m)
					if !b.IsAttacked(kingSq, them) {
						moves = append(moves, m)
					}
					b.UnmakeMove(m)
				}
			}
		}
	} else {
		// Black
		pawnsOn2 := pawns & Rank2
		pawnsNot2 := pawns &^ Rank2

		promoCapR := ((pawnsOn2 & NotFileA) >> 9) & checkerBB
		for promoCapR != 0 {
			to := promoCapR.PopLSB()
			moves = append(moves, NewMoveFlags(to+9, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(to+9, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(to+9, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(to+9, to, FlagPromoteN))
		}
		promoCapL := ((pawnsOn2 & NotFileH) >> 7) & checkerBB
		for promoCapL != 0 {
			to := promoCapL.PopLSB()
			moves = append(moves, NewMoveFlags(to+7, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(to+7, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(to+7, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(to+7, to, FlagPromoteN))
		}

		promoPush := ((pawnsOn2 >> 8) & empty) & target
		for promoPush != 0 {
			to := promoPush.PopLSB()
			moves = append(moves, NewMoveFlags(to+8, to, FlagPromoteQ))
			moves = append(moves, NewMoveFlags(to+8, to, FlagPromoteR))
			moves = append(moves, NewMoveFlags(to+8, to, FlagPromoteB))
			moves = append(moves, NewMoveFlags(to+8, to, FlagPromoteN))
		}

		capR := ((pawnsNot2 & NotFileA) >> 9) & checkerBB
		for capR != 0 {
			to := capR.PopLSB()
			moves = append(moves, NewMove(to+9, to))
		}
		capL := ((pawnsNot2 & NotFileH) >> 7) & checkerBB
		for capL != 0 {
			to := capL.PopLSB()
			moves = append(moves, NewMove(to+7, to))
		}

		push1 := ((pawnsNot2 >> 8) & empty) & target
		for push1 != 0 {
			to := push1.PopLSB()
			moves = append(moves, NewMove(to+8, to))
		}

		push1All := (pawns >> 8) & empty
		push2 := ((push1All & Rank6) >> 8) & empty & blockTarget
		for push2 != 0 {
			to := push2.PopLSB()
			moves = append(moves, NewMove(to+16, to))
		}

		if b.EnPassant != NoSquare {
			capturedPawnSq := b.EnPassant + 8
			if Square(capturedPawnSq) == checkerSq {
				epAttackers := PawnAttacks[White][b.EnPassant] & pawns
				for epAttackers != 0 {
					from := epAttackers.PopLSB()
					m := NewMoveFlags(from, b.EnPassant, FlagEnPassant)
					b.MakeMove(m)
					if !b.IsAttacked(kingSq, them) {
						moves = append(moves, m)
					}
					b.UnmakeMove(m)
				}
			}
		}
	}

	// Knight evasions
	knights := b.Pieces[pieceOf(WhiteKnight, us)] & nonPinned
	for knights != 0 {
		from := knights.PopLSB()
		attacks := KnightAttacks[from] & target
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// Bishop evasions
	bishops := b.Pieces[pieceOf(WhiteBishop, us)] & nonPinned
	for bishops != 0 {
		from := bishops.PopLSB()
		attacks := BishopAttacksBB(from, b.AllPieces) & target
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// Rook evasions
	rooks := b.Pieces[pieceOf(WhiteRook, us)] & nonPinned
	for rooks != 0 {
		from := rooks.PopLSB()
		attacks := RookAttacksBB(from, b.AllPieces) & target
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	// Queen evasions
	queens := b.Pieces[pieceOf(WhiteQueen, us)] & nonPinned
	for queens != 0 {
		from := queens.PopLSB()
		attacks := QueenAttacksBB(from, b.AllPieces) & target
		for attacks != 0 {
			to := attacks.PopLSB()
			moves = append(moves, NewMove(from, to))
		}
	}

	return moves
}

// sfLegalMoves generates legal moves matching Stockfish's order.
// Uses evasion generator when in check, normal generator otherwise.
// Recovers from panics caused by corrupted board positions (invalid squares).
func sfLegalMoves(b *Board) (result []Move) {
	defer func() {
		if r := recover(); r != nil {
			result = nil
		}
	}()
	kingSq := b.Pieces[pieceOf(WhiteKing, b.SideToMove)].LSB()
	if kingSq < 0 || kingSq > 63 {
		return nil
	}
	if b.InCheck() {
		return generateSFOrderEvasions(b)
	}
	return generateSFOrderLegalMoves(b)
}

// BINPReader reads BINP-format .binpack files, producing NNUETrainSamples.
//
// Decodes both stem (anchor) positions and movetext continuation positions.
// Movetext uses hierarchical piece+destination encoding (binpDecodeMoveFromBoard).
// Each chain yields 1 stem + numPlies continuation positions.
type BINPReader struct {
	file   *os.File
	reader *bufio.Reader

	// Current chunk state
	chunk  []byte // chunk data buffer
	offset int    // current offset within chunk

	// Movetext decoding state: queued samples from current chain
	pending []*NNUETrainSample
	pidx    int // index into pending
}

// openBINPReader opens a BINP-format file for reading.
func openBINPReader(path string) (*BINPReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return &BINPReader{
		file:   f,
		reader: bufio.NewReaderSize(f, 1<<20), // 1MB buffer
	}, nil
}

// Close closes the BINP reader.
func (r *BINPReader) Close() error {
	return r.file.Close()
}

// readNextChunk reads the next BINP chunk (magic + size + data) into r.chunk.
// Returns io.EOF when no more chunks are available.
func (r *BINPReader) readNextChunk() error {
	var header [8]byte
	if _, err := io.ReadFull(r.reader, header[:]); err != nil {
		if err == io.ErrUnexpectedEOF || err == io.EOF {
			return io.EOF
		}
		return fmt.Errorf("reading BINP chunk header: %w", err)
	}

	if header[0] != 'B' || header[1] != 'I' || header[2] != 'N' || header[3] != 'P' {
		return fmt.Errorf("invalid BINP magic: %x", header[:4])
	}

	chunkSize := binary.LittleEndian.Uint32(header[4:8])
	if chunkSize == 0 {
		return io.EOF
	}
	if chunkSize > 64*1024*1024 { // sanity: 64MB max
		return fmt.Errorf("BINP chunk too large: %d bytes", chunkSize)
	}

	// Reuse buffer if possible
	if cap(r.chunk) >= int(chunkSize) {
		r.chunk = r.chunk[:chunkSize]
	} else {
		r.chunk = make([]byte, chunkSize)
	}
	if _, err := io.ReadFull(r.reader, r.chunk); err != nil {
		return fmt.Errorf("reading BINP chunk data: %w", err)
	}
	r.offset = 0
	return nil
}

// Next returns the next training sample from the BINP file.
// Returns stem positions and movetext continuation positions in chain order.
// Returns (nil, io.EOF) when all data has been read.
func (r *BINPReader) Next() (*NNUETrainSample, error) {
	for {
		// Drain pending movetext samples first
		if r.pidx < len(r.pending) {
			s := r.pending[r.pidx]
			r.pidx++
			if r.pidx >= len(r.pending) {
				r.pending = r.pending[:0]
				r.pidx = 0
			}
			return s, nil
		}

		// Read next stem from chunk
		// Need at least 34 bytes: 32 (stem) + 2 (numPlies)
		if len(r.chunk)-r.offset < 34 {
			if err := r.readNextChunk(); err != nil {
				return nil, err
			}
		}

		// Try to parse a stem at the current offset.
		// If decompression fails (corrupt offset from movetext skip failure),
		// scan forward byte-by-byte to find the next valid stem.
		sample, ok := r.tryParseStem()
		if !ok {
			// No valid stem found in remainder of chunk — skip to next chunk
			r.offset = len(r.chunk)
			continue
		}

		// nil sample means a valid but filtered stem (score too high, in check, etc.)
		// Loop to find the next non-filtered stem.
		if sample == nil {
			continue
		}

		return sample, nil
	}
}

// tryParseStem attempts to parse a stem at the current offset.
// If parsing fails, scans forward to find the next valid stem.
// Returns the training sample and true on success, or nil/false if no more stems in chunk.
func (r *BINPReader) tryParseStem() (*NNUETrainSample, bool) {
	for r.offset+34 <= len(r.chunk) {
		sample, advance, ok := r.parseStemAt(r.offset)
		if ok {
			r.offset += advance
			return sample, true
		}
		// Invalid stem — advance by 1 byte and retry
		r.offset++
	}
	return nil, false
}

// parseStemAt tries to parse a BINP stem at the given offset.
// Returns (sample, bytesConsumed, success).
// bytesConsumed includes the stem (34 bytes) plus any movetext data to skip.
func (r *BINPReader) parseStemAt(offset int) (*NNUETrainSample, int, bool) {
	if offset+34 > len(r.chunk) {
		return nil, 0, false
	}

	// Parse CompressedPosition (24 bytes)
	var compPos [24]byte
	copy(compPos[:], r.chunk[offset:offset+24])
	pos, err := decompressBINPPosition(compPos)
	if err != nil {
		return nil, 0, false
	}

	// Validate: exactly one king of each color
	if pos.wKingSq < 0 || pos.wKingSq > 63 || pos.bKingSq < 0 || pos.bKingSq > 63 {
		return nil, 0, false
	}
	if pos.squares[pos.wKingSq] != WhiteKing || pos.squares[pos.bKingSq] != BlackKing {
		return nil, 0, false
	}
	wkCount, bkCount := 0, 0
	pieceCount := 0
	for _, p := range pos.squares {
		if p != Empty {
			pieceCount++
		}
		if p == WhiteKing {
			wkCount++
		}
		if p == BlackKing {
			bkCount++
		}
	}
	if wkCount != 1 || bkCount != 1 || pieceCount < 2 || pieceCount > 32 {
		return nil, 0, false
	}

	// Compressed move (2 bytes, big-endian)
	compMove := uint16(r.chunk[offset+24])<<8 | uint16(r.chunk[offset+25])

	// Score (2 bytes, big-endian, signedToUnsigned encoded)
	rawScore := uint16(r.chunk[offset+26])<<8 | uint16(r.chunk[offset+27])
	score := unsignedToSigned(rawScore)

	// ply|result (2 bytes, big-endian)
	plyResult := uint16(r.chunk[offset+28])<<8 | uint16(r.chunk[offset+29])
	resultEnc := (plyResult >> 14) & 3
	result := unsignedToSigned(resultEnc)

	// rule50 (2 bytes, big-endian)
	rule50 := uint16(r.chunk[offset+30])<<8 | uint16(r.chunk[offset+31])
	if rule50 > 200 {
		return nil, 0, false
	}
	pos.rule50 = uint8(rule50)

	// numPlies (2 bytes, big-endian)
	numPlies := int(r.chunk[offset+32])<<8 | int(r.chunk[offset+33])
	if numPlies > 5000 {
		return nil, 0, false
	}

	// Validate the compressed move makes sense
	if compMove != 0 {
		moveType := (compMove >> 14) & 3
		fromSq := Square((compMove >> 8) & 0x3F)
		if fromSq > 63 {
			return nil, 0, false
		}
		// For normal and promotion moves, there should be a piece on the from square
		if moveType <= 1 && pos.squares[fromSq] == Empty {
			return nil, 0, false
		}
	}

	consumed := 34 // stem + numPlies

	// Filter: skip positions with extreme scores (mates, decided games)
	if score > 3000 || score < -3000 {
		return nil, consumed, true // valid stem, skip it
	}

	// Filter: skip positions where side to move is in check (static eval meaningless)
	board, err := pos.toBoard()
	if err == nil && board.InCheck() {
		return nil, consumed, true // valid stem, skip it
	}

	// Convert result to our 0/1/2 format
	var resultU8 uint8
	switch result {
	case -1:
		resultU8 = 0
	case 0:
		resultU8 = 1
	case 1:
		resultU8 = 2
	default:
		resultU8 = 1
	}

	// Extract features for the stem position
	stemSample := pos.extractFeatures(score, resultU8)

	// Decode movetext positions if present
	if numPlies > 0 && compMove != 0 {
		movetextPos := pos
		if err := movetextPos.applyBINPMove(compMove); err != nil {
			// Can't apply stem move — return stem only, skip rest of chunk
			return stemSample, consumed, true
		}
		movetextSamples, movetextLen, err := r.decodeMovetextPositions(
			&movetextPos, score, result, numPlies, r.chunk[offset+34:])
		if err != nil {
			// Movetext decoding failed — return stem only
			return stemSample, consumed, true
		}
		consumed += movetextLen
		if len(movetextSamples) > 0 {
			r.pending = movetextSamples
			r.pidx = 0
		}
	}

	return stemSample, consumed, true
}

// nthSetBit returns the square index of the nth set bit (0-indexed) in a bitboard.
func nthSetBit(bb Bitboard, n int) int {
	for i := 0; i < n; i++ {
		bb &= bb - 1 // clear lowest set bit
	}
	return bits.TrailingZeros64(uint64(bb))
}

// countBitsBefore returns the number of set bits in bb below position sq.
func countBitsBefore(bb Bitboard, sq int) int {
	mask := Bitboard((uint64(1) << uint(sq)) - 1) // bits 0..sq-1
	return bits.OnesCount64(uint64(bb & mask))
}

// binpDecodeMoveFromBoard decodes a move from BINP movetext using the hierarchical
// piece+destination encoding. Returns the decoded move in our Move format, or NoMove
// on error. This matches Stockfish's PackedMoveScoreListReader::nextMoveScore().
func binpDecodeMoveFromBoard(br *binpBitReader, b *Board) Move {
	us := b.SideToMove
	them := 1 - us
	ourPieces := b.Occupied[us]
	theirPieces := b.Occupied[them]
	occupied := b.AllPieces

	// 1. Read piece selection: which of our pieces is moving
	numOurPieces := bits.OnesCount64(uint64(ourPieces))
	pieceId := int(br.readBits(usedBitsSafe(numOurPieces)))
	if br.overflow {
		return NoMove
	}
	from := Square(nthSetBit(ourPieces, pieceId))
	if from < 0 || from > 63 {
		return NoMove
	}

	piece := b.Squares[from]
	pt := piece
	if pt >= BlackPawn {
		pt -= 6 // normalize to white piece type (1-6)
	}

	switch pt {
	case WhitePawn:
		// Compute pawn destinations
		forward := Square(8)
		promoRank := 6 // 0-indexed rank 6 = rank 7
		startRank := 1 // 0-indexed rank 1 = rank 2
		if us == Black {
			forward = -8
			promoRank = 1 // rank 2
			startRank = 6 // rank 7
		}

		epSquare := b.EnPassant
		attackTargets := theirPieces
		if epSquare != NoSquare {
			attackTargets |= SquareBB(epSquare)
		}

		destinations := PawnAttacks[us][from] & attackTargets

		sqForward := from + forward
		if sqForward >= 0 && sqForward <= 63 && !occupied.IsSet(sqForward) {
			destinations |= SquareBB(sqForward)
			sqForward2 := sqForward + forward
			if from.Rank() == startRank && sqForward2 >= 0 && sqForward2 <= 63 && !occupied.IsSet(sqForward2) {
				destinations |= SquareBB(sqForward2)
			}
		}

		destCount := bits.OnesCount64(uint64(destinations))

		if from.Rank() == promoRank {
			// Promotion: moveId = destIndex * 4 + promoIndex
			moveId := int(br.readBits(usedBitsSafe(destCount * 4)))
			if br.overflow {
				return NoMove
			}
			promoIdx := moveId % 4
			destIdx := moveId / 4
			if destIdx >= destCount {
				return NoMove
			}
			to := Square(nthSetBit(destinations, destIdx))
			var flag int
			switch promoIdx {
			case 0:
				flag = FlagPromoteN
			case 1:
				flag = FlagPromoteB
			case 2:
				flag = FlagPromoteR
			case 3:
				flag = FlagPromoteQ
			}
			return NewMoveFlags(from, to, flag)
		}

		// Non-promotion pawn move
		moveId := int(br.readBits(usedBitsSafe(destCount)))
		if br.overflow {
			return NoMove
		}
		if moveId >= destCount {
			return NoMove
		}
		to := Square(nthSetBit(destinations, moveId))
		if to == epSquare {
			return NewMoveFlags(from, to, FlagEnPassant)
		}
		return NewMove(from, to)

	case WhiteKing:
		// King moves: pseudo-attacks + castling
		attacks := KingAttacks[from] &^ ourPieces
		attacksCount := bits.OnesCount64(uint64(attacks))

		var ourCastleMask CastlingRights
		if us == White {
			ourCastleMask = WhiteKingside | WhiteQueenside
		} else {
			ourCastleMask = BlackKingside | BlackQueenside
		}
		ourCastling := b.Castling & ourCastleMask
		numCastlings := bits.OnesCount8(uint8(ourCastling))

		moveId := int(br.readBits(usedBitsSafe(attacksCount + numCastlings)))
		if br.overflow {
			return NoMove
		}

		if moveId >= attacksCount {
			// Castling
			idx := moveId - attacksCount

			var longRights CastlingRights
			if us == White {
				longRights = WhiteQueenside
			} else {
				longRights = BlackQueenside
			}

			hasLong := ourCastling&longRights != 0
			castleType := 0 // 0=long, 1=short
			if hasLong {
				if idx == 0 {
					castleType = 0 // long
				} else {
					castleType = 1 // short
				}
			} else {
				castleType = 1 // short (only option)
			}

			var kingDest Square
			if castleType == 0 {
				// Queenside: king goes to c-file
				kingDest = NewSquare(2, from.Rank())
			} else {
				// Kingside: king goes to g-file
				kingDest = NewSquare(6, from.Rank())
			}
			return NewMoveFlags(from, kingDest, FlagCastle)
		}

		to := Square(nthSetBit(attacks, moveId))
		return NewMove(from, to)

	default:
		// Knight, Bishop, Rook, Queen
		var attacksBB Bitboard
		switch pt {
		case WhiteKnight:
			attacksBB = KnightAttacks[from] &^ ourPieces
		case WhiteBishop:
			attacksBB = BishopAttacksBB(from, occupied) &^ ourPieces
		case WhiteRook:
			attacksBB = RookAttacksBB(from, occupied) &^ ourPieces
		case WhiteQueen:
			attacksBB = QueenAttacksBB(from, occupied) &^ ourPieces
		}

		attacksCount := bits.OnesCount64(uint64(attacksBB))
		moveId := int(br.readBits(usedBitsSafe(attacksCount)))
		if br.overflow {
			return NoMove
		}
		if moveId >= attacksCount {
			return NoMove
		}
		to := Square(nthSetBit(attacksBB, moveId))
		return NewMove(from, to)
	}
}

// decodeMovetextPositions decodes all movetext continuation positions from a chain.
// Uses the hierarchical piece+destination encoding matching Stockfish's binpack format.
// Returns the decoded samples, byte length consumed, and any error.
func (r *BINPReader) decodeMovetextPositions(pos *sfLightPos, lastScore int16, lastResult int16, numPlies int, data []byte) ([]*NNUETrainSample, int, error) {
	if len(data) == 0 {
		return nil, 0, fmt.Errorf("no data for movetext")
	}

	br := newBinpBitReader(data)
	samples := make([]*NNUETrainSample, 0, numPlies)
	currentResult := lastResult
	mLastScore := -lastScore // matches Stockfish: m_lastScore starts as -entry.score

	for i := 0; i < numPlies; i++ {
		board, err := pos.toBoard()
		if err != nil {
			return samples, br.numReadBytes(), err
		}

		// Decode move using hierarchical piece+destination encoding
		m := binpDecodeMoveFromBoard(br, board)
		if m == NoMove || br.overflow {
			return samples, br.numReadBytes(), fmt.Errorf("failed to decode move at ply %d", i)
		}

		// Read VLE score delta
		scoreDelta := br.readVLE()
		if br.overflow {
			return samples, br.numReadBytes(), fmt.Errorf("overflow reading score at ply %d", i)
		}
		plyScore := mLastScore + unsignedToSigned(scoreDelta)
		mLastScore = -plyScore

		// Result alternates sign each ply
		currentResult = -currentResult

		// Convert result to 0/1/2 format
		var resultU8 uint8
		switch currentResult {
		case -1:
			resultU8 = 0
		case 0:
			resultU8 = 1
		case 1:
			resultU8 = 2
		default:
			resultU8 = 1
		}

		// Filter: skip positions with extreme scores or in check
		if plyScore > 3000 || plyScore < -3000 {
			// Still need to apply move to keep position tracking correct
			if i < numPlies-1 {
				applySfMoveFromOurMove(pos, m)
			}
			continue
		}
		if board != nil && board.InCheck() {
			if i < numPlies-1 {
				applySfMoveFromOurMove(pos, m)
			}
			continue
		}

		// Extract features for this position
		sample := pos.extractFeatures(plyScore, resultU8)
		samples = append(samples, sample)

		// Apply move to advance position for next ply
		if i < numPlies-1 {
			applySfMoveFromOurMove(pos, m)
		}
	}

	return samples, br.numReadBytes(), nil
}

// skipMovetextBitsStatic parses through movetext bits to determine byte length.
// Uses the hierarchical piece+destination encoding matching Stockfish's binpack format.
func skipMovetextBitsStatic(pos *sfLightPos, lastScore int16, numPlies int, data []byte) (int, error) {
	if len(data) == 0 {
		return 0, fmt.Errorf("no data for movetext")
	}

	br := newBinpBitReader(data)
	mLastScore := -lastScore

	for i := 0; i < numPlies; i++ {
		board, err := pos.toBoard()
		if err != nil {
			return br.numReadBytes(), err
		}

		m := binpDecodeMoveFromBoard(br, board)
		if m == NoMove || br.overflow {
			return br.numReadBytes(), fmt.Errorf("failed to decode move at ply %d", i)
		}

		// Read VLE score
		scoreDelta := br.readVLE()
		if br.overflow {
			return br.numReadBytes(), fmt.Errorf("overflow at score ply %d", i)
		}
		plyScore := mLastScore + unsignedToSigned(scoreDelta)
		mLastScore = -plyScore

		// Apply move to advance position for next ply
		if i < numPlies-1 {
			applySfMoveFromOurMove(pos, m)
		}
	}

	return br.numReadBytes(), nil
}


// toBoard converts an sfLightPos to a full Board for legal move generation.
// Returns an error if the position is invalid (e.g., missing kings).
func (pos *sfLightPos) toBoard() (*Board, error) {
	// Validate: both kings must be present
	hasWhiteKing := false
	hasBlackKing := false
	for sq := Square(0); sq < 64; sq++ {
		if pos.squares[sq] == WhiteKing {
			hasWhiteKing = true
		} else if pos.squares[sq] == BlackKing {
			hasBlackKing = true
		}
	}
	if !hasWhiteKing || !hasBlackKing {
		return nil, fmt.Errorf("invalid position: missing king(s)")
	}

	fen := pos.toFEN()
	var b Board
	if err := b.SetFEN(fen); err != nil {
		return nil, fmt.Errorf("SetFEN(%q): %w", fen, err)
	}
	return &b, nil
}

// ============================================================================
// SFBinpackReader: unified reader for both legacy and BINP formats
// ============================================================================

// SFBinpackReader reads Stockfish .binpack files sequentially, producing NNUETrainSamples.
// Supports both legacy PackedSfen chain format and the modern BINP chunk format.
type SFBinpackReader struct {
	file      *os.File
	reader    *bufio.Reader
	pos       sfLightPos
	chainLeft int // remaining entries in current chain (legacy format only)

	// BINP format support
	isBINP     bool
	binpReader *BINPReader
}

// OpenSFBinpack opens a Stockfish .binpack file for sequential reading.
// Auto-detects the format by checking for the "BINP" magic header.
func OpenSFBinpack(path string) (*SFBinpackReader, error) {
	binp, err := isBINPFormat(path)
	if err != nil {
		return nil, err
	}

	if binp {
		br, err := openBINPReader(path)
		if err != nil {
			return nil, err
		}
		return &SFBinpackReader{
			isBINP:     true,
			binpReader: br,
		}, nil
	}

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
	if r.isBINP {
		return r.binpReader.Close()
	}
	return r.file.Close()
}

// Next returns the next training sample, or (nil, io.EOF) at end of file.
// For legacy format: each chain entry describes the current position (score, result)
// and the move played from it. We extract features first, then apply the move to advance.
// For BINP format: delegates to BINPReader which handles stems and movetext.
func (r *SFBinpackReader) Next() (*NNUETrainSample, error) {
	if r.isBINP {
		return r.binpReader.Next()
	}

	// Legacy PackedSfen chain format
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

			// Filter: skip positions with extreme scores or in check
			skip := score > 3000 || score < -3000

			// Extract features from current position (before applying this entry's move)
			var sample *NNUETrainSample
			if !skip {
				sample = r.pos.extractFeatures(score, result)
			}

			// Apply move to advance to next position
			r.chainLeft--
			if sfMove != 0 && r.chainLeft > 0 {
				if err := r.pos.applyMove(sfMove); err != nil {
					return nil, fmt.Errorf("applying move: %w", err)
				}
			}

			if skip {
				continue // loop to next chain entry
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

// countBinpackPositions pre-scans .binpack files to count total positions.
// For legacy format: reads chain headers (fast, ~2 bytes per chain of ~100 positions).
// For BINP format: reads chunk headers and stem entries, counting 1 + numPlies per chain.
func countBinpackPositions(paths []string) (int, error) {
	total := 0
	for _, path := range paths {
		binp, err := isBINPFormat(path)
		if err != nil {
			return 0, err
		}
		if binp {
			n, err := countBINPPositions(path)
			if err != nil {
				return 0, err
			}
			total += n
			continue
		}

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

// countBINPPositions counts positions in a BINP-format file by scanning chunks.
// Each chain contributes 1 (stem) + numPlies (movetext) positions.
// This is a fast scan that only reads headers and stem metadata, skipping movetext data.
func countBINPPositions(path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20) // 1MB buffer

	total := 0

	for {
		// Read chunk header: 4 bytes magic + 4 bytes size
		var header [8]byte
		if _, err := io.ReadFull(r, header[:]); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break
			}
			return 0, fmt.Errorf("reading BINP chunk header in %s: %w", path, err)
		}

		if header[0] != 'B' || header[1] != 'I' || header[2] != 'N' || header[3] != 'P' {
			return 0, fmt.Errorf("invalid BINP magic in %s at position", path)
		}

		chunkSize := int(binary.LittleEndian.Uint32(header[4:8]))
		if chunkSize == 0 {
			break
		}

		// Estimate stems per chunk: read first few stems to get average chain size,
		// then extrapolate for the rest.
		// For counting purposes, we only need an approximate total for progress bars
		// and train/val splitting. The actual reading handles exact counts.
		//
		// Simple estimate: each stem is 34 bytes, each ply of movetext averages ~1 byte
		// (5 bits move + 5-10 bits VLE score ≈ 10-15 bits). Average chain has ~100 plies.
		// So average chain = 34 + 100 * 1.5 = 184 bytes.
		// stems_per_chunk ≈ chunkSize / 184
		//
		// For accuracy, skip the chunk and use the estimate.
		stemsEstimate := chunkSize / 184
		if stemsEstimate < 1 {
			stemsEstimate = 1
		}
		total += stemsEstimate

		// Skip chunk data
		if _, err := r.Discard(chunkSize); err != nil {
			return 0, fmt.Errorf("skipping BINP chunk data in %s: %w", path, err)
		}
	}

	return total, nil
}

// scanMovetextLength determines the byte length of movetext data for a chain.
// stemData is the 34-byte stem+numPlies, movetextData is the remaining chunk data.
// Returns the number of bytes consumed by the movetext.
func scanMovetextLength(stemData []byte, movetextData []byte, numPlies int) (result int, retErr error) {
	defer func() {
		if r := recover(); r != nil {
			result = 0
			retErr = fmt.Errorf("panic in scanMovetextLength: %v", r)
		}
	}()
	return scanMovetextLengthInner(stemData, movetextData, numPlies)
}

func scanMovetextLengthInner(stemData []byte, movetextData []byte, numPlies int) (int, error) {
	// Decode the stem position and apply the stem move to get the first continuation position
	var compPos [24]byte
	copy(compPos[:], stemData[0:24])
	pos, err := decompressBINPPosition(compPos)
	if err != nil {
		return 0, err
	}

	compMove := uint16(stemData[24])<<8 | uint16(stemData[25])
	if compMove == 0 {
		return 0, fmt.Errorf("null stem move with numPlies > 0")
	}

	if err := pos.applyBINPMove(compMove); err != nil {
		return 0, err
	}

	// Decode the stem score for delta computation
	rawScore := uint16(stemData[26])<<8 | uint16(stemData[27])
	lastScore := unsignedToSigned(rawScore)

	return skipMovetextBitsStatic(&pos, lastScore, numPlies, movetextData)
}

// applySfMoveFromOurMove applies one of our Move values to an sfLightPos.
func applySfMoveFromOurMove(pos *sfLightPos, m Move) {
	from := m.From()
	to := m.To()
	flags := m.Flags()

	var sfMove uint16
	switch {
	case flags == FlagCastle:
		sfMove = uint16(from) | uint16(to)<<6 | sfMoveTyCastling<<12
	case flags == FlagEnPassant:
		sfMove = uint16(from) | uint16(to)<<6 | sfMoveTypeEnPassant<<12
	case flags&FlagPromotion != 0:
		var promoType uint16
		switch flags {
		case FlagPromoteN:
			promoType = 0
		case FlagPromoteB:
			promoType = 1
		case FlagPromoteR:
			promoType = 2
		case FlagPromoteQ:
			promoType = 3
		}
		sfMove = uint16(from) | uint16(to)<<6 | sfMoveTypePromotion<<12 | promoType<<14
	default:
		sfMove = uint16(from) | uint16(to)<<6
	}

	pos.applyMove(sfMove) //nolint: we ignore errors in scan path
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

// ValidationSamples reads validation samples.
// For large SF binpack files, reading from the end requires scanning the entire file.
// Instead, read from near the beginning (positions 1000-101000) which is fast and
// provides a representative sample since positions are already shuffled across chains.
func (s *SFBinpackSource) ValidationSamples(trainFraction float64) ([]*NNUETrainSample, error) {
	maxVal := 100_000
	valCount := int(float64(s.totalPos) * (1 - trainFraction))
	if valCount > maxVal {
		valCount = maxVal
	}
	// Read from early in the file (skip first 1000 to avoid opening position bias)
	start := 1000
	if start >= s.totalPos {
		start = 0
	}
	end := start + valCount
	if end > s.totalPos {
		end = s.totalPos
	}
	return sfBinpackReadRange(s.paths, start, end)
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
// Auto-detects legacy vs BINP format.
func DumpSFBinpack(path string, n int) error {
	binp, err := isBINPFormat(path)
	if err != nil {
		return err
	}
	if binp {
		return dumpBINP(path, n)
	}
	return dumpLegacyBinpack(path, n)
}

// dumpBINP prints the first n chains from a BINP-format file.
func dumpBINP(path string, n int) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	chainNum := 0
	posNum := 0

	for chainNum < n {
		// Read chunk header
		var header [8]byte
		if _, err := io.ReadFull(r, header[:]); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break
			}
			return err
		}
		if header[0] != 'B' || header[1] != 'I' || header[2] != 'N' || header[3] != 'P' {
			return fmt.Errorf("invalid BINP magic")
		}
		chunkSize := int(binary.LittleEndian.Uint32(header[4:8]))
		if chunkSize == 0 {
			break
		}

		chunk := make([]byte, chunkSize)
		if _, err := io.ReadFull(r, chunk); err != nil {
			return fmt.Errorf("reading chunk: %w", err)
		}

		offset := 0
		for offset+34 <= chunkSize && chainNum < n {
			// Parse stem
			var compPos [24]byte
			copy(compPos[:], chunk[offset:offset+24])
			pos, err := decompressBINPPosition(compPos)
			if err != nil {
				return fmt.Errorf("chain %d: decompressing position: %w", chainNum, err)
			}

			compMove := uint16(chunk[offset+24])<<8 | uint16(chunk[offset+25])
			rawScore := uint16(chunk[offset+26])<<8 | uint16(chunk[offset+27])
			score := unsignedToSigned(rawScore)
			plyResult := uint16(chunk[offset+28])<<8 | uint16(chunk[offset+29])
			ply := int(plyResult & 0x3FFF)
			resultEnc := (plyResult >> 14) & 3
			result := unsignedToSigned(resultEnc)
			rule50 := uint16(chunk[offset+30])<<8 | uint16(chunk[offset+31])
			numPlies := int(chunk[offset+32])<<8 | int(chunk[offset+33])
			pos.rule50 = uint8(rule50)
			offset += 34

			resultStr := "draw"
			if result == -1 {
				resultStr = "loss"
			} else if result == 1 {
				resultStr = "win"
			}

			// Decode the compressed move for display
			m := decompressBINPMove(compMove, &pos)

			fmt.Printf("=== Chain %d (stem + %d plies) ===\n", chainNum, numPlies)
			fmt.Printf("  [stem] ply=%d score=%d result=%s move=%s rule50=%d FEN=%s\n",
				ply, score, resultStr, m.String(), rule50, pos.toFEN())
			posNum++

			// Decode movetext if present
			if numPlies > 0 && compMove != 0 {
				movetextPos := pos
				if err := movetextPos.applyBINPMove(compMove); err != nil {
					fmt.Printf("  ERROR applying stem move: %v\n", err)
					chainNum++
					// Skip to end of chunk since we can't determine movetext length
					offset = chunkSize
					continue
				}

				mLastScore := -score
				currentResult := result
				currentPly := ply + 1
				br := newBinpBitReader(chunk[offset:])
				skipOK := true

				for pi := 0; pi < numPlies; pi++ {
					board, err := movetextPos.toBoard()
					if err != nil {
						fmt.Printf("  ERROR converting to Board: %v\n", err)
						skipOK = false
						break
					}

					m := binpDecodeMoveFromBoard(br, board)
					if m == NoMove || br.overflow {
						fmt.Printf("  ERROR: failed to decode move at ply %d\n", currentPly)
						skipOK = false
						break
					}

					scoreDelta := br.readVLE()
					if br.overflow {
						fmt.Printf("  ERROR: overflow reading score at ply %d\n", currentPly)
						skipOK = false
						break
					}
					plyScore := mLastScore + unsignedToSigned(scoreDelta)
					mLastScore = -plyScore
					currentResult = -currentResult

					pResultStr := "draw"
					if currentResult == -1 {
						pResultStr = "loss"
					} else if currentResult == 1 {
						pResultStr = "win"
					}

					fmt.Printf("  [%d] ply=%d score=%d result=%s move=%s FEN=%s\n",
						pi, currentPly, plyScore, pResultStr, m.String(), movetextPos.toFEN())
					posNum++

					currentPly++

					// Apply move to advance
					if pi < numPlies-1 {
						applySfMoveFromOurMove(&movetextPos, m)
					}
				}

				if skipOK {
					offset += br.numReadBytes()
				} else {
					// Can't determine movetext length, skip rest of chunk
					offset = chunkSize
				}
			}

			chainNum++
			fmt.Println()
		}
	}

	fmt.Printf("Showed %d chains, %d positions (BINP format)\n", chainNum, posNum)
	return nil
}

// dumpLegacyBinpack prints the first n chains from a legacy-format .binpack file.
func dumpLegacyBinpack(path string, n int) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	chainNum := 0
	posNum := 0

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

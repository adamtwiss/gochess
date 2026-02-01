package chess

import "math/rand"

// ZobristKeys holds all random numbers for Zobrist hashing
type ZobristKeys struct {
	Pieces     [13][64]uint64 // [piece][square] - index 0 (Empty) unused
	SideToMove uint64
	Castling   [4]uint64  // K, Q, k, q
	EnPassant  [8]uint64  // one per file
}

var Zobrist ZobristKeys

func init() {
	InitZobrist(0x1234567890ABCDEF)
}

// InitZobrist initializes the Zobrist keys with a given seed
// Using a fixed seed ensures reproducible hashes across runs
func InitZobrist(seed int64) {
	r := rand.New(rand.NewSource(seed))

	// Generate keys for each piece on each square
	for piece := WhitePawn; piece <= BlackKing; piece++ {
		for sq := 0; sq < 64; sq++ {
			Zobrist.Pieces[piece][sq] = r.Uint64()
		}
	}

	// Side to move
	Zobrist.SideToMove = r.Uint64()

	// Castling rights
	for i := 0; i < 4; i++ {
		Zobrist.Castling[i] = r.Uint64()
	}

	// En passant files
	for i := 0; i < 8; i++ {
		Zobrist.EnPassant[i] = r.Uint64()
	}
}

// Hash computes the Zobrist hash for the current board position
func (b *Board) Hash() uint64 {
	var hash uint64

	// Hash all pieces
	for sq := 0; sq < 64; sq++ {
		piece := b.Squares[sq]
		if piece != Empty {
			hash ^= Zobrist.Pieces[piece][sq]
		}
	}

	// Hash side to move
	if b.SideToMove == Black {
		hash ^= Zobrist.SideToMove
	}

	// Hash castling rights
	if b.Castling&WhiteKingside != 0 {
		hash ^= Zobrist.Castling[0]
	}
	if b.Castling&WhiteQueenside != 0 {
		hash ^= Zobrist.Castling[1]
	}
	if b.Castling&BlackKingside != 0 {
		hash ^= Zobrist.Castling[2]
	}
	if b.Castling&BlackQueenside != 0 {
		hash ^= Zobrist.Castling[3]
	}

	// Hash en passant file (only if there's a valid en passant square)
	if b.EnPassant != NoSquare {
		hash ^= Zobrist.EnPassant[b.EnPassant.File()]
	}

	return hash
}

// PawnHash computes the Zobrist hash for pawn positions only.
// Used for pawn structure caching since pawns change infrequently.
func (b *Board) PawnHash() uint64 {
	var hash uint64
	for _, piece := range [2]Piece{WhitePawn, BlackPawn} {
		bb := b.Pieces[piece]
		for bb != 0 {
			sq := bb.PopLSB()
			hash ^= Zobrist.Pieces[piece][sq]
		}
	}
	return hash
}

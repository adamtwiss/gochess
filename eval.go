package chess

// Piece values in centipawns
const (
	PawnValue   = 100
	KnightValue = 320
	BishopValue = 330
	RookValue   = 500
	QueenValue  = 900
	KingValue   = 20000 // High value to detect checkmate situations
)

// Mobility bonus per square attacked (in centipawns)
const (
	KnightMobility = 4
	BishopMobility = 5
	RookMobility   = 2
	QueenMobility  = 1
)

// Evaluate returns a static evaluation of the position in centipawns
// Positive values favor White, negative values favor Black
func (b *Board) Evaluate() int {
	score := 0

	// Material counting (simple and fast)
	score += b.evaluateMaterial(White)
	score -= b.evaluateMaterial(Black)

	// Mobility - count squares each piece can attack
	score += b.evaluateMobility(White)
	score -= b.evaluateMobility(Black)

	return score
}

// evaluateMaterial returns simple material count for a color
func (b *Board) evaluateMaterial(color Color) int {
	score := 0

	score += b.Pieces[pieceOf(WhitePawn, color)].Count() * PawnValue
	score += b.Pieces[pieceOf(WhiteKnight, color)].Count() * KnightValue
	score += b.Pieces[pieceOf(WhiteBishop, color)].Count() * BishopValue
	score += b.Pieces[pieceOf(WhiteRook, color)].Count() * RookValue
	score += b.Pieces[pieceOf(WhiteQueen, color)].Count() * QueenValue

	return score
}

// EvaluateRelative returns the evaluation from the perspective of the side to move
// Positive values are good for the side to move
func (b *Board) EvaluateRelative() int {
	score := b.Evaluate()
	if b.SideToMove == Black {
		return -score
	}
	return score
}

// evaluateMobility calculates mobility bonus for a given color
func (b *Board) evaluateMobility(color Color) int {
	score := 0

	// Knights
	knights := b.Pieces[pieceOf(WhiteKnight, color)]
	for knights != 0 {
		sq := knights.PopLSB()
		attacks := KnightAttacks[sq]
		score += attacks.Count() * KnightMobility
	}

	// Bishops
	bishops := b.Pieces[pieceOf(WhiteBishop, color)]
	for bishops != 0 {
		sq := bishops.PopLSB()
		attacks := BishopAttacksBB(sq, b.AllPieces)
		score += attacks.Count() * BishopMobility
	}

	// Rooks
	rooks := b.Pieces[pieceOf(WhiteRook, color)]
	for rooks != 0 {
		sq := rooks.PopLSB()
		attacks := RookAttacksBB(sq, b.AllPieces)
		score += attacks.Count() * RookMobility
	}

	// Queens
	queens := b.Pieces[pieceOf(WhiteQueen, color)]
	for queens != 0 {
		sq := queens.PopLSB()
		attacks := QueenAttacksBB(sq, b.AllPieces)
		score += attacks.Count() * QueenMobility
	}

	return score
}

package chess

// Mobility bonus per square attacked (in centipawns)
const (
	KnightMobility = 2
	BishopMobility = 3
	RookMobility   = 1
	QueenMobility  = 1
)

// Evaluate returns a static evaluation of the position in centipawns.
// Positive values favor White, negative values favor Black.
// Uses tapered evaluation blending middlegame and endgame PST scores.
func (b *Board) Evaluate() int {
	wMG, wEG := b.evaluatePST(White)
	bMG, bEG := b.evaluatePST(Black)

	wMob := b.evaluateMobility(White)
	bMob := b.evaluateMobility(Black)

	// Pawn structure (cached via pawn hash table)
	if b.PawnTable == nil {
		b.PawnTable = NewPawnTable(1) // 1 MB default
	}
	pawnEntry := b.probePawnEval()

	// King safety (per-node, not cached)
	wKSmg, wKSeg := b.evaluateKingSafety(White)
	bKSmg, bKSeg := b.evaluateKingSafety(Black)

	mg := wMG - bMG + wMob - bMob +
		int(pawnEntry.WhiteMG) - int(pawnEntry.BlackMG) +
		wKSmg - bKSmg
	eg := wEG - bEG + wMob - bMob +
		int(pawnEntry.WhiteEG) - int(pawnEntry.BlackEG) +
		wKSeg - bKSeg

	phase := b.computePhase()
	score := (mg*(TotalPhase-phase) + eg*phase) / TotalPhase

	return score
}

// EvaluateRelative returns the evaluation from the perspective of the side to move.
// Positive values are good for the side to move.
func (b *Board) EvaluateRelative() int {
	score := b.Evaluate()
	if b.SideToMove == Black {
		return -score
	}
	return score
}

// computePhase returns the game phase from 0 (opening/middlegame) to TotalPhase (endgame).
// Phase increases as pieces are traded off.
func (b *Board) computePhase() int {
	phase := TotalPhase

	phase -= (b.Pieces[WhiteKnight].Count() + b.Pieces[BlackKnight].Count()) * KnightPhase
	phase -= (b.Pieces[WhiteBishop].Count() + b.Pieces[BlackBishop].Count()) * BishopPhase
	phase -= (b.Pieces[WhiteRook].Count() + b.Pieces[BlackRook].Count()) * RookPhase
	phase -= (b.Pieces[WhiteQueen].Count() + b.Pieces[BlackQueen].Count()) * QueenPhase

	if phase < 0 {
		phase = 0
	}
	return phase
}

// evaluatePST returns the middlegame and endgame PST scores for a color.
// Includes both material values and positional bonuses.
func (b *Board) evaluatePST(color Color) (mg, eg int) {
	for pt := WhitePawn; pt <= WhiteKing; pt++ {
		piece := pieceOf(pt, color)
		bb := b.Pieces[piece]
		mgTable := mgPST[pt]
		egTable := egPST[pt]
		mgMat := mgMaterial[pt]
		egMat := egMaterial[pt]

		for bb != 0 {
			sq := bb.PopLSB()
			idx := int(sq)
			if color == Black {
				idx ^= 56 // Mirror rank for Black
			}
			mg += mgMat + mgTable[idx]
			eg += egMat + egTable[idx]
		}
	}
	return
}

// evaluateMobility calculates mobility bonus for a given color.
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

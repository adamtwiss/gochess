package chess

// Mobility bonuses per safe square (MG/EG), declared as var for future tuning
var (
	KnightMobilityMG = 4
	KnightMobilityEG = 4
	BishopMobilityMG = 5
	BishopMobilityEG = 5
	RookMobilityMG   = 2
	RookMobilityEG   = 3
	QueenMobilityMG  = 1
	QueenMobilityEG  = 2
)

// Piece evaluation bonuses (MG/EG)
var (
	BishopPairMG = 30
	BishopPairEG = 50

	KnightOutpostMG          = 15
	KnightOutpostEG          = 10
	KnightOutpostSupportedMG = 30
	KnightOutpostSupportedEG = 15

	RookOpenFileMG     = 20
	RookOpenFileEG     = 15
	RookSemiOpenFileMG = 10
	RookSemiOpenFileEG = 8
	RookOn7thMG        = 20
	RookOn7thEG        = 30

	BishopOpenPositionMG = 3
	BishopOpenPositionEG = 3

	PassedPawnNotBlockedEG = 10

	RookBehindPassedMG = 15
	RookBehindPassedEG = 25
)

// Evaluate returns a static evaluation of the position in centipawns.
// Positive values favor White, negative values favor Black.
// Uses tapered evaluation blending middlegame and endgame PST scores.
func (b *Board) Evaluate() int {
	wMG, wEG := b.evaluatePST(White)
	bMG, bEG := b.evaluatePST(Black)

	// Pawn structure (cached via pawn hash table)
	if b.PawnTable == nil {
		b.PawnTable = NewPawnTable(1) // 1 MB default
	}
	pawnEntry := b.probePawnEval()

	// Piece evaluation (mobility + positional bonuses)
	wPmg, wPeg := b.evaluatePieces(White, &pawnEntry)
	bPmg, bPeg := b.evaluatePieces(Black, &pawnEntry)

	// Passed pawn enhancements (piece-dependent, not cached)
	wPPmg, wPPeg := b.evaluatePassedPawns(White, &pawnEntry)
	bPPmg, bPPeg := b.evaluatePassedPawns(Black, &pawnEntry)

	// King safety (per-node, not cached)
	wKSmg, wKSeg := b.evaluateKingSafety(White)
	bKSmg, bKSeg := b.evaluateKingSafety(Black)

	mg := wMG - bMG +
		wPmg - bPmg +
		int(pawnEntry.WhiteMG) - int(pawnEntry.BlackMG) +
		wPPmg - bPPmg +
		wKSmg - bKSmg
	eg := wEG - bEG +
		wPeg - bPeg +
		int(pawnEntry.WhiteEG) - int(pawnEntry.BlackEG) +
		wPPeg - bPPeg +
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

// evaluatePieces computes mobility and positional bonuses for knights, bishops,
// rooks, and queens in a single pass. Replaces the old evaluateMobility().
func (b *Board) evaluatePieces(color Color, pawnEntry *PawnEntry) (mg, eg int) {
	friendly := b.Occupied[color]
	enemyPawns := b.Pieces[pieceOf(WhitePawn, 1-color)]
	friendlyPawns := b.Pieces[pieceOf(WhitePawn, color)]
	totalPawns := b.Pieces[WhitePawn].Count() + b.Pieces[BlackPawn].Count()

	// Precompute friendly pawn attacks for outpost support detection
	var friendlyPawnAttacks Bitboard
	if color == White {
		friendlyPawnAttacks = friendlyPawns.NorthWest() | friendlyPawns.NorthEast()
	} else {
		friendlyPawnAttacks = friendlyPawns.SouthWest() | friendlyPawns.SouthEast()
	}

	// Passed pawns for rook-behind-passer detection
	passedPawns := pawnEntry.Passed[color]

	// --- Knights ---
	knights := b.Pieces[pieceOf(WhiteKnight, color)]
	for knights != 0 {
		sq := knights.PopLSB()
		attacks := KnightAttacks[sq] &^ friendly
		count := attacks.Count()
		mg += count * KnightMobilityMG
		eg += count * KnightMobilityEG

		// Outpost: relative rank 4-6 (ranks 3-5 zero-indexed for White, 2-4 for Black)
		rank := sq.Rank()
		relRank := rank
		if color == Black {
			relRank = 7 - rank
		}
		if relRank >= 3 && relRank <= 5 {
			if OutpostMask[color][sq]&enemyPawns == 0 {
				// No enemy pawn can attack this square
				if SquareBB(sq)&friendlyPawnAttacks != 0 {
					mg += KnightOutpostSupportedMG
					eg += KnightOutpostSupportedEG
				} else {
					mg += KnightOutpostMG
					eg += KnightOutpostEG
				}
			}
		}
	}

	// --- Bishops ---
	bishops := b.Pieces[pieceOf(WhiteBishop, color)]
	bishopCount := bishops.Count()

	// Bishop pair bonus (checked once before loop)
	if bishopCount >= 2 {
		mg += BishopPairMG
		eg += BishopPairEG
	}

	for bishops != 0 {
		sq := bishops.PopLSB()
		attacks := BishopAttacksBB(sq, b.AllPieces) &^ friendly
		count := attacks.Count()
		mg += count * BishopMobilityMG
		eg += count * BishopMobilityEG

		// Open position bonus: more valuable with fewer pawns
		missingPawns := 16 - totalPawns
		mg += missingPawns * BishopOpenPositionMG
		eg += missingPawns * BishopOpenPositionEG
	}

	// --- Rooks ---
	rooks := b.Pieces[pieceOf(WhiteRook, color)]
	for rooks != 0 {
		sq := rooks.PopLSB()
		attacks := RookAttacksBB(sq, b.AllPieces) &^ friendly
		count := attacks.Count()
		mg += count * RookMobilityMG
		eg += count * RookMobilityEG

		file := sq.File()
		fileMask := FileMasks[file]

		// Open file: no pawns at all on this file
		if fileMask&(friendlyPawns|enemyPawns) == 0 {
			mg += RookOpenFileMG
			eg += RookOpenFileEG
		} else if fileMask&friendlyPawns == 0 {
			// Semi-open file: no friendly pawns on this file
			mg += RookSemiOpenFileMG
			eg += RookSemiOpenFileEG
		}

		// Rook on 7th rank (relative)
		rank := sq.Rank()
		relRank := rank
		if color == Black {
			relRank = 7 - rank
		}
		if relRank == 6 {
			mg += RookOn7thMG
			eg += RookOn7thEG
		}

		// Rook behind passed pawn: rook on same file, behind the passer
		if passedPawns&fileMask != 0 {
			// Find the most advanced passed pawn on this file
			filePassed := passedPawns & fileMask
			if color == White {
				// White: rook should be on a lower rank than the passer
				passerSq := filePassed.MSB()
				if rank < passerSq.Rank() {
					mg += RookBehindPassedMG
					eg += RookBehindPassedEG
				}
			} else {
				// Black: rook should be on a higher rank than the passer
				passerSq := filePassed.LSB()
				if rank > passerSq.Rank() {
					mg += RookBehindPassedMG
					eg += RookBehindPassedEG
				}
			}
		}
	}

	// --- Queens ---
	queens := b.Pieces[pieceOf(WhiteQueen, color)]
	for queens != 0 {
		sq := queens.PopLSB()
		attacks := QueenAttacksBB(sq, b.AllPieces) &^ friendly
		count := attacks.Count()
		mg += count * QueenMobilityMG
		eg += count * QueenMobilityEG
	}

	return
}

// evaluatePassedPawns computes piece-dependent passed pawn bonuses.
// These depend on piece positions so they cannot be cached in the pawn table.
func (b *Board) evaluatePassedPawns(color Color, pawnEntry *PawnEntry) (mg, eg int) {
	passed := pawnEntry.Passed[color]

	for passed != 0 {
		sq := passed.PopLSB()

		// Check if the square directly ahead is empty (not blocked by any piece)
		var aheadSq Square
		if color == White {
			aheadSq = sq + 8
		} else {
			aheadSq = sq - 8
		}

		if aheadSq >= 0 && aheadSq < 64 {
			if !b.AllPieces.IsSet(aheadSq) {
				eg += PassedPawnNotBlockedEG
			}
		}
	}

	return
}

package chess

// Mobility bonuses per safe square (MG/EG), declared as var for future tuning
var (
	KnightMobilityMG = 6
	KnightMobilityEG = 5
	BishopMobilityMG = 7
	BishopMobilityEG = 6
	RookMobilityMG   = 3
	RookMobilityEG   = 4
	QueenMobilityMG  = 2
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

	// Trapped rook: rook on back-rank corner with king blocking escape
	TrappedRookPenaltyMG = -40
	TrappedRookPenaltyEG = -20

	BishopOpenPositionMG = 3
	BishopOpenPositionEG = 3

	// Passed pawn: not-blocked bonus scaled by relative rank
	PassedPawnNotBlockedMG = [8]int{0, 0, 0, 5, 8, 12, 20, 0}
	PassedPawnNotBlockedEG = [8]int{0, 5, 5, 10, 15, 25, 40, 0}

	// Passed pawn: entire path to promotion clear
	PassedPawnFreePathMG = [8]int{0, 0, 0, 0, 5, 10, 20, 0}
	PassedPawnFreePathEG = [8]int{0, 0, 0, 5, 15, 30, 60, 0}

	// King proximity (EG only, per Chebyshev distance unit)
	PassedPawnFriendlyKingDistEG = -5 // closer = better
	PassedPawnEnemyKingDistEG    = 5  // farther = better
	PassedPawnKingScale          = [8]int{0, 0, 0, 1, 2, 3, 4, 0}

	// Protected passer (defended by own pawn)
	PassedPawnProtectedMG = 10
	PassedPawnProtectedEG = 15

	// Connected passers (friendly passer on adjacent file)
	PassedPawnConnectedMG = 10
	PassedPawnConnectedEG = 20

	RookBehindPassedMG = 15
	RookBehindPassedEG = 25

	// King attack weights per attacked square in king zone (MG only)
	KnightKingAttack = 2
	BishopKingAttack = 2
	RookKingAttack   = 3
	QueenKingAttack  = 5

	// Castling rights bonus (MG only, per retained right)
	CastlingRightsMG = 10

	// Space evaluation (per safe square in center files, ranks 4-6 relative)
	SpaceBonusMG = 2
	SpaceBonusEG = 0

	// Minor piece centralization (by center distance 0-3)
	KnightCentralityMG = [4]int{8, 4, 0, -4}
	KnightCentralityEG = [4]int{4, 2, 0, -2}
	BishopCentralityMG = [4]int{4, 2, 0, -2}
	BishopCentralityEG = [4]int{2, 1, 0, -1}

	// Knight closed position bonus (per pawn on the board)
	KnightClosedPositionMG = 2
	KnightClosedPositionEG = 1

	// Pawn threat bonuses (pawns attacking enemy pieces)
	PawnThreatMinorMG = 15
	PawnThreatMinorEG = 10
	PawnThreatRookMG  = 25
	PawnThreatRookEG  = 15
	PawnThreatQueenMG = 30
	PawnThreatQueenEG = 20
)

// EvalEntry is a single eval cache entry.
type EvalEntry struct {
	Key   uint64
	Score int16
}

// EvalTable caches Evaluate() results keyed by Zobrist hash.
type EvalTable struct {
	entries []EvalEntry
	mask    uint64
}

// NewEvalTable creates a new eval cache with the given size in MB.
func NewEvalTable(sizeMB int) *EvalTable {
	entrySize := uint64(16) // sizeof(EvalEntry) with padding
	numEntries := uint64(sizeMB*1024*1024) / entrySize
	size := uint64(1)
	for size*2 <= numEntries {
		size *= 2
	}
	return &EvalTable{
		entries: make([]EvalEntry, size),
		mask:    size - 1,
	}
}

// Evaluate returns a static evaluation of the position in centipawns.
// Positive values favor White, negative values favor Black.
// Uses tapered evaluation blending middlegame and endgame PST scores.
func (b *Board) Evaluate() int {
	// Eval cache probe
	if b.EvalTable == nil {
		b.EvalTable = NewEvalTable(1)
	}
	idx := b.HashKey & b.EvalTable.mask
	if e := b.EvalTable.entries[idx]; e.Key == b.HashKey {
		return int(e.Score)
	}

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

	// Space evaluation
	wSPmg, wSPeg := b.evaluateSpace(White)
	bSPmg, bSPeg := b.evaluateSpace(Black)

	// Pawn threats
	wTmg, wTeg := b.evaluateThreats(White)
	bTmg, bTeg := b.evaluateThreats(Black)

	// Castling rights bonus (middlegame only)
	castlingMG := 0
	if b.Castling&WhiteKingside != 0 {
		castlingMG += CastlingRightsMG
	}
	if b.Castling&WhiteQueenside != 0 {
		castlingMG += CastlingRightsMG
	}
	if b.Castling&BlackKingside != 0 {
		castlingMG -= CastlingRightsMG
	}
	if b.Castling&BlackQueenside != 0 {
		castlingMG -= CastlingRightsMG
	}

	mg := wMG - bMG +
		wPmg - bPmg +
		int(pawnEntry.WhiteMG) - int(pawnEntry.BlackMG) +
		wPPmg - bPPmg +
		wKSmg - bKSmg +
		wSPmg - bSPmg +
		wTmg - bTmg +
		castlingMG
	eg := wEG - bEG +
		wPeg - bPeg +
		int(pawnEntry.WhiteEG) - int(pawnEntry.BlackEG) +
		wPPeg - bPPeg +
		wKSeg - bKSeg +
		wSPeg - bSPeg +
		wTeg - bTeg

	phase := b.computePhase()
	score := (mg*(TotalPhase-phase) + eg*phase) / TotalPhase

	// Eval cache store
	b.EvalTable.entries[idx] = EvalEntry{Key: b.HashKey, Score: int16(score)}

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

		// Per-piece-type PST scaling
		var scaleMG, scaleEG int
		switch pt {
		case WhitePawn:
			scaleMG = PawnPSTScaleMG
			scaleEG = PawnPSTScaleEG
		case WhiteKing:
			scaleMG = KingPSTScaleMG
			scaleEG = KingPSTScaleEG
		default: // Knight, Bishop, Rook, Queen
			scaleMG = PiecePSTScaleMG
			scaleEG = PiecePSTScaleEG
		}

		for bb != 0 {
			sq := bb.PopLSB()
			idx := int(sq)
			if color == Black {
				idx ^= 56 // Mirror rank for Black
			}
			mg += mgMat + mgTable[idx]*scaleMG/100
			eg += egMat + egTable[idx]*scaleEG/100
		}
	}
	return
}

// evaluatePieces computes mobility and positional bonuses for knights, bishops,
// rooks, and queens in a single pass. Replaces the old evaluateMobility().
func (b *Board) evaluatePieces(color Color, pawnEntry *PawnEntry) (mg, eg int) {
	friendly := b.Occupied[color]
	enemy := color ^ 1
	enemyPawns := b.Pieces[pieceOf(WhitePawn, enemy)]
	friendlyPawns := b.Pieces[pieceOf(WhitePawn, color)]
	totalPawns := b.Pieces[WhitePawn].Count() + b.Pieces[BlackPawn].Count()

	// King attack tracking
	enemyKingSq := b.Pieces[pieceOf(WhiteKing, enemy)].LSB()
	kingZone := KingAttacks[enemyKingSq]
	var attackerCount, attackWeight int

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

		if kzAttacks := attacks & kingZone; kzAttacks != 0 {
			attackerCount++
			attackWeight += KnightKingAttack * kzAttacks.Count()
		}

		// Centralization bonus
		cd := centerDistance(sq)
		mg += KnightCentralityMG[cd]
		eg += KnightCentralityEG[cd]

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

		// Knight closed position bonus (more valuable with more pawns)
		mg += totalPawns * KnightClosedPositionMG
		eg += totalPawns * KnightClosedPositionEG
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
		// Diminishing returns: first 7 squares at full rate, excess at half
		effective := count
		if count > 7 {
			effective = 7 + (count-7)/2
		}
		mg += effective * BishopMobilityMG
		eg += effective * BishopMobilityEG

		if kzAttacks := attacks & kingZone; kzAttacks != 0 {
			attackerCount++
			attackWeight += BishopKingAttack * kzAttacks.Count()
		}

		// Centralization bonus
		cd := centerDistance(sq)
		mg += BishopCentralityMG[cd]
		eg += BishopCentralityEG[cd]

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
		// Diminishing returns: first 7 squares at full rate, excess at half
		effective := count
		if count > 7 {
			effective = 7 + (count-7)/2
		}
		mg += effective * RookMobilityMG
		eg += effective * RookMobilityEG

		if kzAttacks := attacks & kingZone; kzAttacks != 0 {
			attackerCount++
			attackWeight += RookKingAttack * kzAttacks.Count()
		}

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

		// Trapped rook: corner rook with king blocking escape route
		backRank := 0
		if color == Black {
			backRank = 7
		}
		if rank == backRank {
			kingSq := b.Pieces[pieceOf(WhiteKing, color)].LSB()
			kingFile := kingSq.File()
			kingRank := kingSq.Rank()
			if kingRank == backRank {
				rookFile := sq.File()
				// Kingside trap: rook on h-file, king on f/g-file
				if rookFile == 7 && (kingFile == 5 || kingFile == 6) {
					mg += TrappedRookPenaltyMG
					eg += TrappedRookPenaltyEG
				}
				// Queenside trap: rook on a-file, king on b/c-file
				if rookFile == 0 && (kingFile == 1 || kingFile == 2) {
					mg += TrappedRookPenaltyMG
					eg += TrappedRookPenaltyEG
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

		if kzAttacks := attacks & kingZone; kzAttacks != 0 {
			attackerCount++
			attackWeight += QueenKingAttack * kzAttacks.Count()
		}
	}

	// King attack penalty (quadratic scaling, MG only)
	if attackerCount >= 2 {
		mg += attackWeight * attackWeight / 4
	}

	return
}

// centerDistance returns the Chebyshev distance from a square to the
// center (d4/d5/e4/e5). Returns 0-3.
func centerDistance(sq Square) int {
	file := sq.File()
	rank := sq.Rank()
	fd := file - 3
	if file >= 4 {
		fd = file - 4
	}
	if fd < 0 {
		fd = -fd
	}
	rd := rank - 3
	if rank >= 4 {
		rd = rank - 4
	}
	if rd < 0 {
		rd = -rd
	}
	d := fd
	if rd > d {
		d = rd
	}
	if d > 3 {
		d = 3
	}
	return d
}

// chebyshevDistance returns the Chebyshev (king) distance between two squares.
func chebyshevDistance(sq1, sq2 Square) int {
	fd := sq1.File() - sq2.File()
	if fd < 0 {
		fd = -fd
	}
	rd := sq1.Rank() - sq2.Rank()
	if rd < 0 {
		rd = -rd
	}
	if fd > rd {
		return fd
	}
	return rd
}

// evaluatePassedPawns computes piece-dependent passed pawn bonuses.
// These depend on piece positions so they cannot be cached in the pawn table.
func (b *Board) evaluatePassedPawns(color Color, pawnEntry *PawnEntry) (mg, eg int) {
	allPassed := pawnEntry.Passed[color]
	passed := allPassed
	friendlyPawns := b.Pieces[pieceOf(WhitePawn, color)]
	friendlyKingSq := b.Pieces[pieceOf(WhiteKing, color)].LSB()
	enemyKingSq := b.Pieces[pieceOf(WhiteKing, 1-color)].LSB()

	// Precompute friendly pawn attacks for protected passer detection
	var friendlyPawnAttacks Bitboard
	if color == White {
		friendlyPawnAttacks = friendlyPawns.NorthWest() | friendlyPawns.NorthEast()
	} else {
		friendlyPawnAttacks = friendlyPawns.SouthWest() | friendlyPawns.SouthEast()
	}

	for passed != 0 {
		sq := passed.PopLSB()
		rank := sq.Rank()
		file := sq.File()
		relRank := rank
		if color == Black {
			relRank = 7 - rank
		}

		// 1. King proximity (EG only)
		scale := PassedPawnKingScale[relRank]
		if scale > 0 {
			friendlyDist := chebyshevDistance(friendlyKingSq, sq)
			enemyDist := chebyshevDistance(enemyKingSq, sq)
			eg += scale * (enemyDist*PassedPawnEnemyKingDistEG + friendlyDist*PassedPawnFriendlyKingDistEG)
		}

		// 2. Not blocked (rank-scaled)
		var aheadSq Square
		if color == White {
			aheadSq = sq + 8
		} else {
			aheadSq = sq - 8
		}

		notBlocked := aheadSq >= 0 && aheadSq < 64 && !b.AllPieces.IsSet(aheadSq)
		if notBlocked {
			mg += PassedPawnNotBlockedMG[relRank]
			eg += PassedPawnNotBlockedEG[relRank]

			// 3. Free path: entire path to promotion is clear
			if ForwardFileMask[color][sq]&b.AllPieces == 0 {
				mg += PassedPawnFreePathMG[relRank]
				eg += PassedPawnFreePathEG[relRank]
			}
		}

		// 4. Protected passer (defended by own pawn)
		if SquareBB(sq)&friendlyPawnAttacks != 0 {
			mg += PassedPawnProtectedMG
			eg += PassedPawnProtectedEG
		}

		// 5. Connected passers (friendly passer on adjacent file)
		if allPassed&AdjacentFiles[file] != 0 {
			mg += PassedPawnConnectedMG
			eg += PassedPawnConnectedEG
		}
	}

	return
}

// evaluateSpace rewards controlling territory in the opponent's half.
// Uses only pawn bitboards (cheap). Counts safe squares in center files, ranks 4-6 relative.
func (b *Board) evaluateSpace(color Color) (mg, eg int) {
	enemyPawns := b.Pieces[pieceOf(WhitePawn, 1-color)]

	// Enemy pawn attacks
	var enemyPawnAttacks Bitboard
	if color == White {
		enemyPawnAttacks = enemyPawns.SouthWest() | enemyPawns.SouthEast()
	} else {
		enemyPawnAttacks = enemyPawns.NorthWest() | enemyPawns.NorthEast()
	}

	// Space region: ranks 4-6 (relative), center files (c-f)
	centerFiles := FileC | FileD | FileE | FileF
	var spaceRegion Bitboard
	if color == White {
		spaceRegion = (Rank4 | Rank5 | Rank6) & centerFiles
	} else {
		spaceRegion = (Rank3 | Rank4 | Rank5) & centerFiles
	}

	// Safe space: in region, not attacked by enemy pawns
	safeSpace := spaceRegion &^ enemyPawnAttacks
	count := safeSpace.Count()
	mg += count * SpaceBonusMG
	eg += count * SpaceBonusEG
	return
}

// evaluateThreats rewards pawns attacking enemy pieces.
func (b *Board) evaluateThreats(color Color) (mg, eg int) {
	friendlyPawns := b.Pieces[pieceOf(WhitePawn, color)]
	enemy := color ^ 1

	var pawnAttacks Bitboard
	if color == White {
		pawnAttacks = friendlyPawns.NorthWest() | friendlyPawns.NorthEast()
	} else {
		pawnAttacks = friendlyPawns.SouthWest() | friendlyPawns.SouthEast()
	}

	minors := b.Pieces[pieceOf(WhiteKnight, enemy)] | b.Pieces[pieceOf(WhiteBishop, enemy)]
	mg += (pawnAttacks & minors).Count() * PawnThreatMinorMG
	eg += (pawnAttacks & minors).Count() * PawnThreatMinorEG
	mg += (pawnAttacks & b.Pieces[pieceOf(WhiteRook, enemy)]).Count() * PawnThreatRookMG
	eg += (pawnAttacks & b.Pieces[pieceOf(WhiteRook, enemy)]).Count() * PawnThreatRookEG
	mg += (pawnAttacks & b.Pieces[pieceOf(WhiteQueen, enemy)]).Count() * PawnThreatQueenMG
	eg += (pawnAttacks & b.Pieces[pieceOf(WhiteQueen, enemy)]).Count() * PawnThreatQueenEG
	return
}

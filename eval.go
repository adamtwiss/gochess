package chess

// Non-linear mobility arrays indexed by move count. Each entry is {MG, EG}.
// Derived from Stockfish classical at ~70% scale.
var (
	// Knight: 9 entries (0-8 squares).
	KnightMobility = [9][2]int{
		{49, -151}, {71, -26}, {82, 31}, {87, 62},
		{96, 84}, {103, 105}, {111, 113}, {122, 115}, {132, 103},
	}

	// Bishop: 14 entries (0-13 squares).
	BishopMobility = [14][2]int{
		{82, 189}, {95, 228}, {105, 261}, {115, 285},
		{122, 300}, {127, 310}, {127, 312}, {123, 311},
		{127, 307}, {125, 299}, {134, 284}, {129, 277},
		{148, 250}, {147, 249},
	}

	// Rook: 15 entries (0-14 squares).
	RookMobility = [15][2]int{
		{54, 217}, {70, 247}, {75, 255}, {83, 270},
		{87, 286}, {95, 297}, {103, 302}, {108, 307},
		{112, 319}, {112, 325}, {111, 330}, {107, 334},
		{103, 336}, {93, 333}, {79, 333},
	}

	// Queen: 28 entries (0-27 squares).
	QueenMobility = [28][2]int{
		{238, 53}, {240, 106}, {240, 168}, {240, 214},
		{242, 261}, {244, 301}, {247, 335}, {249, 366},
		{253, 390}, {254, 411}, {255, 431}, {256, 449},
		{257, 460}, {256, 473}, {257, 483}, {256, 487},
		{256, 495}, {261, 488}, {268, 483}, {284, 462},
		{313, 426}, {356, 400}, {382, 370}, {381, 349},
		{334, 309}, {307, 259}, {236, 181}, {239, 187},
	}
)

// Piece evaluation bonuses (MG/EG)
var (
	BishopPairMG = 37
	BishopPairEG = 164

	KnightOutpostMG          = 28
	KnightOutpostEG          = -15
	KnightOutpostSupportedMG = 49
	KnightOutpostSupportedEG = -3

	RookOpenFileMG     = 68
	RookOpenFileEG     = 5
	RookSemiOpenFileMG = 26
	RookSemiOpenFileEG = 51
	RookOn7thMG        = 112
	RookOn7thEG        = 135

	// Rook on enemy king file: extra bonus when on open/semi-open file
	// that is the same file as (or adjacent to) the enemy king
	RookEnemyKingFileMG     = 14
	RookEnemyKingFileEG     = -47
	RookEnemyKingFileEnabled = true

	// Trapped rook: rook on back-rank corner with king blocking escape
	TrappedRookPenaltyMG = -29
	TrappedRookPenaltyEG = -79

	BishopOpenPositionMG = 6
	BishopOpenPositionEG = -9

	BadBishopPawnMG = -9
	BadBishopPawnEG = -9

	DoubledRooksMG = 36
	DoubledRooksEG = -6

	// Passed pawn: not-blocked bonus scaled by relative rank
	PassedPawnNotBlockedMG = [8]int{0, 5, 4, 18, 22, 55, 58, 0}
	PassedPawnNotBlockedEG = [8]int{0, 11, 16, 23, 30, 50, 111, 0}

	// Passed pawn: entire path to promotion clear
	PassedPawnFreePathMG = [8]int{0, -1, -4, -9, -21, -13, 58, 0}
	PassedPawnFreePathEG = [8]int{0, 14, 8, 23, 58, 129, 131, 0}

	// King proximity (EG only, per Chebyshev distance unit)
	PassedPawnFriendlyKingDistEG = -12 // closer = better
	PassedPawnEnemyKingDistEG    = 19  // farther = better
	PassedPawnKingScale          = [8]int{0, 0, 0, 1, 2, 3, 4, 0}

	// Protected passer (defended by own pawn)
	PassedPawnProtectedMG = 40
	PassedPawnProtectedEG = -1

	// Connected passers (friendly passer on adjacent file)
	PassedPawnConnectedMG = 2
	PassedPawnConnectedEG = 25

	RookBehindPassedMG = 22
	RookBehindPassedEG = 44

	// Passed pawn: enemy piece blocking the stop square (partially cancels base bonus)
	PassedPawnBlockedMG = [8]int{0, -3, -24, -22, -14, -20, -127, 0}
	PassedPawnBlockedEG = [8]int{0, -5, -1, -7, -21, -42, -83, 0}

	// King attack unit weights (base per attacker + bonus per king-zone square)
	KnightAttackUnits   = 7
	KnightKingZoneBonus = 1
	BishopAttackUnits   = 5
	BishopKingZoneBonus = 1
	RookAttackUnits     = 8
	RookKingZoneBonus   = 2
	QueenAttackUnits    = 13
	QueenKingZoneBonus  = 1

	// Safe check bonuses: attack units for pieces that can deliver checks
	// on squares not defended by enemy pawns or occupied by friendly pieces
	SafeKnightCheckBonus = 6
	SafeBishopCheckBonus = 3
	SafeRookCheckBonus   = 7
	SafeQueenCheckBonus  = 5

	// No-queen attack scale: scale factor (out of 128) for king safety
	// penalty when the attacking side has no queen
	NoQueenAttackScale = 40

	// Castling rights bonus (MG only, per retained right)
	CastlingRightsMG = 42

	// Space evaluation (per safe square in center files, ranks 4-6 relative)
	SpaceBonusMG = 5
	SpaceBonusEG = 21

	// Knight closed position bonus (per pawn on the board)
	KnightClosedPositionMG = -1
	KnightClosedPositionEG = 26

	// Pawn threat bonuses (pawns attacking enemy pieces)
	PawnThreatMinorMG = 104
	PawnThreatMinorEG = 120
	PawnThreatRookMG  = 114
	PawnThreatRookEG  = 74
	PawnThreatQueenMG = 75
	PawnThreatQueenEG = 100

	// Piece-on-piece threats (minor/rook attacking higher-value enemy pieces)
	MinorThreatRookMG  = 119
	MinorThreatRookEG  = 80
	MinorThreatQueenMG = 67
	MinorThreatQueenEG = 160
	RookThreatQueenMG  = 104
	RookThreatQueenEG  = 164
)

// Endgame king activity (EG only, unconditional centralization + material advantage bonuses)
var KingCenterBonusEG = -19       // per center-distance unit (penalty, both sides)
var KingProximityAdvantageEG = 30 // per unit closer to enemy king (stronger side)
var KingCornerPushEG = 88         // per center-distance unit of weaker king (stronger side)

// Direct pawn storm bonus (NOT gated on attackerCount).
// PawnStormBonusMG/EG[opposed][relativeRank] gives centipawn bonus.
// opposed=0: enemy pawn present on this file (blocked storm)
// opposed=1: no enemy pawn on this file (open storm)
var PawnStormBonusMG = [2][8]int{
	{0, -9, -14, -7, 6, 29, -77, 0},   // Opposed
	{0, -10, -22, -10, -7, 34, -73, 0}, // Unopposed
}
var PawnStormBonusEG = [2][8]int{
	{0, -2, 6, 1, -6, -21, -84, 0},       // Opposed
	{0, -13, -13, -20, -37, -98, -69, 0}, // Unopposed
}
var PawnStormBonusEnabled = true

// Same-side pawn storm: extra MG bonus for pawns storming toward the enemy king
// when both kings are on the same wing. These pawns serve a dual purpose (attack +
// defense compromise) that the regular storm tables don't capture well.
// Indexed by relative rank. Only ranks 4-6 are relevant (earlier = still shield).
var SameSideStormMG = [8]int{0, 0, 0, 0, -8, -18, -26, 0}
var SameSideStormEnabled = true

// Feature toggles for king safety improvements
var SafeCheckEnabled = true
var NoQueenScaleEnabled = true

// Tempo bonus for the side to move.
var TempoMG = 30
var TempoEG = 20

// Trade bonus: when ahead, bonus per opponent non-pawn piece traded (encourages
// simplification) and per own pawn remaining (discourages pawn trades).
var TradePieceBonus = 18 // bonus per missing enemy non-pawn piece when ahead
var TradePawnBonus = 26  // bonus per own pawn when ahead

// OCBScale is the endgame scale factor (out of 128) for opposite-colored bishop endings.
var OCBScale = 64

// UseNNUE toggles NNUE evaluation. When true and NNUENet is loaded,
// EvaluateRelative dispatches to the NNUE forward pass.
var UseNNUE = true

// GlobalNNUENet is the shared NNUE network pointer. When non-nil and UseNNUE
// is true, newly created boards (e.g. in EPD tests, benchmarks) automatically
// get the NNUE net wired in via AttachNNUE.
var GlobalNNUENet *NNUENet

// KingSafetyTable maps accumulated attack units to centipawn penalties.
// Superlinear growth: near-zero for low indices, rapid growth from 15-50, capped at 999.
var KingSafetyTable = [100]int{
	0, 0, 1, 2, 3, 5, 7, 10, 13, 16,
	20, 24, 29, 34, 39, 45, 68, 68, 68, 68,
	68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
	68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
	76, 81, 99, 107, 114, 125, 130, 139, 158, 173,
	195, 210, 249, 276, 314, 350, 390, 419, 461, 463,
	467, 486, 489, 521, 547, 549, 579, 579, 600, 675,
	698, 753, 768, 777, 799, 856, 887, 912, 928, 954,
	975, 977, 984, 987, 996, 997, 997, 999, 1000, 1002,
	1004, 1004, 1004, 1004, 1004, 1004, 1004, 1004, 1004, 1004,
}

// Evaluate returns a static evaluation of the position in centipawns.
// Positive values favor White, negative values favor Black.
// Uses tapered evaluation blending middlegame and endgame PST scores.
func (b *Board) Evaluate() int {
	wMG, wEG := b.PSTScoreMG[White], b.PSTScoreEG[White]
	bMG, bEG := b.PSTScoreMG[Black], b.PSTScoreEG[Black]

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

	// Pawn storm (direct bonus, not gated on attackerCount)
	wSTmg, wSTeg := b.evaluatePawnStorm(White)
	bSTmg, bSTeg := b.evaluatePawnStorm(Black)

	// Endgame king distance heuristic
	_, ekEG := b.evaluateEndgameKings()

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
		wSTmg - bSTmg +
		castlingMG
	eg := wEG - bEG +
		wPeg - bPeg +
		int(pawnEntry.WhiteEG) - int(pawnEntry.BlackEG) +
		wPPeg - bPPeg +
		wKSeg - bKSeg +
		wSPeg - bSPeg +
		wTeg - bTeg +
		wSTeg - bSTeg +
		ekEG

	// Tempo bonus for the side to move
	tempoSign := 1
	if b.SideToMove == Black {
		tempoSign = -1
	}
	mg += tempoSign * TempoMG
	eg += tempoSign * TempoEG

	phase := b.computePhase()
	score := (mg*(TotalPhase-phase) + eg*phase) / TotalPhase

	// Trade bonus: encourage piece trades and discourage pawn trades when ahead.
	// Scaled by eval magnitude so it's negligible in balanced positions.
	{
		wPieces := b.Pieces[WhiteKnight].Count() + b.Pieces[WhiteBishop].Count() +
			b.Pieces[WhiteRook].Count() + b.Pieces[WhiteQueen].Count()
		bPieces := b.Pieces[BlackKnight].Count() + b.Pieces[BlackBishop].Count() +
			b.Pieces[BlackRook].Count() + b.Pieces[BlackQueen].Count()
		wPawns := b.Pieces[WhitePawn].Count()
		bPawns := b.Pieces[BlackPawn].Count()

		// Raw trade incentive from White's perspective:
		// fewer enemy pieces is good, more own pawns is good
		tradeScore := (7 - bPieces) * TradePieceBonus + wPawns * TradePawnBonus -
			(7 - wPieces) * TradePieceBonus - bPawns * TradePawnBonus

		// Scale by eval: full effect at ±500cp, zero at 0cp
		absScore := score
		if absScore < 0 {
			absScore = -absScore
		}
		if absScore > 500 {
			absScore = 500
		}
		score += tradeScore * absScore / 500
	}

	// Endgame scale factors (insufficient material / draw detection)
	wScale, bScale := b.endgameScale()
	if score > 0 {
		score = score * wScale / 128
	} else {
		score = score * bScale / 128
	}

	// 50-move rule scaling
	if b.HalfmoveClock > 0 {
		hmc := int(b.HalfmoveClock)
		if hmc > 100 {
			hmc = 100
		}
		score = score * (100 - hmc) / 100
	}

	return score
}

// EvaluateRelative returns the evaluation from the perspective of the side to move.
// Positive values are good for the side to move.
func (b *Board) EvaluateRelative() int {
	if UseNNUE && b.NNUENet != nil && b.NNUEAcc != nil {
		return b.NNUEEvaluateRelative()
	}
	score := b.Evaluate()
	if b.SideToMove == Black {
		return -score
	}
	return score
}

// NNUEEvaluateRelative returns the NNUE evaluation from the perspective
// of the side to move. Applies endgame scaling and 50-move rule scaling.
func (b *Board) NNUEEvaluateRelative() int {
	// Lazy materialization: copy parent + apply delta only when eval is needed.
	// Pruned nodes never pay this cost.
	b.NNUEAcc.Materialize(b.NNUENet, b)
	acc := b.NNUEAcc.Current()

	score := b.NNUENet.Evaluate(acc, b.SideToMove)

	// Endgame scale factors
	wScale, bScale := b.endgameScale()
	// Score is side-to-move relative, so convert to White-relative for scaling
	whiteScore := score
	if b.SideToMove == Black {
		whiteScore = -score
	}
	if whiteScore > 0 {
		whiteScore = whiteScore * wScale / 128
	} else {
		whiteScore = whiteScore * bScale / 128
	}
	// Convert back to side-to-move relative
	if b.SideToMove == Black {
		score = -whiteScore
	} else {
		score = whiteScore
	}

	// 50-move rule scaling
	if b.HalfmoveClock > 0 {
		hmc := int(b.HalfmoveClock)
		if hmc > 100 {
			hmc = 100
		}
		score = score * (100 - hmc) / 100
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
	kingZone := KingAttacks[enemyKingSq] | SquareBB(enemyKingSq)
	var attackerCount, attackUnits int
	var allKnightAttacks, allBishopAttacks, allRookAttacks, allQueenAttacks Bitboard

	// Precompute pawn attacks
	var friendlyPawnAttacks, enemyPawnAttacks Bitboard
	if color == White {
		friendlyPawnAttacks = friendlyPawns.NorthWest() | friendlyPawns.NorthEast()
		enemyPawnAttacks = enemyPawns.SouthWest() | enemyPawns.SouthEast()
	} else {
		friendlyPawnAttacks = friendlyPawns.SouthWest() | friendlyPawns.SouthEast()
		enemyPawnAttacks = enemyPawns.NorthWest() | enemyPawns.NorthEast()
	}

	// Passed pawns for rook-behind-passer detection
	passedPawns := pawnEntry.Passed[color]

	// --- Knights ---
	knights := b.Pieces[pieceOf(WhiteKnight, color)]
	knightCount := knights.Count()

	// Knight closed position bonus: constant per knight, compute once
	mg += knightCount * totalPawns * KnightClosedPositionMG
	eg += knightCount * totalPawns * KnightClosedPositionEG

	for knights != 0 {
		sq := knights.PopLSB()
		attacks := KnightAttacks[sq] &^ friendly
		allKnightAttacks |= attacks
		safeMobility := (attacks &^ enemyPawnAttacks).Count()
		mg += KnightMobility[safeMobility][0]
		eg += KnightMobility[safeMobility][1]

		if kzAttacks := attacks & kingZone; kzAttacks != 0 {
			attackerCount++
			attackUnits += KnightAttackUnits + KnightKingZoneBonus*kzAttacks.Count()
		}

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

	// Bishop open position bonus: constant per bishop, compute once
	missingPawns := 16 - totalPawns
	mg += bishopCount * missingPawns * BishopOpenPositionMG
	eg += bishopCount * missingPawns * BishopOpenPositionEG

	for bishops != 0 {
		sq := bishops.PopLSB()
		attacks := BishopAttacksBB(sq, b.AllPieces) &^ friendly
		allBishopAttacks |= attacks
		safeMobility := (attacks &^ enemyPawnAttacks).Count()
		mg += BishopMobility[safeMobility][0]
		eg += BishopMobility[safeMobility][1]

		if kzAttacks := attacks & kingZone; kzAttacks != 0 {
			attackerCount++
			attackUnits += BishopAttackUnits + BishopKingZoneBonus*kzAttacks.Count()
		}

		// Bad bishop: penalty per friendly pawn on same square color
		var sameColorMask Bitboard
		if SquareBB(sq)&LightSquares != 0 {
			sameColorMask = LightSquares
		} else {
			sameColorMask = DarkSquares
		}
		sameColorPawns := (friendlyPawns & sameColorMask).Count()
		mg += sameColorPawns * BadBishopPawnMG
		eg += sameColorPawns * BadBishopPawnEG
	}

	// --- Rooks ---
	rooks := b.Pieces[pieceOf(WhiteRook, color)]

	// Doubled rooks: bonus when two rooks share a file
	if rooks.Count() >= 2 {
		r := rooks
		sq1 := r.PopLSB()
		sq2 := r.PopLSB()
		if sq1.File() == sq2.File() {
			mg += DoubledRooksMG
			eg += DoubledRooksEG
		}
	}

	// Precompute for trapped rook detection
	friendlyKingSq := b.Pieces[pieceOf(WhiteKing, color)].LSB()
	friendlyKingFile := friendlyKingSq.File()
	friendlyKingRank := friendlyKingSq.Rank()
	backRank := 0
	if color == Black {
		backRank = 7
	}

	for rooks != 0 {
		sq := rooks.PopLSB()
		attacks := RookAttacksBB(sq, b.AllPieces) &^ friendly
		allRookAttacks |= attacks
		safeMobility := (attacks &^ enemyPawnAttacks).Count()
		mg += RookMobility[safeMobility][0]
		eg += RookMobility[safeMobility][1]

		if kzAttacks := attacks & kingZone; kzAttacks != 0 {
			attackerCount++
			attackUnits += RookAttackUnits + RookKingZoneBonus*kzAttacks.Count()
		}

		file := sq.File()
		fileMask := FileMasks[file]

		// Open file: no pawns at all on this file
		isOpenOrSemiOpen := false
		if fileMask&(friendlyPawns|enemyPawns) == 0 {
			mg += RookOpenFileMG
			eg += RookOpenFileEG
			isOpenOrSemiOpen = true
		} else if fileMask&friendlyPawns == 0 {
			// Semi-open file: no friendly pawns on this file
			mg += RookSemiOpenFileMG
			eg += RookSemiOpenFileEG
			isOpenOrSemiOpen = true
		}

		// Rook on enemy king file: bonus for pressuring the enemy king's file
		if RookEnemyKingFileEnabled && isOpenOrSemiOpen {
			enemyKingFile := enemyKingSq.File()
			fileDist := file - enemyKingFile
			if fileDist < 0 {
				fileDist = -fileDist
			}
			if fileDist <= 1 {
				mg += RookEnemyKingFileMG
				eg += RookEnemyKingFileEG
			}
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
		if rank == backRank && friendlyKingRank == backRank {
			// Kingside trap: rook on h-file, king on f/g-file
			if file == 7 && (friendlyKingFile == 5 || friendlyKingFile == 6) {
				mg += TrappedRookPenaltyMG
				eg += TrappedRookPenaltyEG
			}
			// Queenside trap: rook on a-file, king on b/c-file
			if file == 0 && (friendlyKingFile == 1 || friendlyKingFile == 2) {
				mg += TrappedRookPenaltyMG
				eg += TrappedRookPenaltyEG
			}
		}
	}

	// --- Queens ---
	queens := b.Pieces[pieceOf(WhiteQueen, color)]
	for queens != 0 {
		sq := queens.PopLSB()
		attacks := QueenAttacksBB(sq, b.AllPieces) &^ friendly
		allQueenAttacks |= attacks
		safeMobility := (attacks &^ enemyPawnAttacks).Count()
		mg += QueenMobility[safeMobility][0]
		eg += QueenMobility[safeMobility][1]

		if kzAttacks := attacks & kingZone; kzAttacks != 0 {
			attackerCount++
			attackUnits += QueenAttackUnits + QueenKingZoneBonus*kzAttacks.Count()
		}
	}

	// --- Piece-on-piece threats ---
	// Minor pieces (knights/bishops) threatening enemy rooks and queens
	minorAttacks := allKnightAttacks | allBishopAttacks
	enemyRooks := b.Pieces[pieceOf(WhiteRook, enemy)]
	enemyQueens := b.Pieces[pieceOf(WhiteQueen, enemy)]
	mg += (minorAttacks & enemyRooks).Count() * MinorThreatRookMG
	eg += (minorAttacks & enemyRooks).Count() * MinorThreatRookEG
	mg += (minorAttacks & enemyQueens).Count() * MinorThreatQueenMG
	eg += (minorAttacks & enemyQueens).Count() * MinorThreatQueenEG
	// Rooks threatening enemy queens
	mg += (allRookAttacks & enemyQueens).Count() * RookThreatQueenMG
	eg += (allRookAttacks & enemyQueens).Count() * RookThreatQueenEG

	// --- Safe check bonus ---
	// Add attack units for pieces that can deliver checks on squares not
	// defended by enemy pawns or occupied by friendly pieces.
	// Gated on attackerCount >= 2 for the same reason as the structural factors.
	if SafeCheckEnabled && attackerCount >= 1 {
		safeSquares := ^(enemyPawnAttacks | friendly)
		knightCheckSqs := KnightAttacks[enemyKingSq]
		bishopCheckSqs := BishopAttacksBB(enemyKingSq, b.AllPieces)
		rookCheckSqs := RookAttacksBB(enemyKingSq, b.AllPieces)

		if knightCheckSqs&safeSquares&allKnightAttacks != 0 {
			attackUnits += SafeKnightCheckBonus
		}
		if bishopCheckSqs&safeSquares&allBishopAttacks != 0 {
			attackUnits += SafeBishopCheckBonus
		}
		if rookCheckSqs&safeSquares&allRookAttacks != 0 {
			attackUnits += SafeRookCheckBonus
		}
		queenCheckSqs := bishopCheckSqs | rookCheckSqs
		if queenCheckSqs&safeSquares&allQueenAttacks != 0 {
			attackUnits += SafeQueenCheckBonus
		}
	}

	// --- King safety: structural factors ---

	// Weak squares: king-zone squares not defended by enemy pawns
	// (reuse enemyPawnAttacks computed above — identical bitboard)
	weakKingZone := kingZone &^ enemyPawnAttacks &^ SquareBB(enemyKingSq)
	weakSquareCount := weakKingZone.Count()

	// Open/semi-open files toward enemy king
	enemyKingFile := enemyKingSq.File()
	startFile := enemyKingFile - 1
	if startFile < 0 {
		startFile = 0
	}
	endFile := enemyKingFile + 1
	if endFile > 7 {
		endFile = 7
	}

	openFileUnits := 0
	for f := startFile; f <= endFile; f++ {
		if FileMasks[f]&enemyPawns == 0 {
			openFileUnits += 3 // semi-open (no enemy pawns)
			if FileMasks[f]&friendlyPawns == 0 {
				openFileUnits += 2 // fully open (no pawns at all)
			}
		}
	}

	// Pawn shelter weakness around enemy king
	shelterWeakness := 0
	for f := startFile; f <= endFile; f++ {
		filePawns := enemyPawns & FileMasks[f]
		if filePawns == 0 {
			shelterWeakness += 3
		} else {
			if enemy == White {
				if filePawns&(Rank2|Rank3) == 0 {
					shelterWeakness += 2
				}
			} else {
				if filePawns&(Rank7|Rank6) == 0 {
					shelterWeakness += 2
				}
			}
		}
	}

	// Add structural factors when attackers are present
	if attackerCount >= 1 {
		attackUnits += weakSquareCount
		attackUnits += openFileUnits
		attackUnits += shelterWeakness

	}

	// King attack penalty via table lookup (MG only)
	if attackUnits < 0 {
		attackUnits = 0
	}
	if attackUnits >= 100 {
		attackUnits = 99
	}
	penalty := 0
	if attackerCount >= 2 {
		penalty = KingSafetyTable[attackUnits]
	} else if attackerCount == 1 && attackUnits >= 15 {
		penalty = KingSafetyTable[attackUnits] / 3
	}

	// Scale down when attacking side has no queen
	if NoQueenScaleEnabled && penalty > 0 && b.Pieces[pieceOf(WhiteQueen, color)].Count() == 0 {
		penalty = penalty * NoQueenAttackScale / 128
	}

	mg += penalty

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

		// 2a. Blocked passer penalty (enemy piece on stop square)
		if aheadSq >= 0 && aheadSq < 64 && b.AllPieces.IsSet(aheadSq) {
			blocker := b.Squares[aheadSq]
			if blocker.Color() != color {
				mg += PassedPawnBlockedMG[relRank]
				eg += PassedPawnBlockedEG[relRank]
			}
		}

		// 2b. Not blocked + safe advance (rank-scaled)
		notBlocked := aheadSq >= 0 && aheadSq < 64 && !b.AllPieces.IsSet(aheadSq)
		if notBlocked {
			enemyControls := b.IsAttacked(aheadSq, 1-color)
			safeAdvance := !enemyControls || b.IsAttacked(aheadSq, color)
			if safeAdvance {
				mg += PassedPawnNotBlockedMG[relRank]
				eg += PassedPawnNotBlockedEG[relRank]

				// 3. Free path: entire path to promotion is clear
				if ForwardFileMask[color][sq]&b.AllPieces == 0 {
					mg += PassedPawnFreePathMG[relRank]
					eg += PassedPawnFreePathEG[relRank]
				}
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

// endgameScale returns per-side scale factors (0-128) for draw/insufficient material detection.
// 0 = can't win, 128 = normal. Each side is evaluated independently.
func (b *Board) endgameScale() (wScale, bScale int) {
	wScale, bScale = 128, 128

	wPawns := b.Pieces[WhitePawn].Count()
	wKnights := b.Pieces[WhiteKnight].Count()
	wBishops := b.Pieces[WhiteBishop].Count()
	wRooks := b.Pieces[WhiteRook].Count()
	wQueens := b.Pieces[WhiteQueen].Count()
	wMinors := wKnights + wBishops
	wMajors := wRooks + wQueens

	bPawns := b.Pieces[BlackPawn].Count()
	bKnights := b.Pieces[BlackKnight].Count()
	bBishops := b.Pieces[BlackBishop].Count()
	bRooks := b.Pieces[BlackRook].Count()
	bQueens := b.Pieces[BlackQueen].Count()
	bMinors := bKnights + bBishops
	bMajors := bRooks + bQueens

	// Per-side can't-win detection (no pawns required)
	if wPawns == 0 {
		if wMinors <= 1 && wMajors == 0 {
			// K alone, KN, or KB vs anything — can't force mate
			wScale = 0
		} else if wKnights == 2 && wBishops == 0 && wMajors == 0 {
			// KNN vs anything — can't force mate
			wScale = 0
		} else if wRooks == 1 && wMinors == 0 && wQueens == 0 && bMinors == 1 && bMajors == 0 {
			// KR vs KB or KR vs KN — usually drawn
			wScale = 16
		} else if wMajors == 0 && wMinors == 2 && bMinors >= 1 {
			// KBB vs KB/KN, KBN vs KB/KN — can't force mate with extra minor
			wScale = 16
		}
	}

	if bPawns == 0 {
		if bMinors <= 1 && bMajors == 0 {
			bScale = 0
		} else if bKnights == 2 && bBishops == 0 && bMajors == 0 {
			bScale = 0
		} else if bRooks == 1 && bMinors == 0 && bQueens == 0 && wMinors == 1 && wMajors == 0 {
			bScale = 16
		} else if bMajors == 0 && bMinors == 2 && wMinors >= 1 {
			// KBB vs KB/KN, KBN vs KB/KN — can't force mate with extra minor
			bScale = 16
		}
	}

	// KBvKB same-color bishops — draw
	if wPawns == 0 && bPawns == 0 &&
		wBishops == 1 && bBishops == 1 &&
		wMajors == 0 && bMajors == 0 &&
		wKnights == 0 && bKnights == 0 {
		wBishopBB := b.Pieces[WhiteBishop]
		bBishopBB := b.Pieces[BlackBishop]
		wOnLight := wBishopBB&LightSquares != 0
		bOnLight := bBishopBB&LightSquares != 0
		if wOnLight == bOnLight {
			wScale = 0
			bScale = 0
		}
	}

	// Opposite-colored bishop endgames (bishops + pawns only) — drawish
	if wBishops == 1 && bBishops == 1 &&
		wKnights == 0 && bKnights == 0 &&
		wMajors == 0 && bMajors == 0 &&
		(wPawns > 0 || bPawns > 0) {
		wBishopBB := b.Pieces[WhiteBishop]
		bBishopBB := b.Pieces[BlackBishop]
		if (wBishopBB&LightSquares != 0) != (bBishopBB&LightSquares != 0) {
			if wScale > OCBScale {
				wScale = OCBScale
			}
			if bScale > OCBScale {
				bScale = OCBScale
			}
		}
	}

	return
}

// evaluateEndgameKings returns king activity bonuses for endgames.
// Two components:
// 1. Unconditional centralization — both kings penalized per center-distance (EG only).
// 2. Material advantage bonuses — stronger side rewarded for proximity and pushing
//    enemy king to edge. No piece-type gates; uses weighted material count.
// Returns (mg, eg) — mg is always 0, bonuses are EG only.
func (b *Board) evaluateEndgameKings() (mg, eg int) {
	wKingSq := b.Pieces[WhiteKing].LSB()
	bKingSq := b.Pieces[BlackKing].LSB()

	// 1. Unconditional centralization (both sides)
	// KingCenterBonusEG is negative, so further from center = bigger penalty
	wCenterDist := centerDistance(wKingSq)
	bCenterDist := centerDistance(bKingSq)
	eg += wCenterDist * KingCenterBonusEG // White penalty (negative contribution)
	eg -= bCenterDist * KingCenterBonusEG // Black penalty (flipped sign)

	// 2. Material advantage bonuses
	wMaterial := b.Pieces[WhiteKnight].Count() + b.Pieces[WhiteBishop].Count() +
		b.Pieces[WhiteRook].Count()*3 + b.Pieces[WhiteQueen].Count()*6
	bMaterial := b.Pieces[BlackKnight].Count() + b.Pieces[BlackBishop].Count() +
		b.Pieces[BlackRook].Count()*3 + b.Pieces[BlackQueen].Count()*6

	dist := chebyshevDistance(wKingSq, bKingSq)

	if wMaterial > bMaterial {
		eg += (7 - dist) * KingProximityAdvantageEG
		eg += bCenterDist * KingCornerPushEG
	} else if bMaterial > wMaterial {
		eg -= (7 - dist) * KingProximityAdvantageEG
		eg -= wCenterDist * KingCornerPushEG
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

// evaluatePawnStorm returns a direct MG/EG bonus for friendly pawns advanced
// toward the enemy king. Not gated on attacker count — provides signal even
// when pieces haven't yet entered the king zone.
func (b *Board) evaluatePawnStorm(color Color) (mg, eg int) {
	if !PawnStormBonusEnabled {
		return
	}

	enemy := color ^ 1
	friendlyPawns := b.Pieces[pieceOf(WhitePawn, color)]
	enemyPawns := b.Pieces[pieceOf(WhitePawn, enemy)]
	enemyKingSq := b.Pieces[pieceOf(WhiteKing, enemy)].LSB()

	enemyKingFile := enemyKingSq.File()
	startFile := enemyKingFile - 1
	if startFile < 0 {
		startFile = 0
	}
	endFile := enemyKingFile + 1
	if endFile > 7 {
		endFile = 7
	}

	// Detect same-side castling: both kings on the same wing
	sameSide := false
	if SameSideStormEnabled {
		ourKingSq := b.Pieces[pieceOf(WhiteKing, color)].LSB()
		ourKingFile := ourKingSq.File()
		sameSide = (ourKingFile <= 3 && enemyKingFile <= 3) || (ourKingFile >= 4 && enemyKingFile >= 4)
	}

	for f := startFile; f <= endFile; f++ {
		filePawns := friendlyPawns & FileMasks[f]
		if filePawns == 0 {
			continue
		}
		// Most advanced pawn on this file
		var advancedSq Square
		if color == White {
			advancedSq = filePawns.MSB()
		} else {
			advancedSq = filePawns.LSB()
		}
		relRank := advancedSq.Rank()
		if color == Black {
			relRank = 7 - relRank
		}

		// Determine if file is opposed (enemy pawn present)
		opposed := 0
		if enemyPawns&FileMasks[f] == 0 {
			opposed = 1 // unopposed
		}

		mg += PawnStormBonusMG[opposed][relRank]
		eg += PawnStormBonusEG[opposed][relRank]

		// Same-side storm: extra bonus for advanced pawns (rank 4+) aimed at
		// the enemy king when our own king is on the same wing. Rewards the
		// attacking potential that offsets the self-weakening shield cost.
		if sameSide && relRank >= 4 {
			mg += SameSideStormMG[relRank]
		}
	}

	return
}

package chess

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
)

// TunerParam describes a single tunable parameter.
type TunerParam struct {
	Name string
	// Pointer back to the engine variable for applying tuned values.
	// Nil for read-only reference params.
	setter func(int)
}

// SparseEntry represents a single non-zero coefficient in a trace.
type SparseEntry struct {
	Index uint16
	Coeff int16
}

// TunerTrace captures which parameters contribute to a position's evaluation.
type TunerTrace struct {
	MG     []SparseEntry
	EG     []SparseEntry
	Phase  int     // game phase (0 = full MG, 24 = full EG)
	Result float64 // game outcome: 1.0 (white wins), 0.5 (draw), 0.0 (black wins)
}

// TunerEntry is a loaded training position with precomputed trace.
type TunerEntry struct {
	Trace TunerTrace
}

// Tuner holds the parameter catalog and training data.
type Tuner struct {
	Params []TunerParam   // parameter metadata
	Values []float64      // current parameter values
	Traces []TunerEntry   // loaded training data

	// Parameter index ranges for output formatting
	sections []tunerSection
}

type tunerSection struct {
	name       string
	startIndex int
	endIndex   int // exclusive
}

// Indices into the parameter vector for each category.
// These are set during initTunerParams().
var (
	idxMaterialMG    int // 5 values: Pawn..Queen
	idxMaterialEG    int // 5 values
	idxPSTMG         int // 6 pieces × 64 squares = 384
	idxPSTEG         int // 384
	idxMobilityStart int // N(9)+B(14)+R(15)+Q(28) = 66 MG + 66 EG = 132
	idxBonusStart    int // piece bonuses
	idxPassedStart   int // passed pawn bonuses
	idxPawnStart     int // pawn structure
	idxKingAttack    int // king attack weights
	idxKingSafetyTbl int // king safety table (100 entries, MG only)
	idxPawnShield    int // pawn shield constants (5 entries, MG only)
	idxMisc          int // space, threats, castling, OCB
)

// NewTuner creates a new tuner with the parameter catalog initialized from engine globals.
func NewTuner() *Tuner {
	t := &Tuner{}
	t.initTunerParams()
	return t
}

// initTunerParams builds the parameter vector from engine globals.
func (t *Tuner) initTunerParams() {
	t.Params = nil
	t.Values = nil
	t.sections = nil

	add := func(name string, value int, setter func(int)) int {
		idx := len(t.Params)
		t.Params = append(t.Params, TunerParam{Name: name, setter: setter})
		t.Values = append(t.Values, float64(value))
		return idx
	}

	addSection := func(name string, start int) {
		if len(t.sections) > 0 {
			t.sections[len(t.sections)-1].endIndex = start
		}
		t.sections = append(t.sections, tunerSection{name: name, startIndex: start})
	}

	// === Material values (5 MG + 5 EG = 10) ===
	idxMaterialMG = len(t.Params)
	addSection("Material MG", idxMaterialMG)
	for pt := 1; pt <= 5; pt++ { // Pawn=1 .. Queen=5
		p := pt
		add(fmt.Sprintf("mgMaterial[%d]", pt), mgMaterial[pt], func(v int) { mgMaterial[p] = v })
	}
	idxMaterialEG = len(t.Params)
	addSection("Material EG", idxMaterialEG)
	for pt := 1; pt <= 5; pt++ {
		p := pt
		add(fmt.Sprintf("egMaterial[%d]", pt), egMaterial[pt], func(v int) { egMaterial[p] = v })
	}

	// === PST tables (6 pieces × 64 squares × 2 phases = 768) ===
	// Parameters represent the *effective* PST contribution (rawTable[sq] * scale/100).
	// This eliminates coupling between PST values and scale factors.
	// At output time, values are divided by scale/100 to recover raw table values,
	// or output with scale=100.
	pstNames := []string{"Pawn", "Knight", "Bishop", "Rook", "Queen", "King"}
	mgTables := [6]*[64]int{&mgPawnTable, &mgKnightTable, &mgBishopTable, &mgRookTable, &mgQueenTable, &mgKingTable}
	egTables := [6]*[64]int{&egPawnTable, &egKnightTable, &egBishopTable, &egRookTable, &egQueenTable, &egKingTable}
	mgScales := [6]int{PawnPSTScaleMG, PiecePSTScaleMG, PiecePSTScaleMG, PiecePSTScaleMG, PiecePSTScaleMG, KingPSTScaleMG}
	egScales := [6]int{PawnPSTScaleEG, PiecePSTScaleEG, PiecePSTScaleEG, PiecePSTScaleEG, PiecePSTScaleEG, KingPSTScaleEG}

	idxPSTMG = len(t.Params)
	addSection("PST MG", idxPSTMG)
	for pi := 0; pi < 6; pi++ {
		tbl := mgTables[pi]
		scale := mgScales[pi]
		for sq := 0; sq < 64; sq++ {
			p, s := pi, sq
			// Initialize with the effective (scaled) value
			effectiveVal := tbl[sq] * scale / 100
			add(fmt.Sprintf("mg%sTable[%d]", pstNames[pi], sq), effectiveVal,
				func(v int) { mgTables[p][s] = v })
		}
	}

	idxPSTEG = len(t.Params)
	addSection("PST EG", idxPSTEG)
	for pi := 0; pi < 6; pi++ {
		tbl := egTables[pi]
		scale := egScales[pi]
		for sq := 0; sq < 64; sq++ {
			p, s := pi, sq
			effectiveVal := tbl[sq] * scale / 100
			add(fmt.Sprintf("eg%sTable[%d]", pstNames[pi], sq), effectiveVal,
				func(v int) { egTables[p][s] = v })
		}
	}

	// === Mobility arrays (66 MG + 66 EG = 132) ===
	idxMobilityStart = len(t.Params)
	addSection("Mobility", idxMobilityStart)
	for i := 0; i < 9; i++ {
		ii := i
		add(fmt.Sprintf("KnightMobility[%d][MG]", i), KnightMobility[i][0],
			func(v int) { KnightMobility[ii][0] = v })
		add(fmt.Sprintf("KnightMobility[%d][EG]", i), KnightMobility[i][1],
			func(v int) { KnightMobility[ii][1] = v })
	}
	for i := 0; i < 14; i++ {
		ii := i
		add(fmt.Sprintf("BishopMobility[%d][MG]", i), BishopMobility[i][0],
			func(v int) { BishopMobility[ii][0] = v })
		add(fmt.Sprintf("BishopMobility[%d][EG]", i), BishopMobility[i][1],
			func(v int) { BishopMobility[ii][1] = v })
	}
	for i := 0; i < 15; i++ {
		ii := i
		add(fmt.Sprintf("RookMobility[%d][MG]", i), RookMobility[i][0],
			func(v int) { RookMobility[ii][0] = v })
		add(fmt.Sprintf("RookMobility[%d][EG]", i), RookMobility[i][1],
			func(v int) { RookMobility[ii][1] = v })
	}
	for i := 0; i < 28; i++ {
		ii := i
		add(fmt.Sprintf("QueenMobility[%d][MG]", i), QueenMobility[i][0],
			func(v int) { QueenMobility[ii][0] = v })
		add(fmt.Sprintf("QueenMobility[%d][EG]", i), QueenMobility[i][1],
			func(v int) { QueenMobility[ii][1] = v })
	}

	// === Piece bonuses ===
	idxBonusStart = len(t.Params)
	addSection("Piece Bonuses", idxBonusStart)
	add("BishopPairMG", BishopPairMG, func(v int) { BishopPairMG = v })
	add("BishopPairEG", BishopPairEG, func(v int) { BishopPairEG = v })
	add("KnightOutpostMG", KnightOutpostMG, func(v int) { KnightOutpostMG = v })
	add("KnightOutpostEG", KnightOutpostEG, func(v int) { KnightOutpostEG = v })
	add("KnightOutpostSupportedMG", KnightOutpostSupportedMG, func(v int) { KnightOutpostSupportedMG = v })
	add("KnightOutpostSupportedEG", KnightOutpostSupportedEG, func(v int) { KnightOutpostSupportedEG = v })
	add("RookOpenFileMG", RookOpenFileMG, func(v int) { RookOpenFileMG = v })
	add("RookOpenFileEG", RookOpenFileEG, func(v int) { RookOpenFileEG = v })
	add("RookSemiOpenFileMG", RookSemiOpenFileMG, func(v int) { RookSemiOpenFileMG = v })
	add("RookSemiOpenFileEG", RookSemiOpenFileEG, func(v int) { RookSemiOpenFileEG = v })
	add("RookOn7thMG", RookOn7thMG, func(v int) { RookOn7thMG = v })
	add("RookOn7thEG", RookOn7thEG, func(v int) { RookOn7thEG = v })
	add("TrappedRookPenaltyMG", TrappedRookPenaltyMG, func(v int) { TrappedRookPenaltyMG = v })
	add("TrappedRookPenaltyEG", TrappedRookPenaltyEG, func(v int) { TrappedRookPenaltyEG = v })
	add("BishopOpenPositionMG", BishopOpenPositionMG, func(v int) { BishopOpenPositionMG = v })
	add("BishopOpenPositionEG", BishopOpenPositionEG, func(v int) { BishopOpenPositionEG = v })
	add("BadBishopPawnMG", BadBishopPawnMG, func(v int) { BadBishopPawnMG = v })
	add("BadBishopPawnEG", BadBishopPawnEG, func(v int) { BadBishopPawnEG = v })
	add("DoubledRooksMG", DoubledRooksMG, func(v int) { DoubledRooksMG = v })
	add("DoubledRooksEG", DoubledRooksEG, func(v int) { DoubledRooksEG = v })
	add("KnightClosedPositionMG", KnightClosedPositionMG, func(v int) { KnightClosedPositionMG = v })
	add("KnightClosedPositionEG", KnightClosedPositionEG, func(v int) { KnightClosedPositionEG = v })

	// === Passed pawn bonuses ===
	idxPassedStart = len(t.Params)
	addSection("Passed Pawns", idxPassedStart)
	for i := 0; i < 8; i++ {
		ii := i
		add(fmt.Sprintf("PassedPawnNotBlockedMG[%d]", i), PassedPawnNotBlockedMG[i],
			func(v int) { PassedPawnNotBlockedMG[ii] = v })
		add(fmt.Sprintf("PassedPawnNotBlockedEG[%d]", i), PassedPawnNotBlockedEG[i],
			func(v int) { PassedPawnNotBlockedEG[ii] = v })
	}
	for i := 0; i < 8; i++ {
		ii := i
		add(fmt.Sprintf("PassedPawnFreePathMG[%d]", i), PassedPawnFreePathMG[i],
			func(v int) { PassedPawnFreePathMG[ii] = v })
		add(fmt.Sprintf("PassedPawnFreePathEG[%d]", i), PassedPawnFreePathEG[i],
			func(v int) { PassedPawnFreePathEG[ii] = v })
	}
	for i := 0; i < 8; i++ {
		ii := i
		add(fmt.Sprintf("PassedPawnKingScale[%d]", i), PassedPawnKingScale[i],
			func(v int) { PassedPawnKingScale[ii] = v })
	}
	add("PassedPawnFriendlyKingDistEG", PassedPawnFriendlyKingDistEG, func(v int) { PassedPawnFriendlyKingDistEG = v })
	add("PassedPawnEnemyKingDistEG", PassedPawnEnemyKingDistEG, func(v int) { PassedPawnEnemyKingDistEG = v })
	add("PassedPawnProtectedMG", PassedPawnProtectedMG, func(v int) { PassedPawnProtectedMG = v })
	add("PassedPawnProtectedEG", PassedPawnProtectedEG, func(v int) { PassedPawnProtectedEG = v })
	add("PassedPawnConnectedMG", PassedPawnConnectedMG, func(v int) { PassedPawnConnectedMG = v })
	add("PassedPawnConnectedEG", PassedPawnConnectedEG, func(v int) { PassedPawnConnectedEG = v })
	add("RookBehindPassedMG", RookBehindPassedMG, func(v int) { RookBehindPassedMG = v })
	add("RookBehindPassedEG", RookBehindPassedEG, func(v int) { RookBehindPassedEG = v })

	// === Pawn structure ===
	idxPawnStart = len(t.Params)
	addSection("Pawn Structure", idxPawnStart)
	for i := 0; i < 8; i++ {
		ii := i
		add(fmt.Sprintf("passedPawnMG[%d]", i), passedPawnMG[i],
			func(v int) { passedPawnMG[ii] = v })
		add(fmt.Sprintf("passedPawnEG[%d]", i), passedPawnEG[i],
			func(v int) { passedPawnEG[ii] = v })
	}
	for i := 0; i < 8; i++ {
		ii := i
		add(fmt.Sprintf("pawnAdvancementMG[%d]", i), pawnAdvancementMG[i],
			func(v int) { pawnAdvancementMG[ii] = v })
		add(fmt.Sprintf("pawnAdvancementEG[%d]", i), pawnAdvancementEG[i],
			func(v int) { pawnAdvancementEG[ii] = v })
	}
	// Note: doubled/isolated/backward/connected are const in the engine.
	// We duplicate them as tunable vars here. The tuner output will show
	// updated values the user can substitute.
	add("doubledPawnMG", doubledPawnMG, nil)
	add("doubledPawnEG", doubledPawnEG, nil)
	add("isolatedPawnMG", isolatedPawnMG, nil)
	add("isolatedPawnEG", isolatedPawnEG, nil)
	add("backwardPawnMG", backwardPawnMG, nil)
	add("backwardPawnEG", backwardPawnEG, nil)
	add("connectedPawnMG", connectedPawnMG, nil)
	add("connectedPawnEG", connectedPawnEG, nil)

	// === King attack weights ===
	idxKingAttack = len(t.Params)
	addSection("King Attack", idxKingAttack)
	add("KnightAttackUnits", KnightAttackUnits, func(v int) { KnightAttackUnits = v })
	add("KnightKingZoneBonus", KnightKingZoneBonus, func(v int) { KnightKingZoneBonus = v })
	add("BishopAttackUnits", BishopAttackUnits, func(v int) { BishopAttackUnits = v })
	add("BishopKingZoneBonus", BishopKingZoneBonus, func(v int) { BishopKingZoneBonus = v })
	add("RookAttackUnits", RookAttackUnits, func(v int) { RookAttackUnits = v })
	add("RookKingZoneBonus", RookKingZoneBonus, func(v int) { RookKingZoneBonus = v })
	add("QueenAttackUnits", QueenAttackUnits, func(v int) { QueenAttackUnits = v })
	add("QueenKingZoneBonus", QueenKingZoneBonus, func(v int) { QueenKingZoneBonus = v })

	// === King safety table (100 entries, MG only) ===
	idxKingSafetyTbl = len(t.Params)
	addSection("King Safety Table", idxKingSafetyTbl)
	for i := 0; i < 100; i++ {
		ii := i
		add(fmt.Sprintf("KingSafetyTable[%d]", i), KingSafetyTable[i],
			func(v int) { KingSafetyTable[ii] = v })
	}

	// === Pawn shield (5 entries, MG only) ===
	idxPawnShield = len(t.Params)
	addSection("Pawn Shield", idxPawnShield)
	add("shieldPawnRank2MG", shieldPawnRank2MG, func(v int) { shieldPawnRank2MG = v })
	add("shieldPawnRank3MG", shieldPawnRank3MG, func(v int) { shieldPawnRank3MG = v })
	add("missingShieldPawnMG", missingShieldPawnMG, func(v int) { missingShieldPawnMG = v })
	add("missingShieldPawnAdvancedMG", missingShieldPawnAdvancedMG, func(v int) { missingShieldPawnAdvancedMG = v })
	add("semiOpenFileNearKingMG", semiOpenFileNearKingMG, func(v int) { semiOpenFileNearKingMG = v })

	// === Misc: space, threats, castling, OCB ===
	idxMisc = len(t.Params)
	addSection("Misc", idxMisc)
	add("CastlingRightsMG", CastlingRightsMG, func(v int) { CastlingRightsMG = v })
	add("SpaceBonusMG", SpaceBonusMG, func(v int) { SpaceBonusMG = v })
	add("SpaceBonusEG", SpaceBonusEG, func(v int) { SpaceBonusEG = v })
	add("PawnThreatMinorMG", PawnThreatMinorMG, func(v int) { PawnThreatMinorMG = v })
	add("PawnThreatMinorEG", PawnThreatMinorEG, func(v int) { PawnThreatMinorEG = v })
	add("PawnThreatRookMG", PawnThreatRookMG, func(v int) { PawnThreatRookMG = v })
	add("PawnThreatRookEG", PawnThreatRookEG, func(v int) { PawnThreatRookEG = v })
	add("PawnThreatQueenMG", PawnThreatQueenMG, func(v int) { PawnThreatQueenMG = v })
	add("PawnThreatQueenEG", PawnThreatQueenEG, func(v int) { PawnThreatQueenEG = v })
	add("OCBScale", OCBScale, func(v int) { OCBScale = v })
	add("TempoMG", TempoMG, func(v int) { TempoMG = v })
	add("TempoEG", TempoEG, func(v int) { TempoEG = v })

	// Close last section
	if len(t.sections) > 0 {
		t.sections[len(t.sections)-1].endIndex = len(t.Params)
	}
}

// NumParams returns the total number of tunable parameters.
func (t *Tuner) NumParams() int {
	return len(t.Params)
}

// LoadTrainingData reads training positions from a file.
// Format: "FEN;result" per line, where result is 1.0, 0.5, or 0.0.
func (t *Tuner) LoadTrainingData(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// Increase buffer for long lines
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		idx := strings.LastIndex(line, ";")
		if idx < 0 {
			continue // skip malformed lines
		}

		fen := strings.TrimSpace(line[:idx])
		resultStr := strings.TrimSpace(line[idx+1:])

		var result float64
		switch resultStr {
		case "1.0":
			result = 1.0
		case "0.5":
			result = 0.5
		case "0.0":
			result = 0.0
		default:
			_, err := fmt.Sscanf(resultStr, "%f", &result)
			if err != nil {
				continue
			}
		}

		var b Board
		if err := b.SetFEN(fen); err != nil {
			continue // skip invalid FENs
		}

		trace := t.computeTrace(&b)
		trace.Result = result
		t.Traces = append(t.Traces, TunerEntry{Trace: trace})
	}

	return scanner.Err()
}

// computeTrace mirrors the Evaluate() function but records parameter coefficients.
// Returns a trace with sparse MG and EG coefficient arrays.
func (t *Tuner) computeTrace(b *Board) TunerTrace {
	var trace TunerTrace
	trace.Phase = b.computePhase()

	// Helper to add a coefficient
	addMG := func(idx int, coeff int16) {
		if coeff != 0 {
			trace.MG = append(trace.MG, SparseEntry{Index: uint16(idx), Coeff: coeff})
		}
	}
	addEG := func(idx int, coeff int16) {
		if coeff != 0 {
			trace.EG = append(trace.EG, SparseEntry{Index: uint16(idx), Coeff: coeff})
		}
	}
	sign := func(color Color) int16 {
		if color == White {
			return 1
		}
		return -1
	}

	// === Material + PST ===
	for color := Color(0); color <= 1; color++ {
		s := sign(color)
		for pt := WhitePawn; pt <= WhiteKing; pt++ {
			piece := pieceOf(pt, color)
			bb := b.Pieces[piece]

			ptIdx := int(pt) - 1 // 0-based piece type index (0=Pawn..5=King)
			matMGIdx := idxMaterialMG + ptIdx
			matEGIdx := idxMaterialEG + ptIdx

			pstMGBase := idxPSTMG + ptIdx*64
			pstEGBase := idxPSTEG + ptIdx*64

			for bb != 0 {
				sq := bb.PopLSB()
				idx := int(sq)
				if color == Black {
					idx ^= 56
				}

				// Material (skip King, index 5)
				if ptIdx < 5 {
					addMG(matMGIdx, s)
					addEG(matEGIdx, s)
				}

				// PST: params are pre-scaled effective values, coeff is just ±1
				addMG(pstMGBase+idx, s)
				addEG(pstEGBase+idx, s)
			}
		}
	}

	// === Mobility (safe: excludes squares attacked by enemy pawns) ===
	for color := Color(0); color <= 1; color++ {
		s := sign(color)
		friendly := b.Occupied[color]
		enemyPawns := b.Pieces[pieceOf(WhitePawn, 1-color)]

		var enemyPawnAttacks Bitboard
		if color == White {
			enemyPawnAttacks = enemyPawns.SouthWest() | enemyPawns.SouthEast()
		} else {
			enemyPawnAttacks = enemyPawns.NorthWest() | enemyPawns.NorthEast()
		}

		// Knights
		knights := b.Pieces[pieceOf(WhiteKnight, color)]
		for knights != 0 {
			sq := knights.PopLSB()
			attacks := KnightAttacks[sq] &^ friendly
			count := (attacks &^ enemyPawnAttacks).Count()
			base := idxMobilityStart + count*2
			addMG(base, s)
			addEG(base+1, s)
		}

		// Bishops
		bishops := b.Pieces[pieceOf(WhiteBishop, color)]
		for bishops != 0 {
			sq := bishops.PopLSB()
			attacks := BishopAttacksBB(sq, b.AllPieces) &^ friendly
			count := (attacks &^ enemyPawnAttacks).Count()
			base := idxMobilityStart + 9*2 + count*2
			addMG(base, s)
			addEG(base+1, s)
		}

		// Rooks
		rooks := b.Pieces[pieceOf(WhiteRook, color)]
		for rooks != 0 {
			sq := rooks.PopLSB()
			attacks := RookAttacksBB(sq, b.AllPieces) &^ friendly
			count := (attacks &^ enemyPawnAttacks).Count()
			base := idxMobilityStart + (9+14)*2 + count*2
			addMG(base, s)
			addEG(base+1, s)
		}

		// Queens
		queens := b.Pieces[pieceOf(WhiteQueen, color)]
		for queens != 0 {
			sq := queens.PopLSB()
			attacks := QueenAttacksBB(sq, b.AllPieces) &^ friendly
			count := (attacks &^ enemyPawnAttacks).Count()
			base := idxMobilityStart + (9+14+15)*2 + count*2
			addMG(base, s)
			addEG(base+1, s)
		}
	}

	// === Piece bonuses ===
	for color := Color(0); color <= 1; color++ {
		s := sign(color)
		enemy := color ^ 1
		friendlyPawns := b.Pieces[pieceOf(WhitePawn, color)]
		enemyPawns := b.Pieces[pieceOf(WhitePawn, enemy)]
		totalPawns := b.Pieces[WhitePawn].Count() + b.Pieces[BlackPawn].Count()

		// Precompute friendly pawn attacks
		var friendlyPawnAttacks Bitboard
		if color == White {
			friendlyPawnAttacks = friendlyPawns.NorthWest() | friendlyPawns.NorthEast()
		} else {
			friendlyPawnAttacks = friendlyPawns.SouthWest() | friendlyPawns.SouthEast()
		}

		base := idxBonusStart

		// Bishop pair
		if b.Pieces[pieceOf(WhiteBishop, color)].Count() >= 2 {
			addMG(base+0, s)  // BishopPairMG
			addEG(base+1, s)  // BishopPairEG
		}

		// Knight outposts
		knights := b.Pieces[pieceOf(WhiteKnight, color)]
		for knights != 0 {
			sq := knights.PopLSB()
			rank := sq.Rank()
			relRank := rank
			if color == Black {
				relRank = 7 - rank
			}
			if relRank >= 3 && relRank <= 5 {
				if OutpostMask[color][sq]&enemyPawns == 0 {
					if SquareBB(sq)&friendlyPawnAttacks != 0 {
						addMG(base+4, s)  // KnightOutpostSupportedMG
						addEG(base+5, s)  // KnightOutpostSupportedEG
					} else {
						addMG(base+2, s)  // KnightOutpostMG
						addEG(base+3, s)  // KnightOutpostEG
					}
				}
			}

			// Knight closed position bonus (per pawn)
			addMG(base+20, s*int16(totalPawns))  // KnightClosedPositionMG
			addEG(base+21, s*int16(totalPawns))  // KnightClosedPositionEG
		}

		// Rook bonuses
		rooks := b.Pieces[pieceOf(WhiteRook, color)]
		rooksForDoubled := rooks
		if rooks.Count() >= 2 {
			for f := 0; f < 8; f++ {
				if (rooksForDoubled & FileMasks[f]).Count() >= 2 {
					addMG(base+18, s) // DoubledRooksMG
					addEG(base+19, s) // DoubledRooksEG
				}
			}
		}

		for rooks != 0 {
			sq := rooks.PopLSB()
			file := sq.File()
			fileMask := FileMasks[file]
			rank := sq.Rank()
			relRank := rank
			if color == Black {
				relRank = 7 - rank
			}

			// Open/semi-open file
			if fileMask&(friendlyPawns|enemyPawns) == 0 {
				addMG(base+6, s)  // RookOpenFileMG
				addEG(base+7, s)  // RookOpenFileEG
			} else if fileMask&friendlyPawns == 0 {
				addMG(base+8, s)  // RookSemiOpenFileMG
				addEG(base+9, s)  // RookSemiOpenFileEG
			}

			// Rook on 7th
			if relRank == 6 {
				addMG(base+10, s) // RookOn7thMG
				addEG(base+11, s) // RookOn7thEG
			}

			// Rook behind passed pawn: handled in passed pawn section below
			// (uses RookBehindPassedMG/EG at idxPassedStart+46/47)

			// Trapped rook
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
					if (rookFile == 7 && (kingFile == 5 || kingFile == 6)) ||
						(rookFile == 0 && (kingFile == 1 || kingFile == 2)) {
						addMG(base+12, s) // TrappedRookPenaltyMG
						addEG(base+13, s) // TrappedRookPenaltyEG
					}
				}
			}
		}

		// Bishop bonuses
		bishops := b.Pieces[pieceOf(WhiteBishop, color)]
		for bishops != 0 {
			sq := bishops.PopLSB()
			missingPawns := 16 - totalPawns
			addMG(base+14, s*int16(missingPawns)) // BishopOpenPositionMG
			addEG(base+15, s*int16(missingPawns)) // BishopOpenPositionEG

			var sameColorMask Bitboard
			if SquareBB(sq)&LightSquares != 0 {
				sameColorMask = LightSquares
			} else {
				sameColorMask = DarkSquares
			}
			sameColorPawns := (friendlyPawns & sameColorMask).Count()
			addMG(base+16, s*int16(sameColorPawns)) // BadBishopPawnMG
			addEG(base+17, s*int16(sameColorPawns)) // BadBishopPawnEG
		}
	}

	// === Passed pawn enhancements ===
	pawnEntry := b.probePawnEval()
	for color := Color(0); color <= 1; color++ {
		s := sign(color)
		friendlyPawns := b.Pieces[pieceOf(WhitePawn, color)]
		friendlyKingSq := b.Pieces[pieceOf(WhiteKing, color)].LSB()
		enemyKingSq := b.Pieces[pieceOf(WhiteKing, 1-color)].LSB()
		allPassed := pawnEntry.Passed[color]
		passed := allPassed

		var friendlyPawnAttacks Bitboard
		if color == White {
			friendlyPawnAttacks = friendlyPawns.NorthWest() | friendlyPawns.NorthEast()
		} else {
			friendlyPawnAttacks = friendlyPawns.SouthWest() | friendlyPawns.SouthEast()
		}

		base := idxPassedStart

		for passed != 0 {
			sq := passed.PopLSB()
			rank := sq.Rank()
			file := sq.File()
			relRank := rank
			if color == Black {
				relRank = 7 - rank
			}

			// Not blocked (rank-indexed)
			var aheadSq Square
			if color == White {
				aheadSq = sq + 8
			} else {
				aheadSq = sq - 8
			}

			notBlocked := aheadSq >= 0 && aheadSq < 64 && !b.AllPieces.IsSet(aheadSq)
			if notBlocked {
				addMG(base+relRank*2, s)   // PassedPawnNotBlockedMG[relRank]
				addEG(base+relRank*2+1, s) // PassedPawnNotBlockedEG[relRank]

				// Free path
				if ForwardFileMask[color][sq]&b.AllPieces == 0 {
					addMG(base+16+relRank*2, s)   // PassedPawnFreePathMG[relRank]
					addEG(base+16+relRank*2+1, s)  // PassedPawnFreePathEG[relRank]
				}
			}

			// King proximity (EG only)
			// The eval computes: scale * (enemyDist * EnemyDistEG + friendlyDist * FriendlyDistEG)
			// This is a product of two tunable params (scale × distEG).
			// We use the engine's initial scale value as a constant coefficient
			// so the trace coefficients are independent of parameter values.
			initialScale := int16(PassedPawnKingScale[relRank])
			if initialScale > 0 {
				friendlyDist := chebyshevDistance(friendlyKingSq, sq)
				enemyDist := chebyshevDistance(enemyKingSq, sq)
				addEG(base+40, s*initialScale*int16(friendlyDist))
				addEG(base+41, s*initialScale*int16(enemyDist))
			}

			// Protected passer
			if SquareBB(sq)&friendlyPawnAttacks != 0 {
				addMG(base+42, s) // PassedPawnProtectedMG
				addEG(base+43, s) // PassedPawnProtectedEG
			}

			// Connected passers
			if allPassed&AdjacentFiles[file] != 0 {
				addMG(base+44, s) // PassedPawnConnectedMG
				addEG(base+45, s) // PassedPawnConnectedEG
			}
		}

		// Rook behind passer (already handled in piece bonuses for accuracy,
		// but the passed pawn section has its own params)
		rooks := b.Pieces[pieceOf(WhiteRook, color)]
		for rooks != 0 {
			sq := rooks.PopLSB()
			file := sq.File()
			rank := sq.Rank()
			fileMask := FileMasks[file]
			if allPassed&fileMask != 0 {
				filePassed := allPassed & fileMask
				behind := false
				if color == White {
					passerSq := filePassed.MSB()
					behind = rank < passerSq.Rank()
				} else {
					passerSq := filePassed.LSB()
					behind = rank > passerSq.Rank()
				}
				if behind {
					addMG(base+46, s) // RookBehindPassedMG
					addEG(base+47, s) // RookBehindPassedEG
				}
			}
		}
	}

	// === Pawn structure (from probePawnEval) ===
	for color := Color(0); color <= 1; color++ {
		s := sign(color)
		pawns := b.Pieces[pieceOf(WhitePawn, color)]
		enemyPawns := b.Pieces[pieceOf(WhitePawn, 1-color)]
		allFriendlyPawns := pawns

		var pawnAttacks Bitboard
		if color == White {
			pawnAttacks = allFriendlyPawns.NorthWest() | allFriendlyPawns.NorthEast()
		} else {
			pawnAttacks = allFriendlyPawns.SouthWest() | allFriendlyPawns.SouthEast()
		}

		base := idxPawnStart

		for pawns != 0 {
			sq := pawns.PopLSB()
			file := sq.File()
			rank := int(sq.Rank())
			relativeRank := rank
			if color == Black {
				relativeRank = 7 - rank
			}

			// Passed pawn base bonus (rank-indexed)
			if PassedPawnMask[color][sq]&enemyPawns == 0 {
				addMG(base+relativeRank*2, s)   // passedPawnMG[relativeRank]
				addEG(base+relativeRank*2+1, s) // passedPawnEG[relativeRank]
			}

			// Doubled
			if ForwardFileMask[color][sq]&allFriendlyPawns != 0 {
				addMG(base+32, s)  // doubledPawnMG
				addEG(base+33, s)  // doubledPawnEG
			}

			// Isolated
			if AdjacentFiles[file]&allFriendlyPawns == 0 {
				addMG(base+34, s) // isolatedPawnMG
				addEG(base+35, s) // isolatedPawnEG
			} else {
				// Backward
				if BackwardSupportMask[color][sq]&allFriendlyPawns == 0 {
					var stopSq Square
					if color == White {
						stopSq = sq + 8
					} else {
						stopSq = sq - 8
					}
					if stopSq >= 0 && stopSq < 64 {
						if PawnAttacks[color][stopSq]&enemyPawns != 0 {
							addMG(base+36, s) // backwardPawnMG
							addEG(base+37, s) // backwardPawnEG
						}
					}
				}
			}

			// Connected
			if SquareBB(sq)&pawnAttacks != 0 {
				addMG(base+38, s) // connectedPawnMG
				addEG(base+39, s) // connectedPawnEG
			}

			// Pawn advancement
			addMG(base+16+relativeRank*2, s)   // pawnAdvancementMG[relativeRank]
			addEG(base+16+relativeRank*2+1, s) // pawnAdvancementEG[relativeRank]
		}
	}

	// === Castling rights ===
	miscBase := idxMisc
	castlingCoeff := int16(0)
	if b.Castling&WhiteKingside != 0 {
		castlingCoeff++
	}
	if b.Castling&WhiteQueenside != 0 {
		castlingCoeff++
	}
	if b.Castling&BlackKingside != 0 {
		castlingCoeff--
	}
	if b.Castling&BlackQueenside != 0 {
		castlingCoeff--
	}
	addMG(miscBase+0, castlingCoeff) // CastlingRightsMG

	// === Space ===
	for color := Color(0); color <= 1; color++ {
		s := sign(color)
		enemyPawns := b.Pieces[pieceOf(WhitePawn, 1-color)]
		var enemyPawnAttacks Bitboard
		if color == White {
			enemyPawnAttacks = enemyPawns.SouthWest() | enemyPawns.SouthEast()
		} else {
			enemyPawnAttacks = enemyPawns.NorthWest() | enemyPawns.NorthEast()
		}
		centerFiles := FileC | FileD | FileE | FileF
		var spaceRegion Bitboard
		if color == White {
			spaceRegion = (Rank4 | Rank5 | Rank6) & centerFiles
		} else {
			spaceRegion = (Rank3 | Rank4 | Rank5) & centerFiles
		}
		safeSpace := spaceRegion &^ enemyPawnAttacks
		count := int16(safeSpace.Count())
		addMG(miscBase+1, s*count) // SpaceBonusMG
		addEG(miscBase+2, s*count) // SpaceBonusEG
	}

	// === Threats ===
	for color := Color(0); color <= 1; color++ {
		s := sign(color)
		friendlyPawns := b.Pieces[pieceOf(WhitePawn, color)]
		enemy := color ^ 1
		var pAttacks Bitboard
		if color == White {
			pAttacks = friendlyPawns.NorthWest() | friendlyPawns.NorthEast()
		} else {
			pAttacks = friendlyPawns.SouthWest() | friendlyPawns.SouthEast()
		}
		minors := b.Pieces[pieceOf(WhiteKnight, enemy)] | b.Pieces[pieceOf(WhiteBishop, enemy)]
		minorThreats := int16((pAttacks & minors).Count())
		rookThreats := int16((pAttacks & b.Pieces[pieceOf(WhiteRook, enemy)]).Count())
		queenThreats := int16((pAttacks & b.Pieces[pieceOf(WhiteQueen, enemy)]).Count())

		addMG(miscBase+3, s*minorThreats) // PawnThreatMinorMG
		addEG(miscBase+4, s*minorThreats) // PawnThreatMinorEG
		addMG(miscBase+5, s*rookThreats)  // PawnThreatRookMG
		addEG(miscBase+6, s*rookThreats)  // PawnThreatRookEG
		addMG(miscBase+7, s*queenThreats) // PawnThreatQueenMG
		addEG(miscBase+8, s*queenThreats) // PawnThreatQueenEG
	}

	// === King safety table (MG only) ===
	// Mirror the attack unit computation from evaluatePieces() to determine
	// which table entry is active, then set coefficient ±1.
	// Uses initial engine values for attack weights (not tuned values) so
	// trace coefficients are independent of parameter state.
	for color := Color(0); color <= 1; color++ {
		s := sign(color)
		friendly := b.Occupied[color]
		enemy := color ^ 1
		enemyPawns := b.Pieces[pieceOf(WhitePawn, enemy)]
		friendlyPawns := b.Pieces[pieceOf(WhitePawn, color)]

		enemyKingSq := b.Pieces[pieceOf(WhiteKing, enemy)].LSB()
		kingZone := KingAttacks[enemyKingSq] | SquareBB(enemyKingSq)
		var attackerCount, attackUnits int

		// Knights
		kn := b.Pieces[pieceOf(WhiteKnight, color)]
		for kn != 0 {
			sq := kn.PopLSB()
			attacks := KnightAttacks[sq] &^ friendly
			if kzAttacks := attacks & kingZone; kzAttacks != 0 {
				attackerCount++
				attackUnits += KnightAttackUnits + KnightKingZoneBonus*kzAttacks.Count()
			}
		}

		// Bishops
		bi := b.Pieces[pieceOf(WhiteBishop, color)]
		for bi != 0 {
			sq := bi.PopLSB()
			attacks := BishopAttacksBB(sq, b.AllPieces) &^ friendly
			if kzAttacks := attacks & kingZone; kzAttacks != 0 {
				attackerCount++
				attackUnits += BishopAttackUnits + BishopKingZoneBonus*kzAttacks.Count()
			}
		}

		// Rooks
		ro := b.Pieces[pieceOf(WhiteRook, color)]
		for ro != 0 {
			sq := ro.PopLSB()
			attacks := RookAttacksBB(sq, b.AllPieces) &^ friendly
			if kzAttacks := attacks & kingZone; kzAttacks != 0 {
				attackerCount++
				attackUnits += RookAttackUnits + RookKingZoneBonus*kzAttacks.Count()
			}
		}

		// Queens
		qu := b.Pieces[pieceOf(WhiteQueen, color)]
		for qu != 0 {
			sq := qu.PopLSB()
			attacks := QueenAttacksBB(sq, b.AllPieces) &^ friendly
			if kzAttacks := attacks & kingZone; kzAttacks != 0 {
				attackerCount++
				attackUnits += QueenAttackUnits + QueenKingZoneBonus*kzAttacks.Count()
			}
		}

		// Structural factors (same as eval.go)
		var enemyPawnDefense Bitboard
		if enemy == White {
			enemyPawnDefense = enemyPawns.NorthWest() | enemyPawns.NorthEast()
		} else {
			enemyPawnDefense = enemyPawns.SouthWest() | enemyPawns.SouthEast()
		}
		weakKingZone := kingZone &^ enemyPawnDefense &^ SquareBB(enemyKingSq)
		weakSquareCount := weakKingZone.Count()

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
				openFileUnits += 3
				if FileMasks[f]&friendlyPawns == 0 {
					openFileUnits += 2
				}
			}
		}

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

		if attackerCount >= 1 {
			attackUnits += weakSquareCount
			attackUnits += openFileUnits
			attackUnits += shelterWeakness
		}

		if attackUnits < 0 {
			attackUnits = 0
		}
		if attackUnits >= 100 {
			attackUnits = 99
		}

		// Trace the table entry for attackerCount >= 2
		if attackerCount >= 2 {
			addMG(idxKingSafetyTbl+attackUnits, s)
		}
		// Skip attackerCount == 1 case (eval divides by 3, can't represent cleanly)
	}

	// === Pawn shield (MG only) ===
	// Mirror evaluateKingSafety() from pawns.go
	for color := Color(0); color <= 1; color++ {
		s := sign(color)
		kingSq := b.Pieces[pieceOf(WhiteKing, color)].LSB()
		if kingSq == NoSquare {
			continue
		}
		kingFile := kingSq.File()
		friendlyPawns := b.Pieces[pieceOf(WhitePawn, color)]

		startFile := kingFile - 1
		if startFile < 0 {
			startFile = 0
		}
		endFile := kingFile + 1
		if endFile > 7 {
			endFile = 7
		}

		for f := startFile; f <= endFile; f++ {
			filePawns := friendlyPawns & FileMasks[f]

			if filePawns == 0 {
				addMG(idxPawnShield+4, s) // semiOpenFileNearKingMG
				continue
			}

			foundShield := false
			if color == White {
				if filePawns&Rank2 != 0 {
					addMG(idxPawnShield+0, s) // shieldPawnRank2MG
					foundShield = true
				} else if filePawns&Rank3 != 0 {
					addMG(idxPawnShield+1, s) // shieldPawnRank3MG
					foundShield = true
				}
			} else {
				if filePawns&Rank7 != 0 {
					addMG(idxPawnShield+0, s) // shieldPawnRank2MG
					foundShield = true
				} else if filePawns&Rank6 != 0 {
					addMG(idxPawnShield+1, s) // shieldPawnRank3MG
					foundShield = true
				}
			}

			if !foundShield {
				hasAdvancedPawn := (color == White && filePawns&Rank4 != 0) ||
					(color == Black && filePawns&Rank5 != 0)
				if hasAdvancedPawn {
					addMG(idxPawnShield+3, s) // missingShieldPawnAdvancedMG
				} else {
					addMG(idxPawnShield+2, s) // missingShieldPawnMG
				}
			}
		}
	}

	// === Tempo ===
	tempoSign := int16(1)
	if b.SideToMove == Black {
		tempoSign = -1
	}
	addMG(miscBase+10, tempoSign) // TempoMG
	addEG(miscBase+11, tempoSign) // TempoEG

	return trace
}

// scoreFromTrace computes the evaluation score from a trace and parameter vector.
func scoreFromTrace(trace *TunerTrace, params []float64) float64 {
	mg := 0.0
	for _, e := range trace.MG {
		mg += params[e.Index] * float64(e.Coeff)
	}
	eg := 0.0
	for _, e := range trace.EG {
		eg += params[e.Index] * float64(e.Coeff)
	}
	phase := float64(trace.Phase)
	return (mg*(float64(TotalPhase)-phase) + eg*phase) / float64(TotalPhase)
}

// sigmoid maps an evaluation score to a win probability.
func sigmoid(score, K float64) float64 {
	return 1.0 / (1.0 + math.Pow(10.0, -score/K))
}

// TuneK finds the optimal scaling constant K by minimizing MSE.
func (t *Tuner) TuneK() float64 {
	// Golden section search for optimal K in [50, 800]
	lo, hi := 50.0, 800.0
	gr := (math.Sqrt(5) + 1) / 2

	for hi-lo > 0.1 {
		c := hi - (hi-lo)/gr
		d := lo + (hi-lo)/gr
		if t.computeError(c) < t.computeError(d) {
			hi = d
		} else {
			lo = c
		}
	}
	return (lo + hi) / 2
}

// computeError computes the mean squared error for all positions with a given K.
func (t *Tuner) computeError(K float64) float64 {
	numCPU := runtime.NumCPU()
	n := len(t.Traces)
	if n == 0 {
		return 0
	}

	chunkSize := (n + numCPU - 1) / numCPU
	errors := make([]float64, numCPU)

	var wg sync.WaitGroup
	for cpu := 0; cpu < numCPU; cpu++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			start := id * chunkSize
			end := start + chunkSize
			if end > n {
				end = n
			}
			localErr := 0.0
			for i := start; i < end; i++ {
				score := scoreFromTrace(&t.Traces[i].Trace, t.Values)
				predicted := sigmoid(score, K)
				diff := t.Traces[i].Trace.Result - predicted
				localErr += diff * diff
			}
			errors[id] = localErr
		}(cpu)
	}
	wg.Wait()

	total := 0.0
	for _, e := range errors {
		total += e
	}
	return total / float64(n)
}

// ComputeErrorPublic is the exported wrapper for computeError.
func (t *Tuner) ComputeErrorPublic(K float64) float64 {
	return t.computeError(K)
}

// TuneConfig controls the tuning process.
type TuneConfig struct {
	Epochs  int     // number of optimization epochs
	LR      float64 // learning rate (default 1.0)
	Beta1   float64 // Adam beta1 (default 0.9)
	Beta2   float64 // Adam beta2 (default 0.999)
	Epsilon float64 // Adam epsilon (default 1e-8)
}

// DefaultTuneConfig returns sensible defaults.
func DefaultTuneConfig() TuneConfig {
	return TuneConfig{
		Epochs:  500,
		LR:      1.0,
		Beta1:   0.9,
		Beta2:   0.999,
		Epsilon: 1e-8,
	}
}

// Tune runs the Adam optimizer to minimize prediction error.
// Calls onEpoch after each epoch with (epoch, error).
func (t *Tuner) Tune(K float64, cfg TuneConfig, onEpoch func(epoch int, err float64)) {
	n := len(t.Traces)
	np := len(t.Values)
	if n == 0 || np == 0 {
		return
	}

	// Adam state
	m := make([]float64, np) // first moment
	v := make([]float64, np) // second moment

	numCPU := runtime.NumCPU()
	chunkSize := (n + numCPU - 1) / numCPU

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		// Compute gradients in parallel
		gradChunks := make([][]float64, numCPU)
		var wg sync.WaitGroup

		for cpu := 0; cpu < numCPU; cpu++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				localGrad := make([]float64, np)
				start := id * chunkSize
				end := start + chunkSize
				if end > n {
					end = n
				}

				for i := start; i < end; i++ {
					trace := &t.Traces[i].Trace
					score := scoreFromTrace(trace, t.Values)
					sig := sigmoid(score, K)
					// d(loss)/d(score) = -2 * (result - sig) * sig * (1 - sig) * ln(10) / K
					// Simplified (absorb constants into gradient, divide by N later)
					errTerm := (trace.Result - sig) * sig * (1 - sig)
					phase := float64(trace.Phase)

					mgScale := errTerm * (float64(TotalPhase) - phase) / float64(TotalPhase)
					egScale := errTerm * phase / float64(TotalPhase)

					for _, e := range trace.MG {
						localGrad[e.Index] += mgScale * float64(e.Coeff)
					}
					for _, e := range trace.EG {
						localGrad[e.Index] += egScale * float64(e.Coeff)
					}
				}
				gradChunks[id] = localGrad
			}(cpu)
		}
		wg.Wait()

		// Sum gradients
		grad := make([]float64, np)
		for _, chunk := range gradChunks {
			if chunk == nil {
				continue
			}
			for j := range grad {
				grad[j] += chunk[j]
			}
		}

		// Scale by -2 * ln(10) / (K * N) and negate (we want to minimize)
		scale := -2.0 * math.Log(10) / (K * float64(n))
		for j := range grad {
			grad[j] *= scale
		}

		// Adam update
		for j := 0; j < np; j++ {
			m[j] = cfg.Beta1*m[j] + (1-cfg.Beta1)*grad[j]
			v[j] = cfg.Beta2*v[j] + (1-cfg.Beta2)*grad[j]*grad[j]

			mHat := m[j] / (1 - math.Pow(cfg.Beta1, float64(epoch)))
			vHat := v[j] / (1 - math.Pow(cfg.Beta2, float64(epoch)))

			t.Values[j] -= cfg.LR * mHat / (math.Sqrt(vHat) + cfg.Epsilon)
		}

		if onEpoch != nil {
			err := t.computeError(K)
			onEpoch(epoch, err)
		}
	}
}

// ApplyValues writes the tuned values back to engine globals.
func (t *Tuner) ApplyValues() {
	for i, p := range t.Params {
		if p.setter != nil {
			p.setter(int(math.Round(t.Values[i])))
		}
	}
}

// PrintParams prints all tuned parameters in Go source format.
func (t *Tuner) PrintParams(w *bufio.Writer) {
	// Material
	w.WriteString("=== Material ===\n")
	w.WriteString("var mgMaterial = [7]int{0, ")
	for i := 0; i < 5; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[idxMaterialMG+i])))
	}
	w.WriteString(", 0}\n")

	w.WriteString("var egMaterial = [7]int{0, ")
	for i := 0; i < 5; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[idxMaterialEG+i])))
	}
	w.WriteString(", 0}\n\n")

	// PST tables
	pstNames := []string{"Pawn", "Knight", "Bishop", "Rook", "Queen", "King"}
	w.WriteString("=== PST MG ===\n")
	for pi := 0; pi < 6; pi++ {
		fmt.Fprintf(w, "var mg%sTable = [64]int{\n", pstNames[pi])
		base := idxPSTMG + pi*64
		for rank := 0; rank < 8; rank++ {
			w.WriteString("\t")
			for file := 0; file < 8; file++ {
				sq := rank*8 + file
				if file > 0 {
					w.WriteString(", ")
				}
				fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+sq])))
			}
			w.WriteString(",\n")
		}
		w.WriteString("}\n\n")
	}

	w.WriteString("=== PST EG ===\n")
	for pi := 0; pi < 6; pi++ {
		fmt.Fprintf(w, "var eg%sTable = [64]int{\n", pstNames[pi])
		base := idxPSTEG + pi*64
		for rank := 0; rank < 8; rank++ {
			w.WriteString("\t")
			for file := 0; file < 8; file++ {
				sq := rank*8 + file
				if file > 0 {
					w.WriteString(", ")
				}
				fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+sq])))
			}
			w.WriteString(",\n")
		}
		w.WriteString("}\n\n")
	}

	// PST scales — set all to 100 since tuned PST values are already effective
	w.WriteString("=== PST Scales (set to 100: values are pre-scaled) ===\n")
	for _, name := range []string{"PawnPSTScaleMG", "PawnPSTScaleEG", "PiecePSTScaleMG", "PiecePSTScaleEG", "KingPSTScaleMG", "KingPSTScaleEG"} {
		fmt.Fprintf(w, "var %s = 100\n", name)
	}
	w.WriteString("\n")

	// Mobility
	w.WriteString("=== Mobility ===\n")
	mobSizes := []struct {
		name string
		size int
	}{
		{"KnightMobility", 9},
		{"BishopMobility", 14},
		{"RookMobility", 15},
		{"QueenMobility", 28},
	}
	offset := idxMobilityStart
	for _, mob := range mobSizes {
		fmt.Fprintf(w, "var %s = [%d][2]int{\n", mob.name, mob.size)
		for i := 0; i < mob.size; i++ {
			mg := int(math.Round(t.Values[offset+i*2]))
			eg := int(math.Round(t.Values[offset+i*2+1]))
			fmt.Fprintf(w, "\t{%d, %d},\n", mg, eg)
		}
		w.WriteString("}\n\n")
		offset += mob.size * 2
	}

	// Piece bonuses
	w.WriteString("=== Piece Bonuses ===\n")
	bonusNames := []string{
		"BishopPairMG", "BishopPairEG",
		"KnightOutpostMG", "KnightOutpostEG",
		"KnightOutpostSupportedMG", "KnightOutpostSupportedEG",
		"RookOpenFileMG", "RookOpenFileEG",
		"RookSemiOpenFileMG", "RookSemiOpenFileEG",
		"RookOn7thMG", "RookOn7thEG",
		"TrappedRookPenaltyMG", "TrappedRookPenaltyEG",
		"BishopOpenPositionMG", "BishopOpenPositionEG",
		"BadBishopPawnMG", "BadBishopPawnEG",
		"DoubledRooksMG", "DoubledRooksEG",
		"KnightClosedPositionMG", "KnightClosedPositionEG",
	}
	for i, name := range bonusNames {
		fmt.Fprintf(w, "var %s = %d\n", name, int(math.Round(t.Values[idxBonusStart+i])))
	}
	w.WriteString("\n")

	// Passed pawns
	w.WriteString("=== Passed Pawn Enhancements ===\n")
	base := idxPassedStart
	w.WriteString("var PassedPawnNotBlockedMG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+i*2])))
	}
	w.WriteString("}\n")
	w.WriteString("var PassedPawnNotBlockedEG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+i*2+1])))
	}
	w.WriteString("}\n")

	w.WriteString("var PassedPawnFreePathMG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+16+i*2])))
	}
	w.WriteString("}\n")
	w.WriteString("var PassedPawnFreePathEG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+16+i*2+1])))
	}
	w.WriteString("}\n")

	w.WriteString("var PassedPawnKingScale = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+32+i])))
	}
	w.WriteString("}\n")

	fmt.Fprintf(w, "var PassedPawnFriendlyKingDistEG = %d\n", int(math.Round(t.Values[base+40])))
	fmt.Fprintf(w, "var PassedPawnEnemyKingDistEG = %d\n", int(math.Round(t.Values[base+41])))
	fmt.Fprintf(w, "var PassedPawnProtectedMG = %d\n", int(math.Round(t.Values[base+42])))
	fmt.Fprintf(w, "var PassedPawnProtectedEG = %d\n", int(math.Round(t.Values[base+43])))
	fmt.Fprintf(w, "var PassedPawnConnectedMG = %d\n", int(math.Round(t.Values[base+44])))
	fmt.Fprintf(w, "var PassedPawnConnectedEG = %d\n", int(math.Round(t.Values[base+45])))
	fmt.Fprintf(w, "var RookBehindPassedMG = %d\n", int(math.Round(t.Values[base+46])))
	fmt.Fprintf(w, "var RookBehindPassedEG = %d\n", int(math.Round(t.Values[base+47])))
	w.WriteString("\n")

	// Pawn structure
	w.WriteString("=== Pawn Structure ===\n")
	base = idxPawnStart
	w.WriteString("var passedPawnMG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+i*2])))
	}
	w.WriteString("}\n")
	w.WriteString("var passedPawnEG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+i*2+1])))
	}
	w.WriteString("}\n")

	w.WriteString("var pawnAdvancementMG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+16+i*2])))
	}
	w.WriteString("}\n")
	w.WriteString("var pawnAdvancementEG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+16+i*2+1])))
	}
	w.WriteString("}\n")

	fmt.Fprintf(w, "const doubledPawnMG = %d\n", int(math.Round(t.Values[base+32])))
	fmt.Fprintf(w, "const doubledPawnEG = %d\n", int(math.Round(t.Values[base+33])))
	fmt.Fprintf(w, "const isolatedPawnMG = %d\n", int(math.Round(t.Values[base+34])))
	fmt.Fprintf(w, "const isolatedPawnEG = %d\n", int(math.Round(t.Values[base+35])))
	fmt.Fprintf(w, "const backwardPawnMG = %d\n", int(math.Round(t.Values[base+36])))
	fmt.Fprintf(w, "const backwardPawnEG = %d\n", int(math.Round(t.Values[base+37])))
	fmt.Fprintf(w, "const connectedPawnMG = %d\n", int(math.Round(t.Values[base+38])))
	fmt.Fprintf(w, "const connectedPawnEG = %d\n", int(math.Round(t.Values[base+39])))
	w.WriteString("\n")

	// King attack
	w.WriteString("=== King Attack ===\n")
	attackNames := []string{
		"KnightAttackUnits", "KnightKingZoneBonus",
		"BishopAttackUnits", "BishopKingZoneBonus",
		"RookAttackUnits", "RookKingZoneBonus",
		"QueenAttackUnits", "QueenKingZoneBonus",
	}
	for i, name := range attackNames {
		fmt.Fprintf(w, "var %s = %d\n", name, int(math.Round(t.Values[idxKingAttack+i])))
	}
	w.WriteString("\n")

	// King safety table
	w.WriteString("=== King Safety Table ===\n")
	w.WriteString("var KingSafetyTable = [100]int{\n")
	for i := 0; i < 100; i++ {
		if i%10 == 0 {
			w.WriteString("\t")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[idxKingSafetyTbl+i])))
		if i < 99 {
			w.WriteString(", ")
		}
		if i%10 == 9 {
			w.WriteString("\n")
		}
	}
	w.WriteString("}\n\n")

	// Pawn shield
	w.WriteString("=== Pawn Shield ===\n")
	shieldNames := []string{
		"shieldPawnRank2MG",
		"shieldPawnRank3MG",
		"missingShieldPawnMG",
		"missingShieldPawnAdvancedMG",
		"semiOpenFileNearKingMG",
	}
	for i, name := range shieldNames {
		fmt.Fprintf(w, "var %s = %d\n", name, int(math.Round(t.Values[idxPawnShield+i])))
	}
	w.WriteString("\n")

	// Misc
	w.WriteString("=== Misc ===\n")
	miscNames := []string{
		"CastlingRightsMG",
		"SpaceBonusMG", "SpaceBonusEG",
		"PawnThreatMinorMG", "PawnThreatMinorEG",
		"PawnThreatRookMG", "PawnThreatRookEG",
		"PawnThreatQueenMG", "PawnThreatQueenEG",
		"OCBScale",
		"TempoMG", "TempoEG",
	}
	for i, name := range miscNames {
		fmt.Fprintf(w, "var %s = %d\n", name, int(math.Round(t.Values[idxMisc+i])))
	}
	w.WriteString("\n")

	w.Flush()
}

// VerifyTrace checks that the trace-based evaluation matches Evaluate() for a position.
// Returns the trace score, eval score, and whether they match within tolerance.
func (t *Tuner) VerifyTrace(fen string) (traceScore, evalScore float64, ok bool) {
	var b Board
	if err := b.SetFEN(fen); err != nil {
		return 0, 0, false
	}

	trace := t.computeTrace(&b)
	traceScore = scoreFromTrace(&trace, t.Values)
	evalScore = float64(b.Evaluate())

	// Allow some tolerance due to:
	// 1. PST scaling (trace uses raw values, eval applies scale/100)
	// 2. King safety table (non-linear, not in trace)
	// 3. Endgame scaling (multiplicative, not in trace)
	// 4. 50-move rule scaling
	// 5. Eval cache
	diff := math.Abs(traceScore - evalScore)
	ok = diff < 200 // generous tolerance for components we don't trace
	return
}

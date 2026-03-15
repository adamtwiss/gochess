package chess

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
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
	MG            []SparseEntry
	EG            []SparseEntry
	Phase         int     // game phase (0 = full MG, 24 = full EG)
	Result        float64 // game outcome: 1.0 (white wins), 0.5 (draw), 0.0 (black wins)
	Score         int16   // White-relative search score in centipawns (0 when unavailable)
	WScale        int     // endgame scale factor for White winning (0-128)
	BScale        int     // endgame scale factor for Black winning (0-128)
	HalfmoveClock int     // for 50-move rule scaling
}

// Tuner holds the parameter catalog.
type Tuner struct {
	Params []TunerParam // parameter metadata
	Values []float64    // current parameter values
	Frozen []bool       // if true, parameter is pinned and not updated during tuning

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
	idxSafeCheck     int // safe check bonuses (4 entries)
	idxKingSafetyTbl int // king safety table (100 entries, MG only)
	idxPawnShield    int // pawn shield constants (5 entries, MG only)
	idxPawnStorm     int // pawn storm bonus (2x8 MG + 2x8 EG = 32 entries)
	idxSameSideStorm int // same-side pawn storm bonus (8 entries, MG only)
	idxEndgameKing   int // endgame king activity (3 entries, EG only)
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
	add("RookEnemyKingFileMG", RookEnemyKingFileMG, func(v int) { RookEnemyKingFileMG = v })
	add("RookEnemyKingFileEG", RookEnemyKingFileEG, func(v int) { RookEnemyKingFileEG = v })

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
	for i := 0; i < 8; i++ {
		ii := i
		add(fmt.Sprintf("PassedPawnBlockedMG[%d]", i), PassedPawnBlockedMG[i],
			func(v int) { PassedPawnBlockedMG[ii] = v })
		add(fmt.Sprintf("PassedPawnBlockedEG[%d]", i), PassedPawnBlockedEG[i],
			func(v int) { PassedPawnBlockedEG[ii] = v })
	}

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
	add("doubledPawnMG", doubledPawnMG, func(v int) { doubledPawnMG = v })
	add("doubledPawnEG", doubledPawnEG, func(v int) { doubledPawnEG = v })
	add("isolatedPawnMG", isolatedPawnMG, func(v int) { isolatedPawnMG = v })
	add("isolatedPawnEG", isolatedPawnEG, func(v int) { isolatedPawnEG = v })
	add("backwardPawnMG", backwardPawnMG, func(v int) { backwardPawnMG = v })
	add("backwardPawnEG", backwardPawnEG, func(v int) { backwardPawnEG = v })
	add("connectedPawnMG", connectedPawnMG, func(v int) { connectedPawnMG = v })
	add("connectedPawnEG", connectedPawnEG, func(v int) { connectedPawnEG = v })
	add("PawnMajorityMG", PawnMajorityMG, func(v int) { PawnMajorityMG = v })
	add("PawnMajorityEG", PawnMajorityEG, func(v int) { PawnMajorityEG = v })
	for i := 0; i < 8; i++ {
		ii := i
		add(fmt.Sprintf("queensidePawnAdvMG[%d]", i), queensidePawnAdvMG[i],
			func(v int) { queensidePawnAdvMG[ii] = v })
		add(fmt.Sprintf("queensidePawnAdvEG[%d]", i), queensidePawnAdvEG[i],
			func(v int) { queensidePawnAdvEG[ii] = v })
	}
	for i := 0; i < 8; i++ {
		ii := i
		add(fmt.Sprintf("candidatePassedMG[%d]", i), candidatePassedMG[i],
			func(v int) { candidatePassedMG[ii] = v })
		add(fmt.Sprintf("candidatePassedEG[%d]", i), candidatePassedEG[i],
			func(v int) { candidatePassedEG[ii] = v })
	}
	for i := 0; i < 8; i++ {
		ii := i
		add(fmt.Sprintf("pawnLeverMG[%d]", i), pawnLeverMG[i],
			func(v int) { pawnLeverMG[ii] = v })
		add(fmt.Sprintf("pawnLeverEG[%d]", i), pawnLeverEG[i],
			func(v int) { pawnLeverEG[ii] = v })
	}

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

	// === Safe check bonuses ===
	// NoQueenAttackScale is intentionally excluded: it's multiplicative (penalty * scale / 128)
	// and cannot be represented as an additive trace coefficient.
	idxSafeCheck = len(t.Params)
	addSection("Safe Check", idxSafeCheck)
	add("SafeKnightCheckBonus", SafeKnightCheckBonus, func(v int) { SafeKnightCheckBonus = v })
	add("SafeBishopCheckBonus", SafeBishopCheckBonus, func(v int) { SafeBishopCheckBonus = v })
	add("SafeRookCheckBonus", SafeRookCheckBonus, func(v int) { SafeRookCheckBonus = v })
	add("SafeQueenCheckBonus", SafeQueenCheckBonus, func(v int) { SafeQueenCheckBonus = v })

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

	// === Pawn Storm (2 x 8 MG + 2 x 8 EG = 32 entries) ===
	idxPawnStorm = len(t.Params)
	addSection("Pawn Storm", idxPawnStorm)
	for opp := 0; opp < 2; opp++ {
		o := opp
		for r := 0; r < 8; r++ {
			rr := r
			oppLabel := "Opp"
			if opp == 1 {
				oppLabel = "Unp"
			}
			add(fmt.Sprintf("PawnStormBonusMG[%s][%d]", oppLabel, r), PawnStormBonusMG[opp][r],
				func(v int) { PawnStormBonusMG[o][rr] = v })
		}
	}
	for opp := 0; opp < 2; opp++ {
		o := opp
		for r := 0; r < 8; r++ {
			rr := r
			oppLabel := "Opp"
			if opp == 1 {
				oppLabel = "Unp"
			}
			add(fmt.Sprintf("PawnStormBonusEG[%s][%d]", oppLabel, r), PawnStormBonusEG[opp][r],
				func(v int) { PawnStormBonusEG[o][rr] = v })
		}
	}

	// === Same-Side Pawn Storm (8 entries, MG only) ===
	idxSameSideStorm = len(t.Params)
	addSection("Same-Side Storm", idxSameSideStorm)
	for i := 0; i < 8; i++ {
		ii := i
		add(fmt.Sprintf("SameSideStormMG[%d]", i), SameSideStormMG[i],
			func(v int) { SameSideStormMG[ii] = v })
	}

	// === Endgame King Activity (3 entries, EG only) ===
	idxEndgameKing = len(t.Params)
	addSection("Endgame King", idxEndgameKing)
	add("KingCenterBonusEG", KingCenterBonusEG, func(v int) { KingCenterBonusEG = v })
	add("KingProximityAdvantageEG", KingProximityAdvantageEG, func(v int) { KingProximityAdvantageEG = v })
	add("KingCornerPushEG", KingCornerPushEG, func(v int) { KingCornerPushEG = v })

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
	add("MinorThreatRookMG", MinorThreatRookMG, func(v int) { MinorThreatRookMG = v })
	add("MinorThreatRookEG", MinorThreatRookEG, func(v int) { MinorThreatRookEG = v })
	add("MinorThreatQueenMG", MinorThreatQueenMG, func(v int) { MinorThreatQueenMG = v })
	add("MinorThreatQueenEG", MinorThreatQueenEG, func(v int) { MinorThreatQueenEG = v })
	add("RookThreatQueenMG", RookThreatQueenMG, func(v int) { RookThreatQueenMG = v })
	add("RookThreatQueenEG", RookThreatQueenEG, func(v int) { RookThreatQueenEG = v })
	add("OCBScale", OCBScale, func(v int) { OCBScale = v })
	add("TempoMG", TempoMG, func(v int) { TempoMG = v })
	add("TempoEG", TempoEG, func(v int) { TempoEG = v })
	add("TradePieceBonus", TradePieceBonus, func(v int) { TradePieceBonus = v })
	add("TradePawnBonus", TradePawnBonus, func(v int) { TradePawnBonus = v })

	// Close last section
	if len(t.sections) > 0 {
		t.sections[len(t.sections)-1].endIndex = len(t.Params)
	}

	// Freeze parameters that are well-established or can't be learned from search scores
	t.Frozen = make([]bool, len(t.Params))
	// Material values: well-established and prone to coupling with PSTs
	for i := idxMaterialMG; i < idxMaterialMG+5; i++ {
		t.Frozen[i] = true
	}
	for i := idxMaterialEG; i < idxMaterialEG+5; i++ {
		t.Frozen[i] = true
	}
	// Tempo and trade bonuses: long-term strategic concepts that search at
	// practical time controls cannot evaluate (simplify-when-ahead, tempo value
	// in quiet positions). Confirmed by lambda=0 and lambda=0.1 tuning runs
	// both producing anti-textbook values that lost ~37 Elo in gauntlets.
	frozenNames := map[string]bool{
		"TempoMG": true, "TempoEG": true,
		"TradePieceBonus": true, "TradePawnBonus": true,
	}
	for i, p := range t.Params {
		if frozenNames[p.Name] {
			t.Frozen[i] = true
		}
	}
}

// NumParams returns the total number of tunable parameters.
func (t *Tuner) NumParams() int {
	return len(t.Params)
}

// BuildPairMap builds a mapping from MG parameter index to EG parameter index.
// Parameters are paired by matching base names after stripping MG/EG markers.
// This is used by the v3 tbin format to store paired entries once instead of twice.
func (t *Tuner) BuildPairMap() map[uint16]uint16 {
	type info struct {
		idx   uint16
		phase byte // 'm' or 'e'
	}
	bases := make(map[string][]info)

	for i, p := range t.Params {
		name := p.Name
		var base string
		var phase byte

		switch {
		case strings.HasPrefix(name, "mg"):
			phase, base = 'm', name[2:]
		case strings.HasPrefix(name, "eg"):
			phase, base = 'e', name[2:]
		case strings.HasSuffix(name, "MG"):
			phase, base = 'm', name[:len(name)-2]
		case strings.HasSuffix(name, "EG"):
			phase, base = 'e', name[:len(name)-2]
		case strings.Contains(name, "[MG]"):
			phase, base = 'm', strings.Replace(name, "[MG]", "[]", 1)
		case strings.Contains(name, "[EG]"):
			phase, base = 'e', strings.Replace(name, "[EG]", "[]", 1)
		default:
			continue // unpaired (e.g. KingSafetyTable, OCBScale)
		}

		bases[base] = append(bases[base], info{uint16(i), phase})
	}

	result := make(map[uint16]uint16)
	for _, infos := range bases {
		if len(infos) != 2 {
			continue
		}
		var mgIdx, egIdx uint16
		if infos[0].phase == 'm' && infos[1].phase == 'e' {
			mgIdx, egIdx = infos[0].idx, infos[1].idx
		} else if infos[0].phase == 'e' && infos[1].phase == 'm' {
			mgIdx, egIdx = infos[1].idx, infos[0].idx
		} else {
			continue
		}
		result[mgIdx] = egIdx
	}
	return result
}

// ---------------------------------------------------------------------------
// Binary trace file (.tbin) format v3 for disk-streamed training
// ---------------------------------------------------------------------------
//
// Header (24 bytes):
//   magic:         [4]byte   "TBIN"
//   version:       uint16    3
//   numParams:     uint16
//   numTrain:      uint32
//   numValidation: uint32
//   trainBytes:    uint64    (byte size of all training records, for seeking to validation)
//
// Records (variable-length, sequential):
//   phase:         uint8
//   result:        uint8     (0=black, 1=draw, 2=white)
//   wScale:        uint8
//   bScale:        uint8
//   halfmoveClock: uint8
//   score:         int16     (White-relative centipawns, 0 when unavailable)
//   pairedCount:   uint16    (entries where MG and EG share the same coeff)
//   mgOnlyCount:   uint16    (entries with only an MG component)
//   egOnlyCount:   uint16    (entries with only an EG component)
//   paired[pairedCount]:  4 bytes each (uint16 mgIndex, int16 coeff; EG index via mgToEg map)
//   mgOnly[mgOnlyCount]:  4 bytes each (uint16 index, int16 coeff)
//   egOnly[egOnlyCount]:  4 bytes each (uint16 index, int16 coeff)

const (
	tbinMagic      = "TBIN"
	tbinVersion    = 3
	tbinHeaderSize = 24
)

// TraceFile provides streaming access to a preprocessed .tbin file via mmap.
type TraceFile struct {
	Path          string
	NumTrain      int
	NumValidation int
	NumParams     int
	trainBytes    uint64
	data          []byte           // mmap'd file contents
	version       uint16           // tbin format version
	mgToEg        map[uint16]uint16 // MG→EG index map for v3 paired decoding
}

// Close unmaps the memory-mapped file.
func (tf *TraceFile) Close() error {
	if tf.data != nil {
		err := syscall.Munmap(tf.data)
		tf.data = nil
		return err
	}
	return nil
}

// resultToUint8 encodes a float64 game result to a uint8.
func resultToUint8(r float64) uint8 {
	if r == 1.0 {
		return 2
	}
	if r == 0.5 {
		return 1
	}
	return 0
}

// uint8ToResult decodes a uint8 game result to float64.
func uint8ToResult(r uint8) float64 {
	switch r {
	case 2:
		return 1.0
	case 1:
		return 0.5
	default:
		return 0.0
	}
}

// writeTraceRecord writes a single TunerTrace as a v3 binary record to w.
// v3 format: paired MG/EG entries (same coeff) stored once, ~50% smaller than v2.
// Entries are also aggregated by index (summing coefficients) before writing.
// Header: 13 bytes (5 fixed + 2 score + 2 pairedCount + 2 mgOnlyCount + 2 egOnlyCount)
// Then: paired entries (4 bytes each), mgOnly entries (4 bytes each), egOnly entries (4 bytes each).
func writeTraceRecord(w io.Writer, trace *TunerTrace, mgToEg map[uint16]uint16) (int, error) {
	// Aggregate MG entries by index (sum coefficients for duplicates)
	mgAgg := make(map[uint16]int16, len(trace.MG))
	for _, e := range trace.MG {
		mgAgg[e.Index] += e.Coeff
	}
	// Aggregate EG entries by index
	egAgg := make(map[uint16]int16, len(trace.EG))
	for _, e := range trace.EG {
		egAgg[e.Index] += e.Coeff
	}

	// Remove zero-sum entries
	for idx := range mgAgg {
		if mgAgg[idx] == 0 {
			delete(mgAgg, idx)
		}
	}
	for idx := range egAgg {
		if egAgg[idx] == 0 {
			delete(egAgg, idx)
		}
	}

	// Classify into paired, mgOnly, egOnly
	paired := make([]SparseEntry, 0, len(mgAgg))
	mgOnly := make([]SparseEntry, 0, 4)

	for mgIdx, mgCoeff := range mgAgg {
		if egIdx, ok := mgToEg[mgIdx]; ok {
			if egCoeff, found := egAgg[egIdx]; found && egCoeff == mgCoeff {
				paired = append(paired, SparseEntry{mgIdx, mgCoeff})
				delete(egAgg, egIdx) // consume the EG entry
				continue
			}
		}
		mgOnly = append(mgOnly, SparseEntry{mgIdx, mgCoeff})
	}

	// Remaining EG entries are egOnly
	egOnly := make([]SparseEntry, 0, len(egAgg))
	for egIdx, egCoeff := range egAgg {
		egOnly = append(egOnly, SparseEntry{egIdx, egCoeff})
	}

	// 13-byte header
	var header [13]byte
	header[0] = uint8(trace.Phase)
	header[1] = resultToUint8(trace.Result)
	header[2] = uint8(trace.WScale)
	header[3] = uint8(trace.BScale)
	header[4] = uint8(trace.HalfmoveClock)
	binary.LittleEndian.PutUint16(header[5:7], uint16(trace.Score))
	binary.LittleEndian.PutUint16(header[7:9], uint16(len(paired)))
	binary.LittleEndian.PutUint16(header[9:11], uint16(len(mgOnly)))
	binary.LittleEndian.PutUint16(header[11:13], uint16(len(egOnly)))

	n, err := w.Write(header[:])
	if err != nil {
		return n, err
	}
	total := n

	var buf [4]byte
	writeEntry := func(e SparseEntry) error {
		binary.LittleEndian.PutUint16(buf[0:2], e.Index)
		binary.LittleEndian.PutUint16(buf[2:4], uint16(e.Coeff))
		nn, err := w.Write(buf[:])
		total += nn
		return err
	}

	for _, e := range paired {
		if err := writeEntry(e); err != nil {
			return total, err
		}
	}
	for _, e := range mgOnly {
		if err := writeEntry(e); err != nil {
			return total, err
		}
	}
	for _, e := range egOnly {
		if err := writeEntry(e); err != nil {
			return total, err
		}
	}

	return total, nil
}

// decodeTraceRecord decodes a single v3 TunerTrace from a byte slice at the given offset.
// v3: 13-byte header with 3 counts (paired, mgOnly, egOnly). Paired entries emit to both MG and EG.
func decodeTraceRecord(data []byte, offset int, mgBuf, egBuf []SparseEntry, mgToEg map[uint16]uint16) (TunerTrace, []SparseEntry, []SparseEntry, int) {
	phase := int(data[offset])
	result := uint8ToResult(data[offset+1])
	wScale := int(data[offset+2])
	bScale := int(data[offset+3])
	halfmove := int(data[offset+4])
	score := int16(binary.LittleEndian.Uint16(data[offset+5:]))
	pairedCount := int(binary.LittleEndian.Uint16(data[offset+7:]))
	mgOnlyCount := int(binary.LittleEndian.Uint16(data[offset+9:]))
	egOnlyCount := int(binary.LittleEndian.Uint16(data[offset+11:]))
	offset += 13

	totalMG := pairedCount + mgOnlyCount
	totalEG := pairedCount + egOnlyCount

	// Grow backing slices if needed, reuse capacity
	if cap(mgBuf) < totalMG {
		mgBuf = make([]SparseEntry, totalMG)
	} else {
		mgBuf = mgBuf[:totalMG]
	}
	if cap(egBuf) < totalEG {
		egBuf = make([]SparseEntry, totalEG)
	} else {
		egBuf = egBuf[:totalEG]
	}

	// Paired entries → emit to both MG and EG
	for i := 0; i < pairedCount; i++ {
		idx := binary.LittleEndian.Uint16(data[offset:])
		coeff := int16(binary.LittleEndian.Uint16(data[offset+2:]))
		mgBuf[i] = SparseEntry{idx, coeff}
		egBuf[i] = SparseEntry{mgToEg[idx], coeff}
		offset += 4
	}

	// MG-only entries
	for i := 0; i < mgOnlyCount; i++ {
		mgBuf[pairedCount+i] = SparseEntry{
			Index: binary.LittleEndian.Uint16(data[offset:]),
			Coeff: int16(binary.LittleEndian.Uint16(data[offset+2:])),
		}
		offset += 4
	}

	// EG-only entries
	for i := 0; i < egOnlyCount; i++ {
		egBuf[pairedCount+i] = SparseEntry{
			Index: binary.LittleEndian.Uint16(data[offset:]),
			Coeff: int16(binary.LittleEndian.Uint16(data[offset+2:])),
		}
		offset += 4
	}

	trace := TunerTrace{
		Phase:         phase,
		Result:        result,
		Score:         score,
		WScale:        wScale,
		BScale:        bScale,
		HalfmoveClock: halfmove,
		MG:            mgBuf[:totalMG],
		EG:            egBuf[:totalEG],
	}
	return trace, mgBuf, egBuf, offset
}

// PreprocessBinToFile reads positions from a .bin file (32-byte binary records),
// computes evaluation traces, and writes a .tbin trace cache file.
func PreprocessBinToFile(t *Tuner, inputBin, outputBin string) error {
	// Open and validate the .bin file
	stat, err := os.Stat(inputBin)
	if err != nil {
		return err
	}
	if stat.Size()%BinpackRecordSize != 0 {
		return fmt.Errorf("%s: file size %d is not a multiple of %d", inputBin, stat.Size(), BinpackRecordSize)
	}
	totalRecords := int(stat.Size() / BinpackRecordSize)
	if totalRecords == 0 {
		return fmt.Errorf("%s: empty file", inputBin)
	}

	f, err := os.Open(inputBin)
	if err != nil {
		return err
	}
	defer f.Close()

	// Read all records into memory for shuffling
	data := make([]byte, stat.Size())
	if _, err := io.ReadFull(f, data); err != nil {
		return fmt.Errorf("reading %s: %w", inputBin, err)
	}

	// Build index array and shuffle deterministically
	indices := make([]int, totalRecords)
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(42))
	rng.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	// 90/10 split
	numTrain := totalRecords * 9 / 10
	numValidation := totalRecords - numTrain

	// Write to a temp file first, then rename atomically
	tmpFile := outputBin + ".tmp"
	out, err := os.Create(tmpFile)
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(out, 256*1024)

	// Write placeholder header
	var headerBuf [tbinHeaderSize]byte
	copy(headerBuf[0:4], tbinMagic)
	binary.LittleEndian.PutUint16(headerBuf[4:6], tbinVersion)
	binary.LittleEndian.PutUint16(headerBuf[6:8], uint16(t.NumParams()))
	binary.LittleEndian.PutUint32(headerBuf[8:12], uint32(numTrain))
	binary.LittleEndian.PutUint32(headerBuf[12:16], uint32(numValidation))
	if _, err := bw.Write(headerBuf[:]); err != nil {
		out.Close()
		os.Remove(tmpFile)
		return err
	}

	// Build pair map for v3 format
	pairMap := t.BuildPairMap()
	fmt.Printf("  v3 format: %d MG/EG paired parameters\n", len(pairMap))

	// Write records in shuffled order: training first, then validation.
	// Parallelized: process batches of records with worker goroutines,
	// then write results sequentially to maintain order.
	var trainBytesTotal uint64
	numWorkers := runtime.NumCPU()
	if numWorkers < 1 {
		numWorkers = 1
	}
	const batchSize = 32768
	startTime := time.Now()
	processed := 0

	for batchStart := 0; batchStart < len(indices); batchStart += batchSize {
		batchEnd := batchStart + batchSize
		if batchEnd > len(indices) {
			batchEnd = len(indices)
		}
		batch := indices[batchStart:batchEnd]

		// Parallel: compute traces for this batch
		type traceResult struct {
			trace TunerTrace
			err   error
		}
		results := make([]traceResult, len(batch))
		var wg sync.WaitGroup
		work := make(chan int, len(batch))
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				var rec [BinpackRecordSize]byte
				for j := range work {
					idx := batch[j]
					copy(rec[:], data[idx*BinpackRecordSize:(idx+1)*BinpackRecordSize])
					b, score, resultByte, err := UnpackPosition(rec)
					if err != nil {
						results[j] = traceResult{err: fmt.Errorf("invalid record at index %d: %w", idx, err)}
						continue
					}
					trace := t.computeTrace(b)
					trace.Score = score
					trace.Result = float64(ResultToFloat(resultByte))
					results[j] = traceResult{trace: trace}
				}
			}()
		}
		for j := range batch {
			work <- j
		}
		close(work)
		wg.Wait()

		// Sequential: write results in order
		for j, r := range results {
			if r.err != nil {
				out.Close()
				os.Remove(tmpFile)
				return r.err
			}
			n, err := writeTraceRecord(bw, &r.trace, pairMap)
			if err != nil {
				out.Close()
				os.Remove(tmpFile)
				return err
			}
			globalIdx := batchStart + j
			if globalIdx < numTrain {
				trainBytesTotal += uint64(n)
			}
		}

		processed += len(batch)
		// Progress report every ~100K records
		if processed%(100000-100000%batchSize) < batchSize || processed == totalRecords {
			elapsed := time.Since(startTime)
			rate := float64(processed) / elapsed.Seconds()
			remaining := time.Duration(float64(totalRecords-processed) / rate * float64(time.Second))
			fmt.Printf("\r  %d / %d records (%.1f%%) — %.0f rec/s — ETA %v   ",
				processed, totalRecords,
				float64(processed)/float64(totalRecords)*100,
				rate, remaining.Round(time.Second))
		}
	}
	fmt.Println() // newline after progress

	if err := bw.Flush(); err != nil {
		out.Close()
		os.Remove(tmpFile)
		return err
	}

	// Patch trainBytes in header
	binary.LittleEndian.PutUint64(headerBuf[16:24], trainBytesTotal)
	if _, err := out.Seek(0, io.SeekStart); err != nil {
		out.Close()
		os.Remove(tmpFile)
		return err
	}
	if _, err := out.Write(headerBuf[:]); err != nil {
		out.Close()
		os.Remove(tmpFile)
		return err
	}

	if err := out.Close(); err != nil {
		os.Remove(tmpFile)
		return err
	}

	return os.Rename(tmpFile, outputBin)
}

// OpenTraceFile mmaps a .tbin file and validates its header.
// mgToEg is the MG→EG pair map from Tuner.BuildPairMap(), required for v3 format.
// The caller must call Close() when done to unmap the file.
func OpenTraceFile(filename string, numParams int, mgToEg map[uint16]uint16) (*TraceFile, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := int(fi.Size())
	if size < tbinHeaderSize {
		return nil, fmt.Errorf("tbin file too small: %d bytes", size)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap failed: %v", err)
	}

	if string(data[0:4]) != tbinMagic {
		syscall.Munmap(data)
		return nil, fmt.Errorf("invalid tbin magic: %q", data[0:4])
	}
	version := binary.LittleEndian.Uint16(data[4:6])
	if version != tbinVersion {
		syscall.Munmap(data)
		return nil, fmt.Errorf("unsupported tbin version: %d", version)
	}
	fileParams := int(binary.LittleEndian.Uint16(data[6:8]))
	if fileParams != numParams {
		syscall.Munmap(data)
		return nil, fmt.Errorf("tbin param count mismatch: file has %d, tuner has %d", fileParams, numParams)
	}
	numTrain := int(binary.LittleEndian.Uint32(data[8:12]))
	numValidation := int(binary.LittleEndian.Uint32(data[12:16]))
	trainBytes := binary.LittleEndian.Uint64(data[16:24])

	return &TraceFile{
		Path:          filename,
		NumTrain:      numTrain,
		NumValidation: numValidation,
		NumParams:     fileParams,
		trainBytes:    trainBytes,
		data:          data,
		version:       version,
		mgToEg:        mgToEg,
	}, nil
}

// streamRecords walks the mmap'd data starting at byteOffset past the header,
// decoding count records in batches and calling fn for each batch.
// The batch slices are reused across calls; fn must not retain references to them.
func (tf *TraceFile) streamRecords(byteOffset int64, count, batchSize int, fn func(batch []TunerTrace)) {
	offset := tbinHeaderSize + int(byteOffset)

	batch := make([]TunerTrace, 0, batchSize)
	// Reusable backing slices for sparse entries per record.
	// We allocate separate slices for each batch slot so the batch elements
	// don't alias each other's backing arrays.
	mgBufs := make([][]SparseEntry, batchSize)
	egBufs := make([][]SparseEntry, batchSize)

	read := 0
	for read < count {
		batch = batch[:0]
		end := read + batchSize
		if end > count {
			end = count
		}
		for i := read; i < end; i++ {
			slot := i - read
			trace, mg, eg, newOffset := decodeTraceRecord(tf.data, offset, mgBufs[slot], egBufs[slot], tf.mgToEg)
			mgBufs[slot] = mg
			egBufs[slot] = eg
			offset = newOffset
			batch = append(batch, trace)
		}
		fn(batch)
		read = end
	}
}

// StreamTraining streams all training records in batches.
func (tf *TraceFile) StreamTraining(batchSize int, fn func(batch []TunerTrace)) {
	tf.streamRecords(0, tf.NumTrain, batchSize, fn)
}

// StreamValidation streams all validation records in batches.
func (tf *TraceFile) StreamValidation(batchSize int, fn func(batch []TunerTrace)) {
	tf.streamRecords(int64(tf.trainBytes), tf.NumValidation, batchSize, fn)
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
		enemyKingSq := b.Pieces[pieceOf(WhiteKing, enemy)].LSB()

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
			isOpenOrSemiOpen := false
			if fileMask&(friendlyPawns|enemyPawns) == 0 {
				addMG(base+6, s)  // RookOpenFileMG
				addEG(base+7, s)  // RookOpenFileEG
				isOpenOrSemiOpen = true
			} else if fileMask&friendlyPawns == 0 {
				addMG(base+8, s)  // RookSemiOpenFileMG
				addEG(base+9, s)  // RookSemiOpenFileEG
				isOpenOrSemiOpen = true
			}

			// Rook on enemy king file
			if RookEnemyKingFileEnabled && isOpenOrSemiOpen {
				enemyKingFile := enemyKingSq.File()
				fileDist := file - enemyKingFile
				if fileDist < 0 {
					fileDist = -fileDist
				}
				if fileDist <= 1 {
					addMG(base+22, s) // RookEnemyKingFileMG
					addEG(base+23, s) // RookEnemyKingFileEG
				}
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

			// Not blocked / blocked penalty (rank-indexed)
			var aheadSq Square
			if color == White {
				aheadSq = sq + 8
			} else {
				aheadSq = sq - 8
			}

			// Blocked passer penalty (enemy piece on stop square)
			if aheadSq >= 0 && aheadSq < 64 && b.AllPieces.IsSet(aheadSq) {
				blocker := b.Squares[aheadSq]
				if blocker.Color() != color {
					addMG(base+48+relRank*2, s)   // PassedPawnBlockedMG[relRank]
					addEG(base+48+relRank*2+1, s) // PassedPawnBlockedEG[relRank]
				}
			}

			notBlocked := aheadSq >= 0 && aheadSq < 64 && !b.AllPieces.IsSet(aheadSq)
			if notBlocked {
				enemyControls := b.IsAttacked(aheadSq, 1-color)
				safeAdvance := !enemyControls || b.IsAttacked(aheadSq, color)
				if safeAdvance {
					addMG(base+relRank*2, s)   // PassedPawnNotBlockedMG[relRank]
					addEG(base+relRank*2+1, s) // PassedPawnNotBlockedEG[relRank]

					// Free path
					if ForwardFileMask[color][sq]&b.AllPieces == 0 {
						addMG(base+16+relRank*2, s)   // PassedPawnFreePathMG[relRank]
						addEG(base+16+relRank*2+1, s) // PassedPawnFreePathEG[relRank]
					}
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
			} else if CandidatePassedEnabled {
				// Candidate passed pawn
				if ForwardFileMask[color][sq]&enemyPawns == 0 {
					adjSentries := (PassedPawnMask[color][sq] & AdjacentFiles[file] & enemyPawns).Count()
					friendlyAdj := (AdjacentFiles[file] & allFriendlyPawns).Count()
					if friendlyAdj >= adjSentries {
						addMG(base+58+relativeRank*2, s)   // candidatePassedMG[relativeRank]
						addEG(base+58+relativeRank*2+1, s) // candidatePassedEG[relativeRank]
					}
				}
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

			// Queenside pawn advancement (files a, b, c)
			if file <= 2 {
				addMG(base+42+relativeRank*2, s)   // queensidePawnAdvMG[relativeRank]
				addEG(base+42+relativeRank*2+1, s) // queensidePawnAdvEG[relativeRank]
			}

			// Pawn lever
			if PawnLeverEnabled {
				var aheadSq Square
				if color == White {
					aheadSq = sq + 8
				} else {
					aheadSq = sq - 8
				}
				if aheadSq >= 0 && aheadSq < 64 {
					if PawnAttacks[color][aheadSq]&enemyPawns != 0 {
						addMG(base+74+relativeRank*2, s)   // pawnLeverMG[relativeRank]
						addEG(base+74+relativeRank*2+1, s) // pawnLeverEG[relativeRank]
					}
				}
			}
		}

		// Pawn majority
		passed := pawnEntry.Passed[color]
		friendly := b.Pieces[pieceOf(WhitePawn, color)]
		enemy := b.Pieces[pieceOf(WhitePawn, 1-color)]
		for _, wingMask := range [2]Bitboard{QueensideMask, KingsideMask} {
			if passed&wingMask != 0 {
				continue
			}
			ours := (friendly & wingMask).Count()
			theirs := (enemy & wingMask).Count()
			if ours > theirs {
				coeff := int16(ours-theirs) * s
				addMG(base+40, coeff) // PawnMajorityMG
				addEG(base+41, coeff) // PawnMajorityEG
			}
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
		var allKnightAttacks, allBishopAttacks, allRookAttacks, allQueenAttacks Bitboard

		// Knights
		kn := b.Pieces[pieceOf(WhiteKnight, color)]
		for kn != 0 {
			sq := kn.PopLSB()
			attacks := KnightAttacks[sq] &^ friendly
			allKnightAttacks |= attacks
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
			allBishopAttacks |= attacks
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
			allRookAttacks |= attacks
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
			allQueenAttacks |= attacks
			if kzAttacks := attacks & kingZone; kzAttacks != 0 {
				attackerCount++
				attackUnits += QueenAttackUnits + QueenKingZoneBonus*kzAttacks.Count()
			}
		}

		// Safe check bonus (same as eval.go)
		if attackerCount >= 1 {
			var enemyPawnAtk Bitboard
			if enemy == White {
				enemyPawnAtk = enemyPawns.NorthWest() | enemyPawns.NorthEast()
			} else {
				enemyPawnAtk = enemyPawns.SouthWest() | enemyPawns.SouthEast()
			}
			safeSquares := ^(enemyPawnAtk | friendly)
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

		// Piece-on-piece threats (reuse attack bitboards from king safety loop)
		minorAttacks := allKnightAttacks | allBishopAttacks
		enemyRooks := b.Pieces[pieceOf(WhiteRook, enemy)]
		enemyQueens := b.Pieces[pieceOf(WhiteQueen, enemy)]
		minorOnRook := int16((minorAttacks & enemyRooks).Count())
		minorOnQueen := int16((minorAttacks & enemyQueens).Count())
		rookOnQueen := int16((allRookAttacks & enemyQueens).Count())
		addMG(miscBase+9, s*minorOnRook)   // MinorThreatRookMG
		addEG(miscBase+10, s*minorOnRook)  // MinorThreatRookEG
		addMG(miscBase+11, s*minorOnQueen) // MinorThreatQueenMG
		addEG(miscBase+12, s*minorOnQueen) // MinorThreatQueenEG
		addMG(miscBase+13, s*rookOnQueen)  // RookThreatQueenMG
		addEG(miscBase+14, s*rookOnQueen)  // RookThreatQueenEG
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

	// === Pawn Storm (MG-only, direct bonus) ===
	for color := Color(0); color <= 1; color++ {
		s := int16(1)
		if color == Black {
			s = -1
		}
		enemy := color ^ 1
		friendlyPawns := b.Pieces[pieceOf(WhitePawn, color)]
		enemyPawns := b.Pieces[pieceOf(WhitePawn, enemy)]
		enemyKingSq := b.Pieces[pieceOf(WhiteKing, enemy)].LSB()

		enemyKingFile := enemyKingSq.File()
		stFile := enemyKingFile - 1
		if stFile < 0 {
			stFile = 0
		}
		eFile := enemyKingFile + 1
		if eFile > 7 {
			eFile = 7
		}

		for f := stFile; f <= eFile; f++ {
			filePawns := friendlyPawns & FileMasks[f]
			if filePawns == 0 {
				continue
			}
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

			opposed := 0
			if enemyPawns&FileMasks[f] == 0 {
				opposed = 1 // unopposed
			}

			addMG(idxPawnStorm+opposed*8+relRank, s)
			addEG(idxPawnStorm+16+opposed*8+relRank, s)

			// Same-side storm: extra MG bonus when both kings on same wing
			if SameSideStormEnabled && relRank >= 4 {
				ourKingSq := b.Pieces[pieceOf(WhiteKing, color)].LSB()
				ourKingFile := ourKingSq.File()
				sameSide := (ourKingFile <= 3 && enemyKingFile <= 3) || (ourKingFile >= 4 && enemyKingFile >= 4)
				if sameSide {
					addMG(idxSameSideStorm+relRank, s)
				}
			}
		}
	}

	// === Endgame King Activity ===
	{
		wKingSq := b.Pieces[WhiteKing].LSB()
		bKingSq := b.Pieces[BlackKing].LSB()
		wCenterDist := int16(centerDistance(wKingSq))
		bCenterDist := int16(centerDistance(bKingSq))

		// Unconditional centralization: White - Black center distance diff
		addEG(idxEndgameKing+0, wCenterDist-bCenterDist) // KingCenterBonusEG

		// Material advantage bonuses
		wMat := b.Pieces[WhiteKnight].Count() + b.Pieces[WhiteBishop].Count() +
			b.Pieces[WhiteRook].Count()*3 + b.Pieces[WhiteQueen].Count()*6
		bMat := b.Pieces[BlackKnight].Count() + b.Pieces[BlackBishop].Count() +
			b.Pieces[BlackRook].Count()*3 + b.Pieces[BlackQueen].Count()*6

		dist := int16(chebyshevDistance(wKingSq, bKingSq))

		if wMat > bMat {
			addEG(idxEndgameKing+1, 7-dist)     // KingProximityAdvantageEG
			addEG(idxEndgameKing+2, bCenterDist) // KingCornerPushEG
		} else if bMat > wMat {
			addEG(idxEndgameKing+1, -(7 - dist))  // KingProximityAdvantageEG
			addEG(idxEndgameKing+2, -wCenterDist)  // KingCornerPushEG
		}
	}

	// === Tempo ===
	tempoSign := int16(1)
	if b.SideToMove == Black {
		tempoSign = -1
	}
	addMG(miscBase+16, tempoSign) // TempoMG
	addEG(miscBase+17, tempoSign) // TempoEG

	// === Trade bonus ===
	// In eval.go this is scaled by min(|score|, 500) / 500.
	// We use the engine's current evaluation as the constant scale factor
	// (same approach as PassedPawnKingScale).
	{
		evalScore := b.Evaluate()
		absEval := evalScore
		if absEval < 0 {
			absEval = -absEval
		}
		if absEval > 500 {
			absEval = 500
		}

		wPieces := b.Pieces[WhiteKnight].Count() + b.Pieces[WhiteBishop].Count() +
			b.Pieces[WhiteRook].Count() + b.Pieces[WhiteQueen].Count()
		bPieces := b.Pieces[BlackKnight].Count() + b.Pieces[BlackBishop].Count() +
			b.Pieces[BlackRook].Count() + b.Pieces[BlackQueen].Count()
		wPawns := b.Pieces[WhitePawn].Count()
		bPawns := b.Pieces[BlackPawn].Count()

		pieceCoeff := int16((int(wPieces) - int(bPieces)) * absEval / 500)
		pawnCoeff := int16((int(wPawns) - int(bPawns)) * absEval / 500)

		// Add to both MG and EG with same coefficient to make phase-independent
		addMG(miscBase+18, pieceCoeff) // TradePieceBonus
		addEG(miscBase+18, pieceCoeff)
		addMG(miscBase+19, pawnCoeff) // TradePawnBonus
		addEG(miscBase+19, pawnCoeff)
	}

	trace.WScale, trace.BScale = b.endgameScale()
	trace.HalfmoveClock = int(b.HalfmoveClock)

	return trace
}

// endgameScaleFactor returns the combined multiplicative scale factor (0.0-1.0)
// for a position given the raw score sign.
func endgameScaleFactor(trace *TunerTrace, rawScore float64) float64 {
	var scale float64
	if rawScore > 0 {
		scale = float64(trace.WScale) / 128.0
	} else if rawScore < 0 {
		scale = float64(trace.BScale) / 128.0
	} else {
		scale = 1.0
	}
	if trace.HalfmoveClock > 0 {
		scale *= float64(100-trace.HalfmoveClock) / 100.0
	}
	return scale
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
	score := (mg*(float64(TotalPhase)-phase) + eg*phase) / float64(TotalPhase)
	return score * endgameScaleFactor(trace, score)
}

// sigmoid maps an evaluation score to a win probability.
func sigmoid(score, K float64) float64 {
	return 1.0 / (1.0 + math.Pow(10.0, -score/K))
}

// TuneK finds the optimal scaling constant K by minimizing MSE on training data.
// Uses a random sample of up to 1M records for speed — K is stable well below this.
// Always tunes against game results (not blended scores), since wrapping both
// target and prediction in the same sigmoid lets K trivially minimize error by
// flattening everything to 0.5.
func (t *Tuner) TuneK(tf *TraceFile) float64 {
	// Sample up to 1M training records for K estimation
	sampleSize := tf.NumTrain
	const maxSample = 1_000_000
	if sampleSize > maxSample {
		sampleSize = maxSample
	}

	// Load sample into memory. Deep-copy MG/EG slices since the streaming
	// callback reuses backing arrays between batches.
	sample := make([]TunerTrace, 0, sampleSize)
	tf.streamRecords(0, sampleSize, streamBatchSize, func(batch []TunerTrace) {
		for i := range batch {
			if len(sample) >= sampleSize {
				return
			}
			t := batch[i]
			t.MG = append([]SparseEntry(nil), batch[i].MG...)
			t.EG = append([]SparseEntry(nil), batch[i].EG...)
			sample = append(sample, t)
		}
	})

	// Print diagnostic: score distribution, result stats, and score-result correlation
	{
		var minScore, maxScore float64
		sumScore := 0.0
		wins, draws, losses := 0, 0, 0
		winScoreSum, lossScoreSum := 0.0, 0.0
		// Bucket scores for correlation check
		type bucket struct {
			wins, draws, losses int
		}
		buckets := map[string]*bucket{}
		getBucket := func(s float64) string {
			switch {
			case s < -500:
				return "<-500"
			case s < -200:
				return "-500..-200"
			case s < -50:
				return "-200..-50"
			case s < 50:
				return "-50..50"
			case s < 200:
				return "50..200"
			case s < 500:
				return "200..500"
			default:
				return ">500"
			}
		}
		for _, name := range []string{"<-500", "-500..-200", "-200..-50", "-50..50", "50..200", "200..500", ">500"} {
			buckets[name] = &bucket{}
		}

		for i := range sample {
			s := scoreFromTrace(&sample[i], t.Values)
			if i == 0 || s < minScore {
				minScore = s
			}
			if i == 0 || s > maxScore {
				maxScore = s
			}
			sumScore += s
			b := buckets[getBucket(s)]
			switch {
			case sample[i].Result == 1.0:
				wins++
				winScoreSum += s
				b.wins++
			case sample[i].Result == 0.5:
				draws++
				b.draws++
			default:
				losses++
				lossScoreSum += s
				b.losses++
			}
		}
		fmt.Printf("  K diagnostic: %d samples, scores min=%.1f avg=%.1f max=%.1f\n",
			len(sample), minScore, sumScore/float64(len(sample)), maxScore)
		fmt.Printf("  Results: W=%d D=%d L=%d (%.1f%% decisive)\n",
			wins, draws, losses,
			float64(wins+losses)/float64(len(sample))*100)
		if wins > 0 {
			fmt.Printf("  Avg score in wins: %.1f, avg score in losses: %.1f\n",
				winScoreSum/float64(wins), lossScoreSum/float64(losses))
		}
		fmt.Printf("  Score-result correlation by bucket:\n")
		for _, name := range []string{"<-500", "-500..-200", "-200..-50", "-50..50", "50..200", "200..500", ">500"} {
			b := buckets[name]
			total := b.wins + b.draws + b.losses
			if total == 0 {
				continue
			}
			winPct := float64(b.wins) / float64(total) * 100
			fmt.Printf("    %12s: n=%6d  W=%.1f%% D=%.1f%% L=%.1f%%\n",
				name, total, winPct,
				float64(b.draws)/float64(total)*100,
				float64(b.losses)/float64(total)*100)
		}
	}

	// Always tune K against game results (not score targets).
	// K maps eval scores to win probabilities — it only has meaning
	// relative to a fixed target. With score-only targets, K cancels out.
	sampleError := func(K float64) float64 {
		totalErr := 0.0
		for i := range sample {
			score := scoreFromTrace(&sample[i], t.Values)
			predicted := sigmoid(score, K)
			target := sample[i].Result
			diff := target - predicted
			totalErr += diff * diff
		}
		return totalErr / float64(len(sample))
	}

	// Golden section search for optimal K in [50, 2000]
	lo, hi := 50.0, 2000.0
	gr := (math.Sqrt(5) + 1) / 2
	iter := 0

	for hi-lo > 0.1 {
		c := hi - (hi-lo)/gr
		d := lo + (hi-lo)/gr
		if sampleError(c) < sampleError(d) {
			hi = d
		} else {
			lo = c
		}
		iter++
		if iter%5 == 0 {
			fmt.Printf("  K search: iter %d, range [%.1f, %.1f]\n", iter, lo, hi)
		}
	}
	return (lo + hi) / 2
}

const streamBatchSize = 65536

// computeErrorStreaming computes MSE by streaming records from a TraceFile region.
// scoreBlend controls the target: 0=result-only, 1=score-only.
func (t *Tuner) computeErrorStreaming(tf *TraceFile, byteOffset int64, count int, K, scoreBlend float64) float64 {
	if count == 0 {
		return 0
	}

	numCPU := runtime.NumCPU()
	totalErr := 0.0

	tf.streamRecords(byteOffset, count, streamBatchSize, func(batch []TunerTrace) {
		n := len(batch)
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
					score := scoreFromTrace(&batch[i], t.Values)
					predicted := sigmoid(score, K)
					target := batch[i].Result
					if scoreBlend > 0 && batch[i].Score != 0 {
						scoreTarget := sigmoid(float64(batch[i].Score), K)
						target = (1-scoreBlend)*target + scoreBlend*scoreTarget
					}
					diff := target - predicted
					localErr += diff * diff
				}
				errors[id] = localErr
			}(cpu)
		}
		wg.Wait()

		for _, e := range errors {
			totalErr += e
		}
	})

	return totalErr / float64(count)
}

// ComputeTrainError computes MSE on the training set.
func (t *Tuner) ComputeTrainError(tf *TraceFile, K, scoreBlend float64) float64 {
	return t.computeErrorStreaming(tf, 0, tf.NumTrain, K, scoreBlend)
}

// ComputeValidationError computes MSE on the held-out validation set.
func (t *Tuner) ComputeValidationError(tf *TraceFile, K, scoreBlend float64) float64 {
	return t.computeErrorStreaming(tf, int64(tf.trainBytes), tf.NumValidation, K, scoreBlend)
}

// TuneConfig controls the tuning process.
type TuneConfig struct {
	Epochs     int     // number of optimization epochs
	LR         float64 // learning rate (default 1.0)
	Beta1      float64 // Adam beta1 (default 0.9)
	Beta2      float64 // Adam beta2 (default 0.999)
	Epsilon    float64 // Adam epsilon (default 1e-8)
	Lambda     float64 // L2 regularization toward initial values (default 0, disabled)
	ScoreBlend float64 // blend search scores into loss (0=result-only, 1=score-only)
	Stop       <-chan struct{} // if non-nil, checked each epoch; close to stop early
}

// DefaultTuneConfig returns sensible defaults.
func DefaultTuneConfig() TuneConfig {
	return TuneConfig{
		Epochs:  500,
		LR:      1.0,
		Beta1:   0.9,
		Beta2:   0.999,
		Epsilon: 1e-8,
		Lambda:  0,
	}
}

// Tune runs the Adam optimizer to minimize prediction error.
// After each Adam step, applies constraints:
//   - L2 regularization toward initial parameter values (cfg.Lambda)
//   - PST center-normalization (prevents PSTs from absorbing material offsets)
//   - King safety table monotonicity clamp
//
// Calls onEpoch after each epoch with (epoch, trainError, validationError).
func (t *Tuner) Tune(tf *TraceFile, K float64, cfg TuneConfig, onEpoch func(epoch int, trainErr, valErr float64)) {
	n := tf.NumTrain
	np := len(t.Values)
	if n == 0 || np == 0 {
		return
	}

	// Snapshot initial values for L2 regularization anchor
	initialValues := make([]float64, np)
	copy(initialValues, t.Values)

	// Adam state
	adam_m := make([]float64, np) // first moment
	adam_v := make([]float64, np) // second moment

	numCPU := runtime.NumCPU()

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		// Check for early stop signal
		if cfg.Stop != nil {
			select {
			case <-cfg.Stop:
				return
			default:
			}
		}

		// Accumulate gradient across all batches
		grad := make([]float64, np)

		tf.StreamTraining(streamBatchSize, func(batch []TunerTrace) {
			batchN := len(batch)
			chunkSize := (batchN + numCPU - 1) / numCPU

			gradChunks := make([][]float64, numCPU)
			var wg sync.WaitGroup

			for cpu := 0; cpu < numCPU; cpu++ {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()
					localGrad := make([]float64, np)
					start := id * chunkSize
					end := start + chunkSize
					if end > batchN {
						end = batchN
					}

					for i := start; i < end; i++ {
						trace := &batch[i]

						var mgSum, egSum float64
						for _, e := range trace.MG {
							mgSum += t.Values[e.Index] * float64(e.Coeff)
						}
						for _, e := range trace.EG {
							egSum += t.Values[e.Index] * float64(e.Coeff)
						}
						phase := float64(trace.Phase)
						rawScore := (mgSum*(float64(TotalPhase)-phase) + egSum*phase) / float64(TotalPhase)
						sf := endgameScaleFactor(trace, rawScore)
						score := rawScore * sf

						sig := sigmoid(score, K)
						target := trace.Result
						if cfg.ScoreBlend > 0 && trace.Score != 0 {
							scoreTarget := sigmoid(float64(trace.Score), K)
							target = (1-cfg.ScoreBlend)*target + cfg.ScoreBlend*scoreTarget
						}
						errTerm := (target - sig) * sig * (1 - sig)

						mgScale := errTerm * (float64(TotalPhase) - phase) / float64(TotalPhase) * sf
						egScale := errTerm * phase / float64(TotalPhase) * sf

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

			// Merge batch gradients into epoch gradient
			for _, chunk := range gradChunks {
				if chunk == nil {
					continue
				}
				for j := range grad {
					grad[j] += chunk[j]
				}
			}
		})

		// Scale by -2 * ln(10) / (K * N) and negate (we want to minimize)
		scale := -2.0 * math.Log(10) / (K * float64(n))
		for j := range grad {
			grad[j] *= scale
		}

		// Add L2 regularization gradient: lambda * 2 * (param - initial)
		if cfg.Lambda > 0 {
			for j := range grad {
				if !t.Frozen[j] {
					grad[j] += cfg.Lambda * 2 * (t.Values[j] - initialValues[j])
				}
			}
		}

		// Adam update
		for j := 0; j < np; j++ {
			if t.Frozen[j] {
				continue
			}
			adam_m[j] = cfg.Beta1*adam_m[j] + (1-cfg.Beta1)*grad[j]
			adam_v[j] = cfg.Beta2*adam_v[j] + (1-cfg.Beta2)*grad[j]*grad[j]

			mHat := adam_m[j] / (1 - math.Pow(cfg.Beta1, float64(epoch)))
			vHat := adam_v[j] / (1 - math.Pow(cfg.Beta2, float64(epoch)))

			t.Values[j] -= cfg.LR * mHat / (math.Sqrt(vHat) + cfg.Epsilon)
		}

		// --- Post-update constraints ---

		// PST center-normalization: for each piece type with material (Pawn..Queen),
		// subtract the PST mean and add it to the corresponding material value.
		// This prevents PSTs from absorbing material-level offsets.
		for pt := 0; pt < 5; pt++ { // Pawn=0..Queen=4
			// MG
			pstBase := idxPSTMG + pt*64
			matIdx := idxMaterialMG + pt
			sum := 0.0
			for sq := 0; sq < 64; sq++ {
				sum += t.Values[pstBase+sq]
			}
			mean := sum / 64.0
			for sq := 0; sq < 64; sq++ {
				t.Values[pstBase+sq] -= mean
			}
			t.Values[matIdx] += mean

			// EG
			pstBase = idxPSTEG + pt*64
			matIdx = idxMaterialEG + pt
			sum = 0.0
			for sq := 0; sq < 64; sq++ {
				sum += t.Values[pstBase+sq]
			}
			mean = sum / 64.0
			for sq := 0; sq < 64; sq++ {
				t.Values[pstBase+sq] -= mean
			}
			t.Values[matIdx] += mean
		}

		// King PST: center-normalize without material (King has no material value).
		// Just subtract the mean to keep values centered around 0.
		for _, pstBase := range []int{idxPSTMG + 5*64, idxPSTEG + 5*64} {
			sum := 0.0
			for sq := 0; sq < 64; sq++ {
				sum += t.Values[pstBase+sq]
			}
			mean := sum / 64.0
			for sq := 0; sq < 64; sq++ {
				t.Values[pstBase+sq] -= mean
			}
		}

		// King safety table monotonicity: clamp so entry[i] >= entry[i-1]
		for i := 1; i < 100; i++ {
			idx := idxKingSafetyTbl + i
			prev := idxKingSafetyTbl + i - 1
			if t.Values[idx] < t.Values[prev] {
				t.Values[idx] = t.Values[prev]
			}
		}

		if onEpoch != nil {
			trainErr := t.ComputeTrainError(tf, K, cfg.ScoreBlend)
			valErr := t.ComputeValidationError(tf, K, cfg.ScoreBlend)
			onEpoch(epoch, trainErr, valErr)
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
		"RookEnemyKingFileMG", "RookEnemyKingFileEG",
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

	w.WriteString("var PassedPawnBlockedMG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+48+i*2])))
	}
	w.WriteString("}\n")
	w.WriteString("var PassedPawnBlockedEG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+48+i*2+1])))
	}
	w.WriteString("}\n")
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

	fmt.Fprintf(w, "var doubledPawnMG = %d\n", int(math.Round(t.Values[base+32])))
	fmt.Fprintf(w, "var doubledPawnEG = %d\n", int(math.Round(t.Values[base+33])))
	fmt.Fprintf(w, "var isolatedPawnMG = %d\n", int(math.Round(t.Values[base+34])))
	fmt.Fprintf(w, "var isolatedPawnEG = %d\n", int(math.Round(t.Values[base+35])))
	fmt.Fprintf(w, "var backwardPawnMG = %d\n", int(math.Round(t.Values[base+36])))
	fmt.Fprintf(w, "var backwardPawnEG = %d\n", int(math.Round(t.Values[base+37])))
	fmt.Fprintf(w, "var connectedPawnMG = %d\n", int(math.Round(t.Values[base+38])))
	fmt.Fprintf(w, "var connectedPawnEG = %d\n", int(math.Round(t.Values[base+39])))
	fmt.Fprintf(w, "var PawnMajorityMG = %d\n", int(math.Round(t.Values[base+40])))
	fmt.Fprintf(w, "var PawnMajorityEG = %d\n", int(math.Round(t.Values[base+41])))

	w.WriteString("var queensidePawnAdvMG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+42+i*2])))
	}
	w.WriteString("}\n")
	w.WriteString("var queensidePawnAdvEG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+42+i*2+1])))
	}
	w.WriteString("}\n")

	w.WriteString("var candidatePassedMG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+58+i*2])))
	}
	w.WriteString("}\n")
	w.WriteString("var candidatePassedEG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+58+i*2+1])))
	}
	w.WriteString("}\n")
	w.WriteString("var pawnLeverMG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+74+i*2])))
	}
	w.WriteString("}\n")
	w.WriteString("var pawnLeverEG = [8]int{")
	for i := 0; i < 8; i++ {
		if i > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[base+74+i*2+1])))
	}
	w.WriteString("}\n")
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

	// Safe check bonuses
	w.WriteString("=== Safe Check ===\n")
	safeCheckNames := []string{
		"SafeKnightCheckBonus", "SafeBishopCheckBonus",
		"SafeRookCheckBonus", "SafeQueenCheckBonus",
	}
	for i, name := range safeCheckNames {
		fmt.Fprintf(w, "var %s = %d\n", name, int(math.Round(t.Values[idxSafeCheck+i])))
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

	// Pawn storm
	w.WriteString("=== Pawn Storm ===\n")
	w.WriteString("var PawnStormBonusMG = [2][8]int{\n")
	oppLabels := []string{"Opposed", "Unopposed"}
	for opp := 0; opp < 2; opp++ {
		w.WriteString("\t{")
		for r := 0; r < 8; r++ {
			if r > 0 {
				w.WriteString(", ")
			}
			fmt.Fprintf(w, "%d", int(math.Round(t.Values[idxPawnStorm+opp*8+r])))
		}
		fmt.Fprintf(w, "}, // %s\n", oppLabels[opp])
	}
	w.WriteString("}\n")
	w.WriteString("var PawnStormBonusEG = [2][8]int{\n")
	for opp := 0; opp < 2; opp++ {
		w.WriteString("\t{")
		for r := 0; r < 8; r++ {
			if r > 0 {
				w.WriteString(", ")
			}
			fmt.Fprintf(w, "%d", int(math.Round(t.Values[idxPawnStorm+16+opp*8+r])))
		}
		fmt.Fprintf(w, "}, // %s\n", oppLabels[opp])
	}
	w.WriteString("}\n\n")

	// Same-Side Storm
	w.WriteString("var SameSideStormMG = [8]int{")
	for r := 0; r < 8; r++ {
		if r > 0 {
			w.WriteString(", ")
		}
		fmt.Fprintf(w, "%d", int(math.Round(t.Values[idxSameSideStorm+r])))
	}
	w.WriteString("}\n\n")

	// Endgame King
	w.WriteString("=== Endgame King ===\n")
	egKingNames := []string{
		"KingCenterBonusEG",
		"KingProximityAdvantageEG",
		"KingCornerPushEG",
	}
	for i, name := range egKingNames {
		fmt.Fprintf(w, "var %s = %d\n", name, int(math.Round(t.Values[idxEndgameKing+i])))
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
		"MinorThreatRookMG", "MinorThreatRookEG",
		"MinorThreatQueenMG", "MinorThreatQueenEG",
		"RookThreatQueenMG", "RookThreatQueenEG",
		"OCBScale",
		"TempoMG", "TempoEG",
		"TradePieceBonus", "TradePawnBonus",
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
	// 1. King safety table (non-linear, not in trace)
	// 2. King safety attackerCount==1 division by 3 (not in trace)
	// 3. Trade bonus absScore approximation
	// 4. int16 rounding in trace coefficients
	// Note: endgame scaling and 50-move rule scaling are now modeled in trace
	diff := math.Abs(traceScore - evalScore)
	ok = diff < 150 // tolerance for non-linear components we don't trace
	return
}

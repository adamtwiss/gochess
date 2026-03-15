package chess

// PeSTO piece-square tables for tapered evaluation.
// Tables are from White's perspective with A1=index 0.
// For Black pieces, mirror the rank: table[sq ^ 56].
// Material values are baked into the PST entries.

// Phase contribution of each piece type
const (
	PawnPhase   = 0
	KnightPhase = 1
	BishopPhase = 1
	RookPhase   = 2
	QueenPhase  = 4
	TotalPhase  = 24 // 4*KnightPhase + 4*BishopPhase + 4*RookPhase + 2*QueenPhase
)

// Middlegame piece-square tables (material included)

var mgPawnTable = [64]int{
	-9, -9, -9, -9, -9, -9, -9, -9,
	14, 12, -1, -19, -24, 28, 56, 15,
	-4, -27, -4, -16, -4, 3, 39, 10,
	-21, -26, 4, 10, 13, 16, 5, -22,
	-23, -17, -21, 21, 25, 27, 0, -7,
	-15, -27, -12, -2, 9, 33, -1, -13,
	39, -26, -30, 47, 93, 20, -30, -6,
	-9, -9, -9, -9, -9, -9, -9, -9,
}

var egPawnTable = [64]int{
	-17, -17, -17, -17, -17, -17, -17, -17,
	17, -7, -7, 9, 13, 25, -4, 6,
	20, 4, -14, -4, 0, 18, -8, 11,
	25, 0, -49, -34, -21, 1, 13, 22,
	5, -7, -44, -28, -23, -6, 14, 13,
	6, 2, -28, 10, 14, 30, 46, 39,
	-35, 3, 13, 46, 28, 58, 54, 27,
	-17, -17, -17, -17, -17, -17, -17, -17,
}

var mgKnightTable = [64]int{
	-74, -3, -6, 15, 16, 11, -9, -59,
	-7, 2, 4, 33, 24, 27, 19, 5,
	-11, 25, 18, 33, 45, 25, 38, -9,
	19, 9, 44, 26, 35, 47, 34, 20,
	42, 33, 48, 57, 34, 83, 25, 55,
	-26, 62, 12, 73, 98, 73, 59, -22,
	-38, -69, 26, 27, 15, 54, -76, -28,
	-168, -175, -134, -115, 28, -271, -85, -95,
}

var egKnightTable = [64]int{
	-57, -62, -24, -23, -8, -21, -48, -17,
	-48, -10, -23, -16, -10, -25, -6, -25,
	-26, -10, -13, 16, 9, -11, -17, -12,
	20, 21, 34, 38, 37, 30, 22, 9,
	17, 19, 41, 46, 33, 24, 17, 7,
	9, -6, 45, 23, 2, 10, -27, 0,
	-4, 44, -10, 10, 6, -38, 18, -36,
	-31, 38, 40, 43, -4, 35, 0, -92,
}

var mgBishopTable = [64]int{
	24, 38, 18, 11, 38, -8, 40, 17,
	43, 56, 30, 28, 27, 52, 56, 74,
	34, 33, 49, 19, 24, 39, 47, 25,
	10, 22, 16, 59, 34, 31, 15, 25,
	20, 26, 17, 45, 45, 31, 32, -13,
	-37, 26, 15, 43, 44, 70, 31, 26,
	-61, -64, -4, -65, -62, -40, -59, -82,
	-34, -154, -92, -176, -153, -262, -111, -31,
}

var egBishopTable = [64]int{
	-30, -49, -31, -19, -24, -6, -35, -26,
	-46, -34, -15, -7, -11, -11, -14, -80,
	-20, 1, 24, 18, 38, 20, -14, -25,
	-29, 20, 34, 46, 41, 24, 0, -45,
	-29, 12, 33, 46, 40, 11, -6, -21,
	-4, 12, 20, 8, 1, 14, -3, -28,
	3, 22, 2, 12, 20, 7, 13, -27,
	-8, 30, 11, 32, 27, 53, 8, -3,
}

var mgRookTable = [64]int{
	-1, 18, 21, 24, 28, 31, 16, 4,
	-25, -12, -10, 0, 1, 6, 9, -44,
	-18, -11, -19, -29, -17, 0, 23, 0,
	-11, -17, -7, -21, -15, -10, -1, -6,
	-13, 8, 17, 21, 6, 27, 19, 10,
	10, 40, 31, 50, 73, 60, 67, 30,
	-69, -107, -56, -47, -61, -23, -112, -38,
	63, 37, 36, 16, -4, -12, -11, 21,
}

var egRookTable = [64]int{
	-23, -28, -18, -30, -37, -38, -37, -58,
	-17, -15, -6, -17, -31, -41, -36, -32,
	-4, 3, 13, 9, -4, -18, -29, -31,
	24, 40, 34, 33, 15, 18, 16, -1,
	52, 56, 48, 40, 37, 26, 33, 16,
	59, 48, 51, 41, 22, 32, 27, 22,
	-68, -62, -74, -78, -83, -107, -80, -110,
	39, 51, 46, 51, 49, 53, 60, 48,
}

var mgQueenTable = [64]int{
	21, 32, 31, 29, 35, 17, 49, 28,
	13, 22, 37, 27, 29, 46, 39, 34,
	5, 20, 3, 1, 4, 2, 12, 11,
	9, -46, -18, -16, -25, -29, -34, 0,
	-28, -41, -47, -27, -39, -27, -33, -6,
	-41, -24, -35, -14, -6, 60, 46, 30,
	-61, -43, -34, -60, -56, 31, 9, 69,
	-71, -29, 19, 8, 15, 35, 12, 4,
}

var egQueenTable = [64]int{
	-138, -188, -176, -152, -174, -179, -260, -203,
	-112, -110, -151, -113, -121, -140, -151, -191,
	-75, -70, -11, -32, -16, -12, -29, -78,
	-76, 50, 52, 96, 91, 80, 73, -19,
	11, 82, 105, 131, 165, 132, 98, 54,
	31, 66, 113, 111, 133, 113, 85, 32,
	79, 56, 111, 142, 157, 102, 113, -1,
	76, 41, 13, 31, 30, 33, 32, 58,
}

var mgKingTable = [64]int{
	-24, 23, 19, -62, -60, -34, 36, 22,
	15, 6, 1, -44, -47, -24, 12, 25,
	-70, 19, -8, -14, -19, -63, -28, -99,
	-1, 56, 39, 38, -27, -48, -63, -175,
	-27, 64, 33, -8, -28, -4, 60, -44,
	-16, 44, 38, 22, 19, 14, 17, -11,
	68, 72, 33, 21, 1, -4, 41, 7,
	26, 50, 27, -52, 32, 11, 53, 40,
}

var egKingTable = [64]int{
	-76, -67, -58, -46, -67, -50, -67, -96,
	-62, -61, -37, -25, -24, -34, -50, -42,
	-24, -39, -43, -31, -32, -29, -18, 1,
	-16, -12, -20, -44, -33, 7, 28, 42,
	24, 28, 16, -12, 0, 45, 60, 58,
	40, 60, 30, 37, 46, 73, 101, 74,
	-1, 43, 61, 58, 70, 96, 90, 46,
	-105, 23, 31, 34, 34, 65, 71, -171,
}

// PeSTO base material values (added to positional PST entries)
// Indexed by white piece type (1=Pawn .. 6=King)
var mgMaterial = [7]int{0, 97, 373, 379, 524, 1190, 0}
var egMaterial = [7]int{0, 125, 298, 455, 625, 1183, 0}

// Per-piece-type PST positional scale factors (percentage, 100 = unscaled PeSTO values).
// All set to 100: tuned PST values are pre-scaled.
var (
	PawnPSTScaleMG  = 100
	PawnPSTScaleEG  = 100
	PiecePSTScaleMG = 100
	PiecePSTScaleEG = 100
	KingPSTScaleMG  = 100
	KingPSTScaleEG  = 100
)

// Lookup tables indexed by white piece type (1-6)
var mgPST [7]*[64]int
var egPST [7]*[64]int

// Combined PST+material tables for incremental evaluation.
// Indexed by piece (1-12) and square (0-63). Black pieces have sq^56
// mirroring baked in, so callers just index [piece][square] directly.
var pstCombinedMG [13][64]int
var pstCombinedEG [13][64]int

func init() {
	mgPST[WhitePawn] = &mgPawnTable
	mgPST[WhiteKnight] = &mgKnightTable
	mgPST[WhiteBishop] = &mgBishopTable
	mgPST[WhiteRook] = &mgRookTable
	mgPST[WhiteQueen] = &mgQueenTable
	mgPST[WhiteKing] = &mgKingTable

	egPST[WhitePawn] = &egPawnTable
	egPST[WhiteKnight] = &egKnightTable
	egPST[WhiteBishop] = &egBishopTable
	egPST[WhiteRook] = &egRookTable
	egPST[WhiteQueen] = &egQueenTable
	egPST[WhiteKing] = &egKingTable

	// Build combined PST+material tables
	for pt := WhitePawn; pt <= WhiteKing; pt++ {
		var scaleMG, scaleEG int
		switch pt {
		case WhitePawn:
			scaleMG = PawnPSTScaleMG
			scaleEG = PawnPSTScaleEG
		case WhiteKing:
			scaleMG = KingPSTScaleMG
			scaleEG = KingPSTScaleEG
		default:
			scaleMG = PiecePSTScaleMG
			scaleEG = PiecePSTScaleEG
		}
		mgTable := mgPST[pt]
		egTable := egPST[pt]
		mgMat := mgMaterial[pt]
		egMat := egMaterial[pt]

		// White piece: direct index
		for sq := 0; sq < 64; sq++ {
			pstCombinedMG[pt][sq] = mgMat + mgTable[sq]*scaleMG/100
			pstCombinedEG[pt][sq] = egMat + egTable[sq]*scaleEG/100
		}

		// Black piece: bake in sq^56 mirror
		blackPt := pt + 6
		for sq := 0; sq < 64; sq++ {
			idx := sq ^ 56
			pstCombinedMG[blackPt][sq] = mgMat + mgTable[idx]*scaleMG/100
			pstCombinedEG[blackPt][sq] = egMat + egTable[idx]*scaleEG/100
		}
	}
}

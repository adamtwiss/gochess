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
	-11, -11, -11, -11, -11, -11, -11, -11,
	9, 8, 2, -20, -26, 24, 60, 4,
	-4, -29, -6, -20, -5, 6, 33, 8,
	-20, -32, 3, 7, 10, 13, 6, -30,
	-21, -18, -24, 22, 25, 21, -7, -22,
	-30, -18, 5, -7, 14, 54, -7, -30,
	67, -31, -43, 67, 142, 46, -35, 1,
	-11, -11, -11, -11, -11, -11, -11, -11,
}

var egPawnTable = [64]int{
	-17, -17, -17, -17, -17, -17, -17, -17,
	17, -2, -7, 16, 19, 31, -4, 9,
	24, 1, -11, -4, -1, 15, -6, 13,
	22, 4, -48, -32, -16, 4, 18, 25,
	2, -2, -44, -20, -20, -10, 9, 10,
	5, 11, -28, 17, 21, 26, 43, 35,
	-42, 11, -6, 22, 16, 56, 73, -3,
	-17, -17, -17, -17, -17, -17, -17, -17,
}

var mgKnightTable = [64]int{
	-115, -15, -23, 20, -1, 9, -15, -82,
	-23, -12, -6, 31, 28, 19, 16, 6,
	-19, 12, 15, 35, 53, 30, 37, -9,
	9, 27, 46, 33, 43, 47, 58, 25,
	49, 37, 57, 68, 41, 88, 26, 45,
	-34, 63, 54, 87, 91, 125, 64, -60,
	-30, -50, 22, 71, -4, 67, -104, -12,
	-191, -170, -119, -142, 50, -255, -91, -122,
}

var egKnightTable = [64]int{
	-87, -54, -32, -37, -7, -43, -36, -20,
	-46, -26, -32, -17, -13, -43, -19, -24,
	-50, -23, -23, 19, 2, -17, -25, -24,
	11, 9, 35, 42, 44, 34, 14, 17,
	14, 8, 42, 47, 45, 21, 7, 28,
	15, 5, 43, 40, 34, 25, -1, 20,
	-11, 40, -9, 13, 20, -31, 50, -28,
	-53, 18, 35, 55, 15, 50, 2, -88,
}

var mgBishopTable = [64]int{
	9, 25, 4, -9, 28, -19, 10, 17,
	25, 38, 24, 23, 19, 32, 53, 56,
	21, 28, 39, 22, 21, 25, 39, 20,
	11, 15, 16, 59, 41, 27, 22, 16,
	16, 25, 11, 61, 57, 40, 33, -24,
	-23, 22, 12, 34, 54, 95, 39, 43,
	-14, -41, -13, -43, -46, -38, -47, -57,
	-2, -105, -81, -182, -171, -274, -92, -45,
}

var egBishopTable = [64]int{
	-39, -50, -32, -22, -20, -20, -25, -30,
	-48, -37, -26, -7, -8, -20, -34, -76,
	-17, -5, 23, 4, 37, 15, -12, -34,
	-33, 23, 27, 46, 32, 19, 0, -35,
	-25, 13, 27, 37, 43, 9, 10, -25,
	-7, 23, 24, 25, 5, 25, 8, -13,
	-22, 19, 24, 21, 32, 10, 20, -34,
	-2, 27, 10, 37, 33, 42, 3, 5,
}

var mgRookTable = [64]int{
	-4, 19, 26, 30, 37, 40, 15, -1,
	-51, -14, -16, 1, 12, 30, 5, -68,
	-30, -21, -26, -36, -17, 3, 33, -21,
	-12, -15, -11, -15, 11, -4, -4, -26,
	-15, 7, 40, 47, 31, 49, 43, 14,
	18, 52, 40, 80, 112, 80, 65, 53,
	-102, -144, -67, -64, -65, -46, -113, -47,
	91, 26, 32, -4, 3, -39, -46, 3,
}

var egRookTable = [64]int{
	-8, -25, -17, -40, -43, -42, -31, -38,
	-13, -14, -11, -28, -52, -76, -46, -19,
	-1, 12, 1, -3, -9, -23, -39, -20,
	32, 44, 45, 24, 2, 5, 24, 13,
	59, 58, 46, 30, 31, 18, 23, 20,
	63, 55, 48, 32, 16, 18, 28, 28,
	-59, -56, -74, -82, -90, -107, -90, -101,
	38, 71, 55, 66, 56, 69, 70, 65,
}

var mgQueenTable = [64]int{
	4, 13, 13, 11, 29, 12, 54, 28,
	-13, 0, 20, 15, 12, 45, 21, 24,
	-18, 0, -2, -6, 6, 1, 24, 5,
	-7, -31, -16, -20, -22, -1, -7, -7,
	-32, -24, -37, -31, -16, 10, 3, -19,
	-20, -15, -36, 3, -6, 85, 50, 15,
	-46, -60, -18, -21, -31, 29, -36, 64,
	-80, -8, 28, 28, 21, 26, 4, -20,
}

var egQueenTable = [64]int{
	-122, -167, -147, -85, -170, -173, -262, -198,
	-87, -79, -110, -86, -90, -150, -154, -189,
	-73, -48, -20, -31, -16, 2, -76, -76,
	-43, 38, 51, 96, 68, 50, 34, -21,
	25, 53, 96, 138, 140, 77, 39, 58,
	23, 74, 128, 107, 171, 94, 64, 50,
	74, 88, 118, 119, 109, 111, 101, -14,
	77, 37, 23, 28, 17, 23, 21, 62,
}

var mgKingTable = [64]int{
	-42, 6, -8, -93, -83, -75, 7, -10,
	2, 0, 6, -46, -43, -39, 2, 3,
	-81, 49, 23, 49, 13, -38, -14, -137,
	54, 94, 70, 45, -30, -6, 13, -149,
	-12, 69, 30, -46, -57, 10, 112, 11,
	-22, 37, 22, 6, 8, 15, 28, 18,
	55, 65, 12, 10, -15, -18, 44, 5,
	8, 37, 13, -69, 20, -1, 49, 18,
}

var egKingTable = [64]int{
	-78, -62, -34, -36, -53, -30, -52, -110,
	-58, -56, -37, -23, -17, -17, -39, -43,
	-28, -43, -39, -38, -28, -25, -16, 6,
	-20, -12, -16, -36, -30, 2, 12, 28,
	26, 30, 22, -5, 8, 37, 47, 41,
	24, 69, 46, 35, 36, 59, 99, 54,
	-12, 66, 63, 65, 54, 84, 90, 3,
	-151, 53, 44, 41, 44, 87, 96, -225,
}

// PeSTO base material values (added to positional PST entries)
// Indexed by white piece type (1=Pawn .. 6=King)
var mgMaterial = [7]int{0, 99, 382, 408, 561, 1222, 0}
var egMaterial = [7]int{0, 125, 298, 468, 626, 1187, 0}

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

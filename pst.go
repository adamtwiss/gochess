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
	-7, -7, -7, -7, -7, -7, -7, -7,
	7, 12, 8, -12, -15, 23, 51, 11,
	-9, -23, -3, -14, -4, 2, 32, 9,
	-20, -24, 4, 6, 12, 13, 3, -16,
	-24, -18, -18, 16, 17, 18, -1, -11,
	-21, -18, -11, -8, 1, 28, 2, -13,
	30, -13, -34, 56, 101, 2, -5, -15,
	-7, -7, -7, -7, -7, -7, -7, -7,
}

var egPawnTable = [64]int{
	-13, -13, -13, -13, -13, -13, -13, -13,
	16, -5, -7, 6, 7, 22, 0, 0,
	20, 5, -11, -10, -2, 15, -6, 10,
	23, 3, -39, -35, -26, 0, 11, 25,
	5, -8, -41, -30, -25, -11, 4, 14,
	11, 4, -20, 3, 8, 26, 34, 42,
	-27, 5, 12, 31, 10, 50, 61, 27,
	-13, -13, -13, -13, -13, -13, -13, -13,
}

var mgKnightTable = [64]int{
	-55, 1, 2, 20, 21, 21, 3, -32,
	-3, 5, 13, 34, 31, 26, 31, 16,
	-1, 24, 21, 32, 48, 30, 41, 5,
	19, 17, 40, 30, 38, 47, 38, 28,
	36, 29, 36, 47, 37, 69, 25, 51,
	-27, 52, 11, 66, 87, 74, 46, -10,
	-29, -61, 10, 38, 10, 38, -93, -31,
	-205, -178, -126, -126, 25, -266, -87, -135,
}

var egKnightTable = [64]int{
	-43, -52, -22, -8, 3, -22, -29, -2,
	-44, -8, -19, -6, -5, -23, -1, -8,
	-34, -16, -13, 17, 8, -1, -8, -12,
	16, 18, 30, 29, 34, 34, 12, 8,
	7, 16, 36, 44, 32, 25, 7, 1,
	3, -9, 40, 20, -3, -7, -15, 6,
	0, 33, -12, -4, -4, -35, 21, -37,
	-28, 29, 34, 41, -16, 37, -4, -91,
}

var mgBishopTable = [64]int{
	29, 31, 18, 17, 43, -2, 30, 34,
	33, 50, 30, 28, 26, 48, 53, 60,
	32, 30, 45, 18, 23, 36, 42, 28,
	10, 19, 14, 57, 34, 24, 20, 24,
	16, 26, 10, 43, 41, 35, 39, -8,
	-26, 17, 7, 39, 32, 64, 21, 28,
	-48, -48, -9, -66, -46, -36, -56, -63,
	-52, -129, -73, -175, -154, -245, -96, -73,
}

var egBishopTable = [64]int{
	-26, -39, -25, -19, -29, -17, -11, -33,
	-35, -37, -21, -10, -10, -15, -17, -71,
	-24, -3, 14, 6, 35, 19, -18, -29,
	-14, 10, 29, 41, 25, 23, -7, -42,
	-25, 10, 28, 36, 33, -1, -10, -15,
	-4, 11, 18, 3, -2, 15, -1, -14,
	-6, 20, -2, 13, 12, 11, 14, -25,
	2, 36, 17, 47, 36, 70, 21, 5,
}

var mgRookTable = [64]int{
	-14, 11, 13, 16, 21, 23, 16, -6,
	-31, -9, -15, -6, 9, 3, 20, -33,
	-24, -16, -19, -18, -23, -7, 19, 1,
	-20, -19, -12, -22, -9, -10, 18, 7,
	-14, 0, 15, 26, 11, 36, 19, 10,
	1, 36, 27, 52, 90, 77, 67, 31,
	-68, -107, -67, -50, -55, -7, -102, -49,
	62, 42, 50, 13, 0, -7, -12, 9,
}

var egRookTable = [64]int{
	-3, -12, -5, -14, -17, -9, -17, -36,
	-15, -10, 4, -8, -31, -22, -39, -40,
	-5, 4, 19, 7, 6, -8, -20, -26,
	20, 29, 31, 26, 9, 16, 13, -4,
	41, 47, 33, 25, 21, 18, 18, 4,
	53, 35, 40, 29, 1, 13, 19, 12,
	-63, -57, -64, -78, -87, -105, -75, -99,
	37, 53, 40, 48, 40, 49, 61, 48,
}

var mgQueenTable = [64]int{
	36, 41, 37, 38, 46, 39, 53, 66,
	12, 25, 39, 31, 37, 51, 50, 38,
	-2, 17, 2, 6, 7, 10, 17, 13,
	8, -38, -17, -14, -21, -22, -21, -3,
	-30, -40, -49, -33, -41, -22, -26, -11,
	-32, -29, -48, -15, -19, 53, 33, 22,
	-61, -48, -35, -44, -47, 8, -23, 37,
	-106, -24, 27, 16, 17, 37, -4, -44,
}

var egQueenTable = [64]int{
	-134, -165, -153, -140, -142, -168, -246, -181,
	-102, -97, -127, -102, -105, -119, -155, -175,
	-64, -78, -14, -29, -9, 0, -23, -61,
	-82, 32, 37, 65, 83, 71, 55, -2,
	11, 62, 100, 122, 151, 108, 73, 51,
	19, 53, 112, 109, 147, 96, 78, 38,
	68, 62, 99, 128, 136, 101, 106, -13,
	82, 36, 18, 31, 26, 39, 28, 54,
}

var mgKingTable = [64]int{
	-37, 10, 8, -58, -62, -39, 26, 16,
	22, 2, 7, -41, -40, -26, 2, 12,
	-83, 19, -2, 7, 3, -53, -26, -115,
	15, 70, 56, 55, -12, -31, -38, -177,
	-25, 63, 35, -16, -39, 0, 80, -25,
	-17, 37, 37, 17, 20, 16, 16, 0,
	63, 64, 21, 13, -8, -16, 39, 4,
	20, 43, 20, -59, 26, 3, 45, 32,
}

var egKingTable = [64]int{
	-60, -44, -26, -32, -53, -27, -44, -83,
	-57, -42, -21, -2, 0, -12, -25, -22,
	-18, -25, -24, -14, -13, -7, -3, 25,
	-26, -21, -22, -32, -18, 15, 26, 50,
	12, 14, 11, 1, 7, 36, 44, 48,
	30, 36, 23, 29, 43, 56, 72, 59,
	-17, 18, 34, 37, 46, 61, 62, 16,
	-125, 16, 21, 31, 26, 56, 51, -196,
}

// PeSTO base material values (added to positional PST entries)
// Indexed by white piece type (1=Pawn .. 6=King)
var mgMaterial = [7]int{0, 95, 371, 375, 518, 1192, 0}
var egMaterial = [7]int{0, 121, 289, 448, 617, 1170, 0}

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

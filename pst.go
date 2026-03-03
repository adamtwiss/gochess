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
	10, 10, 3, -18, -29, 20, 55, 4,
	-3, -29, -5, -21, -9, 5, 29, 6,
	-18, -30, 2, 4, 7, 7, 6, -30,
	-20, -20, -22, 20, 26, 14, -7, -27,
	-37, -14, 11, -7, 15, 60, -9, -28,
	75, -36, -42, 40, 126, 45, -17, 29,
	-9, -9, -9, -9, -9, -9, -9, -9,
}

var egPawnTable = [64]int{
	-15, -15, -15, -15, -15, -15, -15, -15,
	20, 0, -7, 6, 15, 29, -2, 13,
	23, 2, -10, -4, -1, 12, -6, 16,
	23, 3, -47, -29, -19, 3, 14, 29,
	0, -2, -43, -20, -17, -8, 8, 12,
	1, 14, -23, 12, 22, 31, 40, 36,
	-45, 9, 0, 34, 21, 50, 49, -25,
	-15, -15, -15, -15, -15, -15, -15, -15,
}

var mgKnightTable = [64]int{
	-95, -12, -28, 27, 4, 19, -12, -77,
	-20, -14, -3, 35, 31, 21, 19, 10,
	-17, 9, 19, 38, 62, 33, 41, -7,
	13, 22, 53, 37, 47, 50, 57, 24,
	45, 40, 53, 65, 41, 71, 21, 39,
	-70, 63, 40, 85, 64, 77, 47, -110,
	10, -55, 21, 43, -22, 21, -111, 22,
	-150, -124, -110, -143, 26, -218, -77, -89,
}

var egKnightTable = [64]int{
	-94, -42, -33, -44, -9, -47, -35, -4,
	-28, -25, -31, -21, -10, -33, -18, -25,
	-37, -19, -20, 19, -5, -16, -31, -28,
	5, 9, 31, 40, 36, 28, 8, 22,
	11, 6, 36, 42, 41, 17, -11, 17,
	28, -2, 50, 32, 40, 34, -11, 32,
	-12, 38, -7, 19, 24, -5, 49, -53,
	-42, 11, 43, 38, 16, 38, 14, -76,
}

var mgBishopTable = [64]int{
	15, 15, 7, -2, 30, -18, 8, 10,
	22, 39, 26, 26, 22, 30, 53, 49,
	28, 27, 38, 23, 21, 28, 42, 13,
	12, 17, 16, 58, 39, 30, 20, 15,
	25, 28, 12, 56, 54, 34, 36, -32,
	-20, 23, 24, 23, 44, 73, 30, 29,
	-32, -35, -10, -65, -60, -30, -43, -67,
	23, -79, -58, -185, -175, -229, -115, -41,
}

var egBishopTable = [64]int{
	-40, -51, -31, -24, -18, -14, -28, -24,
	-52, -32, -27, -5, -6, -12, -28, -66,
	-25, 1, 26, 3, 43, 18, -16, -24,
	-34, 28, 27, 48, 33, 19, 2, -37,
	-18, 9, 28, 42, 47, 9, 1, -28,
	-4, 31, 20, 30, 12, 24, 8, -19,
	-10, 15, 32, 25, 36, 0, 15, -35,
	-14, 14, -5, 42, 24, 21, 0, -7,
}

var mgRookTable = [64]int{
	1, 25, 31, 34, 42, 44, 12, 3,
	-59, -2, -7, 8, 27, 33, -3, -77,
	-23, -11, -23, -39, -9, 12, 22, -17,
	-4, -11, 1, -16, 7, -4, -18, -26,
	-1, 9, 53, 54, 39, 53, 2, 14,
	19, 48, 52, 73, 113, 43, 28, 30,
	-64, -106, -54, -50, -62, -61, -99, -42,
	84, 4, 20, -17, -15, -65, -73, 14,
}

var egRookTable = [64]int{
	-4, -29, -20, -43, -47, -45, -30, -36,
	-12, -24, -19, -37, -62, -74, -42, -16,
	-6, 8, -8, -5, -15, -31, -36, -26,
	25, 39, 34, 20, -1, -2, 25, 5,
	49, 51, 36, 21, 25, 13, 28, 17,
	55, 46, 33, 26, 9, 22, 32, 33,
	-39, -36, -46, -56, -63, -69, -57, -73,
	34, 66, 48, 61, 49, 73, 71, 56,
}

var mgQueenTable = [64]int{
	16, 17, 16, 14, 28, -14, 54, 21,
	-16, 5, 24, 19, 14, 45, 7, 18,
	-34, -1, 0, -5, 7, 1, 24, -6,
	-8, -29, -20, -16, -20, -6, -13, -16,
	-31, -21, -34, -31, -13, 7, -4, -33,
	-12, 2, -38, 18, -13, 84, 39, -6,
	-41, -66, -10, -2, -20, 0, -40, 54,
	-67, 10, 43, 41, 32, 49, 1, -19,
}

var egQueenTable = [64]int{
	-120, -183, -147, -74, -155, -109, -224, -187,
	-81, -78, -103, -78, -68, -142, -106, -173,
	-53, -29, -32, -26, -9, 0, -84, -49,
	-41, 27, 57, 93, 62, 43, 34, -24,
	30, 44, 80, 129, 116, 59, 21, 51,
	28, 66, 131, 74, 163, 77, 68, 34,
	70, 104, 109, 98, 93, 127, 92, -24,
	68, 23, 9, 25, 5, 26, 20, 45,
}

var mgKingTable = [64]int{
	-47, 7, -8, -85, -74, -66, 8, -7,
	15, 8, 8, -27, -18, -22, 11, 0,
	-56, 61, 39, 67, 46, -1, 3, -110,
	33, 51, 45, 41, 3, 14, 28, -96,
	-17, 38, 28, -13, -43, 2, 45, 3,
	-31, 15, 22, -5, -2, 3, 24, -5,
	29, 36, -5, -7, -20, -7, 43, -5,
	0, 21, 2, -71, 6, -7, 33, 16,
}

var egKingTable = [64]int{
	-78, -67, -35, -41, -52, -30, -53, -116,
	-67, -55, -37, -26, -22, -20, -40, -45,
	-31, -40, -39, -38, -32, -33, -15, -3,
	-12, 0, -9, -32, -34, -2, 12, 16,
	30, 30, 27, -7, 4, 36, 58, 33,
	15, 72, 40, 35, 39, 68, 95, 52,
	11, 65, 62, 70, 51, 86, 90, 10,
	-144, 50, 43, 38, 39, 85, 91, -193,
}

// PeSTO base material values (added to positional PST entries)
// Indexed by white piece type (1=Pawn .. 6=King)
var mgMaterial = [7]int{0, 97, 367, 389, 541, 1162, 0}
var egMaterial = [7]int{0, 123, 285, 433, 590, 1114, 0}

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

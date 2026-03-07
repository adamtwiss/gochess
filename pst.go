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
	12, 14, 2, -20, -23, 26, 58, 11,
	-3, -29, -3, -19, -5, 6, 36, 12,
	-21, -28, 6, 9, 14, 15, 5, -23,
	-23, -18, -20, 22, 26, 26, -2, -8,
	-21, -20, -5, -5, 9, 36, 0, -14,
	42, -25, -33, 44, 96, 22, -28, -6,
	-9, -9, -9, -9, -9, -9, -9, -9,
}

var egPawnTable = [64]int{
	-18, -18, -18, -18, -18, -18, -18, -18,
	17, -9, -7, 8, 9, 27, -3, 9,
	21, 1, -13, -8, -2, 16, -5, 14,
	24, 1, -50, -33, -23, 5, 14, 27,
	3, -5, -44, -24, -21, -6, 13, 12,
	3, 4, -28, 9, 15, 31, 43, 39,
	-35, 4, 9, 44, 32, 65, 59, 29,
	-18, -18, -18, -18, -18, -18, -18, -18,
}

var mgKnightTable = [64]int{
	-77, -3, -5, 24, 19, 12, -3, -60,
	-6, -1, 2, 32, 27, 25, 20, 10,
	-13, 21, 17, 31, 47, 25, 39, -10,
	14, 7, 39, 23, 32, 41, 28, 22,
	44, 26, 44, 53, 42, 80, 17, 57,
	-29, 69, 9, 81, 105, 74, 66, -27,
	-37, -73, 28, 30, 14, 59, -77, -25,
	-165, -175, -134, -116, 29, -269, -84, -92,
}

var egKnightTable = [64]int{
	-60, -57, -26, -25, -6, -26, -38, -19,
	-50, -15, -27, -16, -12, -28, -10, -24,
	-30, -11, -23, 21, 7, -10, -16, -13,
	23, 21, 39, 37, 39, 36, 25, 12,
	16, 25, 44, 52, 32, 27, 15, 8,
	5, -9, 45, 21, -3, 11, -28, -1,
	-4, 45, -14, 12, 5, -36, 20, -36,
	-30, 36, 40, 43, -4, 35, 0, -90,
}

var mgBishopTable = [64]int{
	22, 39, 15, 8, 43, -6, 39, 19,
	41, 57, 30, 28, 25, 48, 58, 74,
	34, 30, 52, 18, 21, 38, 45, 30,
	13, 16, 8, 64, 33, 20, 11, 21,
	21, 30, 13, 50, 54, 33, 44, -17,
	-40, 25, 21, 47, 46, 76, 33, 34,
	-66, -69, -8, -67, -64, -42, -59, -86,
	-28, -153, -90, -177, -154, -263, -110, -28,
}

var egBishopTable = [64]int{
	-29, -50, -29, -20, -26, -12, -37, -23,
	-49, -34, -19, -12, -12, -12, -16, -82,
	-14, 1, 23, 15, 44, 24, -12, -23,
	-26, 22, 37, 46, 43, 31, 3, -43,
	-32, 11, 32, 46, 40, 9, -8, -21,
	-4, 9, 17, 7, 0, 14, -5, -25,
	0, 19, -3, 13, 19, 5, 12, -29,
	-5, 33, 12, 35, 28, 54, 9, 1,
}

var mgRookTable = [64]int{
	2, 23, 26, 29, 33, 38, 24, 5,
	-32, -12, -14, -3, 4, 11, 8, -48,
	-20, -13, -26, -35, -22, 2, 28, -2,
	-13, -22, -10, -27, -17, -13, -4, -6,
	-12, 2, 19, 20, 4, 30, 15, 14,
	6, 42, 28, 53, 76, 58, 68, 32,
	-67, -113, -54, -48, -63, -22, -116, -34,
	70, 42, 38, 15, -2, -10, -9, 24,
}

var egRookTable = [64]int{
	-19, -28, -21, -35, -37, -41, -37, -50,
	-21, -16, -7, -23, -39, -45, -37, -34,
	-3, 3, 11, 5, -4, -17, -26, -28,
	24, 39, 35, 28, 12, 15, 17, 2,
	52, 58, 49, 37, 35, 27, 31, 18,
	64, 50, 53, 40, 22, 32, 31, 25,
	-67, -61, -75, -82, -86, -105, -80, -106,
	45, 55, 46, 51, 52, 57, 64, 51,
}

var mgQueenTable = [64]int{
	24, 43, 41, 41, 48, 23, 58, 33,
	12, 26, 44, 34, 37, 55, 45, 40,
	2, 16, 1, 0, 5, 11, 13, 16,
	5, -57, -28, -26, -33, -38, -46, -5,
	-33, -51, -55, -33, -45, -34, -43, -13,
	-45, -28, -37, -16, -5, 60, 43, 29,
	-64, -40, -33, -63, -60, 32, 5, 67,
	-62, -23, 21, 12, 17, 36, 13, 8,
}

var egQueenTable = [64]int{
	-138, -190, -181, -147, -174, -180, -260, -203,
	-112, -109, -148, -116, -125, -138, -150, -191,
	-75, -69, -12, -32, -15, -11, -27, -77,
	-80, 46, 50, 95, 95, 79, 69, -21,
	10, 79, 103, 132, 165, 132, 94, 57,
	31, 64, 114, 112, 135, 113, 85, 35,
	79, 54, 112, 141, 155, 104, 113, -3,
	83, 42, 13, 33, 30, 34, 32, 63,
}

var mgKingTable = [64]int{
	-25, 27, 20, -64, -64, -43, 34, 19,
	16, 7, 5, -47, -43, -30, 9, 21,
	-73, 23, 0, -4, -9, -55, -27, -108,
	0, 59, 45, 45, -20, -42, -59, -176,
	-28, 65, 33, -9, -29, -2, 62, -43,
	-17, 43, 38, 20, 18, 12, 16, -11,
	66, 71, 31, 19, -1, -6, 41, 5,
	23, 48, 25, -55, 30, 9, 52, 36,
}

var egKingTable = [64]int{
	-75, -67, -54, -52, -68, -46, -65, -102,
	-61, -60, -34, -18, -19, -27, -47, -43,
	-23, -33, -35, -22, -23, -24, -14, 6,
	-20, -11, -21, -45, -37, 9, 28, 44,
	19, 28, 13, -18, -8, 42, 54, 57,
	39, 58, 30, 32, 43, 67, 99, 75,
	-4, 45, 58, 56, 66, 94, 94, 43,
	-110, 25, 31, 31, 34, 68, 76, -179,
}

// PeSTO base material values (added to positional PST entries)
// Indexed by white piece type (1=Pawn .. 6=King)
var mgMaterial = [7]int{0, 97, 373, 379, 524, 1188, 0}
var egMaterial = [7]int{0, 126, 299, 455, 627, 1181, 0}

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

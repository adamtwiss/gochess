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
	0, 0, 0, 0, 0, 0, 0, 0,
	6, 17, 17, -8, -7, 21, 31, -5,
	-9, -10, 6, -8, -6, 10, 10, -3,
	-24, -20, 7, -3, 11, 12, -10, -21,
	-15, -21, 2, 11, 9, 16, -11, -19,
	-36, 31, -30, -24, 34, 33, -7, -15,
	89, -77, -62, -29, 68, -6, 23, 120,
	0, 0, 0, 0, 0, 0, 0, 0,
}

var egPawnTable = [64]int{
	0, 0, 0, 0, 0, 0, 0, 0,
	9, 15, 18, 28, 24, 10, 6, 4,
	16, 15, 0, 13, 18, 4, 2, 12,
	22, 1, -15, 2, -7, -1, 13, 18,
	6, -3, -19, -9, -9, 4, 13, 5,
	-3, -17, 0, 0, -1, -5, 32, 15,
	-74, 7, 52, 3, 36, -1, 40, -86,
	0, 0, 0, 0, 0, 0, 0, 0,
}

var mgKnightTable = [64]int{
	-47, 32, 0, 24, 54, 3, 35, -61,
	53, -7, 33, 52, 48, 46, 64, 6,
	19, 44, 35, 44, 69, 41, 48, 21,
	30, 35, 33, 40, 46, 52, 38, 25,
	30, 36, 23, 50, 36, 21, 15, 44,
	-132, -16, -3, 84, -16, 51, 17, -150,
	-33, -41, -19, -28, -30, -51, -131, 91,
	-188, -56, -116, -182, 20, -180, -58, -38,
}

var egKnightTable = [64]int{
	-72, -8, -23, -52, -36, 13, 15, 4,
	5, -23, -40, -25, -6, -35, 31, 10,
	-10, -14, -23, -4, -25, -29, 16, -33,
	-4, 3, 13, 28, 12, 13, 12, 31,
	13, -21, 7, 6, -3, 38, -3, 2,
	59, -5, 21, 8, 20, 19, -17, -1,
	-2, 42, 10, -1, 13, -6, 38, -67,
	-79, 85, 31, -28, 10, 42, 49, -23,
}

var mgBishopTable = [64]int{
	23, -2, 25, 9, -3, 8, 73, 8,
	-1, 42, 48, 34, 29, 53, 41, 39,
	12, 40, 36, 18, 30, 27, 32, -4,
	-20, 34, 7, 39, 43, 23, -39, 48,
	37, 11, 25, 28, 35, 25, 36, -9,
	-11, 2, 32, 0, -17, 25, 41, -4,
	-74, -30, -17, -113, -33, -7, -63, -63,
	37, -17, 19, -156, -176, -166, -102, -48,
}

var egBishopTable = [64]int{
	-39, -44, -32, -23, -36, -20, -30, -11,
	-57, -7, -47, -8, 1, -47, -15, -60,
	-11, -19, -4, 4, 39, 3, -20, -5,
	-28, -6, 13, 26, 25, 4, -8, -31,
	-19, -7, 12, 24, 20, 18, -25, -9,
	2, 46, 18, 30, 28, 29, -17, -9,
	37, 31, 20, 27, 46, 2, 20, -22,
	11, 42, 21, 32, 23, 36, 19, 9,
}

var mgRookTable = [64]int{
	-18, 9, 0, 22, 26, 26, 17, -4,
	-71, -25, 8, 6, 25, 36, -2, -33,
	-14, 5, -24, -59, -9, 8, -2, 5,
	11, -13, 42, -33, 26, 19, 10, 36,
	-1, 15, 47, 37, 18, 66, -45, 57,
	37, 118, 15, 31, 96, 32, 16, 10,
	-102, -64, -98, 20, -25, -98, -88, -16,
	69, -34, 35, -12, 11, -72, -102, -4,
}

var egRookTable = [64]int{
	2, -12, -1, -18, -34, -22, -3, 5,
	-3, -6, -17, -8, -30, -38, -17, -13,
	-4, -8, 26, 15, 14, 19, -15, -26,
	-16, 13, -19, 30, -2, -7, 5, -11,
	29, 20, 5, 17, 17, 4, 25, 7,
	24, -8, 17, 8, 1, 12, 22, 12,
	-22, -35, -17, -64, -51, -19, -18, -50,
	10, 41, 32, 35, 15, 53, 62, 20,
}

var mgQueenTable = [64]int{
	13, 45, 17, 4, 14, 0, 49, 61,
	2, 2, 17, 17, 2, 27, -5, 46,
	-25, -6, 13, 10, 12, -6, 32, -25,
	-5, -14, -12, -16, -6, -15, -12, -7,
	-38, -25, -3, -32, -12, -10, -25, -27,
	-23, 14, -38, 6, -49, 56, 11, -12,
	-46, -63, -34, -2, -4, -59, -34, 75,
	-89, 28, 46, 75, 66, 40, 20, -38,
}

var egQueenTable = [64]int{
	-90, -177, -106, -93, -62, -23, -182, -156,
	-62, -38, -23, -72, -32, -97, -31, -162,
	-52, -29, -13, -16, -25, -6, -38, 10,
	-55, -10, 40, 105, 51, 29, 59, -29,
	24, 11, 55, 99, 70, 36, -13, 16,
	36, 57, 122, 31, 138, 46, 46, 6,
	61, 90, 69, 62, 72, 103, 84, -42,
	39, -1, -21, 17, 24, 16, 20, 12,
}

var mgKingTable = [64]int{
	-65, 23, 16, -15, -56, -34, 2, 13,
	29, -1, -20, -7, -8, -4, -38, -29,
	-9, 24, 2, -16, -20, 6, 22, -22,
	-17, -20, -12, -27, -30, -25, -14, -36,
	-49, -1, -27, -39, -46, -44, -33, -51,
	-14, -14, -22, -46, -44, -30, -15, -27,
	1, 7, -8, -64, -43, -16, 9, 8,
	-15, 36, 12, -54, 8, -28, 24, 14,
}

var egKingTable = [64]int{
	5, -52, -31, -45, -37, -34, -44, -90,
	-52, -49, -26, -17, -9, -23, -30, -50,
	-17, -18, -22, -21, -26, -32, -20, -24,
	-13, 8, 4, -1, -15, -10, -17, -30,
	9, 24, 29, 23, 43, 35, 11, 18,
	61, 72, 31, 52, 16, 47, 43, 39,
	-21, 42, 72, 56, 58, 45, 23, 19,
	-170, 82, 64, 111, 41, 44, 34, -212,
}

// PeSTO base material values (added to positional PST entries)
// Indexed by white piece type (1=Pawn .. 6=King)
var mgMaterial = [7]int{0, 91, 374, 392, 555, 1184, 0}
var egMaterial = [7]int{0, 125, 315, 442, 637, 1163, 0}

// Per-piece-type PST positional scale factors (percentage, 100 = unscaled PeSTO values).
// Only the positional component is scaled; material values are unchanged.
// Pawns are scaled most aggressively (rank-2 values of 98-134 are extreme).
// Pieces need stronger signals for placement quality differentiation.
var (
	PawnPSTScaleMG  = 100 // Tuned: values are pre-scaled
	PawnPSTScaleEG  = 100 // Tuned: values are pre-scaled
	PiecePSTScaleMG = 100 // Tuned: values are pre-scaled
	PiecePSTScaleEG = 100 // Tuned: values are pre-scaled
	KingPSTScaleMG  = 60  // Kept: using original MG king table
	KingPSTScaleEG  = 100 // Tuned: values are pre-scaled
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

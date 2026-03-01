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
	-5, -5, -5, -5, -5, -5, -5, -5,
	10, 12, 11, -29, -24, 15, 42, 7,
	0, -23, -3, -28, -11, -4, 16, 5,
	-18, -31, 5, 0, 4, 2, -6, -25,
	-15, -23, -13, 13, 22, 12, 1, -37,
	-43, 10, 0, -7, 39, 55, -28, -47,
	89, -63, -33, 1, 95, 6, 37, 78,
	-5, -5, -5, -5, -5, -5, -5, -5,
}

var egPawnTable = [64]int{
	-8, -8, -8, -8, -8, -8, -8, -8,
	9, 8, 9, 14, -1, 26, -2, 4,
	14, 1, 5, 11, -7, 1, -12, 9,
	12, 14, -33, -27, -17, -6, 10, 19,
	1, 4, -35, -10, -17, -14, 5, 10,
	-10, 1, -3, 0, 8, 20, 22, 36,
	-70, 4, 39, 31, 47, 20, 55, -75,
	-8, -8, -8, -8, -8, -8, -8, -8,
}

var mgKnightTable = [64]int{
	-82, 9, -17, 48, 24, 10, 1, -59,
	6, 5, -3, 51, 37, 33, 27, 37,
	-11, 14, 30, 50, 84, 38, 41, 1,
	23, 8, 58, 51, 49, 47, 56, 47,
	41, 52, 42, 60, 41, 50, 18, 60,
	-119, 16, 8, 101, 13, 34, 45, -144,
	-4, -79, -7, -2, -33, -25, -130, 73,
	-160, -86, -116, -168, 24, -191, -69, -60,
}

var egKnightTable = [64]int{
	-97, -43, -47, -28, -9, -19, -18, 23,
	-29, -26, -46, -32, -11, -50, -2, -1,
	-9, -15, -24, -5, -17, -20, -28, -3,
	3, -12, 18, 21, 26, 12, 22, 26,
	16, -14, 15, 21, 19, 15, -28, 22,
	53, 19, 28, -8, 39, 22, -29, 16,
	20, 41, 6, 28, 28, 8, 57, -50,
	-54, 52, 44, -4, 30, 41, 38, -50,
}

var mgBishopTable = [64]int{
	21, -22, 7, 2, 21, -19, 33, -16,
	18, 38, 29, 35, 22, 46, 43, 61,
	22, 27, 31, 20, 26, 41, 39, -1,
	1, 9, 18, 63, 30, 34, -7, 31,
	42, 36, 21, 36, 45, 35, 42, -48,
	-9, 6, 41, 14, 14, 45, 58, 16,
	-53, -20, -17, -89, -48, -38, -50, -52,
	40, -52, -18, -169, -170, -195, -119, -48,
}

var egBishopTable = [64]int{
	-41, -47, -16, -32, -8, -16, -38, -21,
	-42, -29, -35, -8, -10, -27, -33, -66,
	11, -10, 20, -3, 26, 10, -19, -20,
	-41, 20, 9, 49, 27, 18, 6, -24,
	-36, 4, 21, 27, 49, 16, 3, -37,
	-3, 30, 12, 27, 14, 24, -2, -22,
	19, 14, 33, 45, 30, -13, 19, -23,
	7, 20, -9, 42, 32, 17, 21, 7,
}

var mgRookTable = [64]int{
	-7, 23, 19, 31, 42, 38, 11, -12,
	-81, 6, -5, 5, 23, 29, -7, -62,
	-35, -1, -13, -53, -22, 7, 27, -14,
	-12, -2, 0, -46, 16, 4, -25, -1,
	9, 13, 67, 42, 37, 57, -22, 17,
	50, 104, 41, 59, 113, 46, 11, 36,
	-73, -74, -77, 0, -37, -88, -105, -28,
	83, -5, 14, -29, -10, -64, -84, 18,
}

var egRookTable = [64]int{
	10, -15, -10, -46, -39, -23, -5, -5,
	-5, -10, -17, -36, -65, -58, -35, -32,
	-2, -2, -12, -1, -5, -15, -26, -9,
	9, 24, 6, 34, -5, -6, 20, -26,
	37, 33, 21, 23, 28, 13, 29, 16,
	39, 10, 30, 27, 5, 19, 27, 27,
	-12, -37, -21, -68, -54, -31, -24, -72,
	24, 56, 33, 40, 26, 62, 68, 38,
}

var mgQueenTable = [64]int{
	-2, 14, 34, 11, 17, -13, 44, 28,
	-17, -8, 20, 20, 16, 30, 4, 16,
	-28, -9, 0, -11, 5, -5, 29, 1,
	-2, -41, -19, -10, -15, -2, -12, -7,
	-17, -28, -9, -25, -21, 8, -11, -36,
	0, 22, -21, 25, -27, 82, 23, -32,
	-22, -56, -16, 19, -19, -30, -17, 42,
	-66, 9, 37, 54, 34, 51, -5, -32,
}

var egQueenTable = [64]int{
	-111, -210, -151, -92, -100, -59, -196, -179,
	-80, -40, -77, -90, -57, -118, -57, -172,
	-35, -25, -11, -25, -12, 4, -54, -1,
	-38, 9, 53, 89, 54, 40, 63, -18,
	34, 22, 69, 111, 88, 43, -1, 35,
	42, 58, 135, 50, 149, 65, 62, 16,
	72, 101, 88, 80, 77, 120, 89, -37,
	53, 7, -7, 14, 0, 24, 14, 23,
}

var mgKingTable = [64]int{
	-73, 23, 15, -46, -43, -42, 14, 3,
	6, 29, -7, 0, 11, 3, 12, 5,
	-35, 36, 33, 28, 20, 28, 32, -47,
	11, 13, 19, 12, 6, 7, 26, -41,
	-19, 16, 8, -11, -27, -4, 14, -5,
	-23, -9, 10, -16, -2, -4, 21, -17,
	16, 25, -15, -19, -22, 1, 33, -4,
	-6, 18, 5, -56, 6, -7, 20, 13,
}

var egKingTable = [64]int{
	-34, -47, -44, -62, -47, -38, -42, -108,
	-58, -63, -36, -25, -22, -28, -35, -42,
	-27, -36, -36, -37, -26, -23, -16, -26,
	-15, 19, -12, -17, -29, -4, 19, 3,
	18, 30, 21, -5, 16, 38, 50, 54,
	29, 51, 45, 28, 46, 67, 80, 45,
	1, 66, 47, 80, 49, 74, 63, 6,
	-165, 58, 53, 71, 46, 74, 54, -197,
}

// PeSTO base material values (added to positional PST entries)
// Indexed by white piece type (1=Pawn .. 6=King)
var mgMaterial = [7]int{0, 93, 356, 374, 527, 1141, 0}
var egMaterial = [7]int{0, 116, 274, 411, 571, 1082, 0}

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

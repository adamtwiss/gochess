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
	8, 8, 3, -20, -26, 26, 62, 6,
	-6, -29, -6, -20, -5, 8, 35, 9,
	-22, -32, 4, 7, 10, 15, 8, -28,
	-21, -18, -23, 22, 26, 22, -6, -21,
	-31, -19, 4, -6, 13, 55, -6, -27,
	69, -31, -42, 69, 143, 42, -46, -5,
	-11, -11, -11, -11, -11, -11, -11, -11,
}

var egPawnTable = [64]int{
	-17, -17, -17, -17, -17, -17, -17, -17,
	19, -2, -7, 18, 21, 31, -4, 9,
	24, 1, -11, -3, 0, 15, -6, 13,
	23, 4, -48, -32, -16, 3, 17, 25,
	3, -2, -44, -20, -21, -11, 9, 9,
	6, 11, -28, 15, 20, 26, 42, 34,
	-41, 11, -7, 22, 16, 56, 75, -2,
	-17, -17, -17, -17, -17, -17, -17, -17,
}

var mgKnightTable = [64]int{
	-121, -16, -23, 20, -1, 8, -15, -81,
	-24, -13, -7, 30, 27, 19, 15, 5,
	-20, 11, 14, 34, 53, 30, 37, -10,
	8, 26, 45, 32, 42, 47, 59, 24,
	48, 36, 57, 69, 42, 90, 28, 47,
	-32, 63, 53, 89, 96, 138, 69, -44,
	-31, -52, 23, 78, 4, 78, -101, -9,
	-210, -195, -121, -141, 69, -272, -94, -132,
}

var egKnightTable = [64]int{
	-83, -57, -31, -38, -7, -44, -38, -21,
	-47, -28, -33, -19, -14, -44, -21, -27,
	-53, -25, -24, 18, 2, -19, -26, -26,
	11, 9, 35, 44, 46, 35, 15, 18,
	13, 9, 42, 49, 48, 23, 12, 30,
	13, 3, 42, 41, 35, 23, 1, 17,
	-11, 39, -10, 10, 18, -35, 48, -24,
	-49, 24, 35, 56, 11, 57, 5, -85,
}

var mgBishopTable = [64]int{
	9, 24, 4, -9, 28, -20, 10, 15,
	24, 38, 23, 23, 18, 32, 53, 55,
	21, 27, 39, 22, 21, 25, 39, 19,
	10, 15, 16, 60, 41, 26, 21, 17,
	16, 24, 11, 62, 57, 41, 32, -22,
	-23, 22, 12, 35, 54, 101, 43, 46,
	-13, -44, -14, -41, -46, -36, -48, -58,
	-13, -114, -88, -175, -163, -290, -67, -50,
}

var egBishopTable = [64]int{
	-39, -47, -33, -23, -22, -21, -25, -31,
	-46, -37, -27, -9, -10, -21, -36, -77,
	-17, -4, 22, 3, 36, 14, -12, -35,
	-32, 22, 27, 46, 31, 19, 0, -35,
	-26, 12, 27, 36, 43, 10, 11, -24,
	-9, 22, 23, 24, 6, 25, 8, -12,
	-23, 19, 23, 20, 33, 11, 22, -32,
	0, 29, 12, 36, 32, 49, 1, 10,
}

var mgRookTable = [64]int{
	-8, 15, 23, 25, 31, 33, 12, -4,
	-54, -17, -20, -5, 6, 22, 1, -70,
	-32, -23, -30, -43, -24, -2, 34, -22,
	-14, -17, -15, -20, 4, -7, 2, -27,
	-18, 6, 37, 42, 25, 47, 55, 17,
	17, 50, 39, 78, 109, 93, 83, 67,
	-107, -151, -75, -74, -75, -50, -115, -41,
	99, 36, 41, 8, 19, -14, -23, 19,
}

var egRookTable = [64]int{
	-11, -29, -19, -36, -36, -30, -21, -37,
	-20, -17, -12, -23, -41, -57, -32, -19,
	-8, 9, 1, 2, 3, -7, -27, -19,
	25, 39, 43, 28, 12, 20, 31, 14,
	52, 53, 44, 33, 40, 30, 26, 19,
	55, 50, 45, 33, 21, 23, 27, 21,
	-83, -75, -92, -96, -100, -115, -103, -124,
	29, 66, 53, 67, 62, 74, 75, 61,
}

var mgQueenTable = [64]int{
	3, 12, 12, 10, 28, 18, 57, 32,
	-14, -2, 19, 14, 11, 44, 23, 26,
	-16, -1, -4, -7, 6, 0, 23, 8,
	-7, -32, -16, -21, -23, -2, -6, -6,
	-32, -26, -38, -31, -18, 11, 4, -14,
	-22, -17, -36, -2, -3, 87, 60, 22,
	-48, -60, -20, -29, -35, 43, -35, 70,
	-86, -17, 21, 27, 19, 23, 13, -17,
}

var egQueenTable = [64]int{
	-123, -165, -149, -90, -176, -191, -277, -202,
	-89, -82, -115, -91, -97, -154, -167, -195,
	-81, -52, -20, -34, -21, -1, -74, -84,
	-47, 40, 49, 97, 69, 53, 34, -16,
	20, 52, 99, 141, 149, 83, 48, 64,
	21, 75, 125, 122, 172, 102, 60, 59,
	75, 82, 119, 130, 117, 97, 107, -9,
	80, 45, 32, 31, 25, 28, 24, 73,
}

var mgKingTable = [64]int{
	-36, 9, -5, -90, -81, -72, 12, -7,
	3, -4, -1, -52, -49, -43, 3, 6,
	-83, 40, 4, 20, -13, -48, -18, -131,
	67, 99, 55, 23, -53, -26, 4, -152,
	-11, 80, 22, -68, -68, 6, 144, 17,
	-17, 46, 17, 6, 11, 20, 27, 33,
	76, 83, 20, 19, -13, -26, 45, 14,
	14, 48, 20, -67, 31, 3, 60, 25,
}

var egKingTable = [64]int{
	-72, -61, -32, -33, -51, -29, -52, -101,
	-51, -55, -37, -21, -16, -17, -39, -36,
	-22, -42, -36, -33, -23, -24, -15, 11,
	-19, -14, -14, -32, -26, 4, 14, 35,
	27, 26, 22, -1, 8, 36, 40, 44,
	23, 64, 44, 34, 34, 56, 97, 52,
	-19, 59, 58, 63, 53, 84, 87, 3,
	-159, 44, 38, 41, 42, 84, 90, -233,
}

// PeSTO base material values (added to positional PST entries)
// Indexed by white piece type (1=Pawn .. 6=King)
var mgMaterial = [7]int{0, 99, 388, 411, 569, 1259, 0}
var egMaterial = [7]int{0, 125, 306, 483, 649, 1234, 0}

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

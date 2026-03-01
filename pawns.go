package chess

// Pawn hash table, pawn structure evaluation, and king safety.

// PawnEntry stores cached pawn structure evaluation for a pawn configuration.
type PawnEntry struct {
	Key     uint64
	WhiteMG int16
	WhiteEG int16
	BlackMG int16
	BlackEG int16
	Passed  [2]Bitboard // passed pawn bitboards per color
}

// PawnTable is a hash table for caching pawn structure evaluations.
type PawnTable struct {
	entries []PawnEntry
	mask    uint64
	probes  uint64
	hits    uint64
}

// Stats returns probe and hit counts for the pawn hash table.
func (pt *PawnTable) Stats() (probes, hits uint64) {
	return pt.probes, pt.hits
}

// NewPawnTable creates a new pawn hash table with the given size in MB.
func NewPawnTable(sizeMB int) *PawnTable {
	entrySize := uint64(32) // sizeof(PawnEntry): uint64 + 4×int16 + [2]uint64
	numEntries := uint64(sizeMB*1024*1024) / entrySize

	// Round down to power of 2
	size := uint64(1)
	for size*2 <= numEntries {
		size *= 2
	}

	return &PawnTable{
		entries: make([]PawnEntry, size),
		mask:    size - 1,
	}
}

// Probe looks up a pawn hash entry.
func (pt *PawnTable) Probe(key uint64) (PawnEntry, bool) {
	pt.probes++
	entry := pt.entries[key&pt.mask]
	if entry.Key == key {
		pt.hits++
		return entry, true
	}
	return PawnEntry{}, false
}

// Store saves a pawn hash entry (always-replace policy).
func (pt *PawnTable) Store(entry PawnEntry) {
	pt.entries[entry.Key&pt.mask] = entry
}

// Precomputed masks for pawn evaluation
var (
	FileMasks          [8]Bitboard     // one mask per file
	AdjacentFiles      [8]Bitboard     // neighboring file(s)
	PassedPawnMask     [2][64]Bitboard // squares that must be empty of enemy pawns for passed
	ForwardFileMask    [2][64]Bitboard // squares ahead on same file (for doubled detection)
	OutpostMask        [2][64]Bitboard // squares on adjacent files from rank upward (for outpost detection)
	BackwardSupportMask [2][64]Bitboard // adjacent file squares at same rank or behind (for backward detection)
)

func init() {
	// File masks
	files := [8]Bitboard{FileA, FileB, FileC, FileD, FileE, FileF, FileG, FileH}
	for i := 0; i < 8; i++ {
		FileMasks[i] = files[i]
	}

	// Adjacent files
	for i := 0; i < 8; i++ {
		if i > 0 {
			AdjacentFiles[i] |= FileMasks[i-1]
		}
		if i < 7 {
			AdjacentFiles[i] |= FileMasks[i+1]
		}
	}

	// Forward file masks and passed pawn masks
	for sq := 0; sq < 64; sq++ {
		file := sq % 8
		rank := sq / 8

		// White: squares ahead on same file
		var forwardWhite Bitboard
		for r := rank + 1; r < 8; r++ {
			forwardWhite |= SquareBB(Square(r*8 + file))
		}
		ForwardFileMask[White][sq] = forwardWhite

		// Black: squares ahead (downward) on same file
		var forwardBlack Bitboard
		for r := rank - 1; r >= 0; r-- {
			forwardBlack |= SquareBB(Square(r*8 + file))
		}
		ForwardFileMask[Black][sq] = forwardBlack

		// Passed pawn mask for White: no enemy pawns ahead on same or adjacent files
		PassedPawnMask[White][sq] = Bitboard(0)
		for r := rank + 1; r < 8; r++ {
			PassedPawnMask[White][sq] |= SquareBB(Square(r*8 + file))
			if file > 0 {
				PassedPawnMask[White][sq] |= SquareBB(Square(r*8 + file - 1))
			}
			if file < 7 {
				PassedPawnMask[White][sq] |= SquareBB(Square(r*8 + file + 1))
			}
		}

		// Passed pawn mask for Black: no enemy pawns ahead (downward) on same or adjacent files
		PassedPawnMask[Black][sq] = Bitboard(0)
		for r := rank - 1; r >= 0; r-- {
			PassedPawnMask[Black][sq] |= SquareBB(Square(r*8 + file))
			if file > 0 {
				PassedPawnMask[Black][sq] |= SquareBB(Square(r*8 + file - 1))
			}
			if file < 7 {
				PassedPawnMask[Black][sq] |= SquareBB(Square(r*8 + file + 1))
			}
		}

		// Outpost masks: adjacent file squares from this rank upward (White) or downward (Black)
		// If enemyPawns & OutpostMask[color][sq] == 0, no enemy pawn can attack sq
		OutpostMask[White][sq] = Bitboard(0)
		if file > 0 {
			for r := rank; r < 8; r++ {
				OutpostMask[White][sq] |= SquareBB(Square(r*8 + file - 1))
			}
		}
		if file < 7 {
			for r := rank; r < 8; r++ {
				OutpostMask[White][sq] |= SquareBB(Square(r*8 + file + 1))
			}
		}

		OutpostMask[Black][sq] = Bitboard(0)
		if file > 0 {
			for r := rank; r >= 0; r-- {
				OutpostMask[Black][sq] |= SquareBB(Square(r*8 + file - 1))
			}
		}
		if file < 7 {
			for r := rank; r >= 0; r-- {
				OutpostMask[Black][sq] |= SquareBB(Square(r*8 + file + 1))
			}
		}

		// BackwardSupportMask: adjacent file squares at same rank or behind
		// White: ranks 0..rank on adjacent files
		BackwardSupportMask[White][sq] = Bitboard(0)
		for r := 0; r <= rank; r++ {
			if file > 0 {
				BackwardSupportMask[White][sq] |= SquareBB(Square(r*8 + file - 1))
			}
			if file < 7 {
				BackwardSupportMask[White][sq] |= SquareBB(Square(r*8 + file + 1))
			}
		}
		// Black: ranks rank..7 on adjacent files
		BackwardSupportMask[Black][sq] = Bitboard(0)
		for r := rank; r <= 7; r++ {
			if file > 0 {
				BackwardSupportMask[Black][sq] |= SquareBB(Square(r*8 + file - 1))
			}
			if file < 7 {
				BackwardSupportMask[Black][sq] |= SquareBB(Square(r*8 + file + 1))
			}
		}
	}
}

// Pawn structure bonuses/penalties (centipawns)
var passedPawnMG = [8]int{0, 6, -5, -3, 6, 66, 149, 0} // by rank (0=rank1, 7=rank8)
var passedPawnEG = [8]int{0, 4, 26, 28, 46, 67, 166, 0}

var (
	doubledPawnMG   = -15
	doubledPawnEG   = -17
	isolatedPawnMG  = -17
	isolatedPawnEG  = -10
	backwardPawnMG  = -12
	backwardPawnEG  = -22
	connectedPawnMG = 10
	connectedPawnEG = 24
)

// Pawn advancement bonus by relative rank (index 0=rank1, 7=rank8).
// Rewards pawns that have advanced beyond their starting squares.
var pawnAdvancementMG = [8]int{0, -10, -2, 9, 22, 39, 129, 0}
var pawnAdvancementEG = [8]int{0, 32, 34, 34, 57, 71, 51, 0}

// Candidate passed pawn: no enemy pawn ahead on own file, friendly support >= enemy sentries
var candidatePassedMG = [8]int{0, 10, 0, 6, 1, -27, 0, 0}
var candidatePassedEG = [8]int{0, 13, 27, 53, 68, 207, 0, 0}
var CandidatePassedEnabled = true

// Pawn majority: bonus per pawn advantage on a wing (queenside/kingside)
var PawnMajorityMG = 20
var PawnMajorityEG = 17
var PawnMajorityEnabled = true

// Queenside pawn advancement bonus by relative rank (files a, b, c only).
// Stacks on top of base pawnAdvancement bonus. Rewards advancing queenside
// pawns which are strategically dangerous (further from king, create outside passers).
var queensidePawnAdvMG = [8]int{0, -12, 5, 6, 8, 12, 39, 0}
var queensidePawnAdvEG = [8]int{0, 10, 8, 17, 23, 32, 7, 0}

// Pawn lever: bonus for a pawn that can advance one square to attack an enemy pawn.
// Creates tension, opens lines, and is the mechanism behind most strategic pawn advances.
var pawnLeverMG = [8]int{0, 11, -1, 3, 3, 5, 0, 0}
var pawnLeverEG = [8]int{0, -6, -4, 1, 8, 3, 0, 0}
var PawnLeverEnabled = true

// evaluatePawnStructure evaluates pawn structure for one color.
// Returns mg and eg scores and a bitboard of passed pawns.
func evaluatePawnStructure(b *Board, color Color) (mg, eg int, passed Bitboard) {
	pawns := b.Pieces[pieceOf(WhitePawn, color)]
	enemyPawns := b.Pieces[pieceOf(WhitePawn, 1-color)]
	allFriendlyPawns := pawns

	// Compute pawn attacks for connected pawn detection
	var pawnAttacks Bitboard
	if color == White {
		pawnAttacks = allFriendlyPawns.NorthWest() | allFriendlyPawns.NorthEast()
	} else {
		pawnAttacks = allFriendlyPawns.SouthWest() | allFriendlyPawns.SouthEast()
	}

	for pawns != 0 {
		sq := pawns.PopLSB()
		file := sq.File()
		rank := int(sq.Rank())

		// For black, mirror the rank for table lookups
		relativeRank := rank
		if color == Black {
			relativeRank = 7 - rank
		}

		// Doubled pawns: another friendly pawn ahead on same file
		if ForwardFileMask[color][sq]&allFriendlyPawns != 0 {
			mg += doubledPawnMG
			eg += doubledPawnEG
		}

		// Isolated pawns: no friendly pawns on adjacent files
		if AdjacentFiles[file]&allFriendlyPawns == 0 {
			mg += isolatedPawnMG
			eg += isolatedPawnEG
		} else {
			// Backward pawn: not isolated, but no friendly pawn on adjacent files
			// at same rank or behind, and stop square controlled by enemy pawns
			if BackwardSupportMask[color][sq]&allFriendlyPawns == 0 {
				var stopSq Square
				if color == White {
					stopSq = sq + 8
				} else {
					stopSq = sq - 8
				}
				if stopSq >= 0 && stopSq < 64 {
					if PawnAttacks[color][stopSq]&enemyPawns != 0 {
						mg += backwardPawnMG
						eg += backwardPawnEG
					}
				}
			}
		}

		// Passed pawns: no enemy pawns ahead on same or adjacent files
		if PassedPawnMask[color][sq]&enemyPawns == 0 {
			mg += passedPawnMG[relativeRank]
			eg += passedPawnEG[relativeRank]
			passed |= SquareBB(sq)
		} else if CandidatePassedEnabled {
			// Candidate passed pawn: no enemy on own file ahead, friendly support >= enemy sentries
			if ForwardFileMask[color][sq]&enemyPawns == 0 {
				adjSentries := (PassedPawnMask[color][sq] & AdjacentFiles[file] & enemyPawns).Count()
				friendlyAdj := (AdjacentFiles[file] & allFriendlyPawns).Count()
				if friendlyAdj >= adjSentries {
					mg += candidatePassedMG[relativeRank]
					eg += candidatePassedEG[relativeRank]
				}
			}
		}

		// Connected pawns: defended by another pawn
		if SquareBB(sq)&pawnAttacks != 0 {
			mg += connectedPawnMG
			eg += connectedPawnEG
		}

		// Pawn advancement bonus
		mg += pawnAdvancementMG[relativeRank]
		eg += pawnAdvancementEG[relativeRank]

		// Queenside pawn advancement bonus (files a, b, c)
		if file <= 2 {
			mg += queensidePawnAdvMG[relativeRank]
			eg += queensidePawnAdvEG[relativeRank]
		}

		// Pawn lever: pawn can advance to attack an enemy pawn
		if PawnLeverEnabled {
			var aheadSq Square
			if color == White {
				aheadSq = sq + 8
			} else {
				aheadSq = sq - 8
			}
			if aheadSq >= 0 && aheadSq < 64 {
				if PawnAttacks[color][aheadSq]&enemyPawns != 0 {
					mg += pawnLeverMG[relativeRank]
					eg += pawnLeverEG[relativeRank]
				}
			}
		}
	}

	// Pawn majority: bonus when we have more pawns on a wing than the opponent
	if PawnMajorityEnabled {
		friendly := b.Pieces[pieceOf(WhitePawn, color)]
		enemy := b.Pieces[pieceOf(WhitePawn, 1-color)]
		for _, wingMask := range [2]Bitboard{QueensideMask, KingsideMask} {
			if passed&wingMask != 0 {
				continue
			}
			ours := (friendly & wingMask).Count()
			theirs := (enemy & wingMask).Count()
			if ours > theirs {
				advantage := ours - theirs
				mg += PawnMajorityMG * advantage
				eg += PawnMajorityEG * advantage
			}
		}
	}

	return
}

// probePawnEval probes the pawn hash table or computes pawn evaluation.
func (b *Board) probePawnEval() PawnEntry {
	if b.PawnTable != nil {
		if entry, ok := b.PawnTable.Probe(b.PawnHashKey); ok {
			return entry
		}
	}

	wMG, wEG, wPassed := evaluatePawnStructure(b, White)
	bMG, bEG, bPassed := evaluatePawnStructure(b, Black)

	entry := PawnEntry{
		Key:     b.PawnHashKey,
		WhiteMG: int16(wMG),
		WhiteEG: int16(wEG),
		BlackMG: int16(bMG),
		BlackEG: int16(bEG),
		Passed:  [2]Bitboard{wPassed, bPassed},
	}

	if b.PawnTable != nil {
		b.PawnTable.Store(entry)
	}

	return entry
}

// King safety constants (vars for tuner access)
var (
	shieldPawnRank2MG          = 24
	shieldPawnRank3MG          = 9
	missingShieldPawnMG        = -4
	missingShieldPawnAdvancedMG = -7
	semiOpenFileNearKingMG     = -37
)

// evaluateKingSafety evaluates king safety for one color.
// Not cached since king position changes frequently.
func (b *Board) evaluateKingSafety(color Color) (mg, eg int) {
	kingSq := b.Pieces[pieceOf(WhiteKing, color)].LSB()
	if kingSq == NoSquare {
		return 0, 0
	}

	kingFile := kingSq.File()
	friendlyPawns := b.Pieces[pieceOf(WhitePawn, color)]

	// Check files around the king (king file and adjacent files)
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
			// No friendly pawn on this file near king — semi-open file penalty
			mg += semiOpenFileNearKingMG
			continue
		}

		// Check for shield pawns on ranks 2 and 3 relative to the king's side
		foundShield := false
		if color == White {
			// White king: shield pawns on ranks 2-3
			if filePawns&Rank2 != 0 {
				mg += shieldPawnRank2MG
				foundShield = true
			} else if filePawns&Rank3 != 0 {
				mg += shieldPawnRank3MG
				foundShield = true
			}
		} else {
			// Black king: shield pawns on ranks 7-6
			if filePawns&Rank7 != 0 {
				mg += shieldPawnRank2MG
				foundShield = true
			} else if filePawns&Rank6 != 0 {
				mg += shieldPawnRank3MG
				foundShield = true
			}
		}

		if !foundShield {
			// Reduced penalty if pawn is on rank 4 (advanced but still present)
			hasAdvancedPawn := (color == White && filePawns&Rank4 != 0) ||
				(color == Black && filePawns&Rank5 != 0)
			if hasAdvancedPawn {
				mg += missingShieldPawnAdvancedMG
			} else {
				mg += missingShieldPawnMG
			}
		}
	}

	return
}

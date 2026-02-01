package chess

// SEE (Static Exchange Evaluation) determines the material outcome
// of a sequence of captures on a single square.
// Returns the material gain/loss in centipawns from the perspective
// of the side making the initial capture.

// SEEPieceValues for SEE calculation (simpler than eval values)
var SEEPieceValues = [13]int{
	0,   // Empty
	100, // WhitePawn
	320, // WhiteKnight
	330, // WhiteBishop
	500, // WhiteRook
	900, // WhiteQueen
	0,   // WhiteKing (can't be captured)
	100, // BlackPawn
	320, // BlackKnight
	330, // BlackBishop
	500, // BlackRook
	900, // BlackQueen
	0,   // BlackKing
}

// SEE returns the static exchange evaluation for a capture move
// Positive values mean the capture wins material
func (b *Board) SEE(m Move) int {
	from := m.From()
	to := m.To()

	// Get the initial captured piece value
	captured := b.Squares[to]
	if captured == Empty {
		// En passant
		if m.Flags() == FlagEnPassant {
			captured = WhitePawn // Value is same for both colors
		} else {
			return 0 // Not a capture
		}
	}

	// Track which pieces have been "removed" from the board
	occupied := b.AllPieces

	// The piece making the initial capture
	attacker := b.Squares[from]

	// Remove the initial attacker from consideration
	occupied &^= SquareBB(from)

	// Build the gain array on the stack (max 32 captures on a single square)
	// gain[i] = material balance after i-th capture, from perspective of side that just captured
	var gain [32]int
	gainLen := 1
	gain[0] = SEEPieceValues[captured] // Initial capture gains the victim

	// The piece that will be captured next (initially, the attacker that just moved)
	nextVictimValue := SEEPieceValues[attacker]

	// Determine side to move (starts with opponent after initial capture)
	sideToMove := 1 - b.SideToMove

	// Find the least valuable attacker for each side alternately
	for gainLen < 32 {
		// Find least valuable attacker of 'sideToMove' to 'to'
		lva, lvaValue := b.leastValuableAttacker(to, sideToMove, occupied)

		if lva == NoSquare {
			break // No more attackers
		}

		// The current side captures with their LVA
		// They gain nextVictimValue but from their perspective, need to negate previous gain
		gain[gainLen] = nextVictimValue - gain[gainLen-1]
		gainLen++

		// The piece that just captured becomes the next potential victim
		nextVictimValue = lvaValue

		// Remove the attacker from occupied
		occupied &^= SquareBB(lva)

		sideToMove = 1 - sideToMove
	}

	// Negamax the gain list to find the actual result
	// Work backwards: at each step, the moving side can choose to capture or stand pat
	// gain[i] = max(stand_pat, -opponent's_best) = max(gain[i], -gain[i+1])
	for i := gainLen - 2; i >= 0; i-- {
		if -gain[i+1] < gain[i] {
			gain[i] = -gain[i+1]
		}
	}

	return gain[0]
}

// leastValuableAttacker finds the least valuable piece of 'color' attacking 'sq'
// considering only pieces in 'occupied'. Returns the square and piece value.
func (b *Board) leastValuableAttacker(sq Square, color Color, occupied Bitboard) (Square, int) {
	// Check pawns first (least valuable)
	pawns := b.Pieces[pieceOf(WhitePawn, color)] & occupied
	pawnAttackers := PawnAttacks[1-color][sq] & pawns
	if pawnAttackers != 0 {
		return pawnAttackers.LSB(), SEEPieceValues[WhitePawn]
	}

	// Knights
	knights := b.Pieces[pieceOf(WhiteKnight, color)] & occupied
	knightAttackers := KnightAttacks[sq] & knights
	if knightAttackers != 0 {
		return knightAttackers.LSB(), SEEPieceValues[WhiteKnight]
	}

	// Bishops
	bishops := b.Pieces[pieceOf(WhiteBishop, color)] & occupied
	bishopAttacks := BishopAttacksBB(sq, occupied)
	bishopAttackers := bishopAttacks & bishops
	if bishopAttackers != 0 {
		return bishopAttackers.LSB(), SEEPieceValues[WhiteBishop]
	}

	// Rooks
	rooks := b.Pieces[pieceOf(WhiteRook, color)] & occupied
	rookAttacks := RookAttacksBB(sq, occupied)
	rookAttackers := rookAttacks & rooks
	if rookAttackers != 0 {
		return rookAttackers.LSB(), SEEPieceValues[WhiteRook]
	}

	// Queens
	queens := b.Pieces[pieceOf(WhiteQueen, color)] & occupied
	queenAttacks := bishopAttacks | rookAttacks
	queenAttackers := queenAttacks & queens
	if queenAttackers != 0 {
		return queenAttackers.LSB(), SEEPieceValues[WhiteQueen]
	}

	// King (highest value, only if no other attackers)
	kings := b.Pieces[pieceOf(WhiteKing, color)] & occupied
	kingAttackers := KingAttacks[sq] & kings
	if kingAttackers != 0 {
		return kingAttackers.LSB(), 20000 // High value, king can capture but is risky
	}

	return NoSquare, 0
}

// SEESign returns true if SEE >= threshold (faster than full SEE for pruning)
// Commonly used with threshold=0 to check if capture doesn't lose material
func (b *Board) SEESign(m Move, threshold int) bool {
	// Quick checks before doing full SEE
	from := m.From()
	to := m.To()

	captured := b.Squares[to]
	if captured == Empty && m.Flags() != FlagEnPassant {
		return threshold <= 0 // Not a capture
	}

	capturedValue := SEEPieceValues[captured]
	if m.Flags() == FlagEnPassant {
		capturedValue = SEEPieceValues[WhitePawn]
	}

	attacker := b.Squares[from]
	attackerValue := SEEPieceValues[attacker]

	// If we capture something worth more than our piece, it's definitely good
	if capturedValue >= attackerValue+threshold {
		return true
	}

	// If even capturing for free doesn't meet threshold, it's definitely bad
	if capturedValue < threshold {
		return false
	}

	// Need full SEE calculation
	return b.SEE(m) >= threshold
}

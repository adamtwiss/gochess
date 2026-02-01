package chess

import (
	"fmt"
	"strings"
)

// ParseSAN parses a move in Standard Algebraic Notation and returns the corresponding Move
// Examples: e4, Nf3, Bxc6, O-O, O-O-O, Qxf7+, e8=Q, Rad1, N1c3
func (b *Board) ParseSAN(san string) (Move, error) {
	// Remove check/checkmate indicators
	san = strings.TrimSuffix(san, "+")
	san = strings.TrimSuffix(san, "#")

	// Handle castling
	if san == "O-O" || san == "0-0" {
		if b.SideToMove == White {
			return NewMoveFlags(NewSquare(4, 0), NewSquare(6, 0), FlagCastle), nil
		}
		return NewMoveFlags(NewSquare(4, 7), NewSquare(6, 7), FlagCastle), nil
	}
	if san == "O-O-O" || san == "0-0-0" {
		if b.SideToMove == White {
			return NewMoveFlags(NewSquare(4, 0), NewSquare(2, 0), FlagCastle), nil
		}
		return NewMoveFlags(NewSquare(4, 7), NewSquare(2, 7), FlagCastle), nil
	}

	// Parse the SAN components
	var pieceType Piece
	var fromFile, fromRank int = -1, -1
	var toFile, toRank int
	var promotion Piece
	isCapture := false

	i := 0
	runes := []rune(san)

	// Check for piece type (uppercase letter, not a file)
	if i < len(runes) && runes[i] >= 'A' && runes[i] <= 'Z' && runes[i] != 'O' {
		switch runes[i] {
		case 'N':
			pieceType = WhiteKnight
		case 'B':
			pieceType = WhiteBishop
		case 'R':
			pieceType = WhiteRook
		case 'Q':
			pieceType = WhiteQueen
		case 'K':
			pieceType = WhiteKing
		default:
			return NoMove, fmt.Errorf("unknown piece: %c", runes[i])
		}
		i++
	} else {
		pieceType = WhitePawn
	}

	// Adjust piece for black
	if b.SideToMove == Black && pieceType != Empty {
		pieceType += 6
	}

	// Parse disambiguation and destination
	// Patterns: e4, exd5, Nf3, Nxf3, Nbd2, N1c3, Nbxd2, R1a3, Qh4xe1
	var coords []struct{ file, rank int }

	for i < len(runes) {
		if runes[i] == 'x' {
			isCapture = true
			i++
			continue
		}
		if runes[i] == '=' {
			// Promotion
			i++
			if i < len(runes) {
				switch runes[i] {
				case 'Q':
					promotion = WhiteQueen
				case 'R':
					promotion = WhiteRook
				case 'B':
					promotion = WhiteBishop
				case 'N':
					promotion = WhiteKnight
				}
				if b.SideToMove == Black && promotion != Empty {
					promotion += 6
				}
				i++
			}
			continue
		}
		if runes[i] >= 'a' && runes[i] <= 'h' {
			file := int(runes[i] - 'a')
			rank := -1
			i++
			if i < len(runes) && runes[i] >= '1' && runes[i] <= '8' {
				rank = int(runes[i] - '1')
				i++
			}
			coords = append(coords, struct{ file, rank int }{file, rank})
		} else if runes[i] >= '1' && runes[i] <= '8' {
			// Just a rank (rare disambiguation like N1c3)
			rank := int(runes[i] - '1')
			coords = append(coords, struct{ file, rank int }{-1, rank})
			i++
		} else {
			i++ // Skip unknown
		}
	}

	// Interpret coords: last one with both file and rank is destination
	// Earlier ones are disambiguation
	if len(coords) == 0 {
		return NoMove, fmt.Errorf("no destination square in: %s", san)
	}

	// Find destination (last complete coordinate)
	destIdx := -1
	for j := len(coords) - 1; j >= 0; j-- {
		if coords[j].rank != -1 && coords[j].file != -1 {
			destIdx = j
			break
		}
	}
	if destIdx == -1 {
		return NoMove, fmt.Errorf("incomplete destination in: %s", san)
	}

	toFile = coords[destIdx].file
	toRank = coords[destIdx].rank

	// Disambiguation from earlier coords
	for j := 0; j < destIdx; j++ {
		if coords[j].file != -1 {
			fromFile = coords[j].file
		}
		if coords[j].rank != -1 {
			fromRank = coords[j].rank
		}
	}

	to := NewSquare(toFile, toRank)

	// Find the matching legal move
	moves := b.GenerateLegalMoves()
	var candidates []Move

	for _, m := range moves {
		// Check piece type
		movingPiece := b.Squares[m.From()]
		if movingPiece != pieceType {
			continue
		}

		// Check destination
		if m.To() != to {
			continue
		}

		// Check capture (for pawns, captures must change file)
		if pieceType == WhitePawn || pieceType == BlackPawn {
			if isCapture && m.From().File() == m.To().File() {
				continue
			}
			if !isCapture && m.From().File() != m.To().File() {
				continue
			}
		}

		// Check disambiguation
		if fromFile != -1 && m.From().File() != fromFile {
			continue
		}
		if fromRank != -1 && m.From().Rank() != fromRank {
			continue
		}

		// Check promotion
		if promotion != Empty {
			promoPiece := m.PromotionPiece()
			if b.SideToMove == Black {
				promoPiece += 6
			}
			if promoPiece != promotion {
				continue
			}
		} else if m.IsPromotion() {
			// If no promotion specified but move is promotion, skip non-queen promotions
			// (standard convention is to default to queen)
			if m.Flags() != FlagPromoteQ {
				continue
			}
		}

		candidates = append(candidates, m)
	}

	if len(candidates) == 0 {
		return NoMove, fmt.Errorf("no legal move matches: %s", san)
	}
	if len(candidates) > 1 {
		return NoMove, fmt.Errorf("ambiguous move: %s (matches %d moves)", san, len(candidates))
	}
	return candidates[0], nil
}

// pieceToSANChar returns the SAN character for a piece type (white pieces)
func pieceToSANChar(p Piece) byte {
	switch p {
	case WhiteKnight:
		return 'N'
	case WhiteBishop:
		return 'B'
	case WhiteRook:
		return 'R'
	case WhiteQueen:
		return 'Q'
	case WhiteKing:
		return 'K'
	default:
		return 0
	}
}

// ToSAN converts a move to Standard Algebraic Notation
func (b *Board) ToSAN(m Move) string {
	if m == NoMove {
		return "--"
	}

	from := m.From()
	to := m.To()
	piece := b.Squares[from]
	captured := b.Squares[to]
	flags := m.Flags()

	// Handle castling
	if flags == FlagCastle {
		if to.File() == 6 {
			return "O-O"
		}
		return "O-O-O"
	}

	var san strings.Builder

	// Piece letter (not for pawns)
	pieceType := piece
	if piece >= BlackPawn {
		pieceType -= 6
	}

	if pieceType != WhitePawn {
		san.WriteByte(pieceToSANChar(pieceType))
	}

	// Disambiguation for non-pawns
	if pieceType != WhitePawn {
		// Check if other pieces of same type can reach same square
		moves := b.GenerateLegalMoves()
		needFile, needRank := false, false
		for _, other := range moves {
			if other == m {
				continue
			}
			if b.Squares[other.From()] == piece && other.To() == to {
				if other.From().File() != from.File() {
					needFile = true
				} else {
					needRank = true
				}
			}
		}
		if needFile {
			san.WriteByte(byte('a' + from.File()))
		}
		if needRank {
			san.WriteByte(byte('1' + from.Rank()))
		}
	}

	// Capture indicator
	isCapture := captured != Empty || flags == FlagEnPassant
	if isCapture {
		if pieceType == WhitePawn {
			san.WriteByte(byte('a' + from.File()))
		}
		san.WriteByte('x')
	}

	// Destination square
	san.WriteByte(byte('a' + to.File()))
	san.WriteByte(byte('1' + to.Rank()))

	// Promotion
	if m.IsPromotion() {
		san.WriteByte('=')
		promoPiece := m.PromotionPiece()
		san.WriteByte(pieceToSANChar(promoPiece))
	}

	// Check/checkmate indicator
	b.MakeMove(m)
	inCheck := b.InCheck()
	hasMoves := len(b.GenerateLegalMoves()) > 0
	b.UnmakeMove(m)

	if inCheck {
		if !hasMoves {
			san.WriteByte('#')
		} else {
			san.WriteByte('+')
		}
	}

	return san.String()
}

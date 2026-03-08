package chess

import (
	"chess/syzygy"
	"sync"
)

// Syzygy tablebase configuration
var (
	// SyzygyPath is the path to Syzygy tablebase files.
	// Empty string means tablebases are disabled.
	SyzygyPath string

	// SyzygyProbeDepth is the minimum depth to probe WDL tables during search.
	// Higher values reduce the overhead of probing at shallow depths.
	SyzygyProbeDepth = 1

	// SyzygyEnabled is true when tablebases are loaded and available.
	SyzygyEnabled bool

	// tbRootMu protects ProbeRoot calls which are NOT thread-safe in Fathom.
	// ProbeWDL is thread-safe and does not need this mutex.
	tbRootMu sync.Mutex
)

// SyzygyMaxPieceCount returns the largest tablebase available.
func SyzygyMaxPieceCount() int {
	return syzygy.MaxPieceCount
}

// TBScore constants for tablebase results in search
const (
	TBWinScore  = MateScore - 200 // Just below mate scores but above all eval scores
	TBLossScore = -TBWinScore
)

// SyzygyInit initializes the Syzygy tablebase with the configured path.
func SyzygyInit(path string) bool {
	if path == "" {
		SyzygyEnabled = false
		return false
	}
	ok := syzygy.Init(path)
	SyzygyEnabled = ok && syzygy.MaxPieceCount > 0
	SyzygyPath = path
	return SyzygyEnabled
}

// SyzygyFree releases Syzygy tablebase resources.
func SyzygyFree() {
	if SyzygyEnabled {
		syzygy.Free()
		SyzygyEnabled = false
	}
}

// tbPieceCount returns the number of pieces on the board.
func (b *Board) tbPieceCount() int {
	return b.AllPieces.Count()
}

// tbCanProbeWDL checks if this position can be probed for WDL.
func (b *Board) tbCanProbeWDL(depth int) bool {
	if !SyzygyEnabled {
		return false
	}
	if b.Castling != NoCastling {
		return false
	}
	cnt := b.tbPieceCount()
	if cnt > syzygy.MaxPieceCount {
		return false
	}
	if cnt == syzygy.MaxPieceCount && depth < SyzygyProbeDepth {
		return false
	}
	return true
}

// tbCanProbeRoot checks if this position can be probed at root (DTZ).
func (b *Board) tbCanProbeRoot() bool {
	if !SyzygyEnabled {
		return false
	}
	if b.Castling != NoCastling {
		return false
	}
	return b.tbPieceCount() <= syzygy.MaxPieceCount
}

// tbGetBitboards extracts the bitboard data needed by the Syzygy probing code.
func (b *Board) tbGetBitboards() (white, black, kings, queens, rooks, bishops, knights, pawns uint64, ep uint, isWhite bool) {
	white = uint64(b.Occupied[White])
	black = uint64(b.Occupied[Black])
	kings = uint64(b.Pieces[WhiteKing] | b.Pieces[BlackKing])
	queens = uint64(b.Pieces[WhiteQueen] | b.Pieces[BlackQueen])
	rooks = uint64(b.Pieces[WhiteRook] | b.Pieces[BlackRook])
	bishops = uint64(b.Pieces[WhiteBishop] | b.Pieces[BlackBishop])
	knights = uint64(b.Pieces[WhiteKnight] | b.Pieces[BlackKnight])
	pawns = uint64(b.Pieces[WhitePawn] | b.Pieces[BlackPawn])

	if b.EnPassant != NoSquare {
		ep = uint(b.EnPassant)
	}
	isWhite = b.SideToMove == White
	return
}

// TBProbeWDL probes the WDL table during search.
// Returns (score, ok) where score is relative to the side to move.
// Only call when tbCanProbeWDL() returns true.
func (b *Board) TBProbeWDL() (int, bool) {
	white, black, kings, queens, rooks, bishops, knights, pawns, ep, isWhite := b.tbGetBitboards()

	wdl, ok := syzygy.ProbeWDL(white, black, kings, queens, rooks, bishops, knights, pawns,
		uint(b.HalfmoveClock), uint(b.Castling), ep, isWhite)
	if !ok {
		return 0, false
	}

	switch wdl {
	case syzygy.WDLWin:
		return TBWinScore, true
	case syzygy.WDLCursedWin:
		return 1, true // Winning but drawn by 50-move rule
	case syzygy.WDLDraw:
		return 0, true
	case syzygy.WDLBlessedLoss:
		return -1, true // Losing but drawn by 50-move rule
	case syzygy.WDLLoss:
		return TBLossScore, true
	}
	return 0, false
}

// tbWDLToScore converts a raw WDL value to an engine score.
func tbWDLToScore(wdl int) int {
	switch wdl {
	case syzygy.WDLWin:
		return TBWinScore
	case syzygy.WDLCursedWin:
		return 1
	case syzygy.WDLDraw:
		return 0
	case syzygy.WDLBlessedLoss:
		return -1
	case syzygy.WDLLoss:
		return TBLossScore
	}
	return 0
}

// TBProbeRoot probes the DTZ table at the root to find the best move.
// Returns (move, wdl, dtz, ok). The move is matched against legal moves.
// Thread-safe: uses a mutex because Fathom's ProbeRoot is not reentrant.
func (b *Board) TBProbeRoot() (Move, int, int, bool) {
	if !b.tbCanProbeRoot() {
		return NoMove, 0, 0, false
	}

	white, black, kings, queens, rooks, bishops, knights, pawns, ep, isWhite := b.tbGetBitboards()

	tbRootMu.Lock()
	result, ok := syzygy.ProbeRoot(white, black, kings, queens, rooks, bishops, knights, pawns,
		uint(b.HalfmoveClock), uint(b.Castling), ep, isWhite)
	tbRootMu.Unlock()
	if !ok {
		return NoMove, 0, 0, false
	}

	// Match the Fathom move to our legal moves
	moves := b.GenerateLegalMoves()
	for _, m := range moves {
		from := int(m.From())
		to := int(m.To())
		if from == result.From && to == result.To {
			if result.Promotes != 0 {
				// Fathom: 1=queen, 2=rook, 3=bishop, 4=knight
				if !m.IsPromotion() {
					continue
				}
				flags := m.Flags()
				match := false
				switch result.Promotes {
				case 1:
					match = flags == FlagPromoteQ
				case 2:
					match = flags == FlagPromoteR
				case 3:
					match = flags == FlagPromoteB
				case 4:
					match = flags == FlagPromoteN
				}
				if !match {
					continue
				}
			}
			return m, result.WDL, result.DTZ, true
		}
	}

	return NoMove, 0, 0, false
}

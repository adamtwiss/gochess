//go:build !cgo

package syzygy

// Stub implementations when CGO is not available.
// All probes return failure so the engine runs without tablebase support.

// CGOAvailable indicates whether this binary was built with CGO support.
const CGOAvailable = false

var MaxPieceCount int

const (
	WDLLoss        = 0
	WDLBlessedLoss = 1
	WDLDraw        = 2
	WDLCursedWin   = 3
	WDLWin         = 4
)

func Init(path string) bool { return false }
func Free()                 {}

type RootResult struct {
	From     int
	To       int
	Promotes int
	WDL      int
	DTZ      int
}

func ProbeWDL(white, black, kings, queens, rooks, bishops, knights, pawns uint64,
	rule50 uint, castling uint, ep uint, isWhite bool) (int, bool) {
	return 0, false
}

func ProbeRoot(white, black, kings, queens, rooks, bishops, knights, pawns uint64,
	rule50 uint, castling uint, ep uint, isWhite bool) (RootResult, bool) {
	return RootResult{}, false
}

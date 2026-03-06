//go:build cgo

package syzygy

// #cgo CFLAGS: -O3 -std=gnu11 -w
// #include "tbwrappers.h"
// #include <stdlib.h>
import "C"
import (
	"strings"
	"unsafe"
)

// MaxPieceCount is the largest tablebase available (e.g. 5 for 5-piece TBs).
// Set after Init().
var MaxPieceCount int

// Init initializes the Syzygy tablebase with the given path.
// The path may contain multiple directories separated by ';' (Windows) or ':' (Unix).
func Init(path string) bool {
	cPath := C.CString(strings.TrimSpace(path))
	defer C.free(unsafe.Pointer(cPath))
	ok := C.tb_init(cPath)
	MaxPieceCount = int(C.TB_LARGEST)
	return bool(ok)
}

// Free releases resources allocated by Init.
func Free() {
	C.tb_free()
	MaxPieceCount = 0
}

// WDL result constants
const (
	WDLLoss       = C.TB_LOSS
	WDLBlessedLoss = C.TB_BLESSED_LOSS
	WDLDraw       = C.TB_DRAW
	WDLCursedWin  = C.TB_CURSED_WIN
	WDLWin        = C.TB_WIN
)

// ProbeWDL probes the Win-Draw-Loss table for the given position.
// Returns the WDL result and true if successful, or (0, false) on failure.
//
// Requirements for success:
// - No castling rights
// - Fifty-move counter is 0
// - Piece count <= MaxPieceCount
func ProbeWDL(white, black, kings, queens, rooks, bishops, knights, pawns uint64,
	rule50 uint, castling uint, ep uint, isWhite bool) (int, bool) {

	result := uint(C.tb_probe_wdl(
		C.uint64_t(white),
		C.uint64_t(black),
		C.uint64_t(kings),
		C.uint64_t(queens),
		C.uint64_t(rooks),
		C.uint64_t(bishops),
		C.uint64_t(knights),
		C.uint64_t(pawns),
		C.uint(rule50),
		C.uint(castling),
		C.uint(ep),
		C.bool(isWhite),
	))
	if result == uint(C.TB_RESULT_FAILED) {
		return 0, false
	}
	return int(result), true
}

// RootResult holds the result of a root DTZ probe.
type RootResult struct {
	From     int
	To       int
	Promotes int // 0=none, 1=queen, 2=rook, 3=bishop, 4=knight
	WDL      int
	DTZ      int
}

// ProbeRoot probes the DTZ table at the root position.
// Returns the result and true if successful.
// This function is NOT thread safe - call only once at root.
func ProbeRoot(white, black, kings, queens, rooks, bishops, knights, pawns uint64,
	rule50 uint, castling uint, ep uint, isWhite bool) (RootResult, bool) {

	result := uint(C.tb_probe_root(
		C.uint64_t(white),
		C.uint64_t(black),
		C.uint64_t(kings),
		C.uint64_t(queens),
		C.uint64_t(rooks),
		C.uint64_t(bishops),
		C.uint64_t(knights),
		C.uint64_t(pawns),
		C.uint(rule50),
		C.uint(castling),
		C.uint(ep),
		C.bool(isWhite),
		nil,
	))
	if result == uint(C.TB_RESULT_FAILED) ||
		result == uint(C.TB_RESULT_CHECKMATE) ||
		result == uint(C.TB_RESULT_STALEMATE) {
		return RootResult{}, false
	}

	return RootResult{
		From:     int(C.tb_get_from(C.uint(result))),
		To:       int(C.tb_get_to(C.uint(result))),
		Promotes: int(C.tb_get_promotes(C.uint(result))),
		WDL:      int(C.tb_get_wdl(C.uint(result))),
		DTZ:      int(C.tb_get_dtz(C.uint(result))),
	}, true
}

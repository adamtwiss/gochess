# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Build and Test Commands

```bash
go test                    # Run all tests (includes WAC/ECM suites, ~3min)
go test -v                 # Verbose output
go test -run TestX         # Run a specific test
go test -bench .           # Run benchmarks
go test -run 'Test(Print|Zobrist|Bitboard|Perft|SAN|Search|SEE|Eval)' # Skip EPD suites

go build -o chess ./cmd/chess    # Build CLI binary
./chess -e testdata/wac.epd -t 5000 -n 20   # Run EPD test suite
```

## Project Structure

Chess engine in Go using bitboard representation. Core library is `package chess` in the root; CLI is `cmd/chess/`.

```
cmd/chess/main.go    CLI entry point (EPD test runner)
testdata/            EPD test suite files (wac.epd, ecm.epd)
board.go             Board struct, piece types, FEN parsing, pieceOf() helper
move.go              Move encoding (16-bit), flags, NoMove sentinel
bitboard.go          Bitboard type, bit manipulation, file/rank masks
attacks.go           Magic bitboard tables, pre-computed attack lookups
movegen.go           Pseudo-legal/legal move generation, IsAttacked, InCheck
makemove.go          MakeMove/UnmakeMove, null move, UndoInfo
movepicker.go        Staged move ordering for search, IsPseudoLegal
search.go            Negamax, alpha-beta, iterative deepening, LMR, null-move pruning
eval.go              Material + mobility evaluation
tt.go                Transposition table (depth-preferred replacement)
zobrist.go           Zobrist hash keys, incremental hashing
see.go               Static Exchange Evaluation for capture ordering
san.go               SAN parsing (ParseSAN) and formatting (ToSAN)
epd.go               EPD file loading and test suite runner
```

## Architecture

### Board Representation

- **Hybrid storage**: `Board.Squares[64]` for piece-by-square lookup, `Board.Pieces[13]` bitboards for piece-type iteration
- **Occupancy**: `Board.Occupied[2]` (by color) and `Board.AllPieces` for attack generation
- **Undo stack**: `Board.UndoStack []UndoInfo` stores captured piece, castling, en passant, halfmove clock, hash per move

### Piece Indexing

Pieces 1-12 (Empty=0). White 1-6 (Pawn..King), Black 7-12. Converting: `blackPiece = whitePiece + 6`. Use `pieceOf(WhiteKnight, color)` instead of raw arithmetic.

### Squares

0-63: `square = rank*8 + file` (a1=0, h1=7, a8=56, h8=63). Use `NewSquare(file, rank)`.

### Move Encoding

16 bits: bits 0-5 = from, 6-11 = to, 12-15 = flags.

Flag values: `FlagNone=0, FlagEnPassant=1, FlagCastle=2, FlagPromoteN=4, FlagPromoteB=5, FlagPromoteR=6, FlagPromoteQ=7`. `FlagPromotion=4` is a bitmask for "any promotion".

**Critical rule**: Check non-promotion flags with equality (`flags == FlagEnPassant`), not bitwise AND. Promotion flags 4-7 have bit 0 set for some values, so `flags & FlagEnPassant != 0` gives false positives on promotions. `IsPromotion()` using `flags & FlagPromotion` is safe because bit 2 is never set in en passant or castle flags.

### Move Generation

Magic bitboards for sliding pieces (attacks.go). Pre-computed tables for knight, king, pawn attacks.

- `GenerateAllMoves()` — pseudo-legal moves (fast, doesn't verify king safety)
- `GenerateLegalMoves()` — filters via `IsLegal()` (make/unmake + king check)
- `GenerateCaptures()` — captures + promotions (for quiescence)
- `GenerateQuiets()` — non-captures, non-promotions

### Move Ordering (MovePicker)

Staged generation for search efficiency:

1. TT move (from transposition table)
2. Good captures (SEE >= 0, scored by MVV-LVA)
3. Killer moves (2 per ply, caused beta cutoffs in sibling nodes)
4. Quiet moves (scored by history heuristic)
5. Bad captures (SEE < 0, last resort)

Selection sort within each stage (partial sort, only finds next-best on demand).

### Search

Negamax with alpha-beta pruning, iterative deepening with time control.

- **Transposition table**: Probe before search, store after. Depth-preferred replacement. Mate scores adjusted by ply distance to prevent stale mate evaluations.
- **Null-move pruning**: Skip turn and search with reduced depth (R=3 if depth>=7, else R=2). Disabled when in check.
- **Late Move Reductions (LMR)**: Reduce quiet moves searched late in the move list by 1 ply (moveCount >= 8, depth >= 5). Re-search at full depth if score exceeds alpha. Disabled for captures, promotions, killers, check-giving moves, and high-history moves.
- **Quiescence search**: Captures only at leaf nodes, pruned by SEE >= 0. Stand-pat evaluation as lower bound. Depth-limited to 32.
- **Killer moves**: 2 slots per ply, updated on beta cutoff with quiet moves.
- **History heuristic**: `history[from][to] += depth * depth` on beta cutoff. Used to score quiet moves in move ordering.
- **Time management**: Checks clock every 4096 nodes. Iterative deepening allows stopping between depths.

### Evaluation

Centipawn-based. Material (P=100, N=320, B=330, R=500, Q=900) plus mobility bonuses per attacked square (N=4, B=5, R=2, Q=1 centipawns). `Evaluate()` returns White-relative; `EvaluateRelative()` returns side-to-move relative.

### Zobrist Hashing

Incrementally updated in `MakeMove`/`UnmakeMove` via XOR. Keys cover piece-square, side to move, castling rights (4 individual keys), and en passant file (8 keys, by file not square). Full recompute only in `SetFEN()`. Fixed seed for deterministic hashing.

### SEE (Static Exchange Evaluation)

Simulates alternating captures on a single square. Builds gain array, then negamax backward to find optimal result. `SEESign(move, threshold)` provides fast boolean check with early exits. Used in quiescence pruning and move ordering.

### EPD / Test Suites

`LoadEPDFile()` parses EPD format (4-field FEN + operations like `bm`, `am`, `id`). `RunEPDTest()` searches a position and checks if found move matches expected best move(s). Test suites: WAC (300 positions), ECM (201 positions) in `testdata/`.

### CLI

`cmd/chess/main.go` — EPD test runner. Flags: `-e` (EPD file), `-t` (ms per position), `-n` (max positions), `-d` (max depth), `-hash` (TT size MB).

## Key Gotchas

- Move flag equality vs bitwise: see "Critical rule" above.
- `UnmakeMove()` panics on undo stack mismatch — always pair with `MakeMove()`.
- `GenerateAllMoves()` returns pseudo-legal moves; search must call `IsLegal()`.
- Castling rights are lost both when a rook/king moves AND when a rook is captured on its home square.
- En passant hash uses file only (8 keys), not full square.
- TT mate scores are ply-adjusted: stored as `mate + ply`, retrieved as `mate - ply`.

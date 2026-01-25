# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
go test              # Run all tests
go test -v           # Run tests with verbose output
go test -run TestX   # Run a specific test
go test -bench .     # Run benchmarks
```

## Architecture

This is a chess engine written in Go using bitboard representation. All code is in the `chess` package.

### Board Representation

- **Bitboard-based**: Uses 64-bit integers where each bit represents a square (bit 0 = a1, bit 63 = h8)
- **Hybrid storage**: `Board.Squares[64]` array for piece lookup by square, plus `Board.Pieces[13]` bitboards for piece-type iteration
- **Occupancy tracking**: `Board.Occupied[2]` (by color) and `Board.AllPieces` bitboards for fast attack generation

### Key Type Mappings

Pieces are indexed 1-12 (Empty=0, WhitePawn=1...WhiteKing=6, BlackPawn=7...BlackKing=12). Converting between white and black pieces: `blackPiece = whitePiece + 6`.

Squares are 0-63 with `square = rank*8 + file`. Use `NewSquare(file, rank)` to create.

Moves are encoded in 16 bits: bits 0-5 = from, bits 6-11 = to, bits 12-15 = flags.

### Move Generation

Uses magic bitboards for sliding pieces (rook/bishop/queen). Attack tables are pre-computed at init time in `attacks.go`.

Move generation produces pseudo-legal moves via `GenerateAllMoves()`. Legal move filtering happens in `GenerateLegalMoves()` by making each move and checking if the king is in check.

**Important**: Move flags must be checked with equality (`flags == FlagEnPassant`), not bitwise AND, because promotion flags (4-7) contain the en passant bit pattern.

### Search

- Negamax with alpha-beta pruning (`search.go`)
- Iterative deepening with time control
- Quiescence search for captures at leaf nodes
- Move ordering: TT move first, then MVV-LVA for captures
- Transposition table with Zobrist hashing (`tt.go`)

### Zobrist Hashing

Hash keys are incrementally updated in `MakeMove`/`UnmakeMove`. The hash includes piece positions, side to move, castling rights, and en passant file.

### Evaluation

Simple evaluation in centipawns (`eval.go`): material counting plus mobility bonuses per piece type.

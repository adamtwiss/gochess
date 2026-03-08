# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Build and Test Commands

```bash
go test                    # Run all tests (includes WAC/ECM suites, ~3min)
go test -v                 # Verbose output
go test -run TestX         # Run a specific test
go test -bench .           # Run benchmarks
go test -short             # Run unit tests only (skips slow EPD suites)

go build -o chess ./cmd/chess    # Build CLI binary
go build -o tuner ./cmd/tuner   # Build Texel tuner binary
./chess -e testdata/wac.epd -t 5000 -n 20              # Run EPD test suite
./chess -e testdata/wac.epd -t 5000 -n 20 -threads 4   # Run EPD with Lazy SMP (4 threads)
./chess -uci                                            # Start UCI mode
./chess -benchmark -t 200                               # Run multi-suite benchmark (quick)
./chess -benchmark -t 200 -save base.json               # Save benchmark results to JSON
./chess -benchmark -t 200 -compare base.json            # Compare against saved baseline
./chess -buildbook -pgn testdata/2600.pgn -bookout book.bin  # Build Polyglot opening book

./tuner selfplay -games 20000 -time 200 -concurrency 6 -output training.dat  # Generate training data
./tuner tune -data training.dat -epochs 500 -lr 1.0                          # Tune eval parameters
./tuner tune -data training.dat -epochs 500 -lr 1.0 -lambda 0.5             # Tune with blended loss (default lambda=0)
./tuner nnue-train -data training.dat -epochs 100 -lr 0.01 -output net.nnue          # Train NNUE
./tuner nnue-train -data training.dat -resume net-v1.nnue -epochs 100 -output net-v2.nnue # Resume

./chess -nnue net.nnue -uci                             # UCI with NNUE
./chess -syzygy /path/to/tablebases -uci                # UCI with Syzygy tablebases

# Self-play testing with cutechess-cli (ALWAYS use these patterns)
cutechess-cli \
  -tournament gauntlet \
  -engine name=GoChess-new cmd=./chess proto=uci \
  -engine name=GoChess-old cmd=./chess.older proto=uci \
  -each tc=0/10+0.1 option.Hash=64 option.MoveOverhead=100 \
  -rounds 200 -concurrency 4 \
  -openings file=testdata/noob_3moves.epd format=epd order=random \
  -pgnout gauntlet.pgn -recover -ratinginterval 20 \
  -draw movenumber=20 movecount=10 score=10 \
  -resign movecount=3 score=500 twosided=true

# SPRT testing (stops early when statistically significant):
cutechess-cli \
  -tournament gauntlet \
  -engine name=GoChess-new cmd=./chess arg=-uci \
  -engine name=GoChess-old cmd=./chess.older arg=-uci \
  -each tc=0/10+0.1 option.Hash=64 \
  -rounds 5000 -concurrency 8 \
  -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 \
  -openings file=testdata/noob_3moves.epd format=epd order=random \
  -pgnout sprt.pgn -recover -ratinginterval 20
```

## Project Structure

Chess engine in Go using bitboard representation. Core library is `package chess` in the root; CLI is `cmd/chess/`.

```
cmd/chess/main.go    CLI entry point (EPD runner, UCI mode, book builder, interactive CLI)
cmd/tuner/main.go    Texel tuner CLI (selfplay data generation, parameter optimization)
testdata/            EPD suites (wac, ecm, arasan, lct, sbd, etc.), noob_3moves.epd, 2600.pgn
board.go             Board struct, piece types, FEN parsing
move.go              Move encoding (16-bit), flags, NoMove sentinel
bitboard.go          Bitboard type, bit manipulation, masks
attacks.go           Magic bitboard tables, attack lookups
movegen.go           Pseudo-legal/legal move generation, evasions
makemove.go          MakeMove/UnmakeMove, null move, UndoInfo
movepicker.go        Staged move ordering for search
search.go            Negamax, alpha-beta, iterative deepening, Lazy SMP
eval.go              Tapered eval: PST, mobility, king safety, positional bonuses
pst.go               PeSTO piece-square tables, material values
pawns.go             Pawn structure eval, pawn hash table, pawn shield
tt.go                Transposition table (lockless, 4-slot buckets)
zobrist.go           Zobrist hash keys, incremental hashing
see.go               Static Exchange Evaluation
san.go               SAN parsing and formatting
epd.go               EPD file loading and test suite runner
pgn.go               PGN game parsing
benchmark.go         Multi-suite benchmark with JSON save/compare
book.go / polyglot.go  Polyglot opening book (build and load)
uci.go               UCI protocol implementation
cli.go               Interactive CLI engine
selfplay.go          Self-play game generation for tuning data
tuner.go             Texel tuner: parameter catalog, traces, .tbin cache, Adam optimizer
nnue.go              NNUE inference: HalfKP network, lazy accumulators, incremental updates
nnue_amd64.go/s      AVX2 SIMD (runtime detected)
nnue_arm64.go/s      NEON SIMD
nnue_nosimd.go       Fallback stubs
nnue_train.go        NNUE training: backprop, Adam, quantization
tb.go                Syzygy tablebase integration (WDL/DTZ probing)
syzygy/              Fathom C library (CGO wrapper)
```

## Key Conventions

### Piece Indexing
Pieces 1-12 (Empty=0). White 1-6 (Pawn..King), Black 7-12. Use `pieceOf(WhiteKnight, color)` instead of raw `+6` arithmetic.

### Squares
0-63: `square = rank*8 + file` (a1=0, h8=63). Use `NewSquare(file, rank)`.

### Move Encoding
16 bits: bits 0-5 = from, 6-11 = to, 12-15 = flags. Flags: `FlagNone=0, FlagEnPassant=1, FlagCastle=2, FlagPromoteN=4, FlagPromoteB=5, FlagPromoteR=6, FlagPromoteQ=7`.

**Critical rule**: Check non-promotion flags with equality (`flags == FlagEnPassant`), not bitwise AND. Promotion flags 4-7 have bit 0 set for some values, so `flags & FlagEnPassant != 0` gives false positives. `IsPromotion()` using `flags & FlagPromotion` is safe.

### Board Representation
- Hybrid: `Board.Squares[64]` (piece-by-square) + `Board.Pieces[13]` (bitboards per piece type)
- Incremental PST: `Board.PSTScoreMG/EG[color]` updated by `putPiece`/`removePiece`/`movePiece`
- Undo stack: `Board.UndoStack []UndoInfo` for MakeMove/UnmakeMove pairing

### Move Generation
- `GenerateAllMoves()` returns pseudo-legal moves; search must call `IsLegal()`
- `GenerateEvasionsAppend()` produces fully legal evasions when in check (no IsLegal needed)
- `IsLegal(m, pinned, inCheck)` uses pin-aware fast paths; `PinnedAndCheckers()` called once per node

### Search Overview
Negamax with alpha-beta, iterative deepening, PVS, aspiration windows. Features: null-move pruning, reverse futility, futility, LMR, LMP, SEE pruning, ProbCut, history pruning, IIR, singular extensions, check/recapture/passed-pawn extensions. Quiescence with SEE filtering and evasion handling. Move ordering: TT move -> good captures -> killers -> counter-move -> quiets -> bad captures. History tables: main history, capture history, continuation history.

### Lazy SMP
All threads share only the TT (lockless via XOR-verified packed atomics). Board, SearchInfo, pawn table, and NNUE accumulator stack are per-thread.

### NNUE
HalfKP (40960 -> 2x256 -> 32 -> 32 -> 1). Lazy accumulator: MakeMove stores deltas, Materialize() applies on demand (saves ~17% NPS). King moves trigger full recompute. SIMD: AVX2 (x86-64, runtime detected) and NEON (ARM64).

### Texel Tuner
~1268 parameters optimized via Adam. Training data: `FEN;score;result` from selfplay, preprocessed to `.tbin` binary cache, disk-streamed. `computeTrace()` mirrors `Evaluate()` recording sparse coefficients. Frozen params: material (coupling), tempo, trade bonuses. PST values are pre-scaled (effective = raw * scale/100).

### Syzygy Tablebases
Via bundled Fathom (CGO). Root: DTZ probe before search. Interior nodes: WDL probe (requires HalfmoveClock==0). Fathom's `ProbeRoot` is NOT thread-safe (main thread only); `tb_probe_wdl` IS thread-safe.

## Key Gotchas

- Move flag equality vs bitwise: see "Critical rule" above
- `UnmakeMove()` panics on empty undo stack or move mismatch — always pair with `MakeMove()`
- Castling rights lost when rook/king moves AND when rook captured on home square
- En passant hash uses file only (8 keys), not full square
- TT mate scores are ply-adjusted: stored as `mate + ply`, retrieved as `mate - ply`
- TT `Probe`/`Store` are lockless via packed atomics — do not add non-atomic fields to `ttSlot`
- Lazy SMP: `Stopped` and `Deadline` accessed atomically. New shared state must use atomics or be per-thread
- Tuner traces must mirror `Evaluate()` exactly. When modifying eval, update `computeTrace()` to match
- `.tbin` cache must be rebuilt (delete or touch `.dat`) when `computeTrace()` or param catalog changes
- `StreamTraining`/`StreamValidation` callbacks must not retain `[]TunerTrace` batch references (reused)
- NNUE accumulator stack must stay in sync with undo stack (push on MakeMove, pop on UnmakeMove, null moves skip)
- NNUE `putPiece`/`removePiece`/`movePiece` hooks read king bitboards — call when kings are on the board
- NNUE incremental updates skip kings; king moves trigger `RecomputeAccumulator`. Castling moves king first, then rook
- `.nnbin` cache must be rebuilt when training data format or feature indexing changes
- Syzygy `tbchess.inc` is `#include`d by `tbprobe.c` — must NOT be compiled separately
- Syzygy WDL probes require `HalfmoveClock == 0`; DTZ probes accept any value
- **cutechess-cli**: Each flag and value must be separate `arg=` params (`arg=-nnue arg=/path/to/net.nnue`). `-uci` flag NOT needed (auto-detected). Use absolute paths

## Maintenance Reminders

- **Keep CLAUDE.md and README.md up to date** when changing search, eval, tuner, CLI, or architecture
- **New eval parameter**: Add to `initTunerParams()`, `computeTrace()`, `PrintParams()`. Update "What's tuned" lists. Delete `.tbin` cache

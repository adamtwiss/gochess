# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Build and Test Commands

```bash
go test                    # Run all tests (includes WAC/ECM suites, ~3min)
go test -v                 # Verbose output
go test -run TestX         # Run a specific test
go test -bench .           # Run benchmarks
go test -short             # Run unit tests only (skips slow EPD suites)

go build -o chess ./cmd/chess    # Build engine binary
go build -o tuner ./cmd/tuner   # Build Texel tuner binary
./chess fetch-net                                       # Download NNUE net from GitHub releases
./chess -e testdata/wac.epd -t 5000 -n 20              # Run EPD test suite
./chess -e testdata/wac.epd -t 5000 -n 20 -threads 4   # Run EPD with Lazy SMP (4 threads)
./chess                                                 # Start UCI mode (default)
./chess -benchmark -t 200                               # Run multi-suite benchmark (quick)
./chess -benchmark -t 200 -save base.json               # Save benchmark results to JSON
./chess -benchmark -t 200 -compare base.json            # Compare against saved baseline
./chess -buildbook -pgn testdata/2600.pgn -bookout book.bin  # Build Polyglot opening book

./tuner selfplay -games 20000 -time 200 -concurrency 6                       # Generate training data (.bin)
./tuner selfplay -games 20000 -time 200 -concurrency 6 -syzygy /path/to/tb  # With Syzygy tablebases
./tuner tune -data training.bin -epochs 500 -lr 1.0                          # Tune eval parameters
./tuner tune -data training.bin -epochs 500 -lr 1.0 -lambda 0.5             # Tune with blended loss (default lambda=0)
./tuner nnue-train -data training.bin -epochs 100 -lr 0.01 -output net.nnue  # Train NNUE
./tuner nnue-train -data a.bin,b.bin -epochs 100 -lr 0.01                    # Train from multiple files
./tuner nnue-train -data training.bin -resume net-v1.nnue -epochs 100 -output net-v2.nnue # Resume
cat a.bin b.bin > combined.bin                                               # Concatenate .bin files

./tuner convert-binpack -input data.binpack -output data.bin                  # Convert SF .binpack to .bin
./tuner convert-binpack -input data.binpack                                   # Output defaults to data.bin
./tuner rescore -data training.bin -depth 10 -concurrency 8 -hash 512        # Rescore .bin in-place
./tuner rescore -data training.bin -depth 8 -concurrency 4 -syzygy /path/to/tb  # Rescore with Syzygy
./tuner shuffle -data training.bin                                            # Shuffle .bin in-place
./tuner check-net -net net.nnue                                              # NNUE health check
./tuner compare-nets -net1 a.nnue -net2 b.nnue                              # Compare two networks
./tuner convert-net -input old.nnue -output new.nnue                         # Convert net versions

./chess -nnue net.nnue                                   # UCI with specific NNUE net
./chess -syzygy /path/to/tablebases                      # UCI with Syzygy tablebases

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
  -engine name=GoChess-new cmd=./chess proto=uci \
  -engine name=GoChess-old cmd=./chess.older proto=uci \
  -each tc=0/10+0.1 option.Hash=64 \
  -rounds 5000 -concurrency 8 \
  -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 \
  -openings file=testdata/noob_3moves.epd format=epd order=random \
  -pgnout sprt.pgn -recover -ratinginterval 20
```

## Project Structure

Chess engine in Go using bitboard representation. Core library is `package chess` in the root; entry point is `cmd/chess/`. NNUE net files are hosted in GitHub releases (not committed to git); `net.txt` references the current net URL.

```
cmd/chess/main.go    Entry point (UCI mode, EPD runner, book builder, fetch-net)
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
netload.go           NNUE net loading from net.txt (URL resolution, fetch-net support)
selfplay.go          Self-play game generation for tuning data (binpack output)
rescore.go           Rescore .bin training data with deeper search
binpack.go           Fixed-size binary training data format (32 bytes/record, no header)
tuner.go             Texel tuner: parameter catalog, traces, .tbin cache, Adam optimizer
nnue.go              NNUE inference: HalfKA network, lazy accumulators, incremental updates
nnue_v5.go           NNUE v5 inference: shallow wide net (Bullet GPU training), CReLU/SCReLU, pairwise
nnue_amd64.go/s      AVX2 SIMD (runtime detected) — v4 and v5 kernels
nnue_arm64.go/s      NEON SIMD
nnue_nosimd.go       Fallback stubs
nnue_train.go        NNUE training: backprop, Adam, quantization
sfbinpack.go         Stockfish .binpack format reader (legacy PackedSfen chain + modern BINP chunk)
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
Negamax with alpha-beta, iterative deepening, PVS, aspiration windows. Features: null-move pruning, reverse futility, futility, LMR (separate quiet/capture tables, threat-aware adjustments), LMP, SEE pruning, ProbCut, history pruning, IIR, hindsight reduction (reduce quiet positions where both evals are high), singular extensions, recapture extensions, alpha-reduce (reduce after alpha raise), failing heuristic (aggressive pruning when eval is deteriorating), fail-high score blending (dampen inflated cutoff scores). Quiescence with SEE filtering, evasion handling, and beta blending. Move ordering: TT move -> good captures -> killers -> counter-move -> quiets -> bad captures. History tables: main history, capture history, continuation history, pawn-hash correction history.

### Lazy SMP
All threads share only the TT (lockless via XOR-verified packed atomics). Board, SearchInfo, pawn table, and NNUE accumulator stack are per-thread.

### NNUE
Two architectures supported, selected by network file version:

- **v4 (HalfKA deep)**: 12288 → 2×256 → 32 → 32 → 8. Two hidden layers with int8 quantized layer 1 (VPMADDUBSW/SMULL+SADALP for doubled throughput). 16 king buckets × 12 piece types × 64 squares = 12288 inputs. 8 material-based output buckets.
- **v5 (Bullet shallow wide)**: (12288 → N)×2 → 1×8. Single hidden layer, dynamic width (1024/1536/any, auto-detected from file). Supports CReLU and SCReLU activations, optional pairwise multiplication (halves effective width). Quantization: QA=255, QB=64. Designed for Bullet GPU trainer.

Both share: lazy accumulator (MakeMove stores deltas, Materialize() applies on demand), incremental updates skip kings. SIMD: AVX2 (x86-64, runtime detected) and NEON (ARM64). Width-generic SIMD kernels (`nnueAccAddN`, `nnueAccSubN`, `nnueAccSubAddN`, `nnueAccCopySubAddN`, `nnueAccCopySubSubAddN`) process the full hidden width in a single call, avoiding per-256-chunk function call overhead.

Finny table: per-perspective `[NNUEKingBuckets][mirror]` cache of accumulated weights and piece bitboards. On king bucket changes, `RefreshAccumulator` diffs cached vs current piece bitboards and applies only changed features (~5 delta ops vs ~30 full recompute ops). Invalidated on SetFEN/Reset. Deep-copied for Lazy SMP threads.

### Training Data Formats
- **Binpack** (`.bin`): Fixed-size 32-byte records, no file header. Stores packed board position (occupancy bitmap + piece nibbles), score (int16), result (uint8). Files can be concatenated with `cat`. Features extracted at training time. Block-shuffled reader (64KB = 2048 records/block) for efficient I/O.

### Texel Tuner
~1268 parameters optimized via Adam. Training data: `.bin` binpack from selfplay, preprocessed to `.tbin` binary cache, disk-streamed. `computeTrace()` mirrors `Evaluate()` recording sparse coefficients. Frozen params: material (coupling), tempo, trade bonuses. PST values are pre-scaled (effective = raw * scale/100).

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
- `.tbin` cache must be rebuilt (delete the `.tbin` file) when `computeTrace()` or param catalog changes
- `.bin` files have no header — file size must be a multiple of 32. Features are extracted from packed positions at training time, so binpack data survives feature set changes
- `StreamTraining`/`StreamValidation` callbacks must not retain `[]TunerTrace` batch references (reused)
- NNUE accumulator stack must stay in sync with undo stack (push on MakeMove, pop on UnmakeMove, null moves skip)
- NNUE `putPiece`/`removePiece`/`movePiece` hooks read king bitboards — call when kings are on the board
- NNUE incremental updates skip kings; king moves trigger `RecomputeAccumulator`. Castling moves king first, then rook
- NNUE v5 hidden size is auto-detected from file — do not hardcode dimensions
- Syzygy `tbchess.inc` is `#include`d by `tbprobe.c` — must NOT be compiled separately
- Syzygy WDL probes require `HalfmoveClock == 0`; DTZ probes accept any value
- **cutechess-cli**: Each flag and value must be separate `arg=` params (`arg=-nnue arg=/path/to/net.nnue`). Use `proto=uci`. Use absolute paths

## Search/Eval Optimization Workflow

Changes to search, eval, or move ordering must be validated by self-play Elo, not just benchmarks or NPS. Follow this workflow:

1. **Work in a worktree** — `isolation: worktree` for agents, or manual `git worktree add`. Keep main clean until the change is proven.
2. **Build a binary** — `go build -o chess-<variant> ./cmd/chess` from the worktree.
3. **SPRT test against HEAD** — non-regression test with both engines using identical settings (same NNUE net, same hash, `OwnBook=false`):
   ```bash
   cutechess-cli -tournament gauntlet \
     -engine name=New cmd=./chess-new proto=uci option.UseNNUE=true option.NNUEFile=$(pwd)/net.nnue option.OwnBook=false option.Hash=64 option.MoveOverhead=100 \
     -engine name=Base cmd=./chess-base proto=uci option.UseNNUE=true option.NNUEFile=$(pwd)/net.nnue option.OwnBook=false option.Hash=64 option.MoveOverhead=100 \
     -each tc=0/10+0.1 \
     -rounds 5000 -concurrency 6 \
     -sprt elo0=-10 elo1=5 alpha=0.05 beta=0.05 \
     -openings file=testdata/noob_3moves.epd format=epd order=random \
     -pgnout sprt_test.pgn -recover -ratinginterval 20
   ```
4. **Be patient** — small gains (+5 Elo) are real and valuable. Stockfish gained hundreds of Elo from many +5 patches. SPRT at this level needs 1000-3000 games to converge.
5. **Tune before rejecting** — if a sound idea tests negative, try 2-3 parameter variants before giving up. Run variants in parallel at lower concurrency.
6. **One change at a time** — never stack untested changes. Each commit should be independently validated.
7. **Merge on SPRT acceptance** — merge the worktree branch to main, commit, push. Then move to the next idea.
8. **Log results in `experiments.md`** — record every experiment (pass or fail) with the change, SPRT result, baseline, and lessons learned. This builds institutional knowledge and prevents re-testing failed ideas.

CPU budget: on a 16-thread machine, 7 concurrency is the max for 10s+0.1s games without time-pressure noise. Two parallel experiments at 4 each is fine.

### Research Pipeline (Parallel SPRT)

For sustained optimization, run a continuous pipeline of 4 parallel experiments (4 threads each = 16 total):

1. **Build all variants in worktrees** — `git worktree add .claude/worktrees/<name> HEAD`, edit the parameter, build with `go build -o chess-<name> ./cmd/chess`.
2. **Run SPRTs in background** — redirect output to `sprt_<name>.log`. Monitor with `tail` on the log files.
3. **Check results hourly** — parse the last `SPRT:` and `Elo difference:` lines from each log. Report a summary table (games, Elo, LLR, status).
4. **When an experiment finishes** (H0 or H1 accepted), immediately:
   - Log the result in `experiments.md`
   - Clean up the worktree and binary
   - Queue a replacement experiment to keep all 16 threads busy
5. **When an experiment is rejected (negative Elo)**, consider testing the opposite direction. If tightening a margin loses Elo, try loosening it — a strong negative result implies the optimum may be on the other side. If both directions lose, the parameter is well-calibrated.
6. **When an experiment is accepted (positive Elo)**, merge to main, rebuild the baseline binary, and consider testing a further step in the same direction to find the true optimum.
7. **Bracket the optimum** — for any parameter change that gains Elo, test values on both sides (e.g. if X=50 gains, test X=40 and X=60). Two zero-Elo results on either side confirms you've found the peak.

Key principles:
- **Report progress proactively** — post a summary table (game count, Elo, LLR, status) at least every hour, and immediately when any experiment reaches a conclusion (H0/H1). The user may be monitoring via remote control and needs regular updates without having to ask.
- **Never leave threads idle** — always have experiments queued. Small gains compound.
- **Test one parameter at a time per experiment** — but run multiple independent experiments in parallel.
- **Revisit failed experiments when conditions change** — a new NNUE net or new search feature can shift optimal parameters. Check `experiments.md` before re-testing.
- **Self-play Elo ≈ 2-3x cross-engine Elo** for search changes. A +5 self-play gain is real and worth merging.

## Maintenance Reminders

- **Keep CLAUDE.md and README.md up to date** when changing search, eval, tuner, CLI, or architecture
- **New eval parameter**: Add to `initTunerParams()`, `computeTrace()`, `PrintParams()`. Update "What's tuned" lists. Delete `.tbin` cache

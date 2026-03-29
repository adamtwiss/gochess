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

./tuner check-net -net net.nnue                                              # NNUE health check
./tuner compare-nets -net1 a.nnue -net2 b.nnue                              # Compare two networks
./tuner convert-bullet -v5 -input quantised.bin -output net.nnue             # Convert Bullet → v5
./tuner convert-bullet -v5 -input quantised.bin -output net.nnue -screlu     # Convert Bullet → v5 SCReLU
./tuner convert-bullet -v5 -input quantised.bin -output net.nnue -hidden 16 -hidden2 32 -screlu  # v7 with L1+L2

# Legacy trainer (selfplay, tune, nnue-train) — see docs/legacy_trainer.md

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
  -rounds 5000 -concurrency 16 \
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
Three architecture generations, selected by network file version:

- **v4 (HalfKA deep)**: 12288 → 2×256 → 32 → 32 → 8. Legacy, not actively used.
- **v5/v6 (Bullet direct output)**: (12288 → N)×2 → 1×8. Dynamic width (1024/1536). CReLU or SCReLU (v6 adds flags byte). Quantization: QA=255, QB=64. Production net is v5 1024 CReLU.
- **v7 (Bullet with hidden layers)**: (12288 → N)×2 → L1 → [L2 →] 1×8. Adds explicit hidden layers between accumulator and output. Header stores FTSize, L1Size, L2Size. Target architecture: 1024 → 16 → 32 → 1×8 SCReLU (matching top engines).

All share: lazy accumulator (MakeMove stores deltas, Materialize() applies on demand), incremental updates skip kings. SIMD: AVX2 (x86-64, runtime detected) and NEON (ARM64). Width-generic SIMD kernels for accumulator updates and forward pass.

Finny table: per-perspective `[NNUEKingBuckets][mirror]` cache of accumulated weights and piece bitboards. On king bucket changes, `RefreshAccumulator` diffs cached vs current piece bitboards and applies only changed features (~5 delta ops vs ~30 full recompute ops). Invalidated on SetFEN/Reset. Deep-copied for Lazy SMP threads.

### NNUE Training (Bullet GPU)

We train on **Bullet** (Rust, CUDA) using T80 binpack data (~12B positions across 6 files). Training produces `quantised.bin` which is converted to `.nnue` via `./tuner convert-bullet`.

Key findings:
- **CReLU kills hidden layer neurons** during long training (dying ReLU). At e800-s400, 15/16 L1 neurons were dead. SCReLU prevents this — all 16 survived e800.
- **LR warmup is critical for hidden layers**: Standard cosine LR (0.001→0.0001) destroys narrow hidden layers (16→32 neurons) before they can establish structure. The fix: **5 SB linear warmup 0.0001→0.001, then cosine 0.001→0.0001 for remaining SBs**. Without warmup, L1 mean|w| dropped to 0.7 by SB40 (dead). With warmup, L1 mean|w| reached 17.7 by SB20 (healthy). This is because the FT layer (12M params) benefits from high LR, but hidden layers (33K params) get washed out by it.
- **SCReLU scale chain**: SCReLU squares accumulator values (scale QA²). The full v² must be preserved through the L1 matmul — dividing by QA before the matmul loses too much precision. Correct chain: bias×QA² + sum(v²×w), then /QA² after matmul.
- **Hidden→output activation is linear** in Bullet (no SCReLU on the final layer before output buckets). The `l2.forward(hidden)` call in Bullet has no `.screlu()`.
- **L1 quantisation**: int16 at QA=255. L2 quantisation: int16 at QA=255. Output weights: int16 at QB=64. Output bias: int32 at QA×QB.
- **v5 architecture is saturated**: 1024 CReLU, 1024 SCReLU, 1536 CReLU, 1536 SCReLU all within ~20 Elo of each other cross-engine. Hidden layers are needed to break through.
- **Float inference for small layers**: Viridithas/top engines use float for the 16→32 layers. Worth considering if quantization precision becomes an issue.

Bullet LR schedule for hidden layer nets:
```rust
// CRITICAL: warmup prevents hidden layer collapse
lr_scheduler: Sequence {
    first: LinearDecayLR { initial_lr: 0.0001, final_lr: 0.001, final_superbatch: 5 },   // ramp UP
    second: CosineDecayLR { initial_lr: 0.001, final_lr: 0.0001, final_superbatch: N-5 }, // decay
    first_scheduler_final_superbatch: 5,
}
```

### Monitoring Hidden Layer Health During Training

Hidden layer neurons (L1=16, L2=32) can die during training. **Monitor at every checkpoint.**

**Using check-net:**
```bash
./tuner check-net net.nnue          # Reports architecture, evals test positions
```
Check-net flags: scores should be differentiated (knight ≠ bishop ≠ rook ≠ queen for piece-down positions). Collapsed evals (identical scores for different piece-down) indicate dead neurons.

**Using weight analysis** (from raw quantised.bin or converted .nnue):
```go
// Load net and inspect per-neuron weight magnitude
net, _ := chess.LoadNNUEV5("net.nnue")
for i := 0; i < net.L1Size; i++ {
    // Sum |weight| for neuron i across all inputs
    var sum float64
    for j := 0; j < 2*net.HiddenSize; j++ {
        sum += math.Abs(float64(net.L1Weights[j*net.L1Size+i]))
    }
    meanW := sum / float64(2*net.HiddenSize)
    fmt.Printf("Neuron %d: mean|w|=%.1f\n", i, meanW)
}
```

**Healthy indicators:**
- L1 mean|w| > 5 (growing over training)
- L1 zeros < 10%
- Per-neuron mean|w| differentiated (not all identical)
- L1 biases active (not all zero)
- L2 mean|w| > 20 for int16, > 5 for int8

**Dead neuron indicators:**
- L1 mean|w| < 1 or declining across checkpoints
- L1 zeros > 50%
- Per-neuron mean|w| uniform (e.g. all ~0.7 = no differentiation)
- L1 biases all zero
- check-net shows identical scores for different positions

**Timeline from our experience:**
- CReLU e800: 16/16 alive at SB20 → 1/16 alive at SB400 (dying ReLU)
- SCReLU e800 with warmup: 16/16 alive through all 800 SBs
- Without warmup, even SCReLU collapsed by SB40 on long schedules (e100+)
- **Always check at SB20, SB50, SB100** — if L1 mean|w| is declining, kill and adjust

**Key training requirements for hidden layers:**
1. **SCReLU activation** (not CReLU) — CReLU has zero gradient at 0, kills neurons permanently
2. **LR warmup** — 5 SB linear ramp 0.0001→0.001 before cosine decay (see schedule above)
3. **int8 L1 (QA_L1=64)** gives 69% faster inference via VPMADDUBSW kernel
4. **Float L2→output inference** — avoids integer truncation in narrow layers
5. **Score filter `unsigned_abs() < 10000`** in Bullet data loader (consider tightening to 2000)

### Texel Tuner

See `docs/legacy_trainer.md` for the Go CPU tuner (selfplay, parameter optimization). Still used for Texel tuning of eval parameters.

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
- NNUE accumulator stack must stay in sync with undo stack (push on MakeMove, pop on UnmakeMove, null moves skip)
- NNUE `putPiece`/`removePiece`/`movePiece` hooks read king bitboards — call when kings are on the board
- NNUE incremental updates skip kings; king moves trigger `RecomputeAccumulator`. Castling moves king first, then rook
- NNUE v5/v6 hidden size is auto-detected from file size; v7 stores dimensions in header
- NNUE v7 SCReLU: do NOT divide by QA before L1 matmul — keep v² at scale QA² through the matmul for precision
- NNUE v7 hidden→output: Bullet uses linear (no activation) before output. Do NOT apply SCReLU at L1→output or L2→output boundary
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
     -rounds 5000 -concurrency 16 \
     -sprt elo0=-10 elo1=5 alpha=0.05 beta=0.05 \
     -openings file=testdata/noob_3moves.epd format=epd order=random \
     -pgnout sprt_test.pgn -recover -ratinginterval 20
   ```
4. **Be patient** — small gains (+5 Elo) are real and valuable. Stockfish gained hundreds of Elo from many +5 patches. SPRT at this level needs 1000-3000 games to converge.
5. **Tune before rejecting** — if a sound idea tests negative, try 2-3 parameter variants before giving up. Run variants in parallel at lower concurrency.
6. **One change at a time** — never stack untested changes. Each commit should be independently validated.
7. **Merge on SPRT acceptance** — merge the worktree branch to main, commit, push. Then move to the next idea.
8. **Log results in `experiments.md`** — record every experiment (pass or fail) with the change, SPRT result, baseline, and lessons learned. This builds institutional knowledge and prevents re-testing failed ideas.

CPU budget: Single-thread engines without pondering only use 1 CPU thread when it's their turn — they are idle while the opponent thinks. So each game uses ~1 active thread, NOT 2. On a 16-thread machine, use **concurrency 16** for single-thread engine games. Two parallel experiments at concurrency 8 each is fine.

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

### CRITICAL: Self-Play vs Cross-Engine Transfer (discovered 2026-03-22)

**Self-play SPRT is actively misleading for pruning-class changes.** Extensive cross-engine ablation testing revealed that pruning tightening gains Elo in self-play while LOSING Elo against other engines. This is not a discount — it is anti-correlated.

**Why:** Pruning tightening exploits shared eval blind spots. In self-play, both sides miss the same positions, so pruning them is "free." Against other engines with different evals, those pruned positions contain moves the opponent plays and we miss. The more aggressively we prune, the more we gain in self-play AND the more we lose cross-engine.

**Revised transfer model (2026-03-23, from combined Hercules + Atlas gauntlet data):**

| Change type | Example | Self-play | Cross-engine | Mechanism |
|------------|---------|-----------|-------------|-----------|
| Accuracy/information | multi-corr-hist | +28.6 | **+25** | Eval-agnostic, both sides benefit |
| Self-correcting reduction | LMR C=1.30 | +13.3 | **~+8** | Re-search prevents permanent blindspots |
| Eval-agnostic reduction | hindsight 200 | +16.2 | **+10** | Quiet detection works regardless of opponent |
| NPS improvement | Finny tables | +165 | **+50** | ~0.3:1 discount |
| Structural search | aspiration contraction | +48.9 | **~+20** | ~0.4:1 discount |
| Hard capture pruning | SEE cap 80 | +25.2 | **-10** | Captures are where engines diverge most |
| Capture reduction overfit | cap LMR continuous | +10.6 | **-26** | Fine-grained self-play optimization = overfitting |
| Pruning tightening | hist-prune, badnoisy | +14-32 | **-18 to -26** | Shared blindspots exploited |
| Bad extensions | check ext (SEE) | -11 | **-30 to -39** | 3:1 amplification — wasted nodes on eval-biased positions |
| Bad extensions | singular ext | -60 to -140 | **-41** | Verification search too costly |

**Key principle:** Any search decision tuned to our eval's biases gets amplified cross-engine — whether pruning what we think is unimportant (positive self-play, negative cross-engine) or extending what we think is important (negative self-play, *even more* negative cross-engine). Only eval-agnostic changes transfer cleanly.

**Rules for search changes:**

1. **Accuracy/information improvements** (correction history, eval refinement, move ordering): Trust self-play SPRT. ~1:1 transfer.
2. **Self-correcting reductions** (LMR with re-search): Trust self-play SPRT. Re-search mechanism prevents permanent blindspots.
3. **Eval-agnostic reductions** (hindsight — both sides agree position is quiet): Trust self-play SPRT. ~1:1 transfer.
4. **Structural changes** (aspiration windows, time management): Trust self-play SPRT with ~2:1 discount.
5. **NPS improvements** (SIMD, caching): Trust self-play SPRT with ~3:1 discount.
6. **Hard capture pruning/reduction** (SEE thresholds, capture LMR tuning): **DO NOT trust self-play.** Must validate with cross-engine gauntlet. Captures are the most opponent-dependent search decisions.
7. **Extensions** (check ext, singular ext): **DO NOT trust self-play.** Extensions invest nodes based on eval judgment, which is amplified 3:1 against diverse opponents. Only recapture extensions (forced tactical resolution) are proven beneficial.
8. **Eval scale changes** (SCReLU activation, quantization): **Must test cross-engine.** Different eval dynamic ranges interact with all search thresholds simultaneously. SCReLU needs ×0.80 scale correction to match CReLU-tuned thresholds.

**Three-tier validation (discovered 2026-03-25):**

Self-play SPRT gives **contradictory signals** to cross-engine testing. Confirmed examples:
- TT staticEval fix: -9 self-play, **+12 cross-engine**
- Evasion capture ordering fix: unknown self-play, **+26 cross-engine**
- Previous "Titan reverts" were features that gained +10-30 self-play but hurt cross-engine — tested against a baseline with the evasion bug, so cross-engine results may have been contaminated

**Tier 1: Self-play SPRT (fast fail, ~10 min)**
- Catches disasters (-50 Elo) quickly
- **Pass even if mildly negative (down to -10)** — real cross-engine gains can show as -9 in self-play
- Only reject if clearly catastrophic

**Tier 2: Rival engine gauntlet (the real test, ~20 min)**
- 3-5 engines near our Elo (e.g. Texel, Ethereal, Laser), 200 games each, concurrency 16
- This is the **acceptance criterion** — positive delta here means merge
- Catches eval-dependent effects, TT interactions, and move ordering issues that self-play misses

**Tier 3: Broad RR (validation before production, ~2 hours)**
- 10-12 engines in our competitive range (Tucano through Laser tier)
- Confirms the gauntlet result against diverse opponents
- Guards against overfitting to the 3 gauntlet engines

**Only merge if Tier 2 is positive.** Tier 1 rejection is not grounds to skip Tier 2 unless clearly catastrophic. Tier 3 is for production releases and major changes.

## Maintenance Reminders

- **Keep CLAUDE.md and README.md up to date** when changing search, eval, tuner, CLI, or architecture
- **New eval parameter**: see `docs/legacy_trainer.md` for tuner update checklist

## NNUE Model Naming Convention

Model files follow one of two formats depending on architecture:

**v5 (direct FT→output):**
```
net-v5-{width}{activation}-w{wdl}-e{epochs}s{snap}.nnue
```

**v7 (FT→hidden→output, with hidden layers):**
```
net-v7-{ftWidth}h{layers}{activation}-w{wdl}-e{epochs}s{snap}.nnue
```

Where:
- **ftWidth**: accumulator width: `1024`, `1536`, `768pw` (768 pairwise)
- **h{layers}**: hidden layer stack (v7 only): `h16` (single L1=16), `h16x32` (L1=16, L2=32)
- **activation**: omit for CReLU (default), `s` for SCReLU
- **w{wdl}**: WDL proportion as integer hundredths: `w0` (0.0), `w5` (0.05), `w10` (0.1), `w50` (0.5)
- **e{epochs}**: total superbatches in the cosine LR schedule (determines LR curve)
- **s{snap}**: snapshot checkpoint within that run

### Examples
```
net-v5-1024-w0-e120s120.nnue      # 1024 CReLU, wdl=0.0, cosine/120, final checkpoint (production)
net-v5-1024s-w0-e200s200.nnue     # 1024 SCReLU, wdl=0.0, cosine/200, final
net-v5-1024-w5-e200s200.nnue      # 1024 CReLU, wdl=0.05, cosine/200, final
net-v5-1536-w0-e800s400.nnue      # 1536 CReLU, wdl=0.0, cosine/800, snapshot at sb400
net-v5-1536-w0-e800s800.nnue      # 1536 CReLU, wdl=0.0, cosine/800, final
net-v5-1536-w5-e800s600.nnue      # 1536 CReLU, wdl=0.05, cosine/800, snap at 600
net-v5-768pw-w0-e400s400.nnue     # 768 pairwise, wdl=0.0, cosine/400, final
net-v7-1024h16-w0-e100s100.nnue      # 1024 FT, L1=16, CReLU, cosine/100, final
net-v7-1024h16s-w0-e800s800.nnue    # 1024 FT, L1=16, SCReLU, cosine/800, final
net-v7-1024h16x32s-w0-e800s800.nnue # 1024 FT, L1=16→L2=32, SCReLU, cosine/800, final
net-v7-1024h16x32s-w5-e200s200.nnue # 1024 FT, L1=16→L2=32, SCReLU, wdl=0.05, final
```

### Key distinctions this prevents
- `e800s400` vs `e400s400`: same SB count but **different LR** (800-run is at 50% LR, 400-run is fully decayed)
- `w5` vs `w50`: unambiguous (0.05 vs 0.5)
- `1024s` vs `1024`: SCReLU vs CReLU
- `v5` vs `v7`: direct output vs hidden layer architecture
- `h16` vs `h16x32`: single hidden layer vs two hidden layers

### Legacy names
Old files use inconsistent naming (e.g. `net-v5-120sb-sb120.nnue`, `net-v5-1536-wdl00-sb400.nnue`). These should be renamed when practical. The production model `net-v5-120sb-sb120.nnue` keeps its name in GitHub releases for backward compatibility.

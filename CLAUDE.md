# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Build and Test Commands

```bash
go test                    # Run all tests (includes WAC/ECM suites, ~3min)
go test -v                 # Verbose output
go test -run TestX         # Run a specific test
go test -bench .           # Run benchmarks
go test -short              # Run unit tests only (skips slow EPD suites)

go build -o chess ./cmd/chess    # Build CLI binary
go build -o tuner ./cmd/tuner   # Build Texel tuner binary
./chess -e testdata/wac.epd -t 5000 -n 20              # Run EPD test suite
./chess -e testdata/wac.epd -t 5000 -n 20 -threads 4   # Run EPD with Lazy SMP (4 threads)
./chess -uci                                            # Start UCI mode
./chess -benchmark -t 200                               # Run multi-suite benchmark (quick)
./chess -benchmark -t 200 -save base.json               # Save benchmark results to JSON
./chess -benchmark -t 200 -compare base.json            # Compare against saved baseline
./chess -buildbook -pgn testdata/2600.pgn -bookout book.bin  # Build Polyglot opening book

./tuner selfplay -games 20000 -time 200 -concurrency 6 -output training.dat  # Generate training data (FEN;score;result)
./tuner tune -data training.dat -epochs 500 -lr 1.0                          # Tune eval parameters
./tuner tune -data training.dat -epochs 500 -lr 1.0 -lambda 0.5             # Tune with blended score+result loss
./tuner nnue-train -data training.dat -epochs 100 -lr 0.01 -output net.nnue          # Train NNUE network
./tuner nnue-train -data training.dat -positions 50000 -epochs 50 -output net-v1.nnue # Train on subset
./tuner nnue-train -data training.dat -resume net-v1.nnue -epochs 100 -output net-v2.nnue # Resume training

./chess -nnue net.nnue -uci                                                  # UCI with NNUE
./chess -syzygy /path/to/tablebases -uci                                      # UCI with Syzygy tablebases

# Self-play testing with cutechess-cli (ALWAYS use these patterns)
# New vs old binary (200 games, 10s+0.1s increment, draw/resign adjudication):
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

# Gauntlet vs external engines:
cutechess-cli \
  -tournament gauntlet \
  -engine name=GoChess cmd=./chess arg=-uci proto=uci option.MoveOverhead=100 \
  -engine name=GnuChess cmd=gnuchessu dir=/usr/share/games/gnuchess proto=uci \
  -engine name=Rodent3 cmd=rodentIII proto=uci \
  -each tc=0/10+0.1 \
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
testdata/            Test data (wac.epd, ecm.epd, arasan.epd, lct.epd, sbd.epd, wac300.epd, wac2018.epd, zugzwang.epd, noob_3moves.epd, 2600.pgn)
book.bin             Opening book (Polyglot .bin format)
board.go             Board struct, piece types, FEN parsing, pieceOf() helper
move.go              Move encoding (16-bit), flags, NoMove sentinel
bitboard.go          Bitboard type, bit manipulation, file/rank masks
attacks.go           Magic bitboard tables, pre-computed attack lookups
movegen.go           Pseudo-legal/legal move generation, IsAttacked, InCheck
makemove.go          MakeMove/UnmakeMove, null move, UndoInfo
movepicker.go        Staged move ordering for search, IsPseudoLegal
search.go            Negamax, alpha-beta, iterative deepening, LMR, LMP, PVS, Lazy SMP
eval.go              Tapered eval: PST, mobility, king safety, positional bonuses
pst.go               PeSTO piece-square tables, material values, phase constants
pawns.go             Pawn structure eval (doubled/isolated/passed), pawn hash table, pawn shield
tt.go                Transposition table (lockless, 4-slot buckets with Stockfish-style age/depth replacement)
zobrist.go           Zobrist hash keys, incremental hashing
see.go               Static Exchange Evaluation for capture ordering and quiet move pruning
san.go               SAN parsing (ParseSAN) and formatting (ToSAN)
epd.go               EPD file loading and test suite runner
pgn.go               PGN game parsing (tags, moves)
benchmark.go         Multi-suite benchmark: continuous scoring (time-to-solve), JSON save/load, comparison
book.go              Opening book: build Polyglot .bin from PGN
polyglot.go          Polyglot book format: load, hash, move encoding/matching, weighted selection
uci.go               UCI protocol (position, go, setoption, ponder)
cli.go               Interactive CLI engine (set, fen, board, eval, moves, search, epd, perft)
selfplay.go          Self-play game generation for tuning data
tuner.go             Texel tuner: parameter catalog, trace-based eval, .tbin binary cache, disk-streamed Adam optimizer
nnue.go              NNUE inference: HalfKP network, accumulators, forward pass, incremental updates, load/save
nnue_amd64.go        NNUE SIMD stubs for x86-64 (AVX2 runtime detection)
nnue_amd64.s         NNUE AVX2 assembly: CReLU, MatMul, accumulator ops
nnue_arm64.go        NNUE SIMD stubs for ARM64 (NEON always available)
nnue_arm64.s         NNUE NEON assembly: CReLU, MatMul, accumulator ops
nnue_nosimd.go       NNUE fallback stubs for non-SIMD platforms
nnue_train.go        NNUE training: float32 net, backprop, Adam optimizer, binary cache, quantization, resume from .nnue
tb.go                Syzygy tablebase integration: init/free, WDL/DTZ probing, move matching
syzygy/              Fathom C library (CGO wrapper for Syzygy tablebase probing)
syzygy/syzygy.go     CGO wrapper: Init, Free, ProbeWDL, ProbeRoot
syzygy/syzygy_stub.go  No-op stubs when CGO is disabled
syzygy/tbprobe.c     Fathom probing implementation (includes tbchess.inc)
syzygy/tbchess.inc   Fathom move generation (included by tbprobe.c, not compiled separately)
syzygy/tbprobe.h     Fathom public API
syzygy/tbconfig.h    Engine-specific Fathom configuration (builtin popcount/lsb, no helper API)
syzygy/tbwrappers.h  Go-callable inline functions wrapping Fathom C macros
```

## Architecture

### Board Representation

- **Hybrid storage**: `Board.Squares[64]` for piece-by-square lookup, `Board.Pieces[13]` bitboards for piece-type iteration
- **Occupancy**: `Board.Occupied[2]` (by color) and `Board.AllPieces` for attack generation
- **Incremental PST**: `Board.PSTScoreMG[2]` and `Board.PSTScoreEG[2]` track running PST+material totals per color, updated by `putPiece`/`removePiece`/`movePiece`
- **Undo stack**: `Board.UndoStack []UndoInfo` stores move, captured piece, castling, en passant, halfmove clock, HashKey, and PawnHashKey per move

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
- `GenerateLegalMoves()` — filters via `IsLegal()` (pin-aware fast path)
- `GenerateCaptures()` — captures + promotions (for quiescence)
- `GenerateQuiets()` — non-captures, non-promotions
- `GenerateEvasionsAppend()` — fully legal evasion moves when in check (no IsLegal filtering needed)

### Move Legality (Pin-Aware Fast Path)

`IsLegal(m, pinned, inCheck)` uses precomputed data to avoid full make/unmake for most moves:

- **Precomputed tables**: `LineBB[64][64]` (all squares on line through two squares) and `BetweenBB[64][64]` (squares strictly between two aligned squares). Initialized in `initLineBB()`.
- **`PinnedAndCheckers(us)`**: Returns `(pinned, checkers Bitboard)` in a single pass. Finds enemy snipers (sliders that could see the king on an empty board), then classifies: 0 pieces between → checker, 1 friendly piece between → pinned.
- **Fast paths in IsLegal**:
  1. Castling → always true (validated during generation)
  2. En passant → full make/unmake (rare, tricky discovered checks)
  3. King moves → `isAttackedWithOcc(to, them, occ)` with king removed from occupancy
  4. Non-king, in check → full make/unmake (must verify check resolution)
  5. Non-king, not pinned, not in check → always true (single bitboard AND)
  6. Pinned piece → legal iff moves along pin line: `LineBB[kingSq][from] & SquareBB(to) != 0`
- **`isAttackedWithOcc(sq, by, occ)`**: Like `IsAttacked` but uses custom occupancy for sliding pieces. Shared by IsLegal king path and evasion generator.
- **Search integration**: `negamax` and `quiescence` call `PinnedAndCheckers` once per node. When in check, `InitEvasion` uses `GenerateEvasionsAppend` to produce fully legal moves, skipping `IsLegal`. When not in check, `IsLegal` filters pseudo-legal moves as before.
- **`IsLegalSlow(m)`**: Convenience wrapper computing both pinned and inCheck internally. Used in non-hot paths (PV extraction, `GenerateLegalMoves`).

### Evasion Move Generator

`GenerateEvasionsAppend(moves, checkers, pinned)` generates all legal moves when in check, without needing `IsLegal` filtering:

- **King evasions** (always): moves to squares not attacked by enemy, using occupancy with king removed so sliders see through.
- **Double check**: only king moves are legal (non-king section skipped entirely).
- **Single check**: non-pinned pieces can capture the checker or block the check ray (`BetweenBB[kingSq][checkerSq]`).
- **Key insight**: pinned pieces can never resolve check (the checker must be on a different line than the pin), so they're excluded from non-king evasion generation.
- **En passant**: only generated if the captured pawn IS the checking piece, validated via make/unmake.
- **Integration**: `MovePicker.InitEvasion()` uses a flat scored list (captures above quiets) with selection sort. Used by both negamax (when in check) and quiescence (fixes correctness bug where QS missed quiet evasions).

### Move Ordering (MovePicker)

Staged generation for search efficiency:

**Normal mode** (not in check):
1. TT move (from transposition table)
2. Good captures (SEE >= 0, scored by MVV-LVA + capture history)
3. Killer moves (2 per ply, caused beta cutoffs in sibling nodes)
4. Counter-move (move that refuted opponent's previous move)
5. Quiet moves (scored by history + continuation history)
6. Bad captures (SEE < 0, scored by SEE + capture history)

**Evasion mode** (in check): TT move → all evasion moves in a single scored list (captures: 10000+MVV-LVA+captHist, queen promotions: 9000, quiets: history+contHist, underpromotions: -1000).

Selection sort within each stage (partial sort, only finds next-best on demand).

### Search

Negamax with alpha-beta pruning, iterative deepening with time control.

- **Transposition table**: Probe before search, store after. 4-slot buckets with Stockfish-style replacement scoring (`depth - 4*age`): stale entries from older generations are cheaply evicted, current-generation deep entries are preserved. Lockless thread-safe via packed atomic `uint64` fields with XOR verification (see Lazy SMP section). Mate scores adjusted by ply distance to prevent stale mate evaluations.
- **Null-move pruning**: Skip turn and search with reduced depth. Adaptive reduction: R = 3 + depth/3 + min(3, (eval-beta)/200). Verification search at depth >= 12: re-search at depth-1-R to catch zugzwang. Requires depth >= 3, non-pawn material, not in check.
- **Reverse Futility Pruning**: At shallow depths (depth <= 6), prune whole node if static eval minus margin (depth * 120) exceeds beta.
- **Futility pruning**: At depth <= 6, skip quiet non-checking moves when static eval plus margin (depth * 200) cannot raise alpha.
- **Late Move Reductions (LMR)**: Logarithmic reduction table. Quiet moves searched late in the move list are reduced. Re-search at full depth if score exceeds alpha. Disabled for captures, promotions, killers, and check-giving moves. Continuous history adjustment: good history reduces less, bad history reduces more (histScore / 5000). Reduced less at PV nodes and when position is improving. **Cut-node adjustment**: +1 reduction at expected cut nodes (zero window search, not first move).
- **Late Move Pruning (LMP)**: Skip quiet moves at shallow depths (depth 1-8) after searching enough moves (threshold: `3 + depth*depth`). When the position is improving and depth >= 3, thresholds are increased by 50% to search more moves. Disabled when in check or giving check.
- **SEE quiet pruning**: At depth <= 8, prune quiet moves where `SEEAfterQuiet` indicates the piece lands on a square where it would be captured for material loss exceeding `depth * 80` centipawns. Computed before MakeMove, applied after (to exempt check-giving moves). Exempts TT move, killers, counter-move, captures, promotions. Controlled by `SEEQuietPruneEnabled` toggle.
- **SEE capture pruning**: At depth <= 6, prune captures where SEE < -depth*100 (losing too much material). Exempts TT move and positions where we might be in a mating attack. Applied before MakeMove.
- **ProbCut**: At depth >= 5, if static eval + 100 >= beta + 200, search good captures (SEE >= 0) at reduced depth (depth-4) with raised beta (beta+200). If any capture scores above the raised beta, prune the node. Leverages accurate shallow searches to cut deep subtrees.
- **History-based pruning**: At depth <= 3, prune quiet moves with deeply negative combined history + continuation history scores (threshold: -2000*depth). Only when position is NOT improving (protecting tactical lines). Exempts TT move, killers, counter-move, captures, promotions.
- **Internal Iterative Reduction (IIR)**: At depth >= 6, when no TT move exists and not in check, reduce depth by 1. Searching without a good move to try first is less efficient, so we save time by searching shallower. Gated on `IIREnabled`.
- **Singular extensions**: At depth >= 10, if the TT move is significantly better than alternatives (verified by a reduced-depth search excluding the TT move), extend the TT move by 1 ply.
- **Principal Variation Search (PVS)**: After first move, search with zero window (alpha, alpha+1). Re-search with full window if it fails high.
- **Aspiration windows**: Starting at depth 4, iterative deepening uses a narrow window (delta=25) around previous score. Widens progressively on fail high/low.
- **Check extensions**: Extend search by 1 ply when move gives check.
- **Recapture extensions**: Extend by 1 ply when recapturing on the same square the opponent just captured on, to fully resolve tactical exchanges. Gated on `RecaptureExtEnabled`.
- **Passed pawn push extensions**: Extend by 1 ply for quiet pawn pushes to 6th or 7th rank that are verified as passed pawns (via `PassedPawnMask`). Helps resolve promotion races and endgame tactics. Gated on `PassedPawnExtEnabled`.
- **Quiescence search**: Captures only at leaf nodes, pruned by SEE >= 0. Stand-pat evaluation as lower bound. Depth-limited to 32. Uses fail-soft (returns bestScore, not alpha/beta). TT probed at top for cutoffs; results stored on all exit paths with depth 0. When in check: uses evasion generator for all legal moves (not just captures), skips stand-pat, detects checkmate (moveCount == 0).
- **Killer moves**: 2 slots per ply, updated on beta cutoff with quiet moves.
- **Counter-move heuristic**: `CounterMoves[piece][toSquare]` indexed by opponent's previous move. Stored on beta cutoff, used as a MovePicker stage between killers and quiets.
- **History heuristic**: `history[from][to] += depth * depth` on beta cutoff. Quiet moves tried before the cutoff move receive a matching penalty (`-= depth * depth`). Used to score quiet moves in move ordering and to adjust LMR reductions.
- **Capture history**: `CaptHistory[piece][toSquare][capturedPieceType]` (int16, ~11.4KB per thread). Tracks which captures caused beta cutoffs. Updated with `depth * depth` bonus on cutoff, penalty for captures tried before. Added to MVV-LVA scores for good captures and to SEE scores for bad captures, improving ordering among equal-value captures.
- **Continuation history**: `ContHistory[prevPiece][prevTo][curPiece][curTo]` (int16, ~1.3MB per thread). Captures the pattern "after piece X moved to square Y, quiet move Z tends to be good/bad". Updated alongside History on quiet beta cutoffs (bonus) and for quiet moves tried before cutoff (penalty). Added to quiet move scores in MovePicker and to the LMR history adjustment. Nil-safe: disabled at root and after null moves.
- **Time management**: Dual soft/hard deadline system. `computeSearchTime()` returns soft (base allocation) and hard limits. Soft allocation: `timeLeft/25 + inc*4/5`, capped at half remaining. Emergency mode: when `timeLeft < 1000ms`, soft is capped to `min(timeLeft/10, inc)`. Hard limit: 3× soft for sudden death, 2× soft for tournament TC (movestogo), with absolute cap at `timeLeft/5 + inc` (never exceeding 75% remaining). Between iterations (depth ≥ 4), the soft deadline is dynamically scaled based on best-move stability (stable → scale down to 0.5×) and score instability (large delta → scale up to 1.4×). Mate-range score swings are ignored. The adjusted soft deadline is clamped to the hard deadline. Checks clock every 4096 nodes against hard deadline. Helper threads, EPD, benchmark, and `go movetime` use hard deadline only (SoftDeadline=0). `go infinite`/`go depth` set no deadlines.
- **Lazy SMP**: Multi-threaded search via `SearchParallel()`. All threads search the same root position independently, sharing only the transposition table. Each thread has its own `Board` copy (with undo stack for repetition detection), `SearchInfo` (killers, history, counter-moves), and pawn hash table. Helper threads use depth diversification (a skip table indexed by thread index and depth) to ensure threads are at different depths at any given time, improving TT entry diversity. The main thread (thread 0) runs normally with the `OnDepth` callback; helper threads run a stripped-down iterative deepening loop. Node counts from all threads are aggregated for NPS reporting. Default: 1 thread (no behavior change). Configurable via UCI `Threads` option (1-256) and `-threads` CLI flag.

### Syzygy Tablebases

Optional endgame tablebase support via the bundled Fathom C library (CGO). Controlled by `SyzygyPath` UCI option or `-syzygy` CLI flag.

- **Root DTZ probing** (`SearchWithInfo`): Before iterative deepening, if the position has ≤`MaxPieceCount` pieces and no castling rights, probe DTZ tables to get the optimal move immediately. Returns without searching. The move from Fathom is matched against `GenerateLegalMoves()` by from/to squares and promotion piece.
- **Search WDL probing** (`negamax`): At interior nodes (ply > 0), after draw detection and before the TT probe, if the position has ≤`MaxPieceCount` pieces, no castling, and `HalfmoveClock == 0`, probe WDL tables. Results are stored in TT with depth `MaxPly` so they persist. Win/loss scores use `TBWinScore/TBLossScore` (28800/-28800), adjusted by ply like mate scores. Cursed wins and blessed losses are scored as ±1.
- **Score constants**: `TBWinScore = MateScore - 200` (28800). Below mate scores but above all eval scores.
- **Probing eligibility**: `tbCanProbeWDL(depth)` checks: SyzygyEnabled, no castling, piece count ≤ max (with depth gate at exactly max). `tbCanProbeRoot()` checks: SyzygyEnabled, piece count ≤ max.
- **Bitboard mapping**: `tbGetBitboards()` extracts Fathom's required format from our Board: `white/black` from `Occupied[color]`, piece-type bitboards by OR-ing both colors (e.g. `kings = Pieces[WhiteKing] | Pieces[BlackKing]`).
- **CGO wrapper** (`syzygy/syzygy.go`): Thin wrapper around Fathom's `tb_init`, `tb_free`, `tb_probe_wdl`, `tb_probe_root`. C macros (`TB_GET_WDL` etc.) are wrapped in inline functions in `tbwrappers.h` since CGO can't call macros. Stub file (`syzygy_stub.go`) provides no-op implementations when CGO is disabled.
- **UCI options**: `SyzygyPath` (string, path to `.rtbw`/`.rtbz` files), `SyzygyProbeDepth` (spin, 1-100, minimum depth for WDL probes). CLI flag: `-syzygy <path>`.
- **Statistics**: `SearchInfo.TBHits` counts probes. Reported as `tbhits` in UCI info strings.

### Transposition Table Thread Safety

The TT uses a lockless scheme for concurrent access by multiple search threads:
- Each entry is stored as two `uint64` fields: `keyXor` and `data`
- `data` packs move (16 bits), flag (8 bits), score (16 bits), and depth (8 bits) into a single `uint64`
- `keyXor = key ^ data` — on read, the stored key is recovered as `keyXor ^ data` and verified against the requested key
- All reads/writes use `atomic.LoadUint64`/`atomic.StoreUint64`
- Torn reads (where one field is from an old write and the other from a new write) are detected by the XOR verification and treated as misses
- Stats counters use `atomic.AddUint64`

### NNUE Evaluation

Optional neural network evaluation behind `UseNNUE` toggle. Both classical and NNUE evals coexist; `EvaluateRelative()` dispatches automatically.

- **Architecture**: HalfKP input (40960) -> 2x256 accumulators -> concat (512) -> 32 -> 32 -> 1
- **Feature indexing**: `HalfKPIndex(perspective, kingSq, piece, pieceSq)`. 64 king squares x 10 piece types (excluding kings) x 64 piece squares = 40960. Black perspective mirrors squares (^56) and swaps piece colors.
- **Incremental updates**: `putPiece`, `removePiece`, `movePiece` call `AddFeature`/`RemoveFeature` on the accumulator. King pieces are skipped (king moves trigger full `RecomputeAccumulator`).
- **SIMD forward pass**: Platform-specific assembly for AVX2 (x86-64) and NEON (ARM64). `nnueUseSIMD` flag controls dispatch. Five SIMD functions: `nnueCReLU256` (activation), `nnueMatMul32x512` (hidden layer), `nnueAccAdd256`/`nnueAccSubAdd256`/`nnueAccSubSubAdd256` (accumulator updates). `PrepareWeights()` transposes hidden weights for SIMD matmul. AVX2 uses runtime detection (`cpu.X86.HasAVX2`); NEON is always available on ARM64.
- **Accumulator stack**: `NNUEAccumulatorStack` with Push/Pop mirrors the undo stack. `MakeMove` pushes before modifications, `UnmakeMove` pops at end. Null moves skip push/pop (no pieces change, child moves push/pop their own copies).
- **Thread safety**: `NNUENet` is shared read-only across Lazy SMP threads. Each thread gets its own `NNUEAccumulatorStack` via `DeepCopy()`.
- **Quantization**: Training uses float32 (`NNUETrainNet`), inference uses int16 (`NNUENet`). Scale factors: input=127, hidden=64, output=127*64=8128. `DequantizeNetwork` reverses this for resuming training from a saved `.nnue` file.
- **Training**: `NNUETrainer` with Adam optimizer. Loss: `lambda * MSE(sigmoid(nnue/K), result) + (1-lambda) * MSE(sigmoid(nnue/K), sigmoid(score/K))`. Binary cache (`.nnbin`) for disk-streamed training data. Supports `-resume` to load weights from an existing `.nnue` file and `-positions` to cap training data size per epoch.
- **UCI options**: `UseNNUE` (check), `NNUEFile` (string path). CLI: `nnue load/on/off/eval`. CLI flag: `-nnue <file>`.

### Evaluation

Tapered evaluation blending middlegame and endgame scores based on game phase (piece count). `Evaluate()` returns White-relative centipawns; `EvaluateRelative()` returns side-to-move relative.

- **Piece-square tables** (pst.go): PeSTO tables for all piece types, separate MG/EG values with per-piece-type scaling factors (35-85%). Material values added separately: MG (P=82, N=337, B=365, R=477, Q=1025), EG (P=94, N=281, B=297, R=512, Q=936). SEE uses simplified values (P=100, N=320, B=330, R=500, Q=900). **Incremental PST**: Combined PST+material tables (`pstCombinedMG[piece][sq]`, `pstCombinedEG[piece][sq]`) bake material + scaled positional values together. `Board.PSTScoreMG[color]` and `Board.PSTScoreEG[color]` are maintained incrementally by `putPiece`/`removePiece`/`movePiece`, eliminating per-Evaluate full-board scans. `evaluatePST()` is retained for testing/verification.
- **Pawn structure** (pawns.go): Cached via pawn hash table. Evaluates doubled, isolated, backward, connected, and passed pawns. Pawn advancement bonus. **Queenside pawn advancement**: additional rank-indexed bonus for pawns on files a-c (`queensidePawnAdvMG/EG[8]`), stacking on top of the base advancement bonus; rewards pushing queenside pawns which create outside passers. **Candidate passed pawns**: pawns with no enemy pawn ahead on their own file and enough friendly adjacent pawns to outnumber enemy sentries; bonus by rank via `candidatePassedMG/EG[8]` (~45% of passed pawn base). Gated on `CandidatePassedEnabled`. **Pawn majority**: per-wing (queenside files a-d, kingside files e-h) bonus when one side has more pawns than the opponent; scales linearly with the pawn count advantage (`PawnMajorityMG/EG` per extra pawn). Wings where the side already has a passed pawn are skipped to avoid double-counting. Gated on `PawnMajorityEnabled`. **Pawn lever**: bonus for a pawn that can advance one square to attack an enemy pawn on an adjacent file, rewarding tension creation and strategic pawn advances; rank-indexed via `pawnLeverMG/EG[8]` with peak bonus at relative rank 4 (5th rank). Gated on `PawnLeverEnabled`. Precomputed masks: `PassedPawnMask`, `ForwardFileMask`, `OutpostMask`, `AdjacentFiles`.
- **Mobility** (eval.go): Non-linear bonus arrays indexed by move count — `KnightMobility[9]`, `BishopMobility[14]`, `RookMobility[15]`, `QueenMobility[28]` — each with separate MG/EG values.
- **King safety** (eval.go + pawns.go): Table-driven system. Per-piece attack unit weights (Knight=7, Bishop=5, Rook=8, Queen=13) plus king-zone square bonuses accumulate into an attack score. **Safe check bonus**: after piece loops, for each piece type, checks if any reachable check square is "safe" (not defended by enemy pawns, not occupied by friendly pieces); if so, adds a fixed bonus per piece type (Knight=6, Bishop=3, Rook=7, Queen=5). Uses binary detection (any safe check exists), not per-square counting. Gated on `attackerCount >= 1`. **Pawn storm**: adds attack units for friendly pawns advanced near the enemy king, indexed by relative rank via `PawnStormUnits[8]` (max 3 units at rank 6, up to 9 total across 3 king-adjacent files). Gated on `attackerCount >= 1` and `PawnStormEnabled`. **No-queen scaling**: when the attacking side has no queen, the final king safety penalty is scaled down to `NoQueenAttackScale/128` (~31%). `KingSafetyTable[100]` maps total attack units to centipawn penalties. Pawn shield evaluation (ranks 2-3 around king) and semi-open file penalty near king in pawns.go.
- **Positional bonuses** (eval.go): Bishop pair, knight outposts (supported/unsupported), knight closed-position bonus (scales with pawn count), rook on open/semi-open file, rook on 7th rank, doubled rooks on same file, trapped rook penalty, bad bishop penalty (per friendly pawn on same square color), bishop open position scaling, castling rights MG bonus.
- **Passed pawn enhancements** (eval.go): Blocked passer penalty (rank-indexed penalty when enemy piece occupies stop square), control-aware not-blocked bonus (stop square must be physically empty AND not solely enemy-controlled), free path to promotion, king proximity (friendly close / enemy far), protected passers, connected passers, rook behind passer. These depend on piece positions so they are not cached in the pawn table.
- **Space and threats** (eval.go): Space evaluation (safe squares in center files), pawn threats (pawns attacking enemy pieces), piece-on-piece threats (minor pieces threatening rooks/queens, rooks threatening queens).
- **Pawn storm bonus** (eval.go): Direct MG+EG bonus for friendly pawns advanced toward the enemy king, not gated on attacker count. `PawnStormBonusMG/EG[2][8]` indexed by opposed/unopposed and relative rank. Evaluated over 3 files around the enemy king (king file ±1). Separate from the legacy `PawnStormUnits` system in king safety.
- **Endgame king activity** (eval.go): Two EG-only components. (1) Unconditional centralization: both kings penalized per center-distance unit (`KingCenterBonusEG`), tapered eval scales this out in middlegames. (2) Material advantage bonuses: when one side has more material (weighted: N/B=1, R=3, Q=6), reward king proximity (`KingProximityAdvantageEG`) and pushing enemy king to edge (`KingCornerPushEG`). No piece-type gates.
- **Endgame scaling** (eval.go): Per-side scale factors (0-128) for draw/insufficient material detection. Handles KNN, KR vs KB/KN, pawnless 2 minors vs 1+ minor (KBB/KBN vs KB/KN, scale 16), opposite-colored bishop drawishness (OCBScale=64), and 50-move rule scaling.
- **Static eval caching via TT**: Instead of a separate eval cache, static eval is stored in TT entries (`StaticEval` field, int8*4, ±508cp range). Main search and qsearch read it back on TT hits to skip `EvaluateRelative()` calls.
- **Phase calculation**: Knight=1, Bishop=1, Rook=2, Queen=4, Total=24. Phase increases as pieces are traded.

### Zobrist Hashing

Incrementally updated in `MakeMove`/`UnmakeMove` via XOR. Keys cover piece-square, side to move, castling rights (4 individual keys), and en passant file (8 keys, by file not square). Separate `PawnHashKey` for pawn structure caching. Full recompute only in `SetFEN()`. Fixed seed for deterministic hashing.

### SEE (Static Exchange Evaluation)

Simulates alternating captures on a single square. Builds gain array, then negamax backward to find optimal result. `SEESign(move, threshold)` provides fast boolean check with early exits. Used in quiescence pruning and move ordering. `SEEAfterQuiet(move)` evaluates the exchange on the destination square after a quiet move, returning 0 (safe) or negative (material loss); used for quiet move pruning in search.

### Opening Book

Standard Polyglot `.bin` format. Any Polyglot book (downloaded or self-built) works. `LoadOpeningBook()` reads the binary file. `PickMove()` computes a Polyglot Zobrist hash (separate from the engine's internal hash), looks up book moves, and matches them against legal moves to handle castling encoding differences and en passant detection. `BuildOpeningBook()` can generate a Polyglot book from PGN games. Integrated into UCI engine via `OwnBook` and `BookFile` options. The Polyglot hash uses 781 fixed random numbers from the specification; the engine's internal Zobrist hash is unchanged.

### UCI Protocol

`uci.go` implements the Universal Chess Interface. Supports: `position` (startpos/FEN + moves), `go` (time controls, depth, movetime, infinite, ponder), `stop`, `setoption` (Hash, Threads, Ponder, OwnBook, BookFile, UseNNUE, NNUEFile, SyzygyPath, SyzygyProbeDepth), `ponderhit`. Search runs in a goroutine using `SearchParallel` with the configured thread count; `stop` signals all threads via `SearchInfo.Stopped`. Opening book consulted before search when enabled. Syzygy tablebases probed at root (DTZ) and during search (WDL) when enabled.

### EPD / Test Suites

`LoadEPDFile()` parses EPD format (4-field FEN + operations like `bm`, `am`, `id`). `RunEPDTest()` searches a position and checks if found move matches expected best move(s). Test suites in `testdata/`: WAC (201 positions), WAC300 (300), ECM (200), plus arasan, lct, sbd, wac2018, zugzwang.

### PGN Parsing

`ParsePGN()` / `ParsePGNFile()` parse PGN format into `PGNGame` structs (tag pairs + move list). Used by the opening book builder. Handles brace comments, NAGs, and result tokens.

### Texel Tuner

`cmd/tuner/main.go` — Two-phase system for optimizing ~1268 evaluation parameters.

**Self-play data generation** (`selfplay.go`): Plays engine-vs-engine games to produce training data. Each game uses `SearchParallel()` with configurable time/depth per move. Opening diversity from `testdata/noob_3moves.epd` (150K positions). Game termination: checkmate, stalemate, 50-move rule, threefold repetition, insufficient material, or adjudication (eval exceeds ±1000cp for 5 consecutive moves). Positions are filtered (skip first 8 plies, skip checks, skip mate scores) and written as `FEN;score;result` lines (score is White-relative centipawns). Games run concurrently with independent Board+TT per game.

**Binary cache and disk-streamed training** (`tuner.go`): Training data is preprocessed from the `.dat` FEN file into a `.tbin` binary cache, then streamed from disk during tuning. This keeps memory usage at O(batch_size) (~few MB) regardless of dataset size.

- `.tbin` file format (v2): 24-byte header (magic `"TBIN"`, version `uint16`, numParams `uint16`, numTrain `uint32`, numValidation `uint32`, trainBytes `uint64`) followed by variable-length records. Each record stores phase, result, scale factors, halfmove clock, search score (int16), and sparse MG/EG coefficient arrays. ~730 bytes/position average.
- `PreprocessToFile()` reads all FEN lines, shuffles deterministically (seed 42), splits 90/10 train/validation, computes traces via `computeTrace()`, writes binary records.
- `OpenTraceFile()` reads and validates the `.tbin` header, returning a `TraceFile` handle.
- `StreamTraining()` / `StreamValidation()` stream records in batches (default 65536) with reusable buffers. The callback receives a `[]TunerTrace` batch that must not be retained.
- The `.tbin` is auto-created on first run and auto-rebuilt when the source `.dat` file is newer.

**Parameter optimization** (`tuner.go`): Texel tuning via Adam optimizer. The `Tuner` holds the parameter catalog (no in-memory training data):

- `initTunerParams()` builds a flat parameter vector from engine globals: material (10), PST (768, pre-scaled by PST scale factors), mobility (132), piece bonuses (24), passed pawn enhancements (64), pawn structure (90), king attack weights (8), safe check bonuses (4), king safety table (100), pawn shield (5), pawn storm (32), same-side storm (8), endgame king (3), misc (20).
- `computeTrace()` mirrors `Evaluate()` but records sparse MG/EG coefficients per parameter instead of computing a score. Each position produces a `TunerTrace` with `[]SparseEntry` for MG and EG contributions.
- `scoreFromTrace()` evaluates: `(mg * (24 - phase) + eg * phase) / 24`
- `sigmoid(score, K)` maps score to win probability: `1 / (1 + 10^(-score/K))`
- `TuneK(tf, scoreBlend)` finds optimal K via golden section search on MSE over streamed training data
- `Tune(tf, K, cfg, onEpoch)` runs Adam optimizer: per epoch, streams training batches from disk, computes parallel gradients within each batch, aggregates across all batches, then applies Adam update. `cfg.ScoreBlend` (0-1, derived from CLI `-lambda` as `1-lambda`) blends search scores into the loss target: `target = lambda*result + (1-lambda)*sigmoid(score/K)`.
- `ComputeTrainError(tf, K, scoreBlend)` / `ComputeValidationError(tf, K, scoreBlend)` compute MSE by streaming from the respective region of the `.tbin` file

**What's tuned**: Material values, PST tables, mobility arrays, piece bonuses (bishop pair, outposts, rook file, trapped rook, rook-on-enemy-king-file, etc.), passed pawn enhancements (blocked penalty, not-blocked, free path, king proximity, connected, protected), pawn structure (passed base, doubled, isolated, backward, connected, advancement, pawn majority, queenside pawn advancement, candidate passed pawns, pawn lever), king attack unit weights, safe check bonuses (knight, bishop, rook, queen), king safety table (100-entry nonlinear lookup), pawn shield constants (shield rank bonuses, missing shield penalties, semi-open file penalty), pawn storm bonus (MG+EG, 32 params), same-side pawn storm bonus (MG, 8 params), endgame king activity (centralization, proximity, corner push), space/pawn threat/piece threat/castling bonuses, OCB scale.

**What's frozen** (in parameter catalog but not optimized): Material values (10 params, coupling with PSTs), tempo (2 params), trade bonuses (2 params). Tempo and trade are frozen because they encode long-term strategic concepts (simplify-when-ahead, tempo value in quiet positions) that search at practical time controls cannot learn — confirmed by lambda=0 and lambda=0.1 runs both producing anti-textbook values that lost ~37 Elo.

**What's NOT tuned**: Endgame scale factors (multiplicative), phase constants, PST scale factors (folded into PST values), 50-move rule scaling, no-queen attack scale (multiplicative, can't represent in trace).

**PST scaling note**: Tuned PST values are *effective* values (raw table value × scale/100). Output sets all PST scale factors to 100. This avoids coupling between table values and scale factors during optimization.

### CLI

`cmd/chess/main.go` — Five modes:

- **Interactive CLI**: Default when stdin is a terminal and no flags given. Provides commands: `set`, `fen`, `board`, `reset`, `moves`, `search`, `eval`, `epd`, `perft`, `uci`. Implemented in `cli.go` (`CLIEngine`).
- **EPD testing**: `-e` (EPD file), `-t` (ms per position), `-n` (max positions), `-d` (max depth), `-hash` (TT size MB), `-v` (verbose per-position output), `-threads` (Lazy SMP thread count)
- **Benchmark**: `-benchmark` runs WAC, ECM, SBD, Arasan suites with continuous time-to-solve scoring. `-save FILE` writes JSON results, `-compare FILE` shows delta against a baseline. Reuses `-t`, `-hash`, `-d`, `-threads`.
- **UCI mode**: `-uci`, with optional `-book` (opening book path), `-nnue` (NNUE network file), `-syzygy` (tablebase path)
- **Book building**: `-buildbook`, `-pgn`, `-bookout`, `-bookdepth`, `-bookminfreq`, `-booktop`

## Key Gotchas

- Move flag equality vs bitwise: see "Critical rule" above.
- `UnmakeMove()` panics on empty undo stack or move mismatch — always pair with `MakeMove()`.
- `GenerateAllMoves()` returns pseudo-legal moves; search must call `IsLegal()`.
- Castling rights are lost both when a rook/king moves AND when a rook is captured on its home square.
- En passant hash uses file only (8 keys), not full square.
- TT mate scores are ply-adjusted: stored as `mate + ply`, retrieved as `mate - ply`.
- Pawn hash table auto-initializes on first `Evaluate()` call if nil.
- Lazy SMP threads share only the TT. Board, SearchInfo, and pawn table are per-thread. The `Stopped` and `Deadline` fields on `SearchInfo` are accessed atomically (`int32`/`int64`). When adding new shared state, it must use atomic operations or be per-thread.
- TT `Probe`/`Store` are lockless via packed atomics — do not add non-atomic fields to `ttSlot`.
- Tuner PST parameters are pre-scaled (raw × scale/100). Adding a new PST-related eval term requires initializing the tuner param with the effective value, not the raw table value.
- Tuner traces must mirror `Evaluate()` exactly. When modifying eval, update `computeTrace()` in tuner.go to match. Non-additive eval terms (king safety table lookup, endgame scaling, 50-move scaling) cannot be represented in the trace.
- Tuner's `PassedPawnKingScale` uses initial engine values as constant coefficients to avoid circular dependency with distance parameters (product of two tunable params).
- `.tbin` binary cache must be rebuilt (delete it or touch the `.dat` file) whenever `computeTrace()` or the parameter catalog changes. The header stores `numParams` as a sanity check, but structural changes within the same param count won't be caught.
- `StreamTraining`/`StreamValidation` callbacks must not retain references to the `[]TunerTrace` batch — the backing arrays are reused across batches.
- NNUE accumulator stack must stay in sync with the undo stack. Every `MakeMove` pushes, every `UnmakeMove` pops. Null moves skip push/pop (no pieces change). If adding new make/unmake paths, maintain this invariant.
- NNUE `putPiece`/`removePiece`/`movePiece` hooks use `b.Pieces[WhiteKing].LSB()` and `b.Pieces[BlackKing].LSB()` to get king squares for feature indexing. These must be called when king bitboards are valid (kings on the board).
- NNUE incremental updates skip king pieces. King moves trigger `RecomputeAccumulator` at the end of `MakeMove`. Castling moves the king first then the rook — the rook move goes through `movePiece` but the accumulator is not computed yet (will be recomputed at end).
- `.nnbin` binary cache must be rebuilt when the training data format changes or feature indexing changes.
- Syzygy `tbchess.inc` is `#include`d by `tbprobe.c` — it must NOT be compiled as a separate C file (hence the `.inc` extension, which CGO ignores). If adding new C files to `syzygy/`, ensure they don't get compiled twice.
- Syzygy WDL probes require `HalfmoveClock == 0` (Fathom rejects non-zero rule50 for WDL). DTZ probes accept any rule50 value.
- Fathom's `ProbeRoot` is NOT thread-safe. It is only called in `SearchWithInfo` on the main thread before iterative deepening begins. The WDL probe (`tb_probe_wdl`) IS thread-safe.

## Maintenance Reminders

- **Keep CLAUDE.md and README.md up to date.** When making changes to search, evaluation, tuner, CLI, or architecture, update the corresponding sections in both files so they stay accurate and useful.
- **When adding a new evaluation parameter**, consider whether it should be tunable. If so, add it to `initTunerParams()` in `tuner.go`, add the corresponding trace coefficient logic to `computeTrace()`, add it to `PrintParams()`, and update the "What's tuned" list in this file and the README. Delete any existing `.tbin` cache so it gets rebuilt with the new param count.

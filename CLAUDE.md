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
./chess -buildbook -pgn testdata/2600.pgn -eco testdata/eco.pgn -bookout book.bin  # Build opening book

./tuner selfplay -games 20000 -time 200 -concurrency 6 -output training.dat  # Generate training data
./tuner tune -data training.dat -epochs 500 -lr 1.0                          # Tune eval parameters
```

## Project Structure

Chess engine in Go using bitboard representation. Core library is `package chess` in the root; CLI is `cmd/chess/`.

```
cmd/chess/main.go    CLI entry point (EPD runner, UCI mode, book builder, interactive CLI)
cmd/tuner/main.go    Texel tuner CLI (selfplay data generation, parameter optimization)
testdata/            Test data (wac.epd, ecm.epd, arasan.epd, lct.epd, sbd.epd, wac300.epd, wac2018.epd, zugzwang.epd, noob_3moves.epd, 2600.pgn, eco.pgn)
book.bin             Compiled opening book (binary format)
board.go             Board struct, piece types, FEN parsing, pieceOf() helper
move.go              Move encoding (16-bit), flags, NoMove sentinel
bitboard.go          Bitboard type, bit manipulation, file/rank masks
attacks.go           Magic bitboard tables, pre-computed attack lookups
movegen.go           Pseudo-legal/legal move generation, IsAttacked, InCheck
makemove.go          MakeMove/UnmakeMove, null move, UndoInfo
movepicker.go        Staged move ordering for search, IsPseudoLegal
search.go            Negamax, alpha-beta, iterative deepening, LMR, LMP, PVS, Lazy SMP
eval.go              Tapered eval: PST, mobility, king safety, positional bonuses, eval cache
pst.go               PeSTO piece-square tables, material values, phase constants
pawns.go             Pawn structure eval (doubled/isolated/passed), pawn hash table, pawn shield
tt.go                Transposition table (lockless, 4-slot buckets with Stockfish-style age/depth replacement)
zobrist.go           Zobrist hash keys, incremental hashing
see.go               Static Exchange Evaluation for capture ordering and quiet move pruning
san.go               SAN parsing (ParseSAN) and formatting (ToSAN)
epd.go               EPD file loading and test suite runner
pgn.go               PGN game parsing (tags, moves)
benchmark.go         Multi-suite benchmark: continuous scoring (time-to-solve), JSON save/load, comparison
book.go              Opening book: build from PGN, binary format, weighted move selection
uci.go               UCI protocol (position, go, setoption, ponder)
cli.go               Interactive CLI engine (set, fen, board, eval, moves, search, epd, perft)
selfplay.go          Self-play game generation for tuning data
tuner.go             Texel tuner: parameter catalog, trace-based eval, .tbin binary cache, disk-streamed Adam optimizer
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
- **Null-move pruning**: Skip turn and search with reduced depth (R=3 if depth>=7, else R=2). Requires depth >= 3, non-pawn material, not in check.
- **Reverse Futility Pruning**: At shallow depths (depth <= 3), prune whole node if static eval minus margin (depth * 120) exceeds beta.
- **Futility pruning**: At depth <= 2, skip quiet non-checking moves when static eval plus margin cannot raise alpha.
- **Late Move Reductions (LMR)**: Logarithmic reduction table. Quiet moves searched late in the move list are reduced. Re-search at full depth if score exceeds alpha. Disabled for captures, promotions, killers, and check-giving moves. Continuous history adjustment: good history reduces less, bad history reduces more (histScore / 5000). Reduced less at PV nodes and when position is improving.
- **Late Move Pruning (LMP)**: Skip quiet moves at shallow depths (depth 1-8) after searching enough moves (threshold from `lmpThreshold[depth]` table). Disabled when in check or giving check.
- **SEE quiet pruning**: At depth <= 8, prune quiet moves where `SEEAfterQuiet` indicates the piece lands on a square where it would be captured for material loss exceeding `depth * 80` centipawns. Computed before MakeMove, applied after (to exempt check-giving moves). Exempts TT move, killers, counter-move, captures, promotions. Controlled by `SEEQuietPruneEnabled` toggle.
- **Singular extensions**: At depth >= 10, if the TT move is significantly better than alternatives (verified by a reduced-depth search excluding the TT move), extend the TT move by 1 ply.
- **Principal Variation Search (PVS)**: After first move, search with zero window (alpha, alpha+1). Re-search with full window if it fails high.
- **Aspiration windows**: Starting at depth 4, iterative deepening uses a narrow window (delta=25) around previous score. Widens progressively on fail high/low.
- **Check extensions**: Extend search by 1 ply when move gives check.
- **Recapture extensions**: Extend by 1 ply when recapturing on the same square the opponent just captured on, to fully resolve tactical exchanges. Gated on `RecaptureExtEnabled`.
- **Passed pawn push extensions**: Extend by 1 ply for quiet pawn pushes to 6th or 7th rank that are verified as passed pawns (via `PassedPawnMask`). Helps resolve promotion races and endgame tactics. Gated on `PassedPawnExtEnabled`.
- **Quiescence search**: Captures only at leaf nodes, pruned by SEE >= 0. Stand-pat evaluation as lower bound. Depth-limited to 32. When in check: uses evasion generator for all legal moves (not just captures), skips stand-pat, detects checkmate (moveCount == 0).
- **Killer moves**: 2 slots per ply, updated on beta cutoff with quiet moves.
- **Counter-move heuristic**: `CounterMoves[piece][toSquare]` indexed by opponent's previous move. Stored on beta cutoff, used as a MovePicker stage between killers and quiets.
- **History heuristic**: `history[from][to] += depth * depth` on beta cutoff. Quiet moves tried before the cutoff move receive a matching penalty (`-= depth * depth`). Used to score quiet moves in move ordering and to adjust LMR reductions.
- **Capture history**: `CaptHistory[piece][toSquare][capturedPieceType]` (int16, ~11.4KB per thread). Tracks which captures caused beta cutoffs. Updated with `depth * depth` bonus on cutoff, penalty for captures tried before. Added to MVV-LVA scores for good captures and to SEE scores for bad captures, improving ordering among equal-value captures.
- **Continuation history**: `ContHistory[prevPiece][prevTo][curPiece][curTo]` (int16, ~1.3MB per thread). Captures the pattern "after piece X moved to square Y, quiet move Z tends to be good/bad". Updated alongside History on quiet beta cutoffs (bonus) and for quiet moves tried before cutoff (penalty). Added to quiet move scores in MovePicker and to the LMR history adjustment. Nil-safe: disabled at root and after null moves.
- **Time management**: Dual soft/hard deadline system. `computeSearchTime()` returns soft (base allocation) and hard (3× soft, capped at 75% remaining) limits. Between iterations (depth ≥ 4), the soft deadline is dynamically scaled based on best-move stability (stable → scale down to 0.5×) and score instability (large delta → scale up to 1.4×). Mate-range score swings are ignored. The adjusted soft deadline is clamped to the hard deadline. Checks clock every 4096 nodes against hard deadline. Helper threads, EPD, benchmark, and `go movetime` use hard deadline only (SoftDeadline=0). `go infinite`/`go depth` set no deadlines.
- **Lazy SMP**: Multi-threaded search via `SearchParallel()`. All threads search the same root position independently, sharing only the transposition table. Each thread has its own `Board` copy (with undo stack for repetition detection), `SearchInfo` (killers, history, counter-moves), eval cache, and pawn hash table. Helper threads use depth diversification (a skip table indexed by thread index and depth) to ensure threads are at different depths at any given time, improving TT entry diversity. The main thread (thread 0) runs normally with the `OnDepth` callback; helper threads run a stripped-down iterative deepening loop. Node counts from all threads are aggregated for NPS reporting. Default: 1 thread (no behavior change). Configurable via UCI `Threads` option (1-256) and `-threads` CLI flag.

### Transposition Table Thread Safety

The TT uses a lockless scheme for concurrent access by multiple search threads:
- Each entry is stored as two `uint64` fields: `keyXor` and `data`
- `data` packs move (16 bits), flag (8 bits), score (16 bits), and depth (8 bits) into a single `uint64`
- `keyXor = key ^ data` — on read, the stored key is recovered as `keyXor ^ data` and verified against the requested key
- All reads/writes use `atomic.LoadUint64`/`atomic.StoreUint64`
- Torn reads (where one field is from an old write and the other from a new write) are detected by the XOR verification and treated as misses
- Stats counters use `atomic.AddUint64`

### Evaluation

Tapered evaluation blending middlegame and endgame scores based on game phase (piece count). `Evaluate()` returns White-relative centipawns; `EvaluateRelative()` returns side-to-move relative.

- **Piece-square tables** (pst.go): PeSTO tables for all piece types, separate MG/EG values with per-piece-type scaling factors (35-85%). Material values added separately: MG (P=82, N=337, B=365, R=477, Q=1025), EG (P=94, N=281, B=297, R=512, Q=936). SEE uses simplified values (P=100, N=320, B=330, R=500, Q=900). **Incremental PST**: Combined PST+material tables (`pstCombinedMG[piece][sq]`, `pstCombinedEG[piece][sq]`) bake material + scaled positional values together. `Board.PSTScoreMG[color]` and `Board.PSTScoreEG[color]` are maintained incrementally by `putPiece`/`removePiece`/`movePiece`, eliminating per-Evaluate full-board scans. `evaluatePST()` is retained for testing/verification.
- **Pawn structure** (pawns.go): Cached via pawn hash table. Evaluates doubled, isolated, backward, connected, and passed pawns. Pawn advancement bonus. **Candidate passed pawns**: pawns with no enemy pawn ahead on their own file and enough friendly adjacent pawns to outnumber enemy sentries; bonus by rank via `candidatePassedMG/EG[8]` (~45% of passed pawn base). Gated on `CandidatePassedEnabled`. **Pawn majority**: per-wing (queenside files a-d, kingside files e-h) bonus when one side has more pawns than the opponent; scales linearly with the pawn count advantage (`PawnMajorityMG/EG` per extra pawn). Wings where the side already has a passed pawn are skipped to avoid double-counting. Gated on `PawnMajorityEnabled`. Precomputed masks: `PassedPawnMask`, `ForwardFileMask`, `OutpostMask`, `AdjacentFiles`.
- **Mobility** (eval.go): Non-linear bonus arrays indexed by move count — `KnightMobility[9]`, `BishopMobility[14]`, `RookMobility[15]`, `QueenMobility[28]` — each with separate MG/EG values.
- **King safety** (eval.go + pawns.go): Table-driven system. Per-piece attack unit weights (Knight=7, Bishop=5, Rook=8, Queen=13) plus king-zone square bonuses accumulate into an attack score. **Safe check bonus**: after piece loops, for each piece type, checks if any reachable check square is "safe" (not defended by enemy pawns, not occupied by friendly pieces); if so, adds a fixed bonus per piece type (Knight=6, Bishop=3, Rook=7, Queen=5). Uses binary detection (any safe check exists), not per-square counting. Gated on `attackerCount >= 1`. **Pawn storm**: adds attack units for friendly pawns advanced near the enemy king, indexed by relative rank via `PawnStormUnits[8]` (max 3 units at rank 6, up to 9 total across 3 king-adjacent files). Gated on `attackerCount >= 1` and `PawnStormEnabled`. **No-queen scaling**: when the attacking side has no queen, the final king safety penalty is scaled down to `NoQueenAttackScale/128` (~31%). `KingSafetyTable[100]` maps total attack units to centipawn penalties. Pawn shield evaluation (ranks 2-3 around king) and semi-open file penalty near king in pawns.go.
- **Positional bonuses** (eval.go): Bishop pair, knight outposts (supported/unsupported), knight closed-position bonus (scales with pawn count), rook on open/semi-open file, rook on 7th rank, doubled rooks on same file, trapped rook penalty, bad bishop penalty (per friendly pawn on same square color), bishop open position scaling, castling rights MG bonus.
- **Passed pawn enhancements** (eval.go): Not-blocked bonus, free path to promotion, king proximity (friendly close / enemy far), protected passers, connected passers, rook behind passer. These depend on piece positions so they are not cached in the pawn table.
- **Space and threats** (eval.go): Space evaluation (safe squares in center files), pawn threats (pawns attacking enemy pieces).
- **Endgame king activity** (eval.go): Two EG-only components. (1) Unconditional centralization: both kings penalized per center-distance unit (`KingCenterBonusEG`), tapered eval scales this out in middlegames. (2) Material advantage bonuses: when one side has more material (weighted: N/B=1, R=3, Q=6), reward king proximity (`KingProximityAdvantageEG`) and pushing enemy king to edge (`KingCornerPushEG`). No piece-type gates.
- **Endgame scaling** (eval.go): Per-side scale factors (0-128) for draw/insufficient material detection. Handles KNN, KR vs KB/KN, pawnless 2 minors vs 1+ minor (KBB/KBN vs KB/KN, scale 16), opposite-colored bishop drawishness (OCBScale=64), and 50-move rule scaling.
- **Eval cache** (eval.go): `EvalTable` caches full `Evaluate()` results keyed by Zobrist hash. Auto-initialized at 1 MB. Avoids redundant recomputation on transpositions.
- **Phase calculation**: Knight=1, Bishop=1, Rook=2, Queen=4, Total=24. Phase increases as pieces are traded.

### Zobrist Hashing

Incrementally updated in `MakeMove`/`UnmakeMove` via XOR. Keys cover piece-square, side to move, castling rights (4 individual keys), and en passant file (8 keys, by file not square). Separate `PawnHashKey` for pawn structure caching. Full recompute only in `SetFEN()`. Fixed seed for deterministic hashing.

### SEE (Static Exchange Evaluation)

Simulates alternating captures on a single square. Builds gain array, then negamax backward to find optimal result. `SEESign(move, threshold)` provides fast boolean check with early exits. Used in quiescence pruning and move ordering. `SEEAfterQuiet(move)` evaluates the exchange on the destination square after a quiet move, returning 0 (safe) or negative (material loss); used for quiet move pruning in search.

### Opening Book

Binary format built from PGN games (e.g. `testdata/2600.pgn`) and ECO classifications (`testdata/eco.pgn`). `BuildOpeningBook()` processes games up to a configurable depth, tracking move frequency. `PickMove()` selects from book entries using weighted random selection. Integrated into UCI engine via `OwnBook` and `BookFile` options.

### UCI Protocol

`uci.go` implements the Universal Chess Interface. Supports: `position` (startpos/FEN + moves), `go` (time controls, depth, movetime, infinite, ponder), `stop`, `setoption` (Hash, Threads, Ponder, OwnBook, BookFile), `ponderhit`. Search runs in a goroutine using `SearchParallel` with the configured thread count; `stop` signals all threads via `SearchInfo.Stopped`. Opening book consulted before search when enabled.

### EPD / Test Suites

`LoadEPDFile()` parses EPD format (4-field FEN + operations like `bm`, `am`, `id`). `RunEPDTest()` searches a position and checks if found move matches expected best move(s). Test suites in `testdata/`: WAC (201 positions), WAC300 (300), ECM (200), plus arasan, lct, sbd, wac2018, zugzwang.

### PGN Parsing

`ParsePGN()` / `ParsePGNFile()` parse PGN format into `PGNGame` structs (tag pairs + move list). Used by the opening book builder. Handles brace comments, NAGs, and result tokens.

### Texel Tuner

`cmd/tuner/main.go` — Two-phase system for optimizing ~1147 evaluation parameters.

**Self-play data generation** (`selfplay.go`): Plays engine-vs-engine games to produce training data. Each game uses `SearchParallel()` with configurable time/depth per move. Opening diversity from `testdata/noob_3moves.epd` (150K positions). Game termination: checkmate, stalemate, 50-move rule, threefold repetition, insufficient material, or adjudication (eval exceeds ±1000cp for 5 consecutive moves). Positions are filtered (skip first 8 plies, skip checks, skip mate scores) and written as `FEN;result` lines. Games run concurrently with independent Board+TT per game.

**Binary cache and disk-streamed training** (`tuner.go`): Training data is preprocessed from the `.dat` FEN file into a `.tbin` binary cache, then streamed from disk during tuning. This keeps memory usage at O(batch_size) (~few MB) regardless of dataset size.

- `.tbin` file format: 24-byte header (magic `"TBIN"`, version `uint16`, numParams `uint16`, numTrain `uint32`, numValidation `uint32`, trainBytes `uint64`) followed by variable-length records. Each record stores phase, result, scale factors, halfmove clock, and sparse MG/EG coefficient arrays. ~730 bytes/position average.
- `PreprocessToFile()` reads all FEN lines, shuffles deterministically (seed 42), splits 90/10 train/validation, computes traces via `computeTrace()`, writes binary records.
- `OpenTraceFile()` reads and validates the `.tbin` header, returning a `TraceFile` handle.
- `StreamTraining()` / `StreamValidation()` stream records in batches (default 65536) with reusable buffers. The callback receives a `[]TunerTrace` batch that must not be retained.
- The `.tbin` is auto-created on first run and auto-rebuilt when the source `.dat` file is newer.

**Parameter optimization** (`tuner.go`): Texel tuning via Adam optimizer. The `Tuner` holds the parameter catalog (no in-memory training data):

- `initTunerParams()` builds a flat parameter vector from engine globals: material (10), PST (768, pre-scaled by PST scale factors), mobility (132), piece bonuses (22), passed pawn enhancements (48), pawn structure (40), king attack weights (8), king safety table (100), pawn shield (5), misc (14).
- `computeTrace()` mirrors `Evaluate()` but records sparse MG/EG coefficients per parameter instead of computing a score. Each position produces a `TunerTrace` with `[]SparseEntry` for MG and EG contributions.
- `scoreFromTrace()` evaluates: `(mg * (24 - phase) + eg * phase) / 24`
- `sigmoid(score, K)` maps score to win probability: `1 / (1 + 10^(-score/K))`
- `TuneK(tf)` finds optimal K via golden section search on MSE over streamed training data
- `Tune(tf, K, cfg, onEpoch)` runs Adam optimizer: per epoch, streams training batches from disk, computes parallel gradients within each batch, aggregates across all batches, then applies Adam update
- `ComputeTrainError(tf, K)` / `ComputeValidationError(tf, K)` compute MSE by streaming from the respective region of the `.tbin` file

**What's tuned**: Material values, PST tables, mobility arrays, piece bonuses (bishop pair, outposts, rook file, trapped rook, etc.), passed pawn enhancements (not-blocked, free path, king proximity, connected, protected), pawn structure (passed base, doubled, isolated, backward, connected, advancement), king attack unit weights, safe check bonuses (knight, bishop, rook, queen), king safety table (100-entry nonlinear lookup), pawn shield constants (shield rank bonuses, missing shield penalties, semi-open file penalty), endgame king activity (centralization, proximity, corner push), space/threat/castling bonuses, tempo, trade bonuses, OCB scale.

**What's NOT tuned**: Endgame scale factors (multiplicative), phase constants, PST scale factors (folded into PST values), 50-move rule scaling, no-queen attack scale (multiplicative, can't represent in trace).

**PST scaling note**: Tuned PST values are *effective* values (raw table value × scale/100). Output sets all PST scale factors to 100. This avoids coupling between table values and scale factors during optimization.

### CLI

`cmd/chess/main.go` — Five modes:

- **Interactive CLI**: Default when stdin is a terminal and no flags given. Provides commands: `set`, `fen`, `board`, `reset`, `moves`, `search`, `eval`, `epd`, `perft`, `uci`. Implemented in `cli.go` (`CLIEngine`).
- **EPD testing**: `-e` (EPD file), `-t` (ms per position), `-n` (max positions), `-d` (max depth), `-hash` (TT size MB), `-v` (verbose per-position output), `-threads` (Lazy SMP thread count)
- **Benchmark**: `-benchmark` runs WAC, ECM, SBD, Arasan suites with continuous time-to-solve scoring. `-save FILE` writes JSON results, `-compare FILE` shows delta against a baseline. Reuses `-t`, `-hash`, `-d`, `-threads`.
- **UCI mode**: `-uci`, with optional `-book` (opening book path)
- **Book building**: `-buildbook`, `-pgn`, `-eco`, `-bookout`, `-bookdepth`, `-bookminfreq`, `-booktop`

## Key Gotchas

- Move flag equality vs bitwise: see "Critical rule" above.
- `UnmakeMove()` panics on empty undo stack or move mismatch — always pair with `MakeMove()`.
- `GenerateAllMoves()` returns pseudo-legal moves; search must call `IsLegal()`.
- Castling rights are lost both when a rook/king moves AND when a rook is captured on its home square.
- En passant hash uses file only (8 keys), not full square.
- TT mate scores are ply-adjusted: stored as `mate + ply`, retrieved as `mate - ply`.
- Eval cache uses full `HashKey` (includes side-to-move). `Evaluate()` is White-relative, so same position with different side-to-move gets separate cache entries — correct but slightly lower hit rate.
- Pawn hash table and eval cache auto-initialize on first `Evaluate()` call if nil.
- Lazy SMP threads share only the TT. Board, SearchInfo, eval cache, and pawn table are per-thread. The `Stopped` and `Deadline` fields on `SearchInfo` are accessed atomically (`int32`/`int64`). When adding new shared state, it must use atomic operations or be per-thread.
- TT `Probe`/`Store` are lockless via packed atomics — do not add non-atomic fields to `ttSlot`.
- Tuner PST parameters are pre-scaled (raw × scale/100). Adding a new PST-related eval term requires initializing the tuner param with the effective value, not the raw table value.
- Tuner traces must mirror `Evaluate()` exactly. When modifying eval, update `computeTrace()` in tuner.go to match. Non-additive eval terms (king safety table lookup, endgame scaling, 50-move scaling) cannot be represented in the trace.
- Tuner's `PassedPawnKingScale` uses initial engine values as constant coefficients to avoid circular dependency with distance parameters (product of two tunable params).
- `.tbin` binary cache must be rebuilt (delete it or touch the `.dat` file) whenever `computeTrace()` or the parameter catalog changes. The header stores `numParams` as a sanity check, but structural changes within the same param count won't be caught.
- `StreamTraining`/`StreamValidation` callbacks must not retain references to the `[]TunerTrace` batch — the backing arrays are reused across batches.

## Maintenance Reminders

- **Keep CLAUDE.md and README.md up to date.** When making changes to search, evaluation, tuner, CLI, or architecture, update the corresponding sections in both files so they stay accurate and useful.
- **When adding a new evaluation parameter**, consider whether it should be tunable. If so, add it to `initTunerParams()` in `tuner.go`, add the corresponding trace coefficient logic to `computeTrace()`, add it to `PrintParams()`, and update the "What's tuned" list in this file and the README. Delete any existing `.tbin` cache so it gets rebuilt with the new param count.

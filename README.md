# gochess

A chess engine written in Go, built entirely through collaboration with Claude as an experiment in AI-assisted software development. The engine uses bitboard representation, magic bitboards for sliding piece attacks, and implements standard search and evaluation techniques found in modern chess engines.

## Features

- Bitboard-based board representation with magic bitboard move generation
- Negamax search with alpha-beta pruning, iterative deepening, and PVS
- Lazy SMP multi-threaded search (configurable thread count)
- Lockless transposition table, null-move pruning, late move reductions, late move pruning
- Tapered evaluation with piece-square tables, pawn structure, mobility, king safety
- Optional NNUE evaluation (HalfKA architecture, SIMD-accelerated on x86-64 and ARM64)
- Texel tuner for automated evaluation parameter optimization via self-play (disk-streamed, constant memory)
- Syzygy endgame tablebase support (3-4-5-6 piece, via bundled Fathom C library)
- Polyglot opening book support (standard .bin format, compatible with any Polyglot book)
- Full UCI protocol support for use with chess GUIs
- EPD test suite runner (WAC, ECM)

## Building

Requires Go 1.21 or later and a C compiler (for Syzygy tablebase support via CGO).

```bash
go build -o chess ./cmd/chess   # Chess engine
go build -o tuner ./cmd/tuner   # Texel tuner
```

The engine bundles the [Fathom](https://github.com/jdart1/Fathom) C library for Syzygy tablebase probing. This is compiled automatically via CGO during `go build` — no separate build step is needed. CGO is enabled by default in Go; if you've disabled it (`CGO_ENABLED=0`), the engine will still build but without tablebase support.

## Usage

The engine has five modes of operation: interactive CLI (default), UCI mode, EPD test suite runner, benchmark, and opening book builder.

### UCI Mode

Start the engine in UCI protocol mode for use with a chess GUI:

```bash
./chess -uci
```

The engine also enters UCI mode automatically when stdin is not a terminal (e.g., when launched by a GUI).

#### UCI Options

| Option | Default | Description |
|--------|---------|-------------|
| `Hash` | 64 | Transposition table size in MB |
| `Threads` | 1 | Number of search threads (Lazy SMP) |
| `Ponder` | false | Enable pondering |
| `UseNNUE` | true | Enable NNUE evaluation (auto-loads `net.nnue` from working directory) |
| `NNUEFile` | | Path to NNUE network file (`.nnue`); overrides auto-detection |
| `OwnBook` | false | Use the engine's opening book |
| `BookFile` | | Path to opening book file |
| `SyzygyPath` | | Path to Syzygy tablebase files |
| `SyzygyProbeDepth` | 1 | Minimum depth for WDL probing during search |

#### Connecting to a Chess GUI

The engine speaks the [UCI protocol](https://www.chessprogramming.org/UCI) and works with any UCI-compatible GUI. Some popular options:

- [Arena](http://www.playwitharena.de/) (Windows, free)
- [CuteChess](https://cutechess.com/) (cross-platform, free)
- [Lucas Chess](https://lucaschess.pythonanywhere.com/) (Windows, free)
- [Banksia](https://banksiagui.com/) (cross-platform, free)
- [Scid vs. PC](https://scidvspc.sourceforge.net/) (cross-platform, free)

To add the engine to your GUI:

1. Build the binary with `go build -o chess ./cmd/chess`
2. In your GUI, find the engine management settings (usually under "Engines" or "Engine Management")
3. Add a new engine and point it to the `chess` binary
4. The GUI will communicate with the engine over UCI automatically

To use an opening book with a GUI, either set the `OwnBook` and `BookFile` UCI options through the GUI's engine configuration, or start the engine with the `-book` flag. Any standard Polyglot `.bin` book file will work:

```bash
./chess -book book.bin
```

### EPD Test Suites

Run standard test suites (e.g., WAC, ECM) to evaluate engine strength:

```bash
# Run the WAC suite with 5 seconds per position
./chess -e testdata/wac.epd -t 5000

# Run with 4 search threads (Lazy SMP)
./chess -e testdata/wac.epd -t 5000 -threads 4

# Run the first 20 positions with verbose output
./chess -e testdata/wac.epd -t 5000 -n 20 -v

# Run with custom depth limit and hash size
./chess -e testdata/ecm.epd -d 20 -hash 128
```

| Flag | Default | Description |
|------|---------|-------------|
| `-e` | | EPD file to run |
| `-t` | 5000 | Time per position in milliseconds |
| `-n` | 0 | Number of positions to run (0 = all) |
| `-d` | 64 | Maximum search depth |
| `-hash` | 64 | Transposition table size in MB |
| `-v` | false | Verbose output with board display and per-depth info |
| `-threads` | 1 | Number of search threads (Lazy SMP) |

### Building an Opening Book

Build a Polyglot `.bin` opening book from a PGN database of games:

```bash
./chess -buildbook -pgn testdata/2600.pgn -bookout book.bin
```

| Flag | Default | Description |
|------|---------|-------------|
| `-pgn` | | PGN file with games (required) |
| `-bookout` | book.bin | Output file path |
| `-bookdepth` | 30 | Max full moves to include |
| `-bookminfreq` | 3 | Minimum frequency to include a move |
| `-booktop` | 8 | Maximum moves per position |

You can also use any pre-built Polyglot `.bin` book downloaded from the internet.

### Texel Tuner

The tuner optimizes ~1268 evaluation parameters (material values, piece-square tables, mobility, positional bonuses, pawn structure, king safety table, pawn shield, king attack weights, trade bonuses) by minimizing prediction error against game outcomes. Training data is preprocessed into a compact binary cache (`.tbin`) and streamed from disk during optimization, keeping memory usage constant regardless of dataset size. The workflow has two steps: generate training data via self-play, then run gradient-descent optimization.

#### Step 1: Generate training data

```bash
./tuner selfplay -games 20000 -time 200 -concurrency 6
./tuner selfplay -games 20000 -depth 8 -concurrency 6
./tuner selfplay -games 20000 -time 200 -concurrency 6 -classical   # Use classical eval
```

Selfplay uses NNUE evaluation by default (auto-loads `net.nnue` from the working directory). Use `-classical` to fall back to handcrafted eval, or `-nnue path/to/net.nnue` to specify a network file.

This plays self-play games using opening positions from `testdata/noob_3moves.epd` for diversity. Each game records positions with the search score and game result in binpack format (`.bin`). Games are adjudicated when eval exceeds ±1000cp for 5 consecutive moves. Positions are filtered to skip the first 8 plies, positions where the side to move is in check, and positions with mate scores.

Use `-time` for time-limited or `-depth` for depth-limited search (mutually exclusive). Depth-limited mode ensures consistent data quality across different machines.

| Flag | Default | Description |
|------|---------|-------------|
| `-games` | 1000 | Number of games to play |
| `-time` | 0 | Time per move in ms (mutually exclusive with `-depth`) |
| `-depth` | 0 | Fixed search depth per move (mutually exclusive with `-time`) |
| `-concurrency` | 1 | Parallel games (set to ~CPU core count) |
| `-threads` | 1 | Search threads per game (Lazy SMP) |
| `-hash` | 16 | TT size in MB per game |
| `-openings` | `testdata/noob_3moves.epd` | EPD file with starting positions |
| `-output` | `training.bin` | Output file for training data |
| `-nnue` | | NNUE network file (default: auto-load `net.nnue`) |
| `-classical` | false | Disable NNUE, use classical eval only |

If neither `-time` nor `-depth` is specified, defaults to `-time 200`. With `-time 200 -concurrency 6`, expect roughly 1-2 games/second. 20K games produces ~1-2M training positions.

#### Step 2: Tune parameters

```bash
./tuner tune -data training.bin -epochs 500 -lr 1.0
```

On first run, this preprocesses the data file into a binary cache (`training.tbin`) — shuffling positions, computing evaluation traces, and writing a compact binary format (~730 bytes/position). Subsequent runs reuse the cache automatically; the cache is rebuilt if the source data file is newer.

During tuning, training data is streamed from the `.tbin` file in batches of 65536, keeping memory usage at ~50-100 MB regardless of dataset size. The tuner finds the optimal sigmoid scaling constant K via golden section search, then runs the Adam optimizer. It prints the error every 10 epochs and outputs all tuned parameters in Go source format at the end.

| Flag | Default | Description |
|------|---------|-------------|
| `-data` | `training.bin` | Training data file from step 1 (.bin) |
| `-epochs` | 500 | Number of optimization epochs |
| `-lr` | 1.0 | Learning rate |
| `-lambda` | 1.0 | Result vs score weight: 1=result-only, 0=score-only |
| `-l2` | 0 | L2 regularization strength toward initial values |

#### Step 3: Apply tuned values

The tuner prints updated values as Go source code. Copy the relevant sections into the engine source:

- **`pst.go`** — material values (`mgMaterial`, `egMaterial`), PST tables (`mg*Table`, `eg*Table`). Set all `*PSTScale*` variables to `100` (tuned values are pre-scaled).
- **`eval.go`** — mobility arrays, piece bonuses, passed pawn bonuses, king attack weights, king safety table, space/threat/castling parameters.
- **`pawns.go`** — passed pawn base arrays, pawn advancement arrays, structure constants, pawn shield constants.

#### Step 4: Verify

```bash
./chess -e testdata/wac.epd -t 5000 -n 20
```

Compare pass rate and log-scores against the baseline to confirm the tuned values are an improvement.

#### Tips

- 20K games at 200ms/move is a reasonable minimum. 50K+ games at 500ms produces higher quality data.
- More training positions generally helps more than more epochs.
- Error should decrease monotonically. If it stalls early, generate more data.
- The tuner does not optimize endgame scale factors (multiplicative) or phase constants.
- The `.tbin` cache is tied to the parameter catalog. If you add or remove tunable parameters, delete the `.tbin` file to force a rebuild.

### NNUE Evaluation

The engine supports an optional NNUE (Efficiently Updatable Neural Network) evaluation that can replace the classical handcrafted eval. The network uses a HalfKA architecture: 12288 inputs (16 king buckets × 12 piece types × 64 squares), two 256-neuron accumulators (one per perspective), concatenated into a 512→32→32→1 output. The hidden layer uses int8 quantized weights for doubled SIMD throughput. The forward pass is SIMD-accelerated with AVX2 on x86-64 and NEON on ARM64.

#### Training an NNUE network

**Step 1: Generate training data**

```bash
./tuner selfplay -games 20000 -time 200 -concurrency 6
```

Selfplay records the engine's search score alongside each position and game result. Both the game result and the search score are used as training targets.

**Step 2: Train the network**

```bash
./tuner nnue-train -data training.bin -epochs 100 -lr 0.01 -output net.nnue
```

This trains a quantized NNUE network from scratch. The loss function blends game-result prediction with score prediction: `lambda * MSE(sigmoid(nnue/K), result) + (1-lambda) * MSE(sigmoid(nnue/K), sigmoid(score/K))`.

| Flag | Default | Description |
|------|---------|-------------|
| `-data` | `training.bin` | Training data file (.bin, must include scores) |
| `-epochs` | 100 | Training epochs |
| `-lr` | 0.01 | Learning rate |
| `-lambda` | 0.5 | Blend between result (1.0) and score (0.0) targets |
| `-K` | 400 | Sigmoid scaling constant |
| `-output` | `net.nnue` | Output network file |
| `-positions` | 0 | Limit training positions per epoch (0=use all) |
| `-resume` | | Resume training from an existing `.nnue` network file |

Training uses float32 weights internally, then quantizes to int16 for inference (int8 for the hidden layer).

**Two-phase training workflow:**

You can train on a subset first, then fine-tune on the full dataset:

```bash
# Phase 1: Quick initial training on 50K positions
./tuner nnue-train -data training.bin -positions 50000 -epochs 50 -lr 0.01 -output net-v1.nnue

# Phase 2: Fine-tune on full dataset starting from Phase 1 weights
./tuner nnue-train -data training.bin -resume net-v1.nnue -epochs 100 -lr 0.005 -output net-v2.nnue
```

#### Using NNUE in UCI mode

The engine auto-loads `net.nnue` from the working directory on startup. To use a different network:

```bash
./chess -nnue /path/to/custom.nnue -uci
```

Use `-classical` to disable NNUE and use the handcrafted eval. You can also toggle NNUE at runtime via UCI options:

```
setoption name UseNNUE value true
setoption name NNUEFile value /path/to/net.nnue
```

#### Using NNUE in the interactive CLI

NNUE is enabled by default when `net.nnue` is present. Use `-classical` for handcrafted eval:

```bash
./chess              # Auto-loads net.nnue if present
./chess -classical   # Force classical eval
```

In the interactive CLI, additional NNUE commands are available:

| Command | Description |
|---------|-------------|
| `nnue load <file>` | Load a network file |
| `nnue on` | Enable NNUE evaluation |
| `nnue off` | Switch back to classical evaluation |
| `nnue eval` | Show NNUE evaluation of the current position |

#### Performance

The NNUE forward pass is accelerated with platform-specific SIMD assembly:

- **x86-64 (AVX2)**: ~10× faster than the generic Go implementation
- **ARM64 (NEON)**: SIMD always available, no runtime detection needed

Accumulator updates are incremental — only changed features are updated when pieces move. King moves trigger a full recompute since they change the perspective for all features.

### Syzygy Tablebases

The engine supports [Syzygy endgame tablebases](https://syzygy-tables.info/) for perfect play in positions with few pieces. When enabled, the engine will:

- **At the root**: Probe DTZ (Distance To Zeroing move) tables to immediately return the optimal move without searching.
- **During search**: Probe WDL (Win/Draw/Loss) tables to get exact scores for positions with ≤N pieces, cutting off the search tree early.

#### Obtaining tablebase files

Syzygy tables come in two file types — `.rtbw` (WDL) and `.rtbz` (DTZ). Both are needed. You can download them from:

- https://tablebase.lichess.ovh/tables/standard/ (3-4-5 piece: ~1GB, 6-piece: ~150GB)
- Various chess tablebase mirrors

Download the WDL and DTZ files into the same directory:

```bash
mkdir -p /path/to/tablebases
# Download 3-4-5 piece tables (recommended minimum)
# See https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/ and 3-4-5-dtz/
```

#### Using tablebases

Via CLI flag:

```bash
./chess -syzygy /path/to/tablebases -uci
```

Via UCI option (from a GUI or at runtime):

```
setoption name SyzygyPath value /path/to/tablebases
setoption name SyzygyProbeDepth value 1
```

| Option | Default | Description |
|--------|---------|-------------|
| `SyzygyPath` | | Path to directory containing `.rtbw` and `.rtbz` files |
| `SyzygyProbeDepth` | 1 | Minimum search depth to probe WDL tables (higher = less overhead at shallow depths) |

The engine reports `tbhits` in UCI info strings when tablebases are active.

## Running Tests

```bash
# Run all tests (includes WAC/ECM suites, takes a few minutes)
go test

# Run unit tests only (skip slow EPD suites)
go test -short

# Run benchmarks
go test -bench .
```

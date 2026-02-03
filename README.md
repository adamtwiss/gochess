# gochess

A chess engine written in Go, built entirely through collaboration with Claude as an experiment in AI-assisted software development. The engine uses bitboard representation, magic bitboards for sliding piece attacks, and implements standard search and evaluation techniques found in modern chess engines.

## Features

- Bitboard-based board representation with magic bitboard move generation
- Negamax search with alpha-beta pruning, iterative deepening, and PVS
- Lazy SMP multi-threaded search (configurable thread count)
- Lockless transposition table, null-move pruning, late move reductions, late move pruning
- Tapered evaluation with piece-square tables, pawn structure, mobility, king safety
- Texel tuner for automated evaluation parameter optimization via self-play
- Opening book support (built from PGN databases)
- Full UCI protocol support for use with chess GUIs
- EPD test suite runner (WAC, ECM)

## Building

Requires Go 1.21 or later.

```bash
go build -o chess ./cmd/chess   # Chess engine
go build -o tuner ./cmd/tuner   # Texel tuner
```

## Usage

The engine has three modes of operation: UCI mode, EPD test suite runner, and opening book builder.

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
| `OwnBook` | false | Use the engine's opening book |
| `BookFile` | | Path to opening book file |

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

To use the opening book with a GUI, either set the `OwnBook` and `BookFile` UCI options through the GUI's engine configuration, or start the engine with the `-book` flag:

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

Build an opening book from a PGN database of games:

```bash
./chess -buildbook -pgn testdata/2600.pgn -eco testdata/eco.pgn -bookout book.bin
```

| Flag | Default | Description |
|------|---------|-------------|
| `-pgn` | | PGN file with games (required) |
| `-eco` | | ECO PGN file for opening names |
| `-bookout` | book.bin | Output file path |
| `-bookdepth` | 30 | Max full moves to include |
| `-bookminfreq` | 3 | Minimum frequency to include a move |
| `-booktop` | 8 | Maximum moves per position |

### Texel Tuner

The tuner optimizes ~1143 evaluation parameters (material values, piece-square tables, mobility, positional bonuses, pawn structure, king safety table, pawn shield, king attack weights) by minimizing prediction error against game outcomes. The workflow has two steps: generate training data via self-play, then run gradient-descent optimization.

#### Step 1: Generate training data

```bash
./tuner selfplay -games 20000 -time 200 -concurrency 6 -output training.dat
```

This plays self-play games using opening positions from `testdata/noob_3moves.epd` for diversity. Each game records positions with the game result (1.0/0.5/0.0 from White's perspective). Games are adjudicated when eval exceeds ±1000cp for 5 consecutive moves. Positions are filtered to skip the first 8 plies, positions where the side to move is in check, and positions with mate scores.

| Flag | Default | Description |
|------|---------|-------------|
| `-games` | 1000 | Number of games to play |
| `-time` | 200 | Time per move in milliseconds |
| `-concurrency` | 1 | Parallel games (set to ~CPU core count) |
| `-threads` | 1 | Search threads per game (Lazy SMP) |
| `-hash` | 16 | TT size in MB per game |
| `-openings` | `testdata/noob_3moves.epd` | EPD file with starting positions |
| `-output` | `training.dat` | Output file for training data |

With `-time 200 -concurrency 6`, expect roughly 1-2 games/second. 20K games produces ~1-2M training positions.

#### Step 2: Tune parameters

```bash
./tuner tune -data training.dat -epochs 500 -lr 1.0
```

This loads the training data, finds the optimal sigmoid scaling constant K via golden section search, then runs the Adam optimizer. It prints the error every 10 epochs and outputs all tuned parameters in Go source format at the end.

| Flag | Default | Description |
|------|---------|-------------|
| `-data` | `training.dat` | Training data file from step 1 |
| `-epochs` | 500 | Number of optimization epochs |
| `-lr` | 1.0 | Learning rate |

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

## Running Tests

```bash
# Run all tests (includes WAC/ECM suites, takes a few minutes)
go test

# Run unit tests only (skip slow EPD suites)
go test -short

# Run benchmarks
go test -bench .
```

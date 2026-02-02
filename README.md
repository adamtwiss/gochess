# gochess

A chess engine written in Go, built entirely through collaboration with Claude as an experiment in AI-assisted software development. The engine uses bitboard representation, magic bitboards for sliding piece attacks, and implements standard search and evaluation techniques found in modern chess engines.

## Features

- Bitboard-based board representation with magic bitboard move generation
- Negamax search with alpha-beta pruning, iterative deepening, and PVS
- Lazy SMP multi-threaded search (configurable thread count)
- Lockless transposition table, null-move pruning, late move reductions, late move pruning
- Tapered evaluation with piece-square tables, pawn structure, mobility, king safety
- Opening book support (built from PGN databases)
- Full UCI protocol support for use with chess GUIs
- EPD test suite runner (WAC, ECM)

## Building

Requires Go 1.21 or later.

```bash
go build -o chess ./cmd/chess
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

## Running Tests

```bash
# Run all tests (includes WAC/ECM suites, takes a few minutes)
go test

# Run unit tests only (skip slow EPD suites)
go test -short

# Run benchmarks
go test -bench .
```

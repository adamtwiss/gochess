package chess

import (
	"encoding/binary"
	"fmt"
	"os"
	"sort"
)

// BookBuildOptions controls how the opening book is built.
type BookBuildOptions struct {
	MaxPly  int // Max ply depth to record (default 60 = 30 moves)
	MinFreq int // Min frequency to include a move (default 3)
	TopN    int // Max moves per position (default 8)
}

// BuildOpeningBook builds a Polyglot .bin opening book from a PGN file.
func BuildOpeningBook(pgnFile, outFile string, opts BookBuildOptions) error {
	if opts.MaxPly <= 0 {
		opts.MaxPly = 60
	}
	if opts.MinFreq <= 0 {
		opts.MinFreq = 3
	}
	if opts.TopN <= 0 {
		opts.TopN = 8
	}

	// Parse GM games for move frequencies using Polyglot hashes
	type moveKey struct {
		hash uint64
		move uint16 // Polyglot-encoded move
	}
	freqs := make(map[moveKey]int)

	games, err := ParsePGNFile(pgnFile)
	if err != nil {
		return fmt.Errorf("parsing PGN file: %w", err)
	}

	gamesProcessed := 0
	for _, g := range games {
		var b Board
		b.Reset()
		for ply, san := range g.Moves {
			if ply >= opts.MaxPly {
				break
			}
			hash := PolyglotHash(&b)
			m, err := b.ParseSAN(san)
			if err != nil {
				break
			}
			pm := moveToPolyglot(m, &b)
			freqs[moveKey{hash, pm}]++
			b.MakeMove(m)
		}
		gamesProcessed++
	}
	fmt.Printf("Games: %d processed\n", gamesProcessed)

	// Group by hash, trim by frequency and top-N
	type entry struct {
		key    uint64
		move   uint16
		weight uint16
	}
	grouped := make(map[uint64][]entry)
	for mk, freq := range freqs {
		if freq < opts.MinFreq {
			continue
		}
		w := freq
		if w > 65535 {
			w = 65535
		}
		grouped[mk.hash] = append(grouped[mk.hash], entry{mk.hash, mk.move, uint16(w)})
	}

	// Sort each position's moves by weight desc, trim to TopN
	var allEntries []entry
	for _, moves := range grouped {
		sort.Slice(moves, func(i, j int) bool {
			return moves[i].weight > moves[j].weight
		})
		if len(moves) > opts.TopN {
			moves = moves[:opts.TopN]
		}
		allEntries = append(allEntries, moves...)
	}

	// Sort all entries by key (Polyglot requirement)
	sort.Slice(allEntries, func(i, j int) bool {
		if allEntries[i].key != allEntries[j].key {
			return allEntries[i].key < allEntries[j].key
		}
		return allEntries[i].weight > allEntries[j].weight
	})

	fmt.Printf("Book: %d positions, %d entries after trimming (minFreq=%d, topN=%d)\n",
		len(grouped), len(allEntries), opts.MinFreq, opts.TopN)

	// Write Polyglot .bin format
	f, err := os.Create(outFile)
	if err != nil {
		return err
	}
	defer f.Close()

	var buf [16]byte
	for _, e := range allEntries {
		binary.BigEndian.PutUint64(buf[0:8], e.key)
		binary.BigEndian.PutUint16(buf[8:10], e.move)
		binary.BigEndian.PutUint16(buf[10:12], e.weight)
		binary.BigEndian.PutUint32(buf[12:16], 0) // learn data
		if _, err := f.Write(buf[:]); err != nil {
			return err
		}
	}

	return nil
}

// moveToPolyglot converts an internal Move to Polyglot encoding.
// Polyglot encodes castling as king-to-rook (e.g. e1g1 -> e1h1).
func moveToPolyglot(m Move, b *Board) uint16 {
	from := m.From()
	to := m.To()
	fromFile := from.File()
	fromRank := from.Rank()

	// Convert castling from king-to-destination to king-to-rook
	if m.Flags() == FlagCastle {
		if fromFile == 4 { // king on e-file
			if fromRank == 0 { // white
				if to.File() == 6 { // g1 -> h1 (kingside)
					to = Square(7)
				} else if to.File() == 2 { // c1 -> a1 (queenside)
					to = Square(0)
				}
			} else if fromRank == 7 { // black
				if to.File() == 6 { // g8 -> h8
					to = Square(63)
				} else if to.File() == 2 { // c8 -> a8
					to = Square(56)
				}
			}
		}
	}

	var promo uint16
	switch m.Flags() {
	case FlagPromoteN:
		promo = 1
	case FlagPromoteB:
		promo = 2
	case FlagPromoteR:
		promo = 3
	case FlagPromoteQ:
		promo = 4
	}

	return uint16(to.File()) |
		uint16(to.Rank())<<3 |
		uint16(from.File())<<6 |
		uint16(from.Rank())<<9 |
		promo<<12
}


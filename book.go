package chess

import (
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"
)

// BookMove is a single move option in an opening book position.
type BookMove struct {
	Move   Move   // 16-bit encoded move
	Weight uint16 // Frequency count
}

// BookEntry holds all known moves for a single position hash.
type BookEntry struct {
	Hash  uint64
	Moves []BookMove // Sorted by weight descending
	Name  string     // Opening name (empty if none)
}

// OpeningBook provides weighted random move selection from a prebuilt book.
type OpeningBook struct {
	entries map[uint64]*BookEntry
	rng     *rand.Rand
	useBook bool // OwnBook toggle
}

// BookBuildOptions controls how the opening book is built.
type BookBuildOptions struct {
	MaxPly  int // Max ply depth to record (default 60 = 30 moves)
	MinFreq int // Min frequency to include a move (default 3)
	TopN    int // Max moves per position (default 8)
}

const (
	bookMagic   = "CBOK"
	bookVersion = uint16(1)
	noNameTag   = uint32(0xFFFFFFFF)
)

// BuildOpeningBook builds an opening book from PGN files and writes it to outFile.
func BuildOpeningBook(pgnFile, ecoFile, outFile string, opts BookBuildOptions) error {
	if opts.MaxPly <= 0 {
		opts.MaxPly = 60
	}
	if opts.MinFreq <= 0 {
		opts.MinFreq = 3
	}
	if opts.TopN <= 0 {
		opts.TopN = 8
	}

	// Phase 1: Parse ECO file for opening names
	ecoNames := make(map[uint64]string)
	if ecoFile != "" {
		ecoGames, err := ParsePGNFile(ecoFile)
		if err != nil {
			return fmt.Errorf("parsing ECO file: %w", err)
		}
		for _, g := range ecoGames {
			name := buildECOName(g.Tags)
			if name == "" {
				continue
			}
			var b Board
			b.Reset()
			for _, san := range g.Moves {
				m, err := b.ParseSAN(san)
				if err != nil {
					break
				}
				b.MakeMove(m)
			}
			// Record name at the final position
			ecoNames[b.HashKey] = name
		}
		fmt.Printf("ECO: %d named positions from %d entries\n", len(ecoNames), len(ecoGames))
	}

	// Phase 2: Parse GM games for move frequencies
	type posData struct {
		moves map[Move]uint16
	}
	positions := make(map[uint64]*posData)

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
			hash := b.HashKey
			m, err := b.ParseSAN(san)
			if err != nil {
				break // stop this game on parse error
			}
			pd := positions[hash]
			if pd == nil {
				pd = &posData{moves: make(map[Move]uint16)}
				positions[hash] = pd
			}
			w := pd.moves[m]
			if w < 65535 {
				pd.moves[m] = w + 1
			}
			b.MakeMove(m)
		}
		gamesProcessed++
	}
	fmt.Printf("Games: %d processed, %d positions recorded\n", gamesProcessed, len(positions))

	// Phase 3: Build entries, merge ECO names, trim
	entries := make(map[uint64]*BookEntry, len(positions))
	for hash, pd := range positions {
		var moves []BookMove
		for m, w := range pd.moves {
			if int(w) >= opts.MinFreq {
				moves = append(moves, BookMove{Move: m, Weight: w})
			}
		}
		if len(moves) == 0 {
			continue
		}
		// Sort by weight descending
		sort.Slice(moves, func(i, j int) bool {
			return moves[i].Weight > moves[j].Weight
		})
		if len(moves) > opts.TopN {
			moves = moves[:opts.TopN]
		}
		entry := &BookEntry{
			Hash:  hash,
			Moves: moves,
			Name:  ecoNames[hash],
		}
		entries[hash] = entry
	}
	fmt.Printf("Book: %d positions after trimming (minFreq=%d, topN=%d)\n",
		len(entries), opts.MinFreq, opts.TopN)

	// Phase 4: Serialize to binary
	return writeBookFile(outFile, entries)
}

func buildECOName(tags map[string]string) string {
	parts := []string{}
	if eco := tags["ECO"]; eco != "" {
		parts = append(parts, eco)
	}
	if opening := tags["Opening"]; opening != "" {
		parts = append(parts, opening)
	}
	if variation := tags["Variation"]; variation != "" {
		parts = append(parts, variation)
	}
	return strings.Join(parts, ", ")
}

func writeBookFile(filename string, entries map[uint64]*BookEntry) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	// Build string table
	stringTable := []byte{}
	stringOffsets := make(map[string]uint32)
	for _, e := range entries {
		if e.Name == "" {
			continue
		}
		if _, ok := stringOffsets[e.Name]; !ok {
			stringOffsets[e.Name] = uint32(len(stringTable))
			stringTable = append(stringTable, []byte(e.Name)...)
			stringTable = append(stringTable, 0) // null terminator
		}
	}

	// Write header (32 bytes)
	header := make([]byte, 32)
	copy(header[0:4], bookMagic)
	binary.LittleEndian.PutUint16(header[4:6], bookVersion)
	// header[6:8] reserved
	binary.LittleEndian.PutUint32(header[8:12], uint32(len(entries)))
	binary.LittleEndian.PutUint32(header[12:16], uint32(len(stringTable)))
	// header[16:32] reserved
	if _, err := f.Write(header); err != nil {
		return err
	}

	// Write string table
	if _, err := f.Write(stringTable); err != nil {
		return err
	}

	// Write entries
	for _, e := range entries {
		// Hash (8 bytes)
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], e.Hash)
		if _, err := f.Write(buf[:]); err != nil {
			return err
		}

		// NameOffset (4 bytes)
		var nameOff uint32 = noNameTag
		if e.Name != "" {
			nameOff = stringOffsets[e.Name]
		}
		var nbuf [4]byte
		binary.LittleEndian.PutUint32(nbuf[:], nameOff)
		if _, err := f.Write(nbuf[:]); err != nil {
			return err
		}

		// MoveCount (1 byte)
		if _, err := f.Write([]byte{byte(len(e.Moves))}); err != nil {
			return err
		}

		// Moves (4 bytes each: uint16 move + uint16 weight)
		for _, m := range e.Moves {
			var mbuf [4]byte
			binary.LittleEndian.PutUint16(mbuf[0:2], uint16(m.Move))
			binary.LittleEndian.PutUint16(mbuf[2:4], m.Weight)
			if _, err := f.Write(mbuf[:]); err != nil {
				return err
			}
		}
	}

	return nil
}

// LoadOpeningBook loads an opening book from a binary file.
func LoadOpeningBook(filename string) (*OpeningBook, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return parseBookData(data)
}

func parseBookData(data []byte) (*OpeningBook, error) {
	if len(data) < 32 {
		return nil, fmt.Errorf("book file too small")
	}

	// Validate header
	if string(data[0:4]) != bookMagic {
		return nil, fmt.Errorf("invalid book magic: %q", string(data[0:4]))
	}
	version := binary.LittleEndian.Uint16(data[4:6])
	if version != bookVersion {
		return nil, fmt.Errorf("unsupported book version: %d", version)
	}
	entryCount := binary.LittleEndian.Uint32(data[8:12])
	stringTableSize := binary.LittleEndian.Uint32(data[12:16])

	// Parse string table
	strStart := uint32(32)
	strEnd := strStart + stringTableSize
	if uint32(len(data)) < strEnd {
		return nil, fmt.Errorf("truncated string table")
	}
	stringTable := data[strStart:strEnd]

	// Parse entries
	pos := strEnd
	entries := make(map[uint64]*BookEntry, entryCount)
	for i := uint32(0); i < entryCount; i++ {
		if int(pos)+13 > len(data) {
			return nil, fmt.Errorf("truncated entry %d", i)
		}

		hash := binary.LittleEndian.Uint64(data[pos : pos+8])
		nameOff := binary.LittleEndian.Uint32(data[pos+8 : pos+12])
		moveCount := int(data[pos+12])
		pos += 13

		if int(pos)+moveCount*4 > len(data) {
			return nil, fmt.Errorf("truncated moves in entry %d", i)
		}

		var name string
		if nameOff != noNameTag && nameOff < uint32(len(stringTable)) {
			end := nameOff
			for end < uint32(len(stringTable)) && stringTable[end] != 0 {
				end++
			}
			name = string(stringTable[nameOff:end])
		}

		moves := make([]BookMove, moveCount)
		for j := 0; j < moveCount; j++ {
			mv := binary.LittleEndian.Uint16(data[pos : pos+2])
			wt := binary.LittleEndian.Uint16(data[pos+2 : pos+4])
			moves[j] = BookMove{Move: Move(mv), Weight: wt}
			pos += 4
		}

		entries[hash] = &BookEntry{
			Hash:  hash,
			Moves: moves,
			Name:  name,
		}
	}

	return &OpeningBook{
		entries: entries,
		rng:     rand.New(rand.NewSource(time.Now().UnixNano())),
		useBook: true,
	}, nil
}

// Lookup returns the book entry for a position hash, or nil.
func (book *OpeningBook) Lookup(hash uint64) *BookEntry {
	return book.entries[hash]
}

// PickMove selects a weighted random move for the given position hash.
// Returns (move, openingName, found).
func (book *OpeningBook) PickMove(hash uint64) (Move, string, bool) {
	if !book.useBook {
		return NoMove, "", false
	}
	entry := book.entries[hash]
	if entry == nil || len(entry.Moves) == 0 {
		return NoMove, "", false
	}

	// Sum weights
	total := 0
	for _, m := range entry.Moves {
		total += int(m.Weight)
	}

	// Weighted random selection
	r := book.rng.Intn(total)
	cumulative := 0
	for _, m := range entry.Moves {
		cumulative += int(m.Weight)
		if r < cumulative {
			return m.Move, entry.Name, true
		}
	}

	// Fallback (shouldn't reach)
	return entry.Moves[0].Move, entry.Name, true
}

// SetUseBook enables or disables book lookups.
func (book *OpeningBook) SetUseBook(use bool) {
	book.useBook = use
}

// Size returns the number of positions in the book.
func (book *OpeningBook) Size() int {
	return len(book.entries)
}

// serializeForTest is a helper to serialize and deserialize in memory for testing.
func serializeBookEntries(entries map[uint64]*BookEntry) ([]byte, error) {
	r, w := io.Pipe()
	errCh := make(chan error, 1)
	go func() {
		defer w.Close()

		// Build string table
		stringTable := []byte{}
		stringOffsets := make(map[string]uint32)
		for _, e := range entries {
			if e.Name == "" {
				continue
			}
			if _, ok := stringOffsets[e.Name]; !ok {
				stringOffsets[e.Name] = uint32(len(stringTable))
				stringTable = append(stringTable, []byte(e.Name)...)
				stringTable = append(stringTable, 0)
			}
		}

		header := make([]byte, 32)
		copy(header[0:4], bookMagic)
		binary.LittleEndian.PutUint16(header[4:6], bookVersion)
		binary.LittleEndian.PutUint32(header[8:12], uint32(len(entries)))
		binary.LittleEndian.PutUint32(header[12:16], uint32(len(stringTable)))
		if _, err := w.Write(header); err != nil {
			errCh <- err
			return
		}
		if _, err := w.Write(stringTable); err != nil {
			errCh <- err
			return
		}

		for _, e := range entries {
			var buf [8]byte
			binary.LittleEndian.PutUint64(buf[:], e.Hash)
			w.Write(buf[:])

			var nameOff uint32 = noNameTag
			if e.Name != "" {
				nameOff = stringOffsets[e.Name]
			}
			var nbuf [4]byte
			binary.LittleEndian.PutUint32(nbuf[:], nameOff)
			w.Write(nbuf[:])

			w.Write([]byte{byte(len(e.Moves))})

			for _, m := range e.Moves {
				var mbuf [4]byte
				binary.LittleEndian.PutUint16(mbuf[0:2], uint16(m.Move))
				binary.LittleEndian.PutUint16(mbuf[2:4], m.Weight)
				w.Write(mbuf[:])
			}
		}
		errCh <- nil
	}()

	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	if err := <-errCh; err != nil {
		return nil, err
	}
	return data, nil
}

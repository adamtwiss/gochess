package chess

// Transposition table for storing search results

// TTFlag indicates the type of score stored
type TTFlag uint8

const (
	TTNone  TTFlag = iota
	TTExact        // Exact score
	TTLower        // Lower bound (score >= beta, cut-off)
	TTUpper        // Upper bound (score <= alpha, failed low)
)

// TTEntry is a single transposition table entry
type TTEntry struct {
	Key   uint64 // Zobrist hash key (for collision detection)
	Depth int8   // Search depth
	Score int16  // Evaluation score
	Flag  TTFlag // Type of score
	Move  Move   // Best move found
}

// TranspositionTable stores search results for position lookup
type TranspositionTable struct {
	entries []TTEntry
	size    uint64
	mask    uint64 // size - 1, for fast modulo
	probes  uint64 // Stats: total probes
	hits    uint64 // Stats: successful probes
	writes  uint64 // Stats: entries written
}

// NewTranspositionTable creates a new TT with the given size in MB
func NewTranspositionTable(sizeMB int) *TranspositionTable {
	// Calculate number of entries
	entrySize := uint64(16) // sizeof(TTEntry): uint64 + int8 + pad + int16 + uint8 + pad + uint16
	numEntries := uint64(sizeMB*1024*1024) / entrySize

	// Round down to power of 2 for fast indexing
	size := uint64(1)
	for size*2 <= numEntries {
		size *= 2
	}

	return &TranspositionTable{
		entries: make([]TTEntry, size),
		size:    size,
		mask:    size - 1,
	}
}

// Clear resets all entries in the table
func (tt *TranspositionTable) Clear() {
	for i := range tt.entries {
		tt.entries[i] = TTEntry{}
	}
	tt.probes = 0
	tt.hits = 0
	tt.writes = 0
}

// index returns the table index for a hash key
func (tt *TranspositionTable) index(key uint64) uint64 {
	return key & tt.mask
}

// Probe looks up a position in the table
// Returns the entry and whether it was found
func (tt *TranspositionTable) Probe(key uint64) (TTEntry, bool) {
	tt.probes++
	entry := tt.entries[tt.index(key)]
	if entry.Key == key && entry.Flag != TTNone {
		tt.hits++
		return entry, true
	}
	return TTEntry{}, false
}

// Store saves a position to the table
// Uses depth-preferred replacement: only replace if new depth >= stored depth
func (tt *TranspositionTable) Store(key uint64, depth int, score int, flag TTFlag, move Move) {
	idx := tt.index(key)
	entry := &tt.entries[idx]

	// Replace if:
	// - Empty slot
	// - New search is at least as deep
	// For same position, we still require depth >= to avoid overwriting deep results
	if entry.Flag == TTNone || int8(depth) >= entry.Depth {
		entry.Key = key
		entry.Depth = int8(depth)
		entry.Score = int16(score)
		entry.Flag = flag
		entry.Move = move
		tt.writes++
	}
}

// Stats returns probe count, hit count and write count
func (tt *TranspositionTable) Stats() (probes, hits, writes uint64) {
	return tt.probes, tt.hits, tt.writes
}

// Hashfull returns permill of table entries used (for UCI info)
func (tt *TranspositionTable) Hashfull() int {
	used := 0
	// Sample first 1000 entries for speed
	sample := min(1000, len(tt.entries))
	for i := 0; i < sample; i++ {
		if tt.entries[i].Flag != TTNone {
			used++
		}
	}
	return used * 1000 / sample
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

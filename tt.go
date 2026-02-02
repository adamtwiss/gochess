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

// TTBucket holds two TT entries: slot 0 is depth-preferred, slot 1 is always-replace
type TTBucket struct {
	entries [2]TTEntry
}

// TranspositionTable stores search results for position lookup
type TranspositionTable struct {
	buckets []TTBucket
	size    uint64 // number of buckets
	mask    uint64 // size - 1, for fast modulo
	probes  uint64 // Stats: total probes
	hits    uint64 // Stats: successful probes
	writes  uint64 // Stats: entries written
}

// NewTranspositionTable creates a new TT with the given size in MB
func NewTranspositionTable(sizeMB int) *TranspositionTable {
	// Calculate number of buckets (32 bytes each: 2 x 16-byte entries)
	bucketSize := uint64(32)
	numBuckets := uint64(sizeMB*1024*1024) / bucketSize

	// Round down to power of 2 for fast indexing
	size := uint64(1)
	for size*2 <= numBuckets {
		size *= 2
	}

	return &TranspositionTable{
		buckets: make([]TTBucket, size),
		size:    size,
		mask:    size - 1,
	}
}

// Clear resets all entries in the table
func (tt *TranspositionTable) Clear() {
	for i := range tt.buckets {
		tt.buckets[i] = TTBucket{}
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
	bucket := &tt.buckets[tt.index(key)]

	// Check slot 0 (depth-preferred)
	if bucket.entries[0].Key == key && bucket.entries[0].Flag != TTNone {
		tt.hits++
		return bucket.entries[0], true
	}

	// Check slot 1 (always-replace)
	if bucket.entries[1].Key == key && bucket.entries[1].Flag != TTNone {
		tt.hits++
		return bucket.entries[1], true
	}

	return TTEntry{}, false
}

// Store saves a position to the table
// Slot 0: depth-preferred replacement. Slot 1: always-replace.
func (tt *TranspositionTable) Store(key uint64, depth int, score int, flag TTFlag, move Move) {
	bucket := &tt.buckets[tt.index(key)]
	slot0 := &bucket.entries[0]
	slot1 := &bucket.entries[1]

	d := int8(depth)

	// If same position already in slot 0, update if depth >=
	if slot0.Key == key {
		if d >= slot0.Depth {
			slot0.Depth = d
			slot0.Score = int16(score)
			slot0.Flag = flag
			slot0.Move = move
			tt.writes++
		}
		return
	}

	// If same position already in slot 1, always update
	if slot1.Key == key {
		slot1.Depth = d
		slot1.Score = int16(score)
		slot1.Flag = flag
		slot1.Move = move
		tt.writes++
		return
	}

	// New position: try depth-preferred into slot 0, otherwise always-replace into slot 1
	if slot0.Flag == TTNone || d >= slot0.Depth {
		slot0.Key = key
		slot0.Depth = d
		slot0.Score = int16(score)
		slot0.Flag = flag
		slot0.Move = move
		tt.writes++
	} else {
		slot1.Key = key
		slot1.Depth = d
		slot1.Score = int16(score)
		slot1.Flag = flag
		slot1.Move = move
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
	// Sample first 1000 buckets, count non-empty slots across both
	sampleBuckets := min(1000, len(tt.buckets))
	totalSlots := sampleBuckets * 2
	for i := 0; i < sampleBuckets; i++ {
		if tt.buckets[i].entries[0].Flag != TTNone {
			used++
		}
		if tt.buckets[i].entries[1].Flag != TTNone {
			used++
		}
	}
	return used * 1000 / totalSlots
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

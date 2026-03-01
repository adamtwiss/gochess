package chess

import "sync/atomic"

// Transposition table for storing search results

// TTFlag indicates the type of score stored
type TTFlag uint8

const (
	TTNone  TTFlag = iota
	TTExact        // Exact score
	TTLower        // Lower bound (score >= beta, cut-off)
	TTUpper        // Upper bound (score <= alpha, failed low)
)

// TTEntry is a single transposition table entry (used by callers)
type TTEntry struct {
	Key        uint64 // Zobrist hash key (for collision detection)
	Depth      int8   // Search depth
	Score      int16  // Evaluation score
	Flag       TTFlag // Type of score
	Move       Move   // Best move found
	StaticEval int16  // Static eval (stored as int8 * 4, range ±508)
}

// ttSlot stores a TT entry as two uint64 fields for lockless concurrent access.
// keyXor = key XOR data — torn reads are detected by checking key XOR keyXor == data.
type ttSlot struct {
	keyXor uint64
	data   uint64
}

// packTTData packs entry fields into a single uint64:
//
//	bits  0-15: Move (uint16)
//	bits 16-23: Flag (uint8)
//	bits 24-39: Score (int16, stored as uint16)
//	bits 40-47: Depth (int8, stored as uint8)
//	bits 48-55: Generation (uint8)
//	bits 56-63: StaticEval (int8, actual eval = int16(int8) * 4)
func packTTData(depth int8, score int16, flag TTFlag, move Move, gen uint8, staticEval int16) uint64 {
	// Compress static eval to int8 by dividing by 4 (range ±508cp)
	se := staticEval / 4
	if se > 127 {
		se = 127
	} else if se < -128 {
		se = -128
	}
	return uint64(move) |
		uint64(flag)<<16 |
		uint64(uint16(score))<<24 |
		uint64(uint8(depth))<<40 |
		uint64(gen)<<48 |
		uint64(uint8(int8(se)))<<56
}

// unpackTTData unpacks a data uint64 into entry fields.
func unpackTTData(data uint64) (depth int8, score int16, flag TTFlag, move Move, gen uint8, staticEval int16) {
	move = Move(data & 0xFFFF)
	flag = TTFlag((data >> 16) & 0xFF)
	score = int16(uint16((data >> 24) & 0xFFFF))
	depth = int8(uint8((data >> 40) & 0xFF))
	gen = uint8((data >> 48) & 0xFF)
	staticEval = int16(int8(uint8((data >> 56) & 0xFF))) * 4
	return
}

// TTBucket holds four TT slots (64 bytes = 1 cache line).
type TTBucket struct {
	slots [4]ttSlot
}

// TranspositionTable stores search results for position lookup
type TranspositionTable struct {
	buckets    []TTBucket
	size       uint64 // number of buckets
	mask       uint64 // size - 1, for fast modulo
	generation uint8  // incremented each search for age-based replacement
	probes     uint64 // Stats: total probes (atomic)
	hits       uint64 // Stats: successful probes (atomic)
	writes     uint64 // Stats: entries written (atomic)
}

// NewTranspositionTable creates a new TT with the given size in MB
func NewTranspositionTable(sizeMB int) *TranspositionTable {
	// Calculate number of buckets (64 bytes each: 4 x 16-byte slots)
	bucketSize := uint64(64)
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

// NewSearch increments the generation counter. Call once at the start of each
// new search (before iterative deepening). Wraps naturally at 256.
func (tt *TranspositionTable) NewSearch() {
	tt.generation++
}

// Clear resets all entries in the table
func (tt *TranspositionTable) Clear() {
	for i := range tt.buckets {
		tt.buckets[i] = TTBucket{}
	}
	tt.generation = 0
	atomic.StoreUint64(&tt.probes, 0)
	atomic.StoreUint64(&tt.hits, 0)
	atomic.StoreUint64(&tt.writes, 0)
}

// index returns the table index for a hash key
func (tt *TranspositionTable) index(key uint64) uint64 {
	return key & tt.mask
}

// Probe looks up a position in the table.
// Uses atomic loads and XOR verification for lockless thread safety.
func (tt *TranspositionTable) Probe(key uint64) (TTEntry, bool) {
	atomic.AddUint64(&tt.probes, 1)
	bucket := &tt.buckets[tt.index(key)]

	// Check all 4 slots
	for i := 0; i < 4; i++ {
		slot := &bucket.slots[i]
		data := atomic.LoadUint64(&slot.data)
		keyXor := atomic.LoadUint64(&slot.keyXor)

		// Verify: stored key = keyXor XOR data; check it matches requested key
		if keyXor^data != key {
			continue
		}

		depth, score, flag, move, _, staticEval := unpackTTData(data)
		if flag == TTNone {
			continue
		}

		atomic.AddUint64(&tt.hits, 1)
		return TTEntry{
			Key:        key,
			Depth:      depth,
			Score:      score,
			Flag:       flag,
			Move:       move,
			StaticEval: staticEval,
		}, true
	}

	return TTEntry{}, false
}

// Store saves a position to the table.
// Uses atomic stores with XOR key verification for lockless thread safety.
// Replacement uses Stockfish-style scoring: depth - 4*age. Stale entries
// from older generations are cheap to evict; current-generation deep entries
// are preserved.
func (tt *TranspositionTable) Store(key uint64, depth int, score int, flag TTFlag, move Move, staticEval int) {
	bucket := &tt.buckets[tt.index(key)]
	d := int8(depth)
	gen := tt.generation

	newData := packTTData(d, int16(score), flag, move, gen, int16(staticEval))

	// Scan all 4 slots: look for key match, empty slot, and worst-scoring slot
	replaceIdx := 0
	replaceScore := int(1<<30) // impossibly high so any real slot beats it
	for i := 0; i < 4; i++ {
		slotData := atomic.LoadUint64(&bucket.slots[i].data)
		slotKeyXor := atomic.LoadUint64(&bucket.slots[i].keyXor)
		slotKey := slotKeyXor ^ slotData

		slotDepth, _, slotFlag, _, slotGen, _ := unpackTTData(slotData)

		// Empty slot: use immediately
		if slotFlag == TTNone {
			atomic.StoreUint64(&bucket.slots[i].data, newData)
			atomic.StoreUint64(&bucket.slots[i].keyXor, key^newData)
			atomic.AddUint64(&tt.writes, 1)
			return
		}

		// Key match: update if newer generation or deeper/equal depth
		if slotKey == key {
			if d >= slotDepth || gen != slotGen {
				atomic.StoreUint64(&bucket.slots[i].data, newData)
				atomic.StoreUint64(&bucket.slots[i].keyXor, key^newData)
				atomic.AddUint64(&tt.writes, 1)
			}
			return
		}

		// Track worst slot for replacement: score = depth - 4*age
		// Lower score = better replacement candidate
		age := int(gen-slotGen) & 0xFF
		slotScore := int(slotDepth) - 4*age
		if slotScore < replaceScore {
			replaceScore = slotScore
			replaceIdx = i
		}
	}

	// No key match and no empty slot: replace worst-scoring slot
	atomic.StoreUint64(&bucket.slots[replaceIdx].data, newData)
	atomic.StoreUint64(&bucket.slots[replaceIdx].keyXor, key^newData)
	atomic.AddUint64(&tt.writes, 1)
}

// Stats returns probe count, hit count and write count
func (tt *TranspositionTable) Stats() (probes, hits, writes uint64) {
	return atomic.LoadUint64(&tt.probes), atomic.LoadUint64(&tt.hits), atomic.LoadUint64(&tt.writes)
}

// Hashfull returns permill of table entries used (for UCI info)
func (tt *TranspositionTable) Hashfull() int {
	used := 0
	sampleBuckets := min(1000, len(tt.buckets))
	totalSlots := sampleBuckets * 4
	for i := 0; i < sampleBuckets; i++ {
		for j := 0; j < 4; j++ {
			data := atomic.LoadUint64(&tt.buckets[i].slots[j].data)
			_, _, flag, _, _, _ := unpackTTData(data)
			if flag != TTNone {
				used++
			}
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

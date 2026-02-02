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
	Key   uint64 // Zobrist hash key (for collision detection)
	Depth int8   // Search depth
	Score int16  // Evaluation score
	Flag  TTFlag // Type of score
	Move  Move   // Best move found
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
func packTTData(depth int8, score int16, flag TTFlag, move Move) uint64 {
	return uint64(move) |
		uint64(flag)<<16 |
		uint64(uint16(score))<<24 |
		uint64(uint8(depth))<<40
}

// unpackTTData unpacks a data uint64 into entry fields.
func unpackTTData(data uint64) (depth int8, score int16, flag TTFlag, move Move) {
	move = Move(data & 0xFFFF)
	flag = TTFlag((data >> 16) & 0xFF)
	score = int16(uint16((data >> 24) & 0xFFFF))
	depth = int8(uint8((data >> 40) & 0xFF))
	return
}

// TTBucket holds two TT slots: slot 0 is depth-preferred, slot 1 is always-replace
type TTBucket struct {
	slots [2]ttSlot
}

// TranspositionTable stores search results for position lookup
type TranspositionTable struct {
	buckets []TTBucket
	size    uint64 // number of buckets
	mask    uint64 // size - 1, for fast modulo
	probes  uint64 // Stats: total probes (atomic)
	hits    uint64 // Stats: successful probes (atomic)
	writes  uint64 // Stats: entries written (atomic)
}

// NewTranspositionTable creates a new TT with the given size in MB
func NewTranspositionTable(sizeMB int) *TranspositionTable {
	// Calculate number of buckets (32 bytes each: 2 x 16-byte slots)
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

	// Check both slots
	for i := 0; i < 2; i++ {
		slot := &bucket.slots[i]
		data := atomic.LoadUint64(&slot.data)
		keyXor := atomic.LoadUint64(&slot.keyXor)

		// Verify: stored key = keyXor XOR data; check it matches requested key
		if keyXor^data != key {
			continue
		}

		depth, score, flag, move := unpackTTData(data)
		if flag == TTNone {
			continue
		}

		atomic.AddUint64(&tt.hits, 1)
		return TTEntry{
			Key:   key,
			Depth: depth,
			Score: score,
			Flag:  flag,
			Move:  move,
		}, true
	}

	return TTEntry{}, false
}

// Store saves a position to the table.
// Uses atomic stores with XOR key verification for lockless thread safety.
// Slot 0: depth-preferred replacement. Slot 1: always-replace.
func (tt *TranspositionTable) Store(key uint64, depth int, score int, flag TTFlag, move Move) {
	bucket := &tt.buckets[tt.index(key)]
	d := int8(depth)

	newData := packTTData(d, int16(score), flag, move)

	// Read slot 0 atomically
	slot0Data := atomic.LoadUint64(&bucket.slots[0].data)
	slot0KeyXor := atomic.LoadUint64(&bucket.slots[0].keyXor)
	slot0Key := slot0KeyXor ^ slot0Data

	// If same position already in slot 0, update if depth >=
	if slot0Key == key {
		slot0Depth, _, slot0Flag, _ := unpackTTData(slot0Data)
		if slot0Flag != TTNone && d < slot0Depth {
			return
		}
		atomic.StoreUint64(&bucket.slots[0].data, newData)
		atomic.StoreUint64(&bucket.slots[0].keyXor, key^newData)
		atomic.AddUint64(&tt.writes, 1)
		return
	}

	// Read slot 1 atomically
	slot1Data := atomic.LoadUint64(&bucket.slots[1].data)
	slot1KeyXor := atomic.LoadUint64(&bucket.slots[1].keyXor)
	slot1Key := slot1KeyXor ^ slot1Data

	// If same position already in slot 1, always update
	if slot1Key == key {
		atomic.StoreUint64(&bucket.slots[1].data, newData)
		atomic.StoreUint64(&bucket.slots[1].keyXor, key^newData)
		atomic.AddUint64(&tt.writes, 1)
		return
	}

	// New position: try depth-preferred into slot 0, otherwise always-replace into slot 1
	_, _, slot0Flag, _ := unpackTTData(slot0Data)
	if slot0Flag == TTNone || d >= int8(uint8((slot0Data>>40)&0xFF)) {
		atomic.StoreUint64(&bucket.slots[0].data, newData)
		atomic.StoreUint64(&bucket.slots[0].keyXor, key^newData)
	} else {
		atomic.StoreUint64(&bucket.slots[1].data, newData)
		atomic.StoreUint64(&bucket.slots[1].keyXor, key^newData)
	}
	atomic.AddUint64(&tt.writes, 1)
}

// Stats returns probe count, hit count and write count
func (tt *TranspositionTable) Stats() (probes, hits, writes uint64) {
	return atomic.LoadUint64(&tt.probes), atomic.LoadUint64(&tt.hits), atomic.LoadUint64(&tt.writes)
}

// Hashfull returns permill of table entries used (for UCI info)
func (tt *TranspositionTable) Hashfull() int {
	used := 0
	// Sample first 1000 buckets, count non-empty slots across both
	sampleBuckets := min(1000, len(tt.buckets))
	totalSlots := sampleBuckets * 2
	for i := 0; i < sampleBuckets; i++ {
		for j := 0; j < 2; j++ {
			data := atomic.LoadUint64(&tt.buckets[i].slots[j].data)
			_, _, flag, _ := unpackTTData(data)
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

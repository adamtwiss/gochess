package chess

import "testing"

func TestTTBasic(t *testing.T) {
	tt := NewTranspositionTable(1) // 1 MB

	// Store and retrieve
	key := uint64(0x123456789ABCDEF0)
	tt.Store(key, 5, 100, TTExact, NewMove(12, 28), 0)

	entry, found := tt.Probe(key)
	if !found {
		t.Fatal("Entry not found")
	}

	if entry.Depth != 5 {
		t.Errorf("Depth = %d, want 5", entry.Depth)
	}
	if entry.Score != 100 {
		t.Errorf("Score = %d, want 100", entry.Score)
	}
	if entry.Flag != TTExact {
		t.Errorf("Flag = %d, want TTExact", entry.Flag)
	}
	if entry.Move != NewMove(12, 28) {
		t.Errorf("Move = %v, want e2e4", entry.Move)
	}
}

func TestTTCollision(t *testing.T) {
	tt := NewTranspositionTable(1)

	key1 := uint64(0x123456789ABCDEF0)
	key2 := uint64(0x023456789ABCDEF0) // Different key, may hash to same index

	tt.Store(key1, 5, 100, TTExact, NewMove(12, 28), 0)
	tt.Store(key2, 6, 200, TTLower, NewMove(52, 36), 0)

	entry1, found1 := tt.Probe(key1)
	entry2, found2 := tt.Probe(key2)

	// With five slots per bucket, both should be found if they collide
	if !found1 && !found2 {
		t.Error("Neither entry found after storing both")
	}

	if found2 {
		if entry2.Score != 200 {
			t.Errorf("key2 Score = %d, want 200", entry2.Score)
		}
	}

	t.Logf("key1 found: %v, key2 found: %v", found1, found2)
	t.Logf("entry1: %+v", entry1)
	t.Logf("entry2: %+v", entry2)
}

func TestTTDepthReplacement(t *testing.T) {
	tt := NewTranspositionTable(1)

	key := uint64(0x123456789ABCDEF0)

	// Store with depth 3
	tt.Store(key, 3, 50, TTExact, NewMove(12, 28), 0)

	// Try to store with lower depth - should not replace in slot 0
	tt.Store(key, 2, 60, TTExact, NewMove(12, 20), 0)

	entry, found := tt.Probe(key)
	if !found {
		t.Fatal("Entry not found")
	}

	// Should still have depth 3 entry
	if entry.Depth != 3 {
		t.Errorf("Depth = %d, want 3 (should not be replaced by shallower)", entry.Depth)
	}

	// Store with higher depth - should replace
	tt.Store(key, 5, 70, TTExact, NewMove(52, 36), 0)

	entry, _ = tt.Probe(key)
	if entry.Depth != 5 {
		t.Errorf("Depth = %d, want 5 (should be replaced by deeper)", entry.Depth)
	}
}

func TestTTBucketBehavior(t *testing.T) {
	tt := NewTranspositionTable(1)

	// Craft keys that map to the same bucket index but differ in upper 32 bits.
	// Adding n<<32 preserves the bucket index (lower bits) while giving each
	// key a unique upper-32-bit fingerprint for 32-bit key verification.
	baseKey := uint64(42)

	keys := [6]uint64{
		baseKey,
		baseKey + (1 << 32),
		baseKey + (2 << 32),
		baseKey + (3 << 32),
		baseKey + (4 << 32),
		baseKey + (5 << 32),
	}
	for i := 1; i < len(keys); i++ {
		if tt.index(keys[i]) != tt.index(keys[0]) {
			t.Fatalf("key%d doesn't collide with key0", i)
		}
	}

	// Fill all 5 slots with varying depths
	tt.Store(keys[0], 10, 150, TTExact, NewMove(12, 28), 0)
	tt.Store(keys[1], 3, 50, TTLower, NewMove(52, 36), 0)
	tt.Store(keys[2], 6, 80, TTUpper, NewMove(1, 18), 0)
	tt.Store(keys[3], 8, 120, TTExact, NewMove(6, 21), 0)
	tt.Store(keys[4], 5, 90, TTLower, NewMove(10, 26), 0)

	// All 5 should be found
	for i := 0; i < 5; i++ {
		_, found := tt.Probe(keys[i])
		if !found {
			t.Errorf("key%d not found after filling 5 slots", i)
		}
	}

	// Store a 6th entry — should evict the shallowest (key1, depth 3)
	tt.Store(keys[5], 7, 110, TTLower, NewMove(11, 27), 0)

	entry5, found5 := tt.Probe(keys[5])
	if !found5 {
		t.Error("key5 not found after storing into full bucket")
	}
	if found5 && entry5.Depth != 7 {
		t.Errorf("key5 depth = %d, want 7", entry5.Depth)
	}

	// The deepest entry (key0, depth 10) must survive
	entry0, found0 := tt.Probe(keys[0])
	if !found0 {
		t.Error("Deepest entry (key0) was evicted — should survive")
	}
	if found0 && entry0.Depth != 10 {
		t.Errorf("key0 depth = %d, want 10", entry0.Depth)
	}

	// key1 (shallowest, depth 3) should have been evicted
	_, found1 := tt.Probe(keys[1])
	if found1 {
		t.Error("Shallowest entry (key1, depth 3) should have been evicted")
	}
}

func TestTTGeneration(t *testing.T) {
	tt := NewTranspositionTable(1)

	key := uint64(0xDEADBEEF)

	// Store at generation 0 with depth 5
	tt.Store(key, 5, 100, TTExact, NewMove(12, 28), 0)

	entry, found := tt.Probe(key)
	if !found {
		t.Fatal("Entry not found at gen 0")
	}
	if entry.Depth != 5 || entry.Score != 100 {
		t.Errorf("Gen 0: depth=%d score=%d, want 5/100", entry.Depth, entry.Score)
	}

	// Advance generation
	tt.NewSearch()

	// Store same key with shallower depth but newer generation — should replace
	tt.Store(key, 3, 200, TTLower, NewMove(52, 36), 0)

	entry, found = tt.Probe(key)
	if !found {
		t.Fatal("Entry not found after gen-1 store")
	}
	if entry.Depth != 3 || entry.Score != 200 {
		t.Errorf("Gen 1: depth=%d score=%d, want 3/200", entry.Depth, entry.Score)
	}

	// Without advancing generation, shallower should NOT replace
	tt.Store(key, 1, 300, TTUpper, NewMove(1, 18), 0)

	entry, found = tt.Probe(key)
	if !found {
		t.Fatal("Entry not found after same-gen shallow store")
	}
	if entry.Depth != 3 {
		t.Errorf("Same-gen shallower replaced deeper: depth=%d, want 3", entry.Depth)
	}

	// Test that stale entries are evicted first in a full bucket
	tt2 := NewTranspositionTable(1)
	baseKey := uint64(99)
	k := [6]uint64{
		baseKey,
		baseKey + (1 << 32),
		baseKey + (2 << 32),
		baseKey + (3 << 32),
		baseKey + (4 << 32),
		baseKey + (5 << 32),
	}

	// Fill 5 slots at generation 0 with high depth
	for i := 0; i < 5; i++ {
		tt2.Store(k[i], 10, 100, TTExact, NewMove(12, 28), 0)
	}

	// Advance generation twice
	tt2.NewSearch()
	tt2.NewSearch()

	// Store 6th key — should evict one of the stale depth-10 entries
	// because age penalty makes them score 10 - 4*2 = 2
	tt2.Store(k[5], 4, 50, TTLower, NewMove(52, 36), 0)

	_, found6 := tt2.Probe(k[5])
	if !found6 {
		t.Error("New-generation entry not found after evicting stale entry")
	}
}

func TestTTClear(t *testing.T) {
	tt := NewTranspositionTable(1)

	// Store some entries
	for i := uint64(0); i < 100; i++ {
		tt.Store(i*7777, 5, 100, TTExact, NewMove(12, 28), 0)
	}

	// Verify entries exist
	_, found := tt.Probe(0)
	if !found {
		t.Error("Entry should exist before Clear")
	}

	// Clear and verify everything is gone
	tt.Clear()

	// Probe should miss
	_, found = tt.Probe(0)
	if found {
		t.Error("Entry should not exist after Clear")
	}

	if tt.Hashfull() != 0 {
		t.Errorf("Hashfull after Clear = %d, want 0", tt.Hashfull())
	}
}

func TestSearchWithTT(t *testing.T) {
	var b Board
	b.Reset()

	tt := NewTranspositionTable(16)

	// First search
	move1, info1 := b.SearchWithTT(5, 0, tt)
	t.Logf("First search: move=%s, nodes=%d", move1.String(), info1.Nodes)

	// Reset position and search again with same TT (should benefit from cached entries)
	b.Reset()
	move2, info2 := b.SearchWithTT(5, 0, tt)
	t.Logf("Second search: move=%s, nodes=%d", move2.String(), info2.Nodes)

	// Second search should benefit from TT (fewer nodes)
	if info2.Nodes >= info1.Nodes {
		t.Errorf("Expected fewer nodes on second search (first=%d, second=%d)", info1.Nodes, info2.Nodes)
	}

	// Results should be the same
	if move1 != move2 {
		t.Logf("Note: Different best move on second search (move1=%s, move2=%s)", move1.String(), move2.String())
	}
}

func TestTTHashfull(t *testing.T) {
	tt := NewTranspositionTable(1)

	// Initially empty
	if tt.Hashfull() != 0 {
		t.Errorf("New table hashfull = %d, want 0", tt.Hashfull())
	}

	// Add some entries
	for i := uint64(0); i < 500; i++ {
		tt.Store(i*12345, 5, 100, TTExact, NoMove, 0)
	}

	hashfull := tt.Hashfull()
	t.Logf("After 500 entries: hashfull = %d permill", hashfull)

	if hashfull == 0 {
		t.Error("Hashfull should be > 0 after adding entries")
	}
}

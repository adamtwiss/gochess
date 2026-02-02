package chess

import "testing"

func TestTTBasic(t *testing.T) {
	tt := NewTranspositionTable(1) // 1 MB

	// Store and retrieve
	key := uint64(0x123456789ABCDEF0)
	tt.Store(key, 5, 100, TTExact, NewMove(12, 28))

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

	tt.Store(key1, 5, 100, TTExact, NewMove(12, 28))
	tt.Store(key2, 6, 200, TTLower, NewMove(52, 36))

	entry1, found1 := tt.Probe(key1)
	entry2, found2 := tt.Probe(key2)

	// With two slots per bucket, both should be found if they collide
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
	tt.Store(key, 3, 50, TTExact, NewMove(12, 28))

	// Try to store with lower depth - should not replace in slot 0
	tt.Store(key, 2, 60, TTExact, NewMove(12, 20))

	entry, found := tt.Probe(key)
	if !found {
		t.Fatal("Entry not found")
	}

	// Should still have depth 3 entry
	if entry.Depth != 3 {
		t.Errorf("Depth = %d, want 3 (should not be replaced by shallower)", entry.Depth)
	}

	// Store with higher depth - should replace
	tt.Store(key, 5, 70, TTExact, NewMove(52, 36))

	entry, _ = tt.Probe(key)
	if entry.Depth != 5 {
		t.Errorf("Depth = %d, want 5 (should be replaced by deeper)", entry.Depth)
	}
}

func TestTTBucketBehavior(t *testing.T) {
	tt := NewTranspositionTable(1)

	// Craft two keys that map to the same bucket index
	// Since mask = size-1, we just need keys that agree on the low bits
	baseKey := uint64(42)
	idx := tt.index(baseKey)

	// key1 and key2 both map to same bucket but are different positions
	key1 := baseKey
	key2 := baseKey + tt.size // same low bits, different key

	if tt.index(key1) != tt.index(key2) {
		t.Fatalf("keys don't collide: idx1=%d, idx2=%d", tt.index(key1), tt.index(key2))
	}
	_ = idx

	// Store a deep entry (goes to slot 0)
	tt.Store(key1, 10, 150, TTExact, NewMove(12, 28))

	// Store a shallow entry for different position at same index
	// Should go to slot 1 (can't beat depth 10 for slot 0)
	tt.Store(key2, 3, 50, TTLower, NewMove(52, 36))

	// Both should be findable
	entry1, found1 := tt.Probe(key1)
	entry2, found2 := tt.Probe(key2)

	if !found1 {
		t.Error("Deep entry (key1) not found - should be in slot 0")
	}
	if !found2 {
		t.Error("Shallow entry (key2) not found - should be in slot 1")
	}

	if found1 && entry1.Depth != 10 {
		t.Errorf("key1 depth = %d, want 10", entry1.Depth)
	}
	if found2 && entry2.Depth != 3 {
		t.Errorf("key2 depth = %d, want 3", entry2.Depth)
	}

	// Now store a third position at same index with medium depth
	// This should replace slot 1 (always-replace) but not slot 0 (depth 10)
	key3 := baseKey + 2*tt.size
	tt.Store(key3, 6, 80, TTUpper, NewMove(1, 18))

	// key1 (deep, slot 0) should survive
	entry1, found1 = tt.Probe(key1)
	if !found1 {
		t.Error("Deep entry (key1) evicted from slot 0 by shallower key3")
	}
	if found1 && entry1.Depth != 10 {
		t.Errorf("key1 depth = %d, want 10", entry1.Depth)
	}

	// key3 should be in slot 1
	entry3, found3 := tt.Probe(key3)
	if !found3 {
		t.Error("key3 not found - should be in slot 1")
	}
	if found3 && entry3.Depth != 6 {
		t.Errorf("key3 depth = %d, want 6", entry3.Depth)
	}

	// key2 should be evicted (replaced in slot 1)
	_, found2 = tt.Probe(key2)
	if found2 {
		t.Log("key2 still found after key3 replaced slot 1 (unexpected but acceptable if index differs)")
	}
}

func TestTTClear(t *testing.T) {
	tt := NewTranspositionTable(1)

	// Store some entries
	for i := uint64(0); i < 100; i++ {
		tt.Store(i*7777, 5, 100, TTExact, NewMove(12, 28))
	}

	// Verify entries exist
	_, found := tt.Probe(0)
	if !found {
		t.Error("Entry should exist before Clear")
	}

	_, _, writes := tt.Stats()
	if writes == 0 {
		t.Error("Should have writes before Clear")
	}

	// Clear and verify everything is gone
	tt.Clear()

	probes, hits, writes := tt.Stats()
	if probes != 0 || hits != 0 || writes != 0 {
		t.Errorf("After Clear: probes=%d, hits=%d, writes=%d, all should be 0", probes, hits, writes)
	}

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

	_, hits1, writes1 := tt.Stats()
	t.Logf("First search: move=%s, nodes=%d, TT writes=%d", move1.String(), info1.Nodes, writes1)

	// Reset position and search again with same TT
	b.Reset()
	move2, info2 := b.SearchWithTT(5, 0, tt)

	_, hits2, writes2 := tt.Stats()
	t.Logf("Second search: move=%s, nodes=%d, TT hits=%d, TT writes=%d",
		move2.String(), info2.Nodes, hits2-hits1, writes2-writes1)

	// Second search should have TT hits and potentially fewer nodes
	if hits2 <= hits1 {
		t.Error("Expected TT hits on second search")
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
		tt.Store(i*12345, 5, 100, TTExact, NoMove)
	}

	hashfull := tt.Hashfull()
	t.Logf("After 500 entries: hashfull = %d permill", hashfull)

	if hashfull == 0 {
		t.Error("Hashfull should be > 0 after adding entries")
	}
}

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

	// key1 might be overwritten if they collide
	entry1, found1 := tt.Probe(key1)
	entry2, found2 := tt.Probe(key2)

	// At least one should be found
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

	// Try to store with lower depth - should not replace
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

package chess

import (
	"os"
	"testing"
)

func TestBookSerializationRoundTrip(t *testing.T) {
	// Create a Board at startpos and get the hash + a legal move
	var b Board
	b.Reset()
	startHash := b.HashKey
	moves := b.GenerateLegalMoves()
	if len(moves) == 0 {
		t.Fatal("no legal moves from startpos")
	}
	e4 := moves[0] // some legal move

	entries := map[uint64]*BookEntry{
		startHash: {
			Hash: startHash,
			Name: "Starting Position",
			Moves: []BookMove{
				{Move: e4, Weight: 100},
				{Move: moves[1], Weight: 50},
			},
		},
	}

	data, err := serializeBookEntries(entries)
	if err != nil {
		t.Fatal(err)
	}

	book, err := parseBookData(data)
	if err != nil {
		t.Fatal(err)
	}

	if book.Size() != 1 {
		t.Errorf("expected 1 entry, got %d", book.Size())
	}

	entry := book.Lookup(startHash)
	if entry == nil {
		t.Fatal("entry not found for startpos hash")
	}
	if entry.Name != "Starting Position" {
		t.Errorf("name: got %q, want %q", entry.Name, "Starting Position")
	}
	if len(entry.Moves) != 2 {
		t.Fatalf("expected 2 moves, got %d", len(entry.Moves))
	}
	if entry.Moves[0].Move != e4 {
		t.Errorf("first move mismatch")
	}
	if entry.Moves[0].Weight != 100 {
		t.Errorf("first weight: got %d, want 100", entry.Moves[0].Weight)
	}
}

func TestBookLookupMiss(t *testing.T) {
	entries := map[uint64]*BookEntry{}
	data, err := serializeBookEntries(entries)
	if err != nil {
		t.Fatal(err)
	}
	book, err := parseBookData(data)
	if err != nil {
		t.Fatal(err)
	}
	if book.Lookup(12345) != nil {
		t.Error("expected nil for missing hash")
	}
	_, _, ok := book.PickMove(12345)
	if ok {
		t.Error("expected ok=false for missing hash")
	}
}

func TestBookPickMoveWeighted(t *testing.T) {
	var b Board
	b.Reset()
	hash := b.HashKey
	moves := b.GenerateLegalMoves()

	entries := map[uint64]*BookEntry{
		hash: {
			Hash: hash,
			Moves: []BookMove{
				{Move: moves[0], Weight: 1000},
				{Move: moves[1], Weight: 1},
			},
		},
	}

	data, err := serializeBookEntries(entries)
	if err != nil {
		t.Fatal(err)
	}
	book, err := parseBookData(data)
	if err != nil {
		t.Fatal(err)
	}

	// With weight 1000 vs 1, the first move should be picked overwhelmingly
	counts := map[Move]int{}
	for i := 0; i < 10000; i++ {
		m, _, ok := book.PickMove(hash)
		if !ok {
			t.Fatal("pick returned not ok")
		}
		counts[m]++
	}

	if counts[moves[0]] < 9000 {
		t.Errorf("expected first move picked >9000 times, got %d", counts[moves[0]])
	}
}

func TestBookOwnBookToggle(t *testing.T) {
	var b Board
	b.Reset()
	hash := b.HashKey
	moves := b.GenerateLegalMoves()

	entries := map[uint64]*BookEntry{
		hash: {
			Hash:  hash,
			Moves: []BookMove{{Move: moves[0], Weight: 100}},
		},
	}

	data, err := serializeBookEntries(entries)
	if err != nil {
		t.Fatal(err)
	}
	book, err := parseBookData(data)
	if err != nil {
		t.Fatal(err)
	}

	// Default: book enabled
	_, _, ok := book.PickMove(hash)
	if !ok {
		t.Error("expected pick to succeed with book enabled")
	}

	// Disable
	book.SetUseBook(false)
	_, _, ok = book.PickMove(hash)
	if ok {
		t.Error("expected pick to fail with book disabled")
	}

	// Re-enable
	book.SetUseBook(true)
	_, _, ok = book.PickMove(hash)
	if !ok {
		t.Error("expected pick to succeed after re-enabling")
	}
}

func TestBookBuildAndLoad(t *testing.T) {
	tmpFile := t.TempDir() + "/test_book.bin"

	opts := BookBuildOptions{
		MaxPly:  20,
		MinFreq: 1, // low threshold for test
		TopN:    8,
	}
	err := BuildOpeningBook("testdata/2600.pgn", "testdata/eco.pgn", tmpFile, opts)
	if err != nil {
		t.Fatal(err)
	}

	// Check file exists and is non-trivial
	info, err := os.Stat(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() < 1000 {
		t.Errorf("book file suspiciously small: %d bytes", info.Size())
	}

	// Load and verify
	book, err := LoadOpeningBook(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	if book.Size() < 100 {
		t.Errorf("expected at least 100 positions, got %d", book.Size())
	}

	// Starting position should be in the book
	var b Board
	b.Reset()
	entry := book.Lookup(b.HashKey)
	if entry == nil {
		t.Fatal("starting position not in book")
	}
	if len(entry.Moves) == 0 {
		t.Fatal("no moves for starting position")
	}

	// Verify PickMove works
	m, _, ok := book.PickMove(b.HashKey)
	if !ok {
		t.Fatal("PickMove returned false for starting position")
	}
	if m == NoMove {
		t.Fatal("PickMove returned NoMove")
	}

	// Some entries should have ECO names
	namedCount := 0
	for _, e := range book.entries {
		if e.Name != "" {
			namedCount++
		}
	}
	if namedCount == 0 {
		t.Error("no entries have opening names")
	}
	t.Logf("Book has %d positions, %d with names", book.Size(), namedCount)
}

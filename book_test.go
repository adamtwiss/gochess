package chess

import (
	"encoding/binary"
	"os"
	"testing"
)

func TestPolyglotHash(t *testing.T) {
	// Known Polyglot hashes from the specification
	var b Board
	b.Reset()
	hash := PolyglotHash(&b)
	expected := uint64(0x463B96181691FC9C)
	if hash != expected {
		t.Errorf("startpos hash: got 0x%016X, want 0x%016X", hash, expected)
	}

	// After 1.e4
	b.Reset()
	m, err := b.ParseSAN("e4")
	if err != nil {
		t.Fatal(err)
	}
	b.MakeMove(m)
	hash = PolyglotHash(&b)
	expected = uint64(0x823C9B50FD114196)
	if hash != expected {
		t.Errorf("after 1.e4 hash: got 0x%016X, want 0x%016X", hash, expected)
	}

	// After 1.e4 e5
	m, err = b.ParseSAN("e5")
	if err != nil {
		t.Fatal(err)
	}
	b.MakeMove(m)
	hash = PolyglotHash(&b)
	expected = uint64(0x0844931A6EF4B9A0)
	if hash != expected {
		t.Errorf("after 1.e4 e5 hash: got 0x%016X, want 0x%016X", hash, expected)
	}
}

func TestBookBuildAndLoad(t *testing.T) {
	tmpFile := t.TempDir() + "/test_book.bin"

	opts := BookBuildOptions{
		MaxPly:  20,
		MinFreq: 1,
		TopN:    8,
	}
	err := BuildOpeningBook("testdata/2600.pgn", tmpFile, opts)
	if err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() < 100 {
		t.Errorf("book file suspiciously small: %d bytes", info.Size())
	}
	// Polyglot format: must be multiple of 16
	if info.Size()%16 != 0 {
		t.Errorf("book file size %d not a multiple of 16", info.Size())
	}

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
	m, ok := book.PickMove(&b)
	if !ok {
		t.Fatal("PickMove returned false for starting position")
	}
	if m == NoMove {
		t.Fatal("PickMove returned NoMove")
	}

	t.Logf("Book has %d positions", book.Size())
}

func TestBookPickMoveWeighted(t *testing.T) {
	// Build a book and verify weighted selection works
	tmpFile := t.TempDir() + "/test_book.bin"

	opts := BookBuildOptions{
		MaxPly:  10,
		MinFreq: 1,
		TopN:    8,
	}
	err := BuildOpeningBook("testdata/2600.pgn", tmpFile, opts)
	if err != nil {
		t.Fatal(err)
	}

	book, err := LoadOpeningBook(tmpFile)
	if err != nil {
		t.Fatal(err)
	}

	var b Board
	b.Reset()

	// Pick many times and verify we get valid moves
	seen := make(map[Move]int)
	for i := 0; i < 1000; i++ {
		m, ok := book.PickMove(&b)
		if !ok {
			t.Fatal("PickMove returned false for starting position")
		}
		seen[m]++
	}

	if len(seen) < 2 {
		t.Logf("Only saw %d distinct moves (book may have very skewed weights)", len(seen))
	}

	// All picked moves should be legal
	legalMoves := b.GenerateLegalMoves()
	legalSet := make(map[Move]bool)
	for _, m := range legalMoves {
		legalSet[m] = true
	}
	for m := range seen {
		if !legalSet[m] {
			t.Errorf("PickMove returned illegal move: %v", m)
		}
	}
}

func TestBookOwnBookToggle(t *testing.T) {
	tmpFile := t.TempDir() + "/test_book.bin"
	opts := BookBuildOptions{MaxPly: 10, MinFreq: 1, TopN: 8}
	err := BuildOpeningBook("testdata/2600.pgn", tmpFile, opts)
	if err != nil {
		t.Fatal(err)
	}

	book, err := LoadOpeningBook(tmpFile)
	if err != nil {
		t.Fatal(err)
	}

	var b Board
	b.Reset()

	// Default: book enabled
	_, ok := book.PickMove(&b)
	if !ok {
		t.Error("expected pick to succeed with book enabled")
	}

	// Disable
	book.SetUseBook(false)
	_, ok = book.PickMove(&b)
	if ok {
		t.Error("expected pick to fail with book disabled")
	}

	// Re-enable
	book.SetUseBook(true)
	_, ok = book.PickMove(&b)
	if !ok {
		t.Error("expected pick to succeed after re-enabling")
	}
}

func TestBookLookupMiss(t *testing.T) {
	// Create a minimal empty book
	tmpFile := t.TempDir() + "/empty_book.bin"
	err := os.WriteFile(tmpFile, []byte{}, 0644)
	if err != nil {
		t.Fatal(err)
	}

	book, err := LoadOpeningBook(tmpFile)
	if err != nil {
		t.Fatal(err)
	}

	// A random position shouldn't be in the book
	var b Board
	b.SetFEN("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
	_, ok := book.PickMove(&b)
	if ok {
		t.Error("expected ok=false for position not in book")
	}
}

func TestLoadExternalPolyglotBook(t *testing.T) {
	// Create a Polyglot book file manually (as an external tool would)
	// Startpos with e2e4 (weight 100) and d2d4 (weight 50)
	var b Board
	b.Reset()
	hash := PolyglotHash(&b)

	type rawEntry struct {
		key    uint64
		move   uint16
		weight uint16
	}

	// e2e4: from=(4,1) to=(4,3) -> Polyglot encoding
	e4Move := uint16(4) | uint16(3)<<3 | uint16(4)<<6 | uint16(1)<<9
	// d2d4: from=(3,1) to=(3,3)
	d4Move := uint16(3) | uint16(3)<<3 | uint16(3)<<6 | uint16(1)<<9

	entries := []rawEntry{
		{hash, e4Move, 100},
		{hash, d4Move, 50},
	}

	// Write Polyglot .bin file
	tmpFile := t.TempDir() + "/external.bin"
	f, err := os.Create(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	var buf [16]byte
	for _, e := range entries {
		binary.BigEndian.PutUint64(buf[0:8], e.key)
		binary.BigEndian.PutUint16(buf[8:10], e.move)
		binary.BigEndian.PutUint16(buf[10:12], e.weight)
		binary.BigEndian.PutUint32(buf[12:16], 0) // learn
		f.Write(buf[:])
	}
	f.Close()

	// Load and verify
	book, err := LoadOpeningBook(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	if book.Size() != 1 { // 1 unique position
		t.Errorf("expected 1 position, got %d", book.Size())
	}

	// Pick moves — should get valid legal moves
	seen := make(map[string]int)
	for i := 0; i < 1000; i++ {
		m, ok := book.PickMove(&b)
		if !ok {
			t.Fatal("PickMove failed")
		}
		seen[m.String()]++
	}

	// Should see both e2e4 and d2d4
	if seen["e2e4"] == 0 {
		t.Error("never picked e2e4")
	}
	if seen["d2d4"] == 0 {
		t.Error("never picked d2d4")
	}
	// e2e4 should be picked roughly 2:1 over d2d4 (100 vs 50 weight)
	ratio := float64(seen["e2e4"]) / float64(seen["d2d4"])
	if ratio < 1.5 || ratio > 2.5 {
		t.Errorf("weight ratio off: e2e4=%d d2d4=%d ratio=%.2f (expected ~2.0)", seen["e2e4"], seen["d2d4"], ratio)
	}
}

func TestMoveToPolyglotRoundTrip(t *testing.T) {
	// Verify that moveToPolyglot -> matchPolyglotMove round-trips correctly
	var b Board
	b.Reset()
	legalMoves := b.GenerateLegalMoves()

	for _, m := range legalMoves {
		pm := moveToPolyglot(m, &b)
		matched, ok := matchPolyglotMove(pm, legalMoves, &b)
		if !ok {
			t.Errorf("matchPolyglotMove failed for %v (polyglot: 0x%04x)", m, pm)
			continue
		}
		if matched != m {
			t.Errorf("round-trip mismatch: %v -> 0x%04x -> %v", m, pm, matched)
		}
	}
}

func TestPolyglotCastlingEncoding(t *testing.T) {
	// Set up a position where castling is legal
	var b Board
	b.SetFEN("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
	legalMoves := b.GenerateLegalMoves()

	// Find the kingside castle move
	var ksCastle Move
	for _, m := range legalMoves {
		if m.Flags() == FlagCastle && m.To() == Square(6) { // g1
			ksCastle = m
			break
		}
	}
	if ksCastle == NoMove {
		t.Fatal("no kingside castle found")
	}

	// Polyglot should encode as e1h1 (king to rook)
	pm := moveToPolyglot(ksCastle, &b)
	toFile := int(pm & 7)
	toRank := int((pm >> 3) & 7)
	fromFile := int((pm >> 6) & 7)
	fromRank := int((pm >> 9) & 7)

	if fromFile != 4 || fromRank != 0 || toFile != 7 || toRank != 0 {
		t.Errorf("kingside castle: expected e1h1 encoding, got from=(%d,%d) to=(%d,%d)",
			fromFile, fromRank, toFile, toRank)
	}

	// Should round-trip back
	matched, ok := matchPolyglotMove(pm, legalMoves, &b)
	if !ok || matched != ksCastle {
		t.Errorf("kingside castle round-trip failed: ok=%v, matched=%v, expected=%v", ok, matched, ksCastle)
	}
}

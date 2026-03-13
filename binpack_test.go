package chess

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestBinpackRoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		fen    string
		score  int16
		result uint8
	}{
		{"starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 15, 2},
		{"after e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", -20, 0},
		{"after d4 d5", "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2", 5, 1},
		{"no castling", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1", 0, 1},
		{"white kingside only", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w K - 0 1", 100, 2},
		{"all castling", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", -50, 0},
		{"endgame KPK", "8/8/8/8/8/5K2/5P2/5k2 w - - 0 1", 500, 2},
		{"black to move", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", -15, 1},
		{"max score", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 32000, 2},
		{"min score", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -32000, 0},
		{"halfmove clock 50", "8/8/8/8/8/5K2/5P2/5k2 w - - 50 1", 0, 1},
		{"en passant file a", "rnbqkbnr/1ppppppp/8/pP6/8/8/P1PPPPPP/RNBQKBNR w KQkq a6 0 3", 10, 2},
		{"en passant file h", "rnbqkbnr/ppppppp1/8/7p/6P1/8/PPPPPP1P/RNBQKBNR w KQkq h6 0 2", 10, 2},
		{"en passant black", "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2", -5, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var b Board
			if err := b.SetFEN(tt.fen); err != nil {
				t.Fatalf("SetFEN(%q): %v", tt.fen, err)
			}

			rec := PackPosition(&b, tt.score, tt.result)
			b2, score2, result2, err := UnpackPosition(rec)
			if err != nil {
				t.Fatalf("UnpackPosition: %v", err)
			}

			// Check score and result
			if score2 != tt.score {
				t.Errorf("score: got %d, want %d", score2, tt.score)
			}
			if result2 != tt.result {
				t.Errorf("result: got %d, want %d", result2, tt.result)
			}

			// Check board state
			if b2.SideToMove != b.SideToMove {
				t.Errorf("SideToMove: got %d, want %d", b2.SideToMove, b.SideToMove)
			}
			if b2.Castling != b.Castling {
				t.Errorf("Castling: got %d, want %d", b2.Castling, b.Castling)
			}
			if b2.EnPassant != b.EnPassant {
				t.Errorf("EnPassant: got %d, want %d", b2.EnPassant, b.EnPassant)
			}
			if b2.HalfmoveClock != b.HalfmoveClock {
				t.Errorf("HalfmoveClock: got %d, want %d", b2.HalfmoveClock, b.HalfmoveClock)
			}

			// Check all squares
			for sq := Square(0); sq < 64; sq++ {
				if b2.Squares[sq] != b.Squares[sq] {
					t.Errorf("Square %s: got %d, want %d", sq.String(), b2.Squares[sq], b.Squares[sq])
				}
			}

			// Check bitboards
			if b2.AllPieces != b.AllPieces {
				t.Errorf("AllPieces: got %016x, want %016x", b2.AllPieces, b.AllPieces)
			}
			for p := WhitePawn; p <= BlackKing; p++ {
				if b2.Pieces[p] != b.Pieces[p] {
					t.Errorf("Pieces[%d]: got %016x, want %016x", p, b2.Pieces[p], b.Pieces[p])
				}
			}
		})
	}
}

func TestBinpackScoreClamping(t *testing.T) {
	var b Board
	b.Reset()

	// Score exceeding max should be clamped
	rec := PackPosition(&b, 32767, 1)
	_, score, _, _ := UnpackPosition(rec)
	if score != 32000 {
		t.Errorf("expected clamped score 32000, got %d", score)
	}

	rec = PackPosition(&b, -32767, 1)
	_, score, _, _ = UnpackPosition(rec)
	if score != -32000 {
		t.Errorf("expected clamped score -32000, got %d", score)
	}
}

func TestBinpackResultConversion(t *testing.T) {
	tests := []struct {
		floatResult float64
		uintResult  uint8
	}{
		{0.0, 0},
		{0.5, 1},
		{1.0, 2},
	}
	for _, tt := range tests {
		got := ResultToUint8(tt.floatResult)
		if got != tt.uintResult {
			t.Errorf("ResultToUint8(%v) = %d, want %d", tt.floatResult, got, tt.uintResult)
		}
		gotFloat := ResultToFloat(tt.uintResult)
		if gotFloat != float32(tt.floatResult) {
			t.Errorf("ResultToFloat(%d) = %v, want %v", tt.uintResult, gotFloat, tt.floatResult)
		}
	}
}

func TestBinpackFeatureExtraction(t *testing.T) {
	fen := "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
	var b Board
	if err := b.SetFEN(fen); err != nil {
		t.Fatal(err)
	}

	// Get features via the existing ParseNNUETrainData path
	expected, err := ParseNNUETrainData(fen + ";15;0.5")
	if err != nil {
		t.Fatal(err)
	}

	// Get features via binpack
	rec := PackPosition(&b, 15, 1)
	got := ExtractFeaturesFromBinpack(rec)

	// Compare feature sets (order may differ, so sort both)
	if len(got.WhiteFeatures) != len(expected.WhiteFeatures) {
		t.Fatalf("WhiteFeatures count: got %d, want %d", len(got.WhiteFeatures), len(expected.WhiteFeatures))
	}
	if len(got.BlackFeatures) != len(expected.BlackFeatures) {
		t.Fatalf("BlackFeatures count: got %d, want %d", len(got.BlackFeatures), len(expected.BlackFeatures))
	}

	// Build sets for comparison
	wSet := make(map[int]bool)
	for _, f := range expected.WhiteFeatures {
		wSet[f] = true
	}
	for _, f := range got.WhiteFeatures {
		if !wSet[f] {
			t.Errorf("unexpected WhiteFeature %d", f)
		}
	}

	bSet := make(map[int]bool)
	for _, f := range expected.BlackFeatures {
		bSet[f] = true
	}
	for _, f := range got.BlackFeatures {
		if !bSet[f] {
			t.Errorf("unexpected BlackFeature %d", f)
		}
	}
}

func TestBinpackFileIO(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bin")

	// Write some records
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}

	fens := []string{
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
		"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
		"8/8/8/8/8/5K2/5P2/5k2 w - - 0 1",
	}

	for i, fen := range fens {
		var b Board
		if err := b.SetFEN(fen); err != nil {
			t.Fatal(err)
		}
		rec := PackPosition(&b, int16(i*10), uint8(i))
		if _, err := f.Write(rec[:]); err != nil {
			t.Fatal(err)
		}
	}
	f.Close()

	// Read them back
	bf, err := OpenBinpackFiles(path)
	if err != nil {
		t.Fatal(err)
	}
	defer bf.Close()

	if bf.NumRecords() != 3 {
		t.Fatalf("NumRecords: got %d, want 3", bf.NumRecords())
	}

	for i := 0; i < 3; i++ {
		rec, err := bf.ReadRecord(i)
		if err != nil {
			t.Fatalf("ReadRecord(%d): %v", i, err)
		}
		_, score, result, err := UnpackPosition(rec)
		if err != nil {
			t.Fatalf("UnpackPosition(%d): %v", i, err)
		}
		if score != int16(i*10) {
			t.Errorf("record %d: score got %d, want %d", i, score, i*10)
		}
		if result != uint8(i) {
			t.Errorf("record %d: result got %d, want %d", i, result, i)
		}
	}
}

func TestBinpackMultiFile(t *testing.T) {
	dir := t.TempDir()

	// Create two files
	for fileIdx, fens := range [][]string{
		{"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
		{
			"8/8/8/8/8/5K2/5P2/5k2 w - - 0 1",
			"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
		},
	} {
		path := filepath.Join(dir, fmt.Sprintf("data%d.bin", fileIdx))
		f, _ := os.Create(path)
		for i, fen := range fens {
			var b Board
			b.SetFEN(fen)
			rec := PackPosition(&b, int16(fileIdx*100+i), 1)
			f.Write(rec[:])
		}
		f.Close()
	}

	bf, err := OpenBinpackFiles(
		filepath.Join(dir, "data0.bin"),
		filepath.Join(dir, "data1.bin"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer bf.Close()

	if bf.NumRecords() != 3 {
		t.Fatalf("NumRecords: got %d, want 3", bf.NumRecords())
	}

	// Read all records
	scores := make([]int16, 3)
	for i := 0; i < 3; i++ {
		rec, err := bf.ReadRecord(i)
		if err != nil {
			t.Fatalf("ReadRecord(%d): %v", i, err)
		}
		_, scores[i], _, _ = UnpackPosition(rec)
	}

	// File 0 has 1 record (score 0), File 1 has 2 records (scores 100, 101)
	if scores[0] != 0 {
		t.Errorf("record 0: got score %d, want 0", scores[0])
	}
	if scores[1] != 100 {
		t.Errorf("record 1: got score %d, want 100", scores[1])
	}
	if scores[2] != 101 {
		t.Errorf("record 2: got score %d, want 101", scores[2])
	}
}

func TestBinpackEpochReader(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bin")

	// Write 100 records
	f, _ := os.Create(path)
	var b Board
	b.Reset()
	for i := 0; i < 100; i++ {
		rec := PackPosition(&b, int16(i), 1)
		f.Write(rec[:])
	}
	f.Close()

	bf, err := OpenBinpackFiles(path)
	if err != nil {
		t.Fatal(err)
	}
	defer bf.Close()

	// Read all records via epoch reader (90% train)
	rng := rand.New(rand.NewSource(42))
	reader := bf.NewEpochReader(rng, 0.9)

	expectedTrain := reader.NumTrainRecords()
	if expectedTrain != 90 {
		t.Fatalf("NumTrainRecords: got %d, want 90", expectedTrain)
	}

	seen := make(map[float32]bool)
	total := 0
	sampleBuf := make([]*NNUETrainSample, 0, 32)
	for {
		batch, err := reader.NextBatch(32, sampleBuf)
		if err != nil {
			t.Fatal(err)
		}
		if batch == nil {
			break
		}
		for _, s := range batch {
			seen[s.Score] = true
			total++
		}
	}

	if total != 90 {
		t.Errorf("total samples: got %d, want 90", total)
	}
	// All 90 training scores should be unique (scores 0-89, shuffled)
	if len(seen) != 90 {
		t.Errorf("unique scores: got %d, want 90", len(seen))
	}
}

func TestBinpackCatConcatenation(t *testing.T) {
	dir := t.TempDir()

	// Create two small binpack files
	var b Board
	b.Reset()

	path1 := filepath.Join(dir, "a.bin")
	path2 := filepath.Join(dir, "b.bin")
	pathCombined := filepath.Join(dir, "combined.bin")

	f1, _ := os.Create(path1)
	rec1 := PackPosition(&b, 100, 2)
	f1.Write(rec1[:])
	f1.Close()

	f2, _ := os.Create(path2)
	rec2 := PackPosition(&b, 200, 0)
	f2.Write(rec2[:])
	f2.Close()

	// Simulate "cat a.bin b.bin > combined.bin"
	combined, _ := os.Create(pathCombined)
	d1, _ := os.ReadFile(path1)
	d2, _ := os.ReadFile(path2)
	combined.Write(d1)
	combined.Write(d2)
	combined.Close()

	// Read concatenated file
	bf, err := OpenBinpackFiles(pathCombined)
	if err != nil {
		t.Fatal(err)
	}
	defer bf.Close()

	if bf.NumRecords() != 2 {
		t.Fatalf("NumRecords: got %d, want 2", bf.NumRecords())
	}

	r1, _ := bf.ReadRecord(0)
	_, s1, _, _ := UnpackPosition(r1)
	r2, _ := bf.ReadRecord(1)
	_, s2, _, _ := UnpackPosition(r2)

	if s1 != 100 {
		t.Errorf("record 0 score: got %d, want 100", s1)
	}
	if s2 != 200 {
		t.Errorf("record 1 score: got %d, want 200", s2)
	}
}

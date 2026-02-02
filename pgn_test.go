package chess

import (
	"strings"
	"testing"
)

func TestParseTag(t *testing.T) {
	input := `[ECO "A00"]
[Opening "Polish (Sokolsky) opening"]

1. b4 *
`
	games, err := ParsePGN(strings.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	if len(games) != 1 {
		t.Fatalf("expected 1 game, got %d", len(games))
	}
	g := games[0]
	if g.Tags["ECO"] != "A00" {
		t.Errorf("ECO tag: got %q, want %q", g.Tags["ECO"], "A00")
	}
	if g.Tags["Opening"] != "Polish (Sokolsky) opening" {
		t.Errorf("Opening tag: got %q, want %q", g.Tags["Opening"], "Polish (Sokolsky) opening")
	}
	if len(g.Moves) != 1 || g.Moves[0] != "b4" {
		t.Errorf("Moves: got %v, want [b4]", g.Moves)
	}
}

func TestParsePGNMultipleGames(t *testing.T) {
	input := `[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Nf3 1-0

[Event "Test2"]
[Result "0-1"]

1. d4 d5 0-1
`
	games, err := ParsePGN(strings.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	if len(games) != 2 {
		t.Fatalf("expected 2 games, got %d", len(games))
	}
	if len(games[0].Moves) != 3 {
		t.Errorf("game 0: expected 3 moves, got %d: %v", len(games[0].Moves), games[0].Moves)
	}
	if len(games[1].Moves) != 2 {
		t.Errorf("game 1: expected 2 moves, got %d: %v", len(games[1].Moves), games[1].Moves)
	}
}

func TestParsePGNBraceComments(t *testing.T) {
	input := `[Event "Test"]

1. e4 {great move} e5 2. Nf3 Nc6 *
`
	games, err := ParsePGN(strings.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	if len(games) != 1 {
		t.Fatalf("expected 1 game, got %d", len(games))
	}
	expected := []string{"e4", "e5", "Nf3", "Nc6"}
	if len(games[0].Moves) != len(expected) {
		t.Fatalf("expected %d moves, got %d: %v", len(expected), len(games[0].Moves), games[0].Moves)
	}
	for i, m := range expected {
		if games[0].Moves[i] != m {
			t.Errorf("move %d: got %q, want %q", i, games[0].Moves[i], m)
		}
	}
}

func TestParsePGNMultiLineBraceComment(t *testing.T) {
	input := `{
This is a multi-line comment
at the top of the file.
}

[ECO "A00"]
[Opening "Test"]

1. b4 *
`
	games, err := ParsePGN(strings.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	if len(games) != 1 {
		t.Fatalf("expected 1 game, got %d", len(games))
	}
	if games[0].Tags["ECO"] != "A00" {
		t.Errorf("ECO: got %q, want %q", games[0].Tags["ECO"], "A00")
	}
}

func TestParsePGNNAGs(t *testing.T) {
	input := `[Event "Test"]

1. e4 $1 e5 $2 2. Nf3 *
`
	games, err := ParsePGN(strings.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	if len(games[0].Moves) != 3 {
		t.Errorf("expected 3 moves, got %d: %v", len(games[0].Moves), games[0].Moves)
	}
}

func TestParsePGNAnnotationGlyphs(t *testing.T) {
	input := `[Event "Test"]

1. e4! e5? 2. Nf3!! Nc6?! *
`
	games, err := ParsePGN(strings.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	expected := []string{"e4", "e5", "Nf3", "Nc6"}
	if len(games[0].Moves) != len(expected) {
		t.Fatalf("expected %d moves, got %d: %v", len(expected), len(games[0].Moves), games[0].Moves)
	}
	for i, m := range expected {
		if games[0].Moves[i] != m {
			t.Errorf("move %d: got %q, want %q", i, games[0].Moves[i], m)
		}
	}
}

func TestTokenizeMoveText(t *testing.T) {
	tests := []struct {
		line     string
		expected []string
	}{
		{"1. e4 e5 2. Nf3 Nc6", []string{"e4", "e5", "Nf3", "Nc6"}},
		{"1. e4 {comment} e5", []string{"e4", "e5"}},
		{"1... Nf6", []string{"Nf6"}},
		{"12. Qxf7+ 1-0", []string{"Qxf7+", "1-0"}},
		{"1. e4 $1 e5 $23", []string{"e4", "e5"}},
		{"1. e4! e5? Nf3!!", []string{"e4", "e5", "Nf3"}},
	}
	for _, tt := range tests {
		got := tokenizeMoveText(tt.line)
		if len(got) != len(tt.expected) {
			t.Errorf("tokenize(%q): got %v, want %v", tt.line, got, tt.expected)
			continue
		}
		for i := range got {
			if got[i] != tt.expected[i] {
				t.Errorf("tokenize(%q)[%d]: got %q, want %q", tt.line, i, got[i], tt.expected[i])
			}
		}
	}
}

func TestParsePGNFileECO(t *testing.T) {
	games, err := ParsePGNFile("testdata/eco.pgn")
	if err != nil {
		t.Fatal(err)
	}
	if len(games) < 100 {
		t.Errorf("expected at least 100 ECO games, got %d", len(games))
	}
	// Check first game
	g := games[0]
	if g.Tags["ECO"] != "A00" {
		t.Errorf("first game ECO: got %q, want %q", g.Tags["ECO"], "A00")
	}
	if len(g.Moves) < 1 {
		t.Error("first game has no moves")
	}
	// All games should have ECO tag
	for i, g := range games {
		if g.Tags["ECO"] == "" {
			t.Errorf("game %d missing ECO tag", i)
			break
		}
	}
}

func TestParsePGNReplayAndHash(t *testing.T) {
	// Parse a short PGN game, replay all moves on a Board, verify zobrist hash
	// at the end matches a full recompute. This catches regressions in:
	// - PGN parsing (tokenizing, comment stripping)
	// - SAN parsing (disambiguation, captures, promotions, castling)
	// - Incremental zobrist hashing (MakeMove correctness)
	input := `[Event "Immortal Game"]
[White "Anderssen"]
[Black "Kieseritzky"]
[Result "1-0"]

1. e4 e5 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6
6. Nf3 Qh6 7. d3 Nh5 8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6
11. Rg1 cxb5 12. h4 Qg6 13. h5 Qg5 14. Qf3 Ng8 15. Bxf4 Qf6
16. Nc3 Bc5 17. Nd5 Qxb2 18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2 Na6
21. Nxg7+ Kd8 22. Qf6+ Nxf6 23. Be7# 1-0
`
	games, err := ParsePGN(strings.NewReader(input))
	if err != nil {
		t.Fatalf("ParsePGN error: %v", err)
	}
	if len(games) != 1 {
		t.Fatalf("expected 1 game, got %d", len(games))
	}

	g := games[0]
	if len(g.Moves) == 0 {
		t.Fatal("game has no moves")
	}

	var b Board
	b.Reset()

	for i, san := range g.Moves {
		m, err := b.ParseSAN(san)
		if err != nil {
			t.Fatalf("move %d (%q): ParseSAN error: %v", i+1, san, err)
		}
		b.MakeMove(m)

		// Verify hash stays in sync after every move
		computed := b.Hash()
		if b.HashKey != computed {
			t.Fatalf("move %d (%q): hash mismatch: stored=%x computed=%x", i+1, san, b.HashKey, computed)
		}
	}

	// Record the final hash for regression detection
	finalHash := b.HashKey
	t.Logf("Immortal Game: %d moves replayed, final hash=%x", len(g.Moves), finalHash)

	// Verify the final position FEN is correct
	// After 23. Be7# the position should have specific characteristics
	if b.SideToMove != Black {
		t.Errorf("Expected Black to move after White's last move, got White")
	}
	// Black king should be on d8
	bk := b.Pieces[BlackKing].LSB()
	if bk != NewSquare(3, 7) {
		t.Errorf("Black king on %s, expected d8", bk)
	}
}

func TestParsePGNReplayECOFile(t *testing.T) {
	// Replay first 10 games from eco.pgn and verify hashes after each game
	games, err := ParsePGNFile("testdata/eco.pgn")
	if err != nil {
		t.Fatal(err)
	}

	limit := 10
	if len(games) < limit {
		limit = len(games)
	}

	for gi := 0; gi < limit; gi++ {
		g := games[gi]
		var b Board
		b.Reset()

		for i, san := range g.Moves {
			m, err := b.ParseSAN(san)
			if err != nil {
				t.Errorf("game %d (ECO=%s) move %d (%q): ParseSAN error: %v",
					gi, g.Tags["ECO"], i+1, san, err)
				break
			}
			b.MakeMove(m)
		}

		// Verify final hash matches full recompute
		computed := b.Hash()
		if b.HashKey != computed {
			t.Errorf("game %d (ECO=%s): final hash mismatch: stored=%x computed=%x",
				gi, g.Tags["ECO"], b.HashKey, computed)
		}

		// Verify pawn hash too
		pawnComputed := b.PawnHash()
		if b.PawnHashKey != pawnComputed {
			t.Errorf("game %d (ECO=%s): final pawn hash mismatch: stored=%x computed=%x",
				gi, g.Tags["ECO"], b.PawnHashKey, pawnComputed)
		}
	}
}

func TestParsePGNFile2600(t *testing.T) {
	games, err := ParsePGNFile("testdata/2600.pgn")
	if err != nil {
		t.Fatal(err)
	}
	if len(games) < 1000 {
		t.Errorf("expected at least 1000 games, got %d", len(games))
	}
	// Check first game
	g := games[0]
	if g.Tags["White"] != "Kasparov, Gary" {
		t.Errorf("first game White: got %q", g.Tags["White"])
	}
	if len(g.Moves) < 10 {
		t.Errorf("first game has too few moves: %d", len(g.Moves))
	}
}

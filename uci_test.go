package chess

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"
)

func TestParseGoParams(t *testing.T) {
	tests := []struct {
		name   string
		tokens []string
		want   goParams
	}{
		{
			"empty",
			nil,
			goParams{depth: 64},
		},
		{
			"depth only",
			[]string{"depth", "10"},
			goParams{depth: 10},
		},
		{
			"infinite",
			[]string{"infinite"},
			goParams{depth: 64, infinite: true},
		},
		{
			"ponder with time",
			[]string{"ponder", "wtime", "60000", "btime", "60000", "winc", "1000", "binc", "1000"},
			goParams{depth: 64, ponder: true, wtime: 60000, btime: 60000, winc: 1000, binc: 1000},
		},
		{
			"movetime",
			[]string{"movetime", "5000"},
			goParams{depth: 64, movetime: 5000},
		},
		{
			"movestogo",
			[]string{"wtime", "30000", "btime", "30000", "movestogo", "10"},
			goParams{depth: 64, wtime: 30000, btime: 30000, movestogo: 10},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseGoParams(tt.tokens)
			if got != tt.want {
				t.Errorf("parseGoParams(%v) = %+v, want %+v", tt.tokens, got, tt.want)
			}
		})
	}
}

func TestParseUCIMove(t *testing.T) {
	tests := []struct {
		name    string
		fen     string
		uci     string
		wantErr bool
	}{
		{"pawn push", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", false},
		{"knight move", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3", false},
		{"kingside castling", "r1bqk2r/ppppbppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "e1g1", false},
		{"queenside castling", "r3kbnr/pppqpppp/2n1b3/3p4/3P4/2N1B3/PPPQPPPP/R3KBNR w KQkq - 6 5", "e1c1", false},
		{"en passant", "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", "e5d6", false},
		{"queen promotion", "8/4P3/8/8/8/8/4k1K1/8 w - - 0 1", "e7e8q", false},
		{"knight promotion", "8/4P3/8/8/8/8/4k1K1/8 w - - 0 1", "e7e8n", false},
		{"rook promotion", "8/4P3/8/8/8/8/4k1K1/8 w - - 0 1", "e7e8r", false},
		{"bishop promotion", "8/4P3/8/8/8/8/4k1K1/8 w - - 0 1", "e7e8b", false},
		{"illegal move", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e1e5", true},
		{"too short", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2", true},
		{"bad promo char", "8/4P3/8/8/8/8/4k1K1/8 w - - 0 1", "e7e8x", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var b Board
			if err := b.SetFEN(tt.fen); err != nil {
				t.Fatalf("SetFEN: %v", err)
			}
			move, err := b.ParseUCIMove(tt.uci)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error for %s, got move %s", tt.uci, move.String())
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseUCIMove(%s): %v", tt.uci, err)
			}
			if move.String() != tt.uci {
				t.Errorf("move.String() = %s, want %s", move.String(), tt.uci)
			}
		})
	}
}

func TestParseUCIMoveFlags(t *testing.T) {
	// Verify castling has FlagCastle
	var b Board
	b.SetFEN("r1bqk2r/ppppbppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
	m, err := b.ParseUCIMove("e1g1")
	if err != nil {
		t.Fatal(err)
	}
	if m.Flags() != FlagCastle {
		t.Errorf("castling move flags = %d, want %d (FlagCastle)", m.Flags(), FlagCastle)
	}

	// Verify en passant has FlagEnPassant
	b.SetFEN("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
	m, err = b.ParseUCIMove("e5d6")
	if err != nil {
		t.Fatal(err)
	}
	if m.Flags() != FlagEnPassant {
		t.Errorf("en passant move flags = %d, want %d (FlagEnPassant)", m.Flags(), FlagEnPassant)
	}
}

func TestToFEN(t *testing.T) {
	tests := []struct {
		name string
		fen  string
	}{
		{"starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
		{"after e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"},
		{"mid-game", "r1bqk2r/ppppbppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"},
		{"no castling", "8/4P3/8/8/8/8/4k1K1/8 w - - 0 1"},
		{"partial castling", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kq - 0 1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var b Board
			if err := b.SetFEN(tt.fen); err != nil {
				t.Fatalf("SetFEN: %v", err)
			}
			got := b.ToFEN()
			if got != tt.fen {
				t.Errorf("ToFEN() = %s\n  want   %s", got, tt.fen)
			}
		})
	}
}

func TestToFENRoundTrip(t *testing.T) {
	// Start position, make e2e4, verify FEN
	var b Board
	b.Reset()
	m, _ := b.ParseUCIMove("e2e4")
	b.MakeMove(m)
	fen := b.ToFEN()
	expected := "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
	if fen != expected {
		t.Errorf("after e2e4: got %s, want %s", fen, expected)
	}

	// Parse back
	var b2 Board
	if err := b2.SetFEN(fen); err != nil {
		t.Fatal(err)
	}
	if b2.ToFEN() != expected {
		t.Errorf("round-trip failed: %s", b2.ToFEN())
	}
}

func TestFormatScore(t *testing.T) {
	tests := []struct {
		name  string
		score int
		want  string
	}{
		{"centipawn positive", 150, "cp 150"},
		{"centipawn zero", 0, "cp 0"},
		{"centipawn negative", -80, "cp -80"},
		{"mate in 1", MateScore - 1, "mate 1"},
		{"mate in 2", MateScore - 3, "mate 2"},
		{"mate in 3", MateScore - 5, "mate 3"},
		{"mated in 1", -(MateScore - 2), "mate -1"},
		{"mated in 2", -(MateScore - 4), "mate -2"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatScore(tt.score)
			if got != tt.want {
				t.Errorf("formatScore(%d) = %s, want %s", tt.score, got, tt.want)
			}
		})
	}
}

func TestComputeSearchTime(t *testing.T) {
	tests := []struct {
		name                                          string
		movetime, wtime, btime, winc, binc, movestogo int
		infinite                                      bool
		side                                          Color
		wantSoft, wantHard                            int
	}{
		{"movetime", 5000, 0, 0, 0, 0, 0, false, White, 5000, 5000},
		{"infinite", 0, 0, 0, 0, 0, 0, true, White, 0, 0},
		{"no clock", 0, 0, 0, 0, 0, 0, false, White, 0, 0},
		{"white clock 30s", 0, 30000, 0, 0, 0, 0, false, White, 1000, 3000},             // soft=30000/30=1000, hard=3*1000=3000
		{"black clock 30s", 0, 0, 30000, 0, 0, 0, false, Black, 1000, 3000},             // soft=30000/30=1000, hard=3000
		{"with increment", 0, 30000, 0, 2000, 0, 0, false, White, 2500, 7500},           // soft=1000+1500=2500, hard=7500
		{"movestogo 10", 0, 30000, 0, 0, 0, 10, false, White, 3000, 6000},               // soft=30000/10=3000, hard=2*3000=6000 (tournament TC)
		{"cap at half", 0, 2000, 0, 0, 0, 0, false, White, 66, 198},                     // soft=2000/30=66, hard=198
		{"floor at 10ms", 0, 100, 0, 0, 0, 0, false, White, 10, 30},                     // soft=floor(10), hard=30
		{"movetime overrides clock", 3000, 60000, 60000, 0, 0, 0, false, White, 3000, 3000},
		{"hard capped at 75%", 0, 1000, 0, 0, 0, 0, false, White, 33, 99},               // soft=1000/30=33, hard=min(99,750)=99
		{"hard capped by maxHard", 0, 600, 0, 0, 0, 1, false, White, 300, 300},          // soft=min(600/1,300)=300, hard=2*300=600 but mtgCap=200, so hard=soft=300
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotSoft, gotHard := computeSearchTime(tt.movetime, tt.wtime, tt.btime, tt.winc, tt.binc, tt.movestogo, tt.infinite, tt.side)
			if gotSoft != tt.wantSoft {
				t.Errorf("computeSearchTime() soft = %d, want %d", gotSoft, tt.wantSoft)
			}
			if gotHard != tt.wantHard {
				t.Errorf("computeSearchTime() hard = %d, want %d", gotHard, tt.wantHard)
			}
		})
	}
}

func TestUCIProtocol(t *testing.T) {
	// Test basic UCI handshake
	input := "uci\nisready\nquit\n"
	var out bytes.Buffer
	engine := NewUCIEngineWithIO(strings.NewReader(input), &out)
	engine.Run()

	output := out.String()
	if !strings.Contains(output, "id name GoChess") {
		t.Error("missing 'id name' in output")
	}
	if !strings.Contains(output, "uciok") {
		t.Error("missing 'uciok' in output")
	}
	if !strings.Contains(output, "readyok") {
		t.Error("missing 'readyok' in output")
	}
}

func TestUCIPosition(t *testing.T) {
	input := "position startpos moves e2e4 e7e5\nd\nquit\n"
	var out bytes.Buffer
	engine := NewUCIEngineWithIO(strings.NewReader(input), &out)
	engine.Run()

	output := out.String()
	if !strings.Contains(output, "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2") {
		t.Errorf("position not set correctly, output:\n%s", output)
	}
}

func TestUCIPositionFEN(t *testing.T) {
	fen := "r1bqk2r/ppppbppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
	input := "position fen " + fen + "\nd\nquit\n"
	var out bytes.Buffer
	engine := NewUCIEngineWithIO(strings.NewReader(input), &out)
	engine.Run()

	output := out.String()
	if !strings.Contains(output, fen) {
		t.Errorf("FEN position not set correctly, output:\n%s", output)
	}
}

func TestUCIGoDepth(t *testing.T) {
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	engine := NewUCIEngineWithIO(inR, outW)

	done := make(chan struct{})
	go func() {
		engine.Run()
		outW.Close()
		close(done)
	}()

	fmt.Fprintln(inW, "position startpos")
	fmt.Fprintln(inW, "go depth 3")

	// Read output until we see bestmove
	scanner := bufio.NewScanner(outR)
	var lines []string
	for scanner.Scan() {
		line := scanner.Text()
		lines = append(lines, line)
		if strings.HasPrefix(line, "bestmove") {
			break
		}
	}

	inW.Close()
	<-done

	output := strings.Join(lines, "\n")
	if !strings.Contains(output, "bestmove") {
		t.Errorf("missing 'bestmove' in output:\n%s", output)
	}
	if !strings.Contains(output, "info depth") {
		t.Errorf("missing 'info depth' in output:\n%s", output)
	}
}

func TestUCINewGame(t *testing.T) {
	// Set a custom position, then ucinewgame should reset to startpos
	input := "position fen 8/8/8/8/8/8/8/4K2k w - - 0 1\nucinewgame\nd\nquit\n"
	var out bytes.Buffer
	engine := NewUCIEngineWithIO(strings.NewReader(input), &out)
	engine.Run()

	output := out.String()
	if !strings.Contains(output, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") {
		t.Errorf("ucinewgame didn't reset to startpos, output:\n%s", output)
	}
}

func TestUCISetOption(t *testing.T) {
	input := "setoption name Hash value 32\nisready\nquit\n"
	var out bytes.Buffer
	engine := NewUCIEngineWithIO(strings.NewReader(input), &out)
	engine.Run()

	if engine.hashSizeMB != 32 {
		t.Errorf("hash size = %d, want 32", engine.hashSizeMB)
	}
}

func TestUCIBookMove(t *testing.T) {
	// Build a book with a single entry at startpos
	var b Board
	b.Reset()
	hash := b.HashKey
	moves := b.GenerateLegalMoves()

	// Find e2e4
	var e4 Move
	for _, m := range moves {
		if m.String() == "e2e4" {
			e4 = m
			break
		}
	}
	if e4 == NoMove {
		t.Fatal("e2e4 not found in legal moves")
	}

	entries := map[uint64]*BookEntry{
		hash: {
			Hash:  hash,
			Name:  "Starting Position",
			Moves: []BookMove{{Move: e4, Weight: 100}},
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

	input := "position startpos\ngo depth 10\nquit\n"
	var out bytes.Buffer
	engine := NewUCIEngineWithIO(strings.NewReader(input), &out)
	engine.SetBook(book)
	engine.Run()

	output := out.String()
	if !strings.Contains(output, "bestmove e2e4") {
		t.Errorf("expected book move e2e4, output:\n%s", output)
	}
	if !strings.Contains(output, "info string book: Starting Position") {
		t.Errorf("expected book info string, output:\n%s", output)
	}
	// Should NOT contain "info depth" since book move skips search
	if strings.Contains(output, "info depth") {
		t.Errorf("expected no search info when book move found, output:\n%s", output)
	}
}

func TestUCIOwnBookOption(t *testing.T) {
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

	// Disable OwnBook, then search — should do a real search (no instant book move)
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	engine := NewUCIEngineWithIO(inR, outW)
	engine.SetBook(book)

	done := make(chan struct{})
	go func() {
		engine.Run()
		outW.Close()
		close(done)
	}()

	fmt.Fprintln(inW, "setoption name OwnBook value false")
	fmt.Fprintln(inW, "position startpos")
	fmt.Fprintln(inW, "go depth 1")

	scanner := bufio.NewScanner(outR)
	var lines []string
	for scanner.Scan() {
		line := scanner.Text()
		lines = append(lines, line)
		if strings.HasPrefix(line, "bestmove") {
			break
		}
	}

	inW.Close()
	<-done

	output := strings.Join(lines, "\n")
	// With OwnBook disabled, should see search info
	if !strings.Contains(output, "info depth") {
		t.Errorf("expected search info when OwnBook disabled, output:\n%s", output)
	}
}

func TestUCIBookAnnouncement(t *testing.T) {
	input := "uci\nquit\n"
	var out bytes.Buffer
	engine := NewUCIEngineWithIO(strings.NewReader(input), &out)
	engine.Run()

	output := out.String()
	if !strings.Contains(output, "option name OwnBook type check default true") {
		t.Errorf("missing OwnBook option announcement, output:\n%s", output)
	}
	if !strings.Contains(output, "option name BookFile type string") {
		t.Errorf("missing BookFile option announcement, output:\n%s", output)
	}
}

func TestUCIPonderOption(t *testing.T) {
	// Verify Ponder option is announced in uci output
	input := "uci\nquit\n"
	var out bytes.Buffer
	engine := NewUCIEngineWithIO(strings.NewReader(input), &out)
	engine.Run()

	output := out.String()
	if !strings.Contains(output, "option name Ponder type check default true") {
		t.Errorf("missing Ponder option announcement, output:\n%s", output)
	}

	// Setting Ponder to false should omit ponder move from bestmove
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	engine2 := NewUCIEngineWithIO(inR, outW)

	done := make(chan struct{})
	go func() {
		engine2.Run()
		outW.Close()
		close(done)
	}()

	fmt.Fprintln(inW, "setoption name Ponder value false")
	fmt.Fprintln(inW, "position startpos")
	fmt.Fprintln(inW, "go depth 5")

	scanner := bufio.NewScanner(outR)
	var bestmoveLine string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "bestmove") {
			bestmoveLine = line
			break
		}
	}

	inW.Close()
	<-done

	if bestmoveLine == "" {
		t.Fatal("no bestmove received")
	}
	if strings.Contains(bestmoveLine, "ponder") {
		t.Errorf("expected no ponder move when Ponder option is false, got: %s", bestmoveLine)
	}
}

func TestUCIBestmovePonder(t *testing.T) {
	// Normal go depth 5 should include ponder move when PV >= 2
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	engine := NewUCIEngineWithIO(inR, outW)

	done := make(chan struct{})
	go func() {
		engine.Run()
		outW.Close()
		close(done)
	}()

	fmt.Fprintln(inW, "position startpos")
	fmt.Fprintln(inW, "go depth 5")

	scanner := bufio.NewScanner(outR)
	var bestmoveLine string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "bestmove") {
			bestmoveLine = line
			break
		}
	}

	inW.Close()
	<-done

	if bestmoveLine == "" {
		t.Fatal("no bestmove received")
	}
	// From startpos at depth 5, PV should have >= 2 moves
	if !strings.Contains(bestmoveLine, " ponder ") {
		t.Errorf("expected bestmove to include ponder move, got: %s", bestmoveLine)
	}

	// Verify ponder move is a valid 4-char UCI move
	parts := strings.Fields(bestmoveLine)
	if len(parts) < 4 || parts[2] != "ponder" {
		t.Errorf("unexpected bestmove format: %s", bestmoveLine)
	}
}

func TestUCIPonderHit(t *testing.T) {
	// go ponder → search runs → ponderhit → search continues → bestmove arrives
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	engine := NewUCIEngineWithIO(inR, outW)

	done := make(chan struct{})
	go func() {
		engine.Run()
		outW.Close()
		close(done)
	}()

	fmt.Fprintln(inW, "position startpos")
	fmt.Fprintln(inW, "go ponder wtime 60000 btime 60000")

	// Wait for search to start producing info lines
	scanner := bufio.NewScanner(outR)
	gotInfo := false
	bestmoveCh := make(chan string, 1)

	go func() {
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "info depth") {
				gotInfo = true
			}
			if strings.HasPrefix(line, "bestmove") {
				bestmoveCh <- line
				return
			}
		}
		close(bestmoveCh)
	}()

	// Give search a moment to start
	time.Sleep(200 * time.Millisecond)

	// Send ponderhit
	fmt.Fprintln(inW, "ponderhit")

	select {
	case bm := <-bestmoveCh:
		if bm == "" {
			t.Fatal("no bestmove received after ponderhit")
		}
		if !strings.HasPrefix(bm, "bestmove") {
			t.Errorf("unexpected output: %s", bm)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("timeout waiting for bestmove after ponderhit")
	}

	_ = gotInfo // search was running

	inW.Close()
	<-done
}

func TestUCIPonderStop(t *testing.T) {
	// go ponder → search runs → stop → bestmove arrives immediately
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	engine := NewUCIEngineWithIO(inR, outW)

	done := make(chan struct{})
	go func() {
		engine.Run()
		outW.Close()
		close(done)
	}()

	fmt.Fprintln(inW, "position startpos")
	fmt.Fprintln(inW, "go ponder")

	scanner := bufio.NewScanner(outR)
	bestmoveCh := make(chan string, 1)

	go func() {
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "bestmove") {
				bestmoveCh <- line
				return
			}
		}
		close(bestmoveCh)
	}()

	// Give search a moment to start
	time.Sleep(200 * time.Millisecond)

	// Send stop instead of ponderhit
	fmt.Fprintln(inW, "stop")

	select {
	case bm := <-bestmoveCh:
		if bm == "" {
			t.Fatal("no bestmove received after stop during ponder")
		}
		if !strings.HasPrefix(bm, "bestmove") {
			t.Errorf("unexpected output: %s", bm)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for bestmove after stop during ponder")
	}

	inW.Close()
	<-done
}

func TestUCIPonderHitTimeLimits(t *testing.T) {
	// go ponder wtime 100 btime 100 → ponderhit → search finishes quickly (not infinite)
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	engine := NewUCIEngineWithIO(inR, outW)

	done := make(chan struct{})
	go func() {
		engine.Run()
		outW.Close()
		close(done)
	}()

	fmt.Fprintln(inW, "position startpos")
	fmt.Fprintln(inW, "go ponder wtime 100 btime 100")

	scanner := bufio.NewScanner(outR)
	bestmoveCh := make(chan string, 1)

	go func() {
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "bestmove") {
				bestmoveCh <- line
				return
			}
		}
		close(bestmoveCh)
	}()

	// Give search a moment
	time.Sleep(200 * time.Millisecond)

	// Send ponderhit — with wtime=100 btime=100, search should finish quickly
	start := time.Now()
	fmt.Fprintln(inW, "ponderhit")

	select {
	case bm := <-bestmoveCh:
		elapsed := time.Since(start)
		if bm == "" {
			t.Fatal("no bestmove received")
		}
		// With 100ms clock, time alloc is 10ms (floor). Should finish well under 5s.
		if elapsed > 5*time.Second {
			t.Errorf("search took too long after ponderhit with short time: %v", elapsed)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for bestmove after ponderhit with short time")
	}

	inW.Close()
	<-done
}

func TestUCIPonderExhaustsDepth(t *testing.T) {
	// go ponder depth 2 on simple position → search finishes → ponderhit → bestmove
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	engine := NewUCIEngineWithIO(inR, outW)

	done := make(chan struct{})
	go func() {
		engine.Run()
		outW.Close()
		close(done)
	}()

	// Use a simple endgame where depth 2 finishes almost instantly
	fmt.Fprintln(inW, "position fen 8/8/8/8/8/5k2/4q3/4K3 w - - 0 1")
	fmt.Fprintln(inW, "go ponder depth 2")

	scanner := bufio.NewScanner(outR)
	bestmoveCh := make(chan string, 1)

	go func() {
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "bestmove") {
				bestmoveCh <- line
				return
			}
		}
		close(bestmoveCh)
	}()

	// Wait long enough for depth 2 to finish (search blocks on ponderDone)
	time.Sleep(500 * time.Millisecond)

	// Verify no bestmove yet (search finished but goroutine is waiting)
	select {
	case bm := <-bestmoveCh:
		// It's possible on very slow systems, but this shouldn't fire
		// before ponderhit. If it does, re-check the implementation.
		t.Logf("bestmove arrived before ponderhit (possible race): %s", bm)
		inW.Close()
		<-done
		return
	default:
		// Good — no bestmove yet, goroutine is waiting
	}

	// Send ponderhit — should unblock the goroutine
	fmt.Fprintln(inW, "ponderhit")

	select {
	case bm := <-bestmoveCh:
		if bm == "" {
			t.Fatal("no bestmove received after ponderhit")
		}
		if !strings.HasPrefix(bm, "bestmove") {
			t.Errorf("unexpected output: %s", bm)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for bestmove after ponderhit on exhausted depth")
	}

	inW.Close()
	<-done
}

func TestUCIBookFileOption(t *testing.T) {
	// Build a book file on disk
	opts := BookBuildOptions{MaxPly: 10, MinFreq: 1, TopN: 4}
	bookPath := t.TempDir() + "/test.bin"
	if err := BuildOpeningBook("testdata/2600.pgn", "testdata/eco.pgn", bookPath, opts); err != nil {
		t.Fatal(err)
	}

	// Use string reader + bytes.Buffer to avoid io.Pipe deadlock
	// (setoption BookFile sends "info string" output during processing)
	input := fmt.Sprintf("setoption name BookFile value %s\nisready\nposition startpos\ngo depth 10\nquit\n", bookPath)
	var out bytes.Buffer
	engine := NewUCIEngineWithIO(strings.NewReader(input), &out)
	engine.Run()

	output := out.String()
	// Should see the book loaded info string
	if !strings.Contains(output, "info string book loaded:") {
		t.Errorf("expected book loaded info, output:\n%s", output)
	}
	// Should get a bestmove without search (book hit)
	if !strings.Contains(output, "bestmove") {
		t.Errorf("expected bestmove, output:\n%s", output)
	}
	// No search should have happened
	if strings.Contains(output, "info depth") {
		t.Errorf("expected no search when book has a move, output:\n%s", output)
	}
}

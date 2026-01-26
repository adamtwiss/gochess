package chess

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strings"
	"testing"
)

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
		want                                          int
	}{
		{"movetime", 5000, 0, 0, 0, 0, 0, false, White, 5000},
		{"infinite", 0, 0, 0, 0, 0, 0, true, White, 0},
		{"no clock", 0, 0, 0, 0, 0, 0, false, White, 0},
		{"white clock 30s", 0, 30000, 0, 0, 0, 0, false, White, 1000},             // 30000/30 = 1000
		{"black clock 30s", 0, 0, 30000, 0, 0, 0, false, Black, 1000},             // 30000/30 = 1000
		{"with increment", 0, 30000, 0, 2000, 0, 0, false, White, 2500},           // 30000/30 + 2000*3/4 = 1000+1500 = 2500
		{"movestogo 10", 0, 30000, 0, 0, 0, 10, false, White, 3000},               // 30000/10 = 3000
		{"cap at half", 0, 2000, 0, 0, 0, 0, false, White, 66},                    // 2000/30=66, cap=1000, 66<1000
		{"floor at 10ms", 0, 100, 0, 0, 0, 0, false, White, 10},                   // 100/30=3 → floor at 10
		{"movetime overrides clock", 3000, 60000, 60000, 0, 0, 0, false, White, 3000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := computeSearchTime(tt.movetime, tt.wtime, tt.btime, tt.winc, tt.binc, tt.movestogo, tt.infinite, tt.side)
			if got != tt.want {
				t.Errorf("computeSearchTime() = %d, want %d", got, tt.want)
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

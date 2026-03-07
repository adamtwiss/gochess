package chess

import (
	"fmt"
	"testing"
)

// TestNNUEDeadNeurons checks neuron activation across many diverse positions
// to determine if neurons are truly dead (never activate) vs just inactive at startpos.
func TestNNUEDeadNeurons(t *testing.T) {
	net, err := LoadNNUE("net.nnue")
	if err != nil {
		t.Skipf("no net.nnue found: %v", err)
	}
	net.PrepareWeights()
	trainNet := DequantizeNetwork(net)

	positions := []string{
		// Starting and early game
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
		"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
		"rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
		// Middlegame positions
		"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
		"r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 5",
		"r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 6 8",
		"r1bq1rk1/pp2bppp/2n1pn2/3p4/3P1B2/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
		// Unbalanced material
		"r1bqkbnr/pppppppp/2n5/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",     // extra knight black
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",       // missing queen white
		"rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1",       // missing pawn white
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1",        // missing rook white
		"4k3/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",              // stripped black
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/4K3 w kq - 0 1",              // stripped white
		// Endgames
		"4k3/8/8/8/8/8/4PPPP/4K2R w K - 0 1",
		"4k3/pppp4/8/8/8/8/PPPP4/4K3 w - - 0 1",
		"8/5k2/8/8/3Q4/8/8/4K3 w - - 0 1",
		"8/5k2/8/8/3r4/8/5PP1/4K3 w - - 0 1",
		"8/8/3k4/8/8/3K4/3P4/8 w - - 0 1",
		"8/3k4/8/8/8/8/3KR3/8 w - - 0 1",
		// Tactical positions
		"r1bqk2r/ppppbppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
		"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
		"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
		"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
		// Pawn structures
		"4k3/pp3ppp/4p3/3p4/3P4/4P3/PP3PPP/4K3 w - - 0 1",
		"4k3/1p4pp/p2p4/P1pPp3/2P1P3/6PP/1P6/4K3 w - - 0 1",
		// Kings on unusual squares
		"8/8/8/4k3/8/4K3/8/8 w - - 0 1",
		"r1bq1rk1/ppp2ppp/2np4/8/3NP1n1/2N5/PPP2PPP/R1BQKB1R w KQ - 0 1",
		// Opposite castling
		"r3kb1r/pp1n1ppp/1qp1pn2/3p1b2/3P1B2/2NBPN2/PPP2PPP/R2QK2R w KQkq - 0 8",
		"2kr3r/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP2PPP/R1B1K2R w KQ - 0 8",
	}

	// Track which neurons ever activate across ALL positions
	hidden1EverActive := make([]bool, NNUEHiddenSize*2)
	hidden2EverActive := make([]bool, NNUEHidden2Size)
	hidden3EverActive := make([]bool, NNUEHidden3Size)

	for _, fen := range positions {
		var b Board
		b.Reset()
		if err := b.SetFEN(fen); err != nil {
			t.Fatalf("bad FEN: %v", err)
		}

		sample := extractFeatures(&b)
		_, hidden1, hidden2, hidden3 := trainNet.Forward(sample)

		for i := 0; i < NNUEHiddenSize*2; i++ {
			if hidden1[i] > 0 {
				hidden1EverActive[i] = true
			}
		}
		for i := 0; i < NNUEHidden2Size; i++ {
			if hidden2[i] > 0 {
				hidden2EverActive[i] = true
			}
		}
		for i := 0; i < NNUEHidden3Size; i++ {
			if hidden3[i] > 0 {
				hidden3EverActive[i] = true
			}
		}
	}

	// Count truly dead neurons
	var h1Dead, h2Dead, h3Dead int
	for i := range hidden1EverActive {
		if !hidden1EverActive[i] {
			h1Dead++
		}
	}
	for i := range hidden2EverActive {
		if !hidden2EverActive[i] {
			h2Dead++
		}
	}
	for i := range hidden3EverActive {
		if !hidden3EverActive[i] {
			h3Dead++
		}
	}

	fmt.Printf("Tested %d diverse positions\n\n", len(positions))
	fmt.Printf("Hidden1 (CReLU, %d neurons):\n", NNUEHiddenSize*2)
	fmt.Printf("  Ever active: %d/%d (%.1f%%)\n", NNUEHiddenSize*2-h1Dead, NNUEHiddenSize*2, 100*float64(NNUEHiddenSize*2-h1Dead)/float64(NNUEHiddenSize*2))
	fmt.Printf("  Truly dead:  %d/%d (%.1f%%)\n", h1Dead, NNUEHiddenSize*2, 100*float64(h1Dead)/float64(NNUEHiddenSize*2))

	fmt.Printf("\nHidden2 (%d neurons):\n", NNUEHidden2Size)
	fmt.Printf("  Ever active: %d/%d (%.1f%%)\n", NNUEHidden2Size-h2Dead, NNUEHidden2Size, 100*float64(NNUEHidden2Size-h2Dead)/float64(NNUEHidden2Size))
	fmt.Printf("  Truly dead:  %d/%d (%.1f%%)\n", h2Dead, NNUEHidden2Size, 100*float64(h2Dead)/float64(NNUEHidden2Size))

	fmt.Printf("\nHidden3 (%d neurons):\n", NNUEHidden3Size)
	fmt.Printf("  Ever active: %d/%d (%.1f%%)\n", NNUEHidden3Size-h3Dead, NNUEHidden3Size, 100*float64(NNUEHidden3Size-h3Dead)/float64(NNUEHidden3Size))
	fmt.Printf("  Truly dead:  %d/%d (%.1f%%)\n", h3Dead, NNUEHidden3Size, 100*float64(h3Dead)/float64(NNUEHidden3Size))
}

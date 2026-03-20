package chess

import (
	"testing"
)

func BenchmarkSearchNNUE(b *testing.B) {
	net, err := LoadNNUE("net.nnue")
	if err != nil {
		b.Skipf("no net.nnue: %v", err)
	}
	net.PrepareWeights()

	oldUseNNUE := UseNNUE
	UseNNUE = true
	defer func() { UseNNUE = oldUseNNUE }()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var board Board
		board.Reset()
		board.NNUENet = net
		accStack := NewNNUEAccumulatorStack(64)
		board.NNUEAcc = accStack
		net.RecomputeAccumulator(accStack.Current(), &board)
		board.Search(8, 0)
	}
}

// BenchmarkSearchNNUEV5 benchmarks a depth-8 search with v5 NNUE.
func BenchmarkSearchNNUEV5(b *testing.B) {
	netV5 := loadTestV5Net(b)

	oldUseNNUE := UseNNUE
	UseNNUE = true
	defer func() { UseNNUE = oldUseNNUE }()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var board Board
		board.Reset()
		board.AttachNNUEV5(netV5)
		board.Search(8, 0)
	}
}

// BenchmarkMaterializeV5Quiet benchmarks MaterializeV5 on quiet moves (Type 1).
// This is the hottest path: ~70% of all materialization calls.
func BenchmarkMaterializeV5Quiet(b *testing.B) {
	netV5 := loadTestV5Net(b)

	// Set up a position and make a quiet move to get a Type 1 dirty piece
	var board Board
	board.Reset()
	board.AttachNNUEV5(netV5)
	board.NNUEAccV5.MaterializeV5(netV5, &board)

	// e2e4 — quiet pawn push
	move := NewMove(NewSquare(4, 1), NewSquare(4, 3))
	board.MakeMove(move)

	// Verify we got a Type 1 dirty piece
	acc := board.NNUEAccV5.Current()
	if acc.Dirty.Type != 1 {
		b.Fatalf("expected dirty type 1 (quiet), got %d", acc.Dirty.Type)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		acc.Computed = false
		board.NNUEAccV5.MaterializeV5(netV5, &board)
	}
}

// BenchmarkMaterializeV5Capture benchmarks MaterializeV5 on captures (Type 2).
func BenchmarkMaterializeV5Capture(b *testing.B) {
	netV5 := loadTestV5Net(b)

	// Set up a position with a capture available
	var board Board
	board.SetFEN("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
	board.AttachNNUEV5(netV5)
	board.NNUEAccV5.MaterializeV5(netV5, &board)

	// exd5 — capture
	move := NewMove(NewSquare(4, 3), NewSquare(3, 4))
	board.MakeMove(move)

	acc := board.NNUEAccV5.Current()
	if acc.Dirty.Type != 2 {
		b.Fatalf("expected dirty type 2 (capture), got %d", acc.Dirty.Type)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		acc.Computed = false
		board.NNUEAccV5.MaterializeV5(netV5, &board)
	}
}

// BenchmarkForwardV5 benchmarks the v5 forward pass in isolation.
func BenchmarkForwardV5(b *testing.B) {
	netV5 := loadTestV5Net(b)

	var board Board
	board.Reset()
	board.AttachNNUEV5(netV5)
	board.NNUEAccV5.MaterializeV5(netV5, &board)
	acc := board.NNUEAccV5.Current()
	pieceCount := board.AllPieces.Count()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		netV5.Forward(acc, White, pieceCount)
	}
}

// loadTestV5Net loads the best available v5 net for benchmarking.
func loadTestV5Net(b *testing.B) *NNUENetV5 {
	b.Helper()
	// Try common v5 net names
	for _, name := range []string{"net-v5-multi-sb120.nnue", "net-v5-120sb-sb120.nnue"} {
		net, err := LoadNNUEV5(name)
		if err == nil {
			return net
		}
	}
	b.Skip("no v5 net available")
	return nil
}

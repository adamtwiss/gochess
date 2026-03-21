package chess

import "testing"

func BenchmarkSearchNNUEV5_1536(b *testing.B) {
	net, err := LoadNNUEV5("net-v5-1536-wdl00-sb200.nnue")
	if err != nil {
		b.Skipf("no 1536 net: %v", err)
	}
	oldUseNNUE := UseNNUE
	UseNNUE = true
	defer func() { UseNNUE = oldUseNNUE }()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var board Board
		board.Reset()
		board.AttachNNUEV5(net)
		board.Search(8, 0)
	}
}

func BenchmarkMaterializeV5_1536(b *testing.B) {
	net, err := LoadNNUEV5("net-v5-1536-wdl00-sb200.nnue")
	if err != nil {
		b.Skipf("no 1536 net: %v", err)
	}
	var board Board
	board.Reset()
	board.AttachNNUEV5(net)
	board.NNUEAccV5.MaterializeV5(net, &board)
	move := NewMove(NewSquare(4, 1), NewSquare(4, 3))
	board.MakeMove(move)
	acc := board.NNUEAccV5.Current()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		acc.Computed = false
		board.NNUEAccV5.MaterializeV5(net, &board)
	}
}

func BenchmarkForwardV5_1536(b *testing.B) {
	net, err := LoadNNUEV5("net-v5-1536-wdl00-sb200.nnue")
	if err != nil {
		b.Skipf("no 1536 net: %v", err)
	}
	var board Board
	board.Reset()
	board.AttachNNUEV5(net)
	board.NNUEAccV5.MaterializeV5(net, &board)
	acc := board.NNUEAccV5.Current()
	pieceCount := board.AllPieces.Count()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		net.Forward(acc, White, pieceCount)
	}
}

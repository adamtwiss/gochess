package chess

import "testing"

func BenchmarkSearchNNUEV5_1536(b *testing.B) {
	net := loadTest1536Net(b)
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
	net := loadTest1536Net(b)
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
	net := loadTest1536Net(b)
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

func BenchmarkForwardV5_1536_CReLUDotOnly(b *testing.B) {
	net := loadTest1536Net(b)
	if net.UseSCReLU {
		b.Skip("net is SCReLU, need CReLU for this bench")
	}
	var board Board
	board.Reset()
	board.AttachNNUEV5(net)
	board.NNUEAccV5.MaterializeV5(net, &board)
	acc := board.NNUEAccV5.Current()
	H := net.HiddenSize
	outW := net.OutputWeights[:H]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nnueV5CReLUDotN(&acc.White[0], &outW[0], H)
	}
}

func loadTest1536Net(b *testing.B) *NNUENetV5 {
	b.Helper()
	for _, name := range []string{
		"net-v5-1536-w5-e800s800.nnue",
		"net-v5-1536-wdl00-sb800.nnue",
		"net-v5-1536-wdl00-sb200.nnue",
	} {
		net, err := LoadNNUEV5(name)
		if err == nil {
			return net
		}
	}
	b.Skip("no 1536 net available")
	return nil
}

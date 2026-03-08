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

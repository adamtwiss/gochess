package chess

import "testing"

func BenchmarkSearchNNUEV5_1536_d9(b *testing.B) {
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
		board.Search(9, 0)
	}
}

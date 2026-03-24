package chess

import "testing"

func BenchmarkSearchNNUEV5_1536_d9(b *testing.B) {
	net := loadTest1536Net(b)
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

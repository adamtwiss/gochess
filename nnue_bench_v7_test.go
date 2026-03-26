package chess

import "testing"

// loadTestV7Net loads a v7 net for benchmarking.
func loadTestV7Net(b *testing.B) *NNUENetV5 {
	b.Helper()
	for _, name := range []string{
		"net-v7-1024h16x32s-w0-e800i8s800.nnue",
	} {
		net, err := LoadNNUEV5(name)
		if err == nil {
			if net.L1Size == 0 {
				continue
			}
			return net
		}
	}
	b.Skip("no v7 net available")
	return nil
}

// BenchmarkForwardV7 benchmarks the full v7 forward pass (L1+L2+output).
func BenchmarkForwardV7(b *testing.B) {
	net := loadTestV7Net(b)
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

// BenchmarkForwardV7SCReLUPack benchmarks just the SCReLU pack step (int16→uint8).
func BenchmarkForwardV7SCReLUPack(b *testing.B) {
	net := loadTestV7Net(b)
	var board Board
	board.Reset()
	board.AttachNNUEV5(net)
	board.NNUEAccV5.MaterializeV5(net, &board)
	acc := board.NNUEAccV5.Current()
	H := net.HiddenSize
	var buf [1536]byte
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nnueSCReLUPack(&acc.White[0], &buf[0], H)
	}
}

// BenchmarkForwardV7Int8MatMul benchmarks just the int8 L1 matmul kernel.
func BenchmarkForwardV7Int8MatMul(b *testing.B) {
	net := loadTestV7Net(b)
	if net.L1Weights8T == nil {
		b.Skip("no int8 weights")
	}
	var board Board
	board.Reset()
	board.AttachNNUEV5(net)
	board.NNUEAccV5.MaterializeV5(net, &board)
	acc := board.NNUEAccV5.Current()
	H := net.HiddenSize
	L1 := net.L1Size

	// Pre-pack accumulator
	var buf [1536]byte
	nnueSCReLUPack(&acc.White[0], &buf[0], H)

	var hidden [64]int32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < L1; j++ {
			hidden[j] = 0
		}
		nnueV5L1Int8MatMulN(&buf[0], &net.L1Weights8T[0], &hidden[0], H, L1)
	}
}

// BenchmarkSearchV7 benchmarks a depth-8 search with v7 int8 NNUE.
func BenchmarkSearchV7(b *testing.B) {
	net := loadTestV7Net(b)
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

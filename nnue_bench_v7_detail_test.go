package chess

import "testing"

// BenchmarkForwardV7Breakdown benchmarks individual stages of the v7 forward pass.
func BenchmarkForwardV7Breakdown(b *testing.B) {
	net := loadTestV7Net(b)
	var board Board
	board.Reset()
	board.AttachNNUEV5(net)
	board.NNUEAccV5.MaterializeV5(net, &board)
	acc := board.NNUEAccV5.Current()
	H := net.HiddenSize
	L1 := net.L1Size

	stmAcc := acc.White
	ntmAcc := acc.Black

	qaInt := int32(nnueV5InputScale)
	qaL1 := int32(net.L1Scale)
	if qaL1 == 0 {
		qaL1 = int32(nnueV5InputScale)
	}

	// Pre-pack accumulators
	var stmBuf, ntmBuf [1536]byte
	nnueSCReLUPack(&stmAcc[0], &stmBuf[0], H)
	nnueSCReLUPack(&ntmAcc[0], &ntmBuf[0], H)

	b.Run("PackOnly", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			nnueSCReLUPack(&stmAcc[0], &stmBuf[0], H)
			nnueSCReLUPack(&ntmAcc[0], &ntmBuf[0], H)
		}
	})

	b.Run("MatMulOnly", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var hidden32 [64]int32
			for j := 0; j < L1; j++ {
				hidden32[j] = int32(net.L1Biases[j]) * qaInt
			}
			nnueV5L1Int8MatMulN(&stmBuf[0], &net.L1Weights8T[0], &hidden32[0], H, L1)
			nnueV5L1Int8MatMulN(&ntmBuf[0], &net.L1Weights8T[L1*H], &hidden32[0], H, L1)
		}
	})

	b.Run("DequantSCReLU", func(b *testing.B) {
		var hidden32 [64]int32
		for j := 0; j < L1; j++ {
			hidden32[j] = int32(net.L1Biases[j]) * qaInt
		}
		nnueV5L1Int8MatMulN(&stmBuf[0], &net.L1Weights8T[0], &hidden32[0], H, L1)
		nnueV5L1Int8MatMulN(&ntmBuf[0], &net.L1Weights8T[L1*H], &hidden32[0], H, L1)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			qaL1sq := float32(qaL1) * float32(qaL1)
			var l1f [64]float32
			for j := 0; j < L1; j++ {
				h := hidden32[j] / qaInt
				if h < 0 {
					h = 0
				}
				if h > int32(qaL1) {
					h = int32(qaL1)
				}
				hsq := h * h
				l1f[j] = float32(hsq) / qaL1sq
			}
			_ = l1f
		}
	})

	b.Run("L2Forward", func(b *testing.B) {
		// Prepare L1 output
		var hidden32 [64]int32
		for j := 0; j < L1; j++ {
			hidden32[j] = int32(net.L1Biases[j]) * qaInt
		}
		nnueV5L1Int8MatMulN(&stmBuf[0], &net.L1Weights8T[0], &hidden32[0], H, L1)
		nnueV5L1Int8MatMulN(&ntmBuf[0], &net.L1Weights8T[L1*H], &hidden32[0], H, L1)

		qaL1sq := float32(qaL1) * float32(qaL1)
		var l1f [64]float32
		for j := 0; j < L1; j++ {
			h := hidden32[j] / qaInt
			if h < 0 {
				h = 0
			}
			if h > int32(qaL1) {
				h = int32(qaL1)
			}
			hsq := h * h
			l1f[j] = float32(hsq) / qaL1sq
		}

		bucket := 0
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			net.forwardL2SCReLUFloat(l1f[:L1], bucket)
		}
	})
}

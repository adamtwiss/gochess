//go:build arm64

package chess

// NEON is always available on ARM64, no runtime detection needed.
var nnueUseSIMD = true
var nnueUseSIMDV5 = true

// nnueCReLU256 applies ClippedReLU (clamp to [0, 127]) to 256 int16 values.
//
//go:noescape
func nnueCReLU256(src *int16, dst *int16)

// nnueCReLU256to8 applies ClippedReLU to 256 int16 values and packs into uint8.
// Output values are clamped to [0, 127] and narrowed to bytes.
//
//go:noescape
func nnueCReLU256to8(src *int16, dst *byte)

// nnueMatMul32x512 computes the hidden layer matrix-vector multiply:
//
//	output[j] = biases[j] + sum_i(input[i] * weightsT[j][i])
//
// for j=0..31, i=0..511. weightsT is transposed: [32][512] row-major.
// Uses SMULL/SMLAL2 for int16 multiply-accumulate.
//
//go:noescape
func nnueMatMul32x512(input *int16, weightsT *int16, biases *int32, output *int32)

// nnueMatMul32x512_i8 computes the hidden layer matrix-vector multiply using int8 weights:
//
//	output[j] = biases[j] + sum_i(input[i] * weightsT[j][i])
//
// for j=0..31, i=0..511. input is uint8 [0,127], weightsT is int8 [-128,127].
// Uses SMULL/SMLAL2 on .8B for 2× element throughput over int16.
//
//go:noescape
func nnueMatMul32x512_i8(input *byte, weightsT *int8, biases *int32, output *int32)

// nnueAccSubAdd256 computes acc[i] += newW[i] - oldW[i] for i=0..255.
// Used for quiet moves (SubAddFeature).
//
//go:noescape
func nnueAccSubAdd256(acc *int16, oldW *int16, newW *int16)

// nnueAccSubSubAdd256 computes acc[i] += newW[i] - oldW[i] - capW[i] for i=0..255.
// Used for captures (SubSubAddFeature).
//
//go:noescape
func nnueAccSubSubAdd256(acc *int16, oldW *int16, newW *int16, capW *int16)

// nnueAccAdd256 computes acc[i] += weights[i] for i=0..255.
// Used in RecomputeAccumulator.
//
//go:noescape
func nnueAccAdd256(acc *int16, weights *int16)

// nnueAccSub256 computes acc[i] -= weights[i] for i=0..255.
// Used in RemoveFeature.
//
//go:noescape
func nnueAccSub256(acc *int16, weights *int16)

// nnueAccCopySubAdd256 computes dst[i] = src[i] + newW[i] - oldW[i] for i=0..255.
// Fused copy+update for quiet moves in Materialize.
//
//go:noescape
func nnueAccCopySubAdd256(dst *int16, src *int16, oldW *int16, newW *int16)

// nnueAccCopySubSubAdd256 computes dst[i] = src[i] + newW[i] - oldW[i] - capW[i] for i=0..255.
// Fused copy+update for captures in Materialize.
//
//go:noescape
func nnueAccCopySubSubAdd256(dst *int16, src *int16, oldW *int16, newW *int16, capW *int16)

// nnueMatMul32x32ReLU computes: output[k] = biases[k] + sum_j(ReLU(input[j]>>6) * weightsT[k*32+j])
// for k=0..31. weightsT is transposed: [32][32] row-major.
//
//go:noescape
func nnueMatMul32x32ReLU(input *int32, weightsT *int16, biases *int32, output *int32)

// nnueDotReLU32 computes: sum_k(ReLU(input[k]>>6) * int32(weights[k])) for k=0..31.
// Used for the output layer (32→1).
//
//go:noescape
func nnueDotReLU32(input *int32, weights *int16) int32

// nnueAccAddN computes acc[i] += weights[i] for i=0..count-1.
// count must be a multiple of 16.
//
//go:noescape
func nnueAccAddN(acc *int16, weights *int16, count int)

// nnueAccSubN computes acc[i] -= weights[i] for i=0..count-1.
// count must be a multiple of 16.
//
//go:noescape
func nnueAccSubN(acc *int16, weights *int16, count int)

// nnueAccSubAddN computes acc[i] += newW[i] - oldW[i] for i=0..count-1.
// count must be a multiple of 16.
//
//go:noescape
func nnueAccSubAddN(acc *int16, oldW *int16, newW *int16, count int)

// nnueAccCopySubAddN computes dst[i] = src[i] + newW[i] - oldW[i] for i=0..count-1.
// count must be a multiple of 16.
//
//go:noescape
func nnueAccCopySubAddN(dst *int16, src *int16, oldW *int16, newW *int16, count int)

// nnueAccCopySubSubAddN computes dst[i] = src[i] + newW[i] - oldW[i] - capW[i] for i=0..count-1.
// count must be a multiple of 16.
//
//go:noescape
func nnueAccCopySubSubAddN(dst *int16, src *int16, oldW *int16, newW *int16, capW *int16, count int)

// nnueV5CReLUDot1024 computes clamped dot product for v5 output layer.
//
//go:noescape
func nnueV5CReLUDot1024(acc *int16, weights *int16) int32

// nnueV5CReLUDotN computes CReLU dot product for any width (multiple of 16).
//
//go:noescape
func nnueV5CReLUDotN(acc *int16, weights *int16, count int) int32

// nnueV5SCReLUDotN computes exact SCReLU dot product for any width (multiple of 16).
// Returns int64 (caller divides by QA=255).
//
//go:noescape
func nnueV5SCReLUDotN(acc *int16, weights *int16, count int) int64

// nnueV5PairwiseDotN computes pairwise dot product for v5 768pw.
// Returns int64 (caller divides by QA=255).
//
//go:noescape
func nnueV5PairwiseDotN(accFirst *int16, accSecond *int16, weights *int16, count int) int64

// nnueV5L1MatMulN computes the L1 hidden layer matmul for one perspective:
//
//	hidden[i] += sum_j( clamp(acc[j], 0, 255) * wT[i*accLen + j] )
//
// for i=0..l1-1, j=0..accLen-1. accLen must be a multiple of 16.
// hidden must be pre-initialized (e.g. with biases). Accumulates, does not overwrite.
//
//go:noescape
func nnueV5L1MatMulN(acc *int16, wT *int16, hidden *int32, accLen int, l1 int)

// nnueV5L1SCReLUMatMulN computes the SCReLU L1 hidden layer matmul for one perspective:
//
//	hidden[i] += sum_j( clamp(acc[j], 0, 255)² * wT[i*accLen + j] )
//
// for i=0..l1-1, j=0..accLen-1. accLen must be a multiple of 16.
// hidden must be pre-initialized (e.g. with biases scaled to QA³).
//
//go:noescape
func nnueV5L1SCReLUMatMulN(acc *int16, wT *int16, hidden *int64, accLen int, l1 int)

//go:build arm64

package chess

// NEON is always available on ARM64, no runtime detection needed.
var nnueUseSIMD = true

// nnueCReLU256 applies ClippedReLU (clamp to [0, 127]) to 256 int16 values.
//
//go:noescape
func nnueCReLU256(src *int16, dst *int16)

// nnueMatMul32x512 computes the hidden layer matrix-vector multiply:
//
//	output[j] = biases[j] + sum_i(input[i] * weightsT[j][i])
//
// for j=0..31, i=0..511. weightsT is transposed: [32][512] row-major.
// Uses SMULL/SMLAL2 for int16 multiply-accumulate.
//
//go:noescape
func nnueMatMul32x512(input *int16, weightsT *int16, biases *int32, output *int32)

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

// nnueMatMul32x32ReLU and nnueDotReLU32 — Go scalar fallback for ARM64.
// TODO: implement NEON assembly versions for ~20% speedup.

func nnueMatMul32x32ReLU(input *int32, weightsT *int16, biases *int32, output *int32) {
	// output[k] = biases[k] + sum_j(max(0, input[j] >> 6) * weightsT[k*32+j])
	inp := (*[32]int32)(input)
	wt := (*[32 * 32]int16)(weightsT)
	bi := (*[32]int32)(biases)
	out := (*[32]int32)(output)
	for k := 0; k < 32; k++ {
		acc := bi[k]
		base := k * 32
		for j := 0; j < 32; j++ {
			v := inp[j] >> 6
			if v > 0 {
				acc += v * int32(wt[base+j])
			}
		}
		out[k] = acc
	}
}

func nnueDotReLU32(input *int32, weights *int16) int32 {
	// sum_k(max(0, input[k] >> 6) * int32(weights[k]))
	inp := (*[32]int32)(input)
	w := (*[32]int16)(weights)
	var acc int32
	for k := 0; k < 32; k++ {
		v := inp[k] >> 6
		if v > 0 {
			acc += v * int32(w[k])
		}
	}
	return acc
}

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

//go:build arm64

package chess

import "unsafe"

// NEON is always available on ARM64, no runtime detection needed.
var nnueUseSIMD = true
var nnueUseSIMDV5 = false // v5 dot-product kernels not yet implemented in NEON

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

// Width-generic stubs — delegate to 256-wide NEON kernels
func nnueAccAddN(acc *int16, weights *int16, count int) {
	// Cast to slices for safe indexing
	a := unsafe.Slice(acc, count)
	w := unsafe.Slice(weights, count)
	for off := 0; off < count; off += 256 {
		nnueAccAdd256(&a[off], &w[off])
	}
}

func nnueAccSubN(acc *int16, weights *int16, count int) {
	a := unsafe.Slice(acc, count)
	w := unsafe.Slice(weights, count)
	for off := 0; off < count; off += 256 {
		nnueAccSub256(&a[off], &w[off])
	}
}

func nnueAccSubAddN(acc *int16, oldW *int16, newW *int16, count int) {
	a := unsafe.Slice(acc, count)
	o := unsafe.Slice(oldW, count)
	n := unsafe.Slice(newW, count)
	for off := 0; off < count; off += 256 {
		nnueAccSub256(&a[off], &o[off])
		nnueAccAdd256(&a[off], &n[off])
	}
}

func nnueAccCopySubAddN(dst *int16, src *int16, oldW *int16, newW *int16, count int) {
	d := unsafe.Slice(dst, count)
	s := unsafe.Slice(src, count)
	o := unsafe.Slice(oldW, count)
	n := unsafe.Slice(newW, count)
	for off := 0; off < count; off += 256 {
		nnueAccCopySubAdd256(&d[off], &s[off], &o[off], &n[off])
	}
}

func nnueAccCopySubSubAddN(dst *int16, src *int16, oldW *int16, newW *int16, capW *int16, count int) {
	d := unsafe.Slice(dst, count)
	s := unsafe.Slice(src, count)
	o := unsafe.Slice(oldW, count)
	n := unsafe.Slice(newW, count)
	c := unsafe.Slice(capW, count)
	for off := 0; off < count; off += 256 {
		nnueAccCopySubSubAdd256(&d[off], &s[off], &o[off], &n[off], &c[off])
	}
}

// nnueV5CReLUDot1024 computes clamped dot product for v5 output layer.
// TODO: implement NEON version
//go:noescape
func nnueV5CReLUDot1024(acc *int16, weights *int16) int32

//go:noescape
func nnueV5SCReLUDot1024(acc *int16, weights *int16) int32

//go:noescape
func nnueV5CReLUDotN(acc *int16, weights *int16, count int) int32

//go:noescape
func nnueV5SCReLUDotN(acc *int16, weights *int16, count int) int64

// nnueV5PairwiseDotN computes pairwise dot product for v5 768pw.
// TODO: implement NEON version
//go:noescape
func nnueV5PairwiseDotN(accFirst *int16, accSecond *int16, weights *int16, count int) int64

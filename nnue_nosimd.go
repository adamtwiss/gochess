//go:build !amd64 && !arm64

package chess

var nnueUseSIMD = false
var nnueUseSIMDV5 = false

// Stub functions — never called when nnueUseSIMD is false,
// but needed for compilation on platforms without SIMD support.

func nnueCReLU256(src *int16, dst *int16)                                           { panic("SIMD not available") }
func nnueCReLU256to8(src *int16, dst *byte)                                         { panic("SIMD not available") }
func nnueMatMul32x512(input *int16, weightsT *int16, biases *int32, output *int32)  { panic("SIMD not available") }
func nnueMatMul32x512_i8(input *byte, weightsT *int8, biases *int32, output *int32) { panic("SIMD not available") }
func nnueAccSubAdd256(acc *int16, oldW *int16, newW *int16)                    { panic("SIMD not available") }
func nnueAccSubSubAdd256(acc *int16, oldW *int16, newW *int16, capW *int16)    { panic("SIMD not available") }
func nnueAccAdd256(acc *int16, weights *int16)                                 { panic("SIMD not available") }
func nnueAccSub256(acc *int16, weights *int16)                                 { panic("SIMD not available") }
func nnueAccCopySubAdd256(dst *int16, src *int16, oldW *int16, newW *int16)    { panic("SIMD not available") }
func nnueAccCopySubSubAdd256(dst *int16, src *int16, oldW *int16, newW *int16, capW *int16) { panic("SIMD not available") }
func nnueMatMul32x32ReLU(input *int32, weightsT *int16, biases *int32, output *int32) { panic("SIMD not available") }
func nnueV5CReLUDot1024(acc *int16, weights *int16) int32 { panic("SIMD not available") }
func nnueDotReLU32(input *int32, weights *int16) int32 { panic("SIMD not available") }
func nnueAccAddN(acc *int16, weights *int16, count int)                              { panic("SIMD not available") }
func nnueAccSubN(acc *int16, weights *int16, count int)                              { panic("SIMD not available") }
func nnueAccSubAddN(acc *int16, oldW *int16, newW *int16, count int)                 { panic("SIMD not available") }
func nnueAccCopySubAddN(dst *int16, src *int16, oldW *int16, newW *int16, count int) { panic("SIMD not available") }
func nnueAccCopySubSubAddN(dst *int16, src *int16, oldW *int16, newW *int16, capW *int16, count int) { panic("SIMD not available") }
func nnueV5CReLUDotN(acc *int16, weights *int16, count int) int32  { panic("SIMD not available") }
func nnueV5SCReLUDotN(acc *int16, weights *int16, count int) int64 { panic("SIMD not available") }
func nnueV5PairwiseDotN(accFirst *int16, accSecond *int16, weights *int16, count int) int64 { panic("SIMD not available") }
func nnueV5L1MatMulN(acc *int16, wT *int16, hidden *int32, accLen int, l1 int) { panic("SIMD not available") }
func nnueSCReLUPack(src *int16, dst *byte, count int) { panic("SIMD not available") }
func nnueV5L1Int8MatMulN(acc8 *byte, wT8 *int8, hidden *int32, accLen int, l1 int) { panic("SIMD not available") }
func nnueV5L1SCReLUMatMulN(acc *int16, wT *int16, hidden *int64, accLen int, l1 int) { panic("SIMD not available") }

//go:build !amd64 && !arm64

package chess

var nnueUseSIMD = false

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
func nnueV5SCReLUDot1024(acc *int16, weights *int16) int32 { panic("SIMD not available") }

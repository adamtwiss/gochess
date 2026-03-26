//go:build amd64

#include "textflag.h"

// NNUE AVX2 SIMD implementations for forward pass and accumulator updates.
//
// All functions use ABI0 calling convention (arguments on stack via FP).
// VZEROUPPER is called before every RET to avoid AVX/SSE transition penalties.

// ============================================================================
// Constants
// ============================================================================

// 16 x int16(127) for ClippedReLU max clamp
DATA nnue_clamp_127<>+0(SB)/8, $0x007f007f007f007f
DATA nnue_clamp_127<>+8(SB)/8, $0x007f007f007f007f
DATA nnue_clamp_127<>+16(SB)/8, $0x007f007f007f007f
DATA nnue_clamp_127<>+24(SB)/8, $0x007f007f007f007f
GLOBL nnue_clamp_127<>(SB), NOPTR+RODATA, $32

// 16 x int16(1) for VPMADDWD widening in int8 matmul
DATA nnue_ones_16<>+0(SB)/8, $0x0001000100010001
DATA nnue_ones_16<>+8(SB)/8, $0x0001000100010001
DATA nnue_ones_16<>+16(SB)/8, $0x0001000100010001
DATA nnue_ones_16<>+24(SB)/8, $0x0001000100010001
GLOBL nnue_ones_16<>(SB), NOPTR+RODATA, $32

// ============================================================================
// nnueCReLUPackN(src *int16, dst *byte, count int)
// Packs int16 to uint8 with CReLU: dst[i] = clamp(src[i], 0, 255)
// count must be a multiple of 32. Processes 32 elements per iteration.
// ============================================================================
TEXT ·nnueCReLUPackN(SB), NOSPLIT, $0-24
	MOVQ src+0(FP), AX
	MOVQ dst+8(FP), BX
	MOVQ count+16(FP), CX
	SHRQ $5, CX                         // count / 32

	VPXOR Y0, Y0, Y0                    // zero (floor)
	VMOVDQU nnue_clamp_255<>(SB), Y1    // 255 (ceiling)

crelu_pack_loop:
	VMOVDQU (AX), Y2
	VPMAXSW Y0, Y2, Y2
	VPMINSW Y1, Y2, Y2
	VMOVDQU 32(AX), Y3
	VPMAXSW Y0, Y3, Y3
	VPMINSW Y1, Y3, Y3
	VPACKUSWB Y3, Y2, Y4
	VPERMQ $0xD8, Y4, Y4                // fix lane interleaving
	VMOVDQU Y4, (BX)
	ADDQ $64, AX
	ADDQ $32, BX
	DECQ CX
	JNZ crelu_pack_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueInt8DotN(a *byte, b *int8, count int) int32
// Dot product of uint8 × int8 using VPMADDUBSW + VPMADDWD.
// count must be a multiple of 64 (2x unrolled, 32 per group).
// ============================================================================
TEXT ·nnueInt8DotN(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ count+16(FP), CX
	SHRQ $6, CX                         // count / 64

	VMOVDQU nnue_ones_16<>(SB), Y0      // ones for VPMADDWD widening
	VPXOR Y6, Y6, Y6                    // accumulator 1
	VPXOR Y7, Y7, Y7                    // accumulator 2

i8dot_loop:
	// First 32
	VMOVDQU (AX), Y2
	VMOVDQU (BX), Y3
	VPMADDUBSW Y3, Y2, Y4
	VPMADDWD Y0, Y4, Y4
	VPADDD Y4, Y6, Y6
	// Second 32
	VMOVDQU 32(AX), Y2
	VMOVDQU 32(BX), Y3
	VPMADDUBSW Y3, Y2, Y5
	VPMADDWD Y0, Y5, Y5
	VPADDD Y5, Y7, Y7

	ADDQ $64, AX
	ADDQ $64, BX
	DECQ CX
	JNZ i8dot_loop

	// Horizontal sum
	VPADDD Y7, Y6, Y6
	VEXTRACTI128 $1, Y6, X5
	VPADDD X5, X6, X6
	VPSHUFD $0x4E, X6, X5
	VPADDD X5, X6, X6
	VPSHUFD $0x01, X6, X5
	VPADDD X5, X6, X6
	VMOVD X6, AX
	MOVL AX, ret+24(FP)

	VZEROUPPER
	RET

// 16 x uint16(257) for SCReLU v²/255 approximation: VPMULHUW(v², 257) ≈ v²/255
DATA nnue_257_16<>+0(SB)/8, $0x0101010101010101
DATA nnue_257_16<>+8(SB)/8, $0x0101010101010101
DATA nnue_257_16<>+16(SB)/8, $0x0101010101010101
DATA nnue_257_16<>+24(SB)/8, $0x0101010101010101
GLOBL nnue_257_16<>(SB), NOPTR+RODATA, $32

// ============================================================================
// ttPrefetch(bucket *TTBucket)
// Issues PREFETCHT0 to bring a TT bucket into L1 cache.
// ============================================================================
TEXT ·ttPrefetch(SB), NOSPLIT, $0-8
	MOVQ bucket+0(FP), AX
	PREFETCHT0 (AX)
	RET

// ============================================================================
// nnueSCReLUPack(src *int16, dst *byte, count int)
// Packs int16 accumulator to uint8 with SCReLU: dst[i] = clamp(src[i],0,255)²/255
// count must be a multiple of 32. Processes 32 elements per iteration.
// ============================================================================
TEXT ·nnueSCReLUPack(SB), NOSPLIT, $0-24
	MOVQ src+0(FP), AX
	MOVQ dst+8(FP), BX
	MOVQ count+16(FP), CX
	SHRQ $5, CX                         // count / 32

	VPXOR Y0, Y0, Y0                    // zero (floor)
	VMOVDQU nnue_clamp_255<>(SB), Y1    // 255 (ceiling)
	VMOVDQU nnue_257_16<>(SB), Y6       // 257 for /255 approximation

screlu_pack_loop:
	// First 16 int16 values
	VMOVDQU (AX), Y2
	VPMAXSW Y0, Y2, Y2                  // clamp floor
	VPMINSW Y1, Y2, Y2                  // clamp ceiling [0, 255]
	VPMULLW Y2, Y2, Y3                  // v² (uint16, max 65025)
	VPMULHUW Y6, Y3, Y3                 // v² * 257 >> 16 ≈ v²/255 [0, 255]

	// Second 16 int16 values
	VMOVDQU 32(AX), Y4
	VPMAXSW Y0, Y4, Y4
	VPMINSW Y1, Y4, Y4
	VPMULLW Y4, Y4, Y5
	VPMULHUW Y6, Y5, Y5

	// Pack two 16×uint16 → 32×uint8
	VPACKUSWB Y5, Y3, Y7
	// VPACKUSWB interleaves lanes: need to fix with VPERMQ
	VPERMQ $0xD8, Y7, Y7               // de-interleave: 0,2,1,3 → 0,1,2,3
	VMOVDQU Y7, (BX)

	ADDQ $64, AX                        // 32 × int16 = 64 bytes
	ADDQ $32, BX                        // 32 × uint8 = 32 bytes
	DECQ CX
	JNZ screlu_pack_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueCReLU256(src *int16, dst *int16)
// Clamps 256 int16 values to [0, 127].
// 256 elements / 16 per YMM = 16 iterations.
// ============================================================================
TEXT ·nnueCReLU256(SB), NOSPLIT, $0-16
	MOVQ src+0(FP), AX
	MOVQ dst+8(FP), BX
	VPXOR Y0, Y0, Y0                    // Y0 = zero (floor)
	VMOVDQU nnue_clamp_127<>(SB), Y1    // Y1 = 127 (ceiling)
	MOVQ $16, CX                        // 16 iterations

crelu_loop:
	VMOVDQU (AX), Y2
	VPMAXSW Y0, Y2, Y2                  // max(0, x)
	VPMINSW Y1, Y2, Y2                  // min(127, x)
	VMOVDQU Y2, (BX)
	ADDQ $32, AX
	ADDQ $32, BX
	DECQ CX
	JNZ crelu_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueCReLU256to8(src *int16, dst *byte)
// Clamps 256 int16 values to [0, 127] and packs into 256 uint8 values.
// Processes 32 int16 → 32 uint8 per iteration (8 iterations).
// Uses VPACKUSWB + VPERMQ to pack and fix lane ordering.
// ============================================================================
TEXT ·nnueCReLU256to8(SB), NOSPLIT, $0-16
	MOVQ src+0(FP), AX
	MOVQ dst+8(FP), BX
	VPXOR Y0, Y0, Y0                    // Y0 = zero (floor)
	VMOVDQU nnue_clamp_127<>(SB), Y1    // Y1 = 127 (ceiling)
	MOVQ $8, CX                         // 8 iterations (32 int16 → 32 uint8 each)

crelu8_loop:
	VMOVDQU (AX), Y2                    // load 16 int16
	VMOVDQU 32(AX), Y3                  // load next 16 int16
	VPMAXSW Y0, Y2, Y2                  // max(0, x)
	VPMINSW Y1, Y2, Y2                  // min(127, x)
	VPMAXSW Y0, Y3, Y3
	VPMINSW Y1, Y3, Y3
	VPACKUSWB Y3, Y2, Y4                // pack 2×16 int16 → 32 uint8 (lane-interleaved)
	VPERMQ $0xD8, Y4, Y4                // fix lane ordering: [0,2,1,3]
	VMOVDQU Y4, (BX)                    // store 32 uint8
	ADDQ $64, AX                        // 32 int16 = 64 bytes
	ADDQ $32, BX                        // 32 uint8 = 32 bytes
	DECQ CX
	JNZ crelu8_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueMatMul32x512_i8(input *byte, weightsT *int8, biases *int32, output *int32)
//
// Computes: output[j] = biases[j] + sum_i(input[i] * weightsT[j][i])
// for j = 0..31, i = 0..511.
//
// input is uint8 [0,127], weightsT is int8 [-128,127].
// weightsT layout: [32][512] int8, row-major. Each row = 512 bytes.
// Uses VPMADDUBSW (u8×i8→i16) + VPMADDWD (i16→i32) for 2× throughput.
//
// Processes 4 output neurons at a time (8 groups of 4).
// Register allocation:
//   AX  = input base pointer (constant)
//   BX  = weightsT[j] base (advances by 4 rows per outer iteration)
//   DX  = biases pointer (advances by 16 per outer iteration)
//   SI  = output pointer (advances by 16 per outer iteration)
//   R8  = row stride (512 bytes, constant)
//   R9  = weightsT[j+1] base
//   R10 = weightsT[j+2] base
//   R11 = byte offset for inner loop (0..512 by 32)
//   R12 = weightsT[j+3] base
//   R13 = outer loop counter (8..0)
//   Y8-Y11 = accumulators for 4 outputs (int32)
//   Y12 = constant: 16 × int16(1) for VPMADDWD widening
//   Y0  = input load (32 uint8)
//   Y1-Y4 = scratch for VPMADDUBSW/VPMADDWD results
//   X5  = scratch for horizontal reduction
// ============================================================================
TEXT ·nnueMatMul32x512_i8(SB), NOSPLIT, $0-32
	MOVQ input+0(FP), AX
	MOVQ weightsT+8(FP), BX
	MOVQ biases+16(FP), DX
	MOVQ output+24(FP), SI

	MOVQ $512, R8                       // stride = 512 bytes per row
	MOVQ $8, R13                        // 8 groups of 4 outputs
	VMOVDQU nnue_ones_16<>(SB), Y12     // Y12 = 16 × int16(1)

i8_outer:
	// Base pointers for 4 weight rows
	LEAQ (BX)(R8*1), R9                // weightsT[j+1]
	LEAQ (BX)(R8*2), R10               // weightsT[j+2]
	LEAQ (R9)(R8*2), R12               // weightsT[j+3] = BX + 3*stride

	// Zero accumulators
	VPXOR Y8, Y8, Y8
	VPXOR Y9, Y9, Y9
	VPXOR Y10, Y10, Y10
	VPXOR Y11, Y11, Y11

	// Inner loop: 512 elements in groups of 32 (16 iterations)
	XORQ R11, R11

i8_inner:
	VMOVDQU (AX)(R11*1), Y0            // load 32 uint8 activations

	VPMADDUBSW (BX)(R11*1), Y0, Y1     // 32 (u8×i8) → 16 int16
	VPMADDWD Y12, Y1, Y1               // 16 int16 → 8 int32
	VPADDD Y1, Y8, Y8                  // accumulate

	VPMADDUBSW (R9)(R11*1), Y0, Y2     // weights[j+1]
	VPMADDWD Y12, Y2, Y2
	VPADDD Y2, Y9, Y9

	VPMADDUBSW (R10)(R11*1), Y0, Y3    // weights[j+2]
	VPMADDWD Y12, Y3, Y3
	VPADDD Y3, Y10, Y10

	VPMADDUBSW (R12)(R11*1), Y0, Y4    // weights[j+3]
	VPMADDWD Y12, Y4, Y4
	VPADDD Y4, Y11, Y11

	ADDQ $32, R11
	CMPQ R11, $512
	JNE i8_inner

	// --- Horizontal reduce Y8 → output[j] ---
	VEXTRACTI128 $1, Y8, X5
	VPADDD X5, X8, X8
	VPSHUFD $0x4E, X8, X5
	VPADDD X5, X8, X8
	VPSHUFD $0xB1, X8, X5
	VPADDD X5, X8, X8
	VMOVD X8, R11
	ADDL (DX), R11
	MOVL R11, (SI)

	// --- Horizontal reduce Y9 → output[j+1] ---
	VEXTRACTI128 $1, Y9, X5
	VPADDD X5, X9, X9
	VPSHUFD $0x4E, X9, X5
	VPADDD X5, X9, X9
	VPSHUFD $0xB1, X9, X5
	VPADDD X5, X9, X9
	VMOVD X9, R11
	ADDL 4(DX), R11
	MOVL R11, 4(SI)

	// --- Horizontal reduce Y10 → output[j+2] ---
	VEXTRACTI128 $1, Y10, X5
	VPADDD X5, X10, X10
	VPSHUFD $0x4E, X10, X5
	VPADDD X5, X10, X10
	VPSHUFD $0xB1, X10, X5
	VPADDD X5, X10, X10
	VMOVD X10, R11
	ADDL 8(DX), R11
	MOVL R11, 8(SI)

	// --- Horizontal reduce Y11 → output[j+3] ---
	VEXTRACTI128 $1, Y11, X5
	VPADDD X5, X11, X11
	VPSHUFD $0x4E, X11, X5
	VPADDD X5, X11, X11
	VPSHUFD $0xB1, X11, X5
	VPADDD X5, X11, X11
	VMOVD X11, R11
	ADDL 12(DX), R11
	MOVL R11, 12(SI)

	// Advance outer pointers
	ADDQ $2048, BX                      // weightsT += 4 rows (4 * 512)
	ADDQ $16, DX                        // biases += 4 int32
	ADDQ $16, SI                        // output += 4 int32
	DECQ R13
	JNZ i8_outer

	VZEROUPPER
	RET

// ============================================================================
// nnueMatMul32x512(input *int16, weightsT *int16, biases *int32, output *int32)
//
// Computes: output[j] = biases[j] + sum_i(input[i] * weightsT[j][i])
// for j = 0..31, i = 0..511.
//
// weightsT layout: [32][512] int16, row-major. Each row = 1024 bytes.
// Uses VPMADDWD: multiplies 16 int16 pairs, sums adjacent → 8 int32.
//
// Processes 4 output neurons at a time (8 groups of 4).
// Register allocation:
//   AX  = input base pointer (constant)
//   BX  = weightsT[j] base (advances by 4 rows per outer iteration)
//   DX  = biases pointer (advances by 16 per outer iteration)
//   SI  = output pointer (advances by 16 per outer iteration)
//   R8  = row stride (1024 bytes, constant)
//   R9  = weightsT[j+1] base (per outer iteration)
//   R10 = weightsT[j+2] base (per outer iteration)
//   R11 = byte offset for inner loop (0..1024 by 32)
//   R12 = weightsT[j+3] base (per outer iteration)
//   R13 = outer loop counter (8..0)
//   Y8-Y11 = accumulators for 4 outputs
//   Y0  = input load
//   Y1-Y4 = VPMADDWD results
//   X5  = scratch for horizontal reduction
// ============================================================================
TEXT ·nnueMatMul32x512(SB), NOSPLIT, $0-32
	MOVQ input+0(FP), AX
	MOVQ weightsT+8(FP), BX
	MOVQ biases+16(FP), DX
	MOVQ output+24(FP), SI

	MOVQ $1024, R8              // stride = 512 * 2 bytes per row
	MOVQ $8, R13                // 8 groups of 4 outputs

outer:
	// Base pointers for 4 weight rows
	LEAQ (BX)(R8*1), R9        // weightsT[j+1]
	LEAQ (BX)(R8*2), R10       // weightsT[j+2]
	LEAQ (R9)(R8*2), R12       // weightsT[j+3] = BX + 3*stride

	// Zero accumulators
	VPXOR Y8, Y8, Y8
	VPXOR Y9, Y9, Y9
	VPXOR Y10, Y10, Y10
	VPXOR Y11, Y11, Y11

	// Inner loop: 512 elements in groups of 16 (32 iterations)
	XORQ R11, R11

inner:
	VMOVDQU (AX)(R11*1), Y0          // load 16 int16 activations

	VMOVDQU (BX)(R11*1), Y1          // load weights[j]
	VPMADDWD Y0, Y1, Y1              // madd → 8 int32
	VPADDD Y1, Y8, Y8                // accumulate

	VMOVDQU (R9)(R11*1), Y2          // load weights[j+1]
	VPMADDWD Y0, Y2, Y2
	VPADDD Y2, Y9, Y9

	VMOVDQU (R10)(R11*1), Y3         // load weights[j+2]
	VPMADDWD Y0, Y3, Y3
	VPADDD Y3, Y10, Y10

	VMOVDQU (R12)(R11*1), Y4         // load weights[j+3]
	VPMADDWD Y0, Y4, Y4
	VPADDD Y4, Y11, Y11

	ADDQ $32, R11
	CMPQ R11, $1024
	JNE inner

	// --- Horizontal reduce Y8 → output[j] ---
	VEXTRACTI128 $1, Y8, X5          // X5 = high 128 bits
	VPADDD X5, X8, X8                // X8 += high
	VPSHUFD $0x4E, X8, X5            // swap 64-bit halves
	VPADDD X5, X8, X8                // add
	VPSHUFD $0xB1, X8, X5            // swap 32-bit pairs
	VPADDD X5, X8, X8                // X8[0] = total
	VMOVD X8, R11
	ADDL (DX), R11                    // add bias[j]
	MOVL R11, (SI)                    // store output[j]

	// --- Horizontal reduce Y9 → output[j+1] ---
	VEXTRACTI128 $1, Y9, X5
	VPADDD X5, X9, X9
	VPSHUFD $0x4E, X9, X5
	VPADDD X5, X9, X9
	VPSHUFD $0xB1, X9, X5
	VPADDD X5, X9, X9
	VMOVD X9, R11
	ADDL 4(DX), R11
	MOVL R11, 4(SI)

	// --- Horizontal reduce Y10 → output[j+2] ---
	VEXTRACTI128 $1, Y10, X5
	VPADDD X5, X10, X10
	VPSHUFD $0x4E, X10, X5
	VPADDD X5, X10, X10
	VPSHUFD $0xB1, X10, X5
	VPADDD X5, X10, X10
	VMOVD X10, R11
	ADDL 8(DX), R11
	MOVL R11, 8(SI)

	// --- Horizontal reduce Y11 → output[j+3] ---
	VEXTRACTI128 $1, Y11, X5
	VPADDD X5, X11, X11
	VPSHUFD $0x4E, X11, X5
	VPADDD X5, X11, X11
	VPSHUFD $0xB1, X11, X5
	VPADDD X5, X11, X11
	VMOVD X11, R11
	ADDL 12(DX), R11
	MOVL R11, 12(SI)

	// Advance outer pointers
	ADDQ $4096, BX                    // weightsT += 4 rows (4 * 1024)
	ADDQ $16, DX                      // biases += 4 int32
	ADDQ $16, SI                      // output += 4 int32
	DECQ R13
	JNZ outer

	VZEROUPPER
	RET

// ============================================================================
// nnueAccSubAdd256(acc *int16, oldW *int16, newW *int16)
// Computes: acc[i] += newW[i] - oldW[i] for i = 0..255
// 256 elements / 16 per YMM = 16 iterations.
// ============================================================================
TEXT ·nnueAccSubAdd256(SB), NOSPLIT, $0-24
	MOVQ acc+0(FP), AX
	MOVQ oldW+8(FP), BX
	MOVQ newW+16(FP), CX
	MOVQ $16, DX

subadd_loop:
	VMOVDQU (CX), Y0                 // new weights
	VPSUBW (BX), Y0, Y0              // Y0 = new - old (memory operand)
	VPADDW (AX), Y0, Y0              // Y0 = acc + (new - old)
	VMOVDQU Y0, (AX)                 // store back
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, CX
	DECQ DX
	JNZ subadd_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueAccSubSubAdd256(acc *int16, oldW *int16, newW *int16, capW *int16)
// Computes: acc[i] += newW[i] - oldW[i] - capW[i] for i = 0..255
// ============================================================================
TEXT ·nnueAccSubSubAdd256(SB), NOSPLIT, $0-32
	MOVQ acc+0(FP), AX
	MOVQ oldW+8(FP), BX
	MOVQ newW+16(FP), CX
	MOVQ capW+24(FP), DX
	MOVQ $16, SI

subsubadd_loop:
	VMOVDQU (CX), Y0                 // new weights
	VPSUBW (BX), Y0, Y0              // Y0 = new - old
	VPSUBW (DX), Y0, Y0              // Y0 = new - old - cap
	VPADDW (AX), Y0, Y0              // Y0 = acc + delta
	VMOVDQU Y0, (AX)                 // store back
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, CX
	ADDQ $32, DX
	DECQ SI
	JNZ subsubadd_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueAccAdd256(acc *int16, weights *int16)
// Computes: acc[i] += weights[i] for i = 0..255
// ============================================================================
TEXT ·nnueAccAdd256(SB), NOSPLIT, $0-16
	MOVQ acc+0(FP), AX
	MOVQ weights+8(FP), BX
	MOVQ $16, CX

accadd_loop:
	VMOVDQU (AX), Y0                 // load accumulator
	VPADDW (BX), Y0, Y0              // add weights (memory operand)
	VMOVDQU Y0, (AX)                 // store back
	ADDQ $32, AX
	ADDQ $32, BX
	DECQ CX
	JNZ accadd_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueAccSub256(acc *int16, weights *int16)
// Computes: acc[i] -= weights[i] for i = 0..255
// ============================================================================
TEXT ·nnueAccSub256(SB), NOSPLIT, $0-16
	MOVQ acc+0(FP), AX
	MOVQ weights+8(FP), BX
	MOVQ $16, CX

accsub_loop:
	VMOVDQU (AX), Y0                 // load accumulator
	VPSUBW (BX), Y0, Y0              // subtract weights (memory operand)
	VMOVDQU Y0, (AX)                 // store back
	ADDQ $32, AX
	ADDQ $32, BX
	DECQ CX
	JNZ accsub_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueAccAddN(acc *int16, weights *int16, count int)
// Computes: acc[i] += weights[i] for i = 0..count-1
// count must be a multiple of 16. Width-generic for v5 dynamic hidden sizes.
// ============================================================================
TEXT ·nnueAccAddN(SB), NOSPLIT, $0-24
	MOVQ acc+0(FP), AX
	MOVQ weights+8(FP), BX
	MOVQ count+16(FP), CX
	SHRQ $4, CX                      // count / 16 = number of YMM iterations

accaddn_loop:
	VMOVDQU (AX), Y0
	VPADDW (BX), Y0, Y0
	VMOVDQU Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, BX
	DECQ CX
	JNZ accaddn_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueAccSubN(acc *int16, weights *int16, count int)
// Computes: acc[i] -= weights[i] for i = 0..count-1
// count must be a multiple of 16. Width-generic for v5 dynamic hidden sizes.
// ============================================================================
TEXT ·nnueAccSubN(SB), NOSPLIT, $0-24
	MOVQ acc+0(FP), AX
	MOVQ weights+8(FP), BX
	MOVQ count+16(FP), CX
	SHRQ $4, CX

accsubn_loop:
	VMOVDQU (AX), Y0
	VPSUBW (BX), Y0, Y0
	VMOVDQU Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, BX
	DECQ CX
	JNZ accsubn_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueAccSubAddN(acc *int16, oldW *int16, newW *int16, count int)
// Computes: acc[i] += newW[i] - oldW[i] for i = 0..count-1
// Fused sub+add for moved pieces. count must be a multiple of 16.
// ============================================================================
TEXT ·nnueAccSubAddN(SB), NOSPLIT, $0-32
	MOVQ acc+0(FP), AX
	MOVQ oldW+8(FP), BX
	MOVQ newW+16(FP), CX
	MOVQ count+24(FP), DX
	SHRQ $4, DX

accsubaddn_loop:
	VMOVDQU (AX), Y0                 // load acc
	VPSUBW (BX), Y0, Y0              // acc -= oldW
	VPADDW (CX), Y0, Y0              // acc += newW
	VMOVDQU Y0, (AX)                 // store
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, CX
	DECQ DX
	JNZ accsubaddn_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueAccCopySubAddN(dst *int16, src *int16, oldW *int16, newW *int16, count int)
// Computes: dst[i] = src[i] + newW[i] - oldW[i] for i = 0..count-1
// count must be a multiple of 16. Width-generic.
// ============================================================================
TEXT ·nnueAccCopySubAddN(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), AX
	MOVQ src+8(FP), BX
	MOVQ oldW+16(FP), CX
	MOVQ newW+24(FP), DX
	MOVQ count+32(FP), SI
	SHRQ $4, SI

copysubaddn_loop:
	VMOVDQU (DX), Y0
	VPSUBW (CX), Y0, Y0
	VPADDW (BX), Y0, Y0
	VMOVDQU Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, CX
	ADDQ $32, DX
	DECQ SI
	JNZ copysubaddn_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueAccCopySubSubAddN(dst *int16, src *int16, oldW *int16, newW *int16, capW *int16, count int)
// Computes: dst[i] = src[i] + newW[i] - oldW[i] - capW[i] for i = 0..count-1
// count must be a multiple of 16. Width-generic.
// ============================================================================
TEXT ·nnueAccCopySubSubAddN(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+8(FP), BX
	MOVQ oldW+16(FP), CX
	MOVQ newW+24(FP), DX
	MOVQ capW+32(FP), SI
	MOVQ count+40(FP), DI
	SHRQ $4, DI

copysubsubaddn_loop:
	VMOVDQU (DX), Y0
	VPSUBW (CX), Y0, Y0
	VPSUBW (SI), Y0, Y0
	VPADDW (BX), Y0, Y0
	VMOVDQU Y0, (AX)
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, CX
	ADDQ $32, DX
	ADDQ $32, SI
	DECQ DI
	JNZ copysubsubaddn_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueAccCopySubAdd256(dst *int16, src *int16, oldW *int16, newW *int16)
// Computes: dst[i] = src[i] + newW[i] - oldW[i] for i = 0..255
// Fused copy+update: reads from src (parent), writes to dst (child).
// ============================================================================
TEXT ·nnueAccCopySubAdd256(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), AX
	MOVQ src+8(FP), BX
	MOVQ oldW+16(FP), CX
	MOVQ newW+24(FP), DX
	MOVQ $16, SI

copysubadd_loop:
	VMOVDQU (DX), Y0                 // new weights
	VPSUBW (CX), Y0, Y0              // Y0 = new - old
	VPADDW (BX), Y0, Y0              // Y0 = src + (new - old)
	VMOVDQU Y0, (AX)                 // store to dst
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, CX
	ADDQ $32, DX
	DECQ SI
	JNZ copysubadd_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueMatMul32x32ReLU(input *int32, weightsT *int16, biases *int32, output *int32)
//
// Computes: output[k] = biases[k] + sum_j(max(0, input[j] >> 6) * weightsT[k*32+j])
// for k = 0..31, j = 0..31.
//
// weightsT layout: [32][32] int16, row-major (each row = 64 bytes).
// Applies ReLU(x >> 6) to input once, then reuses across all outputs.
// Uses VPMOVSXWD to sign-extend int16 weights to int32, VPMULLD for multiply.
//
// Register allocation:
//   AX  = input pointer (precompute only)
//   BX  = weightsT pointer (advances by 64 per output)
//   CX  = biases pointer (advances by 4 per output)
//   DX  = output pointer (advances by 4 per output)
//   R8  = loop counter (32..0)
//   R9  = scratch for scalar result
//   Y0-Y3 = activated values (constant after precompute)
//   Y4  = zero register
//   Y5  = scratch for weights/products
//   Y8  = accumulator per output
//   X5  = scratch for horizontal reduction
// ============================================================================
TEXT ·nnueMatMul32x32ReLU(SB), NOSPLIT, $0-32
	MOVQ input+0(FP), AX
	MOVQ weightsT+8(FP), BX
	MOVQ biases+16(FP), CX
	MOVQ output+24(FP), DX

	// Precompute activated = ReLU(input >> 6)
	VPXOR Y4, Y4, Y4
	VMOVDQU (AX), Y0
	VPSRAD $6, Y0, Y0
	VPMAXSD Y4, Y0, Y0
	VMOVDQU 32(AX), Y1
	VPSRAD $6, Y1, Y1
	VPMAXSD Y4, Y1, Y1
	VMOVDQU 64(AX), Y2
	VPSRAD $6, Y2, Y2
	VPMAXSD Y4, Y2, Y2
	VMOVDQU 96(AX), Y3
	VPSRAD $6, Y3, Y3
	VPMAXSD Y4, Y3, Y3

	MOVQ $32, R8

matmul32x32_loop:
	VPXOR Y8, Y8, Y8              // zero accumulator

	// Group 0: activated[0..7] × weights[0..7]
	VMOVDQU (BX), X5              // load 8 int16 (128 bits)
	VPMOVSXWD X5, Y5              // sign-extend to 8 int32
	VPMULLD Y0, Y5, Y5
	VPADDD Y5, Y8, Y8

	// Group 1: activated[8..15] × weights[8..15]
	VMOVDQU 16(BX), X5
	VPMOVSXWD X5, Y5
	VPMULLD Y1, Y5, Y5
	VPADDD Y5, Y8, Y8

	// Group 2: activated[16..23] × weights[16..23]
	VMOVDQU 32(BX), X5
	VPMOVSXWD X5, Y5
	VPMULLD Y2, Y5, Y5
	VPADDD Y5, Y8, Y8

	// Group 3: activated[24..31] × weights[24..31]
	VMOVDQU 48(BX), X5
	VPMOVSXWD X5, Y5
	VPMULLD Y3, Y5, Y5
	VPADDD Y5, Y8, Y8

	// Horizontal reduce Y8 → R9
	VEXTRACTI128 $1, Y8, X5
	VPADDD X5, X8, X8
	VPSHUFD $0x4E, X8, X5
	VPADDD X5, X8, X8
	VPSHUFD $0xB1, X8, X5
	VPADDD X5, X8, X8
	VMOVD X8, R9

	// Add bias and store output
	ADDL (CX), R9
	MOVL R9, (DX)

	ADDQ $64, BX                  // next weight row (32 int16)
	ADDQ $4, CX                   // next bias
	ADDQ $4, DX                   // next output
	DECQ R8
	JNZ matmul32x32_loop

	VZEROUPPER
	RET

// ============================================================================
// nnueDotReLU32(input *int32, weights *int16) int32
//
// Returns: sum_k(max(0, input[k] >> 6) * int32(weights[k])) for k = 0..31
// Used for the output layer (32 → 1 with ReLU activation).
// ============================================================================
TEXT ·nnueDotReLU32(SB), NOSPLIT, $0-24
	MOVQ input+0(FP), AX
	MOVQ weights+8(FP), BX

	VPXOR Y4, Y4, Y4              // zero for ReLU
	VPXOR Y8, Y8, Y8              // accumulator

	// Group 0: input[0..7]
	VMOVDQU (AX), Y0
	VPSRAD $6, Y0, Y0
	VPMAXSD Y4, Y0, Y0
	VMOVDQU (BX), X5
	VPMOVSXWD X5, Y5
	VPMULLD Y0, Y5, Y5
	VPADDD Y5, Y8, Y8

	// Group 1: input[8..15]
	VMOVDQU 32(AX), Y0
	VPSRAD $6, Y0, Y0
	VPMAXSD Y4, Y0, Y0
	VMOVDQU 16(BX), X5
	VPMOVSXWD X5, Y5
	VPMULLD Y0, Y5, Y5
	VPADDD Y5, Y8, Y8

	// Group 2: input[16..23]
	VMOVDQU 64(AX), Y0
	VPSRAD $6, Y0, Y0
	VPMAXSD Y4, Y0, Y0
	VMOVDQU 32(BX), X5
	VPMOVSXWD X5, Y5
	VPMULLD Y0, Y5, Y5
	VPADDD Y5, Y8, Y8

	// Group 3: input[24..31]
	VMOVDQU 96(AX), Y0
	VPSRAD $6, Y0, Y0
	VPMAXSD Y4, Y0, Y0
	VMOVDQU 48(BX), X5
	VPMOVSXWD X5, Y5
	VPMULLD Y0, Y5, Y5
	VPADDD Y5, Y8, Y8

	// Horizontal reduce Y8
	VEXTRACTI128 $1, Y8, X5
	VPADDD X5, X8, X8
	VPSHUFD $0x4E, X8, X5
	VPADDD X5, X8, X8
	VPSHUFD $0xB1, X8, X5
	VPADDD X5, X8, X8
	VMOVD X8, AX
	MOVL AX, ret+16(FP)

	VZEROUPPER
	RET

// ============================================================================
// nnueAccCopySubSubAdd256(dst *int16, src *int16, oldW *int16, newW *int16, capW *int16)
// Computes: dst[i] = src[i] + newW[i] - oldW[i] - capW[i] for i = 0..255
// Fused copy+update for captures: reads from src (parent), writes to dst (child).
// ============================================================================
TEXT ·nnueAccCopySubSubAdd256(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), AX
	MOVQ src+8(FP), BX
	MOVQ oldW+16(FP), CX
	MOVQ newW+24(FP), DX
	MOVQ capW+32(FP), SI
	MOVQ $16, DI

copysubsubadd_loop:
	VMOVDQU (DX), Y0                 // new weights
	VPSUBW (CX), Y0, Y0              // Y0 = new - old
	VPSUBW (SI), Y0, Y0              // Y0 = new - old - cap
	VPADDW (BX), Y0, Y0              // Y0 = src + delta
	VMOVDQU Y0, (AX)                 // store to dst
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, CX
	ADDQ $32, DX
	ADDQ $32, SI
	DECQ DI
	JNZ copysubsubadd_loop

	VZEROUPPER
	RET

// ============================================================================
// 16 x int16(255) for v5 ClippedReLU max clamp
// ============================================================================
DATA nnue_clamp_255<>+0(SB)/8, $0x00ff00ff00ff00ff
DATA nnue_clamp_255<>+8(SB)/8, $0x00ff00ff00ff00ff
DATA nnue_clamp_255<>+16(SB)/8, $0x00ff00ff00ff00ff
DATA nnue_clamp_255<>+24(SB)/8, $0x00ff00ff00ff00ff
GLOBL nnue_clamp_255<>(SB), NOPTR+RODATA, $32

// ============================================================================
// nnueV5CReLUDot1024(acc *int16, weights *int16) int32
// Computes: sum = sum_i( clamp(acc[i], 0, 255) * weights[i] ) for i=0..1023
// Uses VPMAXSW/VPMINSW for clamping, VPMADDWD for multiply-accumulate.
// Returns the 32-bit sum.
// 1024 elements / 16 per YMM = 64 iterations.
// ============================================================================
TEXT ·nnueV5CReLUDot1024(SB), NOSPLIT, $0-24
	MOVQ acc+0(FP), AX
	MOVQ weights+8(FP), BX
	VPXOR Y0, Y0, Y0                    // Y0 = zero (floor)
	VMOVDQU nnue_clamp_255<>(SB), Y1    // Y1 = 255 (ceiling)
	VPXOR Y5, Y5, Y5                    // Y5 = accumulator (int32 sum)
	MOVQ $64, CX                        // 64 iterations

v5dot_loop:
	VMOVDQU (AX), Y2                    // load 16 acc values
	VPMAXSW Y0, Y2, Y2                  // max(0, x) 
	VPMINSW Y1, Y2, Y2                  // min(255, x) = CReLU
	VMOVDQU (BX), Y3                    // load 16 weights
	VPMADDWD Y2, Y3, Y4                 // multiply pairs, accumulate to 8 int32
	VPADDD Y4, Y5, Y5                   // add to running sum
	ADDQ $32, AX
	ADDQ $32, BX
	DECQ CX
	JNZ v5dot_loop

	// Horizontal sum of Y5 (8 x int32 -> 1 int32)
	VEXTRACTI128 $1, Y5, X6             // high 128 bits
	VPADDD X6, X5, X5                   // add high + low (4 x int32)
	VPSHUFD $0x4E, X5, X6               // swap high/low 64-bit halves
	VPADDD X6, X5, X5                   // 2 x int32
	VPSHUFD $0x01, X5, X6               // swap 32-bit halves
	VPADDD X6, X5, X5                   // 1 x int32
	VMOVD X5, AX                        // move to register
	MOVL AX, ret+16(FP)                 // return value

	VZEROUPPER
	RET

// ============================================================================
// nnueV5CReLUDotN(acc *int16, weights *int16, count int) int32
// Generic width CReLU dot product.
// Computes: sum = sum_i( clamp(acc[i], 0, 255) * weights[i] ) for i=0..count-1
// count must be a multiple of 16.
// ============================================================================
TEXT ·nnueV5CReLUDotN(SB), NOSPLIT, $0-28
	MOVQ acc+0(FP), AX
	MOVQ weights+8(FP), BX
	MOVQ count+16(FP), CX
	SHRQ $5, CX                         // count / 32 = iterations (2x unrolled)
	VPXOR Y0, Y0, Y0                    // Y0 = zero (floor)
	VMOVDQU nnue_clamp_255<>(SB), Y1    // Y1 = 255 (ceiling)
	VPXOR Y5, Y5, Y5                    // Y5 = accumulator 1
	VPXOR Y7, Y7, Y7                    // Y7 = accumulator 2

v5dotn_loop:
	// First 16 elements
	VMOVDQU (AX), Y2
	VPMAXSW Y0, Y2, Y2
	VPMINSW Y1, Y2, Y2
	VMOVDQU (BX), Y3
	VPMADDWD Y2, Y3, Y4
	VPADDD Y4, Y5, Y5
	// Second 16 elements
	VMOVDQU 32(AX), Y2
	VPMAXSW Y0, Y2, Y2
	VPMINSW Y1, Y2, Y2
	VMOVDQU 32(BX), Y3
	VPMADDWD Y2, Y3, Y4
	VPADDD Y4, Y7, Y7
	ADDQ $64, AX
	ADDQ $64, BX
	DECQ CX
	JNZ v5dotn_loop

	VPADDD Y7, Y5, Y5                   // merge accumulators

	// Horizontal sum of Y5
	VEXTRACTI128 $1, Y5, X6
	VPADDD X6, X5, X5
	VPSHUFD $0x4E, X5, X6
	VPADDD X6, X5, X5
	VPSHUFD $0x01, X5, X6
	VPADDD X6, X5, X5
	VMOVD X5, AX
	MOVL AX, ret+24(FP)

	VZEROUPPER
	RET

// ============================================================================
// nnueV5SCReLUDotN(acc *int16, weights *int16, count int) int64
// Optimized SCReLU dot product using byte decomposition + VPMADDWD.
// Computes: sum = sum_i( clamp(acc[i], 0, 255)^2 * weights[i] ) for i=0..count-1
// Returns int64 (caller divides by QA=255).
//
// Key optimization: decompose x^2 = byte0 + byte1 * 256 where:
//   byte0 = x^2 & 0xFF (max 255, safe for signed VPMADDWD pairwise)
//   byte1 = x^2 >> 8   (max 254, safe for signed VPMADDWD pairwise)
// Both terms accumulate safely in int32 for up to 256 elements per batch:
//   max pair = 2 * 255 * 32767 = 16.7M, × 64 pairs (256 elements) = 1.07B < INT32_MAX
// Drains to int64 every 256 elements (8 iterations of 32).
// 2x unrolled inner loop. count must be multiple of 256.
//
// Register allocation:
//   Y0  = zero (floor), Y1 = 0x00FF (clamp ceiling = byte0 mask)
//   Y3-Y7 = temporaries
//   Y8-Y9 = int32 byte0 accumulators (2x unroll)
//   Y10-Y11 = int32 byte1 accumulators (2x unroll)
//   Y12-Y13 = int64 drain accumulators (byte0)
//   Y14-Y15 = int64 drain accumulators (byte1)
// ============================================================================
TEXT ·nnueV5SCReLUDotN(SB), NOSPLIT, $0-32
	MOVQ acc+0(FP), AX
	MOVQ weights+8(FP), BX
	MOVQ count+16(FP), CX

	// Outer loop = count / 256 (drain cycles), inner = 8 iterations of 32 elements
	SHRQ $8, CX                         // CX = count / 256 = drain cycles

	VPXOR Y0, Y0, Y0                    // Y0 = zero (floor)
	VMOVDQU nnue_clamp_255<>(SB), Y1    // Y1 = 0x00FF (ceiling + byte0 mask)

	// Int64 drain accumulators
	VPXOR Y12, Y12, Y12
	VPXOR Y13, Y13, Y13
	VPXOR Y14, Y14, Y14
	VPXOR Y15, Y15, Y15

v5screlu_drain_loop:
	// Reset int32 batch accumulators
	VPXOR Y8, Y8, Y8
	VPXOR Y9, Y9, Y9
	VPXOR Y10, Y10, Y10
	VPXOR Y11, Y11, Y11
	MOVQ $8, DX                          // 8 inner iterations × 32 elements = 256

v5screlu_inner:
	// ======== First 16 elements ========
	VMOVDQU (AX), Y3                    // acc[0..15]
	VPMAXSW Y0, Y3, Y3                  // clamp lower
	VPMINSW Y1, Y3, Y3                  // clamp upper
	VPMULLW Y3, Y3, Y4                  // x_sq = x^2 (uint16)
	VMOVDQU (BX), Y5                    // weights[0..15]
	VPAND Y1, Y4, Y6                    // byte0 = x_sq & 0xFF
	VPSRLW $8, Y4, Y7                   // byte1 = x_sq >> 8
	VPMADDWD Y5, Y6, Y6                 // byte0 pairs (int32)
	VPMADDWD Y5, Y7, Y7                 // byte1 pairs (int32)
	VPADDD Y6, Y8, Y8                   // byte0 acc
	VPADDD Y7, Y10, Y10                 // byte1 acc

	// ======== Second 16 elements ========
	VMOVDQU 32(AX), Y3
	VPMAXSW Y0, Y3, Y3
	VPMINSW Y1, Y3, Y3
	VPMULLW Y3, Y3, Y4
	VMOVDQU 32(BX), Y5
	VPAND Y1, Y4, Y6
	VPSRLW $8, Y4, Y7
	VPMADDWD Y5, Y6, Y6
	VPMADDWD Y5, Y7, Y7
	VPADDD Y6, Y9, Y9
	VPADDD Y7, Y11, Y11

	ADDQ $64, AX
	ADDQ $64, BX
	DECQ DX
	JNZ v5screlu_inner

	// === Drain int32 → int64 ===

	// Merge byte0 accumulators and drain
	VPADDD Y9, Y8, Y8                   // Y8 = 8 x int32 (byte0 batch total)
	VPMOVSXDQ X8, Y2
	VPADDQ Y2, Y12, Y12
	VEXTRACTI128 $1, Y8, X2
	VPMOVSXDQ X2, Y2
	VPADDQ Y2, Y13, Y13

	// Merge byte1 accumulators, shift left 8, and drain
	VPADDD Y11, Y10, Y10                // Y10 = 8 x int32 (byte1 batch total)
	VPMOVSXDQ X10, Y2
	VPSLLQ $8, Y2, Y2                   // × 256
	VPADDQ Y2, Y14, Y14
	VEXTRACTI128 $1, Y10, X2
	VPMOVSXDQ X2, Y2
	VPSLLQ $8, Y2, Y2
	VPADDQ Y2, Y15, Y15

	DECQ CX
	JNZ v5screlu_drain_loop

	// === Epilogue: combine all int64 accumulators ===
	VPADDQ Y13, Y12, Y12                // byte0: 4 x int64
	VPADDQ Y15, Y14, Y14                // byte1: 4 x int64
	VPADDQ Y14, Y12, Y12                // combined: 4 x int64

	// Horizontal sum of Y12 (4 x int64 → 1 int64)
	VEXTRACTI128 $1, Y12, X9
	VPADDQ X9, X12, X12                 // 2 x int64
	VPSHUFD $0x4E, X12, X9              // swap 64-bit halves
	VPADDQ X9, X12, X12                 // 1 x int64
	VMOVQ X12, AX
	MOVQ AX, ret+24(FP)

	VZEROUPPER
	RET

// ============================================================================
// nnueV5PairwiseDotN(accFirst *int16, accSecond *int16, weights *int16, count int) int64
//
// Optimized pairwise dot product using byte decomposition + VPMADDWD.
// Computes: sum = sum_i( clamp(a[i],0,255) * clamp(b[i],0,255) * weights[i] )
// for i=0..count-1, where a=accFirst, b=accSecond (second half of accumulator).
//
// Key optimization: a*b ∈ [0,65025] = byte0 + byte1*256 (same as SCReLU x²).
//   byte0 = (a*b) & 0xFF, byte1 = (a*b) >> 8
// Both accumulate safely in int32 for 128-element batches.
// count must be a multiple of 128. Returns int64 (caller divides by QA=255).
//
// Register allocation:
//   AX = accFirst, BX = accSecond, DX = weights
//   CX = drain counter (count/128), SI = inner counter
//   Y0 = zero, Y1 = 0x00FF (clamp + byte0 mask)
//   Y2 = clamped a, Y3 = clamped b, Y4 = product, Y5 = weights
//   Y6, Y7 = byte0, byte1 terms
//   Y8-Y9 = int32 byte0 accumulators
//   Y10-Y11 = int32 byte1 accumulators
//   Y12-Y13 = int64 drain byte0
//   Y14-Y15 = int64 drain byte1
// ============================================================================
TEXT ·nnueV5PairwiseDotN(SB), NOSPLIT, $0-40
	MOVQ accFirst+0(FP), AX
	MOVQ accSecond+8(FP), BX
	MOVQ weights+16(FP), DX
	MOVQ count+24(FP), CX
	SHRQ $7, CX                         // count / 128 = drain cycles

	VPXOR Y0, Y0, Y0                    // zero (floor)
	VMOVDQU nnue_clamp_255<>(SB), Y1    // 0x00FF (ceiling + byte0 mask)

	// Int64 drain accumulators
	VPXOR Y12, Y12, Y12
	VPXOR Y13, Y13, Y13
	VPXOR Y14, Y14, Y14
	VPXOR Y15, Y15, Y15

v5pw_drain_loop:
	// Reset int32 batch accumulators
	VPXOR Y8, Y8, Y8
	VPXOR Y9, Y9, Y9
	VPXOR Y10, Y10, Y10
	VPXOR Y11, Y11, Y11
	MOVQ $4, SI                          // 4 inner iterations × 32 elements = 128

v5pw_inner:
	// ======== First 16 elements ========
	VMOVDQU (AX), Y2                    // load a[0..15]
	VPMAXSW Y0, Y2, Y2
	VPMINSW Y1, Y2, Y2                  // clamp a
	VMOVDQU (BX), Y3                    // load b[0..15]
	VPMAXSW Y0, Y3, Y3
	VPMINSW Y1, Y3, Y3                  // clamp b
	VPMULLW Y2, Y3, Y4                  // a*b (uint16, max 65025)
	VMOVDQU (DX), Y5                    // weights[0..15]
	VPAND Y1, Y4, Y6                    // byte0 = product & 0xFF
	VPSRLW $8, Y4, Y7                   // byte1 = product >> 8
	VPMADDWD Y5, Y6, Y6                 // byte0 pairs
	VPMADDWD Y5, Y7, Y7                 // byte1 pairs
	VPADDD Y6, Y8, Y8
	VPADDD Y7, Y10, Y10

	// ======== Second 16 elements ========
	VMOVDQU 32(AX), Y2
	VPMAXSW Y0, Y2, Y2
	VPMINSW Y1, Y2, Y2
	VMOVDQU 32(BX), Y3
	VPMAXSW Y0, Y3, Y3
	VPMINSW Y1, Y3, Y3
	VPMULLW Y2, Y3, Y4
	VMOVDQU 32(DX), Y5
	VPAND Y1, Y4, Y6
	VPSRLW $8, Y4, Y7
	VPMADDWD Y5, Y6, Y6
	VPMADDWD Y5, Y7, Y7
	VPADDD Y6, Y9, Y9
	VPADDD Y7, Y11, Y11

	ADDQ $64, AX
	ADDQ $64, BX
	ADDQ $64, DX
	DECQ SI
	JNZ v5pw_inner

	// === Drain int32 → int64 ===
	VPADDD Y9, Y8, Y8
	VPMOVSXDQ X8, Y2
	VPADDQ Y2, Y12, Y12
	VEXTRACTI128 $1, Y8, X2
	VPMOVSXDQ X2, Y2
	VPADDQ Y2, Y13, Y13

	VPADDD Y11, Y10, Y10
	VPMOVSXDQ X10, Y2
	VPSLLQ $8, Y2, Y2
	VPADDQ Y2, Y14, Y14
	VEXTRACTI128 $1, Y10, X2
	VPMOVSXDQ X2, Y2
	VPSLLQ $8, Y2, Y2
	VPADDQ Y2, Y15, Y15

	DECQ CX
	JNZ v5pw_drain_loop

	// === Combine all int64 accumulators ===
	VPADDQ Y13, Y12, Y12
	VPADDQ Y15, Y14, Y14
	VPADDQ Y14, Y12, Y12

	VEXTRACTI128 $1, Y12, X9
	VPADDQ X9, X12, X12
	VPSHUFD $0x4E, X12, X9
	VPADDQ X9, X12, X12
	VMOVQ X12, AX
	MOVQ AX, ret+32(FP)

	VZEROUPPER
	RET

// ============================================================================
// nnueV5L1MatMulN(acc *int16, wT *int16, hidden *int32, accLen int, l1 int)
//
// L1 hidden layer matmul for one perspective (called twice: stm + ntm).
// Computes: hidden[i] += sum_j( clamp(acc[j], 0, 255) * wT[i*accLen + j] )
// for i=0..l1-1, j=0..accLen-1.
// accLen must be a multiple of 32. l1 can be any positive value.
//
// Uses transposed weight layout [l1][accLen] for sequential memory access.
// For each L1 neuron: CReLU dot product of accumulator against weight row,
// using VPMADDWD (pairs of int16 → int32 partial sums).
//
// Register allocation:
//   Y0  = zero (CReLU floor)
//   Y1  = 255 (CReLU ceiling)
//   Y8, Y9 = int32 accumulators (2x unrolled)
//   Y2, Y3 = loaded+clamped acc values
//   Y4, Y5 = loaded weights
//   Y6, Y7 = VPMADDWD results
//   SI  = acc base pointer
//   DI  = weight pointer (advances through rows)
//   R8  = hidden pointer (advances per neuron)
//   CX  = inner loop count (accLen / 32)
//   DX  = outer loop count (l1)
//   R9  = weight row stride in bytes (accLen * 2)
// ============================================================================
TEXT ·nnueV5L1MatMulN(SB), NOSPLIT, $0-40
	MOVQ acc+0(FP), SI
	MOVQ wT+8(FP), DI
	MOVQ hidden+16(FP), R8
	MOVQ accLen+24(FP), CX
	MOVQ l1+32(FP), DX

	// Weight row stride in bytes
	MOVQ CX, R9
	SHLQ $1, R9                         // R9 = accLen * 2

	// Inner loop count
	SHRQ $5, CX                         // CX = accLen / 32

	VPXOR Y0, Y0, Y0                    // Y0 = zero (floor)
	VMOVDQU nnue_clamp_255<>(SB), Y1    // Y1 = 255 (ceiling)

l1mm_outer:
	VPXOR Y8, Y8, Y8                    // accumulator 1
	VPXOR Y9, Y9, Y9                    // accumulator 2
	MOVQ SI, R10                        // R10 = acc cursor (reset per neuron)
	MOVQ DI, R11                        // R11 = weight cursor
	MOVQ CX, R12                        // R12 = inner loop counter

l1mm_inner:
	// First 16 elements
	VMOVDQU (R10), Y2
	VPMAXSW Y0, Y2, Y2                  // clamp floor
	VPMINSW Y1, Y2, Y2                  // clamp ceiling
	VMOVDQU (R11), Y4
	VPMADDWD Y2, Y4, Y6                 // pairwise mul-add → 8 int32
	VPADDD Y6, Y8, Y8

	// Second 16 elements (2x unrolled)
	VMOVDQU 32(R10), Y3
	VPMAXSW Y0, Y3, Y3
	VPMINSW Y1, Y3, Y3
	VMOVDQU 32(R11), Y5
	VPMADDWD Y3, Y5, Y7
	VPADDD Y7, Y9, Y9

	ADDQ $64, R10
	ADDQ $64, R11
	DECQ R12
	JNZ l1mm_inner

	// Horizontal sum: Y8 + Y9 → scalar
	VPADDD Y9, Y8, Y8
	VEXTRACTI128 $1, Y8, X6
	VPADDD X6, X8, X8
	VPSHUFD $0x4E, X8, X6
	VPADDD X6, X8, X8
	VPSHUFD $0x01, X8, X6
	VPADDD X6, X8, X8
	VMOVD X8, AX

	// Accumulate into hidden[i]
	ADDL (R8), AX
	MOVL AX, (R8)

	ADDQ $4, R8                          // next hidden slot
	ADDQ R9, DI                          // next weight row
	DECQ DX
	JNZ l1mm_outer

	VZEROUPPER
	RET

// ============================================================================
// nnueV5L1Int8MatMulN(acc8 *byte, wT8 *int8, hidden *int32, accLen int, l1 int)
//
// Int8 L1 matmul: hidden[i] += sum_j( acc8[j] * wT8[i*accLen + j] )
// Uses VPMADDUBSW (u8 × i8 → i16 pairwise) then VPMADDWD with ones to
// widen to int32. Processes 32 elements per iteration = 2x throughput
// over the int16 VPMADDWD kernel.
// accLen must be a multiple of 32. l1 can be any positive value.
//
// Register allocation:
//   Y0 = ones (16 x int16(1)) for VPMADDWD widening
//   Y8, Y9 = int32 accumulators (2x unrolled)
//   Y2-Y5 = loaded data, Y6-Y7 = VPMADDUBSW/VPMADDWD results
// ============================================================================
// nnueV5L1Int8MatMulN: 4-neuron blocked int8 matmul.
// Processes 4 L1 neurons simultaneously, sharing accumulator loads across
// all 4 weight rows. This gives 4× fewer acc loads and better ILP
// (4 independent VPMADDUBSW chains for out-of-order execution).
// L1 must be a multiple of 4. accLen must be a multiple of 32.
//
// Register allocation:
//   SI = acc8 base, DI = current weight group base
//   R8 = hidden pointer, R9 = stride (accLen bytes)
//   R10 = acc cursor, R11-R14 = weight cursors (4 rows)
//   R15 = inner loop counter, DX = outer counter (L1/4)
//   CX = inner count (accLen/32, constant)
//   Y0 = ones for VPMADDWD, Y2 = acc data (shared)
//   Y3-Y6 = weight data, Y8-Y11 = neuron accumulators
TEXT ·nnueV5L1Int8MatMulN(SB), NOSPLIT, $0-40
	MOVQ acc8+0(FP), SI
	MOVQ wT8+8(FP), DI
	MOVQ hidden+16(FP), R8
	MOVQ accLen+24(FP), CX
	MOVQ l1+32(FP), DX

	MOVQ CX, R9                         // R9 = stride = accLen bytes
	SHRQ $5, CX                         // CX = accLen / 32 = inner iterations
	SHRQ $2, DX                         // DX = L1 / 4 = outer iterations

	VMOVDQU nnue_ones_16<>(SB), Y0      // ones for VPMADDWD widening

l1i8_outer:
	VPXOR Y8, Y8, Y8                    // acc neuron 0
	VPXOR Y9, Y9, Y9                    // acc neuron 1
	VPXOR Y10, Y10, Y10                 // acc neuron 2
	VPXOR Y11, Y11, Y11                 // acc neuron 3

	MOVQ SI, R10                        // acc cursor
	MOVQ DI, R11                        // weight row 0
	LEAQ (DI)(R9*1), R12               // weight row 1
	LEAQ (DI)(R9*2), R13               // weight row 2
	LEAQ (R12)(R9*2), R14              // weight row 3 = DI + 3*stride
	MOVQ CX, R15                        // inner loop counter

l1i8_inner:
	// Load 32 uint8 accumulator values (shared by all 4 neurons)
	VMOVDQU (R10), Y2

	// Neuron 0: u8×i8 → i16 → i32, accumulate
	VMOVDQU (R11), Y3
	VPMADDUBSW Y3, Y2, Y3
	VPMADDWD Y0, Y3, Y3
	VPADDD Y3, Y8, Y8

	// Neuron 1
	VMOVDQU (R12), Y4
	VPMADDUBSW Y4, Y2, Y4
	VPMADDWD Y0, Y4, Y4
	VPADDD Y4, Y9, Y9

	// Neuron 2
	VMOVDQU (R13), Y5
	VPMADDUBSW Y5, Y2, Y5
	VPMADDWD Y0, Y5, Y5
	VPADDD Y5, Y10, Y10

	// Neuron 3
	VMOVDQU (R14), Y6
	VPMADDUBSW Y6, Y2, Y6
	VPMADDWD Y0, Y6, Y6
	VPADDD Y6, Y11, Y11

	ADDQ $32, R10
	ADDQ $32, R11
	ADDQ $32, R12
	ADDQ $32, R13
	ADDQ $32, R14
	DECQ R15
	JNZ l1i8_inner

	// Horizontal sum neuron 0 → hidden[0]
	VEXTRACTI128 $1, Y8, X3
	VPADDD X3, X8, X8
	VPSHUFD $0x4E, X8, X3
	VPADDD X3, X8, X8
	VPSHUFD $0x01, X8, X3
	VPADDD X3, X8, X8
	VMOVD X8, AX
	ADDL (R8), AX
	MOVL AX, (R8)

	// Horizontal sum neuron 1 → hidden[1]
	VEXTRACTI128 $1, Y9, X3
	VPADDD X3, X9, X9
	VPSHUFD $0x4E, X9, X3
	VPADDD X3, X9, X9
	VPSHUFD $0x01, X9, X3
	VPADDD X3, X9, X9
	VMOVD X9, AX
	ADDL 4(R8), AX
	MOVL AX, 4(R8)

	// Horizontal sum neuron 2 → hidden[2]
	VEXTRACTI128 $1, Y10, X3
	VPADDD X3, X10, X10
	VPSHUFD $0x4E, X10, X3
	VPADDD X3, X10, X10
	VPSHUFD $0x01, X10, X3
	VPADDD X3, X10, X10
	VMOVD X10, AX
	ADDL 8(R8), AX
	MOVL AX, 8(R8)

	// Horizontal sum neuron 3 → hidden[3]
	VEXTRACTI128 $1, Y11, X3
	VPADDD X3, X11, X11
	VPSHUFD $0x4E, X11, X3
	VPADDD X3, X11, X11
	VPSHUFD $0x01, X11, X3
	VPADDD X3, X11, X11
	VMOVD X11, AX
	ADDL 12(R8), AX
	MOVL AX, 12(R8)

	// Advance to next group of 4 neurons
	ADDQ $16, R8                         // hidden += 4 * sizeof(int32)
	MOVQ R14, DI                         // DI = end of row 3 = start of row 4
	DECQ DX
	JNZ l1i8_outer

	VZEROUPPER
	RET

// ============================================================================
// nnueV5L1SCReLUMatMulN(acc *int16, wT *int16, hidden *int64, accLen int, l1 int)
//
// SCReLU L1 matmul: hidden[i] += sum_j( clamp(acc[j], 0, 255)² * wT[i*accLen+j] )
// Uses lo/hi decomposition of v² to stay in VPMADDWD:
//   v² = lo + hi * 32768, where lo = v² & 0x7FFF, hi = v² >> 15
//   sum(v² * w) = sum(lo * w) + sum(hi * w) * 32768
// Accumulates into int64 hidden[] to avoid overflow.
// accLen must be a multiple of 16. l1 can be any positive value.
//
// Register allocation:
//   Y0  = zero, Y1 = 255 ceiling, Y2 = 0x7FFF mask
//   Y3  = loaded acc (clamped), Y4 = v² (VPMULLW)
//   Y5  = lo (v² & 0x7FFF), Y6 = hi (v² >> 15)
//   Y7  = loaded weights
//   Y8  = VPMADDWD(lo, w), Y9 = VPMADDWD(hi, w)
//   Y10 = lo accumulator (int32), Y11 = hi accumulator (int32)
// ============================================================================
TEXT ·nnueV5L1SCReLUMatMulN(SB), NOSPLIT, $0-40
	MOVQ acc+0(FP), SI
	MOVQ wT+8(FP), DI
	MOVQ hidden+16(FP), R8
	MOVQ accLen+24(FP), CX
	MOVQ l1+32(FP), DX

	// Weight row stride in bytes
	MOVQ CX, R9
	SHLQ $1, R9                         // R9 = accLen * 2

	// Inner loop count: accLen / 16
	SHRQ $4, CX

	VPXOR Y0, Y0, Y0                    // Y0 = zero
	VMOVDQU nnue_clamp_255<>(SB), Y1    // Y1 = 255 ceiling
	VPCMPEQW Y2, Y2, Y2                 // all 1s
	VPSRLW $1, Y2, Y2                   // Y2 = 0x7FFF mask

l1screlu_outer:
	VPXOR Y10, Y10, Y10                 // lo accumulator (int32)
	VPXOR Y11, Y11, Y11                 // hi accumulator (int32)
	MOVQ SI, R10                        // R10 = acc cursor
	MOVQ DI, R11                        // R11 = weight cursor
	MOVQ CX, R12                        // R12 = inner loop counter

l1screlu_inner:
	// Load 16 acc values, clamp to [0, 255]
	VMOVDQU (R10), Y3
	VPMAXSW Y0, Y3, Y3
	VPMINSW Y1, Y3, Y3

	// Square: v² = v * v (unsigned result, fits uint16)
	VPMULLW Y3, Y3, Y4

	// Decompose: lo = v² & 0x7FFF (safe signed int16), hi = v² >> 15 (0 or 1)
	VPAND Y4, Y2, Y5                    // lo
	VPSRLW $15, Y4, Y6                  // hi

	// Load weights
	VMOVDQU (R11), Y7

	// Multiply-accumulate both halves
	VPMADDWD Y5, Y7, Y8                 // lo * w → 8 int32
	VPMADDWD Y6, Y7, Y9                 // hi * w → 8 int32
	VPADDD Y8, Y10, Y10
	VPADDD Y9, Y11, Y11

	ADDQ $32, R10
	ADDQ $32, R11
	DECQ R12
	JNZ l1screlu_inner

	// Horizontal sum of lo accumulator (Y10) → single int32
	VEXTRACTI128 $1, Y10, X12
	VPADDD X12, X10, X10
	VPSHUFD $0x4E, X10, X12
	VPADDD X12, X10, X10
	VPSHUFD $0x01, X10, X12
	VPADDD X12, X10, X10
	VMOVD X10, AX                       // AX = lo_sum (int32)

	// Horizontal sum of hi accumulator (Y11) → single int32
	VEXTRACTI128 $1, Y11, X12
	VPADDD X12, X11, X11
	VPSHUFD $0x4E, X11, X12
	VPADDD X12, X11, X11
	VPSHUFD $0x01, X11, X12
	VPADDD X12, X11, X11
	VMOVD X11, BX                       // BX = hi_sum (int32)

	// Combine: total = lo_sum + hi_sum * 32768 (in int64)
	MOVLQSX AX, AX                      // sign-extend int32 → int64
	MOVLQSX BX, BX
	SHLQ $15, BX                        // hi_sum * 32768
	ADDQ BX, AX                         // AX = total (int64)

	// Accumulate into hidden[i] (int64)
	ADDQ (R8), AX
	MOVQ AX, (R8)

	ADDQ $8, R8                          // next hidden slot (int64 = 8 bytes)
	ADDQ R9, DI                          // next weight row
	DECQ DX
	JNZ l1screlu_outer

	VZEROUPPER
	RET

// ============================================================================
// nnueFloatMatVecFMA(dst *float32, input *float32, weights *float32, inputLen int, outputLen int)
//
// Float32 matrix-vector multiply using AVX2 FMA:
//   dst[k] += sum_i( input[i] * weights[i * outputLen + k] )
//
// dst must be pre-initialized (e.g. with biases). outputLen must be a multiple
// of 8. Uses VBROADCASTSS + VFMADD231PS for each input element, processing
// 8 output neurons per YMM register.
//
// Register allocation:
//   AX = dst, BX = input cursor, CX = weight cursor
//   DX = inputLen counter, R8 = outputLen, R9 = outputLen * 4 (row stride bytes)
//   Y0 = broadcast input value
//   Y8+ = accumulators (loaded from dst, stored back at end)
// ============================================================================
TEXT ·nnueFloatMatVecFMA(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), AX
	MOVQ input+8(FP), BX
	MOVQ weights+16(FP), CX
	MOVQ inputLen+24(FP), DX
	MOVQ outputLen+32(FP), R8

	// Row stride in bytes = outputLen * 4
	LEAQ (R8)(R8*1), R9                // R9 = outputLen * 2
	LEAQ (R9)(R9*1), R9                // R9 = outputLen * 4? No...
	// Simpler: outputLen * 4
	MOVQ R8, R9
	SHLQ $2, R9                        // R9 = outputLen * 4 bytes per row

	// Compute number of YMM groups = outputLen / 8
	MOVQ R8, R10
	SHRQ $3, R10                        // R10 = outputLen / 8

	// Load dst into accumulators
	// We handle up to 8 groups (outputLen <= 64)
	MOVQ AX, R11                        // R11 = dst cursor for initial load
	MOVQ R10, R12                       // R12 = group counter
	// Load all groups into Y8..Y15
	VMOVUPS (R11), Y8
	CMPQ R12, $1
	JE fmv_loaded
	VMOVUPS 32(R11), Y9
	CMPQ R12, $2
	JE fmv_loaded
	VMOVUPS 64(R11), Y10
	CMPQ R12, $3
	JE fmv_loaded
	VMOVUPS 96(R11), Y11
	CMPQ R12, $4
	JE fmv_loaded
	VMOVUPS 128(R11), Y12
	CMPQ R12, $5
	JE fmv_loaded
	VMOVUPS 160(R11), Y13
	CMPQ R12, $6
	JE fmv_loaded
	VMOVUPS 192(R11), Y14
	CMPQ R12, $7
	JE fmv_loaded
	VMOVUPS 224(R11), Y15

fmv_loaded:
	// Main loop: for each input[i], broadcast and FMA
fmv_input_loop:
	VBROADCASTSS (BX), Y0              // broadcast input[i] to all 8 lanes

	// Inner loop: FMA for each group of 8 outputs
	MOVQ CX, R11                       // R11 = weight row cursor
	VFMADD231PS (R11), Y0, Y8
	CMPQ R10, $1
	JE fmv_next_input
	VFMADD231PS 32(R11), Y0, Y9
	CMPQ R10, $2
	JE fmv_next_input
	VFMADD231PS 64(R11), Y0, Y10
	CMPQ R10, $3
	JE fmv_next_input
	VFMADD231PS 96(R11), Y0, Y11
	CMPQ R10, $4
	JE fmv_next_input
	VFMADD231PS 128(R11), Y0, Y12
	CMPQ R10, $5
	JE fmv_next_input
	VFMADD231PS 160(R11), Y0, Y13
	CMPQ R10, $6
	JE fmv_next_input
	VFMADD231PS 192(R11), Y0, Y14
	CMPQ R10, $7
	JE fmv_next_input
	VFMADD231PS 224(R11), Y0, Y15

fmv_next_input:
	ADDQ $4, BX                         // next input value (float32 = 4 bytes)
	ADDQ R9, CX                         // next weight row
	DECQ DX
	JNZ fmv_input_loop

	// Store accumulators back to dst
	MOVQ AX, R11
	VMOVUPS Y8, (R11)
	CMPQ R10, $1
	JE fmv_done
	VMOVUPS Y9, 32(R11)
	CMPQ R10, $2
	JE fmv_done
	VMOVUPS Y10, 64(R11)
	CMPQ R10, $3
	JE fmv_done
	VMOVUPS Y11, 96(R11)
	CMPQ R10, $4
	JE fmv_done
	VMOVUPS Y12, 128(R11)
	CMPQ R10, $5
	JE fmv_done
	VMOVUPS Y13, 160(R11)
	CMPQ R10, $6
	JE fmv_done
	VMOVUPS Y14, 192(R11)
	CMPQ R10, $7
	JE fmv_done
	VMOVUPS Y15, 224(R11)

fmv_done:
	VZEROUPPER
	RET

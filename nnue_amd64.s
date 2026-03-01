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

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

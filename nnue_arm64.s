//go:build arm64

#include "textflag.h"

// NNUE ARM64 NEON implementations for forward pass and accumulator updates.
//
// All functions use Go's ABI0 calling convention (arguments on stack via FP).
// NEON registers V0-V31 are 128-bit (8 x int16 or 4 x int32).
// 256 int16 elements = 32 NEON iterations (vs 16 AVX2 iterations).
//
// NEON arithmetic uses WORD encodings because Go's ARM64 assembler
// does not support all NEON mnemonics natively. Each WORD is annotated
// with its ARM64 instruction equivalent.
//
// Reserved registers (NOT used):
//   R18 (platform register, reserved on Darwin)
//   R26, R27 (g), R28 (RSB) — reserved by Go runtime

// ============================================================================
// nnueCReLU256(src *int16, dst *int16)
// Clamps 256 int16 values to [0, 127].
// 256 elements / 8 per NEON register = 32 iterations.
// ============================================================================
TEXT ·nnueCReLU256(SB), NOSPLIT, $0-16
	MOVD src+0(FP), R0
	MOVD dst+8(FP), R1

	// V1 = [127, 127, ..., 127] (8 x int16)
	MOVD $127, R2
	WORD $0x4E020C41              // DUP V1.8H, W2

	// V0 = zero vector (for max with 0)
	VEOR V0.B16, V0.B16, V0.B16

	MOVD $16, R3                  // 16 iterations (2 vectors each)

crelu_loop:
	VLD1 (R0), [V2.B16]          // load 8 int16
	WORD $0x4E606442              // SMAX V2.8H, V2.8H, V0.8H  (max with 0)
	WORD $0x4E616C42              // SMIN V2.8H, V2.8H, V1.8H  (min with 127)
	VST1 [V2.B16], (R1)          // store result
	ADD $16, R0, R0
	ADD $16, R1, R1
	VLD1 (R0), [V3.B16]          // load next 8 int16
	WORD $0x4E606463              // SMAX V3.8H, V3.8H, V0.8H
	WORD $0x4E616C63              // SMIN V3.8H, V3.8H, V1.8H
	VST1 [V3.B16], (R1)
	ADD $16, R0, R0
	ADD $16, R1, R1
	SUBS $1, R3, R3
	BNE crelu_loop

	RET

// ============================================================================
// nnueCReLU256to8(src *int16, dst *byte)
// Clamps 256 int16 values to [0, 127] and packs into 256 uint8 values.
// Processes 16 int16 → 16 uint8 per iteration (16 iterations).
// Uses SMAX/SMIN for clamping, SQXTUN/SQXTUN2 for narrowing.
// ============================================================================
TEXT ·nnueCReLU256to8(SB), NOSPLIT, $0-16
	MOVD src+0(FP), R0
	MOVD dst+8(FP), R1

	// V1 = [127, 127, ..., 127] (8 x int16)
	MOVD $127, R2
	WORD $0x4E020C41              // DUP V1.8H, W2

	// V0 = zero vector (for max with 0)
	VEOR V0.B16, V0.B16, V0.B16

	MOVD $16, R3                  // 16 iterations (16 int16 → 16 uint8 each)

crelu8_loop:
	VLD1 (R0), [V2.B16]          // load 8 int16
	WORD $0x4E606442              // SMAX V2.8H, V2.8H, V0.8H  (max with 0)
	WORD $0x4E616C42              // SMIN V2.8H, V2.8H, V1.8H  (min with 127)
	ADD $16, R0, R0
	VLD1 (R0), [V3.B16]          // load next 8 int16
	WORD $0x4E606463              // SMAX V3.8H, V3.8H, V0.8H
	WORD $0x4E616C63              // SMIN V3.8H, V3.8H, V1.8H
	ADD $16, R0, R0
	WORD $0x2E212844              // SQXTUN V4.8B, V2.8H   (narrow low 8)
	WORD $0x6E212864              // SQXTUN2 V4.16B, V3.8H (narrow high 8)
	VST1 [V4.B16], (R1)          // store 16 uint8
	ADD $16, R1, R1
	SUBS $1, R3, R3
	BNE crelu8_loop

	RET

// ============================================================================
// nnueMatMul32x512_i8(input *byte, weightsT *int8, biases *int32, output *int32)
//
// Computes: output[j] = biases[j] + sum_i(input[i] * weightsT[j][i])
// for j = 0..31, i = 0..511.
//
// input is uint8 [0,127] (safe to treat as signed int8).
// weightsT layout: [32][512] int8, row-major. Each row = 512 bytes.
// Uses SMULL/SMLAL2 on .8B for int8×int8→int16, SADALP for int16→int32.
// Processes 16 elements per inner iteration (vs 8 for int16 path).
//
// Register allocation:
//   R0  = input base pointer (constant)
//   R1  = weightsT group base (advances by 2048 per outer)
//   R2  = biases pointer (advances by 16 per outer)
//   R3  = output pointer (advances by 16 per outer)
//   R4  = outer loop counter (8..0)
//   R5  = stride constant (512)
//   R6-R9   = weight row cursors j..j+3 (inner loop)
//   R10 = input cursor (inner loop)
//   R11 = inner loop counter (32..0)
//   R12-R13 = scratch for reduction
//   V16-V19 = int32 accumulators for 4 outputs
//   V20     = int16 scratch for SMULL/SMLAL2
//   V24     = input data (16 bytes)
//   V25     = weight data (16 bytes)
// ============================================================================
TEXT ·nnueMatMul32x512_i8(SB), NOSPLIT, $0-32
	MOVD input+0(FP), R0
	MOVD weightsT+8(FP), R1
	MOVD biases+16(FP), R2
	MOVD output+24(FP), R3

	MOVD $8, R4                   // 8 groups of 4 outputs
	MOVD $512, R5                 // row stride (512 bytes)

i8_outer:
	// Set up weight row cursors for j, j+1, j+2, j+3
	MOVD R1, R6                   // weights[j]
	ADD R5, R1, R7                // weights[j+1]
	ADD R5, R7, R8                // weights[j+2]
	ADD R5, R8, R9                // weights[j+3]

	// Zero accumulators
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16

	// Inner loop: 512 elements in groups of 16 (32 iterations)
	MOVD R0, R10                  // input cursor
	MOVD $32, R11                 // 32 iterations

i8_inner:
	VLD1 (R10), [V24.B16]        // load 16 uint8/int8 inputs

	// Output j: SMULL + SMLAL2 + SADALP
	VLD1 (R6), [V25.B16]
	WORD $0x0E39C314              // SMULL  V20.8H, V24.8B, V25.8B
	WORD $0x4E398314              // SMLAL2 V20.8H, V24.16B, V25.16B
	WORD $0x4E606A90              // SADALP V16.4S, V20.8H

	// Output j+1
	VLD1 (R7), [V25.B16]
	WORD $0x0E39C314              // SMULL  V20.8H, V24.8B, V25.8B
	WORD $0x4E398314              // SMLAL2 V20.8H, V24.16B, V25.16B
	WORD $0x4E606A91              // SADALP V17.4S, V20.8H

	// Output j+2
	VLD1 (R8), [V25.B16]
	WORD $0x0E39C314              // SMULL  V20.8H, V24.8B, V25.8B
	WORD $0x4E398314              // SMLAL2 V20.8H, V24.16B, V25.16B
	WORD $0x4E606A92              // SADALP V18.4S, V20.8H

	// Output j+3
	VLD1 (R9), [V25.B16]
	WORD $0x0E39C314              // SMULL  V20.8H, V24.8B, V25.8B
	WORD $0x4E398314              // SMLAL2 V20.8H, V24.16B, V25.16B
	WORD $0x4E606A93              // SADALP V19.4S, V20.8H

	// Advance cursors
	ADD $16, R10, R10
	ADD $16, R6, R6
	ADD $16, R7, R7
	ADD $16, R8, R8
	ADD $16, R9, R9
	SUBS $1, R11, R11
	BNE i8_inner

	// --- Horizontal reduce V16 → output[j] ---
	WORD $0x4EB0BE10              // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x4EB0BE10              // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x1E26020C              // FMOV W12, S16
	MOVW (R2), R13                // bias[j]
	ADDW R13, R12, R12
	MOVW R12, (R3)                // store output[j]

	// --- Horizontal reduce V17 → output[j+1] ---
	WORD $0x4EB1BE31              // ADDP V17.4S, V17.4S, V17.4S
	WORD $0x4EB1BE31              // ADDP V17.4S, V17.4S, V17.4S
	WORD $0x1E26022C              // FMOV W12, S17
	MOVW 4(R2), R13
	ADDW R13, R12, R12
	MOVW R12, 4(R3)

	// --- Horizontal reduce V18 → output[j+2] ---
	WORD $0x4EB2BE52              // ADDP V18.4S, V18.4S, V18.4S
	WORD $0x4EB2BE52              // ADDP V18.4S, V18.4S, V18.4S
	WORD $0x1E26024C              // FMOV W12, S18
	MOVW 8(R2), R13
	ADDW R13, R12, R12
	MOVW R12, 8(R3)

	// --- Horizontal reduce V19 → output[j+3] ---
	WORD $0x4EB3BE73              // ADDP V19.4S, V19.4S, V19.4S
	WORD $0x4EB3BE73              // ADDP V19.4S, V19.4S, V19.4S
	WORD $0x1E26026C              // FMOV W12, S19
	MOVW 12(R2), R13
	ADDW R13, R12, R12
	MOVW R12, 12(R3)

	// Advance outer pointers
	ADD $2048, R1, R1             // weightsT += 4 rows (4 * 512)
	ADD $16, R2, R2               // biases += 4 int32
	ADD $16, R3, R3               // output += 4 int32
	SUBS $1, R4, R4
	BNE i8_outer

	RET

// ============================================================================
// nnueMatMul32x512(input *int16, weightsT *int16, biases *int32, output *int32)
//
// Computes: output[j] = biases[j] + sum_i(input[i] * weightsT[j][i])
// for j = 0..31, i = 0..511.
//
// weightsT layout: [32][512] int16, row-major. Each row = 1024 bytes.
// Processes 4 output neurons at a time (8 groups of 4).
//
// Uses SMULL/SMLAL2 pairs: for each group of 8 int16 inputs,
// SMULL multiplies the low 4 pairs → 4 int32,
// SMLAL2 multiplies the high 4 pairs and accumulates.
//
// Register allocation:
//   R0  = input base pointer (constant)
//   R1  = weightsT group base (advances by 4096 per outer)
//   R2  = biases pointer (advances by 16 per outer)
//   R3  = output pointer (advances by 16 per outer)
//   R4  = outer loop counter (8..0)
//   R5  = stride constant (1024)
//   R6-R9   = weight row cursors j..j+3 (inner loop)
//   R10 = input cursor (inner loop)
//   R11 = inner loop counter (64..0)
//   R12-R13 = scratch for reduction
//   V0      = input data
//   V4-V7   = weight data
//   V16-V19 = accumulators for 4 outputs (4 x int32)
//   V20-V23 = scratch for multiply-accumulate
// ============================================================================
TEXT ·nnueMatMul32x512(SB), NOSPLIT, $0-32
	MOVD input+0(FP), R0
	MOVD weightsT+8(FP), R1
	MOVD biases+16(FP), R2
	MOVD output+24(FP), R3

	MOVD $8, R4                   // 8 groups of 4 outputs
	MOVD $1024, R5                // row stride (512 * 2 bytes)

outer:
	// Set up weight row cursors for j, j+1, j+2, j+3
	MOVD R1, R6                   // weights[j]
	ADD R5, R1, R7                // weights[j+1]
	ADD R5, R7, R8                // weights[j+2]
	ADD R5, R8, R9                // weights[j+3]

	// Zero accumulators
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16

	// Inner loop: 512 elements in groups of 8 (64 iterations)
	MOVD R0, R10                  // input cursor
	MOVD $64, R11                 // 64 iterations

inner:
	VLD1 (R10), [V0.B16]         // load 8 int16 inputs

	// Output j: multiply-accumulate directly into V16
	VLD1 (R6), [V4.B16]
	WORD $0x0E648010              // SMLAL  V16.4S, V0.4H, V4.4H
	WORD $0x4E648010              // SMLAL2 V16.4S, V0.8H, V4.8H

	// Output j+1: multiply-accumulate directly into V17
	VLD1 (R7), [V5.B16]
	WORD $0x0E658011              // SMLAL  V17.4S, V0.4H, V5.4H
	WORD $0x4E658011              // SMLAL2 V17.4S, V0.8H, V5.8H

	// Output j+2: multiply-accumulate directly into V18
	VLD1 (R8), [V6.B16]
	WORD $0x0E668012              // SMLAL  V18.4S, V0.4H, V6.4H
	WORD $0x4E668012              // SMLAL2 V18.4S, V0.8H, V6.8H

	// Output j+3: multiply-accumulate directly into V19
	VLD1 (R9), [V7.B16]
	WORD $0x0E678013              // SMLAL  V19.4S, V0.4H, V7.4H
	WORD $0x4E678013              // SMLAL2 V19.4S, V0.8H, V7.8H

	// Advance cursors
	ADD $16, R10, R10
	ADD $16, R6, R6
	ADD $16, R7, R7
	ADD $16, R8, R8
	ADD $16, R9, R9
	SUBS $1, R11, R11
	BNE inner

	// --- Horizontal reduce V16 → output[j] ---
	// ADDP twice: [a,b,c,d] → [a+b,c+d,...] → [(a+b)+(c+d),...]
	WORD $0x4EB0BE10              // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x4EB0BE10              // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x1E26020C              // FMOV W12, S16
	MOVW (R2), R13                // bias[j]
	ADDW R13, R12, R12
	MOVW R12, (R3)                // store output[j]

	// --- Horizontal reduce V17 → output[j+1] ---
	WORD $0x4EB1BE31              // ADDP V17.4S, V17.4S, V17.4S
	WORD $0x4EB1BE31              // ADDP V17.4S, V17.4S, V17.4S
	WORD $0x1E26022C              // FMOV W12, S17
	MOVW 4(R2), R13
	ADDW R13, R12, R12
	MOVW R12, 4(R3)

	// --- Horizontal reduce V18 → output[j+2] ---
	WORD $0x4EB2BE52              // ADDP V18.4S, V18.4S, V18.4S
	WORD $0x4EB2BE52              // ADDP V18.4S, V18.4S, V18.4S
	WORD $0x1E26024C              // FMOV W12, S18
	MOVW 8(R2), R13
	ADDW R13, R12, R12
	MOVW R12, 8(R3)

	// --- Horizontal reduce V19 → output[j+3] ---
	WORD $0x4EB3BE73              // ADDP V19.4S, V19.4S, V19.4S
	WORD $0x4EB3BE73              // ADDP V19.4S, V19.4S, V19.4S
	WORD $0x1E26026C              // FMOV W12, S19
	MOVW 12(R2), R13
	ADDW R13, R12, R12
	MOVW R12, 12(R3)

	// Advance outer pointers
	ADD $4096, R1, R1             // weightsT += 4 rows (4 * 1024)
	ADD $16, R2, R2               // biases += 4 int32
	ADD $16, R3, R3               // output += 4 int32
	SUBS $1, R4, R4
	BNE outer

	RET

// ============================================================================
// nnueAccSubAdd256(acc *int16, oldW *int16, newW *int16)
// Computes: acc[i] += newW[i] - oldW[i] for i = 0..255
// 256 elements / 8 per NEON register = 32 iterations.
// ============================================================================
TEXT ·nnueAccSubAdd256(SB), NOSPLIT, $0-24
	MOVD acc+0(FP), R0
	MOVD oldW+8(FP), R1
	MOVD newW+16(FP), R2
	MOVD $16, R3                  // 16 iterations (2 vectors each)

subadd_loop:
	VLD1 (R2), [V0.B16]          // new weights
	VLD1 (R1), [V1.B16]          // old weights
	WORD $0x6E618400              // SUB V0.8H, V0.8H, V1.8H  (new - old)
	VLD1 (R0), [V1.B16]          // accumulator
	WORD $0x4E608420              // ADD V0.8H, V1.8H, V0.8H  (acc + delta)
	VST1 [V0.B16], (R0)          // store back
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	VLD1 (R2), [V2.B16]
	VLD1 (R1), [V3.B16]
	WORD $0x6E638442              // SUB V2.8H, V2.8H, V3.8H
	VLD1 (R0), [V3.B16]
	WORD $0x4E628460              // ADD V0.8H, V3.8H, V2.8H
	VST1 [V0.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	SUBS $1, R3, R3
	BNE subadd_loop

	RET

// ============================================================================
// nnueAccSubSubAdd256(acc *int16, oldW *int16, newW *int16, capW *int16)
// Computes: acc[i] += newW[i] - oldW[i] - capW[i] for i = 0..255
// ============================================================================
TEXT ·nnueAccSubSubAdd256(SB), NOSPLIT, $0-32
	MOVD acc+0(FP), R0
	MOVD oldW+8(FP), R1
	MOVD newW+16(FP), R2
	MOVD capW+24(FP), R3
	MOVD $16, R4                  // 16 iterations (2 vectors each)

subsubadd_loop:
	VLD1 (R2), [V0.B16]          // new weights
	VLD1 (R1), [V1.B16]          // old weights
	WORD $0x6E618400              // SUB V0.8H, V0.8H, V1.8H  (new - old)
	VLD1 (R3), [V1.B16]          // capture weights
	WORD $0x6E618400              // SUB V0.8H, V0.8H, V1.8H  (new - old - cap)
	VLD1 (R0), [V1.B16]          // accumulator
	WORD $0x4E608420              // ADD V0.8H, V1.8H, V0.8H  (acc + delta)
	VST1 [V0.B16], (R0)          // store back
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	VLD1 (R2), [V2.B16]
	VLD1 (R1), [V3.B16]
	WORD $0x6E638442              // SUB V2.8H, V2.8H, V3.8H
	VLD1 (R3), [V3.B16]
	WORD $0x6E638442              // SUB V2.8H, V2.8H, V3.8H
	VLD1 (R0), [V3.B16]
	WORD $0x4E628460              // ADD V0.8H, V3.8H, V2.8H
	VST1 [V0.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	SUBS $1, R4, R4
	BNE subsubadd_loop

	RET

// ============================================================================
// nnueAccAdd256(acc *int16, weights *int16)
// Computes: acc[i] += weights[i] for i = 0..255
// ============================================================================
TEXT ·nnueAccAdd256(SB), NOSPLIT, $0-16
	MOVD acc+0(FP), R0
	MOVD weights+8(FP), R1
	MOVD $16, R2                  // 16 iterations (2 vectors each)

accadd_loop:
	VLD1 (R0), [V0.B16]
	VLD1 (R1), [V1.B16]
	WORD $0x4E618400              // ADD V0.8H, V0.8H, V1.8H
	VST1 [V0.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	VLD1 (R0), [V2.B16]
	VLD1 (R1), [V3.B16]
	WORD $0x4E638442              // ADD V2.8H, V2.8H, V3.8H
	VST1 [V2.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	SUBS $1, R2, R2
	BNE accadd_loop

	RET

// ============================================================================
// nnueAccSub256(acc *int16, weights *int16)
// Computes: acc[i] -= weights[i] for i = 0..255
// ============================================================================
TEXT ·nnueAccSub256(SB), NOSPLIT, $0-16
	MOVD acc+0(FP), R0
	MOVD weights+8(FP), R1
	MOVD $16, R2                  // 16 iterations (2 vectors each)

accsub_loop:
	VLD1 (R0), [V0.B16]
	VLD1 (R1), [V1.B16]
	WORD $0x6E618400              // SUB V0.8H, V0.8H, V1.8H
	VST1 [V0.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	VLD1 (R0), [V2.B16]
	VLD1 (R1), [V3.B16]
	WORD $0x6E638442              // SUB V2.8H, V2.8H, V3.8H
	VST1 [V2.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	SUBS $1, R2, R2
	BNE accsub_loop

	RET

// ============================================================================
// nnueAccCopySubAdd256(dst *int16, src *int16, oldW *int16, newW *int16)
// Computes: dst[i] = src[i] + newW[i] - oldW[i] for i = 0..255
// Fused copy+update: reads from src (parent), writes to dst (child).
// ============================================================================
TEXT ·nnueAccCopySubAdd256(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD oldW+16(FP), R2
	MOVD newW+24(FP), R3
	MOVD $16, R4                  // 16 iterations (2 vectors each)

copysubadd_loop:
	VLD1 (R3), [V0.B16]          // new weights
	VLD1 (R2), [V1.B16]          // old weights
	WORD $0x6E618400              // SUB V0.8H, V0.8H, V1.8H  (new - old)
	VLD1 (R1), [V1.B16]          // src (parent accumulator)
	WORD $0x4E608420              // ADD V0.8H, V1.8H, V0.8H  (src + delta)
	VST1 [V0.B16], (R0)          // store to dst
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	VLD1 (R3), [V2.B16]
	VLD1 (R2), [V3.B16]
	WORD $0x6E638442              // SUB V2.8H, V2.8H, V3.8H
	VLD1 (R1), [V3.B16]
	WORD $0x4E628460              // ADD V0.8H, V3.8H, V2.8H
	VST1 [V0.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	SUBS $1, R4, R4
	BNE copysubadd_loop

	RET

// ============================================================================
// nnueAccCopySubSubAdd256(dst *int16, src *int16, oldW *int16, newW *int16, capW *int16)
// Computes: dst[i] = src[i] + newW[i] - oldW[i] - capW[i] for i = 0..255
// Fused copy+update for captures: reads from src (parent), writes to dst (child).
// ============================================================================
TEXT ·nnueAccCopySubSubAdd256(SB), NOSPLIT, $0-40
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD oldW+16(FP), R2
	MOVD newW+24(FP), R3
	MOVD capW+32(FP), R4
	MOVD $16, R5                  // 16 iterations (2 vectors each)

copysubsubadd_loop:
	VLD1 (R3), [V0.B16]          // new weights
	VLD1 (R2), [V1.B16]          // old weights
	WORD $0x6E618400              // SUB V0.8H, V0.8H, V1.8H  (new - old)
	VLD1 (R4), [V1.B16]          // capture weights
	WORD $0x6E618400              // SUB V0.8H, V0.8H, V1.8H  (new - old - cap)
	VLD1 (R1), [V1.B16]          // src (parent accumulator)
	WORD $0x4E608420              // ADD V0.8H, V1.8H, V0.8H  (src + delta)
	VST1 [V0.B16], (R0)          // store to dst
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	ADD $16, R4, R4
	VLD1 (R3), [V2.B16]
	VLD1 (R2), [V3.B16]
	WORD $0x6E638442              // SUB V2.8H, V2.8H, V3.8H
	VLD1 (R4), [V3.B16]
	WORD $0x6E638442              // SUB V2.8H, V2.8H, V3.8H
	VLD1 (R1), [V3.B16]
	WORD $0x4E628460              // ADD V0.8H, V3.8H, V2.8H
	VST1 [V0.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	ADD $16, R4, R4
	SUBS $1, R5, R5
	BNE copysubsubadd_loop

	RET

// ============================================================================
// nnueMatMul32x32ReLU(input *int32, weightsT *int16, biases *int32, output *int32)
//
// Computes: output[k] = biases[k] + sum_j(max(0, input[j] >> 6) * weightsT[k*32+j])
// for k = 0..31, j = 0..31.
//
// weightsT layout: [32][32] int16, row-major (each row = 64 bytes).
// Precomputes activated = ReLU(input >> 6) into V0-V7 (8 regs × 4 int32).
// Uses SSHLL/SSHLL2 to sign-extend int16 weights to int32, then MUL+ADD.
// Note: MLA is NOT available for .4S on NEON (that encoding is SDOT).
//
// Register allocation:
//   R0  = input pointer (precompute only)
//   R1  = weightsT pointer (advances by 64 per output)
//   R2  = biases pointer (advances by 4 per output)
//   R3  = output pointer (advances by 4 per output)
//   R4  = loop counter (32..0)
//   R5  = temp address for weight loads
//   R12 = scratch for scalar result
//   R13 = scratch for bias
//   V0-V7  = activated input (constant after precompute)
//   V8     = zero register
//   V16    = accumulator per output
//   V24    = weight load (8 int16)
//   V25    = sign-extended weights / products
// ============================================================================
TEXT ·nnueMatMul32x32ReLU(SB), NOSPLIT, $0-32
	MOVD input+0(FP), R0
	MOVD weightsT+8(FP), R1
	MOVD biases+16(FP), R2
	MOVD output+24(FP), R3

	// Zero register for ReLU
	VEOR V8.B16, V8.B16, V8.B16

	// Precompute activated[0..31] = max(0, input[i] >> 6)
	// 8 NEON registers × 4 int32 = 32 values

	VLD1 (R0), [V0.B16]
	WORD $0x4F3A0400              // SSHR V0.4S, V0.4S, #6
	WORD $0x4EA86400              // SMAX V0.4S, V0.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V1.B16]
	WORD $0x4F3A0421              // SSHR V1.4S, V1.4S, #6
	WORD $0x4EA86421              // SMAX V1.4S, V1.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V2.B16]
	WORD $0x4F3A0442              // SSHR V2.4S, V2.4S, #6
	WORD $0x4EA86442              // SMAX V2.4S, V2.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V3.B16]
	WORD $0x4F3A0463              // SSHR V3.4S, V3.4S, #6
	WORD $0x4EA86463              // SMAX V3.4S, V3.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V4.B16]
	WORD $0x4F3A0484              // SSHR V4.4S, V4.4S, #6
	WORD $0x4EA86484              // SMAX V4.4S, V4.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V5.B16]
	WORD $0x4F3A04A5              // SSHR V5.4S, V5.4S, #6
	WORD $0x4EA864A5              // SMAX V5.4S, V5.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V6.B16]
	WORD $0x4F3A04C6              // SSHR V6.4S, V6.4S, #6
	WORD $0x4EA864C6              // SMAX V6.4S, V6.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V7.B16]
	WORD $0x4F3A04E7              // SSHR V7.4S, V7.4S, #6
	WORD $0x4EA864E7              // SMAX V7.4S, V7.4S, V8.4S

	MOVD $32, R4                  // 32 output neurons

matmul32_loop:
	// Zero accumulator
	VEOR V16.B16, V16.B16, V16.B16

	// Group 0: weights[0..7] × activated[0..7]
	VLD1 (R1), [V24.B16]         // load 8 int16 weights
	WORD $0x0F10A719              // SSHLL  V25.4S, V24.4H, #0  (sign-extend low 4)
	WORD $0x4EA09F39              // MUL    V25.4S, V25.4S, V0.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S
	WORD $0x4F10A719              // SSHLL2 V25.4S, V24.8H, #0  (sign-extend high 4)
	WORD $0x4EA19F39              // MUL    V25.4S, V25.4S, V1.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S

	// Group 1: weights[8..15] × activated[8..15]
	ADD $16, R1, R5
	VLD1 (R5), [V24.B16]
	WORD $0x0F10A719              // SSHLL  V25.4S, V24.4H, #0
	WORD $0x4EA29F39              // MUL    V25.4S, V25.4S, V2.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S
	WORD $0x4F10A719              // SSHLL2 V25.4S, V24.8H, #0
	WORD $0x4EA39F39              // MUL    V25.4S, V25.4S, V3.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S

	// Group 2: weights[16..23] × activated[16..23]
	ADD $32, R1, R5
	VLD1 (R5), [V24.B16]
	WORD $0x0F10A719              // SSHLL  V25.4S, V24.4H, #0
	WORD $0x4EA49F39              // MUL    V25.4S, V25.4S, V4.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S
	WORD $0x4F10A719              // SSHLL2 V25.4S, V24.8H, #0
	WORD $0x4EA59F39              // MUL    V25.4S, V25.4S, V5.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S

	// Group 3: weights[24..31] × activated[24..31]
	ADD $48, R1, R5
	VLD1 (R5), [V24.B16]
	WORD $0x0F10A719              // SSHLL  V25.4S, V24.4H, #0
	WORD $0x4EA69F39              // MUL    V25.4S, V25.4S, V6.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S
	WORD $0x4F10A719              // SSHLL2 V25.4S, V24.8H, #0
	WORD $0x4EA79F39              // MUL    V25.4S, V25.4S, V7.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S

	// Horizontal reduce V16 → scalar
	WORD $0x4EB0BE10              // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x4EB0BE10              // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x1E26020C              // FMOV W12, S16
	MOVW (R2), R13                // bias[k]
	ADDW R13, R12, R12
	MOVW R12, (R3)                // store output[k]

	// Advance to next output neuron
	ADD $64, R1, R1               // next weight row (32 int16 = 64 bytes)
	ADD $4, R2, R2                // next bias
	ADD $4, R3, R3                // next output
	SUBS $1, R4, R4
	BNE matmul32_loop

	RET

// ============================================================================
// nnueDotReLU32(input *int32, weights *int16) int32
//
// Returns: sum_k(max(0, input[k] >> 6) * int32(weights[k])) for k = 0..31
// Used for the output layer (32 → 1 with ReLU activation).
//
// Register allocation:
//   R0  = input pointer (advances during precompute)
//   R1  = weights pointer
//   R5  = temp address
//   V0-V7  = activated input (precomputed)
//   V8     = zero register
//   V16    = accumulator
//   V24    = weight load (8 int16)
//   V25    = sign-extended weights
// ============================================================================
TEXT ·nnueDotReLU32(SB), NOSPLIT, $0-24
	MOVD input+0(FP), R0
	MOVD weights+8(FP), R1

	// Zero register for ReLU and accumulator
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V16.B16, V16.B16, V16.B16

	// Precompute activated[0..31] = max(0, input[i] >> 6)
	VLD1 (R0), [V0.B16]
	WORD $0x4F3A0400              // SSHR V0.4S, V0.4S, #6
	WORD $0x4EA86400              // SMAX V0.4S, V0.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V1.B16]
	WORD $0x4F3A0421              // SSHR V1.4S, V1.4S, #6
	WORD $0x4EA86421              // SMAX V1.4S, V1.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V2.B16]
	WORD $0x4F3A0442              // SSHR V2.4S, V2.4S, #6
	WORD $0x4EA86442              // SMAX V2.4S, V2.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V3.B16]
	WORD $0x4F3A0463              // SSHR V3.4S, V3.4S, #6
	WORD $0x4EA86463              // SMAX V3.4S, V3.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V4.B16]
	WORD $0x4F3A0484              // SSHR V4.4S, V4.4S, #6
	WORD $0x4EA86484              // SMAX V4.4S, V4.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V5.B16]
	WORD $0x4F3A04A5              // SSHR V5.4S, V5.4S, #6
	WORD $0x4EA864A5              // SMAX V5.4S, V5.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V6.B16]
	WORD $0x4F3A04C6              // SSHR V6.4S, V6.4S, #6
	WORD $0x4EA864C6              // SMAX V6.4S, V6.4S, V8.4S

	ADD $16, R0, R0
	VLD1 (R0), [V7.B16]
	WORD $0x4F3A04E7              // SSHR V7.4S, V7.4S, #6
	WORD $0x4EA864E7              // SMAX V7.4S, V7.4S, V8.4S

	// Group 0: weights[0..7] × activated[0..7]
	VLD1 (R1), [V24.B16]
	WORD $0x0F10A719              // SSHLL  V25.4S, V24.4H, #0
	WORD $0x4EA09F39              // MUL    V25.4S, V25.4S, V0.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S
	WORD $0x4F10A719              // SSHLL2 V25.4S, V24.8H, #0
	WORD $0x4EA19F39              // MUL    V25.4S, V25.4S, V1.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S

	// Group 1: weights[8..15] × activated[8..15]
	ADD $16, R1, R5
	VLD1 (R5), [V24.B16]
	WORD $0x0F10A719              // SSHLL  V25.4S, V24.4H, #0
	WORD $0x4EA29F39              // MUL    V25.4S, V25.4S, V2.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S
	WORD $0x4F10A719              // SSHLL2 V25.4S, V24.8H, #0
	WORD $0x4EA39F39              // MUL    V25.4S, V25.4S, V3.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S

	// Group 2: weights[16..23] × activated[16..23]
	ADD $32, R1, R5
	VLD1 (R5), [V24.B16]
	WORD $0x0F10A719              // SSHLL  V25.4S, V24.4H, #0
	WORD $0x4EA49F39              // MUL    V25.4S, V25.4S, V4.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S
	WORD $0x4F10A719              // SSHLL2 V25.4S, V24.8H, #0
	WORD $0x4EA59F39              // MUL    V25.4S, V25.4S, V5.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S

	// Group 3: weights[24..31] × activated[24..31]
	ADD $48, R1, R5
	VLD1 (R5), [V24.B16]
	WORD $0x0F10A719              // SSHLL  V25.4S, V24.4H, #0
	WORD $0x4EA69F39              // MUL    V25.4S, V25.4S, V6.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S
	WORD $0x4F10A719              // SSHLL2 V25.4S, V24.8H, #0
	WORD $0x4EA79F39              // MUL    V25.4S, V25.4S, V7.4S
	WORD $0x4EB98610              // ADD    V16.4S, V16.4S, V25.4S

	// Horizontal reduce V16 → scalar
	WORD $0x4EB0BE10              // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x4EB0BE10              // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x1E26020C              // FMOV W12, S16
	MOVW R12, ret+16(FP)

	RET

// ============================================================================
// nnueAccAddN(acc *int16, weights *int16, count int)
// Computes: acc[i] += weights[i] for i = 0..count-1
// count must be a multiple of 16.
// ============================================================================
TEXT ·nnueAccAddN(SB), NOSPLIT, $0-24
	MOVD acc+0(FP), R0
	MOVD weights+8(FP), R1
	MOVD count+16(FP), R2
	LSR $4, R2, R2                // count / 16

accaddn_loop:
	VLD1 (R0), [V0.B16]
	VLD1 (R1), [V1.B16]
	WORD $0x4E618400              // ADD V0.8H, V0.8H, V1.8H
	VST1 [V0.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	VLD1 (R0), [V2.B16]
	VLD1 (R1), [V3.B16]
	WORD $0x4E638442              // ADD V2.8H, V2.8H, V3.8H
	VST1 [V2.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	SUBS $1, R2, R2
	BNE accaddn_loop

	RET

// ============================================================================
// nnueAccSubN(acc *int16, weights *int16, count int)
// Computes: acc[i] -= weights[i] for i = 0..count-1
// count must be a multiple of 16.
// ============================================================================
TEXT ·nnueAccSubN(SB), NOSPLIT, $0-24
	MOVD acc+0(FP), R0
	MOVD weights+8(FP), R1
	MOVD count+16(FP), R2
	LSR $4, R2, R2

accsubn_loop:
	VLD1 (R0), [V0.B16]
	VLD1 (R1), [V1.B16]
	WORD $0x6E618400              // SUB V0.8H, V0.8H, V1.8H
	VST1 [V0.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	VLD1 (R0), [V2.B16]
	VLD1 (R1), [V3.B16]
	WORD $0x6E638442              // SUB V2.8H, V2.8H, V3.8H
	VST1 [V2.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	SUBS $1, R2, R2
	BNE accsubn_loop

	RET

// ============================================================================
// nnueAccSubAddN(acc *int16, oldW *int16, newW *int16, count int)
// Computes: acc[i] += newW[i] - oldW[i] for i = 0..count-1
// count must be a multiple of 16.
// ============================================================================
TEXT ·nnueAccSubAddN(SB), NOSPLIT, $0-32
	MOVD acc+0(FP), R0
	MOVD oldW+8(FP), R1
	MOVD newW+16(FP), R2
	MOVD count+24(FP), R3
	LSR $4, R3, R3

accsubaddn_loop:
	VLD1 (R2), [V0.B16]          // new weights
	VLD1 (R1), [V1.B16]          // old weights
	WORD $0x6E618400              // SUB V0.8H, V0.8H, V1.8H  (new - old)
	VLD1 (R0), [V1.B16]          // accumulator
	WORD $0x4E608420              // ADD V0.8H, V1.8H, V0.8H  (acc + delta)
	VST1 [V0.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	VLD1 (R2), [V2.B16]
	VLD1 (R1), [V3.B16]
	WORD $0x6E638442              // SUB V2.8H, V2.8H, V3.8H
	VLD1 (R0), [V3.B16]
	WORD $0x4E628460              // ADD V0.8H, V3.8H, V2.8H
	VST1 [V0.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	SUBS $1, R3, R3
	BNE accsubaddn_loop

	RET

// ============================================================================
// nnueAccCopySubAddN(dst *int16, src *int16, oldW *int16, newW *int16, count int)
// Computes: dst[i] = src[i] + newW[i] - oldW[i] for i = 0..count-1
// count must be a multiple of 16.
// ============================================================================
TEXT ·nnueAccCopySubAddN(SB), NOSPLIT, $0-40
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD oldW+16(FP), R2
	MOVD newW+24(FP), R3
	MOVD count+32(FP), R4
	LSR $4, R4, R4

copysubaddn_loop:
	VLD1 (R3), [V0.B16]          // new weights
	VLD1 (R2), [V1.B16]          // old weights
	WORD $0x6E618400              // SUB V0.8H, V0.8H, V1.8H  (new - old)
	VLD1 (R1), [V1.B16]          // src (parent accumulator)
	WORD $0x4E608420              // ADD V0.8H, V1.8H, V0.8H  (src + delta)
	VST1 [V0.B16], (R0)          // store to dst
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	VLD1 (R3), [V2.B16]
	VLD1 (R2), [V3.B16]
	WORD $0x6E638442              // SUB V2.8H, V2.8H, V3.8H
	VLD1 (R1), [V3.B16]
	WORD $0x4E628460              // ADD V0.8H, V3.8H, V2.8H
	VST1 [V0.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	SUBS $1, R4, R4
	BNE copysubaddn_loop

	RET

// ============================================================================
// nnueAccCopySubSubAddN(dst *int16, src *int16, oldW *int16, newW *int16, capW *int16, count int)
// Computes: dst[i] = src[i] + newW[i] - oldW[i] - capW[i] for i = 0..count-1
// count must be a multiple of 16.
// ============================================================================
TEXT ·nnueAccCopySubSubAddN(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD oldW+16(FP), R2
	MOVD newW+24(FP), R3
	MOVD capW+32(FP), R4
	MOVD count+40(FP), R5
	LSR $4, R5, R5

copysubsubaddn_loop:
	VLD1 (R3), [V0.B16]          // new weights
	VLD1 (R2), [V1.B16]          // old weights
	WORD $0x6E618400              // SUB V0.8H, V0.8H, V1.8H  (new - old)
	VLD1 (R4), [V1.B16]          // capture weights
	WORD $0x6E618400              // SUB V0.8H, V0.8H, V1.8H  (new - old - cap)
	VLD1 (R1), [V1.B16]          // src (parent accumulator)
	WORD $0x4E608420              // ADD V0.8H, V1.8H, V0.8H  (src + delta)
	VST1 [V0.B16], (R0)          // store to dst
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	ADD $16, R4, R4
	VLD1 (R3), [V2.B16]
	VLD1 (R2), [V3.B16]
	WORD $0x6E638442              // SUB V2.8H, V2.8H, V3.8H
	VLD1 (R4), [V3.B16]
	WORD $0x6E638442              // SUB V2.8H, V2.8H, V3.8H
	VLD1 (R1), [V3.B16]
	WORD $0x4E628460              // ADD V0.8H, V3.8H, V2.8H
	VST1 [V0.B16], (R0)
	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	ADD $16, R3, R3
	ADD $16, R4, R4
	SUBS $1, R5, R5
	BNE copysubsubaddn_loop

	RET

// ============================================================================
// nnueV5CReLUDot1024(acc *int16, weights *int16) int32
//
// Computes: sum = sum_i( clamp(acc[i], 0, 255) * weights[i] ) for i=0..1023
// Uses SMAX/SMIN for clamping, SMULL/SMLAL2 for multiply-accumulate.
// 1024 elements / 16 per iteration (2x unrolled) = 64 iterations.
// ============================================================================
TEXT ·nnueV5CReLUDot1024(SB), NOSPLIT, $0-24
	MOVD acc+0(FP), R0
	MOVD weights+8(FP), R1

	VEOR V0.B16, V0.B16, V0.B16  // V0 = zero (floor)
	MOVD $255, R3
	WORD $0x4E020C61              // DUP V1.8H, W3 (255 broadcast)
	VEOR V16.B16, V16.B16, V16.B16  // int32 accumulator 1
	VEOR V17.B16, V17.B16, V17.B16  // int32 accumulator 2

	MOVD $64, R2                  // 64 iterations

v5crelu1024_loop:
	// First 8 elements
	VLD1 (R0), [V2.B16]          // load 8 acc int16
	WORD $0x4E606442              // SMAX V2.8H, V2.8H, V0.8H
	WORD $0x4E616C42              // SMIN V2.8H, V2.8H, V1.8H
	VLD1 (R1), [V3.B16]          // load 8 weights
	WORD $0x0E63C044              // SMULL V4.4S, V2.4H, V3.4H
	WORD $0x4E638044              // SMLAL2 V4.4S, V2.8H, V3.8H
	WORD $0x4EA48610              // ADD V16.4S, V16.4S, V4.4S
	ADD $16, R0, R0
	ADD $16, R1, R1
	// Second 8 elements
	VLD1 (R0), [V2.B16]
	WORD $0x4E606442              // SMAX V2.8H, V2.8H, V0.8H
	WORD $0x4E616C42              // SMIN V2.8H, V2.8H, V1.8H
	VLD1 (R1), [V3.B16]
	WORD $0x0E63C044              // SMULL V4.4S, V2.4H, V3.4H
	WORD $0x4E638044              // SMLAL2 V4.4S, V2.8H, V3.8H
	WORD $0x4EA48631              // ADD V17.4S, V17.4S, V4.4S
	ADD $16, R0, R0
	ADD $16, R1, R1
	SUBS $1, R2, R2
	BNE v5crelu1024_loop

	// Merge accumulators
	WORD $0x4EB18610              // ADD V16.4S, V16.4S, V17.4S

	// Horizontal sum: V16.4S -> scalar
	WORD $0x4EB0BE10              // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x4EB0BE10              // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x1E26020C              // FMOV W12, S16
	MOVW R12, ret+16(FP)

	RET

// ============================================================================
// nnueV5CReLUDotN(acc *int16, weights *int16, count int) int32
//
// Generic width CReLU dot product. count must be a multiple of 16.
// 2x unrolled, count/16 iterations.
// ============================================================================
TEXT ·nnueV5CReLUDotN(SB), NOSPLIT, $0-28
	MOVD acc+0(FP), R0
	MOVD weights+8(FP), R1
	MOVD count+16(FP), R2
	LSR $4, R2, R2                // count / 16

	VEOR V0.B16, V0.B16, V0.B16
	MOVD $255, R3
	WORD $0x4E020C61              // DUP V1.8H, W3
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16

v5creludotn_loop:
	// First 8 elements
	VLD1 (R0), [V2.B16]
	WORD $0x4E606442              // SMAX V2.8H, V2.8H, V0.8H
	WORD $0x4E616C42              // SMIN V2.8H, V2.8H, V1.8H
	VLD1 (R1), [V3.B16]
	WORD $0x0E63C044              // SMULL V4.4S, V2.4H, V3.4H
	WORD $0x4E638044              // SMLAL2 V4.4S, V2.8H, V3.8H
	WORD $0x4EA48610              // ADD V16.4S, V16.4S, V4.4S
	ADD $16, R0, R0
	ADD $16, R1, R1
	// Second 8 elements
	VLD1 (R0), [V2.B16]
	WORD $0x4E606442              // SMAX V2.8H, V2.8H, V0.8H
	WORD $0x4E616C42              // SMIN V2.8H, V2.8H, V1.8H
	VLD1 (R1), [V3.B16]
	WORD $0x0E63C044              // SMULL V4.4S, V2.4H, V3.4H
	WORD $0x4E638044              // SMLAL2 V4.4S, V2.8H, V3.8H
	WORD $0x4EA48631              // ADD V17.4S, V17.4S, V4.4S
	ADD $16, R0, R0
	ADD $16, R1, R1
	SUBS $1, R2, R2
	BNE v5creludotn_loop

	WORD $0x4EB18610              // ADD V16.4S, V16.4S, V17.4S
	WORD $0x4EB0BE10              // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x4EB0BE10              // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x1E26020C              // FMOV W12, S16
	MOVW R12, ret+24(FP)

	RET

// ============================================================================
// nnueV5SCReLUDotN(acc *int16, weights *int16, count int) int64
//
// Exact SCReLU dot product for any width (multiple of 16).
// Computes: sum = sum_i( clamp(acc[i], 0, 255)^2 * weights[i] )
// Returns int64 (caller divides by QA=255).
//
// Per 8 elements: clamp, widen to int32, square, multiply by widened weights,
// widen to int64 and accumulate.
//
// Register allocation:
//   R0   = acc pointer
//   R1   = weights pointer
//   R2   = loop counter (count/8)
//   V0   = zero
//   V1   = 255 broadcast
//   V2   = loaded acc values (8 int16)
//   V3   = loaded weights (8 int16)
//   V4,V5 = widened values (int32)
//   V6   = product (int32)
//   V7   = scratch for widening
//   V16-V19 = int64 accumulators
// ============================================================================
TEXT ·nnueV5SCReLUDotN(SB), NOSPLIT, $0-32
	MOVD acc+0(FP), R0
	MOVD weights+8(FP), R1
	MOVD count+16(FP), R2
	LSR $3, R2, R2                // count / 8

	VEOR V0.B16, V0.B16, V0.B16
	MOVD $255, R3
	WORD $0x4E020C61              // DUP V1.8H, W3
	VEOR V16.B16, V16.B16, V16.B16  // int64 acc 0
	VEOR V17.B16, V17.B16, V17.B16  // int64 acc 1
	VEOR V18.B16, V18.B16, V18.B16  // int64 acc 2
	VEOR V19.B16, V19.B16, V19.B16  // int64 acc 3

v5screludotn_loop:
	VLD1 (R0), [V2.B16]          // load 8 acc int16
	WORD $0x4E606442              // SMAX V2.8H, V2.8H, V0.8H
	WORD $0x4E616C42              // SMIN V2.8H, V2.8H, V1.8H
	VLD1 (R1), [V3.B16]          // load 8 weights

	// --- Low 4 elements ---
	WORD $0x0F10A444              // SSHLL V4.4S, V2.4H, #0  (x[0..3] -> int32)
	WORD $0x0F10A465              // SSHLL V5.4S, V3.4H, #0  (w[0..3] -> int32)
	WORD $0x4EA49C86              // MUL V6.4S, V4.4S, V4.4S (x^2)
	WORD $0x4EA59CC6              // MUL V6.4S, V6.4S, V5.4S (x^2 * w)
	WORD $0x0F20A4C7              // SSHLL V7.2D, V6.2S, #0  (low 2 -> int64)
	WORD $0x4EE78610              // ADD V16.2D, V16.2D, V7.2D
	WORD $0x4F20A4C7              // SSHLL2 V7.2D, V6.4S, #0 (high 2 -> int64)
	WORD $0x4EE78631              // ADD V17.2D, V17.2D, V7.2D

	// --- High 4 elements ---
	WORD $0x4F10A444              // SSHLL2 V4.4S, V2.8H, #0 (x[4..7] -> int32)
	WORD $0x4F10A465              // SSHLL2 V5.4S, V3.8H, #0 (w[4..7] -> int32)
	WORD $0x4EA49C86              // MUL V6.4S, V4.4S, V4.4S
	WORD $0x4EA59CC6              // MUL V6.4S, V6.4S, V5.4S
	WORD $0x0F20A4C7              // SSHLL V7.2D, V6.2S, #0
	WORD $0x4EE78652              // ADD V18.2D, V18.2D, V7.2D
	WORD $0x4F20A4C7              // SSHLL2 V7.2D, V6.4S, #0
	WORD $0x4EE78673              // ADD V19.2D, V19.2D, V7.2D

	ADD $16, R0, R0
	ADD $16, R1, R1
	SUBS $1, R2, R2
	BNE v5screludotn_loop

	// Horizontal sum: V16+V17+V18+V19 -> single int64
	WORD $0x4EF18610              // ADD V16.2D, V16.2D, V17.2D
	WORD $0x4EF38652              // ADD V18.2D, V18.2D, V19.2D
	WORD $0x4EF28610              // ADD V16.2D, V16.2D, V18.2D
	WORD $0x4EF0BE10              // ADDP V16.2D, V16.2D, V16.2D
	WORD $0x9E66020C              // FMOV X12, D16
	MOVD R12, ret+24(FP)

	RET

// ============================================================================
// nnueV5PairwiseDotN(accFirst *int16, accSecond *int16, weights *int16, count int) int64
//
// Pairwise dot product for v5 768pw architecture.
// Computes: sum = sum_i( clamp(a[i],0,255) * clamp(b[i],0,255) * weights[i] )
// for i=0..count-1. Returns int64 (caller divides by QA=255).
//
// Register allocation:
//   R0   = accFirst pointer
//   R1   = accSecond pointer
//   R2   = weights pointer
//   R3   = loop counter (count/8)
//   V0   = zero
//   V1   = 255 broadcast
//   V2   = clamped a
//   V3   = clamped b
//   V5   = weights
//   V4,V6,V7,V8 = scratch
//   V16-V19 = int64 accumulators
// ============================================================================
TEXT ·nnueV5PairwiseDotN(SB), NOSPLIT, $0-40
	MOVD accFirst+0(FP), R0
	MOVD accSecond+8(FP), R1
	MOVD weights+16(FP), R2
	MOVD count+24(FP), R3
	LSR $3, R3, R3                // count / 8

	VEOR V0.B16, V0.B16, V0.B16
	MOVD $255, R4
	WORD $0x4E020C81              // DUP V1.8H, W4
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16

v5pwdotn_loop:
	// Load and clamp a
	VLD1 (R0), [V2.B16]
	WORD $0x4E606442              // SMAX V2.8H, V2.8H, V0.8H
	WORD $0x4E616C42              // SMIN V2.8H, V2.8H, V1.8H
	// Load and clamp b
	VLD1 (R1), [V3.B16]
	WORD $0x4E606463              // SMAX V3.8H, V3.8H, V0.8H
	WORD $0x4E616C63              // SMIN V3.8H, V3.8H, V1.8H
	// Load weights
	VLD1 (R2), [V5.B16]

	// --- Low 4 elements ---
	WORD $0x0F10A444              // SSHLL V4.4S, V2.4H, #0  (a[0..3] -> int32)
	WORD $0x0F10A466              // SSHLL V6.4S, V3.4H, #0  (b[0..3] -> int32)
	WORD $0x0F10A4A7              // SSHLL V7.4S, V5.4H, #0  (w[0..3] -> int32)
	WORD $0x4EA69C84              // MUL V4.4S, V4.4S, V6.4S (a*b)
	WORD $0x4EA79C84              // MUL V4.4S, V4.4S, V7.4S (a*b*w)
	WORD $0x0F20A488              // SSHLL V8.2D, V4.2S, #0  (low 2 -> int64)
	WORD $0x4EE88610              // ADD V16.2D, V16.2D, V8.2D
	WORD $0x4F20A488              // SSHLL2 V8.2D, V4.4S, #0 (high 2 -> int64)
	WORD $0x4EE88631              // ADD V17.2D, V17.2D, V8.2D

	// --- High 4 elements ---
	WORD $0x4F10A444              // SSHLL2 V4.4S, V2.8H, #0 (a[4..7] -> int32)
	WORD $0x4F10A466              // SSHLL2 V6.4S, V3.8H, #0 (b[4..7] -> int32)
	WORD $0x4F10A4A7              // SSHLL2 V7.4S, V5.8H, #0 (w[4..7] -> int32)
	WORD $0x4EA69C84              // MUL V4.4S, V4.4S, V6.4S
	WORD $0x4EA79C84              // MUL V4.4S, V4.4S, V7.4S
	WORD $0x0F20A488              // SSHLL V8.2D, V4.2S, #0
	WORD $0x4EE88652              // ADD V18.2D, V18.2D, V8.2D
	WORD $0x4F20A488              // SSHLL2 V8.2D, V4.4S, #0
	WORD $0x4EE88673              // ADD V19.2D, V19.2D, V8.2D

	ADD $16, R0, R0
	ADD $16, R1, R1
	ADD $16, R2, R2
	SUBS $1, R3, R3
	BNE v5pwdotn_loop

	// Horizontal sum
	WORD $0x4EF18610              // ADD V16.2D, V16.2D, V17.2D
	WORD $0x4EF38652              // ADD V18.2D, V18.2D, V19.2D
	WORD $0x4EF28610              // ADD V16.2D, V16.2D, V18.2D
	WORD $0x4EF0BE10              // ADDP V16.2D, V16.2D, V16.2D
	WORD $0x9E66020C              // FMOV X12, D16
	MOVD R12, ret+32(FP)

	RET

// ============================================================================
// nnueV5L1MatMulN(acc *int16, wT *int16, hidden *int32, accLen int, l1 int)
//
// L1 hidden layer matmul for one perspective (called twice: stm + ntm).
// Computes: hidden[i] += sum_j( clamp(acc[j], 0, 255) * wT[i*accLen + j] )
// for i=0..l1-1, j=0..accLen-1.
// accLen must be a multiple of 16. l1 can be any positive value.
//
// Uses transposed weight layout [l1][accLen].
// For each L1 neuron: CReLU dot product using SMULL/SMLAL2.
//
// Register allocation:
//   R0  = acc base pointer
//   R1  = weight pointer (advances through rows)
//   R2  = hidden pointer (advances per neuron)
//   R3  = inner loop count (accLen / 16)
//   R4  = outer loop count (l1)
//   R5  = weight row stride in bytes (accLen * 2)
//   R6  = acc cursor (reset per neuron)
//   R7  = weight cursor
//   R8  = inner loop counter
//   V0  = zero (CReLU floor)
//   V1  = 255 (CReLU ceiling)
//   V16, V17 = int32 accumulators
// ============================================================================
TEXT ·nnueV5L1MatMulN(SB), NOSPLIT, $0-40
	MOVD acc+0(FP), R0
	MOVD wT+8(FP), R1
	MOVD hidden+16(FP), R2
	MOVD accLen+24(FP), R3
	MOVD l1+32(FP), R4

	// Weight row stride in bytes
	LSL $1, R3, R5                       // R5 = accLen * 2

	// Inner loop count
	LSR $4, R3, R3                       // R3 = accLen / 16

	// Constants
	VEOR V0.B16, V0.B16, V0.B16         // V0 = zero
	MOVD $255, R9
	WORD $0x4E020D21                     // DUP V1.8H, W9   (broadcast 255)

l1mm_neon_outer:
	VEOR V16.B16, V16.B16, V16.B16      // accumulator 1
	VEOR V17.B16, V17.B16, V17.B16      // accumulator 2
	MOVD R0, R6                          // R6 = acc cursor
	MOVD R1, R7                          // R7 = weight cursor
	MOVD R3, R8                          // R8 = inner loop counter

l1mm_neon_inner:
	// First 8 elements
	VLD1 (R6), [V2.B16]                 // load acc[j:j+8]
	WORD $0x4E606442                     // SMAX V2.8H, V2.8H, V0.8H
	WORD $0x4E616C42                     // SMIN V2.8H, V2.8H, V1.8H
	VLD1 (R7), [V3.B16]                 // load wT[i*accLen+j:+8]
	WORD $0x0E63C044                     // SMULL V4.4S, V2.4H, V3.4H
	WORD $0x4E638044                     // SMLAL2 V4.4S, V2.8H, V3.8H
	WORD $0x4EA48610                     // ADD V16.4S, V16.4S, V4.4S
	ADD $16, R6, R6
	ADD $16, R7, R7

	// Second 8 elements
	VLD1 (R6), [V2.B16]
	WORD $0x4E606442                     // SMAX V2.8H, V2.8H, V0.8H
	WORD $0x4E616C42                     // SMIN V2.8H, V2.8H, V1.8H
	VLD1 (R7), [V3.B16]
	WORD $0x0E63C044                     // SMULL V4.4S, V2.4H, V3.4H
	WORD $0x4E638044                     // SMLAL2 V4.4S, V2.8H, V3.8H
	WORD $0x4EA48631                     // ADD V17.4S, V17.4S, V4.4S
	ADD $16, R6, R6
	ADD $16, R7, R7

	SUBS $1, R8, R8
	BNE l1mm_neon_inner

	// Horizontal sum: V16 + V17 → scalar
	WORD $0x4EB18610                     // ADD V16.4S, V16.4S, V17.4S
	WORD $0x4EB0BE10                     // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x4EB0BE10                     // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x1E26020C                     // FMOV W12, S16

	// Accumulate into hidden[i]
	MOVW (R2), R9
	ADDW R12, R9, R9
	MOVW R9, (R2)

	ADD $4, R2, R2                       // next hidden slot
	ADD R5, R1, R1                       // next weight row
	SUBS $1, R4, R4
	BNE l1mm_neon_outer

	RET

// ============================================================================
// nnueSCReLUPack(src *int16, dst *byte, count int)
// Packs int16 to uint8 with SCReLU: dst[i] = clamp(src[i],0,255)²/255
// count must be a multiple of 16. Processes 8 elements at a time.
// ============================================================================
TEXT ·nnueSCReLUPack(SB), NOSPLIT, $0-24
	MOVD src+0(FP), R0
	MOVD dst+8(FP), R1
	MOVD count+16(FP), R2
	LSR $3, R2, R2                       // count / 8

	VEOR V0.B16, V0.B16, V0.B16         // zero
	MOVD $255, R3
	WORD $0x4E020C61                     // DUP V1.8H, W3 (broadcast 255)

screlu_pack_neon_loop:
	// Load 8 int16 values
	VLD1 (R0), [V2.B16]

	// Clamp [0, 255]
	WORD $0x4E606442                     // SMAX V2.8H, V2.8H, V0.8H
	WORD $0x4E616C42                     // SMIN V2.8H, V2.8H, V1.8H

	// Square: v*v → need to widen to avoid overflow
	// v is in [0,255] as int16, v*v max=65025 fits uint16
	WORD $0x4E620042                     // MUL V2.8H, V2.8H, V2.8H (v²)

	// Divide by 255: approximate as (v² + 128) >> 8 + (v² >> 16)
	// Actually simpler: use UMULL to widen, then take high half
	// Or: since we have v² as uint16 and need v²/255 as uint8,
	// use: (v² * 257) >> 16 via UMULL + high half extraction
	// But NEON doesn't have PMULHUW directly. Use UMULL for 4 at a time.

	// Low 4: widen v²[0:3] to uint32, multiply by 257, take bits [16:23]
	WORD $0x2F20A043                     // USHLL V3.4S, V2.4H, #0 (v²[0:3] → uint32)
	MOVD $257, R4
	WORD $0x4E040C84                     // DUP V4.4S, W4 (broadcast 257)
	WORD $0x4EA49C63                     // MUL V3.4S, V3.4S, V4.4S (v² * 257)
	WORD $0x4F300463                     // USHR V3.4S, V3.4S, #16 (>> 16)

	// High 4: same for v²[4:7]
	WORD $0x6F20A045                     // USHLL2 V5.4S, V2.8H, #0
	WORD $0x4EA49CA5                     // MUL V5.4S, V5.4S, V4.4S
	WORD $0x4F3004A5                     // USHR V5.4S, V5.4S, #16

	// Narrow back: uint32 → uint16 → uint8
	WORD $0x0E612863                     // XTN V3.4H, V3.4S
	WORD $0x4E6128A3                     // XTN2 V3.8H, V5.4S
	WORD $0x0E212863                     // XTN V3.8B, V3.8H

	// Store 8 uint8 values
	WORD $0x0D008023                     // ST1 {V3.8B}, [R1]

	ADD $16, R0, R0                      // 8 × int16 = 16 bytes
	ADD $8, R1, R1                       // 8 × uint8 = 8 bytes
	SUBS $1, R2, R2
	BNE screlu_pack_neon_loop

	RET

// ============================================================================
// nnueCReLUPackN(src *int16, dst *byte, count int)
// CReLU pack: clamp [0,255] and narrow to uint8. count must be multiple of 16.
// ============================================================================
TEXT ·nnueCReLUPackN(SB), NOSPLIT, $0-24
	MOVD src+0(FP), R0
	MOVD dst+8(FP), R1
	MOVD count+16(FP), R2
	LSR $4, R2, R2                       // count / 16

	VEOR V0.B16, V0.B16, V0.B16
	MOVD $255, R3
	WORD $0x4E020C61                     // DUP V1.8H, W3

crelu_packn_neon_loop:
	VLD1 (R0), [V2.B16]
	WORD $0x4E606442                     // SMAX V2.8H, V2.8H, V0.8H
	WORD $0x4E616C42                     // SMIN V2.8H, V2.8H, V1.8H
	// Narrow int16 → uint8 (saturating)
	WORD $0x0E212842                     // XTN V2.8B, V2.8H
	WORD $0x0D008022                     // ST1 {V2.8B}, [R1]
	ADD $16, R0, R0
	ADD $8, R1, R1
	SUBS $1, R2, R2
	BNE crelu_packn_neon_loop
	RET

// ============================================================================
// nnueInt8DotN(a *byte, b *int8, count int) int32
// Dot product u8 × i8 → i32. count must be multiple of 16.
// ============================================================================
TEXT ·nnueInt8DotN(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD count+16(FP), R2
	LSR $4, R2, R2

	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16

i8dot_neon_loop:
	VLD1 (R0), [V2.B16]                 // 16 x uint8
	VLD1 (R1), [V3.B16]                 // 16 x int8
	// Widen and multiply low 8
	WORD $0x2F08A444                     // USHLL V4.8H, V2.8B, #0
	WORD $0x0F08A465                     // SSHLL V5.8H, V3.8B, #0
	WORD $0x0E65C086                     // SMULL V6.4S, V4.4H, V5.4H
	WORD $0x4E658086                     // SMLAL2 V6.4S, V4.8H, V5.8H
	WORD $0x4EA68610                     // ADD V16.4S, V16.4S, V6.4S
	// High 8
	WORD $0x6F08A444                     // USHLL2 V4.8H, V2.16B, #0
	WORD $0x4F08A465                     // SSHLL2 V5.8H, V3.16B, #0
	WORD $0x0E65C086                     // SMULL V6.4S, V4.4H, V5.4H
	WORD $0x4E658086                     // SMLAL2 V6.4S, V4.8H, V5.8H
	WORD $0x4EA68631                     // ADD V17.4S, V17.4S, V6.4S

	ADD $16, R0, R0
	ADD $16, R1, R1
	SUBS $1, R2, R2
	BNE i8dot_neon_loop

	WORD $0x4EB18610                     // ADD V16.4S, V16.4S, V17.4S
	WORD $0x4EB0BE10                     // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x4EB0BE10                     // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x1E26020C                     // FMOV W12, S16
	MOVW R12, ret+24(FP)
	RET

// ============================================================================
// nnueV5L1Int8MatMulN(acc8 *byte, wT8 *int8, hidden *int32, accLen int, l1 int)
//
// Int8 L1 matmul using NEON. Processes 16 bytes at a time.
// Widens u8→i16, i8→i16, multiplies i16, accumulates i32.
// accLen must be a multiple of 16.
// ============================================================================
TEXT ·nnueV5L1Int8MatMulN(SB), NOSPLIT, $0-40
	MOVD acc8+0(FP), R0
	MOVD wT8+8(FP), R1
	MOVD hidden+16(FP), R2
	MOVD accLen+24(FP), R3
	MOVD l1+32(FP), R4

	// Weight row stride = accLen bytes
	MOVD R3, R5

	// Inner loop count: accLen / 16
	LSR $4, R3, R3

l1i8_neon_outer:
	VEOR V16.B16, V16.B16, V16.B16      // int32 accumulator
	VEOR V17.B16, V17.B16, V17.B16
	MOVD R0, R6                          // acc cursor
	MOVD R1, R7                          // weight cursor
	MOVD R3, R8                          // inner loop counter

l1i8_neon_inner:
	// Load 16 uint8 acc values and 16 int8 weights
	VLD1 (R6), [V2.B16]                 // 16 x uint8
	VLD1 (R7), [V3.B16]                 // 16 x int8

	// Widen u8 → i16 (unsigned), i8 → i16 (signed)
	WORD $0x2F08A444                     // USHLL V4.8H, V2.8B, #0 (low 8 u8 → u16)
	WORD $0x0F08A465                     // SSHLL V5.8H, V3.8B, #0 (low 8 i8 → i16)

	// Multiply i16 × i16, accumulate to i32
	WORD $0x0E65C086                     // SMULL V6.4S, V4.4H, V5.4H
	WORD $0x4E658086                     // SMLAL2 V6.4S, V4.8H, V5.8H
	WORD $0x4EA68610                     // ADD V16.4S, V16.4S, V6.4S

	// High 8 elements
	WORD $0x6F08A444                     // USHLL2 V4.8H, V2.16B, #0 (high 8 u8 → u16)
	WORD $0x4F08A465                     // SSHLL2 V5.8H, V3.16B, #0 (high 8 i8 → i16)
	WORD $0x0E65C086                     // SMULL V6.4S, V4.4H, V5.4H
	WORD $0x4E658086                     // SMLAL2 V6.4S, V4.8H, V5.8H
	WORD $0x4EA68631                     // ADD V17.4S, V17.4S, V6.4S

	ADD $16, R6, R6
	ADD $16, R7, R7
	SUBS $1, R8, R8
	BNE l1i8_neon_inner

	// Horizontal sum
	WORD $0x4EB18610                     // ADD V16.4S, V16.4S, V17.4S
	WORD $0x4EB0BE10                     // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x4EB0BE10                     // ADDP V16.4S, V16.4S, V16.4S
	WORD $0x1E26020C                     // FMOV W12, S16

	// Accumulate into hidden[i]
	MOVW (R2), R9
	ADDW R12, R9, R9
	MOVW R9, (R2)

	ADD $4, R2, R2
	ADD R5, R1, R1
	SUBS $1, R4, R4
	BNE l1i8_neon_outer

	RET

// ============================================================================
// nnueV5L1SCReLUMatMulN(acc *int16, wT *int16, hidden *int64, accLen int, l1 int)
//
// SCReLU L1 matmul: hidden[i] += sum_j( clamp(acc[j], 0, 255)² * wT[i*accLen+j] )
// Processes 4 elements at a time: widen to int32, square, multiply, accumulate int64.
// accLen must be a multiple of 4. l1 can be any positive value.
//
// Register allocation:
//   R0 = acc base, R1 = weight pointer, R2 = hidden pointer
//   R3 = inner loop count (accLen/4), R4 = outer loop count (l1)
//   R5 = weight row stride, V0 = zero, V1 = 255
//   V16-V17 = int64 accumulators
// ============================================================================
TEXT ·nnueV5L1SCReLUMatMulN(SB), NOSPLIT, $0-40
	MOVD acc+0(FP), R0
	MOVD wT+8(FP), R1
	MOVD hidden+16(FP), R2
	MOVD accLen+24(FP), R3
	MOVD l1+32(FP), R4

	// Weight row stride in bytes
	LSL $1, R3, R5                       // R5 = accLen * 2

	// Inner loop count: accLen / 4
	LSR $2, R3, R3

	// Constants
	VEOR V0.B16, V0.B16, V0.B16         // V0 = zero
	MOVD $255, R9
	WORD $0x4E040D21                     // DUP V1.4S, W9   (broadcast 255 as int32)

l1screlu_neon_outer:
	VEOR V16.B16, V16.B16, V16.B16      // int64 accumulator low
	VEOR V17.B16, V17.B16, V17.B16      // int64 accumulator high
	MOVD R0, R6                          // acc cursor
	MOVD R1, R7                          // weight cursor
	MOVD R3, R8                          // inner loop counter

l1screlu_neon_inner:
	// Load 4 acc values as int16, widen to int32
	MOVD (R6), R10
	WORD $0x4E080D42                     // DUP V2.2D, X10 (load 8 bytes into V2)
	WORD $0x0F10A442                     // SSHLL V2.4S, V2.4H, #0 (widen int16 → int32)

	// Clamp to [0, 255]
	WORD $0x4EA06442                     // SMAX V2.4S, V2.4S, V0.4S
	WORD $0x4EA16C42                     // SMIN V2.4S, V2.4S, V1.4S

	// Square: V3 = V2 * V2
	WORD $0x4EA29C43                     // MUL V3.4S, V2.4S, V2.4S

	// Load 4 weights as int16, widen to int32
	MOVD (R7), R10
	WORD $0x4E080D44                     // DUP V4.2D, X10
	WORD $0x0F10A484                     // SSHLL V4.4S, V4.4H, #0

	// Multiply: V5 = v² × w (int32)
	WORD $0x4EA49C65                     // MUL V5.4S, V3.4S, V4.4S

	// Widen to int64 and accumulate
	WORD $0x0F20A4A6                     // SSHLL V6.2D, V5.2S, #0 (low 2 → int64)
	WORD $0x4EA68610                     // ADD V16.2D, V16.2D, V6.2D
	WORD $0x4F20A4A6                     // SSHLL2 V6.2D, V5.4S, #0 (high 2 → int64)
	WORD $0x4EA68631                     // ADD V17.2D, V17.2D, V6.2D

	ADD $8, R6, R6                       // 4 × int16 = 8 bytes
	ADD $8, R7, R7
	SUBS $1, R8, R8
	BNE l1screlu_neon_inner

	// Horizontal sum: V16 + V17 → single int64
	WORD $0x4EF18610                     // ADD V16.2D, V16.2D, V17.2D
	WORD $0x4EF0BE10                     // ADDP V16.2D, V16.2D, V16.2D
	WORD $0x9E66020C                     // FMOV X12, D16

	// Accumulate into hidden[i] (int64)
	MOVD (R2), R9
	ADD R12, R9, R9
	MOVD R9, (R2)

	ADD $8, R2, R2                       // next hidden slot (int64)
	ADD R5, R1, R1                       // next weight row
	SUBS $1, R4, R4
	BNE l1screlu_neon_outer

	RET

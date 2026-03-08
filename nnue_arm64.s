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

	MOVD $32, R3                  // 32 iterations

crelu_loop:
	VLD1 (R0), [V2.B16]          // load 8 int16
	WORD $0x4E606442              // SMAX V2.8H, V2.8H, V0.8H  (max with 0)
	WORD $0x4E616C42              // SMIN V2.8H, V2.8H, V1.8H  (min with 127)
	VST1 [V2.B16], (R1)          // store result
	ADD $16, R0, R0
	ADD $16, R1, R1
	SUBS $1, R3, R3
	BNE crelu_loop

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

	// Output j: multiply-accumulate into V16
	VLD1 (R6), [V4.B16]
	WORD $0x0E64C014              // SMULL  V20.4S, V0.4H, V4.4H
	WORD $0x4E648014              // SMLAL2 V20.4S, V0.8H, V4.8H
	WORD $0x4EB48610              // ADD    V16.4S, V16.4S, V20.4S

	// Output j+1: multiply-accumulate into V17
	VLD1 (R7), [V5.B16]
	WORD $0x0E65C015              // SMULL  V21.4S, V0.4H, V5.4H
	WORD $0x4E658015              // SMLAL2 V21.4S, V0.8H, V5.8H
	WORD $0x4EB58631              // ADD    V17.4S, V17.4S, V21.4S

	// Output j+2: multiply-accumulate into V18
	VLD1 (R8), [V6.B16]
	WORD $0x0E66C016              // SMULL  V22.4S, V0.4H, V6.4H
	WORD $0x4E668016              // SMLAL2 V22.4S, V0.8H, V6.8H
	WORD $0x4EB68652              // ADD    V18.4S, V18.4S, V22.4S

	// Output j+3: multiply-accumulate into V19
	VLD1 (R9), [V7.B16]
	WORD $0x0E67C017              // SMULL  V23.4S, V0.4H, V7.4H
	WORD $0x4E678017              // SMLAL2 V23.4S, V0.8H, V7.8H
	WORD $0x4EB78673              // ADD    V19.4S, V19.4S, V23.4S

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
	MOVD $32, R3

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
	MOVD $32, R4

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
	MOVD $32, R2

accadd_loop:
	VLD1 (R0), [V0.B16]          // load accumulator
	VLD1 (R1), [V1.B16]          // load weights
	WORD $0x4E618400              // ADD V0.8H, V0.8H, V1.8H
	VST1 [V0.B16], (R0)          // store back
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
	MOVD $32, R2

accsub_loop:
	VLD1 (R0), [V0.B16]          // load accumulator
	VLD1 (R1), [V1.B16]          // load weights
	WORD $0x6E618400              // SUB V0.8H, V0.8H, V1.8H
	VST1 [V0.B16], (R0)          // store back
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
	MOVD $32, R4

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
	MOVD $32, R5

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
	SUBS $1, R5, R5
	BNE copysubsubadd_loop

	RET

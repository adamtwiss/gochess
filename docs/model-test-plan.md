# NNUE Model Test Plan

## Motivation

We've explored many model dimensions (activation, width, WDL, training length, data diversity) across multiple GPUs over several days. However, the results are hard to interpret because:

1. **Confounding variables**: Different GPUs used different data files (1 vs 6 T80 files), different Bullet versions, different LR schedules
2. **Naming inconsistency**: Filenames didn't distinguish cosine schedule length from snapshot point (e.g. sb400 from an e800 run ≠ sb400 from an e400 run)
3. **Search changes during testing**: The engine binary changed significantly during model testing (Finny tables, multi-corr, pruning changes and reverts)
4. **Small cross-engine margins**: Most model changes are within ±10 Elo of production cross-engine, making it hard to distinguish signal from noise

We need a systematic, controlled experiment series to establish ground truth.

## Controlled Variables (same for ALL runs)

- **Data**: All 6 T80 2024 .min-v2.v6 binpack files via `new_concat_multiple`
- **Bullet version**: Current HEAD (pin exact commit)
- **Batch size**: 16384
- **Batches per SB**: 6104
- **Threads**: 16, batch_queue_size: 128
- **SavedFormat**: NO .transpose()
- **Naming convention**: `net-v5-{width}{activation}-w{wdl}-e{epochs}s{snap}.nnue` (per CLAUDE.md)
- **Validation**: Run `check-net` on every checkpoint, record loss at each snapshot

## Testing Binary

All SPRT tests use a **frozen binary** built from a specific commit. No search changes during the model test phase. Record the commit hash.

## Phase 1: Establish Baseline (3 runs)

**Goal**: Confirm our production baseline and test if 1024 CReLU benefits from more training.

| Run | Config | Hypothesis | Priority |
|-----|--------|-----------|----------|
| A1 | 1024 CReLU w0 e120 | Reproduce production baseline — should match within ±5 Elo | HIGH |
| A2 | 1024 CReLU w0 e400 | Does more training help at 1024? Previous test said no (sb200=sb120) but Bullet version changed | HIGH |
| A3 | 1024 CReLU w0 e400 (different seed) | How much does random seed matter? Establishes noise floor | MEDIUM |

**SPRT tests**: A1 vs production (expect neutral), A2 vs A1 (expect neutral or slight positive), A3 vs A2 (measures seed variation)

## Phase 2: Activation Function (2 runs)

**Goal**: Does SCReLU help at 1024? Controlled comparison.

| Run | Config | Hypothesis | Priority |
|-----|--------|-----------|----------|
| B1 | 1024 SCReLU w0 e400 | SCReLU gets lower loss (0.00884 vs 0.00925 at e200). At e400 it should converge and may beat CReLU | HIGH |
| B2 | 1024 SCReLU w0 e200 | Matches our existing SCReLU test — confirms reproducibility | LOW |

**SPRT tests**: B1 vs A2 (isolates SCReLU effect at same training length)

## Phase 3: WDL Blend (3 runs)

**Goal**: Find optimal WDL. We think 0.05 > 0.1 > 0.0 in self-play but cross-engine data is ambiguous.

| Run | Config | Hypothesis | Priority |
|-----|--------|-----------|----------|
| C1 | 1024 CReLU w5 e400 | wdl=0.05 was +44 self-play at e200. At e400 with controlled setup, should be clearer | HIGH |
| C2 | 1024 CReLU w7 e400 | Bracket between 0.05 and 0.1. May find the true optimum | HIGH |
| C3 | 1024 CReLU w10 e400 | wdl=0.1 was +17 self-play. Controlled retest | MEDIUM |

**SPRT tests**: C1 vs A2, C2 vs A2, C3 vs A2 (all vs same CReLU w0 e400 baseline). Also C1 vs C2, C2 vs C3 for direct comparison.

## Phase 4: Width (2 runs)

**Goal**: Does 1536 help when properly trained? Use the best activation and WDL from phases 2-3.

| Run | Config | Hypothesis | Priority |
|-----|--------|-----------|----------|
| D1 | 1536 {best-activation} {best-wdl} e800 | 1536 needs ~800 SBs. With optimal activation+WDL it might clearly beat 1024 | HIGH |
| D2 | 1536 {best-activation} {best-wdl} e400 | Intermediate checkpoint — is it competitive at e400? | MEDIUM |

**SPRT tests**: D1 vs best-1024 (the real architecture question)

## Phase 5: Data Diversity (1 run)

**Goal**: Does training data diversity matter for the best architecture?

| Run | Config | Hypothesis | Priority |
|-----|--------|-----------|----------|
| E1 | {best-config} on 1 T80 file only | If neutral vs 6-file, data diversity doesn't matter. If negative, it does. | LOW |

**SPRT test**: E1 vs best model from phase 4

## Execution Plan

With 3×4070 GPUs:

**Day 1** (~12 hours): Phase 1 + Phase 2 (5 runs, 3 parallel)
- GPU-A: A1 (e120, ~1.5h) then A3 (e400, ~4h)
- GPU-B: A2 (e400, ~4h) then B2 (e200, ~2h)
- GPU-C: B1 (e400, ~4h)

**Day 2** (~8 hours): Phase 3 (3 runs, all parallel)
- GPU-A: C1 (e400)
- GPU-B: C2 (e400)
- GPU-C: C3 (e400)

**Day 3** (~8 hours): Phase 4 + 5
- GPU-A: D1 (e800, ~8h)
- GPU-B: D2 (e400, ~4h) then E1 (e400, ~4h)
- GPU-C: SPRT testing on Hercules/Atlas

**SPRT testing** runs continuously on Hercules as models arrive. Each SPRT at concurrency 8, ~500-1000 games to resolve.

## Success Criteria

A model is a "production upgrade" if it:
1. Beats current production by +10 in self-play SPRT (H1)
2. Shows +5 or better in an 8×200 cross-engine gauntlet (or better, an RR)
3. Has no NPS regression >5% vs production

## What We Expect To Learn

1. **Is 1024 CReLU fully trained?** (Phase 1)
2. **Does SCReLU give free Elo at same width/NPS?** (Phase 2)
3. **What's the optimal WDL blend?** (Phase 3)
4. **Does width help when all other variables are optimised?** (Phase 4)
5. **Combined: what's the best achievable model?** (Phases 2-4 combined)

## Phase 6: Hidden Layers (the big architectural leap)

**Motivation**: Every engine ranked above us in the RR has multiple hidden layers between the accumulator and output. Our architecture goes directly from FT→output (single layer). The top engines go through FT→L1→L2→output which adds non-linear pattern combination. This is likely the single biggest architectural gap.

Engine review data:
```
#1  Reckless:    768pw → 16 → 32 → 1×8  (3 layers after FT)
#2  Obsidian:    1536pw → 16 → 32 → 1×8  (3 layers)
#3  Berserk:     1024 → 16 → 32 → 1      (3 layers)
#4  Alexandria:  1536pw → 16 → 32 → 1×8  (3 layers)
    GoChess:     1024 → 1×8               (1 layer - output only)
```

### Phase 6a: Validation (quick, ~30 min GPU time)

**Goal**: Verify that Bullet can train hidden layers with wdl=0.0 without gradient vanishing.

Previous attempt with deep layers + Bullet's sigmoid loss produced dead L2/L3 weights (98% zero after quantization). That was with wdl=0.5 on the old Bullet. The situation may be different with wdl=0.0 on current Bullet.

| Run | Config | What we're checking |
|-----|--------|-------------------|
| V1 | (768×16→1024)×2 → 16 → 1×8, wdl=0.0, e10 | Are L1 (16-wide) weights non-zero after 10 SBs? |
| V2 | (768×16→1024)×2 → 16 → 32 → 1×8, wdl=0.0, e10 | Are L2 (32-wide) weights also non-zero? |

Check with `check-net` and by inspecting raw weight statistics (mean |w|, % non-zero after quantization).

**If weights are dead**: Try wdl=0.05 (provides stronger gradient signal), or MSE loss if Bullet supports it, or gradient clipping.

**If weights are alive**: Proceed to full training runs.

### Phase 6b: Full Training (if validation passes)

| Run | Config | Hypothesis | Priority |
|-----|--------|-----------|----------|
| F1 | (1024)×2 → 16 → 1×8, {best-wdl}, e400 | Minimum hidden layer — does even 16 neurons help? | HIGH |
| F2 | (1024)×2 → 16 → 32 → 1×8, {best-wdl}, e400 | Full Alexandria/Berserk-style — the target architecture | HIGH |
| F3 | (1024pw + dual)×2 → 16 → 1×8, {best-wdl}, e400 | Pairwise + dual activation + hidden (Reckless-style) | MEDIUM |

**SPRT tests**: F1, F2, F3 each vs best-1024 single-layer from phases 1-3.

### Phase 6c: Combined Optimisation

Take the winning hidden layer config and combine with the best findings from phases 2-4:

| Run | Config | Goal |
|-----|--------|------|
| G1 | {best-hidden} + SCReLU + {best-wdl} + 1536-wide, e800 | The ultimate architecture combining all winning dimensions |

This is speculative — only run if phases 2-6b each show clear gains.

### GPU Host Prompt for Validation (Phase 6a)

```
Phase 6a: Hidden Layer Validation

Goal: Test if Bullet can train hidden layers with wdl=0.0 without gradient vanishing.

Run 1 — Single hidden layer:
  Architecture: (768×16 → 1024)×2 → 16 → 1×8
  - Feature transformer: 768*16 inputs → 1024 (same as current)
  - Hidden layer 1: 2*1024 = 2048 → 16 (new!)
  - Output: 16 → 1 × 8 buckets

  Bullet config structure:
    let l0 = builder.new_affine("l0", 768 * 16, 1024);
    let l1 = builder.new_affine("l1", 2 * 1024, 16);  // NEW hidden layer
    let l2 = builder.new_affine("l2", 16, NUM_OUTPUT_BUCKETS);

    let stm = l0.forward(stm_inputs).crelu();
    let ntm = l0.forward(ntm_inputs).crelu();
    let hidden = l1.forward(stm.concat(ntm)).crelu();  // NEW
    let out = l2.forward(hidden).select(output_buckets);

  Training: wdl=0.0, cosine over 10 SBs, 6 T80 files
  Save at SB 5 and SB 10.

Run 2 — Two hidden layers:
  Architecture: (768×16 → 1024)×2 → 16 → 32 → 1×8

  Bullet config structure:
    let l0 = builder.new_affine("l0", 768 * 16, 1024);
    let l1 = builder.new_affine("l1", 2 * 1024, 16);
    let l2 = builder.new_affine("l2", 16, 32);          // NEW second hidden
    let l3 = builder.new_affine("l3", 32, NUM_OUTPUT_BUCKETS);

    let stm = l0.forward(stm_inputs).crelu();
    let ntm = l0.forward(ntm_inputs).crelu();
    let h1 = l1.forward(stm.concat(ntm)).crelu();
    let h2 = l2.forward(h1).crelu();                    // NEW
    let out = l3.forward(h2).select(output_buckets);

  Training: same as Run 1.

Validation checks after training:
1. Convert both checkpoints to .nnue
2. Run check-net on each — do they produce non-zero evals?
3. Report weight statistics for each layer:
   - Mean |weight|
   - % of weights that are zero after int16 quantization
   - If L1 or L2 weights are >90% zero, gradient vanishing is occurring

Key question: Do the hidden layer weights learn meaningful values,
or do they collapse to near-zero like our previous v4 deep net attempt?

Quantization for the new layers:
- L0 (FT): int16, QA=255 (same as current)
- L1 (hidden): experiment with int8 (QA=64) or keep int16
- L2 (hidden2): float or int8
- Output: int16, QB=64 (same as current)

Note: We already have multi-layer inference code from v4. The engine
can load and run these nets. The converter (tuner convert-bullet) needs
updating for the extra layers — use the v4 converter as reference.
```

## What Might Be Wrong In Our Current Understanding

- "1024 is converged at sb120" — tested once with old Bullet, never retested
- "wdl=0.05 is best WDL" — based on one self-play test, cross-engine was ambiguous, never tested 0.07
- "SCReLU needs more training" — based on one sb200 test, could also be that SCReLU + our search is a bad fit
- "1536 = 1024 cross-engine" — tested with suboptimal WDL and variable search code
- "6 files = 1 file" — only tested at 1024 CReLU w0, may differ for other configs
- "Hidden layers don't train on Bullet" — tested with wdl=0.5 on old Bullet, gradient vanishing was the issue. Never tested with wdl=0.0 on current Bullet
- "Our architecture ceiling is the training data" — could actually be the missing hidden layers limiting pattern combination

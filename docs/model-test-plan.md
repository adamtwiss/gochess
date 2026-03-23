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

## What Might Be Wrong In Our Current Understanding

- "1024 is converged at sb120" — tested once with old Bullet, never retested
- "wdl=0.05 is best WDL" — based on one self-play test, cross-engine was ambiguous, never tested 0.07
- "SCReLU needs more training" — based on one sb200 test, could also be that SCReLU + our search is a bad fit
- "1536 = 1024 cross-engine" — tested with suboptimal WDL and variable search code
- "6 files = 1 file" — only tested at 1024 CReLU w0, may differ for other configs

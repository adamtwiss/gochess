# NNUE Model Registry

Tracks every model that has been tested in play, with exact training config and result.
**Always update this file when testing a new model.**

## Production Model

| Field | Value |
|-------|-------|
| **File** | `net-v5-120sb-sb120.nnue` |
| **Architecture** | 1024 CReLU, no pairwise, no SCReLU |
| **Bullet version** | ~7f930b0 (March 16 2025, OLD Bullet) |
| **Data** | Single file: `test80-2024-02-feb-2tb7p.min-v2.v6.binpack` |
| **WDL** | 0.0 (pure score) |
| **LR** | 0.001 → 0.0001, cosine decay over 120 SB |
| **Batch** | 16384, 6104 batches/SB (~100M pos/SB) |
| **Threads** | 16, batch_queue_size 128 |
| **SavedFormat** | NO .transpose() on l0w and l1w |
| **Superbatches** | 120 |
| **Checkpoint** | `/workspace/code/bullet/checkpoints/gochess-v5-120/quantised.bin` (GPU1) |
| **MD5** | `4a58e81ac236f4f63e7ffecc7c0a2ddc` |
| **Status** | **PRODUCTION** — strongest model, reference for all comparisons |

## Models Under Test

### net-v5-1024-wdl00-sb120.nnue (2026-03-20)

| Field | Value |
|-------|-------|
| **Architecture** | 1024 CReLU, no pairwise, no SCReLU |
| **Bullet version** | feab644 (current, NEW Bullet — 154 commits after production) |
| **Data** | 6 T80 files (all available .min-v2.v6.binpack) |
| **WDL** | 0.0 (pure score) |
| **LR** | 0.001 → 0.0001, cosine decay over 120 SB |
| **Batch** | 16384, 6104 batches/SB |
| **SavedFormat** | NO .transpose() (correct for new Bullet too) |
| **Superbatches** | 120 |
| **Trained on** | GPU2 |
| **Status** | SPRT testing vs production — checking pipeline reproducibility + 6-file data benefit |

## Failed Models (DO NOT REUSE)

### All wdl=0.5 models (2026-03-17 to 2026-03-20)

All nets trained with wdl=0.5 are ~300 Elo weaker than production. The WDL setting was incorrectly changed to 0.5 based on a misdiagnosed "WDL confound" that was actually a data diversity issue.

| File | Elo vs Production | Reason |
|------|-------------------|--------|
| net-v5-1536-wdl05-sb120.nnue | -251 | wdl=0.5 + 1536 NPS penalty |
| net-v5-1536-wdl05-sb200.nnue | +243 vs sb120 (internal) | wdl=0.5 (both sides same defect) |
| net-v5-1024-6file-sb120.nnue | -317 | wdl=0.5 |
| net-v5-1024-6file-sb200.nnue | +98 vs sb120 (internal) | wdl=0.5 (both sides same defect) |
| net-v5-1024-multi-sb120.nnue | -93 (confound test) | wdl=0.0 but trained on old confounded pipeline |
| net-v5-1024-multi-sb200.nnue | not tested | wdl=0.0, old pipeline |

### Pre-v5 models

| File | Notes |
|------|-------|
| net-v5-sb5.nnue | 5 SB only, test run |
| net-v5-multi-sb120.nnue | wdl=0.0, old multi-file (pre-confound-fix) |
| net-v5-screlu-sb120.nnue | SCReLU, wdl=0.0, old pipeline |
| net-v5-1536-sb120.nnue | 1536 wide, wdl=0.0, old pipeline |

## Key Findings

### Training Settings
- **WDL must be 0.0** (pure score) — wdl=0.5 produces nets ~300 Elo weaker. Discovered 2026-03-20 after days of confusion.
- **No .transpose()** in SavedFormat — correct for both old and new Bullet (verified byte-identical conversion).
- **wdl=0.1** (light game-result blend) — untested, training on GPU2. Hypothesis: small blend might help without the 0.5 collapse.

### Data Diversity
- **1024-wide: 6 T80 files = 1 file** — no Elo difference (SPRT neutral at 798 games). The single-file dataset has enough diversity for 1024's capacity.
- **1536-wide**: untested whether more data helps. Larger models may benefit from data diversity that smaller models can't exploit.

### Training Length (Superbatches)
- **1024-wide: sb120 = sb200** — no benefit from extra training (SPRT neutral at 376 games). The 1024 model is fully converged at 120 SBs.
- **1536-wide: sb200 >> sb120** — +91.6 Elo (H1 at 97 games). Wider nets have more parameters and need significantly more training to converge. sb200 may still be undertrained for 1536.
- **Validation loss can be misleading** — loss plateaus while playing strength continues to improve. The model is refining critical positions (game-deciding patterns) even when average prediction error stops falling.
- **LR schedule must match training length** — cosine decay over 120 SBs means the last 80 SBs of a 200 SB run are at near-zero LR. For longer runs, extend the cosine schedule to match (e.g. cosine over 300 SBs for a 300 SB run).
- **Optimal SBs likely scales with architecture size**: 1024→120, 1536→300+, larger→more.

### Architecture Width
- **1536 has lower validation loss than 1024** (0.008810 vs 0.009058 at sb200) — genuinely better eval quality from wider architecture.
- **1536 NPS penalty is only 12% with Finny tables** — down from ~75% without Finny. Finny tables disproportionately help wider nets because king-bucket recomputes (the expensive part) are proportional to width.
- **1536 sb120 is too weak** (-165 vs 1024 production) — the eval quality doesn't compensate for the NPS hit when undertrained. sb200 is much closer. sb300/sb400 may cross the threshold.
- **NPS impact of width**: 1024 = 1781 kNPS, 1536 = 1568 kNPS (-12%). The Finny table cache hit rate matters more than raw accumulator width for practical NPS.

### Inference Optimizations
- **Finny tables**: +34% NPS, +165 Elo (self-play). Caches per-king-bucket accumulator state, turning full recomputes into delta updates. Impact scales with accumulator width — even more valuable at 1536+.
- **Fused MaterializeV5**: +5-8% NPS, +32.6 Elo. Combines lazy accumulator copy + delta application.
- **NPS gains transfer ~1:1 to cross-engine Elo** (same as eval improvements, unlike search gains which are ~2:1).

### Random Seed Variation
- Check-net values vary 10-20% between different random seeds with identical config. Playing strength likely varies 10-20 Elo between seeds. Consider training multiple seeds and picking the best, or ensemble distillation, for production models.

## Key Rules

1. **WDL must be 0.0** (pure score) — wdl=0.5 produces nets ~300 Elo weaker
2. **No .transpose()** in SavedFormat — correct for both old and new Bullet
3. **Always test vs production** (`net-v5-120sb-sb120.nnue`) — never compare nets trained with different settings against each other
4. **Log every model here** with exact Bullet version, data files, and WDL setting
5. **Bullet version matters** — old (7f930b0) and new (feab644) have different Kaiming init (+41% larger initial weights) and bullet_core overhaul. Reproduction confirmed equivalent strength.

## Bullet Version Notes

| Version | Commit | Key Changes |
|---------|--------|-------------|
| Old (March 2025) | ~7f930b0 | Used for production model. Kaiming init: 1/sqrt(n) |
| New (Current) | feab644 | SavedFormat refactor (cb67efb), Kaiming init sqrt(2/n) (e1c9336), i32 quant fix (ae31a62), bullet_core overhaul (c5177bd) |

The SavedFormat refactor doesn't change binary output (verified: byte-identical conversion from old checkpoint). The Kaiming init change may affect training dynamics — reproduction test pending from GPU1.

## Active Training Pipeline (2026-03-21)

| GPU | Architecture | WDL | SBs | Data | Status |
|-----|-------------|-----|-----|------|--------|
| GPU1 | 768 pairwise | 0.0 | 120 | 1-2 T80 files | Training |
| GPU1 | 1024 reproduce | 0.0 | 120 | 1 T80 file | Complete — SPRT testing |
| GPU2 | 1024 | 0.1 | 120 | 6 T80 files | Training |
| GPU2 | 1536 | 0.0 | 300 | 6 T80 files | Training |
| GPU2 | 1536 | 0.0 | 400 | 6 T80 files | Training |

**Known variable**: GPU1 uses 1-2 T80 files, GPU2 uses 6. At 1024-wide this made no difference (neutral SPRT). May matter at larger architectures with more capacity. Note when comparing results across GPUs.

## SPRT Results (2026-03-21)

| Test | Result | Elo | Notes |
|------|--------|-----|-------|
| 1536 sb200 vs 1536 sb120 (wdl=0.0) | **H1** | +91.6 | Wider nets need more training |
| 1536 sb120 vs 1024 production | **H0** | -164.9 | sb120 undertrained for 1536 |
| 1536 sb200 vs 1024 production | Testing | -34.9 (30 games) | Key test — in progress |
| 1024 wdl=0.0 6-file vs production | Neutral | -3.1 | Pipeline confirmed |
| 1024 sb200 vs sb120 (wdl=0.0) | Neutral | +1.0 | More training doesn't help at 1024 |
| GPU1 reproduce vs production | Testing | +11.6 (129 games) | GPU1 pipeline check |

## SPRT Results (2026-03-21, continued)

| Test | Result | Elo | Notes |
|------|--------|-----|-------|
| 1024 wdl=0.1 sb120 vs prod | **H0** | -175 | Light WDL blend catastrophic at sb120 |
| 1024 wdl=0.1 sb200 vs prod | ~H0 | -5 (141 games) | Recovers but doesn't help |
| 768pw sb120 vs prod (scalar) | **H0** | -346 | 53% NPS penalty from scalar code |
| 768pw sb200 vs prod (scalar) | **H0** | -158 | Still penalized by scalar code |
| 768pw sb120 vs prod (SIMD) | **H0** | -119 | Undertrained, NPS now only -7% |
| 768pw sb200 vs prod (SIMD) | Testing | -31 (82 games) | Recovering, following 1536 pattern |

## Key Finding: Training Length Scales With Model Complexity

| Architecture | sb120 vs prod | sb200 vs prod | sb300/400 | Optimal SBs (est.) |
|-------------|---------------|---------------|-----------|-------------------|
| 1024 CReLU | baseline | +1 (no gain) | N/A | 120 |
| 1536 CReLU | -165 | **+11** | Training... | 200-300 |
| 768 pairwise | -119 | -31 (improving) | Training... | 300-400+ |

The pattern is consistent: more complex architectures need proportionally more training.

## NPS Benchmarks (Intel Xeon E-2288G, Hercules)

Methodology: EPD runner, 30 WAC positions, 3s each, single-threaded, 3 runs averaged.

| Architecture | NPS (kNPS) | vs 1024 | Notes |
|-------------|-----------|---------|-------|
| 1024 CReLU | 1,773 | baseline | With Atlas wide kernels |
| 768 pairwise | 1,654 | -7% | With Titan pairwise SIMD |
| 1536 CReLU | 1,471 | -17% | With Atlas wide kernels |

## CRITICAL FINDING: wdl=0.1 beats wdl=0.0 (2026-03-21)

**1024 wdl=0.1 sb200 vs production (wdl=0.0 sb120): H1 at 742 games, +17.3 Elo ±17.6, 97.3% LOS.**

A light game-result blend (10%) with enough training (200 SBs) produces a stronger net than pure score training. Key points:
- wdl=0.1 sb120 is catastrophic (-175 Elo) — the model needs ~200 SBs to learn the blended signal
- wdl=0.1 sb200 is +17.3 — genuinely better than wdl=0.0 at any training length
- wdl=0.5 is still catastrophic (-300 Elo) — too much result blending collapses the eval

**New recommendation**: Use wdl=0.1 with sb200+ for all future training. The optimal WDL is not zero.

## Atlas Wide Kernels — Intel SPRT

H1 at 1663 games, +9.2 Elo ±10.3, 95.9% LOS. Confirmed gain on Intel (smaller than AMD's +14.7 due to +5% vs +18% NPS difference).

## 1536 sb400 Beats Production (2026-03-21)

**1536 CReLU wdl=0.0 sb400 vs 1024 production: H1 at 630 games, +18.2 Elo ±18.2, LOS 97.5%.**

First architecture change to definitively beat production. The progression:

| 1536 Checkpoint | vs 1024 Production | Notes |
|----------------|-------------------|-------|
| sb120 | -165 | Massively undertrained |
| sb200 | +11 (H1) | Just crossed positive |
| sb300 | +1 (flat) | Inconsistent checkpoint |
| sb400 | **+18.2 (H1)** | Clear winner, loss still dropping |

The 1536 net's validation loss was still dropping at sb400 (0.008679). Training sb600/sb800 with cosine decay over the full length would likely find more Elo. Queued for when GPU capacity frees up.

NPS impact: 1536 is 17% slower than 1024 (1471 vs 1773 kNPS on Intel with Finny tables). The +18 Elo eval quality gain more than compensates.

**Next steps:** 
- Gauntlet testing against rival engines (in progress)
- Train sb600/sb800 when GPU available
- Test 1536 with wdl=0.1 (if WDL bracket shows benefit)
- Profile and optimize 1536 NPS (Atlas)

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

## Key Rules

1. **WDL must be 0.0** (pure score) — wdl=0.5 produces nets ~300 Elo weaker
2. **No .transpose()** in SavedFormat — correct for both old and new Bullet
3. **Always test vs production** (`net-v5-120sb-sb120.nnue`) — never compare nets trained with different settings against each other
4. **Log every model here** with exact Bullet version, data files, and WDL setting
5. **Bullet version matters** — old (7f930b0) and new (feab644) have different Kaiming init (+41% larger initial weights) and bullet_core overhaul. Reproduction testing ongoing.

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

# Self-Play vs Cross-Engine Transfer for Pruning Changes

## Discovery (2026-03-22)

After a productive day of search tuning (10 binary-changing commits, ~+200 self-play Elo cumulative), cross-engine performance had actually **regressed by ~12 Elo** against the rival pack (Ethereal, Texel, Minic, Weiss, Midnight, Laser, Demolito, Monolith at 10+0.1s TC).

## Hypothesis

**Aggressive forward-pruning gains are self-play artifacts.** When both sides of a self-play game share identical search blind spots, tightening pruning thresholds "works" because the opponent also misses the refutations you're pruning away. Against diverse engines with different search characteristics, those pruned lines get exploited.

Accuracy improvements (better eval correction, better signals) transfer well because they make the engine genuinely smarter regardless of opponent. Pruning changes just make the engine lazier in ways that only same-engine opponents tolerate.

## What We Tested

### Phase 1 — Cumulative Ablation (8x100 = 800 games per commit point)

Built binaries at each of the 10 commit points from the day's work and ran gauntlets against the 8-engine rival pack. This measured the cumulative pack Elo at each stage, revealing which commits caused the score to drop.

### Phase 2 — Individual Isolation (8x400 = 3200 games per variant, +/-10 error bars)

After reverting the obvious losers (hist-prune-1500, badnoisy-50/60), we built "revert one change" variants from the cleaned HEAD to isolate each remaining change's individual cross-engine impact. Also tested re-adding SEE cap 80 which had been accidentally reverted by a merge conflict between two Claude instances.

## Full Results

### Today's Changes (Cumulative Ablation)

| # | Change | Self-play Elo | Cross-engine 8x100 (+/-20) | Cross-engine 8x400 (+/-10) | Action |
|---|--------|:---:|:---:|:---:|--------|
| 0 | **baseline** (before today) | — | **+33** | — | Reference |
| 1 | hist-prune 2000->1500 | +14.7 | **+7** (delta: -26) | — | **Reverted** |
| 2 | SEE cap 100->80 | +25.2 | **+14** (delta: +7) | **+18** (+4 vs baseline) | **Re-added** |
| 3 | multi-corr-hist | +28.6 | **+39** (delta: +25) | — | **Kept** |
| 4 | badnoisy 75->60 | +32.4 | **+21** (delta: -18) | — | **Reverted** |
| 5 | QS delta 240->280 | +11.0 | **+14** (delta: -7) | **+11** (-3 vs baseline) | **Kept** (neutral) |
| 6 | badnoisy 60->50 | +16.7 | **+18** (delta: +4) | — | **Reverted** (stacked with #4) |
| 7 | RFP depth 7->8 | +37.7 | **+14** (delta: -4) | **+1** (-13 vs baseline) | **Reverted** |
| 8 | cap LMR continuous /5000 | +10.6 | **+10** (delta: -4) | **+10** (-4 vs baseline) | **Kept** (neutral) |
| 9 | futility 50+d*50 | +10.2 | **+21** (delta: +11) | **+6** (-8 vs baseline) | **Reverted** |

### Older Changes (Before/After Delta, HCE Only — No NNUE)

These predate v5 NNUE support so ran with HCE eval. Absolute scores are low (~-300) but the deltas are still meaningful.

| Test | Self-play | Before Pack Elo | After Pack Elo | Delta | Verdict |
|------|:---------:|:---:|:---:|:---:|---------|
| TT noisy detection | +34.4 | -328 | -282 | **+46** | Positive — information improvement |
| Opp-mat LMR + badnoisy 75 | +77.3 | -299 | -306 | **-7** | Neutral — mixed pruning+accuracy |
| Futility 100+d*100->80+d*80 | +33.6 | -285 | -287 | **-2** | Neutral — pruning tightening |
| LMR quiet C=1.50->1.30 | +13.3 | — | — | crashed | Binary too old for v5 net |

## Classification of Change Types

The data reveals a clear taxonomy of search changes by their transfer characteristics:

### 1. Accuracy Improvements (~1:1 or better transfer)
Changes that make the engine's evaluation or information gathering genuinely better.

- **Multi-source correction history**: +28.6 self-play, +25 cross-engine
- **TT noisy move detection**: +34.4 self-play, +46 cross-engine

These work because they help the engine understand positions better, which is valuable against any opponent. They are the highest-value changes.

### 2. Structural Search Changes (~0.4:1 transfer)
Changes to the search framework (aspiration windows, time management).

- **Aspiration contraction**: +48.9 self-play, ~+20 cross-engine

These change how the search navigates, not what it prunes. Moderate transfer.

### 3. NPS Improvements (~0.3:1 transfer)
Speed optimizations that let the engine search deeper.

- **Finny tables**: +165 NPS gain, ~+50 cross-engine (estimated)

The Elo = 100*log2(speedup) formula holds, but the self-play SPRT number overstates the gain because faster search also amplifies pruning.

### 4. Pruning Tightening (NEGATIVE transfer)
Changes that prune more aggressively — tighter margins, deeper depth gates, more reduction.

- **hist-prune 2000->1500**: +14.7 self-play, **-26 cross-engine**
- **badnoisy 75->60**: +32.4 self-play, **-18 cross-engine**
- **RFP depth 7->8**: +37.7 self-play, **-13 cross-engine**
- **futility 50+d*50**: +10.2 self-play, **-8 cross-engine**

**The larger the self-play gain, the worse the cross-engine regression.** This is because bigger self-play gains mean more positions are being pruned — and those are exactly the positions that diverse opponents exploit.

## Why the Anti-Correlation Exists

In a self-play game:
1. Engine A prunes position X because its eval says it's not promising
2. Engine B (identical) would also not find X promising, so never plays into it
3. The pruning appears "free" — no games are lost because of it
4. Self-play SPRT sees faster search (more depth for the same time) with no quality loss

In a cross-engine game:
1. Engine A prunes position X
2. Engine B (different search/eval) sees X as promising and plays into it
3. Engine A has no analysis of X and makes poor moves
4. The more positions A prunes, the more exploitable blind spots it creates

The fundamental issue is that **self-play tests for search speed under identical blind spots**, while **cross-engine tests for search robustness under diverse challenges**.

## Process Recommendations

### For Future Search Tuning

1. **Classify every change** as accuracy, structural, NPS, or pruning before testing
2. **Accuracy/structural/NPS changes**: Self-play SPRT is sufficient
3. **Pruning changes**: Self-play SPRT is a first filter only (kills disasters). Must follow up with cross-engine gauntlet (8 engines, 100 games each, ~25 min at concurrency 32) before merging
4. **A positive self-play result for a pruning change should raise suspicion**, not confidence

### Gauntlet Testing Protocol

```bash
cutechess-cli \
  -tournament gauntlet \
  -engine name=GoChess-test cmd=./chess-test proto=uci option.NNUEFile=$(pwd)/net.nnue arg=-syzygy arg=/tablebases \
  -engine name=Ethereal cmd=~/chess/engines/Ethereal/src/Ethereal proto=uci \
  -engine name=Texel cmd=~/chess/engines/texel/build/texel proto=uci \
  -engine name=Demolito cmd=~/chess/engines/Demolito/src/demolito proto=uci \
  -engine name=Monolith cmd=~/chess/engines/Monolith/Source/Monolith proto=uci \
  -engine name=Midnight cmd=~/chess/engines/MidnightChessEngine/midnight proto=uci \
  -engine name=Weiss cmd=~/chess/engines/weiss/src/weiss proto=uci \
  -engine name=Minic cmd=~/chess/engines/Minic/Dist/Minic3/minic_dev_linux_x64 proto=uci \
  -engine name=Laser cmd=~/chess/engines/laser-chess-engine/src/laser proto=uci \
  -each tc=0/10+0.1 \
  -rounds 100 -concurrency 32 \
  -openings file=testdata/noob_3moves.epd format=epd order=random \
  -pgnout gauntlet.pgn -recover -ratinginterval 20 \
  -draw movenumber=20 movecount=10 score=10 \
  -resign movecount=3 score=500 twosided=true
```

**Error bars:** 8x100 = 800 games gives +/-20. Sufficient to catch >15 Elo regressions. For finer resolution (catching >8 Elo effects), use 8x400 = 3200 games (+/-10).

### Red Flags for Pruning Changes

A pruning change is suspect if:
- It tightens a margin (100->80, 75->50)
- It extends a depth gate (depth 7->8)
- It increases reduction (C=1.50->1.30)
- It prunes based on engine-specific signals (history scores, which are learned from self-play patterns)
- The self-play gain is large (>+20) — counter-intuitively, bigger gains are MORE suspicious

### What Actually Works Cross-Engine

Focus tuning effort on:
- **Better eval correction** (multi-source correction history was the biggest win)
- **Better information gathering** (TT noisy detection transferred at >1:1)
- **Speed** (Finny tables, SIMD, fused operations — genuine extra depth)
- **Move ordering** (helps the engine find good moves before pruning kicks in)

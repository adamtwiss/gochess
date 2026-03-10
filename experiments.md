# Experiment Log

Structured record of all search/eval tuning experiments. Each entry captures the change, SPRT result, baseline context, and lessons learned. Revisit failed experiments when conditions change (new NNUE net, new search features, etc.).

**SPRT settings** (unless noted): `elo0=-10 elo1=5 alpha=0.05 beta=0.05`, tc=10+0.1, Hash=64, OwnBook=false, openings=noob_3moves.epd.

---

## 2026-03-09: Correction History v1
- **Change**: Pawn-hash indexed correction table. Full strength (÷GRAIN), all node types updated.
- **Result**: -12 Elo (rejected early, ~200 games)
- **Baseline**: NNUE net-halfka.nnue, pre-RFP/LMR tuning
- **Notes**: Fail-low nodes provided unreliable upper bounds as update signal, adding noise. Full strength correction on noisy data hurt.

## 2026-03-09: Correction History v2
- **Change**: Half strength correction (÷GRAIN*2), exact/fail-high updates only.
- **Result**: -11 Elo (rejected, ~300 games)
- **Baseline**: NNUE net-halfka.nnue, pre-RFP/LMR tuning
- **Notes**: Halving correction made it too weak to help. The issue was noise quality, not magnitude.

## 2026-03-09: Correction History v3 (MERGED)
- **Change**: Full strength, tight clamp (corrHistMax=128 vs 256), depth>=3 gate, exact/fail-high only.
- **Result**: **+11.9 Elo**, H1 accepted. W123-L105-D157 (385 games). LOS 91.8%.
- **Baseline**: NNUE net-halfka.nnue, pre-RFP/LMR tuning
- **Commit**: fe1edb5
- **Notes**: Tight clamping + depth gate addressed noise while keeping full correction strength. Third attempt — persistence paid off. Sound idea proven in Stockfish.

## 2026-03-09: RFP Tightening v1 (MERGED)
- **Change**: Margins depth*120/85 -> depth*85/60, depth limit 6->7.
- **Result**: **+15.7 Elo**, H1 accepted. W191-L165-D268 (~624 games). LOS 94.8%.
- **Baseline**: NNUE net-halfka.nnue, pre-LMR tuning
- **Commit**: ad9f603
- **Notes**: Same change failed pre-NNUE. NNUE eval accuracy enables tighter node-level pruning. Big win.

## 2026-03-09: NMP v1 (divisor + improving penalty)
- **Change**: Eval divisor 200->150, +1R when not improving.
- **Result**: **-14.0 Elo**, H0 accepted. W266-L299-D391 (956 games). LOS 3.3%.
- **Baseline**: NNUE net-halfka.nnue, post-RFP tuning
- **Notes**: The !improving +1R was too aggressive — over-pruning in declining positions. Divisor change alone tested separately in v2.

## 2026-03-09: NMP v2 (divisor only)
- **Change**: Eval divisor 200->150, no improving penalty.
- **Result**: **-22.6 Elo**, H0 accepted. W172-L216-D289 (677 games). LOS 1.3%.
- **Baseline**: NNUE net-halfka.nnue, post-RFP tuning
- **Notes**: Divisor itself is too aggressive. Current NMP formula (R=3+depth/3, divisor 200) is well-calibrated. Don't revisit unless eval changes significantly.

## 2026-03-09: LMR Aggressiveness v1 (MERGED)
- **Change**: LMR table constant C=2.0 -> 1.75 (more reduction for late moves).
- **Result**: **+16.2 Elo**, H1 accepted. W170-L147-D240 (~700 games). LOS 95.1%.
- **Baseline**: NNUE net-halfka.nnue, post-RFP tuning (no correction history in baseline binary)
- **Commit**: 474cd58
- **Notes**: More aggressive LMR saves time to search important moves deeper. NNUE's accurate eval makes it safe to reduce late moves more. Draw ratio increased (42% vs ~38%), indicating fewer blunders.

## 2026-03-09: Razoring Tightening v1
- **Change**: Margin 400+depth*100 -> 300+depth*75.
- **Result**: **-32.2 Elo**, H0 accepted. ~400 games. LOS 0.4%.
- **Baseline**: NNUE net-halfka.nnue, post-RFP tuning
- **Notes**: Razoring at depth 1-2 needs the slack. Current margins are well-calibrated. Quick, decisive rejection.

## 2026-03-09: Futility Pruning v1
- **Change**: Move futility margin 100+lmrDepth*100 -> 75+lmrDepth*75 (uniform 25% tightening).
- **Result**: +0.7 Elo after 2128 games, inconclusive (killed at LLR 1.35/2.94).
- **Baseline**: NNUE net-halfka.nnue, post-RFP tuning
- **Notes**: Essentially zero effect — too tight at low lmrDepth (base margin of 75 barely any slack). Unlike RFP (node-level), per-move futility pruning errors compound. Replaced with v2.

## 2026-03-09: Futility Pruning v2 (RUNNING)
- **Change**: Move futility margin 100+lmrDepth*100 -> 100+lmrDepth*75 (tighten slope only, keep base).
- **Result**: In progress.
- **Baseline**: NNUE net-halfka.nnue, post-RFP tuning
- **Notes**: Preserves safe base margin of 100, only tightens per-depth scaling. Hypothesis: v1 failed because base was too tight, not the depth scaling.

## 2026-03-09: ProbCut v1
- **Change**: probCutBeta from beta+200 -> beta+150, pre-filter staticEval+100 -> +75.
- **Result**: -3.4 Elo, inconclusive (killed at 1007 games). W285-L292-D430. LOS 34%.
- **Baseline**: NNUE net-halfka.nnue, post-RFP tuning
- **Notes**: Looked promising early (+11.6 at 547 games) but faded to zero. Classic early noise. ProbCut margin of 200 appears well-calibrated already.

## 2026-03-09: SEE Pruning v1
- **Change**: Capture threshold -depth*100 -> -depth*75, quiet threshold -20*depth^2 -> -15*depth^2.
- **Result**: -7.7 Elo, killed at ~650 games (trending to rejection). W186-L199-D261. LOS 22.8%.
- **Baseline**: NNUE net-halfka.nnue, post-LMR tuning
- **Notes**: SEE thresholds are about tactical accuracy, not eval quality — NNUE doesn't change material exchange calculations. Current thresholds are correct.

## 2026-03-09: RFP v2 — deeper
- **Change**: RFP depth limit 7->8 (margins unchanged at 85/60).
- **Result**: **-68.6 Elo**, killed at ~200 games (near-rejection). W46-L84-D68. LOS 0.0%.
- **Baseline**: NNUE net-halfka.nnue, post-LMR tuning
- **Notes**: Depth 7 is the sweet spot. At depth 8, max margin is 8*85=680cp — far too large. Pruning positions that shouldn't be pruned. Decisive and fast failure.

## 2026-03-09: Futility Pruning v2
- **Change**: Move futility margin 100+lmrDepth*100 -> 100+lmrDepth*75 (tighten slope only, keep base).
- **Result**: -8.7 Elo, killed at ~462 games (trending to rejection). W130-L142-D190. LOS 23.7%.
- **Baseline**: NNUE net-halfka.nnue, post-RFP tuning
- **Notes**: Neither the uniform tightening (v1) nor the slope-only tightening (v2) helped. Per-move futility margins are already well-tuned. Unlike node-level RFP, per-move errors compound.

## 2026-03-09: LMR v2 — more aggressive (MERGED)
- **Change**: LMR constant C=1.75 -> 1.5.
- **Result**: **+44.4 Elo**, H1 accepted. W112-L73-D122 (307 games). LOS 99.8%.
- **Baseline**: NNUE net-halfka.nnue, post-LMR v1 tuning
- **Commit**: (pending)
- **Notes**: Second consecutive LMR tightening win. C=2.0→1.75 was +16 Elo, C=1.75→1.5 is +44 Elo. Testing C=1.25 next to find the optimum.

## 2026-03-10: NNUE net-new2.nnue (MERGED)
- **Change**: Deploy fine-tuned NNUE net (lower learning rate training).
- **Result**: **+10.4 Elo**, H1 accepted. W332-L300-D465 (1097 games). LLR 2.99.
- **Baseline**: net-halfka.nnue (previous net), same binary
- **Notes**: Net-new2 trained at lower LR squeezed additional quality. Deployed as net-halfka.nnue / net.nnue.

## 2026-03-10: LMR v3 — C=1.25
- **Change**: LMR constant C=1.5 -> 1.25 (even more reduction).
- **Result**: +3 Elo, killed at 374 games. W97-L94-D183 (50.4%). Inconclusive → zero effect.
- **Baseline**: NNUE net-halfka.nnue, post-LMR v2
- **Notes**: C=1.25 overshoots the optimum. Too much reduction causes missed tactical moves.

## 2026-03-10: LMR v3b — C=1.375
- **Change**: LMR constant C=1.5 -> 1.375 (bracketing between 1.25 and 1.5).
- **Result**: +0 Elo, killed at 395 games. W106-L107-D182 (49.9%). Dead flat.
- **Baseline**: NNUE net-halfka.nnue, post-LMR v2
- **Notes**: C=1.5 is at or very near the optimum. No further LMR constant tuning is worthwhile.

## 2026-03-10: LMP v1 — tighter base
- **Change**: LMP base 3+depth^2 -> 2+depth^2.
- **Result**: -10 Elo, killed at 197 games. W51-L57-D89 (48.5%).
- **Baseline**: NNUE net-halfka.nnue, post-LMR v2
- **Notes**: Base of 3 is the right floor. Reducing to 2 prunes too many early moves.

## 2026-03-10: LMP v2 — no improving bonus
- **Change**: Remove the +50% LMP limit bonus when improving (always use base formula).
- **Result**: **-38 Elo**, killed at 147 games. W32-L48-D67 (44.6%).
- **Baseline**: NNUE net-halfka.nnue, post-LMR v2
- **Notes**: The improving bonus is critical — it prevents over-pruning when the position is getting better. Decisive failure.

## 2026-03-10: History pruning divisor v1 — 4000
- **Change**: LMR history adjustment divisor 5000 -> 4000 (more history influence).
- **Result**: **-32 Elo**, killed at 174 games. W42-L58-D74 (45.4%).
- **Baseline**: NNUE net-halfka.nnue, post-LMR v2
- **Notes**: More history influence causes over-fitting to noisy history data.

## 2026-03-10: History pruning divisor v2 — 6000
- **Change**: LMR history adjustment divisor 5000 -> 6000 (less history influence).
- **Result**: -14 Elo, killed at 245 games. W67-L76-D102 (48.2%).
- **Baseline**: NNUE net-halfka.nnue, post-LMR v2
- **Notes**: Less history influence also hurts. Divisor of 5000 is well-calibrated in both directions.

## 2026-03-10: Singular extension margin — depth*2
- **Change**: Singular beta = ttScore - depth*2 (was depth*3; narrower margin = extend more often).
- **Result**: **-85 Elo**, killed at 116 games. W21-L48-D47 (38.4%). Catastrophic.
- **Baseline**: NNUE net-halfka.nnue, post-LMR v2
- **Notes**: depth*2 extends far too many moves, wasting search time. depth*3 is well-calibrated.

---

## Ideas Not Yet Tested

- **Aspiration windows**: Currently delta=15. Small effect expected — low priority.
- **Singular extension depth threshold**: Currently (depth-1)/2. Could try depth/2 or depth/3.
- **Double singular threshold**: Currently singularBeta - depth*3. Could try depth*2.
- **Continuation history weight**: Currently added raw to histScore. Could scale.

## Key Patterns Observed

1. **Node-level pruning benefits from NNUE tightening** (RFP, LMR) — these prune entire subtrees based on eval, and NNUE's accuracy makes this safe.
2. **Per-move pruning is more sensitive** (futility, SEE, LMP) — errors compound across many moves, so margins need more slack.
3. **NMP, razoring, LMP, history divisor, singular margin are all well-tuned** — don't revisit unless eval changes significantly.
4. **Self-play Elo ~2x cross-engine Elo** for search changes. Calibrate expectations.
5. **Persistence matters** — correction history took 3 attempts. Tune before rejecting sound ideas.
6. **LMR constant C=1.5 is near-optimal** — C=1.25 and C=1.375 both tested at zero. Further tuning has diminishing returns.
7. **LMP improving bonus is critical** — removing it costs ~38 Elo. The improving heuristic correctly identifies positions where more moves should be searched.
8. **History divisor 5000 is optimal** — both directions (4000 and 6000) lose Elo.

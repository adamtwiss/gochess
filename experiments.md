# Experiment Log

Structured record of all search/eval tuning experiments. Each entry captures the change, SPRT result, baseline context, and lessons learned. Revisit failed experiments when conditions change (new NNUE net, new search features, etc.).

**SPRT settings** (unless noted): `elo0=-10 elo1=5 alpha=0.05 beta=0.05`, tc=10+0.1, Hash=64, OwnBook=false, openings=noob_3moves.epd.

**Net convention**: All experiments use the checked-in `net.nnue`, referenced by commit hash.

---

## 2026-03-09: Correction History v1
- **Change**: Pawn-hash indexed correction table. Full strength (÷GRAIN), all node types updated.
- **Result**: -12 Elo (rejected early, ~200 games)
- **Baseline**: net.nnue @ 69a797e, pre-RFP/LMR tuning
- **Notes**: Fail-low nodes provided unreliable upper bounds as update signal, adding noise. Full strength correction on noisy data hurt.

## 2026-03-09: Correction History v2
- **Change**: Half strength correction (÷GRAIN*2), exact/fail-high updates only.
- **Result**: -11 Elo (rejected, ~300 games)
- **Baseline**: net.nnue @ 69a797e, pre-RFP/LMR tuning
- **Notes**: Halving correction made it too weak to help. The issue was noise quality, not magnitude.

## 2026-03-09: Correction History v3 (MERGED)
- **Change**: Full strength, tight clamp (corrHistMax=128 vs 256), depth>=3 gate, exact/fail-high only.
- **Result**: **+11.9 Elo**, H1 accepted. W123-L105-D157 (385 games). LOS 91.8%.
- **Baseline**: net.nnue @ 69a797e, pre-RFP/LMR tuning
- **Commit**: fe1edb5
- **Notes**: Tight clamping + depth gate addressed noise while keeping full correction strength. Third attempt — persistence paid off. Sound idea proven in Stockfish.

## 2026-03-09: RFP Tightening v1 (MERGED)
- **Change**: Margins depth*120/85 -> depth*85/60, depth limit 6->7.
- **Result**: **+15.7 Elo**, H1 accepted. W191-L165-D268 (~624 games). LOS 94.8%.
- **Baseline**: net.nnue @ 69a797e, pre-LMR tuning
- **Commit**: ad9f603
- **Notes**: Same change failed pre-NNUE. NNUE eval accuracy enables tighter node-level pruning. Big win.

## 2026-03-09: NMP v1 (divisor + improving penalty)
- **Change**: Eval divisor 200->150, +1R when not improving.
- **Result**: **-14.0 Elo**, H0 accepted. W266-L299-D391 (956 games). LOS 3.3%.
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning
- **Notes**: The !improving +1R was too aggressive — over-pruning in declining positions. Divisor change alone tested separately in v2.

## 2026-03-09: NMP v2 (divisor only)
- **Change**: Eval divisor 200->150, no improving penalty.
- **Result**: **-22.6 Elo**, H0 accepted. W172-L216-D289 (677 games). LOS 1.3%.
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning
- **Notes**: Divisor itself is too aggressive. Current NMP formula (R=3+depth/3, divisor 200) is well-calibrated. Don't revisit unless eval changes significantly.

## 2026-03-09: LMR Aggressiveness v1 (MERGED)
- **Change**: LMR table constant C=2.0 -> 1.75 (more reduction for late moves).
- **Result**: **+16.2 Elo**, H1 accepted. W170-L147-D240 (~700 games). LOS 95.1%.
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning (no correction history in baseline binary)
- **Commit**: 474cd58
- **Notes**: More aggressive LMR saves time to search important moves deeper. NNUE's accurate eval makes it safe to reduce late moves more. Draw ratio increased (42% vs ~38%), indicating fewer blunders.

## 2026-03-09: Razoring Tightening v1
- **Change**: Margin 400+depth*100 -> 300+depth*75.
- **Result**: **-32.2 Elo**, H0 accepted. ~400 games. LOS 0.4%.
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning
- **Notes**: Razoring at depth 1-2 needs the slack. Current margins are well-calibrated. Quick, decisive rejection.

## 2026-03-09: Futility Pruning v1
- **Change**: Move futility margin 100+lmrDepth*100 -> 75+lmrDepth*75 (uniform 25% tightening).
- **Result**: +0.7 Elo after 2128 games, inconclusive (killed at LLR 1.35/2.94).
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning
- **Notes**: Essentially zero effect — too tight at low lmrDepth (base margin of 75 barely any slack). Unlike RFP (node-level), per-move futility pruning errors compound. Replaced with v2.

## 2026-03-09: ProbCut v1
- **Change**: probCutBeta from beta+200 -> beta+150, pre-filter staticEval+100 -> +75.
- **Result**: -3.4 Elo, inconclusive (killed at 1007 games). W285-L292-D430. LOS 34%.
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning
- **Notes**: Looked promising early (+11.6 at 547 games) but faded to zero. Classic early noise. ProbCut margin of 200 appears well-calibrated already.

## 2026-03-09: SEE Pruning v1
- **Change**: Capture threshold -depth*100 -> -depth*75, quiet threshold -20*depth^2 -> -15*depth^2.
- **Result**: -7.7 Elo, killed at ~650 games (trending to rejection). W186-L199-D261. LOS 22.8%.
- **Baseline**: net.nnue @ 69a797e, post-LMR tuning
- **Notes**: SEE thresholds are about tactical accuracy, not eval quality — NNUE doesn't change material exchange calculations. Current thresholds are correct.

## 2026-03-09: RFP v2 — deeper
- **Change**: RFP depth limit 7->8 (margins unchanged at 85/60).
- **Result**: **-68.6 Elo**, killed at ~200 games (near-rejection). W46-L84-D68. LOS 0.0%.
- **Baseline**: net.nnue @ 69a797e, post-LMR tuning
- **Notes**: Depth 7 is the sweet spot. At depth 8, max margin is 8*85=680cp — far too large. Pruning positions that shouldn't be pruned. Decisive and fast failure.

## 2026-03-09: Futility Pruning v2
- **Change**: Move futility margin 100+lmrDepth*100 -> 100+lmrDepth*75 (tighten slope only, keep base).
- **Result**: -8.7 Elo, killed at ~462 games (trending to rejection). W130-L142-D190. LOS 23.7%.
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning
- **Notes**: Neither the uniform tightening (v1) nor the slope-only tightening (v2) helped. Per-move futility margins are already well-tuned. Unlike node-level RFP, per-move errors compound.

## 2026-03-09: LMR v2 — more aggressive (MERGED)
- **Change**: LMR constant C=1.75 -> 1.5.
- **Result**: **+44.4 Elo**, H1 accepted. W112-L73-D122 (307 games). LOS 99.8%.
- **Baseline**: net.nnue @ 69a797e, post-LMR v1 tuning
- **Commit**: d188f53
- **Notes**: Second consecutive LMR tightening win. C=2.0→1.75 was +16 Elo, C=1.75→1.5 is +44 Elo. Testing C=1.25 next to find the optimum.

## 2026-03-10: NNUE net-new2 (MERGED)
- **Change**: Deploy fine-tuned NNUE net (lower learning rate training).
- **Result**: **+10.4 Elo**, H1 accepted. W332-L300-D465 (1097 games). LLR 2.99.
- **Baseline**: net.nnue @ 911b468, same binary
- **Commit**: fb7519b
- **Notes**: Net-new2 trained at lower LR squeezed additional quality.

## 2026-03-10: LMR v3 — C=1.25
- **Change**: LMR constant C=1.5 -> 1.25 (even more reduction).
- **Result**: +3 Elo, killed at 374 games. W97-L94-D183 (50.4%). Inconclusive → zero effect.
- **Baseline**: net.nnue @ 911b468, post-LMR v2
- **Notes**: C=1.25 overshoots the optimum. Too much reduction causes missed tactical moves.

## 2026-03-10: LMR v3b — C=1.375
- **Change**: LMR constant C=1.5 -> 1.375 (bracketing between 1.25 and 1.5).
- **Result**: +0 Elo, killed at 395 games. W106-L107-D182 (49.9%). Dead flat.
- **Baseline**: net.nnue @ 911b468, post-LMR v2
- **Notes**: C=1.5 is at or very near the optimum. No further LMR constant tuning is worthwhile.

## 2026-03-10: LMP v1 — tighter base
- **Change**: LMP base 3+depth^2 -> 2+depth^2.
- **Result**: -10 Elo, killed at 197 games. W51-L57-D89 (48.5%).
- **Baseline**: net.nnue @ 911b468, post-LMR v2
- **Notes**: Base of 3 is the right floor. Reducing to 2 prunes too many early moves.

## 2026-03-10: LMP v2 — no improving bonus
- **Change**: Remove the +50% LMP limit bonus when improving (always use base formula).
- **Result**: **-38 Elo**, killed at 147 games. W32-L48-D67 (44.6%).
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: The improving bonus is critical — it prevents over-pruning when the position is getting better. Decisive failure.

## 2026-03-10: History pruning divisor v1 — 4000
- **Change**: LMR history adjustment divisor 5000 -> 4000 (more history influence).
- **Result**: **-32 Elo**, killed at 174 games. W42-L58-D74 (45.4%).
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: More history influence causes over-fitting to noisy history data.

## 2026-03-10: History pruning divisor v2 — 6000
- **Change**: LMR history adjustment divisor 5000 -> 6000 (less history influence).
- **Result**: -14 Elo, killed at 245 games. W67-L76-D102 (48.2%).
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Less history influence also hurts. Divisor of 5000 is well-calibrated in both directions.

## 2026-03-10: Singular extension margin — depth*2
- **Change**: Singular beta = ttScore - depth*2 (was depth*3; narrower margin = extend more often).
- **Result**: **-85 Elo**, killed at 116 games. W21-L48-D47 (38.4%). Catastrophic.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: depth*2 extends far too many moves, wasting search time. depth*3 is well-calibrated.

## 2026-03-10: Aspiration window delta=12
- **Change**: Aspiration window initial delta 15 → 12 (tighter windows).
- **Result**: **-18.4 Elo**, H0 accepted. W196-L239-D376 (811 games). LOS 2.0%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Tighter windows cause more re-searches, wasting time. Delta=15 is well-calibrated. Testing delta=18 (opposite direction) next.

## 2026-03-10: RFP improving margin 60→50
- **Change**: RFP improving margin depth*60 → depth*50 (tighter pruning when position improving).
- **Result**: -6.3 Elo, killed at 1391 games (inconclusive, trending reject). W374-L396-D621. LOS 18.3%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Even with NNUE's better eval, the improving margin can't be tightened further. 60 is already aggressive.

## 2026-03-10: Razoring gentle (375+d*90)
- **Change**: Razoring margin 400+depth*100 → 375+depth*90 (~7% tightening).
- **Result**: -2.5 Elo, killed at 1402 games (dead flat). W399-L409-D594. LOS 36.2%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Gentler than the v1 attempt (-32 Elo at 300+d*75). Still no gain — razoring is well-calibrated in both directions.

## 2026-03-10: IIR deeper reduction (2 at d≥10)
- **Change**: IIR reduces by 2 plies when depth≥10 and no TT move (was always 1).
- **Result**: +1.8 Elo, killed at 1394 games (inconclusive, essentially zero). W396-L387-D611. LOS 59.9%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Slight positive trend but too small to matter. Double IIR at deep nodes is neutral — the extra reduction doesn't save enough time to compensate.

## 2026-03-10: Aspiration window delta=18
- **Change**: Aspiration window initial delta 15 → 18 (wider windows, opposite of delta=12).
- **Result**: -7.0 Elo, killed at 1253 games (trending reject). W348-L372-D533. LOS 23.0%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Both directions tested (12 and 18). Delta=15 is optimal — tighter wastes time on re-searches, wider loses precision.

## 2026-03-10: Razoring loosening (425+d*110)
- **Change**: Razoring margin 400+depth*100 → 425+depth*110 (opposite of gentle tightening).
- **Result**: **-21.5 Elo**, H0 accepted. W186-L231-D307 (724 games). LOS 1.6%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Loosening loses even more than tightening. Both directions confirm 400+d*100 is optimal. Wider margins waste time searching hopeless positions.

## 2026-03-10: Singular extension depth threshold depth/2
- **Change**: Singular verification depth (depth-1)/2 → depth/2 (deeper verification, fewer extensions).
- **Result**: -14.5 Elo, killed at 841 games (trending reject). W231-L267-D343. LOS 6.4%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Fewer extensions = miss important moves. (depth-1)/2 is well-calibrated.

## 2026-03-10: NMP verification depth 12→10
- **Change**: NMP verification search threshold depth≥12 → depth≥10 (verify at shallower depths).
- **Result**: -13.1 Elo, killed at 850 games (trending reject). W223-L256-D371. LOS 7.2%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Verification at depth 10-11 costs too much time for insufficient zugzwang protection. Depth 12 is the right threshold.

## 2026-03-10: Improving gate on razoring
- **Change**: Skip razoring when position is improving (`&& !improving`).
- **Result**: -6.6 Elo, killed at 1484 games (trending reject). W397-L424-D663. LOS 16.4%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Improving detection doesn't help razoring. At depth 1-2, the improving signal is noisy — positions that are "improving" at these shallow depths aren't reliably getting better.

## 2026-03-10: Eval-based LMR adjustment
- **Change**: Reduce less when staticEval+200 < alpha (losing), reduce more when staticEval-200 > beta (winning).
- **Result**: -5.9 Elo, killed at 1480 games (trending reject). W391-L416-D673. LOS 18.9%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: The improving heuristic already captures position trajectory. Adding raw eval distance to alpha/beta is redundant and slightly harmful — it fights with the existing improving adjustment.

## 2026-03-10: History-based LMP threshold
- **Change**: Raise LMP limit by +2 for moves with history score > 4000 (harder to prune good-history moves).
- **Result**: -3.1 Elo, killed at 1476 games (flat/slightly negative). W413-L421-D642. LOS 32.5%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: LMP already benefits from move ordering — high-history moves appear early and avoid pruning naturally. Explicit history gate is redundant.

## 2026-03-10: 2-ply continuation history in LMR (full weight)
- **Change**: Add ply-2 continuation history to LMR reduction adjustment and history pruning. Full weight (same as ply-1).
- **Result**: -3.8 Elo, killed at 1925 games (dead flat then faded). W526-L543-D856. LOS 26.0%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Early positive signal (+1.4 at 1471 games) was noise. Ply-2 piece lookup is lossy (piece may be captured), adding noise. Half-weight variant still running.

## 2026-03-10: 2-ply cont history in move ordering
- **Change**: Add ply-2 continuation history to MovePicker quiet move scoring AND LMR/pruning.
- **Result**: -27.1 Elo, killed at 194 games (strongly negative early). W50-L68-D76. LOS 9.1%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Adding noisy ply-2 signal to move ordering is actively harmful — bad ordering cascades through the entire search. Pruning-only is safer.

## 2026-03-10: 2-ply cont history half weight (MERGED)
- **Change**: Add ply-2 continuation history to LMR and history pruning, at half weight (÷2).
- **Result**: **+11.0 Elo**, H1 accepted. W300-L268-D439 (1007 games). LOS 91.0%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Commit**: ab25488
- **Notes**: Full weight was flat (-3.8 at 1925 games); half weight works. Ply-2 history is noisier than ply-1 (piece may have been captured), so down-weighting is essential. Adding to move ordering was actively harmful (-27 Elo) — pruning/reduction only.

## 2026-03-10: Futility improving (+50 margin)
- **Change**: Add +50 to futility margin when position is improving.
- **Result**: -1.9 Elo, killed at 1115 games (flat). W313-L323-D479. LOS 40.6%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Early noise showed +15.6 at 205 games, faded to zero. Futility margins are already well-calibrated.

## 2026-03-10: SEE quiet improving gate
- **Change**: Loosen SEE quiet pruning threshold by -50 when improving.
- **Result**: -0.4 Elo, killed at 902 games (flat). W243-L244-D415. LOS 48.2%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Early +10.3 at 442 games faded to zero. SEE thresholds are about material exchange, not eval trajectory.

## 2026-03-10: History pruning with improving (tighter threshold)
- **Change**: Allow history pruning when improving, but with 2x stricter threshold (-4000*d vs -2000*d).
- **Result**: +0.0 Elo, killed at 898 games (dead flat). W251-L258-D389. LOS 50.0%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: The existing `!improving` gate is correct — extending history pruning to improving positions, even with a stricter threshold, doesn't help.

## 2026-03-10: LMP PV-node exemption (MERGED)
- **Change**: Add `beta-alpha == 1` to LMP gate, exempting PV nodes from late move pruning.
- **Result**: **+16.9 Elo**, H1 accepted. W217-L183-D301 (701 games). LOS 95.5%. LLR 2.96.
- **Baseline**: net.nnue @ ab25488, post-ContHist2
- **Commit**: d92b873
- **Notes**: At PV nodes, accuracy matters more than speed. Pruning late quiet moves at PV nodes risks missing the principal variation. Gate change, not margin — confirms that structural changes are higher-leverage than parameter tuning.

## 2026-03-10: Eval instability heuristic (MERGED)
- **Change**: Detect sharp eval swings from parent node (`|staticEval - (-parentEval)| > 200`). When unstable: skip history pruning, reduce LMR by 1, loosen SEE quiet threshold by 100cp.
- **Result**: **+12.8 Elo**, H1 accepted. ~830 games. LOS 92.6%. LLR 2.95.
- **Baseline**: net.nnue @ ab25488, post-ContHist2
- **Commit**: (pending)
- **Notes**: Novel heuristic — detects tactically volatile positions where pruning is dangerous. NNUE's accurate eval makes the 200cp swing meaningful (not eval noise). Opens a family of follow-up experiments: using instability to gate NMP, RFP, singular extensions; tuning the 200cp threshold.

## 2026-03-10: Capture history in SEE pruning
- **Change**: Modulate SEE capture pruning threshold by capture history: `seeThreshold += captHistVal/20`.
- **Result**: -0.7 Elo, killed at 940 games (fully regressed). W261-L263-D416. LOS 46.5%.
- **Baseline**: net.nnue @ ab25488, post-ContHist2
- **Notes**: Early signal was +50 at 90 games, +17 at 150, +5 at 600, then zero by 940. Textbook early noise fade. SEE thresholds are about material exchange accuracy — capture history doesn't meaningfully improve the threshold. The capture history signal is already used in capture ordering (MVV-LVA + captHist), which is the right place for it.

## 2026-03-10: Counter-move LMR reduction
- **Change**: Reduce LMR by 1 for moves matching the counter-move heuristic (`if move == counterMove { reduction-- }`).
- **Result**: -1.7 Elo, killed at 608 games (dead flat). W174-L177-D257. LOS 43.6%.
- **Baseline**: net.nnue @ ab25488, post-ContHist2
- **Notes**: Counter-move already gets priority in move ordering (tried before quiets). Reducing LMR for it doesn't add value — the ordering benefit is sufficient.

## 2026-03-10: Double extension depth gate (depth≥12)
- **Change**: Add `depth >= 12` to double singular extension condition (was no depth gate beyond singular's depth≥10).
- **Result**: -15 Elo, killed at 437 games (consistently negative). W120-L138-D179. Win% 47.9%.
- **Baseline**: net.nnue @ ab25488, post-ContHist2
- **Notes**: Restricting double extensions to deeper nodes loses them when they matter most (depth 10-11). The existing threshold is correct. ~40-50 games may have been affected by CPU contention from concurrent training.

## 2026-03-10: Pawn history table
- **Change**: New pawn-structure-aware history table indexed by `[pawnHash%512][piece][toSquare]`. Added to quiet and evasion move scoring, updated on beta cutoffs (bonus for cutoff move, penalty for tried quiets). ~832KB per thread.
- **Result**: **+22.2 Elo**, H1 accepted. 533 games. LLR 3.01.
- **Baseline**: net.nnue @ 7f836a2, post-EvalInstability
- **Commit**: (this commit)
- **Notes**: Pawn structure changes slowly, making it a stable, low-noise signal for move ordering. The table captures which piece-to-square patterns work well in specific pawn structures. Major win — validates the hypothesis that move ordering has significant room for improvement.

## 2026-03-10: Continuation history 2x weight
- **Change**: Double continuation history weight in quiet and evasion move scoring: `score += 2 * int(mp.contHist[piece][m.To()])` (was `1 *`).
- **Result**: **+27.9 Elo**, H1 accepted. 461 games. LLR 3.00.
- **Baseline**: net.nnue @ 7f836a2, post-EvalInstability
- **Commit**: (this commit)
- **Notes**: Continuation history (what worked after the opponent's previous move) is a highly predictive ordering signal. Doubling its weight amplifies this signal relative to main history. Combined with pawn history, total move ordering improvement is ~50 Elo from this session. Note: ply-2 cont hist at full weight was previously harmful (-27 Elo), but ply-1 benefits from amplification.

## 2026-03-11: New NNUE net (MERGED)
- **Change**: Updated net.nnue from net-new.nnue (additional training).
- **Result**: **+28.6 Elo**, H1 accepted. W167-L128-D180 (475 games). LOS 98.8%.
- **Baseline**: net.nnue @ 1e9f490, post-PawnHist
- **Commit**: 16d6592

## 2026-03-11: QS TT move (MERGED)
- **Change**: Pass TT move to quiescence search InitQuiescence, start at stageTTMove instead of stageGenerateCaptures.
- **Result**: **+7.1 Elo**, H1 accepted. W386-L358-D624 (1368 games). LOS 84.8%.
- **Baseline**: net.nnue @ 1e9f490, post-PawnHist
- **Commit**: 204e62d
- **Notes**: Low-hanging fruit — QS was ignoring TT information for move ordering.

## 2026-03-11: Continuation history 3x weight (MERGED)
- **Change**: Cont history weight in quiet and evasion move scoring 2x→3x.
- **Result**: **+3.8 Elo**, H1 accepted. W644-L620-D921 (2185 games). LOS 75.0%.
- **Baseline**: net.nnue @ 9ef020a, post-QS-TTMove
- **Commit**: dab4bd4
- **Notes**: Continuing the trend from 1x→2x (+27.9 Elo). Smaller gain as expected. Testing 4x next to bracket the optimum.

## 2026-03-11: Pawn history in LMR
- **Change**: Add pawn history scores to LMR reduction adjustment (alongside main/cont history).
- **Result**: **-7.8 Elo**, H0 accepted. W742-L801-D1100 (2643 games). LOS 6.7%.
- **Baseline**: net.nnue @ 9ef020a, post-QS-TTMove
- **Notes**: Pawn history is useful for move ordering but too noisy for LMR adjustment. Consistent with pattern: ordering signals don't always transfer to pruning/reduction.

## 2026-03-11: Eval instability threshold 150
- **Change**: Instability threshold 200→150 (detect more volatile positions).
- **Result**: -2.5 Elo at 3281 games (flat, killed). LOS 29.1%.
- **Baseline**: net.nnue @ 9ef020a, post-QS-TTMove
- **Notes**: Lower threshold fires too often, diluting the signal. 200 is better.

## 2026-03-11: Eval instability threshold 300
- **Change**: Instability threshold 200→300 (only flag extreme swings).
- **Result**: **-6.8 Elo**, H0 accepted. W892-L955-D1353 (3200 games). LOS 7.1%.
- **Baseline**: net.nnue @ 9ef020a, post-QS-TTMove
- **Notes**: Higher threshold misses too many volatile positions. 200 is near-optimal. Bracketed: 150 flat, 300 negative.

---

## Ideas Not Yet Tested
- **Singular extension depth threshold**: Currently (depth-1)/2. Could try depth/2 or depth/3.
- **Double singular threshold**: Currently singularBeta - depth*3. Could try depth*2.
- **Continuation history weight tuning**: 3x validated. Testing 4x to bracket the optimum.
- **Pawn history weight tuning**: Currently 1x in ordering. Could try 2x.

## Key Patterns Observed

1. **Node-level pruning benefits from NNUE tightening** (RFP, LMR) — these prune entire subtrees based on eval, and NNUE's accuracy makes this safe.
2. **Per-move pruning is more sensitive** (futility, SEE, LMP) — errors compound across many moves, so margins need more slack.
3. **NMP, razoring, LMP, history divisor, singular margin are all well-tuned** — don't revisit unless eval changes significantly.
4. **Self-play Elo ~2x cross-engine Elo** for search changes. Calibrate expectations.
5. **Persistence matters** — correction history took 3 attempts. Tune before rejecting sound ideas.
6. **LMR constant C=1.5 is near-optimal** — C=1.25 and C=1.375 both tested at zero. Further tuning has diminishing returns.
7. **LMP improving bonus is critical** — removing it costs ~38 Elo. The improving heuristic correctly identifies positions where more moves should be searched.
8. **History divisor 5000 is optimal** — both directions (4000 and 6000) lose Elo.
9. **Aspiration delta=15 is optimal** — both directions (12 and 18) lose Elo. Bracketed.
10. **Razoring 400+d*100 is optimal** — three attempts (300+d*75, 375+d*90, 425+d*110) all lose. Do not revisit.
11. **Singular depth (depth-1)/2, NMP verify depth 12 are well-calibrated** — don't change.
12. **Margin-tuning has diminishing returns** — after the initial NNUE-driven RFP/LMR wins, most parameters are already near-optimal. New features (structural changes) are more likely to gain than parameter adjustments.
13. **2-ply continuation history needs half weight** — full weight adds noise (ply-2 piece may be captured); adding to move ordering is harmful (-27 Elo). Pruning/reduction only, at ÷2.
14. **Improving heuristic doesn't help per-move pruning** — futility (+50), SEE (-50), history pruning (stricter threshold) all tested neutral. The improving signal is already captured by RFP and LMR adjustments.
15. **Move ordering has massive room for improvement** — pawn history (+22.2) and cont-hist 2x weight (+27.9) combined for ~50 Elo from a single session. New history signals and weight tuning are high-value experiments.
16. **Eval instability threshold 200 is optimal** — 150 flat, 300 negative. Bracketed.
17. **Pawn history doesn't transfer to LMR** — useful for ordering (-7.8 Elo in LMR). Ordering signals don't always work for pruning/reduction.

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

## 2026-03-11: Continuation history 4x weight
- **Change**: Cont history weight in quiet and evasion move scoring 3x→4x.
- **Result**: -5.0 Elo at 1055 games (killed, trending negative). LOS 26.9%. LLR -0.57.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Overshoots the optimum. 3x is the sweet spot — confirmed by bracketing (2x +27.9, 3x +3.8, 4x negative). Do not increase further.

## 2026-03-11: Pawn history pruning (weight=2)
- **Change**: Apply pawn history score to quiet move pruning and LMR reduction (scaled to ±2x weight).
- **Result**: **-17.0 Elo**, H0 accepted. W265-L313-D402 (980 games). LOS 2.3%.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Confirms pawn history in pruning/LMR is harmful (first attempt -7.8 Elo). Ordering only.

## 2026-03-11: QS delta pruning margin
- **Change**: Tighten quiescence delta pruning margin.
- **Result**: -4.6 Elo at 1069 games (killed, trending negative). LOS 28.7%. LLR -0.47.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Current delta margin is well-calibrated. Do not tighten.

## 2026-03-11: Correction history clamp 128→96
- **Change**: Reduce correction history clamp from 128 to 96 (less aggressive corrections).
- **Result**: -0.7 Elo at 1048 games (killed, flat). LOS 46.8%. LLR 0.40.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Slightly worse. Testing 192 next (more aggressive corrections).

## 2026-03-11: Node fraction time management
- **Change**: Track best move's share of root nodes. High fraction (>0.9) → 0.8x time, low fraction (<0.3) → 1.5x time, (<0.5) → 1.3x time.
- **Result**: +2 Elo at ~960 games (killed, flat). LOS ~53%.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Sound concept (used in Stockfish) but thresholds may need tuning, or the benefit is too small at these time controls. Could revisit with different scaling factors.

## 2026-03-11: Pawn history 2x weight in ordering
- **Change**: Double pawn history weight in quiet move scoring: `score += 2 * pawnHist[...]` (was 1x).
- **Result**: -4 Elo at ~960 games (killed, slightly negative). LOS ~38%.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: 1x is the right weight. Unlike cont history (which benefited from 1x→3x amplification), pawn history is already well-scaled.

## 2026-03-11: Main history 2x weight in ordering
- **Change**: Double main history weight in quiet move scoring.
- **Result**: -1 Elo at ~950 games (killed, flat). LOS ~47%.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Main history weight is already well-calibrated. The cont history amplification trick doesn't generalize to all history tables.

## 2026-03-11: Correction history clamp 128→192
- **Change**: Increase correction history clamp from 128 to 192 (more aggressive corrections).
- **Result**: +1 Elo at 968 games (killed, flat). LOS ~52%.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Bracketed: 96 was -0.7, 192 is +1. Both flat. 128 is near-optimal. Do not revisit.

---

## 2026-03-11: Net blended (lambda=0.05)
- **Change**: NNUE net trained with lambda=0.05 result blending (same epochs, LR, dataset as current net).
- **Result**: -3.1 Elo at 1667 games (killed, flat/negative). LOS 32.4%.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Lambda=0.05 didn't help. User plans further training iterations before resubmitting.

## 2026-03-11: TM directional score drop
- **Change**: Replace abs(scoreDelta) with directional scoreChange. Score drop (<-30 → 1.5x, <-15 → 1.25x time), score improve (>30 → 0.85x time).
- **Result**: -0.6 Elo at 1686 games (killed, flat). LOS 46.2%. LLR 0.68.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Directional TM doesn't help at these thresholds. The existing abs(scoreDelta) approach may already capture what matters. Could revisit with more aggressive thresholds or combined with node fraction.

## 2026-03-11: Quiet check bonus in move ordering
- **Change**: +5000 score bonus for quiet moves that give direct check in generateAndScoreQuiets().
- **Result**: **-10.9 Elo**, H0 accepted. 1682 games. LOS 4.6%.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Checking moves are already handled well by check extension. Boosting them in ordering disrupts history-based quality. Do not revisit.

## 2026-03-11: Counter-move history table
- **Change**: New [13][64][13][64]int16 history table indexed by [prevPiece][prevTo][piece][to]. Used in quiet/evasion scoring (1x weight), history pruning, and LMR adjustment.
- **Result**: -7.2 Elo at 1683 games (killed, trending H0). LOS 13.0%. LLR -1.73.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Counter-move history adds noise rather than signal. The existing cont history (ply-1 piece/to) already captures this relationship. The extra [prevPiece][prevTo] indexing fragments the table too much for reliable statistics. Do not revisit without much larger games/deeper searches.

## 2026-03-11: Singular extension depth 10→8
- **Change**: Lower singular extension depth threshold from depth>=10 to depth>=8.
- **Result**: **-66.8 Elo** at 100 games (killed immediately). LOS 0.8%.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Catastrophic. The verification search at half of depth 8 (=depth 4) is too expensive and unreliable for the benefit. Singular extensions need to remain at deep nodes where the TT entry is trustworthy. Testing wider singular margin (depth*2 instead of depth*3) next, which fires more often at depth>=10 without the cost of shallower verification.

## 2026-03-11: Check Ext SEE Filter (Killed ~250 games)
- **Change**: Only extend checks where SEE(check move) >= 0. Filters out checks where the checking piece can be captured for material gain.
- **Result**: -8.4 Elo at 251 games (killed, trending negative). W66-L72-D113.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: SEE filter is too coarse — many valuable checks involve sacrificial piece placement. The issue isn't material cost but whether the check restricts the king.

## 2026-03-11: NMP +1 Reduction with Rooks in EG (Killed ~252 games)
- **Change**: In endgame (non-pawn-king pieces < 10), when both sides have rooks, add +1 to NMP reduction. Theory: rook endgames are drawish and NMP can be more aggressive.
- **Result**: -12 Elo at 252 games (killed, trending negative). W68-L75-D109.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: More aggressive NMP in rook endgames actually hurts — rook endgames have subtle zugzwang-like positions where NMP is already borderline. Existing NMP calibration is correct.

## 2026-03-11: EG Futility 75% Margin (Killed ~244 games)
- **Change**: Tighten futility pruning margin by 25% in endgame (non-pawn-king < 10): `margin = (100 + lmrDepth*100) * 3/4`.
- **Result**: -14 Elo at 244 games (killed, trending negative). W69-L74-D101.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Endgame positions are more sensitive to futility errors (single pawn = game-deciding). Tighter margins prune moves that actually matter. Confirms pattern #2: per-move pruning needs slack.

## 2026-03-11: Singular Margin depth*2 (Killed ~254 games)
- **Change**: Widen singular extension margin from `ttScore - depth*3` to `ttScore - depth*2`, making singular extensions fire more often.
- **Result**: +3.5 Elo at 254 games (killed, regressed from early +27 to flat). Also retested with `restart=on`: -10 Elo at 122 games, confirming no gain.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Early results were noise. The current singular margin (depth*3) is well-calibrated. Wider margin fires more but many of those extra firings don't identify truly singular moves.

## 2026-03-11: LMR Reduction -1 in EG (Killed ~69 games)
- **Change**: Reduce LMR reduction by 1 in endgame (non-pawn-king pieces < 10). Theory: EG has higher LMR re-search rate, suggesting reductions too aggressive.
- **Result**: -25 Elo at 69 games (killed early).
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Less LMR in endgame loses Elo. The higher re-search rate in EG is acceptable; reducing less means searching too many moves fully.

## 2026-03-12: History Divisor 5000→7000 (Killed ~100 games, vs PersistHistory baseline)
- **Change**: Increase LMR history adjustment divisor from 5000 to 7000, dampening history's influence on reductions. Tested on top of persist-history fix (richer history data).
- **Result**: -27 Elo at ~100 games (killed).
- **Baseline**: chess-persist-history (history tables persist across searches)
- **Notes**: Dampening history influence is wrong direction. With persistent history providing richer/higher-magnitude data, the current divisor of 5000 may already be correct or even too high. Don't increase further; consider testing lower (4000 or 3500).

---

## Ideas Not Yet Tested
- **Singular extension margin widening**: Tested, no gain. Do not revisit.
- **EG loose check filter** (in progress): Skip check extension when endgame AND king has 4+ escape squares. Data: 98% of EG checks are loose (4+ escapes) with only 7.2% cutoff rate vs 25-35% for tight checks. Targets 2.9M wasted nodes.
- **Counter-move before killers in EG** (in progress): Data shows counter-move is 23% effective in EG vs 15% MG.
- **LMR reduction -1 in EG** (in progress): EG has 1.5% LMR re-search rate vs 1.1% MG, suggesting reductions are slightly too aggressive.
- **Endgame-specific move ordering**: TT move dominates EG cutoffs (51.4% vs 13.9% MG), suggesting other ordering signals need different weights.
- **Queen check extension filter**: Queen checks are 86% of EG checks but mostly shuffling. Could selectively reduce extension for distant queen checks.

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
18. **EG-specific search parameters lose Elo** — Tested EG futility (75% margin), NMP +1 with rooks, check ext SEE filter. All negative. The existing "one size fits all" parameters are well-calibrated across phases. History tables naturally adapt during gradual MG→EG transition.
19. **Check extension quality varies dramatically by phase** — MG: 35% loose (4+ escapes, 10% cutoff), EG: 98% loose (7% cutoff). 86% of EG checks are queen shuffling. Structural check quality filters > phase-specific parameter tweaks.

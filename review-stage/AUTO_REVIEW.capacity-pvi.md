# Auto Review Loop — capacity-matched independent leakage measure

## Round 1 (2026-06-20)

### Assessment (Summary)
- Score: 5/10
- Verdict: almost (central claim NOT established; "almost" only for the narrow L12-DP statement)
- Key criticisms (ranked):
  1. Single intervention/single layer (L12 input-DP) — tracking one noise knob ≠ prediction. Fix: multi-layer + one non-DP intervention (split-depth/obfuscation).
  2. Selectivity carries the result (ρ 0.93 vs raw 0.64); shuffle floor drifts −3.79→−1.85 → correction is part of signal. Fix: predeclare selectivity, decompose raw/real/shuffle, leave-one-out, partial-rank corr controlling for noise ratio r; tune dim/reg by floor health not TTRSR.
  3. n=7 too small. Fix: seeds/prompt-split CIs (bootstrap), denser ε between 512–256.
  4. randproj floor −3.9 not ≈0 (pca −1.9, knn −1.1 healthier). Fix: sweep dim {32,64,128,256}+reg, pick smallest with floor∈[−1,+1], then evaluate faithfulness.
  5. Independence structurally plausible but redundancy w/ retrieval-PVI untested. Fix: per-instance collinearity (logit/CKA/R²); does cap-PVI add beyond retrieval-PVI/freq/norm/position.
  6. Family fragility (gauss fails, knn weak). Fix: pick family by M0/M1 only, freeze for M2; also run pca_softmax + lower-dim randproj M2.

### Reviewer Raw Response
<details><summary>full</summary>

Score 5/10. Verdict ALMOST, not established. Established: capacity matching fixes the broken class-PVI estimator and preserves structural independence. Not established: that the family GENERALLY/faithfully predicts TTRSR. [W1] single L12 DP sweep — both TTRSR and cap-PVI driven by same noise knob → tracking along one path, not prediction; fix: several layers + one non-DP intervention. [W2] selectivity does the work (raw ρ0.64 vs sel ρ0.93); floor not near zero, drifts −3.79→−1.85 so correction is part of signal; nuance: dropping ε=256 still leaves ρ≈0.89, but pre-collapse selectivity barely moves (8.05→7.68) while TTRSR moves a lot → "flat then cliff"; fix: predeclare selectivity, decompose, leave-one-out, partial/rank corr controlling for r, tune by floor health only. [W3] n=7: ρ0.929 → permutation p≈0.007 if only planned test, but with family/dim/raw-vs-sel choices not enough; fix seeds/CIs + denser ε. [W4] floor −3.9 not ≈0; fix dim+reg sweep pick floor∈[−1,1]. [W5] not circular under your defn, but test redundancy w/ retrieval-PVI per-instance + added value over controls. [W6] family fragility → freeze family by M0/M1, run pca + lower-dim randproj M2. Bottom line: Almost for narrow "capacity-matched token-ID selectivity tracks TTRSR on this L12 DP sweep"; NOT established for the general claim.
</details>

### Status
- continuing to round 2. Difficulty: medium.

## Round 2 (2026-06-20)
### Assessment
- Score: 6.5/10 — Verdict: almost (STOP threshold met; continuing one round for the cheap path to "established")
- Accepts the scoped claim: capacity-matching + control-anchored reg fixes the pathological class-PVI estimator → structurally independent token-ID measure that moderately-strongly tracks TTRSR across layer×DP, exposing meaningful divergences. Not yet the unqualified "faithfully tracks."
- Remaining (ranked): (1) single defense — add ONE non-DP intervention (cheap rep-space: iso noise / PCA ablation / split-depth); (2) make L20 divergence a RESULT — show token-id accuracy stable while embedding cosine/retrieval collapses; (3) pooled ρ hides repeated-measure structure — macro-avg per-layer ρ + mixed/fixed-effects; (4) partial ρ|retr 0.275 weak — give CI/perm-p + residual plot; (5) l2 selection auditability — show l2 sweep, selection by floor health not TTRSR; (6) floor precision — report mean±sd, −1.53 is just outside [−1.5,1.5].
### Status: continuing to round 3 (non-DP intervention + L20 mechanism). Difficulty: medium.

## Round 3 (2026-06-20)
### Assessment
- Score: 7/10 — Verdict: ESTABLISHED (scoped) IF reframed; ALMOST if still claiming "PVI-in-bits faithfully tracks".
- Conceptual center (per reviewer): original failed object = high-d class-PVI **bits**; FIXED object = capacity-matched independent **token-ID reader**; best readout = bounded **token-ID accuracy**; PVI/selectivity = useful but calibration-sensitive (partially rescued).
- Must-fix (framing, not experiments): (1) STOP calling accuracy "V-information" — split all claims into `reader accuracy` (robust predictor) vs `PVI bits/selectivity` (fragile, partially rescued). (2) calibration diagnostic (NLL/conf/ECE) on the bad cells to prove the log-loss-artifact story. (3) report within-layer/macro-avg ρ as PRIMARY, partial ρ|r as secondary (class-PVI's +0.613 partial|r shows the statistic can mislead under layer×ε structure). (4) one model only → scope to gemma or add cheapest cross-model subset. (5) dim16 sensitivity (floor −1.10 vs dim64 −1.53).

### Status: STOP — positive assessment (7/10, established-scoped). Loop complete at round 3/4.

## FINAL SUMMARY
**Objective** (fix the independent token-id V-family to track the attack, or formally retire it): **ACHIEVED, scoped + honest.**
- class-PVI's failure was an estimator regime (d>n_val), not the family. **Capacity-matching (PCA→dim<n + linear reader)** removes the catastrophe: shuffle floor −49→−1.5 b (dim-anchored, NOT l2-anchored), monotone, ~0.3× class-PVI cost; gauss reader fails, pca_softmax is the pick.
- **The robust fixed measure is the reader's token-ID accuracy** (bounded), which tracks the inversion attack (TTRSR) at ρ **0.82–1.0 across all layers** under representation-space defenses (PCA-ablation, isotropic noise), and at early/mid layers under input-DP (partial ρ|r≈0.71, beyond the noise knob). The **unfixed class-PVI is within-layer anti-correlated** everywhere.
- **PVI-in-bits is only partially rescued**: the −48 floor and "rise under noise" are unbounded-log-loss artifacts; bits track only with regularization and remain fragile under iso-noise / late-layer DP.
- **Characterized divergence (a result, not a failure)**: under propagated input-DP at late layers (L20), token-ID decodability survives while embedding-reconstruction (the attack) collapses → cap diverges from TTRSR (ρ −0.21, in the bounded readout too). This localizes WHAT input-DP protects (embedding geometry) vs not (id-decodability).
- **Independence**: token-ID target, never the embedding table; ρ(cap, retrieval-PVI)=0.66–0.76 (<0.9); the capacity fix changes only `dim`, not the target → not "becoming the attack."

## Method Description
`v_information_capacity` (src/talens/measures/vinfo_capacity.py): an attack-independent leakage probe over token-ID classes. Pipeline: standardize hidden state X (train stats) → reduce dimensionality to `dim<n_val` (PCA via GPU covariance-eigh, or random projection) → a cheap reader (linear softmax / Gaussian class-conditional / kNN) predicts the token id; null = class prior; Hewitt–Liang shuffle control. Two readouts: PVI-in-bits (= log2 q(y|x) − log2 prior, calibration-sensitive) and the reader's bounded top-1 token-ID accuracy (robust). It never reads the token-embedding table, so it is structurally independent of the ridge→embedding inversion attack whose top-1 recovery (TTRSR) is the ground truth. Faithfulness is evaluated by Spearman vs TTRSR across (layer × defense-strength) grids for input-DP, PCA-subspace-ablation, and isotropic-hidden-state-noise defenses.

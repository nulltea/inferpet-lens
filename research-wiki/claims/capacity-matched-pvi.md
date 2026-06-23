---
type: claim
node_id: claim:capacity-matched-pvi
name: "Capacity matching repairs the independent token-id V-family (scoped, gemma-2-2b)"
description: "Empirical + estimator-theory, jury-PARTIAL/scoped: PCA capacity matching removes the d>n_val class-PVI shuffle-floor catastrophe (−49.7→−1.9 bits, dim-anchored, 0.57x cost) into an attack-INDEPENDENT token-id probe whose bounded reader accuracy tracks the ridge inversion attack at rho 0.80–1.00 under at-layer representation-space defenses; PVI-bits is auxiliary (log-loss unbounded-below)."
node_type: claim
status: drafted
provenance: ".aris/traces/result-to-claim/2026-06-23_run01/ ; refine-logs/resid-capacity-pvi/"
tags: ["headline", "empirical", "theory", "partial", "resid-capacity-pvi", "PVI", "capacity-matched"]
date: 2026-06-23
updated: 2026-06-23
---

# Capacity matching repairs the independent token-id V-family (scoped, gemma-2-2b)

**status:** `drafted` (empirical jury `PARTIAL`/scoped; estimator-theory lemma `verified` by proof-checker — see Theory)

## Statement (scoped after jury, 2026-06-23)
The independent token-id V-information family (class-PVI: a reader that predicts the *token id* from a
hidden state and never reads the embedding table) failed not because the family is wrong but because
of an **estimator-regime pathology**. With `d > n_val` features the reader interpolates and its
held-out / shuffle-control PVI-in-bits is driven unbounded below — the catastrophic shuffle floor
(−44.9 to −51.4 bits across depth in gemma-2-2b). **Capacity matching** — reduce to `dim < n_val` via
a train-only PCA (covariance-eigh) before a calibrated linear-softmax reader — bounds the floor
(−1.23 to −1.27 bits at every depth, −1.9 in the M1 screen), at **0.57× the cost** of class-PVI, with
the fix **dim-anchored, not l2-anchored**. The repaired probe is attack-independent by construction
(only the dim reduction changes; no embedding/attack information is added), and its **bounded reader
top-1 accuracy** tracks the ridge embedding-inversion attack (TTRSR) at **Spearman 0.80–1.00 under
at-layer representation-space defenses (PCA-subspace ablation, isotropic hidden-state noise) at
L5/L12/L20**, where the unfixed class-PVI is uninterpretable. **PVI-in-bits is only auxiliary**: it is
calibration-sensitive (unbounded log-loss) and is *more fragile across these runs* than accuracy; do
not call the accuracy "V-information".

**Honest scope:** single model (gemma-2-2b), single seed; the accuracy readout (not the bits) is the
object that tracks; generality across models/seeds is a named firm-up, not claimed.

## Empirical status (jury-gated, NOT self-certified)
`PARTIAL` / scoped (Codex xhigh result-to-claim, 2026-06-23, confidence medium-high; trace
`.aris/traces/result-to-claim/2026-06-23_run01/`). Integrity audit **WARN, no FAIL** — probe≠attack
circularity check **PASS** (`refine-logs/resid-capacity-pvi/EXPERIMENT_AUDIT.md`; Codex independently
re-derived the Spearmans). Prior 3-round auto-review reached 7/10 "established-scoped".

## Evidence chain (bits canonical + per-secret readout; all gemma-2-2b)
- **R1 floor screen** (`results/capacity_screen.json`, L12, 3 seeds): class-PVI real 4.96 b / shuffle
  **−49.7 b** (catastrophe) → pca_softmax 5.39 b / **−1.9 b**, cost **0.57×**; gauss fails (−32.6),
  knn weak. Fix is dim-anchored (l2 only trades signal). Floor repair holds at **every depth**
  (`results/localdp_depth_L0_5_12_20.json`): cap floor −1.25/−1.23/−1.24/−1.27 vs class
  −44.9/−50.2/−51.4/−48.1 at L0/L5/L12/L20.
- **R2 representation-space defenses** (`results/nondp_intervention.json`, dim64): Spearman(cap reader
  accuracy, TTRSR) = PCA-ablation 0.80/0.90/0.90, iso-noise 1.00/1.00/1.00 at L5/L12/L20; CLUB tracks
  identically. This is the attack-predictive regime — no depth dependence.
- **R4 readout split**: reader accuracy (bounded) tracks at ρ 0.80–1.00; PVI-bits fragile under
  iso-noise (ρ 0.40–0.70). Report accuracy primary, bits auxiliary.
- **Independence**: `v_information_capacity(X, y)` takes only features + token-id labels, never the
  embedding table (`vinfo_capacity.py`); ρ(cap, retrieval-PVI)=0.66–0.76 (<0.9). The only change
  class-PVI→cap-PVI is the PCA dim reduction.

## Theory — the estimator-regime lemma (proof inline; verified by /proof-checker, thread 019ef5d7)

Reader = a conditional model `q(·|x)` over `K ≥ 2` token classes with marginal prior `p`. Per-example
PVI-in-bits `pvi(x,y) = log2 q(y|x) − log2 p(y)`; the functional `PVI(q) = E[pvi(x,Y)]`. Using
`E[−log2 p(Y)] = H(Y)` (prior entropy, finite), `PVI(q) = E[log2 q(Y|x)] + H(Y)`. Top-1 accuracy
`acc(q) = P(argmax_c q(c|x) = Y)` under any fixed measurable tie-break.

### Lemma 1 (PVI-in-bits is unbounded below over readers; accuracy is not)
Over the class of readers permitted to assign arbitrarily small positive class probabilities, `PVI(q)`
has **no finite lower bound**, whereas `acc(q) ∈ [0,1]`. ∎

**Proof.** Fix a class `c` with `π_c := P(Y=c) > 0`. For `ε ∈ (0,1)` define the reader
`q_ε(c|x) = ε` for all `x`, distributing `1−ε` over the other classes. Since `log2 q ≤ 0` for every
class, `E[log2 q_ε(Y|x)] = Σ_{c'} π_{c'} E[log2 q_ε(c'|x) | Y=c'] ≤ π_c · log2 ε` (drop the other
non-positive terms). Hence `PVI(q_ε) ≤ π_c log2 ε + H(Y) → −∞` as `ε → 0+`. The construction depends
only on `x` (never on the realized label), so it is a valid reader. Accuracy is the expectation of an
indicator, hence in `[0,1]` regardless of confidence. ∎

### Proposition 2 (interpolation realizes the catastrophe on the empirical control)
Consider the *empirical* control PVI the estimator actually computes: a reader fit on `n` training
points with `d ≥ n` features, evaluated by an empirical mean over a control in which the labels `Ỹ_i`
are a non-identity permutation of the originals `Y_i`. Suppose the fit **interpolates** —
`q_t(Y_i | x_i) → 1` for every `i` as training proceeds (the generic attractor of unregularized
log-loss GD on separable data: `‖θ_t‖ → ∞` toward the max-margin direction, Soudry et al., JMLR 2018,
with the multiclass extension of Lyu–Li 2020). Let `S = {i : Ỹ_i ≠ Y_i}`; a non-identity permutation
of a non-constant labeling gives `|S| ≥ 1`. For `i ∈ S`, `argmax_c q_t(c|x_i) = Y_i ≠ Ỹ_i`, so
`q_t(Ỹ_i | x_i) → 0` and `log2 q_t(Ỹ_i|x_i) → −∞`. The empirical control PVI is a *finite* sum
`(1/n) Σ_i [log2 q_t(Ỹ_i|x_i) − log2 p̂(Ỹ_i)]`; every term is bounded above and at least one (`i ∈ S`)
`→ −∞`, so `PVI_control(θ_t) → −∞`. ∎

**Scope (per proof-checker).** A finite-sum divergence on the **in-sample interpolation regime** — no
limit/expectation interchange, no held-out-confidence claim. It gives the asymptotic `−∞` *mechanism*
consistent with the observed finite floors (−44.9…−51.4 bits); the realized magnitude depends on
stopping time, precision, and any logit cap, so the theory explains the catastrophe's *existence*, not
the exact numbers. `K=2` is fully rigorous via Soudry et al.; `K>2` uses the multiclass max-margin
extension.

### Proposition 3 (a bounded-logit reader has a finite, dimension-independent PVI floor)
If every evaluated logit satisfies `|z_c(x)| ≤ B` for a cap `B` fixed **independently of the feature
dimension** (e.g. an explicit confidence cap, or `‖x‖ ≤ R` after projection and `‖W‖ ≤ C` giving
`B = RC`), then for every class
`q(c|x) = e^{z_c}/Σ_{c'} e^{z_{c'}} ≥ e^{−B}/(e^{−B} + (K−1)e^{B}) = 1/(1+(K−1)e^{2B}) =: σ_min > 0`.
Hence `log2 q(Y|x) ≥ log2 σ_min`, and `PVI(q) = E[log2 q(Y|x)] + H(Y) ≥ log2 σ_min + H(Y) ≥
log2 σ_min`, a **finite floor depending only on `(B,K)`**, independent of `d`. ∎

**What the theory does and does not give (scoped per proof-checker).** It proves: (i) PVI-in-bits can
diverge under overconfidence while accuracy cannot (Lemma 1) — so accuracy is the readout that *cannot*
blow up, not "robust" in any broader sense; (ii) an interpolating fit drives the empirical control PVI
to `−∞` (Prop 2) — the existence-mechanism of the floor; (iii) a **bounded-logit** reader has a finite,
`d`-independent floor (Prop 3). Crucially, **`k < n_val` is not a step in the Prop 3 bound** — the
finite floor follows from the logit cap alone. The dimension reduction is the *practical enabler*: it
keeps the fit out of the Prop-2 interpolation regime so a fixed, modest `B` is attainable. This
reconciles the empirical "dim-anchored, not l2-anchored" finding (l2 at fixed `d ≥ n` tunes `B` but
does not escape interpolation; cutting `dim < n_val` does) — but that reconciliation is an
*interpretation*, not part of the proof. The theory does **not** prove the bounded accuracy *tracks the
attack* — that correlation is the empirical content (R2), scoped to gemma-2-2b.

## Open (queued firm-ups, not this phase)
Multi-seed R1/R2 with bootstrap CIs; PCA floor screen reported at all depths (data exists, fold in);
cross-model replication (gemma-2-2b only); dim16 sensitivity (floor −1.10 vs dim64 −1.53);
calibration diagnostic (NLL/ECE) on the bits-fragile cells to *measure* Lemma 1's mechanism.

## Connections
Companion empirical-divergence claim [[depth-decoupling-input-dp]] (same surface, the L20 limit).
Independence backbone [[threat-model-fairness]]. MI comparator behaviour [[mi-monotone-gaussian]].
_Edges recorded in `graph/edges.jsonl`._

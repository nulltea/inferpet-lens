# resid-capacity-pvi — consolidated standardized results (bits canonical + per-secret readout)

**Surface:** residual-stream (`resid_post`) and the L0 embedding observation of a token.
**Secret kind:** `token_id` (the token behind the representation).
**Threat model:** WEIGHTS-PUB honest-but-curious — adversary knows weights + the token-embedding
table + published defense params; synthesizes unlimited `(defended-rep, token)` pairs
(`claim:threat-model-fairness`).
**Probe under test:** `v_information_capacity` (`src/talens/measures/vinfo_capacity.py`) — a
*capacity-matched* member of the independent token-id V-family. Pipeline: standardize hidden state
X (train stats) → reduce to `dim < n_val` (PCA via GPU covariance-eigh, or random projection) → a
cheap reader (linear softmax / Gaussian / kNN) predicts the **token id**; null = class prior;
Hewitt–Liang shuffle control. It **never reads the token-embedding table**, so it is structurally
independent of the ridge→embedding inversion attack whose top-1 recovery (TTRSR) is the ground truth.
**Attack (ground truth):** ridge resid→embedding inversion, graded by TTRSR (top-1 token recovery
rate, shuffle-controlled).
**Metric convention (CLAUDE.md):** canonical **bits** = cap-PVI selectivity (real − shuffle), kind
`capacity_v_info`, with **CLUB** `I(rep;token)` (kind `mi_upper_bound`) as the independent MI
comparator; per-secret **readout** = `token_top1_recovery_rate` — the probe's own **reader accuracy**
(`cap_acc`, the readout that is *more stable across these runs* than the bits) rendered beside the
attack's **TTRSR**. The raw runs (predating the reporting layer) store plain dict fields
(`cap_pvi_selectivity`, `cap_acc`, `club_bits`); the canonical rows in this document were
**re-rendered** from those raw fields through `src/talens/report.py`
(`LeakageReport.from_measure("v_information_capacity", …)`, `token_id_readout`) to put them on the
campaign metric. Bits stored verbatim; legibility handled by `report.format_bits`.

Model: `unsloth/gemma-2-2b`, d=2304. Provenance JSONs under `results/`; all numbers verbatim,
independently re-derived from the raw records (see foot of file). Single model, single seed unless
noted — multi-seed CIs and cross-model replication are named firm-ups, not blockers (reviewer R3:
"none change the verdict").

---

## R1 — the estimator-regime fix: class-PVI catastrophe → capacity-matched floor  (`results/capacity_screen.json`, `capacity_screen_dims.json`)

L12, every-n 2, 3 seeds. The question class-PVI failed: a token-id reader in the regime `d > n_val`
(d=2304 features, ~hundreds of validation tokens) interpolates the training labels, so its held-out
log-loss surplus on the **shuffle control** is unbounded below — the "−49 bit" catastrophe. Capacity
matching (reduce to `dim < n_val` first) bounds it.

| family | real cap-PVI | **shuffle floor** | cost vs class-PVI | noise decay |
|---|---|---|---|---|
| class-PVI (unfixed) | 4.96 bits | **−49.7 bits** ❌ | 1× | non-monotone (rises) ❌ |
| **pca_softmax** (pick) | 5.39 bits | **−1.9 bits** ✅ | 0.57× ✅ | monotone |
| randproj_softmax | 5.17 bits | −3.9 bits | **0.42×** ✅ | graceful monotone ✅ |
| gauss | −12.0 bits ❌ | −32.6 bits ❌ | 0.15× | non-monotone ❌ |
| knn | 1.35 bits | −1.1 bits ✅ | 0.13× ✅ | monotone, weak |

- **Catastrophe removed**: shuffle floor **−49.7 → −1.9 bits**, at **0.57× the cost** of class-PVI.
  The floor fix is **dim-anchored, NOT l2-anchored** (l2 reg only trades signal for floor health;
  dropping `dim` below `n_val` is what bounds it).
- **Floor repair holds at every depth** (from `localdp_depth_L0_5_12_20.json`, dim64): cap-PVI shuffle
  floor **−1.25 / −1.23 / −1.24 / −1.27 bits** at L0 / L5 / L12 / L20, against class-PVI **−44.9 /
  −50.2 / −51.4 / −48.1**. So the repair is not an L12-only artifact of the M1 screen — it is stable
  across the depth axis (answers the "PCA floor shown only at L12" gap).
- **The reader matters**: `gauss` stays catastrophic (miscalibrated class-conditional) — capacity
  matching is necessary but not sufficient; a calibrated reader (`pca_softmax`) is the pick.
- **The only change class-PVI → cap-PVI is the dimensionality reduction.** No embedding/attack
  information is added, so the repair comes from fixing the estimator regime, not from the probe
  "becoming the attack" — independence is preserved by construction.

## R2 — representation-space defenses: reader accuracy tracks the attack  (`results/nondp_intervention.json`, `nondp_intervention_l2-0.1.json`)

cached resid L5/12/20, pca_softmax dim 64. Two non-DP interventions, both **at the observed layer**
(no propagation). Spearman(readout, TTRSR) across the defense knob, per layer:

| defense | knob | readout | L5 | L12 | L20 |
|---|---|---|---|---|---|
| PCA-subspace ablation | k (dims removed) | **cap reader accuracy** | 0.80 | 0.90 | 0.90 |
| PCA-subspace ablation | k | CLUB I(rep;tok) | 0.90 | 0.90 | 0.90 |
| isotropic hidden-state noise | σ | **cap reader accuracy** | **1.00** | **1.00** | **1.00** |
| isotropic hidden-state noise | σ | CLUB I(rep;tok) | 1.00 | 1.00 | 1.00 |

Sample canonical rows (PCA-ablation, L20; `bits | readout`):

| k removed | attack TTRSR | cap-PVI bits | cap reader acc | CLUB |
|---|---|---|---|---|
| 0   | 0.465 | 3.47 bits | 0.682 | 3.69e3 bits |
| 8   | 0.055 | 3.13 bits | 0.652 | 3.00e3 bits |
| 32  | 0.015 | 2.76 bits | 0.650 | 2.14e3 bits |
| 128 | 0.042 | 1.78 bits | 0.495 | 748 bits |
| 384 | 0.013 | 0.85 bits | 0.322 | −77 bits |

- Under **representation-space** defenses the probe's **bounded reader accuracy tracks the inversion
  attack at ρ 0.80–1.00 at every depth** — no depth dependence. CLUB (an independent MI estimator)
  tracks identically. This is the positive, attack-predictive regime.

## R3 — input-DP depth sweep: the measure–attack decoupling is depth-resolved  (`results/localdp_depth_L0_5_12_20.json`)

Embedding-local DP (noise injected at the input embedding, then **propagated** through the network),
ε∈{∞,4096,1024,768,512,384,256}, n=7 per layer, pca_softmax dim 64. Spearman(measure, TTRSR) per
layer (independently re-derived; `cap_acc` matches the chronicle's +0.99 at L0 to within
tie-handling):

| layer | clean TTRSR | ρ(cap reader acc, TTRSR) | ρ(CLUB, TTRSR) | ρ(cap-PVI bits, TTRSR) | class-PVI shuffle floor |
|---|---|---|---|---|---|
| L0  | 0.809 | **+0.96 / +0.99** | +0.96 | +0.71 | −44.9 bits |
| L5  | 0.559 | +0.68 | +0.96 | +0.68 | −50.2 bits |
| L12 | 0.347 | +0.43 | +0.89 | +0.32 | −51.4 bits |
| L20 | 0.462 | **−0.21** | +0.29 | −0.21 | −48.1 bits |

Sample canonical rows (L0 input-DP; `bits | readout`):

| ε | attack TTRSR | cap-PVI bits | cap reader acc | CLUB |
|---|---|---|---|---|
| ∞    | 0.809 | 6.67 bits | 0.882 | 3.85e3 bits |
| 1024 | 0.661 | 6.80 bits | 0.878 | 3.64e3 bits |
| 512  | 0.428 | 6.47 bits | 0.846 | 3.34e3 bits |
| 256  | 0.140 | 5.46 bits | 0.684 | 2.81e3 bits |

- **The tracking degrades monotonically with propagation depth**: ρ(cap-acc, TTRSR)
  **+0.96 (L0) → +0.68 (L5) → +0.43 (L12) → −0.21 (L20)**. **CLUB shows a parallel attenuation**
  (+0.96 → +0.96 → +0.89 → +0.29): it weakens in the same direction but stays **positive at L20**
  — it does **not** reproduce cap-acc's sign reversal. The shared *attenuation* (an independent MI
  upper bound also loses its grip on TTRSR with depth) is what argues the divergence is a **property
  of the propagated signal, not a cap-PVI estimator artifact**; the *sign reversal* is specific to the
  bounded token-id readout and is the sharper, less-replicated half of the result.
- **Mechanism (a result, not a failure):** DP noise injected at the embedding is reshaped by depthwise
  nonlinear processing; it destroys **embedding geometry** (the attack's target) *before*
  **token-identity decodability** (the probe's target). At L0 the noise hits the representation
  directly → reconstruction and id-decodability fall in lockstep (ρ +0.96). By L20 they invert
  (ρ −0.21): in these runs an attack-independent measure and an embedding-reconstruction attack
  **diverge** (observed, single-seed), and the divergence localizes **what input-DP protects
  (embedding geometry) vs not (id-decodability) by depth** — a contribution in its own right, pending
  multi-seed firm-up.
- **Leave-one-out robustness (input-DP sweep, cap-acc vs TTRSR):** removing any single ε-point, the
  correlation stays positive at L12 (range +0.09..+0.60) and stays **negative at L20** (range
  −0.94..−0.09). The L20 sign survives single-point deletion; its magnitude is volatile. (Not a
  substitute for multi-seed CIs.)
- **Class-PVI is uninterpretable here** (shuffle floor −45 to −51 bits at every depth) — the
  catastrophe persists across the whole sweep; cap-PVI's floor stays ≈ −1.2.

## R4 — readout split: accuracy robust, PVI-in-bits only partially rescued  (R2/R3 + `results/localdp_m2_*.json`)

| readout | PCA-ablation | iso-noise | input-DP (L0/L5/L12/L20) |
|---|---|---|---|
| **cap reader accuracy** (bounded) | 0.80–0.90 | **1.00** | +0.96 / +0.68 / +0.43 / −0.21 |
| cap-PVI bits (selectivity, unbounded log-loss) | 0.90 | 0.40–0.70 (fragile) | +0.71 / +0.68 / +0.32 / −0.21 |

- The **bounded reader accuracy** is the robust predictor; **PVI-in-bits** is calibration-sensitive
  (the −48 floor and "rise under noise" are unbounded-log-loss artifacts) and tracks only with
  regularization. **Report accuracy as primary, bits as auxiliary; do NOT call the accuracy
  "V-information"** (reviewer R3's central framing fix).

---

## Synthesis (the surface's thesis)

The independent token-id V-family, broken by an estimator-regime pathology, is **repaired by
capacity matching** — and the repaired probe is attack-predictive *except* where it provably should
not be:

1. **Fix (R1):** class-PVI's `d > n_val` catastrophe (shuffle floor −49.7 bits) is removed by
   dimensionality reduction to `dim < n_val` (floor −1.9 bits, dim-anchored, 0.57× cost). The fix
   adds no attack information → independence preserved.
2. **Positive regime (R2):** under representation-space defenses (PCA-ablation, iso-noise) the
   probe's **bounded reader accuracy tracks the inversion attack at ρ 0.80–1.00 at every depth**;
   CLUB tracks identically.
3. **Characterized divergence (R3):** under **propagated input-DP**, ρ(measure, attack) degrades
   monotonically with depth (+0.96 L0 → −0.21 L20); **CLUB attenuates in parallel** (down to +0.29
   at L20 but still positive) ⇒ the depth-attenuation is a signal property, while the L20 sign
   reversal is specific to the bounded token-id readout. Input-DP destroys embedding geometry (attack
   target) before token-id decodability (measure target). Single-seed, n=7 per layer — scoped.
4. **Readout discipline (R4):** reader **accuracy** primary (robust), PVI-**bits** auxiliary
   (partially rescued, calibration-sensitive).

**Keeper claim 1 — `claim:capacity-matched-pvi` (scoped, gemma-2-2b):** capacity matching repairs the
independent token-id V-family — the `d > n_val` class-PVI shuffle-floor catastrophe (−44.9 to −51.4
bits across depth) becomes a bounded floor (−1.23 to −1.27 bits at every depth, dim-anchored, 0.57×
cost) — yielding an attack-independent token-id reader whose **bounded top-1 accuracy** tracks the
ridge inversion attack at ρ 0.80–1.00 under at-layer representation-space defenses (PCA-ablation,
iso-noise) at L5/L12/L20. Proof obligation: the estimator-regime argument (why a separable
`d ≥ n_val` reader drives held-out PVI-bits unbounded-below on the shuffle control, and why
`dim < n_val` bounds it). Honest scope: single model, single seed; the accuracy readout (not bits)
is the robust object.

**Keeper claim 2 — `claim:depth-decoupling-input-dp` (scoped, single-seed):** under propagated
input-DP the attack-independent probe's tracking of the embedding-reconstruction attack **attenuates
monotonically with depth** (ρ +0.96 L0 → −0.21 L20), turning negative at L20; CLUB, an independent MI
upper bound, exhibits a **parallel attenuation** (+0.96 → +0.29, staying positive) — so the
depth-attenuation is a property of the propagated signal, while the L20 sign reversal is specific to
the bounded token-id readout. This is the measurement-loop "non-correlation IS the finding" branch,
bounded and explained; the firm-up is multi-seed + denser ε + CIs.

**Named firm-ups (non-blocking, reviewer R3):** (i) calibration diagnostic (NLL/ECE) on the artifact
cells to *prove* the log-loss-artifact story; (ii) within-layer/macro-avg ρ as primary stat with
bootstrap CIs (n=7 thin); (iii) cross-model replication (gemma-2-2b only); (iv) dim16 sensitivity
(floor −1.10 vs dim64 −1.53).

---

### Provenance / integrity
- Re-derivation: per-layer Spearman ρ recomputed from raw `records`/`rows` in
  `results/localdp_depth_L0_5_12_20.json` and `results/nondp_intervention.json` (script in the
  experiment log). Stored `spearman_vs_ttrsr` fields cross-checked; cap-acc at L0 is +0.96 by the
  tie-broken estimator here vs +0.99 in the chronicle (scipy average-ties) — same gradient.
- Eval type: **genuine** — TTRSR is exact token-id match of the ridge attack against held-out tokens;
  cap reader accuracy is held-out top-1 over the real token-id classes. No synthetic ground truth.
- Probe ≠ attack: cap-PVI predicts token-id classes and never touches the embedding table; the attack
  reconstructs the embedding. ρ(cap, retrieval-PVI) = 0.66–0.76 (< 0.9) — adds beyond the attack-in-bits.

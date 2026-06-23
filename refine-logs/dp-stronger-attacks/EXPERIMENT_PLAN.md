---
type: plan
status: current
created: 2026-06-22
updated: 2026-06-22
tags: [experiment-plan, nns-pvi, table-softmax, voronoi-mi, bnn, dp-probe]
companion: [EXPERIMENT_PLAN, EXPERIMENT_RESULTS]
supersedes: []
---

# Experiment Plan — NNS-PVI: table-geometry V-information probe matched to BNN attacks

**Problem**: BNN (Bayes-NN / NNS-against-table) is the MAP-optimal attack under
embedding-space DP noise, achieves TTRSR=0.969 at r=3.63 (ε=64), and is completely
invisible to CapPVI (ρ=+0.45) and weakly visible to CLUB (ρ=+0.22). The existing
probe hypothesis classes — linear softmax in PCA-64 (CapPVI) and a variational
Gaussian bound on the continuous X→emb channel (CLUB) — both discard the full-d
Voronoi geometry of the embedding table that BNN exploits. No existing probe is
matched to the BNN decision rule.

**Method thesis**: A **table-softmax V-information** probe — `q(v|Y) ∝
exp(-‖Y − clip(e_v)‖²/τ)` over the known embedding pool — uses the identical decision
geometry as BNN. When temperature τ=2σ² (model-matched), this IS the Bayes posterior
under isotropic Gaussian DP noise, so V-info → I(V;Y) (an achievability lower bound
from the NNS hypothesis class). This probe should track BNN TTRSR with ρ≥0.90 across
the ε-sweep where CapPVI has ρ=0.45 and CLUB has ρ=0.22.

**Date**: 2026-06-22

---

## Definitions

| Term | Meaning |
|------|---------|
| **BNN / NNS** | Bayes-NN: `v* = argmin_v ‖Y − clip(e_v)‖²` over pool; MAP under isotropic Gaussian noise. |
| **CapPVI** | Capacity-matched PVI with `pca_softmax` hypothesis class — linear softmax in top-64 PCA components of Y. |
| **CLUB** | Contrastive log-ratio upper bound on MI for continuous X→emb channel. |
| **NNS-PVI** | Table-softmax V-information: `PVI = E[log q_NNS(v*｜Y) − log p(v*)]` where `q_NNS` is the table-distance softmax. The new probe this plan designs and tests. |
| **r** | Noise ratio = σ√d / C (dimensionless noise amplitude relative to clip radius). |
| **τ** | NNS-PVI temperature. τ=2σ² (model-matched) → q_NNS = Bayes posterior. |
| **Voronoi posterior** | `q(v｜Y)` under the table-softmax; concentrates on the Voronoi cell of the true embedding as τ→0 or noise→0. |
| **selectivity** | real − shuffle-control floor; the fraction of a probe's raw value that is not memorisation. |

---

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|-------|----------------|-----------------------------|--------|
| **C1 (primary)** — NNS-PVI(model-matched) tracks BNN TTRSR with ρ≥0.90 across the ε-sweep, where CapPVI has ρ=0.45 and CLUB has ρ=0.22. | This is the headline: a probe that is calibrated to the dominant L0 attack. Without it the MI framework has a known blind spot for Voronoi-geometry attacks. | Spearman ρ(NNS-PVI-sel, BNN-TTRSR) ≥ 0.90 over the 5-point ε-sweep (∞,1024,512,256,64), compared against the existing CapPVI (0.45) and CLUB (0.22) baselines. | B1, B2 |
| **C2 (supporting)** — NNS-PVI strictly dominates CapPVI as an MI lower bound at every ε; both are lower bounds, but NNS-PVI is tighter. | This is the theoretical justification: the NNS hypothesis class subsumes any linear projection class. | NNS-PVI ≥ CapPVI at every ε point (in bits); gap grows with noise as PCA-64 loses the near-orthogonal signal BNN uses. | B2, B3 |
| **C3 (design check)** — The gain is the geometry match, not the temperature fitting. | Anti-claim: "NNS-PVI just picked a better regulariser." If CapPVI with tuned τ matched NNS-PVI, the pool-distance feature would be irrelevant. | A cross-validated PCA-softmax temperature (same τ fitting procedure, different features) does NOT achieve the BNN-tracking gain. | B3 |
| **C4 (scope)** — NNS-PVI is specific to the embedding-space DP channel (L0); it degrades gracefully at L20 where BNN does not apply. | Characterises probe scope — the plan does not claim NNS-PVI is the right probe for all layers. | NNS-PVI@L20 shows lower ρ vs ridge@L20 than CLUB@L20 does; the L0 advantage is layer-specific. | B2 |

**Anti-claims to rule out**:
- **A1 "Just temperature."** → B3 compares same-τ-procedure on PCA vs table features.
- **A2 "Probe == attack (circular)."** → NNS-PVI is a selectivity scalar in bits; BNN is a TTRSR in recovery rate. Different spaces, scored independently.
- **A3 "CapPVI non-monotone invalidates the baseline."** → Already documented as artefact (F4 in unified-dp-sweep). Use the same 5-point sweep; report the non-monotone point explicitly.

---

## Paper Storyline

- **Main paper proves**: C1 (B2) — NNS-PVI tracks BNN; C2 (B3) — it's an MI lower bound that CapPVI is not tight on.
- **Appendix**: C3 (B3) anti-claim ablation; C4 scope check.
- **Cut**: full-vocab NNS-PVI (V=256K, O(n·V·d) cost); NNS-PVI@L>0 with unembedding-matrix lookup (interesting but not the core story).

---

## Analytical Background (to brief reviewer)

BNN succeeds because in d=2304, with a pool of V=2048 tokens:
```
‖Δ‖ ≈ √(2·C²) ≈ 5.87   (near-orthogonal embeddings)
Decision SNR = ‖Δ‖/(2σ) ≈ 9.35 at r=3.63
P(BNN correct per pair) ≈ Φ(9.35) ≈ 1−10⁻²⁰
```

CapPVI uses PCA-64. At r=3.63, per-dimension noise/signal ≈ 13:1; every PCA direction
is noise-dominated. The 64-dim projection discards exactly the near-orthogonal
dimensions that make BNN work.

NNS-PVI computes the Voronoi posterior:
```
q(v|Y) = softmax(-‖Y − clip(e_v)‖²/τ)  for v in pool
```
This softmax concentrates on the correct Voronoi cell with logit-gap ≈ ‖Δ‖²/τ.
When τ=2σ²:
```
q(v*|Y) = P(v*|Y, Gaussian noise) = exact Bayes posterior
NNS-PVI → I(V;Y)  [achievability lower bound from NNS class]
```
At r=3.63, σ²≈0.099, τ=0.198, logit-gap = 5.87²/0.198 ≈ 174 → q concentrates
entirely on v* → NNS-PVI → H(V). At r=0 (ε=∞), Y=clip(e_v) → same. Both match
BNN's near-1.0 TTRSR. In between: NNS-PVI is monotone in decision SNR, same as BNN.

CapPVI at r=3.63: all 64 PCA logits are noise; q_PCA → uniform → PVI → 0. That is
the F1 finding from the unified sweep. NNS-PVI does not have this collapse.

---

## Experiment Blocks

### B1 — Implement NNS-PVI — MUST-RUN FIRST (no GPU needed)

**Claim tested**: design correctness (precondition for B2–B4)

**Why**: The probe does not exist in `src/talens/measures/`. Must be built and
unit-tested before running the sweep.

**Implementation location**: `src/talens/measures/vinfo_capacity.py` — add
`nns_v_information()` alongside the existing `v_information_capacity()`. Same interface:
- inputs: `X (n,d)`, `y (n,) token ids`, `table (V,d)`, optional `pool (k,)` int indices, `tau`, `control`, `seed`
- outputs: dict with `"reader_top1_acc"` (fraction where argmax q_NNS = true v), `"nns_pvi_bits"`, `"nns_pvi_nats"`

**Core formula**:
```python
# pool_emb: (k, d) clip(table[pool])
# dists_sq: (n, k) = ||Y_i - pool_emb_j||^2 = ||Y||^2 - 2 Y@E.T + ||E||^2
# logits = -dists_sq / tau
# q = softmax(logits, axis=1)
# pvi_bits = mean( logits[i, true_pos[i]] - logsumexp(logits[i]) - log(1/k) ) / ln2
```

**Temperature τ handling**:
- `tau=None` (default): cross-validate τ ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0, ∞}
  on the train split; pick τ that maximises mean log q(v*|Y) on the val split.
- `tau="model"`: requires passing `sigma` kwarg; sets τ=2*sigma²
- `tau=float`: use directly

**Selectivity / shuffle control**: same as CapPVI — `control="shuffle"` permutes y
labels, nns_pvi_bits(shuffle) ≈ 0 for a healthy estimator.

**Clip handling**: probe must use `clip(table[pool], C)` matching BNN, not raw table.
`C` is passed as `clip_norm` kwarg (default=None → no clip).

**Unit tests** (CPU, < 5 min):
- T1: at σ=0, Y=clip(e_v) for each v → NNS-PVI ≥ 0.95·H(V) (nearly maximal)
- T2: at σ→∞, NNS-PVI → 0 and shuffle floor → 0
- T3: argmax q(v|Y) == v* matches BNN prediction for the same (Y, pool)
- T4: model-matched τ=2σ² gives NNS-PVI ≥ NNS-PVI(τ=1.0) at moderate noise (r≈0.5)
- T5: `control="shuffle"` reduces NNS-PVI to ≈ 0 on clean data

**Success criterion**: all T1–T5 pass on CPU with synthetic d=32 embeddings.
**Cost**: < 30 min coding + < 5 min test.
**Priority**: MUST-RUN.

---

### B2 — Full ε-sweep with NNS-PVI column — MUST-RUN

**Claim tested**: C1 (ρ≥0.90), C2 (NNS-PVI ≥ CapPVI in bits), C4 (L0 advantage)

**Why**: This is the decisive block. Adds NNS-PVI to `_run_probes()` in
`scripts/spikes/unified_dp_sweep.py` and reruns the full 5-point ε-sweep.

**Dataset / split**: gemma-2-2b, corpora/release-gate-512.txt, 256 prompts,
vocab-disjoint split, pool=2048, seed=20260622. Same as the prior sweep.

**Compared systems (probes)**:
- CapPVI (pca_softmax, dim=64) — existing baseline, ρ_baseline=0.45
- CLUB — existing baseline, ρ_baseline=0.22
- NNS-PVI (model-matched, τ=2σ², same pool as BNN, clip_norm=C_raw)
- NNS-PVI (cross-validated τ) — variant in same run

**Metrics**:
- Primary: Spearman ρ(NNS-PVI-sel, BNN-TTRSR) over the 5-point ε sweep (report both τ variants)
- Secondary: NNS-PVI-bits vs CapPVI-bits at each ε (is NNS-PVI ≥ CapPVI?)
- Tertiary: ρ(NNS-PVI-sel, ridge-TTRSR-L0) — does it also track ridge?

**Setup**:
- Add `nns_pvi_bits` to `_run_probes()` return dict alongside `club`, `pvi`, `mdl`
- Pass `C_raw`, `sigma`, `pool` to `nns_v_information()`
- For τ cross-validation: use the probe's train split (no extra prompts needed)
- Report selectivity = real − shuffle as for all other probes
- Estimated extra wall-clock: +10–15 min per ε (pool=2048, d=2304, n≈3000)
- Total run: ~3 hours on ROCm container

**Success criterion** (C1 confirmed): ρ(NNS-PVI-sel, BNN-TTRSR) ≥ 0.90
**Failure interpretation**: If ρ < 0.90 despite model-matched τ, check whether:
  (a) the BNN TTRSR values are themselves non-monotone (they weren't in prior sweep)
  (b) τ calibration is off (use σ directly from ε,C values — not from cross-val)
  (c) pool clip norm is mismatched with BNN's clip norm

**Success criterion** (C2): NNS-PVI-bits ≥ CapPVI-bits at ε∈{256, 64} (where PCA fails)
**Failure interpretation** (C2): If NNS-PVI < CapPVI, implies the pool-distance features
carry less MI signal than PCA directions — unexpected given the BNN geometry argument.
Would require revisiting the τ-calibration or pool definition.

**Table / figure target**: main results table (columns: ε, r, ridge@L0, BNN@L0,
  | CLUB-sel, CapPVI-sel, NNS-PVI-sel (model-matched), NNS-PVI-sel (cv-τ)).
  ρ matrix below the table.

**Priority**: MUST-RUN.

---

### B3 — Temperature ablation (C3 geometry-vs-τ isolation) — MUST-RUN

**Claim tested**: C3 — the gain is geometry (table distances), not temperature fitting.

**Why**: The reviewer will ask: "did you just tune τ on PCA-softmax?" If yes, the gain
might be trivially reproducible by tuning CapPVI's implicit temperature. This block
rules out that explanation.

**Design**: Run two additional probe variants:
- `PCA-τ`: pca_softmax with cross-validated temperature τ (same τ-sweep as NNS-PVI cv)
- `RandProj-τ`: randproj_softmax (already in `vinfo_capacity.py`) with cv-τ
Compare both against NNS-PVI-cv-τ.

**Expected outcome**: PCA-τ and RandProj-τ do not gain ρ-vs-BNN even with temperature
tuning, because the feature space (projected Y) does not contain the Voronoi structure.
NNS-PVI gains specifically from the `‖Y − clip(e_v)‖²` feature.

**Cost**: No new GPU run needed — add these columns to the B2 run script. Same prompts/
split, just additional probe variants inside `_run_probes()`.

**Success criterion**: ρ(PCA-τ-sel, BNN) < ρ(NNS-PVI-cv-τ-sel, BNN) − 0.2.
**Failure interpretation**: If PCA-τ matches NNS-PVI-cv-τ, temperature is the key
variable and the table-distance feature is not necessary. Claim C3 must be weakened.

**Table / figure target**: Appendix ablation table.
**Priority**: MUST-RUN (required to defend C3).

---

### B4 — Pool size ablation — NICE-TO-HAVE

**Claim tested**: Scope / practical guidance.

**Why**: BNN uses pool=2048. NNS-PVI with pool=2048 gives a pool-conditioned MI.
A reviewer may ask whether NNS-PVI(full-vocab=256K) tracks more or less well.
This is also a cost question: pool=256K is ~128× more expensive.

**Design**: Add a `pool_size` sweep — pool∈{256, 2048, 8192, full-vocab} — at the
single most informative ε (ε=256, r=0.91, the cliff point).
Full-vocab run: V≈256K, n≈3000, d=2304 → ~1.8B floats per batch; needs chunking.

**Expected outcome**: NNS-PVI(pool=2048) ≈ NNS-PVI(pool=8192) within noise (the
top-k nearest neighbors dominate the softmax). Pool=256 loses because the pool
is too small to represent the decision space accurately.

**Cost**: ~30 min extra GPU at ε=256 only.
**Priority**: NICE-TO-HAVE.

---

### B5 — NNS-PVI at L20 (scope check, C4) — NICE-TO-HAVE

**Claim tested**: C4 — NNS-PVI is L0-specific.

**Why**: At L20, the activations are NOT in embedding space. BNN does not apply.
NNS-PVI at L20 uses the embedding table distances against L20 activation vectors —
a mismatch. This should perform poorly relative to CLUB@L20.

**Design**: Add `nns_pvi_l20` column to the B2 run (negligible extra cost).
Use `clip_norm=C_runtime` for L20 (same as the hook).

**Expected outcome**: ρ(NNS-PVI@L20, ridge@L20) < ρ(CLUB@L20, ridge@L20) = 0.90.
NNS-PVI gives a valid MI lower bound but is a weak probe for L20 recovery.

**Priority**: NICE-TO-HAVE (5 min added to B2).

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Implement + unit test NNS-PVI | B1 coding + `pytest tests/test_vinfo_capacity.py` | All T1–T5 pass | <1h CPU | Low — pure math |
| M1 | Full sweep with NNS-PVI col | B2 (+ B3 freeride + B5 freeride) in ROCm | ρ(NNS-PVI, BNN)≥0.80 to continue | ~3–3.5h GPU | Medium — τ calibration |
| M2 | Verify claims, write wiki entry | B2 results analysis | ρ≥0.90 → C1 confirmed; 0.7–0.90 → partial; <0.7 → diagnose | 30 min | Low |
| M3 (optional) | Pool ablation | B4 at ε=256 | NNS-PVI(2048) ≈ NNS-PVI(8192) → stop; else characterise | ~30 min GPU | Low |

**Must-run**: M0 → M1 → M2.
**Nice-to-have**: M3.

### Execution order within M1 (B2)

In `unified_dp_sweep.py`, extend `_run_probes()`:

```python
from talens.measures.vinfo_capacity import nns_v_information

def _run_probes(X, y, table, seed, max_rows, cap_dim, pool=None, C=None, sigma=0.0):
    # existing CLUB / CapPVI / MDL ...
    tau_model = 2.0 * sigma**2 if sigma > 0 else None
    rn = _safe(nns_v_information(X, y, table, pool=pool, tau=tau_model,
                                  clip_norm=C).get("nns_pvi_bits"))
    sn = _safe(nns_v_information(X, y, table, pool=pool, tau=tau_model,
                                  clip_norm=C, control="shuffle").get("nns_pvi_bits"))
    # cv-tau variant
    rn_cv = _safe(nns_v_information(X, y, table, pool=pool, tau=None,
                                    clip_norm=C).get("nns_pvi_bits"))
    sn_cv = _safe(nns_v_information(X, y, table, pool=pool, tau=None,
                                    clip_norm=C, control="shuffle").get("nns_pvi_bits"))
    real = {..., "nns_pvi": rn, "nns_pvi_cv": rn_cv}
    sel  = {..., "nns_pvi": rn - sn, "nns_pvi_cv": rn_cv - sn_cv}
    return real, sel
```

---

## Compute and Data Budget

| Item | Estimate |
|------|----------|
| M0 unit tests | < 5 min CPU (synthetic d=32) |
| M1 B2 full sweep | ~3.5h GPU (same as prior sweep + ~15% overhead) |
| M3 pool ablation (optional) | ~30 min GPU at ε=256 |
| **Total GPU** | **~4h** |
| Data preparation | None — reuse release-gate-512.txt, pool=2048 from prior sweep |
| Human evaluation | None |
| **Biggest bottleneck** | τ calibration correctness (M0 unit tests must validate T4) |

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| τ calibration wrong → NNS-PVI non-monotone | Medium | T4 unit test validates model-matched τ on synthetic; inspect logit-gap at each ε before computing selectivity |
| Pool clip mismatch (C_raw vs C_runtime) | Low | Pass explicit `clip_norm=C_raw` (L0) or `clip_norm=C_runtime` (L20); log both |
| NNS-PVI@ε=∞ not near H(V) | Low | At σ=0 the table-softmax collapses to argmin (one-hot); use large τ or skip model-matched at ε=∞ and report only cv-τ variant |
| ρ plateau: BNN is near-1.0 at ε=∞,1024,512 → ρ computed over 3 distinct values only | Medium | Report Spearman ρ over all 5 points; note that the 3-way tie compresses ρ and is expected. Also report point-by-point NNS-PVI vs BNN scatter. |
| Shuffle floor non-zero for NNS-PVI | Low | The table-softmax with shuffled labels is NOT zero if Y clusters happen to align with wrong token embeddings. Verify T5 empirically and add a per-position shuffle (not global) if needed. |

---

## Final Checklist

- [ ] B1: `nns_v_information()` implemented in `src/talens/measures/vinfo_capacity.py`
- [ ] B1: T1–T5 unit tests pass in `tests/test_vinfo_capacity.py`
- [ ] B2: `_run_probes()` in `unified_dp_sweep.py` extended with NNS-PVI columns
- [ ] B2: Full 5-point ε-sweep run; ρ(NNS-PVI-sel, BNN) computed and logged
- [ ] B3: PCA-τ and RandProj-τ variants added and ρ compared (geometry vs temperature check)
- [ ] B5: NNS-PVI@L20 column added (freeride in B2 run)
- [ ] C1 confirmed: ρ(NNS-PVI-model-matched, BNN) ≥ 0.90
- [ ] C2 confirmed: NNS-PVI-bits ≥ CapPVI-bits at ε∈{256, 64}
- [ ] C3 confirmed: PCA-τ does not match NNS-PVI-cv-τ
- [ ] Wiki: new experiment node added, `claim:nns-pvi-geometry-match` created
- [ ] Must-run vs nice-to-have separated

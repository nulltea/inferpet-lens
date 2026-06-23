---
type: plan
status: current
created: 2026-06-22
updated: 2026-06-22
tags: [experiment-plan, spectral-mi, matched-probe, embedding-inversion, vec2text, dp, localization, fano, validation]
companion: [EXPERIMENT_TRACKER]
supersedes: []
---

# Experiment Plan — Empirically validate the spectral channel-MI probe

**Problem**: `claim:spectral-channel-mi-embedding-inversion` is PASS-verified but is a set of
CONVERSE bounds. The proof flags three things it does NOT establish, only bounds: (i) that the
geometry-only ceilings actually hold against a real SOTA attack and how tight they are; (ii) that
the probe *predicts* (not just bounds) recovery, and does so better than CLUB/capPVI; (iii) the
*achievability* of T4 localization (that the recoverable leakage really lives in the top
eigendirections an attack reads). The probe also is not yet implemented. This plan validates the
verified theory — it does not re-prove it.

**Method Thesis**: The spectral channel-MI `I_G(σ)=½Σ log2(1+λi/σ²)`, computed from the
clean-embedding covariance spectrum + σ **alone**, is the matched probe for embedding inversion
under DP: it upper-bounds SOTA Vec2Text recovery (geometry-only ceiling), rank-predicts it across
the privacy budget ≥ CLUB and ≫ capPVI at a fraction of the cost, and its per-mode profile
correctly localizes *where* the leakage lives (eigen-ablation).

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| **C1** (PRIMARY — matched probe) | Is the cheap geometry-only probe a faithful predictor of the strong attack, beating the incumbents? | Over the ε sweep, Spearman(I_G, Vec2Text recovery) ≥ Spearman(CLUB, ·) and ≫ Spearman(capPVI, ·); I_G is closed-form (one eigh) vs CLUB's trained net / capPVI's trained reader | B1, B2 |
| **C2** (ceiling holds + tightness) | A converse is only useful if it actually bounds the real attack | Vec2Text exact-match ≤ Fano ceiling `(min{H(e0),I_G}+1)/H(X)` and per-token error ≥ RD floor at every ε (no violation); report the gap (attack sub-optimality + bound looseness) | B3 |
| **C3** (localization / "where" — supporting) | The headline novelty over a scalar MI: it says *which directions* leak | Eigen-ablation of Y: Vec2Text recovery vs #kept top principal modes k saturates by k≈d_eff(σ); dropping the top-d_eff modes collapses recovery; the {t_i}/tail(k) profile predicts the curve | B4 |

**Anti-claims to rule out**:
- *"I_G tracks only because everything decays monotonically with ε (common-cause)."* Mitigation:
  C3's eigen-ablation varies recovery at **fixed ε** (the monotone-ε knob is held constant), and
  I_G's per-mode structure is something CLUB's scalar cannot produce — so the win is not just
  shared monotonicity. (Same confound flagged in the attack-side B3/B4.)
- *"I_G is a disguised attack."* Refuted by construction: it is `cov-eigh(Σ_clean) + σ`, never runs
  an inverter, never sees the attack's outputs — the geometry-only invariant of
  [[bnn-error-bounds-bhattacharyya-fano]].

## Paper Storyline
- **Main paper must prove**: C1 (the matched probe predicts SOTA recovery, beats CLUB/capPVI,
  closed-form) and C3 (it localizes the leakage — eigen-ablation). Together: *a free, closed-form
  spectral quantity forecasts and localizes what the strongest faithful inversion recovers.*
- **Appendix**: C2 (ceiling tightness; depends on H(X)/H(e0) estimates), d_eff-vs-knee, H(e0) collisions.
- **Cut**: re-deriving any bound (done in the proof); other encoders/datasets (separate scope).

## Experiment Blocks

### Block 1 — Implement I_G (geometry-only) + sanity — MUST-RUN (M0/M1)
- **Claim**: C1 (enabler). **Why**: the probe must exist as a geometry-only measure and be
  correct before any comparison.
- **Implementation**: `src/talens/measures/spectral_channel_mi.py` — input: clean-embedding matrix
  `E0 (n×d)` (or precomputed `Σ`), `σ` (scalar or list), optional `H_X`, `H_e0`. Output dict:
  `i_g_bits`, per-mode `t_i`, `d_eff`, `tail(k)` profile, eigenvalues `λ`, and the ceilings
  `fano_exact_ceiling=(min{H_e0,I_G}+1)/H_X`, `rd_pertoken_floor`. Eigenvalues via covariance eigh
  (torch `linalg.eigh` on GPU when available, numpy fallback) — mirror `channel_error_bounds.py`
  (numpy in / dict out, dependency-light).
- **Metrics / tests** (model-free, host `.venv`): (a) Gaussian `E0` ⇒ `i_g_bits` matches
  `½ log2 det(I+Σ/σ²)` to float tol; (b) `σ→∞` ⇒ `I_G→0`; `σ→0` ⇒ `I_G→∞` (and `min{H_e0,I_G}=H_e0`);
  (c) monotone decreasing in σ; (d) per-mode `Σ t_i == i_g_bits`; (e) `d_eff` counts `λi≥σ²`.
  Lock in `tests/test_spectral_channel_mi.py`.
- **Success**: unit tests green; on the GTR sweep `I_G` is finite, monotone↓ in ε, same rank order
  as CLUB (both estimate `I(X;Y)`). **Failure**: eigh instability / scale mismatch ⇒ fix before B2.
- **Table/figure**: none (enabler); report I_G vs CLUB sanity row.

### Block 2 — Matched-probe comparison: I_G vs CLUB vs capPVI (C1) — MUST-RUN (M2)
- **Claim**: C1 (headline). **Why**: the central question — does the free probe predict SOTA
  recovery as well as / better than the incumbents?
- **Dataset**: the pooled-GTR DP sweep, ε∈{∞,1024,512,256,128}, N≈128, 32-tok (reuse
  `scripts/eval/vec2text_attack.py` + the B8 pipeline).
- **Compared systems (probes)**: **I_G** (ours, geometry-only) vs **CLUB** `I(e';e0)` (trained
  variational net) vs **capPVI** (trained PCA-softmax cluster reader). Recovery = Vec2Text
  token-F1 / exact / cos (the thing being predicted).
- **Metrics**: Spearman(probe, recovery) over ε for each probe and each recovery metric; **wall-clock
  cost** per probe (eigh vs net-train vs reader-train); decisive: token-F1 & cos Spearman.
- **Success**: Spearman(I_G, ·) ≥ Spearman(CLUB, ·) (expect both ≈+1.0) and ≫ Spearman(capPVI, ·)
  (expect capPVI flat/low), with I_G cheapest. **Failure**: if I_G < CLUB the closed-form bound
  loses to the variational estimate — report honestly (still cheaper/decomposable).
- **Table/figure**: Table "probe ↔ Vec2Text-recovery Spearman + cost, over ε".

### Block 3 — Ceiling holds + tightness (C2) — MUST-RUN (M2)
- **Claim**: C2. **Why**: validate the converse is real (no attack exceeds it) and measure slack.
- **Setup**: per ε compute the geometry-only **Fano exact-match ceiling** `(min{H_e0,I_G}+1)/H_X`
  and the **RD per-token-error floor**; overlay actual Vec2Text exact-match and per-token error.
  `H_X` ≈ `log2(#distinct texts)` (uniform proxy, an upper bound on the true H(X)); `H_e0` via
  empirical embedding-collision/quantization estimate. **Both flagged as estimates** — report the
  ceiling as a band over plausible H estimates.
- **Metrics**: violation count (must be 0); ceiling−actual gap vs ε.
- **Success**: no ε violates the ceiling/floor; gap shrinks toward high noise (where bound tightens).
  **Failure**: a violation ⇒ H-estimate too small (proxy issue) OR a bug — diagnose, don't dismiss.
- **Table/figure**: Figure "Vec2Text recovery under the geometry-only Fano/RD ceiling vs ε".

### Block 4 — Eigen-ablation: where the leakage lives (C3) — MUST-RUN (M3)
- **Claim**: C3 (T4 achievability). **Why**: the unique localization claim — turn the per-mode
  profile into an empirical "which directions matter".
- **Design**: at ε∈{∞,512,128}, form the rank-k projected release `Y_k = U_k U_kᵀ Y` (U_k = top-k
  eigenvectors of clean Σ; ambient 768-d preserved) and the drop-top-k complement `Y_{-k}`; sweep
  k∈{8,32,64,128,256,768}. Run Vec2Text on each; recovery vs k.
- **Metrics**: recovery(token-F1) vs k for keep-top-k and drop-top-k; compare the knee to
  `d_eff(σ)`; compare recovery-retained to `1 − tail(k)/I_G` shape.
- **Success**: keep-top-k recovery **saturates by k≈d_eff(σ)** (low modes droppable); drop-top-k
  **collapses** recovery; the {t_i}/tail(k) profile predicts the curve ordering. **Failure**: if
  recovery needs modes far below d_eff, localization is looser than claimed — report the gap (still
  a converse). **OOD caveat**: projected embeddings are off the GTR manifold; note that as a
  confound (compare against a CLUB-of-Y_k sanity).
- **Table/figure**: Figure "Vec2Text recovery vs #principal modes kept/dropped, with d_eff markers".

### Block 5 — d_eff vs privacy knee + H(e0) collisions — NICE-TO-HAVE (M4)
- d_eff(σ) overlaid on the recovery-vs-ε knee (does d_eff predict where recovery falls?); empirical
  embedding-collision count → H(e0) estimate sharpening C2's band. Table/figure: appendix.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| **M0** | implement `spectral_channel_mi.py` + host unit tests | B1 tests (CPU, model-free) | tests green; Gaussian-exact, monotone | ~mins | eigh scale/units |
| **M1** | probes on the GTR sweep | compute I_G/CLUB/capPVI + recovery (reuse B8) | I_G sane, monotone, ranks like CLUB | ~10 min | recompute sweep if B8 JSON stale |
| **M2** | C1 comparison + C2 ceilings | Spearman table + ceiling figure | I_G ≥ CLUB, ≫ capPVI; no ceiling violation | ~10 min | H_X/H_e0 estimates (C2) |
| **M3** | C3 eigen-ablation | 3 ε × ~6 k × {keep,drop} Vec2Text | keep saturates by d_eff; drop collapses | ~15 min | OOD projection; inversion cost |
| **M4** | nice-to-haves | d_eff-knee, H(e0) collisions | — | ~10 min | — |

Must-run: **M0 → M1 → M2 → M3**. Nice-to-have: M4.

## Compute and Data Budget
- **Total**: ~1 GPU-hour. eigh on 768-d is trivial; the cost is Vec2Text inversions (reused handler,
  ~0.66 s/text at beam1). B4 is the driver: ~3 ε × 6 k × 2 (keep/drop) × N inversions — cap N≈96
  and the k-grid at 6 to stay ≤20 min/run.
- **Data**: the existing 32-tok GTR text set; no new data, no training (pretrained corrector).
- **Biggest bottleneck**: B4 inversion count, and the H(X)/H(e0) estimates for C2.

## Risks and Mitigations
- **H(X)/H(e0) estimates (C2).** Uniform proxy `H_X≈log2(#distinct texts)` over-estimates true H(X);
  `H_e0` via collision counting is noisy at N≈128. Mitigation: report the Fano/RD ceiling as a
  **band** over a plausible H range; treat C2 as "no violation + qualitative tightness," not a sharp
  number. Flagged as the main caveat (also an Open Risk in the claim).
- **Eigen-ablation OOD (B4).** `U_k U_kᵀ Y` is off the GTR manifold; Vec2Text may degrade for reasons
  unrelated to information. Mitigation: report CLUB(Y_k; e0) alongside (an info measure on the same
  projected input) — if recovery and CLUB(Y_k) drop together, it's information not OOD.
- **Common-cause monotone confound (C1).** Mitigated by the fixed-ε ablation axis (C3) and the
  per-mode structure (CLUB can't produce it).
- **Stale B8 sweep JSON.** If reusing cached recovery, re-run the clean handler once to materialize
  `results/vec2text_dp_eval.json` at the documented path.

## Final Checklist
- [x] Main result covered (C1 probe-comparison table; C3 localization figure)
- [x] Novelty isolated (I_G vs CLUB vs capPVI head-to-head; localization is unique to the spectral probe)
- [x] Simplicity defended (closed-form eigh vs trained net/reader; cost reported)
- [x] Frontier component justified (the SOTA Vec2Text attack is the *target* the probe must predict — not decoration)
- [x] Nice-to-have (M4) separated from must-run (M0–M3)
- [x] Geometry-only / not-a-disguised-attack invariant preserved (probe never runs an inverter)

# Experiment Plan — KV/QKV accumulation / BSS attacks (Task B-1)

**Problem**: Can a blind-source-separation (BSS) / accumulation adversary recover the
secret behind the KV/QKV surface of Qwen3-4B from captured activations, and does an
*attack-independent* matched probe predict that recovery as the observation count T grows?
**Method thesis**: On the *plaintext* KV/QKV surface (Identity transform, WEIGHTS-PUB),
porting the aloepri BSS family (gram_error / jade / jd) lets us (a) quantify single-observation
ICA recovery per layer×kind, (b) test whether recovery accumulates with stack size T, and
(c) test whether a geometry-only probe tracks recovery across the T sweep — establishing the
plaintext baseline against which the later KV-CLOAK defense (Task B-2) is read.
**Date**: 2026-06-24

## Definitions (IT vocabulary)

- **BSS / ICA** — blind source separation: recover latent sources `S` from a linear mixture
  `U = A·S` knowing only `U`, up to a per-source permutation + sign ambiguity.
- **JADE** — Joint Approximate Diagonalization of Eigenmatrices (Cardoso & Souloumiac 1993):
  ICA via joint-diagonalizing 4th-order cumulant matrices of the whitened observation. Single
  observation matrix.
- **JD (accumulation)** — joint-diagonalization across a *stack* of T observation covariances
  `{C_t = U_t·U_tᵀ/d}` (different prompts, same layer/kind). Recovers one shared demixing.
  The accumulation axis is T ∈ {1,2,4,8,16}.
- **gram_error** — row-Gram fingerprint: cos-normalised Frobenius distance between
  `G_U = U·Uᵀ` and `G_H = H·Hᵀ`; range [0,√2]; lower = more fingerprintable. On plaintext
  `U = H` ⇒ ≈ 0 (the trivial baseline that the row-Gram *is* the activation fingerprint).
- **Surface kinds**: `kq` = pre-softmax Q·Kᵀ per-head scores `(n_heads, n_q, n_kv)`;
  `kqv_out` = per-head attention output pre-W_o `(n_q, heads·head_dim)`; `resid_post` reference.
- **Recovery** (graded): p95 of the Hungarian-aligned |cosine| between recovered sources and
  true rows. The Hungarian alignment is maximum-benefit-of-the-doubt to the attacker.
- **Matched probe** (attack-independent, bits canonical): a geometry-only statistic of the
  channel the BSS attack exploits, computable WITHOUT running joint-diagonalization.

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Linked blocks |
|-------|----------------|-----------------------------|---------------|
| **C1 (primary): the accumulation question.** On the plaintext KV/QKV surface the jd p95-cosine curve is **flat in T** (no accumulation) — distinct fresh-per-prompt activations share no common demixing — OR it **climbs** because activations share a common anisotropic subspace. The plan *measures which*, with a chance-floor control. | This is the literal Task objective and the baseline that makes the KV-CLOAK mask-reuse test (B-2) interpretable: any climb-with-T under a defense is then attributable to mask correlation, not to BSS per se. | jd p95-cosine(T) for T∈{1,2,4,8,16}, per layer×kind, vs a random-rows Hungarian floor; slope of cosine-vs-log₂T with CI excluding/including 0. | B1, B2, B3 |
| **C2 (supporting): probe predicts attack.** An attack-independent geometry probe — negentropy/excess-kurtosis (jade separability) and shared-spectral-capacity of the averaged covariance (jd accumulation) — correlates with BSS recovery across the T sweep and across layer×kind. | The repo thesis: does an attack-independent IT measure predict a separately-run attack? Clean test because the probe is pure geometry, never the demixing. | Spearman/Pearson of probe-bits vs recovery across all (layer×kind×T) cells; ≥0.7 |ρ| ⇒ correlate; else identify weak-attack vs non-matched-probe. | B3 |

**Anti-claim to rule out**: "recovery" is just Hungarian-alignment cheating (optimally matching
recovered sources to ground truth inflates cosine even for noise). **Control**: random-rows /
Gaussian-rows baseline run through the same Hungarian-aligned-cosine pipeline → the
alignment-chance floor. Recovery claims must clear this floor.
**Anti-claim 2**: a "correlation" that is the attack in disguise. Mitigation: the probe is
computed from covariance eigenspectra / whitened-row kurtosis only — never from the JADE/JD
demixing output. Stated explicitly in B3.

## Not-applicable family members (documented, no phase)

- **sda / tfma** (aloepri): operate on a recovered **token-id sequence**, not on activations —
  they do not cross the activation boundary under WEIGHTS-PUB. `not_applicable`.
- **ia weight-axis**: needs an obfuscated **weight** pair `(W_plain, W_obf)`; WEIGHTS-PUB gives
  the adversary the *true* weights, so there is no weight secret to attack. `not_applicable`.
These are recorded in the experiment log, not forced into a phase.

## Paper storyline

- **Main paper must prove**: C1 (the T-accumulation behaviour on plaintext, with floor control)
  and C2 (probe-tracks-recovery or the bounded reason it doesn't).
- **Appendix can support**: gram_error fingerprint baseline (≈0 on plaintext confirms protocol);
  per-head vs pooled kq breakdown; layer profile across all 36 layers (dev-24 is cheap).
- **Intentionally cut**: learned-inverter comparison (covered by hidden_state attack elsewhere);
  any defense sweep (KV-CLOAK = Task B-2).

## Experiment Blocks

### Block 0: Port + unit-sanity (MUST-RUN, CPU)
- **Claim tested**: implementation faithfulness (pre-req for C1/C2).
- **Why**: the 3 attacks + matched probe must be faithful array-math ports living as
  `src/talens/attacks/{bss_gram,bss_jade,bss_jd}.py` (or one `bss.py`) and
  `src/talens/measures/bss_separability.py`, consuming a `CaptureSet` through the `Transform`
  seam — no vendored `sys.path`, no snapshot loader.
- **Data**: synthetic — (i) known orthogonal mixture `U=A·S` with non-Gaussian S ⇒ JADE
  recovers S up to perm/sign (cosine≈1); (ii) independent random stacks ⇒ jd flat in T;
  (iii) `U=H` ⇒ gram_error cos_norm≈0.
- **Metrics**: recovery cosine, flatness, fingerprint distance vs known answers.
- **Success**: synthetic recoveries match analytic expectation; numbers reproduce the aloepri
  driver on a shared toy input within 1e-3.
- **Failure interpretation**: port bug — fix before any real data.
- **Priority**: MUST-RUN.

### Block 1: dev-24 CPU pilot — the cheap full-layer probe (MUST-RUN, CPU, NO GPU)
- **Claim tested**: C1, C2 at pilot scale + scope decision for B2.
- **Why**: the dev-24 capture (24 prompts, all 36 layers, kq/kqv_out/resid_post) is already
  cached — full attack+probe sweep with zero GPU. Establishes signal shape before paying GPU.
- **Data**: cached `capture-3e3a86a58abf0727` (results/capture_cache). Layers: profile all 36
  for jade/gram_error/probe (cheap); jd T-sweep at L0, L12, L20.
- **Compared systems**: gram_error, jade, jd(T-sweep), + random-rows floor control.
- **Metrics**: bits canonical (probe) + readout (p95 cosine for jade/jd, cos_norm for
  gram_error); T-curve for jd.
- **Setup**: `max_dim=64` row cap, `max_features=256` feature cap (matches aloepri defaults);
  Hungarian-aligned p95 cosine; seeds for the random control = 3.
- **Success criterion**: pilot produces a coherent (bits, recovery) table; jd T-curve has a
  decided shape (flat vs climbing) distinguishable from the floor at 24 prompts (T≤16 → ≥1
  stack). **GO to B2** if either (a) jade recovery clears the floor on ≥1 layer×kind, or (b)
  the jd T-curve shows a non-flat trend worth confirming at scale.
- **Failure interpretation**: if everything sits at the floor, BSS is too weak on plaintext
  KV/QKV → that is a first-class negative result; still report it, skip the GPU scale-up.
- **Table/figure target**: main table (bits+readout per layer×kind), jd T-curve figure.
- **Priority**: MUST-RUN.

### Block 2: 512-corpus scale-up (CONDITIONAL — only if B1 GO; GPU, ONE capture job)
- **Claim tested**: C1 at statistical strength (T=16/32 with many non-overlapping stacks).
- **Why**: 24 prompts give only 1 stack at T=16; the release-gate-512 corpus gives 32 — needed
  for a CI on the cosine-vs-T slope.
- **Data**: GPU capture kq+kqv_out at **L0, L12, L20** for the 512-prompt corpus
  (`corpora/release-gate-512.txt`); resid_post at those layers already cached. ONE container,
  serial, GPU-wrapped via `scripts/run_in_rocm.sh`. PERF GATE before launch.
- **Metrics/setup**: identical to B1.
- **Success criterion**: cosine-vs-log₂T slope CI confirms (or refutes) the B1 trend.
- **Failure interpretation**: pilot trend was noise → fall back to the B1 negative result.
- **Priority**: NICE-TO-HAVE → promoted to MUST if B1 GO.

### Block 3: probe↔recovery correlation + decision (MUST-RUN, CPU)
- **Claim tested**: C2.
- **Why**: the measurement-loop verdict — does the attack-independent probe predict recovery?
- **Compared systems**: probe-bits vs recovery across every (layer×kind×T) cell from B1 (+B2).
- **Metrics**: Spearman ρ, Pearson r, scatter.
- **Setup**: probe = negentropy/excess-kurtosis of whitened rows (jade channel) +
  shared-spectral-capacity ½Σlog₂(1+λ_shared/λ_resid) of the **averaged** covariance over the
  T stack (jd channel), reusing `spectral_channel_mi.py`. **Attack-independence statement**:
  both are functions of covariance eigenspectra / whitened-row moments only; neither calls the
  joint-diagonalization routine. Could be computed with the BSS attacks deleted.
- **Success criterion**: |ρ| ≥ 0.7 ⇒ probe predicts attack → C2 holds, draft claim + proof.
- **Failure interpretation**: |ρ| < 0.7 ⇒ the finding is *why*: (a) attack too weak
  (queue stronger BSS / more T) or (b) probe not channel-matched (queue a matched probe).
  Bound the gap in theory; append a spawn-depth-1 follow-up.
- **Priority**: MUST-RUN.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | port + synthetic sanity | B0 | recoveries match analytic | ~min, CPU | port bug |
| M1 | dev-24 pilot | B1 | GO/NO-GO to B2 | ~min, CPU | thin stats at 24 prompts |
| M2 | scale-up (if GO) | B2 | slope CI | 1 GPU capture ~10–20 min | GPU saturation |
| M3 | correlation verdict | B3 | |ρ|≥0.7? | ~min, CPU | probe not matched |
| M4 | claim/negative-result + HTML | — | jury (auto-review) | ~min | — |

## Compute and data budget

- **GPU**: at most ONE capture job (512 prompts × Qwen3-4B forward × kq+kqv_out at 3 layers).
  Everything else (all 3 attacks, the probe, the correlation) is CPU numpy/BLAS on cached
  operands. Estimate capture wall-time ~10–20 min; if >10 min, confirm iGPU saturation
  (batch the forward passes, eager attention already required for kq/kqv_out capture) per the
  perf gate before launch. One GPU process at a time.
- **Data**: dev-24 (cached, free); release-gate-512 (cached prompts; needs kq/kqv_out capture).
- **Biggest bottleneck**: JADE 4th-order cumulant `O(m⁴·T)` — bounded by `max_dim=64`,
  `max_features=256`; numba-JIT joint-diag is CPU-parallel.

## Risks and mitigations

- **Hungarian-alignment inflation** → random-rows floor control (3 seeds); recovery must clear it.
- **Probe = attack in disguise** → probe is eigenspectrum/kurtosis only; stated + asserted in code.
- **Thin T=16 statistics at dev-24** → B2 scale-up gated on a B1 signal; otherwise honest
  negative result.
- **kq raggedness** (`n_kv` varies per prompt) → the existing `_flatten_operand` zero-pads to a
  global max_kv; BSS row axis = query positions, so pad rows are excluded by the strip rule.

## Final checklist

- [ ] Main table (bits + readout per layer×kind) covered — B1/B2
- [ ] Accumulation question (C1) decided with a chance floor — B1/B2/B3
- [ ] Probe isolated as attack-independent (C2) — B3
- [ ] sda/tfma/ia-weight documented not-applicable — experiment log
- [ ] GPU only if pilot signal; perf gate before capture — B2
- [ ] Negative result is first-class if recovery sits at floor or |ρ|<0.7

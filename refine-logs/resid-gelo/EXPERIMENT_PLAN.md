# Experiment Plan — resid-gelo (Task 5, campaign-B-expand)

**Problem**: GELO (arXiv 2603.05035, github.com/noskill/gelo) is a confidential-inference defense
that exposes only `U = A·H` for remote GPU compute — `A ∈ R^{n×n}` is a *fresh secret per-batch*
row-mixing over the `n` token rows of the residual `H ∈ R^{n×d}` (`A^{-1}=A^T` when orthogonal),
optionally padded with appended *shield rows* (decoy tokens). Correctness holds because left
row-mix commutes with right projection: `U W = A H W`, un-mixed in the TEE by `A^{-1}`.
**Method Thesis**: GELO is the *canonical blind-source-separation (BSS) setting* the plaintext
kv-accumulation phase flagged as the place where BSS becomes informative (there `U=H`, no mixing).
Under GELO there IS an unknown linear row-mixing — so the question is whether ICA/JD recover the
real rows, whether the matched geometry-only probe tracks that recovery across the privacy sweep,
and whether the orthogonal-A choice (κ=1) leaks the feature Gram by construction.
**Date**: 2026-06-24

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|------------------------------|---------------|
| **C0 (theory, structural)** orthogonal-A Gram-invariance leak: when `A` is orthogonal the column (feature) Gram `UᵀU = HᵀAᵀAH = HᵀH` is *exactly* invariant; the row Gram `UUᵀ = A G_H Aᵀ` is only orthogonally *conjugated* (Frobenius norm preserved, entries not). | A defense advertised as "secret mixing" still publishes an exact `d×d` functional of the secret at its recommended (orthogonal) setting — a structural leak, attack-independent. | Proof inline + empirical rel-err `‖UᵀU−HᵀH‖/‖HᵀH‖ ≈ 0` at κ=1, growing with κ. | B1, B3 |
| **C1 (empirical)** BSS recovery under GELO vs κ(A) and shield fraction: does ICA/JD (graded against a matched random-demixing floor) recover the real token rows, and how does the *genuine margin* move across the privacy sweep? Anchor ridge (amortized linear `U→H`) must fail under fresh-per-batch A. | Quantifies what the mixing actually buys vs the plaintext (U=H) baseline; positions GELO against KV-CLOAK. | Per-cell `p95 Hungarian cosine − random-demixing floor` swept over κ∈{1,3,10,30,100} × shield∈{0,0.5,1.0}; ridge recovery ≈ chance. | B2 |
| **C2 (measurement loop)** matched probe tracks recovery: the geometry-only negentropy / shared-spectral-capacity probe (bits) correlates with the genuine BSS margin across the κ/shield sweep. | The probe≠attack predictivity test for the BSS channel under a real mixing defense (the kv-accumulation C2 only had the Identity surface). | Spearman(probe bits, genuine margin) across all sweep cells, with a probe-independence check. | B2, B3 |

**Anti-claims to rule out**: (a) "recovery is just subspace-membership artifact" — graded against
the matched random-demixing floor, as in kv-accumulation; (b) "the probe is the attack in disguise"
— probe is whitened-moment/eigenspectrum only, computable with the joint-diag deleted; (c) "ridge
works so the mixing is useless / works so it's broken" — ridge is the amortized-inversion anchor and
is *expected* to fail (fresh A per prompt), which is itself the point.

## Paper Storyline
- **Main paper must prove**: C0 (orthogonal-A leak, the headline structural finding) + the C1/C2
  measurement-loop verdict (correlate or not — either is first-class).
- **Appendix**: shield-energy fine sweep; manifold vs Gaussian vs Student-t shield rows.
- **Cut**: gradient/learned inversion of A (out of scope; ridge is the linear anchor).

## Experiment Blocks

### Block 1 (B1): Channel-decoupling / Gram-invariance sanity — MUST-RUN
- **Claim tested**: C0.
- **Why**: establishes the structural identities the sweep rests on, before any attack.
- **Data**: one representative `resid_post` operand (L12) from the cached capture.
- **Compared systems**: GELO channels — `orth` (κ=1), `ill` (κ controlled), `shield-only`.
- **Metrics**: feature-Gram rel-err `‖UᵀU−HᵀH‖_F/‖HᵀH‖_F`; row-Gram conjugation residual
  `‖UUᵀ−G_H‖_F` vs Frobenius-norm-preservation `|‖UUᵀ‖_F−‖G_H‖_F|`; correctness `‖A⁻¹U−H‖`.
- **Success**: feature-Gram rel-err ≈ 1e-6 at κ=1, monotone↑ with κ; row-Gram entries change but
  Frobenius norm preserved at κ=1; un-mix exact.
- **Failure interpretation**: identity broken ⇒ implementation bug (fix before sweep).
- **Figure target**: identities table.

### Block 2 (B2): κ × shield privacy sweep — MUST-RUN
- **Claim tested**: C1, C2.
- **Why**: the core measurement loop (recovery + probe across the privacy parameters).
- **Data**: cached `resid_post` at layers {0,12,20} (subset of available {0..32}); ≤96 prompts.
- **Compared systems**: attacks `jade`, `jd`(T-stack), `gram_error`; **anchor** ridge `U→H`;
  **floor** random-orthogonal demixing (matched).
- **Metrics**: per-cell `p95 Hungarian cosine` recovery (readout) and its genuine margin over the
  matched floor; probe **bits** (`negentropy_bits`, `shared_spectral_capacity_bits`); ridge p95.
- **Setup**: κ(A)∈{1,3,10,30,100} (κ=1 ⇒ orthogonal), shield-fraction∈{0,0.5,1.0} (shield rows =
  Gaussian, energy-matched to median real-row norm); `max_dim=48`, `max_features=256`; seed 0;
  CPU on cached operands.
- **Success (C1)**: genuine margin is interpretable across κ/shield (any monotone trend reported);
  ridge p95 ≈ floor (amortized inversion fails). **Success (C2)**: |Spearman| ≥ 0.6 probe-vs-margin,
  or a documented non-correlation with a bounded explanation (weak-attack vs non-matched-probe).
- **Failure interpretation**: probe flat while recovery moves ⇒ non-matched probe → queue follow-up;
  recovery flat everywhere ⇒ mixing makes BSS as ill-posed as plaintext → that is the finding.
- **Figure target**: recovery-and-bits vs κ (faceted by shield); correlation scatter.

### Block 3 (B3): probe-independence + leak corroboration — MUST-RUN (light)
- **Claim tested**: C0 corroboration + C2 integrity.
- **Why**: confirm the probe is computable with the attack deleted, and that the feature-Gram leak
  is what an attacker would exploit at κ=1.
- **Metrics**: probe value with joint-diag code path disabled (must be unchanged); feature-Gram
  cosine recovery at κ=1 (should be ≈ exact) as the concrete C0 readout.
- **Success**: probe unchanged; feature-Gram exactly recovered at κ=1.

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | B1 sanity (identities) | gelo_sweep --sanity | identities hold | <1 min CPU | low |
| M1 | B2 pilot @ L12 | gelo_sweep (L12 only) | recovery+probe finite, ridge fails | ~2-4 min CPU | med (ICA convergence) |
| M2 | B2 full {0,12,20} + B3 | gelo_sweep | C1/C2 verdict | ~5-8 min CPU | med |

## Compute and Data Budget
- **GPU-hours: 0** — reuses cached `capture-28a0ee6c41330ee3.pt`; transform/attacks/probe are numpy.
  Perf gate: optimal scope (no redundant capture; 3-layer subset; capped dims), CPU is correct here.
- Biggest bottleneck: JADE/JD cumulant cost O(m⁴) — bounded by `max_dim=48`.

## Risks and Mitigations
- **Shield rows change row count** (breaks the same-row-count Transform contract): handle shields in
  the sweep spike (append before attack), keep the core GELO Transform shield-free / contract-clean.
- **ICA on correlated token rows** (sources not independent) may under-recover regardless of κ: that
  is a real finding, not a bug — graded against the matched floor it is interpretable.
- **κ control**: build A = orthogonal · diag(singular values with target κ) · orthogonal (SVD form).

## Final Checklist
- [x] Main result (C0 leak + C1/C2 loop) covered
- [x] Novelty isolated (matched floor; probe≠attack)
- [x] Simplicity defended (reuse Task-1 attacks + probe; one new Transform)
- [x] Frontier component: n/a (geometry/linear-algebra defense + classical ICA)
- [x] Nice-to-have (shield-energy/type fine sweep) separated from must-run

---
type: plan
status: current
created: 2026-06-22
updated: 2026-06-22
tags: [experiment-plan, anisotropic-dp, mahalanobis, sparse, bnn, mi-leakage, smarter-noise]
companion: [bnn-error-bounds/PROOF_PACKAGE, EXPERIMENT_TRACKER]
supersedes: []
---

> **⚠️ GATE FIRED (2026-06-22): M-1/B(-1) = GO-UNSTABLE → PIVOT-to-narrow.** The broad-dominance
> premise is NOT supported by the codebook geometry (low-noise scatter is spiky: eff-rank 7, ~6
> morphological pairs, ~3% tokens; low/high-noise eigenspaces near-orthogonal, cosθ=0.24). Per the
> pre-registered pivot clause, the strong **B0 theorem and M2 broad headline are on hold** pending the
> direction decision. See `EXPERIMENT_RESULTS.md`. Live track → narrow: nullspace-removal (11%) +
> morphological-direction targeting, quantified.

# Experiment Plan — Does *shaped* DP noise leak less token-MI and weaken BNN at smaller budget?

**Problem**: We proved & validated that BNN@L0 (nearest-neighbour MAP on the embedding table) defeats
**isotropic** DP noise at every utility-preserving ε; only utility-destroying noise stops it (the cliff,
`claim:bnn-error-bounds-bhattacharyya-fano`). Open question: can **anisotropic / "smarter" noise** move
that cliff — achieve the *same* token-MI reduction and BNN suppression at a **smaller total noise budget** —
and do the prominent shaped-noise schemes (SPARSE, metric-DP/Mahalanobis) actually realize this for the
*exact-token* threat, or only for their own target?

**Method thesis**: The token-MAP error is governed by the **Mahalanobis margins**
`m_vu = √(Δ_vuᵀ Σ⁻¹ Δ_vu)`. **Leakage objective** (signs fixed explicitly): privacy = *maximize*
attacker error `P_e` / channel equivocation `H(V|Y)`, equivalently *minimize* `I(V;Y)` and BNN success.
Adding variance along a confusable axis `Δ` **lowers** that pair's margin `m_vu`, which **raises** its
pairwise error and **raises** the union error-surrogate `U_B(Σ)=Σ_{v≠u}exp(−Δ_vuᵀΣ⁻¹Δ_vu/8)` — good for
privacy. At fixed **in-span distortion budget**, isotropic `Σ∝I` is *non-stationary* for this objective iff
the **weighted pairwise scatter** `S(Σ)=Σ_{v≠u} w_vu(Σ)·Δ_vuΔ_vuᵀ` (weights from the Bhattacharyya terms)
is **not** proportional to the identity on the codebook span — and the improving direction shifts variance
onto the **top eigenvectors of `S`**, not naively the smallest-`‖Δ‖` directions. Our probe — generalized for
free from `‖Δ‖` to `√(Δᵀ Σ⁻¹ Δ)`, `ε∼N(0,Σ)` — both **derives** the improving `Σ*` and **audits** whether
SPARSE's concept-aligned `Σ` lands on `S`'s top eigenvectors or misses them. (No new attack, no new probe.)

**Budget = privacy-relevant, not raw distortion.** `tr(Σ)` alone is gameable: the centered codebook spans a
rank-`≤(K−1)` subspace (≤2047 of d=2304), so isotropic wastes ~11% of trace in a **nullspace** that cannot
affect token identity. We therefore (a) constrain comparisons to **in-span trace** with an explicit
**isotropic-in-span** baseline (isolating the trivial nullspace-removal win), and (b) report the
**condition-number / Mahalanobis-DP ε** axis so "smaller noise" is a genuine privacy budget, not anisotropy
hiding noise in don't-care directions.

**Date**: 2026-06-22 · relies on user-supplied descriptions of SPARSE (arXiv:2602.07090, **elliptical-Laplace**,
concept-specific) and the Mahalanobis metric-DP line (Xu 2020; Chatzikokolakis 2013) — reproduction risk (R1).

## Claim Map
| Claim | Why it matters | Minimum convincing evidence | Blocks |
|-------|----------------|-----------------------------|--------|
| **C1 (primary, formal — surrogate-scoped)** — under `ε∼N(0,Σ)` the Mahalanobis bracket holds; and at fixed **in-span** budget, isotropic `Σ∝I` has a **strict trace-preserving ascent direction for the error-surrogate `U_B`** (equivalently descent for the success-surrogate) **iff** the weighted pairwise scatter `S` is not ∝ I on the span — the improving direction is `S`'s top eigenvectors. | The headline made a *correctly-signed, conditional* theorem, not a slogan: "smarter noise can leak less MI at smaller budget — and exactly when." | proof-checker PASS on the **surrogate** theorem (B0); the *exact* dominance (exact-MAP error ↑ and exact MC `I(V;Y)` ↓ at matched in-span budget) is the **empirical** claim (B2), reported with CIs and the surrogate-vs-exact gap. | B0, B2 |
| **C2 (primary, audit)** — SPARSE's concept-aware shaping does **not** necessarily minimise *token*-MI; quantify its gap to `Σ*` and to isotropic. | Resolves "is a prominent shaped-noise scheme actually token-private, or only private for its own concept?" — the probe adjudicates. | Two separate comparisons: (i) **SPARSE-cov Gaussian** (its covariance *shape* under our Gaussian probe = mechanism-class control) vs `Σ*` vs iso; (ii) **actual SPARSE (elliptical-Laplace) black-box** scored by empirical recovery (Laplace-MAP decoder) + **exact Laplace-posterior log-loss** for MI (generic kNN/MINE only as a cross-check; our Gaussian bracket does NOT model Laplace). Plus principal-angle overlap between SPARSE's high-noise subspace and `S`'s top eigenvectors. | B2, B4 |
| **C3 (supporting, anti-claim)** — the gain is **alignment to `S`'s top eigenvectors**, not anisotropy per se, and not nullspace removal. | Rules out "any elliptical noise helps" and "the win is just not wasting the nullspace." | A random-rotation control `Σ_rand` (same in-span spectrum as `Σ*`, random eigenvectors) shows **no** gain; and the **isotropic-in-span** baseline isolates how much of any gain is mere nullspace-removal (trivial) vs margin-alignment (the real claim). | B2 |
| **C4 (supporting, utility)** — the `I(V;Y)`/BNN gain survives a **matched-utility** budget, not just matched in-span trace. | "Leaks less MI" must not be bought by extra distortion. | Re-index budget by a downstream-utility proxy that is **not** circular with the attacker's quantity (NOT retained-MI — see B3); `Σ*` still dominates isotropic. | B3 |

**Anti-claims to rule out**
- **A1 "anisotropy itself helps"** → C3 random-rotation control gets no gain.
- **A2 "lower MI is just more total noise"** → matched **in-span** trace (B2) **and** matched-utility (B3).
- **A3 "the win is just nullspace removal"** → the **isotropic-in-span** baseline isolates the trivial ~11% nullspace effect; the real claim is the gain *beyond* it.
- **A4 "the probe is the attack again"** → probe is geometry-only (proven T3, extends to Σ); BNN is the attack; independence preserved.
- **A5 "lower BNN error ⟹ lower MI"** (Fano is one-sided) → `I(V;Y)`/`H(V|Y)` measured **directly** as a co-primary quantity, never inferred from error.
- **A6 "SPARSE was crippled"** → two faithful comparisons (Gaussian-cov control + actual elliptical-Laplace black-box at its stated ε); report SPARSE's own metric too.

## Paper storyline
- **Main paper proves (two separate strengths)**: (formal) C1/B0 — under anisotropic `S`, isotropic-in-span has a **surrogate local improving direction** (geometry-aware shaping *can* leak less, and exactly when); (empirical) B2 — geometry-aware `geom_optimal` **demonstrably** lowers *exact* `I(V;Y)` and raises *exact* BNN error vs `iso_inspan` at matched in-span budget. (Do not conflate: the proof is surrogate+local; the exact-attack dominance is empirical.) **C2** (B2/B4) — the prominent scheme (SPARSE) only realises this if its concept subspace coincides with `S`'s top eigenvectors (frontier-necessity verdict). + C3 novelty isolation.
- **Appendix**: C4 matched-utility; B4 SPARSE faithfulness + subspace-overlap diagnosis; B5 which token pairs `Σ*` newly protects.
- **Cut**: NVDP / DP-Forward (need a different attack + probe — out of scope, noted as future); a **certified-global / convex-relaxation** `Σ` (B0 gives only the *local* improving-direction & suboptimality-of-isotropic; global optimization of `Σ exp(−ΔᵀΣ⁻¹Δ/8)` is open and not claimed convex — nice-to-have).

## Experiment Blocks

### B(-1) — Geometry diagnostic FIRST (go/no-go before the proof) — MUST-RUN, FIRST
- **Why first**: the entire headline is conditional on the codebook's **in-span margin geometry being anisotropic**. If the weighted pairwise scatter `S` is ≈ isotropic on the span, shaping cannot beat isotropic-in-span and only the trivial nullspace win remains. Cheap to check; gates B0.
- **Compute (gemma-2-2b table, pool=2048, geometry-only) — across the full B2 budget grid, not one σ**: (i) centered-span rank + **nullspace trace fraction** (expect ~11%); (ii) eigen-spectrum & **effective rank** of the weighted pairwise scatter `S(B) = Σ_{v≠u} w_vu Δ_vuΔ_vuᵀ` at each budget `B` (weights `w_vu=exp(−Δ_vuᵀΣ⁻¹Δ_vu/8)`), with **eigenspace stability** across the transition (do the top eigenvectors drift/cross?); (iii) **gradient anisotropy** of `U_B` at isotropic (is `∇_Σ U_B|_{iso}=+⅛Σ⁻¹SΣ⁻¹` ∝ I on the span?); (iv) top-pair mass (does a handful of morphological pairs dominate `S`?).
- **Decision gate**: if `S` is meaningfully anisotropic in-span (eff-rank ≪ span-rank, or gradient clearly ≠ ∝ I) → proceed to B0/B2 with the broad-dominance headline. If near-isotropic → **pivot** the headline to "shaping helps only via nullspace-removal + a few morphological directions" (still a clean, honest finding) and skip the strong B0 theorem.
- **Cost**: <10 min. **Priority**: MUST-RUN, FIRST.

### B0 — Proof gate: anisotropic improving-direction theorem (surrogate-scoped) — MUST-RUN (after B(-1) go)
- **Claim**: C1 (formal core, correctly signed, conditional). **Gate**: no empirical "smarter noise wins" claim asserted until PASS.
- **Theorem (extends `claim:bnn-error-bounds-bhattacharyya-fano`)**: under `Y=clip(e_V)+N(0,Σ)`, uniform prior, K≥3:
  (a) **Mahalanobis bracket**: pairwise MAP error `= Q(½√(Δ_vuᵀΣ⁻¹Δ_vu))`; the union/Bhattacharyya upper and Fano-equivocation lower brackets hold verbatim with Euclidean `‖Δ‖` → Mahalanobis `√(ΔᵀΣ⁻¹Δ)`, `ε∼N(0,Σ)`.
  (b) **isotropic non-stationarity (correctly signed)**: with `U_B(Σ)=Σ_{v≠u}exp(−Δ_vuᵀΣ⁻¹Δ_vu/8)`, the covariance gradient is **positive PSD**, `∇_Σ U_B = +(1/8)·Σ⁻¹ S(Σ) Σ⁻¹` with `S(Σ)=Σ_{v≠u} w_vu Δ_vuΔ_vuᵀ`, `w_vu=exp(−Δ_vuᵀΣ⁻¹Δ_vu/8)`. (Derivation: `∂(ΔᵀΣ⁻¹Δ)/∂Σ = −Σ⁻¹ΔΔᵀΣ⁻¹`; chain rule gives the `+1/8`.) On the trace-preserving manifold `{Σ⪰0 : tr_span Σ = B}` the projection of this gradient off the identity is nonzero **iff `S` is not ∝ I on the span**; the **ascent** direction (which **raises** the error-**surrogate** `U_B` ⟹ lowers the success-surrogate; *exact* error/MI movement is empirical, B2) loads `S`'s **top eigenvectors**. So isotropic-in-span is not surrogate-optimal under anisotropic `S`. (Assumes isotropic is interior to the condition-number constraint.)
  (c) **smaller-budget corollary (LOCAL/surrogate only)**: by continuity along that ascent direction, a *nearby* shaped `Σ` reaches isotropic's surrogate value at *slightly* smaller in-span trace. This is a local statement — it does **not** assert broad Pareto dominance across all budgets (that is the empirical B2 claim).
- **Honest scope / caveats to encode**: (i) statement is about the **surrogate `U_B`**, not exact `P_e` or exact `I(V;Y)` — exact dominance is empirical (B2); (ii) it is a **local/first-order** improving-direction result — NOT a global optimum; maximizing `Σ exp(−ΔᵀΣ⁻¹Δ/8)` under trace/κ is **not known to be convex**, so `geom_optimal` is a **first-order / projected-gradient (local) heuristic**, not a certified global `Σ*` (a convex relaxation is nice-to-have, not claimed); (iii) `I(V;Y)` is NOT inferred from the error result — measured separately; (iv) budget is **in-span trace** with a condition-number floor; the in-span Gaussian is singular in full space, so the channel is defined as a **projected (pseudoinverse-Mahalanobis) channel on the centered span**, OR an explicit `λ_min I` floor outside the span with that out-of-span trace reported separately (valid full-rank DP mechanism); (v) δ/clip as in base proof.
- **Deliverable**: `refine-logs/PROOF_PACKAGE.md` + proof-checker PASS (≥ sound-modulo-imports). **Failure**: weaken C1 to empirical-only.
- **Priority**: MUST-RUN (gated by B(-1)).

### B1 — Generalize probe to Σ + build the four noise shapes — MUST-RUN (CPU sanity)
- **Builds**: extend `src/talens/measures/channel_error_bounds.py` to accept a covariance `Σ` (via Cholesky / eigh): `union_bhattacharyya(..., cov=Σ)`, `fano_equivocation(..., cov=Σ)` (draw `ε∼N(0,Σ)`; Mahalanobis distances). New `src/talens/measures/noise_shapes.py`, all normalized to equal **in-span trace** with a condition-number floor `κ_max`:
  - `iso_full(B,d)` = `(B/d)I` (raw isotropic, includes nullspace — the *gameable* reference);
  - `iso_inspan(table,pool,B)` = isotropic on the codebook span only (the **honest** isotropic baseline; isolates nullspace-removal);
  - `geom_optimal(table,pool,B,κ_max)` = variance loaded onto the **top eigenvectors of the weighted scatter `S`** (the B0 improving direction), in-span, condition-capped;
  - `sparse_cov(table,pool,B,concept)` = SPARSE's covariance *shape* (concept-mask → elliptical loading) used as a **Gaussian** mechanism-class control;
  - `rand_aniso(spectrum_of(geom_optimal), seed)` = `Σ*` in-span spectrum, random orthonormal basis.
- **Sanity tests** (`tests/test_anisotropic_bounds.py`, CPU, synthetic): Σ=σ²I reproduces the isotropic probe exactly; in-span trace + κ_max enforced for all constructors; Mahalanobis `Ĥ_M` unbiased vs brute force under Σ; `geom_optimal` **raises** the error-surrogate vs `iso_inspan` at equal in-span B (the correctly-signed B0 claim, numerically); `rand_aniso` ≈ `iso_inspan`.
- **Success**: tests pass; isotropic-reduction exact; signs correct. **Cost**: <45 min CPU. **Priority**: MUST-RUN.

### B2 — Main: (budget × noise-shape) dominance of BNN-error and I(V;Y) — MUST-RUN (GPU)
- **Claim**: C1 (exact-empirical), C2 (Gaussian-cov arm), C3.
- **Setup**: gemma-2-2b table, pool=2048, seed=20260622 (reuse `bnn_error_bounds_validation.py` harness). Budget grid = matched **in-span trace** spanning the transition (~8 pts), condition-number floor `κ_max` fixed. Shapes: {`iso_full`, `iso_inspan`, `geom_optimal`, `sparse_cov`, `rand_aniso`}.
- **Per (B, shape)**: measured **exact** BNN error (uniform-prior MAP, Mahalanobis decode) **and** **exact MC** `H(V|Y)`/`I(V;Y)` (Fano-MC under Σ), each with CIs (Hoeffding / MC-SE); plus the union/Fano bracket and the surrogate `U_B` (to report surrogate-vs-exact gap).
- **Metrics**: (i) **exact** BNN-err and **exact** `I(V;Y)` vs in-span B per shape (both co-primary, neither inferred from the other); (ii) **budget saving** = `B_inspan,iso / B_geom` at equal exact BNN-err; (iii) decomposition: `iso_full → iso_inspan` (trivial nullspace win) vs `iso_inspan → geom_optimal` (the real margin-alignment win); (iv) `sparse_cov` vs `geom_optimal` vs `iso_inspan` gap (C2); (v) `rand_aniso ≈ iso_inspan` (C3).
- **Success (C1)**: `geom_optimal` strictly above `iso_inspan` on exact BNN-err **and** below on exact `I(V;Y)` at matched in-span B (CIs disjoint), budget saving > 1 *beyond* the nullspace effect. **(C3)**: `rand_aniso` within CI of `iso_inspan`. **Failure**: if `geom_optimal ≈ iso_inspan`, the in-span margin geometry is near-isotropic (B(-1) should have caught this) → headline pivots to nullspace + morphological only.
- **Table/fig**: dominance plot (exact BNN-err & exact I(V;Y) vs in-span B, 5 curves); bracket bands; nullspace-vs-alignment decomposition bar; surrogate-vs-exact gap.
- **Priority**: MUST-RUN. **Cost**: ~25 min GPU (geometry-only + MC, ⊥ n).

### B3 — Matched-utility budget (non-circular) — MUST-RUN-light / appendix
- **Claim**: C4. Re-index the x-axis from in-span trace to a **utility** proxy that is **not** the attacker's quantity. **Not** retained-MI (circular). Use: (a) **downstream task** — a cheap linear probe (e.g. SST-2-style sentiment / a POS tag) trained on clean embeddings, evaluated on noised ones; OR (b) **cosine fidelity** to clean embedding *restricted to a task-relevant readout direction* (not the token-discriminative one). Recompute B2 dominance at matched utility.
- **Success**: `geom_optimal` still dominates `iso_inspan` at equal utility. **Failure**: the gain was a distortion artefact → C1 (matched in-span trace) stands but C4 retracted, and we report that geometry-aware noise trades the *same* utility for less leakage only under the trace budget, not the task budget.
- **Priority**: MUST-RUN (proxy) / NICE (full real-task). **Cost**: ~20 min.

### B4 — SPARSE: faithful black-box + concept≠token diagnosis — NICE-TO-HAVE (appendix)
- **Claim**: sharpen C2 with the *actual* mechanism (not just its covariance shape). **Two arms**: (i) **SPARSE-cov Gaussian** already in B2 (mechanism-class control, our bracket applies); (ii) **actual SPARSE** — elliptical-**Laplace** per its spec (concept-mask + generalized-Laplace) as a **black-box sampled defense**: draw its noise, run the **Laplace-MAP decoder** (the elliptical-Laplace density is known in closed form, so the matched MAP decoder — not Gaussian NN — is the right attack), and compute `I(V;Y)`/log-loss **directly from the known Laplace posterior** (NOT kNN/MINE, which are unreliable in d=2304; kNN/MINE only as a cross-check, never headline). Token recovery is reported as **empirical** (the Gaussian bracket does not certify Laplace). Evaluate at SPARSE's stated ε=10 and report SPARSE's *own* metric (attribute leakage) alongside token recovery. Compute **principal angles** between SPARSE's high-noise subspace and `S`'s top eigenvectors → explains match/miss.
- **Priority**: NICE-TO-HAVE. **Cost**: ~40 min + concept labels (R1/R2).

### B5 — Which pairs Σ* newly protects — NICE-TO-HAVE
- Qualitative: list token pairs that flip from BNN-recovered (isotropic) to BNN-confused (`Σ*`) at matched B — expect the morphological twins get the targeted variance. Heatmap: `Σ*` eigen-directions vs confusable-pair axes. **Priority**: NICE-TO-HAVE. **Cost**: trivial (post-process).

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| **M-1** | **Geometry diagnostic FIRST** | B(-1) | weighted scatter `S` anisotropic in-span? go/pivot | <10 min | **Gates everything** |
| M0 | Improving-direction theorem (surrogate) | B0 (proof-writer→checker) | PASS ≥ sound-modulo-imports | ~2–3 h | Med (surrogate-only scope) |
| M1 | Σ-probe + 5 shapes + sanity | B1 + pytest | tests pass; iso-reduction exact; signs correct | <1 h CPU | Low |
| M2 | Main dominance (exact err + exact MI) | B2 | `geom`≻`iso_inspan` (CIs disjoint), saving>1 beyond nullspace; `rand`≈`iso_inspan` | ~25 min GPU | Med (in-span spectrum may be ~iso) |
| M3 | Non-circular utility + SPARSE audit | B3, B4 | `geom`≻`iso_inspan` at matched utility; C2 verdict | ~60 min | Med (SPARSE Laplace faithfulness) |
| M4 | Polish | B5 | — | trivial | Low |

**Must-run**: M-1 → (go) → M0 → M1 → M2 → B3-proxy. **Nice-to-have**: B4, B5, real-task utility, certified-global `Σ` relaxation (open).
**If M-1 says pivot**: skip M0's strong theorem; headline becomes the honest "nullspace-removal + few morphological directions" audit (C2/C3 still run).

## Compute and Data Budget
- Total GPU ≈ **<1.5 h** (probe is geometry-only + MC, ⊥ test-set size). Proof gate is the wall-clock driver (M0).
- Data: reuse gemma-2-2b table + release-gate-512 pool. **SPARSE needs a sensitive-concept label set** (its mask) — the main data dependency (B4); for B2 use a proxy concept or a synthetic sensitive subspace, labelled as "SPARSE-style."
- Human eval: none. Biggest bottleneck: the B0 proof scope (surrogate vs exact `P_e`).

## Risks and Mitigations
- **R1 unverified sources** (SPARSE 2602.07090 / NVDP 2601.02307 post-cutoff; reviewer confirmed SPARSE exists): implement the *mechanism class* from the description; the Gaussian arm is a "SPARSE-cov" control, the Laplace arm is the faithful black-box — neither claimed as exact paper reproduction.
- **R2 SPARSE needs concept labels**: use a proxy/synthetic sensitive subspace for the B2 Gaussian-cov arm; full label set only for B4.
- **R3 surrogate ≠ exact P_e / MI**: B0 theorem is **surrogate-only**; B2 is the **exact** empirical test (exact-MAP error + exact MC `I(V;Y)`, CIs) and reports the surrogate-vs-exact gap.
- **R4 in-span margin spectrum near-isotropic** (no room beyond nullspace): **checked FIRST in B(-1)**; if near-flat, headline pivots to the nullspace+morphological audit rather than failing silently.
- **R5 budget gameable / "matched budget" contested**: in-span trace + condition-number floor + the `iso_full→iso_inspan→geom` decomposition isolate trivial vs real gains; B3 adds a non-circular utility axis.
- **R6 MI inferred from error** (Fano one-sided): `I(V;Y)` measured directly as co-primary (A5); never inferred from BNN error.
- **R7 DP accounting must be explicit** (before proof packaging): state the privacy axis precisely — `(ε,δ)` / RDP / zCDP / Mahalanobis-sensitivity — and the channel definition (projected pseudoinverse-Mahalanobis on the centered span **vs** full-rank with a `λ_min I` floor outside the span, out-of-span trace reported separately). The structure is in place; the accountant must be named, not implied.

## Final Checklist
- [ ] **Geometry diagnostic FIRST** (B(-1)): weighted scatter anisotropic in-span? go/pivot gate fired
- [ ] Improving-direction theorem, **correctly signed + surrogate-scoped** (B0, proof-checker PASS) — C1 formal
- [ ] Σ-generalized probe + 5 shape constructors (incl. `iso_inspan`) + sign-checked sanity tests (B1)
- [ ] Main dominance on **exact** err **and** **exact** I(V;Y) with CIs; nullspace-vs-alignment decomposition; `rand`≈`iso_inspan` (B2) — C1, C3
- [ ] SPARSE: Gaussian-cov control + Laplace black-box + principal-angle overlap (B2/B4) — C2 verdict
- [ ] Non-circular matched-utility dominance (B3) — C4
- [ ] Budget is privacy-relevant (in-span trace + κ floor), not gameable raw `tr(Σ)`
- [ ] MI measured directly, never inferred from error (A5)
- [ ] Frontier-necessity answered: is learned/concept noise necessary, or does closed-form `geom_optimal` suffice?
- [ ] Simplicity defended: closed-form `geom_optimal` (top-`S`-eigenvectors) vs trained SPARSE mask
- [ ] Must-run vs nice-to-have separated; pivot path defined if M-1 says no
- [ ] /auto-review-loop positive assessment on this proposal

# Auto Review — shaped/anisotropic-noise proposal (refine-logs/EXPERIMENT_PLAN.md)

Reviewer: gpt-5.5 via Codex MCP (xhigh), difficulty=medium. Thread: 019ef087-3db5-75c0-b01d-1684de91c05e.
Reviewing the proposal to test anisotropic DP noise (Mahalanobis family + SPARSE) vs BNN@L0 + our probe.

## Round 1 (2026-06-22)

### Assessment (summary)
- **Score: 5/10 · Verdict: not ready.** C1 NOT provable as stated.
1. **Sign/objective bug** — more variance along a confusable Δ *raises* the error surrogate / lowers MI; plan said "lowers U." Define objective signs explicitly.
2. **Local optimum = weighted pairwise scatter**, not "small-‖Δ‖". Isotropic non-stationary iff weighted-scatter gradient ≠ ∝ I.
3. **tr(Σ) = distortion, not privacy; gameable** — centered codebook span rank ≤ 2047 < d=2304 ⟹ isotropic wastes ~11% trace in a token-irrelevant nullspace; shaped Σ wins trivially. Add in-span baseline + condition-number / Mahalanobis-ε axis.
4. **Surrogate ≠ exact P_e / exact MI** — theorem surrogate-only; require empirical exact-MAP error + exact MC H(V|Y) with CIs.
5. **BNN error ≠ MI** (Fano one-sided) — measure I(V;Y) directly as primary.
6. **SPARSE is concept-specific + Laplace** — split into Gaussian-cov mechanism control vs actual Laplace black-box with its own metrics. (arXiv:2602.07090 confirmed to exist.)
7. **Check geometry anisotropy FIRST** (before the proof): weighted-scatter spectrum/eff-rank, gradient anisotropy, top-pair mass, nullspace waste. Headline dies if in-span geometry is near-isotropic.

### Reviewer raw response
<details><summary>expand</summary>

Score 5/10, Not ready, C1 provable as stated: No.
1. Sign/objective: increasing variance along Δ lowers that pair's Mahalanobis margin and INCREASES U_B=Σexp(−ΔᵀΣ⁻¹Δ/8); plan said "lowers U(Σ)" while raising BNN error — backwards unless U redefined as leakage/success. Fix: define objective; prove correctly-signed local derivative.
2. Pairwise direction locally right, globally insufficient: trace reallocation helps some pairs, hurts others; valid local direction governed by a weighted pairwise scatter matrix, not "small-‖Δ‖". Fix: isotropic non-stationary iff weighted scatter gradient not ∝ identity.
3. tr(Σ)=B distortion not privacy; gameable. K=2048,d=2304 → centered span rank ≤2047, isotropic wastes ~11% trace in nullspace; shaped cov wins by removing nullspace waste (true but weak). Fix: isotropic-in-span baseline + report in-span trace; constrain min-eigenvalue/condition or use Mahalanobis sensitivity/ε.
4. Union-Bhattacharyya optimization not proof of exact P_e/MI; base proof notes upper can be vacuous, Fano loose, mid gap wide. Fix: surrogate-only theorem + empirical exact-MAP + exact MC H(V|Y) with CIs.
5. BNN error and MI not interchangeable (Fano one-sided). Fix: H(V|Y)/I(V;Y) primary measured, not inferred.
6. SPARSE concept-specific, mask-learning + Mahalanobis/generalized-Laplace, evaluates sensitive-token leakage/utility, not Gaussian token-codebook MAP. Fix: SPARSE-cov Gaussian (mechanism control) and actual SPARSE/Laplace black-box, separately.
7. R4 before proof: proof only conditional/local; headline dies if weighted margin geometry near-isotropic. Add M−1 diagnostic across budget transition.
Bottom line: promising empirical geometry audit, not yet top-venue formal claim. Rewrite C1: "Mahalanobis bounds extend; under non-isotropic weighted-pair scatter, isotropic has a local trace-preserving ascent direction for the error surrogate; exact MI/error dominance is empirical."

</details>

### Actions taken (Round 1 → plan revision)
All 7 folded into refine-logs/EXPERIMENT_PLAN.md (see Round-2 diff).

### Status: continuing to Round 2.

## Round 2 (2026-06-22)

### Assessment (summary)
- **Score: 7/10 · Verdict: almost** (stop condition ≥6 ∧ almost MET) — conditioned on fixing one math error.
- Remaining items (all applied in Round 2→3 revision):
  1. **Gradient SIGN error** (the one blocker): `∇_Σ U_B = +(1/8)Σ⁻¹SΣ⁻¹` (positive PSD), not `−(1/4)…`. "Top eigenvectors of S" conclusion survives. FIXED.
  2. "Global Σ* is a convex program" likely false → call `geom_optimal` a first-order/local heuristic. FIXED.
  3. `iso_inspan` is singular in full space → define projected (pseudoinverse) channel on span OR `λ_min I` floor outside span, trace reported separately. FIXED.
  4. Smaller-budget corollary → state LOCAL only, not broad Pareto. FIXED.
  5. B(-1) → compute `S(B)` across the whole budget grid + eigenspace stability, not one σ. FIXED.
  6. Laplace MI → use Laplace-MAP decoder + exact known-posterior log-loss; kNN/MINE only cross-check, never headline. FIXED.
  7. Storyline overstated ("proves … beats isotropic against optimal attack") → separate surrogate-local-proof from exact-empirical. FIXED.
- Budget framing + SPARSE/Laplace split judged "mostly sound now."

### Reviewer raw response
<details><summary>expand</summary>

Score 7/10. Verdict Almost, but not ready until the theorem statement is corrected. C1 as stated: No, literally — B0 covariance-gradient sign wrong: ∇_Σ U_B = +(1/8)Σ⁻¹S Σ⁻¹ (negative sign would be wrt precision-like directions). "Top eigenvectors of S" still right after fixing. Weaknesses: (1) sign error → +const·Σ⁻¹SΣ⁻¹, define prefactor; (2) "global Σ* convex program" likely false — maximizing Σexp(−ΔᵀΣ⁻¹Δ/8) under trace/κ not obviously convex → call geom_optimal first-order/heuristic local; (3) iso_inspan singular in full space → projected pseudoinverse channel OR λ_min I floor outside span, report separately; (4) smaller-budget corollary local not broad; (5) B(-1) grid-based across budget, eigenspace stability; (6) generic kNN/MINE MI in 2304-d weak + define Laplace decoder → use known Laplace posterior/log-loss, don't lean on kNN/MINE for headline; (7) storyline line 59 overstates — separate proves-surrogate-local from empirically-demonstrates-exact. Net: one math correction + wording discipline → ready for execution.

</details>

### Actions taken
All 7 applied to refine-logs/EXPERIMENT_PLAN.md (sign +⅛Σ⁻¹SΣ⁻¹; geom_optimal=local heuristic; iso_inspan DP channel defined; corollary local; B(-1) grid+stability; Laplace-MAP+known-posterior MI; storyline split). Round 3 = confirmation.

## Round 3 (2026-06-22) — TERMINAL

### Assessment (summary)
- **Score: 8/10 · Verdict: READY for execution.** Stop condition met (≥6 ∧ ready). Loop terminates positively.
- C1 confirmed mathematically sound as a **local, in-span, surrogate improving-direction** theorem; corrected gradient `∇_Σ U_B = +(1/8)Σ⁻¹SΣ⁻¹` verified; top-eigenvector loading defensible (isotropic interior to κ-constraint).
- 4 residual NON-BLOCKING wording fixes — all applied: (1) "raises error"→"raises the error-surrogate (exact empirical)"; (2) claim-map C2 now says Laplace-posterior log-loss primary (matches B4); (3) dropped "SDP/global-optimal" → "certified-global/convex-relaxation, open, not claimed convex"; (4) added R7 — name the DP accountant (ε,δ/RDP/zCDP) + channel (projected-pseudoinverse vs λ_min-floor) before proof packaging.

### Reviewer raw response
<details><summary>expand</summary>

Score 8/10. Verdict Ready for execution, with minor wording cleanup before proof packaging. C1 now sound as a local, in-span, surrogate improving-direction theorem; corrected gradient ∇_Σ U_B=+(1/8)Σ⁻¹SΣ⁻¹ right; trace-preserving projected gradient nonzero iff S not ∝ I on span; top-eigenvector loading defensible assuming isotropic interior to condition-number constraints. Residual non-blocking: (1) "raises U_B raises error" → "raises error-surrogate/upper-bound surrogate", exact error empirical B2; (2) claim-map line says actual SPARSE generic MI estimate but B4 says Laplace posterior log-loss primary — sync; (3) lines still mention "global-optimal Σ via SDP" — convexity dropped, call it relaxation or remove; (4) before execution pin DP accounting: projected pseudoinverse vs λ_min floor, and whether axis is approx-DP/RDP/zCDP/Mahalanobis sensitivity. No remaining methodological blocker. Run M-1 first; only proceed to strong empirical headline if S(B) anisotropic and stable across the budget grid.

</details>

### Actions taken
4 wording fixes applied to refine-logs/EXPERIMENT_PLAN.md. Plan is execution-ready.

### Status: TERMINATED — positive assessment (8/10, ready). Score progression 5 → 7 → 8.

## Method Description
Audit whether *shaped* (anisotropic, covariance-Σ) DP noise on token embeddings leaks less token mutual information and weakens the Bayes-NN (nearest-neighbour MAP) embedding attack at a smaller privacy budget than isotropic noise. The verified geometry-only error-bounds probe (union-Bhattacharyya upper + Fano-equivocation lower) extends from Euclidean to Mahalanobis margins (`‖Δ‖ → √(ΔᵀΣ⁻¹Δ)`, `ε∼N(0,Σ)`), so it brackets the attack under any Σ without running it. The formal result (surrogate, local): at fixed in-span trace, isotropic-in-span noise has a strict trace-preserving ascent direction for the error-surrogate `U_B`, with gradient `+⅛Σ⁻¹SΣ⁻¹` (`S` = error-weighted pairwise scatter); the improving direction loads `S`'s top eigenvectors — i.e. put more noise along the confusable-token directions. Empirically (gemma-2-2b), a geometry-aligned Σ is compared against isotropic-in-span, a random-rotation control, and SPARSE's covariance shape on exact BNN error and exact MC `I(V;Y)` at matched in-span budget, decomposing any gain into trivial nullspace-removal vs real margin-alignment, and auditing whether the prominent concept-aware scheme (SPARSE) actually aligns with the token-margin geometry or misses it.

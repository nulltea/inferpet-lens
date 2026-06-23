# Proof Audit — claim:capacity-matched-pvi (estimator-regime lemma)

**Verdict: PASS** (Codex gpt-5.5 xhigh, thread 019ef5d7, 2 rounds). Proof inline in
`research-wiki/claims/capacity-matched-pvi.md` §Theory.

## Round 1 issues → fixes
- **Lemma 1** (quantifier/hidden-assumption): construction must not condition on the realized label,
  and the "bounded elsewhere" remainder must be uniform. **Fix:** label-independent reader
  `q_ε(c|x)=ε`; bound `E[log2 q_ε(Y|x)] ≤ π_c log2 ε`, so `PVI ≤ π_c log2 ε + H(Y) → −∞`.
- **Proposition 2** (CRITICAL: reference-mismatch + case-incomplete + illegal-interchange):
  Cover is binary; Soudry multiclass needs care; training-point confidence ≠ held-out; the
  existential Lemma 1 does not give a sequence limit. **Fix:** restricted to the **in-sample
  interpolation regime** with a **direct finite-sum** divergence (≥1 term →−∞, rest bounded above),
  multiclass via Lyu–Li 2020; narrowed to the asymptotic **existence-mechanism** of the floor.
- **Proposition 3** (CRITICAL: scope-overclaim + quantifier): `k<n_val` was unused; `−log2 p(y)` is
  wrong (y random); "independent of d" needs B fixed. **Fix:** `PVI ≥ log2 σ_min + H(Y) ≥ log2 σ_min`
  with `σ_min=1/(1+(K−1)e^{2B})`, B fixed independent of d; **`k<n_val` demoted** to the practical
  enabler of a bounded-logit fit (not a proof step) — reconciling the empirical "dim-anchored" finding.

## Round 2: PASS
All three gaps closed in substance. Non-blocking note: "generic attractor" is informal but harmless
because Prop 2 takes interpolation as an explicit hypothesis.

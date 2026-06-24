# Experiment Audit Report — resid-gelo (Task 5)

**Date**: 2026-06-24
**Auditor**: Codex gpt-5.5, xhigh reasoning (cross-model, read-only) — thread 019ef843-47ca-7880-bace-f396f1e16c39
**Project**: transformer-attacks-lens / GELO defense on Qwen3-4B residual

## Overall Verdict: WARN
## Integrity Status: warn (no integrity-critical fraud; scope-hygiene warnings only)

## Checks

### A. Ground Truth Provenance: PASS
Plaintext rows come from cached `results/capture_cache/capture-28a0ee6c41330ee3.pt` (gelo_sweep.py:44,225,230),
real `resid_post` prompt matrices. Recovery target `real_h = Hs[:n_real]` (gelo_sweep.py:179) is the actual
H fed into `U = A·H_aug` (gelo_sweep.py:109) — self-consistent, no attack-derived GT.

### B. Score Normalization / Baseline Honesty: PASS
Floor = matched random-orthogonal demix (same whitening gelo_sweep.py:145, random rotation :148, same
Hungarian p95 :149). Genuine margin = `jade_m - floor_m` (:191). Raw p95 Hungarian |cosine| (bss.py:156).
No self-max normalization.

### C. Result File Existence: PASS
analysis.json reads only sweep.json; headline C0 sequence (RESULTS.md:27) traces to shield-0 raw cells;
C1 table (RESULTS.md:41-43) matches analysis.json:10-43; ridge 0.288 / floor 0.667 / Spearman 0.293,0.507
present in analysis.json. No phantom numbers.

### D. Probe ≠ Attack: PASS (integrity-critical, confirmed)
bss_separability imports `_whiten`,`_subsample` but NOT `_joint_diag` (bss_separability.py:30). Probe =
row moments on whitened data (`_row_negentropy_nats`, :38). Sweep's `_negentropy_bits` (gelo_sweep.py:152)
whitens then calls it — no joint-diag. Feature-Gram metric is `U.T@U` vs `H.T@H` (:163), geometry-only.

### E. Identity Authenticity: PASS
`make_mixing` returns orthogonal `q1` at κ=1 (gelo.py:51); `Q1 diag(s) Q2` with log-spaced singular values
for κ>1 (:53). Sanity does real algebra + `np.linalg.solve(A,U)` un-mix (gelo_sweep.py:66); sanity.json
confirms cond=1, feat-Gram relerr 2.5e-16, Frobenius preserved, unmix < 5e-15. Not a normalization artifact.

### F. Ridge Anchor Honesty: PASS
`_ridge_anchor` fits W on `obs[:n_tr]`, evaluates on disjoint `obs[n_tr:]` (gelo_sweep.py:213), n=20 held-out.
Failure expected by design (fresh-per-prompt A; seed includes prompt_index, :123) — correct negative control.

### G. Scope: WARN
Actual: 3 layers {0,12,20} × 5 κ × 3 shield = 45 cells, 48 prompts, max_dim 32, seed 0 (single seed).
RESULTS.md reports scope and the partial verdict / C2 failure honestly. Warnings:
- EXPERIMENT_PLAN.md planned `jd`, `gram_error`, `shared_spectral_capacity_bits` (line 52) but the sweep ran
  only JADE + matched random-demix + negentropy + feat-Gram relerr + ridge. (Deferred to follow-up.)
- "JADE/JD" wording → tightened to "JADE (joint-diag)" in RESULTS.md (distinct JD-stack attack not run).

## Action Items (addressed)
- [x] RESULTS.md "JADE/JD" → "JADE (joint-diag)".
- [x] Scope (single seed, JADE-only) recorded as a caveat in the claim + follow-up Task 10.
- [ ] (follow-up) run JD-stack / gram_error / stronger BSS + shared-spectral / feature-Gram-matched probe.

## Claim Impact
- C0: supported.
- C1: supported, scoped (tested JADE attack, single seed).
- C2: unsupported as stated (ρ below 0.6 bar); feature-Gram-mismatch diagnosis is a hypothesis.

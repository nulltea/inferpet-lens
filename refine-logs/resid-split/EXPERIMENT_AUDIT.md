# Experiment Audit Report — resid-split / PriPert (Task 6)

**Date**: 2026-06-24
**Auditor**: Codex xhigh (cross-model, read-only), thread 019ef87c-ab65-72f0-bb61-0b84f006b54c
**Project**: transformer-attacks-lens

## Overall Verdict: WARN  (no fraud; documentation hygiene only — both flags now FIXED)

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
GT = tokenizer token ids from corpus prompts (`capture.py:159`, `types.py:121`), never model
outputs. Vocab-disjoint split genuine by token-id partition (`splits.py:67,82,86`); every inverter
called with `split_mode="vocab"` + shuffle control (`pripert_sweep.py:86`); selectivity =
real − shuffle (`pripert_sweep.py:92`).

### B. Score Normalization: PASS
TTRSR top-1 raw hit rate (`ridge.py:113,116`); I_G raw bits `½log1p(λ/σ²)/log2` (`spectral_channel_mi.py:159,161`);
CLUB nats→bits only (`club.py:198`); Fano `min(1,(accessible+1)/H_X)`. No prediction-derived denominator.

### C. Probe ≠ Attack (integrity-critical): PASS
I_G receives only `cov(Sparsify_ρ(H))`, σ, and entropy metadata (`pripert_sweep.py:76,78,104`) — `S`
is computed separately from the attack's `U`. CLUB trains its own estimator on `(U, embedding Y)`
(`club.py:101`). Neither consumes inverter predictions/labels. Fano's pool size `M` comes from the
inverter, **but `i_g_bits` does not depend on `M`**, so the 0.958 correlation is NOT circular.

### D. Result File Existence / Number Match: WARN → FIXED
All headline numbers match `runs/sweep/pripert_sweep.json` (spearman_ig_vs_bestrec=0.9582…,
ig_vs_mlp2=0.9147…, club_vs_bestrec=0.9765…, n_finite_ig=24/32, n_converse_violations=0; L8/ρ0.25/β2
I_G=10.0136, Fano=0.9862, best=0.00242). Flagged: the L0 within-layer Spearman cited as 0.543
recomputes fragilely (auditor got 0.400 under a different best-definition), and EXPERIMENT_AUDIT.json
was absent. **Fixed**: L0 estimate softened to "≈0.5, n=6, fragile point estimate"; audit files written.

### E. Phantom Exclusion / Cherry-Picking: PASS
β=0 ⇒ σ=0 ⇒ I_G infinite/vacuous by design (`spectral_channel_mi.py:144`), serialized as
`null` + `i_g_is_inf` (`pripert_sweep.py:110`), excluded from rank correlations and disclosed
(`RESULTS_STANDARDIZED.md:9`; `n_finite_ig=24/32` in JSON). Principled, not selective.

### F. Scope Honesty: WARN → FIXED
Limits disclosed: Gaussian/PCA proxy not adversarial-optimized δ (`pripert.py:24`), single seed +
one model/corpus (`RESULT_TO_CLAIM.md:21`), 3-ρ/depth grid (`RESULTS_STANDARDIZED.md:13`); C3
correctly slack/loose; C2 scoped to "observed/tested attacks" in RESULT_TO_CLAIM.md. Flagged:
`RESULTS_STANDARDIZED.md:25` said "no probe–attack gap" without the qualifier. **Fixed**: now reads
"no probe–attack gap is observed for the tested inverters (monotone tracker, not tight predictor)."

### Evaluation Type: real_gt (dataset-provided token ids)

## Action Items
- [x] Soften L0 within-layer Spearman to a robust statement (done).
- [x] Add the "observed/tested attacks" qualifier to RESULTS_STANDARDIZED.md (done).
- [x] Write EXPERIMENT_AUDIT.json/md (done).

## Claim Impact
- **C1**: supported — scope to depth-dependent suppression (L0 resists).
- **C2**: supported — scope to "no observed gap for tested attacks"; probe is a monotone capacity
  tracker, not a tight predictor.
- **C3**: supported — valid but slack converse, 0 violations.

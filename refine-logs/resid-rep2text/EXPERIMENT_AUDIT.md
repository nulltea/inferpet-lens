# Experiment Audit Report — resid-rep2text

**Date**: 2026-06-24 · **Auditor**: Codex gpt-5.5 xhigh (cross-model, read-only) ·
**Project**: transformer-attacks-lens / Rep2Text on Qwen3 L10 residual

## Overall Verdict: WARN  ·  Integrity Status: warn  (no fraud found)

## Checks
### A. Ground Truth Provenance: WARN
GT = true input token ids from corpus text (PIQA/MMLU/IFEval), not model-derived (PASS on source).
Train/test residuals disjoint. WARN: shuffled null is a random permutation, not a strict derangement
(possible fixed points → the gap is *conservative*); residual standardization uses the full-ensemble
mean/std incl. test (negligible; applied identically to real + controls).

### B. Score Normalization: PASS
Raw token-F1 / ROUGE-L (LCS); plain means; low raw values (mean control 0.002). No self-normalization.

### C. Result File Existence: PASS
All headline numbers (I_G=2856.48, per-bucket gaps +0.015..+0.089 with bootstrap CIs, across-σ
Spearman 1.0/0.943, rd_proxy ρ=0.176, n=36) exist in `runs/rep2text_results.json` and match the
narrative rounding; computed by code on the live run path.

### D. Dead Code Detection: WARN
Core metrics, probe, and correlations are live. WARN: the planned third "prior-only/random" control is
absent (only mean + shuffled implemented); one leftover unused variable. Non-blocking.

### E. Scope Assessment: WARN
Narrow by design: one source/decoder pair (Qwen3-4B→1.7B), L10, single seed, single adapter,
N=23/bucket. Narrative scoped "for this setup"; multi-seed/decoder/layer sweeps explicitly cut. Strong
phrasings softened with "for this setup".

### F. Probe-Is-Not-The-Attack: PASS (critical)
`spectral_channel_mi` is covariance-spectrum + fixed geometry-only σ_ref math ONLY; it never receives
generated text, target tokens, or decoder outputs and could not be computed by running the attack.
NOT circular. Minor WARN: probe σ = √(inject²+σ_ref²) (scalar offset by the floor); H_X=L·log₂(vocab)
honestly flagged as an upper proxy. Bootstrap is correctly paired over examples.

### F(eval-type). real_gt

## Action Items (all addressed in EXPERIMENT_RESULTS.md "Caveats")
- Scope every conclusion to *this setup* (done).
- Note the conservative-null (fixed-point) and standardization caveats (done).
- Note σ_ref offset + H_X upper-proxy (done).
- Follow-up (resid-rep2text-v2): add prior-only control, derangement null, empirical H_token,
  multi-seed, and the matched V-information probe.

## Claim Impact
- C1 (bottleneck length-decay): **REFUTED** — supported.
- genuine-leakage-significant: **supported** (all bootstrap CIs exclude 0; conservative null).
- C2 capacity probe vacuous across length (ρ=0.18): **supported**.
- C2 across-σ ordinal (ρ=1.0/0.94): **needs qualifier** — binding only after >80 % capacity destroyed.

# Experiment Audit Report — resid-dp-attacks

**Date**: 2026-06-23
**Auditor**: Codex gpt-5.5 xhigh (cross-model, read-only) · thread 019ef51c-6a8b-78e2-8642-597e244f80c4
**Scope**: consolidated b2/b6 residual-stream-under-DP attacks (R1–R6) vs source JSONs.

## Overall Verdict: WARN  (no fraud FAIL)

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN
GT = token ids from tokenization / cached prompt ids — EXCEPT the stored **L0 run (R1) uses a synthetic
Zipf token-id sample over the real embedding table** (N=7000 matches the Zipf fallback path). Vocab-disjoint
splits are real (split by unique token id, then collect rows); candidate pools explicitly retain all test
true ids (the bug that previously dropped them is fixed in `l0_fast.txt`). **Action taken:** R1 relabeled
`synthetic_proxy` in RESULTS_STANDARDIZED.md. Shuffle floors (R2/R5) computed once on clean activations and
reused across noise — disclosed assumption; **Action taken:** noted in the doc.

### B. Score Normalization: PASS
Recovery = raw exact-match mean; selectivity = raw recovery − shuffle floor; uplift = raw difference.
No metric divided by model max/min/mean. Near-1.0 R1 scores are raw L0 exact-channel recovery.

### C. Probe ≠ Attack Circularity: PASS
CLUB trains q(Y|X) from representations+embeddings; capPVI trains a reduced-dim token-class reader; MDL
computes online code length from (X,y). None call attack recovery. (`mdl_probe_check.py` loads the lpos
recovery numbers only for the correlation *report* — MDL bits themselves are computed independently.)

### D. Result File Existence + Number Match: PASS
All files exist; spot-checks match — R5 deep +0.828571 / ridge −0.085714 (b6_strong_decoder.json),
R6 fmv 0.7375 / uplift +0.525889 (b6c_forward_model.json), R3 mdl +0.80 (mdl_probe_check.json),
R1 ridge 0.020 / Bayes 1.000 / uplift +0.980 (l0_fast.txt).

### E. L0 Pre/Post-Bugfix Provenance Split: PASS
Disclosed in the doc: recovery from post-fix `l0_fast.txt`, bits from prior `b2_l0_bayes.json`. The bug
(pool truncation dropping true ids) affected only the *attack* candidate pool; probe bits use (Y, ids) /
(Y, table[ids]) independent of the pool → bits not invalidated. Mixed provenance, honestly labeled.

### F. Scope Assessment: WARN
R4 = 4 ε / 1 seed (doc says "Suggestive"); R6 = 3 ε; R5 = 6 ε / **1 seed** — was tagged "Headline run".
**Action taken:** added single-seed caveat + "no robustness claim; multi-seed CIs are the named firm-up".

### G. Evaluation Type: mixed
R1 synthetic_proxy (Zipf over real embeddings); R2/R4/R5 real_gt; R3 real_gt probe-only (recovery copied
from R2); R6 real_gt with oracle teacher-forced prefix + 400-position subset.

## Claim Impact
- claim:restore-correlation — supported (partial), R5 carries single-seed qualifier.
- claim:bayes-gap-diagnosis — supported (partial); L0 directly, depth directionally.

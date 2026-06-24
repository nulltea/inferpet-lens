# Experiment Audit Report — resid-depth-inversion (Task 4)

**Date**: 2026-06-24
**Auditor**: Codex gpt-5.5, xhigh reasoning (cross-model, read-only)
**Project**: transformer-attacks-lens · campaign-B-expand
**Thread**: 019ef7fe-7000-7741-a8a5-1b12b35b9b26 (2 rounds: initial FAIL on reporting → fixes → WARN)

## Overall Verdict: WARN
## Integrity Status: warn (scope only — no fraud patterns present)

## Checks

### A. Ground Truth Provenance: PASS
GT token ids are tokenizer-derived input labels (`capture.py:159`, `types.py:121`), not model
outputs. Vocab-disjoint split is real — distinct token ids partitioned train/val/test
(`splits.py:67`). Shuffle is a genuine label permutation (`_inversion.py:60`).

### B. Score Normalization: PASS
No metric divided by model-output statistics. Cosine retrieval normalizes embeddings
(`ridge.py:107`); top-1 is mean correctness (`ridge.py:116`); selectivity = raw real − shuffle
(`depth_inversion_sweep.py:119`). Raw + shuffle floors both reported.

### C. Result File Existence: PASS (fixed this round)
Headline numbers match JSON: L32 ridge 0.3995/sel 0.3898, mlp2 0.5424; Spearman cap +0.85, CLUB
+0.78; bootstrap CIs present. `run.exit`=0. **Fixed**: header now states the inverter split
correctly (9469 positions/layer captured; n_train 3373 / n_test 413; CIs over the 413 test rows/layer)
— the earlier "~7.5k test rows" misstatement is removed; tracker R001–R004 marked DONE.

### D. Probe-vs-Attack Independence: PASS (CRITICAL for this project)
cap-PVI maps token ids to CLASSES and trains a PCA-softmax reader (`vinfo_capacity.py:162,204,218`)
— it does NOT use the attack's embedding-table retrieval. CLUB is a separate variational MI
estimator (`club.py:101,175,189`), not the inverter's loss. Caveat: cap-PVI uses a row split /
shared class set rather than the attack's vocab-disjoint split — noted, not circular.

### E. Scope Assessment: WARN
Single model (Qwen3-4B), single corpus (release-gate-512), single seed. Prose avoids
all-transformer overclaiming and RESULT_TO_CLAIM explicitly scopes the claims as partial. WARN is
the honest-disclosure state, not an overclaim.

### F. Evaluation Type: real_gt
Real input-token ground truth from a real corpus/model capture (`capture.py:159`, `types.py:126`).

## Action Items
- (done) corrected test-row / CI sample-size statement to 413 rows/layer.
- (done) tracker R001–R004 → DONE with artifact pointers.
- (carry to claim) state scope precisely: Qwen3-4B / resid_post / release-gate-512 / sampled depths,
  single seed; CIs over test rows not seeds.

## Claim Impact
- C1 (depth ≠ privacy): supported, scoped.
- DECISION (mlp2 beats ridge at L32): supported (disjoint CIs at L32).
- C2 (probe tracks recovery across depth): supported, scoped; probe independence verified (Check D).

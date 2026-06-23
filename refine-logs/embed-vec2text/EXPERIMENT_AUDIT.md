# Experiment Audit Report — embed-vec2text surface (Task 4)

**Date**: 2026-06-23
**Auditor**: Codex gpt-5.5 xhigh (cross-model, read-only, thread 019ef5ee)
**Project**: transformer-attacks-lens · surface embed-vec2text (Vec2Text + spectral channel-MI)

## Overall Verdict: WARN

## Integrity Status: warn (no fraud; two documentation fixes applied)

## Checks

### A. Ground Truth Provenance: PASS
B9 grades recovered text against corpus-derived `texts`, not model outputs
(spectral_mi_probe_eval.py:83,106). B7/B6c grade token recovery against corpus prompt token IDs;
B7 confirms the post-bug-fix disjoint train/test token split, scoring test positions only
(b7_vec2text_corrector.py:152,179; b6c:93). capPVI is a disclosed self-supervised cluster proxy.

### B. Score Normalization: PASS
All recovery metrics are raw fractions/cosine/exact/token-F1; I_G is raw bits. No metric divided by a
prediction max/min/mean (spectral_mi_probe_eval.py:127; b7:278; b6c:144). Vector norm used only for
cosine/top-k matching.

### C. Result File Existence / Number Match: WARN → fixed
Numbers in logs/claim match the JSONs after rounding (B9 Spearman I_G/CLUB +1.00, capPVI +0.6156, 0
ceiling violations; B7 deltas −0.0335/−0.0402/−0.0251). WARN: the claim's "Spearman=+1.00" held for
token-F1/cos/pos-token-acc only; exact-match is +0.71 (I_G & CLUB) / +0.54 (capPVI).
**Fix applied**: claim evidence chain now states the exact-match qualifier explicitly.

### D. PROBE != ATTACK (integrity-critical): PASS
Confirmed. `spectral_channel_mi` computes eigenvalues from the clean embedding covariance, then
t_i = ½log₂(1+λ/σ²) and sums bits (spectral_channel_mi.py:14,140,159). In B9 the I_G call consumes only
`e0_clip`, `sigma`, and entropy/vocab proxies — NOT recon, token-F1, exact, cosine, CLUB, or capPVI
(spectral_mi_probe_eval.py:109). Geometry-only; not the attack in disguise; the +1.0 correlation is
not circular.

### E. Scope / Disclosure: WARN → fixed
N=96<d=768 rank deficiency disclosed in both log and claim (undersampled d_eff/localization, T4/M3 not
run). B7 genuinely negative (feedback final < no-feedback at every ε). WARN: B7 log said "No teacher
forcing" while code fixes train-token context. **Fix applied**: reworded to "no full-prefix teacher
forcing of test tokens; train-token context held fixed, scored on test positions only."

### F. Evaluation Type:
- Spectral probe unit tests: simulation_only (model-free math).
- B9 recovery: real_gt (corpus texts); capPVI + entropy ceilings: self_supervised_proxy / explicit proxy.
- B7 feedback eval: real_gt token-level corpus labels + simulated Gaussian DP + disjoint pools.
- B6c forward-model: real_gt token labels, teacher-forced-prefix oracle variant + simulated DP.

## Action Items
- [x] Claim: add exact-match Spearman qualifier (+0.71) beside the +1.00 token-F1/cos figure.
- [x] B7 log: reword the teacher-forcing line.

## Claim Impact
- claim:spectral-channel-mi-embedding-inversion — **supported** (integrity_status: warn → fixed). The
  central integrity claim (I_G geometry-only, non-circular) passes.

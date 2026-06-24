# Experiment Audit Report — KV-CLOAK (Task B-2)

**Date**: 2026-06-24
**Auditor**: External reviewer backend, gpt-5.5, xhigh reasoning (cross-model, read-only)
**Project**: transformer-attacks-lens

## Overall Verdict: WARN
## Integrity Status: warn

No fabricated ground truth, no phantom core numbers, no probe/attack circularity, no score
normalization fraud. WARNs are reporting/faithfulness items, none of which invalidate C1/C2.

## Checks

### A. Ground Truth Provenance: WARN
Real captured per-head K (pre `repeat_kv`, `capture.py:51/56/58`); `per_prompt_matrices` keeps
plaintext H and exposed U=transform(H). Recovery graded vs `jd_floor` (real chance floor) with
attacker-favorable Hungarian (`bss.py:156/171/278/365`). WARN: sweep stores only `jd_floor_t1`
(`kv_cloak_sweep.py:129`); RESULTS says JD stays "at the floor across T" without `jd_floor_t4`.
→ FIXED: added `jd_floor_t4` to the L0 JD table; wording grounded.

### B. Score Normalization: PASS
No metric normalized by model output statistics; bits via `/ln2`; M-only gram_error=0.124
feature-subsample caveat disclosed (`RESULTS.md`).

### C. Result File Existence: PASS
273 records; B1/L0-table/correlations/b-flatness/JD all trace to sanity.json + analysis.json.

### D. Dead Code / Probe≠Attack: PASS
All metric functions called (`kv_cloak_sweep.py:103`). Negentropy = whitened-row skew/kurtosis
only (`bss_separability.py:30/73`), no joint-diagonalization → not the demixing in disguise.

### E. Scope: WARN
RESULTS scope correctly bounded and the b-inertness caveat bounds it to b∈{16,32,64} and this
adversary. WARN: `EXPERIMENT_PLAN.md` stale (kqv_out/dev24/b={2,4,8}). → FIXED: plan marked
superseded with a header pointing to the actual run + RESULTS.md.

### F. Faithfulness: WARN
`K' = S·P̂·(K+A)·M` order, orthogonal S/M, per-(seed,prompt,block) one-time-pad P̂ all correct
(`kv_cloak.py`). Stored-K surface appropriate (KV-CLOAK leaves o=kqv_out invariant by design).
WARN: `A` built over the flattened full-width row, not strictly per-head. → FIXED: docstring
corrected to describe the full-row beacon; RESULTS already carries the "stylized beacon" caveat.
Does not affect the M / S·P̂ conclusions.

### F-eval. Evaluation Type: real_gt

## Claim Impact
- **C1** (M is the load-bearing channel; S·P̂/b inert; A→spectrum): supported.
- **C2** (negentropy = between-channel diagnostic, ρ=0.71 aggregate / 0.77 channel-mean; not
  within-channel): supported as narrowed.

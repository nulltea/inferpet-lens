# Experiment Audit Report — perm-cover

**Date**: 2026-06-23
**Auditor**: External reviewer (Codex gpt-5.5, xhigh, read-only) — thread `019ef64a-7557-7f30-aaba-b96e2b0f75fc`
**Project**: transformer-attacks-lens / permutation-cover surface

## Overall Verdict: WARN
## Integrity Status: warn

No fabrication, no synthetic ground truth, no score-normalization fraud. The one material issue is
**interpretive** and is corrected below: the standardized writeup originally over-labeled
retrieval-PVI as an *independent* probe when source code + the spike's own comments mark it as the
*dependent* "VMA-in-bits" measure.

## Checks

### A. Ground Truth Provenance: PASS
VMA recovery is graded against the **stored secret permutation τ** (`aloepri.py:134` `tau =
rng.permutation(n)`, `:136` `obf[tau] = transformed`, `:137` returned as `perm`), scored via
`pair.inverse_perm()` (`vma.py:63`) / `np.argsort(perm)` (`vma_stronger.py:61`). Real cached model
weights, obfuscated synthetically (`model_id="aloepri-synthetic"`). Not attack-derived.

### B. Score Normalization: PASS
No metric divided by its own max/min. VMA = direct fraction-correct mean. `α_e=0 → 1.0` is
legitimate (zero noise + the sorted-quantile signature is column-permutation-invariant by
construction, `features.py:30`). Bits are raw estimates; the keymat CLUB `−2.4` is not clipped.

### C. Result File Existence: PASS
All cited numbers exist and match (rounded): uplift `+0.434 @α=0.2`, `+0.600 @α=0.35`
(`vma_stronger.json:34,42`); Spearman CLUB `+0.976`, retr-PVI `+1.00` (`aloepri_vma_sweep.json:18`);
`rs_vs_club_all 1.0`, `fe_vs_club_all 0.991` (`vma_stronger.json:74`); keymat VMA `0.0`, CLUB
`−2.3937` (`aloepri_vma_sweep.json:87`); cover_break p95 `0.917@L0` (`fullcheck-L0-10.json:286`).

### D. Dead Code Detection: WARN
`cover_break` `fastica_anchor` branch raises `NotImplementedError` (`cover_break.py:81`); only the
ridge variant runs. Disclosed as deferred in the writeup — acceptable.

### E. Scope Assessment: WARN
Single cached embedding table; R1 is 3-seed (N=1000), R2 is 1-seed (N=1200), both disclosed. **R3
(`fullcheck-L0-10.json`) is Qwen3-4B activations** (`model_id Qwen/Qwen3-4B`), a *different surface*
— it is the plaintext anchor baseline, not a matched AloePri cover-break.

### F. Evaluation Type: real_gt
Controlled real-weight-table exact secret-permutation recovery.

## Probe ≠ Attack Verdict (integrity-critical)
- **CLUB-on-φ — INDEPENDENT (the clean probe).** Runs no Hungarian assignment and no nearest-neighbor
  matching; uses τ only to form positive paired samples, then estimates MI with a variational CLUB
  network (`measures.py:45`, `club.py:101`).
- **retrieval-PVI — DEPENDENT ("VMA-in-bits").** Uses plaintext signatures as the candidate table,
  obfuscated signatures as queries, fits an inverter, and scores the true candidate under
  cosine-softmax retrieval (`measures.py:61`, `vinfo.py:165`, `_retrieval.py:44`). The spike
  `aloepri_vma_sweep.py:12` itself labels it "dependent / the VMA in bits". **Its ρ=1.0 with VMA is
  partially circular and must not be cited as independent confirmation.**

## Action Items (APPLIED)
1. Demote retrieval-PVI to a *dependent reference (VMA-in-bits)* in `RESULTS_STANDARDIZED.md` and the
   claim — done.
2. Thesis-confirmation rests on **CLUB-on-φ only** (ρ +0.976 over the α_e sweep; +1.00/+0.99
   recorrelation in `vma_stronger.json`) — done.
3. Clarify R3 cover_break is Qwen3-4B activations, a separate plaintext-baseline surface — done.

## Claim Impact
> Superseded by the proof audit (`PROOF_AUDIT.md`, verdict PASS): the theory part below is now
> **verified**, not pending. Left here for the round trail.
- `perm-llr-threshold` empirical (full-sort ≫ RowSort-64): **supported**.
- `perm-llr-threshold` theory (per-row maximal-invariant profile-MLE + DPI domination; 2 log n cited):
  **verified** (proof-checker PASS, thread `019ef653`).
- thesis-confirmation: **needs qualifier** — rests on the independent CLUB-on-φ; retrieval-PVI is the
  attack-in-bits; bidirectional language softened to "faithful monotone indicator + keymat negative
  control collapses probe and recovery together".

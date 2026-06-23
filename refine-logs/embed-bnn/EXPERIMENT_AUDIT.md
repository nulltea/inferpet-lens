# Experiment Audit Report — embed-bnn (BNN error bounds)

**Date**: 2026-06-23
**Auditor**: Codex `gpt-5.5` xhigh (cross-model, read-only, thread `019ef60a`)
**Project**: transformer-attacks-lens

## Overall Verdict: WARN
## Integrity Status: warn

No evidence of attack-observation leakage, ground-truth reuse, or prediction-stat
normalization fraud. Both WARNs are reporting/interpretation hygiene, not integrity failures.

## Checks

### A. Ground Truth Provenance: PASS
BNN ground truth is the real channel label index `idx` (`true = np.repeat(idx, M)`), compared
against NN predictions — not derived from probe outputs. Fano (`seed+1`) and BNN measurement
(`seed+2`) draw separate noise. (`scripts/spikes/bnn_error_bounds_validation.py:64,88,90,99,189`)

### B. Score Normalization: WARN
No metric divided by max/min/mean of the model's own predictions. BNN error is a raw
misclassification rate; bounds are analytic/MC quantities normalized by `K` or `log₂(K−1)`.
WARN: the spike records `p_e_ub_raw` but omits `p_e_lb_raw` / `p_e_lb_lcb_raw` although the
measure returns them. Completeness gap only — not fraud. (Not re-running GPU to add a field;
the clamped `p_e_lb` + `p_e_lb_lcb` are persisted and sufficient for the bracketing claim.)

### C. Result File Existence & Number Match: PASS
Dense JSON matches the experiment log: `pool_size=2048`, `n_inside=10`, `rho_ub_bnn=0.887625…`,
`rho_lb_bnn=0.937436…`, `floor_epsilon=16.0`. Tracker status `DONE`. Side note: coarse
`results/bnn_error_bounds_validation.json` has `rho_lb_bnn: NaN` (degenerate BNN~0 range, documented).

### D. Dead Code: PASS
`union_bhattacharyya` and `fano_equivocation` are imported, called in the ε loop, and their
returned fields are written to the JSON records.

### E. Scope Assessment: WARN
Actual scope: one model/table (`unsloth/gemma-2-2b`), one pool, one seed, uniform prior, L0
channel, 5 coarse + 10 dense ε points. Docs mostly state this clearly and explicitly reject
universal claims. WARN: the C2 morphology wording said top pairs are "exactly case/space/number
neighbours," but the dense JSON's closest pairs include rare Unicode/PUA glyph tokens (`'𞤥'`,
`'⸏'`, `''`, `' vooz'`) *before* the clean case/space examples. **Fixed**: C2 wording in
the experiment log now states the closest pairs are dominated by rare glyph tokens, with
case/space/number twins the recognizable subset.

### F. Attack-Independence (THE KEY CHECK, probe≠attack): PASS
The bounds path consumes only `pool_clip`, `sigma`, precomputed codebook distances, and Fano's
own internal RNG. BNN observations are generated only afterward in `_bnn_uniform_error`. Fano
creates fresh synthetic noise internally. `tests/test_channel_error_bounds.py::test_no_observation_argument`
asserts no observation/label parameter exists. (Blacklist test — an exact-signature whitelist
would be stricter; recorded as a future hardening, but the current path is provably attack-independent.)
(`src/talens/measures/channel_error_bounds.py:68,124,179,192`)

## Action Items
- [done] Fix C2 morphology overstatement in the experiment log.
- [deferred, non-blocking] Persist `p_e_lb_raw` if the JSON is ever regenerated.
- [deferred, non-blocking] Tighten `test_no_observation_argument` to an exact-signature whitelist.

## Claim Impact
- claim:bnn-error-bounds-bhattacharyya-fano: **supported** (scoped: uniform-prior, L0, one table);
  C2 wording qualified. Attack-independence (the load-bearing property) PASS.

---
type: dev-log
status: current
created: 2026-06-23
updated: 2026-06-23
tags: [metric, reporting, bits, readout, infrastructure, shakedown]
companion: docs/html/metric-std.html
---

# Metric-standardization reporting layer (Block A, Task 1 — harness shakedown)

## Objective

Build/extend the `src/talens` reporting layer so **every** probe emits the campaign's
canonical metric: **bits canonical + a per-secret human readout** (perplexity / token-F1 /
recovery-rate / cosine / AUC), and retrofit the existing measures onto it. Also the supervised
shakedown of the autonomous campaign harness (gate paths, ARIS output locations, GPU teardown).

## What was built

`src/talens/report.py` (numpy-only, runs in the host CPU venv):

1. **`canonical_bits(measure, result) → (bits, bits_kind)`** — the single place that extracts the
   comparable bits scalar from each measure's heterogeneous output dict. Registry covers all eight
   families:

   | measure registry name      | dict key pulled                       | `bits_kind`                |
   |-----------------------------|---------------------------------------|----------------------------|
   | `v_information`             | `v_information_bits`                  | `v_info`                   |
   | `v_information_retrieval`   | `v_information_bits`                  | `v_info`                   |
   | `v_information_capacity`    | `v_information_bits`                  | `capacity_v_info`          |
   | `club`                      | `club_mi_bits`                        | `mi_upper_bound`           |
   | `mdl`                       | `surplus_description_length_bits`     | `sdl`                      |
   | `pid`                       | `i_joint_bits`                        | `pid_total_mi`             |
   | `spectral_channel_mi`       | `i_g_bits`                            | `channel_mi_upper_bound`   |
   | `fano_equivocation`         | `i_channel_bits`                      | `channel_mi`               |

   The `bits_kind` tag keeps a mixed table interpretable (an MI upper bound, an SDL, a channel-MI
   *ceiling*, and a Fano channel-MI *point estimate* are all "bits" but are NOT the same quantity).
   A declined measure (`{"v_information_bits": None}`, or `fano_equivocation` at `K<3`) maps to
   `bits=None`, not 0.

   **Error-bounds retrofit (the 8th family).** The geometry-only BNN/MAP error-bound probe
   (`channel_error_bounds`) is the one measure whose native output is an *error probability*, not
   bits. To bring it onto the canonical scale honestly: `fano_equivocation` now also emits
   `i_channel_bits = log₂K − H(V|Y)` (uniform-prior channel MI; `Ĥ_M` is unbiased for `H(V|Y)`, so
   this is an unbiased estimate of `I(V;Y)` — sign-consistent leakage, *not* the equivocation
   dual). `LeakageReport.from_error_bounds(fano, ub)` pairs the two functions into one row: bits =
   the Fano channel-MI estimate; readout = the recovery band `[1−P_e^ub, 1−P_e^lb]` bracketed by
   the union-Bhattacharyya error upper bound and the Fano error lower bound. Both halves are
   functions of codebook geometry + σ only → the row never reports the attack's own recovery.

2. **`format_bits(x)`** — the **legibility fix** CLAUDE.md mandates ("fix the 1/100-of-a-bit
   illegibility in the readout, not the stored value"). Adaptive precision: ≥0.1 bit → 3 sig figs
   `"X.XX bits"`; below 0.1 bit → millibits `"X mbit"` with adaptive decimals so a genuine 0.01-bit
   leak renders `"10.0 mbit"`, never `"0.00 bits"`. The stored float is **untouched** (verified by
   test). `∞`/`−∞`/`None` handled.

3. **Per-secret `Readout` builders**, one per secret kind, each carrying the graded primary
   recovery scalar + the human-readable fields:
   - `token_id_readout` — top-1 recovery (primary) + top-k + perplexity (`2^H`)
   - `text_readout` — token-F1 (primary) + ROUGE-L
   - `permutation_readout` — recovery-rate (primary) + Kendall-τ
   - `embedding_readout` — cosine (primary) + decoded token-F1
   - `membership_readout` — AUC (primary)
   - `error_band_readout` — token-id recovery *ceiling* `1−P_e^lb` (primary) + recovery floor
     `1−P_e^ub` and both raw MAP error bounds (the two-sided geometry-only BNN probe)

   Plus `token_f1(pred_ids, gt_ids)` (multiset F1) and `perplexity_from_bits(H)` helpers.

4. **`LeakageReport`** — one standardized row: `bits` (verbatim) + `bits_kind` + paired `Readout`
   + sweep context (`surface`, `layer`, `transform`, `sigma`). `from_measure(...)` builds it by
   calling `canonical_bits` and attaching the (already-computed, attack-side) readout; `.render()`
   prints **both axes** on one line; `.to_dict()` is JSON-serializable with a `bits_legible` field.

Also exported `spectral_channel_mi` from `talens.measures.__init__` (was reachable only via its
submodule) and surfaced the whole reporting API from `talens.__init__`.

## Validation

- `tests/test_report.py` — **30 tests**, all green: legibility contract (nonzero never renders 0;
  stored value unmutated), `canonical_bits` against the **real** output schema of every measure
  (incl. the `fano_equivocation` channel-MI point estimate, distinct from the spectral upper
  bound), each readout builder (incl. `error_band_readout`), the two-sided `from_error_bounds` row,
  declined/unknown handling, end-to-end `LeakageReport` + JSON round-trip.
- Full model-free suite: **128 passed** (host CPU venv) — no regressions.
- **Edge-case honesty (round-2 review fixes).** `canonical_bits` now *raises* on a missing
  canonical key (schema drift) instead of masking it as a declined `None`; declined measures carry
  the key explicitly (mdl's too-few-rows paths now set `surplus_description_length_bits: None`). The
  σ=0 Fano channel-MI is computed from the duplicate-group sizes (`I = log₂K − Σ_g (n_g/K)log₂n_g`),
  so it never overstates leakage as `log₂K` when codewords collide after clipping (verified:
  distinct→`log₂8=3`; 4 groups of 2→`I=2, H=1`). The paired `union_bhattacharyya` σ=0 branch is
  likewise collision-honest: it returns the exact deterministic MAP error `(K−G)/K` (G distinct
  groups) instead of a false 0, so the rendered recovery floor is `1−(K−G)/K`, not 1 (regression
  tests in `tests/test_channel_error_bounds.py`). `from_error_bounds` suppresses the misleading
  `recovery_ceiling=1` for a declined (K<3) Fano row and carries the `note`/SE/raw bounds into `extra`.
- **Live retrofit smoke** (real `fano_equivocation` + `union_bhattacharyya`, 64×16 pool, σ=1, CPU
  numpy fallback): `fano_equivocation[channel_mi]=5.14 bits | recovery_ceiling=1 (recovery_floor=0.533, …)`;
  verified the identity `I(V;Y)+H(V|Y)=log₂K` to 1e-6.
- **End-to-end retrofit smoke** on real measure outputs (no GPU, synthetic arrays):
  - `v_information` → `v_information[v_info]=-22 mbit | token_top1_recovery_rate=0.55 (top10=0.8, perplexity=4.05)`
    — the small negative PVI renders as millibits, not `0.00` (legibility fix exercised on live output).
  - `spectral_channel_mi` → `spectral_channel_mi[channel_mi_upper_bound]=48.5 bits | cosine=0.91 (token_f1=0.42)`
    — bits stored verbatim (`48.53125…`).

## Integrity note (probe ≠ attack)

The reporting layer is **rendering only**. It stores the measure's bits verbatim and attaches the
attack's recovery readout as a *separate* paired field — it never derives bits from the attack or
vice versa, so it cannot manufacture a correlation. `bits_kind` prevents conflating distinct
quantities (MI-UB vs SDL vs channel-MI) on the shared bits axis.

## Scope / non-claims

This is **infrastructure**, not a leakage finding — there is no research-wiki Claim node here.
Downstream surface tasks (2–7) consume this layer to render their `(bits, recovery)` sweep tables
in the mandated convention. Shakedown outcome (harness): gate path
`refine-logs/metric-std/REVIEW_STATE.json`, ARIS output locations, and Codex auth exercised by the
review skills in this phase.

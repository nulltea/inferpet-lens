# Queued onto Task 7 (leakage–utility, the single GPU phase) — from Task 6 (readout-metrics)

**Result of the Task 6 audit: no NEW model-required readout gaps.**

Every bits column on `synthesis.html` and the per-surface / per-probe pages already carries a
same-row per-secret recovery readout, sourced from on-disk run JSON (source-order 1) or already
recomputable model-free by `talens.report` (source-order 2). The full table-by-table verification is
in `refine-logs/readout-metrics/AUDIT.md`. There were therefore no source-order-(3) items —
readouts that genuinely require loading the model and are not stored — to emit.

## What Task 7 still owes (already queued elsewhere — listed here only so Task 7 sees one merged view)

These are **not** Task-6 readout gaps; they are the plaintext-reference baselines that Task 4
already queued. Repeated here so Task 7 has a single consolidated worklist and does not load the
model twice.

| item | symbol | owner queue | backfill target |
|---|---|---|---|
| plaintext SDL selectivity per layer across depth | `SDL` | `refine-logs/probe-pages/queued-for-utility.md` (Task 4) | `probe-sdl.html` §04 |
| shared spectral capacity per layer × channel kind, clean KV | shared-spectral-capacity | `refine-logs/probe-pages/queued-for-utility.md` (Task 4) | `probe-shared-spectral-capacity.html` §04 |

## Model-required precision debt (not a readout gap)

| item | symbol | what to emit | why deferred | backfill target |
|---|---|---|---|---|
| full-precision BNN equivocation `H(V|Y)` for the low-noise rows | `H(V|Y)` | recompute/export `H(V|Y)` to ≥4 dp for the rows currently source-rounded to `0.00` (ε≈96–128) | the on-disk source (`refine-logs/bnn-error-bounds/EXPERIMENT_RESULTS.md`) prints these at 2 dp; finer precision requires the gemma-2-2b embedding table = a model load = GPU, forbidden in this no-GPU phase | `bnn-attack.html` / `synthesis.html` bnn `H(V|Y)` column (currently displayed as `≈0`) |

This is **not** a missing per-secret recovery readout — the BNN error-band readout is already paired
in those rows. It is a display-precision gap: the rows now show `≈0` (honest negligible-equivocation
marker, explicitly not an asserted exact zero). When Task 7 loads the model it can export full-precision
`H(V|Y)` so the millibit branch of the legibility contract can render the true sub-0.1-bit value.

## Note on perplexity / top-k token-id readouts

The token-id-secret tables (V_cap reader-accuracy, depth-inversion selectivity, CLUB DP sweeps) use
**recovery-rate / accuracy / selectivity** as the primary readout, which is sufficient under the
metric convention (the *primary* graded recovery scalar). A predictive-distribution **perplexity /
top-k** companion would be a richer optional readout but requires the model's output distribution
(source-order 3). It is **not required** for any bits column to be legible — all are already
paired — so it is explicitly **not** queued. If Task 7 emits the clean-model captures above and the
per-token predictive distribution is cheap to dump in the same pass, perplexity could be added to
the token-id rows as an optional enhancement, but no page is blocked on it.
</content>


---

**RESOLVED by Task 7 (c7-utility) 2026-06-25.** All queued items emitted/backfilled; see `refine-logs/utility-tradeoff/queue_results.json`. SDL + shared-spectral-capacity plaintext baselines were emitted CPU-only from clean Qwen3 captures already on disk (no GPU); BNN H(V|Y) was a display-precision backfill from `results/bnn_error_bounds_validation_dense.json` (full-precision values were already stored). Probe pages `probe-sdl.html` / `probe-shared-spectral-capacity.html` §04 and `bnn-attack.html` H(V|Y) column backfilled.

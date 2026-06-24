# Task 6 — readout-metrics audit (no-GPU render pass)

**Phase:** c6-readout-metrics · recipe `consolidate` · gpu false · 2026-06-24
**Objective:** every bits column on `synthesis.html` and the per-surface/per-probe pages must
carry a paired per-secret human readout (perplexity / token-F1 / cosine / recovery-rate / AUC /
error-band / margin / accuracy); the millibit / `n/a`-vs-measured-zero legibility contract from
`src/talens/report.py` must hold. No GPU; source readouts from on-disk run JSON (1) or recompute
model-free via `talens.report` (2); queue genuinely model-required ones (3) to Task 7.

## Method

Grepped every `<th>` whose header text contains "bit" across all 22 `docs/html/*.html`, then
inspected each bits-bearing `.spec` table for (a) a same-row recovery-readout column, (b) any bits
*value* rendered as a bare `0.00`/sub-0.1 decimal that could hide a real leak, (c) `—`/`n/a` cells
distinct from measured zero. Cross-checked the suspicious near-zero cells against their source
files. Three parallel read-only sweeps (residual / embedding+KV+defenses / probe pages) plus a
direct read of the densest page (`synthesis.html` §02–§03) and the JS-populated `bnn-attack` table.

## Finding — the pairing already holds (prior standardization passes)

The task's stated problem ("most surface tables still show bare bits") is **stale**: the earlier
consolidate/standardization passes (Block A/B surface pages, Task 4 probe pages) already attached a
same-row recovery readout to every bits column. Table-by-table verdict:

| page | bits-bearing table(s) | paired recovery readout (same row) | verdict |
|---|---|---|---|
| synthesis.html | §02 overview + 14 §03 sweep tables | probe symbols in §02 (no raw bits); §03 each bits col paired with recovery / token-F1 / V_cap acc / selectivity / error-band | PASS |
| resid-rep2text | noise σ table; **length table** | noise: I_G ↔ real F1; **length: recovery shown, constant I_G was prose-only → FIXED (added constant `I_G=2856` column)** | FIXED |
| resid-gelo | κ(A) table | J (bits) ↔ genuine margin | PASS |
| resid-split | (correlations only) | — (no raw bits column) | PASS |
| resid-dp-attacks | R1/R2/R7 sweep tables | CLUB/V_cap/reader-bits ↔ ridge/Bayes/decoder recovery, selectivity | PASS |
| resid-depth-inversion | depth grid | MI upper-bound (bits) ↔ ridge/MLP selectivity, reader acc | PASS |
| vec2text | DP calibration | I_G (bits) ↔ token-F1 / exact / cos | PASS |
| embed-sgt | budget table | budget (bits) = swept param ↔ token-F1 by shape | PASS |
| bnn-attack | dense ε sweep (JS `RES`) | H(V|Y) bits ↔ BNN err / lower / upper (error-band readout) | PASS |
| kv-cloak | channel ablation | J (bits) ↔ reconstruction p95 cosine | PASS |
| kv-accumulation | margin-by-kind | J (bits) ↔ genuine margin | PASS |
| perm-cover | noise + converse tables | channel MI (bits) ↔ RowSort/full recovery, gain | PASS |
| probe-ig/club/j/vcap/bhattacharyya-fano | §04 plaintext-reference | each pairs its bits with recovery / accuracy / margin / error | PASS |
| probe-sdl, probe-shared-spectral-capacity | (no plaintext table) | explicit "queued onto Task 7" placeholder (from Task 4) | PASS (queued) |
| index.html | zero-leakage reference | Leakage = "~0 by construction" (qualitative, no sweep) | N/A |

### One legitimately-exempt table
`probe-vcap.html` line 194 — the **estimator-repair calibration** table (`real information (bits)` /
`shuffle floor (bits)` / `cost vs original`). Bits *is* the subject here (it documents the
−49.7 → −1.9-bit shuffle-floor repair across reader families); it is not a leakage sweep and has no
"recovery" to pair. The page's §04 plaintext-reference table *does* pair `V_cap reader acc` with
`ridge recovery`. Exempt, correctly.

## Finding — the legibility contract (one millibit fix applied)

- **The `bnn` equivocation column needed the contract applied — now fixed.** `H(V|Y)` at ε=80 is
  0.03 bits, which falls in the open `(0, 0.1)` interval and was rendering as the bare `0.03`. Under
  `format_bits` it must render as **`30 mbit`**. The ε=128 (and ε=96 in the dense table) equivocation
  comes from a 2-dp source that rounds to zero; it is a genuine *negligible* equivocation (full
  determination at low noise; ε=128 BNN err = 0.0001) but cannot be asserted as a mathematical exact
  zero, so it now renders **`≈0`** — distinct from `n/a` and from an asserted `0 bits`. Applied to
  both `synthesis.html` (static cells) and `bnn-attack.html` (a `fmtBits` JS helper mirroring
  `format_bits`, replacing the old `toFixed(2)` that collapsed sub-0.005 to `0.00`). This corrects an
  earlier draft of this audit that wrongly claimed "no bits value falls in (0, 0.1)".
- **No *other* sub-0.1 *bits* value is collapsed.** The remaining static `0.00`/`0.000` cells are
  either a recovery/param-axis value that is genuinely zero (cosine-NN / nearest-neighbour
  **memorization floors** under the vocab-disjoint split; exact-match = 0 at the 0-step base), or an
  error *probability* (bhattacharyya-fano bounds, to 4 dp) — neither is a *bits* value. Sub-1-bit
  *leakage* bits that exist (e.g. kv-cloak `J` = 0.72, 1.47) are above the 0.1 millibit threshold and
  render correctly as bare decimals.
- **`n/a` is distinct from measured zero.** `report.py` renders a `None` bits value as the string
  `"n/a"`; the HTML tables use the em-dash glyph `—` as that n/a marker (e.g. resid-dp ε=64
  CLUB/V_cap; vec2text base-step I_G). Measured zeros render `0`/`0.00`/`0.000`, and a negligible
  near-zero renders `≈0`. All three are visually distinct from `—`.

## Edits made

1. `docs/html/resid-rep2text.html` — added the constant `capacity I_G (bits)` column (value 2856,
   every row) to the **length** table, so the vacuous-capacity signature (bits flat at 2856 while
   recovery climbs 0.076 → 0.244) is legible *in the table*, not prose-only. Harmonizes with the
   synthesis §03 row that already carries this column. The page prose already explains the
   constant-capacity / undefined-rank-correlation reading.
2. `docs/html/bnn-attack.html` — added a `fmtBits` JS helper implementing the **relevant BNN display
   branch** of `format_bits` (the `≥0.1` bare-decimal branch and the sub-0.1 millibit branch over the
   fixed positive `H(V|Y)` domain of this static table; it does not reproduce the Python function's
   `None`/non-finite/sign/`<1 mbit` two-decimal branches, which this data never hits) and applied it
   to the `H(V|Y)` column, replacing `toFixed(2)`: sub-0.1-bit equivocation now renders as millibits
   (ε=80 → `30 mbit`) and a 2-dp-source value that rounds to zero renders `≈0`, never a misleading
   `0.00`.
   - HTML table unit convention: bit units live in the column header `(bits)` except for millibit
     values, where the unit is printed inline (`30 mbit`) so the scale change is unmissable. This is
     a rendering convention on top of `format_bits`, not a deviation from the stored value.
   - **Precision debt (queued to Task 7):** the ε≈96–128 rows are source-rounded to `0.00` (2-dp
     on-disk source) and shown as `≈0`; full-precision `H(V|Y)` needs a model load (GPU), so it is
     logged in `queued-for-utility.md` as a display-precision gap, not a readout gap.
3. `docs/html/synthesis.html` — same contract applied to the static `H(V|Y)` cells of the §03 bnn
   table (ε=80 → `30 mbit`, ε=128 → `≈0`).
4. `docs/html/probe-vcap.html` — added an in-page label to the estimator-repair table declaring it a
   calibration diagnostic where bits is the subject and there is no per-secret recovery target to
   pair, with a pointer to the §04 plaintext-reference table that carries the recovery readout. This
   makes the one bits-without-recovery table's exemption explicit on the page, not only in this audit.

No other edits: every remaining bits column already carried a same-row readout and (after the bnn
fix) satisfies the legibility contract.

## Source-order routing

- (1) on-disk JSON / (2) model-free `talens.report`: all present readouts come from these — no
  recompute was required because the tables were already standardized.
- (3) genuinely model-required: **none new.** See `queued-for-utility.md`.
</content>
</invoke>

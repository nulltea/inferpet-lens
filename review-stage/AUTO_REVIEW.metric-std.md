# Auto Review Loop — metric-std (Block A, Task 1: reporting-layer shakedown)

Reviewer backend: Codex (gpt-5.5, xhigh) · threadId `019ef5b1-c3e8-7841-be24-636cca2720b2`
Deliverable: the metric-standardization reporting layer — bits canonical + per-secret readout,
retrofitting the 8th measure family (geometry-only BNN/MAP error-bound probe) onto the layer.
Scope: infrastructure, no research Claim node — judged on correctness / integrity / faithfulness.

## Round 1 (2026-06-23)

### Assessment
- Score: 8/10 · Verdict: Almost
- The layer is genuinely rendering/pairing-only (bits from measure dict, recovery from attack —
  never derived from each other). `i_channel_bits = log₂K − H(V|Y)` is honest and sign-consistent
  on the normal path; `channel_mi` (point estimate) vs `channel_mi_upper_bound` (spectral converse)
  is a real, non-misleading distinction. Error-band readout semantics correct.

### Reviewer Raw Response
<details><summary>Round 1</summary>

1. High: `sigma <= 0` overstates `i_channel_bits` if codewords collide after clipping — returns
   `log2(K)` unconditionally; with duplicates deterministic MI is `H(Y)`, not `H(V)`.
2. Medium-high: missing result keys silently treated as declined (`result.get(key)`) — schema drift
   becomes `bits=None`; should raise, reserving explicit `None` for real declines.
3. Medium: declined/edge context not propagated into `from_error_bounds`; a declined Fano row can
   render `recovery_ceiling=1` without saying it is vacuous.
4. Low: HTML subtitle still says "all seven measure families".

Verified: rendering-only property holds; bits convention honest on normal path; readout band
semantics correct; identity I+H=log₂K holds. Ran tests: 28 report, 124 suite. Score 8/10, Almost.
</details>

### Actions Taken
- σ≤0 path now computes deterministic-channel MI from duplicate-group sizes (`H(V|Y)=Σ_g (n_g/K)log₂n_g`).
- `canonical_bits` raises `KeyError` on a missing canonical key; mdl declined paths carry the key as `None`.
- `from_error_bounds` suppresses the misleading ceiling for a declined Fano row and carries note/SE/raw into `extra`.
- HTML subtitle seven→eight; counts → 30 report tests / 126 suite.
- New tests: schema-drift-raises, declined-Fano-no-misleading-ceiling.

## Round 2 (2026-06-23)

### Assessment
- Score: 8.5/10 · Verdict: Almost
- Bits-side fixes solid; but the **paired UB (union_bhattacharyya) σ=0 branch still returned
  `p_e_ub=0` under collision**, so the recovery floor rendered as 1 — false for deterministic ambiguity.

### Reviewer Raw Response
<details><summary>Round 2</summary>

1. High: `union_bhattacharyya` still lies for `sigma<=0` under collision (`p_e_ub=0` with duplicate
   rows → recovery floor renders as 1). Fix: deterministic MAP error `1 − G/K`, or decline UB when
   `min_dist==0`. Add a regression test.
2. Low: Fano docstring stale (`sigma=0 ⇒ H(V|Y)=0`, now false under collisions).
3. Low: collision `i_channel_bits` branch not directly covered by a committed test.

Verified: `canonical_bits` raises on missing keys + preserves explicit None; declined Fano no longer
renders fake `recovery_ceiling=1`; HTML "seven" fixed; counts match. Ran 38 report+bounds, 126 suite.
Score 8.5/10, Almost.
</details>

### Actions Taken
- `union_bhattacharyya` σ≤0 branch returns exact deterministic MAP error `(K−G)/K` (both forms) — 0 if distinct.
- Fano docstring updated (distinct vs duplicate-group equivocation; mentions `i_channel_bits`).
- Two regression tests in `tests/test_channel_error_bounds.py` (distinct full-leakage; collision honesty on both axes).
- Counts → 128 suite; collision-honest UB documented on page + log.

## Round 3 (2026-06-23) — FINAL

### Assessment
- Score: 9.5/10 · Verdict: **Ready**
- No remaining correctness/integrity blockers. σ=0 collision path honest on both axes (Fano
  `i_channel=2, H=1`; UB `p_e_ub=0.5` → recovery floor 0.5). Rendering/pairing-only property intact;
  missing canonical keys raise; docs/log match code and counts (40 report+bounds, 128 suite).

### Reviewer Raw Response
<details><summary>Round 3</summary>

Score 9.5/10. Verdict: Ready. No remaining correctness or integrity blockers. The sigma=0 collision
path is honest on both axes; the reporting layer remains rendering/pairing-only; missing canonical
keys raise instead of being masked; docs/log match. Tests: 40 report+bounds, 128 suite. Only
non-blocking polish: union_bhattacharyya docstring could explicitly mention the sigma=0 collision
override (not a readiness issue).
</details>

### Status
- STOPPED — positive assessment (score 9.5 ≥ 6 AND verdict "ready"). The non-blocking docstring
  polish was applied post-verdict.

## Method Description

`talens.report` standardizes every leakage probe onto the campaign metric — **bits canonical +
per-secret readout**. `canonical_bits(measure, result)` extracts the one comparable bits scalar
(tagged by `bits_kind`) from each measure's heterogeneous output dict for all 8 families
(v_information / capacity / retrieval, club, mdl, pid, spectral_channel_mi, and the geometry-only
BNN error-bound family `fano_equivocation`). `format_bits` keeps sub-0.1-bit leakage legible
(millibits). Per-secret `Readout` builders render recovery in the secret's native units. The
two-sided BNN error-bound probe is folded in via `LeakageReport.from_error_bounds(fano, ub)`: bits =
Fano-derived channel-MI `I(V;Y)=log₂K−H(V|Y)`, readout = recovery band `[1−P_e^ub, 1−P_e^lb]`. The
layer is rendering/pairing-only — bits from the probe dict, recovery from the attack — so it cannot
manufacture a correlation (probe ≠ attack).

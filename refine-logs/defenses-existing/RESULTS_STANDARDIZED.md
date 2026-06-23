# defenses-existing — consolidated standardized results (bits canonical + per-secret readout)

**Surface:** the two *implemented* defenses in `scripts/defenses/`, read as **defenses** (parameter
sweeps), not as the channel-attack studies already consolidated elsewhere.

- **Shredder static-Laplace** (`scripts/defenses/shredder.py`; Mireshghallah et al., ASPLOS'20,
  `1905.11814`) — a **lossy, additive** split-inference cover: i.i.d. Laplace(0, *b*) injected at the
  observed activation. Privacy knob = scale **b** (larger b → lower SNR → more accuracy loss).
  Data: `results/b4_cross_scheme.json` (gemma-2-2b; b∈{0, 0.109, 0.218, 0.381, 0.545, 0.817} ×
  layers {0,5,12,20} × 3 seeds; 72 records. n_prompts=192 is the spike default, not stored in the JSON).
- **AloePri** (`scripts/defenses/aloepri.py`; `2603.01499`) — a **static, invertible**
  obfuscation. Two regimes: *perm-core* (row+column permutation + Gaussian noise α_e — the
  VMA-vulnerable regime) and *keymat* (the dense invertible change-of-basis `W̃ = Π(W+α_eE)P̂`).
  Data: `results/aloepri_vma_sweep.json` (gemma-2 embed table, d=2304, N=1200; **single seed**;
  the perm-core arm is the α_e sweep, the **keymat arm is a single α_e=0 config** — no seed/parameter
  sweep there).

**Metric convention (CLAUDE.md):** canonical **bits** = CLUB / retrieval-PVI from the spike,
re-extracted via `LeakageReport.from_measure(...)`; per-secret **readout** rendered beside it
(permutation → recovery-rate; token-id → top-1 recovery; embedding → cosine). Numbers verbatim from
the provenance JSONs; rows below were re-rendered through `src/talens/report.py`.

> **Probe ≠ attack.** Within each defense the bits column is the attack-*independent* CLUB (and the
> labeled "VMA-in-bits" retrieval-PVI reference, never as independent confirmation). The recovery
> columns are the attacks (VMA / ridge-inversion). The cross-scheme analysis correlates the two
> *across* the sweep — it is not the attack reporting its own bits.

---

## R1 — Shredder is **channel-selective**: it kills the permutation channel, spares token-id

Π (permutation) recovery is the VMA row-correspondence attack on the noised embed-table proxy;
layer-independent, averaged over 3 seeds (`b4_cross_scheme.json`):

| b | Π recovery (VMA) |
|---|---|
| 0.000 | 1.000 |
| 0.109 | 0.977 |
| 0.218 | 0.565 |
| 0.381 | 0.204 |
| 0.545 | 0.099 |
| 0.817 | 0.037 |

Token-id recovery (ridge inversion) on the activation surface, b=0 vs b=0.817, by depth:

| layer | token-id rec @ b=0 | token-id rec @ b=0.817 |
|---|---|---|
| L0  | 0.747 | 0.670 |
| L5  | 0.625 | 0.407 |
| L12 | 0.510 | 0.377 |
| L20 | 0.510 | 0.455 |

**Finding.** The same noise scale that drives Π-recovery from 1.0 → 0.04 (a 27× collapse) leaves
token-id recovery essentially intact (L0 0.747 → 0.670; deeper layers ~0.4–0.5 at *both* ends). A
single scalar "privacy level" does **not** uniformly reduce leakage across secret kinds — protecting
row-correspondence (fine geometry) is cheap, protecting token identity (a quantization-robust
statistic) is not.

## R2 — No depth sign-flip under Shredder (corroborates the input-DP decoupling is injection-specific)

token-id recovery stays **positive at every depth** under Shredder (R1, b=0.817 row: 0.67/0.41/0.38/0.46) —
no L20 sign reversal. Because Shredder injects at the *observed* layer (no propagation), it behaves
like input-DP**@L0** at all depths. This is the orthogonal-defense confirmation that the depth-resolved
decoupling of `claim:depth-decoupling-input-dp` is a **propagation/injection-locus property**, not a
generic lossy-defense artifact.

## R3 — bits→recovery calibration does **not** transfer across defense families

Spearman(probe-bits, attack-recovery), within-family vs pooled (`b4_cross_scheme.json`,
`cross_scheme_transfer`):

| channel | within input-DP | within Shredder | pooled (cross-family) |
|---|---|---|---|
| token-id | 0.642 | 0.389 | 0.453 |
| perm-Π | 0.812 | 0.425 | 0.569 |
| embedding | 0.750 | NaN¹ | NaN¹ |

¹ Shredder embedding arm excluded: `embed_probe` is numerically unstable (values ~3–4×10³, NaN std
across seeds). The token-probe perplexity readout is likewise unstable (seed-1 collapse to 0.086) —
**R1 therefore uses attack recovery, not the probe, for the channel-selectivity claim.**

**Finding.** Within each family the matched probe tracks recovery, but the *slope/shape* differs:
within-DP ρ (0.64–0.81) > within-Shredder ρ (0.39–0.43), and pooling the two families degrades the
correlation (0.45–0.57). The bits→recovery map is **conditional on the defense mechanism** — a probe
calibrated on one family under-predicts another. IT-leakage prediction is per-defense, not universal.

## R4 — AloePri defends only through the keymat change-of-basis (perm-core is breakable)

`aloepri_vma_sweep.json`, `bits | readout`:

| regime | α_e | CLUB | retr-PVI | VMA recovery |
|---|---|---|---|---|
| perm-core | 0.00 | 252 bits | 3.34 bits | 1.000 |
| perm-core | 0.20 | 250 bits | 3.31 bits | 0.561 |
| perm-core | 0.35 | 246 bits | 3.25 bits | 0.212 |
| perm-core | 0.50 | 240 bits | 3.17 bits | 0.088 |
| perm-core | 1.50 | 181 bits | 2.27 bits | 0.007 |
| **keymat** | 0.00 | **−2.4 bits** (estimator floor → ~0) | **0.6 mbit** | **0.000** |

Spearman(bits, VMA-recovery) within perm-core: CLUB 0.976, retr-PVI 1.000 (8-point α_e sweep, single seed).

**Finding.** The dense invertible basis `P̂` is the load-bearing defense: with *zero* added noise the
keymat regime already drives VMA recovery to 0 and the independent CLUB probe to its estimator floor
(~0 bits). The permutation+noise core alone is VMA-vulnerable up to α_e≈0.35 and needs α_e≳0.5 to
suppress recovery below 0.1 — and there the probe tracks it (Spearman ≥0.98). The two regimes are two
ends of a leakage–utility profile: cheap-but-breakable (perm-core) vs information-destroying-to-the-probe
(keymat).

---

### Provenance
- `results/b4_cross_scheme.json` / `results/b4_run.log` — Shredder static-Laplace + input-DP cross-scheme (R1–R3).
- `results/aloepri_vma_sweep.json` — AloePri perm-core vs keymat (R4).
- Re-rendered via `src/talens/report.py` (`LeakageReport`, `permutation_readout`); bits stored verbatim.

---
type: reference
status: current
created: 2026-06-16
updated: 2026-06-16
tags: [control-tasks, selectivity, memorization, PVI, MDL, CLUB, inversion-attack, vocab-disjoint]
companion: [pareto-hv-control-tasks]
---

# Control tasks: measuring memorisation under the three measure probes

How much of a measure's value (PVI / MDL / CLUB) or an attack's recovery is
*genuine recoverable structure* versus *memorisation / estimator artefact*?
This doc specifies the control-task mode (`--control`) that answers that, the
reasoning behind the locked design, and what was deliberately **not** built.

## Definitions

| Term | Meaning |
|---|---|
| **Control task** | Hewitt & Liang (EMNLP 2019, [arXiv:1909.03368](https://arxiv.org/abs/1909.03368)): a task with the real task's output space but **randomised labels**, run with the *identical* probe/budget. Measures the probe's memorisation capacity on this representation. |
| **Selectivity** | `real_measure − control_measure`. High selectivity ⇒ the signal is structure, not memorisation. |
| **Shuffle floor** | Our control task: permute the labels (break X↔Y) over the **same rows / same split**, only the labels differ. |
| **Memorisation gap (M3)** | `recovery(row-split) − recovery(vocab-disjoint)` — how much recovery needs the token id to have been seen in training. |
| **Class-probe family** | `v_information`, `online_code_length`: softmax over a fixed top-`max_classes` id set. Row-split only (a softmax head can't emit an unseen id). |
| **Retrieval family** | `v_information_retrieval`, `online_code_length_retrieval`: ridge→embedding + cosine-softmax. Generalises to unseen ids (resolution B). |

## Why a raw measure / recovery can be inflated

On purely random `X` the held-out test split already kills spurious recovery (a
probe can memorise random *train* rows but cannot generalise to random *test*
rows). The dangerous inflation comes from **real-but-trivial structure that
survives held-out test**, plus estimator floors:

| # | Mechanism | How it inflates | Killed by held-out test alone? |
|---|---|---|---|
| **M1** | Marginal / frequency | degenerate predictor emits frequent token; ridge collapses to mean embedding → centroid token | No for the attack; yes for PVI/MDL (null term subtracts the prior) |
| **M2** | Estimator finite-sample floor | estimator reports >0 even at X⊥Y. **CLUB** overshoots true MI 2–4× and has no held-out prior | Partially — CLUB's only null is its product-of-marginals term |
| **M3** | Per-vocabulary memorisation | row-split shares ids train/test; probe recognises a seen id's fingerprint | No — row-split test shares the class set |
| **M4** | Geometry / conditioning | high-rank `X` ⇒ ridge can hit any linear target | No — but see "Why M4 was dropped" |
| **M5** | Probe capacity | over-expressive probe fits arbitrary labels (Zhang et al. 2017) | Mostly yes |
| **M6** | Chance | `1/pool`, `1/C` | Yes (baseline) |

Built-in defences differ per estimator — so a single control is not equally
informative for all:

| Measure | Built-in defence | Residual threat | Informative control |
|---|---|---|---|
| PVI (class-probe) | `q(y\|∅)` subtracts prior; held-out test | M3 | shuffle reads ≈0 (confirms the null) |
| MDL surplus | floor subtracts prior + complexity | M3 | shuffle reads ≈0 |
| **CLUB** | only product-of-marginals | **M2** | **shuffle** (the absolute floor it lacks) |
| Attack (TTRSR) | none — raw rate | **M1**, M3 | shuffle (→frequency floor) + **M3 (vocab-disjoint)** |

## The locked control set

Two controls, partitioned by which estimators they touch:

| Control | Detects | Estimators | Reported as |
|---|---|---|---|
| **shuffle** | M1 / M2 / M5 / M6 | PVI, MDL, CLUB, attack | `*_shuffle` + `*_selectivity = real − shuffle` |
| **vocab-disjoint** | M3 | attack only | `ttrsr_top1_row`, `ttrsr_top1_vocab`, `ttrsr_mem_gap = row − vocab` |

### Shuffle = Option A (global permutation, split held fixed)

Permute labels over the kept rows **before the same split**, so train and test
see a *consistent* broken mapping. The alternative — shuffling only train labels
and scoring true test labels — **cannot** detect M2/M5: spurious train fit is
scored against unrelated truth and washes out to the prior, so CLUB's estimator
floor (the main reason this control exists) would falsely read ≈0. Option A
keeps the broken mapping consistent, so capacity that *transfers* is exactly
what inflates the floor we subtract. Implemented inside each estimator (after
row/class selection, before the split) so only the labels differ — Hewitt &
Liang's cardinal rule. Single seeded draw (`control_seed = 20260616`).

### M3 (vocab-disjoint) is attack-only

The class-probe PVI/MDL are structurally row-split-only; you cannot compute a
row-vs-vocab gap for them, nor subtract a retrieval-family number from a
class-probe number (different family, null, units). M3 is therefore computed
only for the attack (which honours `split_mode` natively) and reads the
per-vocabulary memorisation directly. (Retrieval-family PVI/MDL *could* carry an
M3 gap; deferred — see "Not built".)

## Why M4 (random/permuted-target) was dropped

The worry was "`kqv_out`'s ~97% is ridge conditioning, not embedding alignment."
On inspection it isolates nothing new for this attack:

- **Held-out test already neutralises bare conditioning.** "Ridge hits any
  target" is a *train* statement; high *test* recovery already implies `X`
  clusters by id (feed Gaussian-noise `X` and test recovery → frequency floor).
- **The genuine-vs-memorised question is M3's job.** Per-id separability that
  needs a seen id collapses vocab-disjoint (M3); genuine linear alignment
  (`X ≈ A·embed`, e.g. `kqv_out`'s self-attention diagonal carrying a linear
  image of the token's own embedding) generalises vocab-disjoint and is a *true*
  leak, correctly certified by M3, not an artefact to subtract.
- A permuted-target control, scored consistently, is ≈0 on row-split (separability
  swamps it) or collapses into M3 on vocab-disjoint.

So `kqv_out`'s recovery is most likely genuine (a near-linear image of the
embedding) — the worst case for confidentiality, correctly measured. M3 +
shuffle cover the decomposition; M4 adds only redundant compute.

## Usage

```bash
python -m talens.cli --corpus corpora/release-gate-512.txt \
    --control all --out results/pass1-controls.json
# --control {none, shuffle, vocab, all}   (default none → output unchanged)
```

Record fields added (per `(kind, layer)` block):

- shuffle: `v_information_bits_shuffle`, `mdl_surplus_bits_shuffle`,
  `club_mi_bits_shuffle`, `ttrsr_top1_shuffle`, and the matching
  `*_selectivity` (`real − shuffle`).
- vocab: `ttrsr_top1_row`, `ttrsr_top1_vocab`, `ttrsr_mem_gap`.

Calibration block gains `*_selectivity` columns (each floor-subtracted measure
vs raw recovery, and vs `ttrsr_selectivity`) — testing whether removing the
memorisation floor tightens the measure↔attack correlation.

Determinism: `control_seed = 20260616` (= split `seed + 1`); real and control
evaluate the same rows. With `--control none` the output is byte-identical to
the pre-control pipeline.

## Forward pointer — Pareto-HV

The shuffle floor (`selectivity = real − shuffled`) is the fixed weight-1
instance of the control-task axis. Pareto-probing
([handoff](../handoffs/2026-06-16-pareto-hv-control-tasks.md)) would generalise
it: "memorisation = shuffled-label accuracy" is precisely the complexity x-axis
Pareto sweeps, turning a single subtraction into a selectivity-vs-capacity
frontier. That is a capacity/attacker-budget **measure layer**, orthogonal to
this memorisation control, and is deferred there — not part of this control set.

## Not built (deliberate)

- **M4 random/permuted-target** — subsumed by M3 + shuffle (above).
- **Retrieval-family M3 gap** for PVI/MDL — would require wiring
  `v_information_retrieval` / `online_code_length_retrieval` into the pipeline;
  the class-probe headline measures rely on shuffle for their memorisation
  signal.
- **CLUB id-disjoint split** — CLUB's dominant threat is M2, fully covered by
  shuffle.
- **K-permutation averaging** of the floor — single seeded draw; revisit if the
  floor's sampling variance proves material.

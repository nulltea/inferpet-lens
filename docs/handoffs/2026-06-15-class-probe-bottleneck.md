---
type: handoff
status: current
created: 2026-06-15
updated: 2026-06-15
tags: [handoff, calibration, probe, performance, rocm, gfx1151]
companion: [it-leakage-estimation-set, attacks-setting]
---

# Handoff: class-probe (family 1) is the calibration bottleneck

## Focus for the next session

Get the **first real 512-prompt calibration** to run in a sane wall-time.
The blocker is the **class-probe PVI/MDL measure (family 1)**. The
**retrieval/ridge probe (family 2)** is implemented + tested but
**deferred** by decision — do *not* pivot to it unless family 1 proves
unsalvageable.

## Where things are

Everything below is **on the GPU and validated** — the infra is done; the
open problem is purely the probe's cost. Pass-1 design + decisions live in
[`docs/plans/it-leakage-estimation-set.md`](../plans/it-leakage-estimation-set.md);
the family-1-vs-family-2 / split-regime rationale is in
[`docs/research/attacks_setting.md`](../research/attacks_setting.md)
(family 1 = "class-probe / resolution A"; family 2 = "retrieval /
resolution B"). Don't re-derive those here.

- **ROCm runs** via `scripts/run_in_rocm.sh` + `Containerfile` — thin image
  `talens-rocm:latest` on AMD's `rocm/pytorch:rocm7.2.3` base (the only
  prebuilt torch with gfx1151 kernels; shared with AloePri). No ROCm
  runtime duplicated.
- **gfx1151 validated**: torch matmul; nnsight 0.7 capture (eager attention
  **does** return per-head weights); GPU CLUB (~16× vs CPU, commit
  `19be88c`); GPU torch probe runs but is slow (see blocker).
- **Qwen3-4B and 1.7B are HF-cached** (no download).
- **CPU dev loop**: `.venv` (system-site torch+cpu + scipy/sklearn/pytest).
  `pytest -q` → **24 green** including the analytic ground-truths and the
  new probe oracle test.

## ⚠ Uncommitted working tree

The **family-1 GPU torch probe** is implemented but **NOT committed** (last
commit is `ec41480`). Working-tree changes:

- `measures/_probe.py` — probe swapped sklearn → **torch LBFGS, auto-device
  (cuda/ROCm)**; sklearn kept as `sklearn_*` oracle functions.
- `measures/vinfo.py`, `measures/mdl.py` — probe param `C` → `l2` for the
  torch probe.
- `cli.py` — reverted thread-pool → **sequential blocks** (everything is
  GPU-bound now, so the thread pool from `ec41480` is moot).
- `tests/test_probe_oracle.py` (new) — asserts torch probe ≈ sklearn on
  held-out cross-entropy (the trust gate for the GPU probe).
- `tests/test_analytic.py`, `tests/test_orchestration.py` — updated for the
  `l2`/sequential changes.

**First action: decide the fix below, apply it, then commit this set.**

## The blocker (measured, not estimated)

One **full-vocab class-probe fit** (N≈7k rows, **C≈2500 token-classes**,
d=2560) costs **~68 s on gfx1151** — and it's **steady-state, not JIT**:
three consecutive fits all measured 68–69 s. A full run does **~648 fits**
(MDL online-coding re-fits the probe on ~6 growing prefixes per block ×
72 (kind,layer) blocks ≈ 3.15 full-fit-equivalents/block), so the
class-probe sweep is **~4 hours on the GPU**.

Root cause is intrinsic: **iterative multinomial logistic over ~2500
classes**, many times. Device doesn't change the order of magnitude:
- **CPU sklearn** is worse — GIL-bound under threading (≈4× on 16 cores,
  not 16×; the `ec41480` thread pool gave only ~400% CPU), and the
  C≈2500/d=2560 fit doesn't converge in `max_iter` (divide-by-zero / "not
  converged" warnings).
- **GPU torch LBFGS+strong_wolfe** is the 68 s above (strong_wolfe does
  ~1000 closure evals/fit on this hard problem).

## Options to unblock family 1 (pick one, in the working tree)

1. **Optimizer swap (recommended first try)** — in `_probe.train_softmax_probe`,
   replace `LBFGS(line_search_fn="strong_wolfe")` with **Adam, fixed steps**
   (~10× fewer closure evals → est. ~25–30 min full run). **Must re-pass
   `test_probe_oracle.py`** — Adam may under-converge vs sklearn; tune
   lr/steps until the held-out CE matches within the 0.05-nat tolerance.
2. **Cap `max_classes`** (e.g. 256) in `vinfo`/`mdl` — ~10× smaller fits →
   ~15 min, but changes the measure's meaning to "leakage about the top-N
   most-frequent tokens" (document it).
3. **Cut fit count** — fewer MDL `block_fractions` (4 not 6) and/or sweep a
   subset of layers (`--layers`) for the first run; a regression over ~12
   layers is plenty.
4. Combine 1+3 for margin.

## Deferred (do not start unless asked): family 2 — retrieval/ridge probe

`measures/_retrieval.py` + `v_information_retrieval` +
`online_code_length_retrieval` (committed in `11eb3d1`, tested in
`tests/test_retrieval.py`). It replaces the multinomial classifier with the
inversion attack as a probability model: a **closed-form ridge** `x→embedding`
solve + temperature-softmax over a candidate pool. ~<1 s/fit → **~6 min**
full run, vocab-disjoint (aligns with the attack). The cost finding above
argues for it, but per decision it stays parked behind family 1 for now.

## Suggested next steps

1. Apply option 1 (Adam) to `_probe.py`; run `pytest -q` (esp. the oracle
   test) in `.venv`; commit the uncommitted set.
2. Launch the 512-prompt run:
   `scripts/run_in_rocm.sh python3 -m talens.cli --model Qwen/Qwen3-4B --corpus corpora/release-gate-512.txt --attack-split-mode row --out results/pass1_512.json`
   (background it; the CLI only prints at the end — `tee` to a log and
   watch the log, not the task stdout).
3. Read out the per-layer table + the 3 calibration blocks (Spearman/R² of
   PVI / MDL-surplus / CLUB vs attack TTRSR). That's the first real result.

## Gotchas

- **`/tmp/profile.py` shadowing**: a stray file under the mounted `/tmp`
  shadows stdlib `profile` when you run `python3 /tmp/foo.py` (script dir on
  `sys.path[0]`). Run scratch scripts from the repo dir (e.g. gitignored
  `results/`), or use `-m`. `python3 -m talens.cli` is unaffected.
- `run_in_rocm.sh` exports `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` (from the
  thread-cap era). Harmless for the GPU path; revisit only if a CPU fallback
  matters.
- Watching a backgrounded run: `tee`'d log has the full stream; the task's
  own stdout is empty until the end because the CLI batches its prints.

## Skills

No special skill needed. `docs-tidy` if the docs accrete; `diagnose` if the
Adam-vs-oracle convergence needs a disciplined loop.

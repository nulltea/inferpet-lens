---
type: handoff
status: current
created: 2026-06-18
updated: 2026-06-18
tags: [PVI, v-information, retrieval-family, class-probe, overfit, calibration, measure-design]
companion: [sae-attack]
---

# Handoff: an independent V-family that predicts (and explains) the attack

## The goal for the next session

Find / design a **V-information family that is (a) independent of the inversion
attack, (b) statistically well-behaved (no overfit), (c) correlates with attack
success, and (d) gives quantitative insight into *why* the attack succeeds or
fails.** The two families we have each satisfy only some of these.

## What we established (the conceptual core)

Both PVI families compute `PVI = mean[ log q[X](y|x) − log q[∅](y) ]` in bits;
they differ only in the predictor `q[X]` and null `q[∅]`. Computation details
and code refs are in `docs/dev/sae-attack.md` (§"PVI overfit diagnosis" and the
measure-taxonomy discussion) — do not duplicate; the summary:

- **Retrieval-PVI (resolution B, `v_information_retrieval` + `_retrieval.py`) IS
  the attack, re-scored in bits.** Same ridge `X→embedding` map + same candidate
  pool + same cosine as `attacks/_inversion.py::ridge_inversion`; the *only*
  difference is the read-out: attack = top-1 hit (TTRSR), retrieval-PVI =
  calibrated `softmax(cos/τ)` log-prob minus a mean-embedding null. ⇒ it
  "predicts" the attack **near-circularly** (mechanical, not independent). Use it
  as the **attack-faithful leakage measure in bits**, never as an independent
  predictor. Validated well-behaved (vocab-disjoint shuffle floor ≈ −0.5…−1.3 b,
  monotone under noise) — see `scripts/spikes/diag_retrieval_vocab.py`.
- **Class-PVI (resolution A, `v_information` + `_probe.py`) is the genuinely
  independent family** — a free `d→256` softmax over token-*id* classes (null =
  class prior), structurally unrelated to the embedding-retrieval attack. This is
  the family we WANT for prediction/explanation — **but it currently overfits**
  and its numbers are unreliable.

## Why class-PVI currently fails (diagnosed — `/diagnose` session)

Root cause is **capacity ≫ data**, not the metric:
- `d=2304` free softmax (≈590k params), 256-way thin target (~27 rows/class),
  and crucially **`d` (2304) > early-stop val rows (~400)** → the probe memorizes
  the *validation* set too → early-stopping is fooled → keeps a confidently-wrong
  iterate → on test `log q(y|x)→−∞`. **Unbounded log-loss** turns this into a
  **shuffle-control floor of ≈ −48 bits** (healthy ⇒ ~0) and a **non-monotonic**
  PVI-vs-noise (reproduced: 5.79→5.87 *rises* at σ=.25 → falls). Evidence:
  `scripts/spikes/diag_pvi.py` (l2 sweep: floor −48.5 → −9.5 at l2=10, degenerates
  at l2≥100).

The user's candidate TODO factors, annotated by the diagnosis:
- **Limited training data / rows-per-class** — ✅ core driver (with `d>n_val`).
- **Limited classes (256)** — ⚠ a *family/target* choice; naively raising classes
  makes the free softmax **worse** (more params, fewer rows/class). "Beyond 256"
  is only sane via a structure-sharing (embedding) family — which is the attack.
- **top-1 vs top-k** — ✗ not the overfit cause (that's a TTRSR-vs-log-loss
  read-out difference, orthogonal to the floor).

So: the independence we want lives in class-PVI; the failure is an **estimator
regime** problem, fixable without abandoning the family.

## Concrete directions for the next session

Design an **independent, capacity-matched** V-family (predict token-id WITHOUT
the attack's ridge→embedding map, kept in-regime so its held-out signal is
faithful):
1. **Capacity-controlled class-probe (smallest change):** PCA `X`→dim `< n_val`
   (so val is no longer memorizable) + **control-anchored regularisation** (pick
   `l2`/dim so the *shuffle floor ≈ 0*, not by val-CE which is itself overfit).
   Then check it correlates with TTRSR as an independent predictor.
2. **Different bounded families** to compare: kNN / Gaussian class-conditional
   (generative, self-regularising), fixed-random-projection + linear readout,
   small-MLP with explicit capacity budget.
3. **Explanatory axes** (the "why"): decompose the chosen measure per-layer /
   per-capacity so it reads as *"how many bits are linearly available to a
   bounded-capacity reader vs the attack's recovery"* — the gap between an
   independent bounded family and the attack-faithful retrieval-PVI is itself the
   interesting quantity.

Open question to resolve first: **what exactly should independence buy us?** If
the only goal is predicting TTRSR, an independent family that correlates is the
prize; if the goal is *explaining* leakage, prefer a family whose knobs (capacity,
layer, sparsity) map to interpretable causes.

## How each quantity is computed (concrete)

All four read the **same** captured `(X, y)` per `(kind, layer)`: `X` = activations
`(rows × d)`, `y` = per-position token-id. `embed_table E` = `(vocab × d)` input
embeddings (gemma ties embed/unembed).

1. **Attack — TTRSR** (`attacks/_inversion.py::ridge_inversion`):
   fit ridge `W: X→E[y]` on train (scan `α∈{1e-4,1e-2,1}` by val top-1) → predict
   `ŷ(x)=Wx` on test → cosine-match `ŷ(x)` against a candidate **pool** (random
   `≤2048` ids ∪ test/val ids) → **TTRSR top-k** = fraction of test rows where the
   true id is rank-1 (or rank-≤10). `split_mode∈{vocab,row}` (vocab = train/test
   share no id). Metric: hit-rate ∈ [0,1].

2. **retrieval-PVI** (`measures/vinfo.py::v_information_retrieval`, `_retrieval.py`):
   **same** ridge `X→E[y]` and **same** pool/cosine; instead of argmax it forms
   `q(y|x)=softmax(cos(ŷ(x),E[y])/τ)` (τ tuned on val by NLL), null
   `q[∅]` = retrieval from the **mean train-embedding**; then
   `PVI = mean_test[ log₂ q(y|x) − log₂ q[∅](y) ]`. **= the attack, scored as
   calibrated log-likelihood (bits) instead of top-1.** Generalizes to unseen ids
   → runs vocab-disjoint.

3. **class-PVI** (`measures/vinfo.py::v_information`, `_probe.py`):
   keep rows of the **top-256** ids (re-index `0…C−1`); **row**-split 70/30; train
   a **free softmax** `q(y|x)=softmax(W·x̃+b)`, `W∈ℝ^{d×256}` (standardize `x`,
   AdamW lr .05 / weight-decay `l2`=.1, 500 steps, early-stop on internal 15% val);
   null `q[∅]` = **class prior** (token frequency); same
   `PVI = mean_test[ log₂ q(y|x) − log₂ prior(y) ]`. Family is a *generic
   classifier over token-id classes* — **independent of the embedding attack**.

4. **CLUB** (`measures/club.py::club_mi_upper_bound`):
   MI **upper bound** between `X` and the **token embedding** `Y=E[y]` (continuous,
   like the attack's regression target). Train a variational
   `q(Y|X)=N(μ(X),σ²(X))` (small MLP, `max_rows≈2500`, ~400 Adam steps); bound =
   `mean_i log q(Y_i|X_i) − mean_{i,j} log q(Y_j|X_i)`, in bits. Different
   *estimator* (variational MI), independent of the probe families. Magnitude is a
   loose envelope — use rank, not absolute bits.

**Independence map:** TTRSR and retrieval-PVI share the ridge→embedding map (#1≈#2,
circular); class-PVI (#3, free token-id classifier) and CLUB (#4, variational MI on
embeddings) are the families *structurally independent* of the attack.

## Replicating the DP-budget (ε) sweep

Runner: `scripts/spikes/localdp_runner.py` (generic input-local-DP; embedding-hook
clip-to-C + Gaussian σ=C·√(2ln(1.25/δ))/ε, **C calibrated from runtime embed
norms** at `--clip-percentile 99.9` so clip-only ≈ clean). Run **only** in the GPU
container; gemma-2-2b + gemma-scope must be HF-authenticated (see the sae-gemma
handoff).

```bash
# retrieval-PVI + CLUB + TTRSR (current default of the runner):
scripts/run_in_rocm.sh python3 scripts/spikes/localdp_runner.py \
  --model unsloth/gemma-2-2b --corpus corpora/release-gate-512.txt \
  --max-prompts 256 --layers 5,12,20 \
  --epsilons inf,8192,4096,2048,1024,512,256 --clip-percentile 99.9 \
  --split-mode vocab --out results/localdp_curve_retrieval.json
```

- **CLUB** and **TTRSR** are produced by the same run (`dp_club_bits`, `dp_top1`).
- **class-PVI** version: the runner's `panel()` currently calls
  `v_information_retrieval`. To reproduce the class-probe column, swap that line to
  `v_information(X, y)` (+ `control="shuffle"` for the floor) — i.e. the pre-fix
  runner that produced `results/localdp_curve.json` (`git show` the diff on
  `scripts/spikes/localdp_runner.py`). Keep everything else identical.
- **Fast, model-free variants** on the cached capture (no DP-propagation, post-hoc
  noise on `X`): `scripts/spikes/diag_pvi.py` (class vs retrieval, l2 sweep, noise
  sweep) and `scripts/spikes/diag_retrieval_vocab.py` (vocab-disjoint retrieval
  floor + monotonicity). These are the quick feedback loops.

## Repo state / artifacts

- **Branch `main`**, last commit `de820cb`. **Uncommitted working-tree changes
  from this session** (not yet committed — decide before continuing):
  - `scripts/spikes/localdp_runner.py` — PVI switched to retrieval family (the
    spike fix; fine there since the runner isn't doing the independence check).
  - `tests/test_retrieval.py` — 2 regression tests
    (`test_classprobe_overfits_high_d_but_retrieval_floor_is_sane`,
    `test_retrieval_pvi_monotone_under_noise`). **5/5 pass** on host `.venv`.
  - `docs/dev/sae-attack.md` — PVI-overfit diagnosis section + measure taxonomy.
  - `scripts/spikes/diag_pvi.py`, `scripts/spikes/diag_retrieval_vocab.py` —
    reproducible diagnostic loops (model-free, use the cached gemma capture).
- **Systemic flag (unaddressed):** the **main pipeline** (`src/talens/cli.py:77`,
  `calibrate_capture`) uses the overfitting **class-probe** `v_information` for
  PVI on `resid_post` (d=2304) → its reported PVI is also contaminated. Left for
  the user's calibration call.
- **Fast loop (no model/GPU needed):** cached gemma capture at
  `results/capture_cache/capture-4ca8a33e16bfbec9.pt` (+ embed
  `embed-b0c6566474cadb27.pt`); `load_capture` → `stack` → run measures. The two
  `diag_*.py` scripts already do this (run via `scripts/run_in_rocm.sh` for GPU,
  or host for CPU).

## Suggested skills for next session

- **`/diagnose`** — continue the estimator work (the fast model-free loop on the
  cached capture is already the feedback loop).
- **`/research`** — V-information / probe-family literature for an independent,
  capacity-matched estimator (and whether "usable info that predicts an attack"
  has a standard formulation).
- **`/grill-me`** — to pin down what "independent + explanatory" must satisfy
  before building, so we don't re-conflate families again.

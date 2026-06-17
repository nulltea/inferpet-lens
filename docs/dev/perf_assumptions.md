---
type: dev-log
status: current
created: 2026-06-16
updated: 2026-06-17
tags: [performance, calibration, club, probe, tradeoffs]
companion: [it-leakage-estimation-set, attacks-setting]
---

# Performance assumptions & fidelity tradeoffs

A living registry of optimizations that buy speed **at a cost** — i.e. they
change *what is measured* or *how faithfully*, not just how fast. Each entry
states the cost, how it was validated, its default, and **how to turn it
off**. Exact/lossless optimizations (no fidelity cost) are listed at the
bottom for completeness but need no gate.

Rule of thumb for this repo: the calibration thesis uses the **cross-layer
rank** of each measure against attack recovery (Spearman/R²), not the
absolute measure value. So an optimization that preserves rank but distorts
magnitude is acceptable for calibration — but only if that's stated, because
it makes the absolute number untrustworthy.

## Legend

| Term | Meaning |
|---|---|
| PVI | pointwise V-information (the class-probe measure) |
| CLUB | Contrastive Log-ratio Upper Bound of MI (the upper bracket) |
| recovery | attack TTRSR (token top-1 recovery rate) — the ground truth |
| rank-faithful | preserves cross-layer ordering, not absolute value |

## Costed optimizations (gated)

| # | Optimization | What changes | Cost / tradeoff | Validation | Default | Disable / override |
|---|---|---|---|---|---|---|
| 1 | **PVI/MDL class cap** `max_classes=256` | probe predicts only the 256 most-frequent token ids; rows of rarer ids dropped | measure becomes "token-identity info about the **top-256** tokens", not the full ~2.5k-token corpus vocab | statistical: full vocab → <3 rows/class (memorisation); 256 → ~27 rows/class | `256` | `--max-classes N` (e.g. 4096 ≈ all) |
| 2 | **CLUB fast mode** `steps=150, max_rows=2500` | variational net trained 150 (not 400) steps on ≤2500 (not ~6.6k) rows | CLUB loss does **not** plateau by 400 steps → 150 steps **undertrains** → bound magnitude ~20% lower; the value is **no longer a converged MI upper bound**, only a rank-faithful proxy | 11-block spread on the cached 512-prompt capture: Spearman(full, fast)=**0.982**; Spearman-vs-recovery **0.989 (full) = 0.989 (fast)**; **~10.3×** faster | `fast` | `--club-fidelity full` or `TALENS_CLUB_FIDELITY=full` |
| 3 | **PVI row cap** `max_rows=2500` (`PVI_MAX_ROWS`) | softmax probe fits on a 2500-row subsample (after class selection) instead of the full ~6.6k-row train split | **rank loss**, unlike entries 1–2: cross-block `Spearman(PVI, recovery)` drops **0.91→0.81**. Acceptable only because PVI is the *secondary* predictor (CLUB ranks recovery at 0.99) and PVI's row count does **not** change its control-task reading (see §3) | 30-block subset of the 512 sweep: see §3. PVI was ~56% of a block (MDL off) → **~3×** faster PVI, ~−17% off the sweep | `2500` | `--pvi-max-rows 0` (uncapped) |

### 1. `max_classes = 256` (PVI / MDL)

The class-probe family fits a softmax classifier `q(y|x)` over token ids. The
512-prompt corpus has ~2.5k distinct ids over ~7–9k rows ⇒ **<3 rows/class** —
the probe memorises rather than estimates, and held-out PVI is dominated by
overfit/noise. Capping to the **256 most-frequent** ids gives ~27 rows/class,
a statistically meaningful fit, **and** an ~10× smaller (cheaper) `(N×d)@(d×C)`
matmul.

Cost: the measure now reads *"usable token-identity information about the
256 most-frequent tokens"*. Rare-token leakage is not measured. To recover the
full-vocab measure (slow, statistically fragile) pass `--max-classes 4096`.

### 2. CLUB fast mode (`steps=150`, `max_rows=2500`)

CLUB was the dominant cost (~12 s/block, ~85% of a full run). Two levers:
fewer training steps and fewer rows. **We validated before adopting** — and
the naïve assumption was wrong in an instructive way:

- **Loss does not plateau at 400 steps** (lr=1e-3): e.g. resid L0 loss
  655→246 between steps 150→399. So 150 steps is *undertrained*, not "free":
  the magnitude drops ~20% and is no longer the converged bound.
- **But the rank is preserved.** Across a 7-resid + 4-attn block spread:
  `Spearman(full, fast)=0.982`, and the calibration number itself —
  `Spearman(club, recovery)` — is **0.989 for both** full and fast. Speedup
  **10.3×** (12 s → ~1.1 s/block; full run ~17 min → ~4–5 min).

Conclusion: acceptable **because CLUB is used only for rank** (it's already
documented as a loose upper bound). Its absolute magnitude under `fast` is
**not** a trustworthy MI estimate — use `--club-fidelity full` if you ever
report the bound's value rather than its ordering.

Future refinement (not yet validated): the loss is still descending at
lr=1e-3, so **raising the learning rate** could reach a *tighter* bound in
fewer steps — cutting cost without undertraining (would recover magnitude
fidelity). Validate the rank before switching.

### 3. PVI row cap (`max_rows = 2500`)

With MDL off, **PVI is the dominant cost** — ~56% of a block (it fits the
softmax probe on the full ~6.6k-row train split; CLUB is already capped at
2500 and ~5× cheaper, and the attack solve is GPU-Cholesky). Capping PVI to
2500 rows (after class selection, so the class set is unchanged) makes it ~3×
faster and takes the full control sweep from ~23 min toward ~12 min.

**Unlike entries 1–2, this one costs cross-block rank**, so it was a
deliberate call, not a free lunch. Validated on a 30-block subset (10 layers ×
3 kinds) of the 512-prompt sweep, vs the full-row PVI in
`results/sweep-controls.json`:

| PVI rows | Spearman(PVI, recovery) | Spearman(PVI selectivity, recovery) | shuffle floor (mean bits) |
|---|---|---|---|
| full (~6.6k, ~26/class) | **0.91** | 0.61 | −41 |
| cap 5000 (~14/class) | 0.91 | 0.52 | −41 |
| **cap 2500 (~7/class)** | **0.81** | 0.59 | −44 |

Why it's acceptable:
- **CLUB, not PVI, is the headline predictor** (Spearman-vs-recovery 0.99 vs
  PVI's 0.81–0.91); PVI is corroborating. A 0.81 rank is still clearly
  informative.
- **The cap does not change PVI's control-task reading.** PVI *selectivity*
  ↔ recovery is ~0.6 at every budget (it is *not* a row-count artifact), and
  the shuffle floor stays ~−41 to −44 bits independent of rows — so the
  CLUB↔PVI selectivity asymmetry is intrinsic, not a budget confound. (This
  cap was added *after* confirming that, so it doesn't muddy the control
  analysis in `control-tasks.md`.)
- `cap 5000` would preserve the 0.91 rank at ~1.5× — a safer knob if PVI's
  baseline rank ever matters; `--pvi-max-rows 5000` selects it, `0` = uncapped.

Note this cap supersedes the "PVI is now the dominant recurring component"
cost noted in the divergence-fix section below (that was at full rows).

## Resolved — PVI probe divergence (was an open correctness bug)

**Symptom:** PVI was strongly negative on deep/attn blocks where the attack
clearly recovers tokens (resid L18: recovery 0.55, PVI −13; attn L18 −132).

**Diagnosis (/diagnose, differential vs sklearn oracle):** the data is
highly decodable — sklearn fit resid L18 to test-CE 0.63 / PVI +5.8. Our
probe got **0.648 test accuracy but test-CE 19.4 bits** = *overconfident,
under-converged logits*. Root cause: **`lr=0.2` jumped from underfit
straight to overconfident-divergent** with no good iterate for early
stopping to keep; weak weight decay let logits blow up.

**Fix:** `lr 0.2→0.05`, weight decay `1e-2→0.1`, steps `300→500` (early
stopping retained). Restored resid L0/L18/L30 PVI to ≈ +5.5–5.9 (matching
sklearn) and pulled attn L18 from −132 to −7.7 (genuinely low-info, now
small/legit). Guarded by `tests/test_analytic.py::
test_vinfo_positive_on_informative_but_memorizable` (informative signal +
memorizable noise dims; the old config scores ≈ −3.8 there).

**Cost of the fix:** 500 (vs 300) probe steps → ~4 s/resid block (vs ~2.5 s)
— PVI is now the dominant *recurring* component. Acceptable: correctness
gates the calibration. Not a fidelity tradeoff, a methodology choice.

**What would have prevented it:** the oracle test only covered *easy,
over-determined* blobs, where regularisation/lr never matter — so it never
exercised the overconfidence-on-ill-conditioned-data failure. The new
regression test closes that seam.

## Exact / lossless optimizations (no fidelity cost — no gate needed)

Listed for completeness; these change speed only, verified equivalent.

| Optimization | Why lossless |
|---|---|
| cover-break dual-form solve (`O(d³)→O(k³)`) | algebraic push-through identity; verified rel-err ~1e-4 (float32) vs primal |
| attack ridge → GPU (device-correct `ridge.py`) | identical math, different device |
| attack ridge **solve** → GPU Cholesky (was CPU LU) | SPD normal equations + bias jitter; matches CPU LU to <1e-6, pipeline TTRSR bit-identical; CPU LU kept as fallback for ill-conditioned under-determined systems |
| dropped `*_NUM_THREADS=1` caps in `run_in_rocm.sh` | environment only |
| stack `(X,y)` once per block | same arrays, computed once instead of 3× |
| block thread-pool (`--workers`) | verified `workers=1 ≡ workers=4` bit-for-bit (CLUB seed+init under a lock) |
| capture cache (`--capture-layers`/`--layers`) | replays the same captured tensors; skips the forward pass |

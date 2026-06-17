---
type: dev-log
status: current
created: 2026-06-17
updated: 2026-06-17
tags: [results, control-tasks, layer-sweep, performance, PVI, CLUB, memorisation]
companion: [control-tasks]
---

# Results chronicles

Running log of headline experimental results and performance characterisations.
Newest first. Raw outputs live in the gitignored `results/`.

## 2026-06-17 — Full 36-layer control-task sweep (Qwen3-4B, 512 prompts)

Config: `--control all` (shuffle + vocab), MDL off, `--workers 4`, GPU
(gfx1151), capture cache hit. 108 blocks (36 layers × {resid_post, kqv_out,
kq}). Wall **1369 s (~23 min)**. Output: `results/sweep-controls.json`.

### Findings — the vocab-disjoint control is the deciding lens

| surface | TTRSR row (mean) | TTRSR vocab (mean) | mem_gap | verdict |
|---|---|---|---|---|
| **resid_post** | 0.80 | **0.65** | 0.15 | genuine leak at **every** depth |
| **kqv_out** | 0.47 | 0.17 | 0.31 | genuine **only at L0**; memorisation at depth |
| **kq** | 0.29 | **0.03** | 0.27 | ~pure memorisation at all depths |

- **resid_post**: TTRSR 0.97@L0 → ~0.74 plateau, but vocab recovery stays
  ~0.5–0.65 across all 36 layers → the residual stream genuinely leaks the
  token throughout the network.
- **kqv_out**: at L0 TTRSR 0.995, mem_gap≈0 (the near-linear-image-of-embedding
  leak). From L2 it collapses (vocab 0.04–0.31, mem_gap 0.30–0.40) → at depth
  its recovery is mostly seen-token memorisation. The L0 "total leak" does
  **not** survive depth.
- **kq**: vocab recovery 0.01–0.06 at every layer → genuine generalisable
  recovery is ~0.01–0.05 network-wide. Raw 0.21–0.43 is almost entirely
  training-vocab memorisation.

### Calibration (108 blocks)

| measure ↔ recovery | Spearman | r² |
|---|---|---|
| CLUB ↔ TTRSR | **0.987** | 0.96 |
| PVI ↔ TTRSR | 0.891 | 0.54 |
| CLUB selectivity ↔ TTRSR | 0.985 | 0.96 |
| PVI selectivity ↔ TTRSR | 0.581 | 0.45 |

CLUB is a near-perfect rank predictor; its shuffle floor is ≈0 across all
layers (M2 estimator-floor worry empirically absent). PVI ranks well but is a
looser linear fit and goes negative on some deep kqv_out blocks (class-probe
underperforms the prior while the retrieval attack still partially recovers).

### Performance breakdown (per-block component timing, L12, GPU)

Measured by profiling single blocks on the cached capture (not a sweep rerun):

| component | kqv_out (9469×4096) | resid_post (9469×2560) |
|---|---|---|
| PVI baseline | 6.45 s | 4.15 s |
| PVI shuffle | 6.50 s | 4.16 s |
| CLUB baseline | 1.31 s | 1.17 s |
| CLUB shuffle | 1.31 s | 1.17 s |
| attack row (base) | 2.47 s | 1.15 s |
| attack row (shuffle) | 2.49 s | 1.15 s |
| attack vocab (M3) | 2.62 s | 1.22 s |
| **block total (all arms)** | **23.1 s** | **14.2 s** |

**PVI (the class-probe) is the dominant cost — ~56–59 % of every block** (with
MDL off). It trains the softmax probe on the **full** ~6.6k-row train split
(256 classes, ~300 Adam steps), whereas CLUB is capped (`max_rows=2500`, 150
steps) and ~5× cheaper, and the attack solve is now GPU-Cholesky (~2.5 s incl.
3-alpha scan).

Control overhead ≈ the baseline doubled: control-all ≈ **2.0–2.3×** the
no-control cost (shuffle re-runs PVI+CLUB+attack; vocab adds one attack arm).
Non-control sweep would be ≈ **45 % of 1369 s ≈ 10 min** (estimated from the
per-block baseline share; not rerun). `--workers 4` bought ~25 % over a serial
estimate (GPU-bound; little further headroom).

### Optimisation backlog (not yet implemented)

1. **Cap PVI training rows (~2500, like CLUB)** — biggest lever. PVI on full
   rows is ~5× CLUB; capping should cut PVI ~2.6× with minor rank impact
   (validate Spearman-vs-recovery, as CLUB's cap was). ≈ −17 % of sweep.
2. **Drop the PVI/MDL *shuffle* floor** — established uninformative in
   magnitude (read PVI selectivity by sign; CLUB + attack are the informative
   floors). Removes ~28 % of the sweep. (1)+(2) together ≈ 23 min → ~12 min.
3. **Fix alpha in control attack arms** (skip re-scanning the 3 alphas in
   shuffle/vocab) — modest now that the solve is on GPU.
4. **On-demand controls** — run controls on a flagged subset, not all 108
   blocks every sweep.

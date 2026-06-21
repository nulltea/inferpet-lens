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

## 2026-06-20 — Capacity-matched class-PVI: fixing the independent family (gemma-2-2b)

Resolves the 2026-06-18 tension (independent class-PVI overfits; retrieval-PVI is the
attack; CLUB upper-bound-only). New measure `talens.measures.vinfo_capacity`
(PCA/randproj/gauss/knn readers over token-id, dim-reduced). Built+validated via
`/experiment-bridge` → `/auto-review-loop` (3 rounds, gpt-5.5 xhigh: 5→6.5→**7/10,
scoped-established**). Scripts: `scripts/spikes/{diag_capacity,diag_nondp,
analyze_faithfulness}.py`; `localdp_runner.py` extended (`--capacity-family/-dim/-l2`,
`--every-n`); results `results/{capacity_screen*,nondp_intervention*,localdp_m2_*}.json`.

**Headline:** class-PVI's failure is the `d>n_val` regime, not the token-id family.
Capacity-matching (PCA→dim<n + linear reader, GPU cov-eigh) removes the catastrophe:
shuffle floor **−49 → −1.5 b** (dim-anchored, *not* l2-anchored), monotone, **~0.3× cost**.
The robust fixed measure is the **reader's token-id accuracy** (bounded), which tracks
the attack (TTRSR):

| defense | reader-accuracy ρ vs TTRSR (L5/L12/L20) |
|---|---|
| PCA-subspace ablation | 0.87 / 0.90 / 0.82 |
| isotropic hidden-state noise | 0.90 / 0.90 / 1.00 |
| input-local-DP | 0.68 / 0.43 / **−0.21** |

- **PVI-in-bits is only partially rescued** — the −48 floor & "rise under noise" are
  unbounded-log-loss artifacts; bits track only with regularisation, fragile under
  iso-noise / late-DP. Report accuracy as primary, bits as auxiliary (NOT "V-information").
- **Unfixed class-PVI is within-layer anti-correlated** (L5/12/20 ≈ −0.86/−0.68/−0.79).
- **Characterised divergence — a depth-resolved decoupling (depth sweep L0/5/12/20,
  `results/localdp_depth_L0_5_12_20.json`):** the fixed measure tracks the attack
  *perfectly at shallow depth and decouples monotonically with depth*:

  | layer | clean TTRSR | ρ(PVI-acc, TTRSR) | ρ(CLUB, TTRSR) |
  |---|---|---|---|
  | L0  | 0.809 | **+0.99** | +0.96 |
  | L5  | 0.559 | +0.68 | +0.96 |
  | L12 | 0.347 | +0.43 | +0.89 |
  | L20 | 0.462 | **−0.21** | +0.29 |

  *Why clean TTRSR is <1*: vocab-disjoint top-1 vs a ~2048 pool + contextualisation —
  L0 (≈input embedding) is 0.81, falling with depth. *Why PVI rises while TTRSR falls
  at depth*: DP noise is injected at the **embedding**; at L0 it hits the representation
  directly so reconstruction and id-decodability fall in lockstep (ρ +0.99), but the more
  transformer blocks it propagates through, the more the network reshapes it into a regime
  where coarse token-id separability is preserved/sharpened while the fine embedding
  geometry the attack needs is destroyed → the two decouple and invert by L20. **CLUB shows
  the same depth gradient → it's a property of the signal under propagated DP, not a probe
  artifact.** Conclusion: input-DP protects *embedding geometry* (the attack's target)
  before it protects *token-identity decodability* (the measure's target), with propagation
  depth as the knob — so an attack-independent measure and an embedding-reconstruction
  attack provably diverge, and that divergence is a measurable, depth-localised quantity.
- **Independence**: token-id target, never the embedding table; ρ(cap, retrieval-PVI)
  0.66–0.76 (<0.9). vs the 2026-06-18 CLUB (pooled DP ρ 0.81), cap-accuracy 0.67 but is
  the only *independent token-target* measure that also tracks (CLUB shares the embedding
  target; retrieval-PVI *is* the attack).

Perf note: an early screen CPU-thrashed 26 min (redundant full SVD); fixed with GPU
covariance-eigh (11s→0.44s, GPU→100%). See `vinfo_capacity._pca_basis`.

## 2026-06-18 — Input local-DP (ε) sweep: PVI families + CLUB vs the attack (gemma-2-2b)

Config: input local-DP runner (`scripts/spikes/localdp_runner.py`), gemma-2-2b,
256 prompts, layers {5,12,20}, vocab split, Gaussian mechanism δ=1e-5, clip at
p99.9 of **runtime** embed-norms (C=199, median=84 → clip-only≈clean; curve is
noise-driven). `r = ‖noise‖/‖signal‖ ≈ σ√d / median`. Outputs:
`results/localdp_curve.json` (class-PVI) + `results/localdp_curve_retrieval.json`
(retrieval-PVI); both share CLUB + TTRSR. Replication: see
`docs/handoffs/2026-06-18-independent-vfamily-attack-correlation.md`.

| ε | r | TTRSR frac L5/12/20 | **class-PVI** L5/12/20 (b) | **retrieval-PVI** L5/12/20 (b) | CLUB L5/12/20 (b) |
|---|---|---|---|---|---|
| ∞    | 0.00 | 1.00/1.00/1.00 | 5.0/4.4/4.9 | 32.1/30.6/31.1 | 3404/3267/3397 |
| 8192 | 0.07 | 0.99/0.97/0.99 | 5.0/4.5/4.9 | 32.1/30.5/**12.6** | 3398/3278/3393 |
| 4096 | 0.13 | 1.01/0.98/0.98 | 5.1/4.5/5.0 | 32.2/**12.3**/30.9 | 3405/3268/3399 |
| 2048 | 0.27 | 1.00/0.97/1.02 | 5.1/4.5/5.0 | 32.0/**12.2**/30.6 | 3413/3262/3387 |
| 1024 | 0.54 | 0.94/0.88/1.01 | 5.3/4.5/5.0 | 32.1/12.2/12.7 | 3328/3238/3386 |
| 512  | 1.08 | 0.62/0.66/0.85 | **5.8**/5.0/5.6 | 11.8/11.6/12.8 | 3054/3217/3504 |
| 256  | 2.15 | 0.36/0.29/0.27 | 5.6/4.0/4.7 | 10.1/9.0/8.9 | 2637/2461/2524 |

**Read per measure (vs TTRSR = ground truth, which falls monotonically 1.0→~0.3):**
- **class-PVI (independent family):** ~flat ≈5 b and **non-monotonic** — it even
  *rises* at ε=512 where the attack is collapsing. **Fails to track the attack.**
  Cause: the free `d→256` softmax overfits in the high-d regime (`d>n_val`); its
  shuffle floor is ≈ −48 b (diagnosis: `docs/dev/sae-attack.md`).
- **retrieval-PVI (= the attack, in bits):** collapses to ~9–12 b exactly when
  TTRSR collapses (ε≤512) → "correlates" — but only because it *is* the attack
  (same ridge→embedding map). Magnitude is also **τ-bimodal** (mid-sweep cells
  flip 32↔12 b from the temperature-grid pick while TTRSR is still 1.0) — an
  estimator artifact, not leakage.
- **CLUB (independent estimator, MI upper bound):** monotone-ish decline
  3404→~2540 b, tracks the attack (consistent with the 2026-06-17 sweep's
  CLUB↔TTRSR Spearman 0.99). Currently the **most usable independent** measure —
  caveat: it shares the *embedding* target with the attack (different estimator).

### The tension (concise)

Class-PVI is the family we *want* — independent of the attack (a free token-id
classifier, not the embedding map) — but it overfits in this high-d regime, stays
~flat/non-monotonic, and so **fails to track the attack**. Retrieval-PVI tracks
the attack only because it **is** the attack (same ridge→embedding, scored in bits
not top-1), so its agreement is **mechanical, not independent validation**. We
still lack a measure that is **both** independent of the attack **and** a faithful
predictor of it (CLUB is the closest today).

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

---

## 2026-06-20 — matched-probe program: B0 (impl+review) + B2 (permutation-Π) 

New thread (`refine-logs/FINAL_PROPOSAL.md`): per-channel matched probes +
decoupling law. Numbers + tables in `refine-logs/EXPERIMENT_RESULTS.md` — **do
not duplicate; read there.** Headlines:

- **B0**: AloePri Algorithm 1 (`scripts/defenses/aloepri.py`) — invertible keymat
  `P̂Q̂=I` validated at d=2304 (err 6e-9, float64-build+solve). + Shredder
  (`shredder.py`) + MMI-PID (`measures/pid.py`). 76/76 tests. gpt-5.5 review
  caught 2 CRITICAL pre-run: Shredder SNR sign inverted; V-info-in-MMI not a sound
  Shannon PID (→ operational reader-atoms + conditional increments + lattice guard).
- **B2** (GPU-free, weight surface, gemma-2-2b embed, N=1200): AloePri perm-core
  α_e sweep, VMA τ-recovery 1.00→0.007. **CLUB-on-φ (independent) tracks at
  Spearman +0.976**; retrieval-PVI +1.000 (mechanical, = VMA in bits). **C4 →
  CLUB-on-φ is the independent Π-probe.** The dense Alg1 keymat drives VMA *and*
  both φ-measures to floor (keymat defeats the RowSort channel; perm-core is the
  vulnerable regime). 
- **Pending**: B3 decoupling-matrix off-diagonal (cross-apply each probe to each
  target) needs a unified GPU activation run; diagonal + L20×DP sign-flip in hand.

### 2026-06-21 — B3 decoupling matrix (headline)

`results/b3_decoupling_matrix.json` (gemma-2-2b, ε×depth×3 seeds, 72 settings,
~13 min GPU). Full numbers in `refine-logs/EXPERIMENT_RESULTS.md`. Honest verdict:
**channel-specificity is depth-resolved, not pooled.** 2/3 matched diagonals
dominate (token Δ+0.087, Π Δ+0.162; CIs exclude 0); embedding's CLUB is generic
(ties). Sign-flip: token-id probe↔attack ρ goes +0.89(L0)→−0.11(L12)→+0.08(L20)
while embedding stays positive (+0.98→+0.36). **Methodological finding:** the
monotone noise-index alone correlates with every attack (−0.73/−0.75/−0.99) →
common-cause decay inflates pooled off-diagonals and deflates Δ_i; the depth axis
carries the real channel-specific signal. Separating channels along the noise axis
needs a 2nd defence family (B4). Random/shuffled controls ≈0; retr-PVI dep +0.885.

### 2026-06-21 — B4 cross-scheme (Shredder vs input-DP)

`results/b4_cross_scheme.json`. 2nd defence family = Shredder static-Laplace
(direct activation inject; clean acts captured once, noise swept in-memory).
Numbers in `refine-logs/EXPERIMENT_RESULTS.md`. Two findings: (1) **the decoupling
is defence-injection-specific** — token-id per-layer diagonal is entirely different
under propagated DP (+0.89→+0.53→−0.11→+0.08) vs direct Shredder (+0.16→−0.16→
−0.18→+0.62); not a universal property. (2) **transfer is channel-specific** —
generic embedding-CLUB transfers across schemes (ρ_DP 0.75 / Shr 0.70 / pooled
0.72), the specific token-id reader does NOT (0.64/0.39/0.45 pooled<within); perm_Π
high seed variance. Same 2/3 diagonal-dominance pattern (token, Π; embedding generic).
Instability: seed-1 CLUB→nan (20/72); finite estimates stable — clamp/retry in club.
Sharpest framing: no single scheme-agnostic leakage scalar; calibration curve =
f(channel, defence-injection-geometry).

### 2026-06-21 — B2+ Π firm-up + CLUB nan fix

`results/b2plus_pi_firmup.json`. Π channel firmed: **per-seed ρ(CLUB-on-φ, VMA
τ-recovery) = +1.000 ± 0.000** across 5 seeds × 2 model widths (gemma d2304, qwen
d2560); match-mode independent (ρ with NN assignment +0.998). B4's apparent Π
"seed variance" was a **pooling artifact** — per-seed curves are monotone (ρ=1),
but pooling raw CLUB magnitudes across heterogeneous token draws adds baseline
offsets that deflate pooled ρ (gemma 0.44, qwen 0.90). **Methodological takeaway:
the calibration unit is the within-condition sweep, not pooled raw magnitudes**
(reframes the B3 monotone-confound and B4 transfer reads). CLUB nan bug fixed
(`measures/club.py`: grad-clip + non-finite skip + None-guard; `test_club_stability.py`;
suite 78/78).

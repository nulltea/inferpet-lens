---
type: plan
status: current
created: 2026-06-25
updated: 2026-06-25
tags: [dp, depth, separability, mdl, bhattacharyya, probe, token-class, measurement-loop, I_G]
companion: docs/html/resid-dp-attacks.html
supersedes: []
---

# Design — token-class separability as a converse+achievable measure on the DP×depth surface

Adapts the MDL-probe separability picture (Voita & Titov, arXiv:1909.01380; the
`regularity-min.png` "does the representation sort the labels" intuition) to **our** setting:
MI-vs-attack-recovery correlation under local DP on the gemma-2-2b residual stream across depth.

## Problem this answers

Handoff `docs/handoffs/2026-06-25-dp-depth-probes-and-attacks.md`, open behaviour #1: **I_G measures
representation-survival, not token-survival at depth.** I_G rises monotonically with depth (residual
SNR grows); CLUB/PVI peak at L20; ridge recovery falls monotonically. These disagree and we cannot
currently tell whether the L20 information peak is *real token information the linear attack can't
read* or a *representation-context artifact*.

A **direct token-class separability** measure adjudicates this. Take a small, fixed set of
closely-related high-frequency tokens, collect their residual representations on the same surface,
and ask how distinguishable the class clouds are across depth × DP noise — **without training the
recovery attack**.

- Separability trajectory **rises to L20** ⇒ genuine token information ridge cannot read → the L20
  peak is real → queue a stronger attack.
- Separability **falls monotonically** ⇒ tracks recovery → the I_G peak is a context-MI artifact →
  reframe/relabel I_G.

Either outcome is a first-class result for the measurement loop (probe ≠ attack), not just a figure.

## Labeled task (closed-class panel, no tagger)

Match by gemma token id (leading-space variants resolved once from the tokenizer at run start):

- **Headline (build first):** `to_be = {is, are, was, were}` — K=4, tense×number of *to be*.
- **Panel (build only after the headline earns it):** `articles = {the, a, an}`,
  `pronouns = {it, they, …}` — shows the behaviour is not *to-be*-specific.

Closed-class sets need **no POS tagger** (rejected dependency) — token-id match only.

## Two metrics, both in bits (converse + achievable)

Same converse/achievable pairing already used in the repo (Bhattacharyya–Fano converse + BNN
achievable; `claim:bnn-error-bounds-bhattacharyya-fano`).

1. **Converse — geometric Bhattacharyya (attack-independent, geometry-only).**
   Reuses `src/talens/probes/channel_error_bounds.py`:
   - class-conditional means `E` (K×d on the subsampled subspace) + pooled within-class noise scale
     `σ` (LDA-style isotropic; pooled within-class RMS),
   - `union_bhattacharyya(E, sigma=σ)` → Bayes-error upper bound `p_e_ub`,
   - `fano_equivocation(...)` → `i_class_bits = log2(K) − H(p_e)`.
   No classifier is trained, so a peak is a real geometric fact, not the attack in disguise.

2. **Achievable — MDL online (prequential) code length on the K-class task.**
   Small linear probe; sort rows, train on first `t`, score `t+1`, accumulate `−log₂ p`. Report
   bits/example and compression vs the `log₂K` uniform baseline (the Voita separability number).
   Because the label set is the K-class closed task (≠ vocab), it is **independent of the ridge
   vocab-recovery attack** — this sidesteps the open-#3 MDL↔ridge circularity, which only bites the
   vocab-retrieval MDL.

Both are computed on the **same captured rows** as recovery and the existing probes (CLUB / V_cap /
I_G), so the trajectories are directly comparable across the layer × ε grid.

## The visual (MDL-blog style) — FIG.05

2D PCA projection (cov-eigh; trivial matrix size, CPU is fine — `# ponytail: PCA on a few hundred
64-d rows, cov-eigh on GPU not warranted`) of the `to_be` class clouds at representative cells:
(L0, L12, L20, L25) × (ε=∞, ε=512), colored by class. Emitted as a coords JSON and rendered as a new
**FIG.05** Plotly scatter grid in `docs/html/resid-dp-attacks.html`. This is the literal "do the
class clouds stay separated as we go deep / add noise" picture.

## Code layout (enforces repo structure rule)

| File | Change |
|---|---|
| `src/talens/probes/class_separability.py` | **new** — reusable logic: resolve class-set token ids, filter+relabel rows, Bhattacharyya converse (calls `channel_error_bounds`), MDL achievable, PCA coords. One `assert`-based `__main__`/`demo()` self-check. |
| `scripts/evals/dp/dp_leakage_sweep.py` | register `sep_bhat`, `sep_mdl` in `PROBES`; add one flag for the class-set name and a PCA-coords dump path. Orchestration only — no separability math inline. |
| `docs/html/resid-dp-attacks.html` | add FIG.05 scatter grid + read-caption; reuse existing Plotly include and `css/site.css` per `docs/html/STYLE.md`. |

The probe functions match the existing registry signature `probe(X, E, y, K, …)` where `y` is the
per-row token id; the separability probe selects rows whose id ∈ the class set and remaps to K class
indices internally.

## Corpus & run plan

- **Corpus:** `corpora/rep2text-stratified.txt` (1965 prompts). Verified counts: `is`=1788,
  `are`=381, `was`=170, `were`=40 (the floor). The 512-prompt `release-gate-512` is too sparse
  (`was`=4, `were`=3) → degenerate class statistics. Recovery + all probes + separability re-run
  together on this one corpus so every trajectory shares a surface.
- **Grid:** layers L0/5/12/20/25 × ε(∞/512/256/128), same as the existing FIGs.
- **Order:** fast-iterate the headline on a single layer first; single-seed full grid next; then
  `--seeds` multi-seed error bars on the headline trajectory (varies only the DP-noise draw).
- **Discipline:** all heavy steps via `scripts/run_in_rocm.sh`; ONE GPU process at a time; verify
  the container with `docker ps | grep rocm` (the `pgrep` self-matches). Refine the run with the
  perf gate (`scripts/harness/perf_gate.md`) before the full grid; estimate wall-time, confirm
  saturation if >10 min.

## Out of scope (YAGNI)

- POS / CCG taxonomy and any tagger dependency.
- UMAP / t-SNE — PCA suffices for the cloud picture and adds no dependency.
- The articles/pronouns panel — deferred until the `to_be` headline shows an interesting trajectory.
- A formal research-wiki claim — written only if the trajectory is decisive (separate `/result-to-claim`
  pass).

## Acceptance

1. `class_separability.py` self-check passes (`.venv/bin/python -m pytest` for the model-free parts;
   GPU parts via the container).
2. `dp_leakage_sweep.py --probes sep_bhat,sep_mdl` runs on rep2text-stratified and emits per-(layer,ε)
   converse bits, achievable bits, and the PCA coords dump.
3. FIG.05 renders the `to_be` class clouds across the chosen cells in `resid-dp-attacks.html`.
4. The separability trajectory across depth is read against recovery (monotone) and I_G/CLUB (L20
   peak), and the open-#1 verdict (token-survival vs rep-survival) is stated in the page prose.

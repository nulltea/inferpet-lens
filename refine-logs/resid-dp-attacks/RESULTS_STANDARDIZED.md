# resid-dp-attacks — consolidated standardized results (bits canonical + per-secret readout)

**Surface:** residual-stream (`resid_post`) + L0 embedding observation of a token under input-DP.
**Secret kind:** `token_id` (the token behind the representation). **Threat model:** WEIGHTS-PUB
honest-but-curious — adversary knows weights + embedding table + published DP params (σ, clip C);
can synthesize unlimited `(noised-rep, token)` pairs (`claim:threat-model-fairness`).
**Metric convention (CLAUDE.md):** canonical **bits** = CLUB MI upper bound `I(rep; token)`
(MDL-SDL auxiliary); per-secret **readout** = token recovery rate (TTRSR / shuffle-selectivity /
pos-token-acc), rendered via `src/talens/report.py` (`token_id_readout`, `canonical_bits("club", …)`).

Provenance JSONs under `results/`; per-run logs in `research-wiki/experiments/`. All numbers verbatim
from those files; bits never rescaled (legibility handled by `report.format_bits`).

---

## R1 — L0 exact-Bayes vs ridge under input-DP  (recovery: `results/l0_fast.txt`; bits: `results/b2_l0_bayes.json`)

gemma-2-2b, d=2304, N≈1945 test (recovery run N=7000), pool=2048, vocab-disjoint. r = σ√d / C.
**Eval type (integrity):** `synthetic_proxy` — the stored L0 run samples token ids from a **Zipf
distribution over the REAL embedding table** (not a natural-text corpus); the recovery is genuine
exact-match against those ids, but the prior is synthetic. The L0 claim is a clean channel
proof-of-principle, not a natural-text recovery rate.
**Provenance note (integrity):** recovery rates (ridge / Bayes-NN TTRSR / uplift) are the **post-bugfix**
run `results/l0_fast.txt` (the fix restored a pool-truncation that had dropped large-valued true ids,
clean 0.616→1.000). CLUB / capPVI **bits** are from the prior `results/b2_l0_bayes.json` run; the bug
affected only the *attack* recovery, not the *probe* (bits) — so mixing the two for this row is sound.

| ε (r)        | recovery: ridge TTRSR | recovery: **Bayes-NN TTRSR** | uplift | bits: CLUB I(rep;tok) | capPVI-acc (readout) |
|--------------|-----------------------|------------------------------|--------|-----------------------|----------------------|
| ∞ (0.00)     | 1.000                 | 1.000                        | +0.000 | 3084 bits             | 0.981                |
| 512 (0.45)   | 0.993                 | 1.000                        | +0.007 | 2942 bits             | 0.977                |
| 256 (0.91)   | 0.202                 | 1.000                        | +0.798 | 2624 bits             | 0.935                |
| 128 (1.82)   | 0.020                 | 1.000                        | +0.980 | 1912 bits             | 0.736                |
| 96  (2.42)   | 0.008                 | 1.000                        | +0.992 | —                     | —                    |
| 64  (3.63)   | 0.002                 | 0.993                        | +0.992 | —                     | —                    |

- **Bayes-NN realizes the preserved information**: recovery ~1.0 to r≈2.4 while ridge collapses 50× by
  r=1.82. Bits (CLUB) decay only −38% over the same range — the information is *preserved*; ridge is
  information-inefficient. (At L0 the Bayes-NN is the exact channel optimum, no approximation slack.)
- **R1 is UPLIFT / ceiling-realization evidence, NOT re-correlation evidence.** Because Bayes recovery is
  pinned at ~1.0, its own Spearman-vs-probe in `b2_l0_bayes.json` is degenerate (`bayes_vs_club=0.0`,
  `bayes_vs_capPVI=0.0`); ridge there is still *positively* correlated with the probes. So R1 evidences
  the **attack-efficiency (Bayes-optimality) gap** — the optimal attack realizes information ridge misses.
  The **re-correlation** claim (a stronger attack tracks MI where the weak one anti-correlates) is carried
  specifically by **R5** (deep +0.83 vs ridge −0.09).
- **Honest limit:** L0 is the easiest layer (observation ≈ noised embedding, exact table known). The hard
  case is L>0 (noise propagated through nonlinear blocks); NN-to-table no longer applies there.

## R2 — L>0 MLP decoder vs ridge under AT-LAYER Gaussian noise  (`results/b2_lpos_decoder.json`)

Cached resid_post L5/12/20, in-memory Gaussian noise (level = σ/act-RMS), shuffle-control selectivity.

| L  | level | recovery: ridge sel | recovery: dec-CA sel | uplift-sel | bits: CLUB | capPVI-acc |
|----|-------|---------------------|----------------------|------------|------------|------------|
| 5  | 0.0   | +0.779              | +0.556               | **−0.223** | 2959 bits  | 0.837      |
| 5  | 1.5   | +0.167              | +0.091               | −0.076     | 617 bits   | 0.420      |
| 12 | 0.0   | +0.704              | +0.406               | **−0.298** | 2902 bits  | 0.789      |
| 12 | 3.0   | −0.002              | −0.014               | −0.011     | 164 bits   | 0.160      |
| 20 | 0.0   | +0.720              | +0.495               | **−0.225** | 3043 bits  | 0.838      |
| 20 | 3.0   | +0.077              | +0.030               | −0.047     | 375 bits   | 0.237      |

- **NEGATIVE for the MLP** (loses to ridge at every depth/level): at depth the clean embedding is not
  directly observable, ridge's closed-form linear map already captures resid→embedding, a vanilla
  250-epoch MLP does not beat it. Beating ridge at depth needs a genuinely stronger decoder.
- **KEY (defines the regime):** under at-layer noise **both attacks track the MI probes perfectly** —
  Spearman(selectivity, CLUB) = Spearman(·, capPVI) = **1.00** at L5/12/20. **Ridge does NOT decorrelate
  here.** So the B3 ridge↔MI decorrelation is *not* a generic property of ridge under noise.

## R3 — MDL/SDL probe completes {PVI, CLUB, MDL} at-layer  (`results/mdl_probe_check.json`)

Cached resid L12, at-layer Gaussian sweep, MDL surplus-description-length with shuffle selectivity.

| level | bits: MDL-SDL sel | recovery: ridge sel | capPVI-acc | bits: CLUB |
|-------|-------------------|---------------------|------------|------------|
| 0.0   | +13898 bits       | +0.704              | 0.789      | 2902 bits  |
| 0.75  | +779 bits         | +0.141              | 0.484      | 834 bits   |
| 1.5   | +134 bits         | +0.060              | 0.260      | 383 bits   |
| 3.0   | +166 bits         | −0.002              | 0.160      | 164 bits   |

- Spearman(MDL-SDL sel, ridge-sel) = (·, capPVI) = (·, CLUB) = **+0.80**. **All three named probes
  (PVI, CLUB, MDL/SDL) track recovery under at-layer noise** (ρ 0.80–1.0). CLUB/capPVI smoother (ρ=1.0);
  MDL noisiest + 6–7× cost → report CLUB/accuracy primary, MDL auxiliary.

## R4 — channel-aware decoder vs ridge under PROPAGATED input-DP  (`results/b2_propagated_dp.json`)

Embedding-DP hook → forward → capture resid @L20, gemma-2-2b, ε∈{∞,1024,512,256}, vocab-disjoint.
**This is the open regime** (B3 L20: ridge decorrelates from MI).

| ε    | L  | recovery: ridge sel | recovery: decoder sel | uplift-sel | bits: CLUB | capPVI-acc |
|------|----|---------------------|-----------------------|------------|------------|------------|
| ∞    | 20 | +0.546              | +0.479                | −0.066     | 2525 bits  | 0.668      |
| 1024 | 20 | +0.459              | +0.484                | **+0.025** | 2612 bits  | 0.693      |
| 512  | 20 | +0.405              | +0.475                | **+0.070** | 2835 bits  | 0.825      |
| 256  | 20 | +0.089              | +0.232                | **+0.143** | 1777 bits  | 0.527      |

- **Decoder increasingly BEATS ridge as noise grows** (uplift −0.07→+0.14). Re-correlation
  Spearman(sel, capPVI): decoder 0.80(L12)/0.40(L20) vs ridge 0.40/0.20. Reverses the R2 negative —
  the decoder's advantage is **propagation-specific**. Suggestive (4 ε, 1 seed).

## R5 — strong/Vec2Text-style decoder under PROPAGATED input-DP @L20  (`results/b6_strong_decoder.json`)

ridge / 1-shot MLP / deep MLP (capacity control) / Vec2Text-style iterative corrector (T=1,2,3);
6 ε ∈ {∞,1024,768,512,384,256}, shuffle floors ≈ chance, WEIGHTS-PUB. **Headline run (single seed —
no robustness claim; multi-seed CIs are the named firm-up).** Shuffle floor computed once on clean
activations and reused across noise levels (disclosed scope assumption, not a per-noise control).

| ε    | ridge | mlp   | deep  | iter T1=T2=T3 | bits: CLUB | capPVI-acc |
|------|-------|-------|-------|---------------|------------|------------|
| ∞    | 0.516 | 0.337 | 0.345 | 0.364         | 2523 bits  | 0.668      |
| 1024 | 0.507 | 0.355 | 0.359 | 0.404         | 2610 bits  | 0.693      |
| 768  | 0.483 | 0.379 | 0.402 | 0.422         | 2659 bits  | 0.741      |
| 512  | 0.470 | 0.401 | 0.405 | 0.409         | 2834 bits  | 0.825      |
| 384  | 0.349 | 0.372 | 0.356 | 0.393         | 2767 bits  | 0.787      |
| 256  | 0.167 | 0.236 | 0.221 | 0.235         | 1780 bits  | 0.527      |

- **HEADLINE re-correlation:** Spearman(selectivity, capPVI) over ε — **deep +0.83, iterative +0.71,
  ridge −0.09**; vs CLUB: iter +0.71, ridge −0.09. The trained decoder's recovery **tracks the MI probes
  where ridge ANTI-correlates** (reproducing B3 L20). A stronger attack restores the MI↔recovery
  correlation the weak ridge breaks.
- **Uplift crossover:** decoder beats ridge at high noise (ε≤384, Δ +0.044/+0.068); ridge wins clean.
- **Iteration null (C7):** `iter_T3 − iter_T1 = +0.000`, `iter − deep = +0.023`. Pure embedding-space
  Vec2Text iteration adds nothing (fixed function of Y); capacity ≈ MLP. Faithful Vec2Text needs the
  forward model in the loop → R6.

## R6 — forward-model-in-loop Vec2Text @L20  (`results/b6c_forward_model.json`)

Re-embed decoder-seeded top-k=16 candidates through the actual model (clip-only), match to observed
noised resid Y_obs; teacher-forced prefix; 400 scored positions, ε∈{∞,512,256}.

| ε   | recovery: ridge | recovery: decoder top1 | recovery: **FMV** | uplift FMV−ridge | uplift FMV−dec |
|-----|-----------------|------------------------|-------------------|------------------|----------------|
| ∞   | 0.212           | 0.380                  | **0.738**         | **+0.526**       | **+0.357**     |
| 512 | 0.489           | 0.431                  | 0.495             | +0.006           | +0.064         |
| 256 | 0.227           | 0.245                  | **0.025**         | −0.202           | −0.220         |

- **Closes the low-noise gap** (+0.53 over ridge at clean) but **noise-fragile** (collapses below ridge
  at ε=256 — matches a clean reference to a single heavily-noised draw). Mirror image of R5's decoder.
- **No single attack dominates the noise range** → optimal attack is **regime-dependent**: forward-model
  match (low noise) ⊕ learned denoiser/decoder (high noise). The named optimum is a **noise-aware FMV**
  (denoise Y_obs / match E[Y|cand]) — the open frontier.

---

## Synthesis (the surface's thesis)

The **MI↔recovery decorrelation is not universal — at depth it is specific to noise PROPAGATION**:
1. **At-layer noise (R2/R3):** even weak ridge tracks MI (ρ=1.0); all three probes ρ 0.80–1.0. No gap.
2. **L0 input-DP (R1) — uplift / ceiling-realization, NOT decorrelation:** the Bayes-NN attack realizes
   the MI-preserved information (+0.98 uplift); ridge's *recovery* collapses but ridge still tracks the
   probes positively (+0.76..+0.88) — an information-efficiency gap, not anti-correlation.
3. **Propagated input-DP @depth (R4/R5):** ridge collapses AND decorrelates from MI (−0.09); a stronger
   trained decoder re-correlates (**+0.83**) — the MI-recovery gap is the **Bayes-optimality gap**
   (`claim:bayes-gap-diagnosis`), closed in the right direction by attack strength.
4. **Open (R6):** no single attack dominates the full noise range; the named optimum (noise-aware FMV /
   stronger depth decoder for propagated-DP) is **not yet built** — the live frontier.

**Keeper claim:** `claim:restore-correlation` — info-efficient attacks restore the MI↔recovery
correlation that weak attacks break. Two distinct evidence types: **R1 = uplift / ceiling-realization**
(Bayes-NN +0.98 over ridge — the attack-efficiency gap), **R5 = re-correlation** (deep decoder +0.83 vs
ridge −0.09 across the ε sweep). Do not conflate them: only R5 is literal MI↔recovery re-correlation.
**Backbone (already verified):** `claim:thm-t1-info-efficient` (T1: Bayes-optimal dominates lossy/linear;
recovery monotone in MI on the Gaussian arm).
**Dead-ends (first-class):** R2 MLP-at-depth-under-at-layer-noise (loses to ridge; defines the regime),
R5-C7 embedding-space iteration null, per-position Vec2Text wrong-surface (B7, see negative-result log).

---
type: research
status: current
created: 2026-06-20
updated: 2026-06-20
tags: [idea-discovery, leakage-measure, MDL, SDL, PVI, CLUB, inversion-attacks, calibration, independence]
companion: [interpretability-leakage-bridge, mdl-vinfo-inversion-toolkit]
---

# Idea Discovery Report — an independent, faithful leakage predictor

**Direction**: IT + interpretability prediction of private-inference attack success
under defenses (DP, static obfuscation, split inference). **Blocker**: find a leakage
measure that is *both* independent of the attack *and* a faithful predictor of
token-recovery (TTRSR). class-PVI is independent but overfits (high-d); retrieval-PVI
tracks the attack only because it *is* the attack in bits; CLUB is the closest but is
upper-bound-only and shares the embedding target.

**Pipeline**: research-lit (delta) → idea synthesis → novelty/scoop check → (pilot) → experiment plan

---

## Executive summary

The standing objective is **to fix the independent family we already have — class-PVI**
— so it tracks the attack, *or* to **formally, decisively** show the token-id
V-information family is the wrong methodology. We do **not** pivot to a new measure to
dodge the overfit. class-PVI is the only family that is *both* independent of the attack
*and* cheap; retrieval-PVI is the attack, CLUB is upper-bound-only. There is also a hard
**cost constraint**: the measure probe must stay cheap enough for fast ablation (PVI is
already ~56–59% of every block — per `results-chronicles.md`), so the fix must keep or
**reduce** probe cost.

> **Recommended thesis (Idea 1, framed by Idea 7):**
> class-PVI overfits because of an **estimator-validity regime** (`capacity ≫ data`,
> `d > n_val`), not because the token-id family is wrong. A **capacity-matched** member
> of the *same independent family* (PCA→dim<n_val, control-anchored regularisation, or a
> non-parametric kNN/Gaussian class-conditional that needs *no* iterative fit) is both
> **cheaper** and **well-posed**, and is hypothesised to track TTRSR across the ε-sweep
> and control sweep where the unconstrained free softmax fails. If no capacity-matched
> member can track the attack, the deliverable flips to a **formal impossibility result**
> for the token-id V-family (the `d>n_val` ill-posedness made rigorous).

> **Caveat on MDL/SDL (de-prioritised):** `mdl.py`'s `online_code_length` uses the
> *same* `train_softmax_probe` family, so on small prequential prefixes it inherits the
> *same* `capacity ≫ data` overfit — and costs ~6–7 fits vs class-PVI's 1, violating the
> cost constraint. It is at most a **robustness check**, never the headline. (This
> corrects an earlier draft that wrongly crowned MDL/SDL.)

---

## The 2×2 that organises everything (analysis of the blocker)

Every measure is characterised by **(axis A) what it targets / what family estimates it**
— which fixes its *independence from the attack* — and **(axis B) is the estimator
capacity-matched to the data regime** — which fixes its *validity/faithfulness* when
`d > n_val`. (Note: this is **not** a "point-estimate vs MDL" axis — MDL lives in the
unconstrained column too, since it reuses the same free-softmax probe.)

| | **Unconstrained capacity** (free `d→C` softmax; class-PVI *and* MDL/SDL) | **Capacity-matched** (PCA<n_val / kNN / Gaussian / random-proj) |
|---|---|---|
| **Token-id target** (independent of attack) | **class-PVI** — independent ✔, but **overfits** (d>n_val) → ✗; MDL inherits this + 6–7× cost | **★ capacity-matched class-PVI — EMPTY CELL = the target measure (cheaper *and* valid)** |
| **Embedding target** (shared with attack) | **retrieval-PVI** — faithful but **mechanical** (= attack) ✗ | (capacity-matching an embedding family still shares the attack map) |
| **Variational MI estimator** | **CLUB** — independent estimator ✔, faithful-ish, **upper-bound only**, shares *target* | (MINE/InfoNCE lower bound — not yet built) |

**Reading**: independence lives in the *token-id target* row (class-PVI's family).
Validity-under-stress lives in the *capacity-matched* column. We have only ever occupied
the **unconstrained** cell of the token-id row (class-PVI) — exactly the cell that
overfits when `d > n_val`. The **capacity-matched token-id cell is empty, and filling it
is both the fix and a cost reduction** (smaller d / non-parametric → cheaper than the
full free-softmax fit).

Root cause (2026-06-18 handoff, `/diagnose`): `d=2304` free softmax, `d > n_val (~400)`
→ the probe memorises the *validation* set, early-stopping is fooled, held-out logprob
→ −∞, shuffle floor ≈ −48 b, PVI non-monotonic. The fix is to make the estimator
**well-posed in this regime** — bound capacity below `n_val`, or use a self-regularising
non-parametric family — and to **anchor regularisation on the shuffle-control floor
(≈ 0), not the (overfit) val-CE.**

---

## Literature landscape (2026-06-20 delta over the 2026-06-15 survey)

The broad field was mapped in `interpretability-leakage-bridge.md` /
`mdl-vinfo-inversion-toolkit.md`. This pass verified the open gaps still stand and
found the concurrent work that defines the contribution boundary.

**Gaps confirmed open (no paper found):**
- **G1** — MDL / SDL / ε-sample-complexity used to *predict or rank* inversion-attack
  recovery. Clean. (Voita–Titov MDL stays a probe-quality tool; Whitney SDL has no
  privacy follow-on.) **→ Idea 1.**
- **G2** — one IT measure *calibrated/transferable across* DP + obfuscation +
  split-depth. Defense benchmarks evaluate many defenses, but each *measure* is
  validated against one. **→ Idea 4.**
- **G3** — quantitative degradation theory for V-info/PVI under the Gaussian/DP
  mechanism (usable info is not bijection-invariant; noise destroys it, but no
  closed degradation curve exists). **→ Idea 5.**
- **G4** — a *formal* definition of estimator↔attack independence; the informal
  "attack-independent (IT) vs empirical (attack-based)" dichotomy is standard but
  never formalised as "the estimator is not a reparameterisation of the attack map."
  Closest formal link: de Chérisey et al. (Inf. Sci. 2019) MI↔success-rate bound in
  side-channels. **→ Idea 7.**
- **G1b** — no *two-sided* (upper+lower) bracket on token recovery for
  activations/embeddings (Fisher gives lower only; PML/α-leakage give upper). **→ Idea 3.**

**Scoop risk / prior art to position against (must-cite):**
- **PAF — "What Does the Server See?"** (arXiv:2605.23158, 2026): Jacobian/Taylor
  layer-sensitivity proxy, validated by *Pearson correlation* (R −0.59…−0.81) vs
  ROUGE-L of a reconstruction attack. Same *shape* of claim ("cheap attack-independent
  proxy predicts reconstruction") but **geometric, not IT, and correlation not
  calibration.** This is the paper most likely cited as prior art.
- **Jacobian Rank-Recovery** (icomputing.0270, 2026): explicitly markets an
  "attack-independent" vulnerability proxy for MIA. Same competitive position.
- **FSInfo** (arXiv:2504.10016, 2025): Fisher-Approximated Shannon Info, per-layer,
  training-free, **lower-bound only, obfuscation-family only, no cross-defense
  calibration.** The must-cite IT prior art to differentiate against.

**Our defensible edge** = (i) the predictor is **information-theoretic** (MDL/SDL),
(ii) we deliver **calibration**, not correlation, (iii) we show **cross-scheme
transfer** (G2), and (iv) we **formalise** the independence/faithfulness frontier (G4)
that none of PAF/Rank-Recovery/FSInfo state.

Adjacent tools the plan can borrow: PML / α-leakage (Saeidian 2023; arXiv:2405.00423)
as the **upper-bound** partner; Fisher/CRB (FSInfo) as the **lower-bound** partner for
the bracket; "Lower Bounds on Bayesian Risk via Information Measures" (arXiv:2303.12497)
as the vehicle to turn a measure into a recovery-error bound (G3/G1b).

---

## Ranked ideas

### 🏆 Idea 1 — Capacity-matched class-PVI: fix the independent family (cheaper *and* valid) — **RECOMMENDED / PRIORITY**
- **Claim**: class-PVI's failure is the `d>n_val` estimator regime, not the token-id
  family. A capacity-matched member of the *same* family — (a) **PCA→dim < n_val** then
  linear softmax, (b) **control-anchored regularisation** (pick dim/l2 so the *shuffle
  floor ≈ 0*, not by overfit val-CE), (c) **non-parametric kNN / Gaussian
  class-conditional** (self-regularising, *no* iterative fit), (d) random-projection +
  linear readout — tracks TTRSR across the ε-sweep and control sweep, with shuffle floor
  ≈ 0 and monotone-under-noise.
- **Cost**: each candidate is **≤** current PVI cost (smaller d / closed-form generative
  → *cheaper*), satisfying the fast-ablation constraint — the opposite of MDL.
- **Novelty**: G2/G4 framing; the capacity-matched-V-info-as-leakage-predictor angle is
  open. **Pilot is model-free** on the cached capture (`diag_pvi.py` loop already does
  the class-vs-retrieval, l2, and noise sweeps).
- **Decisive negative branch**: if *no* capacity-matched member tracks the attack while
  staying independent, the deliverable flips to **Idea 1′ (formal impossibility)** — see
  below. Either outcome closes the standing question.

### Idea 1′ — Formal verdict on the token-id V-family (the "decisively conclude" path)
- If empirics fail, make the failure rigorous: characterise the **estimator-validity
  condition** for V-information with an unconstrained family in `d>n` (when is `I_V`
  identifiable / its plug-in estimator consistent?), and the **bias floor** that makes a
  token-id classifier unable to track an embedding-geometry attack. A clean negative is
  publishable and *required* before abandoning the family. Anchor: McAllester–Stratos
  O(ln N) MI-lower-bound limit; de Chérisey et al. MI↔success-rate relation.

### Idea 7 — Formalise the independence×validity frontier (the paper's spine, not standalone)
- Define **attack-independence** (estimator is not a reparameterisation of the attack's
  map/target) and place class-PVI / capacity-matched-PVI / retrieval-PVI / CLUB on the
  2×2. Conceptual contribution (G4) and the language to beat PAF/Rank-Recovery.

### Idea 4 — Cross-scheme calibration: one measure, three defenses (the validation contribution)
- Once a working independent measure exists, show a single calibration map
  (measure → TTRSR) holds across **DP (have runner), static obfuscation, split-depth**.
  G2 — clearest white space, makes it a paper not a metric note. **Cost**: needs
  obfuscation + split-depth runners (DP runner exists); medium compute. Gated on Idea 1.

### Idea 3 — Two-sided leakage bracket: CLUB/α-leakage (upper) + Fisher-CRB / MINE (lower)
- Bracket recovery from both sides; **bracket width = a calibrated confidence signal**
  (respecting McAllester–Stratos O(ln N)). G1b. Medium cost; complements (does not
  replace) the PVI fix.

### Idea 5 — PVI/usable-info-under-DP degradation theory
- Closed(-ish) degradation curve for usable info under the Gaussian mechanism (G3);
  best delivered as a *theorem + fit* inside Idea 1/1′, not standalone.

### Idea 6 (parked) — Pointwise Maximal Leakage / α-leakage on activations
- Operational per-instance leakage on neural reps (G4-neural, novel). Doubles as Idea 3's
  upper bound. Park until the PVI question is resolved.

### Idea 2-MDL (demoted to robustness check) — Description-length read of the same family
- MDL/SDL on token-id, *only* as a cross-check on whether the data-axis read is any more
  stable than the point estimate — **flagged** as same-probe-family (inherits the
  overfit) and 6–7× costlier. Not on the critical path.

### Idea 8 (parked) — Superposition / SAE effective-DoF ↔ invertibility
- arXiv:2512.13568 shows the capacity↔vulnerability link is regime-dependent (not a clean
  law). Park unless a clean signal appears.

---

## Recommended integrated scope (what the experiment plan should target)

**Resolve the standing question first.** Headline = **Idea 1 (capacity-matched class-PVI)**
— make the independent family track the attack while *cutting* probe cost — with
**Idea 1′ (formal verdict)** as the mandatory fallback so the family is either fixed or
decisively retired. **Idea 7** is the framing spine. **Idea 4 (cross-scheme calibration)**
and **Idea 3 (bracket)** are downstream sections, gated on a working independent measure.
Idea 5 is a theory nugget; Idea 2-MDL is a robustness check; Ideas 6/8 parked.

## Next steps
- [ ] Pilot Idea 1 on cached capture (model-free, `diag_pvi.py` loop): test the four
      capacity-matched variants (PCA<n_val, control-anchored l2, kNN/Gaussian, random-proj)
      vs class-PVI/retrieval-PVI/TTRSR under post-hoc noise — pass = shuffle floor ≈ 0,
      monotone, tracks TTRSR, ≤ current PVI cost.
- [ ] `/experiment-plan` built around Idea 1 (+ 1′ fallback, framed by 7; 4/3 downstream).

## Sources
Literature delta traces: see this session's two research sub-agents (2026-06-20).
Companions: `docs/research/interpretability-leakage-bridge.md`,
`docs/research/mdl-vinfo-inversion-toolkit.md`,
`docs/handoffs/2026-06-18-independent-vfamily-attack-correlation.md`.

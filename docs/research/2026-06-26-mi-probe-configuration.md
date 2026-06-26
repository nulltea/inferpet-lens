---
type: research
status: current
created: 2026-06-26
updated: 2026-06-26
tags: [mi-probe, club, pvi, v-information, mdl, configuration, dp, pythia, l20-peak, voita]
companion: scripts/evals/dp_leakage_sweep.py
---

# Correct configuration of the token-MI probes (CLUB · PVI · MDL) for the DP×depth study

**Question.** What is the most representative, well-posed configuration and training set for our three
`MI(rep; token)` probes — CLUB, PVI (V-information), MDL — now that we are moving to a small frozen
model (Pythia-160M, `d=768`, vocab ≈50k), and does the information peak survive scrutiny? ("L20" below
refers to the **previously observed Gemma-2-2b L20 peak**; Pythia-160M has only 12 blocks, so on Pythia
the object is the *analogous mid/deep-layer peak*, to be confirmed by the diagnostic run.)

**Method.** Primary sources read directly (EdgeQuake-ingested PDFs + arXiv HTML), not paraphrased from
memory: CLUB (Cheng et al. 2020, arXiv:2006.12013); V-information (Xu et al. 2020, arXiv:2002.10689);
PVI (Ethayarajh et al. 2022, arXiv:2110.08420); MDL probing (Voita & Titov 2020, arXiv:2003.12298);
control tasks (Hewitt & Liang 2019, arXiv:1909.03368); depth mechanism (Voita et al. 2019,
arXiv:1909.01380 "Bottom-up Evolution"); and the DP-inversion frame (arXiv:2507.16372).

## Definitions (glossary)

- **rep**: the residual-stream vector at layer L for one token position (the released surface).
- **token-id**: discrete token identity (class). **token-embedding**: the continuous input-embedding vector.
- **MI(A;B)**: Shannon mutual information. **I_V(X→Y)**: predictive V-information (family-restricted).
- **PVI**: pointwise V-usable information, `pvi(x→y)=−log₂ g[∅](y)+log₂ g'[x](y)`.
- **MDL (online)**: prequential code length of labels given reps; `compression = uniform_code/online_code`.
- **selectivity**: metric(real labels) − metric(control/shuffled labels) — the Hewitt–Liang floor subtraction.
- **optional ignorance**: a predictive family V must contain, for every model, one that ignores the input.
- **channel-matched**: the probe is trained on samples from the *same* noised channel (ε) it is evaluated on.

## 1. The three probes measure three different quantities

| Probe | Target | Type | Estimator direction |
|---|---|---|---|
| **CLUB** | `I(rep ; token-embedding)` | continuous↔continuous | **upper** bound (loose: 2–4×) |
| **PVI / V-cap** | `I_V(rep → token-id)` | continuous↔discrete | family-restricted (achievable) |
| **MDL** | code length of `token-id` given rep | continuous↔discrete | achievable (prequential) |

Only PVI and MDL target the **token identity**; CLUB targets the embedding vector (a continuous proxy
ridge also regresses to). None is "the MI" — each is **estimator-/family-dependent**, which is exactly
why configuration determines whether the number is representative.

## 2. What each paper requires for the estimate to be valid (sourced)

### CLUB — valid only when the variational `q_θ(y|x)` approximates `p(y|x)` well
- vCLUB `= E_p(x,y)[log q_θ(y|x)] − E_p(x)E_p(y)[log q_θ(y|x)]` (eq. 15).
- **Theorem 3.2 / Corollary 3.3:** vCLUB is a genuine upper bound **iff** `KL(p(x,y)‖q_θ) ≤
  KL(p(x)p(y)‖q_θ)`; otherwise it is an *estimator* with error `≤ KL(p(y|x)‖q_θ)=ε`. So an
  **under-trained or mis-specified `q` is not even a bound**, and an **over-expressive `q` fit on few
  rows inflates** the value. The paper attains the bound by training `q` to convergence as `n→∞`.
- Toy config: `q_θ=N(μ_θ(x),σ²_θ(x)I)`, single hidden layer, ReLU, lr 5e-3, batch 64, 4000 steps; CLUB
  has the lowest bias among upper bounds, CLUBSample higher variance.
- **Implication for us:** CLUB's *magnitude* is trustworthy only when `q≈p(y|x)`; otherwise report
  **rank only**. Our current eval (`club_max_rows=600`, default `hidden_size=128` ⇒ branch `768→64→768`,
  ≈198k params across `p_mu`+`p_logvar`, 400 steps, weight_decay 1e-4) trains ≈198k params on ≈420 rows
  — far from the convergence regime → magnitude unreliable, and the L20 value is a prime suspect for
  inflation.

### PVI / V-information — the family choice *is* the measurement
- `I_V(X→Y)=H_V(Y|∅)−H_V(Y|X)`. The family V must satisfy **optional ignorance** (Def. 1); else
  V-information can be **negative** or **>0 under independence** (Prop. 2 converse). Our class-prior null
  satisfies it. ✓
- **Monotonicity (Prop. 2.1):** `V⊆U ⇒ H_V(Y)≥H_U(Y)` and `H_V(Y|X)≥H_U(Y|X)` — the conditional and the
  null term each shrink with a bigger family, but *separately*; "a bigger family extracts more
  V-information" follows cleanly only when the null/prior term is fixed or equally well modeled across
  conditions. Either way the practical consequence holds: **V must be held fixed across every (depth, ε)
  cell** or cross-condition comparison is meaningless.
- **Data-processing inequality is violated** (Xu §3.2): representation/processing can *create* usable
  information. Good for representation learning, but means the probe family defines the answer.
- **PAC (Thm 1):** one-sided, `I_V − Î_V ≤ 4·R_|D|(G_V) + 2B√(2log(1/δ)/|D|)` (a bound on the gap, not a
  symmetric estimation error). Complex V ⇒ overfit; simple V ⇒ underfit.
- **Two distinct estimands — do not conflate:** Prop. 1.5's `I_V = R²·tr(Cov Y)` (linear-Gaussian V) holds
  for **continuous** `Y ∈ ℝ^d`, so it is **`V_Gauss(rep → embedding)`** (the ridge-regression channel),
  *not* class PVI. **`PVI_class(rep → token-id)`** is a discrete softmax/feature-family estimand. They
  answer different questions; the report keeps them separate below.
- **PVI is more overfit-sensitive than accuracy** (Ethayarajh): the model "becomes less certain about the
  correct label long before predicting the wrong label," so conditional entropy (bits) rises while
  accuracy holds. **Report the accuracy readout beside the bits** (matches our `capacity-pvi-findings`).

### MDL — the most configuration-robust of the three
- Online (prequential) code: train probe on increasing data prefixes, pay the next block's CE; sum =
  online codelength = area under the learning curve. Variational and online codes **agree**.
- **Codelength is stable across probe size** in their setup (Table 8: MLP-1/MLP-2, h∈{50…1000}) —
  *unlike accuracy*. So in their PoS/control-task setting MDL needs little probe-size tuning to separate
  real from control, whereas Hewitt–Liang selectivity *does*. This makes MDL **expected to be the least
  config-sensitive** of the three — but their result is for low-cardinality linguistic tasks, not
  high-cardinality token-id or retrieval-MDL, so **verify with a probe-size sweep** rather than assume it.
- Control task (Hewitt–Liang): random word-type→label; linguistic task compresses far more than control.

### Control tasks (Hewitt–Liang) — the floor every fitted probe needs
- Control task = random labels by word type; **selectivity = real − control**. High-capacity probes lose
  selectivity (they fit the random control too). The fix is either capacity control (their route) or MDL
  (Voita's route, which makes selectivity automatic).

## 3. Q4 — what is special about L20, and does the literature corroborate?

**Voita et al. 2019 (Bottom-up Evolution)** measures `MI(rep_layer ; input-token-identity)` directly:
- **LM (autoregressive):** declines **monotonically** with depth — "information about the past gets lost,
  predictions about the future get formed."
- **MLM:** **non-monotonic** — context-encoding phase (token identity *forgotten*) then **token
  reconstruction at the top layers** (identity recreated). Their Fig. 8/12 t-SNE of `{is,are,was,were}`:
  **mixed** in LM/MT layers, **disambiguated** only for MLM at the top.

**gemma-2-2b and Pythia-160M are decoder (autoregressive) LMs**, so Voita's LM curve gives a *prior* that
token-identity information **declines with depth** (no top-layer recreation, which is the MLM behavior).
Our **token-class separability declining monotonically with depth** (the `sep` probe) matches that prior.

**External-validity caveat (important).** Voita 2019 used Transformer-base-style 6-layer models on WMT
data, *without* injected noise. Extrapolating to a 12/26-layer modern decoder under input-DP is not
guaranteed: residual streams, tied embeddings, copy behavior, and DP-noise propagation could all bend the
trajectory, and "declines" need not mean strict monotonicity. So Voita is **evidence, not proof**.

**Conclusion (appropriately hedged).** The Voita prior plus the matching `sep` decline make a
*token-identity* reading of the L20 peak **unlikely**, but do **not rule it out** — by definition, if
fixed-family PVI/MDL **selectivity** (real − control) still peaks at L20 under controls, that *is*
token-id usable information. The peak must therefore be **adjudicated**, not declared, by: (i) fixed-family
selectivity across depth, (ii) the type-level/row controls, and (iii) the separability cross-check. The
two live hypotheses for a non-token-identity peak remain (a) the **DP-noise×depth interaction** (deeper
layers partially outrun input-injected noise) and (b) a **probe artifact** (CLUB inflation / PVI overfit /
fixed-capacity window). **arXiv:2507.16372** ("Depth Gives a False Sense of Privacy") corroborates only the
weaker claim — inputs remain recoverable at *all* depths and deep layers need a *different* attack
(SVD-basis / generation), with **no reported peak layer**.

## 4. Conclusion — the most representative, well-posed configuration

> **This is the *target* protocol, not what `dp_leakage_sweep.py` runs today.** Current gaps: probes are
> called on the victim capture only (then split internally); **probe controls/selectivity are not wired
> yet** (records carry *attack* selectivity, not CLUB/PVI/MDL selectivity); `probe_vcap` uses
> `family="pca_softmax", dim=64`; and `probe_mdl` uses the **retrieval** family (circular with ridge — see
> open follow-ups). The per-layer split confound *has* been fixed (split/pool computed once across layers).

**Shared protocol (all three probes):**
1. **Channel-matched, independent-draw:** train on the adversary noised draw, evaluate on an
   *independent* victim noised draw at the same ε (the two-seed capture already in `dp_leakage_sweep.py`).
   Training on plaintext and testing on noised is a *different* (covariate-shifted) measurement — do not.
2. **Split type is forced by the probe family — they cannot all share the attack's split:**
   - **Closed-set class PVI / class-MDL** assign one weight vector per token-id, so they **cannot score a
     token-id unseen in training** — vocab-disjoint is *impossible* for them. They must use a **row-split
     (shared class set)**, with the **type-level control task** (Hewitt–Liang: random label per word
     *type*) supplying the memorization floor instead of disjointness.
   - **CLUB (regress→embedding) and retrieval-MDL** are open-set (decode in embedding space), so they
     **can** use the attack's **vocab-disjoint** split and thereby match ridge's generalization — but that
     is also exactly what makes retrieval-MDL circular with ridge. *Matching the split forces matching the
     mechanism*; there is no free alignment.
3. **Fixed predictive family across every (depth, ε) cell** — monotonicity (Prop. 2.1) makes any
   family/dimension change confound the cross-condition comparison.
4. **Two controls, two jobs — don't conflate them** (especially for token-id, where a random label *per
   word type* is itself a function of token identity, so a type-control floor can subtract the very signal
   you measure): use a **row-shuffle/permutation** control as the **independence / finite-sample floor**
   (the main selectivity floor: CLUB→0 at independence, PVI→0), and the **type-level** control only to ask
   the *separate* question "beyond arbitrary lexical memorization." The probes already support
   `control="shuffle"`; the sweep must pass it.
5. **Report bits AND the readout** (PVI accuracy / MDL compression / CLUB rank) — PVI/CLUB bits are fragile.

**Per-probe configuration on Pythia-160M (`d=768`, vocab≈50k):**

| Probe (estimand) | Family / target | Split | Sample regime | Trust |
|---|---|---|---|---|
| **CLUB** = `I(rep; embedding)` ↑bound | Gaussian `q_θ(emb|rep)`; **train to convergence**, raise `max_rows` (≫600), tune `h`+weight_decay so `q≈p(y|x)` | vocab-disjoint OK (open-set) | rows for `q` to converge without overfit (~tens of thousands) | **rank only**, never magnitude (Thm 3.2 caveat) |
| **PVI_class** = `I_V(rep→token-id)` | fixed-complexity softmax on a **fixed feature reduction** (PLS/linear), *not* unsupervised-PCA-then-softmax (variance-truncation); C bounded by samples (Zipf ⇒ ≈256) | **row-split** (closed set) + type control | PAC: complexity ≪ `|D|`; ~50k rows | watch **bits-vs-accuracy** divergence |
| **V_Gauss** = `R²(rep→embedding)` | linear-Gaussian (Prop. 1.5) — the ridge channel | vocab-disjoint OK | ~thousands rows | matches ridge ⇒ near-circular; use as reference |
| **MDL** = code-len(token-id|rep) | class-probe online code, C≈256 | **row-split** + type control | ~tens of thousands rows | **expected** least config-sensitive; **candidate headline after a probe-size sweep** |

Report **class coverage** for the class probes (PVI_class, MDL): #classes, fraction of rows kept after the
top-C cut, the token-frequency mass covered, and confirm the **same class set is fixed across all
(depth, ε) cells** — top-C restricts the estimand to frequent-token usable information, which matters when
comparing to the ridge/retrieval pool.

**Why the small model was still the right move** (it did *not* lift the class-softmax C ceiling — that is
Zipf-structural): (a) `d=768` halves the covariance/`q` conditioning vs gemma's 2304, directly reducing the
CLUB-inflation and any geometry-probe estimation bias; (b) the 50k English-centric vocab makes the
candidate pool's distractors in-distribution (no CJK freebies) so recovery and the matched readout are
honest; (c) ~16× faster captures make the *multi-run diagnostic program* (control task, dim sweep,
multi-seed, two-seed) affordable. Capture **~1800–4000 prompts** — enough for all three at their
well-posed family sizes; spend surplus budget on **more seeds and ε points**, not more rows.

**Predicted outcome the config buys.** Under this protocol, if the L20 peak is real token information it
survives in **selectivity** across **fixed family/dim**; if it is an artifact it collapses — and either
way the result is interpretable against Voita 2019's LM baseline (token identity declines with depth).

## Open follow-ups
- Replace `probe_vcap`'s unsupervised PCA with a fixed-complexity linear/PLS family; pass a **type-level**
  `control` for class probes; report bits + accuracy.
- Make the probe split explicit per family: **row-split for class PVI/MDL** (with type-level control),
  **vocab-disjoint only for CLUB / retrieval-MDL**. Do *not* force class probes onto the disjoint split.
- Run the control-task + dim-sweep depth profile on Pythia-160M (ε grid that bites: inf,64,32,16,8).
- The retrieval-MDL variant is **circular with ridge** (handoff open-#3) — keep it only as a labeled reference.
- **Done this round:** the per-layer split/pool confound in `dp_leakage_sweep.py` is fixed (split + pool +
  shuffle computed once across layers).

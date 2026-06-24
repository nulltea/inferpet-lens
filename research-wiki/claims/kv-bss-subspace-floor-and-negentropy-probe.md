---
type: claim
node_id: claim:kv-bss-subspace-floor-and-negentropy-probe
name: "On plaintext KV/QKV (Identity, U=H) the BSS attack framing is ill-posed (no hidden mixing; the Hungarian-cosine recovery metric is subspace-membership-confounded): recovery does not accumulate with T, and a geometry-only negentropy probe predicts the genuine separation margin only once recovery is graded against a matched random-orthogonal-demixing floor"
description: ""
node_type: claim
status: verified
provenance: "refine-logs/kv-accumulation/RESULTS.md; analysis_b3.json; c2_robustness.json; scripts/spikes/kv_bss_analysis.py; src/talens/attacks/bss.py; src/talens/measures/bss_separability.py"
tags: ["kv-cache", "qkv", "bss", "ica", "jade", "joint-diagonalization", "subspace-membership-floor", "negentropy-probe", "matched-probe", "geometry-only", "accumulation", "weights-pub", "negative-result"]
date: 2026-06-24
added: 2026-06-24T00:00:00Z
---

# The BSS attack framing is ill-posed on plaintext KV/QKV (no hidden mixing; the recovery metric is subspace-membership-confounded); the matched negentropy probe predicts genuine separation only against the random-demixing floor

> Framing precision (per novelty review): classical ICA with mixing `A = I` is not *inherently*
> ill-posed — it is simply that there are **no hidden sources to recover** and the Hungarian-cosine
> *recovery metric* is confounded by subspace membership. We say "ill-posed" in exactly this sense:
> the **attack framing / recovery metric** on plaintext identity-mixing data, not ICA theory itself.

**status:** `verified` (structural core) — the load-bearing **structural** theory of Lemma L1
(parts 1–3: row-span invariance, the rotation-orbit-within-a-fixed-subspace property, and the
floor-mismatch identity) is proved inline below and checked by the cross-model jury. The
**magnitude** of the floor gap, the no-accumulation finding **C1**, and the probe-vs-recovery
correlation **C2** are **empirical** (the first two with identifiability rationales, C2 exploratory:
n = 9 cells, family granularity, permutation `p = 0.0013`) — reported as measurements, not derived in
closed form. An attempted analytic `1/√s` lower bound on the membership floor was rejected by the
jury and is deliberately not claimed.

## One-line statement
On plaintext KV/QKV activations under WEIGHTS-PUB (Identity transform, so the BSS mixing is the
identity and the observed operand `U` equals the true source rows `H`), every demixing — joint-diag
or random — produces an orthonormal basis of **one fixed subspace** `𝒮₀ = rowspan(Ũ)`; the
Hungarian-aligned cosine recovery metric therefore depends on the demixing only through a rotation
*within* `𝒮₀`, and its Haar-random-rotation expectation `F_in` is a subspace-membership baseline
**invariant to whether joint-diagonalization succeeded** (Lemma L1, proved). The shipped Gaussian-GT
floor measures a structurally different functional `F_out`, so grading recovery against it over-credits
the attack by the floor-mismatch bias `F_in − F_out`; the matched null is the random-demixing floor.
Empirically: `F_in ≈ 0.708 ≫ F_out ≈ 0.155`, leaving a genuine margin of only `0.027`. Two further
**empirical** findings: **(C1)** recovery does **not** accumulate with the number of observations `T`
(median slope `+0.009` per `log₂T`), consistent with the identifiability rationale that under `A = I`
there is no hidden mixing to accumulate; **(C2, exploratory)** the attack-independent negentropy probe
predicts the **genuine** separation margin (raw minus the matched random-demixing floor) at Spearman
`ρ = 0.92` (permutation `p = 0.0013`, n = 9), yet **anti-correlates** (`ρ = −0.43`) with the
uncorrected raw readout — the predictive correlation exists only against the correct floor, and only
at activation-kind-family granularity.

This baseline makes a future climb-with-`T` **under a mixing defense** (Task 2 KV-CLOAK, Task 5 GELO)
attributable to the defense's mixing/mask, not to BSS on the plaintext itself.

---

## Definitions and Setup

- **Surface / threat model.** Qwen3-4B activations, kinds `{kq, kqv_out, resid_post}`, layers
  `{0, 12, 20}`, dev-24 cached capture. WEIGHTS-PUB: adversary knows weights + embeddings. Transform
  = `Identity` (plaintext). Secret = the activation rows behind `kq`/`kqv_out`/`resid_post`.
- **BSS generative model.** Standard BSS assumes observations `Y_t = A·S` for a single **unknown,
  fixed** mixing `A ∈ ℝ^{s×s}` shared across observations `t`, with latent sources `S`. The attack's
  job is to estimate a demixing `B ≈ A⁻¹` (up to permutation/scale) so that `B·Y_t ≈ S`.
- **The plaintext operand.** For each prompt the capture yields an operand matrix
  `U ∈ ℝ^{s×T}` (`s` = subsampled activation dimensions, `s ≤ max_dim = 64`; `T` = features /
  positions, `T ≤ max_features = 256`) and the matched ground-truth source-row matrix
  `H ∈ ℝ^{s×T}`. Under Identity, **`U = H`** and the mixing is `A = I_s`. We assume `U` has full row
  rank `s` and `T ≥ 2s` (the regime the attack code enforces via `_subsample` / `_whiten`).
- **Row-span.** `𝒮 := rowspan(H) ⊆ ℝ^T`, a subspace of dimension `s`.
- **Whitening.** `_whiten(U, s)` returns `(Y, W)` with `Y = W·Ũ` (`Ũ` the centered `U`) and the
  whitened rows satisfying `(1/T)·Y·Yᵀ = I_s`. `W ∈ ℝ^{s×s}` is invertible.
- **JADE / JD demixing.** Both produce an orthogonal rotation `B ∈ O(s)` (from joint-diagonalizing
  whitened cumulants, `_joint_diag`); the recovered sources are `Ŝ = B·Y = B·W·Ũ`.
- **Recovery metric.** `p95cos(Ŝ, H)` := the 95th percentile of the Hungarian-aligned `|cosine|`
  between the rows of `Ŝ` and the rows of `H` (`_p95_cosine_with_hungarian`). A larger value reads
  as "more sources recovered".
- **Two candidate null floors.**
  - **Gaussian-GT floor** (shipped `bss.jd_floor`): replace `H`'s rows with independent Gaussian
    rows (random directions in `ℝ^T`, generically **outside** `𝒮`), keep the pipeline; p95 ≈ `0.155`.
  - **Random-demixing floor** (the matched null, `jade_proper_floor` / `jd_proper_floor`): replace
    the joint-diag `B` by a Haar-random `B ∈ O(s)`, keeping the same whitened data `Y` and the same
    Hungarian pipeline; recovered rows stay **inside** `𝒮`. Median p95 ≈ `0.708`.
- **Genuine margin** (the bias-corrected estimand):
  `margin := p95cos(B_jd·Y, H) − 𝔼_{B∼Haar(O(s))}[ p95cos(B·Y, H) ]`,
  estimated per `(kind, layer)` cell as `jade_p95 − proper_floor_p95`.
- **Matched, attack-independent probe** (`negentropy_bits`): the median over prompts of the summed
  Hyvärinen whitened-row negentropy `J(y) ≈ (1/12)·skew(y)² + (1/48)·exkurt(y)²` (nats → bits). It is
  a function of whitened-row moments only and **never calls `_joint_diag`/`jade`/`jd`** (imports only
  the data-prep helpers `_operands, _subsample, _whiten`) — so it is computable with the attacks
  deleted (probe ≠ attack).

---

## Lemma L1 (subspace-membership floor). Under Identity every demixing output is an orthonormal basis of one fixed subspace, so the recovery metric depends on the rotation only within that subspace — and the Gaussian-GT floor measures a structurally different functional than the matched random-demixing floor.

**Hypotheses.** `U = H`; the centered operand `Ũ` (per-row mean removed) has **full row rank `s`**
(equivalently, the whitening `W` exists and is invertible); `T ≥ 2s`. Let `Y = W·Ũ`, so by whitening
`(1/T)·Y·Yᵀ = I_s`. Write `𝒮₀ := rowspan(Ũ)` (dimension `s`).

> Scope note. The proved content of L1 is parts (1)–(3) below: a **structural invariance** and a
> **floor-mismatch identity**. The *magnitude* of the gap (that the membership floor is large,
> `≈ 0.708`, while the Gaussian-GT floor is small, `≈ 0.155`) is reported as an **empirical
> measurement**, not derived by closed form — an attempt at an analytic `1/√s` lower bound failed
> cross-model review (it required orthonormal true rows, a Hungarian ≥ row-max step, and `p95 ≥
> mean`, none of which hold in general) and is deliberately not claimed.

**Statement (proved parts).**

1. **(Row-span invariance.)** For every invertible `M ∈ ℝ^{s×s}` (in particular `M = B·W` for any
   `B ∈ O(s)`), `rowspan(M·Ũ) = 𝒮₀`. Thus the recovered rows `Ŝ = B·Y` lie in the **same** fixed
   subspace `𝒮₀` for every demixing `B`, whether `B = B_jd` or `B` Haar-random.
2. **(`B` acts only by rotation within `𝒮₀`.)** For every `B ∈ O(s)`, `(1/T)(B·Y)(B·Y)ᵀ = I_s`, so
   `B·Y` is a `(1/T)`-orthonormal basis of `𝒮₀`; varying `B` only rotates this basis inside the fixed
   `𝒮₀`. Hence the map `B ↦ p95cos(B·Y, H)` is a functional of `(𝒮₀, H)` evaluated along a rotation
   orbit, and its Haar-`B` expectation
   `F_in(𝒮₀, H; p95cos) := 𝔼_{B∼Haar(O(s))}[ p95cos(B·Y, H) ]` is a fixed functional of `(𝒮₀, H)` for
   the fixed `p95`/Hungarian/quantile convention — and, by left-invariance of Haar measure on `O(s)`,
   independent of the particular starting whitened basis `Y` — hence **invariant to whether `B_jd`
   succeeded.**
3. **(Floor mismatch / which null is matched.)** The Gaussian-GT floor replaces the fixed target rows
   of `H` by i.i.d. Gaussian rows that, with probability 1, are **not** the rows of `H` and (for
   `s < T`) do not lie in `𝒮₀`; it therefore measures a **different functional** `F_out` (matched
   cosine of a `𝒮₀`-basis against random ambient targets), not `F_in`. Consequently the only
   `B`-dependent (joint-diag-attributable) contrast is the **margin over the matched Haar-`B` null**,
   `margin := p95cos(B_jd·Y, H) − F_in(𝒮₀, H)`,
   and grading against the Gaussian-GT floor instead yields
   `raw − Gaussian_floor = margin + (F_in − F_out)`, which over- (or under-)credits the attack by the
   floor-mismatch bias `F_in − F_out`. That bias is generically nonzero and, empirically here, large
   and positive.

**Proof.**

*(L1.1 — row-span invariance.)* By hypothesis `Ũ ∈ ℝ^{s×T}` has full row rank `s`, so `𝒮₀` has
dimension `s`. For invertible `M`, each row of `M·Ũ` is a fixed linear combination of rows of `Ũ`,
so `rowspan(M·Ũ) ⊆ 𝒮₀`; since `Ũ = M⁻¹(M·Ũ)`, also `𝒮₀ ⊆ rowspan(M·Ũ)`. Hence equality. Taking
`M = B·W` (invertible as a product of invertibles) gives `rowspan(B·Y) = 𝒮₀` for every `B ∈ O(s)`.
∎(L1.1)

*(L1.2 — rotation orbit within `𝒮₀`.)* By whitening `(1/T)·Y·Yᵀ = I_s`. For `B ∈ O(s)`,
`(1/T)(B·Y)(B·Y)ᵀ = B·[(1/T)Y·Yᵀ]·Bᵀ = B·I_s·Bᵀ = I_s`. So the rows of `B·Y` form a
`(1/T)`-orthonormal system; by L1.1 they span `𝒮₀`, hence are a `(1/T)`-orthonormal basis of `𝒮₀`.
As `B` ranges over `O(s)`, `B·Y` traces the orbit of one fixed basis `Y` under the orthogonal group,
i.e. all `(1/T)`-orthonormal bases of `𝒮₀`. The target `H` is fixed throughout, so
`B ↦ p95cos(B·Y, H)` depends on `B` only through this in-`𝒮₀` rotation, and
`F_in(𝒮₀, H) = 𝔼_{Haar}[p95cos(B·Y,H)]` is a deterministic functional of `(𝒮₀, H)` — it does not see
`B_jd`. ∎(L1.2)

*(L1.3 — floor mismatch / matched null.)* `B_jd` is the rotation that joint-diagonalizes the whitened
fourth-order cumulants; a Haar `B` ignores cumulant structure. Both, by L1.1–L1.2, are
`(1/T)`-orthonormal bases of the **same** `𝒮₀` compared against the **same** fixed `H`. The Haar-`B`
null therefore holds fixed every nuisance the attack does not control — the subspace `𝒮₀`, the
whitening, the fixed targets `H`, and the Hungarian matcher — and randomizes only the quantity the
attack does control (the in-`𝒮₀` rotation). It is thus the **matched** null, and the data-defined
contrast `margin = p95cos(B_jd·Y,H) − F_in` differences the membership baseline out **by
construction** (this is a definition of the estimand, not a claim that it equals the population ICA
signal). The Gaussian-GT floor, by contrast, changes a nuisance — it replaces the fixed targets `H`
by random ambient rows, generically outside `𝒮₀` — so it estimates a different functional `F_out`.
Hence `raw − Gaussian_floor = margin + (F_in − F_out)`. The sign and size of the mismatch
`F_in − F_out` are not pinned down in closed form here; they are measured. ∎

**Numerical confirmation (empirical magnitudes).** Median over the 9 cells: `jade_p95 = 0.776`,
matched Haar-`B` floor `F_in ≈ proper_floor_p95 = 0.708`, Gaussian-GT floor `F_out ≈ 0.155`. Genuine
`margin = median of per-cell (jade_p95 − proper_floor_p95) = 0.027`. (Readout caveat: the margin is
the **median of the per-cell differences**, not `median(raw) − median(floor) = 0.067`, which
double-counts cross-cell spread.) The genuine margin `0.027` is only ≈ 3.4% of the apparent raw
recovery `0.776`; the remaining ≈ 96% is subspace-membership/floor artifact. (Separately, the
floor-mismatch bias `F_in − F_out ≈ 0.553` ≈ 71% of raw is how much the wrong Gaussian-GT null
under-counts the membership baseline.) `F_out ≈ 0.155` is the value reported by the shipped
chance-floor routine `bss.jd_floor` on this surface (logged in `pilot_dev24.json`). Per kind:
`kq ≈ 0` (L12 = `−0.003`), `kqv_out ≈ 0.02–0.03`, `resid_post ≈ 0.06–0.07`.

> Centering caveat (made explicit). The recovered rows live in the centered span `𝒮₀ = rowspan(Ũ)`,
> whereas the metric compares them against the **raw** rows of `H`; in general `𝒮₀ ⊆ rowspan(H) +
> span(𝟙)` rather than `𝒮₀ ⊆ rowspan(H)`. This does not affect the proved parts (1)–(3) — they
> compare `B_jd·Y` and Haar-`B·Y` against the *same* fixed `H`, so the additive-mean offset is
> common to both and cancels in `margin`. It only enters the (empirical) value of `F_in`.

---

## Observation C1 (no accumulation in T) — empirical, with an identifiability rationale (not a theorem).

**Statement (empirical).** Across the 9 `(kind, layer)` cells the fitted JD-across-`T` recovery
`p95cos` is flat in `log₂T` (median slope `+0.009`); stacking more observations did **not** improve
recovery in these runs.

**Rationale (heuristic, not a proof).** Joint-diagonalization-across-`T` is identifiable only when a
**single fixed unknown mixing `A`** is shared across the `T` stacked observations, so that the `T`
covariances `{Y_t Y_tᵀ}` share the common eigenstructure `A·diag·Aᵀ` that JD locks onto and that
sharpens as more observations constrain `A`. Under Identity `A = I_s` is **known**, and each operand
is a fresh independent activation matrix rather than a repeated view of one fixed source set through
one fixed `A`; there is no hidden mixing for the `T`-stack to accumulate information about. This makes
*absence of accumulation* the expected behaviour, but it is a heuristic: it does not, on its own,
prove a zero slope (per-observation ICA can still vary with sample geometry, and the slope is an
estimated quantity). We therefore report C1 as an **observed flat slope** that is consistent with the
identifiability rationale, not as a proved corollary of L1.

**Numerical confirmation.** Median `jd p95` slope over 9 cells `= +0.009` per `log₂T`; max genuine
margin over the matched per-`T` floor at any `(cell, T)` `= 0.094` (`resid_post`, L20, `T = 8`);
margins are non-monotone in `T`. (`T = 16` unavailable at dev-24: 0 disjoint stacks; `T` axis
`= {1,2,4,8}`.)

---

## Finding C2 (exploratory). The matched negentropy probe predicts the genuine margin, but only against the random-demixing floor.

The attack-independent negentropy probe predicts the **genuine** margin across the 9 `(kind, layer)`
cells at Spearman `ρ = 0.92`, Pearson `0.95`; against the **uncorrected raw** p95 it
**anti-correlates** (`ρ = −0.43`). Robustness (`c2_robustness.json`): permutation `p = 0.0013`
(exact over `9! = 362880` relabelings); leave-one-kind-out `ρ ∈ {0.77, 0.89, 0.77}`;
across-family-means `ρ = 1.0` (monotone `kq ≪ kqv_out ≪ resid_post` in both probe and margin);
within-family `ρ ∈ {0.5, −0.5, 0.5}` (n = 3 each) — essentially **no** within-family layer
resolution. The JD-accumulation probe (`shared_spectral_capacity`) tracks `jd` p95 at `ρ = 0.56`
(n = 36), below the 0.7 bar — a known-weaker, accumulation-axis probe.

**Scope (why exploratory, not asserted as a law).** n = 9; the `ρ = 0.92` is carried by a clean
3-level family ordering, so the probe predicts genuine BSS separability **at activation-kind-family
granularity**, not at per-layer resolution. The sign flip against the raw readout is the structural
content: **the floor definition (Lemma L1) is what makes or breaks the correlation** — the matched IT
probe predicts genuine BSS recovery once, and only once, recovery is graded against the correct
random-demixing floor.

---

## What this does and does not claim

- **Does** establish (proved, Lemma L1 parts 1–3): on plaintext KV/QKV under WEIGHTS-PUB every
  demixing output is an orthonormal basis of one fixed subspace `𝒮₀`, the recovery metric depends on
  the demixing only through a rotation within `𝒮₀`, the Haar-`B` expectation is a membership baseline
  invariant to whether joint-diag succeeded, and the Gaussian-GT floor measures a structurally
  different functional — so the matched null is the random-demixing floor and the contrast `margin =
  raw − F_in` differences the membership baseline out by construction.
- **Does** establish (empirical measurement): `F_in ≈ 0.708 ≫ F_out ≈ 0.155`, genuine margin
  `≈ 0.027`; recovery does not accumulate in `T` (median slope `+0.009`), consistent with the
  identifiability rationale.
- **Does** establish (exploratory, with permutation test): the geometry-only negentropy probe tracks
  the genuine separation margin at family granularity, and only against the matched floor.
- **Does not** claim a closed-form lower bound on the membership floor (the `1/√s` attempt was
  rejected: it needed orthonormal true rows, `Hungarian ≥ row-max`, and `p95 ≥ mean`); the floor
  magnitude and the flat-in-`T` slope are reported as measurements, not theorems.
- **Does not** claim BSS leakage is "meaningful only under a defense" as a proven statement — that
  requires the mixing-defense sweeps (Task 2 KV-CLOAK, Task 5 GELO). C1 is the **negative-control
  baseline** that makes those sweeps interpretable.
- **Not applicable** family members under WEIGHTS-PUB: `sda`/`tfma` (operate on recovered token-id
  sequences, not activations); `ia` weight-axis (needs an obfuscated weight pair; WEIGHTS-PUB gives
  the true weights). The `gram_error` baseline is trivially exact (`U = H` ⇒ row-Gram is the
  fingerprint) — a protocol-confirming appendix, not a recovery claim.

## Novelty (cross-model assessment: 5.5/10, PROCEED as negative baseline + evaluation correction)
The ingredients are classical — whitening leaves an orthogonal-rotation problem, ICA recovery is
defined up to sign/permutation/scale, the Gaussian/spherical case is rotationally non-identifiable
(Comon 1994; Cardoso–Souloumiac JADE 1993; Hyvärinen negentropy), and Hungarian/Kuhn–Munkres solves
the permutation ambiguity. The **delta**: (i) the matched **Haar-demixing null** for grading BSS
against *known* sources under *identity mixing* is not a standard named evaluation correction (ICA
evaluation usually uses gain-matrix / performance indices, e.g. Ilmonen 2012 arXiv:1212.3953;
rotational non-identification, Mesters–Zwiernik arXiv:2206.13668 — related but not this subtraction);
(ii) applying ICA/JD to transformer **KV/QKV** activations at all (the KV-cache attack literature —
*Shadow in the Cache* / KV-Cloak, arXiv:2508.09442 — uses weight-based inversion/collision, no BSS);
(iii) the empirical near-zero corrected margin + negentropy-predicts-corrected-not-raw finding.
Registered as a **negative baseline + evaluation correction**, not as new ICA theory.

## Related
- [[claim:spectral-channel-mi-embedding-inversion]] — same matched-MI-probe philosophy
  (`shared_spectral_capacity` reuses its `spectral_channel_mi`).
- [[claim:capacity-matched-pvi]] — the probe≠attack / matched-null discipline applied to PVI.
- Successors (will exercise this baseline under mixing): Task 2 KV-CLOAK (arXiv:2508.09442), Task 5 GELO.

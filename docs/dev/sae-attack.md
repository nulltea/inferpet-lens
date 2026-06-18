---
type: dev-log
status: current
created: 2026-06-17
updated: 2026-06-18
tags: [SAE, gemma-scope, gemma-2-2b, token-recovery, manifold-denoising, ridge, spike]
companion: [sae-private-inference-attack-design-space, sae-as-confidential-inference-attack]
---

# Developing an SAE-based token-recovery attack (living process doc)

Concise running record of how we design, spike, and harden an attack that
exploits gemma-scope SAE codes to recover the prompt token-id (and, later,
other secrets) under the split-TEE / `WEIGHTS-PUB` threat model.

## Setup

gemma-scope SAE on residual `x ∈ ℝ²³⁰⁴`:
- **encode** (nonlinear): `z = JumpReLU(xWₑₙ + bₑₙ)`, `z ∈ ℝ¹⁶³⁸⁴`, k-sparse.
- **decode** (linear): `x̂ = zW_dec + b_dec = Σᵢ zᵢ·dᵢ ≈ x`; atoms `dᵢ ∈ ℝ²³⁰⁴`.
Known to the attacker under `WEIGHTS-PUB`: the SAE (`Wₑₙ, W_dec, …`), the model
weights, and the tied embedding/unembedding `E`.

## Subproblem breakdown + conclusions

**SP1 — Are the 16k features orthogonal?** No (provably): ≤2304 orthogonal
vectors fit in ℝ²³⁰⁴; 16384 atoms are necessarily linearly dependent —
overcomplete, at best *low-coherence*. Orthogonality is neither true nor the
property we exploit. ⇒ exploit **sparsity + incoherence + known dictionary**.

**SP2 — Does z preserve token info in x?** Yes, and maybe makes it *more*
linearly accessible. Lower bound: `x̂=Dz≈x` is linear ⇒ any linear readout
`w·x` = `(Dᵀw)·z`, so linear-decodability(z) ≥ (x) minus recon error. Gain:
the *encoder* is nonlinear, so a linear probe on z is a nonlinear probe on x ⇒
V-info(z) can exceed V-info(x) even though Shannon `I(z;y) ≤ I(x;y)` (DPI).
⇒ the earlier spike's negative Δ was **overfitting, not info loss**.

**SP3 — Map feature→token without training?** Yes. `D` and `E` are known ⇒
each feature has a fixed token signature `E·dᵢ`; codebook `A = E·Dᵀ` precomputed
once ⇒ `logits(z) = A z = Σ_{active} zᵢ (E dᵢ)`. No classifier, no sample-
complexity problem. (Caveat SP4.)

**SP4 — Is a learned z→vocab inverter the only option?** No — don't learn the
16k map at all. Route `z →(known D)→ x̂` to collapse 16384→2304 *first*, then
the existing well-conditioned 2304-dim ridge `resid→embedding` + candidate-NN
recovers the token. (At depth, raw NN-in-E fails because resid ≠ embedding;
the cheap linear ridge supplies that map — "minimal training", well-posed.)
⇒ cheap `O(k)` decode + reuse existing attack; the 16k overfit vanishes.

**SP5 — Where does sparse/overcomplete z genuinely beat working with x?**
Not in raw recoverable info (DPI caps at x). It wins on **exploitability**:
zero-shot + attributable recovery (known codebook), `O(k)` cost, and —the real
prize— **cover-denoising**: `encode∘decode` projects onto the learned natural-
activation manifold, so `D·encode(T(x))` can strip a defense/`Transform` `T`
and beat recovery from the raw covered `T(x)`. This is the one structural SAE
*advantage* and it maps onto the split-TEE cover threat model.

## Key principle — nonlinear *reformatting*, not information gain

The recurring claim "a **linear** probe on `z` can recover more than a linear
probe on `x`" sounds like it violates data-processing. It doesn't — two
distinct quantities:

- **Shannon MI `I(·;tok)`** — best *any* decoder can do. `z=encode(x)` is a
  deterministic function of `x`, so `I(z;tok) ≤ I(x;tok)`. **No info created.**
- **Linear-decodability (V-information)** — what a *restricted* (linear/cheap)
  reader extracts. **Not** monotone under processing: a nonlinear map can move
  information into a form a weak reader can reach.

The claim is only about the second. A fixed nonlinear feature map reformats
existing information into a linearly-readable layout (the XOR/kernel trick:
`y=sign(x₁x₂)` is linearly unreadable from `(x₁,x₂)` but trivial from
`(x₁,x₂,x₁x₂)` — same info, no gain). The SAE encoder's **JumpReLU threshold**
is that nonlinearity: it **disentangles superposition**, turning a
"present-and-above-threshold-and-unmasked" (nonlinear) concept in `x` into a
single coordinate `zᵢ` a linear probe reads directly.

Bounds, together: `linear-dec(x) ≲ linear-dec(z) ≤ I(x;tok)` — the lower bound
because the decoder is linear (`w·x ≈ (Dᵀw)·z`) minus recon error; the upper
bound is Shannon. The SAE can push linear-decodability *up toward the ceiling a
nonlinear probe on `x` would reach*. **Only helps a bounded attacker** (an
unbounded one runs a nonlinear probe on `x` and erases the edge); and modern
residuals are already ~linear, so the gain is usually **small** (the 2.2%
result). The threshold also discards sub-threshold signal, competing with the
gain → net often ≈0/slightly negative. **This is why decoding `z→x̂` (decoder-reconstruction inversion) sees
no gain — it collapses back to `x`-space, discarding the reformatting; you must
probe `z` *directly* (direct sparse-code probing) to observe it.**

**The two SAE-advantage regimes are the SAME principle** (nonlinear
reformatting for a bounded reader, DPI-capped by the input, no Shannon gain),
differing in lever, baseline, and room:

| | (1) bounded attacker, clean `x` (direct sparse-code probing) | (2) defense phase, covered `T(x)` (SAE-denoising token recovery) |
|---|---|---|
| SAE input / baseline beaten | `x` / linear-on-`x` | `T(x)` / linear-on-`T(x)` |
| Lever | disentangle superposition | learned **manifold prior** denoises the cover |
| DPI ceiling | `I(x;tok)` | `I(T(x);tok)` — *can't recover what `T` destroyed* |
| Magnitude | small (residual already ~linear) | potentially large (cover crushed linear-accessibility) |
| Failure mode | gain ≈ recon-loss → ~0 | **off-manifold covers (rotation/permutation) defeat it** |

Both are V-information (bounded-reader) effects, not information creation; an
unbounded attacker on the same input subsumes the SAE in both.

## Designs

- **decoder-reconstruction inversion** known-weight decode-then-attack (floor / control).
- **direct sparse-code probing** capacity-controlled linear probe z-vs-x (corrected V-info question).
- **SAE-denoising token recovery** cover-denoising via `encode∘decode` under a `Transform` (the win).

## Decisions

- **Metric:** TTRSR top-1/10 from the existing `hidden_state` attack — bounded
  [0,1], avoids the unbounded-V-info blow-up seen in spike #1.
- **Reuse via `Transform`:** the SAE is itself a `Transform` `x ↦ D·encode(x)`,
  so it feeds the existing attack with zero attack-code changes.
- **PHASING (corrected):** the repo is scheme-agnostic; attack development runs
  on **plaintext (Identity)** only. A cover/defense is an external `Transform`
  evaluated in a **separate later phase** — never baked into the attack. So the
  earlier "2×2 SAE × cover" plan was out of scope (it coupled the attack to one
  specific defense). SAE-denoising token recovery is deferred (see below).
- **Next spike = plaintext only.** decoder-reconstruction inversion (decode `z→x̂` via known `D`, run the
  existing ridge attack, TTRSR, vs `x` baseline = SAE-recon cost) and/or direct sparse-code probing
  (corrected `z`-vs-`x` linear-decodability with bounded metric + capacity
  control). No cover.

## Defense-evaluation phase — folded plan (NOT the current plaintext spike)

**SAE-denoising token recovery cover-denoising:** does `D·encode(T(x))` recover the token better than raw
`T(x)` for a cover/`Transform` `T`? Same principle as direct sparse-code probing (see *Key principle*)
— nonlinear reformatting for a bounded reader, capped by `I(T(x);tok)`; the
lever is the learned **manifold prior** instead of disentangling superposition.
Run after the plaintext direct sparse-code probing result; `T` stays a pluggable external `Transform`.

**Governing constraint — DP covers (local-DP / Laplace–Gaussian activation noise):**
- **Formal:** DP **post-processing immunity** ⇒ the SAE (any `f`) **cannot**
  weaken the `(ε,δ)` bound. We never claim to break ε.
- **Empirical:** the manifold prior strips the **off-manifold** fraction of the
  noise ⇒ can lower reconstruction-loss / raise TTRSR vs raw `T(x)`, exploiting
  (a) the **population prior** DP does not protect against and (b) the gap
  between worst-case ε and average-case reconstruction. **Capped by the DPI
  floor `I(T(x);tok)`**: local-DP is high-noise, so a *tightly-calibrated* ε
  often leaves little to denoise (modest gain), whereas a heuristic "added-σ"
  scheme can be denoised substantially.
- **Framing:** treat every defense (incl. LDP) as a pluggable `Transform`; the
  measurable question is *"does `D·encode(T(x))` beat `T(x)`, and at what ε does
  the DPI floor erase the gain?"* — charting where a stated ε yields **real**
  protection against a prior-armed attacker, never violating ε.
- Covers to sweep: gaussian/Laplace LDP noise (denoise-friendly → the headline),
  rotation/permutation (off-manifold → SAE likely can't denoise), quantization.

## Open forks (plaintext spike, grilling in progress)

- decoder-reconstruction inversion vs direct sparse-code probing vs both as the next plaintext spike.
- Layers, corpus size, SAE width.
- Confirm operand injection through the existing attack's `transform=` path.
- Controls (vocab/shuffle) — which apply.

## Spike log

- **#1 (V-info z vs x, token-id):** infra validated (ROCm image + sae_lens +
  capture + encode). Result uninterpretable — V-info −50…−350 b (overfitting:
  n=4651 ≪ d=16384, sparse-std degeneracy). Lesson → use known weights (no
  learn) + bounded TTRSR. See `results/sae_vinfo_spike.json`.
- **#2 (decoder-reconstruction inversion, plaintext, vocab-split, TTRSR top-1):** recovery from `x̂=D·enc(x)`
  ≈ from raw `x` — L5 .661 vs .668, L12 .599 vs .589, L20 .576 vs .619
  (mean Δ ≈ −0.013). `results/sae_recover_spike.json`.

  **Reframing (key conclusion).** decoder-reconstruction inversion is a **control, not an SAE attack.** It
  decodes `z→x̂≈x` and runs the *existing* learned ridge `x→emb` attack, so by
  construction it **reconstructs `x` and re-runs the `x`-attack** — extra,
  lossy steps, never an advantage (DPI: `I(z;tok) ≤ I(x;tok)`). On plaintext
  with `x` in hand + pub embedding, the SAE adds nothing to *recovery*. The
  result only confirms the round-trip is ~lossless and that spike #1's negative
  V-info was learned-probe overfitting (SP2/SP4), not info loss.

  **Where SAE is NOT "the x-attack + extra steps":** (1) **bounded/linear**
  attacker — linear probe on `z` = nonlinear probe on `x`, so `z` may be *more
  linearly decodable* (direct sparse-code probing; the only plaintext recovery angle, marginal, and
  you must probe `z` *directly*, not decode it); (2) **defense phase** — SAE
  denoises an obfuscated `x` (the real recovery expansion); (3) **cost/zero-
  shot/triage** profile; (4) **`z`-only-leak** scenarios. Decoding to `x̂`
  destroys the sparse structure, so decoder-reconstruction inversion can never show (1)/(3).
- **#3 (direct sparse-code probing, plaintext, vocab-split, TTRSR top-1):** probe raw 16k `z`
  *directly* via the ridge attack vs `x` — `z` **collapses**: L5 .003, L12 .003,
  L20 .001 (top-10 .17/.03/.05) vs `x` ~.6. `results/sae_db_spike.json`.
  **The "z is more linearly decodable" edge does NOT materialize** — naively `z`
  is *far worse*. Not info loss (decode-route `x̂` recovers ~.6, SP2 lower bound):
  it's ridge mis-conditioning on the sparse 16k code + failure to generalize the
  `z→emb` map to vocab-disjoint test tokens (same family as #1's blowup; the
  token-agnostic decoder `D` would generalize but the data-fit ridge doesn't
  recover it). **Ceiling is capped at ≈`x`** anyway (decode realizes the lower
  bound; `x` already ~linear) — a whitened/α-tuned probe could at best lift `z`
  *toward* `x`, not past it. Caveat: ~0 is partly an untuned-conditioning
  artifact, so not the fully-fair V-info verdict — but the ≈`x` ceiling makes
  the qualitative conclusion robust.

  **NET (decoder-reconstruction inversion + direct sparse-code probing): on plaintext the SAE gives no recovery advantage over the
  `x`-attack (naively far worse). SAE value is confined to the defense phase
  (denoising) and the cost/zero-shot/triage profile.**
- **#4 (SAE-denoising token recovery, defense phase, AloePri-*style* additive-
  noise cover via the block-forward-hook seam; gemma-2-2b, 256 prompts, vocab
  split, TTRSR top-1):** `results/sae_denoise_spike.json`.

  | σ | L | clean | obf | denoised | gain |
  |---|---|------|-----|----------|------|
  | .25 | 5/12/20 | .516/.418/.411 | .438/.273/.249 | .470/.353/.331 | +.032/+.080/+.082 |
  | .50 | 5/12/20 | .516/.418/.411 | .242/.092/.066 | .367/.128/.069 | +.125/+.036/+.003 |

  **First demonstrated SAE *advantage*.** Denoised > obf in every cell — the SAE
  projects the noised activation back onto the manifold and **claws back ~40–55%
  of the recovery the cover removed** (mild noise). **Bounded by the DPI floor**:
  at σ=.5 L20 the noise already destroyed the signal → gain ≈ 0. So SAE denoising
  expands the *defense-phase* attack up to what survived the cover, never past it.
  Caveats: (a) additive noise is the *favorable* (denoise-friendly) cover —
  AloePri's core **basis-change/keymat is off-manifold, predicted ~null**, the
  decisive follow-up in the **same hook seam**; (b) clean ~.5 (vs #2's ~.6) is the
  256-prompt/HF path — comparisons are internal so gains hold; (c) the
  **block-forward-hook cover seam works** (attached after load, propagating) — the
  injection point for a faithful intra-model AloePri.
- **#5 (cut-layer-release covers, gemma-2-2b, 256 prompts, vocab split, TTRSR
  top-1).** Threat = split-inference: model runs clean, defense perturbs the
  *single released layer*'s activation (post-hoc, no propagation). Supersedes #4
  (which compounded noise across all layers). `results/sae_denoise_cutlayer_{noise,rotate}.json`.

  **(a) Additive noise (on-manifold, denoise-friendly):** SAE-denoising gain
  **positive everywhere** — σ=.25 +.005/+.040/+.008, σ=.5 +.052/+.073/+.060
  (L5/12/20). Closes ~40–48% of the noise gap at σ=.5. *SAE helps.*

  **(b) Basis-change rotate (off-manifold; GENERIC orthogonal rotation — NOT
  faithful AloePri, see verification note):** decisive, two parts.
  - *Ridge absorbs it* — at σ=0 `obf ≈ clean` exactly (.516/.424/.411); with
    noise `obf ≈ noise-only`. A fixed linear cover is **no defense** vs the
    rotation-invariant linear ridge.
  - *It reverses the SAE advantage* — denoising gain flips **negative**:
    σ=0 −.206/−.129/−.150, σ=.25 −.177/−.096/−.102, σ=.5 −.151/−.073/−.062.
    The SAE (trained un-rotated) projects to the wrong manifold region → `x̂`
    worse than obf. *Routing through the SAE actively hurts.*

  **Conclusion.** SAE-denoising helps **only for on-manifold** perturbations
  (noise); an **off-manifold basis-change neutralizes and reverses** it, while
  itself giving ~no defense against the linear attacker (anti-manifold-prior,
  not anti-ridge). Defender combo: noise (defends ridge, partially undone by
  SAE) + **secret** basis-change (blocks the SAE/manifold-prior denoiser).
  **WEIGHTS-PUB caveat:** holds only if the basis-change is **secret** — a known
  `Q` lets the attacker un-rotate first and the SAE advantage returns (why
  AloePri's keymat/Π must be secret).
  **Faithfulness vs real AloePri (verified against EdgeQuake `rag-privacy`,
  paper 2603.01499 §5.2).** The `rotate` cover is a **generic square orthogonal
  residual rotation — it is NEITHER Algorithm 1 NOR Algorithm 2.**
  - *Alg1 (keymat)* is the closest (also a residual transform) but is
    `x·P̂`, `P̂ ∈ ℝ^{d×(d+2h)}` — **rectangular/dimension-expanding** (+2h decoy
    dims), **non-orthogonal**, structured covariant tower. Not a square rotation.
  - *Alg2* is **per-head intra-attention weight** transforms (`R̂_qk·Ĥ_qk·Ẑ_block`
    on Q, `Û_vo` on V/O, `Π_head`) — a different site entirely, not a residual
    activation transform.
  - What survives: EdgeQuake **confirms the real keymat is ridge-absorbable**
    (57% top-1 4B / 97% 8B), so "linear cover → ridge absorbs" holds for genuine
    Alg1 — though our *orthogonal* rotation cancels exactly while the keymat
    (non-orthogonal + expansion + §5.2.2 noise/Π) leaves a gap. The
    "off-manifold breaks SAE-denoising" result is plausible for the keymat but
    **untested** (and the `d+2h` expansion can't feed a `d`-dim gemma SAE
    directly).
- **#6 (input local-DP, ε-calibrated, embedding-hook seam; gemma-2-2b, 256
  prompts, vocab split, TTRSR top-1).** Per-token embedding clip-to-C + Gaussian
  σ=C·√(2ln(1.25/δ))/ε, δ=1e-5, C=median emb-norm 1.765; noise PROPAGATES through
  the clean model. `results/sae_localdp_spike.json`.

  | ε | dp (L5/12/20) | denoised | SAE gain |
  |---|---|---|---|
  | 1 | .021/.016/.026 | .018/.023/.025 | ~0 |
  | 4 | .016/.019/.032 | .034/.023/.021 | ~0 |
  | 8 | .021/.022/.032 | .022/.016/.018 | ~0 |

  Clean = .516/.418/.411. **Input local-DP obliterates recovery to ~chance
  (≈2%) at ALL ε∈{1,4,8}; SAE gain ≈ 0** (nothing survives to recover → DPI
  floor; SAE neither helps nor hurts). No ε-dependence in range: the Gaussian
  mechanism's per-coord σ=C·z/ε gives **total noise norm ≈ σ·√d**, so at ε=8
  (σ≈1.07, √d≈48) noise norm ≈51 ≫ signal C=1.765 — **per-token embedding LDP
  on a 2304-d vector is devastating at any meaningful ε** (√d blowup; would need
  ε≈100–230 / no privacy before recovery leaves the floor). Post-processing
  immunity holds; empirically the SAE is moot vs input-LDP.

  ⚠️ **Bug fixed before this result:** the first run reseeded the noise RNG per
  forward → identical position-only offset every prompt → ridge absorbed it →
  bogus dp .1–.29 with inverted ε ordering and a spurious "SAE hurts". Fix: draw
  noise fresh per call (seed once per ε). Cut-layer spike (#5) drew noise once
  over the full matrix → unaffected.

- **#7 (generic input local-DP runner — `scripts/spikes/localdp_runner.py`, NO
  SAE; leakage panel TTRSR + PVI + CLUB vs ε; gemma-2-2b, 256 prompts, vocab
  split).** `results/localdp_curve.json`. Clip at p99.9 of **runtime**
  embed-norms (C=199, median=84 — the embed output is ~48× the static table
  norm: the √d normalizer is folded in), so clip-only ≈ clean and the curve is
  *noise*-driven.

  | ε | r | TTRSR frac (L5/12/20) | CLUB b (L5) | PVI b (L5) |
  |---|---|---|---|---|
  | ∞ | 0 | 1.00/1.00/1.00 | 3404 | 5.03 |
  | 2048 | .27 | 1.00/.97/.99 | 3413 | 5.09 |
  | 1024 | .54 | .94/.88/1.01 | 3328 | 5.31 |
  | 512 | 1.08 | .62/.66/.85 | 3054 | 5.82 |
  | 256 | 2.15 | .36/.29/.27 | 2637 | 5.59 |

  **TTRSR and CLUB fall monotonically with noise; ~50% recovery-destruction knee
  at ε≈400–512 (r≈1)**, near-clean for ε≳2048. **PVI is non-monotonic (rises
  ~16% mid-range) — treat as V-info estimator variance, not signal; CLUB+TTRSR
  are the reliable axes.** **DP takeaway:** the knee is at ε≈500 — a *non-private*
  budget; with the floored result at ε≤8, input per-token-embedding LDP is
  **all-or-nothing** (√d blowup) — no ε is both private and utility-preserving.

  ⚠️ **2nd bug fixed:** C was computed from the static embedding *table* (norm
  ~1.765) but the hook clips the *runtime* embed output (~84) → over-clipped to a
  constant at every percentile (clip-only frac ~0.23, clip-confounded). Fix:
  calibrate C from runtime embed_tokens norms.

## Defense-phase synthesis — when does SAE-denoising help?

SAE-denoising helps the attacker **iff the perturbation is off-manifold AT the
observed layer**:

| Cover | off-manifold at obs? | ridge baseline | SAE gain |
|---|---|---|---|
| cut-layer additive noise (added *at* released layer) | yes | degraded (partial) | **+ helps** |
| basis-change rotation (structural linear) | no (just rotated) | absorbed by ridge | **− hurts** |
| input local-DP (noise at input, √d-blown-up) | n/a — signal destroyed | **floored (~chance)** | **≈ 0 (moot)** |

So the SAE's niche is narrow: **only direct additive noise on the released
activation.** Structural transforms are the ridge's job (and the SAE off-manifold
projection actively hurts); propagated input noise becomes on-manifold and the
SAE can't touch it. The manifold-prior attack is not a general cover-breaker.

- **Next (same seam):** faithful Alg1 keymat cover (rectangular `d×(d+2h)` P̂,
  expected null/hurt per the table); faithful Alg2 needs Qwen3 + llama.cpp.

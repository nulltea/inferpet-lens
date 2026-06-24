---
type: claim
node_id: claim:rep2text-capacity-nonbinding-extraction-limited
title: "Rep2Text on the Qwen3 L10 residual is extraction-limited, not capacity-limited: the geometry-only capacity probe is provably vacuous across length in the capacity-slack regime"
status: verified
confidence: high
created: 2026-06-24
updated: 2026-06-24
tags: [rep2text, residual-stream, spectral-channel-mi, capacity, v-information, negative-result, qwen3, inversion]
companion: refine-logs/resid-rep2text/EXPERIMENT_RESULTS.md
supersedes:
superseded_by:
---

# Claim — Rep2Text L10 residual: extraction-limited, capacity probe vacuous across length

## Claim (scoped to Qwen3-4B → Qwen3-1.7B, layer 10, single adapter/seed)

A Rep2Text adapter→frozen-decoder inverter recovers **statistically significant but modest**
residual-specific information from a *single* last-token residual @ L10, and that recovery is
**extraction-limited, not capacity-limited**. Three sub-claims, each evidenced:

1. **(C1 refuted)** The single last-token vector is **not** a binding information-capacity bottleneck
   over the tested length range: the geometry-only spectral channel-MI capacity is
   `I_G ≈ 2856 bits` at the reference noise floor, far exceeding even the upper entropy proxy for the
   longest prompts (`H_X(L=59) ≈ 1026 bits`, and ≈470–650 bits under an 8–11 bit/token estimate).
   Recovery therefore does **not** decay with sequence length; the genuine leakage gap is *largest*
   for the longest prompts (+0.089 token-F1 @ L=59 vs +0.031 @ L=9).

2. **(genuine leakage exists)** Real recovery exceeds a 5-draw shuffled-residual null at **every**
   length bucket; all paired-bootstrap 95 % CIs exclude 0 (p(gap≤0) ≤ 0.009). Magnitude is modest
   (gap +0.015 to +0.089 token-F1; most raw token-F1 is the frozen decoder's LM / common-token prior,
   shuffled ≈ 0.108, mean-residual control ≈ 0.002).

3. **(C2 — matched capacity probe does NOT predict recovery across length)** The rate-distortion
   recoverable-fraction proxy `r(L) = min(I_G, H_X(L))/H_X(L)` is `≈ 1.000` for every tested length
   (Spearman vs token-F1 over the length×σ grid, n=36, **ρ = 0.18**). It orders recovery only deep in
   an artificial high-noise regime (across-σ Spearman(I_G, F1) = 1.0; Spearman(I_G, leakage gap) =
   0.94) — reached only **after >80 % of capacity is destroyed** (I_G 2856→520 bits leaves the leakage
   gap essentially flat, +0.038→+0.034). The probe is therefore **uninformative at the plaintext
   operating point** the task targets.

The mechanism is forced by a simple lemma: when channel capacity is **slack** over the tested length
range, the capacity-based rate-distortion ceiling is identically 1 and carries zero information about
how recovery varies with length. The matched probe for this surface must measure **extractable**
(𝒱-usable) information under a bounded decoder, not channel capacity.

## Theory

**Setup.** Fix a representation channel `Y = e_0 + N(0, σ² I_d)` where `e_0 ∈ ℝ^d` is the (clean)
last-token residual, a deterministic function of the secret text `X`, with covariance `Σ = Cov(e_0)`,
eigenvalues `λ_1 ≥ … ≥ λ_d ≥ 0`. Fix **one** covariance `Σ` estimated on the chosen
length-mixed residual ensemble and **hold it fixed** when evaluating every `L` (no per-length `Σ_L`).
The geometry-only spectral channel mutual information is `I_G(σ) = ½ Σ_{i=1}^d log₂(1 + λ_i/σ²)`
(bits); it is a functional of `(Σ, σ)` alone and, with `Σ` thus fixed, is **independent of the
secret's length** `L`. A length-`L` secret (`L ≥ 1` an integer) has entropy `H_X(L) = L·h`, where
`h ∈ (0, log₂ V]` is the per-token entropy and `V` the vocabulary, so `H_X(L) > 0`. The probe assigns
to each length the **normalized information-throughput ceiling**

    r(L) = min(I_G(σ), H_X(L)) / H_X(L).

(`min` because the channel transmits at most `I_G` bits about the secret and the secret carries at
most `H_X` bits — Shannon source–channel separation `R ≤ min{capacity, source rate}`; dividing by the
positive `H_X` normalizes it to a per-secret fraction. We use `r` only as a **cross-length ranking
proxy** — the vacuity claim below concerns its rank profile, not the literal tightness of any
token-recovery fraction.)

**Lemma (capacity-slack vacuity).** Suppose the channel is **slack** over the tested length range,
i.e. `I_G(σ) ≥ H_X(L_max)` where `L_max = max_L`. Then `r(L) = 1` for every tested `L`. Consequently
(for ≥ 2 tested lengths) the sample variance of `{r(L)}` is exactly 0, so the Spearman rank
correlation between `{r(L)}` and **any** recovery profile `{ρ(L)}` is mathematically undefined
(zero-variance denominator; the common constant-input reporting convention records it as 0 = "no rank
information"). Hence in the slack regime the probe carries no information about how recovery varies
with length, **independent of the attack**.

*Proof.* `H_X(L) = L·h` with `h > 0` is strictly increasing in `L ≥ 1`, so for every tested `L`,
`0 < H_X(L) ≤ H_X(L_max) ≤ I_G(σ)`. Therefore `min(I_G, H_X(L)) = H_X(L)` and, since `H_X(L) > 0`,
`r(L) = H_X(L)/H_X(L) = 1`. For ≥ 2 tested lengths a constant sequence has zero sample variance;
Spearman's ρ = `cov(rank r, rank ρ)/(s_{rank r} s_{rank ρ})` has `s_{rank r} = 0`, so ρ is undefined
(reported as 0 under the constant-input convention, i.e. no rank signal). No property of `ρ(L)` is
used, so the conclusion is independent of the attack. ∎

**Corollary (binding threshold).** For `H_X(L) > 0`: `r(L) < 1 ⇔ I_G(σ) < H_X(L) ⇔ L > I_G(σ)/h`
(at `L = I_G/h` the proxy stays saturated, `r = 1`). The probe becomes length-discriminating only for
`L > I_G/h`. With `I_G ≈ 2856` and `h ≤ log₂ V ≈ 17.2`: `I_G/h ≈ 166.05`, i.e. integer `L ≥ 167`
tokens — beyond the tested `L ≤ 59`, which is exactly why the measured Spearman is ≈ 0. Lowering `h`
to a realistic `8–11` bits raises the threshold further (`h=11 ⇒ L ≥ 260`; `h=8 ⇒ I_G/h = 357`, so
`L ≥ 358`), so the vacuity is robust to the `H_X` proxy.

**Consequence.** A *stronger* attack raises `ρ(L)` toward (but never above) the slack ceiling `r(L)=1`;
since `r(L)` is already saturated, no increase in attack strength can make the capacity probe
length-discriminating in this regime. The correct matched probe is an **extractable-information**
measure (𝒱-information, Xu et al. 2020) computed under the decoder's hypothesis class — left to the
follow-up `resid-rep2text-v2`.

## Evidence

- `refine-logs/resid-rep2text/runs/rep2text_results.json` (run `sweep-with-ci`): `I_G=2856.48` bits;
  per-bucket leakage gaps +0.031/+0.015/+0.041/+0.023/+0.054/+0.089 with paired-bootstrap (5000) 95 %
  CIs all excluding 0 (p ≤ 0.009); across-σ Spearman(I_G, F1)=1.0, Spearman(I_G, gap)=0.94;
  rd_proxy vs F1 Spearman=0.18 (n=36).
- Controls: mean-residual ≈ 0.002 (metric well-behaved); 5-draw shuffled null ≈ 0.108 (across-draw
  std ≈ 0.003). Code: `scripts/spikes/rep2text_run.py`, `scripts/spikes/rep2text_build_corpus.py`.
- Integrity: `refine-logs/resid-rep2text/EXPERIMENT_AUDIT.md` — WARN, no fraud; probe-≠-attack PASS.

## Scope / limitations
Single source/decoder pair (Qwen3-4B → Qwen3-1.7B), layer L10, single seed, single adapter,
N=23/bucket; shuffled null is a permutation (conservative); `H_X` is an upper entropy proxy. The
**lemma** is general; the **empirical** refutation is scoped to this setup. Related: refines the
per-position-resid null [[exp:vec2text-feedback-null]]; reuses the probe of
[[claim:spectral-channel-mi-embedding-inversion]].

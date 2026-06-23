---
type: claim
node_id: claim:defense-channel-selectivity-mechanism-dependent
name: "Lossy defenses are channel-selective and bits→recovery calibration is mechanism-dependent (scoped)"
description: "Empirical, jury-KEEPER/scoped single-model: Shredder static-Laplace collapses the permutation channel (Π VMA-recovery 1.0→0.04 over scale b) while leaving token-id ridge-recovery ≥0.45 at all depths — a single privacy scalar does NOT uniformly reduce leakage across secret kinds; and within-family Spearman(probe-bits, recovery) is higher for input-DP (0.64–0.81) than Shredder (0.39–0.43), pooled cross-family degraded (0.45–0.57) ⇒ the probe→attack calibration is conditional on the defense mechanism. AloePri corroboration: defends only via the dense keymat change-of-basis (VMA 0, ~0 bits); perm-core is VMA-vulnerable with probes tracking (Spearman 0.976/1.0). Theory backbone (critical-scale separation) proved Gaussian-only."
node_type: claim
status: drafted
provenance: ".aris/traces/result-to-claim/2026-06-23_defenses-existing_run01/ ; refine-logs/defenses-existing/PROOF_AUDIT.json"
tags: ["empirical", "scoped", "defenses-existing", "shredder", "aloepri", "channel-selective", "calibration", "leakage-utility"]
date: 2026-06-23
added: 2026-06-23T21:57:46Z
updated: 2026-06-23
---

# Lossy defenses are channel-selective and bits→recovery calibration is mechanism-dependent (scoped)

**status:** `drafted` (empirical; jury verdict KEEPER/scoped, single model)

## Statement (scoped after jury, 2026-06-23)

For the two implemented defenses (`scripts/defenses/{shredder.py,aloepri.py}`) measured on gemma-2:

1. **Channel-selectivity (Shredder static-Laplace).** As the Laplace scale `b` grows 0 → 0.817, the
   **permutation** channel (row-correspondence, VMA Hungarian matching) collapses monotonically —
   Π-recovery **1.000 → 0.977 → 0.565 → 0.204 → 0.099 → 0.037** — while **token-id** recovery (ridge
   inversion) is largely preserved, staying **≥ 0.45 at every depth** (L0/5/12/20) even at the largest
   `b` (L0 0.747 → 0.670). **A single scalar "privacy level" does not uniformly reduce leakage across
   secret kinds**: the same noise that destroys fine row-geometry leaves the quantization-robust
   token-identity statistic recoverable.

2. **Mechanism-dependent calibration (cross-family).** The bits→recovery map is **conditional on the
   defense mechanism**. Within-family Spearman(probe-bits, attack-recovery) is higher for **input-DP**
   (token 0.642, Π 0.812, embed 0.750) than for **Shredder** (token 0.389, Π 0.425); pooling the two
   families **degrades** it (token 0.453, Π 0.569). A probe calibrated on one family under-predicts on
   another. *Softened per jury:* this is evidence for **limited cross-family transfer / mechanism
   dependence**, not a non-transfer theorem (n is small — 6 b-levels × 4 layers — and the Shredder
   embedding arm is NaN, so the embedding comparison is unavailable).

3. **AloePri corroboration (regime).** AloePri defends only through the **dense keymat change-of-basis**:
   at α_e=0 the keymat regime already drives VMA-recovery to **0** and the independent CLUB probe to its
   estimator floor (**≈ −2.4 bits → ~0**), retr-PVI ≈ 0. The **perm-core** regime (row+col permutation +
   Gaussian noise) is VMA-vulnerable up to **α_e ≈ 0.35**, and there the independent CLUB probe and the
   dependent retr-PVI **track** recovery (Spearman **0.976 / 1.000**). *Scope:* keymat is a single config.

**Honest scope:** single model (gemma-2-2b / gemma-2 embed table); Shredder 3 seeds, AloePri perm-core
single seed (8-point α_e sweep), keymat single α_e=0 config; no bootstrap CIs; token_probe (perplexity)
and embed_probe numerically unstable across seeds (seed-1 token_probe collapse to 0.086; Shredder embed
arm NaN) — so (1) uses **attack recovery, not the probe**, and (2)'s embedding row is dropped for Shredder.

## Empirical status (jury-gated, NOT self-certified)

`KEEPER`/scoped. result-to-claim (Codex gpt-5.2 xhigh, 2026-06-23): claims (1)/(3) **yes, scoped**;
(2) **partial** — soften "non-transfer" to mechanism-dependence (confidence medium-low); (4=AloePri)
**yes** for perm-core tracking (confidence high), keymat **medium** (single config). Integrity audit
**WARN, no FAIL** (Codex gpt-5.5 xhigh): load-bearing **A (no GT leak) PASS** and **D (no probe==attack
circularity) PASS** — CLUB is attack-independent, retrieval-PVI is correctly labeled the dependent
"VMA-in-bits" reference. WARNs (reporting granularity: "192 prompts" not in JSON; AloePri single
seed / keymat single config) were addressed in `refine-logs/defenses-existing/RESULTS_STANDARDIZED.md`.

## Evidence chain (bits canonical + per-secret readout)

- **R1 Shredder Π-collapse vs token-id preservation** (`results/b4_cross_scheme.json`, 72 records,
  3 seeds; Π=0.037 is the 3-seed mean of raw 0.036/0.031/0.045):

  | b | Π recovery (VMA) | token-id rec L0 | token-id rec L20 |
  |---|---|---|---|
  | 0.000 | 1.000 | 0.747 | 0.510 |
  | 0.218 | 0.565 | 0.901 | 0.491 |
  | 0.817 | 0.037 | 0.670 | 0.455 |

- **R3 cross-family calibration** (`b4_cross_scheme.json::cross_scheme_transfer`):

  | channel | within input-DP | within Shredder | pooled |
  |---|---|---|---|
  | token-id | 0.642 | 0.389 | 0.453 |
  | perm-Π | 0.812 | 0.425 | 0.569 |
  | embedding | 0.750 | NaN | NaN |

- **R4 AloePri regimes** (`aloepri_vma_sweep.json`): perm-core α_e sweep VMA 1.0 → 0.007, CLUB
  252 → 181 bits, retr-PVI 3.34 → 2.27 bits, Spearman(bits,recovery) 0.976/1.0; **keymat α_e=0**: VMA
  0.000, CLUB −2.39 bits (floor), retr-PVI 0.6 mbit.

## Why this is a result, not a probe failure

The channel-selectivity (R1) is an **attack-side** observation (two attacks on the same noised surface),
immune to the probe instability. The mechanism-dependence (R3) is read from **two independent**
probe-bits estimators (CLUB MI upper bound, V-info) correlated against attacks *across* the sweep — not
the attack reporting its own bits (audit D PASS). The interpretation: Shredder, injected **at the
observed layer** (no propagation), behaves like input-DP@**L0** at every depth, so token recovery never
sign-flips with depth — corroborating [[depth-decoupling-input-dp]] as a **propagation/injection-locus**
property, not a generic lossy-defense artifact.

## Theory backbone — channel-selectivity is a critical-scale separation (proved, Gaussian)

The channel-selectivity (R1) is rationalized by a modeled proposition (fixed-codebook geometry; **not** a
claim about the network — A4 below is the empirical input). Proof verified by cross-model proof-checker
(Codex gpt-5.5 xhigh): Round-1 FAIL → Round-2 **PASS** (`refine-logs/defenses-existing/PROOF_AUDIT.json`,
full package `PROOF_PACKAGE.md`).

**Proposition (critical-scale separation).** Observe $Y=X+bZ$, $Z$ i.i.d. $\mathcal N(0,1)$, scale $b>0$.
Token-id: codebook $\{c_a\}_{a\le K}$, min distance $d_{\mathrm{tok}}$, secret $a^\*$, NN/MAP decode.
Permutation: $N$ points, min gap $d_\Pi$, $\sigma\sim\mathrm{Unif}(S_N)$, observations $y_i=x_{\sigma(i)}+W_i$.
**A4:** $d_\Pi\sqrt{\ln(K-1)}\ll d_{\mathrm{tok}}$ and $\exists\,M\ge1$ disjoint clean-point pairs at gap
$\le\rho=\Theta(d_\Pi)$. Then with $b_{\mathrm{tok}}=d_{\mathrm{tok}}/\sqrt{8\ln(K-1)}$ ($K\ge3$; $+\infty$ for
$K=2$) and $b_\Pi(\eta,M)=\rho/(\sqrt2\,Q^{-1}(1-\eta^{1/M}))=\Theta(d_\Pi)$ (fixed $\eta\in(2^{-M},1),M$):
(i) for $b\le b_{\mathrm{tok}}$ token-id MAP error $<\tfrac12$; (ii) for $b\ge b_\Pi$ exact-permutation
recovery $\le\eta$ ($\le 2^{-M}$ as $b\to\infty$, $\to0$ as $M\to\infty$). Under A4 the band
$(b_\Pi,b_{\mathrm{tok}})$ is nonempty: $\Pi$ destroyed, token-id preserved.

*Proof (sketch; full in `PROOF_PACKAGE.md`).* **Preserved.** NN-correct $\iff\langle W,u_a\rangle\le\tfrac12\delta_a\,\forall a$
(a.s.); union bound + $Q(t)\le\tfrac12e^{-t^2/2}$ gives $\Pr[\text{err}]\le\tfrac{K-1}{2}e^{-d_{\mathrm{tok}}^2/8b^2}\le\tfrac12$
for $b\le b_{\mathrm{tok}}$. **Destroyed.** A genie revealing all assignments except the within-pair
orientations $o_m$ only adds information, so $\Pr[\hat\sigma=\sigma]\le\prod_m(1-P_e^{(m)})$; under
$\mathrm{Unif}(S_N)$ the $o_m$ are independent uniform bits, each an equiprobable binary Gaussian test in
$\mathbb R^{2D}$ between means $(x_i,x_j),(x_j,x_i)$ at distance $\sqrt2\delta_m$, so
$P_e^{(m)}=Q(\delta_m/\sqrt2 b)\ge Q(\rho/\sqrt2 b)$; the product is $\le(1-Q(\rho/\sqrt2 b))^M\to2^{-M}$.
Band nonemptiness follows from $\Theta(d_\Pi)<d_{\mathrm{tok}}/\sqrt{8\ln(K-1)}$ under A4. $\blacksquare$

*Scope of the proof:* Gaussian only (Laplace is a caveated heuristic remark); $b_\Pi,b_{\mathrm{tok}}$ are
order constants, not sharp knees — the model rationalizes the *separation* (Π collapses at $b\approx0.2$,
token-id survives to $b\approx0.8$), not the exact knee locations; the fraction-correct collapse (vs exact
recovery) is a heuristic remark needing an observation-scale density assumption.

## Open (queued firm-ups, not this phase)

Multi-seed Shredder + AloePri with bootstrap/permutation CIs on each Spearman; held-out calibration
test (fit bits→recovery on one family, predict the other's recovery error); cross-model replication;
repair the unstable embed_probe so the embedding arm of (2) is recoverable; keymat parameter sweep
(seeds + λ/h) to lift (3) from "this config" to a defended-regime claim.

## Connections

Corroborates [[depth-decoupling-input-dp]] from the orthogonal-defense (Shredder) side. AloePri perm-core
tracking is the activation-table counterpart of the permutation-channel VMA story in [[perm-llr-threshold]].
Independence backbone [[threat-model-fairness]]. MI comparator [[mi-monotone-gaussian]] (its Laplace
degradation-order DPI underwrites the monotone Π-collapse). Empirical support edge from
[[defenses-existing-leakage-utility]].
_Edges recorded in `graph/edges.jsonl`._

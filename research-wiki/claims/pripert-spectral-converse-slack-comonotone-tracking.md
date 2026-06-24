---
type: claim
node_id: claim:pripert-spectral-converse-slack-comonotone-tracking
name: "PriPert split-inference: spectral channel-MI probe is a valid β-co-monotone converse, but slack on a high-rank residual"
description: ""
node_type: claim
status: verified
provenance: "refine-logs/resid-split/PROOF_CHECK_STATE.json"
tags: [residual, pripert, split-inference, spectral-channel-mi, fano-converse, capacity-slack, perturbation, sparsification, measurement-loop, weights-pub]
date: 2026-06-24
added: 2026-06-24T07:40:53Z
companion: refine-logs/resid-split/RESULTS_STANDARDIZED.md
---

# PriPert on the split residual: a matched channel-MI probe tracks inversion recovery along the perturbation axis and certifies a (slack) converse

**status:** `verified` (theory L1–L3 proof-checked PASS — 2 rounds, thread 019ef887; one INVALID
ρ-monotonicity claim refuted by counterexample and withdrawn; empirics integrity-audited WARN/no-fraud;
measurement-loop verdict: C2 positive along the proved axis, joint tracking empirical)

## Claim

PriPert (arXiv 2605.23158) defends a split-inference cut by **activation sparsification** +
**adversarial perturbation**. On Qwen3-4B `resid_post` (`WEIGHTS-PUB`; secret = token id; recovery =
token-table retrieval TTRSR top-1 over a ~2048–2300 candidate pool under a vocab-disjoint
train/val/test split + label-shuffle control + bootstrap CI), realized as the channel-matched
Gaussian proxy `U = Sparsify_ρ(H) + N(0,σ²I)`, `σ = β·meanRMS(plaintext H)` fixed per layer:

1. **(C1) PriPert suppresses inversion recovery; perturbation is the load-bearing knob;
   depth-dependent.** Best-inverter selectivity collapses with β (L8/L16 to ~0.01–0.02 by β=0.5;
   L24 by β≈1) while sparsification ρ is secondary; **L0 (post-block-0, input-adjacent residual, σ_ref=0.19)
   resists even β=2 (best 0.363)**. Not blanket monotonic — depth-dependent.
2. **(C2, headline) A matched, attack-independent spectral channel-MI probe tracks recovery.**
   `I_G = ½Σ_i log₂(1+λ_i/σ²)` (geometry+budget only, computed with NO inverter) rank-tracks
   inversion recovery — pooled Spearman 0.958, **and 0.915 against the stronger learned mlp2
   inverter** (CLUB cross-check 0.977). Within each layer's β-sweep the tracking is **perfect
   (Spearman 1.0 at L8/L16/L24)**; the β-confound-controlled fixed-β layer×ρ slice is 0.916. No
   probe–attack gap is *observed for the tested inverters*.
3. **(C3) The probe's Fano accuracy ceiling is a valid but slack converse.** `accessible =
   min(H_e0, I_G)`, ceiling `= min(1,(accessible+1)/H_X)`: **0 violations across 32 cells**, yet
   vacuous (=1) in 31/32 — I_G ≫ idealized H_X≈11 bits across the useful range; it binds only at β=2 (L8,
   I_G=10.0 < H_X−1≈10.2) where recovery is already 0.002. The empirical floor is reached far inside
   the IT converse — the capacity-slack pattern ([[rep2text-capacity-nonbinding]]) for a
   perturbation defense.

**Theory.** L1 proves the converse is valid (any estimator obeys the Fano ceiling). L2 proves
`I_G` and the Bayes-optimal accuracy are co-monotone **along the perturbation (β) axis at fixed ρ** —
the structural reason the probe rank-tracks recovery on within-layer β-sweeps (the measured 1.0). The
ρ-axis and joint/mixed-axis tracking are **empirical**: ρ-monotonicity is *false in general* for the
data-dependent magnitude mask (the support pattern itself leaks the secret — explicit counterexample
below). L3 proves the certificate is vacuous whenever `d_eff ≥ 2H_X−2`, so on a high-rank residual it
is slack through the floored regime. **The probe's predictive (rank-tracking) value, not its
certificate (Fano) value, carries the measurement loop here.**

## Measurement-loop verdict

Positive along the proved axis. The matched probe correlates with recovery across the sweep
(Spearman 0.958 pooled, 1.0 within-layer β-sweep) AND tracks the stronger learned inverter (0.915) —
the measurement-loop "Yes" branch, so **no spawn-depth-1 follow-up** is warranted. The honest
limits: joint/ρ tracking is empirical (not theorem); the Fano certificate is slack; the perturbation
is the energy-matched Gaussian proxy, not the paper's adversarial-optimized δ.

## Evidence (bits canonical + readout)

| correlate (full sweep, 24 finite-I_G cells of 32) | Spearman |
|---|---|
| I_G (bits) vs best-inverter recovery | 0.958 |
| I_G (bits) vs mlp2 (stronger learned) recovery | 0.915 |
| CLUB (bits) vs best recovery (secondary) | 0.977 |
| fixed-β=0.25 layer×ρ slice (β-confound removed) | 0.916 |
| within-layer β-sweep, L8/L16/L24 | 1.0 |

C1 readout — best TTRSR top-1 at ρ=0.25: L0 {β0:0.661, β2:0.363}; L8 {β0:0.608, β0.5:0.019,
β2:0.002}; L16 {β0:0.475, β0.5:0.017}; L24 {β0:0.576, β0.5:0.160, β1:0.012}. C3: 0/32 converse
violations; binds only L8 β=2 (I_G=10.0, ceiling 0.986). Artifacts:
`refine-logs/resid-split/runs/{pilot,sweep}/`; defense `scripts/defenses/pripert.py`; driver
`scripts/spikes/pripert_sweep.py`; integrity `refine-logs/resid-split/EXPERIMENT_AUDIT.json` (WARN,
no fraud); result-to-claim `RESULT_TO_CLAIM.md`.

---

## Proof (verified — proof-checker PASS, thread 019ef887, 2 rounds)

### Setup and notation
Secret token id $X$, uniform on an $M$-candidate pool ($M\ge2$), $H_X:=H(X)=\log_2 M$. Clean
residual row $H\in\mathbb R^d$; $S=\mathrm{Sp}_\rho(H)$ keeps the $\lceil\rho d\rceil$
largest-magnitude coordinates per row (ties broken by a fixed coordinate order) and zeroes the rest;
observed $U=S+Z$, $Z\sim\mathcal N(0,\sigma^2 I_d)\perp(X,H,S)$, $\sigma=\beta c$, $c=\text{meanRMS}
(H_{\text{plaintext}})>0$, $\beta>0$. $\Sigma_S=\operatorname{Cov}(S)$, eigenvalues
$\lambda_1\ge\dots\ge\lambda_d\ge0$, $I_G(\sigma,\Sigma_S)=\tfrac12\sum_i\log_2(1+\lambda_i/\sigma^2)$,
$d_{\mathrm{eff}}=\#\{i:\lambda_i\ge\sigma^2\}$. $\mathrm{acc}^\star(\sigma,\rho)=\max_{\hat X}\Pr[\hat X(U)=X]$.

### Lemma A (cited — [[spectral-channel-mi-probe-decision]], verified).
For $U=S+Z$, $Z\sim\mathcal N(0,\sigma^2 I_d)\perp S$, $\sigma>0$, $\operatorname{Cov}(S)=\Sigma_S$:
$$I(S;U)=h(U)-h(Z)\le\tfrac12\log_2\det(I+\Sigma_S/\sigma^2)=I_G,$$
by $h(U)\le\tfrac12\log_2((2\pi e)^d\det\operatorname{Cov}(U))$ (Gaussian max-entropy at fixed
covariance) and $\operatorname{Cov}(U)=\Sigma_S+\sigma^2 I$. Holds for any $S$ of covariance
$\Sigma_S$; equality iff $S$ Gaussian. Invoked within its conditions (independent additive Gaussian
noise, fixed covariance, $\sigma>0$). $\square$

### L1 — converse validity.
$X\to S\to U$ Markov ⇒ $I(X;U)\le I(S;U)\le I_G$ (data-processing + Lemma A); also $I(X;U)\le H_X$.
So $I(X;U)\le A:=\min(H_X,I_G)$. For any $\hat X(U)$ with error $P_e$, Fano (support $M$):
$$H(X\mid U)\le H_b(P_e)+P_e\log_2(M-1)\le 1+P_e H_X,$$
and $H(X\mid U)=H_X-I(X;U)\ge H_X-A$, so $H_X-A\le 1+P_e H_X$, giving
$\mathrm{acc}=1-P_e\le (A+1)/H_X$. Clamping, $\mathrm{acc}\le\min(1,(A+1)/H_X)$, the
`fano_exact_ceiling`. Holds for the inverter ⇒ 0 violations certified. $\blacksquare$

### L2(a) — $I_G$ non-increasing in $\sigma$ (strict iff $\Sigma_S\neq0$).
$\partial_\sigma I_G=-\tfrac1{\ln2}\sum_i\frac{\lambda_i}{\sigma^3+\sigma\lambda_i}\le0$ for $\sigma>0$,
strict iff some $\lambda_i>0$. Since $\sigma=\beta c$, $c>0$, $I_G$ is non-increasing in $\beta$. $\blacksquare$

### L2(c) — $\mathrm{acc}^\star$ non-increasing in $\sigma$.
For $\sigma'>\sigma$, $U(\sigma')\stackrel d=U(\sigma)+W$, $W\sim\mathcal N(0,(\sigma'^2-\sigma^2)I)
\perp(X,S,Z_\sigma)$. $W$ is **data-independent**, so $U(\sigma)\mapsto U(\sigma)+W$ is a valid
stochastic (Blackwell) degradation kernel and $X\to U(\sigma)\to U(\sigma')$ is Markov; Bayes-optimal
accuracy cannot rise under post-processing, so $\mathrm{acc}^\star(\sigma')\le\mathrm{acc}^\star(\sigma)$. $\blacksquare$

**β-axis co-monotonicity corollary.** At fixed $\rho$, $I_G$ (L2a) and $\mathrm{acc}^\star$ (L2c) are
both non-increasing in $\beta$ ⇒ same ranking ⇒ Spearman$(I_G,\mathrm{acc}^\star)=+1$ on distinct
values — the proved cause of the measured within-layer Spearman $=1.0$ (each a fixed-$\rho{=}0.25$
β-sweep). The realized inverters satisfy $\mathrm{acc}_{\text{ridge/mlp2}}\le\mathrm{acc}^\star$;
their tracking (mlp2 $0.915$) is empirical evidence, not a corollary. Joint/mixed-axis tracking
($0.958$; $0.916$) is empirical — one-axis monotonicity does not force a ranking across comparisons
moving both $\rho,\beta$.

### L2-Remark — ρ-monotonicity is NOT claimed (refuted for the magnitude mask).
Take $d=2$, $X\in\{1,2\}$ uniform, $H_1=(a,b)$, $H_2=(b,a)$, $a>b>0$. At $\rho=1$ the means are
$(a,b),(b,a)$, distance $\sqrt2(a-b)$; at $\rho=0.5$ (top-1) they become $(a,0),(0,a)$, distance
$\sqrt2\,a$ — *larger* — so $\mathrm{acc}^\star(\rho{=}0.5)>\mathrm{acc}^\star(\rho{=}1)$ at fixed
$\sigma$. The reduction $X\to U_{\rho'}\to U_\rho$ fails: the clean top-$k$ support is unrecoverable
from noisy $U_{\rho'}$, so no degradation kernel exists. Hence neither $\mathrm{acc}^\star$ nor $I_G$
is provably monotone in $\rho$ under the magnitude mask (empirically $\rho$ is the secondary knob and
$I_G$ moves monotonically with it — observation, not theorem). For *fixed nested supports*, $I_G$ is
monotone in $\rho$ by the principal-submatrix inequality $\det(I+N_{KK})\le\det(I+N)$ for
$N=\Sigma/\sigma^2\succeq0$ (with $M=I+N\succeq I$, $M/M_{KK}\succeq I$ ⇒ $\det M\ge\det M_{KK}$).

### L3 — the converse is slack.
Non-vacuity: $\text{fano}<1\iff I_G<H_X-1$. Each retained mode contributes
$\ge\tfrac12\log_2 2=\tfrac12$ bit, so $I_G\ge\tfrac12 d_{\mathrm{eff}}$. Thus
$$\text{fano}<1\Rightarrow I_G<H_X-1\Rightarrow d_{\mathrm{eff}}<2H_X-2,$$
contrapositively **$d_{\mathrm{eff}}\ge2H_X-2\Rightarrow\text{fano}=1$ (vacuous)**. When
$2\le\lceil2H_X-2\rceil\le d$, non-vacuity needs $\sigma^2>\lambda_{\lceil2H_X-2\rceil}$, i.e.
$\beta>\beta^{\star\star}:=\sqrt{\lambda_{\lceil2H_X-2\rceil}}/c$. (Boundary: if
$\lceil2H_X-2\rceil>d$ the $d_{\mathrm{eff}}$ condition gives no threshold and non-vacuity must be
checked directly via $I_G<H_X-1$; if $M=2$, $H_X-1=0$ and non-vacuity is impossible.)
*Empirical slack (not a theorem):* the recovery floor $\beta^\star$ (estimator-dependent) is reached
when the few predictive directions drown — at L8, $\beta^\star\approx0.5$ gives recovery $0.019$ while
$I_G=55.7$ ($d_{\mathrm{eff}}\gg20$); the ceiling binds only at $\beta^{\star\star}\approx2$
($I_G=10.0<H_X-1\approx10.2$). The interval $[\beta^\star,\beta^{\star\star}]$ (recovery floored,
converse vacuous, ratio $\approx4$) is observed and explained by $d_{\mathrm{eff}}(\beta^\star)\gg
2H_X-2$; not derived from the assumptions. $\blacksquare$

### Theorem.
Under the Gaussian PriPert proxy ($\sigma=\beta c>0$, uniform $M$-pool prior, fixed tie-break): (i)
$I_G$ is a valid attack-independent recovery converse [L1]; (ii) $I_G$ is co-monotone with
$\mathrm{acc}^\star$ along the perturbation axis at fixed $\rho$ [L2], the proved reason the matched
probe rank-tracks recovery on within-layer β-sweeps (Spearman 1.0); the joint/ρ tracking (0.958) is
empirical and ρ-monotonicity is false in general (L2-Remark); (iii) the certificate is vacuous
whenever $d_{\mathrm{eff}}\ge2H_X-2$ [L3], so on a high-rank residual it is slack through the floored
regime. Hence the probe's **predictive** value, not its **certificate** value, carries the
measurement loop. $\blacksquare$

### Scope / open risks
ρ-monotonicity of $I_G$ under the magnitude mask is empirical; the $\beta^{\star\star}\!\gg\!\beta^\star$
gap size is empirical (the "$d_{\mathrm{eff}}\ge2H_X-2\Rightarrow$ vacuous" half is proved); joint
Spearman is empirical; uniform-prior $H_X=\log_2 M$ (non-uniform priors need a separate Fano
formulation, possibly looser); Gaussian-δ proxy (adversarial-optimized δ out of scope); single seed,
one model/corpus.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarized for human readers below._
supports ← exp:b-r4-split (Spearman 0.958; within-layer 1.0; 0/32 converse violations). Related:
[[rep2text-capacity-nonbinding]] (capacity-slack on the L10 residual — same vacuous-converse pattern);
[[spectral-channel-mi-probe-decision]] (the cited Gaussian-channel converse ceiling, reused as
Lemma A); [[depth-inversion-certificate]] (Fano certificate on the plaintext depth sweep);
[[gelo-orthogonal-gram-leak-rowmix-defeats-bss]] (sibling residual defense).

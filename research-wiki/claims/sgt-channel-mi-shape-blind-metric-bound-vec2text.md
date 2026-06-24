---
type: claim
node_id: claim:sgt-channel-mi-shape-blind-metric-bound-vec2text
name: "SGT: scalar channel-MI is a valid converse but shape-blind for a fixed metric-bound Vec2Text decoder"
description: ""
node_type: claim
status: verified
provenance: "refine-logs/embed-sgt/proof/PROOF_AUDIT.json; refine-logs/embed-sgt/runs/sweep/sgt_eval.json; refine-logs/embed-sgt/EXPERIMENT_AUDIT.json"
tags: [embedding, stained-glass, sgt, channel-mi, vec2text, converse, shape-blind, metric-bound, negative-result, weights-pub]
date: 2026-06-24
added: 2026-06-24T09:15:55Z
companion: refine-logs/embed-sgt/EXPERIMENT_PLAN.md
---

# SGT: the scalar channel-MI converse $I_G$ predicts Vec2Text recovery *within* a fixed noise shape but is *shape-blind* across shapes at matched budget — recovery of the deployed fixed corrector tracks read-subspace distortion, not $I_G$

**status:** `verified` (theory L1–L3 proof-checked PASS, 2 rounds, cross-model gpt-5.5 xhigh; empirics integrity-audited PASS, no fraud; measurement-loop verdict: does-NOT-correlate-across-shape — that is the finding)

## Summary of the empirical finding (verified, integrity PASS)

Surface: pooled GTR sentence embedding (`gtr-t5-base`, $d=768$, mean-pooled). Attack: pretrained
`gtr-base` Vec2Text iterative corrector (Morris et al. 2023) — a **fixed** decoder. Defense: Stained
Glass Transform (SGT, arXiv 2506.09452), modelled as heteroscedastic Gaussian release
$Y = e_0 + N$, $N\sim\mathcal N(0,D)$, $D=\mathrm{diag}(v_1,\dots,v_d)$. Probe (attack-independent,
geometry-only): generalized spectral channel-MI

$$ I_G(D) \;=\; \tfrac12 \log_2 \det\!\big(I + D^{-1/2}\,\Sigma\,D^{-1/2}\big) \;=\; \tfrac12\sum_{i=1}^d \log_2(1+\mu_i),\qquad \Sigma=\mathrm{Cov}(e_0), $$

where $\mu_i$ are the eigenvalues of the whitened covariance $M(D)=D^{-1/2}\Sigma D^{-1/2}$.

The sweep fixes a target budget $B\in\{826.8,434.1,196.0,71.4\}$ bits and, at each $B$, builds three
noise **shapes** that all hit the same $I_G=B$: `iso` (isotropic $v_i=\sigma^2$), `sgt_opt`
(reverse-water-filling, the distortion-minimizing SGT optimum), `tail_dump` (adversarial: noise
dumped on the low-$\lambda$ tail). Results ($N=96$ held-out texts, `max_tokens=32`, `num_steps=20`):

- **Within a fixed shape**, $I_G$ is a perfect monotone predictor: $\mathrm{Spearman}(I_G,\text{token-F1})=1.0$ for `iso` and `sgt_opt`.
- **Across shapes at matched budget**, recovery diverges $\sim 12\times$: at $I_G=826.8$ bits, token-F1 $=\{$`sgt_opt` $0.566$, `iso` $0.433$, `tail_dump` $0.048\}$.
- **Across the 12 noisy settings** (finite budget; the plaintext anchor is held out of the correlation): $\mathrm{Spearman}(\text{token-F1},I_G)=0.482$ (below the $0.6$ bar), but $\mathrm{Spearman}(\text{token-F1},\text{relCos})=0.972$ and $\mathrm{Spearman}(\text{token-F1},-D_{\text{tot}})=0.951$, where $\text{relCos}=\cos(e_0,Y)$ and $D_{\text{tot}}=\sum_i v_i$. (Including the plaintext anchor as infinite budget moves the I_G correlation only to $\approx0.59$, still below $0.6$.)

So the scalar $I_G$ is **not** a complete predictor of the deployed attack's recovery; what predicts
recovery is read-subspace distortion (relCos / $D_{\text{tot}}$). The theory below explains exactly
why a *valid converse* fails to be a *matched predictor*.

## Claim (three lemmas bounding the gap)

**L1 (valid converse + within-shape monotonicity).** $I_G(D)$ upper-bounds $I(e_0;Y)$ under the
Gaussian channel-capacity surrogate, hence (data-processing + Fano on the discrete text) upper-bounds
any decoder's exact reconstruction; and along any one-parameter shape ray $v=t\,u$ ($u>0$ fixed,
$t>0$), $I_G(t)$ is strictly decreasing in $t$. Therefore within a fixed shape the converse bound
falls monotonically as noise scale rises.

**L2 ($I_G$ is allocation-blind at fixed budget).** For $d\ge2$ the level set
$\{D\succ0\ \text{diagonal}: I_G(D)=B\}$ is a $(d-1)$-dimensional manifold over which the total
distortion $D_{\text{tot}}=\sum_i v_i$ and the per-mode allocation vary freely; the
distortion-minimizing point is reverse-water-filling (`sgt_opt`) and a tail-loaded allocation attains
the same $B$ with arbitrarily larger $D_{\text{tot}}$. Hence $I_G$ is invariant to *which* modes are
protected — it cannot encode the alignment between the noise allocation and a fixed decoder's read
directions.

**L3 (metric-bound decoder $\Rightarrow$ recovery tracks distortion).** Under the modelling assumption
that the fixed Vec2Text corrector $\hat g$ is $L$-Lipschitz in cosine distance on its normalized
input, expected recovery degradation obeys
$\mathbb E[\Delta\text{score}]\le L\,\mathbb E[1-\text{relCos}(e_0,Y)]\le 2L\,D_{\text{tot}}\,\mathbb E[\|e_0\|^{-2}]$,
and at fixed $I_G=B$ the total distortion $D_{\text{tot}}$ is minimized by `sgt_opt` and unbounded for
`tail_dump` (Lemma 2b). Therefore the predictor *matched to this fixed decoder* is total/read-subspace
distortion (relCos), while $I_G$ is only a loose shape-invariant converse. This **separates**
"non-matched probe" (L2: $I_G$ structurally cannot distinguish allocations — proven, independent of
A4) from "weak attack" (a shape-aware decoder might exploit the tail-mode MI — *not* claimed here;
queued as the follow-up).

## Assumptions

- **(A1) Gaussian channel surrogate.** For the MI ceiling we treat $e_0\sim\mathcal N(0,\Sigma)$. Then
  $Y=e_0+N$ with $N\sim\mathcal N(0,D)$ independent gives $I(e_0;Y)=\tfrac12\log_2\det(I+D^{-1/2}\Sigma D^{-1/2})=I_G$ **exactly**.
  For a non-Gaussian $e_0$ with the same second-order statistics, the Gaussian maximizes entropy at
  fixed covariance, so $I_G\ge I(e_0;Y)$ — i.e. $I_G$ remains a valid **upper bound** (converse). This
  is the campaign's T1 sufficiency convention.
- **(A2) $\Sigma\succ0$, $D\succ0$, $d\ge2$.** Covariance positive-definite (full-rank embedding
  statistics) and every noise variance strictly positive; both hold in the experiment ($d=768$,
  $d_{\text{eff}}$ reported, clip applied). Degenerate $\lambda_i=0$ or $v_i=0$ modes are excluded by
  A2 and discussed in Open Risks. The dimension $d\ge2$ is required by Lemma 2 (at $d=1$ a budget
  fixes the single variance, so there is no allocation freedom — see EDGE note in L2).
- **(A2′) Almost-sure non-degeneracy of directions**, $\|e_0\|>0$ and $\|Y\|>0$ a.s., so the cosine
  $\cos(e_0,Y)$ is a.s. well-defined (automatic for the Gaussian surrogate A1 with $\Sigma\succ0$;
  assumed for a general $e_0$).
- **(A3) Independent additive noise**, $N\perp e_0$.
- **(A4) Lipschitz-in-cosine decoder (MODELLING ASSUMPTION, not measured).** The fixed pretrained
  corrector $\hat g$ satisfies, for a graded score $s(\cdot,x)\in[0,1]$ against the true text $x$,
  $\big|\,\mathbb E\,s(\hat g(Y),x)-\mathbb E\,s(\hat g(e_0),x)\,\big|\le L\,\mathbb E[\,d_{\cos}(Y,e_0)\,]$
  with $d_{\cos}=1-\cos$. This is an *assumed model* of the deployed corrector that **explains** the
  data; it is corroborated post hoc by $\mathrm{Spearman}(\text{recovery},\text{relCos})=0.972$, **not**
  derived from Vec2Text internals. L1–L2 do not depend on A4.

## Notation

- $\Sigma=\mathrm{Cov}(e_0)\in\mathbb R^{d\times d}$, eigendecomposition $\Sigma=Q\Lambda Q^\top$,
  $\Lambda=\mathrm{diag}(\lambda_1\ge\dots\ge\lambda_d>0)$ (PCA basis).
- $D=\mathrm{diag}(v_1,\dots,v_d)$, $v_i>0$. In the PCA basis we write the per-mode noise variance as
  $v_i$ acting on PCA mode $i$ (for `iso`/`sgt_opt`/`tail_dump` the construction is performed in this
  basis, so $D$ is diagonal there).
- $M(D)=D^{-1/2}\Sigma D^{-1/2}$, eigenvalues $\mu_1\ge\dots\ge\mu_d\ge0$.
- $I_G(D)=\tfrac12\sum_i\log_2(1+\mu_i)$; in the PCA basis with diagonal $D$, $M=\mathrm{diag}(\lambda_i/v_i)$ so $\mu_i=\lambda_i/v_i$ and $I_G=\tfrac12\sum_i\log_2(1+\lambda_i/v_i)$.
- $D_{\text{tot}}=\sum_i v_i$ (total distortion, basis-invariant since $=\mathrm{tr}\,D=\mathbb E\|N\|^2$).
- $\text{relCos}=\cos(e_0,Y)$; $B>0$ the target budget in bits.

## Proof Strategy

L1: exact-MI identity for Gaussians + maximum-entropy converse for non-Gaussian $e_0$ + Fano on the
discrete text; strict monotonicity by differentiating $I_G(t)$. L2: regular-value argument for the
level-set dimension + convex-reparametrization Lagrangian (reverse-water-filling) for the distortion
minimizer and an explicit tail-loaded family for the unbounded-distortion direction. L3: telescoping
the score gap through $\hat g(e_0)$ and applying A4, bounding $\mathbb E[1-\text{relCos}]$ by
$D_{\text{tot}}$ (controlled by L2b) plus a DCT tail limit.

## Dependency Map

1. L1(1a) depends on A1+A3; L1(1b) needs only $\mathrm{Cov}(e_0)=\Sigma$ + A3 + a discrete secret
   $X$ with $X\to e_0\to Y$ (max-entropy bound + DPI + Fano); L1(1c) is elementary calculus.
2. L2 depends on A2 ($\Sigma,D\succ0$, $d\ge2$); (2a) regular-value theorem, (2b) strict-convex
   reparametrization + KKT, (2c) the tail-loaded family with a read tail mode.
3. L3 depends on A4 (Lipschitz model), A2′ + $\mathbb E[\|e_0\|^{-2}]<\infty$ (the $D_{\text{tot}}$
   bound), L2b ($D_{\text{tot}}$ control), and DCT (tail limit).
4. The structural verdict "$I_G$ is allocation-blind as a probe" depends on L2 ALONE and is
   independent of A4; only the "deployed decoder is relCos-matched" claim uses A4.

## Proof

### Lemma 1 (valid converse + within-shape monotonicity)

**(1a) $I_G=I(e_0;Y)$ under A1.** With $e_0\sim\mathcal N(0,\Sigma)$ and $N\sim\mathcal N(0,D)$
independent (A1, A3), $Y=e_0+N$ is jointly Gaussian with $\mathrm{Cov}(Y)=\Sigma+D$ and
$\mathrm{Cov}(Y\mid e_0)=D$. For Gaussian vectors
$I(e_0;Y)=\tfrac12\log_2\frac{\det(\Sigma+D)}{\det D}=\tfrac12\log_2\det\!\big(D^{-1}(\Sigma+D)\big)
=\tfrac12\log_2\det\!\big(I+D^{-1}\Sigma\big).$
Since $\det(I+D^{-1}\Sigma)=\det(I+D^{-1/2}\Sigma D^{-1/2})$ (the matrices $D^{-1}\Sigma$ and
$D^{-1/2}\Sigma D^{-1/2}$ are similar via $D^{1/2}$, hence share eigenvalues and determinant of
$I+\cdot$), this equals $I_G(D)$.

**(1b) Converse on recovery of the discrete secret.** Fano's inequality needs a *discrete* secret
and an *error event*, which the continuous $e_0$ does not by itself provide. Introduce the genuine
secret: the token sequence $X$ (the text behind the embedding), drawn from a finite alphabet
$\mathcal X$ with $|\mathcal X|=K\ge2$ (per-token $K=$ vocab; per-sequence $K=$ vocab$^{n}$), and the
deterministic embedding map $X\mapsto e_0$. This gives the Markov chain $X\to e_0\to Y\to\hat X$,
where $\hat X=\hat g(Y)$ is any decoder's reconstruction.

(i) **MI bound.** Drop A1's distributional assumption but keep $\mathrm{Cov}(e_0)=\Sigma$. Then
$I(e_0;Y)=h(Y)-h(Y\mid e_0)=h(Y)-h(N)$, with $h(N)=\tfrac12\log_2\big((2\pi e)^d\det D\big)$ fixed.
The maximum-entropy theorem (Gaussian maximizes differential entropy at fixed covariance) gives
$h(Y)\le\tfrac12\log_2\big((2\pi e)^d\det\mathrm{Cov}(Y)\big)$ with $\mathrm{Cov}(Y)=\Sigma+D$ (using
$N\perp e_0$, A3). Subtracting, $I(e_0;Y)\le\tfrac12\log_2\frac{\det(\Sigma+D)}{\det D}=I_G(D)$ for any
$e_0$ with covariance $\Sigma$ (with equality under A1, by (1a)). By the data-processing inequality
along $X\to e_0\to Y$, $\;I(X;Y)\le I(e_0;Y)\le I_G$.

(ii) **Fano.** For the discrete $X$ with the decoder error event $\{\hat X\ne X\}$ of probability
$P_e$, Fano gives $H(X\mid Y)\le 1+P_e\log_2 K$, i.e.
$P_e\ \ge\ \dfrac{H(X)-I(X;Y)-1}{\log_2 K}\ \ge\ \dfrac{H(X)-I_G-1}{\log_2 K},$
a lower bound on exact-reconstruction error that is non-decreasing as $I_G$ decreases (non-vacuous
exactly when $H(X)-I_G-1\ge0$). Hence the *exact-match* recovery is ceiling-bounded by
$1-P_e\le \min\!\big(1,\,1-\frac{H(X)-I_G-1}{\log_2 K}\big)$. (We state the converse for the
exact-match event, where Fano applies directly; graded scores such as token-F1 are not controlled by
exact-match error without a threshold/list-recovery event, so we make no graded ceiling claim here.)
Thus $I_G$ is a valid converse on exact reconstruction of the discrete secret, tightening as
$I_G\downarrow$.

**(1c) Strict monotonicity of the converse along a shape ray.** Fix the unit allocation
$u=(u_1,\dots,u_d)$, $u_i>0$, and set $v_i=t\,u_i$, $t>0$. In the PCA basis
$\mu_i=\lambda_i/v_i=\lambda_i/(t u_i)$, so $I_G(t)=\tfrac12\sum_i\log_2\!\big(1+\tfrac{\lambda_i}{t u_i}\big)$.
Each summand is strictly decreasing in $t$, so
$\tfrac{d}{dt}I_G(t)=\tfrac1{2\ln2}\sum_i\frac{-\lambda_i/(u_i t^2)}{1+\lambda_i/(t u_i)}<0$ for all
$t>0$ (each term $<0$ since $\lambda_i,u_i,t>0$). Hence the *converse bound* $I_G(t)$ is strictly
decreasing along the ray: more noise at fixed shape lowers the recovery ceiling. We do **not** claim
this forces the empirical recovery rank order (a decreasing upper bound need not force a decreasing
realized recovery); rather, the observed within-shape $\mathrm{Spearman}=1.0$ for `iso`/`sgt_opt` is
*consistent* with a converse that the deployed decoder tracks closely within a fixed shape — and the
across-shape failure (L3) is precisely where the converse and the realized recovery part ways.
($\blacksquare$ L1)

### Lemma 2 ($I_G$ is allocation-blind at fixed budget)

Work in the PCA basis (A2: $\lambda_i>0$, $v_i>0$). Define $F:\mathbb R^d_{>0}\to\mathbb R$,
$F(v)=I_G=\tfrac12\sum_i\log_2(1+\lambda_i/v_i)$.

**(2a) Level set is a $(d-1)$-manifold.** $F$ is $C^\infty$ on $\mathbb R^d_{>0}$ with
$\partial F/\partial v_i=\tfrac1{2\ln2}\cdot\frac{-\lambda_i/v_i^2}{1+\lambda_i/v_i}
=\tfrac{-1}{2\ln2}\cdot\frac{\lambda_i}{v_i(v_i+\lambda_i)}<0$ for every $i$ (A2). Thus $\nabla F$
never vanishes, every $B$ in the range of $F$ is a regular value, and by the regular value (implicit
function) theorem $\{v\succ0:F(v)=B\}$ is a smooth embedded $(d-1)$-dimensional submanifold of
$\mathbb R^d_{>0}$. The single budget constraint leaves $d-1$ free directions.

**(2b) $D_{\text{tot}}$ varies freely on the level set; reverse-water-filling is the global minimizer
($d\ge2$).** Minimize $D_{\text{tot}}(v)=\sum_i v_i$ subject to $F(v)=B$. *Convex reparametrization
(for the global minimum).* Let $b_i=\tfrac12\log_2(1+\lambda_i/v_i)\ge0$ be the per-mode bit
allocation, so $v_i=\lambda_i/(2^{2b_i}-1)$ and the budget is the affine constraint $\sum_i b_i=B$.
Each $v_i(b_i)=\lambda_i/(2^{2b_i}-1)$ is strictly convex and strictly decreasing on $b_i>0$ (its
second derivative is positive), so $D_{\text{tot}}(b)=\sum_i \lambda_i/(2^{2b_i}-1)$ is strictly convex
in $b$ on the open simplex $\{b\succ0,\sum b_i=B\}$. Existence: as any $b_i\downarrow0$,
$v_i=\lambda_i/(2^{2b_i}-1)\to\infty$, so $D_{\text{tot}}\to\infty$ near the relative boundary; hence
the infimum is attained at an interior point of the compact closed simplex. A strictly convex
objective then has a **unique** minimizer there, characterized by KKT. *KKT (corrected sign).* With
Lagrangian $\mathcal L=\sum_i v_i+\eta\big(\tfrac1{2\ln2}\sum_i\ln(1+\lambda_i/v_i)-B\big)$ and
$\partial F/\partial v_i<0$ (from 2a), stationarity $\partial\mathcal L/\partial v_i=0$ reads
$1+\eta\cdot\tfrac1{2\ln2}\cdot\frac{-\lambda_i}{v_i(v_i+\lambda_i)}=0$, i.e.
$v_i(v_i+\lambda_i)=\tfrac{\eta}{2\ln2}\lambda_i$ with multiplier $\eta>0$, giving the
reverse-water-filling solution $v_i^\star=\tfrac12\big(\sqrt{\lambda_i^2+\tfrac{2\eta}{\ln2}\lambda_i}-\lambda_i\big)$
($\eta$ fixed by the budget). This is the `sgt_opt` allocation and, by strict convexity, the global
minimizer of $D_{\text{tot}}$ at fixed $B$. *Unbounded direction (needs $d\ge2$).* Fix $B$ and put
mass on the smallest-$\lambda$ mode $i=d$: as $v_d\to\infty$ its bit contribution
$\tfrac12\log_2(1+\lambda_d/v_d)\approx\tfrac{\lambda_d}{2 v_d\ln2}\to0$, so the remaining $d-1$ modes
($d\ge2$, so at least one exists) can re-absorb the negligible lost bits at finite cost while
$D_{\text{tot}}\ge v_d\to\infty$. Thus on the level set $D_{\text{tot}}$ is unbounded above and bounded
below by $\sum_i v_i^\star$: it is **not** a function of $B$. (At $d=1$ the constraint $F(v)=B$ pins
the single $v$, so $D_{\text{tot}}$ is fixed — the allocation freedom of L2 genuinely requires
$d\ge2$, satisfied here with $d=768$.)

**(2c) Allocation-blindness.** $I_G$ depends on $v$ only through the multiset $\{\lambda_i/v_i\}$; two
allocations $v,v'$ with the same multiset give identical $I_G$ yet place noise on different original
directions. Model any fixed decoder by a nonnegative read-weight vector $w\in\mathbb R^d_{\ge0}$,
$w\ne0$ (the directions whose fidelity drives recovery), and let the realized read-subspace distortion
be $W(v)=\sum_i w_i v_i$. We claim $W$ is **not** a function of $I_G$ on the level set, *provided some
tail mode is read* (i.e. $w_j>0$ for some $j$ with $\lambda_j$ among the smaller eigenvalues — the
generic case, since a fixed text decoder reads a broad subspace). Indeed, by the construction in (2b)
(with the loaded mode chosen as such a $j$, possible because $d\ge2$), the tail-loaded family attains
the same budget $B$ with $v_j\to\infty$, so $W(v)\ge w_j v_j\to\infty$ at fixed $B$, while `sgt_opt`
keeps $W$ finite. Hence at matched $I_G=B$, $W$ ranges over an interval **unbounded above** as
$v_j\to\infty$. (If $w$ happens to be supported only on modes that the construction leaves clean, pick
the loaded coordinate inside $\mathrm{supp}(w)$ — possible whenever $|\mathrm{supp}(w)|\ge2$.) Thus
$I_G$ structurally cannot encode the alignment between the noise allocation and the decoder's read
directions. ($\blacksquare$ L2)

### Lemma 3 (metric-bound decoder ⇒ recovery tracks read-subspace distortion)

**(3a) Telescoping bound.** For the true text $x$ and graded score $s(\cdot,x)\in[0,1]$, write the
recovery on the clean embedding $R_0=\mathbb E\,s(\hat g(e_0),x)$ and on the release
$R(D)=\mathbb E\,s(\hat g(Y),x)$. By A4 applied in expectation,
$R_0-R(D)\le|R_0-R(D)|\le L\,\mathbb E[1-\text{relCos}(e_0,Y)].$
So the recovery *degradation* $R_0-R(D)$ is upper-bounded by the expected cosine distortion of the
release, scaled by the decoder's Lipschitz constant $L$.

**(3b) Cosine distortion is shape-dependent at fixed $I_G$, controlled by total noise variance
$D_{\text{tot}}$.** We bound $\mathbb E[1-\text{relCos}]$ rigorously by the total distortion (which
Lemma 2b governs exactly), avoiding any high-$\lambda$/low-$\lambda$ claim about angular geometry.

*Rigorous global upper bound.* For any nonzero $e,n$ with $y=e+n$, one has the global estimate
$1-\cos(e,y)\le \dfrac{2\|n\|^2}{\|e\|^2}$ (for $\|n\|^2/\|e\|^2\ge1$ the right side is $\ge2\ge1-\cos$;
for $\|n\|^2/\|e\|^2<1$ it dominates the standard small-perturbation bound). Conditioning on $e_0$ and
using $\mathbb E[\|N\|^2\mid e_0]=D_{\text{tot}}$ (independent of $e_0$ by A3),
$$\mathbb E[1-\text{relCos}]\ \le\ 2\,D_{\text{tot}}\;\mathbb E\!\big[\|e_0\|^{-2}\big],$$
provided $\mathbb E[\|e_0\|^{-2}]<\infty$ (holds under A1 for $d\ge3$, $\Sigma\succ0$; assumed for
general $e_0$, flagged in Open Risks). So at **fixed $I_G=B$**, the cosine distortion — and by (3a) the
recovery degradation — has an upper bound proportional to $D_{\text{tot}}$, and by Lemma 2b
$D_{\text{tot}}$ is *minimized* by `sgt_opt` (reverse-water-filling). This certifies a *small* loss
ceiling for `sgt_opt`; the matched mechanism is that the deployed decoder's loss is driven by total
noise energy, which $I_G$ (allocation-blind, L2) does not pin down.

*Tail limit (DCT, separate argument).* For `tail_dump` the loss is forced *large* by a direct limit.
Write the noise on the loaded mode as $N_j=\sqrt{v_j}\,Z_j$, $Z_j\sim\mathcal N(0,1)$. As
$v_j\to\infty$ the release direction $Y/\|Y\|$ concentrates on the $j$-axis: $\cos(e_0,Y)\to
\mathrm{sign}(Z_j)\,e_{0j}/\|e_0\|$ a.s. (not $0$ in general). Taking expectations, since $Z_j$ is
symmetric and independent of $e_0$ the two signs cancel, $\mathbb E[\cos(e_0,Y)]\to0$, and as
$|1-\cos|\le2$ is bounded, dominated convergence gives $\mathbb E[1-\text{relCos}]\to1$. So at matched
$B$, $\mathbb E[1-\text{relCos}]$ ranges from small (`sgt_opt`/`iso`, certified by the $D_{\text{tot}}$
bound) to $\to1$ (`tail_dump`, by this limit). Empirically the certified upper bound is smallest for
`sgt_opt`, and the observed ordering matches: $\text{relCos}=\{0.957,0.964,0.023\}$ for
$\{$`iso`,`sgt_opt`,`tail_dump`$\}$ at $B=826.8$ with $D_{\text{tot}}=\{0.11,0.09,2419\}$ — the
$D_{\text{tot}}$ bound certifies the small-loss end and the DCT limit proves the large-loss end (the
exact ordering between `iso` and `sgt_opt` is empirical, not claimed from the one-sided bound).

*Note on directionality (correcting a tempting error).* It is **not** true in general that loading
noise on high-$\lambda$ modes maximizes angular distortion; angular distortion is driven by the noise
*orthogonal* to $e_0$ relative to $\|e_0\|^2$. The clean and sufficient statement is the
$D_{\text{tot}}$ bound above: among matched-$I_G$ allocations, the one minimizing total variance
(`sgt_opt`) minimizes the cosine-distortion ceiling, while the bit-cheap tail loading inflates total
variance without spending budget, maximizing it. The empirical relCos/$D_{\text{tot}}$ ranking
(above) bears this out.

**(3c) Conclusion.** By (3a), the *deployed* fixed decoder's recovery degradation is controlled by
$\mathbb E[1-\text{relCos}]$, which by (3b) is upper-bounded by a constant times $D_{\text{tot}}$ —
shape-dependent at fixed $I_G$ (Lemma 2b) — meanwhile by L2 $I_G$ is allocation-blind. Hence the
predictor *matched to this fixed metric-bound decoder* is total/read-subspace distortion (relCos),
empirically $\mathrm{Spearman}=0.972$/$0.951$, while the scalar $I_G$ is only a loose shape-invariant
converse ($\mathrm{Spearman}=0.482$). Two claims, of different strength:

- **Structural (L2, independent of A4):** $I_G$ is a scalar function of the whitened spectrum and
  therefore *cannot distinguish* two matched-budget allocations that differ only in which modes carry
  the noise. So no scalar reading of $I_G$ can rank such allocations by recovery — $I_G$ is
  **allocation-blind** as a probe. This is proven.
- **Decoder-model (L3, under A4):** the *deployed* fixed Vec2Text corrector appears matched to
  relCos / $D_{\text{tot}}$ (its degradation is bounded by them and tracks them empirically). This
  rests on the modelling assumption A4, corroborated post hoc.

What is **left open** (not claimed): whether a *shape-aware* decoder — one adapted to the release
distribution, e.g. whitening $Y$ by $D$ before correction or retraining per shape — could recover the
bits $I_G$ certifies are present in the tail modes, thereby re-saturating the converse and
re-correlating recovery with $I_G$. That is the **weak-attack** hypothesis; this experiment neither
demonstrates nor refutes it, and it is queued as the spawn-depth-1 follow-up. ($\blacksquare$ L3)

Therefore the three lemmas hold and jointly explain the does-not-correlate finding. $\blacksquare$

## Corrections or Missing Assumptions

- The original campaign claim "the scalar matched MI converse $I_G$ **predicts** Vec2Text recovery"
  is **corrected** to: "$I_G$ is a valid converse and a within-shape monotone predictor, but is
  shape-blind across noise shapes at matched budget; the deployed fixed corrector's recovery is
  predicted by read-subspace distortion / relCos." The original (uncorrected) claim is refuted by the
  matched-budget counterexample (token-F1 $0.048$ vs $0.566$ at equal $I_G$).

## Open Risks

- **(A4 is a model, not a measurement.)** L3's Lipschitz-in-cosine assumption is corroborated only
  post hoc (relCos Spearman $0.972$). A decoder that reads non-cosine features could deviate; L1–L2
  (incl. the structural allocation-blindness verdict) are unaffected.
- **(Inverse-moment integrability.)** The $D_{\text{tot}}$ bound in (3b) needs
  $\mathbb E[\|e_0\|^{-2}]<\infty$ (holds for the Gaussian surrogate A1 at $d\ge3$, $\Sigma\succ0$;
  $d=768$ here). The clip ($\|e_0\|\le C$) and $\Sigma\succ0$ keep the experiment well inside this; a
  general $e_0$ with mass at the origin would need A2′ strengthened.
- **(3b's pointwise expansion is one-sided / leading-order.)** The bound $1-\cos\le 2\|n\|^2/\|e\|^2$
  gives a rigorous *upper* bound on $\mathbb E[1-\text{relCos}]$ via $D_{\text{tot}}$; a matching
  *lower* bound (showing `tail_dump` is forced large rather than merely permitted) is supplied via the
  DCT tail-limit, not a uniform two-sided estimate. The allocation-blindness verdict rests on L2, not
  on a tight relCos map.
- **(Degenerate modes / dimension.)** A2 excludes $\lambda_i=0$ / $v_i=0$ and assumes $d\ge2$ (L2
  allocation freedom) and $d\ge3$ (3b integrability); the clip and reported $d_{\text{eff}}$ keep the
  experiment inside A2, and $d=768$.
- **(Read-weight support.)** L2(2c) needs the decoder to read at least one tail-loadable mode
  ($w_j>0$ with $|\mathrm{supp}(w)|\ge2$) — generic for a broad-subspace text corrector but an
  explicit modelling choice.
- **(Weak-attack arm unresolved.)** Whether a shape-aware / distribution-matched decoder re-saturates
  $I_G$ on `tail_dump` is open — the spawn-depth-1 follow-up.

## Measurement-loop verdict

Does-NOT-correlate-across-shape (Spearman $0.482<0.6$). Per CLAUDE.md this *is* the finding: the gap
is bounded/explained (L1–L3), the diagnosis is **non-matched probe** (L2, structural) with a
**weak-attack** arm queued. First-class negative result.

## Connections

- Empirical support: `exp:embed-sgt-channel-mi-shape-blindness` (this surface's sweep).
- Contrast: `claim:spectral-channel-mi-embedding-inversion` (Block A) — same probe is a *complete*
  predictor under isotropic DP (single shape); SGT adds the shape axis where it fails.
- Sibling negative (matched-probe-vs-weak-attack ambiguity): `claim:gelo-orthogonal-gram-leak-rowmix-defeats-bss`.

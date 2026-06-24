# Proof Package — PriPert spectral converse: valid, β-co-monotone, but slack

## Claim

Let a split-inference adversary observe the Gaussian PriPert channel
$$U = S + Z,\qquad S=\mathrm{Sp}_\rho(H)\in\mathbb R^{d},\qquad Z\sim\mathcal N(0,\sigma^2 I_d)\ \perp\ (X,H,S),$$
where $X$ is the secret token id, $H$ the clean residual row, $\mathrm{Sp}_\rho$ keeps the
$\lceil\rho d\rceil$ largest-magnitude coordinates per row (ties broken by a fixed coordinate
order) and zeroes the rest, and $\sigma=\beta c$ with $c=\text{meanRMS}(H_{\text{plaintext}})>0$
a fixed per-layer constant and $\beta>0$. Let $\Sigma_S=\operatorname{Cov}(S)$ have eigenvalues
$\lambda_1\ge\cdots\ge\lambda_d\ge0$, $I_G(\sigma,\Sigma_S)=\tfrac12\sum_i\log_2(1+\lambda_i/\sigma^2)$,
and $d_{\mathrm{eff}}=\#\{i:\lambda_i\ge\sigma^2\}$. Under a uniform prior on the $M$-candidate pool
($M\ge2$, $H_X:=H(X)=\log_2 M$):

- **(L1, validity)** For *every* estimator $\hat X(U)$, top-1 accuracy obeys
  $\mathrm{acc}\le \min\!\big(1,\ (\min(H_X,I_G)+1)/H_X\big)=:\text{fano}$. So $I_G$ yields an
  attack-independent converse on recovery (certifying the measured 0/32 violations).
- **(L2, β-axis co-monotonicity)** Along the perturbation axis (fixed $\rho$), $I_G$ is
  non-increasing in $\sigma$ (strictly, if $\Sigma_S\neq0$) and the Bayes-optimal accuracy
  $\mathrm{acc}^\star$ is non-increasing in $\sigma$. Hence on any fixed-$\rho$ noise sweep $I_G$ and
  $\mathrm{acc}^\star$ are co-monotone — a proved one-dimensional rank agreement. (Monotonicity in
  $\rho$ is **not** claimed here: it is false in general for the magnitude mask — see L2-Remark.)
- **(L3, slack)** The ceiling is non-vacuous only if $I_G<H_X-1$, which forces
  $d_{\mathrm{eff}}<2H_X-2$. Equivalently, whenever $d_{\mathrm{eff}}\ge 2H_X-2$ the Fano ceiling
  equals $1$ (vacuous). So if the recovery floor is reached at a noise level where the surviving
  spectrum is still high rank ($d_{\mathrm{eff}}\ge 2H_X-2$), the converse is vacuous there; the
  empirical separation $\beta^{\star\star}\gg\beta^\star$ is then a non-vacuous-versus-floored gap.

## Status

- **L1**: PROVABLE AS STATED (for $\sigma>0$, $M\ge2$).
- **L2 (β-axis)**: PROVABLE AS STATED. The earlier $\rho$-axis monotonicity of $\mathrm{acc}^\star$
  and of $I_G$ under the *magnitude* mask is **withdrawn** — it is false in general (counterexample
  in L2-Remark). $\rho$-monotonicity holds only for *fixed-support* sparsification (L2b').
- **L3**: the deterministic part (non-vacuity $\Rightarrow d_{\mathrm{eff}}<2H_X-2$; contrapositive
  $\Rightarrow$ vacuous) is PROVABLE AS STATED. The quantitative $\beta^{\star\star}\gg\beta^\star$
  separation is **EMPIRICAL/HEURISTIC**, explicitly labeled (Open Risks), with measured evidence
  $\beta^{\star\star}/\beta^\star\approx4$ at L8.
- **Joint-sweep tracking** (across mixed $\rho,\beta$, and across layers) is **EMPIRICAL** — not
  implied by the one-dimensional L2 (co-monotonicity along one axis does not force a consistent
  ranking across mixed-axis comparisons). The proved theory covers fixed-$\rho$ noise slices only.

## Assumptions

1. $Z\sim\mathcal N(0,\sigma^2 I_d)$, $\sigma=\beta c>0$, independent of $(X,H,S)$ (channel-matched
   Gaussian δ proxy; the paper's adversarial-optimized δ is out of scope).
2. Uniform prior on the $M$-token candidate pool, $M\ge2$, so $H_X=\log_2 M$ (max-entropy prior).
3. $\Sigma_S$ finite, PSD. Markov chain $X\to H\to S\to U$ ($S$ a function of $H$; $Z$ independent).
4. $\mathrm{Sp}_\rho$ uses a fixed total-order tie-break, so the kept support is a deterministic
   function of $H$.

## Notation

$X$ token id (secret); $H\in\mathbb R^d$ clean residual row; $S=\mathrm{Sp}_\rho(H)$; $U=S+Z$;
$\Sigma_S$, $\lambda_i$, $I_G$ (bits), $d_{\mathrm{eff}}$, $H_X=\log_2 M$ as above;
$\mathrm{acc}^\star(\sigma,\rho)=\max_{\hat X}\Pr[\hat X(U)=X]$ Bayes-optimal top-1 accuracy;
$h(\cdot)$ differential entropy; $H_b$ binary entropy.

## Proof Strategy

L1: data-processing + Gaussian max-entropy (Lemma A, cited) + Fano. L2: calculus for
$\partial_\sigma I_G$; stochastic degradation (added independent noise is a Blackwell-degradation
kernel) for $\mathrm{acc}^\star$. L3: the elementary bound $I_G\ge d_{\mathrm{eff}}/2$ and the
non-vacuity condition.

## Dependency Map

1. L1 ← Lemma A (cited, verified) + data-processing + Fano.
2. L2(a) ← differentiation of $I_G$ in $\sigma$.
3. L2(c) ← stochastic degradation in $\sigma$ (data-independent noise kernel).
4. L2b' (fixed-support $\rho$, optional) ← principal-submatrix determinant inequality.
5. L3 ← $I_G\ge d_{\mathrm{eff}}/2$ and non-vacuity $I_G<H_X-1$.
No cycles.

## Proof

### Lemma A (cited). Gaussian channel max-entropy bound ($\sigma>0$).
For $U=S+Z$, $Z\sim\mathcal N(0,\sigma^2 I_d)\perp S$, $\sigma>0$, with $\operatorname{Cov}(S)=\Sigma_S$,
$$I(S;U)=h(U)-h(U\mid S)=h(U)-h(Z)\le\tfrac12\log_2\det\!\big(I+\Sigma_S/\sigma^2\big)=I_G,$$
using $h(U)\le\tfrac12\log_2((2\pi e)^d\det\operatorname{Cov}(U))$ and $\operatorname{Cov}(U)=\Sigma_S+\sigma^2 I$.
Holds for any (non-Gaussian) $S$ of covariance $\Sigma_S$; equality iff $S$ Gaussian. This is the
verified converse ceiling of [[spectral-channel-mi-probe-decision]]; invoked within its conditions
(independent additive Gaussian noise, fixed covariance, $\sigma>0$). $\square$

### L1 — converse validity.
By Assumption 3, $X\to S\to U$ is Markov, so data-processing gives $I(X;U)\le I(S;U)$, and with
Lemma A, $I(X;U)\le I_G$. Also $I(X;U)\le H(X)=H_X$. Hence $I(X;U)\le A:=\min(H_X,I_G)$.
For any estimator $\hat X(U)$ with error $P_e=\Pr[\hat X\ne X]$, Fano's inequality with support $M$:
$$H(X\mid U)\le H_b(P_e)+P_e\log_2(M-1)\le 1+P_e\log_2 M = 1+P_e H_X,$$
using $H_b(P_e)\le1$ and $\log_2(M-1)\le\log_2 M$. Since $H(X\mid U)=H_X-I(X;U)\ge H_X-A$,
$$H_X-A\le 1+P_e H_X\ \Longrightarrow\ P_e\ge 1-\frac{A+1}{H_X}\ \Longrightarrow\ \mathrm{acc}=1-P_e\le\frac{A+1}{H_X}.$$
Clamping, $\mathrm{acc}\le\min(1,(A+1)/H_X)=\text{fano}$ (the `fano_exact_ceiling` of
`spectral_channel_mi.py:177`). Applies to the inverter as a special case ⇒ 0 violations certified. $\blacksquare$

### L2(a) — $I_G$ non-increasing in $\sigma$ (strict if $\Sigma_S\neq0$).
$I_G=\frac{1}{2\ln2}\sum_i\ln(1+\lambda_i/\sigma^2)$. For $\sigma>0$,
$$\frac{\partial I_G}{\partial\sigma}=-\frac{1}{\ln2}\sum_i\frac{\lambda_i}{\sigma^3+\sigma\lambda_i}\le0,$$
with strict inequality iff some $\lambda_i>0$ (i.e. $\Sigma_S\neq0$). Since $\sigma=\beta c$, $c>0$,
$I_G$ is non-increasing in $\beta$, strictly when $\Sigma_S\neq0$. (If $\Sigma_S=0$, $I_G\equiv0$.) $\blacksquare$

### L2(c) — $\mathrm{acc}^\star$ non-increasing in $\sigma$.
For $\sigma'>\sigma>0$, $U(\sigma')\stackrel d= U(\sigma)+W$ with $W\sim\mathcal N(0,(\sigma'^2-\sigma^2)I)$
independent of $U(\sigma)$ (and of $X$). This $W$ is **data-independent**, so the map
$U(\sigma)\mapsto U(\sigma)+W$ is a valid stochastic (Blackwell-degradation) kernel and
$X\to U(\sigma)\to U(\sigma')$ is Markov. Bayes-optimal accuracy cannot increase under
post-processing, so $\mathrm{acc}^\star(\sigma')\le\mathrm{acc}^\star(\sigma)$; equivalently
$\mathrm{acc}^\star$ is non-increasing in $\beta$ at fixed $\rho$. $\blacksquare$

**β-axis co-monotonicity corollary.** Along any fixed-$\rho$ noise sweep, $I_G$ (L2a) and
$\mathrm{acc}^\star$ (L2c) are both non-increasing in $\beta$, hence induce the same ranking ⇒
Spearman$(I_G,\mathrm{acc}^\star)=+1$ on distinct values. This is the proved structural cause of the
measured **within-layer** Spearman $=1.0$ at L8/L16/L24 (each a fixed-$\rho{=}0.25$ $\beta$-sweep).
The realized ridge/mlp2 accuracies satisfy $\mathrm{acc}_{\text{ridge/mlp2}}\le\mathrm{acc}^\star$;
their co-monotonicity (measured Spearman$(I_G,\text{mlp2})=0.915$) is empirical evidence the realized
attacks are near-optimally ordered, **not** a corollary. The **joint/mixed-axis and $\rho$-axis**
rank tracking (pooled $0.958$; fixed-$\beta$ layer$\times\rho$ slice $0.916$) is likewise
**empirical** — co-monotonicity along one axis does not force a consistent ranking across
comparisons that move both $\rho$ and $\beta$.

### L2-Remark — why $\rho$-monotonicity is NOT claimed (counterexample).
For the data-dependent magnitude mask, sparsification can *increase* $\mathrm{acc}^\star$ (and change
$I_G$ non-monotonically), because *which* coordinates survive is itself informative. Take $d=2$,
$X\in\{1,2\}$ uniform, $H_1=(a,b)$, $H_2=(b,a)$ with $a>b>0$ (deterministic given $X$). At $\rho=1$
the means are $(a,b),(b,a)$ at distance $\sqrt2\,(a-b)$. At $\rho=0.5$ (top-1) the means become
$(a,0),(0,a)$ at distance $\sqrt2\,a>\sqrt2(a-b)$ — *larger* separation, so at fixed $\sigma$,
$\mathrm{acc}^\star(\rho{=}0.5)>\mathrm{acc}^\star(\rho{=}1)$. The Markov reduction
$X\to U_{\rho'}\to U_\rho$ fails: the clean top-$k$ support cannot be recovered from the noisy
$U_{\rho'}$, so no degradation kernel exists. Hence neither $\mathrm{acc}^\star$ nor $I_G$ is
provably monotone in $\rho$ under the magnitude mask. (Empirically on this surface $\rho$ is the
*secondary* knob and $I_G$ moves monotonically with $\rho$, but this is observation, not theorem.)

### L2b' (fixed-support, optional). $I_G$ monotone in $\rho$ for nested fixed masks.
If $\rho<\rho'$ use **nested fixed** supports $K\subseteq K'$ (a fixed mask, independent of $H$),
then $\Sigma_{S(\rho)}$'s nonzero block is the principal submatrix $(\Sigma_{S(\rho')})_{KK}$, and
with $M:=I+\Sigma_{S(\rho')}/\sigma^2\succeq I$, $\det M=\det M_{KK}\cdot\det(M/M_{KK})$ where the
Schur complement $M/M_{KK}\succeq I_{K^c}$ (a principal Schur complement of a matrix $\succeq I$ is
$\succeq I$), so $\det(M/M_{KK})\ge1$ and $\det M\ge\det M_{KK}=\det(I+(\Sigma_{S(\rho')})_{KK}/\sigma^2)$.
Thus $I_G(\Sigma_{S(\rho)})\le I_G(\Sigma_{S(\rho')})$. This applies only to fixed (oracle/expected)
supports, **not** the data-dependent magnitude mask (L2-Remark). $\square$

### L3 — the converse is slack.
Non-vacuity: $\text{fano}<1\iff\min(H_X,I_G)<H_X-1\iff I_G<H_X-1$ (since $\min(H_X,I_G)=H_X\ge H_X-1$
when $I_G\ge H_X$). Each retained mode ($\lambda_i\ge\sigma^2$) contributes
$\tfrac12\log_2(1+\lambda_i/\sigma^2)\ge\tfrac12\log_2 2=\tfrac12$ bit, so $I_G\ge\tfrac12 d_{\mathrm{eff}}$.
Therefore
$$\text{fano}<1\ \Longrightarrow\ I_G<H_X-1\ \Longrightarrow\ d_{\mathrm{eff}}<2H_X-2.$$
Contrapositive: **if $d_{\mathrm{eff}}\ge 2H_X-2$ then $\text{fano}=1$ (vacuous).** When
$2\le\lceil 2H_X-2\rceil\le d$, $d_{\mathrm{eff}}<2H_X-2$ requires $\sigma^2>\lambda_{\lceil 2H_X-2\rceil}$,
i.e. $\beta>\sqrt{\lambda_{\lceil 2H_X-2\rceil}}/c=:\beta^{\star\star}$. Boundary index conventions:
if $\lceil 2H_X-2\rceil>d$ the necessary condition $d_{\mathrm{eff}}<2H_X-2$ holds automatically and
gives *no* useful eigenvalue threshold — non-vacuity must then be checked directly via $I_G<H_X-1$;
if $M=2$ then $H_X-1=0$ and non-vacuity is impossible ($\text{fano}\equiv1$).

*Empirical slack (not a theorem).* Define the recovery floor $\beta^\star=\inf\{\beta:
\mathrm{acc}_{\text{best}}\le\varepsilon\}$ (estimator-dependent). A recovery attack reads the secret
from a few predictive directions and floors once those drown — long before all but $<2H_X-2$ modes
drown. With $H_X\approx11.2$ ($M\approx2300$ at L8) the converse needs $d_{\mathrm{eff}}<\sim20$,
whereas at L8 $\beta^\star\approx0.5$ already gives $\mathrm{acc}_{\text{best}}=0.019$ while
$I_G=55.7$ ($d_{\mathrm{eff}}\gg20$); the ceiling only binds at $\beta^{\star\star}\approx2$
($I_G=10.0<H_X-1\approx10.2$), so $\beta^{\star\star}/\beta^\star\approx4$. The interval
$[\beta^\star,\beta^{\star\star}]$ — recovery floored, converse vacuous — is observed, and explained
by $d_{\mathrm{eff}}(\beta^\star)\gg 2H_X-2$; we do not prove this gap from Assumptions 1–4. $\blacksquare$

### Theorem (tie-together).
Under Assumptions 1–4, the spectral channel-MI probe is (i) a **valid** attack-independent recovery
converse [L1]; (ii) **co-monotone with $\mathrm{acc}^\star$ along the perturbation ($\beta$) axis at
fixed $\rho$** [L2], the proved structural reason the probe rank-tracks recovery on within-layer
$\beta$-sweeps (measured Spearman $=1.0$); the joint/$\rho$-axis tracking (pooled $0.958$) is
empirical, and $\rho$-monotonicity is false in general (L2-Remark); and (iii) its Fano certificate is
**vacuous whenever $d_{\mathrm{eff}}\ge 2H_X-2$** [L3], so on a high-rank residual the converse is
slack through the floored regime $[\beta^\star,\beta^{\star\star}]$ (empirically
$\beta^{\star\star}/\beta^\star\approx4$). Hence on this surface the probe's **predictive
(rank-tracking)** value, not its **certificate (Fano)** value, carries the measurement loop. $\blacksquare$

## Corrections or Missing Assumptions
- **Withdrawn**: $\rho$-axis monotonicity of $\mathrm{acc}^\star$ and of $I_G$ under the magnitude
  mask (false; L2-Remark counterexample). The proved co-monotonicity is the $\beta$-axis only.
- **Withdrawn**: the joint-sweep "co-monotonicity ⇒ rank tracking" implication; only one-dimensional
  $\beta$-slice tracking is proved, joint Spearman is empirical.
- Added $\sigma=\beta c>0$, $M\ge2$, fixed tie-break, and boundary index conventions.
- "no probe–attack gap" is scoped to the *tested* inverters; L2 bounds $\mathrm{acc}^\star$, not the
  realized attack.

## Open Risks
1. **$\rho$-monotonicity of $I_G$ under the magnitude mask** is empirical only (theorem withdrawn).
2. **$\beta^{\star\star}\gg\beta^\star$ slack separation** is empirical/heuristic (the deterministic
   "$d_{\mathrm{eff}}\ge2H_X-2\Rightarrow$ vacuous" half is proved; the gap size is measured).
3. **Joint-sweep Spearman (0.958 / 0.916)** is empirical; theory proves only fixed-$\rho$ $\beta$-slice rank agreement.
4. **Uniform-prior** $H_X=\log_2 M$ (max-entropy pool prior). Non-uniform priors do **not** reduce
   to "replace $\log_2 M$ by $H(X)$": Fano keeps the support size $M$ in the error denominator
   ($H(X\mid U)\le1+P_e\log_2(M-1)$), so a separate formulation is required and the bound may be much
   looser (a skewed prior raises the base-rate accuracy of guessing the mode). The numeric
   $\approx11$ bits is the uniform-pool value.
5. **Gaussian-δ proxy**; adversarial-optimized δ out of scope.

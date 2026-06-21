---
type: theory
status: current
created: 2026-06-21
updated: 2026-06-21
tags: [proof, information-efficient-attacks, MMSE, I-MMSE, Bayes-risk, DPI, blackwell, calibration]
companion: [interpretability-leakage-bridge, it-leakage-estimation-set]
---

# Theorem T1 — Information-efficient attacks dominate weak attacks and (Gaussian arm) track mutual information

The theoretic-guarantee gate (B0) for the information-efficient-attack experiment
plan (`refine-logs/EXPERIMENT_PLAN.md`). It formalises why the deployed weak
attacks (linear ridge inversion; lossy hand-crafted matchers) recover far below
the information ceiling under noise, why a Bayes-optimal noise-aware attack
strictly improves, and why — on the Gaussian (input-DP) arm — the optimal
attack's recovery is a monotone function of mutual information while a fixed weak
attack need not be. This is what makes "stronger attack ⟹ recovery re-correlates
with the MI probes" a *prediction*, not a hope.

## Definitions

| Symbol / term | Meaning |
|---|---|
| $S$ | the secret to recover — a **token** $S\in\mathcal S$, $\lvert\mathcal S\rvert<\infty$ (top-1 task), or its **embedding** $S\in\mathbb R^{d}$ (reconstruction task). |
| $X$ | the clean observation (hidden state / weight row), jointly distributed with $S$. |
| $Y$ | the **defended** observation: output of an additive-noise channel applied to $X$. |
| $N$ | channel noise, independent of $(S,X)$. Gaussian arm: $N\sim\mathcal N(0,I_d)$. |
| $\mathrm{snr}$ | signal-to-noise ratio of the Gaussian channel $Y_{\mathrm{snr}}=\sqrt{\mathrm{snr}}\,X+N$. |
| $\varphi$ | a **statistic** $\varphi(Y)$ to which a weak attack is restricted (e.g. the best-affine ridge feature, or RowSort $\varphi$). |
| $\delta$ | a decision rule $\delta:Y\mapsto\hat S$; $\delta_w=\psi\circ\varphi$ is $\varphi(Y)$-measurable. |
| $L,\;R(\delta)$ | loss $L(s,\hat s)$ and risk $R(\delta)=\mathbb E[L(S,\delta(Y))]$. 0–1 loss (token), squared loss $\lVert\cdot\rVert^2$ (embedding). |
| $A^\*$ | the **Bayes-optimal** attack on $Y$: $\delta^\*(y)=\arg\min_{\hat s}\mathbb E[L(S,\hat s)\mid Y=y]$ (posterior mean for squared loss; MAP for 0–1). |
| $r^\*(\,\cdot\,)$ | Bayes risk attainable from a $\sigma$-algebra: $r^\*(Y)=\inf_{\delta\ \sigma(Y)\text{-meas.}}R(\delta)$. |
| $\mathrm{mmse}(\mathrm{snr})$ | $\mathbb E\lVert X-\mathbb E[X\mid Y_{\mathrm{snr}}]\rVert^2$, the channel-input MMSE. |
| LMMSE | best **affine** estimator (the population object the ridge attack approximates). |
| DPI | data-processing inequality. **Degradation order**: $Y'$ is degraded from $Y$ if $\exists$ channel with $Y'\sim \mathrm{garble}(Y)$, i.e. $S\!-\!Y\!-\!Y'$ Markov. |

## Status

**VERIFIED — PROVABLE AS STATED** (gpt-5.5 xhigh proof-checker, 2 rounds, verdict
PASS; round-1 issues I1–I7 resolved, N1 closed; `PROOF_AUDIT.json`). Scope made
precise in three places (these are
*statements of the theorem*, not weakenings of a stronger false claim):
(b) strictness is **loss-specific** (squared-loss / non-affine conditional mean),
not implied by MI-loss alone for an arbitrary fixed loss; (c) the **exact**
$I$–MMSE identity holds for the **channel-input** target $S=X$, while the
**token** target $S\neq X$ gets monotonicity via the degradation order (no
$\tfrac12$-MMSE identity); the Laplace/Shredder arm uses the degradation order
throughout. The "weak attack need not track MI" statement is a **non-implication
(Remark)**, not a proven positive theorem.

## Assumptions

- **(A1) Markov structure.** $S\!-\!X\!-\!Y$: the channel acts on $X$ and is
  independent of $S$ given $X$ ($N\perp(S,X)$).
- **(A2) Moments & regularity.** Squared-loss parts require
  $\mathbb E\lVert S\rVert^2<\infty$ and $\mathbb E\lVert Y\rVert^2<\infty$ (for
  the affine/LMMSE comparison; automatic in the Gaussian arm from
  $\mathbb E\lVert X\rVert^2<\infty$, which also defines $\mathrm{mmse}$); 0–1-loss
  parts require $\lvert\mathcal S\rvert<\infty$. A measurable Bayes rule is assumed
  to exist (automatic for these two losses on Euclidean $Y$ / finite $\mathcal S$
  via regular conditional distributions; for a general loss/action space, read the
  statements with the infimum $r^\*$). For genuinely infinite-variance heavy tails
  the squared-loss apparatus is replaced by an $L^1$/conditional-median criterion
  (out of scope; Open Risks).
- **(A3) Gaussian arm.** Part (c)'s exact identity assumes
  $Y_{\mathrm{snr}}=\sqrt{\mathrm{snr}}\,X+N$, $N\sim\mathcal N(0,I_d)$. The
  Laplace/Shredder arm assumes only an additive channel that induces a
  degradation order in its noise level.

## Proof strategy

Three independent, standard pillars, each cited and then specialised to the
attack instances: (a) Bayes optimality + $\sigma$-algebra monotonicity of Bayes
risk (Blackwell comparison of experiments); (b) the MMSE orthogonality
decomposition for the squared-loss/linear-ridge instance; (c) the I-MMSE
identity (Guo–Shamai–Verdú) for $S=X$ and the Gaussian degradation order for
$S\neq X$ and for non-Gaussian noise.

## Dependency map

1. **(a)** depends on: Bayes rule optimality; $\sigma(\varphi(Y))\subseteq\sigma(Y)$; Blackwell (1953) post-processing; Rao–Blackwell (convex loss).
2. **(b)** depends on: MMSE $=\mathbb E[S\mid Y]$ (squared loss); orthogonality/Pythagorean decomposition; LMMSE $=$ MMSE $\iff$ $\mathbb E[S\mid Y]$ affine (Kay Ch.10–12).
3. **(c-exact)** depends on: I-MMSE (Guo–Shamai–Verdú 2005, Thms 1–2); monotonicity of $\mathrm{mmse}$ in $\mathrm{snr}$.
4. **(c-token / Laplace)** depends on: Gaussian channels are ordered by degradation; DPI (Cover–Thomas Thm 2.8.1); Bayes-risk monotone under degradation (Blackwell).
5. **Ceiling (Prop. 0)**: Fano (Cover–Thomas Thm 2.10.1) / de Chérisey et al. (2019).

---

## Proof

### Proposition 0 (the ceiling — context).
Under (A1)–(A2), for the token target the error probability of **any** attack
obeys Fano's inequality (all logs **base 2**; the additive $1$ is the loose
$H_b(P_e)\le1$ bit constant)
$$ P_e \;\ge\; \max\!\Big\{0,\;\frac{H(S\mid Y)-1}{\log_2\lvert\mathcal S\rvert}\Big\} \;=\; \max\!\Big\{0,\;\frac{H(S)-I(S;Y)-1}{\log_2\lvert\mathcal S\rvert}\Big\}, $$
(Cover–Thomas, Thm 2.10.1; in nats replace the constant $1$ by $\log 2$), so the
best achievable recovery $1-P_e$ is bounded by
a monotone function of $I(S;Y)$; de Chérisey et al. (2019, TCHES) give the
matching success-rate form $d_P(P_s)\le I(S;\hat S)\le I(S;Y)$. This fixes the
**information ceiling**: recovery cannot exceed what $I(S;Y)$ allows, but a given
attack may sit arbitrarily far below it. T1(a–c) concerns *closing that gap*. ∎

### Part (a) — Weak domination (every prior; the stated losses; any noise).
We take $L$ to be the **squared loss** (embedding) or **finite-alphabet 0–1 loss**
(token); under (A2) a measurable Bayes rule $\delta^\*$ exists for each by
minimising the posterior expected loss pointwise (regular conditional
distributions exist on Euclidean $Y$ / finite $\mathcal S$, and the action space
admits a measurable selector — for a general loss/action space one would need an
extra lower-semicontinuity/selector hypothesis, or state the result with the
infimum $r^\*$ in place of an attaining rule). Since $\delta^\*$ minimises
$R(\cdot)$ over **all** $\sigma(Y)$-measurable rules,
$$ R(A^\*) \;=\; r^\*(Y) \;=\!\!\inf_{\delta\ \sigma(Y)\text{-meas.}}\!\! R(\delta)\;\le\; R(\delta_w)\quad\text{for any }\delta_w. $$
If moreover $\delta_w=\psi\circ\varphi$ is $\varphi(Y)$-measurable, then because
$\sigma(\varphi(Y))\subseteq\sigma(Y)$ the infimum over the smaller class is no
smaller — this is **elementary by inclusion of $\sigma$-algebras**:
$$ r^\*(Y)\;\le\;r^\*(\varphi(Y))\;\le\;R(\delta_w). $$
Equivalently, $Y$ Blackwell-dominates its garbling $\varphi(Y)$ (Blackwell 1953,
Thms 5–6), but the inclusion argument suffices and needs no comparison-of-
experiments machinery. (Rao–Blackwellisation $\mathbb E[\delta_w\mid Y]$ is *not*
the right device here: if $\delta_w$ is already $Y$-measurable then
$\mathbb E[\delta_w\mid Y]=\delta_w$; it only strictly helps for *randomised*
rules under convex loss.) The linear-ridge attack is one particular
$\sigma(Y)$-measurable rule, hence $R(A^\*)\le R(\text{ridge})$. No assumption on
the noise distribution is used. ∎

### Part (b) — Strict improvement (squared-loss / linear-ridge instance).
Take the embedding target $S\in\mathbb R^d$ and squared loss; assume
$\mathbb E\lVert S\rVert^2<\infty$ **and** $\mathbb E\lVert Y\rVert^2<\infty$ (the
latter is automatic in the additive Gaussian arm from $\mathbb E\lVert X\rVert^2<\infty$),
so that $\mu(Y),\ell^\*(Y)\in L^2$ and the affine class $\{AY+b\}$ is well-posed.
The Bayes rule is the conditional mean $\mu(Y):=\mathbb E[S\mid Y]$ with risk
$\mathrm{MMSE}=\mathbb E\lVert S-\mu(Y)\rVert^2$. The ridge attack is the **best
affine** rule $\ell^\*(Y)=AY+b$ (population LMMSE), with risk $\mathrm{MSE}_{\mathrm{lin}}$.
By the orthogonality principle, $S-\mu(Y)\perp g(Y)$ for every square-integrable
$g$, so with $g=\mu-\ell^\*$ the Pythagorean decomposition gives
$$ \mathrm{MSE}_{\mathrm{lin}}-\mathrm{MMSE} \;=\; \mathbb E\big\lVert \mu(Y)-\ell^\*(Y)\big\rVert^2 \;\ge\;0, $$
**with equality iff $\mu(Y)=\ell^\*(Y)$ a.s., i.e. iff $\mathbb E[S\mid Y]$ is
a.s. affine** (Kay 1993, Ch.10–12; Van Trees §2.4). Joint Gaussianity of $(S,Y)$
is *sufficient* for affinity; absent it, $\mu$ departs from every affine map on a
positive-measure set and the gap is **strict**. Thus the Bayes attack strictly
beats ridge exactly when the conditional mean is non-affine — the generic case
for transformer activations.

**Caveat (the single-metric converse is invalid).** Strict MI loss
$I(S;\varphi(Y))<I(S;Y)$ alone does **not** imply a strict Bayes-risk gap for an
*arbitrary fixed* loss: $\varphi$ may discard information irrelevant to that
loss's optimal action (e.g. preserve the posterior $\arg\max$ but not its shape,
giving equal 0–1 risk). The valid general statement is Blackwell's: if
$\varphi(Y)$ is **not** sufficient for $S$ then there **exists** a prior/loss pair
with a strict gap (quantified over priors/losses), not "for this metric." For the
**token / 0–1** instance, write the exact condition tie-aware: equality
$r^\*(Y)=r^\*(\varphi(Y))$ holds iff there is a $\varphi(Y)$-measurable Bayes
action that is also a full-posterior MAP action a.s.; strictness holds iff no such
common Bayes action exists on a positive-measure set (under an a.s.-unique MAP
this reduces to "$\varphi$ changes $\arg\max_s p(s\mid Y)$ on a positive-measure
set"). We therefore assert strictness via (b)'s **non-affinity** (squared loss) or
the **common-MAP-action** condition (0–1 loss), never via MI-loss alone. ∎

### Part (c) — Monotonicity in MI on the Gaussian arm.

**(c-exact) Channel-input target $S=X$.** On $Y_{\mathrm{snr}}=\sqrt{\mathrm{snr}}X+N$,
$N\sim\mathcal N(0,I_d)$, the I-MMSE theorem (Guo–Shamai–Verdú 2005, Thm 1 scalar,
Thm 2 vector; valid for **arbitrary** input law with $\mathbb E\lVert X\rVert^2<\infty$,
not only Gaussian $X$) gives, in nats,
$$ \frac{d}{d\,\mathrm{snr}}\,I(X;Y_{\mathrm{snr}}) \;=\; \tfrac12\,\mathrm{mmse}(\mathrm{snr}),\qquad \mathrm{mmse}(\mathrm{snr})=\mathbb E\big\lVert X-\mathbb E[X\mid Y_{\mathrm{snr}}]\big\rVert^2 . $$
(The $\tfrac12$ coefficient is for $I$ measured in **nats** — local to this
identity; the base-2 convention of Prop 0 is independent. With $I$ in bits the
coefficient is $\tfrac{1}{2\ln 2}$. Only the *sign* — $\mathrm{mmse}\ge0$ —
matters for the monotonicity conclusion, so the base is immaterial here.)
Since $\mathrm{mmse}\ge0$, $I(X;Y_{\mathrm{snr}})$ is **nondecreasing** in
$\mathrm{snr}$. The I-MMSE identity alone does *not* give $\mathrm{mmse}$
monotonicity; we obtain it from the **same Gaussian degradation order** used in
(c-token), specialised to $S=X$ and squared loss: for
$\mathrm{snr}_1<\mathrm{snr}_2$, $X\!-\!Y_{\mathrm{snr}_2}\!-\!Y_{\mathrm{snr}_1}$
is Markov (construction below), so the optimal squared-error risk
$\mathrm{mmse}(\mathrm{snr}_1)\ge\mathrm{mmse}(\mathrm{snr}_2)$ (Bayes risk is
monotone under degradation; Blackwell). Hence $\mathrm{mmse}(\mathrm{snr})$ is
**nonincreasing** in $\mathrm{snr}$, and the Bayes-optimal attack's accuracy
$-\mathrm{mmse}$ and the mutual information move together monotonically — the
optimal attack's recovery is a **monotone function of MI by construction**.

**(c-token) Token target $S\neq X$, $S$ a function/label of $X$.** The exact
$\tfrac12$-MMSE identity is specific to estimating the channel **input**; it does
not transfer verbatim to $S$. Monotonicity still holds, via the **degradation
order**: for $\mathrm{snr}_1<\mathrm{snr}_2$, a scalar/vector Gaussian observation
satisfies
$$ Y_{\mathrm{snr}_1}\;\overset{d}{=}\;\sqrt{\tfrac{\mathrm{snr}_1}{\mathrm{snr}_2}}\;Y_{\mathrm{snr}_2}+N',\quad N'\sim\mathcal N\!\big(0,(1-\tfrac{\mathrm{snr}_1}{\mathrm{snr}_2})I_d\big)\ \perp\ Y_{\mathrm{snr}_2}, $$
so $S\!-\!Y_{\mathrm{snr}_2}\!-\!Y_{\mathrm{snr}_1}$ is Markov. By DPI
(Cover–Thomas Thm 2.8.1) $I(S;Y_{\mathrm{snr}})$ is nondecreasing in
$\mathrm{snr}$, and by Blackwell's monotonicity of Bayes risk under degradation
the optimal token-recovery risk $r^\*(Y_{\mathrm{snr}})$ is nonincreasing in
$\mathrm{snr}$. Thus optimal recovery and MI are **comonotone** along the path
(without the closed-form derivative). ∎

**(c-Laplace / Shredder arm).** Replace the SNR path by any additive channel
whose noise level induces a degradation order $S\!-\!Y_{\text{less noisy}}\!-\!Y_{\text{more noisy}}$.
For i.i.d. zero-mean Laplace noise of scale $b$ (per coordinate, characteristic
function $\phi_b(t)=\tfrac{1}{1+b^2t^2}$), a larger scale is a degradation of a
smaller one: for $b_2>b_1$,
$$ \frac{\phi_{b_2}(t)}{\phi_{b_1}(t)}=\frac{1+b_1^2t^2}{1+b_2^2t^2} $$
is itself a valid characteristic function, so $Y_{b_2}\overset{d}{=}Y_{b_1}+W$
with $W\perp Y_{b_1}$ — i.e. $Y_{b_2}$ is a garbling of $Y_{b_1}$ (note: the
increment $W$ is **not** itself Laplace; only its existence as an independent
additive term is needed). Hence $S\!-\!Y_{b_1}\!-\!Y_{b_2}$ Markov, and the same
DPI + Blackwell argument gives comonotonicity of $I(S;Y)$ and optimal recovery.
**The $\tfrac12$-MMSE identity is *not* invoked here** (it is Gaussian-specific). ∎

### Remark (why a *fixed weak* attack need not track MI — a non-implication).
For a fixed suboptimal rule $\delta_w$, write its risk along the path as
$R_t(\delta_w)=r^\*(Y_t)+\Delta_t$ with excess risk
$\Delta_t=R_t(\delta_w)-r^\*(Y_t)\ge0$. Parts (a)–(c) control $r^\*(Y_t)$ (it is
comonotone with $I(S;Y_t)$), but $\Delta_t$ depends on how $\delta_w$'s **fixed**
decision boundary interacts with the $t$-varying posterior and is **not**
controlled by $I(S;Y_t)$. There is therefore no theorem forcing $R_t(\delta_w)$
monotone in MI, and it generically is not; the observed L20×input-DP sign-flip
(`refine-logs/EXPERIMENT_RESULTS.md`, B3) is an empirical instance of a
non-monotone $\Delta_t$ for the ridge attack. This is a *non-implication*, stated
as motivation, not a proven positive claim.

## What T1 licenses for the experiment plan
- **C1 (uplift):** Part (a) guarantees the noise-aware Bayes attack weakly
  dominates ridge at **every** noise level; part (b) makes it **strict** wherever
  the conditional mean is non-affine (generic) ⟹ a measurable positive uplift.
- **C2 (re-correlation):** Part (c) makes the *optimal* attack's recovery
  comonotone with $I(S;Y)$ along the defence path ⟹ as the attack approaches
  Bayes-optimality, $\mathrm{Spearman}(\text{recovery},\,\text{MI-probe})\to$ its
  monotone ceiling, while ridge's stays governed by the uncontrolled $\Delta_t$.

## Corrections / scope made precise
- (b) strictness asserted only via **non-affinity (squared loss)** or
  **MAP-altering (0–1 loss)**, never MI-loss alone.
- (c) exact $\tfrac12$-MMSE identity asserted only for **$S=X$**; token target and
  Laplace arm get **degradation-order comonotonicity** instead.

## Open risks
- **R-tail.** (A2) finite second moment: if embeddings are genuinely
  infinite-variance, the squared-loss MMSE objects are undefined — switch to
  $L^1$/median (changes (b)'s decomposition; (a),(c-token) survive as they are
  loss-general / DPI-based).
- **R-path.** (c-token) gives comonotonicity along a *degradation-ordered* path;
  a real ε-sweep is degradation-ordered for input-DP at fixed clip, but if the
  clip norm $C$ co-varies with ε the ordering can break — hold $C$ fixed across ε
  (the runner already calibrates $C$ once).
- **R-empirical.** T1 bounds the *optimal* attack; the experiment must show the
  *implemented* denoise-then-invert attack is close enough to Bayes-optimal that
  the uplift and re-correlation actually appear (B2/B3).

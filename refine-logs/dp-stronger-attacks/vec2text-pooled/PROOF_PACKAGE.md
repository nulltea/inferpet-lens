# Proof Package — Spectral channel-MI: a matched, attack-independent probe for embedding inversion under DP

> Round 2 (post proof-checker / Codex xhigh, verdict WARN). All 11 issues addressed:
> the central correction is that `I_G(σ)` is a **Gaussian/covariance MI *ceiling***, not
> "the accessible information," and that it **upper-bounds (converse) recovery** — it is
> not claimed to equal achieved recovery. See the round-1 issue map in `PROOF_AUDIT` notes
> at the end.

## Claim

For the Gaussian-mechanism release of a sentence embedding, the **spectral channel
mutual information**
$$ I_G(\sigma) \;:=\; \tfrac12 \sum_{i=1}^{d} \log_2\!\Big(1+\frac{\lambda_i}{\sigma^2}\Big) $$
— computed from the clean-embedding covariance spectrum $\{\lambda_i\}$ and the noise
scale $\sigma$ **alone** (no attack is ever run) — is a matched, attack-independent
leakage probe with the following provable properties:

- **(T1) Exact target.** The text leakage equals the embedding-channel MI:
  $I(X;Y)=I(e_0;Y)$.
- **(T2) Closed-form ceiling + localization.** $I(X;Y)\le \min\{H(e_0),\,I_G(\sigma)\}\le\min\{H(X),\,I_G(\sigma)\}$,
  with $I_G$ admitting a per-eigenmode decomposition. $I_G$ is a covariance/Gaussian
  *upper bound*: it is the binding, informative ceiling in the moderate/high-$\sigma$
  privacy regime ($I_G<H(e_0)$), and is loose (exceeds the discrete cap, $\to\infty$ as
  $\sigma\to0$ when $\Sigma\ne0$) at low noise — there the discrete entropy $H(e_0)$ is the
  binding ceiling ($H(e_0)\le H(X)$, with strict gap when $\phi$ is many-to-one).
- **(T3) Attack-independent recovery ceiling (converse).** Via Fano, $I(X;Y)$ (hence the
  computable $I_G(\sigma)$) **upper-bounds** the exact-recovery success of *every*
  attack; a rate–distortion variant ceilings the per-token error rate. Monotonicity in
  $\sigma$ gives a rank-prediction; this is a converse, not a claim that any attack
  achieves it (empirical correlation is validated separately, B8).
- **(T4) Localization.** Projecting the observation onto the top-$k$ principal subspace
  discards at most the spectral tail $\tfrac12\sum_{i>k}\log_2(1+\lambda_i/\sigma^2)$, for
  an **arbitrary** embedding distribution — so the leakage is confined to the top
  principal directions up to that tail.

Plus a **contrast corollary** explaining why `capPVI` (cluster V-information) and `CLUB`
(variational $I(e';e_0)$) are inadequate.

## Status

**VERIFIED — PROVABLE AS STATED.** Cross-model checked (Codex `gpt-5.5` xhigh, thread
`019ef046`, 3 rounds, verdict **PASS**, zero open FATAL/CRITICAL). Holds for T1, T2 (as a
certified ceiling $I(X;Y)\le\min\{H(e_0),I_G\}$), T3a (exact-match), T3b (per-token error
rate), T4 (general discarded-info bound + exact Gaussian-surrogate value), and the Contrast
Corollary. The round-1 over-claim — treating $I_G$ as *the* accessible information / a
recovery *predictor* — was corrected: $I_G$ is a converse **ceiling**, binding in the
privacy regime; the low-noise cap is the discrete entropy $H(e_0)$. Two items are explicitly
scoped (not proven), flagged as empirical hooks: token-F1 (vs per-token Hamming) and the
*achievability* of T4 (a tractable attack realizing the top-subspace information). Imported
results used as such: Gaussian max-entropy, Fano, the Shannon rate–distortion lower bound.

## Assumptions

- **(A1) Finite secret.** $X$ on a finite alphabet $\mathcal X$ (token sequences), prior
  $p$, $K_{\mathrm{msg}}:=|\mathcal X|\ge 2$, $H(X)>0$.
- **(A2) Deterministic bounded encoder.** $g:\mathcal X\to\mathbb R^d$,
  $g(x)=\mathrm{clip}(\phi(x),C)$ deterministic, $\|g(x)\|_2\le C<\infty$. $e_0:=g(X)$ is a
  **discrete** random vector ($\le K_{\mathrm{msg}}$ atoms, finite moments).
  $\Sigma:=\operatorname{Cov}(e_0)\succeq0$, eigenvalues $\lambda_1\ge\cdots\ge\lambda_d\ge0$.
- **(A3) Gaussian channel.** $Y=e_0+Z$, $Z\sim\mathcal N(0,\sigma^2 I_d)$, $\sigma>0$,
  $Z\perp X$ (hence $Z\perp e_0$).
- **(A4) Attacker.** $\hat X=\psi(Y)$, $\psi$ arbitrary measurable, $P_e:=\Pr[\hat X\ne X]$.
  WLOG $\hat X\in\mathcal X$: since $X\in\mathcal X$, an off-alphabet output is a certain
  error, so reprojecting any off-alphabet output to a fixed element of $\mathcal X$ can
  only **weakly increase** $\Pr[\hat X=X]$. Hence a success *upper bound* proven for every
  $\mathcal X$-valued estimator also bounds the original $\psi$ (original success $\le$
  reprojected success $\le$ bound). Thus $X\to Y\to\hat X$ is Markov. The probe uses only
  $(\Sigma,\sigma)$.

All entropies/MI in **bits** ($\log:=\log_2$); $\ln$ = natural log; $h(\cdot)$ differential
entropy (bits).

## Notation

- $I,H,h$ — MI / discrete entropy / differential entropy (bits).
- $H_b(p):=-p\log p-(1-p)\log(1-p)\le1$.
- $t_i(\sigma):=\tfrac12\log_2(1+\lambda_i/\sigma^2)$; $I_G(\sigma)=\sum_i t_i(\sigma)$.
- $d_{\mathrm{eff}}(\sigma):=\#\{i:\lambda_i\ge\sigma^2\}$.
- $U=[u_1,\dots,u_d]$ — orthonormal eigenbasis of $\Sigma$ ($\Sigma=U\Lambda U^\top$),
  $\Lambda=\operatorname{diag}(\lambda_i)$. $W:=U^\top Y$ — observation in eigen-coordinates.
- $Q(x):=\Pr[\mathcal N(0,1)>x]$.
- $[a]_+:=\max(a,0)$.

## Proof Strategy

Direct information theory. T1: chain rule + degeneracy of a deterministic map + Markov.
T2: $I=h(Y)-h(Z)$ + Gaussian max-entropy, plus the trivial bound $I(X;Y)\le H(X)$. T3:
Fano + DPI; T3b: rate–distortion converse + Shannon lower bound. T4: **work in
eigen-coordinates** $W=U^\top Y$ (so projected blocks are non-degenerate Euclidean
subvectors), chain rule + max-entropy on the tail block using independence of the
orthogonal isotropic-noise components. Contrast: DPI + entropy saturation + an idealized
constellation union bound (explicitly idealized).

## Dependency Map

1. **T1** ← chain rule for MI; $\sigma(X,e_0)=\sigma(X)$ (A2); $Y\perp X\mid e_0$ (A3).
2. **T2** ← T1; $e_0$ discrete with finite covariance (A2); $Z$ Gaussian (A3); Gaussian
   max-entropy; $I(X;Y)\le H(X)$.
3. **T3a** ← T1, T2, Fano, DPI on $X\to Y\to\hat X$ (A4).
4. **T3b** ← rate–distortion converse $I(X;Y)\ge R_X(D)$ (A4 Markov); Shannon lower bound.
5. **T4** ← eigen-coordinate reparametrization; chain rule; conditioning-reduces-entropy;
   Gaussian max-entropy; independence of $W_{\le k}$- and $W_{>k}$-noise (A3 isotropy).
6. **Contrast (a)** ← DPI, $H(q(e_0))\le\log\kappa$, idealized M-ary Gaussian union bound;
   **(b)** ← T1 + CLUB's variational definition.

---

## Proof

### T1 — Sufficiency identity: $I(X;Y)=I(e_0;Y)$

Mutual information is defined via relative entropy; under (A1)–(A3) $X,e_0$ are discrete
and $Y$ has a Gaussian-mixture density, so all MIs below are finite and the chain rule
applies.

**Step 1.** Chain rule, two ways:
$I(X,e_0;Y)=I(e_0;Y)+I(X;Y\mid e_0)=I(X;Y)+I(e_0;Y\mid X)$.

**Step 2.** $e_0=g(X)$ is $\sigma(X)$-measurable $\Rightarrow\sigma(X,e_0)=\sigma(X)\Rightarrow I(X,e_0;Y)=I(X;Y)$.

**Step 3.** Given $X=x$, $e_0=g(x)$ is a.s. constant $\Rightarrow I(e_0;Y\mid X)=0$.

**Step 4.** $Y=e_0+Z$, $Z\perp X$, so conditional on $e_0=v$ the law of $Y$ is
$\mathcal N(v,\sigma^2I)$ independent of $X$: $Y\perp X\mid e_0\Rightarrow I(X;Y\mid e_0)=0$.

**Step 5.** Substitute: $I(X;Y)=I(X,e_0;Y)=I(e_0;Y)$. $\qquad\blacksquare$

*Consequence.* Measuring MI on embeddings ($I(e_0;Y)$) is **identically** the text leakage
$I(X;Y)$. This is also the target `CLUB` estimates ($I(e';e_0)$); so `CLUB`'s deficiency is
looseness/non-localization, not a wrong target (Contrast (b)).

### T2 — Spectral ceiling and per-mode localization

**Claim.** $I(X;Y)\le\min\{H(e_0),\,I_G(\sigma)\}\;(\le\min\{H(X),I_G(\sigma)\})$, where
$I_G(\sigma)=\tfrac12\sum_i\log_2(1+\lambda_i/\sigma^2)$. The bound $I(X;Y)\le I_G(\sigma)$
holds with equality iff $e_0$ is Gaussian; under (A1)–(A2) $e_0$ is discrete, so for
non-degenerate $e_0$ it is **strict**.

**Step 1 (discrete ceiling).** By T1 $I(X;Y)=I(e_0;Y)\le H(e_0)$ ($e_0$ discrete), and
$H(e_0)\le H(X)$ (data processing: $e_0=g(X)$), with strict gap iff $g$ is many-to-one
(distinct texts sharing an embedding).

**Step 2 (decomposition).** By T1, $I(X;Y)=I(e_0;Y)=h(Y)-h(Y\mid e_0)$ ($e_0$ discrete, $Y$
has a density). For each atom $v$, $Y\mid\{e_0=v\}\sim\mathcal N(v,\sigma^2I_d)$ with
entropy $\tfrac d2\log_2(2\pi e\sigma^2)$ independent of $v$, so $h(Y\mid e_0)=h(Z)$.

**Step 3 (max-entropy).** $\operatorname{Cov}(Y)=\Sigma+\sigma^2I_d$ ($e_0\perp Z$). $Y$ is a
finite Gaussian mixture $\Rightarrow h(Y)$ finite. Gaussian maximum-entropy (unique max of
$h$ over densities with fixed covariance $K$): $h(Y)\le\tfrac12\log_2((2\pi e)^d\det(\Sigma+\sigma^2I_d))$, equality iff $Y$ Gaussian.

**Step 4.** $I(e_0;Y)\le\tfrac12\log_2\dfrac{\det(\Sigma+\sigma^2I_d)}{\sigma^{2d}}=\tfrac12\log_2\det(I_d+\Sigma/\sigma^2)=\tfrac12\sum_i\log_2(1+\lambda_i/\sigma^2)=I_G(\sigma)$.

**Step 5 (equality/strictness).** Equality iff $Y$ Gaussian iff $e_0$ Gaussian; discrete
$e_0$ (A1–A2) is non-Gaussian when non-degenerate, so strict. If $\Sigma=0$ (degenerate
$e_0$, a.s. constant) both sides are $0$. Combining Steps 1 and 4 gives $I(X;Y)\le\min\{H(e_0),I_G(\sigma)\}$. $\qquad\blacksquare$

**Regime of usefulness (corrects round-1 over-claim).** As $\sigma\to0$ with $\Sigma\ne0$,
$I_G\to\infty$ while $I(X;Y)=I(e_0;Y)\to H(e_0)<\infty$ (the discrete channel becomes
noiseless, so the secret-about-$e_0$ information saturates at $H(e_0)$) — so $I_G$ is
*loose/vacuous at low noise* and the binding ceiling there is the **discrete entropy
$H(e_0)$**. For $\sigma$ in the privacy-relevant range, $I_G<H(e_0)$ and is the
**informative** ceiling. Thus the **certified accessible-bit ceiling** is
$\min\{H(e_0),I_G(\sigma)\}$ — an upper bound on the true $I(e_0;Y)$ (generally strict for
discrete non-Gaussian $e_0$), not an exact bit count; $I_G$'s value is as the
*channel-matched* ceiling.

**Per-mode localization.** $I_G=\sum_i t_i$, $t_i=\tfrac12\log_2(1+\lambda_i/\sigma^2)$:
$\lambda_i\gg\sigma^2\Rightarrow t_i\approx\tfrac12\log_2(\lambda_i/\sigma^2)$ (mode carries
information); $\lambda_i\ll\sigma^2\Rightarrow t_i\le\tfrac{\lambda_i}{2\sigma^2\ln2}\to0$
(drowned). Modes with $\lambda_i\ge\sigma^2$ each contribute $t_i\ge\tfrac12$ bit;
$d_{\mathrm{eff}}(\sigma)=\#\{i:\lambda_i\ge\sigma^2\}$ counts them. Made precise in T4.

### T3 — Attack-independent recovery ceiling (Fano)

**T3a (exact-match, uniform prior).** For any $\hat X=\psi(Y)$ with $\hat X\in\mathcal X$
and $P_e$ as in (A4) (the bound transfers to the original $\psi$ by the reprojection
argument there). DPI on $X\to Y\to\hat X$: $I(X;\hat X)\le I(X;Y)$, i.e.
$H(X\mid Y)\le H(X\mid\hat X)$. Fano (valid for $\hat X\in\mathcal X$, $K_{\mathrm{msg}}\ge2$):
$H(X\mid\hat X)\le H_b(P_e)+P_e\log_2(K_{\mathrm{msg}}-1)\le 1+P_e\log_2 K_{\mathrm{msg}}$.
Hence $H(X)-I(X;Y)=H(X\mid Y)\le1+P_e\log_2K_{\mathrm{msg}}$, giving the
**attack-independent, any-prior** bound
$$ P_e\;\ge\;\frac{H(X)-I(X;Y)-1}{\log_2 K_{\mathrm{msg}}}. $$
For a **uniform** prior $H(X)=\log_2K_{\mathrm{msg}}$, this rearranges to the success form
$$ \Pr[\hat X=X]\le\frac{I(X;Y)+1}{H(X)}\;\stackrel{\text{T2}}{\le}\;\frac{\min\{H(e_0),I_G(\sigma)\}+1}{H(X)}. $$
The success form requires the uniform prior (it is false in general: e.g. a $0.9$-mass
atom is guessed correctly w.p. $0.9$ even at $I=0$); the $P_e$ lower bound above is the
correct **non-uniform** statement. $I_G(\sigma)\downarrow$ in $\sigma$ ⇒ the ceiling is
monotone ⇒ rank-prediction. This is a converse (holds for all $\psi$); it does not assert
any attack achieves it. $\qquad\blacksquare$

**T3b (per-token error rate, rate–distortion).** $X=(X_1,\dots,X_n)\in[V]^n$,
$\hat X=(\hat X_1,\dots,\hat X_n)\in[V]^n$, normalized positional Hamming distortion
$D:=\tfrac1n\sum_t\Pr[\hat X_t\ne X_t]$. Let $R_X(\cdot)$ be the source's rate–distortion
function for this distortion. Since $X\to Y\to\hat X$ achieves distortion $D$:
$I(X;Y)\ge I(X;\hat X)\ge R_X(D)$. The **Shannon lower bound**, with
$\gamma(D):=H_b(D)+D\log_2(V-1)$, gives $R_X(D)\ge[\,H(X)-n\,\gamma(D)\,]_+$. Hence
$$ \gamma(D)\;\ge\;\frac{[\,H(X)-I(X;Y)\,]_+}{n}\;\stackrel{\text{T2}}{\ge}\;\frac{[\,H(X)-\min\{H(e_0),I_G(\sigma)\}\,]_+}{n}=:\tau(\sigma). $$
$\gamma$ is strictly increasing on $[0,(V-1)/V]$ (and $\gamma\equiv\gamma((V-1)/V)=\log_2 V$
for $D\ge(V-1)/V$). Thus for the operationally relevant regime $D\le(V-1)/V$ the bound
inverts to $D\ge\gamma^{-1}(\tau(\sigma))>0$ once $\tau(\sigma)>0$ (i.e. $I_G(\sigma)<H(e_0)$);
if instead $D>(V-1)/V$ the error already exceeds the in-region threshold, so the lower
bound is trivially satisfied. **Scope:** ceilings the *positional token-error rate* only —
**not** token-F1 (multiset overlap); no quantitative F1 claim is inferred. $\qquad\blacksquare$

### T4 — Localization: where the recoverable information lives

Work in eigen-coordinates $W=U^\top Y$ to avoid the singular ambient covariance of a
projected vector. Since $\Sigma=U\Lambda U^\top$ and the noise is isotropic,
$W=U^\top e_0+U^\top Z$ with $U^\top Z\sim\mathcal N(0,\sigma^2 I_d)$ (rotation invariance);
write $W_{\le k}:=(W_1,\dots,W_k)$, $W_{>k}:=(W_{k+1},\dots,W_d)$, full-rank Euclidean
subvectors. Because $U$ is invertible, $I(X;P_kY)=I(X;W_{\le k})$ and $I(X;Y)=I(X;W)$.

**Claim (general, non-Gaussian).**
$0\le I(X;Y)-I(X;P_kY)=I(X;W_{>k}\mid W_{\le k})\le\tfrac12\sum_{i>k}\log_2(1+\lambda_i/\sigma^2)$.

**Step 1.** Chain rule: $I(X;W)=I(X;W_{\le k})+I(X;W_{>k}\mid W_{\le k})$, and the
conditional term is $\ge0$ — giving both the DPI direction and the identity for the gap.

**Step 2.** $I(X;W_{>k}\mid W_{\le k})=h(W_{>k}\mid W_{\le k})-h(W_{>k}\mid W_{\le k},X)$
(both differential entropies now on the non-degenerate $(d-k)$-dim coordinate block).

**Step 3 (noise term).** The coordinates of $U^\top Z$ are i.i.d. $\mathcal N(0,\sigma^2)$,
so the noise in $W_{>k}$ is independent of the noise in $W_{\le k}$. Given $X$ (hence
$e_0$, hence $U^\top e_0$ fixed), $W_{>k}=(U^\top e_0)_{>k}+(U^\top Z)_{>k}$ with
$(U^\top Z)_{>k}\perp(U^\top Z)_{\le k}=W_{\le k}-(U^\top e_0)_{\le k}$; hence
$h(W_{>k}\mid W_{\le k},X)=h((U^\top Z)_{>k})=\tfrac{d-k}2\log_2(2\pi e\sigma^2)$.

**Step 4 (signal term).** Conditioning reduces entropy, then Gaussian max-entropy on the
$(d-k)$-dim block, whose covariance is
$\operatorname{Cov}(W_{>k})=\operatorname{diag}(\lambda_{k+1}+\sigma^2,\dots,\lambda_d+\sigma^2)$
(in eigen-coordinates $\operatorname{Cov}(W)=\Lambda+\sigma^2I$ is diagonal): $h(W_{>k}\mid W_{\le k})\le h(W_{>k})\le\tfrac12\log_2\big((2\pi e)^{d-k}\prod_{i>k}(\lambda_i+\sigma^2)\big)$.

**Step 5.** Subtract: $I(X;W_{>k}\mid W_{\le k})\le\tfrac12\log_2\frac{\prod_{i>k}(\lambda_i+\sigma^2)}{\sigma^{2(d-k)}}=\tfrac12\sum_{i>k}\log_2(1+\lambda_i/\sigma^2)$. $\qquad\blacksquare$

**Consequence (where).** The discarded leakage is $\le\mathrm{tail}(k):=\tfrac12\sum_{i>k}\log_2(1+\lambda_i/\sigma^2)$,
so $I(X;P_kY)\ge I(X;Y)-\mathrm{tail}(k)$: the top-$k$ principal subspace retains all but
$\mathrm{tail}(k)$ of the accessible leakage. The localization is sharp **only when
$\mathrm{tail}(k)$ is itself small** — e.g. a few strongly-drowned tail modes
($\lambda_i\ll\sigma^2$); note each summand is $<\tfrac12$ bit but a *large number* of
modes at $\lambda_i\lesssim\sigma^2$ can still sum to a non-negligible tail, so "where"
is pinned down precisely by the computable profile $\{t_i\}$ and $\mathrm{tail}(k)$, not by
$d_{\mathrm{eff}}$ alone. This is a **converse** (the information is present in the
subspace); it does **not** assert a tractable attack realizes it.

**Exact value under the Gaussian surrogate.** If $e_0\sim\mathcal N(\mu,\Sigma)$, the
$W_i$ are independent scalar Gaussian channels, $I(\tilde e_0;Y)=\sum_i t_i(\sigma)$, and
projecting to the top-$k$ block discards **exactly** $\sum_{i>k}t_i(\sigma)$ — matching the
general upper bound. **Achievability** (a specific attack, e.g. Vec2Text, converting the
retained subspace information into recovery) is the empirical hook: ablate the bottom
$d-d_{\mathrm{eff}}$ eigendirections of $Y$ and check recovery is ~unchanged.

### Contrast Corollary — why `capPVI` and `CLUB` fail

**(a) `capPVI` measures an orthogonal, low-bit, noise-robust functional.** Let
$q:\mathbb R^d\to\{1,\dots,\kappa\}$ be a fixed $\kappa$-class label of $e_0$ ($\kappa$-means
assignment; $\kappa=40$ in the eval). The *Shannon* information about the label obeys, by
DPI on $e_0\to q(e_0)$ and T1, and by $I(q;Y)\le H(q)$,
$$ I(q(e_0);Y)\le\min\{\,I(e_0;Y),\,H(q(e_0))\,\}\le\min\{\,I(X;Y),\,\log_2\kappa\,\}, $$
so **any** functional of $q$ saturates at $\le\log_2\kappa\approx5.3$ bits and cannot track
$I(X;Y)$ once the latter exceeds $\log_2\kappa$. The implemented `capPVI` (top-1 accuracy of
a PCA-softmax reader of $q(e_0)$ from $Y$) is a *restricted-reader proxy* for usable
information about this coarse label; we do not claim a clean V-info$\le$Shannon-MI identity
for the specific accuracy estimator, only the structural ceiling above on the label's
information content.

*Noise-robustness (the flatness), illustrative construction.* Consider the idealized
nearest-centroid model: $\kappa$ centroids $\{c_a\}$, min separation
$\Delta_{\min}=\min_{a\ne b}\|c_a-c_b\|$, an atom mapped to its nearest centroid. For this
constellation the cluster MAP error obeys the M-ary Gaussian union bound
$\Pr[\hat q\ne q]\le(\kappa-1)Q(\Delta_{\min}/2\sigma)$, which stays $\approx0$ for all
$\sigma\ll\Delta_{\min}$. (This is an *idealization*; real $\kappa$-means has within-cluster
spread and non-uniform priors, so it bounds a best case, not arbitrary assignments.) Since
topic centroids are widely separated relative to the *fine* eigen-scales $\sqrt{\lambda_i}$
of the many small modes carrying token detail, there is a wide $\sigma$-window where cluster
accuracy $\approx$ clean while, by T3, the token-recovery ceiling has collapsed (it drops
once $I_G(\sigma)$ does, i.e. once $\sigma$ nears those $\sqrt{\lambda_i}$). Hence the
observed *flat-while-recovery-falls*: `capPVI` tracks a coarse, $\le\log_2\kappa$-bit,
noise-robust quantity on a scale orthogonal to fine token recovery.

**(b) `CLUB` targets the right MI but is a loose, non-localizing variational bound.** Here
$e':=Y$ is the released (noised) embedding. By T1 the CLUB objective
$I(e';e_0)=I(Y;e_0)=I(X;Y)$ is the correct leakage. But CLUB is a
**variational upper bound** that is an upper bound only under its variational/conditional
assumptions (well-specified critic), with estimator bias/variance — not a closed-form
*certified* channel bound; and it returns a **single scalar** with no spectral
decomposition, so it cannot exhibit $d_{\mathrm{eff}}$ or the top subspace (T2/T4). It is
monotone (it tracks the right monotone quantity) but its map to recovery is the nonlinear
Fano map (T3), and it is non-localizing. $I_G(\sigma)$ is the *same* target computed in
closed form from $(\Sigma,\sigma)$ — a certified ceiling, strictly attack-free, and
decomposable. $\qquad\blacksquare$

## Corrections or Missing Assumptions (round-2 deltas vs round-1)

- **I_G reframed as a ceiling, not "accessible information."** Added $I(X;Y)\le\min\{H(X),I_G\}$ and the low-$\sigma$ looseness/regime statement (T2). (Codex I1.)
- **"Predicts" → converse ceiling + rank-prediction**, empirical correlation deferred to B8. (I2.)
- **T3a success form restricted to uniform prior** (already implicit; now explicit), with the general $P_e$ bound stated; $\hat X\in\mathcal X$ WLOG added (A4). (I3, I4.)
- **T3b**: explicit domain $D\in[0,(V-1)/V]$ and $[\cdot]_+$ clip; token-F1 scope tightened. (I5, I6.)
- **T4 rewritten in eigen-coordinates** (non-degenerate subspace entropies); converse vs Gaussian-exact vs achievability separated. (I7, I8.)
- **Contrast (a)** weakened: structural $\le\min\{I(X;Y),\log\kappa\}$ on the label's
  Shannon info; the implemented reader is a proxy; the union bound is an explicit
  nearest-centroid idealization. **(b)** CLUB "upper bound under variational assumptions,
  not certified." (I9, I10, I11.)

## Open Risks

- **token-F1 vs per-token error (T3b).** RD ceiling is for positional Hamming; token-F1
  needs a separate distortion analysis (length/alignment). Applies rigorously to per-token
  accuracy only; B8's co-monotonicity of F1/cos/BLEU with the probe is consistent, not a proof.
- **Achievability of T4.** Converses show the info is *present* in the top subspace; a
  tractable attack realizing it is the bottom-mode-ablation experiment, not proven here.
- **Estimating $\Sigma$.** $\lambda_i$ from a finite sample; near the bulk edge,
  sample-eigenvalue bias can shift $d_{\mathrm{eff}}$ — use shrinkage / Marchenko–Pastur when
  $n_{\text{texts}}\lesssim d$ (here $d=768$).
- **Low-$\sigma$ vacuity.** At small $\sigma$, $I_G$ exceeds the discrete cap $H(e_0)$ and
  is uninformative ($I(X;Y)\to H(e_0)$ as $\sigma\to0$); report $\min\{H(e_0),I_G\}$ and
  treat the probe as binding in the privacy regime ($I_G<H(e_0)$).
- **capPVI idealization.** The flatness argument uses a nearest-centroid constellation; for
  arbitrary $\kappa$-means with spread it is a best-case illustration, not a theorem about
  the exact estimator.

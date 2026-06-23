---
type: claim
node_id: claim:spectral-channel-mi-embedding-inversion
name: "Spectral channel-MI: matched attack-independent converse leakage probe for embedding inversion under DP"
description: ""
node_type: claim
status: verified
provenance: "refine-logs/dp-stronger-attacks/vec2text-pooled/PROOF_PACKAGE.md; Codex gpt-5.5 xhigh thread 019ef046, 3 rounds, PASS"
tags: ["vec2text", "embedding-inversion", "spectral-mi", "gaussian-channel", "water-filling", "fano", "rate-distortion", "matched-probe", "geometry-only", "dp", "capPVI", "club", "pooled-embedding"]
date: 2026-06-22
added: 2026-06-22T17:17:36Z
---

# Spectral channel-MI: matched attack-independent converse leakage probe for embedding inversion under DP

**status:** `verified` — cross-model checked (Codex `gpt-5.5` xhigh, thread `019ef046`,
3 rounds, verdict **PASS**, zero open FATAL/CRITICAL). Imported textbook steps (Gaussian
max-entropy, Fano, Shannon rate–distortion lower bound) flagged as such; the matched-probe
framing, sufficiency identity (T1), localization bound (T4), and contrast results are
established here. Full verified proof: `refine-logs/dp-stronger-attacks/vec2text-pooled/PROOF_PACKAGE.md`.
Continuous-channel analog of [[bnn-error-bounds-bhattacharyya-fano]] (which needs an
enumerable codebook; text has $V^n$ messages, so the spectral bound replaces it).

## Statement

Secret text $X$ (finite alphabet, $H(X)>0$); deterministic clipped embedding
$e_0=\mathrm{clip}(\phi(X),C)\in\mathbb R^d$ (so $e_0$ is **discrete**, $\Sigma=\operatorname{Cov}(e_0)$,
eigenvalues $\lambda_1\ge\cdots\ge\lambda_d\ge0$); Gaussian-mechanism release $Y=e_0+\mathcal N(0,\sigma^2I_d)$.
Define $t_i:=\tfrac12\log_2(1+\lambda_i/\sigma^2)$, $I_G(\sigma):=\sum_i t_i$,
$d_{\mathrm{eff}}:=\#\{i:\lambda_i\ge\sigma^2\}$. Computed from $(\Sigma,\sigma)$ **alone** (no attack run):

- **T1 (sufficiency).** $I(X;Y)=I(e_0;Y)$ — measuring MI on embeddings is exactly the text leakage.
- **T2 (spectral ceiling + localization).** $I(X;Y)\le\min\{H(e_0),\,I_G(\sigma)\}$ (strict $\le I_G$
  unless $e_0$ Gaussian). Low-noise cap is the discrete $H(e_0)$ ($I(X;Y)\to H(e_0)$ as $\sigma\to0$);
  $I_G$ is the informative ceiling in the privacy regime $I_G<H(e_0)$. Per-mode $t_i$ carries info iff
  $\lambda_i\gtrsim\sigma^2$ (water-filling).
- **T3 (attack-independent recovery ceiling).** Fano: uniform-prior exact-match success
  $\Pr[\hat X=X]\le(\min\{H(e_0),I_G\}+1)/H(X)$ for **every** attack $\hat X=\psi(Y)$; rate–distortion
  variant ceilings the per-token error rate. Converse, monotone $\downarrow$ in $\sigma$.
- **T4 (localization).** For any top-$k$ principal projector $P_k$ and **arbitrary** $e_0$:
  $I(X;Y)-I(X;P_kY)\le\tfrac12\sum_{i>k}\log_2(1+\lambda_i/\sigma^2)$ — the recoverable leakage is
  confined to the top eigendirections up to the spectral tail (exact split for Gaussian $e_0$).
- **Contrast.** `capPVI` (V-info of a $\kappa$-cluster label) saturates at $\le\log_2\kappa$ bits and
  stays flat while $\sigma\ll$ centroid gaps — orthogonal to fine token recovery. `CLUB` targets the
  right MI $I(e';e_0)=I(X;Y)$ but is a loose, non-localizing variational upper bound.

## Novelty & positioning (Codex gpt-5.5 xhigh, thread 019ef5f6 — 6.5/10, narrow framing)

Novel as a **framing/finding from known pieces**, not a new IT theorem: the closed-form Gaussian
water-filling spectrum $I_G$ is *newly operationalized* as a **geometry-only, attack-independent**
converse + cheap rank-predictor for DP text-embedding inversion. Closest prior: **Eguard** (arXiv:2411.05034,
AAAI — text-MI as a *learned defense* objective, not closed-form/attack-free); **Zhuang et al. 2024**
(arXiv:2402.12784 — Vec2Text recoverability under Gaussian-noise sweeps, but *runs the attack*, no
spectral probe); **Constructing Privacy Channels** (arXiv:1910.09235 — DP↔channel-capacity *generically*,
no text/inversion); attack priors Morris Vec2Text (arXiv:2310.06816), GEIA (arXiv:2305.03010). Positioned
as the cheap attack-free converse **complementing** learned MI estimators (CLUB/Eguard), not replacing
attacks. **Monotonicity-confound caveat (reviewer-critical):** a $+1.0$ Spearman across a *monotone*
$\sigma$-sweep is partly "everything decreases with $\sigma$"; the discriminating test is anti-confound
controls — $I_G$ vs raw $\sigma$, trace-SNR, effective rank, total variance — and a *non-monotone*
mechanism (Laplace / partial-dim DP). Queued as the matched-probe-program follow-up (tracker R005), not
run in this consolidation.

## Honest scope

$I_G$ is a **converse ceiling**, not the exact leakage nor achieved recovery — empirical probe↔recovery
correlation (experiment B8: CLUB-bits ↔ Vec2Text token-F1/cos/BLEU Spearman $=+1.0$) is validation, not
implied by the converse. Token-F1 is **not** bounded (only positional token-error). T4 achievability (a
tractable attack realizing the top-subspace info) is an empirical hook (bottom-mode ablation). Imported:
Gaussian max-entropy, Fano, Shannon RD lower bound. Finite-sample $\lambda_i$ bias can shift
$d_{\mathrm{eff}}$ when $n_{\text{texts}}\lesssim d$ (use shrinkage).

## Evidence chain

- Proof obligations T1–T4 + contrast: all discharged; see `PROOF_PACKAGE.md` (theorem-by-theorem).
- Jury: Codex `gpt-5.5` xhigh, thread `019ef046`, rounds 1→3 (WARN→WARN→**PASS**); round-1 over-claim
  ($I_G$ as "accessible info"/predictor) corrected to converse ceiling; low-noise cap fixed to $H(e_0)$.
- **Implementation**: `src/talens/measures/spectral_channel_mi.py` (geometry-only; Codex-reviewed, no
  critical; hardened) + `tests/test_spectral_channel_mi.py` (10/10 model-free; Gaussian-exact ½logdet,
  monotone, d_eff, ceilings).
- **Empirical (B9, `results/spectral_mi_probe_eval.json`)**: on the pooled-GTR DP sweep (N=96), **C1
  VALIDATED** — Spearman(I_G, Vec2Text recovery)=+1.00 for token-F1/cos/positional-token-acc (=CLUB)
  ≫ capPVI +0.62; for the harder **exact-match** readout (floored at 0 for every ε<∞) the rank
  correlation is +0.71 (=CLUB), still above capPVI +0.54. All at ~28× lower cost
  (I_G 60 ms eigh vs CLUB 1.7 s); **C2** 0 ceiling violations (RD per-token floor 0→0.81 respected).
  **Caveat:** N<d ⇒ rank-deficient Σ, so d_eff/tail (localization, T4/C3) are undersampled — needs n≫d
  (estimate Σ from a large embedding corpus); M3 eigen-ablation re-scoped accordingly (not yet run).
- companion empirical: [[unified-dp-sweep]] (dp-stronger-attacks B8).

## Connections

- analog-of → [[bnn-error-bounds-bhattacharyya-fano]] (discrete-codebook two-sided bound)
- companion → [[mi-monotone-gaussian]]
- supported-by → [[unified-dp-sweep]] (DP sweep; CLUB↔Vec2Text recovery ρ=+1.0)
- supported-by → [[spectral-mi-probe-eval]] (B9: I_G↔Vec2Text recovery Spearman +1.0, 0 ceiling violations)
- contrasted-by → [[vec2text-feedback-null]] (the matched attack on this surface; feedback-null negative result)
- supersedes-as-probe → `capPVI` (cluster V-info), `CLUB` (variational $I(e';e_0)$) for this channel

---

## Full verified proof (inline)

> Folded in full from `refine-logs/dp-stronger-attacks/vec2text-pooled/PROOF_PACKAGE.md`
> (Codex `gpt-5.5` xhigh, thread `019ef046`, 3 rounds → **PASS**). All MI/entropy in bits
> ($\log:=\log_2$); $h(\cdot)$ differential entropy; $\ln$ natural log.

### Assumptions

- **(A1) Finite secret.** $X$ on a finite alphabet $\mathcal X$ (token sequences), prior $p$,
  $K_{\mathrm{msg}}:=|\mathcal X|\ge 2$, $H(X)>0$.
- **(A2) Deterministic bounded encoder.** $g:\mathcal X\to\mathbb R^d$,
  $g(x)=\mathrm{clip}(\phi(x),C)$ deterministic, $\|g(x)\|_2\le C<\infty$. $e_0:=g(X)$ is a
  **discrete** random vector ($\le K_{\mathrm{msg}}$ atoms). $\Sigma:=\operatorname{Cov}(e_0)\succeq0$,
  eigenvalues $\lambda_1\ge\cdots\ge\lambda_d\ge0$.
- **(A3) Gaussian channel.** $Y=e_0+Z$, $Z\sim\mathcal N(0,\sigma^2 I_d)$, $\sigma>0$, $Z\perp X$.
- **(A4) Attacker.** $\hat X=\psi(Y)$, $\psi$ arbitrary measurable, $P_e:=\Pr[\hat X\ne X]$. WLOG
  $\hat X\in\mathcal X$ (reprojecting an off-alphabet output to a fixed element of $\mathcal X$ can only
  weakly increase $\Pr[\hat X=X]$, so a success upper bound transfers). $X\to Y\to\hat X$ is Markov.
  The probe uses only $(\Sigma,\sigma)$.

Notation: $t_i(\sigma):=\tfrac12\log_2(1+\lambda_i/\sigma^2)$, $I_G(\sigma)=\sum_i t_i$,
$d_{\mathrm{eff}}(\sigma):=\#\{i:\lambda_i\ge\sigma^2\}$; $\Sigma=U\Lambda U^\top$, $W:=U^\top Y$;
$H_b(p):=-p\log p-(1-p)\log(1-p)\le1$; $Q(x):=\Pr[\mathcal N(0,1)>x]$; $[a]_+:=\max(a,0)$.

### T1 — Sufficiency identity: $I(X;Y)=I(e_0;Y)$

Under (A1)–(A3), $X,e_0$ are discrete and $Y$ has a Gaussian-mixture density, so all MIs are finite and
the chain rule applies. Chain rule two ways:
$I(X,e_0;Y)=I(e_0;Y)+I(X;Y\mid e_0)=I(X;Y)+I(e_0;Y\mid X)$. Since $e_0=g(X)$ is $\sigma(X)$-measurable,
$\sigma(X,e_0)=\sigma(X)\Rightarrow I(X,e_0;Y)=I(X;Y)$. Given $X=x$, $e_0=g(x)$ is a.s. constant
$\Rightarrow I(e_0;Y\mid X)=0$. And $Y=e_0+Z$, $Z\perp X$, so $Y\perp X\mid e_0\Rightarrow I(X;Y\mid e_0)=0$.
Substituting: $I(X;Y)=I(X,e_0;Y)=I(e_0;Y)$. $\blacksquare$ — measuring MI on embeddings is identically
the text leakage; it is also the target CLUB estimates ($I(e';e_0)$), so CLUB's deficiency is
looseness/non-localization, not a wrong target.

### T2 — Spectral ceiling and per-mode localization

**Claim.** $I(X;Y)\le\min\{H(e_0),\,I_G(\sigma)\}\le\min\{H(X),I_G(\sigma)\}$, with equality
$I(X;Y)=I_G$ iff $e_0$ Gaussian (so strict for discrete non-degenerate $e_0$).

*Discrete ceiling.* By T1, $I(X;Y)=I(e_0;Y)\le H(e_0)$ ($e_0$ discrete), and $H(e_0)\le H(X)$ (DPI on
$e_0=g(X)$), strict gap iff $g$ many-to-one. *Decomposition.* $I(e_0;Y)=h(Y)-h(Y\mid e_0)$; each atom
$v$ gives $Y\mid\{e_0=v\}\sim\mathcal N(v,\sigma^2I_d)$ with entropy $\tfrac d2\log_2(2\pi e\sigma^2)$
independent of $v$, so $h(Y\mid e_0)=h(Z)$. *Max-entropy.* $\operatorname{Cov}(Y)=\Sigma+\sigma^2I_d$;
Gaussian max-entropy gives $h(Y)\le\tfrac12\log_2((2\pi e)^d\det(\Sigma+\sigma^2I_d))$, so
$I(e_0;Y)\le\tfrac12\log_2\det(I_d+\Sigma/\sigma^2)=\tfrac12\sum_i\log_2(1+\lambda_i/\sigma^2)=I_G(\sigma)$.
Equality iff $Y$ (hence $e_0$) Gaussian. $\blacksquare$

*Regime of usefulness (corrects round-1 over-claim).* As $\sigma\to0$ with $\Sigma\ne0$,
$I_G\to\infty$ while $I(X;Y)\to H(e_0)<\infty$ — $I_G$ is loose at low noise; the binding ceiling there
is the **discrete entropy $H(e_0)$**. For privacy-relevant $\sigma$, $I_G<H(e_0)$ is the informative
ceiling. Certified accessible-bit ceiling $=\min\{H(e_0),I_G(\sigma)\}$. *Per-mode:* $\lambda_i\gg\sigma^2
\Rightarrow t_i\approx\tfrac12\log_2(\lambda_i/\sigma^2)$ (carries info); $\lambda_i\ll\sigma^2\Rightarrow
t_i\le\tfrac{\lambda_i}{2\sigma^2\ln2}\to0$ (drowned); modes with $\lambda_i\ge\sigma^2$ each contribute
$\ge\tfrac12$ bit, counted by $d_{\mathrm{eff}}$.

### T3 — Attack-independent recovery ceiling (Fano)

**T3a (exact-match).** DPI on $X\to Y\to\hat X$: $I(X;\hat X)\le I(X;Y)$, i.e.
$H(X\mid Y)\le H(X\mid\hat X)$. Fano ($\hat X\in\mathcal X$, $K_{\mathrm{msg}}\ge2$):
$H(X\mid\hat X)\le H_b(P_e)+P_e\log_2(K_{\mathrm{msg}}-1)\le1+P_e\log_2K_{\mathrm{msg}}$. Hence the
attack-independent, any-prior bound
$$P_e\ge\frac{H(X)-I(X;Y)-1}{\log_2 K_{\mathrm{msg}}}.$$
For a **uniform** prior $H(X)=\log_2K_{\mathrm{msg}}$ this rearranges to the success form
$$\Pr[\hat X=X]\le\frac{I(X;Y)+1}{H(X)}\stackrel{\text{T2}}\le\frac{\min\{H(e_0),I_G(\sigma)\}+1}{H(X)}.$$
The success form needs the uniform prior (false in general); the $P_e$ bound is the correct non-uniform
statement. $I_G\downarrow$ in $\sigma$ ⇒ monotone rank-prediction. Converse — no claim any attack
achieves it. $\blacksquare$

**T3b (per-token error rate, rate–distortion).** $X,\hat X\in[V]^n$, normalized positional Hamming
$D:=\tfrac1n\sum_t\Pr[\hat X_t\ne X_t]$, rate–distortion $R_X$. Since $X\to Y\to\hat X$ achieves $D$:
$I(X;Y)\ge I(X;\hat X)\ge R_X(D)$. Shannon lower bound with $\gamma(D):=H_b(D)+D\log_2(V-1)$ gives
$R_X(D)\ge[H(X)-n\gamma(D)]_+$, so
$$\gamma(D)\ge\frac{[H(X)-I(X;Y)]_+}{n}\stackrel{\text{T2}}\ge\frac{[H(X)-\min\{H(e_0),I_G(\sigma)\}]_+}{n}=:\tau(\sigma).$$
$\gamma$ strictly increasing on $[0,(V-1)/V]$, so for $D\le(V-1)/V$ it inverts to
$D\ge\gamma^{-1}(\tau(\sigma))>0$ once $\tau(\sigma)>0$. **Scope:** ceilings positional token-error rate
only — **not** token-F1. $\blacksquare$

### T4 — Localization: where the recoverable information lives

Work in eigen-coordinates $W=U^\top Y=U^\top e_0+U^\top Z$, $U^\top Z\sim\mathcal N(0,\sigma^2I_d)$
(rotation invariance); $W_{\le k},W_{>k}$ full-rank subvectors. $I(X;P_kY)=I(X;W_{\le k})$, $I(X;Y)=I(X;W)$.

**Claim (general, non-Gaussian).** $0\le I(X;Y)-I(X;P_kY)=I(X;W_{>k}\mid W_{\le k})\le\tfrac12\sum_{i>k}\log_2(1+\lambda_i/\sigma^2)$.

Chain rule: $I(X;W)=I(X;W_{\le k})+I(X;W_{>k}\mid W_{\le k})$, conditional term $\ge0$. Then
$I(X;W_{>k}\mid W_{\le k})=h(W_{>k}\mid W_{\le k})-h(W_{>k}\mid W_{\le k},X)$. *Noise term:* coordinates
of $U^\top Z$ i.i.d. $\mathcal N(0,\sigma^2)$, so given $X$, $h(W_{>k}\mid W_{\le k},X)=h((U^\top Z)_{>k})=\tfrac{d-k}2\log_2(2\pi e\sigma^2)$.
*Signal term:* conditioning-reduces-entropy + Gaussian max-entropy on the block with
$\operatorname{Cov}(W_{>k})=\operatorname{diag}(\lambda_{k+1}+\sigma^2,\dots,\lambda_d+\sigma^2)$:
$h(W_{>k}\mid W_{\le k})\le\tfrac12\log_2((2\pi e)^{d-k}\prod_{i>k}(\lambda_i+\sigma^2))$. Subtract:
$I(X;W_{>k}\mid W_{\le k})\le\tfrac12\sum_{i>k}\log_2(1+\lambda_i/\sigma^2)$. $\blacksquare$

So $I(X;P_kY)\ge I(X;Y)-\mathrm{tail}(k)$, $\mathrm{tail}(k):=\tfrac12\sum_{i>k}\log_2(1+\lambda_i/\sigma^2)$:
the top-$k$ principal subspace retains all but $\mathrm{tail}(k)$ of accessible leakage (sharp only when
the tail is itself small). Under a Gaussian surrogate $e_0\sim\mathcal N(\mu,\Sigma)$ projecting to the
top-$k$ block discards **exactly** $\sum_{i>k}t_i(\sigma)$. Converse — *achievability* (an attack
realizing the retained subspace info) is the empirical bottom-mode-ablation hook.

### Contrast Corollary — why `capPVI` and `CLUB` fail

**(a) capPVI.** For a fixed $\kappa$-class label $q(e_0)$: by DPI ($e_0\to q$) + T1 + $I(q;Y)\le H(q)$,
$I(q(e_0);Y)\le\min\{I(e_0;Y),H(q(e_0))\}\le\min\{I(X;Y),\log_2\kappa\}$ — saturates at $\le\log_2\kappa
\approx5.3$ bits, cannot track $I(X;Y)$ beyond it. *Noise-robustness (idealized nearest-centroid):*
cluster MAP error $\le(\kappa-1)Q(\Delta_{\min}/2\sigma)\approx0$ for $\sigma\ll\Delta_{\min}$ — so a wide
$\sigma$-window has cluster accuracy $\approx$ clean while token recovery (governed by the fine
$\sqrt{\lambda_i}$) collapses. Hence flat-while-recovery-falls.

**(b) CLUB.** By T1, $I(e';e_0)=I(Y;e_0)=I(X;Y)$ is the *correct* leakage, but CLUB is a variational
upper bound (well-specified-critic assumption, estimator bias/variance — not certified) returning a
single scalar with no spectral decomposition: cannot exhibit $d_{\mathrm{eff}}$ or the top subspace.
$I_G$ is the same target in closed form from $(\Sigma,\sigma)$ — certified, attack-free, decomposable. $\blacksquare$

### Open risks (scoped, not gaps in the verified statements)

token-F1 vs per-token error (T3b ceilings positional Hamming only); achievability of T4 (bottom-mode
ablation, empirical); finite-sample $\lambda_i$ bias can shift $d_{\mathrm{eff}}$ when $n_\text{texts}
\lesssim d$ (use shrinkage / Marchenko–Pastur); low-$\sigma$ vacuity (report $\min\{H(e_0),I_G\}$);
capPVI flatness uses a nearest-centroid idealization.

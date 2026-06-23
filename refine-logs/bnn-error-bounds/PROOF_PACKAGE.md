# Proof Package — Union-Bhattacharyya & Fano error bounds for the BNN/NNS attack

**Date**: 2026-06-22 (round 2, after Codex review thread 019eefac)  
**Probe**: union-Bhattacharyya upper bound + Fano-equivocation lower bound on the
L0 embedding-channel MAP error, computed from the codebook `{e_v}` and noise level
σ (via fresh synthetic noise), independent of the attack's test-set decode.

**Supersedes**: the NNS-PVI proof package (rejected design; archive dir removed in cleanup).

---

## Definitions and Setup

- $\mathcal{P} = \{v_1, \ldots, v_K\}$ — finite token pool, **$K \geq 3$** (the Fano denominator $\log_2(K-1)$ is positive; the realistic pool is $K=2048$; the binary case $K=2$ is handled separately via inverse binary entropy and is out of scope here).
- $e_v = \mathrm{clip}(\tilde{e}_v, C) \in \mathbb{R}^d$ — clipped embedding; **distinct**: $\Delta_{vu} := e_v - e_u$, $\|\Delta_{vu}\| > 0$ for $v \neq u$.
- **Uniform prior** $V \sim \mathrm{Unif}(\mathcal{P})$, $p(v) = 1/K$ — the primary scope of all theorems below. (A non-uniform extension is given as Remark N at the end; it requires prior-aware formulas and is not the headline claim.)
- $\varepsilon \sim \mathcal{N}(0, \sigma^2 I_d)$, $\sigma > 0$, independent of $V$. $Y = e_V + \varepsilon$.
- **Declared-inputs assumption (A0)**: the tuple $(\{e_v\}, \sigma, p, K)$ is **fixed and known a priori** (WEIGHTS-PUB: codebook and noise level public; uniform prior declared) — in particular *not* estimated from the attack's channel observations $\{Y_i\}$.
- **MAP/Bayes decoder** under uniform prior $\hat{V}^*(y) = \arg\max_v \mathcal{N}(y;e_v,\sigma^2 I) = \arg\min_v \|y-e_v\|^2 = \hat{V}_\text{BNN}(y)$. Ties have probability 0.
- **Bayes error** $P_e^*(\sigma) = 1 - \mathbb{E}_Y[\max_v p(v\mid Y)] = \mathbb{P}(\hat V^*(Y) \neq V)$. **BNN TTRSR** (population, uniform) $= 1 - P_e^*$.
- $Q(x) = \mathbb{P}(\mathcal{N}(0,1) > x)$. $H(V\mid Y)$, $I(V;Y)$ in bits; $H(V)=\log_2 K$; $I = H(V) - H(V\mid Y)$.

**The probe** (functionals of $(\{e_v\}, \sigma, K)$ + synthetic RNG only, under A0 + uniform prior):

1. **Upper bound** (exact-pairwise union; Bhattacharyya relaxation):
$$P_e^{\mathrm{ub}}(\sigma) = \frac{1}{K}\sum_{v}\sum_{u\neq v} Q\!\left(\frac{\|\Delta_{vu}\|}{2\sigma}\right) \;\leq\; \frac{1}{2K}\sum_{v}\sum_{u\neq v}\exp\!\left(-\frac{\|\Delta_{vu}\|^2}{8\sigma^2}\right) =: P_e^{\mathrm{ub,B}}(\sigma)$$

2. **Equivocation, fresh-noise MC estimator** ($M$ synthetic draws per codeword):
$$\widehat{H}_M(\sigma) = \frac{1}{KM}\sum_{v}\sum_{j=1}^{M} g(v,\varepsilon_{vj}), \quad
g(v,\varepsilon) = \log_2\frac{\sum_u \exp(-\|e_v+\varepsilon-e_u\|^2/2\sigma^2)}{\exp(-\|\varepsilon\|^2/2\sigma^2)}, \quad \varepsilon_{vj}\overset{\text{iid}}{\sim}\mathcal{N}(0,\sigma^2 I_d)$$
unbiased for $H(V\mid Y)$ (T4). The $\varepsilon_{vj}$ come from an RNG seed independent of $\{Y_i\}$.

3. **Population lower bound** (Fano, exact as $M\to\infty$): $\displaystyle P_{e,\infty}^{\mathrm{lb}}(\sigma) = \frac{H(V\mid Y) - 1}{\log_2(K-1)}$.

3'. **Certified finite-$M$ lower bound** (asymptotic coverage; fix $\alpha\in(0,\tfrac12)$, $M\geq2$ so the per-codeword sample variances are defined): with the stratified standard-error estimate $\widehat{\mathrm{se}}^2 = \frac{1}{K^2 M}\sum_v \widehat{s}_v^2$ (where $\widehat s_v^2$ is the sample variance of $\{g_v(\varepsilon_{vj})\}_j$) and one-sided lower confidence bound $\underline{H}_{M,\alpha} = \widehat H_M - z_{1-\alpha}\,\widehat{\mathrm{se}}$,
$$P_{e,\alpha}^{\mathrm{lb}}(\sigma) = \frac{\underline{H}_{M,\alpha} - 1}{\log_2(K-1)}, \qquad \liminf_{M\to\infty}\ \mathbb{P}\big(P_{e,\alpha}^{\mathrm{lb}}(\sigma) \leq P_e^*(\sigma)\big) \geq 1-\alpha \ \text{(CLT coverage, not a finite-}M\text{ certificate).}$$

---

## Theorem 1 — Two-sided bound validity (uniform prior, K≥3)

### Claim
Under A0 and uniform prior, for every $\sigma>0$:
$$P_{e,\infty}^{\mathrm{lb}}(\sigma) \;\leq\; P_e^*(\sigma) \;\leq\; P_e^{\mathrm{ub}}(\sigma) \;\leq\; P_e^{\mathrm{ub,B}}(\sigma)$$
The upper bounds and $P_{e,\infty}^{\mathrm{lb}}$ are functions of $(\{e_v\},\sigma,K)$ only. The finite-$M$ certified bound $P_{e,\alpha}^{\mathrm{lb}} \leq P_e^*$ holds with probability $\geq 1-\alpha$.

### Status: PROVABLE AS STATED

### Proof
**Upper.** Condition on $V=v$. Under *uniform* prior the MAP rule is nearest-neighbor, so an error requires some $u\neq v$ with $\|Y-e_u\| < \|Y-e_v\|$. Union bound:
$$\mathbb{P}(\hat V^*\neq v\mid V=v) \leq \sum_{u\neq v}\mathbb{P}(\|Y-e_u\|<\|Y-e_v\|\mid V=v).$$
With $Y=e_v+\varepsilon$: $\|Y-e_u\|^2-\|Y-e_v\|^2 = \|\Delta_{vu}\|^2 + 2\langle\varepsilon,\Delta_{vu}\rangle$, negative iff $\langle\varepsilon,\Delta_{vu}\rangle < -\|\Delta_{vu}\|^2/2$. Since $\langle\varepsilon,\Delta_{vu}\rangle\sim\mathcal{N}(0,\sigma^2\|\Delta_{vu}\|^2)$, this equals $Q(\|\Delta_{vu}\|/2\sigma)$. Averaging over $v$ uniformly gives $P_e^*\leq P_e^{\mathrm{ub}}$. The Chernoff inequality $Q(x)\leq\tfrac12 e^{-x^2/2}$ ($x\geq0$) gives $P_e^{\mathrm{ub}}\leq P_e^{\mathrm{ub,B}}$. $\square$

**Lower (population).** Fano (Cover & Thomas Thm 2.10.1) for $V\to Y\to\hat V$, any decoder: $H(V\mid Y) \leq H_b(P_e) + P_e\log_2(K-1)$, $H_b\leq 1$. For the MAP decoder (minimum error $P_e^*$), and $K\geq3$ so $\log_2(K-1)>0$:
$$P_e^* \geq \frac{H(V\mid Y)-1}{\log_2(K-1)} = P_{e,\infty}^{\mathrm{lb}}.$$

**Lower (certified finite-$M$).** The **Fano lower-bound functional** $\ell(h) = (h-1)/\log_2(K-1)$ is increasing in $h$ (denominator $>0$), and $P_e^* \geq \ell(H(V\mid Y))$ by the population bound. $\widehat H_M$ is unbiased for $H(V\mid Y)$ (T4) with $\widehat{\mathrm{se}}\to0$; by the CLT, $\underline{H}_{M,\alpha} = \widehat H_M - z_{1-\alpha}\widehat{\mathrm{se}} \leq H(V\mid Y)$ with asymptotic probability $\geq1-\alpha$. Since $\ell$ is increasing, $P_{e,\alpha}^{\mathrm{lb}} = \ell(\underline{H}_{M,\alpha}) \leq \ell(H(V\mid Y)) \leq P_e^*$ with the same asymptotic coverage. $\square$

### Honest scope / Open risks
- Upper bound **vacuous** ($>1$) at low SNR; report $\min(1,\cdot)$. Tight at high SNR / near-orthogonal pool (min-distance regime).
- Population lower bound **vacuous** ($\leq0$) when $H(V\mid Y)\leq1$ bit.
- Tighter Fano via Arimoto/Sibson $\alpha$-information (Rioul et al. arXiv:2105.07167); deferred.
- The finite-$M$ certified bound is **asymptotic** (CLT coverage). $g_v$ is **unbounded** under Gaussian synthetic noise (take $\varepsilon$ far along $-\Delta_{vu}$), so Hoeffding does **not** apply directly. For a non-asymptotic guarantee, use the fact that $g_v$ is **sub-exponential** (it is bounded below by 0 and above by $\log_2 K + \tfrac{1}{\ln 2}\max_u|\langle\varepsilon,\Delta_{vu}\rangle|/\sigma^2$, a maximum of sub-Gaussian linear forms; see T4), and apply a Bernstein-type concentration inequality with the sub-exponential norm as the explicit constant — or truncate $\varepsilon$ to a high-probability ball.

---

## Theorem 2 — BNN achieves the bracketed error (uniform prior)

### Claim
Under A0 + uniform prior, BNN = MAP, so its population error equals $P_e^*\in[P_{e,\infty}^{\mathrm{lb}}, P_e^{\mathrm{ub}}]$. The measured TTRSR on $n$ iid test points satisfies $|(1-\widehat{\text{TTRSR}}_n) - P_e^*| \leq \sqrt{\log(2/\delta)/(2n)}$ with probability $\geq1-\delta$ (Hoeffding).

### Status: PROVABLE AS STATED
### Proof
Uniform prior $\Rightarrow \arg\max_v p(v)\mathcal N(y;e_v,\sigma^2I)=\arg\min_v\|y-e_v\|^2=\hat V_\text{BNN}$. MAP minimizes error over all decoders, attaining $P_e^*$; T1 brackets it. The empirical error is a mean of $n$ iid $\{0,1\}$ indicators with mean $P_e^*$; Hoeffding gives the deviation bound. $\square$

---

## Theorem 3 — Independence from the attack

### Claim
Under A0, the probe triple $(P_e^{\mathrm{ub}}, \widehat H_M, P_{e,\cdot}^{\mathrm{lb}})$ is a measurable function of $(\{e_v\},\sigma,K)$ and the synthetic-noise RNG only. Given the public inputs $(\{e_v\},\sigma,K)$ and synthetic RNG independent of the channel, the probe is **conditionally independent of the entire attack transcript** $\{(V_i, Y_i)\}_i$ — hence of the attack's predictions and its finite-sample error (which is a function of both $\{Y_i\}$ and the true labels $\{V_i\}$).

### Status: PROVABLE AS STATED (by construction, under A0)
### Proof
$P_e^{\mathrm{ub}}, P_e^{\mathrm{ub,B}}$ are explicit functions of the self-distance multiset $\{\|\Delta_{vu}\|\}_{u\neq v}$ and $\sigma$ — neither $Y_i$ nor $V_i$ appears. $\widehat H_M$ is a function of $\{e_v\}$, $\sigma$, and synthetic $\{\varepsilon_{vj}\}$ drawn from an RNG seed that A0 fixes independently of the channel — no transcript variable appears. Hence $\sigma(\text{probe}) \subseteq \sigma(\{e_v\},\sigma,K,\mathrm{RNG}_\text{synth})$, which by A0 + RNG-independence is independent of $\sigma(\{(V_i,Y_i)\})$. $\square$

### Contrast with NNS-PVI (superseded)
NNS-PVI evaluated $\log q_\tau(V_i\mid Y_i)$ on the **attack's own observations** $Y_i$, sharing the matrix $\|Y_i-e_v\|^2$ with BNN; its "independence" was `softmax-at-truth` vs `argmax` — the same computation. The two-sided bound removes $\{Y_i\}$ entirely; independence is by inspection under A0.

### Caveat
If A0 fails — i.e. $\sigma$, $\{e_v\}$, or the prior are *estimated from* $\{Y_i\}$ — the independence breaks. The claim is conditional independence given the declared public inputs, not unconditional.

---

## Theorem 4 — Unbiasedness and consistency of $\widehat H_M$

### Claim
$\mathbb{E}[\widehat H_M] = H(V\mid Y)$; $\widehat H_M \to H(V\mid Y)$ a.s. as $M\to\infty$; and for fixed finite codebook and fixed $\sigma>0$, $\mathrm{Var}(\widehat H_M) = \frac{C_{K,E,\sigma}}{KM}$ with $C_{K,E,\sigma} = \frac1K\sum_v\mathrm{Var}(g_v)$ (no uniformity claimed as $K$ grows or $\sigma\to0$).

### Status: PROVABLE AS STATED (fixed σ, finite codebook)
### Proof
Under uniform prior, $H(V\mid Y) = -\mathbb{E}_{V,Y}[\log_2 p(V\mid Y)]$ with $p(v\mid y) = \mathcal N(y;e_v,\sigma^2I)/\sum_u\mathcal N(y;e_u,\sigma^2I)$. Conditioning $V=v$, $Y=e_v+\varepsilon$:
$$-\log_2 p(v\mid e_v+\varepsilon) = \log_2\frac{\sum_u\exp(-\|e_v+\varepsilon-e_u\|^2/2\sigma^2)}{\exp(-\|\varepsilon\|^2/2\sigma^2)} = g(v,\varepsilon).$$
Define $g_v(\varepsilon)=g(v,\varepsilon)$.

*Integrability (established first).* $g_v\geq0$ (the $u=v$ term equals 1, so the sum $\geq1$, log $\geq0$). Using $\log_2\sum_u e^{a_u}\leq \log_2 K + (\max_u a_u)/\ln2$ with $a_u = -(\|\Delta_{vu}\|^2 + 2\langle\varepsilon,\Delta_{vu}\rangle)/2\sigma^2$ and $a_v=0$ (so $\max_u a_u \geq 0$):
$$0\leq g_v(\varepsilon) \leq \log_2 K + \frac{1}{\ln2}\,\max\!\Big(0,\ \max_{u\neq v}\frac{-\|\Delta_{vu}\|^2 - 2\langle\varepsilon,\Delta_{vu}\rangle}{2\sigma^2}\Big) \leq \log_2 K + \frac{1}{\ln2}\,\max_{u\neq v}\frac{|\langle\varepsilon,\Delta_{vu}\rangle|}{\sigma^2}.$$
Each $\langle\varepsilon,\Delta_{vu}\rangle$ is Gaussian, so $\max_{u\neq v}|\langle\varepsilon,\Delta_{vu}\rangle|$ has finite moments of all orders; hence $\mathbb{E}[g_v]=:\mu_v<\infty$ and $\mathrm{Var}(g_v)<\infty$ for fixed $\sigma>0$ and finite codebook.

*Unbiasedness.* With integrability secured, linearity over the finite sum gives $\mu_v=\mathbb{E}_\varepsilon[g_v]=\mathbb{E}_{Y\mid v}[-\log_2 p(v\mid Y)]$, $\frac1K\sum_v\mu_v = H(V\mid Y)$, so $\mathbb{E}[\widehat H_M] = \frac1{KM}\sum_{v,j}\mu_v = H(V\mid Y)$.

*Consistency.* The draws $\{\varepsilon_{vj}\}$ are independent across $(v,j)$; for **fixed $v$** the $\{g_v(\varepsilon_{vj})\}_j$ are iid, so by SLLN $\frac1M\sum_j g_v(\varepsilon_{vj})\to\mu_v$ a.s. Averaging over the **finite** set of $K$ codewords: $\widehat H_M = \frac1K\sum_v\big(\frac1M\sum_j g_v(\varepsilon_{vj})\big)\to\frac1K\sum_v\mu_v = H(V\mid Y)$ a.s. (Note: summands are **not** identically distributed across $v$; the SLLN is applied per-$v$ then averaged over finite $K$.)

*Variance.* By independence across $v$ and within-$v$ iid: $\mathrm{Var}(\widehat H_M) = \frac1{K^2}\sum_v \mathrm{Var}\big(\frac1M\sum_j g_v(\varepsilon_{vj})\big) = \frac1{K^2 M}\sum_v\mathrm{Var}(g_v) = \frac{C_{K,E,\sigma}}{KM}$. $\square$

### Open risk
The constant $C_{K,E,\sigma} = \frac1K\sum_v\mathrm{Var}(g_v)$ depends on $\sigma$ (via $1/\sigma^2$), on $\max_{vu}\|\Delta_{vu}\|$, and on the codebook geometry through the $\max$ over $K-1$ Gaussian projections — it is **not** uniform as $\sigma\to0$ or as $K$ grows. Stated for fixed $\sigma>0$, finite $K$. Report the empirical MC standard error per $\sigma$.

---

## Theorem 5 — Monotonicity in σ

### Claim
(a) $P_e^{\mathrm{ub}}(\sigma)$, $P_e^{\mathrm{ub,B}}(\sigma)$, and $H(V\mid Y)(\sigma)$ are non-decreasing in $\sigma$.
(b) $P_e^*(\sigma)$ is non-decreasing in $\sigma$; under uniform prior so is $1-\text{BNN TTRSR}$. Hence the whole two-sided bound and the bracketed quantity move co-monotonically.

### Status: PROVABLE AS STATED (non-strict; strictness of $H(V\mid Y)$ noted as remark)
### Proof
**(a)** For fixed $\|\Delta\|>0$, $Q(\|\Delta\|/2\sigma)$ has argument decreasing in $\sigma$, and $Q$ is decreasing, so it is non-decreasing in $\sigma$; likewise $\tfrac12\exp(-\|\Delta\|^2/8\sigma^2)$. Non-negative-weighted ($1/K$) sums preserve this. For $H(V\mid Y)=\log_2 K - I(V;Y)$: for $\sigma_1<\sigma_2$, $Y_{\sigma_2}\stackrel{d}{=}Y_{\sigma_1}+\eta$ with $\eta\sim\mathcal N(0,(\sigma_2^2-\sigma_1^2)I)$ independent, so $V\to Y_{\sigma_1}\to Y_{\sigma_2}$ is Markov; by the data-processing inequality $I(V;Y_{\sigma_2})\leq I(V;Y_{\sigma_1})$, hence $H(V\mid Y)$ non-decreasing in $\sigma$.

**(b)** The same simulation $Y_{\sigma_2}\stackrel{d}{=}Y_{\sigma_1}+\eta$ shows any decoder acting on $Y_{\sigma_2}$ can be realized as a (randomized) decoder on $Y_{\sigma_1}$; therefore the minimum achievable error cannot be smaller at $\sigma_2$, i.e. $P_e^*(\sigma_2)\geq P_e^*(\sigma_1)$. Under uniform prior BNN attains $P_e^*$, so $1-\text{BNN TTRSR}=P_e^*$ is non-decreasing. $\square$

### Remark (strictness)
Strict monotonicity of $H(V\mid Y)$ requires more than the DPI (which gives $\leq$): e.g. the I-MMSE relation $dI/d\mathrm{snr}=\tfrac12\mathrm{mmse}$ with strictly positive MMSE for a non-degenerate finite constellation gives strict decrease of $I$, hence strict increase of $H(V\mid Y)$. Stated separately to avoid over-claiming from DPI alone.

---

## Remark N — Non-uniform prior extension (not the headline claim)

For empirical prior $\hat p$, the MAP boundary shifts and the bounds become prior-aware. **Caveat on independence**: if $\hat p$ is the class-frequency of the *same* test labels $\{V_i\}$ used to score the attack, then $\hat p$ touches the attack transcript (test-set class imbalance affects both $\hat p$ and the measured finite error), so T3's independence weakens to *conditional* independence given the label multiset/counts. To preserve full T3 independence, estimate $\hat p$ from a **separate public corpus**. The prior-aware formulas:
- Upper: pairwise term $Q\!\big(\tfrac{\|\Delta_{vu}\|}{2\sigma} - \tfrac{\sigma}{\|\Delta_{vu}\|}\log\tfrac{\hat p(u)}{\hat p(v)}\big)$, weighted by $\hat p(v)$; Bhattacharyya form $\sqrt{\hat p(u)\hat p(v)}\exp(-\|\Delta_{vu}\|^2/8\sigma^2)$.
- Equivocation: $\widehat H_M = -\sum_v \hat p(v)\,\mathbb{E}_\varepsilon\log_2\frac{\hat p(v)\exp(-\|\varepsilon\|^2/2\sigma^2)}{\sum_u \hat p(u)\exp(-\|e_v+\varepsilon-e_u\|^2/2\sigma^2)}$.
This is deferred; the uniform-prior geometry-only two-sided bound (T1–T5) is the validated headline. For the experiment, BNN TTRSR is compared under matched prior (macro-uniform over the pool, or empirical via the prior-aware formulas).

---

## Complexity Analysis and Viability

(unchanged from round 1; reproduced for completeness)

| Quantity | Dominant op | FLOP | Wall-clock | ⊥ n? |
|---|---|---|---|---|
| Codebook Gram $\{\|\Delta_{vu}\|^2\}$ | one $K\times K$ via $(K,d)(d,K)$ | $2K^2 d\approx19.3$ GFLOP | ~2 ms, cached over σ | yes |
| $P_e^{\mathrm{ub}},P_e^{\mathrm{ub,B}}$ | re-exp Gram per σ | $O(K^2)$ | sub-ms/σ | yes |
| $\widehat H_M$, $M$/codeword | $KM$ pts × $K$-way LSE | $2K^2 M d$ | $M{=}64$: ~1.2 s/σ | yes |

5-point ε-sweep total ≈ **~6 s**, **independent of test-set size $n$** — structurally cheaper than CLUB/CapPVI/MDL (all $O(n)$ + iterative training). Upper bound alone is essentially free.

**Viability: viable and preferred.** Cheapest non-trivial probe; uniquely $n$-independent; textbook bounds (low proof risk); genuinely attack-independent by construction (T3); predicts BNN-not-1.0 from the real distance histogram (small-$\|\Delta\|$ pairs dominate the union sum, reproducing the ≈0.6% morphological-confusion floor of `claim:bnn-nns-high-d-geometry`). Known limitation: bounds can be loose (upper vacuous at low SNR; Fano loose without α-information) — the **gap between the upper and lower bounds is itself a diagnostic** of where geometry alone determines the outcome.

---

## Summary

| Theorem | Status | Basis |
|---|---|---|
| T1 two-sided bound validity (uniform, K≥3) | PROVABLE AS STATED | union bound + Fano; population vs certified finite-M separated |
| T2 BNN achieves bracketed error | PROVABLE AS STATED | MAP optimality + Hoeffding |
| T3 independence from attack | PROVABLE AS STATED | by construction under A0 |
| T4 unbiased + consistent $\widehat H_M$ | PROVABLE AS STATED | per-v SLLN, stratified variance, fixed σ |
| T5 monotonicity in σ | PROVABLE AS STATED | $Q$/exp monotone; DPI for $H(V\mid Y)$ and $P_e^*$ |

**References**: Proakis *Digital Communications* (M-ary union/min-distance bounds); Cover & Thomas *Elements of IT* §11.9 (Chernoff), Thm 2.10.1 (Fano); de Chérisey et al. TCHES 2019 (MI ↔ optimal-attack success, AWGN); Rioul et al. arXiv:2105.07167 (α-information Fano); Feyisetan et al. WSDM 2020, Mattern et al. NAACL-F 2022 (DP-NLP framing).

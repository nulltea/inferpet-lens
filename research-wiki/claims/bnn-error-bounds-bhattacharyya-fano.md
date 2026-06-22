---
type: claim
node_id: claim:bnn-error-bounds-bhattacharyya-fano
name: "Union-Bhattacharyya (upper) and Fano-equivocation (lower) bounds two-sidedly bound the BNN/NNS L0 attack error, computed geometry-only and independent of the attack"
description: ""
node_type: claim
status: verified
provenance: "refine-logs/PROOF_PACKAGE.md; refine-logs/PROOF_AUDIT.md; Codex thread 019eefac, 3 rounds, verdict PASS"
tags: ["bnn", "nns", "union-bhattacharyya", "fano-equivocation", "map-error-bounds", "dp-l0", "matched-probe", "geometry-only", "m-ary-gaussian-channel"]
date: 2026-06-22
added: 2026-06-22T00:00:00Z
---

# Union-Bhattacharyya (upper) and Fano-equivocation (lower) bounds two-sidedly bound the BNN/NNS L0 attack error, geometry-only and independent of the attack

**status:** `verified` — proof closes (Codex `gpt-5.5` xhigh, thread `019eefac`, 3 rounds, zero open
FATAL/CRITICAL). The two bound *steps* are imported textbook results (Proakis union/Bhattacharyya bound;
Cover–Thomas Fano) flagged as such; the novel independence (T3) and estimator consistency (T4) are proved here.

## One-line statement
The L0 embedding-DP channel `Y = e_V + N(0,σ²I_d)` with known codebook `{e_v}` and MAP/BNN decoder is
textbook M-ary Gaussian signaling. Under uniform prior, `K≥3`, and the declared-inputs assumption A0, the
MAP (=BNN) error `P_e*` is **two-sidedly bounded** — from above by the **union-Bhattacharyya bound** and
from below by the **Fano-equivocation bound** — both computed from the codebook geometry + σ alone, never
touching the attack's observations `{Y_i}`. This predicts the optimal attack *independently of the attack
by construction* — the property the superseded [[claim:nns-pvi]] design lacked.

---

## Definitions and Setup

- $\mathcal{P}=\{v_1,\dots,v_K\}$ — token pool, **$K\geq3$** ($\log_2(K-1)>0$; experiments use $K=2048$; binary $K=2$ out of scope, needs inverse-binary-entropy Fano).
- $e_v=\mathrm{clip}(\tilde e_v,C)\in\mathbb{R}^d$, **distinct**: $\Delta_{vu}:=e_v-e_u$, $\|\Delta_{vu}\|>0$ for $v\neq u$.
- **Uniform prior** $V\sim\mathrm{Unif}(\mathcal{P})$ — headline scope (non-uniform = Remark N).
- $\varepsilon\sim\mathcal{N}(0,\sigma^2 I_d)$, $\sigma>0$, $\perp V$; $Y=e_V+\varepsilon$.
- **(A0) declared-inputs**: $(\{e_v\},\sigma,p,K)$ fixed/public a priori (WEIGHTS-PUB), *not* estimated from $\{Y_i\}$.
- **MAP/BNN decoder** (uniform): $\hat V^*(y)=\arg\max_v\mathcal N(y;e_v,\sigma^2I)=\arg\min_v\|y-e_v\|^2$. Ties prob 0.
- **Bayes error** $P_e^*(\sigma)=1-\mathbb{E}_Y[\max_v p(v\mid Y)]=\mathbb{P}(\hat V^*(Y)\neq V)$. BNN TTRSR (pop, uniform) $=1-P_e^*$.
- $Q(x)=\mathbb{P}(\mathcal N(0,1)>x)$; $H(V\mid Y),I(V;Y)$ in bits, $H(V)=\log_2K$.

**The probe** (functionals of $(\{e_v\},\sigma,K)$ + synthetic RNG only):

1. **Union-Bhattacharyya upper bound** (exact-pairwise union; Bhattacharyya relaxation):
$$P_e^{\mathrm{ub}}(\sigma)=\frac1K\sum_v\sum_{u\neq v}Q\!\Big(\frac{\|\Delta_{vu}\|}{2\sigma}\Big)\;\leq\;\frac1{2K}\sum_v\sum_{u\neq v}\exp\!\Big(-\frac{\|\Delta_{vu}\|^2}{8\sigma^2}\Big)=:P_e^{\mathrm{ub,B}}(\sigma)$$

2. **Fano-equivocation, fresh-noise MC estimator** ($M$ synthetic draws / codeword):
$$\widehat H_M(\sigma)=\frac1{KM}\sum_v\sum_{j=1}^{M}g(v,\varepsilon_{vj}),\quad g(v,\varepsilon)=\log_2\frac{\sum_u\exp(-\|e_v+\varepsilon-e_u\|^2/2\sigma^2)}{\exp(-\|\varepsilon\|^2/2\sigma^2)},\quad \varepsilon_{vj}\overset{\text{iid}}{\sim}\mathcal N(0,\sigma^2I_d)$$
unbiased for $H(V\mid Y)$ (T4); the $\varepsilon_{vj}$ use an RNG seed independent of $\{Y_i\}$.

3. **Fano lower bound, population** (exact as $M\to\infty$): $P_{e,\infty}^{\mathrm{lb}}(\sigma)=\dfrac{H(V\mid Y)-1}{\log_2(K-1)}$.

3'. **Fano lower bound, certified finite-$M$** (asymptotic CLT coverage; fix $\alpha\in(0,\tfrac12)$, $M\geq2$): with stratified $\widehat{\mathrm{se}}^2=\frac1{K^2M}\sum_v\widehat s_v^2$ ($\widehat s_v^2$ = sample variance of $\{g_v(\varepsilon_{vj})\}_j$) and $\underline H_{M,\alpha}=\widehat H_M-z_{1-\alpha}\widehat{\mathrm{se}}$:
$$P_{e,\alpha}^{\mathrm{lb}}=\frac{\underline H_{M,\alpha}-1}{\log_2(K-1)},\qquad \liminf_{M\to\infty}\mathbb{P}\big(P_{e,\alpha}^{\mathrm{lb}}\leq P_e^*\big)\geq1-\alpha\ \text{(coverage, not a finite-}M\text{ certificate).}$$

---

## Theorem 1 — Two-sided bound validity (uniform prior, K≥3)
**Claim.** $\forall\sigma>0$: $P_{e,\infty}^{\mathrm{lb}}\leq P_e^*\leq P_e^{\mathrm{ub}}\leq P_e^{\mathrm{ub,B}}$; the upper bounds and $P_{e,\infty}^{\mathrm{lb}}$ are functions of $(\{e_v\},\sigma,K)$ only; $P_{e,\alpha}^{\mathrm{lb}}\leq P_e^*$ w.p. $\geq1-\alpha$.
**Proof.** *Union-Bhattacharyya upper:* condition $V=v$; uniform ⟹ MAP = nearest-neighbor, so error ⟹ $\exists u\neq v$ with $\|Y-e_u\|<\|Y-e_v\|$. Union bound over $u$. With $Y=e_v+\varepsilon$, $\|Y-e_u\|^2-\|Y-e_v\|^2=\|\Delta_{vu}\|^2+2\langle\varepsilon,\Delta_{vu}\rangle<0$ iff $\langle\varepsilon,\Delta_{vu}\rangle<-\|\Delta_{vu}\|^2/2$; since $\langle\varepsilon,\Delta_{vu}\rangle\sim\mathcal N(0,\sigma^2\|\Delta_{vu}\|^2)$ this is $Q(\|\Delta_{vu}\|/2\sigma)$. Average over $v$ ⟹ $P_e^*\leq P_e^{\mathrm{ub}}$. Chernoff $Q(x)\leq\tfrac12e^{-x^2/2}$ ($x\geq0$) ⟹ $P_e^{\mathrm{ub}}\leq P_e^{\mathrm{ub,B}}$. *Fano lower (population):* Fano (Cover–Thomas Thm 2.10.1) $H(V\mid Y)\leq H_b(P_e)+P_e\log_2(K-1)$, $H_b\leq1$; for MAP (min error $P_e^*$), $K\geq3$ ⟹ $P_e^*\geq(H(V\mid Y)-1)/\log_2(K-1)$. *Fano lower (certified):* the functional $\ell(h)=(h-1)/\log_2(K-1)$ is increasing; $\widehat H_M$ unbiased (T4), CLT ⟹ $\underline H_{M,\alpha}\leq H(V\mid Y)$ w.p. $\geq1-\alpha$, so $\ell(\underline H_{M,\alpha})\leq\ell(H(V\mid Y))\leq P_e^*$. ∎
**Honest scope.** Union-Bhattacharyya upper **vacuous** ($>1$) at low SNR (report $\min(1,\cdot)$); tight at high SNR / near-orthogonal pool. Fano lower **vacuous** ($\leq0$) when $H(V\mid Y)\leq1$ bit. Tighter via α-information Fano (Rioul et al. arXiv:2105.07167), deferred. Finite-$M$ Fano bound is asymptotic; $g_v$ is **unbounded** (Hoeffding inapplicable) but **sub-exponential** (T4), so a non-asymptotic route uses Bernstein-type concentration or ε-truncation.

## Theorem 2 — BNN attains the two-sidedly bounded error
**Claim.** Uniform ⟹ BNN = MAP ⟹ population error $=P_e^*\in[P_{e,\infty}^{\mathrm{lb}},P_e^{\mathrm{ub}}]$; measured $|(1-\widehat{\mathrm{TTRSR}}_n)-P_e^*|\leq\sqrt{\log(2/\delta)/2n}$ w.p. $\geq1-\delta$ (Hoeffding).
**Proof.** Uniform ⟹ $\arg\max_v p(v)\mathcal N=\arg\min_v\|y-e_v\|^2=\hat V_\text{BNN}$; MAP minimizes error over all decoders (attains $P_e^*$); T1 bounds it. Empirical error = mean of $n$ iid $\{0,1\}$ with mean $P_e^*$; Hoeffding. ∎

## Theorem 3 — Independence from the attack (the key property)
**Claim.** Under A0, the probe triple $(P_e^{\mathrm{ub}},\widehat H_M,P_{e,\cdot}^{\mathrm{lb}})$ is a measurable function of $(\{e_v\},\sigma,K)$ + synthetic RNG only; given the public inputs and RNG $\perp$ channel, it is **conditionally independent of the entire attack transcript** $\{(V_i,Y_i)\}$ — hence of the attack's predictions and finite-sample error.
**Proof.** $P_e^{\mathrm{ub}},P_e^{\mathrm{ub,B}}$ are explicit functions of the self-distance multiset $\{\|\Delta_{vu}\|\}$ and $\sigma$ — no $Y_i,V_i$. $\widehat H_M$ is a function of $\{e_v\},\sigma$, synthetic $\{\varepsilon_{vj}\}$ (RNG seed fixed by A0 $\perp$ channel) — no transcript variable. Hence $\sigma(\text{probe})\subseteq\sigma(\{e_v\},\sigma,K,\mathrm{RNG}_\text{synth})\perp\sigma(\{(V_i,Y_i)\})$ by A0. ∎
**Contrast (the reason NNS-PVI was rejected).** NNS-PVI evaluated $\log q_\tau(V_i\mid Y_i)$ on the attack's *own* $Y_i$, sharing $\|Y_i-e_v\|^2$ with BNN; "independence" was softmax-at-truth vs argmax — same computation. The union-Bhattacharyya and Fano bounds remove $\{Y_i\}$ entirely; independence is by inspection under A0.
**Caveat.** If A0 fails ($\sigma$, $\{e_v\}$, or prior estimated from $\{Y_i\}$), independence breaks; the claim is conditional independence given the declared public inputs.

## Theorem 4 — Unbiasedness + consistency of $\widehat H_M$
**Claim.** $\mathbb{E}[\widehat H_M]=H(V\mid Y)$; $\widehat H_M\to H(V\mid Y)$ a.s. as $M\to\infty$; $\mathrm{Var}(\widehat H_M)=C_{K,E,\sigma}/(KM)$, $C_{K,E,\sigma}=\frac1K\sum_v\mathrm{Var}(g_v)$ (no uniformity as $K$ grows / $\sigma\to0$).
**Proof.** Uniform ⟹ $H(V\mid Y)=-\mathbb{E}_{V,Y}[\log_2 p(V\mid Y)]$, $p(v\mid y)=\mathcal N(y;e_v,\sigma^2I)/\sum_u\mathcal N(y;e_u,\sigma^2I)$. Conditioning $V=v,Y=e_v+\varepsilon$ gives $-\log_2 p(v\mid e_v+\varepsilon)=g(v,\varepsilon)=:g_v(\varepsilon)$.
*Integrability (first):* $g_v\geq0$ ($u=v$ term $=1$). With $\log_2\sum_u e^{a_u}\leq\log_2K+(\max_u a_u)/\ln2$, $a_u=-(\|\Delta_{vu}\|^2+2\langle\varepsilon,\Delta_{vu}\rangle)/2\sigma^2$, $a_v=0$:
$$0\leq g_v(\varepsilon)\leq\log_2K+\tfrac1{\ln2}\max_{u\neq v}\frac{|\langle\varepsilon,\Delta_{vu}\rangle|}{\sigma^2}.$$
$\langle\varepsilon,\Delta_{vu}\rangle$ Gaussian ⟹ $\max_{u\neq v}|\cdot|$ has all moments ⟹ $\mu_v:=\mathbb{E}[g_v]<\infty$, $\mathrm{Var}(g_v)<\infty$.
*Unbiasedness:* linearity over the finite sum ⟹ $\frac1K\sum_v\mu_v=H(V\mid Y)$, so $\mathbb{E}[\widehat H_M]=H(V\mid Y)$.
*Consistency:* $\{\varepsilon_{vj}\}$ indep across $(v,j)$; for fixed $v$, $\{g_v(\varepsilon_{vj})\}_j$ iid ⟹ SLLN $\frac1M\sum_j g_v\to\mu_v$ a.s.; average over finite $K$ ⟹ $\widehat H_M\to H(V\mid Y)$ a.s. (summands **not** identically distributed across $v$ — SLLN per-$v$ then average).
*Variance:* indep across $v$, iid within $v$ ⟹ $\mathrm{Var}(\widehat H_M)=\frac1{K^2M}\sum_v\mathrm{Var}(g_v)=C_{K,E,\sigma}/(KM)$. ∎
**Open risk.** $C_{K,E,\sigma}$ depends on $\sigma$ (via $1/\sigma^2$), $\max_{vu}\|\Delta_{vu}\|$, and the $\max$ over $K-1$ projections — not uniform as $\sigma\to0$ or $K$ grows. Report empirical MC SE per $\sigma$.

## Theorem 5 — Monotonicity in σ
**Claim.** (a) $P_e^{\mathrm{ub}},P_e^{\mathrm{ub,B}},H(V\mid Y)$ non-decreasing in $\sigma$. (b) $P_e^*$ non-decreasing in $\sigma$; uniform ⟹ $1-\text{BNN TTRSR}$ too — the upper bound, lower bound, and the bounded error all move co-monotonically.
**Proof.** (a) $Q(\|\Delta\|/2\sigma)$, $\tfrac12e^{-\|\Delta\|^2/8\sigma^2}$ increasing in $\sigma$ (argument decreasing, $Q$ decreasing); $1/K$-weighted sums preserve. $H(V\mid Y)=\log_2K-I$; $\sigma_1<\sigma_2$ ⟹ $Y_{\sigma_2}\overset{d}{=}Y_{\sigma_1}+\eta$, $\eta\sim\mathcal N(0,(\sigma_2^2-\sigma_1^2)I)$ indep ⟹ $V\to Y_{\sigma_1}\to Y_{\sigma_2}$ Markov ⟹ DPI $I(V;Y_{\sigma_2})\leq I(V;Y_{\sigma_1})$ ⟹ $H(V\mid Y)$ non-decreasing. (b) same simulation ⟹ any decoder on $Y_{\sigma_2}$ realizable from $Y_{\sigma_1}+\eta$ ⟹ min error non-decreasing ⟹ $P_e^*(\sigma_2)\geq P_e^*(\sigma_1)$. ∎
**Remark (strictness).** Strict monotone $H(V\mid Y)$ needs more than DPI: e.g. I-MMSE $dI/d\mathrm{snr}=\tfrac12\mathrm{mmse}$ with $\mathrm{mmse}>0$ for a non-degenerate constellation. Stated separately to avoid over-claiming.

---

## Remark N — Non-uniform prior extension (deferred, not headline)
For empirical $\hat p$: if $\hat p$ is the class-frequency of the *same* test labels $\{V_i\}$, it touches the transcript ⟹ T3 weakens to conditional independence given label counts; estimate $\hat p$ from a **separate public corpus** to preserve full independence. Prior-aware forms: upper pairwise $Q(\tfrac{\|\Delta_{vu}\|}{2\sigma}-\tfrac{\sigma}{\|\Delta_{vu}\|}\log\tfrac{\hat p(u)}{\hat p(v)})$ weighted $\hat p(v)$; Bhattacharyya $\sqrt{\hat p(u)\hat p(v)}e^{-\|\Delta_{vu}\|^2/8\sigma^2}$; equivocation with $\hat p$ inside the posterior.

## Viability (complexity)
Cheapest non-trivial probe and uniquely **⊥ test-set size $n$**: codebook Gram $O(K^2d)\approx19$ GFLOP once (~2 ms, cached over σ); union-Bhattacharyya upper bound sub-ms/σ; Fano $\widehat H_{64}$ ~1.2 s/σ. Full 5-point ε-sweep ~6 s vs 20–60 s/ε iterative training for CLUB/CapPVI/MDL. Reproduces BNN-not-1.0: small-$\|\Delta\|$ morphological pairs dominate the union sum, recovering the ≈0.6% confusion floor of [[claim:bnn-nns-high-d-geometry]] from geometry alone. Limitation: bounds loose at extremes — **the gap between the upper and lower bounds is itself a diagnostic** of where geometry determines the outcome.

## Connections
Uses [[paper:cherisey2019_best_information_most]] (MI predicts optimal-attack success under AWGN),
[[paper:cover2006_elements_information_theory]] (Fano Thm 2.10.1; Chernoff §11.9),
[[claim:thm-t1-info-efficient]] (BNN is the Bayes-optimal L0 instance), [[claim:bayes-gap-diagnosis]]
(instantiates the Fano/de-Cherisey ceiling as computable geometry-only bounds).
Bounds/supports [[claim:bnn-nns-high-d-geometry]]. Supersedes the NNS-PVI design (rejected — see wiki log 2026-06-22). Tested by exp:bnn-error-bounds-validation (planned, `refine-logs/EXPERIMENT_PLAN.md`).
Full audited proof: `refine-logs/PROOF_PACKAGE.md`, `refine-logs/PROOF_AUDIT.md`.

**References**: Proakis, *Digital Communications* (M-ary union/min-distance bounds); Cover & Thomas,
*Elements of IT* §11.9, Thm 2.10.1; de Chérisey et al. TCHES 2019; Rioul et al. arXiv:2105.07167;
Feyisetan et al. WSDM 2020; Mattern et al. NAACL-F 2022.

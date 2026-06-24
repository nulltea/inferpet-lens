# Proof Package — Depth-Inversion Certificate Lemma (resid-depth-inversion, Task 4)

## Claim

Let a token $t$ at a fixed position have residual-stream representation
$h_\ell = f_\ell(\text{context})$ at depth $\ell$, where $f_\ell$ is the deterministic
forward map of a public-weight transformer (threat model WEIGHTS-PUB). A retrieval inverter
$g$, trained on a disjoint training vocabulary, outputs $\hat t = g(h_\ell)$ by cosine match
to the public embedding table over a candidate pool $\mathcal{P}$ of size $K$. Let
$a_\ell = \Pr(\hat t = t)$ be the **population** top-1 accuracy on a test set whose token vocabulary
$V_{te}$ is disjoint from the training vocabulary $V_{tr}$, and $a^0_\ell$ the accuracy of the
label-shuffle control. Define selectivity $s_\ell = a_\ell - a^0_\ell$.

**(i) Population information certificate.** Let
$\Phi(a) = H(t) - H_b(1-a) - (1-a)\log(K-1)$ and let $a^\*$ be the unique root of $\Phi$ in
$[1/K, 1]$. Then for any $a_\ell \ge 1/K$,
$$ I(t; h_\ell) \;\ge\; \Phi(a_\ell) \;=\; H(t) - H_b(1-a_\ell) - (1-a_\ell)\log(K-1), $$
and this lower bound is **strictly positive whenever $a_\ell > a^\*$**. Under non-degeneracy
$0 < H(t) < \log K$ (A6) we have $a^\* > 1/K$ strictly; in the uniform special case $H(t)=\log K$,
$a^\* = 1/K$.

**(ii) Falsification of depth-irreversibility.**
- *(ii-a, operational, proved here.)* The hypothesis "the residual becomes irreversible to the input
  token beyond some depth $\ell^\dagger$" entails $a_\ell \to a^0_\ell$, i.e. $s_\ell \to 0$, for
  $\ell \ge \ell^\dagger$. Empirically $s_\ell \in [0.39, 0.69]$ with every bootstrap 95% CI excluding
  $0$ across all nine sampled depths, so $s_\ell \not\to 0$: the operational irreversibility hypothesis
  is falsified on Qwen3-4B `resid_post` over the sampled grid.
- *(ii-b, information-certified, conditional.)* The *strict $I(t;h_\ell)>0$* certificate at depth
  $\ell$ additionally requires the lower confidence bound of $a_\ell$ to exceed $a^\*$ (Corollary 1).
  This per-depth numeric check is queued, not discharged here.

## Status

**PROVABLE AFTER WEAKENING.** Corrections vs the original target: (1) the positivity threshold is
$a^\*$, with $a^\*>1/K$ strictly **only under non-degeneracy** $H(t)<\log K$ (it equals $1/K$ in the
uniform case) — the original "positive whenever $a>1/K$" is false for non-uniform $t$; (2) the bound
is a **population** statement, with an explicit finite-sample corollary; (3) part (ii) is split into
an operational falsification (proved) and an information-certified version (conditional on the
queued $a_\ell>a^\*$ check). Lemma 1 (Fano) and the Step-2 monotonicity hold as written.

## Assumptions

- **A1 (deterministic forward map).** $h_\ell = f_\ell(c)$ is deterministic in the context $c$;
  weights are public (WEIGHTS-PUB).
- **A2 (Markov chain at test time).** The inverter $g$ is fixed before observing the test label, so
  $\hat t = g(h_\ell)$ depends on $t$ only through $h_\ell$: $t \to h_\ell \to \hat t$.
- **A3 (vocab-disjoint split).** $V_{te}\cap V_{tr}=\varnothing$: no test token id appears in training.
- **A4 (finite pool).** $t \in \mathcal P$, $K = |\mathcal P| \ge 2$, and $\hat t \in \mathcal P$.
- **A5 (shuffle control).** Under the shuffle control the pairing $(h_\ell,t)$ is replaced by an
  independent label permutation, so the trained predictor sees $h_\ell \perp t$; its accuracy
  $a^0_\ell$ is the accuracy of *this specific* prior-only (frequency) baseline.
- **A6 (non-degeneracy).** $0 < H(t) \le \log K$ (with $<\log K$ for the strict $a^\*>1/K$ separation).
  If $H(t)=0$ ($t$ deterministic) then $I(t;h_\ell)=0$ trivially and the certificate is vacuous.

## Notation

- $H(\cdot)$: Shannon entropy (bits, $\log=\log_2$); $I(\cdot;\cdot)$: mutual information.
- $H_b(p) = -p\log p - (1-p)\log(1-p)$: binary entropy.
- $P_e = \Pr(\hat t \ne t) = 1 - a_\ell$: top-1 error.
- $a^\*$: the unique root of $\Phi(a)=H(t)-H_b(1-a)-(1-a)\log(K-1)$ in $[1/K, 1]$.

## Proof Strategy

Direct. Fano's inequality on the test-time Markov chain $t\to h_\ell\to\hat t$ lower-bounds
$I(t;h_\ell)$ by $\Phi(a_\ell)$; monotonicity of $\Phi$ on $[1/K,1]$ locates the positivity threshold
$a^\*$. A channel-decomposition argument (A3 + A5) attributes selectivity $s_\ell>0$ to the
representation. A finite-sample corollary lifts the population bound to data. Part (ii) reads the
per-depth empirics against these statements.

## Dependency Map

1. Bound (i) ← Fano (Lemma 1) + monotonicity/unique-root (Step 2).
2. Lemma 1 ← A2 (Markov), A4 (finite $K$).
3. $a^\*$ well-defined and $a^\*>1/K$ ← Step 2 + A6.
4. Channel attribution of $s_\ell$ ← A3 (kills exact-id memorization) + A5 (defines the prior baseline).
5. Corollary 1 (empirical) ← (i) + a one-sided confidence lower bound on $a_\ell$.
6. Part (ii-a) ← per-depth $s_\ell$ CI $>0$; Part (ii-b) ← Corollary 1 applied per depth (queued).

## Proof

**Lemma 1 (Fano).** Under A2 and A4, with $P_e = \Pr(\hat t \ne t)$,
$$ H(t \mid h_\ell) \;\le\; H_b(P_e) + P_e \log(K-1). $$
*Proof.* Standard Fano. Let $E = \mathbf 1[\hat t \ne t]$. By the chain rule applied two ways,
$H(t\mid\hat t) + H(E\mid t,\hat t) = H(E\mid\hat t) + H(t\mid E,\hat t)$. Here
$H(E\mid t,\hat t)=0$ since $E$ is a deterministic function of $(t,\hat t)$, so
$H(t\mid\hat t) = H(E\mid\hat t) + H(t\mid E,\hat t)$. Now $H(E\mid\hat t)\le H(E)=H_b(P_e)$
(conditioning does not increase entropy; $E$ binary). For the second term,
$$H(t\mid E,\hat t) = \Pr(E{=}0)\,H(t\mid \hat t, E{=}0) + \Pr(E{=}1)\,H(t\mid \hat t, E{=}1).$$
On $E=0$, $t=\hat t$ so $H(t\mid\hat t,E{=}0)=0$. On $E=1$, $t$ ranges over the $K-1$ pool elements
$\ne\hat t$, so $H(t\mid\hat t,E{=}1)\le \log(K-1)$ (this is valid for all $K\ge2$, with the
convention $\log(K-1)=0$ at $K=2$). Hence $H(t\mid E,\hat t)\le P_e\log(K-1)$, giving
$H(t\mid\hat t)\le H_b(P_e)+P_e\log(K-1)$. By A2 the chain $t\to h_\ell\to\hat t$ and the
data-processing inequality give $I(t;\hat t)\le I(t;h_\ell)$, equivalently
$H(t\mid h_\ell)\le H(t\mid\hat t)$. $\square$

**Step 1 (information lower bound).** $I(t;h_\ell) = H(t) - H(t\mid h_\ell)$. Substituting Lemma 1
and $P_e=1-a_\ell$,
$$ I(t;h_\ell) \;\ge\; H(t) - H_b(1-a_\ell) - (1-a_\ell)\log(K-1) \;=\; \Phi(a_\ell). \tag{1} $$
This holds for the trained inverter's accuracy $a_\ell$ (Lemma 1 holds for any estimator forming the
chain, A2). **For $a_\ell \ge 1/K$**, a larger $a_\ell$ only increases the RHS (Step 2 monotonicity);
below chance the RHS is not monotone, but the empirical regime has $a_\ell\gg1/K$.

**Step 2 (monotonicity, threshold $a^\*$, endpoints).** Write $\Phi(a) = H(t) - R(1-a)$ with
$R(P_e) = H_b(P_e)+P_e\log(K-1)$. Then
$\frac{dR}{dP_e} = \log\frac{1-P_e}{P_e}+\log(K-1) = \log\frac{(1-P_e)(K-1)}{P_e}$, which is $>0$
iff $(1-P_e)(K-1)>P_e$, i.e. $P_e<1-1/K$. So $R$ strictly increases on $P_e\in[0,1-1/K)$ and attains
its maximum $R(1-1/K)=\log K$ there. Equivalently $\Phi$ is strictly increasing in $a$ on $[1/K,1]$,
with **endpoints** $\Phi(1/K)=H(t)-\log K$ and $\Phi(1)=H(t)$.
- If $0<H(t)<\log K$ (A6): $\Phi(1/K)<0<\Phi(1)$, so by strict monotonicity and the intermediate
  value theorem there is a **unique** root $a^\*\in(1/K,1)$, and $\Phi(a)>0 \iff a>a^\*$. Hence
  $a^\*>1/K$ strictly.
- Boundary case $H(t)=\log K$ (uniform $t$): $\Phi(1/K)=0$, so the root is at $a^\*=1/K$ and
  $\Phi(a)>0$ for all $a>1/K$. This is exactly the (only) regime where the original "positive whenever
  $a>1/K$" holds.
Thus whenever the population accuracy $a_\ell>a^\*$, (1) gives $I(t;h_\ell)>0$. $\square$

**Step 3 (channel decomposition: $s_\ell>0$ isolates the representation channel).** Under A3 a
strategy that *memorizes* training id↔embedding pairs has no information about an unseen test id
beyond what the label marginal supplies; formally, restricted to $V_{te}$ such a strategy is
measurable w.r.t. the prior alone, so its accuracy is at most that of the best prior-only predictor.
The shuffle control (A5) instantiates a prior-only predictor with accuracy $a^0_\ell$. Therefore any
test predictor that does **not** read $h_\ell$ — whether by memorization (dead on unseen ids, A3) or
by frequency — achieves accuracy $\le a^0_\ell$ up to the optimality of the shuffle baseline. The
hypothesis $s_\ell = a_\ell - a^0_\ell > 0$ (CI excluding 0) thus certifies that $h_\ell$ contributes
predictive signal *beyond the measured shuffle-control prior channel and beyond exact train-vocabulary
memorization*. (Scope: this is relative to the implemented shuffle baseline; we do not claim $a^0_\ell$
is the Bayes-optimal prior-only accuracy. The empirical witness for the memorization half is the
cosine-NN baseline `nn`, whose selectivity is $0.000$ at every depth.) This step attributes the
*channel*; it does not by itself produce the bits in (1) — that is Steps 1–2.

**Corollary 1 (finite-sample certificate).** Let $\hat a_\ell$ be the empirical accuracy over $n$
i.i.d. test rows, $\underline a_\ell$ a one-sided $(1-\delta_a)$ lower confidence bound for $a_\ell$
(e.g. Clopper–Pearson), and $\underline H$ a $(1-\delta_H)$ lower bound for the test-pool entropy
$H(t)$. Define the **conservative** bound functional
$\underline\Phi(a) = \underline H - H_b(1-a) - (1-a)\log(K-1)$; since $\underline\Phi(a)\le\Phi(a)$
(as $\underline H\le H(t)$) and $\Phi$ is increasing on $[1/K,1]$, for $\underline a_\ell\ge1/K$,
$$ I(t;h_\ell) \;\ge\; \Phi(a_\ell) \;\ge\; \Phi(\underline a_\ell) \;\ge\; \underline\Phi(\underline a_\ell), $$
which holds with probability $\ge 1-\delta_a-\delta_H$ (union bound). **Certify $I(t;h_\ell)>0$ iff
$\underline\Phi(\underline a_\ell)>0$.** Replacing the unknown $H(t)$ by the *upper* bound $\log K$ is
**not** valid — it would inflate $\Phi$ and could falsely certify positivity for low-entropy $t$; a
lower bound $\underline H$ is required for a sound certificate. *Note:* the headline selectivities
$s_\ell$ are shuffle-subtracted; the raw accuracy $a_\ell=s_\ell+a^0_\ell$ is the quantity entering (1).

**Step 5 (part (ii)).** *(ii-a)* The depth-irreversibility hypothesis asserts
$\exists\,\ell^\dagger:\ \forall\ell\ge\ell^\dagger,\ a_\ell\to a^0_\ell$, i.e. $s_\ell\to0$. By the
per-depth bootstrap CIs, $s_\ell\in[0.39,0.69]$ with every CI excluding $0$ across
$\ell\in\{0,4,\dots,32\}$ (lowest L32 ridge $[0.341,0.438]$). So $s_\ell$ does not vanish at any
sampled depth: the operational hypothesis is falsified on this model/surface/grid. By Step 3 the
non-vanishing $s_\ell$ is genuine generalizing recovery (not memorization, not prior). *(ii-b)* The
strict information-certificate version — $I(t;h_\ell)>0$ at depth $\ell$ — follows from Corollary 1
at each depth where $\underline\Phi(\underline a_\ell)>0$; this per-depth check is queued (Open Risks), so (ii-b)
is stated conditionally, not as an established per-depth bits result. $\blacksquare$

## Corrections or Missing Assumptions

- **Threshold corrected (THR-01/THR-02).** Positivity holds for $a_\ell>a^\*$ where $a^\*$ is the root
  of $\Phi$ in $[1/K,1]$; $a^\*>1/K$ strictly only under A6 ($H(t)<\log K$) and $a^\*=1/K$ in the
  uniform case. Non-degeneracy $0<H(t)$ is now an explicit assumption (A6).
- **Population vs empirical (EMP-01).** (1) is a population bound; the finite-sample certificate is
  Corollary 1 via a one-sided CI on $a_\ell$.
- **Conservative entropy (EMP/F-01).** Corollary 1 certifies via $\underline\Phi$ built from a
  *lower* bound $\underline H\le H(t)$; using the upper bound $\log K$ is unsound (it inflates the
  lower bound on $I$ and can falsely certify low-entropy $t$).
- **Channel attribution narrowed (MEM-01/PRIOR-01).** Step 3 excludes *exact train-vocabulary*
  memorization (A3) and the *measured* shuffle prior baseline (A5); it does not claim $a^0_\ell$ is
  Bayes-optimal over all prior-only predictors.
- **Monotonicity scope (INEQ-01).** "Larger $a$ tightens (1)" is asserted only for $a\ge1/K$.
- **Part (ii) split (STEP5-01).** (ii-a) operational falsification is proved from $s_\ell$ CIs;
  (ii-b) information-certified per-depth positivity is conditional on the queued $a_\ell>a^\*$ check.

## Open Risks

- $K$ (effective candidate-pool size) and a *lower bound* $\underline H$ on $H(t)$ must be
  instantiated to compute the numeric $a^\*$ and discharge (ii-b) via $\underline\Phi$; the bound is
  otherwise parametric in $(K,H(t))$. The unknown $H(t)$ must be lower-bounded, never replaced by
  $\log K$ (Corollary 1).
- Fano is loose, so $I(t;h_\ell)$ is **at least as large as** the bound (1); equality can occur. This
  conservatism strengthens, not weakens, the falsification direction.
- $a_\ell$, $a^0_\ell$ are estimated over 413 test rows/layer, single seed; Corollary 1's CI is over
  rows, not seeds. Same scope caveat as the empirical claim.

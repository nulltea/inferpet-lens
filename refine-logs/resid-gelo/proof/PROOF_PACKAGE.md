# Proof Package — GELO structural leakage (resid-gelo, Task 5)

## Claim

Let the residual operand be $H \in \mathbb{R}^{n \times d}$ ($n$ token rows, feature dimension
$d$). GELO (arXiv 2603.05035) exposes $U = A\,\tilde H$ where $A$ is a secret row-mixing matrix and
$\tilde H$ is either $H$ (no shields) or the row-augmented operand $\tilde H = \begin{bmatrix} H \\
S \end{bmatrix} \in \mathbb{R}^{(n+m)\times d}$ with $m$ appended shield rows $S \in
\mathbb{R}^{m\times d}$ (then $A \in \mathbb{R}^{(n+m)\times(n+m)}$). The adversary observes only
$U$ (threat model WEIGHTS-PUB).

- **L1 (feature-Gram leak).** If $A$ is orthogonal then the feature (column) Gram of the exposed
  operand is exactly invariant: $U^\top U = \tilde H^\top A^\top A\, \tilde H = \tilde H^\top \tilde
  H$. With no shields ($m=0$) this equals $H^\top H$ exactly; with shields it equals the augmented
  Gram $H^\top H + S^\top S$, which equals $H^\top H$ iff $S=0$ (real rows). For $m\ge1$ with
  nonzero shields the leak is of the augmented Gram; $H^\top H$ is recoverable from it iff $S^\top S$
  is known/subtractable.
- **L2 (row-Gram conjugation, not invariance).** If $A$ is orthogonal then the row Gram $UU^\top =
  A\,(\tilde H \tilde H^\top)\,A^\top$ is an orthogonal similarity of $G:=\tilde H\tilde H^\top$;
  hence $\mathrm{spec}(UU^\top)=\mathrm{spec}(G)$ and $\|UU^\top\|_F=\|G\|_F$ exactly, but
  $UU^\top \ne G$ in general (equality iff $A$ commutes with $G$).
- **L3 (rowspace preservation).** If $A$ is invertible then $\operatorname{rowspace}(U) =
  \operatorname{rowspace}(\tilde H)$; with $m=0$, $\operatorname{rowspace}(U)=\operatorname{rowspace}(H)$
  (assuming $\tilde H\ne0$ so the rowspace and normalized alignments are well-defined). Hence the
  set of directions any key-free left-demixing attack can produce is confined to
  $\operatorname{rowspace}(\tilde H)$ for any invertible $A$. **Floor-invariance (orthogonal $A$
  only):** when $A$ is orthogonal the recoverable directions, the per-target oracle alignment
  ceiling, and a *rowspace-uniform* random-demixing chance floor are unchanged by $A$ for fixed
  $\tilde H$ (a non-orthogonal $A$ re-weights the row-coordinate metric and can shift the floor).
  Empirically this caveat is moot: the matched floor is **recomputed on each observed $U$** (same
  whitening of that $U$, then a random rotation instead of joint-diag), so it is matched to that
  specific $U$ by construction at every $\kappa$ — the genuine margin (recovery $-$ floor) is
  well-posed for non-orthogonal $A$ even though cross-$\kappa$ floor *invariance* is not claimed.
- **L4 (fresh-per-prompt $A$ defeats amortized linear inversion).** Let prompts $t$ have operands
  $H_t$ and mixings $A_t$ with $A_t$ i.i.d., $\mathbb{E}[A_t]=0$ (e.g. Haar on $O(n)$, or
  $A_t=Q_1\,\mathrm{diag}(s)\,Q_2$ with $Q_1$ Haar-orthogonal independent of the rest, as in the
  experiment's `make_mixing`), and $A_t$ independent of $H_t$, with $\mathbb{E}\|H_t\|_F^2<\infty$.
  Observe $U_t=A_tH_t$. Then $W=0$ is a global minimizer of the population least-squares (and ridge,
  any $\lambda\ge0$) objective $\min_W \mathbb{E}\|U_tW-H_t\|_F^2(+\lambda\|W\|_F^2)$ over *fixed
  no-intercept* linear maps $W\in\mathbb{R}^{d\times d}$; it is the *unique* minimizer when
  $M+\lambda I\succ0$ (with $M:=\mathbb{E}[U_t^\top U_t]\succeq0$), and any other minimizer $W'$
  satisfies $U_tW'=0$ a.s.. Consequently **no fixed no-intercept linear map improves population
  squared loss over the zero predictor**: the best amortized linear inverter carries no
  prompt-specific information. (Its empirical failure is a correct negative control, not evidence the
  mixing is broken; this says nothing about per-prompt BSS or nonlinear/intercept attacks — see
  Open Risks.)

## Status

**PROVABLE AS STATED (after scope tightening).** L1, L2 are exact finite-dimensional identities.
L3's rowspace identity is exact for invertible $A$; its floor-*invariance* corollary is stated for
orthogonal $A$ only (the empirical floor is recomputed per $U$, so cross-$\kappa$ invariance is not
needed). L4 is proved at the population least-squares/ridge level: $W=0$ is a global minimizer over
no-intercept linear maps and no such map beats the zero predictor; uniqueness needs $M+\lambda
I\succ0$. Caveats on what L4 does *not* claim are in Open Risks.

## Assumptions

- L1, L2: $A$ orthogonal, $A^\top A = AA^\top = I$ (the size is $n$ without shields, $n+m$ with).
- L3: rowspace identity needs only $A$ invertible and $\tilde H\ne0$; the floor-invariance corollary
  additionally needs $A$ orthogonal.
- L4: $\{A_t\}$ i.i.d. with $\mathbb{E}[A_t]=0$ and $A_t \perp H_t$; second moments
  $\mathbb{E}\|H_t\|_F^2<\infty$ and $\mathbb{E}\|A_tH_t\|_F^2<\infty$ finite (the latter is automatic
  for Haar-$O(n)$ and the bounded fixed-condition `make_mixing` mixings, where $\|A_t\|_2$ is
  bounded); predictor class is fixed no-intercept linear maps
  $W\in\mathbb{R}^{d\times d}$. $\mathbb{E}[A_t]=0$ holds for Haar measure on $O(n)$ (any
  $n\ge1$) and on $SO(n)$ for $n\ge 2$.
- Real field throughout; all matrices have finite entries.

## Notation

- $X^\top$ transpose; $\operatorname{rowspace}(X)=\operatorname{span}$ of the rows of $X$ (a
  subspace of $\mathbb{R}^d$); $\mathrm{spec}(\cdot)$ multiset of eigenvalues; $\|\cdot\|_F$
  Frobenius norm; $\langle X,Y\rangle = \operatorname{tr}(X^\top Y)$ the Frobenius inner product.
- $G := \tilde H \tilde H^\top$ (the row Gram of the exposed operand).

## Proof Strategy

Direct computation for L1–L2; an image/rank argument for L3; a first-/second-moment (normal-equation)
optimisation argument for L4. No external theorems beyond elementary linear algebra and the
invariance of Haar measure under a fixed group element.

## Dependency Map

1. L1 depends only on $A^\top A = I$ and associativity of matrix products.
2. L2 depends only on $AA^\top=I$ (orthogonal similarity preserves spectrum; trace/Frobenius are
   similarity invariants) plus a commutation counterexample for non-invariance.
3. L3 depends on: $A$ invertible $\Rightarrow$ left-multiplication is a bijection of column space of
   $\tilde H^\top$, equivalently $\operatorname{rowspace}(A\tilde H)=\operatorname{rowspace}(\tilde H)$.
4. L4 depends on: linearity of expectation, $\mathbb{E}[A_t]=0$ to kill the cross-covariance, and
   convexity of the (ridge-regularised) quadratic objective so the stationary point is the global
   minimiser. $\mathbb{E}[A_t]=0$ itself follows from Haar invariance under a reflection in $O(n)$.

## Proof

### Step 1 — L1 (feature-Gram leak).

By definition $U = A\tilde H$, so
$$U^\top U = (A\tilde H)^\top (A\tilde H) = \tilde H^\top A^\top A\, \tilde H.$$
If $A$ is orthogonal, $A^\top A = I$, hence $U^\top U = \tilde H^\top \tilde H$ exactly. The product
is associative and no approximation is used, so the identity is exact in exact arithmetic.

*No shields ($\tilde H = H$, $m=0$):* $U^\top U = H^\top H$. The adversary, observing $U$, computes
$U^\top U$ and obtains the exact $d\times d$ secret feature Gram $H^\top H$ with no attack — an
attack-independent functional of the secret.

*Shields ($\tilde H=\begin{bmatrix}H\\S\end{bmatrix}$):* block multiplication gives
$$\tilde H^\top \tilde H = \begin{bmatrix}H^\top & S^\top\end{bmatrix}\begin{bmatrix}H\\S\end{bmatrix}
= H^\top H + S^\top S.$$
Thus $U^\top U = H^\top H + S^\top S$. The *identity* $U^\top U = H^\top H$ holds iff $S^\top S = 0$,
i.e. $S=0$ (real rows); when $S\ne0$ the exposed object is $H^\top H + S^\top S$, from which $H^\top
H$ is exactly *recoverable* iff $S^\top S$ is known/subtractable. For $m\ge1$ with unknown $S$ the leak is of
the *augmented* Gram, not $H^\top H$ exactly. $\square$ (L1)

### Step 2 — L2 (row-Gram conjugation, not invariance).

$$UU^\top = (A\tilde H)(A\tilde H)^\top = A\,\tilde H \tilde H^\top A^\top = A\,G\,A^\top.$$
With $A$ orthogonal, $A^{-1}=A^\top$, so $UU^\top = A G A^{-1}$ is a *similarity transform* of $G$.
Similar matrices have identical characteristic polynomials, hence $\mathrm{spec}(UU^\top) =
\mathrm{spec}(G)$; in particular the eigenvalue multiset, $\operatorname{tr}(UU^\top) =
\operatorname{tr}(G)$, and (since $UU^\top$ and $G$ are symmetric with the same eigenvalues)
$\|UU^\top\|_F^2 = \sum_i \lambda_i^2 = \|G\|_F^2$ are all preserved.

Non-invariance: $UU^\top = G$ for all such $A$ would require $AG=GA$, i.e. $A$ commutes with $G$.
A generic orthogonal $A$ does not. Concrete counterexample: $G=\mathrm{diag}(2,1)$, $A=\begin{bmatrix}
0&1\\1&0\end{bmatrix}$ (orthogonal); then $AGA^\top=\mathrm{diag}(1,2)\ne G$, while $\mathrm{spec}$
and $\|\cdot\|_F=\sqrt5$ are preserved. Hence the row Gram is hidden up to an orthogonal conjugation
(spectrum exposed, entries not), in contrast to the feature Gram which L1 exposes outright. $\square$ (L2)

### Step 3 — L3 (rowspace preservation $\Rightarrow$ recoverable set invariant).

$\operatorname{rowspace}(A\tilde H) = \{\,x^\top A\tilde H : x\in\mathbb{R}^{n+m}\,\} =
\{\,(A^\top x)^\top \tilde H : x\in\mathbb{R}^{n+m}\,\}$. As $x$ ranges over $\mathbb{R}^{n+m}$ and
$A$ is invertible, $A^\top x$ ranges over all of $\mathbb{R}^{n+m}$ (bijection), so the set equals
$\{\,y^\top\tilde H : y\in\mathbb{R}^{n+m}\,\} = \operatorname{rowspace}(\tilde H)$. Hence
$\operatorname{rowspace}(U)=\operatorname{rowspace}(\tilde H)$ (assuming $\tilde H\ne0$, so the
rowspace is nontrivial and normalized alignments to nonzero target rows are well-defined), and with
$m=0$, $\operatorname{rowspace}(U)=\operatorname{rowspace}(H)$.

Any key-free left-demixing attack outputs sources of the form $BU$ for some $B$ chosen from the
observation; each output row lies in $\operatorname{rowspace}(U)=\operatorname{rowspace}(\tilde H)$.
So the set of *producible directions* and the per-target oracle alignment ceiling over that set
depend only on $\operatorname{rowspace}(\tilde H)$ and the fixed nonzero target rows — not on which
invertible $A$ was used.

**Floor-invariance requires orthogonality.** The *value* a random demixing $B$ attains depends on
the row-coordinate metric of $U$, which a non-orthogonal $A$ changes. Concretely with $H=I_2$,
$A=\mathrm{diag}(k,1)$, a random unit row $x$ gives $x^\top A H=(k\cos\theta,\sin\theta)$, whose
alignment distribution with $e_1$ depends on $k$ — so the chance floor is *not* invariant across
non-orthogonal $A$. For **orthogonal** $A$, whitening removes the metric and the rowspace-uniform
floor is invariant. Empirically the point is moot: the matched floor is **recomputed on each
observed $U$** (same whitening of that $U$, then a random rotation in place of joint-diag), so it is
matched to that specific $U$ by construction at every $\kappa$, and the genuine margin
(recovery $-$ floor) is a well-posed within-cell contrast for non-orthogonal $A$ even though
cross-$\kappa$ floor invariance is not claimed. $\square$ (L3)

### Step 4 — L4 (fresh-per-prompt $A$ defeats amortized linear inversion).

First, $\mathbb{E}[A_t]=0$. Let $R=\mathrm{diag}(-1,1,\dots,1)\in O(n)$ (a reflection, $\det R=-1$,
$R\in O(n)$). Haar measure on $O(n)$ is invariant under left multiplication by $R$, so $A_t$ and
$RA_t$ are identically distributed, giving $\mathbb{E}[A_t]=\mathbb{E}[RA_t]=R\,\mathbb{E}[A_t]$.
Thus the first row of $\mathbb{E}[A_t]$ equals its own negation, hence is $0$; repeating with a
reflection on each coordinate gives $\mathbb{E}[A_t]=0$. (For $SO(n)$, $n\ge2$, use a rotation by
$\pi$ in two coordinates analogously.) For the experiment's $A_t=Q_1\,\mathrm{diag}(s)\,Q_2$ with
$Q_1$ Haar-orthogonal and independent of $(\mathrm{diag}(s),Q_2)$, linearity and independence give
$\mathbb{E}[A_t]=\mathbb{E}[Q_1]\,\mathbb{E}[\mathrm{diag}(s)\,Q_2]=0$ since $\mathbb{E}[Q_1]=0$ by
the reflection argument. So the hypothesis $\mathbb{E}[A_t]=0$ covers both the orthogonal ($\kappa=1$)
and the conditioned ($\kappa>1$) mixings used in the sweep.

Consider the population ridge objective over fixed $W\in\mathbb{R}^{d\times d}$, $\lambda\ge0$:
$$J(W)=\mathbb{E}\big\|U_tW-H_t\big\|_F^2 + \lambda\|W\|_F^2,\qquad U_t=A_tH_t.$$
Expand the data term:
$$\mathbb{E}\|U_tW-H_t\|_F^2 = \mathbb{E}\,\langle U_tW,U_tW\rangle - 2\,\mathbb{E}\,\langle U_tW,H_t\rangle
+ \mathbb{E}\|H_t\|_F^2.$$
The cross term: using $\langle U_tW,H_t\rangle = \operatorname{tr}(W^\top U_t^\top H_t)$ and
$U_t^\top H_t = H_t^\top A_t^\top H_t$,
$$\mathbb{E}\,\langle U_tW,H_t\rangle = \operatorname{tr}\!\big(W^\top\,\mathbb{E}[H_t^\top A_t^\top H_t]\big).$$
By independence $A_t\perp H_t$ and the tower rule, $\mathbb{E}[H_t^\top A_t^\top H_t] =
\mathbb{E}\big[H_t^\top\,\mathbb{E}[A_t^\top\mid H_t]\,H_t\big] = \mathbb{E}\big[H_t^\top
(\mathbb{E}[A_t])^\top H_t\big] = 0$, since $\mathbb{E}[A_t]=0$. Hence the cross term vanishes for
**every** $W$:
$$J(W) = \mathbb{E}\,\langle U_tW,U_tW\rangle + \lambda\|W\|_F^2 + \mathbb{E}\|H_t\|_F^2.$$
The first term is $\mathbb{E}\,\operatorname{tr}(W^\top U_t^\top U_t W)=\operatorname{tr}(W^\top
M W)$ with $M:=\mathbb{E}[U_t^\top U_t]=\mathbb{E}[H_t^\top A_t^\top A_t H_t]\succeq0$. (We do *not*
assume $A_t$ orthogonal here; $M$ is a PSD matrix in either case. When $A_t$ is orthogonal a.s.,
$A_t^\top A_t=I$ and $M=\mathbb{E}[H_t^\top H_t]$, but the argument below needs only $M\succeq0$.)
So
$$J(W)=\operatorname{tr}\!\big(W^\top(M+\lambda I)W\big) + \mathbb{E}\|H_t\|_F^2,$$
a sum of a PSD quadratic in $W$ and a constant. Hence $J(W)\ge \mathbb{E}\|H_t\|_F^2 = J(0)$ for
every $W$, so $W^\star=0$ is a **global minimizer**. It is the *unique* minimizer iff
$M+\lambda I\succ0$ (e.g. any $\lambda>0$, or $\lambda=0$ with $M\succ0$); when $M+\lambda I$ is
singular the minimizers are exactly $\{W: (M+\lambda I)W=0\}$, every such $W$ has
$\operatorname{tr}(W^\top(M+\lambda I)W)=0$, which at $\lambda=0$ forces $\mathbb{E}\|U_tW\|_F^2=0$,
i.e. $U_tW=0$ a.s. — a degenerate, equally uninformative predictor. In all cases the prediction at
the optimum is $0$ a.s. and the residual is $\mathbb{E}\|H_t\|_F^2$. Therefore **no fixed
no-intercept linear map improves population squared loss over the zero predictor**: the amortized
linear inverter carries no prompt-specific information. (This is a statement about the linear,
intercept-free class only; per-prompt BSS and nonlinear/intercept attacks are out of scope — Open
Risks.) $\square$ (L4)

Therefore L1–L4 all hold under their stated assumptions. ∎

## Corrections or Missing Assumptions

- **L1 scope correction (folded into the claim):** the exact *identity* $U^\top U = H^\top H$ holds
  only without shields ($S=0$); with shields the exposed object is the augmented Gram $H^\top H +
  S^\top S$, from which $H^\top H$ is exactly *recoverable* iff $S^\top S$ is known/subtractable. The
  original "exact $H^\top H$" wording would be an overclaim for $m\ge1$; the corrected statement is
  in the Claim.
- L4 was requested "at the rigor achievable". The clean, fully rigorous statement proved is about
  the **population least-squares/ridge optimum** over fixed *no-intercept* linear maps being the
  zero map (or an a.s.-zero predictor when degenerate); see Open Risks for the gap to finite-sample,
  to an intercept/constant predictor, and to nonlinear attacks. Note "zero predictor" $\ne$ "mean
  predictor" unless $\mathbb{E}[H]=0$: the L4 class excludes an intercept, so the comparison is
  strictly to the zero map.

## Open Risks

- **L4 is about linear amortized inversion only.** It shows no fixed *linear* $W$ helps; it does
  *not* claim per-prompt blind-source-separation fails (that is the empirical C1 question, where
  recovery sits just above the matched floor). A nonlinear amortized attack, or one that first
  estimates $A_t$ per prompt, is outside L4's scope.
- **Finite $T$.** The proof is at the population level; for finite prompts the empirical ridge
  solution is $W\approx0$ up to $O(1/\sqrt T)$ sampling noise, consistent with the measured held-out
  $p95=0.288$ sitting *below* the rowspace-respecting random-demix floor $0.667$ (the zero-ish
  predictor ignores the informative rowspace that L3 says the floor exploits).
- **L2 spectrum exposure.** L2 says the row-Gram *spectrum* is preserved under orthogonal $A$; this
  is itself a (weaker) leak channel, not used by the headline claim but worth noting.

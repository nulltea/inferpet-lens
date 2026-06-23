# Proof Package — perm-llr-threshold

## Claim
For the AloePri column-permutation embedding cover, (L1) the RowSort-64 quantile signature is
information-dominated by the full sorted row via the data-processing inequality, and (L2) on the
per-row column-permutation isotropic-Gaussian channel the full sorted-row negative-squared-Euclidean
matcher with Hungarian assignment is the generalized-likelihood (profile-MLE) / MAP permutation
matcher. A Remark delimits what changes under the actual *shared* column permutation; the
`I_XY ≥ 2 log n` exact-recovery threshold is cited, not re-derived.

## Status
PROVABLE AS STATED (L1 unconditional; L2 exact for the per-row-permutation channel in the
profile-likelihood / GLRT sense; the shared-permutation case is handled honestly in the Remark, not
overclaimed).

## Assumptions
- A1. Plaintext rows $x_1,\dots,x_N \in \mathbb R^d$ are fixed (or i.i.d. draws); $d \ge 1$.
- A2. The cover produces obfuscated rows by $y_{\Pi(i)} = P(x_i + e_i)$ with $P \in \mathcal S_d$ a
  permutation matrix on the $d$ feature coordinates, $e_i \sim \mathcal N(0,\sigma^2 I_d)$ i.i.d.
  across $i$ and independent of everything, and $\Pi$ a uniform-prior secret permutation on $N$ rows.
- A3. Signatures: $s(v) = \operatorname{sort}(v)$ (coordinates of $v$ in nondecreasing order);
  $\varphi(v) = Q(s(v))$ where $Q:\mathbb R^d \to \mathbb R^{64}$ is the fixed 64-quantile binning.
  Any fixed mean-centering + $L^2$-normalization is folded into $Q$ and into a fixed post-map of $s$;
  being deterministic, it does not affect either lemma.
- A4. (L2 only) Per-row channel: each row carries its **own** nuisance permutation $P_i$, treated as
  an unknown parameter maximized out (profile likelihood / GLRT). The shared-$P$ case is the Remark.

## Notation
$\mathcal S_d$ = symmetric group acting on coordinates by $(Pv)_k = v_{P^{-1}(k)}$; $\langle\cdot,
\cdot\rangle$ Euclidean inner product; $I(\cdot;\cdot)$ Shannon mutual information; orbit of $v$ under
$\mathcal S_d$ is $\{Pv : P\in\mathcal S_d\}$.

## Proof Strategy
L1: two applications of the DPI to a deterministic post-map on each side. L2: identify the maximal
invariant of the $\mathcal S_d$ action, then chain the profile Gaussian log-likelihood through the
rearrangement inequality to sorted neg-Euclidean, then observe one-to-one maximization is a linear
assignment (Hungarian). Remark: exhibit a shared-$P$ invariant the per-row statistic discards.

## Dependency Map
1. L1 ← data-processing inequality (Cover–Thomas Thm 2.8.1), applied twice.
2. L2 ← (a) orbit characterization of $\mathcal S_d$; (b) Gaussian density of a permutation-rotated
   isotropic noise; (c) rearrangement inequality (Hardy–Littlewood–Pólya); (d) one-to-one
   log-likelihood maximization = linear assignment problem.
3. Remark ← Gram matrix $G=XX^\top$ is invariant under a shared orthogonal/permutation $P$.
4. Threshold ← cited Dai–Cullina–Kiyavash.

## Proof

### Lemma 1 (DPI domination, unconditional)
By A3, $\varphi(v) = Q(s(v))$ with $Q$ deterministic (measurable). Let $A := s(X)$, $B := s(Y)$ for
any jointly distributed plaintext/obfuscated signature pair. Because $Q(A)$ is a deterministic
function of $A$, the chain $Q(A) - A - B$ is Markov, so the DPI gives
$$I(Q(A); B) \le I(A; B).$$
Because $Q(B)$ is a deterministic function of $B$, the chain $Q(B) - B - Q(A)$ is Markov, so the DPI
gives
$$I(Q(A); Q(B)) \le I(Q(A); B).$$
Chaining, $I(\varphi(X);\varphi(Y)) = I(Q(A);Q(B)) \le I(A;B) = I(s(X); s(Y))$. Thus the RowSort-64
signature carries no more pairwise (cross-table) information than the full sorted row, for **every**
joint law — in particular at every noise level $\sigma$. Any matcher that is a function of
$(\varphi(x_i),\varphi(y_j))$ is therefore information-dominated by one that is a function of
$(s(x_i),s(y_j))$. $\blacksquare$

*$Q$ injective on the support of $s$ is **sufficient** for equality (not necessary; equality can also
hold when $B$ depends on $A$ only through $Q(A)$). For a strict 64-bin quantization of a $d=2304$-dim
sorted row with $d\gg 64$, $Q$ is generically many-to-one, consistent with a strict gap and the
observed RowSort collapse while CLUB-on-$\varphi$ barely moves.*

### Lemma 2 (maximal invariant + profile-MLE = sorted neg-Euclidean; per-row channel)
**(i) Maximal invariant.** $s(Pv)=s(v)$ for all $P\in\mathcal S_d$, so $s$ is $\mathcal S_d$-invariant.
Conversely, $s(u)=s(v)$ iff $u,v$ have the same multiset of coordinates iff $u=Pv$ for some
$P\in\mathcal S_d$, i.e. iff they share an orbit. Hence $s$ separates orbits and is the **maximal
invariant** of the $\mathcal S_d$ action.

**(ii) Per-row likelihood.** Under A2/A4, for a candidate pairing of obfuscated row $y$ with plaintext
row $x$ and nuisance $P\in\mathcal S_d$, $y = Px + Pe$ with $Pe \sim \mathcal N(0,\sigma^2 PP^\top) =
\mathcal N(0,\sigma^2 I_d)$ because a permutation matrix is orthogonal ($PP^\top=I$). Thus
$$p(y\mid x,P) = (2\pi\sigma^2)^{-d/2}\exp\!\Big(-\tfrac{1}{2\sigma^2}\lVert y - Px\rVert^2\Big).$$

**(iii) Profile over the nuisance via rearrangement.** Maximizing the density over $P$ minimizes
$\lVert y-Px\rVert^2 = \lVert y\rVert^2 + \lVert x\rVert^2 - 2\langle y, Px\rangle$ (using
$\lVert Px\rVert=\lVert x\rVert$). So $\min_P \lVert y-Px\rVert^2 \iff \max_P \langle y,Px\rangle =
\max_{P}\sum_{k} y_k x_{P^{-1}(k)}$. By the **rearrangement inequality** (Hardy–Littlewood–Pólya), a
sum $\sum_k a_k b_{\pi(k)}$ over permutations $\pi$ is maximized when $a$ and $b$ are equally
ordered, with maximum $\sum_k s(a)_k s(b)_k = \langle s(y), s(x)\rangle$. Therefore
$$\min_{P\in\mathcal S_d}\lVert y - Px\rVert^2 = \lVert y\rVert^2 + \lVert x\rVert^2 - 2\langle s(y),s(x)\rangle = \lVert s(y) - s(x)\rVert^2,$$
the last equality using $\lVert s(y)\rVert=\lVert y\rVert$, $\lVert s(x)\rVert=\lVert x\rVert$. Hence
the **profile log-likelihood** of the pairing is $-\tfrac{1}{2\sigma^2}\lVert s(y)-s(x)\rVert^2$ plus a
pairing-independent constant.

**(iv) Assignment.** With per-row independence (A2, A4) and a uniform prior on $\Pi$, the joint
profile log-likelihood of a one-to-one assignment $\rho$ (obf $j \mapsto$ plain $\rho(j)$) is
$-\tfrac{1}{2\sigma^2}\sum_j \lVert s(y_j)-s(x_{\rho(j)})\rVert^2 + \text{const}$. Maximizing over
one-to-one $\rho$ is the linear assignment problem with cost $c_{j,i}=\lVert s(y_j)-s(x_i)\rVert^2$,
solved exactly by the Hungarian algorithm (an optimizer; coordinate ties may make the optimizing
permutation/matching non-unique, but the value identity in (iii) is unaffected). Thus **full-sort
negative-squared-Euclidean + Hungarian is the profile-MLE (generalized-likelihood) matcher —
equivalently the joint MAP over $(\Pi,\{P_i\})$ under uniform priors — for the per-row
column-permutation isotropic-Gaussian channel.** It is **not** the marginal MAP over $\Pi$ alone,
which would log-sum-exp over the nuisance permutations rather than maximize over them. $\blacksquare$

*(Cosine on mean-centered, $L^2$-normalized sorted rows induces the same ranking as
neg-Euclidean on those rows, since $\lVert a-b\rVert^2 = 2(1-\langle a,b\rangle)$ for unit vectors —
matching the code's numerically identical `fullsort_cos` and `fullsort_euc`.)*

### Remark (shared-permutation gap — what is and is not proven)
The actual AloePri cover (`aloepri.py:132`, `noisy[:, col]`) applies a **single** $P$ to all rows. Two
points:
1. **L1 is unaffected** — it is a statement about the deterministic map $Q$ on marginals and holds for
   any joint law, shared $P$ included.
2. **L2's optimality is for the per-row model only.** Under a shared $P$ the joint law couples rows:
   $p(\{y_j\}\mid\{x_i\},\Pi,P)=\prod_j \mathcal N(y_j; P x_{\Pi^{-1}(j)},\sigma^2 I)$ with one $P$.
   The per-row sorted statistic discards information that a shared $P$ preserves — e.g. the Gram
   matrix $YY^\top = P(X{+}E)(X{+}E)^\top P^\top$ has $\,YY^\top = (X{+}E)(X{+}E)^\top$ entrywise up to
   the row relabeling $\Pi$ because $P^\top P = I$, so pairwise row inner products are a shared-$P$
   invariant **not** recoverable from per-row multisets alone. Hence per-row $s$ is $\mathcal
   S_d$-invariant and **sound** (and still DPI-dominates RowSort), but it is **not the joint maximal
   invariant**, and the sorted matcher is a relaxation/lower bound, **not** proven to be the exact
   joint MAP for the shared channel. This is the scope boundary; the empirical near-perfect recovery
   at small $\sigma$ is consistent with per-row multisets being near-unique fingerprints, not with a
   shared-channel optimality theorem.

### Cited threshold (external benchmark, not re-derived, not implied by L1/L2)
In the Gaussian database-alignment model of Dai, Cullina, and Kiyavash (*Database Alignment with
Gaussian Features*, arXiv:1903.01422), exact permutation recovery is achievable at feature mutual
information $I_{XY} \ge 2\log n + \omega(1)$, with a matching converse under their
canonical/asymptotic assumptions. We cite this as an external order/threshold benchmark for the
sorted-vector features; it is **not** a consequence of Lemmas 1–2 and does **not** automatically
transfer to the quantized $\varphi$ or to the shared-$P$ AloePri channel (whose exact threshold is
out of scope here).

## Corrections or Missing Assumptions
- The L2 result is stated as **profile-MLE / GLRT** optimality (nuisance $P$ maximized, not
  marginalized). This matches the claim's wording ("profile-MLE") and the implemented matcher; full
  marginal-likelihood optimality is neither claimed nor needed.

## Open Risks
- The $2\log n$ constant is imported under its source's jointly-Gaussian model; transfer to the
  sorted-feature representation is at the level of order/threshold, not an exact constant for this
  representation.
- Shared-$P$ exact MAP is explicitly out of scope (Remark); a Gram-/cross-row-aware matcher could in
  principle dominate per-row sort on the shared channel — a named follow-up, not a defect of L1/L2.

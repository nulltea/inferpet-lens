# Proof Package — channel-selectivity via critical-scale separation

## Claim (corrected, after Round-1 review)

Under $Y = X + bZ$ with **i.i.d. Gaussian** noise $Z$ and scale $b>0$, with $\sigma$ drawn **uniformly**
from $S_N$, there exist critical scales $b_\Pi=\Theta(d_\Pi)$ and $b_{\mathrm{tok}}=\Theta(d_{\mathrm{tok}}/\sqrt{\ln K})$
(for $K\ge 3$; $b_{\mathrm{tok}}=+\infty$ for $K=2$) such that:
(i) for $b\ge b_\Pi(\eta,M)$ with any target $\eta\in(2^{-M},1)$, the **exact-permutation recovery
probability** is $\le\eta$ — and $\le 2^{-M}(1+o(1))$ as $b\to\infty$, so $\to 0$ when the number $M$ of
disjoint near-pairs $\to\infty$; the closest-pair Bayes error $\to 1/2$;
(ii) for $b\le b_{\mathrm{tok}}$ the token-id MAP error is $<1/2$.
Hence, **when $d_\Pi\sqrt{\ln(K-1)}\ll d_{\mathrm{tok}}$** (in particular for fixed $K$ with $d_\Pi\ll d_{\mathrm{tok}}$),
the band $(b_\Pi,b_{\mathrm{tok}})$ is nonempty: $\Pi$ is destroyed while token-id survives.

## Status

**PROVABLE AS STATED (for the corrected Gaussian statement).** The corrected statement above is **proved as stated** for Gaussian noise (Round-2 review: Codex gpt-5.5 xhigh, verdict PASS).
Round-1 review (Codex gpt-5.5 xhigh) forced four substantive corrections, all incorporated: (1) a
**uniform prior** on $\sigma$; (2) the binary permutation-pair Bayes error is $Q(\delta/(\sqrt2 b))$, not
$Q(\delta/(2b))$; (3) the product bound tends to $2^{-M}$ for fixed $M$, so full destruction needs
$M\to\infty$ and the per-$\eta$ threshold needs $\eta>2^{-M}$; (4) the band condition is
$d_\Pi\sqrt{\ln(K-1)}\ll d_{\mathrm{tok}}$. The Laplace case and the fraction-correct collapse are demoted
to clearly-labelled **heuristic remarks** (R2, R3), not part of the proved theorem.

## Assumptions

- **A1 (model).** Token-id: codebook $\mathcal C=\{c_1,\dots,c_K\}\subset\mathbb R^D$, min distance
  $d_{\mathrm{tok}}=\min_{a\ne a'}\lVert c_a-c_{a'}\rVert>0$; secret = index $a^\*$ of the clean point
  $x=c_{a^\*}$. Permutation: $N$ clean points $\{x_1,\dots,x_N\}$, min gap
  $d_\Pi=\min_{i\ne j}\lVert x_i-x_j\rVert>0$; secret = $\sigma$ with observations $\{y_i=x_{\sigma(i)}+W_i\}$.
- **A2 (noise, Gaussian).** $W=bZ$ with $Z$ having i.i.d. $\mathcal N(0,1)$ coordinates; the $W_i$ across
  points are independent; $b>0$. (Laplace is treated only heuristically in Remark R3.)
- **A3 (prior + decoders).** Token-id: uniform prior on $a^\*$, NN/MAP decode
  $\hat a=\arg\min_a\lVert Y-c_a\rVert$. Permutation: **$\sigma\sim\mathrm{Unif}(S_N)$**, and the adversary
  outputs any measurable $\hat\sigma(\{y_i\})$ (the lower bound holds for the Bayes-optimal decoder, hence
  for VMA/Hungarian).
- **A4 (regime).** $d_\Pi\sqrt{\ln(K-1)}\ll d_{\mathrm{tok}}$, and there exist $M\ge 1$ **disjoint** clean-point
  pairs $\{(i_m,j_m)\}_{m=1}^M$ with $\delta_m:=\lVert x_{i_m}-x_{j_m}\rVert\le\rho$, $\rho=\Theta(d_\Pi)$.

## Notation

$Q(t)=\Pr[\mathcal N(0,1)>t]$, $Q^{-1}$ its inverse on $(0,1/2)$. For unit $u$, under A2
$\langle W,u\rangle\sim\mathcal N(0,b^2)$. $\delta_a=\lVert c_a-c_{a^\*}\rVert\ge d_{\mathrm{tok}}$,
$u_a=(c_a-c_{a^\*})/\delta_a$.

## Proof Strategy

Two one-sided bounds. **Preserved:** union bound over $K-1$ codewords + Gaussian tail $\Rightarrow$
$b_{\mathrm{tok}}$. **Destroyed:** uniform prior on $\sigma$ + a **genie-aided data-processing** reduction to
$M$ independent binary tests in $\mathbb R^{2D}$, each with the correct Bayes error $Q(\delta/\sqrt2 b)$
$\Rightarrow$ product bound $\Rightarrow$ $b_\Pi$. Band nonemptiness under A4.

## Dependency Map

1. **Lemma 1** ⟵ A1, A2, A3 (token), Gaussian tail.
2. **Lemma 2** ⟵ A1, A2, A3 (uniform $\sigma$), A4, data-processing, binary Bayes error in $\mathbb R^{2D}$.
3. **Theorem** ⟵ Lemmas 1–2 + A4 band condition.
4. **Remarks R2 (fraction-correct), R3 (Laplace)** — heuristic, gated on extra assumptions; not used by the Theorem.

## Proof

### Lemma 1 (token-id preserved for $b\le b_{\mathrm{tok}}$)

NN decode is correct $\iff\lVert W\rVert\le\lVert W+c_{a^\*}-c_a\rVert\ \forall a$; squaring and cancelling
$\lVert W\rVert^2$ gives (almost surely, ties having probability $0$ under continuous noise)
$$ \hat a=a^\* \iff \langle W,u_a\rangle\le\tfrac12\delta_a\ \ \forall a\ne a^\*. $$
So $E_{\mathrm{tok}}=\bigcup_{a\ne a^\*}\{\langle W,u_a\rangle>\tfrac12\delta_a\}$, and by the union bound with
$\delta_a\ge d_{\mathrm{tok}}$ and $Q(t)\le\tfrac12 e^{-t^2/2}$,
$$ \Pr[E_{\mathrm{tok}}]\le(K-1)\,Q\!\Big(\tfrac{d_{\mathrm{tok}}}{2b}\Big)\le\tfrac{K-1}{2}\exp\!\Big(-\tfrac{d_{\mathrm{tok}}^2}{8b^2}\Big). $$
For $K\ge3$ this is $\le\tfrac12$ once $\exp(-d_{\mathrm{tok}}^2/8b^2)\le 1/(K-1)$, i.e.
$$ b\le b_{\mathrm{tok}}:=\frac{d_{\mathrm{tok}}}{\sqrt{8\ln(K-1)}}=\Theta\!\Big(\frac{d_{\mathrm{tok}}}{\sqrt{\ln K}}\Big). $$
**Edge case $K=2$:** the bound is $\tfrac12 e^{-d_{\mathrm{tok}}^2/8b^2}<\tfrac12$ for *every* finite $b$, so
$b_{\mathrm{tok}}=+\infty$ (no finite threshold; the $\Theta$ form is asserted only for $K\ge3$). $\square$

### Lemma 2 (permutation destroyed for $b\gtrsim d_\Pi$)

Let $\sigma\sim\mathrm{Unif}(S_N)$ and fix the $M$ disjoint clean-point pairs of A4. Define a **genie** that
reveals: (a) for every clean point outside the pairs, the observation assigned to it; (b) for each pair
$P_m=\{i_m,j_m\}$, the *unordered* set of the two observations assigned to $\{x_{i_m},x_{j_m}\}$, but **not**
the orientation $o_m\in\{0,1\}$ (which observation came from $x_{i_m}$ vs $x_{j_m}$). Because the genie only
adds information, the Bayes-optimal genie-aided success probability dominates the true one:
$$ \Pr[\hat\sigma=\sigma]\le\Pr[\text{genie-aided decoder recovers all }o_m]. $$
Under $\mathrm{Unif}(S_N)$, conditioned on the genie information the orientations $\{o_m\}$ are **independent
uniform bits**: swapping the two labels within a disjoint pair is a measure-preserving symmetry of the
uniform law on $S_N$, and disjointness makes these symmetries act on independent coordinates. Each $o_m$ is
thus an equiprobable binary test from the joint observation $(y,y')\in\mathbb R^{2D}$: the two hypotheses have
joint means $(x_{i_m},x_{j_m})$ and $(x_{j_m},x_{i_m})$ at Euclidean distance
$\lVert(x_{i_m}-x_{j_m},\,x_{j_m}-x_{i_m})\rVert=\sqrt2\,\delta_m$, with isotropic noise
$\mathcal N(0,b^2 I_{2D})$. The Bayes error of an equiprobable binary Gaussian test at mean-distance $\Delta$
is $Q(\Delta/2b)$, hence
$$ P_e^{(m)}=Q\!\Big(\tfrac{\sqrt2\,\delta_m}{2b}\Big)=Q\!\Big(\tfrac{\delta_m}{\sqrt2\,b}\Big)\ \ge\ Q\!\Big(\tfrac{\rho}{\sqrt2\,b}\Big). $$
By independence of the $M$ sub-tests under the genie,
$$ \boxed{\ \Pr[\hat\sigma=\sigma]\ \le\ \prod_{m=1}^M\big(1-P_e^{(m)}\big)\ \le\ \Big(1-Q\!\big(\tfrac{\rho}{\sqrt2 b}\big)\Big)^{M}.\ } $$
As $b\to\infty$, $Q(\rho/\sqrt2 b)\to\tfrac12$, so the bound $\to 2^{-M}$; for fixed $M$ this is the floor, and
it $\to 0$ only as $M\to\infty$. Quantitatively, for any target $\eta\in(2^{-M},1)$ we have $\eta^{1/M}>1/2$,
so $1-\eta^{1/M}<1/2$ and $Q^{-1}(1-\eta^{1/M})>0$; the bound is $\le\eta$ once
$$ b\ \ge\ b_\Pi(\eta,M):=\frac{\rho}{\sqrt2\,Q^{-1}\!\big(1-\eta^{1/M}\big)}=\Theta(d_\Pi) $$
(for **fixed** $\eta,M$; the hidden constant depends on $\eta,M$ — via $Q^{-1}(1-\eta^{1/M})$, which grows
with $M$ — and on the comparability constant in $\rho=\Theta(d_\Pi)$; it is not uniform as $M$ varies).
In particular the closest-pair Bayes error $Q(d_\Pi/\sqrt2 b)$ exceeds any fixed $\epsilon<1/2$ once
$b\ge d_\Pi/(\sqrt2\,Q^{-1}(\epsilon))$: at least one orientation is unrecoverable. $\square$

### Theorem (channel-selectivity band)

Pick $\eta\in(2^{-M},1)$. By Lemma 2, for $b\ge b_\Pi(\eta,M)=\Theta(d_\Pi)$ exact-permutation recovery is
$\le\eta$. By Lemma 1, for $b\le b_{\mathrm{tok}}=\Theta(d_{\mathrm{tok}}/\sqrt{\ln K})$ ($K\ge3$; $=+\infty$ for
$K=2$) token-id error is $<1/2$. The band $(b_\Pi,b_{\mathrm{tok}})$ is nonempty iff
$\Theta(d_\Pi)<d_{\mathrm{tok}}/\sqrt{8\ln(K-1)}$, i.e. **$d_\Pi\sqrt{\ln(K-1)}\ll d_{\mathrm{tok}}$** (A4); for
$K=2$ the band is $(b_\Pi,\infty)$, nonempty unconditionally. For every $b$ in the band, $\Pi$ is destroyed
(exact recovery $\le\eta$) while token-id is preserved (error $<1/2$). $\blacksquare$

## Heuristic Remark R2 (fraction-correct collapse — NOT proved)

Intuitively, when $b/d_\Pi\to\infty$ a ball of radius $\Theta(b)$ around each noised point contains many
clean candidates, so the per-point correct probability $\to 0$ and the *fraction-correctly-placed* readout
(the VMA metric) collapses. A rigorous version needs a **posterior/local-density** assumption at the
observation scale (density of clean points within $\Theta(b)$ of a noisy observation), control of the
displacement, and the limit $b/d_\Pi\to\infty$ — none of which the Theorem assumes. Stated as motivation for
why measured VMA recovery falls toward chance, not as a corollary.

## Heuristic Remark R3 (Laplace noise — NOT proved)

For i.i.d. Laplace coordinates, a projection $\langle W,u\rangle$ ($\lVert u\rVert_2=1$, $\lVert u\rVert_\infty\le1$)
is sub-exponential, so by Bernstein $\Pr[\langle W,u\rangle>t]\le\exp(-c\min\{t^2/b^2,t/b\})$ for an absolute
$c>0$; substituted into Lemma 1 this gives $b_{\mathrm{tok}}=\Theta(d_{\mathrm{tok}}/\ln K)$ in the
large-deviation regime, and the binary Bayes error in Lemma 2 still $\to1/2$ by symmetry. **Caveats (why this
is not a theorem):** product-Laplace noise is *not* rotationally isotropic, so Euclidean NN is not the MAP
decoder and the $\sqrt2$/projection geometry of Lemma 2 must be redone in the correct metric; separation then
requires $d_\Pi\ln K\ll d_{\mathrm{tok}}$. The proved Theorem is Gaussian-only.

## Corrections (from Round-1 review)

- Added **uniform prior** on $\sigma$ (DEF-01); permutation-pair Bayes error corrected to $Q(\delta/\sqrt2 b)$
  (L2-01); product bound floor is $2^{-M}$ for fixed $M$ — destruction needs $M\to\infty$, $\eta>2^{-M}$
  (L2-02, PROB-01); band condition tightened to $d_\Pi\sqrt{\ln(K-1)}\ll d_{\mathrm{tok}}$ (BAND-01); $K=2$
  separated, no finite-$\Theta$ claim (K2-01); ties-zero-probability note (L1-01); Corollary 2 → heuristic
  Remark R2 (COR2-01); Laplace → heuristic Remark R3, Theorem restricted to Gaussian (R1-01).

## Open Risks

- $b_\Pi,b_{\mathrm{tok}}$ are order constants ($\Theta$), not sharp knees; the measured Π-collapse at
  $b\approx0.2$ vs token-id survival to $b\approx0.8$ is *consistent with* — not derived from — unmeasured
  $d_\Pi,d_{\mathrm{tok}}$.
- The genie/independence argument uses **only disjoint pairs**; it makes **no** product claim for
  overlapping clusters (an earlier "overlapping clusters only help" sentence was removed as unproven).
- Fixed-codebook geometric model; it does not assert the network's true geometry satisfies A4 — that is the
  empirical input the model encodes.

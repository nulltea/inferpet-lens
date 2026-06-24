---
type: claim
node_id: claim:kv-cloak-channel-decoupling-feature-mix-loadbearing
name: "KV-CLOAK channel decoupling: secret feature mix is the only load-bearing channel"
description: ""
node_type: claim
status: verified
provenance: "refine-logs/kv-cloak/proof/PROOF_AUDIT.md"
tags: [kv-cache, kv-cloak, bss, cover-invariance, permutation-channel, matched-probe, weights-pub]
date: 2026-06-24
added: 2026-06-24T01:45:05Z
companion: refine-logs/kv-cloak/RESULTS.md
---

# KV-CLOAK channel decoupling: the secret feature rotation is the only load-bearing channel; permutation and block size are cover-invariant

**status:** `verified`

## Claim

KV-CLOAK (arXiv 2508.09442, eq. 9) exposes a per-PagedAttention-block key matrix
`K ∈ R^{b×d}` (b tokens, d = head dim) as

    U = K' = S · P̂ · (K + A) · M,

with `S ∈ R^{b×b}` **secret orthogonal**, per-head `M ∈ R^{d×d}` secret orthogonal,
`P̂ ∈ {0,1}^{b×b}` a one-time-pad row (token) permutation, and `A` a structured additive beacon
mask (the paper's eval config draws `S, M` from the orthogonal group). Against a `WEIGHTS-PUB`
adversary that observes only `U` and runs a key-free blind-source-separation (BSS) / left-demixing
attack (recovered sources are rows of `BU`, hence lie in `rowspace(U)`), graded by the p95 over
plaintext key rows of the Hungarian-matched `|cosine|`, the three channels **decouple onto three
orthogonal observables**, and exactly one of them is load-bearing:

1. **The right-orthogonal feature mix `M` is the only load-bearing channel.** `M` is a secret
   rotation of the per-head feature frame that a key-free adversary cannot align to the fixed
   plaintext reference; it is the only channel that drives recovery to the random-demixing chance
   floor, with an attack-independent ceiling `O(√(s/d))` (`s = rank(K) ≤ b`, `d` = head dim;
   precisely `E_M ρ = O(√(s/d)+√(log n/d))`, simplifying to `O(√(s/d))` when `log n = O(s)`), and the
   only one that collapses the matched negentropy probe. It defeats both the single-observation
   (JADE) and the accumulation (JD) attack.
2. **The left token-mix + permutation `S·P̂` — and therefore the block size `b` — are
   cover-invariant.** Being left-invertible they preserve the key-row subspace exactly (for a fixed
   `K`), leaving the recoverable directions, the oracle recovery, and the chance floor unchanged; the
   permutation sub-channel additionally leaves the Gram spectrum and every permutation-symmetric
   probe exactly fixed. Recovery stays far above the floor and accumulates with the observation count
   `T`, and (empirically) is flat across `b ∈ {16,32,64}`. The combinatorial `b!` permutation
   barrier defends only a permutation-order adversary, not a subspace-membership / ICA adversary that
   never needs the token order.
3. **The additive beacon mask `A` is the only channel that perturbs the row-Gram spectrum**, and
   alone it has only a small, non-floor effect on recovery (0.626 → 0.581 at layer 0).

Consequence: under this threat model KV-CLOAK's security against representation inversion comes
entirely from the secret orthogonal feature mix `M`; its block size — the parameter the paper sweeps
({16, 32, 64}) — provides no protection.

## Theory (lemmas L1–L4, proved below)

Write the exposed per-block matrix, omitting the additive mask for L1–L4 (channel decomposition):
for the orthogonal/permutation channels, `U = L · K · M` where `L = S·P̂ ∈ R^{b×b}` with `S`
**orthogonal** and `P̂` a permutation (so `L` is orthogonal, hence invertible), and `M` is orthogonal
applied per head. The row-Gram is `G(X) = X Xᵀ`. Targets are the **unit-normalized** plaintext rows
`k̂_j = k_j/‖k_j‖` over the `n ≤ s` graded nonzero rows (grading at most as many targets as the `s`
recovered directions, so the assignment is a well-defined injection). A key-free demixing `B` produces
recovered sources = rows of `BU`; the grading is `ρ(B) = p95_j |cos((BU)_{σ(j)}, k_j)|` over the `n`
targets, where `σ` is the Hungarian assignment (so each matched cosine is at most the per-target
oracle alignment).

- **L1 (M is row-Gram invariant).** For orthogonal `M` (block-diagonal per-head as a special case),
  `G(KM) = G(K)` exactly; hence the `gram_error` fingerprint channel is blind to `M`.
- **L2 (orthogonal S·P̂ is a Gram similarity ⇒ spectrum invariant).** For orthogonal `L`,
  `G(LK) = L G(K) Lᵀ` is a similarity, so `spec G(LK) = spec G(K)`. (Requires `S` orthogonal; a
  merely-invertible `S` gives a congruence, preserving rank/inertia but not the spectrum.) In the
  permutation sub-case `L = P̂`, the rows are only reordered, so every permutation-symmetric row
  functional (the eigenvalue multiset, the summed-row negentropy, the covariance spectrum) is exactly
  preserved.
- **L3 (S·P̂ preserves the row space ⇒ oracle recovery and chance floor invariant, for fixed K).**
  For invertible `L`, `rowspace(LK) = rowspace(K)`. Every key-free demixing recovers directions inside
  `rowspace(U) = rowspace(K)`, so the recoverable set, the oracle recovery, and the
  random-orthogonal-demixing floor are unchanged by `S·P̂`. This is invariance under the choice of
  `L` for a **fixed** `K`; changing the physical block size also re-blocks the sequence (a different
  `K`, `s`, target set), so cross-`b` constancy is established here for the operators and confirmed
  empirically across `b ∈ {16,32,64}`. (The realized JADE/JD output is not claimed invariant under
  `S`: it alters the whitened cumulants; empirically it stays far above the floor. Exact probe
  invariance is claimed only for the permutation sub-channel, per L2.)
- **L4 (secret M caps recovery at the chance ceiling `O(√(s/d))`).** Assume `0 < s < d`, nonzero key
  rows, `M ~ Haar(O(d))`. Then `rowspace(KM) = V·M =: W` is a uniformly random `s`-subspace of `R^d`,
  independent of the fixed frame `V`. For any rowspace attack and any target `k̂_j`, the best
  achievable alignment is `‖P_W k̂_j‖`, with `‖P_W k̂_j‖² ~ Beta(s/2,(d-s)/2)`, mean `s/d`. Hence
  `ρ ≤ max_j ‖P_W k̂_j‖`, and a Bernstein-type Beta tail + union bound over the `n ≤ s` graded targets
  give `E_M ρ = O(√(s/d) + √(log n /d))`, which is `O(√(s/d))` when `log n = O(s)`, vanishing as
  `d/s → ∞`: an attack-independent ceiling that drives recovery to the chance level. This is the
  security-relevant (upper-bound) direction; it is **not** identified with the random-demixing baseline
  (a random unit in `W` has mean-square overlap `1/d`, an order statistic `O(√(log(ns)/d))` below the
  ceiling). The baseline is not a lower bound: a realized attack may score below it.

L1–L2 are exact identities; L3 is an exact subspace identity (fixed `K`) plus invariance of the
oracle bound and floor; L4 is an attack-independent expectation/concentration ceiling over the secret
key `M ~ Haar(O(d))`. Together they decouple the channels and single out `M`.

### Proofs

**Assumptions and notation.** $K\in\mathbb R^{b\times d}$ is one block of $b$ plaintext key rows
$k_1,\dots,k_b\in\mathbb R^{d}$ (one head; $d$ the head dimension), with row space
$V:=\operatorname{rowspace}(K)\subseteq\mathbb R^{d}$ of dimension $s:=\operatorname{rank}(K)\le\min(b,d)$.
$M\in\mathbb R^{d\times d}$ is orthogonal ($M M^{\top}=I_d$), applied on the right (per head).
$L:=S\widehat P\in\mathbb R^{b\times b}$ with $S$ orthogonal and $\widehat P$ a permutation matrix,
so $L$ is orthogonal, hence invertible. $G(X):=XX^{\top}$. We omit the additive mask $A$ throughout
(it is a separate channel; see the claim). Assume $0<s<d$ and that all graded key rows are nonzero;
write the unit targets $\hat k_j:=k_j/\|k_j\|$ over the $n\le s$ graded nonzero key rows (so the
$n$ targets are assigned to distinct members of the $s$-dimensional recovered basis). A key-free
demixing is a matrix $B$ whose recovered sources are the rows of $BU$; with $\sigma$ the Hungarian
assignment of targets to distinct recovered sources, recovery is graded by
$\rho(B)=\mathrm{p95}_{j}\,|\cos(\,(BU)_{\sigma(j)},\,k_j)|$ over the $n$ targets. Each matched cosine
is at most the per-target oracle alignment
$\max_{u\in\operatorname{rowspace}(U),\,\|u\|=1}|\langle u,\hat k_j\rangle|$, so
$\rho(B)\le\mathrm{p95}_j\max_{u}|\langle u,\hat k_j\rangle|$ for every $B$.
The random-orthogonal-demixing floor $\rho_{\mathrm{fl}}$ is $\rho$ when the rows of $BU$ are a
uniformly random orthonormal basis of $\operatorname{rowspace}(U)$.

**Lemma L1 (M leaves the row-Gram invariant).** With $U=KM$ and $M$ orthogonal,
$$G(KM)=KM(KM)^{\top}=K\,M M^{\top}K^{\top}=K I_d K^{\top}=KK^{\top}=G(K).$$
The identity is exact and uses only $MM^{\top}=I_d$; the per-head block-diagonal structure
$M=\mathrm{diag}(M_1,\dots,M_H)$ is a special case (a block-diagonal matrix with orthogonal blocks
is orthogonal). Since the `gram_error` statistic is a function of $G(U)$ versus $G(K)$, it is
identically zero under the $M$ channel. $\qquad\blacksquare$

**Lemma L2 (S·P̂ is an orthogonal similarity on the row-Gram; spectrum invariant).** With $U=LK$,
$$G(LK)=LK(LK)^{\top}=L\,(KK^{\top})\,L^{\top}=L\,G(K)\,L^{\top}.$$
$L=S\widehat P$ is a product of orthogonal matrices ($\widehat P$ is orthogonal because its columns
are distinct standard basis vectors), hence orthogonal, so $L^{\top}=L^{-1}$ and $G(LK)=L\,G(K)\,L^{-1}$
is a similarity transform. Similar matrices have identical spectra, so
$\operatorname{spec}G(LK)=\operatorname{spec}G(K)$. (This step requires $S$ orthogonal; a
merely-invertible $S$ gives a congruence $L\,G(K)\,L^{\top}$ with $L^{\top}\ne L^{-1}$, preserving
rank and inertia but not the spectrum.) The permutation-only case ($S=I$, $L=\widehat P$) is the same
statement; moreover $\widehat P$ merely reorders the rows, so $G(\widehat P K)=\widehat P\,G(K)\,\widehat P^{\top}$
is $G(K)$ with rows and columns jointly permuted, and every permutation-symmetric functional of the
rows (the eigenvalue multiset; the sum-over-rows negentropy; the covariance spectrum) is exactly
preserved. $\qquad\blacksquare$

**Lemma L3 (S·P̂ preserves the recovered subspace, the oracle recovery, and the chance floor, for
fixed K).** For invertible $L$, $\operatorname{rowspace}(LK)=\operatorname{rowspace}(K)=V$: each row
of $LK$ is a linear combination of rows of $K$ (so $\operatorname{rowspace}(LK)\subseteq V$), and
$K=L^{-1}(LK)$ gives the reverse inclusion. Take $U=LK$ (the $S\widehat P$ channel, $M=I$). For any
demixing $B$, the rows of $BU$ lie in $\operatorname{rowspace}(U)=V$, so the set of directions any
key-free demixing can produce is exactly the unit sphere of $V$, identical to the $L=I$ case, and the
targets $\hat k_j\in V$ are unchanged. Consequently (i) the oracle recovery
$\sup_B\rho=\mathrm{p95}_j\max_{u\in V,\|u\|=1}|\langle u,\hat k_j\rangle|=1$ (each $\hat k_j\in V$),
and (ii) the floor $\rho_{\mathrm{fl}}$ — the law of $\rho$ for a uniformly random orthonormal basis
of $\operatorname{rowspace}(U)=V$ — are both invariant under $S\widehat P$, because neither depends on
$L$.

This is invariance under the choice of the orthogonal/permutation operator $L$ for a **fixed** block
$K$. It does not by itself assert constancy across physical block sizes: changing $b$ re-blocks the
token sequence into a different $K$ (different $s$, different targets). The cross-$b$ statement is
therefore: the operators $S\widehat P$ of any size leave the fixed-$K$ invariants unchanged, and the
experiment confirms recovery and the matched probe are flat across $b\in\{16,32,64\}$. For the
permutation sub-channel $\widehat P$ (the literal $b!$ knob), L2 additionally gives exact invariance of
the Gram spectrum and every permutation-symmetric probe. (The *realized* JADE/JD output is not claimed
invariant under $S$: $S$ alters the whitened cumulants, so the algorithm may return a poorer basis;
empirically it stays far above $\rho_{\mathrm{fl}}$. Exact probe invariance is claimed only for the
permutation sub-channel.) $\qquad\blacksquare$

**Lemma L4 (a secret Haar M caps graded recovery at the chance ceiling $O(\sqrt{s/d})$).** Let
$M\sim\mathrm{Haar}(O(d))$ and $U=KM$. By L3 with $L=I$, every recovered direction lies in
$\operatorname{rowspace}(U)=\{xM:x\in V\}=:W$, an $s$-dimensional subspace. Because
right-multiplication by Haar $M$ is an orthogonal map, $W$ is uniform on the Grassmannian
$\mathrm{Gr}(s,d)$, *independent of the fixed plaintext frame $V$* (Haar on $O(d)$ vs $SO(d)$ does not
affect the law of $W$).

By the grading bound in the setup, for every demixing $B$,
$\rho(B)\le\mathrm{p95}_j\,\max_{u\in W,\|u\|=1}|\langle u,\hat k_j\rangle|\le\max_j\|P_W\hat k_j\|$,
where the inner equality $\max_{u\in W,\|u\|=1}|\langle u,\hat k_j\rangle|=\|P_W\hat k_j\|$ is
Cauchy–Schwarz (attained at $u=P_W\hat k_j/\|P_W\hat k_j\|$ when the projection is nonzero; otherwise
both sides are $0$) and $P_W$ is the orthogonal projector onto $W$. This bounds *any* rowspace attack,
the oracle included.

Distribution of $\|P_W\hat k_j\|^2$: by rotational invariance $\|P_W\hat k_j\|^2\stackrel{d}{=}\|P_V(\hat k_jM^{\top})\|^2$,
and $\hat k_jM^{\top}$ is a uniform unit vector on $S^{d-1}$; the squared norm of its projection onto
the fixed $s$-subspace $V$ is the sum of $s$ coordinates squared of a uniform sphere point, the
standard fact
$$\|P_W\hat k_j\|^2 \;\sim\; \mathrm{Beta}\!\left(\tfrac{s}{2},\,\tfrac{d-s}{2}\right),\qquad
\mathbb E\,\|P_W\hat k_j\|^2=\frac{s}{d},\qquad
\mathrm{Var}=\frac{2s(d-s)}{d^2(d+2)}\le\frac{2s}{d^2}.$$
Per target, $\mathbb E_M\|P_W\hat k_j\|\le\sqrt{s/d}$ (Jensen). Let $n\le s$ be the number of graded
nonzero targets. A $\mathrm{Beta}(s/2,(d-s)/2)$ variable obeys the Bernstein-type tail
$\Pr\!\big(\|P_W\hat k_j\|^2 - s/d \ge t\big)\le \exp\!\big(-c\,\min(d^2t^2/s,\,dt)\big)$ (standard for
the projection of a uniform sphere point onto a fixed $s$-subspace), so a union bound over the $n\le s$
targets gives
$$\mathbb E_M\,\max_{j}\|P_W\hat k_j\|^2 \;\le\; \frac{s}{d} \;+\; C\!\left(\frac{\sqrt{s\,\log n}}{d}+\frac{\log n}{d}\right).$$
Since $\sqrt{s\log n}/d\le (s+\log n)/d$ and (for $\log n\le s$) $\log n/d\le s/d$, the right side is
$O(s/d)$, whence
$$\mathbb E_M\,\rho \;\le\; \mathbb E_M\,\max_j\|P_W\hat k_j\| \;\le\; \sqrt{\,\mathbb E_M\max_j\|P_W\hat k_j\|^2\,}
\;=\; O\!\big(\sqrt{s/d}+\sqrt{\log n/d}\big),$$
which is $O(\sqrt{s/d})$ whenever $\log n = O(s)$, and in all cases $\to 0$ as $d/s\to\infty$ with
$\log n = o(d)$. Thus, *for any rowspace attack without the secret $M$*, the expected graded recovery
is bounded by an attack-independent ceiling tending to $0$: $M$ drives recovery to the chance level.

This ceiling is the security-relevant (upper-bound) direction. It is **not** the random-demixing
baseline: a uniformly random unit direction in $W$ has $\mathbb E|\langle u,\hat k_j\rangle|^2=1/d$
(a uniform direction in $\mathbb R^d$), so the random-demixing baseline is an order statistic of size
$O(\sqrt{\log(ns)/d})$ under separated targets, which lies *below* the oracle ceiling $O(\sqrt{s/d})$.
The baseline is not a lower bound on arbitrary attacks: a realized attack may score below it.

Contrast with L3: under $S\widehat P$ the recoverable subspace is exactly $V$, which *contains* the
targets, so the oracle recovery is $1$ and the realized recovery stays far above the baseline; under
$M$ the recoverable subspace is an independent rotation $W$, so even the oracle is capped at
$O(\sqrt{s/d})$. This is the precise sense in which $M$, and only $M$, carries privacy. (In the
experiment $d=128$ per head and the recovery dimension is capped at $s\le 16$ (so $n\le s\le 16$), so
the oracle ceiling $\sqrt{s/d}\lesssim 0.35$ and $\log n<s$; the measured $m$-channel recovery $0.126$
sits just below the empirical chance baseline $\rho_{\mathrm{fl}}=0.157$, consistent with recovery at
chance.) $\qquad\blacksquare$

**Status.** L1, L2 exact identities (L2 requires $S$ orthogonal; a merely-invertible $S$ gives a
congruence, preserving rank/inertia but not the spectrum). L3 an exact subspace identity for fixed
$K$ plus invariance of the oracle recovery and the floor under the operator $L$; cross-$b$ constancy
is operator-level + empirical, not a pure-theory claim about re-blocking. L4 an attack-independent
expectation ceiling $\mathbb E_M\rho=O(\sqrt{s/d}+\sqrt{\log n/d})$ (which is $O(\sqrt{s/d})$ when
$\log n=O(s)$) over the secret key $M\sim\mathrm{Haar}(O(d))$, for any rowspace (left-demixing) attack
lacking $M$.

Open risks / scope. (1) L4 is an *upper bound* on the expected recovery (the security-relevant
direction); the realized attack is observed at or below the ceiling but is not proved to attain it.
(2) The privacy conclusion is conditional on $s/d$ small: the ceiling is vacuous as $s/d\to1$ (if
$s=d$ then $W=\mathbb R^d$ and the oracle alignment is $1$). In the experiment $s\le16\ll d=128$.
(3) The concentration is *absolute* ($\mathrm{Var}=O(s/d^2)\to0$), not relative for fixed $s$; the
statement is $\mathbb E_M\rho\to0$ as $d/s\to\infty$, not that $\rho/\sqrt{s/d}$ concentrates. (4)
Degenerate cases $s=0$ (zero block, recovery $0$) and $s=d$ (no rotation room) are excluded by
$0<s<d$. (5) Exact matched-probe invariance under the token channel is proved only for the
permutation $\widehat P$; invariance under the orthogonal mix $S$ is empirical (small spread).

## Evidence chain

Real Qwen3-4B raw per-head keys (kind `k`), layers {0,12,20}, 48 prompts (110–178 tokens); 273 cells;
CPU + numba. Full numbers in `refine-logs/kv-cloak/RESULTS.md` and `analysis.json` / `sanity.json`.

- **B1 exact identities** (full-feature): `gram_error(M-only)=2.1e-9`, `spec_rel_err(M-only)=5.7e-10`
  (L1); `spec_rel_err(S·P̂-only)=3.2e-9`, full-Gram cos-dist `1.40` (L2); `spec_rel_err(perm)=1.2e-15`;
  `spec_rel_err(A-only)=0.20`; `S,M` orthogonal to 1e-15. 8/8 unit checks.
- **Channel table, layer 0** (floor 0.157): identity 0.626 → `m`/`naive`/`full` **0.126** (floor),
  `sp`/`scx` **0.612** (≈ plaintext), `a` 0.581. Negentropy 1044b → `M` 1.5b; `S·P̂` ≈ 1075b.
  Consistent at L12/L20 (M 0.151/0.160; S·P̂ 0.497/0.517).
- **JD accumulation, layer 0** (floor 0.157→0.168 over T): identity 0.488→0.751, `sp` 0.486→0.730,
  `scx` 0.488→0.725; `m`/`naive`/`full` ~0.126→~0.12 (at/below floor, flat).
- **b-flatness**: JADE flat over `b∈{16,32,64}` (`m` exactly 0.145; spectral-cap spread 0.000 m, 0.054
  scx, 0.264 sp).
- **Matched probe (C2)**: negentropy ↔ JADE Spearman 0.706 (p=5e-42, n=270) aggregate; channel-mean
  0.77; within-channel weak/sign-flipped (anticorrelates under the mask). Between-channel diagnostic,
  not a within-channel oracle. Spectral-capacity ρ=0.33 (not matched).
- **Integrity**: audit WARN (no fabrication / phantom / probe-attack circularity; reporting fixes
  applied). **Proof**: cross-model PASS (`refine-logs/kv-cloak/proof/PROOF_AUDIT.md`).

## Honest scope / non-claims

- `b`-inertness established for `b ∈ {16,32,64}` on 110–178-token prompts against the BSS/ICA
  adversary; not a claim about a permutation-order adversary.
- Not claimed: that `S·P̂` leaves recovery at *exactly* the plaintext value (the real algorithm drops
  ~15% at L12/L20 while staying far above floor; L3 is about the oracle subspace and the floor, which
  are exactly invariant). Not claimed: that `A` has zero recovery effect (0.626→0.581).
- The empirical `M`-only `gram_error` is 0.124, not the exact 2.1e-9, because the attack subsamples
  256/1024 features (breaks orthogonality of the feature sub-block); the exact identity is L1 on the
  full feature set.
- `A` is a faithful-but-stylized model of the paper's positional beacon (row-spanning, not strictly
  per-head); the M / S·P̂ conclusions do not depend on this choice.

## Connections

_Edges are recorded in `graph/edges.jsonl`; summarized here for human readers._

- Extends [[kv-bss-subspace-floor-and-negentropy-probe]] (Task-1 subspace-membership floor + matched
  negentropy probe) onto a published defense.
- Supported by experiment [[b-kv2-cloak-channel-decoupling]].
- Instantiates the cover-invariance principle (orthogonal/permutation covers leave the
  subspace-membership channel untouched) on KV-CLOAK.

---
type: claim
node_id: claim:perm-llr-threshold
name: "Full sorted-row matching DPI-dominates RowSort-64 and is the maximal-invariant matcher for the column-permutation cover"
description: "Empirical (jury-PARTIAL) + theory (proof-checker PASS): on the AloePri column-permutation embedding cover the full sorted-row matcher recovers the secret permutation +0.43-0.60 above RowSort-64 across the alpha_e sweep while the independent CLUB-on-phi probe barely moves; theory: RowSort phi is a deterministic function of the full sorted row so I(phi)<=I(sort) by DPI, and the sorted row is the maximal invariant of the per-row column-permutation group with sorted-neg-Euclidean = profile-MLE; 2 log n cited (Dai-Cullina-Kiyavash), not re-derived."
node_type: claim
status: verified
provenance: ".aris/traces/result-to-claim/2026-06-23_run01/ ; .aris/traces/experiment-audit/2026-06-23_run01/ ; .aris/traces/proof-checker/2026-06-23_run01/ ; refine-logs/perm-cover/"
tags: ["backbone", "provable", "permutation", "perm-cover", "DPI", "empirical", "theory"]
date: 2026-06-21
updated: 2026-06-23
---

# Full sorted-row matching DPI-dominates RowSort-64 and is the maximal-invariant matcher for the column-permutation cover

**status:** `verified` (proof axis: theory PASS by `/proof-checker` 2026-06-23, thread `019ef653`;
empirical axis jury `PARTIAL`/scoped — Codex result-to-claim 2026-06-23. The two axes are separate:
the proof is verified; the empirical support is scoped to a single embedding table.)

## Statement

On the AloePri column-permutation embedding cover (secret = the row correspondence Π over N token
rows; cover = a shared column permutation `P_col` + per-row isotropic Gaussian, noise knob α_e), the
**Vocabulary-Matching Attack (VMA)** matches obfuscated rows to plaintext rows by a
column-permutation-invariant per-row signature + Hungarian assignment. Two results:

**Empirical (jury-supported, this surface).** The **full sorted-row** matcher (the entire sorted row;
negative-Euclidean = the isotropic-Gaussian MLE on the sorted vector, or cosine — numerically
identical here) recovers the secret permutation **+0.434 above RowSort-64 at α_e=0.2** (0.999 vs
0.565) and **+0.600 at α_e=0.35** (0.804 vs 0.204), while the independent **CLUB-on-φ** probe moves
only ~2% by α_e=0.2 (245→240 bits) and ~4% by α_e=0.35 (→235 bits). RowSort-64's 64-quantile binning
collapses at small noise where the leakage has not gone; the full statistic recovers it.

**Theory (proof inline, verified).** (i) RowSort's 64-quantile signature φ is a *deterministic
function* of the full sorted row, so by the data-processing inequality `I(φ_obf; φ_plain) ≤
I(sort_obf; sort_plain)` — RowSort is **DPI-dominated** and cannot exceed the full-sort matcher's
information. (ii) Under the column-permutation group `S_d` acting on a row, the **sorted vector is the
maximal invariant**, and for the per-row isotropic-Gaussian observation model the **sorted
negative-squared-Euclidean distance equals the profile log-likelihood** (rearrangement inequality),
so sorted-neg-Euclidean + Hungarian is the profile-MLE / joint-MAP assignment for the
per-row-permutation channel. (iii) The exact-recovery information threshold for Gaussian feature
alignment, `I_XY ≥ 2 log n`, is **cited** (Dai–Cullina–Kiyavash), not re-derived here.

## Honest scope (after jury + integrity audit)
- **Empirical:** single embedding table (gemma-2 family cache, d=2304); `vma_stronger` is 3-seed,
  `aloepri_vma_sweep` is 1-seed. The full-sort ≫ RowSort gap and the CLUB stability are the
  jury-checkable content.
- **Theory:** the maximal-invariant / profile-MLE optimality (ii) is exact for the model where each
  row may be *independently* column-permuted. The AloePri cover applies a **single shared** column
  permutation across all rows (`aloepri.py:132`, `noisy[:, col]`); under the shared model the per-row
  sorted statistic is still `S_d`-invariant (sound) and still DPI-dominates RowSort (i is
  unconditional), but it is **not the joint maximal invariant** — it discards cross-row column
  alignment. So (ii) is stated for the per-row-permutation channel and is a sound (DPI-dominating)
  matcher for the shared-permutation channel; this gap is explicit, not hidden.
- The `2 log n` constant (iii) is an external theorem under a jointly-Gaussian feature model; cited,
  not proven here.

## Empirical status (jury-gated, NOT self-certified)
`PARTIAL` / scoped (Codex xhigh result-to-claim, 2026-06-23, confidence medium; trace
`.aris/traces/result-to-claim/2026-06-23_run01/`). The jury directed splitting the empirical part
(supported now) from the theory part (proof pending) — done above. Integrity audit **WARN, no FAIL**
(`refine-logs/perm-cover/EXPERIMENT_AUDIT.md`, Codex read-only, thread `019ef64a`): VMA graded against
the genuine secret τ; no normalization; cited numbers exist. The audit's one correction —
**retrieval-PVI is the attack-in-bits, not an independent probe** — is applied: only **CLUB-on-φ**
carries the independence claim here.

## Evidence chain (bits canonical + per-secret readout)
- **R1 attack-strength sweep** (`results/vma_stronger.json`; N=1000, 3 seeds): full-sort_euc vs
  RowSort-64 recovery, with CLUB-on-φ (independent) as bits: uplift +0.023 / **+0.434** / **+0.600** /
  +0.343 / +0.103 / +0.031 at α_e 0.1 / 0.2 / 0.35 / 0.5 / 0.75 / 1.0; CLUB-on-φ 243→206 bits over
  the same span (~2% by α_e 0.2 where RowSort has already lost half its recovery). Recorrelation
  Spearman(recovery, CLUB): RowSort all-sweep +1.00, full-sort +0.99.
- **R2 probe tracking + keymat negative control** (`results/aloepri_vma_sweep.json`; N=1200, 1 seed):
  Spearman(**independent CLUB-on-φ**, VMA recovery) = **+0.976** across α_e; the dense keymat cover
  drives VMA → 0.0 (chance≈8.3e-4) **and** CLUB-on-φ → ≈0 bits (−2.4, estimator floor) together.
  (retrieval-PVI ρ=1.0 is the dependent attack-in-bits, reported as a labeled reference only.)
- **Independence (audited):** CLUB-on-φ runs no Hungarian/NN matching (`measures.py:45`, variational
  MI on τ-paired signatures); it is structurally independent of the VMA assignment.
- Full standardized table: `refine-logs/perm-cover/RESULTS_STANDARDIZED.md`.

## Theory — proof inline (verified by `/proof-checker`, thread `019ef653`, PASS)

**Setup.** Plaintext rows $x_1,\dots,x_N\in\mathbb R^d$; the cover emits $y_{\Pi(i)}=P(x_i+e_i)$ with
$P\in\mathcal S_d$ a coordinate-permutation matrix, $e_i\sim\mathcal N(0,\sigma^2 I_d)$ i.i.d., $\Pi$
a uniform secret row permutation. Signatures $s(v)=\operatorname{sort}(v)$ and
$\varphi(v)=Q(s(v))$, $Q$ the fixed 64-quantile binning (mean-centering + $L^2$-normalization folded
into the deterministic post-maps).

### Lemma 1 (DPI domination, unconditional)
$\varphi=Q\circ s$ with $Q$ deterministic. For any joint law of $A:=s(X)$, $B:=s(Y)$: $Q(A)-A-B$ is
Markov so $I(Q(A);B)\le I(A;B)$; $Q(B)-B-Q(A)$ is Markov so $I(Q(A);Q(B))\le I(Q(A);B)$. Chaining,
$$I(\varphi(X);\varphi(Y)) \le I(s(X);s(Y)) \quad\text{for every }\sigma.$$
RowSort-64 carries no more cross-table information than the full sorted row; any $\varphi$-matcher is
information-dominated by an $s$-matcher. $Q$ injective on the support of $s$ is *sufficient* for
equality (not necessary); for $d=2304\gg 64$ the quantization is many-to-one (non-injective),
consistent with a strict gap — the loss is real. $\blacksquare$

### Lemma 2 (maximal invariant + profile-MLE = sorted neg-Euclidean; per-row channel)
**(i)** $s(Pv)=s(v)$, and $s(u)=s(v)$ iff $u,v$ share an $\mathcal S_d$-orbit; so $s$ is the **maximal
invariant** of the coordinate-permutation action. **(ii)** For a pairing $(y,x)$ with nuisance
$P\in\mathcal S_d$, $Pe\sim\mathcal N(0,\sigma^2 PP^\top)=\mathcal N(0,\sigma^2 I)$ since $PP^\top=I$,
so $p(y\mid x,P)\propto\exp(-\lVert y-Px\rVert^2/2\sigma^2)$. **(iii)** $\min_P\lVert
y-Px\rVert^2\iff\max_P\langle y,Px\rangle$; by the **rearrangement inequality** (Hardy–Littlewood–
Pólya) the maximum is $\langle s(y),s(x)\rangle$, giving
$$\min_{P\in\mathcal S_d}\lVert y-Px\rVert^2=\lVert s(y)-s(x)\rVert^2,$$
so the per-row profile log-likelihood is $-\lVert s(y)-s(x)\rVert^2/2\sigma^2+\text{const}$. **(iv)**
With per-row independence and uniform prior on $\Pi$, the joint profile log-likelihood of a one-to-one
assignment $\rho$ is $-\tfrac{1}{2\sigma^2}\sum_j\lVert s(y_j)-s(x_{\rho(j)})\rVert^2+\text{const}$;
maximizing over one-to-one $\rho$ is the linear assignment problem solved by Hungarian (an optimizer;
coordinate ties can make the optimizing permutation/matching non-unique, but the value identity in
(iii) is unaffected). Hence **full-sort neg-Euclidean + Hungarian is the profile-MLE matcher —
equivalently the joint MAP over $(\Pi,\{P_i\})$ under uniform priors — for the per-row
column-permutation isotropic-Gaussian channel.** It is *not* the marginal MAP over $\Pi$ alone (that
would log-sum-exp over the nuisance permutations rather than maximize). (Cosine on mean-centered,
$L^2$-normalized sorted rows gives the same ranking, since $\lVert a-b\rVert^2=2(1-\langle
a,b\rangle)$ for unit vectors — matching the code's identical `fullsort_cos`/`fullsort_euc`.) $\blacksquare$

### Remark (shared-permutation gap — scope boundary)
The AloePri cover applies a **single** $P$ to all rows (`aloepri.py:132`). L1 is unaffected (a
statement about $Q$ on marginals). L2's optimality is for the per-row model: under a shared $P$ the
joint law couples rows, and the row-row Gram $YY^\top=(X{+}E)(X{+}E)^\top$ (up to $\Pi$) is a
shared-$P$ invariant **not** recoverable from per-row multisets, so per-row $s$ is invariant and sound
(and still DPI-dominates RowSort) but is **not** the joint maximal invariant — the sorted matcher is a
relaxation, not the proven exact joint MAP for the shared channel. Near-perfect recovery at small
$\sigma$ reflects near-unique per-row fingerprints, not a shared-channel optimality theorem.

### Cited threshold (external benchmark, not re-derived, not implied by L1/L2)
In the Gaussian *database alignment* model of Dai–Cullina–Kiyavash (*Database Alignment with Gaussian
Features*, arXiv:1903.01422), exact permutation recovery is achievable at feature mutual information
$I_{XY}\ge 2\log n + \omega(1)$ (with a matching converse under their canonical/asymptotic
assumptions). We cite this as an external order/threshold benchmark for the sorted-vector features;
it is **not** a consequence of Lemmas 1–2 and does **not** automatically transfer to the quantized
$\varphi$ or to the shared-$P$ AloePri channel.

_Full proof package with dependency map and assumptions: `refine-logs/perm-cover/PROOF_PACKAGE.md`;
audit `refine-logs/perm-cover/PROOF_AUDIT.md`._

## Connections
Permutation-channel analog of the L0 Bayes-NN info-efficiency result [[info-efficient-attacks-findings]]
(weak-attack collapse = attack weakness, not leakage loss). Independence backbone
[[threat-model-fairness]]. Experiment log [[exp:vma-fullsort-vs-rowsort]]; negative companion
[[exp:cover-break-matched-deferred]].
_Edges recorded in `graph/edges.jsonl`._

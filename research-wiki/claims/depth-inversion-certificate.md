---
type: claim
node_id: claim:depth-inversion-certificate
name: "Depth does not confer privacy on the Qwen3-4B residual stream — a vocab-disjoint selectivity certificate, with a learned inverter beating ridge at the deepest layer, and an attack-independent probe that tracks recovery across depth (scoped)"
description: "Empirical (jury PARTIAL/scope-only) + proved certificate lemma. C1: best-inverter vocab-disjoint shuffle-subtracted selectivity stays 0.39–0.69 across all 9 depths L0..L32 of Qwen3-4B resid_post (every CI excludes 0, cosine-NN memorization floor 0.000) — depth-irreversibility falsified. DECISION: a learned 2-layer inverter beats linear ridge at L32 (0.542 vs 0.390, disjoint CIs). C2 (POSITIVE measurement-loop): attack-independent probes track recovery across depth — Spearman(cap reader acc, best recovery)=+0.85, Spearman(CLUB bits, best recovery)=+0.78. A Fano-type lemma (proof inline, jury PASS) lower-bounds I(token;resid) from the RAW retrieval accuracy; the shuffle-subtracted selectivity operationally falsifies irreversibility and attributes signal to the residual, while the per-depth positive bit certificate stays conditional on the pool size K and a lower entropy bound."
node_type: claim
status: drafted
provenance: "refine-logs/resid-depth-inversion/ ; runs/full/depth_sweep.json ; PROOF_PACKAGE.md (proof-checker PASS, .aris/traces/proof-checker/2026-06-24_run01/) ; result-to-claim PARTIAL (RESULT_TO_CLAIM.md) ; experiment-audit WARN/no-FAIL (EXPERIMENT_AUDIT.md)"
tags: ["empirical", "partial", "theory", "resid-depth-inversion", "inversion", "depth", "fano", "measurement-loop-positive", "WEIGHTS-PUB"]
date: 2026-06-24
updated: 2026-06-24
---

# Depth does not confer privacy on the Qwen3-4B residual stream (scoped certificate + positive probe–attack tracking)

**status:** `drafted` (empirical jury verdict `PARTIAL`/scope-only; certificate lemma `verified` by
cross-model proof-checker, PASS). The PROOF axis (`verified`) is the Fano lemma; the empirical
support is the experiment, carried by `supports` edges — the two axes are kept separate.

## Statement (scoped after jury, 2026-06-24)

On **Qwen/Qwen3-4B**, surface **`resid_post`**, corpus **release-gate-512** (9469 token-positions/layer
captured; vocab-disjoint inverter split n_train 3373 / n_test 413 per layer), depth grid every-4
(L0..L32, 9 points), under threat model **WEIGHTS-PUB**:

- **C1 — depth does not confer privacy.** The best inverter's vocab-disjoint, shuffle-subtracted
  retrieval selectivity stays in **[0.39, 0.69]** across all nine depths; every bootstrap 95% CI
  excludes 0 (lowest L32 ridge [0.341, 0.438]); the cosine-NN memorization baseline is **0.000** at
  every depth. The depth curve is flat/non-monotone, never collapsing. Reproduces the headline of
  Dong et al. (arXiv 2507.16372) on this model/surface.
- **DECISION — a learned inverter beats linear ridge at the deepest layer.** mlp2 (learned 2-layer
  head) ≈ ridge through mid-network but at **L32 mlp2 = 0.542 vs ridge = 0.390** (gap +0.153), with
  **disjoint** bootstrap CIs (mlp2 [0.494,0.591] vs ridge [0.341,0.438]); elsewhere mlp2≈ridge
  (overlapping CIs). Ridge's late-layer drop is therefore a *linear-inverter artifact* at the
  deepest layer, recoverable by a nonlinear inverter.
- **C2 — an attack-independent probe tracks recovery across depth (POSITIVE).**
  Spearman(capacity-matched token-id V-information reader accuracy, best recovery) = **+0.85**;
  Spearman(CLUB MI bits, best recovery) = **+0.78**; cap-acc tracks ridge +0.80, mlp2 +0.83. Both
  probes are attack-independent by construction (cap-PVI reads token-id *classes* via a PCA-softmax
  reader, never the embedding table the attack retrieves against; CLUB is a separate variational MI
  estimator). Shuffle floors ≈ 0.001–0.005 (no label leakage). This is the positive
  measurement-loop regime: an attack-independent IT measure predicts inversion recovery across depth.
  **Probe scope (honest):** cap-PVI uses a row split over a shared, capped (≤256) set of frequent
  token classes, NOT the attack's vocab-disjoint split, so accuracy and selectivity are not on
  identical partitions; CLUB is an embedding-space MI estimator, attack-separate but target-adjacent,
  and its absolute bits are an upper-bound score read for rank only (the token-id bit certificate is
  the Fano lemma, not CLUB).

**Honest scope:** single model, single corpus, single sweep seed; n=9 depth points → Spearman +0.85
has a permutation p-value below 0.01; bootstrap CIs are over the 413 vocab-disjoint test rows per layer, not over seeds. This
is NOT an all-transformer claim. Jury (Codex xhigh result-to-claim, 2026-06-24) rated all three
`partial` — scope-limited, NOT integrity/correctness.

## Empirical status (jury-gated, NOT self-certified)

- **result-to-claim** (`RESULT_TO_CLAIM.md`): `PARTIAL`/scoped, positive-correlate branch. Narrowly
  scoped claims strongly supported; ask was to state scope precisely (done above).
- **experiment-audit** (`EXPERIMENT_AUDIT.md`, Codex gpt-5.5 xhigh, 2 rounds): **WARN, no FAIL** —
  fraud patterns A (ground-truth provenance: real tokenizer ids), B (no score normalization), C
  (numbers match JSON, run.exit=0), D (**probe–attack independence verified**: cap-PVI uses token-id
  classes + PCA-softmax reader, not the attack's embedding retrieval; CLUB is a separate MI
  estimator), F (real_gt) all PASS; only E (scope) is WARN, honestly disclosed.

## Evidence chain

Per-layer table (recovery = selectivity = real − shuffle; bits beside readout). Data:
`runs/full/depth_sweep.json` (wall-time 205s, exit 0).

| layer | ridge sel [95% CI] | mlp2 sel | nn (memorization floor) | cap reader acc | CLUB upper-bound bits (rank only) |
|---|---|---|---|---|---|
| L0  | 0.685 [0.639, 0.731] | 0.639 | 0.000 | 0.939 | 3426 |
| L4  | 0.598 [0.550, 0.646] | 0.651 | 0.000 | 0.861 | 3381 |
| L8  | 0.588 [0.540, 0.637] | 0.581 | 0.000 | 0.797 | 3003 |
| L12 | 0.593 [0.547, 0.639] | 0.586 | 0.000 | 0.732 | 2824 |
| L16 | 0.533 [0.482, 0.581] | 0.494 | 0.000 | 0.691 | 2792 |
| L20 | 0.504 [0.455, 0.552] | 0.523 | 0.000 | 0.708 | 2891 |
| L24 | 0.603 [0.552, 0.649] | 0.576 | 0.000 | 0.769 | 3062 |
| L28 | 0.540 [0.492, 0.588] | 0.571 | 0.000 | 0.753 | 3043 |
| L32 | 0.390 [0.341, 0.438] | **0.542** [0.494,0.591] | 0.000 | 0.685 | 2970 |

Cross-depth correlation: Spearman(cap acc, best recovery) +0.85; Spearman(CLUB, best recovery) +0.78.

- **nn = 0.000 everywhere** under the vocab-disjoint split ⇒ recovery is genuine *generalizing*
  inversion, not train-vocabulary memorization.
- Connects [[sweep-controls-findings]] (resid_post leaks at every depth) and answers the open
  question from [[info-efficient-attacks-findings]]: a 250-epoch MLP *lost* to ridge at depth **under
  noise**; on **plaintext** depth the learned inverter is competitive and *wins* at the deepest layer.
- Threat-model coverage: aloepri `nn`/`isa`(=ridge)/`ima_paper_like`(=mlp2) all implemented & swept;
  the 2507.16372 white-box per-sample optimization attack is *cut* (out of scope for a per-position
  TTRSR pipeline — mlp2 is the tractable amortized proxy); the black-box transfer attack is
  *not-applicable* under WEIGHTS-PUB.

## Certificate Lemma (proof inline — cross-model proof-checker PASS, 3 rounds)

Recovery is not merely descriptive: the raw retrieval accuracy lower-bounds extractable token
information (Fano), and the shuffle-subtracted selectivity operationally falsifies irreversibility.

### Lemma (depth-inversion certificate)

Let a token $t$ at a fixed position have residual-stream representation $h_\ell = f_\ell(\text{context})$
at depth $\ell$, where $f_\ell$ is the deterministic forward map of a public-weight transformer
(WEIGHTS-PUB). A retrieval inverter $g$, trained on a disjoint training vocabulary, outputs
$\hat t = g(h_\ell)$ by cosine match to the public embedding table over a candidate pool $\mathcal P$
of size $K$. Let $a_\ell = \Pr(\hat t = t)$ be the **population** top-1 accuracy on a test set whose
token vocabulary $V_{te}$ is disjoint from $V_{tr}$, and $a^0_\ell$ the accuracy of the label-shuffle
control; $s_\ell = a_\ell - a^0_\ell$.

**(i)** Let $\Phi(a) = H(t) - H_b(1-a) - (1-a)\log(K-1)$ and $a^\*$ its unique root in $[1/K,1]$.
For any $a_\ell \ge 1/K$, $I(t;h_\ell) \ge \Phi(a_\ell)$, strictly positive whenever $a_\ell > a^\*$.
Under non-degeneracy $0 < H(t) < \log K$, $a^\* > 1/K$ strictly (in the uniform case $H(t)=\log K$,
$a^\* = 1/K$). **(ii-a)** The operational depth-irreversibility hypothesis ($a_\ell\to a^0_\ell$,
i.e. $s_\ell\to0$, beyond some depth) is falsified wherever $s_\ell$'s CI excludes 0. **(ii-b)** The
strict per-depth $I>0$ certificate additionally requires the conservative finite-sample check
$\underline\Phi(\underline a_\ell)>0$ (Corollary 1).

**Assumptions.** A1 deterministic $f_\ell$ (WEIGHTS-PUB); A2 test-time Markov chain
$t\to h_\ell\to\hat t$ ($g$ fixed before the test label); A3 vocab-disjoint $V_{te}\cap V_{tr}=\varnothing$;
A4 finite pool $K\ge2$; A5 shuffle control = an independent label permutation giving the prior-only
accuracy $a^0_\ell$; A6 non-degeneracy $0<H(t)\le\log K$.

**Notation.** $H_b(p)=-p\log p-(1-p)\log(1-p)$; $P_e=1-a_\ell$; $\log=\log_2$ (bits).

#### Proof

**Lemma 1 (Fano).** Under A2, A4: $H(t\mid h_\ell)\le H_b(P_e)+P_e\log(K-1)$.
*Proof.* Let $E=\mathbf 1[\hat t\ne t]$. Expanding $H(E,t\mid\hat t)$ two ways and using
$H(E\mid t,\hat t)=0$ ($E$ deterministic in $(t,\hat t)$): $H(t\mid\hat t)=H(E\mid\hat t)+H(t\mid E,\hat t)$.
Now $H(E\mid\hat t)\le H(E)=H_b(P_e)$. And
$H(t\mid E,\hat t)=\Pr(E{=}0)\cdot0+\Pr(E{=}1)\,H(t\mid\hat t,E{=}1)$; on $E{=}1$, $t$ ranges over the
$K-1$ pool elements $\ne\hat t$ so $H(t\mid\hat t,E{=}1)\le\log(K-1)$ (with $\log(K-1)=0$ at $K=2$).
Thus $H(t\mid\hat t)\le H_b(P_e)+P_e\log(K-1)$. By A2 and the data-processing inequality,
$I(t;\hat t)\le I(t;h_\ell)$, i.e. $H(t\mid h_\ell)\le H(t\mid\hat t)$. $\square$

**Step 1 (bound).** $I(t;h_\ell)=H(t)-H(t\mid h_\ell)\ge H(t)-H_b(1-a_\ell)-(1-a_\ell)\log(K-1)=\Phi(a_\ell)$.
Valid for any estimator forming the chain (A2). For $a_\ell\ge1/K$, larger $a_\ell$ increases the RHS
(Step 2); below chance the RHS is non-monotone, but the empirical regime has $a_\ell\gg1/K$.

**Step 2 (threshold, endpoints).** With $R(P_e)=H_b(P_e)+P_e\log(K-1)$,
$dR/dP_e=\log\frac{(1-P_e)(K-1)}{P_e}>0\iff P_e<1-1/K$, so $\Phi$ is strictly increasing in $a$ on
$[1/K,1]$ with endpoints $\Phi(1/K)=H(t)-\log K$, $\Phi(1)=H(t)$. Under A6 ($H(t)<\log K$):
$\Phi(1/K)<0<\Phi(1)$, so by IVT + strict monotonicity there is a unique root $a^\*\in(1/K,1)$ with
$\Phi(a)>0\iff a>a^\*$. Boundary $H(t)=\log K$: $a^\*=1/K$. $\square$

**Step 3 (channel decomposition).** Under A3 a memorizing strategy has no information about an unseen
test id beyond the label marginal, so on $V_{te}$ its accuracy is at most that of a prior-only
predictor; the shuffle control (A5) instantiates one with accuracy $a^0_\ell$. Hence any predictor
not reading $h_\ell$ — memorization (dead on unseen ids) or frequency — achieves $\le a^0_\ell$
(relative to the measured shuffle baseline; we do not claim $a^0_\ell$ is Bayes-optimal). So
$s_\ell>0$ (CI excluding 0) certifies $h_\ell$ contributes beyond the measured prior channel and
beyond exact train-vocabulary memorization. Empirical witness: cosine-NN selectivity 0.000 at every
depth. This attributes the *channel*; the bits in Step 1 are separate.

**Corollary 1 (finite-sample certificate).** With $\underline a_\ell$ a one-sided $(1-\delta_a)$ lower
confidence bound for $a_\ell$ and $\underline H\le H(t)$ a $(1-\delta_H)$ lower bound on the test-pool
entropy, define $\underline\Phi(a)=\underline H-H_b(1-a)-(1-a)\log(K-1)\le\Phi(a)$. For
$\underline a_\ell\ge1/K$,
$I(t;h_\ell)\ge\Phi(a_\ell)\ge\Phi(\underline a_\ell)\ge\underline\Phi(\underline a_\ell)$ with
probability $\ge1-\delta_a-\delta_H$. **Certify $I>0$ iff $\underline\Phi(\underline a_\ell)>0$.**
Replacing $H(t)$ by the *upper* bound $\log K$ is invalid (it inflates the lower bound on $I$ and can
falsely certify low-entropy $t$); a lower bound $\underline H$ is required.

**Step 5 (part ii).** *(ii-a)* The hypothesis $\exists\ell^\dagger:\forall\ell\ge\ell^\dagger,\,s_\ell\to0$
is falsified since $s_\ell\in[0.39,0.69]$ with every CI excluding 0 across $\ell\in\{0,..,32\}$; by
Step 3 this is genuine recovery. *(ii-b)* The strict $I>0$ certificate per depth follows from
Corollary 1 wherever $\underline\Phi(\underline a_\ell)>0$ — queued, hence conditional. $\blacksquare$

**Status:** PROVABLE AFTER WEAKENING; verified PASS by cross-model proof-checker (Codex gpt-5.5 xhigh,
3 rounds; trace `.aris/traces/proof-checker/2026-06-24_run01/`). Corrections vs the naive target: the
positivity threshold is $a^\*$ (not $1/K$; equal to $1/K$ only for uniform $t$); the bound is a
population statement with an explicit conservative finite-sample corollary using a *lower* entropy
bound; part (ii) splits into operational (proved) vs information-certified (conditional). Full
proof package: `refine-logs/resid-depth-inversion/PROOF_PACKAGE.md`.

## What this is, and what it is not

- **Is:** a scoped reproduction of "depth ≠ privacy" on Qwen3-4B `resid_post`, a statistically
  significant learned > linear gap at the deepest layer, a positive measurement-loop result (probe
  predicts attack across depth), and a verified Fano certificate turning raw retrieval accuracy into
  an information lower bound.
- **Is not:** a cross-model/corpus generalization (single seed); not a discharged per-depth *bits*
  certificate (ii-b's $\underline\Phi(\underline a_\ell)>0$ numeric check is queued, needs $K$ and
  $\underline H$); not a new attack and not a new information-theoretic theorem.

**Novelty (cross-model verdict, `NOVELTY_CHECK.md`):** C1/DECISION LOW (reproduction of arXiv
2507.16372); the Fano certificate is LOW-as-theory (Fano-from-accuracy is standard, e.g. arXiv
1606.05229) / MEDIUM-as-protocol (the vocab-disjoint + shuffle + conservative lower-entropy
operationalization on residuals); the strongest defensible contribution is the **positive
measurement-loop** — an attack-*independent* token-information probe tracking inversion recovery
across depth (MEDIUM; no direct prior art found). Positioned accordingly: a scoped reproduction +
leakage-measurement result, not a new attack/theorem.

## Open (queued firm-ups, not this phase)

Multi-seed/model/corpus generalization; numeric instantiation of $K$ and $\underline H$ to discharge
(ii-b) and convert raw retrieval accuracy into a per-depth bits floor; the 2507.16372 white-box per-sample
optimization attack; stronger inverter ladder.

## Connections

Builds on [[sweep-controls-findings]] and [[info-efficient-attacks-findings]] (resolves its
plaintext-depth open question). Probe-independence backbone shared with [[capacity-pvi-findings]] and
[[capacity-matched-pvi]]. Contrast with [[depth-decoupling-input-dp]] (under propagated input-DP the
same probe *decouples* from the attack with depth; here, on *plaintext*, it *tracks*). MI comparator
[[mi-monotone-gaussian]].
_Edges recorded in `graph/edges.jsonl`._

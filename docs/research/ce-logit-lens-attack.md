---
type: research
status: current
created: 2026-06-25
updated: 2026-06-25
tags: [attack, residual, token-recovery, logit-lens, tuned-lens, cross-entropy, differential-privacy]
companion: research-wiki/claims/single-position-residual-linearly-saturated.md
supersedes:
superseded_by:
---

# CE logit-lens attack — the objective-matched, fair non-linearity test

Terse design note for the cross-entropy logit-lens residual→token attack. Home:
`src/talens/attacks/dp_inversion.py` (`logit_lens_attack`); driven by `scripts/evals/dp_leakage_sweep.py`.

## Why it exists — the objective mismatch

The cosine `skip_decoder_attack` regresses to the *embedding vector* (`loss = 1 − cos(ê, E_true)`) but
is scored on *argmax over a discrete pool*. A correction can lower mean cosine error (↓ val loss)
while flipping a few test points to the wrong neighbour (↓ recovery) — the "val improves, recovery
slightly worse at depth" artifact. CE removes the mismatch: train on exactly what we score.

## What it is

A **logit lens** (read a residual through the model's frozen unembedding) with a learned head:

- query  `h(x) = A·x + b  [+ gate·MLP(x)]`  (A warm-started to ridge; gate=0 ReZero init → starts as
  the linear lens; MLP tail kept at normal init so the gate has a live gradient — see
  [the dead-branch bug](../../research-wiki/claims/single-position-residual-linearly-saturated.md))
- logits **cosine** softmax: `z = (ĥ · Ê) / τ`, query + embeddings **row-normalised**, fixed τ
  (`E` = frozen embedding/unembedding table; **no per-token trainable params**)
- loss   **cross-entropy** over **full/sampled vocab** against the true token id
- early-stop on **val top-1 recovery** (seeded with the ridge warm-start ⇒ ≥ ridge)
- decode **cosine** nearest-token over the candidate pool (identical convention to ridge)

`nonlinear=False` ⇒ the affine tuned-lens; `nonlinear=True` ⇒ + gated GELU correction.

**Why cosine, not raw dot product** (diagnosed 2026-06-25 from a failed pilot): gemma token-embedding
norms are heterogeneous, so raw-dot-product logits rank by ‖E‖ and badly under-recover unseen tokens
— a pilot CE lens hit 0.281 vs ridge 0.792 at L12. On norm-heterogeneous synthetic data: ridge/cosine
= 1.00, **init dot decode = 0.78**, init cosine decode = 1.00. And early-stopping on val *CE loss*
(misaligned with retrieval) drifts the trainable affine to 0.977; **val top-1** early-stop restores
1.00. Hence cosine-CE + cosine decode + val-top1 early-stop.

## The clean 3-way comparison (all on the SAME vocab-disjoint split)

| attack | objective | geometry | isolates |
|---|---|---|---|
| `ridge` | cosine regression | NN decode | current baseline |
| `lens` (linear CE) | cross-entropy | tied frozen `E` | effect of the *matched objective* |
| `declens` (non-linear CE) | cross-entropy | tied frozen `E` + gated MLP | effect of *non-linearity* |

"Does non-linearity help" = `declens` vs `lens` (same objective, geometry, split — only the MLP added).
This gives non-linearity its **best fair shot at generalization** with every confound removed.

## Why it stays a fair generalization test (not memorization)

Memorization here = learning "(test residual) → test token" from **test tokens as labelled positives
with their residuals**. That never happens:

- inputs = train residuals only; test residuals (`X[te]`) never fed in;
- positive labels = train tokens only; a test token is never the rewarded answer;
- the head has **no per-token parameters** — identity lives in the frozen, already-public `E`.

Test tokens enter only as **softmax negatives** (push-away targets), via their *public* embeddings — no
new token-specific information, and if anything it makes test recovery slightly *harder*. So full-vocab
negatives are **conservative-to-neutral**, not inflationary. (Maximally pure variant: *train-only*
negatives — test tokens touch training nowhere. We default to full/sampled vocab, the standard
tuned-lens recipe.) The repo's `*_sel = recovery − label-shuffle floor` is the built-in tripwire: with
the disjoint split the floor stays ~0, so any leak would surface there.

## Implementation

- **Wrinkle (mechanics only):** CE needs the true class in the softmax denominator, but train tokens
  are absent from the test pool. ⇒ train the softmax over **full vocab via sampled softmax** (candidate
  set = unique train tokens ∪ K random negatives, resampled each epoch; target index via
  `searchsorted`), then **eval cosine nearest-token over the test pool** only. Standard "train
  classification / evaluate retrieval". This says nothing about generalization.
- Sampled-vocab gather (`E[cand]`) is moved to GPU per step (~tens of MB for cand ≈ train-tokens +
  K negatives), so the 2.3 GB full table need not stay resident. Per-step cost ≈ one extra
  `n_train × |cand|` matmul vs the cosine decoder; the per-epoch CPU→GPU gather is small relative to
  that matmul but is **measured in a scoped saturation pilot before the full grid** (perf gate), not
  assumed.
- Eval passes `ytr` (train token ids) and `full_emb` to the attack; for the floor it passes the *same*
  permutation-shuffled `ytr`, so the memorization floor is computed correctly. `ridge`/`decoder`
  swallow the extra kwargs.

## Outcomes

- `declens ≈ lens` → affine-saturation settled **rigorously** (non-linearity given its best fair shot,
  fails) → the only path to more recovery is *information* (multi-position context, LM prior), not a
  bigger per-vector net.
- `declens > lens` → genuine non-linear generalization exists → the residual surface re-opens as a
  richer-attack target. (Either way, decisive on the methodologically clean split.)

## Definitions

- **Logit lens** — read an intermediate residual through the model's *final* unembedding `U` to get
  vocab logits `U·h` (nostalgebraist). **Tuned lens** (Belrose 2023, arXiv:2303.08112) inserts a
  learned affine first, CE-trained: `U·(A·h+b)`.
- **WEIGHTS-PUB** — adversary knows the weights + embedding table; norms/Grams/`softmax(QKᵀ)` and the
  full `E` are known functions of the secret. The motivating threat model here.
- **Vocab-disjoint split** — train/test partitioned so they share *no token*; the fair test of whether
  a learned residual→query *function* generalizes to unseen tokens (removes memorization, the
  opposite of generalization).
- **Sampled softmax** — approximate full-vocab CE by summing the denominator over {positives ∪ K
  sampled negatives} instead of all V classes.
- **`*_sel`** — recovery minus a clean-rep label-shuffle floor (per layer); the project's
  memorization-subtracted leakage readout.
- **ReZero init** — gate scalar = 0 (branch starts off) but the gated sublayer keeps normal init, so
  the gate has a live gradient (zeroing both pins the branch ≡ baseline).

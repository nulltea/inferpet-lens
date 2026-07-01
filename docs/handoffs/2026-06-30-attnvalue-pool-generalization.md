---
type: handoff
status: current
created: 2026-06-30
updated: 2026-06-30
tags: [aloepri, isa-attnvalue, kqv_out, partial-tau, candidate-pool, generalization, inverter]
companion: research-wiki/experiments/aloepri-partial-tau-bootstrap.md
---

# Handoff — why ISA-AttnValue (kqv_out) doesn't bootstrap like ISA-HiddenState / IMA-EmbedRow

## What this session shipped (committed, don't redo)

- `cda69b6` — §D.1 gradient-opt ISA (`isa_grad_attack` in `src/talens/attacks/dp_inversion.py`),
  wired into `aloepri_partial_tau_sweep.py` (`ATTACKS={ridge,decoder,isa_grad}`), regression test,
  FIG·01b "ISA-HiddenState · gradient" subplot in `docs/html/static-obf.html`. Finding: gradient ≈ ridge
  within a few pts, both directions; same privacy verdict. Also **corrected FIG·01b kqv_out prose** to say
  its ≤0.085 is a candidate-pool effect, not structural non-invertibility.
- `40e1a2b` — retired the ROCm container; host `.venv` runs gfx1151 torch directly. Setup/troubleshooting
  now in `~/docs/torch-gpu.md` (installer `~/scripts/install_rocm_torch.sh`, untested install path);
  `run_in_rocm.sh` + `talens-rocm:latest` removed; `rocm/pytorch` base kept (torch source, shared w/ AloePri).

## Deferred idea (park it — user said defer)

You can get a classification "pick-the-token" objective **without** a learned head: use the **frozen
embedding table as the head** and train cross-entropy (`logits = E·q`). Keeps held-out generalization
(every token's head row is the known public embedding) AND gives a CE loss. This is exactly what
`logit_lens_attack` does — but it's only wired into `scripts/evals/dp/dp_leakage_sweep.py`, **not** the
AloePri roster, and it came out ≈ ridge (affine-saturated). Could be applied to ISA-HiddenState /
IMA-EmbedRow to maybe beat ridge; **deferred**, low expected payoff given affine saturation.

## The settled understanding (so the next agent doesn't relitigate it)

ISA-HiddenState and IMA-EmbedRow are correct and their recovery is real: both **generalize** to held-out
token types (residual L0 ~0.882, embed-table ~0.989), and the number is **pool-insensitive** (same under
any candidate pool) because the residual ≈ the token's own embedding (per-token-invertible) and the table
is literally linear `(W+noise)·P̂`.

**The harvested k pairs are used as TRAINING DATA in all three attacks, identically** — see
`cascade_attack` (`dp_inversion.py:69`): rows whose token ∈ harvested_types = train rows; the inverter is
fit on those (rep → true-token-embedding) pairs. For kqv_out the in-set `harvested` recovery is ~0.998, so
it *does* train and *does* memorize the seen tokens. **The earlier phrasing "harvested tokens not used in
training" is a misread — they ARE the training set for kqv_out too.**

The kqv_out gap is a **second, incidental use of the same token IDs**: the eval **candidate pool**
(`prepare()` builds `pool = _pool(unique(y), …)` = harvested + held + random filler, padded to 2048) puts
the harvested IDs in the decode answer-set as *distractors*. Decode = cosine-NN of the predicted embedding
against `table[pool]` (frozen table as head, no learned classifier). Measured (single capture, α=1.0,
controlled — only the pool changes):

| kqv_out keymat L0 | pool = held-out only | pool = all types (incl. harvested) |
|---|---|---|
| random 70/30 split | 0.627 | 0.068 |
| freq-harvest k=1024 | 0.584 | 0.085 |

Both pools are size 2048 with ~1652 distractors; all-types just **substitutes ~921 random distractors for
~921 harvested-token distractors**. Recovery still collapses 9× → the harvested tokens are *special
attractors*, not just "more candidates". FIG·02 (`aloepri_score_surface_sweep.py`) uses pool = test types +
**random** fill (harvested mostly absent) → reports ~0.46–0.58. private-rag `run_isa.py` likewise. So
FIG·01b cascade is the odd one out **only** because its pool includes the harvested tokens.

## The open question for the next session (user's framing)

Why does ISA-AttnValue not behave like the others? Mechanism hypothesis (needs confirming): kqv_out is the
attention **output** — a context-weighted mix across positions, **not** a per-token-invertible function —
so the ridge map memorizes harvested tokens (0.998) but for held-out inputs its predictions land in the
harvested-token embedding region rather than on the correct held-out token. The harvested tokens in the
eval pool then "steal" the nearest-neighbour → 0.07. The held-out-only pool (0.58) is weak *real*
generalization that only surfaces once the memorized confusers are removed.

**Is the all-types pool making the attack artificially weaker, or is it the honest measurement?** This is
the crux to resolve:
- For the bootstrap question ("does a τ-harvest let you read held-out tokens?") the realistic decode is the
  **full vocabulary** (attacker doesn't know which token is held-out) → that's ⊇ all-types pool → ≤ 0.07.
  So the honest answer is "kqv_out does not bootstrap"; FIG·02's 0.58 is pool-flattered.
- But one could argue the held-out-only pool is the right *generalization* probe. Decide and document.

## Concrete next steps (offered, NOT yet run/applied — awaiting decision)

1. **Full-vocab decode control** for held-out kqv_out (expect ~0.05–0.07) → the honest number.
2. **Equal-count random-vs-harvested distractor control** (same N, swap identity) → nails "harvested are
   attractors" (should reproduce 0.58 vs 0.07). Reuse one capture; ~3 min GPU.
3. Decide pool policy for the cascade `unharvested` metric (held-out-only vs full-vocab vs both) — a ~2-line
   change in `prepare()`/`cascade_attack`. Residual/table cells won't move; only kqv_out.
4. Correct the FIG·02 / FIG·01b writeups + `research-wiki/experiments/aloepri-partial-tau-bootstrap.md` to
   state kqv_out's apparent leak is memorization, pool-dependent (not a real held-out leak).
5. (Separate, deferred) isa_grad multi-seed firm-up; CE-frozen-head variant for ISA-HiddenState/IMA-EmbedRow.

## Pointers

- `src/talens/attacks/dp_inversion.py` — `cascade_attack` (:69), `ridge_attack`, `skip_decoder_attack`,
  `isa_grad_attack`, `logit_lens_attack`, `nearest_token`.
- `scripts/evals/static_obf/aloepri_partial_tau_sweep.py` — `prepare()` builds the all-types pool; FIG·01b source.
- `scripts/evals/static_obf/aloepri_score_surface_sweep.py` — FIG·02 (matched ridge, random-fill pool).
- `docs/html/static-obf.html` — FIG·01b/02; kqv_out prose already corrected to "candidate-pool effect".
- GPU runs: just `.venv/bin/python …` (no container). One GPU process at a time.

Suggested skills next session: none mandatory — steps 1–2 are direct GPU runs; use `/result-to-claim` only
if turning the kqv_out-memorization finding into a wiki claim, and `/auto-review-loop` before re-rendering
the report.

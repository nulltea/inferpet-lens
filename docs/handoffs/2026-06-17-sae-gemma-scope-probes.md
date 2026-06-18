---
type: handoff
status: current
created: 2026-06-17
updated: 2026-06-18
tags: [SAE, gemma-scope, gemma-2-2b, sae-lens, probe, spike, v-information, triage, attribute-inference, lumia, membership-inference]
companion: [sae-private-inference-attack-design-space, sae-as-confidential-inference-attack]
---

# Handoff: Gemma-scope SAE probes on gemma-2-2b — spike + deferred work

## Where we are

Adding SAE-based measurement to the talens pipeline, tested on **gemma-2-2b**
with **gemma-scope** SAEs. A grilling session decomposed "SAE probe" into
three distinct measurement programs (they had been conflated). Decision: do a
standalone **spike** on program B's core question first; **defer program C**
(this doc). Programs A and B stay on the roadmap.

## The three programs (do not re-conflate)

| | A — Triage by firing | B — SAE-basis leakage | C — Attribute probe (DEFERRED) |
|---|---|---|---|
| Idea | Hand-pick privacy-relevant **pre-labeled** gemma-scope features (email, wallet, phone via Neuronpedia) → flag a prompt when any fires | Feed `Z = sae.encode(x)` into PVI/MDL on **token-id**, compare to dense `x` | Train a probe over **all** of `Z` to predict a **PII attribute** label |
| Uses feature labels? | Yes | No | No |
| Training? | None (zero-shot) | Probe is the measure | Trained classifier |
| Labeled data | Eval only (AUC) | None | **Train** (AI4Privacy) |
| Maps to synthesis goal | #3 triage | #2 / inversion-recoverability | #1 attribute-inference ("are SAEs useful", the 2.2% result) |
| Metric | detection AUC vs "prompt has secret" | ΔV-info(Z vs x) | probe AUC/macro-F1 vs **linear baseline on x** |

Key correction made during grilling: the user's interpretation #1 == **A**
(read off pre-labeled features). The grilling had drifted into **C** (training
on AI4Privacy) by mistake — that is a *different* program and is now deferred.

## Conceptual guardrail (carry this forward)

`Z = sae.encode(x)` is a **deterministic function of x** ⇒ by data-processing,
Shannon `I(Z; y) ≤ I(x; y)`. So **raw-MI framings (esp. CLUB) on Z are near
meaningless** — the SAE can only preserve/destroy information, never add it.
The only meaningful question is **usable / V-information**: does a *bounded*
probe decode the secret *more easily* from sparse `Z` than dense `x`? That is
what the spike measures. **Drop CLUB for the SAE branch.**

## Libraries to reuse (do not reimplement)

- **gemma-scope weights**: `google/gemma-scope-2b-pt-res` (JumpReLU SAEs, all
  layers, widths 16k/65k, "canonical" picks). Matches our `resid_post` surface.
- **SAELens** (`sae-lens>=6`): `SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical",
  sae_id="layer_{L}/width_16k/canonical")`; works **standalone** on raw
  activation tensors (`sae.encode`) — no TransformerLens needed, slots behind
  our nnsight capture. Added to `Containerfile` + `pyproject` `[sae]` extra.
- **`sae-probes`** (github.com/sae-probes/sae-probes): the protocol + linear
  baseline from "Are SAEs Useful?" — reuse for program C's comparison.
- **Our own** `measures/_probe.train_softmax_probe`, `measures/vinfo.v_information`,
  `capture/` — the SAE probe is the existing probe on `Z` instead of `x`.

## What landed in this session (code)

- `scripts/spikes/sae_vinfo_spike.py` — standalone spike. Captures gemma-2-2b
  `resid_post` (no attention patch needed — `resid_post` is HF `hidden_states`,
  so gemma-2-2b works through the existing nnsight path out-of-the-box; the
  qwen3 eager-attn patch is only for `kq`/`kqv_out`). Then `Z = sae.encode(X)`,
  and compares `v_information(Z, token_id)` vs `v_information(X, token_id)` per
  layer, with shuffle control + L0 sparsity. Writes JSON + prints mean Δ bits.
- `Containerfile`, `pyproject.toml` — added `sae-lens`.

## BLOCKER before running

- **gemma-2-2b and gemma-scope are gated** on HF, and there is **no HF token**
  in the env. User must authenticate once:
  `! huggingface-cli login` (after accepting the Gemma terms on the model pages
  for `google/gemma-2-2b` and `google/gemma-scope-2b-pt-res`), or export
  `HF_TOKEN`. The ROCm container bind-mounts `~/.cache/huggingface`, so a host
  login is visible inside.
- Heavy ⇒ run **only** in the container:
  `scripts/run_in_rocm.sh python3 scripts/spikes/sae_vinfo_spike.py --corpus corpora/release-gate-512.txt --max-prompts 256 --layers 5,12,20 --out results/sae_vinfo_spike.json`
  (first run rebuilds the image to pick up `sae-lens`).

## Spike: how to read the result

- `Δ = vinfo_sparse − vinfo_dense` per layer. `Δ > 0` ⇒ the sparse overcomplete
  basis is *more* linearly decodable for token-id (SAE helps the bounded
  attacker); `Δ < 0` ⇒ the SAE bottleneck *cost* usable signal. Literature
  prior (2.2% result) predicts `Δ ≲ 0` for most layers — confirming that would
  be the headline.
- Sanity: shuffle-control V-info ≈ 0 for both; if not, probe is memorising.

## DEFERRED — Program C (record per user)

Train a probe over `Z` to predict **per-token PII type** vs a **linear-probe
baseline on `x`**, replicating "Are SAEs Useful?" on our stack. Open decisions
captured before deferral:

### LUMIA = the dense-`x` head of Program C (matrix attack #3)

The **linear-probe membership/attribute attack** in
`it-leakage-estimation-set.md` (#3, **LUMIA**, arXiv
[2411.19876](https://arxiv.org/abs/2411.19876)) is **the same machinery as
Program C's baseline**, not a separate effort — fold them:

- Program C's load-bearing baseline (`linear-probe on x`) **is** the LUMIA
  attack. LUMIA's contribution is precisely the **per-layer sweep** ("which
  layer is most attackable") — which is already our sweep axis. Its headline:
  **+15.71% AUC over prior MIA SOTA** (unimodal), AUC>60% in 65% of cases.
- Two probe **heads off the same activations / same pipeline**:
  - **attribute** head → PII-type label (Program C, AI4Privacy) — graded by
    AUC / macro-F1.
  - **membership** head → in/out-of-training label (LUMIA proper) — needs a
    shadow in/out split, not AI4Privacy. This is the only extra data piece.
- Shared blockers already listed below transfer verbatim: the **vocab control
  and retrieval-family measures are token-id-specific and do NOT apply** to an
  attribute/membership target (no embedding candidate pool); scope them out for
  both heads.
- Build order: ship the **attribute head first** (drop-in `y` swap, reuses
  AI4Privacy), add the **membership head** only once a shadow-set protocol is
  decided. The SAE-`Z` vs dense-`x` *gap* is the contribution for the attribute
  head; the per-layer AUC curve is the contribution for the membership head.

- **Probe abstraction**: generalize `measures/_probe.py` so the target is
  pluggable (token-id *or* attribute). Note the **retrieval family** (vinfo/mdl
  `_retrieval`) and the **vocab control** are token-id-specific (embedding
  candidate pool) and **do not transfer** to attribute targets — scope them out.
- **Data**: `ai4privacy/pii-masking-300k` (char spans + per-token BIO labels,
  50+ classes, permissive). Needs char-span → gemma-subword-token alignment;
  decide multi-token-span labeling (all positions vs last).
- **Prediction unit** (was the open fork at deferral): per-token PII-*type*
  (recommended — drop-in y swap, subsumes binary) vs per-token binary vs
  per-prompt presence (needs new pooling). Report **binary-collapse AUC +
  multiclass macro-F1** off the same labels.
- **Imbalance**: >90% of positions are non-PII ⇒ subsample `none`, never report
  raw accuracy.
- **Baseline is load-bearing**: the contribution is the *gap* SAE-probe(Z) −
  linear-probe(x); always run both on identical (rows, y).

## Roadmap after spike

1. Spike (B core, token-id) — pending HF auth.
2. If spike interesting → integrate B as a measure (`Z` as a `Transform`-like
   encoder feeding existing PVI/MDL; drop CLUB). Add an "SAE" row to the
   attack×measure matrix in `it-leakage-estimation-set`.
3. Program A (triage): pick privacy features by Neuronpedia label → firing
   detector → AUC on a PII-labeled eval set (AI4Privacy can serve as eval).
4. Program C (deferred, above).

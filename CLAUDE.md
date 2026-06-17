# Repository conventions for Claude

`transformer-attacks-lens` — an information-theoretic, interpretability-
grounded study of confidential-inference attacks on transformers. See
`README.md` for the premise and document map.

## Always run heavy workflows in the GPU container

The host `.venv` has a **CPU-only torch wheel** (`torch …+cpu`) — it is for
fast model-free unit tests (`pytest`) only. **Never** run capture, the
PVI/MDL probe, CLUB, or the inversion attacks (i.e. `talens.cli`,
`calibrate_capture`, anything touching real Qwen3 activations) under the
`.venv`: it silently falls back to CPU and is unusably slow.

The GPU is an AMD Strix Halo iGPU (gfx1151, "Radeon 8060S"); reach it only
through the ROCm container. Wrap **every** heavy command in
`scripts/run_in_rocm.sh` (auto-builds `talens-rocm:latest` on first use,
bind-mounts the repo + HF cache at identical host paths, exposes
`/dev/kfd`+`/dev/dri`):

```bash
scripts/run_in_rocm.sh python3 -m talens.cli \
    --corpus corpora/dev-24.txt --control all --out results/run.json
# sanity: scripts/run_in_rocm.sh python3 -c 'import torch; print(torch.cuda.is_available())'
```

`pytest` (synthetic, model-free) stays on the host `.venv`. See
`Containerfile` for why the base image is AMD's `rocm/pytorch` (gfx1151
kernels) rather than a pip torch wheel.

## What this repo is

A research repo: surveys, the measurement-method lineage, an experimental
plan, and (as it grows) the estimation code that fits information-theoretic
measures to attack-success ground-truth. The headline thesis and the
decided attack×measure matrix live in `docs/plans/it-leakage-estimation-set.md`.

## Scheme-agnostic by design

The library asserts **nothing** about any confidential-inference defense.
It measures the invertibility/leakage of representations and runs attacks
on whatever it is handed. A "defense" is an external, pluggable
`talens.transforms.Transform` (`Tensor → Tensor`); the only built-in is
`Identity` (the plaintext model). Do not bake a specific cover, noise
scheme, or threat model into the core — keep covers/defenses in callers,
tests, or downstream repos.

## Threat-model context (motivating example, not an assumption)

The work originated in the **private-rag** / GELO split-TEE ⟷
untrusted-GPU project, whose conservative `WEIGHTS-PUB` adversary (knows
weights + embeddings, so rotation-/permutation-*invariant* quantities —
a norm, a Gram, `softmax(QK^T)` — are known functions of the secret) is
the *motivating* use case and a good source of test defenses. But the
library itself makes no such assumption; it is the substrate one would
use to *evaluate* whether a given scheme leaks. When a doc invokes
`WEIGHTS-PUB`, it is framing a motivating example, not a library invariant.

Representations of interest as the "secret" to recover: activations /
hidden states, Q·K·V, attention scores, the KV-cache, and the
prompt/tokens they encode.

## Markdown docs (`docs/**/*.md`)

All markdown docs under `docs/` carry YAML frontmatter:

```yaml
---
type: <handoff|plan|prototype-note|research|theory|dev-log|reference>
status: <current|partial|stale>
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: []
# Optional: superseded_by, supersedes, companion, archive_reason
---
```

Folder mapping (by `type`):

- `handoff`        → `docs/handoffs/YYYY-MM-DD-<slug>.md` (filename date = last update)
- `plan`           → `docs/plans/`
- `prototype-note` → `docs/dev/prototype/`
- `research`       → `docs/research/`
- `theory`         → `docs/research/`
- `dev-log`        → `docs/dev/logs/`
- `reference`      → `docs/plans/` (no dedicated folder until critical mass)

When a handoff is no longer in active reference (more than a few days old
and not driving current work), move it to `docs/archive/handoffs/`. Plans
and other docs that go stale stay in place with `status: stale` and
`archive_reason` set. When one doc supersedes another, set `superseded_by`
on the older and `supersedes: [<slug>, …]` on the newer; for partial
overlap use `companion: [<slug>, …]`.

`companion:` lists should reference docs **in this repo**. Cross-repo
pointers (to private-rag artefacts) go in prose, not frontmatter.

## Design / research docs need a Definitions section

Every `docs/research/` document includes an acronym/term glossary
(Definitions table) near the top — the information-theory vocabulary (MDL,
V-info, PVI, PID, IB, DAS/IIA) is dense and ambiguous across communities.

## Voice

Research docs describe *what is known / measured / decided* and *why*, with
comparative tables over prose-lists. Cite a paper's arXiv/DOI inline.
Citation counts are point-in-time snapshots — note that they drift.

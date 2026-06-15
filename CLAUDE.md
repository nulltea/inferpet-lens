# Repository conventions for Claude

`transformer-attacks-lens` — an information-theoretic, interpretability-
grounded study of confidential-inference attacks on transformers. See
`README.md` for the premise and document map.

## What this repo is

A research repo: surveys, the measurement-method lineage, an experimental
plan, and (as it grows) the estimation code that fits information-theoretic
measures to attack-success ground-truth. The headline thesis and the
decided attack×measure matrix live in `docs/plans/it-leakage-estimation-set.md`.

## Threat-model inheritance

This work inherits the split-TEE ⟷ untrusted-GPU threat model from the
originating **private-rag** / GELO project. The load-bearing assumption is
**`WEIGHTS-PUB`** (the conservative default): the adversary knows the model
weights and embedding tables, so every rotation-/permutation-*invariant*
quantity (a norm, a Gram, `softmax(QK^T)`) is a *known* function of the
secret activations — an algebraic anchor for reconstruction. Any leakage
measure or defence claim must report the `WEIGHTS-PUB` number as the bar;
`WEIGHTS-BLIND` (private fine-tune) is the optimistic case only.

**Secret:** user activations / hidden states, Q·K·V, attention scores, the
KV-cache, and the prompt/tokens they encode. **Public:** weights and
embedding tables.

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

---
type: research
status: current
created: 2026-06-15
updated: 2026-06-15
tags: [methodology, train-test-split, vocab-disjoint, V-information, MDL, calibration, probing]
companion: [it-leakage-estimation-set, mdl-vinfo-inversion-toolkit]
---

# Attack & measure setting: the train/test split regime

The calibration thesis — *an information-theoretic measure predicts an
inversion attack's recovery rate across layers* — only holds cleanly if
the measure and the attack describe **the same quantity over the same
data regime**. The split regime (how token-rows are partitioned into
train/test) is exactly that knob, and the attacks and the class-probe
measures want *different* regimes by construction. This doc pins down
why, what it confounds, and the two resolutions the codebase implements.

## Definitions

| Term | Meaning |
|------|---------|
| **token-row** | One captured representation vector (a residual-stream row, or a flattened attention-score row) paired with its ground-truth token id. |
| **row-split** | Shuffle *rows*. Train and test **share the token vocabulary** — the same id appears on both sides in different contexts. (`splits.train_val_test_split`) |
| **vocab-disjoint split** | Partition the *distinct token ids* into disjoint sets; assign each row to the set holding its id. Train and test **share no id** — the model must reconstruct tokens unseen in training. (`splits.vocab_disjoint_train_val_test_split`) |
| **class-probe family** | Predictive family `V` = multinomial-logistic `q(y\|x)` over a fixed class set. Used by the textbook PVI / MDL measures. |
| **retrieval family** | Predictive family `V` = ridge map `X→embedding` + softmax over cosine similarity to a candidate token-embedding pool. This *is* the inversion attack, recast as a probabilistic model. |

## Why each component is forced into a regime

| Component | Regime | Why it has no choice |
|-----------|--------|----------------------|
| Inversion attacks (`hidden_state`, `attn_score`) | **vocab-disjoint** (default) | The ridge inverter maps `X → token embedding`, then cosine-matches a candidate pool. Recovery flows through **embedding geometry**, so it generalises to unseen ids. Vocab-disjoint is the honest attacker — it forbids memorising a per-token lookup table. |
| **PVI, MDL** (class-probe) | **row-split** (forced) | A softmax classifier can only place mass on classes seen in training. Under vocab-disjoint every test id is an **unseen class** → `q(y_test\|x)≈0` → cross-entropy diverges → V-info collapses to ~0 and MDL → uniform, *no matter how much the representation leaks*. Vocab-disjoint is incoherent for a class probe. |
| **CLUB** | regime-agnostic | Estimates `I(representation; embedding)` over continuous vectors — no class notion, unaffected. A useful tie-breaker: it should track recovery under either regime. |

## The confound

Row-split lets the class probe exploit a **memorisation signal**:
"representations near *this* cluster tend to be token #4823, which I have
seen." That signal inflates PVI/MDL but contributes **nothing** to
vocab-disjoint attack recovery (the attack never trained on #4823). So a
layer can read:

- **high PVI** (row-split: probe memorises seen tokens well) yet
- **low attack TTRSR** (vocab-disjoint: the embedding map doesn't
  generalise) — or the reverse.

Regressing `measure → recovery` across such layers mixes the **true
leakage** signal with the **regime mismatch**. A weak R² would then be
uninterpretable: bad predictor, or just a different regime? This is the
confound the calibration must not carry.

## The two resolutions (both implemented)

### A — run the attack in row-split (the first-run choice)

Run the inversion attacks with `split_mode="row"` so measure and attack
share the regime: apples-to-apples. One-line change in the orchestrator
(`cli.run_pass1(..., attack_split_mode="row")`).

- **Pro:** cheapest; gives a clean first calibration number.
- **Con:** attack recovery now includes memorisation, so it **inflates**
  and is a *weaker* privacy statement than the honest vocab-disjoint
  attacker. Always also report the vocab-disjoint attack number as the
  real threat bar.

### B — retrieval-family measures (the principled target)

Replace the class probe in PVI/MDL with the **retrieval family**: fit the
ridge `X→embedding` map, define `q(y|x) ∝ exp(cos(ŷ, E[y])/τ)` over the
candidate pool (temperature `τ` fit on validation), null model
`q(y|∅)` = the same softmax around the mean target embedding. Then:

- PVI under this family **is literally "the attack's usable
  information,"** realising the plan's framing that for the inversion
  attacks *the probe is the V-family*.
- It **generalises to unseen ids** (retrieval, not classification), so it
  runs **vocab-disjoint throughout** — same regime as the honest attack.
- Implemented as `measures.vinfo.v_information_retrieval` and
  `measures.mdl.online_code_length_retrieval`, sharing
  `measures._retrieval`. The class-probe functions stay as the textbook
  baseline.

### C — run both regimes (future rigour)

Report calibration under both row-split (class) and vocab-disjoint
(retrieval) and treat the regime as a studied variable. Most defensible
in the writeup; ~2× compute. Deferred.

## Decision

- **First real run: A only** — attacks in row-split, textbook class-probe
  PVI/MDL, for a fast clean apples-to-apples calibration; report the
  vocab-disjoint attack alongside as the honest threat bar.
- **B is implemented and tested** (retrieval-family measures) and is the
  principled path for the vocab-disjoint calibration; promote it to the
  headline once the first run validates the pipeline.
- CLUB runs under both as the regime-agnostic cross-check.

## Pointers

- Splits: `src/talens/splits.py`.
- Class-probe measures: `measures/vinfo.py::v_information`,
  `measures/mdl.py::online_code_length`.
- Retrieval-family measures (B): `measures/_retrieval.py`,
  `measures/vinfo.py::v_information_retrieval`,
  `measures/mdl.py::online_code_length_retrieval`.
- Orchestrator regime knob: `cli.run_pass1(attack_split_mode=...)`.

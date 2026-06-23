---
type: reference
status: current
created: 2026-06-23
updated: 2026-06-23
tags: [component-topology, attack, probe, defense, schema, threat-model, metric-convention, workflow]
companion: [it-leakage-estimation-set]
---

# Component topology — attacks × probes × defenses

The shared vocabulary and field schema for every component in this repo. CLAUDE.md's
"Research operating model" section points here; read this on-demand whenever you add an
attack, a probe (measure), or a defense, so a new component is described with the same
fields as the existing ones. The headline thesis and the decided attack×measure matrix
live in [`it-leakage-estimation-set.md`](it-leakage-estimation-set.md).

## Definitions

| Term | Meaning |
|---|---|
| **Target surface** | the representation an attacker observes / a defense protects: residual-stream hidden state at layer ℓ, KV-cache, Q·K·V, attention scores `softmax(QKᵀ)`, pooled sentence embedding, logits. |
| **Secret** | what is recovered: token ids, the prompt text, an attribute/membership bit, or a vocab/permutation key. |
| **Probe** | an *independent* information measure that **correlates with** an attack's recovery — never the attack reporting bits instead of recovery (see invariant below). Reports in **bits** (canonical) + a per-secret human readout. |
| **Defense (scheme)** | an external `talens.transforms.Transform` (`Tensor→Tensor`) applied to a surface; the library ships only `Identity`. |
| **WEIGHTS-PUB** | the motivating adversary: knows weights + embeddings, so rotation-/permutation-*invariant* quantities (a norm, a Gram, `softmax(QKᵀ)`) are known functions of the secret. The default threat-model anchor; per-component deviations are documented in the component's own spec. |
| **Recovery** | the attack's graded success metric (the regression target a probe predicts): token-F1, top-k, BLEU/ROUGE, recovery-rate, AUC, cosine. |

## The probe ≠ attack invariant (integrity-critical)

A probe is an **independent** quantity that *correlates with* attack recovery across a
sweep. A measure that simply re-runs the attack and reports its bits instead of its
recovery rate is **not a probe** — it is the attack in disguise, and a "correlation" with
it is circular. The scientific payload of this repo is: *does an attack-independent IT
measure predict a separately-run attack's recovery?* When in doubt, ask: *could this probe
be computed without ever running the attack?* If no, it is not a probe.

This is why geometry-only / matched-channel probes are preferred (e.g.
[[bnn-error-bounds-bhattacharyya-fano]], [[spectral-channel-mi-probe-decision]]) and why
class-PVI overfit is a standing hazard ([[pvi-fix-priority]]).

## Metric convention — bits canonical + per-secret readout

Every probe reports two things:

1. **Canonical: bits** — MI / V-information / channel capacity / SDL, on a common scale so
   probes are mutually comparable and bound attacks the same way. Small absolute
   differences (the "1/100 of a bit" illegibility) are a *presentation* problem, not a
   measurement one — keep bits as the stored value.
2. **Per-secret human readout** — rendered alongside, chosen by secret type:

| Secret type | Human readout |
|---|---|
| token id | perplexity (`2^{bits}` style) + top-k accuracy |
| prompt text | token-F1 / ROUGE / BLEU |
| vocab / permutation key | recovery rate, Kendall-τ |
| pooled embedding (text behind it) | token-F1 / cosine |
| attribute / membership | AUC |

Tables show **both axes** (bits = comparable; readout = intuitive). The reporting layer
that renders this lives under `src/talens/` (built/standardized in the campaign's pilot
metric-standardization phase).

## Attack schema

Each attack (a module under `src/talens/attacks/`) is described by:

| Field | Values / notes |
|---|---|
| target surface | which representation it consumes |
| secret | what it recovers |
| family / mechanism | linear ridge/regression · learned MLP/transformer inverter · iterative optimizer/corrector (Vec2Text) · adapter+decoder (Rep2Text) · SAE · blind-source-separation (JADE/JD/ICA) · nearest-neighbour · invariant/algebraic · accumulation/side-channel (Gram fingerprint) |
| hyperparameters | probe/inverter dim, #layers, normalization, #observations T (for accumulation) |
| threat model | default WEIGHTS-PUB; note deviations (e.g. needs aux corpus, needs `(W_plain,W_obf)` pair) |
| graded metric | the recovery target a probe predicts |
| split regime | row-split (matches class probe) vs vocab-disjoint (honest generalising attacker); see `docs/research/attacks_setting.md` |

**Ported (clean array math, no vendored `sys.path`):** `hidden_state` (IMA/ISA ridge),
`attn_score` (ISA), `cover_break` (anchor ridge), `vocab_match` (VMA / Hidden-No-More).
**Note (threat-model NA):** aloepri `sda`/`tfma` operate on a token-id sequence and `ia`'s
weight-axis on obfuscated weights — neither crosses the boundary under the activation /
open-weight threat model; document as `not_applicable`, do not force a phase.

## Probe (measure) schema

Each measure (a module under `src/talens/measures/`):

| Field | Values / notes |
|---|---|
| target surface | what it is computed on |
| quantity | PVI / conditional V-info · MDL online-code + SDL · CLUB (MI upper bound) · PID · capacity-matched PVI · spectral channel-MI · Bhattacharyya/Fano error bounds |
| canonical unit | **bits** (always) |
| readout | per-secret (table above) |
| attack-independence | how it is computed *without* running the attack (the invariant); state it explicitly |
| cost | probe-family compute (minimize per [[pvi-fix-priority]]; MDL costs 6–7× PVI) |

## Defense (scheme) schema

A defense is a pluggable `Transform`; defenses live in callers/`scripts/defenses/`, never
baked into the core (scheme-agnostic by design). Each:

| Field | Values / notes |
|---|---|
| surface(s) | what it transforms |
| secret(s) protected | may be multiple |
| mechanism class | static obfuscation · dynamic/hybrid obfuscation · split inference · differential privacy (noise) · cryptographic |
| parameter(s) to sweep | privacy budget ε/σ · permutation/block entropy · shield rank/fraction/energy · split layer · cloak block size · MI-budget |
| formally analysed? | yes/no — is there a leakage/converse argument, or only empirical? can we improve on it? |
| threat model | adversary assumed |

**Implemented:** `aloepri`, `shredder` (`scripts/defenses/`). **Campaign targets:**
KV-CLOAK (KV, perm+rotation+mask), SGT/Stained-Glass (embedding, learned MI-budget noise),
split-inference/PriPert (residual, formal converse), GELO (residual/QKVO, fresh invertible
mixing + shield vectors). **Reference-only (no phase):** cryptographic schemes (Euston,
Fision) and TwinShield-full leak ~0 by construction / need HE/MPC — documented as a
zero-leakage reference, not swept.

## The research workflow loop

For a (surface × attack × probe) cell:

1. Run the attack → recovery ground-truth. Run the probe → bits (+ readout).
2. **Do they correlate** across the sweep (plaintext → defense parameters)?
   - **Yes** → write the claim ([[feedback_write_proofs_into_claims]]: verified proof goes
     in full into `research-wiki/claims/<slug>.md`), prove it, render the HTML page
     (house style: `docs/html/STYLE.md`).
   - **No** → the finding is *why*: (a) the **attack is weak / ill-equipped** — a stronger
     attack exists; or (b) the **probe is wrong** — it is not matched to the channel. Bound
     or explain the gap in theory, report it, and queue the stronger-attack / better-probe
     follow-up.
3. Negative results are first-class — record them, do not hide or manufacture a claim.

Authority over step 2's verdict belongs to a cross-model jury (`result-to-claim`,
`experiment-audit`, `proof-checker`, `auto-review-loop`), never to the agent's own
self-assessment — the outer cadence may **drive, never acquit**
(`.aris/shared-references/external-cadence.md`).

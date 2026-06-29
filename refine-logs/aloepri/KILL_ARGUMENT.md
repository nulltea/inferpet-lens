# Kill Argument Report — vocab-disjoint as the worst-case ISA-HiddenState evaluation

**Date**: 2026-06-28
**Reviewer model**: gpt-5.5 xhigh, fresh threads (no codex-reply)
**Attack thread**: 019f0e1e-fb68-7bd3-baca-ac92e6903bfa
**Adjudicator thread**: 019f0e23-b878-7b50-a49d-29e80858a523
**Verdict**: WARN (`reason_code: partial_critical_or_repeated_major`)

Note: not a LaTeX paper. The protocol (two fresh adversarial Codex threads) was run against the
methodological claim + the report `docs/html/static-obf.html` §04 and the data files.

## Claim under fire
"Vocab-disjoint split is the correct worst-case evaluation for the realistic ISA-HiddenState attacker
on AloePri; the ~0.06 multi-key blind recovery under vocab-disjoint shows the keymat defends the
residual stream against the in-model attacker."

## Net assessment
The claim is NOT killed (0 still-unresolved): the residual conclusion is sound. But the framing
overreaches. Vocab-disjoint is the right worst case for one narrow question (can a no-key residual
inverter generalize to unseen token IDs), not for the attacker's real goal (recover prompt tokens).
Crucially, the memo's own thesis also overreaches: for the residual BLIND attack the split barely
matters (row-split 0.060 vs disjoint 0.064), so vocab-disjoint is not subtracting a stronger residual
attack. The real memorization leverage the counter-argument points to is a τ-leak on the WIRE surface
(TFMA 0.52 / SDA 0.75 at ε1=∞) for a ≤20–293-token subset, not a residual phenomenon, and it does not
let the residual attacker read tokens it has not harvested τ for.

## Attack memo (verbatim)
See `.aris/traces/kill-argument/2026-06-28_run01/attack_thread.md`.

## Adjudication (per-point, claim's perspective)

| # | point | classification | severity |
|---|---|---|---|
| P1 | residual blind ~unchanged across splits (0.060 vs 0.064) | answered_by_current_text | major |
| P2 | matched ceiling high ⇒ info preserved, "defends" = access control not destruction | partially_answered | major |
| P3 | vocab-disjoint is the wrong *realistic* residual headline (row-split is the shared-vocab null) | partially_answered | critical |
| P4 | τ-leak is real but WIRE-surface (Π activation-inert), not residual | partially_answered | major |
| P5 | partial-τ memorization does NOT break the residual blind attack / recover unharvested tokens | answered_by_current_text | minor |

counts: answered_by_current_text 2, partially_answered 3, still_unresolved 0.

## The grounded verdict (synthesis)
1. **Residual ISA-HiddenState blind attack**: vocab-disjoint is NOT a stronger worst case — both splits
   give ~0.06. The blind attacker fails on both because it cannot fit the secret basis at all (no
   deployment pairs); memorization does not help it. The scoped conclusion "the keyless residual
   inverter fails on this artifact" holds under either split.
2. **Attacker's overall token-recovery goal**: vocab-disjoint is NOT the realistic worst case. The
   realistic leverage is a partial-τ frequency-analysis channel on the WIRE surface (TFMA/SDA), which
   recovers a leaked subset of high-frequency/special tokens — and is already reported under the
   memorization-inclusive distribution-aware setting. That is a different surface, not the residual.
3. So memorization IS effective for the goal — on the wire, for ~20–293 tokens — but NOT on the
   residual (blind fails both splits). Neither split is universally "the worst case"; the correct
   realistic-attacker measure is a cross-surface, τ-leak-weighted composite.

## Top action items
1. Report residual ISA-HiddenState as BOTH row-split and vocab-disjoint (row-split = realistic
   shared-vocab headline; vocab-disjoint = unseen-token stress test). They are ~equal for the blind
   attack, which is itself the point worth stating.
2. Add a composite realistic-attacker metric: per-token recovery weighted by a τ-leak prior, combining
   the wire TFMA/SDA channel, known-specials/strong-Π, and the residual blind recovery.
3. Add a partial-τ curve (0, 20, 293, 313, 1024 known pairs): measure both harvested-token recovery and
   generalization to unharvested tokens — this directly answers "is memorizing effective?".

## Recommendation
Writing-level scope fix, not a research failure: (a) stop calling vocab-disjoint "the worst case for
the realistic attacker"; scope it to "structural-generalization stress test". (b) state the residual
blind result is split-insensitive (~0.06 both ways). (c) route the realistic-attacker worst case to the
cross-surface composite (action 2). The residual measurement itself stands.

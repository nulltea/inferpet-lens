# result-to-claim verdict — resid-depth-inversion (Task 4)

**Judge**: Codex (model_reasoning_effort xhigh), 2026-06-24. Evidence pre-check: C2 Spearmans
verified; C1/decision L32 values value_not_found ONLY due to full-precision JSON vs rounded display
(numbers present, not hallucinated).

**Verdict: all three claims `partial` — scope-limited, NOT integrity/correctness.** The narrowly
scoped claims are strongly supported; jury's only ask is to state scope precisely (single model
Qwen3-4B / single corpus / single sweep seed; bootstrap CI is over the 413 vocab-disjoint test rows per layer, not over seeds;
n=9 depth points → Spearman +0.85 has p≈0.004).

- **C1 (depth ≠ privacy)** — supported for this Qwen3-4B / release-gate-512 / resid_post setting;
  not a general all-transformer claim. nonzero vocab-disjoint selectivity every depth, shuffle ≈0.
- **DECISION (learned beats ridge at depth)** — supported at L32 (mlp2 0.542 vs ridge 0.390, gap
  +0.153). NEW: bootstrap CIs are DISJOINT at L32 (mlp2 [0.494,0.591] vs ridge [0.341,0.438]) →
  statistically significant; everywhere else mlp2≈ridge (overlapping CIs). So the ridge late-layer
  drop is a linear-inverter artifact specifically at the deepest layer.
- **C2 (probe tracks recovery across depth + strength)** — cross-depth tracking supported
  (cap_acc +0.85, CLUB +0.78); "across strength" weakly supported (only 3 inverters).

**Routing**: positive-correlate branch with scoped claim. Write the claim at the jury-endorsed
scope (Qwen3-4B / resid_post / release-gate-512 / sampled depths). Confidence: medium-high for the
narrow experimental conclusion, medium for the measurement-loop result.

**Deferred (next experiments, NOT this phase)**: multi-seed/model/corpus generalization; stronger
inverter ladder; white-box per-sample optimization attack from 2507.16372; KV/attention surfaces.

# Experiment Audit Report — resid-capacity-pvi

**Date**: 2026-06-23
**Auditor**: Codex gpt-5.5 xhigh (cross-model, read-only; thread `019ef5cd-722f-7fe2-ac20-2b2645d684ca`)
**Project**: transformer-attacks-lens — capacity-matched class-PVI consolidation

## Overall Verdict: WARN
## Integrity Status: warn

No fraud pattern found. The two WARNs are reporting/scope hygiene, both fixed in
`RESULTS_STANDARDIZED.md` (see Action Items). The CRITICAL probe≠attack circularity check PASSED.

## Checks

### A. Ground-Truth Provenance: PASS
TTRSR is real held-out token recovery, not model-derived. `localdp_runner.py:61` captures real
tokenizer ids as labels; `_inversion.py:96,105` evaluates on held-out test ids and returns
`ttrsr_top1`; exact match `pred_ids.eq(true_ids)` (`ridge.py:113`).

### B. Score Normalization: PASS
Bits stored verbatim (`report.py:16,247`). `frac_top1` is a labeled clean-baseline ratio with raw
`clean_top1`/`dp_top1` stored beside it. The `1.0` Spearman values in `nondp_intervention.json` are
correlations over five intervention knobs, with the raw rows present — not self-normalized.

### C. Result-File Existence: PASS
All JSONs exist with the claimed keys/numbers. `capacity_screen.json` class-PVI shuffle `-49.7314`
(rounds to −49.7), pca-softmax dim256 shuffle `-1.9279` (−1.9). `localdp_depth_L0_5_12_20.json` clean
class floors `-44.92/-50.21/-51.36/-48.08`, cap floors `-1.25/-1.23/-1.24/-1.27`. Codex independently
recomputed the DP Spearmans (cap-acc `0.991/0.679/0.429/-0.214`, CLUB `0.964/0.964/0.893/0.286`) —
they match the standardized rounded claims. The two pre-check `value_not_found` (−49.7, −48.1) are
display-rounding of confirmed raw values, not hallucinations.

### D. Dead Code: WARN → fixed
Probe + readout are live (`diag_capacity.py:26,75`; `localdp_runner.py:154,198`; `diag_nondp.py:57,105`).
The WARN: the audited runners write raw dict fields, not serialized `LeakageReport` rows, so the
phrase "rendered via report.py" was not proven by those runner paths. **Fix:** the doc now states the
canonical rows were *re-rendered* from the raw fields through `report.py`.

### E. Scope Assessment: WARN → fixed
Scope is disclosed honestly (single model / single seed / n=7 per layer, with multi-seed/CI/cross-model
firm-ups listed). The WARN: "robust readout" and "provably diverge" overshoot single-seed small-n
evidence. **Fix:** softened to "more stable across these runs" and "diverge (observed, single-seed),
pending multi-seed firm-up".

### F. Probe ≠ Attack Circularity: PASS (critical)
`v_information_capacity` takes only `(X, y)` — never the embedding table or the attack's output
(`vinfo_capacity.py:135,162,183,199,204,214`; `_probe.py:49` uses class indices/logits only). The only
change class-PVI → cap-PVI is the PCA dim reduction. ρ(cap, retrieval-PVI)=0.66–0.76 (<0.9). The
bits↔recovery correlation is therefore not circular — the probe could be computed without ever running
the attack.

### G. Evaluation Type: real_gt + self_supervised_proxy over simulation_only defenses
TTRSR/`clean_top1`/`dp_top1` = `real_gt`; cap-PVI/CLUB/retrieval-PVI/shuffle controls =
`self_supervised_proxy`; input-DP / non-DP interventions = `simulation_only` defense conditions with
real-token ground-truth outcomes.

## Action Items (all applied)
- Soften "rendered via report.py" → "re-rendered from raw fields through report.py". ✅
- Soften "robust readout" → "more stable across these runs". ✅
- Soften "provably diverge" → "diverge (observed, single-seed), pending multi-seed firm-up". ✅

## Claim Impact
- `capacity-matched-pvi`: **supported (scoped)** — accuracy readout is the object; bits auxiliary.
- `depth-decoupling-input-dp`: **supported (scoped)** — single-seed; CLUB parallel attenuation, not sign reversal.

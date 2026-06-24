# Novelty Check Report — claim:depth-inversion-certificate

**Reviewer:** Codex gpt-5.5 xhigh (thread 019ef813-27e7-7721-ac97-aae28388f1d3), 2026-06-24.

## Per-contribution verdict

| Part | Novelty | Closest prior work | Honest delta |
|---|---|---|---|
| C1 (depth ≠ privacy) | LOW | Dong et al. arXiv:2507.16372 | Direct reproduction on Qwen3-4B — external validity, not a new claim. |
| DECISION (mlp2>ridge@L32) | LOW | Dong et al.; Vec2Text/ALGEN/Zero2Text inversion | Narrow ablation: nonlinear inverter beats linear at depth; not conceptually surprising. |
| NOV-1 (Fano certificate) | LOW as theory / MEDIUM as protocol | Fano-from-accuracy standard: Zheng & Benjamini arXiv:1606.05229; arXiv:2210.13662; "A Lesson From Fano" 2022 | The Fano lemma is NOT novel. The contribution is the LLM-residual operationalization (vocab-disjoint + shuffle/null correction + conservative lower-entropy finite-sample bound). Position as instantiation, not new theory. |
| NOV-2 (probe tracks recovery across depth) | MEDIUM | predictive V-information; CLUB MI; Inf2Guard arXiv:2403.02116; NoPeek arXiv:2008.09161 | No direct prior for an attack-INDEPENDENT token-information probe empirically tracking inversion recovery across depth. Components standard; the positive measurement-loop result is the strongest new piece. Not HIGH until replicated across models/datasets/attacks/defenses. |

## Overall
- **Recommendation:** position as a **scoped reproduction + leakage-measurement** result, NOT a new attack or new IT theorem.
- **Strongest defensible novelty:** NOV-2 (the positive measurement-loop: attack-independent probe predicts inversion recovery across depth).
- **Do NOT claim:** "we show deep states are invertible" (Dong did) or "we introduce a new Fano certificate."
- **DO claim:** "we instantiate a conservative Fano-style leakage certificate and validate an attack-independent measurement loop against actual inversion recovery on Qwen3-4B."
- The claim file already frames C1 as a scoped reproduction and the certificate as an instantiation — consistent with this verdict.

# AloePri attack ledger — all 7 paper attacks on the pythia-160m artifact

Paper arXiv 2603.01499 (ground truth; the `sheng1feng/Aloepri` reference is buggy and was not
trusted). Spec: `/home/timo/repos/private-rag/docs/prototype/aloepri-llm.html`. Release gate:
**TTRSR ≤ 15%** per attack at paper hyperparameters (αₑ=1.0, α_h=0.2, h=128, λ=0.3). Model:
EleutherAI/pythia-160m, covariant reparam (`scripts/defenses/aloepri.py:reparam_pythia`).

TTRSR = Text Token Recovery Success Ratio (lower = better privacy). PASS = ≤ 0.15.

| # | Attack | Surface | Config measured | TTRSR | Gate | Source |
|---|--------|---------|-----------------|-------|------|--------|
| 1 | **VMA** (Vocabulary-Matching) | obf weights | keymat, αₑ=0 | **0.000** | PASS | weight_attacks.json |
| 2 | **IA — Gate-IA** | obf weights | keymat αₑ=0 / αₑ=1.0 | **1.000 / 0.000** | FAIL→PASS | weight_attacks.json |
| 2 | **IA — Attn-IA** | obf weights | keymat | **0.000** | PASS | weight_attacks.json |
| 3 | **ISA-HiddenState** (deep ridge) | activations | full_alg1 αₑ=1.0 (blind / matched) | **0.000 / 0.752** | PASS / FAIL | leakage_sweep.json |
| 4 | **IMA-EmbedRow-ridge** (paired) | obf weights | every αₑ (0→2.0) | **~1.000** | FAIL | weight_attacks.json |
| 4 | **IMA-L0 / IMA-transformer** | activations | = ridge / decoder, full_alg1 | matched 0.75→0.33 | see sweep | leakage_sweep.json |
| 5 | **NN** (cosine-NN) | activations | keymat (blind) | **0.000** | PASS | leakage_sweep.json |
| 6 | **TFMA** (freq-matching) | wire token-id | M1 ε1=∞ / 12 / 8 | **0.52 / 0.18 / 0.03** | FAIL→PASS | tokenid_sweep.json |
| 7 | **SDA** (bigram decipher) | wire token-id | M1 ε1=∞ / 12 / 8 | **0.75 / 0.42 / 0.00** | FAIL→PASS | tokenid_sweep.json |
| 8 | **ISA-AttnScore** | attention scores | plaintext / keymat / **alg2** | **0.074 / 0.074 / 0.066** | PASS | attnscore.json |
| — | QK-norm Γ eigendecomposition | obf weights | **N/A** — pythia (GPT-NeoX) has no QK-norm site | — | — | architecture |

**Algorithm 2 is implemented + verified** (`reparam_pythia(config="alg2")`): intra-head R̂qk rotations
(commute with NeoX partial-rotary) + per-pair scaling + Ûvo (V/O) + inter-head permutation, all
orthogonal and self-cancelling → the logits-identity gate passes (obf logits == plaintext, β=1). A
subtle, correct consequence: **Alg2 preserves the attention SCORES Q·Kᵀ by construction** (the model
must still work), so ISA-AttnScore is ~unchanged under alg2 (0.066≈0.074, the small drop is the αₑ
embedding noise). Alg2 defends the per-head Q/K/V **VALUE coordinates** (an attacker reading Q/K/V
activation vectors sees them in the secret M/Ûvo basis), NOT the scores — so the right attack to
demonstrate Alg2's defense is a Q/K/V-value inverter (queued), not the score reader.

## The synthesis (consistent across all surfaces)

**AloePri's keymat is key-gated ACCESS CONTROL, not information hiding.**

- **Defeated by the keymat STRUCTURE** (no key, no paired data): VMA (P̂'s dense mixing breaks the
  sorted-quantile signature), Attn-IA (the d→d+2h rank-deficiency breaks the quadratic-form invariant —
  a *different* mechanism than the paper's Alg-2 head perms, worth noting), NN (the P̂-basis width
  mismatch makes cross-space cosine matching undefined → chance). All PASS.
- **Defeated only by the αₑ NOISE** (the sole information-theoretic knob): Gate-IA (1.000 on keymat-only
  → 0.05 at αₑ=0.5, exactly the paper's "noise αₑ≥1.0 defends"); and the noise floor on the activation
  inverters at high αₑ.
- **NOT defended at all** against a key/paired-data attacker: IMA-EmbedRow-ridge recovers ~1.000 at
  EVERY αₑ (learns P̂⁻¹ in closed form — the paper's "99.22% gap"); ISA-HiddenState *matched* = 0.752 =
  plaintext; the *oracle* (knows Q̂) recovers identically. Information is preserved (CLUB MI flat across
  the lossless basis change) — just locked behind the key.
- **Wire surface (token-id)** is the one genuine information-reduction sweep: Π alone is a substitution
  cipher (TFMA 0.52 / SDA 0.75 at ε1=∞); the M1 randomized-response budget ε1 is what suppresses both
  (ε1=12 → TFMA 0.18 / SDA 0.42; ε1=8 → ~0). SDA (bigram) is the stronger wire attack.
- **Attention scores**: Alg1 preserves Q·Kᵀ byte-for-byte → keymat inert (0.074 == plaintext). Only
  Algorithm 2 (queued) defends this surface.

This matches `docs/research/2026-06-26-dp-scheme-privacy-utility-fork.md`: key-based obfuscation
escapes the information bound by changing the secret to a key — so its privacy is empirical
(attack-resistance under the no-key / no-paired-data threat model), not information-theoretic.

## Result files
- `aloepri_weight_attacks.json` — VMA, Gate-IA, Attn-IA, IMA-EmbedRow-ridge (αₑ sweep)
- `aloepri_leakage_sweep.json` — ISA-HiddenState / IMA-L0 / IMA-transformer / NN (ridge, decoder) + CLUB/V_cap probes (matched/oracle/blind)
- `aloepri_tokenid_sweep.json` — TFMA + SDA (ε1 sweep) + analytic channel MI
- `aloepri_attnscore.json` — ISA-AttnScore (plaintext vs keymat)

## Queued
- **Q/K/V-value inverter** — the attack Algorithm 2 actually defends (ISA/NN on the per-head Q/K/V
  activation vectors): keymat_only leaves them plaintext (recoverable), alg2 obfuscates them with the
  secret M/Ûvo (blind inversion fails, oracle recovers). The score surface (ISA-AttnScore) is the
  wrong demo — scores are preserved by construction.
- Unsupervised basis-recovery blind attack (Procrustes/CCA/ICA) — the current "blind" is wrong-key,
  not a key-recovery attempt; this is the stronger blind the measurement loop calls for.

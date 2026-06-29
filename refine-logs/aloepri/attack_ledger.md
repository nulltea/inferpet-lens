# AloePri attack ledger — paper attack suite on the pythia-160m artifact

Updated 2026-06-27: dropped IMA-EmbedRow-ridge (out of threat model — needs ≥~1k leaked τ-pairs);
added IMA-EmbedRow-transformer (in-model τ-invariant) and ArrowMatch; ISA-HiddenState reduced to the
in-model blind variant; ISA-AttnScore superseded by the queued kq/kqv_out all-layer sweep.

Paper arXiv 2603.01499 (ground truth; the `sheng1feng/Aloepri` reference is buggy and was not
trusted). Spec: `/home/timo/repos/private-rag/docs/prototype/aloepri-llm.html`. Release gate:
**TTRSR ≤ 15%** per attack at paper hyperparameters (αₑ=1.0, α_h=0.2, h=128, λ=0.3). Model:
EleutherAI/pythia-160m, covariant reparam (`scripts/defenses/aloepri.py:reparam_pythia`).

TTRSR = Text Token Recovery Success Ratio (lower = better privacy). PASS = ≤ 0.15.

All activation/score surfaces captured at **layer 6 of pythia-160m (12 layers → mid-network)** unless noted.
TTRSR threat-model rule (consistent with the IMA critique below): an attack counts only if it needs **no
deployment τ, no secret key, and no (obf, plain) paired data from the live basis**. *matched* (paired
ridge on deployment reps) and *oracle* (knows the key Q̂) are out-of-model and not gate-bearing.

| # | Attack | Surface | Config measured | TTRSR | Gate | Source |
|---|--------|---------|-----------------|-------|------|--------|
| 1 | **VMA** (Vocabulary-Matching) | obf weights | keymat, αₑ=0 | **0.000** | PASS | weight_attacks.json |
| 2 | **IA — Gate-IA** | obf weights | keymat αₑ=0 / αₑ=1.0 | **1.000 / 0.000** | FAIL→PASS | weight_attacks.json |
| 2 | **IA — Attn-IA** | obf weights | keymat | **0.000** | PASS | weight_attacks.json |
| 3 | **ISA-HiddenState** (blind, multi-key K=64) | activations resid_post, **all 12 layers** | keymat / alg2@0 / alg2@1.0, **blind only** | single-key **0.000**; **multi-key ≤0.064** (peak L0, ~0 deeper) — every layer ≤ gate. Matched ceiling 0.91→0.46 (out of model). | PASS — key-gated | multikey_blind.json |
| 4 | **IMA-EmbedRow-transformer** (τ-invariant, multi-key) | obf weights | in-model (own synthetic pseudo-τ/Q̂/P̂) | **paper ~0.0**; talens not run, private-rag driver not yet faithful (fails plain control) | QUEUED | private-rag aloepri-attacks.md §IMA-EmbedRow-transformer |
| 5 | **NN** (cosine-NN) | activations resid_post **@L6** | keymat (blind) | **0.000** | PASS | leakage_sweep.json |
| 6 | **TFMA** (freq-matching) | wire token-id | M1 ε1=∞ / 12 / 8 | **0.52 / 0.18 / 0.03** | FAIL→PASS | tokenid_sweep.json |
| 7 | **SDA** (bigram decipher) | wire token-id | M1 ε1=∞ / 12 / 8 | **0.75 / 0.42 / 0.00** | FAIL→PASS | tokenid_sweep.json |
| 8 | **ISA score/value surface** (kq + kqv_out, all 12 layers) | attention scores + per-head value coords | plaintext / keymat / alg2@0 / alg2@1.0; matched ridge | **kqv_out @L0: plaintext=keymat=alg2@0 = 0.685 (rowsplit) / 0.462 (disjoint) — IDENTICAL; alg2@1.0 = 0.564 / 0.179.** kq @L0 ≈ 0.28 / 0.11. Recovery peaks at L0, declines with depth. | leak (gate n/a; this is a per-surface readout) | score_surface.json |
| 9 | **ArrowMatch** (Game-of-Arrows cosine direction-similarity, Wang USENIX'25) | obf weights | plain control / AloePri keymat | **pythia-160m: 1.000 / 0.000** (top-1); Q3-4B: 0.986 / 0.000 | PASS — Obs2: matrix-mult (P̂) obfuscation is immune; cosine signal destroyed | aloepri_arrowmatch.json + private-rag m2_7 |
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
- **Out-of-threat-model attacks dropped.** *IMA-EmbedRow-ridge* (closed-form P̂⁻¹) recovered ~1.000 at
  every αₑ but **only given ≥~1k leaked (plain,obf) τ-pairs** — far above a realistic harvest (TFMA top-100
  distribution-aware ≤20 pairs; ≤313 pre-strong-Π), so it presupposes the win and is **removed** (see
  private-rag aloepri-attacks.md §"Why AloePri is safe from ridge in practice"). Same logic retires
  ISA-HiddenState *matched* (0.752, needs live-basis paired data) and *oracle* (knows Q̂). The in-model
  replacements are the **multi-key / τ-invariant synthesis** attacks: ISA-HiddenState blind with K=64
  synthetic keymats, and IMA-EmbedRow-**transformer** (paper §F.1). Information is still preserved (CLUB MI
  flat across the lossless basis change) — but a no-key/no-τ attacker cannot read it.
- **Wire surface (token-id)** is the one genuine information-reduction sweep: Π alone is a substitution
  cipher (TFMA 0.52 / SDA 0.75 at ε1=∞); the M1 randomized-response budget ε1 is what suppresses both
  (ε1=12 → TFMA 0.18 / SDA 0.42; ε1=8 → ~0). SDA (bigram) is the stronger wire attack.
- **Attention scores**: Alg1 preserves Q·Kᵀ byte-for-byte → keymat inert (0.074 == plaintext). Only
  Algorithm 2 (queued) defends this surface.

This matches `docs/research/2026-06-26-dp-scheme-privacy-utility-fork.md`: key-based obfuscation
escapes the information bound by changing the secret to a key — so its privacy is empirical
(attack-resistance under the no-key / no-paired-data threat model), not information-theoretic.

## Result files
- `aloepri_weight_attacks.json` — VMA, Gate-IA, Attn-IA (αₑ sweep). *(IMA-EmbedRow-ridge column retained in
  the raw JSON but no longer gate-bearing — out of threat model, see synthesis.)*
- `aloepri_score_surface.json` — ISA kq + kqv_out, all 12 layers, {plaintext, keymat, alg2@0, alg2@1.0}, matched ridge
- `aloepri_multikey_blind.json` — ISA-HiddenState blind (single + K=64 multi-key) + matched ceiling, all 12 layers
- `aloepri_arrowmatch.json` — ArrowMatch (plain control 1.000 / AloePri keymat 0.000)
- `aloepri_tokenid_sweep.json` — TFMA + SDA (ε1 sweep) + analytic channel MI
- `aloepri_leakage_sweep.json` — older residual ISA/NN + CLUB/V_cap/MDL probes (matched/oracle/blind @L6)
- `aloepri_attnscore.json` — ISA-AttnScore post-softmax (superseded by `aloepri_score_surface.json`)
- Report: `docs/html/static-obf.html` (per-attack sections, 6 figures)

## Queued
- **kq / kqv_out score-surface sweep (all 12 layers, plaintext/keymat/alg2)** — the in-model demo of
  Algorithm 2. `kq` = pre-softmax Q·Kᵀ; `kqv_out` = attention output (the per-head Q/K/V value surface
  Alg2 actually rotates). Matched ridge, vocab-disjoint + row-split, per private-rag methodology
  (no multi-key on this surface). Report → `docs/html/static-obf.html`. Expectation from private-rag
  Q3-4B: signal concentrated at L0 (kqv_out plain ~90–97%, alg2 ~47%), floors by L≥5; keymat inert.
- **ISA-HiddenState blind with K=64 multi-key synthesis — DONE** (`multikey_blind.json`, all 12 layers).
  Single-key blind ~0.000 (basis mismatch); multi-key ≤0.064 (peak L0, ~0 deeper), under the gate at
  every depth. AloePri's dense Alg1 keymat family has no linear K-invariant inverse, so the synthesis
  attack does not transfer: the residual stays key-gated. Matched ceiling 0.91→0.46 (out of model)
  confirms the information is present, just locked behind the key.
- **IMA-EmbedRow-transformer** — paper §F.1 τ-invariant trained inverter on attacker's own synthetic
  pseudo-τ/Q̂/P̂. Needs the driver-faithfulness fix (must pass identity-τ plain control first).

# Novelty Check — KV-CLOAK channel decoupling

**Reviewer**: gpt-5.5 high (Codex), thread 019ef74e-90b4-7372-ab7e-ea745c3f5ccf, 2026-06-24.
**Score: 7/10 — PROCEED.**

## Assessment
No public prior work shows KV-CLOAK's block permutation / block size is inert against a row-subspace
/ BSS / ICA adversary. KV-CLOAK argues a b! barrier for ORDER/brute-force recovery and that the
one-time permutation severs positional correspondence — it does not analyze subspace-membership
observables. The methodological template (decompose a composite obfuscation into channels + invariants,
match an IT probe) is routine; the novelty is the threat-model correction + concrete cryptanalysis of a
just-accepted (NDSS 2026) defense, plus measuring it on real Qwen3 keys.

## Closest prior work + delta
- KV-CLOAK / Shadow in the Cache (arXiv 2508.09442, NDSS 2026): the defense + its b! security claim.
  Delta: that barrier applies to order-recovery/collision adversaries; row-space/ICA observables are
  permutation-invariant, so b gives no protection — security rests on the secret feature mix M.
- KV-Shield (arXiv 2409.04040): earlier permutation-in-TEE KV protection. Delta: even one-time block
  permutation is inert for subspace observables.
- Representation inversion (GEIA 2305.03010, Vec2Text 2310.06816, Embed-Parrot 2405.11916): motivate
  the attack surface; do not analyze KV-CLOAK's channel composition.
- Classical ICA/JADE/negentropy: "ICA beats permutations" is prior art; novelty is specializing it to
  KV-CLOAK's advertised security parameter and measuring on real LLM keys.

## Positioning (per reviewer)
Pitch as a targeted security break / threat-model refinement: "the swept block-size security parameter
is not a security parameter for subspace/BSS adversaries; only the secret right feature mix M matters."
Do NOT pitch as a new ICA theorem or general obfuscation framework.

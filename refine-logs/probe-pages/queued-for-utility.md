# Queued onto Task 7 (leakage–utility, the single GPU phase) — from Task 4 (probe pages)

Two registered probes have no clean-model plaintext-across-layers reading on disk, so their
per-probe page (`probe-sdl.html`, `probe-shared-spectral-capacity.html`) carries an explicit
"queued onto Task 7" placeholder in its Plaintext-reference section instead of a measured baseline.
Task 7 must emit these on a CLEAN model (no attack, no defense) in the same capture, then hand the
numbers back so the two pages are backfilled.

| probe | symbol | what to emit (clean model, no defense) | on disk now | backfill target |
|---|---|---|---|---|
| surplus description length | `SDL` | prequential SDL selectivity per layer across depth, token-identity label (MDL was OFF in the 36-layer control sweep) | one at-layer point only: L12 zero-noise, SDL sel 13898 bits (`results/mdl_probe_check.json`) | `docs/html/probe-sdl.html` §04 Plaintext reference |
| shared spectral capacity | — | shared spectral capacity per layer × channel kind on the clean KV stack (averaged row covariance C̄, median-eigenvalue floor) | L0 KV-Cloak channel sweep only: 18.24 b identity / 20.94 b feature-mix-only / 16.32 b full defense (`refine-logs/kv-cloak/RESULTS.md`) | `docs/html/probe-shared-spectral-capacity.html` §04 Plaintext reference |

Notes for Task 7:
- Emit both in the SAME clean-model capture as the other piggybacked emissions; do not load the model twice.
- `SDL` reuses the `src/talens/measures/mdl.py` estimator (token-identity reader, prequential pass).
- shared spectral capacity reuses `src/talens/measures/bss_separability.py` on the clean KV stack.
- Representative layers are sufficient (match the depth grid already used on the depth-inversion page); a full 36-layer sweep is not required for the baseline reading.

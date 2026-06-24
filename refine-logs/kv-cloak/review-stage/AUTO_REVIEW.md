# Auto Review — KV-CLOAK channel decoupling (Task B-2)

Reviewer: gpt-5.5 xhigh (Codex), thread 019ef754-b61c-7ac1-8502-879c239c145b. Medium difficulty.

## Round 1 — 7/10, almost
Flagged a BLOCKING packaging gap: the claim file lacked the inline L1-L4 proof (the wiki add_claim
helper had overwritten the hand-written claim with its template). Plus: "only channel reducing
recovery" too strong (A dents 0.626→0.581); HTML L4 omitted the √(log n/d) term; block-size "every
channel" overgeneralized; repro under-specified. C2 honestly framed; numbers matched.

## Round 2 — 8/10, almost
Restored the full claim + inline proof (273 lines); narrowed wording to "load-bearing/floor-driving";
HTML L4 mirrors O(√(s/d)+√(log n/d)); block-size bounded; experiment record gained exact invocation +
jd_floor at T=1/T=4. Residual: masthead/§05/§06 still carried the old "sole operation that reduces
reconstruction / unchanged" overclaim.

## Round 3 — 8.5/10, READY
Fixed the three HTML overclaims (masthead "drives reconstruction to chance"; §05 A has small non-floor
effect + inflates negentropy; §06 mask perturbs spectrum but not to chance) and the final polish nit
(matched probe "plaintext-like in the sweep"). Reviewer: "the report + claim now read as a bounded,
honest internal research node: C1 supported and scoped, C2 between-channel, L4 an upper bound under
s≪d, the public page matches the evidence."

## STOP: positive verdict (score 8.5 ≥ 6 AND verdict = ready).

## Method Description
KV-CLOAK (K'=S·P̂·(K+A)·M) is implemented as a scheme-agnostic representation Transform over raw
per-head Qwen3-4B keys. A channel ablation crossed with block size, mask energy, and seed (273 cells,
CPU) is attacked by the BSS family (gram_error, jade, jd) and probed by the geometry-only negentropy
and shared-spectral-capacity measures. Four verified lemmas decouple the three operations onto three
observables: the secret right-orthogonal feature mix M is the only load-bearing channel (recovery
ceiling O(√(s/d))); the token-mix/permutation and block size are cover-invariant; the additive mask
only moves the Gram spectrum. A matched negentropy probe predicts recovery between channels (ρ=0.71).

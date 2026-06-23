---
type: claim
node_id: claim:spectral-channel-mi-embedding-inversion
name: "Spectral channel-MI: matched attack-independent converse leakage probe for embedding inversion under DP"
description: ""
node_type: claim
status: verified
provenance: "refine-logs/dp-stronger-attacks/vec2text-pooled/PROOF_PACKAGE.md; Codex gpt-5.5 xhigh thread 019ef046, 3 rounds, PASS"
tags: ["vec2text", "embedding-inversion", "spectral-mi", "gaussian-channel", "water-filling", "fano", "rate-distortion", "matched-probe", "geometry-only", "dp", "capPVI", "club", "pooled-embedding"]
date: 2026-06-22
added: 2026-06-22T17:17:36Z
---

# Spectral channel-MI: matched attack-independent converse leakage probe for embedding inversion under DP

**status:** `verified` — cross-model checked (Codex `gpt-5.5` xhigh, thread `019ef046`,
3 rounds, verdict **PASS**, zero open FATAL/CRITICAL). Imported textbook steps (Gaussian
max-entropy, Fano, Shannon rate–distortion lower bound) flagged as such; the matched-probe
framing, sufficiency identity (T1), localization bound (T4), and contrast results are
established here. Full verified proof: `refine-logs/dp-stronger-attacks/vec2text-pooled/PROOF_PACKAGE.md`.
Continuous-channel analog of [[bnn-error-bounds-bhattacharyya-fano]] (which needs an
enumerable codebook; text has $V^n$ messages, so the spectral bound replaces it).

## Statement

Secret text $X$ (finite alphabet, $H(X)>0$); deterministic clipped embedding
$e_0=\mathrm{clip}(\phi(X),C)\in\mathbb R^d$ (so $e_0$ is **discrete**, $\Sigma=\operatorname{Cov}(e_0)$,
eigenvalues $\lambda_1\ge\cdots\ge\lambda_d\ge0$); Gaussian-mechanism release $Y=e_0+\mathcal N(0,\sigma^2I_d)$.
Define $t_i:=\tfrac12\log_2(1+\lambda_i/\sigma^2)$, $I_G(\sigma):=\sum_i t_i$,
$d_{\mathrm{eff}}:=\#\{i:\lambda_i\ge\sigma^2\}$. Computed from $(\Sigma,\sigma)$ **alone** (no attack run):

- **T1 (sufficiency).** $I(X;Y)=I(e_0;Y)$ — measuring MI on embeddings is exactly the text leakage.
- **T2 (spectral ceiling + localization).** $I(X;Y)\le\min\{H(e_0),\,I_G(\sigma)\}$ (strict $\le I_G$
  unless $e_0$ Gaussian). Low-noise cap is the discrete $H(e_0)$ ($I(X;Y)\to H(e_0)$ as $\sigma\to0$);
  $I_G$ is the informative ceiling in the privacy regime $I_G<H(e_0)$. Per-mode $t_i$ carries info iff
  $\lambda_i\gtrsim\sigma^2$ (water-filling).
- **T3 (attack-independent recovery ceiling).** Fano: uniform-prior exact-match success
  $\Pr[\hat X=X]\le(\min\{H(e_0),I_G\}+1)/H(X)$ for **every** attack $\hat X=\psi(Y)$; rate–distortion
  variant ceilings the per-token error rate. Converse, monotone $\downarrow$ in $\sigma$.
- **T4 (localization).** For any top-$k$ principal projector $P_k$ and **arbitrary** $e_0$:
  $I(X;Y)-I(X;P_kY)\le\tfrac12\sum_{i>k}\log_2(1+\lambda_i/\sigma^2)$ — the recoverable leakage is
  confined to the top eigendirections up to the spectral tail (exact split for Gaussian $e_0$).
- **Contrast.** `capPVI` (V-info of a $\kappa$-cluster label) saturates at $\le\log_2\kappa$ bits and
  stays flat while $\sigma\ll$ centroid gaps — orthogonal to fine token recovery. `CLUB` targets the
  right MI $I(e';e_0)=I(X;Y)$ but is a loose, non-localizing variational upper bound.

## Honest scope

$I_G$ is a **converse ceiling**, not the exact leakage nor achieved recovery — empirical probe↔recovery
correlation (experiment B8: CLUB-bits ↔ Vec2Text token-F1/cos/BLEU Spearman $=+1.0$) is validation, not
implied by the converse. Token-F1 is **not** bounded (only positional token-error). T4 achievability (a
tractable attack realizing the top-subspace info) is an empirical hook (bottom-mode ablation). Imported:
Gaussian max-entropy, Fano, Shannon RD lower bound. Finite-sample $\lambda_i$ bias can shift
$d_{\mathrm{eff}}$ when $n_{\text{texts}}\lesssim d$ (use shrinkage).

## Evidence chain

- Proof obligations T1–T4 + contrast: all discharged; see `PROOF_PACKAGE.md` (theorem-by-theorem).
- Jury: Codex `gpt-5.5` xhigh, thread `019ef046`, rounds 1→3 (WARN→WARN→**PASS**); round-1 over-claim
  ($I_G$ as "accessible info"/predictor) corrected to converse ceiling; low-noise cap fixed to $H(e_0)$.
- **Implementation**: `src/talens/measures/spectral_channel_mi.py` (geometry-only; Codex-reviewed, no
  critical; hardened) + `tests/test_spectral_channel_mi.py` (10/10 model-free; Gaussian-exact ½logdet,
  monotone, d_eff, ceilings).
- **Empirical (B9, `results/spectral_mi_probe_eval.json`)**: on the pooled-GTR DP sweep (N=96), **C1
  VALIDATED** — Spearman(I_G, Vec2Text recovery)=+1.00 (=CLUB) ≫ capPVI +0.62, at ~28× lower cost
  (I_G 60 ms eigh vs CLUB 1.7 s); **C2** 0 ceiling violations (RD per-token floor 0→0.81 respected).
  **Caveat:** N<d ⇒ rank-deficient Σ, so d_eff/tail (localization, T4/C3) are undersampled — needs n≫d
  (estimate Σ from a large embedding corpus); M3 eigen-ablation re-scoped accordingly (not yet run).
- companion empirical: [[unified-dp-sweep]] (dp-stronger-attacks B8).

## Connections

- analog-of → [[bnn-error-bounds-bhattacharyya-fano]] (discrete-codebook two-sided bound)
- companion → [[mi-monotone-gaussian]]
- supported-by → [[unified-dp-sweep]] (DP sweep; CLUB↔Vec2Text recovery ρ=+1.0)
- supersedes-as-probe → `capPVI` (cluster V-info), `CLUB` (variational $I(e';e_0)$) for this channel

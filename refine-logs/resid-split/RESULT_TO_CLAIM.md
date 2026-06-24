# Result-to-Claim Verdict — resid-split / PriPert (Task 6)

**Judge**: Codex (gpt-5.x, xhigh), 2026-06-24. Evidence pre-check: 4/4 cited numbers verified in
`runs/sweep/pripert_sweep.json`. Integrity audit: see `EXPERIMENT_AUDIT.json` (run next).

## Verdict (per claim)
- **C1 — partial.** Supports *depth-dependent* suppression, not blanket monotonic. Perturbation
  strongly suppresses inversion at L8/L16 (collapse by β=0.5) and at L24 by β≈1; **L0
  (input-embedding residual, σ_ref=0.19) stays substantially recoverable even at β=2 (0.363) and is
  non-monotonic**. → scope C1 to depth-dependent.
- **C2 — partial (positive measurement loop, one overclaim to drop).** Pooled tracking is real and
  **robust to the β-axis confound**: fixed-β=0.25 layer×ρ slice still Spearman(I_G,best)=**0.916**;
  within-layer β-tracking is **perfect (1.0) at L8/L16/L24**, weak at L0 (≈0.5, n=6 — fragile
  point estimate) where the defense has no recovery dynamic range to track. But **"no probe–attack gap" is too strong as a general claim** — the probe
  is a *monotone rank/capacity tracker*, not a tight predictor (recovery floors while I_G is still
  tens of bits). → revise to "no *observed* gap for the *tested* inverters (incl. mlp2)."
- **C3 — yes, but loose.** Zero converse violations is supported; the Fano certificate is
  *valid but practically vacuous* (binds in 1/32 cells). Honest negative; keep framed as slack.

## Confidence
High for these artifacts; medium for generalization beyond this sweep (one model/corpus, single
Gaussian-δ seed, Gaussian proxy not adversarial-optimized δ).

## Adopted claim scope (drives the research-wiki claim + proof)
On Qwen3-4B `resid_post` under the fixed-plaintext-RMS Gaussian PriPert proxy (WEIGHTS-PUB):
1. **(C1)** the additive perturbation is the load-bearing knob and suppresses tested token
   inversion **depth-dependently** — strongly at L8/L16 (by β=0.5) and at L24 (by β≈1), while **L0
   resists** (best 0.363 at β=2); sparsification (ρ) is secondary.
2. **(C2, headline)** the matched, attack-independent spectral channel-MI probe `I_G` is a **strong
   monotone tracker** of inversion recovery across the joint (layer×ρ×β) sweep — pooled
   Spearman 0.958, β-confound-controlled 0.916, within-layer 1.0 at L8/L16/L24 — and tracks the
   **stronger learned inverter** (mlp2, 0.915), so **no probe–attack gap is observed for the tested
   attacks**. CLUB cross-check 0.977.
3. **(C3)** the probe's Fano accuracy ceiling is a **valid converse with 0 violations but slack**:
   I_G ≫ H_X=11 bits across the useful range; it binds only at β=2 (I_G=10.0) where recovery is
   already ~0.002 — the empirical floor is reached by the perturbation degrading the inverter far
   inside where the capacity converse would bite (slack-converse / capacity-slack pattern, cf.
   `rep2text-capacity-nonbinding`).

## Unsupported / missing (recorded, not blocking the scoped claim)
adversarial-optimized & PCA-aligned δ arms; multiple noise seeds; second model/corpus; stronger
nonlinear decoders; utility-degradation curves; denser ρ grid. None overturn C1/C2/C3 as scoped;
they bound generalization. (No spawn-depth-1 follow-up: C2 correlates AND tracks the stronger
attack — the measurement-loop "Yes" branch.)

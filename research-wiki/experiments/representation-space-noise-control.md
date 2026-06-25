---
type: research
status: current
created: 2026-06-25
updated: 2026-06-25
tags: [differential-privacy, control, representation-space, at-layer-noise, propagation]
supersedes:
companion: docs/html/resid-dp-attacks.html
---

# Representation-space noise — the control that isolates propagation (NOT a DP scheme)

Relocated from `docs/html/resid-dp-attacks.html` in the 2026-06-25 DP-page overhaul. It is kept as
a **control**, not a defense the site reports: "at-layer noise" is **not** a differential-privacy
deployment. The DP page covers only **local DP** (Gaussian mechanism on the input embedding,
observed across depth). This note preserves the at-layer evidence so the propagation finding is not
lost.

## What "at-layer noise" is

Gaussian noise added **directly to the captured hidden layer** `resid_post[L]` (level = σ / act-RMS),
with the representation otherwise clean. There is no input-embedding mechanism and no propagation
through the blocks. It does not correspond to any deployed DP mechanism — it is a synthetic
representation-space perturbation used purely as a control. (The PCA-subspace-ablation variant is
likewise a representation-space defense, not DP.)

## Why it is load-bearing as a control

The DP-page headline is that under **propagated local DP** a linear ridge attack *decorrelates* from
the MI probes at depth while a non-linear decoder *re-correlates*. The at-layer control shows this
decorrelation is **specific to propagation**, not a generic property of noise:

- **Under at-layer noise, even ridge tracks the MI probes perfectly** (Spearman ρ = 1.0 at L5/12/20),
  and a 250-epoch MLP decoder **loses to ridge** at every depth/level (no gap — ridge is already
  near-sufficient when the noise is applied at the observed layer).
- **Under propagated input/local DP, ridge anti-correlates** at depth (ρ ≈ 0.2 vs CLUB at L20) and
  the stronger decoder **re-correlates** (ρ = 1.0).

So the decorrelation is caused by the noise being **reshaped through the non-linear blocks**
(propagation), not by noise magnitude per se. The at-layer regime is the negative control that
establishes this; it is not itself a defense result.

## Source

Original tables (R2/R3 and the at-layer/PCA-ablation/isotropic-noise rows of R7) on the pre-overhaul
`resid-dp-attacks.html`; spikes `scripts/spikes/{b2_lpos_decoder,mdl_probe_check}.py`;
`refine-logs/resid-capacity-pvi/RESULTS_STANDARDIZED.md` (the at-layer V_cap/CLUB-tracks-at-every-depth
table). Single seed; directional.

# Auto Review — resid-split / PriPert (Task 6, campaign-B-expand)

## Round 1 (2026-06-24) — Score 7/10, verdict Almost
Codex (xhigh, thread 019ef899) reviewed report + claim + raw sweep directly. Confirmed: headline
numbers match pripert_sweep.json (Spearman I_G-vs-best 0.958, I_G-vs-mlp2 0.915, CLUB 0.977,
fixed-β slice 0.916, within-layer 1.0); probe attack-independent (cov(Sparsify(H))+σ, no inverter)
— not circular; proved-vs-empirical boundary honest; slack converse not inflated; limits disclosed.
Weaknesses: (1) Fano uses candidate-pool uniform prior, not the test-token distribution — label
idealized; (2) single seed/model/corpus; (3) frame as fixed-plaintext-RMS Gaussian proxy; (4) wording
"eight plaintext"→noiseless, L0 "input-embedding"→post-block-0; (5) stronger inverter.

### Actions taken
Applied #1 (idealized-prior labels in report §04/§05 + claim already had Assumption 2/Open Risk 4),
#3 (Gaussian-proxy framing), #4 (wording) directly. #2 and #5 recorded as disclosed follow-ups
(experiment breadth, out of scope for this surface deliverable's gate).

## Round 2 (2026-06-24) — Score 8.5/10, verdict READY
Re-review confirmed fixes #1/#3/#4 resolved; one trivial editorial (claim C3 "idealized") then fixed.
Gate met (score ≥6 AND verdict ∈ {ready, almost}). Status: completed.

## Method Description
PriPert (arXiv 2605.23158) implemented as a scheme-agnostic talens Transform: per-row top-ρ magnitude
sparsification + additive perturbation, realized as a channel-matched isotropic Gaussian with σ fixed
to the plaintext row-RMS per layer (the attack-independent proxy for the paper's adversarial δ). Swept
over split layer × ρ × β on cached Qwen3-4B resid_post; attacked by ridge/nn/mlp2 token-embedding
inverters (vocab-disjoint split + shuffle control + bootstrap CI); measured by the matched spectral
channel-MI probe I_G (Fano converse) + CLUB. Measurement loop: I_G rank-tracks recovery (incl. the
stronger mlp2) along the perturbation axis (proved co-monotone) and across the joint sweep
(empirical); the Fano converse is valid (0/32 violations) but slack.

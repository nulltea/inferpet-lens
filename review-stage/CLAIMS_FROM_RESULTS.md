# Claims from results — matched-probe program (as of 2026-06-20)

Generated at auto-review-loop termination (round 1, 6.5/10 "almost"). Claims are
graded by current evidence.

## Supported now
- **C-Π (permutation channel).** On the permutation-core AloePri regime, **CLUB on
  the sorted-quantile signature φ is an independent probe that calibratedly predicts
  VMA τ-recovery** (Spearman +0.976 over the α_e sweep, gemma-2-2b embedding, N=1200),
  where retrieval-PVI-on-φ is the dependent reference (+1.000 = the attack in bits).
  *Evidence:* `results/aloepri_vma_sweep.json`. *Caveat:* 7 α points, 1 seed → B2+ firms it.
- **C-keymat.** The dense AloePri Algorithm-1 key matrix (validated invertible at
  d=2304, P̂Q̂=I err 6e-9) **erases the sorted-quantile channel**: VMA τ-recovery and
  both φ-measures fall to floor. The permutation-core is the VMA-vulnerable regime;
  the full keymat defends it. *Evidence:* same JSON (keymat row) + `test_aloepri.py`.
- **C-impl.** AloePri (Alg1 + obf-table + covers), Shredder (static + learned-noise,
  SNR-sign-corrected), and MMI-PID (operational reader atoms + conditional increments)
  are implemented and oracle-tested (76/76). *Evidence:* `scripts/defenses/`, `measures/pid.py`, tests.

## Supported by prior thread (re-used as channel rows)
- **C-token.** Capacity-PVI reader accuracy tracks ridge TTRSR (ρ 0.82–1.0) under
  representation defences; diverges at L20 under input-DP (the seed decoupling datum).
- **C-embed.** CLUB I(rep;emb) tracks ridge cosine; shows the L20×DP gradient too.

## Not yet supported (the headline — needs B3)
- **C-principle (channel-specific calibration / decoupling).** Matched probe×attack
  pairs calibrate; mismatched pairs decouple, with diagonal-dominance (most Δ_i>0,
  CIs exclude 0) and ≥1 sign-flip over a shared ε×depth grid + negative controls.
  *Status:* diagonal + L20×DP sign-flip in hand; **off-diagonal matrix is the next GPU run.**

## Explicitly NOT claimed
- An information-theoretic "law" (reviewer: needs replication across ≥2 defence
  families with repeated sign-flips).
- A sound Shannon PID (the MMI atoms are operational reader-bound atoms).
- A capacity-reader Π-baseline on the weight table (degenerate; the activation
  surface is where reader-style Π-probes are tested).

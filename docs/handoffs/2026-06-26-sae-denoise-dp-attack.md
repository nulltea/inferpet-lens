---
type: handoff
status: partial
created: 2026-06-26
updated: 2026-06-26
tags: [dp, residual, sae, superposition, denoising, attack, measurement-loop, brainstorm]
companion: docs/html/resid-dp-attacks.html
supersedes:
---

# Handoff — SAE-as-denoiser attack on DP-mixed residuals (brainstorm in progress)

Focus for next session: **finish brainstorming, then spec** an SAE (sparse autoencoder) denoising
*attack* against the local-DP residual surface. This was a `/superpowers:brainstorm` session that the
user interrupted mid-design — no code written, no spec written yet. Pick up the brainstorm where it
stopped (one unanswered question, below) and proceed to a design doc.

## Origin / motivation

User looked at **FIG.05** in `docs/html/resid-dp-attacks.html` — the `{is,are,was,were}` token-class
clouds that are tight/separated at L0 and **smear together with depth and stronger DP noise**. This
"reminds them of the superposition hypothesis" (DP mixes token representations) that SAEs are meant to
address → proposal: **use an SAE to separate token-ids mixed by DP, i.e. denoise the representation**.

This connects directly to a standing open question in the measurement loop (memory
[[token-class-separability-no-l20-peak]]): the Bhattacharyya margin collapses monotonically with depth,
and the loop flags *"a stronger attack should re-correlate — queue it."* An SAE denoiser is a candidate
for exactly that stronger attack.

## Decisions locked this session

1. **SAE role = stronger ATTACK (denoiser), graded on recovery — NOT a probe.** User explicitly chose
   this. Rationale: an SAE reconstruction is a learned decoder; using it as a leakage *probe* would be
   "the attack in disguise" / circular under the project's probe≠attack integrity rule. As an attack
   it's clean: SAE denoises the DP'd residual → ridge/decoder recovers tokens → graded on
   token-F1/top-k recovery.

## The hard ceiling that frames the whole experiment

**DP is closed under post-processing.** No operation on the noised residual — SAE included — can
*increase* mutual information about the token. So an SAE can **never beat the I_G converse**. Its only
possible win is **closing the gap between practical ridge recovery and the information-theoretic
ceiling** — i.e. demonstrating that the FIG.05 margin collapse is a **readout limitation** (info present
but in a form linear ridge can't read: superposition/nonlinear) rather than an **information
limitation** (info genuinely destroyed by noise). Either outcome is a first-class result:
- SAE denoising **raises** recovery toward I_G → margin collapse was readout-limited; superposition story holds.
- SAE denoising **does not help** → margin collapse is real info destruction; negative result, record it.

## OPEN QUESTION the user was answering when they interrupted

"Maybe we need to train the SAE on noised activations?" — the **training-regime fork**. I had just
presented the reasoning + a 4-option selector (rejected, then `/handoff` invoked). Re-ask this as the
next step. The options and my recommendation:

- **(RECOMMENDED) Clean-trained, applied to noisy.** Train SAE unsupervised on *clean* activations;
  at attack time encode→decode the *noisy* residual. Denoising-by-projection onto the clean feature
  manifold. **One SAE per layer, reused across the entire ε sweep** (big win on the single iGPU). It's
  the direct superposition-survival test. Under (assumed) ~isotropic local-DP Gaussian noise, projection
  onto the clean manifold is already near-optimal vs symmetric off-manifold noise.
- **Clean-primary + noise-aware ablation.** Same primary, plus one noise-aware variant to check for
  exploitable residual noise structure. Sharper conclusion, more compute.
- **Noise-trained only.** SAE trained on noised activations per ε. Basis matched to noisy dist but
  spends atoms explaining noise; per-ε retraining multiplies cost; generally weaker denoiser.
- **Supervised denoising AE (noisy→clean).** Strongest denoiser but a per-ε learned decoder with a
  sparsity prior — close to the existing MLP decoder in `dp_inversion.py`, not a probe-clean SAE.

Key adversary capability that makes the fork live: under WEIGHTS-PUB the adversary can generate
**unlimited paired (clean, noisy) data** by running the public model + adding the public DP mechanism.

## Conceptual caveat to validate before building

Superposition (features in overlapping directions) ≠ additive DP noise. The SAE-denoising story really
rests on **sparse-coding denoising**: clean signal is sparse in the dictionary, isotropic noise spreads
across many atoms, sparse reconstruction / thresholding discards it. Confirm the local-DP mechanism's
noise geometry (isotropic Gaussian on the input embedding? — see `scripts/defenses/local_dp.py`); if
anisotropic, the clean-vs-noise-aware calculus changes.

## Where things live (don't re-derive)

- Surface, eval orchestrator, attacks, probes, defense, report: all in handoff
  `docs/handoffs/2026-06-25-dp-depth-probes-and-attacks.md` (read it — current state of the DP×depth study).
- Existing attacks to compare/extend against: `src/talens/attacks/dp_inversion.py` (ridge, linear-skip
  MLP decoder, BeamClean).
- Existing separability probe (the FIG.05/06 source): `src/talens/probes/class_separability.py`,
  `scripts/figs/sep_separability_fig.py`, probe `sep` in `scripts/evals/dp_leakage_sweep.py`.
- New home for the SAE: it's an **attack**, so `src/talens/attacks/` (e.g. `sae_denoise.py`), called by
  `scripts/evals/dp_leakage_sweep.py` (or a new eval). Per repo-structure rule, reusable logic must NOT
  live inline in a spike/eval. A new SAE may start as a `scripts/spikes/` spike, then move once confirmed.
- Note: `pythia-140m` (user's current focus model per this session) vs the gemma-2-2b in FIG.05 — the
  DP study was switched to **Pythia-160M** (memory [[mi-probe-config-and-l20-not-token-id]]); confirm
  the exact model with the user before building (user said "Pythia-140M").

## Next steps for a fresh agent

1. Resume `/superpowers:brainstorm`: re-ask the training-regime selector above (one question, selector
   form per the user's global preference), then continue to design (architecture: dictionary size /
   sparsity / which layers; eval: bolt onto the existing ε×depth sweep, grade recovery vs ridge baseline
   and against the I_G ceiling).
2. Confirm model (Pythia-140M vs 160M) and the local-DP noise geometry.
3. Present design → user approval → write spec to `docs/superpowers/specs/2026-06-26-sae-denoise-dp-design.md`
   → `/writing-plans`.
4. Build under the performance gate (`scripts/harness/perf_gate.md`); reuse one clean SAE across ε.

## Suggested skills next session

`/superpowers:brainstorm` (resume) → `/writing-plans` → `/experiment-plan` → `/experiment-bridge`.
Keep the perf gate + one-GPU-process discipline (memory [[one-gpu-process-at-a-time]],
[[experiment-run-discipline]]).

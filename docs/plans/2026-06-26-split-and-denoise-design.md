---
type: plan
status: current
created: 2026-06-26
updated: 2026-06-26
tags: [defense, split-and-denoise, snd, dx-privacy, local-dp, denoiser, utility-recovery, eaas]
companion: [component-topology]
supersedes: []
---

# Split-and-Denoise (SnD) — defense + denoiser + utility-recovery eval

Design spec for implementing the **Split-and-Denoise (SnD)** defense
(Mai et al. 2023, arXiv:2310.09130, `paper:mai2023_splitanddenoise_protect_large`) in this
repo and evaluating its **utility recovery** across the privacy budget. Reference
implementation: <https://github.com/NusIoraPrivacy/eaas-privacy>.

## Definitions

| Term | Meaning |
|---|---|
| **SnD** | Split-N-Denoise: split LLM inference where the client runs the token-embedding layer, privatizes the embeddings, sends them to the server, and **denoises** the returned output embedding locally. |
| **dχ-privacy** | A metric (`dχ(x,x')=‖x−x'‖`) variant of local DP for text. Mechanism `M'` satisfies `ηdχ`-privacy; **η** is the (single) privacy-budget knob — larger η = weaker privacy. **Not** the Gaussian (ε,δ) of `LocalDP`; the two budgets are not interchangeable. |
| **Local encoder / split point** | The client-side computation = the input token-embedding layer only. Same forward-hook seam `LocalDP` already uses (`model.get_input_embeddings()`). |
| **e_c / e_n / e_d** | Clean / noised / denoised **pooled output embedding** (R^d): mean of the last-hidden states over real (non-pad) tokens. The EaaS "sentence embedding". |
| **X̃, Z** | Privatized token-embedding matrix (`M'(X)`) and noise matrix (`Z = X̃ − X`, post-clip), R^{T×d}. The denoiser's conditioning inputs. |
| **C** | L2 clip bound = high-percentile (default 99.9) of runtime token-embedding norms (reuses the existing clip convention so clip-only ≈ clean). |
| **Utility recovery** | How much of the clean output embedding the denoiser restores: `cos(e_d,e_c)` & `MSE(e_d,e_c)` vs the noised-no-denoise baseline `cos(e_n,e_c)` / `MSE(e_n,e_c)`. Paper's Appendix A.10 metric. |

## Threat model & scope

WEIGHTS-PUB, LDP setting: server is untrusted; privatization is client-side. This spec
covers the **utility axis** of the privacy–utility tradeoff only — "what does budget η cost,
and how much does the denoiser claw back". The **privacy axis** (inversion / leakage attacks,
the `(bits, recovery)` measurement loop) stays in `dp_leakage_sweep.py` and is **not**
re-implemented here, mirroring how `dp_utility_sweep.py` is the utility companion to the
leakage sweep.

## Architecture (3 units)

### 1. dχ privatization — `scripts/defenses/snd.py :: DxPrivacy(C, eta)`

A forward hook on the input-embedding layer (sibling to `LocalDP`). For each token embedding
`x ∈ R^d` in the batch:

- direction `v = g/‖g‖`, `g ~ N(0, I_d)` (uniform on the unit sphere);
- magnitude `l ~ Γ(shape=d, scale=1/η)` (the d-dim Laplacian of Wu et al. 2017);
- `M(x) = x + l·v`;
- clip: `M'(x) = M(x)·min(1, C/‖M(x)‖)`.

Returns `M'(X)` (same shape/dtype as the embedding output). `η = inf` ⇒ noiseless (clip-only,
≈ clean). The hook is **stateless**; the eval recovers `Z = X̃ − X` by differencing a noised
capture against the clean embedding (`X = table[ids]`), so no hook side-channel is needed.

### 2. Denoiser — `scripts/defenses/snd.py :: Denoiser(nn.Module)`

Client-side noise-aware transformer. Input sequence of `2T+1` tokens (each R^d):
`[e_n] ++ X̃₁..X̃_T ++ Z₁..Z_T`. Add a learned **type embedding** (3 types: output / raw / noise)
and a learned/sinusoidal positional embedding; run **L=3** transformer-encoder layers
(`d_model=d=768`, 8 heads, `d_ff=d`, GELU). Read the hidden state at the `e_n` slot at the
final layer → linear `R^d→R^d` → `e_d`. Padding tokens masked via `src_key_padding_mask`.

One model serves the whole η sweep (it conditions on `Z`, so it is noise-level aware). Per-η-group
denoisers (paper §A.5.3) are **deliberately skipped** — add only if recovery collapses at extreme η.

### 3. Utility-recovery eval — `scripts/evals/snd_utility_sweep.py`

Config-driven sweep that only orchestrates `DxPrivacy` + `Denoiser` + capture.

1. **Calibrate** `C` from token-norm percentile (same routine as the other sweeps).
2. **Train denoiser** on a TRAIN split: for each training η (default 3 spanning the sweep),
   one noised forward per prompt → `(e_n, X̃)`, one clean forward → `(e_c, X=table[ids])`,
   `Z = X̃ − X`; minimize `E‖denoiser(e_n,X̃,Z) − e_c‖²` (default 2 epochs, Adam). Save weights.
3. **Sweep** over TEST-split prompts, per η:
   - clean fwd → `e_c, X`;
   - noised fwd → `e_n, X̃`; `Z = X̃ − X`;
   - `e_d = denoiser(e_n, X̃, Z)`;
   - record `cos/MSE(e_n,e_c)` (noised baseline), `cos/MSE(e_d,e_c)` (denoised),
     `recovery_cos = (cos(e_d,e_c) − cos(e_n,e_c)) / (1 − cos(e_n,e_c))`,
     `recovery_mse = 1 − MSE(e_d,e_c)/MSE(e_n,e_c)`;
   - teacher-forced **perplexity** & next-token **acc** under the `DxPrivacy` hook, referenced
     to the η=∞ baseline (`ppl_degradation`, `retention_acc`) — the generation-utility cost of
     the noise (the denoiser does not touch this surface; it contextualizes the budget).
4. Write JSON → `refine-logs/snd/snd_utility_sweep.json` (`defense: "snd_dx"`, η list,
   `budget_note` flagging η ≠ Gaussian ε, records, per-η rows).

## Data flow

```
ids ──embed──> X ──DxPrivacy(C,η)──> X̃ ──(server: rest of model)──> e_n
 │                                    │                               │
 └─ table[ids] = X (clean) ───────────┴── Z = X̃ − X ──────────────────┤
                                                                       ▼
clean fwd (no hook) ──> e_c                       Denoiser(e_n, X̃, Z) ──> e_d
                          └──────── cos/MSE recovery ◄───────────────────┘
```

## Error handling / correctness

- η=∞ path must bypass noise entirely (clip-only) and is the mandatory baseline (∞ row first).
- Token-norm calibration and the clip bound are shared with the existing sweeps for splice-ability.
- fp32, eager attention, `pad_token=eos`, right padding (matches `dp_leakage_sweep`'s batch-invariance reasoning).
- Variable T handled by per-prompt sequences padded to batch max with a key-padding mask.

## Testing

One runnable self-check in `snd.py :: __main__` (asserts, CPU, no GPU/model):
1. `DxPrivacy` output rows obey `‖M'(x)‖ ≤ C + 1e-4`;
2. larger η ⇒ smaller mean `‖Z‖` (noise scales the right way);
3. `Denoiser` forward on a `(B, 2T+1, d)` synthetic batch returns shape `(B, d)`;
4. tiny overfit: on a fixed synthetic `(e_n, X̃, Z, e_c)` batch, a few steps drive denoised
   cos-to-`e_c` above the raw `cos(e_n,e_c)` (denoiser learns *something*).

## Performance gate

Default run sized to stay < 10 min on the single iGPU: ≤200 train prompts, 3 train-η, L=3,
2 epochs, batched capture saturating the iGPU; denoiser train/infer on GPU. All knobs are CLI
flags so a fuller sweep is opt-in. One GPU process at a time (ROCm container).

## Skipped (YAGNI — add when)

- **Per-η-group denoisers** → add if a single noise-aware model under-recovers at extreme η.
- **Labeled downstream task (AUC)** → add when a labeled corpus is wired into the repo.
- **Privacy-axis attack/probe here** → already covered by `dp_leakage_sweep`; do not duplicate.

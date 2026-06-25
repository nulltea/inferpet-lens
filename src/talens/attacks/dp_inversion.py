"""Array-interface inversion attacks for the DP / cross-layer leakage sweeps.

These are the attacks `scripts/evals/dp_leakage_sweep.py` (and future BeamClean) call: each maps a
captured representation matrix to predicted token ids via the embedding table, on a plain numpy
(Xtr, Etr, Xte, pool_emb, pool_ids) interface — deliberately lighter than the CaptureSet/AttackResult
API in `hidden_state.py` / `_inversion.py`, which target the full library pipeline.

Repo rule: attacks live here (src/talens/attacks/), not in scripts. evals/spikes only call them.

  ridge_attack         — linear (ridge) obs→embedding map, then nearest token (the strong
                         single-position baseline; ≈ tuned-lens affine-into-E).
  skip_decoder_attack  — gated linear-skip GELU decoder, ridge-warm-started + frozen linear path,
                         early-stopped on a disjoint val split → clean `decoder ≥ ridge`.
  (BeamClean LM-prior beam decode will land here as beamclean_attack — campaign-D Task 5.)
"""
from __future__ import annotations

import copy

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def nearest_token(pred_emb, pool_emb, pool_ids):
    """Cosine nearest-neighbour decode of predicted embeddings against a candidate pool."""
    p = pred_emb / np.clip(np.linalg.norm(pred_emb, axis=1, keepdims=True), 1e-9, None)
    e = pool_emb / np.clip(np.linalg.norm(pool_emb, axis=1, keepdims=True), 1e-9, None)
    return pool_ids[(p @ e.T).argmax(1)]


def ridge_W(Xtr, Etr, alpha=1.0):
    """Closed-form ridge map X→E (float64 Gram/solve for stability; returned float32 [d_in, d_out])."""
    d = Xtr.shape[1]
    A = (Xtr.T @ Xtr).astype(np.float64) + alpha * np.eye(d, dtype=np.float64)
    return np.linalg.solve(A, (Xtr.T @ Etr).astype(np.float64)).astype(np.float32)


def ridge_attack(Xtr, Etr, Xte, pool_emb, pool_ids, *, alpha=1.0, **_):
    """Linear (ridge) obs→embedding map, then nearest token. The strong single-position baseline."""
    return nearest_token(Xte @ ridge_W(Xtr, Etr, alpha), pool_emb, pool_ids)


class LinearSkipDecoder(torch.nn.Module):
    """Gated linear-skip decoder: pred = Linear(x) + gate · MLP(x), GELU, narrow hidden.

    The linear path is warm-started to ridge and FROZEN (carries the strong affine/tuned-lens
    baseline); the GELU MLP adds a gated non-linear correction with gate init 0, so the decoder
    starts identical to ridge and the non-linearity switches on only if the held-out split supports
    it. GELU (not ReLU) avoids dead-neuron narrowing; a NARROW hidden (≤ input) + early-stop control
    overfitting in the small-data regime. See research-wiki/claims/single-position-residual-linearly-saturated.md.
    """

    def __init__(self, d_in, d_out, hidden):
        super().__init__()
        self.lin = torch.nn.Linear(d_in, d_out)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d_out)
        )
        self.gate = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.lin(x) + self.gate * self.mlp(x)


def skip_decoder_attack(Xtr, Etr, Xte, pool_emb, pool_ids, *, hidden=384, epochs=500, lr=1e-3,
                        seed=0, val_frac=0.15, patience=40, **_):
    """Gated linear-skip GELU decoder, ridge-warm-started + FROZEN linear path, early-stopped.

    The linear path is set to ridge's exact W and frozen; only the gated GELU correction (gate=0,
    zero MLP tail at init) trains, with weight-decay + early-stopping on a held-out split. So the
    decoder starts identical to ridge and can only add a data-supported non-linear correction —
    a clean `decoder ≥ ridge` guarantee with overfitting control.
    """
    torch.manual_seed(seed)
    net = LinearSkipDecoder(Xtr.shape[1], Etr.shape[1], hidden).to(DEV)
    with torch.no_grad():  # warm-start: linear path = ridge, gate=0, MLP tail=0 → starts AT ridge
        net.lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(ridge_W(Xtr, Etr).T)).to(DEV))
        net.lin.bias.zero_()
        net.mlp[-1].weight.zero_()
        net.mlp[-1].bias.zero_()
    for p in net.lin.parameters():  # freeze the ridge baseline; train only the gated correction
        p.requires_grad_(False)

    def _unit(A):
        t = torch.from_numpy(A).to(DEV)
        return t / t.norm(dim=1, keepdim=True).clamp_min(1e-9)

    perm = np.random.default_rng(seed).permutation(Xtr.shape[0])
    nval = max(1, int(val_frac * Xtr.shape[0]))
    vi, ti = perm[:nval], perm[nval:]
    xt, yt = torch.from_numpy(Xtr[ti]).to(DEV), _unit(Etr[ti])
    xv, yv = torch.from_numpy(Xtr[vi]).to(DEV), _unit(Etr[vi])
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-3)
    best, best_state, bad = float("inf"), copy.deepcopy(net.state_dict()), 0  # init state = ridge
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        p = net(xt)
        p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-9)
        (1.0 - (p * yt).sum(1)).mean().backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(xv)
            pv = pv / pv.norm(dim=1, keepdim=True).clamp_min(1e-9)
            vloss = (1.0 - (pv * yv).sum(1)).mean().item()
        if vloss < best - 1e-5:
            best, best_state, bad = vloss, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)  # best held-out state (ridge-at-init is in the running → ≥ ridge)
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(Xte).to(DEV)).cpu().numpy()
    return nearest_token(pred, pool_emb, pool_ids)

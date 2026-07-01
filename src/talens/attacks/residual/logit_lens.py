"""ISA-HiddenState · CE logit-lens (objective-matched non-linearity test)."""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F

from .._common import nearest_token, ridge_W, DEV


class LensHead(torch.nn.Module):
    """Residual→query head for the CE logit lens: q = A·x + b [+ gate·GELU-MLP(x)].

    Logits are formed downstream as E·q against the FROZEN embedding table (no per-token params), so
    the head only learns a query map. ReZero init for the non-linear branch (gate=0 but MLP tail at
    normal init → live gate gradient). See docs/research/ce-logit-lens-attack.md.
    """

    def __init__(self, d_in, d_emb, hidden, nonlinear):
        super().__init__()
        self.lin = torch.nn.Linear(d_in, d_emb)
        self.nonlinear = nonlinear
        if nonlinear:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d_emb)
            )
            self.gate = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        q = self.lin(x)
        if self.nonlinear:
            q = q + self.gate * self.mlp(x)
        return q


def logit_lens_attack(Xtr, Etr, Xte, pool_emb, pool_ids, *, ytr, full_emb, nonlinear=True,
                      neg=2048, hidden=384, epochs=500, lr=1e-3, seed=0, val_frac=0.15,
                      patience=40, **_):
    """CE logit-lens attack — objective-matched, fair non-linearity test (docs/research/ce-logit-lens-attack.md).

    Trains a query head h(x) scored against the frozen embedding table `full_emb` with COSINE-softmax
    cross-entropy (query + embeddings row-normalised, fixed temperature) over a SAMPLED full vocab
    (candidate set = unique train tokens ∪ `neg` random negatives, resampled per epoch). Early-stopped
    on val TOP-1 recovery (seeded with the ridge warm-start → ≥ ridge), decoded by COSINE over the test
    `pool` (identical convention to ridge). Cosine — not raw dot product — because gemma token-embedding
    norms are heterogeneous: dot-product logits rank by ‖E‖ and badly under-recover (init dot 0.78 vs
    cosine 1.0 on norm-heterogeneous synthetic); and val-top1 (not val-CE) early-stop keeps the metric
    aligned. `ytr` = train token ids (CE positives — pass the SAME shuffle as `Etr` for the floor);
    `Etr` warm-starts the affine to ridge. nonlinear=False ⇒ tuned-lens affine; True ⇒ + gated GELU.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d_in, d_emb, V = Xtr.shape[1], full_emb.shape[1], full_emb.shape[0]
    ytr = np.asarray(ytr).astype(np.int64)
    assert ytr.shape[0] == Xtr.shape[0] and (0 <= ytr).all() and (ytr < V).all(), \
        "ytr must be valid token ids aligned to Xtr (CE positives index full_emb)"
    net = LensHead(d_in, d_emb, hidden, nonlinear).to(DEV)
    with torch.no_grad():  # warm-start the affine path to ridge (≈ tuned-lens start)
        net.lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(ridge_W(Xtr, Etr).T)).to(DEV))
        net.lin.bias.zero_()

    perm = rng.permutation(Xtr.shape[0])
    nval = max(1, int(val_frac * Xtr.shape[0]))
    vi, ti = perm[:nval], perm[nval:]
    xt, yt = torch.from_numpy(Xtr[ti]).to(DEV), ytr[ti]
    xv, yv = torch.from_numpy(Xtr[vi]).to(DEV), ytr[vi]
    Efull = torch.from_numpy(np.ascontiguousarray(full_emb))  # CPU; gather candidate rows per step
    train_toks = np.unique(ytr)  # always in the candidate set (CE positives must be in the denominator)

    def _unit(t):  # row-normalise → cosine geometry (decouples from gemma's heterogeneous token norms)
        return t / t.norm(dim=1, keepdim=True).clamp_min(1e-9)

    # COSINE softmax CE (normalise query + embeddings, fixed temperature) — matches the cosine decode;
    # raw dot-product logits would rank by ‖E‖ and badly under-recover (init dot 0.78 vs cosine 1.0).
    tau = 0.05
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)

    def _val_top1():  # early-stop METRIC = val top-1 recovery (train-token retrieval) — aligned with
        with torch.no_grad():  # the test metric, unlike val CE loss (which drifts off it)
            negs = rng.choice(V, size=min(neg, V), replace=False)
            cand = np.unique(np.concatenate([train_toks, negs]))
            Ec = _unit(Efull[cand].to(DEV))
            tgt = torch.from_numpy(np.searchsorted(cand, yv)).to(DEV)
            return float((( _unit(net(xv)) @ Ec.T).argmax(1) == tgt).float().mean())

    best, best_state, bad = _val_top1(), copy.deepcopy(net.state_dict()), 0  # seed with ridge init → ≥ ridge
    for _ in range(epochs):
        negs = rng.choice(V, size=min(neg, V), replace=False)
        # cand sorted+deduped; train_toks ⊆ cand so every positive yt/yv is in it →
        # searchsorted returns the EXACT class index (no off-by-one; collisions dedup'd).
        cand = np.unique(np.concatenate([train_toks, negs]))
        Ec = _unit(Efull[cand].to(DEV))
        tgt_t = torch.from_numpy(np.searchsorted(cand, yt)).to(DEV)
        net.train()
        opt.zero_grad()
        F.cross_entropy((_unit(net(xt)) @ Ec.T) / tau, tgt_t).backward()
        opt.step()
        net.eval()
        vrec = _val_top1()
        if vrec > best + 1e-5:
            best, best_state, bad = vrec, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():  # decode by COSINE over the test pool (identical convention to ridge)
        pred = net(torch.from_numpy(Xte).to(DEV)).cpu().numpy()
    return nearest_token(pred, pool_emb, pool_ids)

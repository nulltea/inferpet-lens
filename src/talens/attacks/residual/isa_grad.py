"""ISA-HiddenState · §D.1 gradient-optimization ISA (analysis-by-synthesis)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .._common import nearest_token, ridge_W, DEV


def isa_grad_attack(Xtr, Etr, Xte, pool_emb, pool_ids, *, alpha=1.0, steps=500, lr=5e-2, seed=0, **_):
    """AloePri §D.1 gradient-optimization ISA — analysis-by-synthesis on input embeddings.

    Ridge fits the DISCRIMINATIVE obs→embedding map and decodes in one closed form. §D.1 instead
    fits the GENERATIVE forward-synthesis map Ŵ_g: clean-embedding → observed deployment rep from
    the harvested pairs (= the obfuscation P̂ at L0), then RECOVERS each held-out observed rep x̃ by
    GRADIENT-OPTIMIZING an input embedding e to minimise ‖e·Ŵ_g − x̃‖² (Adam from e=0), nearest-token e.

    "ridge bypasses the optimization landscape" (private-rag 2B.1 doc, §D.1): this runs the actual
    descent over input-embedding space rather than a closed-form inverse. With a LINEAR synthesis map
    the landscape is convex, so the descent lands at the generative least-squares inverse
    e* = x̃·Ŵ_g⁺ — which still differs from ridge's discriminative map when the obfuscation WIDENS the
    basis (d̃ = d+2h > d_emb, so e·Ŵ_g = x̃ is over-determined) or under αₑ output noise (synthesis
    models it as output noise, the physically-correct direction; ridge folds it into the input). Whether
    that gap survives a harvest — i.e. whether gradient-opt ISA beats or merely equals ridge — is the test.
    """
    torch.manual_seed(seed)
    Wg = torch.from_numpy(ridge_W(Etr, Xtr, alpha)).to(DEV)            # clean emb → obs rep (d_emb, d̃)
    Xt = torch.from_numpy(np.ascontiguousarray(Xte)).to(DEV)
    e = torch.zeros(Xt.shape[0], Wg.shape[0], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([e], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(e @ Wg, Xt).backward()
        opt.step()
    return nearest_token(e.detach().cpu().numpy(), pool_emb, pool_ids)

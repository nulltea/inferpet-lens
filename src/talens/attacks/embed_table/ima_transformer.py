"""IMA-EmbedRow · §F.1 2-layer/8-head transformer inverter."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .._common import nearest_token, DEV


class _IMABlock(torch.nn.Module):
    """One inverter block: pre-LN MultiheadAttention (seq=1 ⇒ ≈ identity) + pre-LN GELU FFN, residual."""

    def __init__(self, hidden, n_heads, ffn_mult=4):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(hidden)
        self.attn = torch.nn.MultiheadAttention(hidden, n_heads, batch_first=True, bias=False)
        self.ln2 = torch.nn.LayerNorm(hidden)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(hidden, hidden * ffn_mult, bias=False),
                                       torch.nn.GELU(), torch.nn.Linear(hidden * ffn_mult, hidden, bias=False))

    def forward(self, x):
        xn = self.ln1(x)
        x = x + self.attn(xn, xn, xn, need_weights=False)[0]
        return x + self.ffn(self.ln2(x))


class IMAInverter(torch.nn.Module):
    """AloePri IMA paper §F.1 inverter: input proj → n_layers×(MHA+GELU-FFN) → output proj.
    Port of private-rag run_ima_paper_like.IMAInverter (2 layers, 8 heads). For the static embedding
    table each row is its own sequence (seq=1), so the attention is near-identity and the block is an
    LN+GELU-FFN MLP; the architecture is kept faithful so the L0/sequence variant can reuse it."""

    def __init__(self, observed_dim, output_dim, hidden=768, n_layers=2, n_heads=8, ffn_mult=4):
        super().__init__()
        self.input_proj = torch.nn.Linear(observed_dim, hidden, bias=False) if observed_dim != hidden else torch.nn.Identity()
        self.blocks = torch.nn.ModuleList([_IMABlock(hidden, n_heads, ffn_mult) for _ in range(n_layers)])
        self.output_proj = torch.nn.Linear(hidden, output_dim, bias=False)

    def forward(self, x):
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.output_proj(h)


def ima_transformer_attack(Xtr, Etr, Xte, pool_emb, pool_ids, *, hidden=768, n_layers=2, n_heads=8,
                           epochs=12, lr=1e-3, batch=4096, seed=0, **_):
    """AloePri IMA-EmbedRow-transformer (paper §F.1): train the 2-layer/8-head IMAInverter on
    (observed-row → plain-embedding) pairs with MSE, then decode test rows by nearest-token. The caller
    decides the threat model via what it passes as Xtr/Etr: a SINGLE key (matched / plain control) or
    many synthetic own-key obfuscations (τ-invariant, the in-model attack). Rows are independent (seq=1)."""
    torch.manual_seed(seed)
    net = IMAInverter(Xtr.shape[1], Etr.shape[1], hidden=hidden, n_layers=n_layers, n_heads=n_heads).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr)).to(DEV)
    Et = torch.from_numpy(np.ascontiguousarray(Etr)).to(DEV)
    n = Xt.shape[0]
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = net(Xt[idx].unsqueeze(1)).squeeze(1)          # (b,1,d)->(b,d)
            torch.nn.functional.mse_loss(pred, Et[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(np.ascontiguousarray(Xte)).to(DEV).unsqueeze(1)).squeeze(1).cpu().numpy()
    return nearest_token(out, pool_emb, pool_ids)

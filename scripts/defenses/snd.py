"""Split-and-Denoise (SnD) defense — dχ-privacy on token embeddings + a noise-aware denoiser.

Mai et al. 2023, arXiv:2310.09130 (paper:mai2023_splitanddenoise_protect_large).
Reference impl: https://github.com/NusIoraPrivacy/eaas-privacy

Repo rule: defenses live in scripts/defenses/. evals import and apply them.
DxPrivacy is a forward hook on the input-embedding layer (sibling to LocalDP), but uses
the dχ-privacy Laplacian (Wu et al. 2017) instead of the Gaussian mechanism:
  direction v = g/||g||, g~N(0,I_d); magnitude l~Gamma(d, 1/eta); M(x)=x+l*v; clip to C.
η is the dχ budget (larger = weaker privacy); it is NOT the Gaussian (ε,δ).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class DxPrivacy:
    """dχ-privacy mechanism as a forward hook on the token-embedding layer.

    eta=inf ⇒ clip-only (≈ clean). Per-row: add d-dim Laplacian noise, then L2-clip to C.
    """

    def __init__(self, C: float, eta: float):
        self.C, self.eta = C, eta

    def __call__(self, mod, inp, out):  # noqa: ARG002 (hook signature)
        f = out.float()
        if math.isfinite(self.eta):
            g = torch.randn_like(f)
            v = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-9)        # unit sphere
            # l ~ Gamma(shape=d, scale=1/eta), one magnitude per row
            d = f.shape[-1]
            gamma = torch.distributions.Gamma(torch.tensor(float(d)), torch.tensor(float(self.eta)))
            l = gamma.sample(f.shape[:-1]).to(f).unsqueeze(-1)          # (..., 1)
            f = f + l * v
        n = f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        f = f * (self.C / n).clamp_max(1.0)                            # clip to C
        return f.to(out.dtype)


class Denoiser(nn.Module):
    """Noise-aware transformer: reconstruct clean pooled embedding e_c from (e_n, X̃, Z).

    Token sequence [e_n] ++ X̃ ++ Z (length 2T+1), + a learned type embedding (out/raw/noise)
    + sinusoidal position, L encoder layers (paper Table 8 uses L=6, d_ff≈d for base-size models);
    read the e_n slot → linear → e_d. Conditions on Z, so one model serves an η range; the paper
    trains a separate model per η-group (route at inference). Padding masked via src_key_padding_mask.

    residual=True (default): e_d = e_n + head(h₀) with head ZERO-INIT, so the denoiser starts as exact
    passthrough and only learns the noise CORRECTION. This fixes the absorbing-denoiser failure (a
    raw-readout denoiser can't even reproduce a lightly-noised embedding → hurts recovery at mild
    noise; its clean ceiling sat at ~0.84). With the residual the clean ceiling is ~1.0 by construction
    and recovery cannot go meaningfully negative. The paper's own h_t^l = h_t^{l-1}+a+m residual stream
    motivates this; here it is made explicit at the readout.
    """

    def __init__(self, d: int, n_layers: int = 3, n_heads: int = 8, d_ff: int | None = None,
                 dropout: float = 0.0, residual: bool = True):
        super().__init__()
        self.d = d
        self.residual = residual
        self.type_emb = nn.Embedding(3, d)              # 0=output, 1=raw, 2=noise
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=(d_ff or d), dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d, d)
        if residual:                                    # zero-init ⇒ e_d == e_n at init (passthrough)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def _posenc(T: int, d: int, device) -> torch.Tensor:
        pos = torch.arange(T, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(T, d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe

    def forward(self, e_n, X_tilde, Z, pad_mask=None):
        B, T, d = X_tilde.shape
        seq = torch.cat([e_n.unsqueeze(1), X_tilde, Z], dim=1)          # (B, 2T+1, d)
        types = torch.cat([
            torch.zeros(1, dtype=torch.long, device=seq.device),
            torch.ones(T, dtype=torch.long, device=seq.device),
            torch.full((T,), 2, dtype=torch.long, device=seq.device),
        ])
        seq = seq + self.type_emb(types).unsqueeze(0)
        pe = self._posenc(T, d, seq.device)
        seq = seq + torch.cat([torch.zeros(1, d, device=seq.device), pe, pe], dim=0).unsqueeze(0)
        kpm = None
        if pad_mask is not None:                                       # (B, T) → (B, 2T+1)
            f = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
            kpm = torch.cat([f, pad_mask, pad_mask], dim=1)
        h = self.enc(seq, src_key_padding_mask=kpm)
        out = self.head(h[:, 0])                                        # e_n slot → correction/e_d
        return e_n + out if self.residual else out

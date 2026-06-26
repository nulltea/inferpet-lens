import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from defenses.snd import DxPrivacy, Denoiser


def _apply(hook, x):
    """Invoke the forward hook the way register_forward_hook would: out is the embedding tensor."""
    return hook(None, None, x)


def test_dxprivacy_clips_to_C():
    torch.manual_seed(0)
    x = torch.randn(4, 7, 16) * 5.0          # (B, T, d), large norms
    out = _apply(DxPrivacy(C=2.0, eta=10.0), x)
    assert out.shape == x.shape
    norms = out.float().norm(dim=-1)
    assert torch.all(norms <= 2.0 + 1e-4), norms.max().item()


def test_dxprivacy_noise_scales_with_eta():
    torch.manual_seed(0)
    x = torch.randn(8, 12, 16)
    # large C so clipping rarely binds → Z reflects the raw mechanism noise
    z_small_eta = (_apply(DxPrivacy(C=1e6, eta=1.0), x) - x).norm(dim=-1).mean()
    z_large_eta = (_apply(DxPrivacy(C=1e6, eta=50.0), x) - x).norm(dim=-1).mean()
    assert z_small_eta > z_large_eta            # smaller η = more noise


def test_dxprivacy_inf_eta_is_clip_only():
    x = torch.randn(3, 5, 16) * 0.1             # norms < C → clip is a no-op
    out = _apply(DxPrivacy(C=100.0, eta=math.inf), x)
    assert torch.allclose(out, x)


def test_denoiser_output_shape():
    torch.manual_seed(0)
    B, T, d = 2, 5, 16
    m = Denoiser(d=d, n_layers=2, n_heads=4)
    e_d = m(torch.randn(B, d), torch.randn(B, T, d), torch.randn(B, T, d))
    assert e_d.shape == (B, d)


def test_denoiser_overfits_toward_clean():
    """A few steps on a fixed batch should pull denoised cos-to-e_c above the raw cos(e_n,e_c)."""
    torch.manual_seed(0)
    B, T, d = 8, 4, 16
    e_c = torch.randn(B, d)
    Z = 0.5 * torch.randn(B, T, d)
    X = torch.randn(B, T, d)
    e_n = e_c + 0.8 * torch.randn(B, d)                 # noised observation of e_c
    cos = torch.nn.functional.cosine_similarity
    base = cos(e_n, e_c).mean().item()
    m = Denoiser(d=d, n_layers=2, n_heads=4)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        loss = ((m(e_n, X, Z) - e_c) ** 2).mean()
        loss.backward(); opt.step()
    after = cos(m(e_n, X, Z), e_c).mean().item()
    assert after > base, (base, after)

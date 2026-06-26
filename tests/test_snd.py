import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from defenses.snd import DxPrivacy


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

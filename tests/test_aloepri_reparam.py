"""Covariant re-parameterization of a transformer (AloePri, paper arXiv 2603.01499).

The faithful obfuscation rewrites every residual-touching weight with one residual
key pair (P̂, Q̂), P̂ Q̂ = I_d, so the obfuscated forward is bit-equivalent to plaintext
while the residual stream the server sees is the obfuscated x' = x P̂. The paper only
treats RMSNorm (§5.2.5, a Gaussian κ approximation); Pythia uses LayerNorm (mean +
bias), so the covariant norm is derived here EXACTLY from P̂ Q̂ = I (no approximation):
a wrapped norm computes LN(x' Q̂) P̂ = LN(x) P̂.

These host tests pin the covariance ALGEBRA (model-free, torch only — the direction of
every read/write rewrite, where the bugs live). The real-pythia logits-identity GATE
needs transformers + the model and runs in the ROCm container (importorskip below).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from defenses.aloepri import (
    CovariantLayerNorm,
    keymat_gen,
    obf_read_weight,
    obf_write_weight,
)


def _PQ(d: int, h: int, seed: int = 0):
    P, Q = keymat_gen(d, h, lam=0.3, seed=seed)
    return torch.from_numpy(P).double(), torch.from_numpy(Q).double()


def test_covariant_layernorm_is_exact_in_obf_basis():
    """Wrapped LayerNorm on the obfuscated residual reproduces the plaintext LayerNorm
    output in the P̂-basis, exactly (the novel LayerNorm covariance, mean + bias)."""
    d, h = 16, 8
    P, Q = _PQ(d, h)
    ln = nn.LayerNorm(d).double()
    nn.init.normal_(ln.weight)
    nn.init.normal_(ln.bias)
    x = torch.randn(5, d, dtype=torch.float64)
    y_plain = ln(x)                                # (5, d)
    cov = CovariantLayerNorm(ln, P, Q)
    y_obf = cov(x @ P)                             # obf residual in → obf residual out
    assert torch.allclose(y_obf, y_plain @ P, atol=1e-5), (y_obf - y_plain @ P).abs().max()


def test_covariant_read_linear_matches_plaintext():
    """A linear that READS the residual (q/k/v, mlp-in): rewritten weight on the obf
    residual gives the identical plaintext output; bias is unchanged (output stays in
    the plaintext within-block space)."""
    d, h, out = 16, 8, 7
    P, Q = _PQ(d, h)
    lin = nn.Linear(d, out, bias=True).double()
    nn.init.normal_(lin.weight)
    nn.init.normal_(lin.bias)
    x = torch.randn(5, d, dtype=torch.float64)
    W2 = obf_read_weight(lin.weight.data, Q)       # (out, d+2h)
    y_plain = lin(x)
    y_obf = (x @ P) @ W2.t() + lin.bias.data       # F.linear on the obf residual
    assert torch.allclose(y_obf, y_plain, atol=1e-5), (y_obf - y_plain).abs().max()


def test_covariant_write_linear_produces_obf_residual():
    """A linear that WRITES the residual (attn-out, mlp-out): output is the plaintext
    output mapped into the P̂-basis. Bias maps too (bias adds in residual space)."""
    d, h, inn = 16, 8, 7
    P, Q = _PQ(d, h)
    lin = nn.Linear(inn, d, bias=True).double()
    nn.init.normal_(lin.weight)
    nn.init.normal_(lin.bias)
    x = torch.randn(5, inn, dtype=torch.float64)
    W2, b2 = obf_write_weight(lin.weight.data, lin.bias.data, P)   # (d+2h,inn), (d+2h,)
    y_plain = lin(x)                               # (5, d)
    y_obf = x @ W2.t() + b2                        # (5, d+2h)
    assert torch.allclose(y_obf, y_plain @ P, atol=1e-5), (y_obf - y_plain @ P).abs().max()


# ───────────────────────── real-pythia logits-identity GATE (container) ─────────────────────────
def test_reparam_pythia_preserves_logits():
    """THE gate: a covariantly re-parameterized pythia-160m (keymat only, Π=I, no noise)
    produces logits identical to plaintext to fp32 tolerance. If this fails the LayerNorm
    covariance / read-write directions are wrong and any recovery result is a confound.

    Needs transformers + the cached model → runs in the ROCm container, skipped on host.
    """
    transformers = pytest.importorskip("transformers")
    from defenses.aloepri import reparam_pythia

    tok = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m", torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids
    with torch.no_grad():
        ref = model(ids, use_cache=False).logits

    reparam_pythia(model, config="keymat_only", h=128, lam=0.3, seed=0)
    with torch.no_grad():
        got = model(ids, use_cache=False).logits

    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-3), (got - ref).abs().max().item()


def _pythia():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m", torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids
    return model, ids


def test_full_alg1_noise_zero_equals_keymat():
    """full_alg1 with αₑ=α_h=0 is exactly keymat_only → logits still identical to plaintext
    (the noise=0 regression path)."""
    from defenses.aloepri import reparam_pythia
    model, ids = _pythia()
    with torch.no_grad():
        ref = model(ids, use_cache=False).logits
    reparam_pythia(model, config="full_alg1", h=128, lam=0.3, seed=0, alpha_e=0.0, alpha_h=0.0)
    with torch.no_grad():
        got = model(ids, use_cache=False).logits
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-3), (got - ref).abs().max().item()


def test_full_alg1_embedding_noise_is_lossy_but_runs():
    """αₑ>0 adds embedding noise (Wₑ+αₑσₑE)P̂ → logits genuinely differ from plaintext (lossy),
    shape preserved, forward runs. The αₑ knob is the only thing that moves residual recovery."""
    from defenses.aloepri import reparam_pythia
    model, ids = _pythia()
    with torch.no_grad():
        ref = model(ids, use_cache=False).logits
    reparam_pythia(model, config="full_alg1", h=128, lam=0.3, seed=0, alpha_e=1.0, alpha_h=0.0)
    with torch.no_grad():
        got = model(ids, use_cache=False).logits
    assert got.shape == ref.shape
    assert not torch.allclose(got, ref, atol=1e-2), "αₑ=1.0 embedding noise should perturb logits"

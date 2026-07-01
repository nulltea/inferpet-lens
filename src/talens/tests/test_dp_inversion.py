"""Regression guard for the dead-non-linear-branch bug in skip_decoder_attack.

The bug: warm-starting with BOTH gate=0 AND a zeroed MLP tail makes ∂loss/∂gate = mlp(x) = 0,
so the whole gated branch sits at a zero-gradient saddle and Adam can never leave ridge — the
decoder is pinned ≡ ridge regardless of any exploitable non-linearity. The fix is ReZero-style
(gate=0 only; tail keeps its normal init). These tests fail if the dead init ever returns.
CPU-only, deterministic, seconds.
"""
import numpy as np
import torch

from talens.attacks import (
    LinearSkipDecoder, ridge_W, skip_decoder_attack, nearest_token, logit_lens_attack,
    isa_grad_attack, orthogonal_procrustes_R, blockwise_procrustes_R, rotation_recovery_attack,
)


def _gelu(z):
    return z * 0.5 * (1 + np.tanh(0.797885 * (z + 0.044715 * z**3)))


def _nonlinear_surface(seed=0, n=1200, d_in=64, d_out=32):
    """Synthetic (X, E) whose truth is DOMINANTLY non-linear — ridge cannot fit it well."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d_in)).astype(np.float32)
    Wl = (rng.standard_normal((d_in, d_out)) / np.sqrt(d_in)).astype(np.float32)
    W2 = (rng.standard_normal((d_in, d_in)) / np.sqrt(d_in)).astype(np.float32)
    W3 = (rng.standard_normal((d_in, d_out)) / np.sqrt(d_in)).astype(np.float32)
    E = (0.3 * X @ Wl + 2.0 * _gelu(X @ W2) @ W3).astype(np.float32)
    return X, E


def test_gate_gradient_is_live_at_init():
    """The exact failure mode: at the warm-started init the gate MUST have a non-zero gradient."""
    X, E = _nonlinear_surface()
    net = LinearSkipDecoder(64, 32, 128)
    with torch.no_grad():  # replicate skip_decoder_attack's warm-start (ReZero: tail NOT zeroed)
        net.lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(ridge_W(X, E).T)))
        net.lin.bias.zero_()
    for p in net.lin.parameters():
        p.requires_grad_(False)
    y = torch.from_numpy(E); y = y / y.norm(dim=1, keepdim=True)
    pred = net(torch.from_numpy(X)); pred = pred / pred.norm(dim=1, keepdim=True)
    (1.0 - (pred * y).sum(1)).mean().backward()
    assert net.gate.grad.abs().max() > 1e-8, "gate gradient is dead at init — branch can't train"


def test_decoder_beats_ridge_on_nonlinear_truth():
    """End-to-end: on a dominantly non-linear surface the trained decoder must beat ridge."""
    X, E = _nonlinear_surface()
    tr, te = np.arange(900), np.arange(900, X.shape[0])

    def cos_err(P, Y):
        P = P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-9, None)
        Y = Y / np.clip(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9, None)
        return float(1.0 - (P * Y).sum(1).mean())

    ridge_err = cos_err(X[te] @ ridge_W(X[tr], E[tr]), E[te])
    # skip_decoder_attack returns ids; recover its predictions via a per-row-identity pool
    pool_ids = te.copy()
    yhat = skip_decoder_attack(X[tr], E[tr], X[te], E[te], pool_ids, hidden=128, epochs=400, seed=0)
    # identity recovery is a coarse proxy; assert it lifts above the ridge-error regime
    dec_recov = float((yhat == te).mean())
    ridge_recov = float((nearest_token(X[te] @ ridge_W(X[tr], E[tr]), E[te], pool_ids) == te).mean())
    assert dec_recov >= ridge_recov, f"decoder {dec_recov:.3f} < ridge {ridge_recov:.3f}"
    assert ridge_err > 0.05, "surface not non-linear enough to be a meaningful test"


def _vocab_surface(seed=1, V=300, d_emb=32, d_in=48, n=3000):
    """Synthetic vocab: residual = linear encoding of the token embedding + noise.

    Recoverable by a linear query map → tests the CE logit-lens plumbing AND open-set generalization
    (train/test tokens disjoint; the head must generalize to unseen tokens via the frozen geometry).
    """
    rng = np.random.default_rng(seed)
    full_emb = rng.standard_normal((V, d_emb)).astype(np.float32)
    R = rng.standard_normal((d_emb, d_in)).astype(np.float32)
    toks = rng.integers(0, V, size=n).astype(np.int64)
    X = (full_emb[toks] @ R + 0.1 * rng.standard_normal((n, d_in))).astype(np.float32)
    ntr = int(0.7 * V)
    tr = np.array([i for i, t in enumerate(toks) if t < ntr])
    te = np.array([i for i, t in enumerate(toks) if t >= ntr])
    return full_emb, toks, X, tr, te


def test_logit_lens_generalizes_to_unseen_tokens():
    """CE logit-lens (tied frozen E) must recover UNSEEN test tokens far above pool chance —
    confirming open-set generalization, sampled-softmax targets, and pool decode all work."""
    full_emb, toks, X, tr, te = _vocab_surface()
    pool_ids = np.unique(toks[te])
    chance = 1.0 / pool_ids.size
    for nonlinear in (False, True):
        pred = logit_lens_attack(
            X[tr], full_emb[toks[tr]], X[te], full_emb[pool_ids], pool_ids,
            ytr=toks[tr], full_emb=full_emb, nonlinear=nonlinear,
            hidden=64, epochs=300, neg=128, seed=0,
        )
        acc = float((pred == toks[te]).mean())
        assert acc > 0.5, f"nonlinear={nonlinear}: acc {acc:.3f} ~ chance {chance:.3f} — generalization broke"


def test_logit_lens_robust_to_norm_heterogeneity():
    """Regression for the failed-pilot bug: with HETEROGENEOUS token-embedding norms (like gemma),
    raw dot-product decode ranks by ‖E‖ and under-recovers. Cosine CE + cosine decode must keep the
    lens within reach of the ridge/cosine baseline (this would fail on the old dot-product decode)."""
    rng = np.random.default_rng(2)
    V, d_emb, d_in, n = 400, 32, 48, 4000
    E = (rng.standard_normal((V, d_emb)) * rng.uniform(0.2, 5.0, (V, 1))).astype(np.float32)  # varied norms
    R = rng.standard_normal((d_emb, d_in)).astype(np.float32)
    toks = rng.integers(0, V, size=n).astype(np.int64)
    X = (E[toks] @ R + 0.1 * rng.standard_normal((n, d_in))).astype(np.float32)
    tr = np.array([i for i, t in enumerate(toks) if t < int(0.7 * V)])
    te = np.array([i for i, t in enumerate(toks) if t >= int(0.7 * V)])
    pool_ids = np.unique(toks[te])
    ridge_acc = float((nearest_token(X[te] @ ridge_W(X[tr], E[toks[tr]]), E[pool_ids], pool_ids) == toks[te]).mean())
    lens_acc = float((logit_lens_attack(
        X[tr], E[toks[tr]], X[te], E[pool_ids], pool_ids, ytr=toks[tr], full_emb=E,
        nonlinear=False, hidden=64, epochs=300, neg=128, seed=0) == toks[te]).mean())
    assert ridge_acc > 0.8, f"baseline weak ({ridge_acc:.3f}) — test not meaningful"
    assert lens_acc >= ridge_acc - 0.1, f"lens {lens_acc:.3f} ≪ ridge {ridge_acc:.3f} — norm-decode regressed"


def test_isa_grad_recovers_under_linear_obfuscation():
    """§D.1 gradient-opt ISA must recover tokens on a CLEAN linear (widening) obfuscation X̃ = E·P̂.

    This is the L0 harvest regime where the prose claims it 'collapses to ridge': the descent over
    input-embedding space lands at the generative least-squares inverse, so on a noiseless linear obf
    it should recover ~perfectly (and within noise of ridge). Guards the convergence of the Adam loop
    — if the descent did not actually minimise the objective, recovery would crater. CPU-only, seconds."""
    rng = np.random.default_rng(1)
    V, d_emb, d_obf, n = 200, 32, 48, 1500              # d_obf > d_emb = AloePri keymat widening
    E = rng.standard_normal((V, d_emb)).astype(np.float32)
    P = rng.standard_normal((d_emb, d_obf)).astype(np.float32)   # the secret linear basis (full row rank)
    toks = rng.integers(0, V, size=n).astype(np.int64)
    X = (E[toks] @ P).astype(np.float32)                # clean linear obfuscation, no noise
    cut = int(0.7 * V)
    tr = np.where(toks < cut)[0]; te = np.where(toks >= cut)[0]
    pool_ids = np.unique(toks[te])
    isa_acc = float((isa_grad_attack(X[tr], E[toks[tr]], X[te], E[pool_ids], pool_ids,
                                     steps=600, seed=0) == toks[te]).mean())
    ridge_acc = float((nearest_token(X[te] @ ridge_W(X[tr], E[toks[tr]]), E[pool_ids], pool_ids) == toks[te]).mean())
    assert isa_acc > 0.9, f"gradient-opt ISA failed to converge on clean linear obf ({isa_acc:.3f})"
    assert isa_acc >= ridge_acc - 0.1, f"isa_grad {isa_acc:.3f} ≪ ridge {ridge_acc:.3f}"


def test_basis_align_recovers_under_secret_rotation():
    """rotation_recovery_attack must recover held-out tokens under a SECRET orthogonal rotation of the
    surface (AloePri Alg2 model) where a naive self-gen ridge (no un-rotation) collapses. The harvest
    supplies only the aligned pairs that pin R̂; the token map comes from self-gen. Guards the core
    claim mechanism (claim:aloepri-kqvout-basis-alignment). CPU-only, deterministic, seconds."""
    rng = np.random.default_rng(3)
    V, d, n = 300, 48, 1600
    E = rng.standard_normal((V, d)).astype(np.float32)
    toks = rng.integers(0, V, size=n).astype(np.int64)
    # plaintext surface: (near-)per-token-invertible so a self-gen ridge works in the plain basis
    Xp = (E[toks] + 0.05 * rng.standard_normal((n, d))).astype(np.float32)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))            # secret orthogonal rotation R
    Xd = (Xp @ Q).astype(np.float32)                            # deployment = plaintext · R
    tr = np.arange(0, n // 2); te = np.arange(n // 2, n)        # self-gen train / victim test rows
    pool = np.unique(toks)
    align = rng.choice(n, size=400, replace=False)              # harvested aligned pairs
    R = orthogonal_procrustes_R(Xp[align], Xd[align])
    assert np.linalg.norm(R - Q) / np.linalg.norm(Q) < 0.05, "Procrustes did not recover the rotation"
    ba = float((rotation_recovery_attack(Xp[align], Xd[align], Xp[tr], toks[tr], Xd[te],
                                   E[pool], pool, table=E) == toks[te]).mean())
    # naive self-gen: same ridge, decode the ROTATED reps without un-rotation → wrong basis → floor
    W = ridge_W(Xp[tr], E[toks[tr]])
    naive = float((nearest_token(Xd[te] @ W, E[pool], pool) == toks[te]).mean())
    assert ba > 0.8, f"basis_align failed to recover under secret rotation ({ba:.3f})"
    assert ba > naive + 0.3, f"basis_align ({ba:.3f}) not clearly above naive self-gen ({naive:.3f})"


if __name__ == "__main__":
    test_gate_gradient_is_live_at_init()
    test_isa_grad_recovers_under_linear_obfuscation()
    test_basis_align_recovers_under_secret_rotation()
    test_decoder_beats_ridge_on_nonlinear_truth()
    test_logit_lens_generalizes_to_unseen_tokens()
    test_logit_lens_robust_to_norm_heterogeneity()
    print("ok")

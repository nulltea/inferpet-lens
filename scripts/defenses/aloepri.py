"""AloePri covariant obfuscation — "Towards Privacy-Preserving LLM Inference
via Covariant Obfuscation" (`2603.01499`; ref impl ``sheng1feng/Aloepri``).

A *static* obfuscation: the transform is fixed offline. The secret is a token
permutation Π plus a set of invertible "key matrices" that re-parameterise the
weights so the untrusted server cannot recover Π by comparing its obfuscated
weights to the public plaintext ones. Two pieces are implemented here:

1. :func:`keymat_gen` — **Algorithm 1 (Key Matrix Generation)**. Builds an
   invertible *change-of-basis pair* ``(P̂, Q̂)`` with ``P̂ Q̂ = I_d`` exactly,
   where ``P̂`` is ``d×(d+2h)`` (the obfuscated width is wider, ``d̃ = d+2h``) and
   ``Q̂`` is ``(d+2h)×d``. Distinctness comes from a null-space construction:
   with ``P̂ = [B  C  E]·Z`` and ``Q̂ = Zᵀ·[B⁻¹; F; D]`` (``Z`` orthogonal),

       P̂ Q̂ = B B⁻¹ + C F + E D = I_d + C F + E D ,

   and we pick ``C`` so ``C F = 0`` (rows of ``C`` in ``null(Fᵀ)``) and ``D`` so
   ``E D = 0`` (columns of ``D`` in ``null(E)``), giving ``P̂ Q̂ = I_d`` while
   ``P̂ ≠ Q̂⁻¹`` in any obvious way. ``λ`` regulates the norm of ``B = U + λV``
   (``U`` orthogonal) so half-precision weights don't overflow (AloePri: accuracy
   collapses at ``λ=3`` in bf16).

2. :func:`obfuscate_embedding_table` — the faithful **embedding obfuscation**
   ``W̃ = Π·(W_e + α_e·E)·P̂`` (``E`` Gaussian, scaled by ``α_e``), packaged as a
   :class:`~talens.weights.types.WeightPair` so it feeds the VMA / measure family
   directly. ``α_e`` is the calibration knob the τ-recovery sweep walks
   (AloePri Fig 3: VMA >30% at ``α_e=0.5`` → defended at ``α_e=1.0``).

Plus two activation-space covers (for the cross-scheme calibration on hidden
states): :class:`AloePriPermCover` (channel permutation — norm/φ-invariant, breaks
coordinate-aligned linear probes) and :class:`AloePriKeyMatCover` (the full
``X·P̂`` change of basis, widening to ``d+2h``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from talens.weights.types import WeightPair


def _orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """A uniform sample from O(n) via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    # fix the sign ambiguity so the draw is genuinely Haar-uniform
    return (q * np.sign(np.diag(r))).astype(np.float64)


def _null_space(M: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Orthonormal basis (columns) of the null space of ``M`` (``M x = 0``)."""
    _, s, vh = np.linalg.svd(M)
    rank = int((s > tol * max(M.shape)).sum())
    return vh[rank:].T.conj()                     # columns span null(M)


def keymat_gen(
    d: int, h: int, *, lam: float = 0.1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """AloePri Algorithm 1. Returns ``(P_hat (d, d+2h), Q_hat (d+2h, d))`` with
    ``P_hat @ Q_hat == I_d`` (to floating-point tolerance). ``h`` must be even."""
    if h % 2 != 0:
        raise ValueError("h must be even (rank-≤h/2 factor construction)")
    rng = np.random.default_rng(seed)
    half = h // 2

    # Build in float64 throughout; dense cancellation (I + CF + ED) loses
    # precision in float32 at large d. Use a solve, not an explicit inverse.
    U = _orthogonal(d, rng)
    V = rng.standard_normal((d, d)) / np.sqrt(d)
    B = U + lam * V
    Binv = np.linalg.solve(B, np.eye(d))

    E = (rng.standard_normal((d, half)) / np.sqrt(d)) @ (rng.standard_normal((half, h)) / np.sqrt(d))
    F = (rng.standard_normal((h, half)) / np.sqrt(d)) @ (rng.standard_normal((half, d)) / np.sqrt(d))
    Z = _orthogonal(d + 2 * h, rng)

    # C: d×h with rows in null(Fᵀ) so that C F = 0  (row c·F = 0 ⟺ c ∈ null(Fᵀ));
    # D: h×d with columns in null(E) so that E D = 0  (E·d_j = 0 ⟺ d_j ∈ null(E)).
    # Coeffs scaled by 1/√(null_dim) so C, D stay O(1) regardless of null rank.
    nullFt = _null_space(F.T)                      # (h, m), m ≥ h/2
    C = (rng.standard_normal((d, nullFt.shape[1])) / np.sqrt(nullFt.shape[1])) @ nullFt.T
    nullE = _null_space(E)                         # (h, m'), m' ≥ h/2
    D = nullE @ (rng.standard_normal((nullE.shape[1], d)) / np.sqrt(nullE.shape[1]))

    P_hat = np.concatenate([B, C, E], axis=1) @ Z                 # (d, d+2h)
    Q_hat = Z.T @ np.concatenate([Binv, F, D], axis=0)            # (d+2h, d)
    # Q̂ is a chosen right inverse (P̂Q̂ = I_d), NOT P̂⁻¹. Never expose P̂/Q̂ to a
    # probe — given P̂, any right inverse undoes W@P̂, so the keymat hides Π only
    # while P̂ stays secret.
    return P_hat.astype(np.float32), Q_hat.astype(np.float32)


def obfuscate_embedding_table(
    W: np.ndarray,
    token_ids: np.ndarray | None = None,
    *,
    alpha_e: float = 1.0,
    h: int | None = None,
    lam: float = 0.1,
    keymat: bool = True,
    seed: int = 1,
) -> WeightPair:
    """Faithful AloePri embedding obfuscation ``W̃ = Π·(W + α_e·E)·P̂`` as a
    :class:`WeightPair` (``plain=W``, ``obf=W̃``, ``perm=Π``).

    ``α_e`` scales the additive Gaussian (units of ``W.std()``) — the sweep knob.
    With ``keymat=False`` the ``P̂`` change-of-basis is dropped (the
    *permutation-core* obfuscation: row-permute + noise only); with ``keymat=True``
    the obfuscated table is ``d+2h`` wide (``h`` defaults to ``d//2``), exercising
    the width-agnostic VMA signature.
    """
    W = np.asarray(W, dtype=np.float32)
    n, d = W.shape
    rng = np.random.default_rng(seed)
    if token_ids is None:
        token_ids = np.arange(n, dtype=np.int64)

    noisy = W + (alpha_e * float(W.std()) * rng.standard_normal(W.shape)).astype(np.float32)
    if keymat:
        # Dense invertible change of basis (Algorithm 1). A dense P̂ is NOT a
        # column permutation, so it defeats the RowSort/sorted-quantile VMA —
        # this is the *strong* regime (the keymat is the defense; attacking it
        # needs the raw-row / trained EmbedRow inverter, not sorted matching).
        h_eff = h if h is not None else max(2, (d // 2) - (d // 2) % 2)
        P_hat, _ = keymat_gen(d, h_eff, lam=lam, seed=seed + 7)
        transformed = noisy @ P_hat                          # (n, d+2h)
    else:
        # Permutation-core regime (AloePri §7.3): rows differ "almost only by
        # row- and column-permutations" → the VMA-vulnerable regime the
        # calibration sweeps. The column permutation is sorted-quantile-invariant.
        col = rng.permutation(d)
        transformed = noisy[:, col]
    tau = rng.permutation(n)                                  # secret Π
    obf = np.empty_like(transformed)
    obf[tau] = transformed                                   # obf[τ[i]] is partner of plain[i]
    return WeightPair(
        plain=W, obf=obf.astype(np.float32), perm=tau.astype(np.int64),
        token_ids=np.asarray(token_ids, dtype=np.int64), kind="embed",
        model_id="aloepri-synthetic",
    )


# ───────────────────────── covariant model re-parameterization ─────────────────────────
# Faithful AloePri offline obfuscation of a whole transformer (paper §5.2). One residual
# key pair (P̂, Q̂), P̂ Q̂ = I_d (Algorithm 1), rewrites every residual-touching weight so the
# obfuscated forward is bit-equivalent to plaintext while the residual the server sees is the
# obfuscated x' = x P̂. The paper treats only RMSNorm (§5.2.5, a Gaussian κ approximation);
# the LayerNorm covariance below is derived EXACTLY from P̂ Q̂ = I — no approximation.


def obf_read_weight(W: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Rewrite a linear that READS the residual (q/k/v, mlp-in, head): W̃ = W Q̂ᵀ.
    On the obfuscated residual x'=xP̂, F.linear gives x'(WQ̂ᵀ)ᵀ = (x'Q̂)Wᵀ = xWᵀ. Bias is
    unchanged (the output is the within-block plaintext space)."""
    return W @ Q.to(W.dtype).t()


def obf_write_weight(W: torch.Tensor, b: torch.Tensor | None, P: torch.Tensor):
    """Rewrite a linear that WRITES the residual (attn-out, mlp-out): W̃ = P̂ᵀW, b̃ = P̂ᵀb.
    Output becomes (Wx+b)P̂ — the plaintext output mapped into the P̂-basis."""
    Pf = P.to(W.dtype)
    W2 = Pf.t() @ W
    b2 = None if b is None else Pf.t() @ b
    return W2, b2


class CovariantLayerNorm(nn.Module):
    """LayerNorm on the obfuscated residual that reproduces the plaintext LayerNorm output
    in the P̂-basis, exactly: LN(x' Q̂) P̂ = LN(x) P̂. Wraps the original norm module (keeps its
    weight/bias/eps) and applies the residual key pair around it."""

    def __init__(self, ln: nn.Module, P: torch.Tensor, Q: torch.Tensor):
        super().__init__()
        self.ln = ln
        self.register_buffer("P", P)
        self.register_buffer("Q", Q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x @ self.Q.to(x.dtype)) @ self.P.to(x.dtype)


def reparam_pythia(model, *, config: str = "keymat_only", h: int = 128, lam: float = 0.3,
                   seed: int = 0, alpha_e: float = 1.0, alpha_h: float = 0.2,
                   noise_seed: int = 0):
    """Covariantly re-parameterize a HuggingFace GPT-NeoX (pythia) model IN PLACE so its
    residual stream is obfuscated to x' = x P̂ (width d → d+2h) while the forward output is
    preserved up to the embedding/head noise. One residual key pair P̂/Q̂ (Alg 1, P̂Q̂=I_d) is
    applied to every residual-touching weight.

      * ``keymat_only`` — P̂/Q̂ alone, lossless: logits identical to plaintext (the gate config).
      * ``full_alg1``   — + embedding noise W̃ₑ = (Wₑ+αₑσₑE)P̂ and head noise (αₑ, α_h). The token
        permutation Π is OMITTED here: it is activation-inert (the residual for a true token is
        (Wₑ[t]+αₑσₑE)P̂ regardless of τ), so Π's protection is measured on the token-id surface
        (TFMA / ε1 sweep), not here.
      * ``alg2`` — + intra-head attention obfuscation (added separately).
    """
    if config == "keymat_only":
        alpha_e = alpha_h = 0.0
    elif config != "full_alg1":
        raise NotImplementedError(f"config={config!r} not yet implemented (have keymat_only, full_alg1)")
    net = model.gpt_neox
    d = net.embed_in.weight.shape[1]
    dev, dtype = net.embed_in.weight.device, net.embed_in.weight.dtype
    P_np, Q_np = keymat_gen(d, h, lam=lam, seed=seed)
    P = torch.from_numpy(P_np).to(device=dev, dtype=dtype)
    Q = torch.from_numpy(Q_np).to(device=dev, dtype=dtype)
    gen = torch.Generator().manual_seed(noise_seed)         # CPU generator (reproducible draw)

    def _noised(W, alpha):
        if alpha <= 0:
            return W
        noise = torch.randn(W.shape, generator=gen, dtype=W.dtype).to(W.device)
        return W + alpha * float(W.std()) * noise

    def _read(linear):
        linear.weight = nn.Parameter(obf_read_weight(linear.weight.data, Q))
        linear.in_features = linear.weight.shape[1]

    def _write(linear):
        W2, b2 = obf_write_weight(linear.weight.data, linear.bias.data if linear.bias is not None else None, P)
        linear.weight = nn.Parameter(W2)
        if b2 is not None:
            linear.bias = nn.Parameter(b2)
        linear.out_features = linear.weight.shape[0]

    with torch.no_grad():
        # embedding writes the residual: looked-up (noised) row → row P̂
        net.embed_in.weight = nn.Parameter(_noised(net.embed_in.weight.data, alpha_e) @ P)
        net.embed_in.embedding_dim = net.embed_in.weight.shape[1]
        for layer in net.layers:
            layer.input_layernorm = CovariantLayerNorm(layer.input_layernorm, P, Q)
            layer.post_attention_layernorm = CovariantLayerNorm(layer.post_attention_layernorm, P, Q)
            _read(layer.attention.query_key_value)
            _write(layer.attention.dense)
            _read(layer.mlp.dense_h_to_4h)
            _write(layer.mlp.dense_4h_to_h)
        net.final_layer_norm = CovariantLayerNorm(net.final_layer_norm, P, Q)
        # head reads the residual: (W_head + α_h noise) Q̂ᵀ (no Π — see docstring)
        model.embed_out.weight = nn.Parameter(obf_read_weight(_noised(model.embed_out.weight.data, alpha_h), Q))
        model.embed_out.in_features = model.embed_out.weight.shape[1]
    return {"P": P, "Q": Q, "d": d, "h": h, "config": config}


class AloePriPermCover:
    """Activation cover: a fixed random **channel (feature-dim) permutation**.
    Norm- and sorted-quantile-invariant (so the Π-channel signature survives) but
    scrambles coordinate-aligned linear readouts. The activation shadow of
    AloePri's hidden-dim permutations (Ẑ / Π)."""

    name = "aloepri_perm"

    def __init__(self, *, seed: int = 0):
        self._seed = seed
        self._perm: np.ndarray | None = None

    def __call__(self, operand: torch.Tensor, *, prompt_index: int) -> torch.Tensor:
        d = operand.shape[-1]
        if self._perm is None or self._perm.shape[0] != d:
            self._perm = np.random.default_rng(self._seed).permutation(d)
        idx = torch.as_tensor(self._perm, device=operand.device)
        return operand.index_select(-1, idx)


class AloePriKeyMatCover:
    """Activation cover: the full invertible key-matrix change of basis
    ``U = X · P̂`` (widening ``d → d+2h``). Information-preserving and exactly
    invertible (the covariant counterpart cancels in a real deployment); here it
    tests whether a linear inverter recovers tokens through an unknown basis."""

    name = "aloepri_keymat"

    def __init__(self, *, h: int | None = None, lam: float = 0.1, seed: int = 0):
        self._h, self._lam, self._seed = h, lam, seed
        self._P: torch.Tensor | None = None

    def __call__(self, operand: torch.Tensor, *, prompt_index: int) -> torch.Tensor:
        d = operand.shape[-1]
        if self._P is None or self._P.shape[0] != d:
            h = self._h if self._h is not None else max(2, (d // 2) - (d // 2) % 2)
            P_hat, _ = keymat_gen(d, h, lam=self._lam, seed=self._seed)
            self._P = torch.from_numpy(P_hat).to(operand.dtype)
        return operand @ self._P.to(operand.device)

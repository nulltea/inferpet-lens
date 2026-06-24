"""PriPert as a scheme-agnostic ``talens`` Transform (Task B-6, resid-split).

Faithful to PriPert (arXiv 2605.23158): a *split-inference* defense for the intermediate
residual exposed at a cut layer. The trusted edge applies two operations to the activation
``H`` before shipping it to the untrusted server:

  1. **Activation sparsification** — keep only the ``⌈ρ·d⌉`` largest-magnitude coordinates
     per row (token position), zero the rest. ``ρ`` (``rho``) is the sparsity knob; ``ρ=1``
     keeps everything. This is the lossy, information-discarding component.

  2. **Adversarial perturbation** — add a bounded perturbation ``δ`` with per-row energy
     budget ``β·RMS(row)`` (``β`` = ``beta``). The paper optimizes ``δ`` against a surrogate
     inverter; for an *attack-independent* measurement we expose two non-adversarial,
     energy-matched realizations that use **no inverter** (so the defense never couples to
     the attack the library is about to run):

       - ``mode="gauss"`` (default, the channel-matched realization): isotropic Gaussian
         ``δ ~ N(0, σ²I)`` with ``σ = β·RMS(Sparsify_ρ(H))``. This is exactly the Gaussian
         channel whose capacity the spectral channel-MI probe ``I_G = ½Σlog2(1+λ_i/σ²)``
         bounds — so the probe is the *converse* of this defense, by construction.
       - ``mode="pca"`` (stronger arm): ``δ`` placed in the top principal directions of the
         sparsified batch (energy-matched, deterministic per seed) — still inverter-free.

  The fully-adversarial optimized-against-an-inverter perturbation is the faithful
  upper-strength variant; it is intentionally NOT implemented here because it would tie the
  defense to a specific attack (the integrity line: a defense the library measures must not
  be defined by the attack). The energy-matched Gaussian / PCA perturbation is the proxy,
  documented as such.

Scheme-agnostic: this lives in ``scripts/defenses/`` (not the core). The Transform contract
(:class:`~talens.transforms.Transform`) takes one prompt's residual ``H`` of shape
``(seq, d)`` and returns ``U`` of the **same shape** — sparsification keeps width ``d`` (it
zeros, never drops, coordinates), and additive δ preserves shape. Determinism: δ is drawn
per ``(seed, prompt_index)`` so the Transform is pure.

The sweep driver (``scripts/spikes/pripert_sweep.py``) reuses the row-wise primitives
:func:`sparsify_rows` and :func:`pripert_apply` directly on the stacked operand ``X``
``(n_rows, d)`` — sparsification and additive noise are row-independent, so no per-prompt
grouping is needed; this keeps the matched probe (which consumes ``Sparsify_ρ(H)`` and ``σ``)
exactly aligned with what the attacks observe.
"""

from __future__ import annotations

import numpy as np
import torch


def sparsify_rows(H: np.ndarray, rho: float) -> np.ndarray:
    """Keep the ``⌈ρ·d⌉`` largest-magnitude coordinates per row, zero the rest.

    ``rho`` in ``(0, 1]``; ``rho >= 1`` is the identity (keep all ``d``)."""
    if rho >= 1.0:
        return np.asarray(H, dtype=np.float64)
    if rho <= 0.0:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    H = np.asarray(H, dtype=np.float64)
    d = H.shape[1]
    k = max(1, int(np.ceil(rho * d)))
    # indices of the (d-k) smallest-|·| coords per row -> zero them
    if k >= d:
        return H
    idx = np.argpartition(np.abs(H), d - k, axis=1)[:, : d - k]
    out = H.copy()
    np.put_along_axis(out, idx, 0.0, axis=1)
    return out


def row_rms(A: np.ndarray) -> np.ndarray:
    """Per-row RMS energy ``sqrt(mean_j A[i,j]²)`` (``(n,)``)."""
    return np.sqrt(np.mean(np.asarray(A, dtype=np.float64) ** 2, axis=1))


def perturbation_sigma(S: np.ndarray, beta: float) -> float:
    """Channel σ for the matched probe: ``β`` × mean per-row RMS of the
    sparsified operand ``S`` — the isotropic-Gaussian energy the probe is the
    converse of. Returns a single scalar σ (the probe's channel noise level)."""
    if beta <= 0.0:
        return 0.0
    return float(beta * float(np.mean(row_rms(S))))


def pripert_apply(
    H: np.ndarray, *, rho: float, beta: float, mode: str = "gauss", seed: int = 0,
    sigma: float | None = None,
) -> tuple[np.ndarray, float]:
    """Row-wise PriPert on a stacked operand ``H`` ``(n, d)``.

    Returns ``(U, sigma)`` where ``U = Sparsify_ρ(H) + δ`` and ``sigma`` is the
    scalar isotropic noise level the matched spectral probe must use. By default
    ``sigma = β·meanRMS(Sparsify_ρ(H))``; pass an explicit ``sigma`` to fix the
    noise floor to a reference (e.g. ``β·meanRMS(H_plaintext)``) so the converse
    is comparable across the ρ-axis (signal falls against a fixed floor).
    ``mode``: ``gauss`` (isotropic Gaussian δ) or ``pca`` (top-PCA-aligned δ,
    energy-matched, inverter-free)."""
    S = sparsify_rows(H, rho)
    if sigma is None:
        sigma = perturbation_sigma(S, beta)
    if sigma == 0.0:
        return S, 0.0
    rng = np.random.default_rng(seed)
    n, d = S.shape
    if mode == "gauss":
        delta = rng.standard_normal((n, d)) * sigma
    elif mode == "pca":
        # place δ energy in the top principal directions of the sparsified batch,
        # matched to total energy n·d·σ² (deterministic, uses no inverter).
        Sc = S - S.mean(axis=0, keepdims=True)
        # economy SVD of the centered batch -> right singular vectors = PCA dirs
        _, _, Vt = np.linalg.svd(Sc, full_matrices=False)
        k = min(d, max(1, int(np.ceil(0.1 * d))))  # top 10% directions
        coeff = rng.standard_normal((n, k))
        delta = coeff @ Vt[:k]
        # rescale to total energy n·d·σ²
        cur = np.sqrt(np.sum(delta ** 2))
        tgt = np.sqrt(n * d) * sigma
        delta = delta * (tgt / max(cur, 1e-12))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return S + delta, sigma


class PriPert:
    """PriPert split-inference Transform ``U = Sparsify_ρ(H) + δ`` over a
    ``(seq, d)`` residual operand. ``rho`` = sparsity ratio (1 ⇒ no sparsify),
    ``beta`` = perturbation energy budget (0 ⇒ no perturbation), ``mode`` in
    {``gauss``, ``pca``}. δ is fresh per ``(seed, prompt_index)`` so the
    Transform is pure. Width ``d`` is preserved (sparsify zeros, never drops)."""

    def __init__(self, *, rho: float = 1.0, beta: float = 0.0,
                 mode: str = "gauss", seed: int = 0) -> None:
        self.rho = float(rho)
        self.beta = float(beta)
        self.mode = str(mode)
        self.seed = int(seed)
        self.name = f"pripert[rho={rho:g},beta={beta:g},mode={mode}]"

    def __call__(self, operand: torch.Tensor, *, prompt_index: int) -> torch.Tensor:
        H = operand.detach().to(torch.float64).cpu().numpy()
        U, _ = pripert_apply(
            H, rho=self.rho, beta=self.beta, mode=self.mode,
            seed=self.seed + int(prompt_index),
        )
        return torch.from_numpy(U).to(operand.device, operand.dtype)

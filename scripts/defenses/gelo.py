"""GELO as a scheme-agnostic ``talens`` Transform (Task B-5).

Faithful to GELO (arXiv 2603.05035, github.com/noskill/gelo): a confidential-inference
defense that exposes only ``U = A H`` for remote GPU compute. ``A ∈ R^{n×n}`` is a *fresh
secret per-batch* row-mixing over the ``n`` token rows of the residual ``H ∈ R^{n×d}``;
the trusted side keeps ``A`` and un-mixes because left row-mixing commutes with the right
linear projection, ``U W = A H W`` and ``A^{-1}(A H W) = H W``. With orthogonal ``A`` the
inverse is just ``A^T``. Privacy is meant to come from (i) ``A`` being secret and fresh
per forward pass, and (ii) optional *shield rows* — decoy token rows appended to the batch
so the adversary cannot tell which rows are real or how many.

The library is scheme-agnostic, so this lives in ``scripts/defenses/`` (not the core). The
operand handed in is one prompt's residual ``H`` of shape ``(seq, d)`` (rows = token
positions). The Transform mixes the ``seq`` rows by ``A`` and returns ``U`` of the **same
row count** (the :class:`~talens.transforms.Transform` contract). *Shield rows change the
row count*, so they are NOT applied here — the GELO Transform stays contract-clean and
shield padding is handled by the sweep driver (``scripts/spikes/gelo_sweep.py``) which
appends decoy rows before mixing. This keeps the core attacks/probe able to consume the
shield-free Transform through the standard ``per_prompt_matrices`` seam.

Condition-number control (the privacy knob ``kappa``): ``A = Q1 · diag(s) · Q2`` with
``Q1, Q2`` Haar-orthogonal and singular values ``s`` log-spaced over ``[1, kappa]`` (so
``cond(A) = kappa``). ``kappa = 1`` ⇒ ``A`` orthogonal (the recommended GELO setting and
the locus of the feature-Gram-invariance leak, since then ``UᵀU = Hᵀ AᵀA H = HᵀH``).
Determinism: ``A`` is drawn per ``(seed, prompt_index, n)`` — *fresh per prompt*, modelling
the per-batch resampling (and defeating any amortized/global linear inverter).
"""

from __future__ import annotations

import numpy as np
import torch


def _orth(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return (q * np.sign(np.diag(r))).astype(np.float64)


def make_mixing(rng: np.random.Generator, n: int, kappa: float) -> np.ndarray:
    """Fresh ``n×n`` mixing matrix with condition number exactly ``kappa`` (>=1).

    ``kappa == 1`` returns a Haar-orthogonal matrix; otherwise ``A = Q1 diag(s) Q2`` with
    ``s`` log-spaced over ``[1, kappa]`` so ``cond(A) = kappa``.
    """
    if kappa < 1.0:
        raise ValueError(f"kappa must be >= 1, got {kappa}")
    if n == 1:
        return np.array([[1.0]], dtype=np.float64)
    q1 = _orth(rng, n)
    if kappa == 1.0:
        return q1
    q2 = _orth(rng, n)
    s = np.exp(np.linspace(np.log(1.0), np.log(kappa), n))  # singular values in [1, kappa]
    return (q1 * s[None, :]) @ q2


class GELO:
    """GELO row-mixing transform ``U = A H`` over a ``(seq, d)`` residual operand.

    A is fresh per ``(seed, prompt_index, seq)``. ``kappa`` controls ``cond(A)``
    (1 ⇒ orthogonal). Shields are NOT applied here (they change the row count and are a
    driver-level concern); see ``scripts/spikes/gelo_sweep.py``.
    """

    def __init__(self, *, kappa: float = 1.0, seed: int = 0) -> None:
        self.kappa = float(kappa)
        self.seed = seed
        self.name = f"gelo[kappa={kappa:g},seed={seed}]"
        self._A: dict[int, np.ndarray] = {}  # cache per (prompt_index) within a run

    def mixing_for(self, prompt_index: int, n: int) -> np.ndarray:
        key = prompt_index
        A = self._A.get(key)
        if A is None or A.shape[0] != n:
            rng = np.random.default_rng((self.seed, 55, prompt_index, n))
            A = make_mixing(rng, n, self.kappa)
            self._A[key] = A
        return A

    def __call__(self, operand: torch.Tensor, *, prompt_index: int) -> torch.Tensor:
        H = operand.detach().cpu().numpy().astype(np.float64)
        n = H.shape[0]
        A = self.mixing_for(prompt_index, n)
        U = A @ H
        return torch.from_numpy(U.astype(np.float32))

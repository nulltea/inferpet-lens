"""KV-CLOAK and its siblings as scheme-agnostic ``talens`` Transforms (Task B-2).

Faithful to arXiv 2508.09442 eq. (9): per PagedAttention block of ``b`` tokens, a head's
stored key matrix ``K ∈ R^{b×d}`` (d = head_dim) is exposed as

    K' = S · P̂ · (K + A) · M                                            (eq. 9)

with ``S ∈ R^{b×b}`` and per-head ``M ∈ R^{d×d}`` secret orthogonal matrices, ``P̂`` an
ephemeral one-time-pad row (token) permutation, and ``A`` a structured additive "beacon"
mask (large-magnitude coordinates: rank preservation + zero-storage de-obfuscation).

The library is scheme-agnostic, so this lives in ``scripts/defenses/`` (not the core).
The operand handed in is the flattened KV-cache ``(seq, n_heads·head_dim)`` (rows = tokens);
the transform reshapes per head, partitions the token axis into blocks of ``b``, applies the
chosen channel(s), and re-flattens. Token-axis operators (``S``, ``P̂``) act on whole token
rows identically across heads (they permute/mix k,v *pairs*); the feature operator ``M`` acts
per head. The structured beacon mask ``A`` is generated over the full flattened key row
``(b, n_heads*head_dim)`` (a row-spanning positional beacon), not strictly per head — a faithful
but stylized model of the paper's beacon; the M / S·P̂ conclusions do not depend on this choice.
Determinism: ``S, M, A`` are fixed per ``seed`` (per-deployment keys);
``P̂`` is drawn per ``(seed, prompt, block)`` (the ephemeral one-time pad).

Channels (``channel`` arg):
  full   K' = S P̂ (K+A) M     (eq. 9)
  naive  K' = S K M            (eq. 7, no perm/mask)
  scx    K' = P̂ K             (permutation-only sibling)
  m      K' = K M              (right-orthogonal feature mix only)
  sp     K' = S P̂ K           (left-orthogonal token mix + perm only)
  a      K' = K + A            (additive beacon mask only)
"""

from __future__ import annotations

import numpy as np
import torch

_CHANNELS = {"full", "naive", "scx", "m", "sp", "a"}


def _orth(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return (q * np.sign(np.diag(r))).astype(np.float64)  # fix sign so det>0-ish, still orthogonal


class KVCloak:
    """KV-CLOAK / sibling transform over a flattened ``(seq, n_heads*head_dim)`` operand."""

    def __init__(
        self,
        *,
        head_dim: int = 128,
        block_size: int = 32,
        channel: str = "full",
        mask_energy: float = 1.0,
        n_beacons: int = 4,
        seed: int = 0,
    ) -> None:
        if channel not in _CHANNELS:
            raise ValueError(f"channel {channel!r} not in {_CHANNELS}")
        self.head_dim = head_dim
        self.block_size = block_size
        self.channel = channel
        self.mask_energy = float(mask_energy)
        self.n_beacons = int(n_beacons)
        self.seed = seed
        self.name = f"kvcloak[{channel},b={block_size},a={mask_energy},seed={seed}]"
        self._S: dict[int, np.ndarray] = {}   # per block length
        self._M: dict[int, np.ndarray] = {}   # per head
        self._A: np.ndarray | None = None     # (block_size, n_heads*head_dim)

    # -- key material (deterministic in seed) --
    def _S_for(self, n: int) -> np.ndarray:
        if n not in self._S:
            rng = np.random.default_rng((self.seed, 11, n))
            self._S[n] = _orth(rng, n)
        return self._S[n]

    def _M_for(self, head: int) -> np.ndarray:
        if head not in self._M:
            rng = np.random.default_rng((self.seed, 22, head))
            self._M[head] = _orth(rng, self.head_dim)
        return self._M[head]

    def _Pi(self, prompt_index: int, block: int, n: int) -> np.ndarray:
        # ephemeral one-time-pad token permutation, per (seed, prompt, block)
        rng = np.random.default_rng((self.seed, 33, prompt_index, block, n))
        return rng.permutation(n)

    def _mask(self, n_feat: int) -> np.ndarray:
        # structured beacon mask A: large positive values at deterministic coords per row,
        # scaled to mask_energy * (median feature scale). Rank-preserving "positional beacon".
        if self._A is None:
            b = self.block_size
            rng = np.random.default_rng((self.seed, 44, n_feat))
            A = np.zeros((b, n_feat), dtype=np.float64)
            val = 1.0 / np.sqrt(self.n_beacons)   # unit beacon row has L2 norm 1
            for i in range(b):
                coords = rng.choice(n_feat, size=self.n_beacons, replace=False)
                A[i, coords] = val
            self._A = A
        return self._A

    def __call__(self, operand: torch.Tensor, *, prompt_index: int) -> torch.Tensor:
        H = operand.detach().cpu().numpy().astype(np.float64)
        seq, feat = H.shape
        h = self.head_dim
        n_heads = feat // h
        b = self.block_size
        ch = self.channel
        # scale for the beacon mask: median row L2 norm, so mask_energy=1 ==> a beacon
        # row whose energy is comparable to a real key row ("beacons dominate magnitude").
        scale = float(np.median(np.linalg.norm(H, axis=1))) + 1e-8
        A_unit = self._mask(feat) if (ch in ("full", "a") and self.mask_energy > 0) else None

        U = H.copy()
        for start in range(0, seq, b):
            stop = min(start + b, seq)
            n = stop - start
            blk = H[start:stop].copy()                      # (n, feat)
            block_idx = start // b
            # additive beacon mask (before perm) — per head, structured
            if A_unit is not None:
                blk = blk + self.mask_energy * scale * A_unit[:n]
            # left token-space ops act on whole rows (k,v pairs) identically across heads
            if ch in ("full", "naive", "sp"):
                S = self._S_for(n)
            if ch in ("full", "scx", "sp"):
                perm = self._Pi(prompt_index, block_idx, n)
                blk = blk[perm]
            if ch in ("full", "naive", "sp"):
                blk = S @ blk
            # right feature-space mix M acts per head
            if ch in ("full", "naive", "m"):
                for hd in range(n_heads):
                    sl = slice(hd * h, (hd + 1) * h)
                    blk[:, sl] = blk[:, sl] @ self._M_for(hd)
            U[start:stop] = blk
        return torch.from_numpy(U.astype(np.float32))

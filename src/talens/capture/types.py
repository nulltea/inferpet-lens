"""``CaptureSet`` — the pure-data interface the attacks and measures
consume.

Deliberately free of any model / nnsight import so the attacks, the IT
measures, and the tests can build a ``CaptureSet`` directly from arrays
(synthetic in tests, nnsight-captured in production). The nnsight
capture (:mod:`talens.capture.capture`) is the only thing that imports
the model stack, and it returns one of these.

A capture holds, per representation ``kind`` (``"resid_post"``,
``"attn_score"``, …) and per ``layer``, a list of per-prompt operands.
Two access paths share one flattening rule:

* :meth:`per_prompt_matrices` — per-prompt ``(H, U)`` pairs, where ``H``
  is the flattened plaintext operand and ``U = transform(H)`` is the
  exposed operand. The cover-break attack needs both.
* :meth:`stack` — the ``(rows, features)`` design matrix of exposed
  operands stacked across prompts, with aligned token-id targets. The
  inversion attacks and the probes train on this.

Both apply the agnostic :class:`~talens.transforms.Transform` seam; under
the default :class:`~talens.transforms.Identity` (plaintext, pass 1) the
transform is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..transforms import Identity, Transform


def _flatten_operand(op: torch.Tensor, max_kv: int) -> torch.Tensor:
    """Flatten one operand to a 2-D ``(n_rows, features)`` matrix.

    * 2-D ``(n_rows, d)`` (residual stream) — returned as-is.
    * 3-D ``(n_heads, n_q, n_kv)`` (attention scores) — permuted to
      ``(n_q, n_heads, n_kv)``, zero-padded along the ragged ``n_kv`` to
      ``max_kv``, then reshaped to ``(n_q, n_heads · max_kv)``.
    """
    if op.ndim == 2:
        return op.to(torch.float32)
    if op.ndim == 3:
        n_heads, n_q, n_kv = op.shape
        t = op.permute(1, 0, 2).to(torch.float32)  # (n_q, n_heads, n_kv)
        if n_kv < max_kv:
            pad = torch.zeros((n_q, n_heads, max_kv - n_kv), dtype=t.dtype)
            t = torch.cat([t, pad], dim=2)
        return t.reshape(n_q, n_heads * max_kv)
    raise ValueError(f"unsupported operand ndim={op.ndim}")


@dataclass
class CaptureSet:
    model_id: str
    prompt_token_ids: list[list[int]]
    # operands[(kind, layer)][prompt_idx] -> tensor.
    # resid_post: (n_rows, d); attn_score: (n_heads, n_q, n_kv).
    operands: dict[tuple[str, int], list[torch.Tensor]]

    def n_prompts(self) -> int:
        return len(self.prompt_token_ids)

    def kinds(self) -> list[str]:
        return sorted({k for (k, _) in self.operands})

    def layers(self, kind: str) -> list[int]:
        return sorted(layer for (k, layer) in self.operands if k == kind)

    def _global_max_kv(self, kind: str, layer: int) -> int:
        ops = self.operands.get((kind, layer), [])
        return max((int(op.shape[2]) for op in ops if op.ndim == 3), default=0)

    def per_prompt_matrices(
        self, kind: str, layer: int, *, transform: Transform | None = None
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Per-prompt ``(prompt_idx, H, U)`` with ``H`` the flattened
        plaintext operand and ``U = transform(H)`` the exposed one, both
        float32 numpy arrays of shape ``(n_rows, features)``.
        """
        transform = transform or Identity()
        ops = self.operands.get((kind, layer))
        if not ops:
            return []
        max_kv = self._global_max_kv(kind, layer)
        out: list[tuple[int, np.ndarray, np.ndarray]] = []
        for pi, op in enumerate(ops):
            h = _flatten_operand(op, max_kv)
            u = transform(h, prompt_index=pi)
            out.append(
                (
                    pi,
                    h.detach().cpu().numpy().astype(np.float32, copy=False),
                    u.detach().cpu().numpy().astype(np.float32, copy=False),
                )
            )
        return out

    def stack(
        self,
        kind: str,
        layer: int,
        *,
        transform: Transform | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Return ``(X, y, prompt_lengths)`` for one ``(kind, layer)``:
        the exposed operands stacked into a ``(total_rows, features)``
        float32 matrix ``X``, the ``(total_rows,)`` int64 token ids ``y``,
        and the per-prompt row counts. Rows are truncated to the shorter
        of operand rows / token ids per prompt.
        """
        mats = self.per_prompt_matrices(kind, layer, transform=transform)
        if not mats:
            return np.zeros((0, 0), np.float32), np.zeros((0,), np.int64), []
        Xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        lengths: list[int] = []
        for pi, _h, u in mats:
            ids = np.asarray(self.prompt_token_ids[pi], dtype=np.int64)
            n = min(u.shape[0], ids.shape[0])
            if n == 0:
                continue
            Xs.append(u[:n])
            ys.append(ids[:n])
            lengths.append(n)
        if not Xs:
            return np.zeros((0, 0), np.float32), np.zeros((0,), np.int64), []
        widths = {x.shape[1] for x in Xs}
        if len(widths) != 1:
            raise ValueError(
                f"stack({kind!r}, {layer}): inconsistent feature widths "
                f"{sorted(widths)} across prompts"
            )
        return np.concatenate(Xs, 0), np.concatenate(ys, 0), lengths

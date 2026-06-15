"""Attack drivers (faithful ports of the aloepri ridge attacks).

Three pass-1 attacks, all operating on a :class:`~talens.capture.types.CaptureSet`
through the agnostic :class:`~talens.transforms.Transform` seam:

* :mod:`hidden_state` — IMA/ISA ridge inversion of the residual stream.
* :mod:`attn_score`   — ISA inversion of per-head attention scores.
* :mod:`cover_break`  — anchor-based linear recovery (p95 cosine).
"""

from __future__ import annotations

from . import attn_score, cover_break, hidden_state

__all__ = ["hidden_state", "attn_score", "cover_break"]

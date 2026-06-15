"""Capture: the pure-data :class:`CaptureSet` (always importable) and
the nnsight-backed capture functions (import the model stack lazily).

``from talens.capture import CaptureSet`` works with no model deps.
``from talens.capture.capture import load_model, capture_representations``
pulls in nnsight/transformers only when you actually capture.
"""

from __future__ import annotations

from .types import CaptureSet

__all__ = ["CaptureSet"]

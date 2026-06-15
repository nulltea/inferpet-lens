"""The scheme-agnostic defense seam.

``talens`` knows nothing about any particular confidential-inference
defense. A defense is modelled as a :class:`Transform` — an arbitrary
map applied to a representation tensor *before* the attacks and IT
measures observe it. The library ships exactly one built-in transform,
:class:`Identity` (the plaintext model). Anyone studying a specific
scheme (an orthogonal cover, additive noise, a learned bottleneck, …)
implements their own :class:`Transform` and injects it — the library
treats it as an opaque ``Tensor -> Tensor`` and asserts nothing about
its security properties.

A transform receives the per-prompt operand ``H`` of shape
``(n_rows, d)`` (e.g. one prompt's ``resid_post`` at a layer) and
returns the exposed operand ``U`` of the same row count. ``U`` need not
share ``H``'s feature dimension — a transform may project, pad, or
expand — so attacks must read ``d`` from the operand they are handed,
never assume it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Transform(Protocol):
    """An opaque representation transform (the "defense" under test).

    Implementations must be pure with respect to a given seed: the same
    ``(operand, prompt_index)`` must map to the same output within a run
    so attacks and measures observe a consistent ``U``.
    """

    name: str

    def __call__(self, operand: torch.Tensor, *, prompt_index: int) -> torch.Tensor:
        """Map plaintext operand ``H`` ``(n_rows, d)`` to exposed ``U``
        ``(n_rows, d')``. ``prompt_index`` lets stateful covers vary per
        prompt (e.g. a per-forward mask); pure transforms may ignore it.
        """
        ...


class Identity:
    """The plaintext model — ``U = H``. The only built-in transform.

    Pass 1 runs entirely under Identity: we measure leakage of the
    clean representations and calibrate IT measures against the attack
    recovery on those same tensors. Defenses are studied later by
    swapping in an external :class:`Transform`.
    """

    name = "identity"

    def __call__(self, operand: torch.Tensor, *, prompt_index: int) -> torch.Tensor:  # noqa: ARG002
        return operand

"""Unified leakage-reporting layer — bits canonical + per-secret readout.

Every probe in :mod:`talens.measures` emits a different result dict (``v_information``
returns ``v_information_bits``; ``club`` returns ``club_mi_bits``; ``mdl`` returns
``surplus_description_length_bits``; ``spectral_channel_mi`` returns ``i_g_bits``; …).
This module is the single place that

1. **extracts the canonical bits** scalar from any measure dict — *one comparable
   scale* (MI / V-info / capacity / SDL / channel-MI all in bits), via
   :func:`canonical_bits`; and
2. **attaches a per-secret human readout** — the recovery axis rendered in the units
   a reader can interpret (token-id → perplexity + top-k; text → token-F1 / ROUGE;
   permutation → recovery-rate / Kendall-τ; embedding → token-F1 / cosine; membership
   → AUC), via the :class:`Readout` builders.

The bits value is stored **verbatim** (never rescaled). The "1/100 of a bit"
illegibility CLAUDE.md warns about is fixed in *rendering* only: :func:`format_bits`
uses adaptive precision so a nonzero small magnitude never collapses to ``0.00 bits``.

Dependency-light (numpy only) so it runs in the host CPU venv alongside the model-free
measures and tests.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Bits legibility — fix the "1/100 of a bit" readout, never the stored value.
# ---------------------------------------------------------------------------


def format_bits(x: float | None, *, unit: bool = True) -> str:
    """Render a bits value with adaptive precision.

    A bare ``round(x, 2)`` collapses a genuine ``0.01``-bit leak to ``"0.00"`` —
    illegible. This keeps enough significant figures that any nonzero value renders
    nonzero, and switches to millibits below ``0.1`` bit so the magnitude is readable.
    The *stored* float is untouched; this only affects display.

    >>> format_bits(0.0123)
    '12.3 mbit'
    >>> format_bits(3.14159)
    '3.14 bits'
    >>> format_bits(0.0)
    '0 bits'
    """
    suffix = lambda s: s if unit else ""
    if x is None:
        return "n/a"
    if not math.isfinite(x):
        return ("∞" if x > 0 else "−∞") + (suffix(" bits"))
    ax = abs(x)
    if ax == 0.0:
        return "0" + suffix(" bits")
    if ax >= 0.1:
        # 3 significant figures, trimmed.
        s = f"{x:.3g}"
        return s + suffix(" bits")
    # Sub-0.1-bit: millibits with adaptive decimals so it never reads as 0.
    mbit = x * 1000.0
    amb = abs(mbit)
    decimals = 0 if amb >= 10 else (1 if amb >= 1 else 2)
    return f"{mbit:.{decimals}f}" + suffix(" mbit")


def perplexity_from_bits(h_bits: float | None) -> float | None:
    """Per-token perplexity ``2^H`` from a per-token conditional entropy in bits."""
    if h_bits is None or not math.isfinite(h_bits):
        return None
    return float(2.0 ** h_bits)


def token_f1(predicted_ids: np.ndarray, ground_truth_ids: np.ndarray) -> float:
    """Multiset token-F1 between two id sequences (order-free overlap).

    Precision = |pred ∩ gt| / |pred|, recall = |pred ∩ gt| / |gt|, counted with
    multiplicity; F1 is their harmonic mean. Matches the text-recovery readout used by
    the inversion attacks (vec2text-style).
    """
    pred = np.asarray(predicted_ids).ravel()
    gt = np.asarray(ground_truth_ids).ravel()
    if pred.size == 0 and gt.size == 0:
        return 1.0
    if pred.size == 0 or gt.size == 0:
        return 0.0
    pv, pc = np.unique(pred, return_counts=True)
    gv, gc = np.unique(gt, return_counts=True)
    gmap = dict(zip(gv.tolist(), gc.tolist()))
    overlap = sum(min(int(c), gmap.get(int(v), 0)) for v, c in zip(pv.tolist(), pc.tolist()))
    if overlap == 0:
        return 0.0
    precision = overlap / pred.size
    recall = overlap / gt.size
    return float(2 * precision * recall / (precision + recall))


# ---------------------------------------------------------------------------
# Per-secret readout — the recovery axis in human-legible units.
# ---------------------------------------------------------------------------

# secret_kind -> the readout fields that are meaningful for it.
SECRET_KINDS = ("token_id", "text", "permutation", "embedding", "membership")


@dataclass
class Readout:
    """One per-secret recovery readout, paired with a bits value in :class:`LeakageReport`.

    ``primary_name``/``primary_value`` is the single graded recovery scalar (the one the
    leakage sweep correlates bits against); ``fields`` carries the rest of the human
    readout (top-k, perplexity, τ, cosine, …).
    """

    secret_kind: str
    primary_name: str
    primary_value: float | None
    fields: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.secret_kind not in SECRET_KINDS:
            raise ValueError(f"unknown secret_kind {self.secret_kind!r}; expected {SECRET_KINDS}")

    def render(self) -> str:
        head = f"{self.primary_name}={_fmt(self.primary_value)}"
        if not self.fields:
            return head
        rest = ", ".join(f"{k}={_fmt(v)}" for k, v in self.fields.items())
        return f"{head} ({rest})"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    if not math.isfinite(v):
        return "∞" if v > 0 else "−∞"
    return f"{v:.3g}"


def token_id_readout(
    *, top1: float | None, topk: float | None = None, k: int = 10,
    h_bits: float | None = None,
) -> Readout:
    """Token-id secret: top-1 recovery rate (primary) + top-k + perplexity."""
    fields: dict[str, float] = {}
    if topk is not None:
        fields[f"top{k}"] = float(topk)
    ppl = perplexity_from_bits(h_bits)
    if ppl is not None:
        fields["perplexity"] = ppl
    return Readout("token_id", "token_top1_recovery_rate", top1, fields)


def text_readout(*, token_f1: float | None, rouge_l: float | None = None) -> Readout:
    """Text secret: token-F1 (primary) + optional ROUGE-L."""
    fields = {} if rouge_l is None else {"rouge_l": float(rouge_l)}
    return Readout("text", "token_f1", token_f1, fields)


def permutation_readout(*, recovery_rate: float | None, kendall_tau: float | None = None) -> Readout:
    """Permutation secret: fraction-correctly-placed (primary) + Kendall-τ."""
    fields = {} if kendall_tau is None else {"kendall_tau": float(kendall_tau)}
    return Readout("permutation", "recovery_rate", recovery_rate, fields)


def embedding_readout(*, cosine: float | None, token_f1: float | None = None) -> Readout:
    """Embedding secret: cosine to the true embedding (primary) + decoded token-F1."""
    fields = {} if token_f1 is None else {"token_f1": float(token_f1)}
    return Readout("embedding", "cosine", cosine, fields)


def membership_readout(*, auc: float | None) -> Readout:
    """Membership secret: attack AUC (primary)."""
    return Readout("membership", "auc", auc)


# ---------------------------------------------------------------------------
# Canonical bits — extract the one comparable scalar from any measure dict.
# ---------------------------------------------------------------------------

# measure name -> (result-dict key holding the canonical bits, what those bits mean).
# Keeps the bits-extraction in ONE place instead of scattered across the CLI/spikes.
_BITS_REGISTRY: dict[str, tuple[str, str]] = {
    "v_information": ("v_information_bits", "v_info"),
    "v_information_retrieval": ("v_information_bits", "v_info"),
    "v_information_capacity": ("v_information_bits", "capacity_v_info"),
    "club": ("club_mi_bits", "mi_upper_bound"),
    "mdl": ("surplus_description_length_bits", "sdl"),
    "pid": ("i_joint_bits", "pid_total_mi"),
    "spectral_channel_mi": ("i_g_bits", "channel_mi_upper_bound"),
}


def canonical_bits(measure: str, result: dict[str, Any]) -> tuple[float | None, str]:
    """Pull the canonical bits scalar + its semantic kind from a measure's output dict.

    ``measure`` is the registry name (``"v_information"``, ``"club"``, ``"mdl"``,
    ``"pid"``, ``"spectral_channel_mi"``, …). Returns ``(bits, bits_kind)``; ``bits`` is
    ``None`` when the measure declined (e.g. ``{"v_information_bits": None}``).
    """
    if measure not in _BITS_REGISTRY:
        raise KeyError(
            f"no canonical-bits mapping for measure {measure!r}; "
            f"known: {sorted(_BITS_REGISTRY)}"
        )
    key, kind = _BITS_REGISTRY[measure]
    val = result.get(key)
    return (None if val is None else float(val)), kind


@dataclass
class LeakageReport:
    """One standardized leakage row: canonical **bits** + per-secret **readout**.

    This is the metric convention CLAUDE.md mandates ("bits canonical + per-secret
    readout"). ``bits`` is stored verbatim on the comparable scale; :meth:`bits_legible`
    renders it. ``readout`` is the paired recovery axis (``None`` for an attack-free
    probe-only row). ``sigma`` carries the defense's privacy parameter so a sweep is a
    list of these.
    """

    measure: str
    bits: float | None
    bits_kind: str
    readout: Readout | None = None
    surface: str | None = None
    layer: int | None = None
    transform: str = "Identity"
    sigma: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_measure(
        cls,
        measure: str,
        result: dict[str, Any],
        *,
        readout: Readout | None = None,
        surface: str | None = None,
        layer: int | None = None,
        transform: str = "Identity",
        sigma: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "LeakageReport":
        """Build a report by extracting canonical bits from ``result`` and attaching
        the (already-computed) recovery ``readout``."""
        bits, kind = canonical_bits(measure, result)
        return cls(
            measure=measure, bits=bits, bits_kind=kind, readout=readout,
            surface=surface, layer=layer, transform=transform, sigma=sigma,
            extra=extra or {},
        )

    def bits_legible(self) -> str:
        return format_bits(self.bits)

    def render(self) -> str:
        """One-line ``bits | readout`` rendering — both axes, always."""
        left = f"{self.measure}[{self.bits_kind}]={self.bits_legible()}"
        right = self.readout.render() if self.readout is not None else "no-attack"
        return f"{left} | {right}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bits_legible"] = self.bits_legible()
        return d

"""Disk cache for nnsight captures + embedding tables.

Capture (running the corpus through Qwen3 under nnsight) is the slowest
phase of a pass. Caching the :class:`~talens.capture.types.CaptureSet`
keyed by ``(model, corpus, kinds)`` lets reruns that only change the
*measured* layer subset (``--layers``) or the probe/measure config skip
the forward pass entirely. The embedding table is cached separately,
keyed by the model alone, so it is shared across corpora — together they
let a cache hit avoid loading the model at all.

The capture records which layers it holds; a hit is reusable iff the
requested ``capture_layers`` are a subset of what was cached (a capture
made for *all* layers satisfies any later subset). Model-free by design
(only torch + ``types``) so importing it never pulls in the model stack.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .types import CaptureSet

DEFAULT_CACHE_DIR = Path("results/capture_cache")


def _corpus_digest(model_id: str, prompts: list[str], kinds: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    h.update(model_id.encode()); h.update(b"\x00")
    h.update(",".join(sorted(kinds)).encode()); h.update(b"\x00")
    for p in prompts:
        h.update(p.encode()); h.update(b"\x00")
    return h.hexdigest()[:16]


def capture_cache_path(
    cache_dir: Path | str, model_id: str, prompts: list[str], kinds: tuple[str, ...]
) -> Path:
    digest = _corpus_digest(model_id, prompts, kinds)
    return Path(cache_dir) / f"capture-{digest}.pt"


def embed_cache_path(cache_dir: Path | str, model_id: str) -> Path:
    digest = hashlib.sha256(model_id.encode()).hexdigest()[:16]
    return Path(cache_dir) / f"embed-{digest}.pt"


def save_capture(cap: CaptureSet, path: Path, *, capture_layers: list[int] | None) -> None:
    """Persist a CaptureSet. ``capture_layers`` is the requested spec
    (``None`` = all layers) recorded so a later all-layers request knows
    whether the cache is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": cap.model_id,
            "prompt_token_ids": cap.prompt_token_ids,
            "operands": cap.operands,
            "capture_layers_spec": None if capture_layers is None else sorted(capture_layers),
        },
        path,
    )


def load_capture(path: Path) -> tuple[CaptureSet, list[int] | None]:
    """Load a CaptureSet and its recorded ``capture_layers`` spec."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    cap = CaptureSet(
        model_id=d["model_id"],
        prompt_token_ids=d["prompt_token_ids"],
        operands=d["operands"],
    )
    return cap, d.get("capture_layers_spec")


def save_embed(emb: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, path)


def load_embed(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def present_layers(cap: CaptureSet) -> set[int]:
    """Union of layers present across all kinds in the capture."""
    return {layer for kind in cap.kinds() for layer in cap.layers(kind)}


def can_reuse(
    present: set[int], cached_spec: list[int] | None, want: list[int] | None
) -> bool:
    """Decide whether a cached capture covers a requested ``capture_layers``.

    * ``want is None`` (caller wants *all* layers): reusable only if the
      cache was itself made for all layers (``cached_spec is None``).
    * ``want`` a list: reusable iff every requested layer is present.
    """
    if want is None:
        return cached_spec is None
    return set(want) <= present


def subset_capture(cap: CaptureSet, layers: list[int] | None) -> CaptureSet:
    """A view of ``cap`` restricted to ``layers`` (all kinds). ``None``
    returns the capture unchanged."""
    if layers is None:
        return cap
    keep = set(layers)
    operands = {(k, layer): v for (k, layer), v in cap.operands.items() if layer in keep}
    return CaptureSet(
        model_id=cap.model_id,
        prompt_token_ids=cap.prompt_token_ids,
        operands=operands,
    )

"""GPU capture of the raw KV-cache (per-head K and V) for the KV-CLOAK study (Task B-2).

KV-CLOAK obfuscates the *stored* KV-cache and leaves the attention output invariant,
so the faithful leakage surface is the raw per-head key/value matrices (secret = K, V),
NOT kq/kqv_out. We capture kinds 'k' and 'v' (stored 2-D as (seq, n_kv_heads*head_dim))
on a long-prompt corpus so the token-axis block size b in {16,32,64} is non-degenerate.

MUST run in the ROCm container (real Qwen3 forward). Small + fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from talens.capture.capture import load_or_capture

CORPUS = Path("corpora/kv-cloak-long-48.txt")
LAYERS = [0, 12, 20]
KINDS = ("k", "v")
OUT = Path("refine-logs/kv-cloak")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = [l.strip() for l in CORPUS.read_text().splitlines() if l.strip()]
    print(f"cuda={torch.cuda.is_available()} prompts={len(prompts)}", flush=True)
    cap, _embed, source = load_or_capture(
        "Qwen/Qwen3-4B", prompts, capture_layers=LAYERS, kinds=KINDS
    )
    print(f"source={source} n_prompts={cap.n_prompts()} kinds={cap.kinds()}", flush=True)
    lens = [len(x) for x in cap.prompt_token_ids]
    k0 = cap.operands[("k", LAYERS[0])][0]
    v0 = cap.operands[("v", LAYERS[0])][0]
    meta = {
        "source": source,
        "n_prompts": cap.n_prompts(),
        "token_len_min": min(lens),
        "token_len_max": max(lens),
        "token_len_mean": sum(lens) / len(lens),
        "layers": LAYERS,
        "kinds": list(KINDS),
        "k_shape_p0": list(k0.shape),
        "v_shape_p0": list(v0.shape),
    }
    (OUT / "capture_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

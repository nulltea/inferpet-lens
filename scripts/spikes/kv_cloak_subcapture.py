"""Build a compact long-prompt sub-capture for the KV-CLOAK block-size sweep (Task B-2).

KV-CLOAK partitions the *token* axis into PagedAttention blocks of size b
(arXiv 2508.09442, eq. 9). The dev-24 prompts are only ~10 tokens, so a
b in {16,32} sweep is degenerate there. The release-gate-512 capture has
longer prompts; we select the longest, truncate each to a fixed token
length L, and keep only the KV-relevant kinds (kq, kqv_out) at a small
layer profile. One CPU load of the 9.9 GB capture; writes a ~tens-of-MB
sub-capture under refine-logs/kv-cloak/.

CPU-only (cached tensors). No GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

SRC = Path("results/capture_cache/capture-0e716f8d4bf330c7.pt")
OUT = Path("refine-logs/kv-cloak")
OUT_CAP = OUT / "subcapture_L32.pt"

LAYERS = [0, 12, 20]
KINDS = ["kq", "kqv_out"]
L = 32           # fixed truncated token length
N_PROMPTS = 64   # number of longest prompts to keep


def _ntok(op_kqv: torch.Tensor) -> int:
    # kqv_out operand is (n_q, heads*head_dim); n_q == n_tokens
    return int(op_kqv.shape[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"loading {SRC} ...", flush=True)
    d = torch.load(SRC, map_location="cpu", weights_only=False)
    ops = d["operands"]
    ids = d["prompt_token_ids"]
    n_all = len(ids)
    print(f"loaded: model={d['model_id']} n_prompts={n_all}", flush=True)

    # token length per prompt from a reference kqv_out layer
    ref_key = ("kqv_out", LAYERS[0])
    ref_ops = ops[ref_key]
    lengths = [_ntok(ref_ops[p]) for p in range(n_all)]
    # select prompts with >= L tokens, take the N_PROMPTS longest
    eligible = [p for p in range(n_all) if lengths[p] >= L]
    eligible.sort(key=lambda p: -lengths[p])
    sel = sorted(eligible[:N_PROMPTS])
    print(f"token-length stats: min={min(lengths)} max={max(lengths)} "
          f">= {L}: {len(eligible)} prompts; selecting {len(sel)}", flush=True)

    new_ops: dict[tuple[str, int], list[torch.Tensor]] = {}
    for kind in KINDS:
        for layer in LAYERS:
            key = (kind, layer)
            if key not in ops:
                continue
            lst = []
            for p in sel:
                op = ops[key][p].to(torch.float32)
                if kind == "kqv_out":          # (n_q, F) -> (L, F)
                    op = op[:L]
                elif kind == "kq":             # (heads, n_q, n_kv) -> (heads, L, L)
                    op = op[:, :L, :L]
                lst.append(op.contiguous())
            new_ops[key] = lst

    out = {
        "model_id": d["model_id"],
        "prompt_token_ids": [ids[p][:L] for p in sel],
        "operands": new_ops,
        "capture_layers_spec": LAYERS,
    }
    torch.save(out, OUT_CAP)
    meta = {
        "source": SRC.name,
        "n_prompts": len(sel),
        "trunc_len_L": L,
        "layers": LAYERS,
        "kinds": KINDS,
        "selected_orig_indices": sel,
        "kqv_out_shape": list(new_ops[("kqv_out", LAYERS[0])][0].shape),
        "kq_shape": list(new_ops[("kq", LAYERS[0])][0].shape),
    }
    (OUT / "subcapture_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"wrote {OUT_CAP} ({OUT_CAP.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()

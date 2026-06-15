"""Pass-1 orchestrator: capture → attacks → measures → calibration.

Runs the plaintext (Identity) pipeline end-to-end on Qwen3 and writes a
JSON report. Capture pulls in nnsight/transformers lazily, so the rest
of the package (and the tests) stay model-free.

Example:
    python -m talens.cli --model Qwen/Qwen3-4B \\
        --corpus corpora/dev-64.txt --out results/pass1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .attacks import attn_score, cover_break, hidden_state
from .calibration import calibrate_records
from .measures import club_mi_upper_bound, online_code_length, v_information
from .transforms import Identity


def _read_corpus(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def run_pass1(
    model_id: str,
    prompts: list[str],
    *,
    layers: list[int] | None = None,
) -> dict:
    """Capture, attack, measure, and calibrate under the Identity
    transform. Returns the full report dict.
    """
    from .capture.capture import (  # lazy import of the model stack
        capture_representations,
        embed_table,
        load_model,
    )

    model = load_model(model_id)
    emb = embed_table(model)
    cap = capture_representations(model, prompts, layers=layers)
    transform = Identity()

    records: list[dict] = []
    for kind in cap.kinds():
        for layer in cap.layers(kind):
            # --- attacks (recovery ground-truth) ---
            if kind == "resid_post":
                atk = hidden_state.run(cap, emb, layer=layer, kind=kind, transform=transform)
                cover = cover_break.run(cap, layer=layer, kind=kind, transform=transform)
            elif kind == "attn_score":
                atk = attn_score.run(cap, emb, layer=layer, transform=transform)
                cover = None
            else:
                continue

            # --- measures (predictors) over the same exposed operand ---
            X, y, _ = cap.stack(kind, layer, transform=transform)
            vinfo = v_information(X, y)
            mdl = online_code_length(X, y)
            # CLUB on (representation, token-embedding).
            if X.shape[0] >= 8:
                Y = emb[torch.from_numpy(y)].numpy()
                club = club_mi_upper_bound(X, Y)
            else:
                club = {"club_mi_bits": None}

            records.append(
                {
                    "kind": kind,
                    "layer": layer,
                    "attack": atk.attack,
                    "primary_metric_value": atk.primary_metric_value,
                    "ttrsr_top1": atk.ttrsr_top1,
                    "cover_break_p95": cover.primary_metric_value if cover else None,
                    "v_information_bits": vinfo.get("v_information_bits"),
                    "mdl_surplus_bits": mdl.get("surplus_description_length_bits"),
                    "mdl_compression": mdl.get("compression"),
                    "club_mi_bits": club.get("club_mi_bits"),
                }
            )

    calibration = {
        key: calibrate_records(records, measure_key=key)
        for key in ("v_information_bits", "mdl_surplus_bits", "club_mi_bits")
    }
    return {
        "model_id": model_id,
        "transform": transform.name,
        "n_prompts": len(prompts),
        "records": records,
        "calibration": calibration,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--layers", default=None, help="comma-separated layer indices; default all")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    prompts = _read_corpus(args.corpus)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    report = run_pass1(args.model, prompts, layers=layers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out}")
    for key, cal in report["calibration"].items():
        print(f"  {key:24s} spearman={cal.get('spearman')!s:>8}  r2={cal.get('r_squared')!s:>8}")


if __name__ == "__main__":
    main()

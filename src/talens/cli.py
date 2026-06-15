"""Pass-1 orchestrator: capture → attacks → measures → calibration.

Runs the plaintext (Identity) pipeline end-to-end on Qwen3 and writes a
JSON report. Capture pulls in nnsight/transformers lazily, so the rest
of the package (and the tests) stay model-free.

The heavy compute — capture, the PVI/MDL softmax probe, and CLUB — all
run on the GPU, so the per-(kind, layer) blocks are processed
sequentially: the GPU is the shared resource and each block uses it
fully. (An earlier thread-pool over blocks didn't help: sklearn fits are
GIL-bound under threading; moving the probe to a torch-GPU fit is what
actually uses the hardware — see docs/research/attacks_setting.md and the
probe oracle test.)

Example:
    python -m talens.cli --model Qwen/Qwen3-4B \\
        --corpus corpora/release-gate-512.txt --out results/pass1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .attacks import attn_score, cover_break, hidden_state
from .calibration import calibrate_records
from .capture.types import CaptureSet
from .measures import club_mi_upper_bound, online_code_length, v_information
from .transforms import Identity, Transform


def _read_corpus(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _process_block(
    cap: CaptureSet,
    embed_table: torch.Tensor,
    kind: str,
    layer: int,
    *,
    transform: Transform,
    attack_split_mode: str,
) -> dict:
    """All attack + measure work for one (kind, layer) block."""
    if kind == "resid_post":
        atk = hidden_state.run(
            cap, embed_table, layer=layer, kind=kind, transform=transform,
            split_mode=attack_split_mode,
        )
        cover = cover_break.run(cap, layer=layer, kind=kind, transform=transform)
    elif kind == "attn_score":
        atk = attn_score.run(
            cap, embed_table, layer=layer, transform=transform, split_mode=attack_split_mode,
        )
        cover = None
    else:
        return {}

    X, y, _ = cap.stack(kind, layer, transform=transform)
    vinfo = v_information(X, y)              # GPU torch probe
    mdl = online_code_length(X, y)          # GPU torch probe
    if X.shape[0] >= 8:
        Y = embed_table[torch.from_numpy(y)].numpy()
        club = club_mi_upper_bound(X, Y)    # GPU
    else:
        club = {"club_mi_bits": None}

    return {
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


def calibrate_capture(
    cap: CaptureSet,
    embed_table: torch.Tensor,
    *,
    transform: Transform | None = None,
    attack_split_mode: str = "row",
) -> dict:
    """Run attacks + measures over every (kind, layer) block and fit the
    IT-measure → recovery calibration. Model-free: a synthetic CaptureSet
    exercises this in tests.
    """
    transform = transform or Identity()
    blocks = [(k, layer) for k in cap.kinds() for layer in cap.layers(k)]
    records = [
        r
        for k, layer in blocks
        if (
            r := _process_block(
                cap, embed_table, k, layer,
                transform=transform, attack_split_mode=attack_split_mode,
            )
        )
    ]
    calibration = {
        key: calibrate_records(records, measure_key=key)
        for key in ("v_information_bits", "mdl_surplus_bits", "club_mi_bits")
    }
    return {
        "transform": transform.name,
        "attack_split_mode": attack_split_mode,
        "records": records,
        "calibration": calibration,
    }


def run_pass1(
    model_id: str,
    prompts: list[str],
    *,
    layers: list[int] | None = None,
    attack_split_mode: str = "row",
) -> dict:
    """Capture on the GPU, then attack/measure/calibrate under Identity.
    ``attack_split_mode`` defaults to ``"row"`` (resolution A — match the
    class-probe measures; see ``docs/research/attacks_setting.md``).
    """
    from .capture.capture import (  # lazy import of the model stack
        capture_representations,
        embed_table,
        load_model,
    )

    model = load_model(model_id)
    emb = embed_table(model)
    cap = capture_representations(model, prompts, layers=layers)
    report = calibrate_capture(cap, emb, attack_split_mode=attack_split_mode)
    report["model_id"] = model_id
    report["n_prompts"] = len(prompts)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--layers", default=None, help="comma-separated layer indices; default all")
    p.add_argument(
        "--attack-split-mode", default="row", choices=["row", "vocab"],
        help="row = resolution A (match class-probe measures); vocab = honest attacker",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    prompts = _read_corpus(args.corpus)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    report = run_pass1(
        args.model, prompts, layers=layers, attack_split_mode=args.attack_split_mode
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out}")
    for key, cal in report["calibration"].items():
        print(f"  {key:24s} spearman={cal.get('spearman')!s:>8}  r2={cal.get('r_squared')!s:>8}")


if __name__ == "__main__":
    main()

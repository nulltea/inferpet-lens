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
import os
from pathlib import Path

import torch

from .attacks import cover_break, hidden_state
from .calibration import calibrate_records
from .capture.types import CaptureSet
from .measures import club_mi_upper_bound, online_code_length, v_information
from .transforms import Identity, Transform

# CLUB fidelity presets. "fast" is rank-faithful but magnitude-loose
# (validated ~10× speedup, Spearman-vs-recovery unchanged); "full" is the
# converged-magnitude bound. See docs/dev/perf_assumptions.md.
CLUB_FIDELITY = {
    "fast": {"steps": 150, "max_rows": 2500},
    "full": {"steps": 400, "max_rows": None},
}


# Control tasks (docs/dev/control-tasks.md). ``shuffle`` = Hewitt–Liang
# label-permutation floor (selectivity = real − shuffled), applied to every
# estimator. ``vocab`` = the per-vocabulary memorisation gap for the attack
# (ttrsr_row − ttrsr_vocab). ``all`` = both. The control runs reuse a fixed
# seed offset so real and control evaluate the same rows.
CONTROL_SEED = 20260616
ALL_CONTROLS = ("shuffle", "vocab")


def _parse_controls(arg: str) -> frozenset[str]:
    if arg in ("", "none"):
        return frozenset()
    if arg == "all":
        return frozenset(ALL_CONTROLS)
    return frozenset(arg.split(","))


def _selectivity(real: float | None, floor: float | None) -> float | None:
    """real − floor, or None if either is missing."""
    if real is None or floor is None:
        return None
    return float(real) - float(floor)


def _default_club_fidelity() -> str:
    val = os.environ.get("TALENS_CLUB_FIDELITY", "fast")
    return val if val in CLUB_FIDELITY else "fast"


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
    with_mdl: bool = True,
    max_classes: int = 256,
    club_fidelity: str = "fast",
    controls: frozenset[str] = frozenset(),
) -> dict:
    """All attack + measure work for one (kind, layer) block. ``with_mdl``
    toggles the (expensive, prequential) MDL online-code probe; PVI and
    CLUB always run. ``club_fidelity`` ∈ {fast, full} — see
    ``CLUB_FIDELITY`` / docs/dev/perf_assumptions.md.

    ``controls`` (see docs/dev/control-tasks.md) adds, per estimator:
    ``shuffle`` → a label-permutation floor + ``*_selectivity`` (real −
    floor) for PVI/MDL/CLUB/attack; ``vocab`` → the attack's vocab-disjoint
    memorisation gap (``ttrsr_top1_row − ttrsr_top1_vocab``)."""
    if kind not in ("resid_post", "kq", "kqv_out"):
        return {}

    # Stack the operands once and reuse for both the attack and the
    # measures (avoids re-flattening the same block 2-3×).
    X, y, _ = cap.stack(kind, layer, transform=transform)

    # Every surface uses the same ridge inverter (resid_post = IMA/ISA;
    # kq / kqv_out = ISA on the attention surfaces). cover-break is only
    # meaningful for the residual stream.
    attack_name = "hidden_state_inversion" if kind == "resid_post" else f"isa_{kind}"
    atk = hidden_state.run(
        cap, embed_table, layer=layer, kind=kind, transform=transform,
        split_mode=attack_split_mode, xy=(X, y), attack_name=attack_name,
    )
    cover = (
        cover_break.run(cap, layer=layer, kind=kind, transform=transform)
        if kind == "resid_post" else None
    )

    vinfo = v_information(X, y, max_classes=max_classes)   # GPU torch probe
    mdl = (
        online_code_length(X, y, max_classes=max_classes)  # GPU torch probe (off by default)
        if with_mdl else {}
    )
    has_Y = X.shape[0] >= 8
    if has_Y:
        Y = embed_table[torch.from_numpy(y)].numpy()
        club = club_mi_upper_bound(X, Y, **CLUB_FIDELITY[club_fidelity])  # GPU
    else:
        club = {"club_mi_bits": None}

    rec = {
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

    if "shuffle" in controls:
        # Hewitt–Liang label-permutation floor, same rows/split, only labels
        # broken (Option A). selectivity = real − shuffled.
        vinfo_s = v_information(
            X, y, max_classes=max_classes, control="shuffle", control_seed=CONTROL_SEED
        )
        rec["v_information_bits_shuffle"] = vinfo_s.get("v_information_bits")
        rec["v_information_selectivity"] = _selectivity(
            rec["v_information_bits"], rec["v_information_bits_shuffle"]
        )
        if with_mdl:
            mdl_s = online_code_length(
                X, y, max_classes=max_classes, control="shuffle", control_seed=CONTROL_SEED
            )
            rec["mdl_surplus_bits_shuffle"] = mdl_s.get("surplus_description_length_bits")
            rec["mdl_selectivity"] = _selectivity(
                rec["mdl_surplus_bits"], rec["mdl_surplus_bits_shuffle"]
            )
        if has_Y:
            club_s = club_mi_upper_bound(
                X, Y, **CLUB_FIDELITY[club_fidelity], control="shuffle", control_seed=CONTROL_SEED
            )
            rec["club_mi_bits_shuffle"] = club_s.get("club_mi_bits")
            rec["club_mi_selectivity"] = _selectivity(
                rec["club_mi_bits"], rec["club_mi_bits_shuffle"]
            )
        atk_s = hidden_state.run(
            cap, embed_table, layer=layer, kind=kind, transform=transform,
            split_mode=attack_split_mode, control="shuffle", control_seed=CONTROL_SEED,
            xy=(X, y), attack_name=attack_name,
        )
        rec["ttrsr_top1_shuffle"] = atk_s.ttrsr_top1
        rec["ttrsr_selectivity"] = _selectivity(rec["ttrsr_top1"], atk_s.ttrsr_top1)

    if "vocab" in controls:
        # Per-vocabulary memorisation gap (M3): real labels, but test ids
        # held out of training. gap = row − vocab. Reuse the baseline arm
        # when its split already matches.
        def _ttrsr(split_mode: str) -> float | None:
            if split_mode == attack_split_mode:
                return atk.ttrsr_top1
            return hidden_state.run(
                cap, embed_table, layer=layer, kind=kind, transform=transform,
                split_mode=split_mode, xy=(X, y), attack_name=attack_name,
            ).ttrsr_top1

        rec["ttrsr_top1_row"] = _ttrsr("row")
        rec["ttrsr_top1_vocab"] = _ttrsr("vocab")
        rec["ttrsr_mem_gap"] = _selectivity(rec["ttrsr_top1_row"], rec["ttrsr_top1_vocab"])

    return rec


def calibrate_capture(
    cap: CaptureSet,
    embed_table: torch.Tensor,
    *,
    transform: Transform | None = None,
    attack_split_mode: str = "row",
    with_mdl: bool = True,
    max_classes: int = 256,
    workers: int = 4,
    club_fidelity: str = "fast",
    controls: frozenset[str] = frozenset(),
) -> dict:
    """Run attacks + measures over every (kind, layer) block and fit the
    IT-measure → recovery calibration. Model-free: a synthetic CaptureSet
    exercises this in tests. ``with_mdl`` toggles the MDL probe (and its
    calibration column).

    Blocks are independent, so they run on a bounded thread pool
    (``workers``): the per-block CPU prep (stacking, splits) overlaps the
    GPU work (probe/CLUB/ridge) of other blocks. numpy/torch release the
    GIL, so threads — not processes — suffice and avoid re-sending the
    capture/embeddings. ``workers=1`` forces sequential. Records are
    re-ordered back to block order for deterministic output.
    """
    from concurrent.futures import ThreadPoolExecutor

    transform = transform or Identity()
    blocks = [(k, layer) for k in cap.kinds() for layer in cap.layers(k)]

    def work(block: tuple[str, int]) -> dict:
        k, layer = block
        return _process_block(
            cap, embed_table, k, layer,
            transform=transform, attack_split_mode=attack_split_mode,
            with_mdl=with_mdl, max_classes=max_classes, club_fidelity=club_fidelity,
            controls=controls,
        )

    n_workers = max(1, min(workers, len(blocks)))
    if n_workers == 1:
        results = [work(b) for b in blocks]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(work, blocks))  # map preserves input order
    records = [r for r in results if r]
    keys = ["v_information_bits", "club_mi_bits"]
    if with_mdl:
        keys.insert(1, "mdl_surplus_bits")
    calibration = {key: calibrate_records(records, measure_key=key) for key in keys}

    if "shuffle" in controls:
        # Does the floor-subtracted measure predict recovery better than the
        # raw measure? Calibrate each *_selectivity column against raw recovery
        # and against the selectivity-adjusted recovery. (docs/dev/control-tasks.md)
        sel_keys = ["v_information_selectivity", "club_mi_selectivity"]
        if with_mdl:
            sel_keys.insert(1, "mdl_selectivity")
        for key in sel_keys:
            calibration[key] = calibrate_records(records, measure_key=key)
            calibration[f"{key}__vs_ttrsr_selectivity"] = calibrate_records(
                records, measure_key=key, recovery_key="ttrsr_selectivity"
            )

    return {
        "transform": transform.name,
        "attack_split_mode": attack_split_mode,
        "controls": sorted(controls),
        "records": records,
        "calibration": calibration,
    }


def run_pass1(
    model_id: str,
    prompts: list[str],
    *,
    layers: list[int] | None = None,
    capture_layers: list[int] | None = None,
    every_n: int = 1,
    attack_split_mode: str = "row",
    with_mdl: bool = False,
    max_classes: int = 256,
    cache_dir: str | None = None,
    refresh_capture: bool = False,
    workers: int = 4,
    club_fidelity: str = "fast",
    controls: frozenset[str] = frozenset(),
) -> dict:
    """Capture on the GPU, then attack/measure/calibrate under Identity.
    ``attack_split_mode`` defaults to ``"row"`` (resolution A — match the
    class-probe measures; see ``docs/research/attacks_setting.md``).

    Capture is cached to disk (keyed by model + corpus): ``capture_layers``
    controls what is captured/cached (default: all), ``layers`` controls
    which captured layers are actually measured. A rerun whose ``layers``
    are within an existing cache skips the model forward pass entirely.
    """
    from .capture.capture import load_or_capture  # lazy: model stack
    from .capture.cache import present_layers, subset_capture

    cap, emb, source = load_or_capture(
        model_id, prompts, capture_layers=capture_layers,
        cache_dir=cache_dir, refresh=refresh_capture,
    )
    # Resolve the measured-layer set: base = explicit ``layers`` or all
    # captured, then keep every ``every_n``-th.
    base = layers if layers is not None else sorted(present_layers(cap))
    if every_n > 1:
        base = base[::every_n]
    if layers is not None or every_n > 1:
        missing = sorted(set(base) - present_layers(cap))
        if missing:
            raise ValueError(
                f"layers {missing} not in capture (have "
                f"{sorted(present_layers(cap))}); widen --capture-layers"
            )
        cap = subset_capture(cap, base)

    report = calibrate_capture(
        cap, emb, attack_split_mode=attack_split_mode,
        with_mdl=with_mdl, max_classes=max_classes, workers=workers,
        club_fidelity=club_fidelity, controls=controls,
    )
    report["model_id"] = model_id
    report["n_prompts"] = len(prompts)
    report["capture_source"] = source
    report["club_fidelity"] = club_fidelity
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument(
        "--layers", default=None,
        help="layers to MEASURE: a single int N = the first N layers (0..N-1); "
             "a comma list = explicit indices; default all captured. "
             "(must be ⊆ --capture-layers)",
    )
    p.add_argument(
        "--every-n", type=int, default=1,
        help="measure every N-th layer of the --layers set (1 = all)",
    )
    p.add_argument(
        "--capture-layers", default=None,
        help="comma-separated layers to CAPTURE + cache; default all. Capture a "
             "wide set once, then rerun with narrower --layers off the cache.",
    )
    p.add_argument(
        "--cache-dir", default=None,
        help="capture/embedding cache dir (default results/capture_cache)",
    )
    p.add_argument(
        "--refresh-capture", action="store_true",
        help="force a fresh capture even if a usable cache exists",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="thread-pool size over (kind,layer) blocks (1 = sequential)",
    )
    p.add_argument(
        "--club-fidelity", choices=sorted(CLUB_FIDELITY), default=_default_club_fidelity(),
        help="fast = rank-faithful/magnitude-loose CLUB (~10× faster, validated); "
             "full = converged-magnitude bound. Env: TALENS_CLUB_FIDELITY. "
             "See docs/dev/perf_assumptions.md",
    )
    p.add_argument(
        "--attack-split-mode", default="row", choices=["row", "vocab"],
        help="row = resolution A (match class-probe measures); vocab = honest attacker",
    )
    p.add_argument(
        "--mdl", action="store_true",
        help="run the MDL online-code probe (OFF by default: it trains ~7 probes "
             "per block and ~doubles the measure phase; PVI + CLUB always run). "
             "Its shuffle-control floor reads ≈0 like PVI — see docs/dev/control-tasks.md",
    )
    p.add_argument(
        "--max-classes", type=int, default=256,
        help="cap distinct token-ids the class-probe predicts (top-N most frequent)",
    )
    p.add_argument(
        "--control", default="none", choices=["none", "shuffle", "vocab", "all"],
        help="control task(s) for memorisation (docs/dev/control-tasks.md): "
             "shuffle = label-permutation floor + *_selectivity on every estimator; "
             "vocab = attack vocab-disjoint memorisation gap; all = both",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    prompts = _read_corpus(args.corpus)

    def _parse_layers(spec: str | None) -> list[int] | None:
        if not spec:
            return None
        if "," in spec:
            return [int(x) for x in spec.split(",")]      # explicit indices
        return list(range(int(spec)))                      # single int = first-N count

    layers = _parse_layers(args.layers)
    capture_layers = _parse_layers(args.capture_layers)
    report = run_pass1(
        args.model, prompts, layers=layers, capture_layers=capture_layers,
        every_n=args.every_n, attack_split_mode=args.attack_split_mode,
        with_mdl=args.mdl, max_classes=args.max_classes,
        cache_dir=args.cache_dir, refresh_capture=args.refresh_capture,
        workers=args.workers, club_fidelity=args.club_fidelity,
        controls=_parse_controls(args.control),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out}  (capture: {report.get('capture_source')})")
    for key, cal in report["calibration"].items():
        print(f"  {key:24s} spearman={cal.get('spearman')!s:>8}  r2={cal.get('r_squared')!s:>8}")


if __name__ == "__main__":
    main()

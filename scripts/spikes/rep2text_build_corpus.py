#!/usr/bin/env python3
"""Build a length-stratified NATURAL-TEXT corpus for the resid-rep2text surface.

Reads cached HF datasets (piqa goals, mmlu questions, if_eval prompts) via pyarrow,
dedupes, filters to clean single-line sentences, and stratifies by *word* count so the
final corpus spans a wide length range roughly uniformly (the eval needs populated
length buckets). Writes ``corpora/rep2text-stratified.txt`` (one prompt per line).

CPU-only; runs in the ROCm container only because pyarrow lives there. Deterministic.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pyarrow.ipc as ipc

REPO = Path(__file__).resolve().parents[2]
HF = Path.home() / ".cache" / "huggingface" / "datasets"
SEED = 20260624
OUT = REPO / "corpora" / "rep2text-stratified.txt"

SOURCES = [
    # (glob under HF datasets, column name)
    ("lighteval___piqa/**/piqa-train.arrow", "goal"),
    ("cais___mmlu/**/mmlu-auxiliary_train.arrow", "question"),
    ("cais___mmlu/**/mmlu-validation.arrow", "question"),
    ("google___if_eval/**/*.arrow", "prompt"),
]


def read_col(glob: str, col: str) -> list[str]:
    out: list[str] = []
    for path in sorted(HF.glob(glob)):
        try:
            with ipc.open_stream(str(path)) as r:
                tbl = r.read_all()
        except Exception:
            try:
                with ipc.open_file(str(path)) as r:
                    tbl = r.read_all()
            except Exception as e:  # pragma: no cover
                print(f"[corpus] skip {path.name}: {e}", file=sys.stderr)
                continue
        if col not in tbl.column_names:
            continue
        out.extend(str(x) for x in tbl.column(col).to_pylist())
    return out


def clean(s: str) -> str | None:
    s = re.sub(r"\s+", " ", s.strip())
    # single sentence-ish, drop multi-line / list / code-y instructions
    if "\n" in s or len(s) < 12:
        return None
    if any(tok in s for tok in ("```", "def ", "import ", "http", "  - ", "1.")):
        return None
    # keep first sentence only if very long, to control upper length
    return s


def main() -> None:
    import random

    rng = random.Random(SEED)
    raw: list[str] = []
    for glob, col in SOURCES:
        got = read_col(glob, col)
        print(f"[corpus] {glob.split('/')[0]}::{col} -> {len(got)} rows", flush=True)
        raw.extend(got)

    seen: set[str] = set()
    cleaned: list[str] = []
    for s in raw:
        c = clean(s)
        if c and c not in seen:
            seen.add(c)
            cleaned.append(c)
    rng.shuffle(cleaned)

    # stratify by word count into buckets, sample evenly to span the range
    buckets: dict[int, list[str]] = {}
    for s in cleaned:
        w = len(s.split())
        if 6 <= w <= 60:
            buckets.setdefault(min(w // 6, 9), []).append(s)  # bucket width 6 words
    per = 220  # target per stratum
    final: list[str] = []
    for b in sorted(buckets):
        chunk = buckets[b][:per]
        final.extend(chunk)
        print(f"[corpus] words[{b*6}-{b*6+5}]: have {len(buckets[b])}, take {len(chunk)}", flush=True)
    rng.shuffle(final)

    OUT.write_text("\n".join(final) + "\n")
    print(f"[corpus] wrote {len(final)} prompts -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

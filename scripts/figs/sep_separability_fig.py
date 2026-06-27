#!/usr/bin/env python3
"""Render token-class separability comparison figures (matplotlib → PNG) from a dp_leakage_sweep run.

Reads a dp_leakage_sweep.py JSON carrying the `sep` probe and writes two PNGs to docs/html/img/:

  sep_cloud_grid.png  — 2D PCA projection of the {is,are,was,were} class clouds, rows = ε, cols =
                        layer. The MDL-blog "does the representation sort the labels?" picture: tight
                        separated clouds at L0 smear together with depth and stronger noise.
  sep_trajectory.png  — separability vs depth, one line per ε, four panels:
                        Bhattacharyya margin · MDL-info (achievable, bits) · channel-MI converse
                        (capped at log2 K) · ridge recovery (contrast). Lets you compare across
                        layers AND ε and read off the conclusion.

Conclusion this surfaces: separability (margin + MDL) DECLINES with depth at every ε — no L20 peak —
opposite to the I_G/CLUB L20 rise (FIG.04). So the L20 information peak is representation-channel
survival, not token-class information.

    .venv/bin/python scripts/figs/sep_separability_fig.py --in refine-logs/sep-test/dp_sep_grid.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_COLOR = {"is": "#e6194b", "are": "#3cb44b", "was": "#4363d8", "were": "#f58231"}
EPS_COLOR = {"∞": "#1b1f24", "1024": "#6a4c93", "512": "#2f7d8a", "256": "#c98a3e", "128": "#b55831",
             "64": "#2f5e87", "32": "#2f7d8a", "16": "#c98a3e", "8": "#b55831"}
INK = "#14181c"

plt.rcParams.update({
    "font.family": "monospace", "font.size": 9, "axes.edgecolor": "#888",
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 150,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
})


def _eps_label(e) -> str:
    return "∞" if e is None else f"{e:g}"


def _by_cell(records):
    return {(r.get("epsilon"), r["layer"]): r for r in records}


def cloud_grid(cells, layers, eps_rows, out_path):
    nr, nc = len(eps_rows), len(layers)
    fig, axes = plt.subplots(nr, nc, figsize=(2.0 * nc, 1.9 * nr), squeeze=False)
    for ri, e in enumerate(eps_rows):
        for ci, L in enumerate(layers):
            ax = axes[ri][ci]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#bbb")
            rec = cells.get((e, L))
            if rec and rec.get("sep_coords"):
                co = rec["sep_coords"]
                for lab, c in CLASS_COLOR.items():
                    xs = [p[0] for p in co if p[2] == lab]
                    ys = [p[1] for p in co if p[2] == lab]
                    if xs:
                        ax.scatter(xs, ys, s=7, c=c, alpha=0.8, linewidths=0, label=lab)
                marg = rec.get("sep_bhat_dist")
                ax.text(0.04, 0.93, f"D_B={marg:.0f}" if marg is not None else "", transform=ax.transAxes,
                        fontsize=7, va="top", color="#444")
            if ri == 0:
                ax.set_title(f"L{L}", fontsize=10, color=INK)
            if ci == 0:
                ax.set_ylabel(f"ε={_eps_label(e)}", fontsize=10, color=INK, rotation=90, labelpad=6)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec="none", ms=6, label=lab)
               for lab, c in CLASS_COLOR.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("token-class clouds (2D PCA) — separability collapses with depth & noise",
                 y=1.04, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def trajectory(cells, layers, eps_list, out_path):
    panels = [
        ("sep_bhat_dist",      "Bhattacharyya margin (separability)", True),
        ("sep_mdl_info_bits",  "MDL-info — achievable (bits)",        False),
        ("sep_mi_converse_bits", "channel-MI converse (bits, cap=log₂K)", False),
        ("ridge",              "ridge recovery (top-1) — contrast",   False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for ax, (key, title, logy) in zip(axes.flat, panels):
        for e in eps_list:
            lab = _eps_label(e)
            xs, ys = [], []
            for L in layers:
                r = cells.get((e, L))
                v = r.get(key) if r else None
                if v is not None:
                    xs.append(L); ys.append(v)
            if xs:
                ax.plot(xs, ys, "-o", ms=4, lw=1.6, color=EPS_COLOR.get(lab, "#666"), label=f"ε={lab}")
        ax.set_title(title, color=INK)
        ax.set_xlabel("layer (observation depth)")
        ax.set_xticks(layers)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, ls=":", lw=0.6, alpha=0.5)
        if key == "sep_mi_converse_bits":
            ax.axhline(2.0, ls="--", lw=0.8, color="#999")
            ax.text(layers[-1], 2.0, " log₂K", va="bottom", ha="right", fontsize=7, color="#999")
    axes.flat[0].legend(frameon=False, loc="upper right")
    fig.suptitle("separability vs depth across ε — margin & MDL fall with depth (no mid-depth peak)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="refine-logs/sep-test/dp_sep_grid.json")
    ap.add_argument("--outdir", default="docs/html/img")
    ap.add_argument("--cloud-layers", default="", help="comma layer subset for the cloud grid "
                    "(too many layers makes 2D panels unreadable); default = all layers in the run")
    args = ap.parse_args()

    d = json.loads(Path(args.inp).read_text())
    cells = _by_cell(d["records"])
    layers = d["layers"]
    eps_list = d["epsilons"]  # None for ∞
    cloud_layers = [int(s) for s in args.cloud_layers.split(",") if s.strip()] or layers
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cloud_grid(cells, cloud_layers, eps_list, outdir / "sep_cloud_grid.png")
    trajectory(cells, layers, eps_list, outdir / "sep_trajectory.png")
    print(f"wrote {outdir}/sep_cloud_grid.png and sep_trajectory.png "
          f"({len(cells)} cells, layers={layers}, eps={[_eps_label(e) for e in eps_list]})")


if __name__ == "__main__":
    main()

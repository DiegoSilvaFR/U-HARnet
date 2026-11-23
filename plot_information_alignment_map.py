#!/usr/bin/env python3
"""Information--alignment map (Theorem 1): kappa (x) versus rho (y).

Reads figures/kappa_rho_summary.csv produced by kappa_rho_analysis.py and
draws one panel per ranking strategy (rows) and fine-tuning regime (columns).
Each point is one dataset (color) and HR head (marker), with 95% subject
bootstrap intervals. Gray curves are constant relative improvement
100 * Delta / R_all = 100 * rho * kappa (labelled in %). Settings in which no
information was detected are drawn hollow on rho = 0 (rho is undefined).

By Corollary 1, logistic alignment cannot change kappa: comparing the
"Raw HR" and "Aligned HR" rows, points should move vertically only.

Usage (repository root):
    python plot_information_alignment_map.py
    python plot_information_alignment_map.py --rows raw_hr /logistic_hr \
        --summary figures/kappa_rho_summary.csv --output-dir figures

Outputs: information_alignment_map.pdf / .png
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Categorical slots 1-5 of the validated reference palette (fixed order).
DATASET_COLORS = OrderedDict([
    ("CAPTURE-24", "#2a78d6"),
    ("HARTH", "#eb6834"),
    ("PAMAP2", "#1baf7a"),
    ("REALWORLD", "#eda100"),
    ("USC-HAD", "#e87ba4"),
])
HEAD_MARKERS = OrderedDict([
    ("gaussian", ("o", "Gaussian NLL")),
    ("student_t", ("s", "Student-t NLL")),
    ("mc_dropout", ("^", "MC Dropout")),
    ("CE", ("D", "CE (0% only)")),
])
ROW_LABELS = OrderedDict([
    ("raw_hr", "Raw HR uncertainty"),
    ("/logistic_hr", "Aligned HR uncertainty"),
    ("/har_entropy", "HAR entropy"),
    ("/fusion", "Fusion"),
    ("motion", "Motion intensity"),
])
SHARED_AT_ZERO = {"/har_entropy", "/logistic_har_entropy", "motion"}
DEPTHS = (0, 25, 100)
TITLES = {0: "Frozen encoder (0%)", 25: "Partial fine-tuning (25%)", 100: "Full fine-tuning (100%)"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#d9d9d6"


def draw_iso_curves(ax, x_max, y_lim, levels=(-0.6, -0.4, -0.2, 0.2, 0.4, 0.6)):
    x = np.linspace(1e-3, x_max, 400)
    for d in levels:
        y = d / x
        ok = np.abs(y) <= y_lim
        if not ok.any():
            continue
        ax.plot(x[ok], y[ok], color=GRID, lw=0.9, zorder=0)
        ax.text(x[ok][-1], y[ok][-1], f"{100 * d:+.0f}%", fontsize=6.5, color=MUTED,
                ha="right", va="bottom" if d > 0 else "top", zorder=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=Path("figures/kappa_rho_summary.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("figures"))
    ap.add_argument("--rows", nargs="*", default=["raw_hr", "/logistic_hr", "/har_entropy", "motion"],
                    choices=list(ROW_LABELS), help="Strategies to show, one row each.")
    ap.add_argument("--no-ce", action="store_true", help="Omit the 0%% CE baseline.")
    ap.add_argument("--fig-width", type=float, default=13.5)
    ap.add_argument("--ylim", type=float, default=1.25,
                    help="Symmetric rho axis limit (default 1.25); longer intervals are clipped.")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.summary)
    if args.no_ce:
        df = df[df["head"] != "CE"]
    df = df[df["strategy"].isin(args.rows)]
    # At 0%, entropy and motion are identical for every head: plot them once.
    zero_shared = (df["fine_tuning_pct"] == 0) & df["strategy"].isin(SHARED_AT_ZERO)
    df = pd.concat([df[~zero_shared],
                    df[zero_shared].drop_duplicates(["dataset", "fine_tuning_pct", "strategy"])])
    if df.empty:
        raise SystemExit("No rows to plot: check --summary and --rows.")

    x_max = float(np.nanmax(np.r_[df["rel_kappa_hi"].to_numpy(), df["rel_kappa"].to_numpy()])) * 1.08
    # |rho| <= 1 in the population; intervals reaching beyond the axis (e.g.
    # PAMAP2, five subjects) are clipped at the panel edge.
    y_lim = args.ylim

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
                         "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    n_rows = len(args.rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(args.fig_width, 2.55 * n_rows + 1.0),
                             sharex=True, sharey=True, squeeze=False)
    for r, strategy in enumerate(args.rows):
        for c, depth in enumerate(DEPTHS):
            ax = axes[r, c]
            draw_iso_curves(ax, x_max, y_lim)
            ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
            sub = df[(df["strategy"] == strategy) & (df["fine_tuning_pct"] == depth)]
            for _, row in sub.iterrows():
                color = DATASET_COLORS.get(row["dataset"], "#888888")
                shared = depth == 0 and strategy in SHARED_AT_ZERO
                marker = "o" if shared else HEAD_MARKERS.get(row["head"], ("o", ""))[0]
                detected = bool(row["information_detected"])
                y = row["rho"] if detected else 0.0
                xerr = [[max(row["rel_kappa"] - row["rel_kappa_lo"], 0)],
                        [max(row["rel_kappa_hi"] - row["rel_kappa"], 0)]]
                yerr = ([[max(row["rho"] - row["rho_lo"], 0)], [max(row["rho_hi"] - row["rho"], 0)]]
                        if detected and np.isfinite(row["rho_lo"]) else None)
                ax.errorbar(row["rel_kappa"], y, xerr=xerr, yerr=yerr, fmt="none",
                            ecolor=color, elinewidth=0.9, alpha=0.75, zorder=2)
                ax.plot(row["rel_kappa"], y, marker=marker, ms=7.5, ls="none",
                        mfc=color if detected else "white", mec=INK if detected else color,
                        mew=0.7 if detected else 1.3, zorder=3)
            ax.set_xlim(0, x_max)
            ax.set_ylim(-y_lim, y_lim)
            ax.grid(False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED); ax.spines[side].set_linewidth(0.6)
            ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
            if r == 0:
                ax.set_title(TITLES[depth], fontsize=11, fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(f"{ROW_LABELS[strategy]}\nalignment " + r"$\hat\rho$", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel(r"information $\hat\kappa$ (relative to $R_{\mathrm{all}}$)", fontsize=9)

    dataset_handles = [Line2D([], [], marker="o", ls="none", mfc=col, mec=INK, mew=0.7, ms=7,
                              label=name) for name, col in DATASET_COLORS.items()
                       if name in set(df["dataset"])]
    heads_present = set(df["head"])
    head_handles = [Line2D([], [], marker=m, ls="none", mfc="#bdbdbd", mec=INK, mew=0.7, ms=7,
                           label=lab) for h, (m, lab) in HEAD_MARKERS.items() if h in heads_present]
    head_handles.append(Line2D([], [], marker="o", ls="none", mfc="white", mec=MUTED, mew=1.3, ms=7,
                               label="no information detected"))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09,
                        hspace=0.18, wspace=0.08)
    for handles, y in ((dataset_handles, 0.04), (head_handles, 0.015)):
        fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                   bbox_to_anchor=(0.5, y), frameon=False, fontsize=8.5,
                   columnspacing=1.6, handletextpad=0.4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "information_alignment_map"
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix("." + ext), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Created:", stem.with_suffix(".pdf"), stem.with_suffix(".png"), sep="\n  ")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot mean R@1 versus mean classification macro-F1 from HAR results JSONs.

Examples
--------
python plot_r1_vs_f1.py results_capture24.json results_harth.json \
    results_pamap.json results_realworld.json results_usc.json \
    --output-dir figures

Input layout (no conversion needed)
-----------------------------------
zero_shot.finetuned_{depth}_results.{method}.by_subject.{user}.R@1
classification.finetuned_0_results.{evaluation_id}.{method}.macro_f1
classification.finetuned_{25,100}_results.{method}.{evaluation_id}.macro_f1

Each data point corresponds to one configuration, not one participant or fold.
Coordinates are arithmetic means and errors are unweighted *sample SDs*
(ddof=1), not standard errors/confidence intervals.
Original frozen-encoder baseline (0%) appears once.
For the combined figure, marker fill+shape encode method, dataset-colored
marker outlines identify each dataset, and size encodes fine-tuning depth.
An additional figure uses faint, unfilled grouping envelopes.
No per-point labels or error bars on either combined plot.
Individual dataset figures retain the old method-color styling.
CAPTURE-24 has subject-level R@1 but fold-level classification macro-F1;
these are calculated separately over their own native evaluation units.

Outputs
-------
One clean combined PDF/PNG using dataset-colored marker outlines,
one optional combined PDF/PNG with dashed convex-hull group envelopes,
and one CSV with the mean, SD, n, and aggregation unit of each coordinate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

METHODS = {
    "gaussian":   {"label": "Gaussian NLL",  "color": "#D95F02", "marker": "p"},
    "student_t":  {"label": "Student-t NLL", "color": "#7B52AB", "marker": "h"},
    "mc_dropout": {"label": "MC Dropout",    "color": "#009E91", "marker": "D"},
}
ORIGINAL = {"label": "Original encoder", "color": "#757575", "marker": "*"}
AREA = {0: 160, 25: 78, 100: 145}
DATASET_NAMES = {"CAPTURE24": "CAPTURE-24", "HARTH": "HARTH", "PAMAP": "PAMAP2",
                 "REALWORLD": "REALWORLD", "USC": "USC-HAD"}

# Dataset-outline colors in the combined plot. The palette is chosen
# to remain reasonably distinguishable for color-vision deficiencies.
DATASET_COLORS = {
    "CAPTURE-24": "#0072B2",
    "HARTH": "#D55E00",
    "PAMAP2": "#009E73",
    "REALWORLD": "#CC79A7",
    "USC-HAD": "#7A60A9",
}


def _stats(metrics_by_id: dict, metric: str, context: str):
    vals = []
    for unit, metrics in metrics_by_id.items():
        if not isinstance(metrics, dict):
            raise ValueError("Invalid metric record at %s[%s]" % (context, unit))
        v = metrics.get(metric)
        if v is not None and math.isfinite(float(v)):
            vals.append(float(v))
    if not vals:
        raise ValueError("Missing %s values at %s" % (metric, context))
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0., len(arr)


def extract_summary(data: dict, dataset: str, source_file: str = ""):
    zs = data["zero_shot"]
    cls = data["classification"]
    rows = []
    for depth in (0, 25, 100):
        key = "finetuned_%s_results" % depth
        if key not in zs or key not in cls:
            continue
        if depth == 0:
            original_method = "CE" if "CE" in zs[key] else next(iter(zs[key]), None)
            if original_method is None:
                continue
            # Exact same embedding and linear probe for all frozen-head methods.
            probe_method = next((m for m in ("CE", "gaussian", "student_t", "mc_dropout")
                                 if all(isinstance(f, dict) and m in f for f in cls[key].values())), None)
            if probe_method is None:
                raise ValueError("%s: missing 0%% classification scores" % dataset)
            configurations = [("original", zs[key][original_method],
                               {unit: scores[probe_method] for unit, scores in cls[key].items()})]
        else:
            configurations = [(m, zs[key][m], cls[key][m])
                              for m in METHODS if m in zs[key] and m in cls[key]]

        for method, retrieval, f1_by_id in configurations:
            r1_by_id = retrieval.get("by_subject")
            if not isinstance(r1_by_id, dict):
                raise ValueError("%s %s%% %s missing by_subject R@1" % (dataset, depth, method))
            xm, xs, nx = _stats(r1_by_id, "R@1", "%s/%s/%s/retrieval" % (dataset, depth, method))
            ym, ys, ny = _stats(f1_by_id, "macro_f1", "%s/%s/%s/classification" % (dataset, depth, method))
            y_unit = "subject" if set(f1_by_id) == set(r1_by_id) else "evaluation fold / split"
            rows.append({
                "dataset": DATASET_NAMES.get(dataset, dataset),
                "method": method,
                "method_label": ORIGINAL["label"] if depth == 0 else METHODS[method]["label"],
                "finetune_pct": depth,
                "r1_mean": xm, "r1_sd": xs, "n_r1": nx, "r1_unit": "subject",
                "macro_f1_mean": ym, "macro_f1_sd": ys, "n_macro_f1": ny, "macro_f1_unit": y_unit,
                "source_file": source_file,
            })
    if not rows:
        raise ValueError("No plot-compatible entries in %s" % dataset)
    return rows


def configure_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
        "axes.linewidth": 0.95, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "savefig.dpi": 350,
    })


def _axis_limits(rows, mean_only=False):
    if mean_only:
        xx = [r["r1_mean"] for r in rows]
        yy = [r["macro_f1_mean"] for r in rows]
    else:
        xx = [q for r in rows for q in (r["r1_mean"] - r["r1_sd"], r["r1_mean"] + r["r1_sd"])]
        yy = [q for r in rows for q in (r["macro_f1_mean"] - r["macro_f1_sd"], r["macro_f1_mean"] + r["macro_f1_sd"])]
    dx = max(max(xx) - min(xx), 0.02)
    dy = max(max(yy) - min(yy), 0.03)
    xlims = (max(0, min(xx) - .07 * dx), min(1, max(xx) + .07 * dx))
    ylims = (max(0, min(yy) - .07 * dy), min(1, max(yy) + .07 * dy))
    return xlims, ylims


def _convex_hull_vertices(points):
    """Andrew's monotone chain convex hull (no scipy dependency)."""
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def plot_combined(by_dataset, outstem, *, group_envelopes=False):
    """One scatter plot. Method is fill+shape; dataset is thin marker halo.

    Optional unfilled convex-hull envelopes visually group each dataset.
    Overlap does not obscure any points because hulls have no fill and are
    drawn behind points. No error bars and no per-point text labels.
    """
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(8.9, 4.9))
    ax.set_axisbelow(True)
    ax.grid(True, color="#D5D5D5", alpha=0.73, linewidth=0.62)

    rows = [r for group in by_dataset.values() for r in group]
    (xmin, xmax), (ymin, ymax) = _axis_limits(rows, mean_only=True)
    # A little extra room if envelopes are drawn around point extremes.
    if group_envelopes:
        xpad, ypad = (xmax - xmin) * .032, (ymax - ymin) * .034
        xmin, xmax = max(0., xmin-xpad), min(1., xmax+xpad)
        ymin, ymax = max(0., ymin-ypad), min(1., ymax+ypad)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if group_envelopes:
        # Convex hull in axis-normalized coordinates. Offset by 1.8% of the
        # plotting size to keep the outline away from each marker.
        for dataset, group in by_dataset.items():
            points = [((r['r1_mean']-xmin)/(xmax-xmin),
                       (r['macro_f1_mean']-ymin)/(ymax-ymin)) for r in group]
            hull = _convex_hull_vertices(points)
            if len(hull) < 3:
                continue
            ctr = np.mean(hull, axis=0)
            padded = []
            for v in hull:
                v = np.array(v, dtype=float)
                radial = v-ctr
                radial_norm = np.linalg.norm(radial)
                if radial_norm > 0:
                    v = v + .022 * radial / radial_norm
                padded.append(v)
            ax.add_patch(Polygon(
                padded, transform=ax.transAxes, closed=True,
                fill=False, edgecolor=DATASET_COLORS[dataset],
                linewidth=1.25, linestyle=(0, (4, 3)), alpha=.52,
                joinstyle='round', zorder=2,
            ))

    # Three layers: dataset-color outer wrapper, slim white separator,
    # method-color filled marker. Every point keeps its genuine coordinates.
    for r in rows:
        d = r['finetune_pct']
        st = ORIGINAL if d == 0 else METHODS[r['method']]
        x, y, marker = r['r1_mean'], r['macro_f1_mean'], st['marker']
        dataset_color = DATASET_COLORS[r['dataset']]
        size = AREA[d]
        # Long points on the star need a stronger outer colored silhouette.
        if d == 0:
            ax.scatter(x, y, s=size+142, marker=marker, facecolor=dataset_color,
                       edgecolor='white', linewidth=.30, zorder=5)
            ax.scatter(x, y, s=size+47, marker=marker, facecolor='white',
                       edgecolor='none', zorder=6)
            ax.scatter(x, y, s=size, marker=marker, facecolor=st['color'],
                       edgecolor='none', zorder=7)
        else:
            ax.scatter(x, y, s=size+90, marker=marker, facecolor=dataset_color,
                       edgecolor='white', linewidth=.65, zorder=5)
            ax.scatter(x, y, s=size+27, marker=marker, facecolor='white',
                       edgecolor='white', linewidth=0, zorder=6)
            ax.scatter(x, y, s=size, marker=marker, facecolor=st['color'],
                       edgecolor='none', zorder=7)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Embedding quality, mean R@1')
    ax.set_ylabel('Classification, mean macro-F1')

    # Two compact, non-overlapping legends, no labels on dots.
    method_handles = [
        Line2D([], [], marker=st['marker'], linestyle='None', color='none',
               markerfacecolor=st['color'], markeredgecolor='#EAEAEA',
               markeredgewidth=1.0, markersize=8.5, label=st['label'])
        for st in METHODS.values()
    ]
    method_handles.append(Line2D([], [], marker='*', linestyle='None', color='none',
                                 markerfacecolor=ORIGINAL['color'],
                                 markeredgecolor='#EAEAEA', markeredgewidth=1.0,
                                 markersize=10, label='Original encoder'))
    ds_handles = [
        Line2D([], [], marker='o', linestyle='None', color='none',
               markerfacecolor='white', markeredgecolor=color,
               markeredgewidth=2.2, markersize=8, label=dataset)
        for dataset, color in DATASET_COLORS.items() if dataset in by_dataset
    ]

    legend_models = ax.legend(handles=method_handles, title='Method (marker fill)',
                              loc='upper left', bbox_to_anchor=(1.015, 1.),
                              fontsize=8.3, title_fontsize=8.6,
                              frameon=True, fancybox=False, borderpad=.6,
                              edgecolor='#C5C5C5')
    ax.add_artist(legend_models)
    ax.legend(handles=ds_handles, title='Dataset (outline)',
              loc='upper left', bbox_to_anchor=(1.015, .49),
              fontsize=8.3, title_fontsize=8.6,
              frameon=True, fancybox=False, borderpad=.6,
              edgecolor='#C5C5C5')
    fig.text(.775, .117,
             'Small: 25% fine-tuning\nLarge: 100% fine-tuning\nStar: 0% original',
             ha='left', va='bottom', fontsize=8.2)
    fig.subplots_adjust(left=.10, right=.73, bottom=.15, top=.965)
    for ext in ('png', 'pdf'):
        fig.savefig(str(outstem) + '.' + ext, dpi=350, facecolor='white')
    plt.close(fig)


def plot_dataset_panels(by_dataset, outstem):
    """One publication figure with five dataset panels and one shared legend.

    Every panel shows its own tick numbers and axis labels, per user request.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4), sharex=True, sharey=True)
    axs = list(axes.flat)
    rows = [r for group in by_dataset.values() for r in group]
    (xmin, xmax), (ymin, ymax) = _axis_limits(rows, mean_only=True)

    xpad = (xmax - xmin) * 0.03
    ypad = (ymax - ymin) * 0.04
    limits = (max(0, xmin-xpad), min(1, xmax+xpad),
              max(0, ymin-ypad), min(1, ymax+ypad))

    dataset_items = list(by_dataset.items())
    panel_positions = [0, 1, 2, 3, 4]

    for (dataset, group), pos in zip(dataset_items, panel_positions):
        ax = axs[pos]
        ax.set_axisbelow(True)
        ax.grid(True, color='#D6D6D6', linewidth=0.6, alpha=0.78)

        for depth in (100, 25, 0):
            for r in group:
                if r['finetune_pct'] != depth:
                    continue
                style = ORIGINAL if depth == 0 else METHODS[r['method']]
                ax.scatter(r['r1_mean'], r['macro_f1_mean'],
                           c=style['color'], s=AREA[depth],
                           marker=style['marker'], edgecolors='white',
                           linewidths=.9, zorder=6 if depth == 0 else (5 if depth == 25 else 4))

        ax.text(0.03, 0.95, dataset, transform=ax.transAxes,
                ha='left', va='top', fontsize=11, weight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.78, pad=1.5))

        ax.set_xlim(limits[:2])
        ax.set_ylim(limits[2:])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(labelsize=9, direction='out', labelleft=True, labelbottom=True)
        ax.set_xlabel('mean R@1', fontsize=10)
        ax.set_ylabel('mean macro-F1', fontsize=10)

    legax = axs[-1]
    legax.axis('off')
    legend_handles = [Line2D([], [], marker=style['marker'], linestyle='None',
                             markerfacecolor=style['color'], markeredgecolor='white',
                             markeredgewidth=.8, color='none', markersize=9,
                             label=style['label'])
                      for style in METHODS.values()]
    legend_handles.append(Line2D([], [], marker=ORIGINAL['marker'], linestyle='None',
                                 markerfacecolor=ORIGINAL['color'], markeredgecolor='white',
                                 markeredgewidth=.8, color='none', markersize=10,
                                 label='Original encoder (0%)'))
    legax.legend(handles=legend_handles, loc='upper left', frameon=False,
                 title='Encoder / HR objective', title_fontsize=11,
                 fontsize=10, borderaxespad=0)
    legax.text(.0, .36,
               'Small markers: 25% fine-tuning\n'
               'Large markers: 100% fine-tuning\n'
               'Star: original encoder (0%)',
               transform=legax.transAxes, ha='left', va='top',
               fontsize=10, linespacing=1.45)

    fig.subplots_adjust(left=.07, bottom=.08, right=.985, top=.965,
                        wspace=.28, hspace=.28)
    for ext in ('png', 'pdf'):
        fig.savefig(str(outstem) + '.' + ext, dpi=350, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--single-panel", action="store_true", help="Also generate the original all-in-one axes version")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    all_rows = []
    datasets = {}
    for inp in args.json_paths:
        with inp.open(encoding="utf-8") as f:
            data = json.load(f)
        name = re.sub(r"\s*\(\d+\)$", "", inp.stem)
        name = re.sub(r"^results_", "", name, flags=re.I).upper()
        pretty = DATASET_NAMES.get(name, name)
        rows = extract_summary(data, name, inp.name)
        all_rows.extend(rows)
        datasets[pretty] = rows
        print("%s: %s configurations read" % (pretty, len(rows)))

    with (args.output_dir / "r1_vs_macro_f1_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    if len(datasets) > 1:
        outstem = args.output_dir / 'r1_vs_macro_f1_dataset_panels'
        plot_dataset_panels(datasets, outstem)
        print('Recommended figure (five datasets):', str(outstem) + '.png/.pdf')
        if args.single_panel:
            outstem = args.output_dir / 'r1_vs_macro_f1_all_datasets'
            plot_combined(datasets, outstem, group_envelopes=False)
            print('Legacy all-in-one:', str(outstem) + '.png/.pdf')
    print("Summary CSV:", args.output_dir / "r1_vs_macro_f1_summary.csv")


if __name__ == "__main__":
    main()

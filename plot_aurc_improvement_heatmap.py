#!/usr/bin/env python3
"""Unified cross-dataset AURC improvement heatmap for uncertainty ranking.

For every evaluation unit (subject or fold) compute:

    relative_improvement_pct = 100 * (R_all - AURC) / R_all

Here R_all is the full-coverage error rate, the random-ranking reference
for risk-coverage AURC. Positive values indicate better selective ranking.

The figure contains three side-by-side panels (0%, 25%, 100% fine-tuning),
with five dataset columns per panel. The rows combine the HR uncertainty
head (Gaussian, Student-t, MC Dropout) and ranking strategy:
  - Raw HR            {head}
  - Aligned HR        {head}/logistic_hr
  - HAR entropy       {head}/har_entropy
  - Fusion            {head}/fusion

CE is a 0%-only baseline and is included by default (N/A elsewhere).
Use --no-ce to omit it.
An optional extra 'Aligned HAR' ablation is available via --include-har-alignment.

IMPORTANT: Average the within-subject percentage improvements, NOT
100*(mean R_all - mean AURC)/mean R_all (the two are generally different).
Units with R_all == 0 have undefined relative improvement and are reported
as excluded in the CSV (never silently replaced with zero).

Default call from the repository root:
    python plot_aurc_improvement_heatmap.py

Explicit input paths, in any order:
    python plot_aurc_improvement_heatmap.py \
      downstream_results/capture24/results_capture24.json \
      downstream_results/harth/results_harth.json \
      downstream_results/pamap2/results_pamap.json \
      downstream_results/realworld/results_realworld.json \
      downstream_results/usc/results_usc.json \
      --output-dir figures

Notebook:
    %run plot_aurc_improvement_heatmap.py --output-dir figures

Outputs: aurc_relative_improvement_heatmap.png / .pdf
         aurc_relative_improvement_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
import numpy as np


DEFAULT_JSON_PATHS = [
    Path("downstream_results/capture24/results_capture24.json"),
    Path("downstream_results/harth/results_harth.json"),
    Path("downstream_results/pamap2/results_pamap.json"),
    Path("downstream_results/realworld/results_realworld.json"),
    Path("downstream_results/usc/results_usc.json"),
]

DATASET_NAMES = OrderedDict([
    ("capture24", "CAPTURE-24"),
    ("harth", "HARTH"),
    ("pamap", "PAMAP2"),
    ("realworld", "REALWORLD"),
    ("usc", "USC-HAD"),
])

HEADS = OrderedDict([
    ("gaussian", "Gaussian NLL"),
    ("student_t", "Student-t NLL"),
    ("mc_dropout", "MC Dropout"),
])

RANKING_MODES = OrderedDict([
    ("", "Raw HR"),
    ("/logistic_hr", "Aligned HR"),
    ("/har_entropy", "HAR entropy"),
    ("/fusion", "Fusion"),
])

DEPTHS = (0, 25, 100)


def dataset_id_from_path(path: Path) -> str:
    """Identify dataset from path stem, allowing e.g. results_pamap(2).json."""
    stem = re.sub(r"\s*\(\d+\)$", "", path.stem).lower()
    text = (stem + " " + str(path.parent).lower()).replace("-", "_")
    if "capture24" in text or "capture_24" in text:
        return "capture24"
    if "harth" in text:
        return "harth"
    if "pamap" in text:
        return "pamap"
    if "realworld" in text or "real_world" in text:
        return "realworld"
    if "usc" in text:
        return "usc"
    raise ValueError(f"Cannot identify dataset from filename: {path}. Use a results_<dataset>.json name.")


def _lookup_units(classification: dict, depth: int, key: str) -> dict:
    """Accommodate the differing 0% and 25/100% JSON layouts."""
    block = classification.get(f"finetuned_{depth}_results", {})
    if not isinstance(block, dict) or not block:
        return {}
    if depth == 0:
        # 0%: unit_id -> head/strategy -> {AURC, R_all, ...}
        return {
            unit_id: row[key]
            for unit_id, row in block.items()
            if isinstance(row, dict)
            and isinstance(row.get(key), dict)
            and "AURC" in row[key]
            and "R_all" in row[key]
        }
    # 25/100%: head/strategy -> unit_id -> {AURC, R_all, ...}
    method = block.get(key, {})
    if not isinstance(method, dict):
        return {}
    return {
        unit_id: row
        for unit_id, row in method.items()
        if isinstance(row, dict) and "AURC" in row and "R_all" in row
    }


def summarize_cell(units: dict, dataset: str, depth: int, head: str,
                   mode: str, source: Path) -> dict:
    """Aggregate independently computed unit-level relative improvements."""
    improvements, raw_deltas, all_rates, aurcs = [], [], [], []
    zero_ref, invalid = 0, 0
    for unit_id, row in units.items():
        try:
            r_all, aurc = float(row["R_all"]), float(row["AURC"])
        except (ValueError, TypeError, KeyError):
            invalid += 1
            continue
        if not (math.isfinite(r_all) and math.isfinite(aurc)) or r_all < 0:
            invalid += 1
            continue
        if r_all == 0:
            zero_ref += 1
            continue
        if r_all > 1 + 1e-8 or aurc < -1e-8 or aurc > 1 + 1e-8:
            warnings.warn(f"Suspicious risk value: {dataset}, {depth}%, "
                          f"{head}{mode}, unit={unit_id}, R_all={r_all}, AURC={aurc}")
        improvements.append(100.0 * (r_all - aurc) / r_all)
        raw_deltas.append(r_all - aurc)
        all_rates.append(r_all)
        aurcs.append(aurc)

    x = np.asarray(improvements, dtype=float)
    n = int(len(x))
    return {
        "dataset": dataset,
        "fine_tuning_pct": depth,
        "head": head,
        "strategy": mode or "raw_hr",
        "n_units": n,
        "n_excluded_zero_R_all": zero_ref,
        "n_excluded_invalid": invalid,
        "mean_improvement_pct": float(x.mean()) if n else math.nan,
        "sd_improvement_pct": float(x.std(ddof=1)) if n > 1 else math.nan,
        "sem_improvement_pct": float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan,
        "median_improvement_pct": float(np.median(x)) if n else math.nan,
        "fraction_units_improved": float(np.mean(x > 0)) if n else math.nan,
        "mean_absolute_risk_reduction": float(np.mean(raw_deltas)) if n else math.nan,
        "mean_R_all": float(np.mean(all_rates)) if n else math.nan,
        "mean_AURC": float(np.mean(aurcs)) if n else math.nan,
        "source_file": str(source),
    }


def load_results(paths: list[Path], include_ce: bool,
                 include_har_alignment: bool) -> tuple[list[dict], list[tuple[str, str, str]]]:
    heads = OrderedDict(HEADS)
    if include_ce:
        heads["CE"] = "CE (0% only)"
    modes = OrderedDict(RANKING_MODES)
    if include_har_alignment:
        # This is an ablation, frequently redundant with HAR entropy.
        modes["/logistic_har_entropy"] = "Aligned HAR"

    rows = [(head, head_label, suffix, mode_label)
            for head, head_label in heads.items()
            for suffix, mode_label in modes.items()]

    by_dataset = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing input JSON: {path}. Pass explicit paths, "
                "or run from the repository root with the expected directories."
            )
        key = dataset_id_from_path(path)
        if key in by_dataset:
            raise ValueError(f"Duplicate file for dataset {key}: {path}")
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload.get("classification"), dict):
            raise KeyError(f"{path}: expected root 'classification' dictionary")
        by_dataset[key] = (payload["classification"], path)

    records = []
    for dataset_key, dataset_name in DATASET_NAMES.items():
        if dataset_key not in by_dataset:
            continue
        cls, path = by_dataset[dataset_key]
        for depth in DEPTHS:
            for head, _, suffix, _ in rows:
                if head == "CE" and depth != 0:
                    continue
                lookup = head + suffix
                units = _lookup_units(cls, depth, lookup)
                rec = summarize_cell(units, dataset_name, depth, head, suffix, path)
                records.append(rec)
                if not units:
                    warnings.warn(f"No entries: {dataset_name} / {depth}% / {lookup}")

    if not any(r["n_units"] > 0 for r in records):
        raise ValueError("No classification AURC / R_all pairs were found in the JSON files")
    return records, [(head, head_label, suffix, mode_label)
                     for head, head_label, suffix, mode_label in rows]


def nice_vlim(values: np.ndarray) -> int:
    """Symmetric zero-centered full-range scale; does not clip by default."""
    max_abs = float(np.nanmax(np.abs(values)))
    return max(10, int(math.ceil((max_abs + 0.001) / 10.0) * 10))


def make_heatmap(records: list[dict], row_layout: list[tuple], dataset_names: list[str],
                 output_stem: Path, *, annotate: bool = True, vlim: float | None = None,
                 fig_width: float = 13.5, group_gap: float = 0.22,
                 panel_gap: float = 0.24, label_rotation: float = 28):
    """Create readable three-panel figure with real white gaps between HR heads.

    The plotted values are unchanged by spacing. All five dataset names are
    written in full, with angled x tick labels. Head-group spacing is measured
    as a fraction of one heatmap cell's height.
    """
    n_rows, n_datasets = len(row_layout), len(dataset_names)
    index = {(r["dataset"], r["fine_tuning_pct"], r["head"], r["strategy"]): r
             for r in records}
    matrices = []
    for depth in DEPTHS:
        data = np.full((n_rows, n_datasets), np.nan, dtype=float)
        for iy, (head, _, suffix, _) in enumerate(row_layout):
            for ix, dataset in enumerate(dataset_names):
                r = index.get((dataset, depth, head, suffix or "raw_hr"))
                if r and r["n_units"]:
                    data[iy, ix] = r["mean_improvement_pct"]
        matrices.append(data)

    observed_arrays = [arr[np.isfinite(arr)] for arr in matrices]
    observed = np.concatenate(observed_arrays)
    limit = nice_vlim(observed) if vlim is None else float(vlim)
    if not (limit > 0 and math.isfinite(limit)):
        raise ValueError("--vlim must be a positive finite number")
    clipping = np.any(np.abs(observed) > limit)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "font.size": 9,
        "axes.titlesize": 11,
        "savefig.dpi": 360,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    cmap = mpl.colormaps["RdBu"].copy()  # negative red, positive blue
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    # A true blank space is introduced *between* groups (not a dark rule).
    row_centers = []
    group_boundaries = []
    prev_head, group_index = None, -1
    for i, (head, _, _, _) in enumerate(row_layout):
        if head != prev_head:
            group_index += 1
            if i:
                group_boundaries.append(i)
            prev_head = head
        row_centers.append(i + group_gap * group_index)
    last_y = row_centers[-1]

    # Slightly wider panels plus more inter-panel breathing room for full names.
    fig_height = 2.25 + 0.33 * n_rows + 0.11 * len(group_boundaries)
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height),
                             sharey=True, constrained_layout=False)
    titles = ["Frozen encoder (0%)", "Partial fine-tuning (25%)",
              "Full fine-tuning (100%)"]

    # Split the left-side labeling into two visual columns:
    # method family (centered over its 4 variations) and the variation labels.
    mode_ylabels = [mode_label for _, _, _, mode_label in row_layout]
    head_blocks = []
    start = 0
    while start < len(row_layout):
        head, head_label, _, _ = row_layout[start]
        end = start
        while end + 1 < len(row_layout) and row_layout[end + 1][0] == head:
            end += 1
        center_y = float(np.mean(row_centers[start:end + 1]))
        head_blocks.append((head_label, center_y))
        start = end + 1

    for panel_index, (ax, data, title) in enumerate(zip(axes, matrices, titles)):
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=9)
        for i, yc in enumerate(row_centers):
            for j in range(n_datasets):
                value = data[i, j]
                face = cmap(norm(value)) if np.isfinite(value) else "#E9E9E9"
                # Rectangle cells leave background untouched in method gaps.
                ax.add_patch(Rectangle((j - 0.5, yc - 0.5), 1, 1,
                                       facecolor=face, edgecolor="white",
                                       linewidth=0.85))
                if annotate:
                    if np.isfinite(value):
                        label = f"{value:+.0f}"
                        color = "white" if abs(value) >= limit * 0.56 else "#222222"
                    else:
                        label, color = "–", "#888888"
                    ax.text(j, yc, label, ha="center", va="center",
                            fontsize=7.7, color=color)

        ax.set_xlim(-0.5, n_datasets - 0.5)
        ax.set_ylim(last_y + 0.5, -0.5)
        ax.set_xticks(np.arange(n_datasets))
        ax.set_xticklabels(dataset_names, fontsize=8.4, rotation=label_rotation,
                           ha="right", rotation_mode="anchor")
        ax.set_yticks(row_centers)
        if panel_index == 0:
            ax.set_yticklabels(mode_ylabels, fontsize=8.8)
            # Method-family labels centered relative to their 4 variations.
            for head_label, center_y in head_blocks:
                ax.text(-0.35, center_y, head_label,
                        transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=9.6)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.tick_params(axis="y", length=0, pad=2, labelleft=(panel_index == 0))
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.285, right=0.987, top=0.927,
                        bottom=0.21, wspace=panel_gap)
    colorbar_ax = fig.add_axes((0.32, 0.079, 0.56, 0.022))
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cb = fig.colorbar(scalar, cax=colorbar_ax, orientation="horizontal")
    cb.set_label("Mean relative AURC improvement vs. $R_{\\mathrm{all}}$ (%)  ·  higher is better",
                 fontsize=9, labelpad=4)
    cb.ax.tick_params(labelsize=8, length=3)
    cb.outline.set_linewidth(0.5)
    if clipping:
        fig.text(.5, .014, f"Cell colors clipped to ±{limit:g}%; printed numbers are not clipped.",
                 fontsize=7, ha="center", color="#555555")

    for ext in ("pdf", "png"):
        fig.savefig(output_stem.with_suffix("." + ext), dpi=360,
                    facecolor="white", bbox_inches="tight", pad_inches=0.06, transparent=True)
    plt.close(fig)
    return [str(output_stem.with_suffix(".pdf")), str(output_stem.with_suffix(".png"))]


def main(argv=None):
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_paths", nargs="*", type=Path,
                        help="Optional: five dataset result JSON files; uses default paths when omitted.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    ce = parser.add_mutually_exclusive_group()
    ce.add_argument("--include-ce", dest="include_ce", action="store_true",
                    help="Include CE in 0%% panel (default; kept for compatibility).")
    ce.add_argument("--no-ce", dest="include_ce", action="store_false",
                    help="Exclude the 0%% CE baseline.")
    parser.set_defaults(include_ce=True)
    parser.add_argument("--include-har-alignment", action="store_true",
                        help="Add /logistic_har_entropy ablation as a fifth strategy.")
    parser.add_argument("--no-annotations", action="store_true",
                        help="Disable numeric labels inside heatmap cells.")
    parser.add_argument("--vlim", type=float, default=None,
                        help="Optional symmetric colorbar limit in percentage points; "
                             "default covers all displayed means without clipping.")
    parser.add_argument("--fig-width", type=float, default=13.5,
                        help="Figure width in inches (default 13.5); output PDF is vector.")
    parser.add_argument("--group-gap", type=float, default=0.22,
                        help="Blank vertical spacing between HR heads, in row-height units.")
    parser.add_argument("--panel-gap", type=float, default=0.24,
                        help="Horizontal whitespace between fine-tuning panels.")
    parser.add_argument("--label-rotation", type=float, default=28,
                        help="Dataset name rotation in degrees (default 28).")
    args = parser.parse_args(argv)

    # When no JSON paths are supplied, load the five repository defaults.
    # Any explicitly supplied paths override this list.
    if not args.json_paths:
        args.json_paths = DEFAULT_JSON_PATHS.copy()

    if len(args.json_paths) != 5:
        parser.error(f"Expected exactly five JSON inputs, got {len(args.json_paths)}")
    if args.fig_width < 8:
        parser.error("--fig-width should be >= 8 inches for readable labels")
    if not (0 <= args.group_gap <= 1.0):
        parser.error("--group-gap should be between 0 and 1")
    if not (0.05 <= args.panel_gap <= 1.0):
        parser.error("--panel-gap should be between 0.05 and 1")

    records, row_layout = load_results(args.json_paths, args.include_ce,
                                       args.include_har_alignment)
    dataset_names = list(DATASET_NAMES.values())
    missing_dataset_names = set(dataset_names) - {r["dataset"] for r in records}
    if missing_dataset_names:
        raise ValueError(f"Five distinct datasets required. Missing: {sorted(missing_dataset_names)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "aurc_relative_improvement_heatmap"
    figs = make_heatmap(records, row_layout, dataset_names, stem,
                        annotate=not args.no_annotations, vlim=args.vlim,
                        fig_width=args.fig_width, group_gap=args.group_gap,
                        panel_gap=args.panel_gap, label_rotation=args.label_rotation)

    csv_path = args.output_dir / "aurc_relative_improvement_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    for d in dataset_names:
        r = [v for v in records if v["dataset"] == d]
        print(f"{d}: {sum(v['n_units'] > 0 for v in r)} cells with data; "
              f"{sum(v['n_excluded_zero_R_all'] for v in r)} zero-reference units excluded")
    print("Created:", *figs, str(csv_path), sep="\n  ")


if __name__ == "__main__":
    main()

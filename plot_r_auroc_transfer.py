#!/usr/bin/env python3
"""Cross-task uncertainty ranking: retrieval vs classification R-AUROC.

Each point represents one (dataset, method, fine-tuning-depth) combination:
  x = unweighted mean per-subject retrieval R-AUROC(error, uncertainty)
  y = unweighted mean per-evaluation-unit classification R-AUROC(error, uncertainty)
For CAPTURE-24, the classification units are folds; for the other datasets they
are subjects. The retrieval units are subjects for all five datasets.

CAPTURE-24 stores `c_auroc` for correctness, so classification R-AUROC is
1 - c_auroc. Other datasets store error-detection `R-AUROC` directly.

0% CE is a separate uncertainty method, unlike an R@1/F1 plot (for which all
0% objectives share the same embeddings and classification predictions).

Usage:
  python plot_r_auroc_transfer.py \
    'results_capture24(1).json' 'results_harth(1).json' \
    'results_pamap(1).json' 'results_realworld(1).json' \
    'results_usc(1).json' --output-dir figures

Outputs:
  r_auroc_transfer_dataset_panels.png / .pdf
  r_auroc_transfer_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

DATASET_NAMES = {
    'CAPTURE24': 'CAPTURE-24',
    'HARTH': 'HARTH',
    'PAMAP': 'PAMAP2',
    'REALWORLD': 'REALWORLD',
    'USC': 'USC-HAD',
}
METHOD_STYLES = {
    'CE':         {'name': 'CE (0% only)',  'color': '#686868', 'marker': '*'},
    'gaussian':   {'name': 'Gaussian NLL', 'color': '#D65A09', 'marker': 'p'},
    'student_t':  {'name': 'Student-t NLL', 'color': '#7955AD', 'marker': 'h'},
    'mc_dropout': {'name': 'MC Dropout',   'color': '#009D91', 'marker': 'D'},
}
AREA = {0: 95, 25: 90, 100: 158}


def _summary(vals, what):
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        raise ValueError(f'No finite values for {what}')
    return float(np.mean(arr)), (float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0), len(arr)


def classification_r_auroc(entry: dict, where: str) -> float:
    """R-AUROC evaluates uncertainty against *error* rather than correctness."""
    if entry.get('R-AUROC') is not None:
        return float(entry['R-AUROC'])
    if entry.get('c_auroc') is not None:
        return 1.0 - float(entry['c_auroc'])
    if entry.get('C-AUROC') is not None:
        return 1.0 - float(entry['C-AUROC'])
    raise KeyError(f'Missing R-AUROC / c_auroc in {where}')


def extract_points(data: dict, name: str, source: str):
    out = []
    for depth in (0, 25, 100):
        key = f'finetuned_{depth}_results'
        if key not in data['zero_shot'] or key not in data['classification']:
            continue
        retrieval = data['zero_shot'][key]
        classification = data['classification'][key]
        methods = ('CE', 'gaussian', 'student_t', 'mc_dropout') if depth == 0 else (
            'gaussian', 'student_t', 'mc_dropout')
        for method in methods:
            if method not in retrieval:
                continue
            by_subject = retrieval[method].get('by_subject')
            if not isinstance(by_subject, dict):
                raise ValueError(f'{name}: {depth}% {method} missing retrieval by_subject')
            x = [float(s['R-AUROC']) for s in by_subject.values() if s.get('R-AUROC') is not None]
            if depth == 0:
                # 0% layout: evaluation-id -> method -> metrics
                cls_by_unit = {id_: m[method] for id_, m in classification.items()
                               if method in m}
            else:
                # 25/100% layout: method -> evaluation-id -> metrics
                cls_by_unit = classification.get(method, {})
            if not cls_by_unit:
                raise ValueError(f'{name}: {depth}% {method} missing classification entries')
            y = [classification_r_auroc(entry, f'{name}/{depth}/{method}/{id_}')
                 for id_, entry in cls_by_unit.items()]
            xm, xs, nx = _summary(x, f'{name}/{depth}/{method}/retrieval')
            ym, ys, ny = _summary(y, f'{name}/{depth}/{method}/classification')
            if nx > 0 and ny > 0:
                cls_unit = 'subject' if set(by_subject) == set(cls_by_unit) else 'evaluation fold/split'
                out.append({
                    'dataset': name, 'method': method,
                    'fine_tuning_pct': depth,
                    'retrieval_r_auroc_mean': xm,
                    'retrieval_r_auroc_sd': xs,
                    'retrieval_n_subjects': nx,
                    'classification_r_auroc_mean': ym,
                    'classification_r_auroc_sd': ys,
                    'classification_n_units': ny,
                    'classification_unit': cls_unit,
                    'source_file': source,
                })
    return out


def plot_panels(by_dataset, outstem, limits=(0.20, 0.80)):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 10.5,
        'axes.linewidth': 0.95,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'savefig.dpi': 350,
    })
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.4), sharex=True, sharey=True)
    axes_flat = list(axes.flat)
    for idx, (dataset, points) in enumerate(by_dataset.items()):
        ax = axes_flat[idx]
        ax.set_axisbelow(True)
        ax.grid(which='major', color='#DEDEDE', linewidth=0.65, alpha=0.9)
        ax.axvline(0.5, color='#8E8E8E', linestyle=(0, (3, 3)), linewidth=0.9, alpha=0.72, zorder=1)
        ax.axhline(0.5, color='#8E8E8E', linestyle=(0, (3, 3)), linewidth=0.9, alpha=0.72, zorder=1)
        # Large filled points first; smaller points and hollow 0% on top.
        for ft in (100, 25, 0):
            for pt in points:
                if pt['fine_tuning_pct'] != ft:
                    continue
                style = METHOD_STYLES[pt['method']]
                x, y = pt['retrieval_r_auroc_mean'], pt['classification_r_auroc_mean']
                if ft == 0 and pt['method'] != 'CE':
                    # Hollow marker distinguishes the frozen uncertainty head.
                    ax.scatter(x, y, s=AREA[ft], marker=style['marker'],
                               facecolor='white', edgecolor=style['color'], linewidth=2.0,
                               zorder=7)
                else:
                    ax.scatter(x, y, s=AREA[ft], marker=style['marker'],
                               facecolor=style['color'], edgecolor='white', linewidth=0.85,
                               zorder=7 if ft == 0 else (6 if ft == 25 else 5))
        ax.text(0.035, 0.96, dataset, ha='left', va='top', transform=ax.transAxes,
                fontsize=12, weight='bold',
                bbox=dict(facecolor='white', edgecolor='none', pad=1.3, alpha=0.82),
                zorder=10)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # Force all tick numbers despite the axes being shared.
        ax.tick_params(axis='both', labelbottom=True, labelleft=True,
                       direction='out', pad=4, length=3.8)
        ax.set_xlabel('Retrieval R-AUROC', labelpad=6)
        ax.set_ylabel('Classification R-AUROC', labelpad=7)

    legax = axes_flat[-1]
    legax.axis('off')
    meth_handles = [Line2D([], [], linestyle='None', marker=style['marker'],
                           markersize=10, markerfacecolor=style['color'],
                           markeredgecolor='white', markeredgewidth=0.8,
                           color='none', label=style['name'])
                    for style in METHOD_STYLES.values()]
    legend = legax.legend(handles=meth_handles, loc='upper left', title='Uncertainty method',
                          title_fontsize=12, fontsize=11, frameon=False,
                          bbox_to_anchor=(0.03, 1.0), borderaxespad=0., labelspacing=0.85)
    legax.add_artist(legend)
    # A separate visual key for fine-tuning markers below method key.
    depth_handles = [
        Line2D([], [], color='none', linestyle='None', marker='o', markersize=8,
               markerfacecolor='white', markeredgecolor='#555555', markeredgewidth=1.75,
               label='0% (hollow; CE = star)'),
        Line2D([], [], color='none', linestyle='None', marker='o', markersize=7,
               markerfacecolor='#555555', markeredgecolor='white',
               label='25% (small filled)'),
        Line2D([], [], color='none', linestyle='None', marker='o', markersize=11,
               markerfacecolor='#555555', markeredgecolor='white',
               label='100% (large filled)'),
    ]
    legax.legend(handles=depth_handles, title='HR fine-tuning depth',
                 title_fontsize=11, fontsize=10, frameon=False,
                 loc='upper left', bbox_to_anchor=(0.03, 0.49), borderaxespad=0.)
    legax.text(.04, .19, '0.50 = random ranking\n'
                          'Below 0.50: inverse direction\n'
                          'Above 0.50: conventional direction',
               fontsize=9, va='top', ha='left', transform=legax.transAxes,
               linespacing=1.5, color='#4C4C4C')

    fig.subplots_adjust(left=.085, right=.985, top=.96, bottom=.08,
                        wspace=.34, hspace=.30)
    for ext in ('png', 'pdf'):
        fig.savefig(str(outstem) + '.' + ext, dpi=350, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('json_paths', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('figures'))
    parser.add_argument('--min-axis', type=float, default=0.20)
    parser.add_argument('--max-axis', type=float, default=0.80)
    args = parser.parse_args()
    if not (0 <= args.min_axis < 0.5 < args.max_axis <= 1):
        raise ValueError('Axis range must include 0.5 and be within [0,1]')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_points, datasets = [], {}
    for path in args.json_paths:
        stem = re.sub(r'\s*\(\d+\)$', '', path.stem)
        dataset_key = re.sub(r'^results_', '', stem, flags=re.I).upper()
        name = DATASET_NAMES.get(dataset_key, dataset_key)
        with path.open(encoding='utf-8') as stream:
            data = json.load(stream)
        points = extract_points(data, name, path.name)
        datasets[name] = points
        all_points.extend(points)
        print(f'{name}: {len(points)} mean points, '
              f'R-AUROC retrieval={min(p["retrieval_r_auroc_mean"] for p in points):.3f}..'
              f'{max(p["retrieval_r_auroc_mean"] for p in points):.3f}, '
              f'classification={min(p["classification_r_auroc_mean"] for p in points):.3f}..'
              f'{max(p["classification_r_auroc_mean"] for p in points):.3f}')
    if not all_points:
        raise ValueError('No results found')
    lo, hi = args.min_axis, args.max_axis
    outside = [p for p in all_points if not (lo <= p['retrieval_r_auroc_mean'] <= hi and
                                             lo <= p['classification_r_auroc_mean'] <= hi)]
    if outside:
        raise ValueError(f'{len(outside)} means outside limits [{lo}, {hi}]. Widen limits to show every point.')
    output = args.output_dir / 'r_auroc_transfer_dataset_panels'
    plot_panels(datasets, output, limits=(lo, hi))
    output_csv = args.output_dir / 'r_auroc_transfer_summary.csv'
    with output_csv.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(all_points[0]))
        writer.writeheader()
        writer.writerows(all_points)
    print('Saved:', output.with_suffix('.png'), 'and', output.with_suffix('.pdf'))
    print('Saved:', output_csv)


if __name__ == '__main__':
    main()

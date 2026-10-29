"""Paper's TWO retained zero-shot retrieval figures + their paired t-tests.

Inputs: existing downstream_experiments.py result JSON files; no model inference.

    python plot_retrieval_paper_figures.py --results \
      capture24=results_capture24.json pamap=results_pamap.json \
      realworld=results_realworld.json usc=results_usc.json \
      harth=results_harth.json --outdir paper_figures

Artifacts: exactly two PNG figures and four CSV tables:
  * embedding_quality_r1.png
  * retrieval_error_r_auroc.png
  * embedding_quality_r1_summary.csv
  * embedding_quality_r1_paired_ttests.csv
  * retrieval_error_r_auroc_summary.csv
  * retrieval_error_r_auroc_paired_ttests.csv

All means and SDs are computed across subjects (never pooled windows).
R@1 refers ONLY to embedding nearest-neighbor quality. R-AUROC is an
error-positive ROC AUC: AUROC(1[nearest neighbor is wrong], HR uncertainty).
"""

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

REGIMES = (0, 25, 100)
REGIME_KEYS = {i: 'finetuned_{}_results'.format(i) for i in REGIMES}
METHODS = ('CE', 'gaussian', 'student_t', 'mc_dropout')
METHOD_DISPLAY = {'CE': 'CE', 'gaussian': 'Gaussian',
                  'student_t': 'Student-t', 'mc_dropout': 'MC Dropout',
                  'original': 'Original (0%)'}
DATASET_DISPLAY = {'capture24': 'CAPTURE-24', 'pamap': 'PAMAP2',
                   'pamap2': 'PAMAP2', 'realworld': 'RealWorld',
                   'usc': 'USC-HAD', 'usc-had': 'USC-HAD', 'harth': 'HARTH'}
COLORS = {'original': '#64748b', 'gaussian': '#2da05a',
          'student_t': '#ed8d28', 'mc_dropout': '#b06aba'}


def _display_name(name):
    return DATASET_DISPLAY.get(str(name).lower(), str(name))


def _canonical_method(name):
    name = str(name)
    aliases = {'u_ce': 'CE', 'Gaussian': 'gaussian', 'Student-t': 'student_t',
               'student-t': 'student_t', 'MC Dropout': 'mc_dropout'}
    return aliases.get(name, name)


def collect_subject_metrics(datasets):
    """Collect independent-user observations for the two allowed metrics.

    datasets: dict dataset_name -> previously loaded results JSON dictionary.
    For 0% the shared embedding is entered only once as method='original'
    (NOT duplicated under four uncertainty estimators).
    """
    rows = []
    for dataset, result in datasets.items():
        zero = result.get('zero_shot', {})
        for ft in REGIMES:
            methods = zero.get(REGIME_KEYS[ft], {})
            if not isinstance(methods, dict):
                continue
            seen_baseline = {}
            for original_method, stats in methods.items():
                method = _canonical_method(original_method)
                if method not in METHODS or (ft > 0 and method == 'CE'):
                    continue
                for subject, values in stats.get('by_subject', {}).items():
                    if not isinstance(values, dict):
                        continue
                    for metric in ('R@1', 'R-AUROC'):
                        val = values.get(metric)
                        if val is None:
                            continue
                        try:
                            val = float(val)
                        except (TypeError, ValueError):
                            continue
                        if not np.isfinite(val):
                            continue
                        if metric == 'R@1' and ft == 0:
                            # All 0% heads use one *identical* encoder.
                            # A discrepancy signals a malformed JSON/mismatched run.
                            key = str(subject)
                            if key in seen_baseline:
                                if not np.isclose(val, seen_baseline[key], atol=1e-7):
                                    raise ValueError('Non-identical 0% R@1 embeddings for '
                                                     '{} subject {}'.format(dataset, subject))
                                continue
                            seen_baseline[key] = val
                            use_method = 'original'
                        else:
                            use_method = method
                        if metric == 'R@1' and ft != 0 and method == 'CE':
                            continue
                        rows.append(dict(dataset=str(dataset), subject=str(subject),
                                         finetuning=ft, method=use_method,
                                         metric=metric, value=val))
    df = pd.DataFrame(rows, columns=['dataset', 'subject', 'finetuning',
                                     'method', 'metric', 'value'])
    if not df.empty:
        duplicates = df.duplicated(['dataset', 'subject', 'finetuning',
                                    'method', 'metric'])
        if duplicates.any():
            raise ValueError('Repeated measurements: ' + str(df.loc[duplicates].head()))
    return df


def summarize_metrics(long_df, metric):
    columns = ['dataset', 'finetuning', 'method', 'mean', 'sd', 'n_subjects']
    sub = long_df[long_df.metric == metric].copy()
    if metric == 'R@1':
        sub = sub[((sub.finetuning == 0) & (sub.method == 'original')) |
                  ((sub.finetuning > 0) & (sub.method != 'original'))]
    if sub.empty:
        return pd.DataFrame(columns=columns)
    df = sub.groupby(['dataset', 'finetuning', 'method'], sort=False).value.agg(
        mean='mean', sd=lambda x: x.std(ddof=1), n_subjects='count'
    ).reset_index()
    return df[columns]


def _paired_ttests_for_metric(long_df, metric):
    """Predefined subject-paired contrasts and Holm correction per dataset/metric.

    R@1: at 25/100%, original vs each finetuned and finetuned vs
    finetuned; and 25 vs 100 within method.
    R-AUROC: between methods within each regime, then within-method
    between regimes. CE is available at 0% only.
    """
    sub = long_df[long_df.metric == metric]
    columns = ['dataset', 'metric', 'comparison', 'condition_a', 'condition_b',
               'n_pairs', 'mean_a', 'mean_b', 'mean_a_minus_b',
               't_statistic', 'p', 'p_holm']
    rows = []
    for dataset, data in sub.groupby('dataset', sort=False):
        wide = data.pivot(index='subject', columns=['finetuning', 'method'],
                          values='value')
        comparisons = []
        if metric == 'R@1':
            reference = (0, 'original')
            for ft in (25, 100):
                available = [(ft, m) for m in ('gaussian', 'student_t', 'mc_dropout')
                             if (ft, m) in wide.columns]
                if reference in wide.columns:
                    comparisons += [('Original vs finetuned', reference, other)
                                    for other in available]
                comparisons += [('Within-regime methods', a, b)
                                for a, b in combinations(available, 2)]
            comparisons += [('25% vs 100%', (25, method), (100, method))
                            for method in ('gaussian', 'student_t', 'mc_dropout')
                            if (25, method) in wide.columns and
                               (100, method) in wide.columns]
        else:
            for ft in REGIMES:
                available = [(ft, m) for m in METHODS
                             if (ft, m) in wide.columns]
                comparisons += [('Within-regime methods', a, b)
                                for a, b in combinations(available, 2)]
            for method in ('gaussian', 'student_t', 'mc_dropout'):
                available = [(ft, method) for ft in REGIMES
                             if (ft, method) in wide.columns]
                comparisons += [('Within-method fine-tuning', a, b)
                                for a, b in combinations(available, 2)]
        for group, a, b in comparisons:
            paired = wide.loc[:, [a, b]].dropna()
            n = len(paired)
            x = paired[a].to_numpy(dtype=float)
            y = paired[b].to_numpy(dtype=float)
            if n < 3:
                stat, p = np.nan, np.nan
            elif np.allclose(x, y, atol=1e-14, rtol=0):
                stat, p = 0., 1.
            else:
                # Degenerate but nonzero identical differences have infinite t.
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = ttest_rel(x, y)
                stat, p = float(result.statistic), float(result.pvalue)
            def fmt(c):
                ft, m = c
                return '{}% / {}'.format(ft, METHOD_DISPLAY.get(m, m))
            rows.append(dict(dataset=dataset, metric=metric, comparison=group,
                             condition_a=fmt(a), condition_b=fmt(b),
                             n_pairs=n, mean_a=float(x.mean()) if n else np.nan,
                             mean_b=float(y.mean()) if n else np.nan,
                             mean_a_minus_b=float((x - y).mean()) if n else np.nan,
                             t_statistic=stat, p=p))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=columns)
    df['p_holm'] = np.nan
    for dataset, indices in df.groupby('dataset').groups.items():
        valid = [i for i in indices if np.isfinite(df.loc[i, 'p'])]
        valid = sorted(valid, key=lambda i: df.loc[i, 'p'])
        if valid:
            m = len(valid)
            raw = np.array([df.loc[i, 'p'] for i in valid], dtype=float)
            adjusted = np.maximum.accumulate(np.minimum(1., raw *
                                                      np.arange(m, 0, -1)))
            df.loc[valid, 'p_holm'] = adjusted
    return df[columns]


def _create_embedding_figure(summary, datasets, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if summary.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(max(13, 2.25 * len(datasets) + 2.5), 5.6),
                             sharey=True)
    methods = ('original', 'gaussian', 'student_t', 'mc_dropout')
    offsets = np.arange(len(methods)) - (len(methods)-1)/2
    # Use the readable, zoomed vertical range in the reference figure.
    # Include every available mean +/- SD before adding padding.
    finite_rows = summary[np.isfinite(summary['mean'])].copy()
    spread = finite_rows['sd'].fillna(0.0).to_numpy(dtype=float)
    lower = np.maximum(0.0, finite_rows['mean'].to_numpy(dtype=float) - spread)
    upper = np.minimum(1.0, finite_rows['mean'].to_numpy(dtype=float) + spread)
    y_min = max(0.0, float(lower.min()) - 0.055)
    y_max = min(1.0, float(upper.max()) + 0.05)
    if y_max - y_min < 0.10:
        y_min = max(0.0, y_min - 0.04)
        y_max = min(1.0, y_max + 0.04)
    for ax, ft in zip(axes, (25, 100)):
        x = np.arange(len(datasets), dtype=float)
        width = 0.19
        for j, method in enumerate(methods):
            level = 0 if method == 'original' else ft
            sub = summary[(summary.finetuning == level) & (summary.method == method)]
            lookup = sub.set_index('dataset') if not sub.empty else pd.DataFrame()
            means = [float(lookup.loc[ds, 'mean']) if ds in lookup.index else np.nan
                     for ds in datasets]
            sds = [float(lookup.loc[ds, 'sd']) if ds in lookup.index else np.nan
                   for ds in datasets]
            pos = x + offsets[j] * width
            ok = np.isfinite(means)
            ax.bar(pos[ok], np.asarray(means)[ok], width=width, color=COLORS[method],
                   label=METHOD_DISPLAY[method], zorder=3)
            std_values = np.asarray(sds, dtype=float)
            stderr = np.where(np.isfinite(std_values), std_values, 0.)
            has_sd = ok & np.isfinite(std_values)
            if np.any(has_sd):
                ax.errorbar(pos[has_sd], np.asarray(means)[has_sd],
                            yerr=stderr[has_sd], fmt='none', ecolor='black',
                            elinewidth=1.2, capsize=0, zorder=4)
        ax.set_title('{}% HR fine-tuning'.format(ft), fontsize=15, pad=13)
        ax.set_xticks(x)
        ax.set_xticklabels([_display_name(ds) for ds in datasets], rotation=27,
                           ha='right')
        ax.set_ylim(y_min, y_max)
        ax.grid(axis='y', alpha=0.25, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('Recall@1 (mean ± subject SD)', fontsize=13)
    fig.legend(handles=[Patch(color=COLORS[m], label=METHOD_DISPLAY[m]) for m in methods],
               loc='upper center', ncol=4, frameon=False, fontsize=11)
    fig.subplots_adjust(top=0.83, bottom=0.22, wspace=0.1, left=0.07, right=0.98)
    path = Path(out_dir) / 'embedding_quality_r1.png'
    fig.savefig(str(path), dpi=230, bbox_inches='tight')
    plt.close(fig)
    return path


def _create_auroc_figure(summary, datasets, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    if summary.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12.6, max(4.3, 0.62 * len(datasets) + 1.9)),
                             sharey=True)
    norm = colors.TwoSlopeNorm(vmin=0.2, vcenter=0.5, vmax=0.8)
    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('#eeeeee')
    for ax, ft in zip(axes, REGIMES):
        mat = np.full((len(datasets), len(METHODS)), np.nan)
        sd = np.full_like(mat, np.nan)
        for i, ds in enumerate(datasets):
            for j, method in enumerate(METHODS):
                sub = summary[(summary.dataset == ds) &
                              (summary.finetuning == ft) &
                              (summary.method == method)]
                if not sub.empty:
                    mat[i, j] = sub.iloc[0]['mean']
                    sd[i, j] = sub.iloc[0]['sd']
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cm, norm=norm,
                       aspect='auto', interpolation='nearest')
        ax.set_title('{}% fine-tuning'.format(ft), fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([METHOD_DISPLAY[m] for m in METHODS],
                           rotation=38, ha='right', fontsize=9)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels([_display_name(ds) for ds in datasets], fontsize=10)
        ax.tick_params(length=0)
        for i in range(len(datasets)):
            for j in range(len(METHODS)):
                if np.isfinite(mat[i, j]):
                    value = mat[i, j]
                    label = '{:.3f}'.format(value)
                    if np.isfinite(sd[i, j]):
                        label += '\nSD {:.3f}'.format(sd[i, j])
                    ax.text(j, i, label, ha='center', va='center',
                            color='white' if value < 0.36 or value > 0.69 else '#303030',
                            fontsize=7.5, fontweight='bold')
                else:
                    ax.text(j, i, '—', ha='center', va='center', color='#999999')
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.32, top=0.86, wspace=0.13)
    cbax = fig.add_axes([0.27, 0.175, 0.46, 0.028])
    cb = fig.colorbar(im, cax=cbax, orientation='horizontal')
    cb.set_label('Error R-AUROC (0.5 = chance)', fontsize=10)
    fig.text(0.22, 0.077, 'Inverse: higher uncertainty, fewer errors',
             ha='center', fontsize=9)
    fig.text(0.78, 0.077, 'Conventional: higher uncertainty, more errors',
             ha='center', fontsize=9)
    path = Path(out_dir) / 'retrieval_error_r_auroc.png'
    fig.savefig(str(path), dpi=230, bbox_inches='tight')
    plt.close(fig)
    return path


def export_paper_results(datasets, outdir='paper_figures', make_plots=True):
    """Export the TWO requested metrics, their paired t-tests and TWO plots."""
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    long = collect_subject_metrics(datasets)
    if long.empty:
        raise ValueError('No zero_shot.by_subject metrics found. Use results JSON '
                         'containing per-user R@1 and error R-AUROC.')
    result = {}
    for metric, prefix in (('R@1', 'embedding_quality_r1'),
                           ('R-AUROC', 'retrieval_error_r_auroc')):
        summary = summarize_metrics(long, metric)
        tests = _paired_ttests_for_metric(long, metric)
        summary_path = output / (prefix + '_summary.csv')
        tests_path = output / (prefix + '_paired_ttests.csv')
        summary.to_csv(str(summary_path), index=False)
        tests.to_csv(str(tests_path), index=False)
        result[metric] = {'summary': summary_path, 'paired_ttests': tests_path,
                          'summary_df': summary, 'tests_df': tests}
    if make_plots:
        order = list(datasets)
        result['R@1']['figure'] = _create_embedding_figure(
            result['R@1']['summary_df'], order, output)
        result['R-AUROC']['figure'] = _create_auroc_figure(
            result['R-AUROC']['summary_df'], order, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs='+', required=True, metavar='DATASET=JSON',
                        help='e.g. capture24=results_capture24.json pamap=results_pamap.json')
    parser.add_argument('--outdir', default='paper_figures')
    args = parser.parse_args()
    datasets = {}
    for spec in args.results:
        if '=' not in spec:
            parser.error('Expected DATASET=JSON, got {}'.format(spec))
        name, path = spec.split('=', 1)
        if not name or not path or name in datasets:
            parser.error('Invalid/duplicate dataset spec: {}'.format(spec))
        with open(path, 'r', encoding='utf-8') as f:
            datasets[name] = json.load(f)
    output = export_paper_results(datasets, outdir=args.outdir)
    for metric in ('R@1', 'R-AUROC'):
        for key in ('figure', 'summary', 'paired_ttests'):
            if output[metric].get(key) is not None:
                print('{} {}: {}'.format(metric, key, output[metric][key]))


if __name__ == '__main__':
    main()

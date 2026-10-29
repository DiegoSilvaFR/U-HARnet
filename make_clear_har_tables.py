"""

Main tables: 0% fine-tuning only, where all methods share the same encoder.
Main figures: three side-by-side fine-tuning panels; each cell shows
mean R-AUROC and its participant/fold SD underneath. Color encodes
mean R-AUROC relative to the random-ranking reference of 0.5.

Example:
    python make_clear_har_tables.py --input results_capture24.json results_pamap.json results_realworld.json results_usc.json results_harth.json --outdir paper_downstream_results

Outputs: focused_zero_shot_0pct.tex, focused_classification_0pct.tex,
zero_shot_r_auroc_panels.png/.svg, classification_r_auroc_panels.png/.svg
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

DATASET_NAMES = {"capture24": "CAPTURE-24", "pamap": "PAMAP2", "pamap2": "PAMAP2",
                 "realworld": "RealWorld", "usc": "USC-HAD", "harth": "HARTH"}
DATASETS = ("CAPTURE-24", "PAMAP2", "RealWorld", "USC-HAD", "HARTH")
METHODS = ("CE", "gaussian", "student_t", "mc_dropout")
COL_NAMES = ("CE", "Gaussian", "Student-t", "MC Dropout")
FTS = (0, 25, 100)


def load_results(paths):
    result = {}
    for p in paths:
        name = Path(p).stem
        key = name[len('results_'):] if name.startswith('results_') else name
        display = DATASET_NAMES.get(key.lower(), key)
        with open(p, encoding='utf-8') as f:
            result[display] = json.load(f)
    return result


def get_method_folds(obj):
    if not obj: return {}
    if all(str(k).isdigit() for k in obj):
        ret = {}
        for uid, vals in obj.items():
            for m, metrics in vals.items():
                ret.setdefault(m, {})[str(uid)] = metrics
        return ret
    return obj


def values_for(r, section, ft, method, metric='R-AUROC'):
    obj = r.get(section, {}).get('finetuned_%d_results' % ft, {})
    if section == 'zero_shot':
        vals = obj.get(method, {}).get('by_subject', {}).values()
    else:
        vals = get_method_folds(obj).get(method, {}).values()
    result = []
    for entry in vals:
        if isinstance(entry, dict) and 'u' in entry and set(entry) == {'u'}:
            entry = entry['u']
        v = entry.get(metric)
        if v is None:
            v = entry.get(metric.lower().replace('-', '_'))
        if v is None and metric == 'R-AUROC':
            v = entry.get('r_auroc')
        if v is None and metric == 'R-AUROC' and entry.get('c_auroc') is not None:
            v = 1. - float(entry['c_auroc'])
        if v is not None and np.isfinite(float(v)):
            result.append(float(v))
    return np.asarray(result, float)


def summaries(results, section):
    rows = []
    for name in DATASETS:
        if name not in results: continue
        for ft in FTS:
            for method in METHODS:
                if method == 'CE' and ft != 0: continue
                values = values_for(results[name], section, ft, method)
                if values.size:
                    rows.append({'dataset':name, 'ft':ft, 'method':method,
                                 'mean':values.mean(),
                                 'sd':values.std(ddof=1) if len(values)>1 else np.nan,
                                 'n':len(values)})
    return rows


def latex_table_0pct(rows, section, path):
    # No forced bold face, because AUROC < 0.5 can be more discriminatory
    # than AUROC > 0.5 in the opposite direction.
    lookup = {(r['dataset'],r['ft'],r['method']):r for r in rows}
    label = 'tab:har_zero_shot_0' if section == 'zero_shot' else 'tab:har_linear_probe_0'
    study = ('zero-shot nearest-neighbor retrieval' if section == 'zero_shot'
             else 'downstream linear-probe classification')
    note = ('per participant' if section == 'zero_shot'
            else 'per held-out unit (CAPTURE-24: 15 ten-subject test folds; other datasets: LOSO subjects)')
    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{R-AUROC for %s with the frozen encoder (0\%% fine-tuning). ' % study +
             r'Each entry is the mean $\pm$ standard deviation across evaluation units.}',
             r'\label{%s}' % label,
             r'\small', r'\setlength{\tabcolsep}{9pt}',
             r'\renewcommand{\arraystretch}{1.17}',
             r'\begin{tabular}{@{}lcccc@{}}',
             r'\toprule',
             r'Dataset & CE & Gaussian & Student-$t$ & MC Dropout \\',
             r'\midrule']
    for name in DATASETS:
        if name not in {r['dataset'] for r in rows}: continue
        cells = []
        for method in METHODS:
            r = lookup.get((name,0,method))
            cells.append((r'$%.3f \pm %.3f$' % (r['mean'],r['sd'])) if r and np.isfinite(r['sd']) else r'---')
        lines.append(name + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',
              r'\par\vspace{3pt}',
              r'\begin{minipage}{0.94\textwidth}',
              r'\footnotesize\emph{Note.} R-AUROC $>0.5$ denotes conventional ' +
              r'error ranking; R-AUROC $<0.5$ denotes inverse ranking. Values are averaged %s. ' % note +
              r'These directions are descriptive, not significance claims.',
              r'\end{minipage}',r'\end{table*}', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def figure_panels(rows, section, filename):
    """Plot R-AUROC mean with subject/fold standard deviation just below it."""
    lookup = {(r['dataset'], r['ft'], r['method']): r for r in rows}
    datasets = [d for d in DATASETS if d in {r['dataset'] for r in rows}]
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#f1f1f1')
    norm = TwoSlopeNorm(vmin=0.2, vcenter=0.5, vmax=0.8)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 6.7),
                             gridspec_kw={'wspace': 0.21})
    for ax, ft in zip(axes, FTS):
        matrix = np.full((len(datasets), len(METHODS)), np.nan)
        for i, dataset in enumerate(datasets):
            for j, method in enumerate(METHODS):
                r = lookup.get((dataset, ft, method))
                if r is not None:
                    matrix[i, j] = r['mean']

        ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(np.arange(len(METHODS)))
        ax.set_xticklabels(COL_NAMES, rotation=25, ha='right', fontsize=10)
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels(datasets if ft == 0 else [''] * len(datasets),
                           fontsize=11)
        ax.set_title(f'{ft}% fine-tuning', fontsize=13, fontweight='bold', pad=12)

        for i, dataset in enumerate(datasets):
            for j, method in enumerate(METHODS):
                r = lookup.get((dataset, ft, method))
                if r is None:
                    ax.text(j, i, '—', ha='center', va='center',
                            fontsize=13, color='#787878')
                    continue
                mean, sd = r['mean'], r['sd']
                ink = 'white' if (mean < 0.29 or mean > 0.71) else '#17212c'
                # Keep SD visually subordinate to the main metric.
                ax.text(j, i - 0.13, f'{mean:.3f}', ha='center', va='center',
                        fontsize=11.0, fontweight='semibold', color=ink)
                ax.text(j, i + 0.19,
                        f'SD {sd:.3f}' if np.isfinite(sd) else 'SD —',
                        ha='center', va='center', fontsize=8.9, color=ink)

        ax.set_xticks(np.arange(-0.5, len(METHODS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(datasets), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=3)
        ax.tick_params(axis='both', which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    kind = ('Zero-shot retrieval' if section == 'zero_shot'
            else 'Linear-probe classification')
 #   fig.suptitle(kind + ': R-AUROC by HR fine-tuning level', fontsize=16, y=.985)
    fig.subplots_adjust(left=.11, right=.99, top=.83, bottom=.31)
    cax = fig.add_axes([.30, .155, .40, .021])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks([.2, .35, .5, .65, .8])
    cb.ax.tick_params(labelsize=10)
    fig.text(.25, .080, 'Inverse: higher uncertainty, fewer errors',
             ha='center', fontsize=18)
    fig.text(.75, .080, 'Conventional: higher uncertainty, more errors',
             ha='center', fontsize=18)
    unit = ('participants' if section == 'zero_shot' else
            'held-out folds for CAPTURE-24; LOSO participants for other datasets')
    #fig.text(.5, .029,
       #      f'Top: mean R-AUROC; bottom: SD across {unit}.\n'
      #       'Colors represent mean direction relative to 0.5, not statistical significance.',
     #        ha='center', fontsize=9)
    fig.savefig(filename, dpi=240, bbox_inches='tight', facecolor='white',transparent=True)
   
    plt.close(fig)



# -----------------------------------------------------------------------------
# Embedding quality: R@1 at 25% and 100% fine-tuning
# -----------------------------------------------------------------------------
def plot_embedding_quality_r1(results, save_path,
                              methods_order=("gaussian", "student_t", "mc_dropout"),
                              figsize=(14, 5), dpi=300):
    """Plot mean R@1 with black vertical SD whiskers for all four bars.

    Each dataset has FOUR bars in both panels (25% and 100% fine-tuning):
        Original embedding (0%), Gaussian, Student-t, MC Dropout.

    The original embedding is plotted once per dataset *per panel*, for easy
    comparison. Its statistics come from the same 0% frozen encoder; values
    from different 0% uncertainty heads are not treated as replicates.

    SD is the sample SD across entries in the per-subject/fold `by_subject`
    field. An aggregate R@1 alone is not sufficient to calculate SD: when
    no subject/fold data exist, that bar has no whisker.
    """
    import warnings
    from matplotlib.patches import Patch

    datasets = [name for name in DATASETS if name in results]
    datasets += [name for name in results if name not in datasets]
    ft_levels = (25, 100)
    plot_methods = ("original",) + tuple(m for m in methods_order if m != "original")
    label_map = {"original": "Original (0%)", "gaussian": "Gaussian",
                 "student_t": "Student-$t$", "mc_dropout": "MC Dropout"}
    color_map = {"original": "#64748b", "gaussian": "#32a358",
                 "student_t": "#e68626", "mc_dropout": "#b26cbe"}

    def summarize_r1(data, ft, method):
        """Return (mean, sample SD, count) or None, using by_subject if present."""
        values = values_for(data, "zero_shot", ft, method, metric="R@1")
        if values.size:
            return (float(values.mean()),
                    float(values.std(ddof=1)) if values.size > 1 else np.nan,
                    int(values.size))
        item = (data.get("zero_shot", {})
                    .get("finetuned_%d_results" % ft, {})
                    .get(method, {}))
        pooled = item.get("R@1") if isinstance(item, dict) else None
        if pooled is not None and np.isfinite(float(pooled)):
            return (float(pooled), np.nan, 0)
        return None

    stats = {}
    bounds = []
    for dataset in datasets:
        data = results[dataset]
        # Shared frozen encoder: choose ONE head's 0% metrics, not their mean.
        refs = {m: rec for m in ("gaussian", "student_t", "mc_dropout", "CE")
                if (rec := summarize_r1(data, 0, m)) is not None}
        if refs:
            # Prefer a head for which we have participant/fold-level SD.
            choice = next((m for m in ("gaussian", "student_t", "mc_dropout", "CE")
                           if m in refs and refs[m][2] > 0), next(iter(refs)))
            stats[(dataset, 0, "original")] = refs[choice]
            values = [rec[0] for rec in refs.values()]
            if max(values) - min(values) > 1e-4:
                warnings.warn(
                    "%s: 0%% R@1 differs between heads (%.4f to %.4f). "
                    "Using '%s' for the original embedding; check the JSON."
                    % (dataset, min(values), max(values), choice), stacklevel=2)
        else:
            warnings.warn("%s: no original (0%%) R@1; original bar omitted."
                          % dataset, stacklevel=2)

        for ft in ft_levels:
            for method in methods_order:
                rec = summarize_r1(data, ft, method)
                if rec is not None:
                    stats[(dataset, ft, method)] = rec

    if not any(k[1] in ft_levels for k in stats):
        raise ValueError(
            "No 25%/100% R@1 found under "
            "zero_shot/finetuned_{25,100}_results/<method>/by_subject."
        )

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    positions = np.arange(len(datasets), dtype=float)
    bar_width = 0.19
    offsets = (np.arange(len(plot_methods)) - (len(plot_methods) - 1) / 2) * 0.205

    for ax, ft in zip(axes, ft_levels):
        for j, method in enumerate(plot_methods):
            for i, dataset in enumerate(datasets):
                # The SAME 0% reference appears as a bar in both panels.
                rec = stats.get((dataset, 0, "original") if method == "original"
                                else (dataset, ft, method))
                if rec is None:
                    continue
                mean, sd, _ = rec
                x = positions[i] + offsets[j]
                ax.bar(x, mean, width=bar_width,
                       color=color_map.get(method, "#6e96b8"),
                       edgecolor="none", zorder=3)
                if np.isfinite(sd):
                    ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor="black",
                                elinewidth=1.35, capsize=0, zorder=5)
                bounds.extend([mean - sd, mean + sd] if np.isfinite(sd)
                              else [mean])

        ax.set_xticks(positions)
        ax.set_xticklabels(datasets, rotation=27, ha="right")
        ax.set_title("%d%% HR fine-tuning" % ft, fontsize=13)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#c8c8c8", alpha=0.65, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("R@1", fontsize=15)
    lower, upper = min(bounds), max(bounds)
    padding = max(0.025, (upper - lower) * 0.18)
    bottom, top = max(0.0, lower - padding), min(1.0, upper + padding)
    if top <= bottom:
        top = min(1.0, bottom + 0.1)
    axes[0].set_ylim(bottom, top)

    legend_handles = [
        Patch(facecolor=color_map.get(m, "#6e96b8"), label=label_map.get(m, m))
        for m in plot_methods
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(plot_methods),
               bbox_to_anchor=(0.5, 1.045), frameon=False)
    #fig.text(0.5, 0.012, "All bars = mean R@1; black vertical whiskers = SD across participants/folds",
    #         ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.93], w_pad=2.5)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white", transparent=True)
  #  fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for (dataset, ft, method), (mean, sd, n) in stats.items():
        print("  %-11s | %3d%% | %-12s | R@1 %.4f ± %s (n=%s)" %
              (dataset, ft, method, mean,
               "%.4f" % sd if np.isfinite(sd) else "N/A",
               str(n) if n else "aggregate"))
    print("Saved:", save_path)
    return stats


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', nargs='+', required=True, help='Path(s) to results JSON')
    ap.add_argument('--outdir',default='har_clear_output')
    a=ap.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    results=load_results(a.input)
    for section in ('zero_shot','classification'):
        s=summaries(results,section)
        latex_table_0pct(s,section,out/('focused_%s_0pct.tex'%section))
        figure_panels(s,section,out/('%s_r_auroc_panels.png'%section))
    plot_embedding_quality_r1(results, out/'embedding_quality_r1.png')
    print('Created focused 0% tables and 0/25/100% panels in:',out)


if __name__ == '__main__':
    main()

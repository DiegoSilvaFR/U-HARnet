"""Upstream HR uncertainty: subject-level risk--coverage only.

Evaluate the ORIGINAL held-out upstream HR test subjects at 0%, 25%, and
100% encoder fine-tuning. For each subject and uncertainty head, reject the
windows with the highest HR uncertainty and measure the retained prediction
MSE. The underlying risk calculation is reused unchanged from
benchmark_upstream_heads.risk_coverage_curve_regression.

The plotted curve is the UNWEIGHTED mean of subjects' MSE values at each
coverage level; its shaded band is +/-1 sample SD BETWEEN subjects.
No subject windows are pooled before rejection/ranking. No new training,
validation-based decisions, or downstream experiments are performed.

Requires the existing benchmark_upstream_heads.py and
benchmark_upstream_finetuned.py helper modules and upstream checkpoints.

Example:
    python upstream_hr_risk_coverage.py --regimes 0,25,100
    python upstream_hr_risk_coverage.py --resume
"""
import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from benchmark_upstream_heads import benchmark_heads, risk_coverage_curve_regression
from benchmark_upstream_finetuned import make_upstream_results


HEAD_TYPES = ('gaussian', 'student_t', 'mc_dropout')
OUTPUT_SIZE = {'mc_dropout': 1, 'gaussian': 2, 'student_t': 3}
CHECKPOINTS = {
    25: {
        'gaussian': 'HR_reg_gaussian_oxfordhub_25_final.mdl',
        'student_t': 'HR_reg_student_t_oxfordhub_25_final.mdl',
        'mc_dropout': 'HR_reg_mc_dropout_oxfordhub_ft25_25_final.mdl',
    },
    100: {
        'gaussian': 'HR_reg_gaussian_oxfordhub_100_final.mdl',
        'student_t': 'HR_reg_student_t_oxfordhub_100_final.mdl',
        'mc_dropout': 'HR_reg_mc_dropout_oxfordhub_ft100_100_final.mdl',
    },
}


def normalize_head(value):
    name = str(value).strip().lower()
    return {
        'student-t': 'student_t', 'student t': 'student_t',
        'mc dropout': 'mc_dropout', 'mcdropout': 'mc_dropout',
    }.get(name, name)


def zero_head_frames(df):
    """Yield per-head 0% frames without requesting 1024-D embeddings.

    Supports the original benchmark_heads() long format (one head per row)
    and the wide format with mu_<head> and <head> uncertainty columns.
    """
    method_col = next(
        (c for c in ('head_name', 'head', 'method', 'head_type', 'model_name')
         if c in df.columns), None)
    if method_col is not None:
        names = df[method_col].map(normalize_head)
        missing = [h for h in HEAD_TYPES if not (names == h).any()]
        if missing:
            raise ValueError('0% benchmark is missing head(s): {}.'.format(missing))
        for head in HEAD_TYPES:
            yield head, df.loc[names == head].reset_index(drop=True)
        return

    if all('mu_' + h in df.columns and h in df.columns for h in HEAD_TYPES):
        shared = ['y', 'raw_hr'] + [
            c for c in ('user_id', 'subject_id', 'subject', 'user') if c in df.columns
        ]
        for head in HEAD_TYPES:
            frame = df.loc[:, shared].copy()
            frame['mu'] = df['mu_' + head].to_numpy()
            frame['var'] = df[head].to_numpy()
            yield head, frame
        return

    raise ValueError(
        'benchmark_heads() must return either head_name with y/mu/var/raw_hr, '
        'or per-head mu_<head> and <head> uncertainty columns. '
        'No encoder embeddings are required.'
    )


def require_users(df):
    """Require real subject identifiers (never silently pool observations)."""
    for key in ('user_id', 'subject_id', 'subject', 'user'):
        if key in df.columns:
            if df[key].isna().any():
                raise ValueError('Missing {} values; cannot aggregate per user.'.format(key))
            users = df[key].astype(str).to_numpy()
            if len(pd.unique(users)) < 2:
                raise ValueError('Need >=2 independent test subjects for mean +/- SD.')
            return users
    raise ValueError('Expected a user_id/subject_id/subject/user column.')


def score_risk(y, predictions, uncertainty, original):
    """Preserve the supplied regression risk--coverage implementation.

    The helper may report achieved coverage rounded to a finite number of
    windows. Use the NOMINAL 100,90,...,10% grid to align users and retain
    achieved coverage separately for reference.
    """
    ranked = risk_coverage_curve_regression(y, predictions, uncertainty, original)
    if not isinstance(ranked, pd.DataFrame) or 'mse' not in ranked:
        raise ValueError('risk_coverage_curve_regression must return DataFrame with mse.')

    expected = np.arange(100.0, 0.0, -10.0)
    if len(ranked) != len(expected):
        raise ValueError('Expected 10 coverage points: 100%, 90%, ..., 10%.')

    if 'coverage' in ranked.columns:
        actual = ranked['coverage'].to_numpy(dtype=float)
        if np.nanmax(actual) <= 1.001:
            actual = actual * 100.0
    elif 'coverage_pct' in ranked.columns:
        actual = ranked['coverage_pct'].to_numpy(dtype=float)
    else:
        actual = expected.copy()

    mse = ranked['mse'].to_numpy(dtype=float)
    order = np.argsort(-actual, kind='stable')
    actual, mse = actual[order], mse[order]
    if not (np.isfinite(actual).all() and np.isfinite(mse).all()):
        raise ValueError('Nonfinite coverage or risk values.')
    if np.max(np.abs(actual - expected)) > 10.01:
        raise ValueError('Unexpected coverage levels from upstream benchmark.')
    return pd.DataFrame({
        'coverage_pct': expected,
        'actual_coverage_pct': actual,
        'mse': mse,
    })


def evaluate_one(df, method, regime):
    """Compute full-held-out-test risk--coverage SEPARATELY per test subject."""
    cols = ({'y': 'y', 'pred': 'mu', 'uncertainty': 'var', 'original': 'raw_hr'}
            if regime == 0 else
            {'y': 'y', 'pred': 'predicted_hr',
             'uncertainty': 'hr_uncertainty', 'original': 'original'})
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise ValueError('{}%/{} missing required columns: {}'.format(
            regime, method, missing))

    subjects = require_users(df)
    numeric = {name: df[col].to_numpy(dtype=float) for name, col in cols.items()}
    if not all(np.isfinite(v).all() for v in numeric.values()):
        raise ValueError('{}%/{} has nonfinite HR data.'.format(regime, method))

    rows = []
    for user in pd.unique(subjects):
        indices = np.flatnonzero(subjects == user)
        if len(indices) < 10:
            raise ValueError('User {} has only {} HR windows (need >=10).'.format(
                user, len(indices)))
        curve = score_risk(*(numeric[key][indices] for key in
                             ('y', 'pred', 'uncertainty', 'original')))
        for row in curve.itertuples(index=False):
            rows.append({
                'finetuning': regime,
                'method': method,
                'user_id': str(user),
                'coverage_pct': row.coverage_pct,
                'actual_coverage_pct': row.actual_coverage_pct,
                'mse': row.mse,
                'n_windows': len(indices),
            })
    return pd.DataFrame(rows)


def aggregate_curves(per_user):
    """Mean and between-subject sample SD; NO pooled-window statistics."""
    return per_user.groupby(
        ['finetuning', 'method', 'coverage_pct'], as_index=False, sort=True
    ).agg(
        mse_mean=('mse', 'mean'),
        mse_sd=('mse', lambda v: v.std(ddof=1)),
        n_users=('user_id', 'nunique'),
        actual_coverage_mean=('actual_coverage_pct', 'mean'),
    )


def plot_risk_coverage(summary, output_dir, regimes):
    """ONE line plot, panels for fine-tuning, HR heads colored consistently."""
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.8 * len(regimes), 4.35),
                             sharey=True, squeeze=False)
    colors = {
        'gaussian': '#2878B5',
        'student_t': '#CC6F38',
        'mc_dropout': '#489674',
    }
    labels = {
        'gaussian': 'Gaussian',
        'student_t': 'Student-t',
        'mc_dropout': 'MC Dropout',
    }
    for ax, regime in zip(axes[0], regimes):
        sub = summary[summary['finetuning'] == regime]
        for head in HEAD_TYPES:
            curve = sub[sub['method'] == head].sort_values(
                'coverage_pct', ascending=False)
            if curve.empty:
                continue
            x = curve['coverage_pct'].to_numpy(dtype=float)
            mean = curve['mse_mean'].to_numpy(dtype=float)
            sd = curve['mse_sd'].fillna(0).to_numpy(dtype=float)
            ax.plot(x, mean, color=colors[head], marker='o', markersize=3.5,
                    linewidth=1.8, label=labels[head])
            ax.fill_between(x, mean - sd, mean + sd,
                            color=colors[head], alpha=0.13, linewidth=0)
        ax.set_title('{}% fine-tuning'.format(regime))
        ax.set_xlim(100, 10)
        ax.set_xticks([100, 80, 60, 40, 20, 10])
        ax.set_xlabel('Coverage (%)')
        ax.grid(alpha=0.22)

    axes[0, 0].set_ylabel('MSE')

    handles, texts = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, texts, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    
    fig.savefig(output_dir / 'risk_coverage.png', dpi=220)
    fig.savefig(output_dir / 'risk_coverage.pdf')
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-root', default='ssl_wearables/output_hr/')
    parser.add_argument('--output-dir', default='upstream_hr_risk_coverage')
    parser.add_argument('--regimes', default='0,25,100',
                        help='Any subset of 0,25,100; default: all three.')
    parser.add_argument('--inference-batch-size', type=int, default=128,
                        help='Fine-tuned evaluation batch size (use 64 if GPU OOM).')
    parser.add_argument('--resume', action='store_true',
                        help='Resume completed head/regime risk curves in output-dir.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    regimes = [int(item.strip()) for item in args.regimes.split(',')]
    if not regimes or len(regimes) != len(set(regimes)) or any(
            item not in (0, 25, 100) for item in regimes):
        raise ValueError('--regimes must be a unique subset of 0,25,100.')
    regimes = sorted(regimes)
    if args.inference_batch_size < 1:
        raise ValueError('--inference-batch-size must be >= 1.')

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    curves_path = outdir / 'risk_coverage_per_user.csv'
    summary_path = outdir / 'risk_coverage_mean_sd.csv'
    config_path = outdir / 'config.json'
    config = {
        'model_root': args.model_root,
        'regimes': regimes,
        'inference_batch_size': args.inference_batch_size,
        'aggregation': 'per_user_risk_coverage_v1',
    }

    curves = pd.DataFrame()
    if args.resume:
        if curves_path.exists() != config_path.exists():
            raise FileNotFoundError('Incomplete risk--coverage checkpoint; need CSV and config.')
        if config_path.exists():
            with config_path.open('r') as f:
                previous = json.load(f)
            for key in ('model_root', 'inference_batch_size', 'aggregation'):
                if previous.get(key) != config[key]:
                    raise ValueError('Cannot resume incompatible configuration: {}.'.format(key))
            curves = pd.read_csv(curves_path, dtype={'user_id': str})
    elif curves_path.exists():
        print('WARNING: replacing existing risk--coverage outputs. '
              'Use --resume to keep completed results.', flush=True)
    with config_path.open('w') as f:
        json.dump(config, f, indent=2)

    def is_done(regime, head):
        if curves.empty:
            return False
        rows = curves[(curves.finetuning == regime) & (curves.method == head)]
        n_users = rows.user_id.nunique()
        return (n_users >= 2 and len(rows) == 10 * n_users and
                rows.groupby('user_id').coverage_pct.nunique().eq(10).all())

    def save_result(frame, regime, head):
        nonlocal curves
        if not curves.empty:
            existing_users = set(curves.user_id.astype(str))
            incoming_users = set(frame.user_id.astype(str))
            if existing_users != incoming_users:
                raise ValueError('{}%/{} has a different set of held-out users.'.format(
                    regime, head))
            curves = curves[~((curves.finetuning == regime) &
                              (curves.method == head))]
        curves = pd.concat([curves, frame], ignore_index=True)
        curves.to_csv(curves_path, index=False)
        aggregate_curves(curves).to_csv(summary_path, index=False)
        print('  Saved {}%/{} ({} users)'.format(
            regime, head, frame.user_id.nunique()), flush=True)

    if 0 in regimes and not all(is_done(0, head) for head in HEAD_TYPES):
        print('Evaluating 0% upstream held-out HR windows...', flush=True)
        zero = benchmark_heads()  # No embeddings or additional kNN inference.
        for head, frame in zero_head_frames(zero):
            if not is_done(0, head):
                save_result(evaluate_one(frame, head, 0), 0, head)
        del zero
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for regime in (25, 100):
        if regime not in regimes:
            continue
        for head in HEAD_TYPES:
            if is_done(regime, head):
                print('Reusing {}%/{}'.format(regime, head), flush=True)
                continue
            checkpoint = str(Path(args.model_root) / CHECKPOINTS[regime][head])
            print('Evaluating {}%/{}: {}'.format(regime, head, checkpoint), flush=True)
            frame = make_upstream_results(
                OUTPUT_SIZE[head], checkpoint, batch_size=args.inference_batch_size)
            save_result(evaluate_one(frame, head, regime), regime, head)
            del frame
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if curves.empty:
        raise ValueError('No risk--coverage results were evaluated.')
    selected = curves[curves.finetuning.isin(regimes)]
    summary = aggregate_curves(selected)
    # Keep all completed regimes in the checkpoint, even when plotting a subset.
    curves.to_csv(curves_path, index=False)
    aggregate_curves(curves).to_csv(summary_path, index=False)
    plot_risk_coverage(summary, outdir, regimes)
    print('\nSaved full-test per-user risk--coverage curves, mean +/- SD, '
          'and one line plot to {}'.format(outdir.resolve()), flush=True)
    return selected, summary


if __name__ == '__main__':
    main()

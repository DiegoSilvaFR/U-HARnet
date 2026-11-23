#!/usr/bin/env python3
"""Information--alignment analysis of cross-task uncertainty (Theorem 1).

Theorem 1 of the paper states, for any uncertainty score U and 0/1 loss L,

    Delta_U = R_all - AURC_U = rho_U * kappa_U,

where kappa_U = SD(m(U)) is the information term (m(u) = P(L = 1 | U = u))
and rho_U = Corr(m(U), ln W) is the alignment term (W = rank percentile).
This script estimates both terms from the per-window files written by
downstream_experiments.py (downstream_results/.../windows/windows_*.parquet
or .csv.gz), for every dataset, fine-tuning regime, HR head and ranking
strategy, plus the motion-intensity baseline.

Estimators (per held-out subject s, then averaged over subjects, exactly as
the relative AURC improvements in Fig. 4):

  Delta_s  = R_all,s - AURC_s, with the tie-aware compute_aurc() used in the
             paper, so 100 * mean_s(Delta_s / R_all,s) reproduces Fig. 4.
  V_s, C_s from B equal-count bins of the score:
             V_s = sum_b pi_b (p_b - R_all,s)^2            (between-bin variance)
             C_s = sum_b pi_b (1-pi_b) p_b (1-p_b)/(n_b-1)  (binomial-noise term)
             E[V_s - C_s] = variance of the binned error-probability function.
  tau      = mean_s (V_s - C_s) / R_all,s^2 (not truncated at zero)
  kappa    = sqrt(max(0, tau)): root-mean-square relative information across
             subjects. Averaging the unbiased variances BEFORE the square root
             avoids the small-sample (Jensen) bias of averaging per-subject
             square roots, which matters with ~150-250 windows per subject.
  rho      = (mean_s Delta_s / R_all,s) / kappa. Because |Delta_s| <= kappa_s
             for every subject, |rho| <= 1 in the population.
  Information is "detected" when the lower bound of the bootstrap CI of tau
  is > 0; rho is reported only then.
  B        = "auto": min(20, max(4, n_s // 25)) per subject (>= ~25 windows
             per bin), or a fixed integer.

Conditional information (Corollary 2): for a score U and a conditioning score
C (motion intensity or HAR entropy),
  tau_cond = mean_s [(V2_s - C2_s) - (V1_s - C1_s)] / R_all,s^2,
with V1/C1 from k equal-count bins of C and V2/C2 from k bins of U nested in
each C bin (k = 3 if n_s < 1500 else 5). It estimates
E[Var(P(L | U, C) | C)] / R_all^2 at the binned resolution: it is > 0 only if
U separates error rates among windows with similar C (conservative, because
binning can only remove information).

All confidence intervals: percentile bootstrap over subjects.

Usage (from the repository root, after running downstream_experiments.py):
    python kappa_rho_analysis.py
    python kappa_rho_analysis.py --windows downstream_results --thin 3 \
        --output-dir figures/kappa_rho_thin3
    python kappa_rho_analysis.py --loss retrieval_loss --output-dir figures/kappa_rho_retrieval

Outputs (in --output-dir, default figures/):
    kappa_rho_summary.csv          one row per dataset x regime x head x strategy
    conditional_information.csv    HR beyond motion, HR beyond entropy,
                                   motion beyond entropy; plus paired tests of
                                   HR vs motion AURC
"""

from __future__ import annotations

import argparse
import re
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from linear_probe import compute_aurc
except ImportError:  # pragma: no cover - fallback if run outside the repo
    def compute_aurc(errors, uncertainty):
        errors = np.asarray(errors, dtype=float).reshape(-1)
        uncertainty = np.asarray(uncertainty, dtype=float).reshape(-1)
        order = np.argsort(uncertainty, kind='stable')
        sorted_errors, sorted_scores, n = errors[order], uncertainty[order], len(errors)
        starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
        group_errors = np.add.reduceat(sorted_errors, starts)
        group_counts = np.diff(np.r_[starts, n])
        before = np.repeat(np.r_[0., np.cumsum(group_errors)[:-1]], group_counts)
        within = np.arange(n) - np.repeat(starts, group_counts) + 1
        expected = before + within * np.repeat(group_errors / group_counts, group_counts)
        return float(np.mean(expected / np.arange(1, n + 1)))


DATASET_NAMES = OrderedDict([
    ("capture24", "CAPTURE-24"),
    ("harth", "HARTH"),
    ("pamap", "PAMAP2"),
    ("realworld", "REALWORLD"),
    ("usc", "USC-HAD"),
])
META_COLUMNS = {"subject", "window", "y_true", "y_pred", "loss", "retrieval_loss",
                "motion", "har_entropy", "logistic_har_entropy"}
STRATEGIES = OrderedDict([
    ("", "raw_hr"),
    ("/logistic_hr", "/logistic_hr"),
    ("/har_entropy", "/har_entropy"),
    ("/fusion", "/fusion"),
    ("/logistic_har_entropy", "/logistic_har_entropy"),
])


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def find_window_files(roots):
    files = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            files.append(root)
        else:
            files += sorted(root.rglob("windows_*.parquet")) + sorted(root.rglob("windows_*.csv.gz"))
    return files


def parse_window_filename(path: Path):
    """windows_<dataset>_ft<depth>[_<head>].<ext> -> (dataset_key, depth, head)."""
    name = path.name.split(".")[0]
    m = re.match(r"windows_(.+?)_ft(\d+)(?:_(.+))?$", name)
    if not m:
        raise ValueError(f"Unexpected per-window file name: {path.name}")
    raw, depth, head = m.group(1).lower(), int(m.group(2)), m.group(3)
    key = next((k for k in DATASET_NAMES if k in raw.replace("-", "").replace("_", "")), raw)
    return key, depth, head


def read_windows(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["subject"] = df["subject"].astype(str)
    return df


def thin_windows(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 1:
        return df
    order = df.sort_values(["subject", "window"]) if "window" in df.columns else df
    return order[order.groupby("subject", sort=False).cumcount() % k == 0]


# ----------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------
def _equal_count_bins(values, n_bins, rng):
    """Equal-count quantile bins; ties broken at random (tied windows are
    indistinguishable to the score)."""
    n = len(values)
    order = np.lexsort((rng.random(n), values))
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    return (ranks * n_bins) // n


def _between_variance_corrected(cells, loss):
    """Noise-corrected between-cell variance of error rates: V - C."""
    n = len(loss)
    _, cell_ids = np.unique(cells, return_inverse=True)
    n_b = np.bincount(cell_ids).astype(float)
    p_b = np.bincount(cell_ids, weights=loss) / n_b
    pi_b = n_b / n
    p_bar = float(np.sum(pi_b * p_b))
    v = float(np.sum(pi_b * (p_b - p_bar) ** 2))
    ok = n_b > 1
    c = float(np.sum((pi_b * (1 - pi_b) * p_b * (1 - p_b))[ok] / (n_b[ok] - 1)))
    return v - c


def resolve_bins(spec, n):
    if spec == "auto":
        return int(min(20, max(4, n // 25)))
    return int(min(int(spec), max(2, n // 2)))


def subject_statistics(loss, score, bins_spec, rng):
    loss = np.asarray(loss, dtype=float)
    score = np.asarray(score, dtype=float)
    r_all = float(loss.mean())
    n = len(loss)
    if r_all <= 0 or r_all >= 1 or n < 8:
        return None
    delta = r_all - compute_aurc(loss, score)
    b = resolve_bins(bins_spec, n)
    var_c = _between_variance_corrected(_equal_count_bins(score, b, rng), loss)
    return dict(n=n, bins=b, r_all=r_all, delta=delta,
                kappa=np.sqrt(max(var_c, 0.0)), var_c=var_c)


def conditional_subject_statistic(loss, score, cond, rng):
    loss = np.asarray(loss, dtype=float)
    r_all = float(loss.mean())
    n = len(loss)
    if r_all <= 0 or r_all >= 1 or n < 18:
        return None
    k = 3 if n < 1500 else 5
    c_bins = _equal_count_bins(np.asarray(cond, dtype=float), k, rng)
    cells = np.empty(n, dtype=np.int64)
    score = np.asarray(score, dtype=float)
    for b in range(k):
        idx = np.flatnonzero(c_bins == b)
        cells[idx] = b * k + _equal_count_bins(score[idx], k, rng)
    tau = (_between_variance_corrected(cells, loss)
           - _between_variance_corrected(c_bins, loss)) / r_all ** 2
    return tau


def _aggregate_arrays(d_rel, v_rel):
    """d_rel = Delta_s / R_s, v_rel = (V_s - C_s) / R_s^2; last axis = subjects."""
    rel_delta = d_rel.mean(axis=-1)
    tau = v_rel.mean(axis=-1)
    rel_kappa = np.sqrt(np.maximum(tau, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(rel_kappa > 0, rel_delta / rel_kappa, np.nan)
    return np.stack([rel_delta, rel_kappa, rho, tau], axis=-1)


def aggregate(stats: pd.DataFrame):
    d_rel = (stats["delta"] / stats["r_all"]).to_numpy(dtype=float)
    v_rel = (stats["var_c"] / stats["r_all"] ** 2).to_numpy(dtype=float)
    return _aggregate_arrays(d_rel, v_rel)


def bootstrap(stats: pd.DataFrame, n_boot, rng):
    """Percentile bootstrap over subjects (vectorized)."""
    d_rel = (stats["delta"] / stats["r_all"]).to_numpy(dtype=float)
    v_rel = (stats["var_c"] / stats["r_all"] ** 2).to_numpy(dtype=float)
    idx = rng.integers(0, len(d_rel), size=(n_boot, len(d_rel)))
    draws = _aggregate_arrays(d_rel[idx], v_rel[idx])
    return np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def summarize_score(df, loss_col, score_col, bins_spec, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for subject, g in df.groupby("subject", sort=False):
        st = subject_statistics(g[loss_col].to_numpy(), g[score_col].to_numpy(), bins_spec, rng)
        if st is not None:
            rows.append(dict(subject=subject, **st))
    stats = pd.DataFrame(rows)
    if len(stats) < 2:
        return None
    est = aggregate(stats)
    lo, hi = bootstrap(stats, n_boot, np.random.default_rng(seed + 1))
    detected = bool(lo[3] > 0)
    out = dict(
        n_subjects=len(stats),
        median_windows_per_subject=float(stats["n"].median()),
        median_bins=float(stats["bins"].median()),
        mean_R_all=float(stats["r_all"].mean()),
        mean_improvement_pct=100 * est[0],
        rel_delta=est[0], rel_delta_lo=lo[0], rel_delta_hi=hi[0],
        rel_kappa=est[1], rel_kappa_lo=lo[1], rel_kappa_hi=hi[1],
        rho=est[2] if detected else np.nan,
        rho_lo=lo[2] if detected else np.nan,
        rho_hi=hi[2] if detected else np.nan,
        tau=est[3], tau_lo=lo[3], tau_hi=hi[3],
        information_detected=detected,
    )
    return out, stats


def summarize_conditional(df, loss_col, score_col, cond_col, n_boot, seed):
    rng = np.random.default_rng(seed)
    taus = []
    for _, g in df.groupby("subject", sort=False):
        t = conditional_subject_statistic(g[loss_col].to_numpy(), g[score_col].to_numpy(),
                                          g[cond_col].to_numpy(), rng)
        if t is not None:
            taus.append(t)
    taus = np.asarray(taus)
    if len(taus) < 2:
        return None
    brng = np.random.default_rng(seed + 1)
    boots = taus[brng.integers(0, len(taus), size=(n_boot, len(taus)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n_subjects=len(taus), tau_cond=float(taus.mean()),
                tau_cond_lo=float(lo), tau_cond_hi=float(hi),
                information_beyond_detected=bool(lo > 0))


def paired_aurc_test(df, loss_col, a_col, b_col):
    """Two-sided paired Wilcoxon on subject-level AURC of score a vs score b."""
    from scipy.stats import wilcoxon
    diffs = []
    for _, g in df.groupby("subject", sort=False):
        loss = g[loss_col].to_numpy(dtype=float)
        if 0 < loss.mean() < 1:
            diffs.append(compute_aurc(loss, g[a_col].to_numpy()) - compute_aurc(loss, g[b_col].to_numpy()))
    diffs = np.asarray(diffs)
    if len(diffs) < 3 or np.allclose(diffs, 0):
        return dict(n_pairs=len(diffs), mean_aurc_diff=float(diffs.mean()) if len(diffs) else np.nan,
                    p=np.nan if len(diffs) < 3 else 1.0)
    return dict(n_pairs=len(diffs), mean_aurc_diff=float(diffs.mean()),
                p=float(wilcoxon(diffs, alternative="two-sided").pvalue))


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if len(ok):
        order = ok[np.argsort(p[ok])]
        out[order] = np.maximum.accumulate(np.minimum(1, p[order] * np.arange(len(order), 0, -1)))
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def analyse(files, loss_col, bins_spec, n_boot, thin, seed):
    summary, conditional = [], []
    shared_cache = {}
    for path in files:
        key, depth, file_head = parse_window_filename(path)
        dataset = DATASET_NAMES.get(key, key)
        df = thin_windows(read_windows(path), thin)
        if loss_col not in df.columns:
            warnings.warn(f"{path.name}: no '{loss_col}' column; skipped")
            continue
        sources = [c for c in df.columns if c not in META_COLUMNS and "/" not in c]
        print(f"{path.name}: {df['subject'].nunique()} subjects, {len(df)} windows, sources={sources}")

        def run(score_col, cache_key=None):
            if cache_key is not None and cache_key in shared_cache:
                return shared_cache[cache_key]
            res = summarize_score(df, loss_col, score_col, bins_spec, n_boot, seed)
            if cache_key is not None:
                shared_cache[cache_key] = res
            return res

        for source in sources:
            for suffix, strategy in STRATEGIES.items():
                col = (source + suffix if suffix in ("", "/logistic_hr", "/fusion")
                       else suffix.lstrip("/"))
                if col not in df.columns:
                    continue
                shared = col in ("har_entropy", "logistic_har_entropy")
                res = run(col, (path, col) if shared else None)
                if res is None:
                    continue
                summary.append(dict(dataset=dataset, fine_tuning_pct=depth, head=source,
                                    strategy=strategy, score_column=col, loss=loss_col,
                                    **res[0]))
            if "motion" in df.columns:
                res = run("motion", (path, "motion"))
                if res is not None:
                    summary.append(dict(dataset=dataset, fine_tuning_pct=depth, head=source,
                                        strategy="motion", score_column="motion", loss=loss_col,
                                        **res[0]))

            # Conditional information and HR-vs-motion tests (raw HR score)
            pairs = []
            if "motion" in df.columns:
                pairs.append(("hr_beyond_motion", source, "motion"))
            if "har_entropy" in df.columns:
                pairs.append(("hr_beyond_entropy", source, "har_entropy"))
            if "motion" in df.columns and "har_entropy" in df.columns:
                pairs.append(("motion_beyond_entropy", "motion", "har_entropy"))
            for label, score_col, cond_col in pairs:
                ckey = (path, "cond", score_col, cond_col)
                if ckey not in shared_cache:
                    shared_cache[ckey] = summarize_conditional(df, loss_col, score_col, cond_col,
                                                               n_boot, seed)
                res = shared_cache[ckey]
                if res is None:
                    continue
                row = dict(dataset=dataset, fine_tuning_pct=depth, head=source,
                           quantity=label, score=score_col, given=cond_col, loss=loss_col, **res)
                if label == "hr_beyond_motion":
                    t = paired_aurc_test(df, loss_col, source, "motion")
                    row.update(hr_minus_motion_aurc=t["mean_aurc_diff"],
                               hr_vs_motion_n=t["n_pairs"], hr_vs_motion_p=t["p"])
                conditional.append(row)

    summary = pd.DataFrame(summary)
    conditional = pd.DataFrame(conditional)
    if len(conditional) and "hr_vs_motion_p" in conditional:
        conditional["hr_vs_motion_p_holm"] = np.nan
        mask = conditional["quantity"] == "hr_beyond_motion"
        for _, ids in conditional[mask].groupby(["dataset", "fine_tuning_pct"]).groups.items():
            conditional.loc[ids, "hr_vs_motion_p_holm"] = holm(conditional.loc[ids, "hr_vs_motion_p"])
    return summary, conditional


def check_against_heatmap(summary, heatmap_csv):
    """Compare 100*Delta with the Fig. 4 summary (same per-subject average)."""
    ref = pd.read_csv(heatmap_csv)
    merged = summary.merge(ref[["dataset", "fine_tuning_pct", "head", "strategy", "mean_improvement_pct"]],
                           on=["dataset", "fine_tuning_pct", "head", "strategy"],
                           suffixes=("", "_fig4"))
    if merged.empty:
        print("check: no matching rows in", heatmap_csv)
        return
    diff = (merged["mean_improvement_pct"] - merged["mean_improvement_pct_fig4"]).abs()
    print(f"check vs {heatmap_csv}: {len(merged)} cells, max |difference| = {diff.max():.2e} "
          "percentage points (should be ~0 when files come from the same run)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", nargs="*", default=["downstream_results"],
                    help="Folders (searched recursively) or files with per-window results.")
    ap.add_argument("--output-dir", type=Path, default=Path("figures"))
    ap.add_argument("--loss", default="loss", choices=["loss", "retrieval_loss"],
                    help="0/1 loss: HAR classification error (default) or retrieval error.")
    ap.add_argument("--bins", default="auto",
                    help="Bins per subject for kappa: 'auto' (default) or an integer.")
    ap.add_argument("--thin", type=int, default=1,
                    help="Keep every k-th window per subject (dependence check, e.g. 3).")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-summary", type=Path,
                    default=Path("figures/aurc_relative_improvement_summary.csv"),
                    help="Fig. 4 summary CSV to verify that 100*Delta reproduces it.")
    args = ap.parse_args(argv)

    files = find_window_files(args.windows)
    if not files:
        raise SystemExit("No windows_*.parquet / windows_*.csv.gz files found. Run "
                         "downstream_experiments.py (without --no-save-windows) first.")
    summary, conditional = analyse(files, args.loss, args.bins, args.n_boot, args.thin, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "kappa_rho_summary.csv", index=False)
    conditional.to_csv(args.output_dir / "conditional_information.csv", index=False)
    print("Created:", args.output_dir / "kappa_rho_summary.csv",
          args.output_dir / "conditional_information.csv", sep="\n  ")
    if args.loss == "loss" and args.thin == 1 and args.check_summary and args.check_summary.is_file():
        check_against_heatmap(summary, args.check_summary)


if __name__ == "__main__":
    main()

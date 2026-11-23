#!/usr/bin/env bash
# Full downstream pipeline + figures + information-alignment analysis.
#
# Run from the repository root (Linux / WSL / Git Bash):
#     bash run_pipeline.sh                      # all five datasets
#     DATASETS="harth usc" bash run_pipeline.sh # a subset
#     SKIP_DOWNSTREAM=1 bash run_pipeline.sh    # figures + analysis only
#
# Each dataset writes to downstream_results/<folder>/ (results JSON, statistics
# CSVs, and per-window files in windows/), i.e. where the plotting scripts look.
# Existing files there are overwritten; commit or copy them first if needed.
# Logs: logs/downstream_<dataset>.log
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATASETS="${DATASETS:-pamap usc realworld harth capture24}"
mkdir -p logs figures

dataset_args() {
  # echo "<dataset-name> <dataset-root> <output-folder>"
  case "$1" in
    capture24) echo "capture24 downstream_datasets/capture24/capture24_subject_ downstream_results/capture24" ;;
    harth)     echo "harth downstream_datasets/harth/harth_subject_ downstream_results/harth" ;;
    realworld) echo "realworld downstream_datasets/realworld/realworld_subject_ downstream_results/realworld" ;;
    usc)       echo "usc downstream_datasets/usc/usc_subject_ downstream_results/usc" ;;
    pamap)
      if [ -f downstream_datasets/pamap2/pamap_subject_1.npz ]; then
        echo "pamap downstream_datasets/pamap2/pamap_subject_ downstream_results/pamap2"
      else
        echo "pamap downstream_datasets/pamap2/pamap_subject_{}_wrist downstream_results/pamap2"
      fi ;;
    *) echo "Unknown dataset: $1" >&2; exit 1 ;;
  esac
}

if [ "${SKIP_DOWNSTREAM:-0}" != "1" ]; then
  for ds in $DATASETS; do
    read -r name root out <<< "$(dataset_args "$ds")"
    echo "=== downstream: $name -> $out (log: logs/downstream_$name.log)"
    "$PYTHON" -u downstream_experiments.py \
      --dataset-name "$name" --dataset-root "$root" --output-dir "$out" \
      > "logs/downstream_$name.log" 2>&1
  done
fi

echo "=== Fig. 4: AURC heatmap with significance markers"
"$PYTHON" plot_aurc_improvement_heatmap.py --output-dir figures

echo "=== Theorem 1: kappa / rho estimation"
"$PYTHON" kappa_rho_analysis.py --windows downstream_results --output-dir figures
"$PYTHON" kappa_rho_analysis.py --windows downstream_results --thin 3 \
  --output-dir figures/kappa_rho_thin3
"$PYTHON" kappa_rho_analysis.py --windows downstream_results --bins 10 \
  --output-dir figures/kappa_rho_bins10
"$PYTHON" kappa_rho_analysis.py --windows downstream_results --bins 50 \
  --output-dir figures/kappa_rho_bins50

echo "=== Information-alignment map"
"$PYTHON" plot_information_alignment_map.py \
  --summary figures/kappa_rho_summary.csv --output-dir figures

echo "Done. Main outputs in figures/:"
echo "  aurc_relative_improvement_heatmap.{pdf,png}   (Fig. 4)"
echo "  kappa_rho_summary.csv, conditional_information.csv"
echo "  information_alignment_map.{pdf,png}"

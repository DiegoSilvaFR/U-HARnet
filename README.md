# U-HARnet: Transferring Heart-Rate Uncertainty to Human Activity Recognition

**U-HARnet** studies whether uncertainty estimated for an **upstream heart-rate (HR) regression task** can help identify unreliable predictions in a **downstream human activity recognition (HAR) task**. The framework compares uncertainty estimators and encoder fine-tuning regimes while evaluating representation quality, activity classification, and uncertainty-based selective prediction.

The code builds on the [OxWearables self-supervised wearable encoder](https://github.com/OxWearables/ssl-wearables) (`harnet10`). Its input is a 10-second, three-axis accelerometer window sampled at 30 Hz (`N × 3 × 300`); the encoder produces a 1,024-dimensional representation. The upstream regression heads estimate HR and its associated uncertainty. On downstream datasets, the learned representation supports nearest-neighbor retrieval and a linear HAR probe, while upstream uncertainty can be used directly, aligned to HAR errors, or fused with HAR predictive entropy.

> **Scope.** This README documents how to **evaluate existing trained checkpoints**, not how to retrain all upstream HR models from scratch. The evaluation scripts require the study's pretrained `.mdl` checkpoints and processed datasets. Those assets must be supplied separately if they are not already available locally. See [Known issues and reproducibility notes](#known-issues-and-reproducibility-notes) before running the complete pipeline.

## Contents

- [Evaluation at a glance](#evaluation-at-a-glance)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Required checkpoints and data](#required-checkpoints-and-data)
- [1. Reproduce upstream HR evaluation](#1-reproduce-upstream-hr-evaluation)
- [2. Reproduce downstream HAR evaluation](#2-reproduce-downstream-har-evaluation)
- [3. Evaluate a new HAR dataset](#3-evaluate-a-new-har-dataset)
- [Results and interpretation](#results-and-interpretation)
- [Known issues and reproducibility notes](#known-issues-and-reproducibility-notes)
- [Acknowledgments](#acknowledgments)

## Evaluation at a glance

```text
                          10 s triaxial accelerometer window
                                      (3 × 300)
                                          │
                                          ▼
                               OxWearables HARNet-10
                                  encoder (1024-D)
                                          │
                   ┌──────────────────────┴─────────────────────┐
                   │                                            │
                   ▼                                            ▼
           Upstream HR regression                      Downstream HAR
         Gaussian / Student-t / MC Dropout       ┌──────────────┴─────────────┐
                   │                             │                            │
          HR estimate + uncertainty         1-NN cosine retrieval        Linear probe
                   │                             │                            │
          HR MSE vs coverage                Macro-R@1, AURC            Macro-F1, AURC
                   │                                                          │
                   └─────────────► HR uncertainty ranking / alignment / fusion ◄┘
```

Three encoder regimes are evaluated:

| Regime | Encoder | HR uncertainty methods |
| --- | --- | --- |
| **0%** | Original, frozen Oxford encoder | Gaussian, Student-t, MC Dropout; Oxford multitask entropy (**CE**) as an additional downstream reference |
| **25%** | Partially HR-fine-tuned encoder | Gaussian, Student-t, MC Dropout |
| **100%** | Fully HR-fine-tuned encoder | Gaussian, Student-t, MC Dropout |

The 0% HR heads share the same frozen embedding; 25% and 100% may yield different embeddings for each HR objective. **CE is a reference entropy score from the original Oxford multitask classification heads, not an HR regression head.**

## Repository structure

```text
U-HARnet/
├── upstream_hr.py                  # Upstream HR risk-coverage evaluation + plot
├── benchmark_upstream_heads.py     # Frozen encoder + 0% HR-head inference
├── benchmark_upstream_finetuned.py # Fine-tuned HR checkpoint loading / inference
├── downstream_experiments.py       # Downstream retrieval, LOSO probe and statistics
├── linear_probe.py                 # HAR linear probe and cross-task alignment
├── plot_r1_vs_f1.py               # Representation-quality vs HAR-performance figure
├── plot_r_auroc_transfer.py       # Legacy R-AUROC plot (older result schema)
├── ssl_wearables/                  # Wearable encoder implementation and model assets
├── downstream_results/            # Generated/stored downstream results
└── test.ipynb                      # Development notebook
```

Most paths in the Python scripts are **relative to the repository root**. Run commands from `U-HARnet/` unless you also update those paths.

## Installation

Clone the repository and set up a Python environment:

```bash
git clone https://github.com/DiegoSilvaFR/U-HARnet.git
cd U-HARnet

python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn matplotlib torch
```

The main evaluation scripts directly use **PyTorch, NumPy, pandas, SciPy, scikit-learn, and Matplotlib**. Other dependencies may be required by the bundled wearable-model implementation or by your environment. There is no verified, pinned root-level environment file in this documentation, so **exact package versions are not claimed**. For paper-level reproducibility, record the actual versions used:

```bash
python -m pip freeze > environment_used.txt
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

A CUDA-capable GPU is strongly recommended and is **required by the current downstream retrieval code**, which explicitly calls `retrieval_metrics_gpu()`. Some upstream inference supports CPU execution, but the full downstream command is not CPU-only. The first 0% encoder evaluation uses `torch.hub.load('OxWearables/ssl-wearables', ...)`, which may need internet access or an existing local Torch Hub cache.

## Required checkpoints and data

### 1. Model files

The scripts expect the following file names (or equivalent paths after editing the script constants):

```text
ssl_wearables/
├── model_check_point/
│   └── mtl_best.mdl
└── output_hr/
    ├── HR_reg_gaussian_oxfordhub_ft0_1_final.mdl
    ├── HR_reg_student_t_oxfordhub_ft0_1_final.mdl
    ├── mc_dropout_head.mdl
    ├── HR_reg_gaussian_oxfordhub_25_final.mdl
    ├── HR_reg_student_t_oxfordhub_25_final.mdl
    ├── HR_reg_mc_dropout_oxfordhub_ft25_25_final.mdl
    ├── HR_reg_gaussian_oxfordhub_100_final.mdl
    ├── HR_reg_student_t_oxfordhub_100_final.mdl
    └── HR_reg_mc_dropout_oxfordhub_ft100_100_final.mdl
```

- `mtl_best.mdl` contains the original Oxford multitask-head weights used to compute the **CE reference score**; the 0% feature extractor is separately loaded with `torch.hub`.
- The three `ft0` files are **head-only** checkpoints for the frozen encoder.
- The 25% and 100% files are **fine-tuned HR model** checkpoints expected by `load_HR_encoder()`.
- Do not interchange head-only and full-model checkpoints. Checkpoint structure and parameter names must match the model implementations.

These paths are **expected inputs**, not a claim that all checkpoint binaries are distributed with the repository. The project currently does not expose a general `--checkpoint-dir` argument for the downstream evaluator: update the hard-coded paths in `downstream_experiments.py` and `benchmark_upstream_heads.py` if yours differ. The upstream `--model-root` applies to the 25% and 100% checkpoints only.

### 2. Upstream HR test data

The upstream scripts expect the **held-out HR test subjects** in the following layout:

```text
ssl_wearables/data/ssl/ssl_ai4wellness/data/test/
├── file_list.csv
├── user_001.npz
├── user_002.npz
└── ...
user_stats.json                  # At repository root
```

Each `user_<id>.npz` contains:

| Key | Shape | Description |
| --- | --- | --- |
| `X` | `(n_windows, 3, 300)` | 10-second, 30-Hz accelerometer windows; axes in the expected Oxford order, acceleration in **g** |
| `hr` | `(n_windows,)` | HR target in beats per minute, one value per window |

`file_list.csv` must have a `file_list` column with accessible NPZ paths. The current helper removes a leading `/home/jovyan/` from paths and derives the subject ID from the specific relative prefix `ssl_wearables/data/ssl/ssl_ai4wellness/data/test/user_`. The safest layout is to retain that naming scheme and use repository-relative paths in the CSV. A typical file is:

```csv
file_list
ssl_wearables/data/ssl/ssl_ai4wellness/data/test/user_001.npz
ssl_wearables/data/ssl/ssl_ai4wellness/data/test/user_002.npz
```

The root `user_stats.json` maps those exact subject IDs to upstream HR normalization statistics:

```json
{
  "001": {"mean": 77.4, "std": 12.1},
  "002": {"mean": 83.0, "std": 10.7}
}
```

The values above are **illustrative**, not real study statistics. Supply the actual per-subject statistics used in the study (`std > 0`). The scripts evaluate against standardized HR targets, `(hr - mean) / std`, so changing those statistics changes the interpretation of upstream MSE and the input normalization expected by the trained heads.

### 3. Downstream HAR data

Each HAR dataset is represented as **one NPZ file per subject**. The pipeline concatenates the prefix supplied with `--dataset-root` and a subject ID, followed by `.npz`:

```text
downstream_datasets/
├── pamap2/pamap_subject_1.npz
├── usc/usc_subject_1.npz
├── realworld/realworld_subject_1.npz
├── harth/harth_subject_0.npz
└── capture24/capture24_subject_1.npz
```

Each file needs **only**:

| Key | Shape | Description |
| --- | --- | --- |
| `X` | `(n_windows, 3, 300)` | 30-Hz tri-axial accelerometer windows, `float32` recommended |
| `y` | `(n_windows,)` | Activity/intensity labels for each window |

All subjects in a dataset must share the **same label semantics**. HR labels are *not* needed for downstream evaluation. Use consistent sensor placement and axis ordering throughout a dataset; the encoder was developed for wrist accelerometry, and evaluating other placements constitutes a domain shift. If the original sampling rate differs from 30 Hz, apply appropriate anti-alias filtering and resampling **before** forming 300-sample windows. Do not allow windows to cross recording gaps, changes in activity labels, or different subjects.

The current subject-ID mapping is inside `get_df_ids()` in `downstream_experiments.py`:

| `--dataset-name` | IDs loaded by the current implementation |
| --- | --- |
| `pamap` | `1, 2, 5, 6, 8` |
| `usc` | `1`–`14` |
| `realworld` | `1`–`15` |
| `harth` | `0`–`21` |
| `capture24` | `1`–`150` |

**Important:** The script does not discover subject IDs automatically. Your local filenames and actual participant list must match this table, or you must modify `get_df_ids()`.

## 1. Reproduce upstream HR evaluation

The upstream evaluation measures whether sorting windows by estimated HR uncertainty and **retaining the lowest-uncertainty windows** reduces HR-regression mean squared error. For each **held-out subject**, it evaluates 100%, 90%, ..., 10% coverage. Plots report the unweighted **mean across subjects** and a band of ±1 between-subject sample standard deviation. The test subjects are not pooled before ranking.

### Run

Before running the 25% and 100% regimes, apply the one-line compatibility fix described in [Known issues](#known-issues-and-reproducibility-notes).

```bash
# All three regimes
python upstream_hr.py --regimes 0,25,100

# Frozen encoder / 0% HR heads only
python upstream_hr.py --regimes 0

# Resume previously completed head/regime combinations
python upstream_hr.py --resume

# Select a destination and the fine-tuned checkpoint directory
python upstream_hr.py \
  --regimes 25,100 \
  --model-root ssl_wearables/output_hr \
  --output-dir upstream_hr_risk_coverage
```

Outputs in `upstream_hr_risk_coverage/`:

| File | Contents |
| --- | --- |
| `risk_coverage_per_user.csv` | MSE for each subject, head, regime and coverage |
| `risk_coverage_mean_sd.csv` | Across-subject mean/SD per head, regime and coverage |
| `risk_coverage.png` / `.pdf` | Three-panel (or selected-regime) HR risk–coverage figure |
| `config.json` | Evaluation settings used for resume/checkpoint bookkeeping |

**Interpretation:** A descending MSE as more uncertain windows are rejected indicates useful uncertainty ordering for the **upstream HR target**. This alone does not establish that the same ordering predicts downstream HAR errors; that is evaluated in the next section. HR MSE is computed on the standardized target used by the script, not automatically in `bpm²`.

## 2. Reproduce downstream HAR evaluation

The downstream pipeline evaluates each dataset using the same 0%, 25% and 100% HR checkpoints. It includes:

1. **Representation quality:** exact cosine nearest-neighbor retrieval (self-match excluded), reported as class-balanced **Macro-R@1**; nearest neighbors are drawn from the pooled dataset embedding gallery, and the results are summarized by subject.
2. **HAR classification:** class-weighted linear probe trained in **leave-one-subject-out (LOSO)** fashion, evaluated using subject-level **macro-F1**.
3. **Selective reliability:** **AURC** of binary classification/retrieval errors when retaining the least-uncertain observations first; lower is better.
4. **Cross-task alignment:** logistic error models fitted **only on the outer-LOSO training subjects**, using inner subject-grouped out-of-fold predictions (`GroupKFold`, default 2 folds).
5. **Statistical comparisons:** paired subject-level tests, Holm adjustments, and comparison of AURC against each subject's full-coverage error rate (`R_all`).

### Run an existing dataset

From the repository root, after preparing the subject NPZ files and checkpoints:

```bash
# PAMAP2
python downstream_experiments.py \
  --dataset-name pamap \
  --dataset-root downstream_datasets/pamap2/pamap_subject_

# USC-HAD
python downstream_experiments.py \
  --dataset-name usc \
  --dataset-root downstream_datasets/usc/usc_subject_

# REALWORLD
python downstream_experiments.py \
  --dataset-name realworld \
  --dataset-root downstream_datasets/realworld/realworld_subject_

# HARTH
python downstream_experiments.py \
  --dataset-name harth \
  --dataset-root downstream_datasets/harth/harth_subject_

# CAPTURE-24 (150 subjects)
python downstream_experiments.py \
  --dataset-name capture24 \
  --dataset-root downstream_datasets/capture24/capture24_subject_
```

To change the number of inner subject-grouped calibration folds:

```bash
python downstream_experiments.py \
  --dataset-name pamap \
  --dataset-root downstream_datasets/pamap2/pamap_subject_ \
  --alignment-inner-folds 3
```

**These are evaluator invocations, not data download commands.** The source datasets must be obtained from their respective providers and transformed into the NPZ schema above. The expected CAPTURE-24 files are indexed `1..150`; the study's pipeline excludes participant `151`. Ensure that your selected activity/intensity labels and recording locations match the intended experimental protocol if you want to reproduce the published comparisons.

### Alignment scores

For each upstream uncertainty source, the linear probe evaluates five risk-ranking scores:

| ID | Score | Meaning |
| --- | --- | --- |
| **A** | Raw HR uncertainty | Use upstream uncertainty directly; no error calibration |
| **B** | Logistic(HR uncertainty) | Learn how HR uncertainty relates to HAR error on training subjects |
| **C** | HAR predictive entropy | Uncertainty from the downstream classifier alone |
| **D** | Logistic(HR uncertainty, HAR entropy) | Fuse both uncertainty signals |
| **E** | Logistic(HAR entropy) | Calibrated HAR-only baseline; controls for the logistic alignment step |

The principal fusion contrast is **D vs E** (does HR uncertainty add information beyond HAR entropy after comparable alignment?). **B vs A** isolates the value of aligning the upstream score. The outer test subject is excluded from training the HAR probe, the normalization used by logistic calibration, and the error-alignment models.

### Re-analyze results without rerunning inference

```bash
python downstream_experiments.py \
  --dataset-name pamap \
  --analyze-json downstream_results/results_pamap.json
```

The existing JSON must contain the per-subject results needed by the statistics functions.

To recompute **retrieval/zero-shot** metrics while preserving classification and alignment results from an earlier compatible run:

```bash
python downstream_experiments.py \
  --dataset-name pamap \
  --dataset-root downstream_datasets/pamap2/pamap_subject_ \
  --reuse-classification-json downstream_results/results_pamap.json
```

Only reuse a previous JSON when the dataset, subjects, preprocessing, checkpoints, and classification protocol are identical. The script checks part of the metadata, but this is not a general provenance check.

### Expected downstream outputs

Files are written under `downstream_results/`:

```text
results_<dataset>.json
classification_df_<dataset>.csv
zero_shot_df_<dataset>.csv
omnibus_tests_<dataset>.csv
posthoc_holm_<dataset>.csv
alignment_pairwise_holm_<dataset>.csv
aurc_vs_random_<dataset>.csv
```

The JSON preserves per-subject metrics, regression-head comparisons and alignment information. The CSV files contain summaries and tests. To regenerate the representation-quality plot from multiple results JSON files, see the metric-key compatibility note below:

```bash
python plot_r1_vs_f1.py \
  downstream_results/results_pamap.json \
  downstream_results/results_usc.json \
  downstream_results/results_realworld.json \
  downstream_results/results_harth.json \
  downstream_results/results_capture24.json \
  --output-dir figures
```

The script `plot_r_auroc_transfer.py` is an **older R-AUROC analysis** and is not the current AURC figure generator; do not treat its output as interchangeable with the AURC statistics above.

## 3. Evaluate a new HAR dataset

A new downstream dataset **does not need HR measurements**. It needs labeled accelerometer windows organized by subject. The recommended path is:

1. Convert acceleration to the same physical units (normally **g**), choose a consistent sensor/axis convention, and resample safely to **30 Hz**.
2. Segment the recordings into **10-second / 300-sample** windows without crossing subject, activity or recording-gap boundaries.
3. Save one `my_subject_<id>.npz` per subject with `X` and `y` arrays.
4. Register the subject IDs in `get_df_ids()`.
5. Run `downstream_experiments.py` with the dataset name and file prefix.

### Example: convert already-resampled 30-Hz CSV to NPZ

This small **preparation example** assumes the CSV has these columns:

```text
user_id, timestamp, acc_x, acc_y, acc_z, target
```

`timestamp` is **seconds** on a monotonic clock within each recording (not milliseconds or text). Rows must be sampled at a nominal 30 Hz. The example detects large temporal gaps and activity transitions, builds **non-overlapping** windows, and creates a global activity-label map. It does **not** perform anti-alias filtering or resampling; do that first if the original data are at a different rate.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

FS = 30
WINDOW = 10 * FS
OUT = Path('downstream_datasets/mydataset')
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv('mydataset_30hz.csv').sort_values(['user_id', 'timestamp'])
labels = sorted(df['target'].dropna().unique().tolist())
label_to_id = {label: idx for idx, label in enumerate(labels)}

for user_id, user_df in df.groupby('user_id', sort=False):
    user_df = user_df.reset_index(drop=True).copy()

    # Split at changes of activity or temporal gaps (> 1.5 expected sample intervals).
    new_segment = (
        user_df['target'].ne(user_df['target'].shift())
        | user_df['timestamp'].diff().gt(1.5 / FS)
    )
    user_df['segment'] = new_segment.cumsum()

    windows, targets = [], []
    for _, segment in user_df.groupby('segment', sort=False):
        activity = segment['target'].iloc[0]
        if pd.isna(activity):
            continue
        xyz = segment[['acc_x', 'acc_y', 'acc_z']].to_numpy(dtype=np.float32)
        for start in range(0, len(xyz) - WINDOW + 1, WINDOW):
            window = xyz[start:start + WINDOW]
            if not np.isfinite(window).all():
                continue
            windows.append(window.T)                 # (3, 300)
            targets.append(label_to_id[activity])

    if not windows:
        raise ValueError(f'No valid 10-second windows for subject {user_id}')
    np.savez_compressed(
        OUT / f'my_subject_{user_id}.npz',
        X=np.stack(windows).astype(np.float32),
        y=np.asarray(targets, dtype=np.int64),
    )

(OUT / 'label_map.json').write_text(json.dumps(
    {str(i): str(label) for label, i in label_to_id.items()}, indent=2
))
```

Check a subject file before running an experiment:

```python
import numpy as np

with np.load('downstream_datasets/mydataset/my_subject_1.npz') as data:
    print('X:', data['X'].shape, data['X'].dtype)  # (N, 3, 300) float32
    print('y:', data['y'].shape, data['y'].dtype)  # (N,) integers
    assert data['X'].ndim == 3 and data['X'].shape[1:] == (3, 300)
    assert len(data['X']) == len(data['y'])
    assert np.isfinite(data['X']).all()
```

This example assumes integer subject IDs such as `1`, `2`, `3`, ...; if yours are strings, either map them to integers or adapt the subject-ID discovery logic. **At least three subjects** are needed for nested subject-grouped alignment, and each outer-training fold must contain multiple activity classes.

### Register and evaluate the dataset

In `downstream_experiments.py`, extend `get_df_ids()` to recognize the new name. For example, add the following branch **before the function ends** (adjust subject IDs to match files):

```python
elif df_type == 1 and dataset_name.lower() == 'mydataset':
    return [1, 2, 3, 4]  # all subject IDs actually present
```

Then run:

```bash
python downstream_experiments.py \
  --dataset-name mydataset \
  --dataset-root downstream_datasets/mydataset/my_subject_ \
  --alignment-inner-folds 2
```

The code reads `my_subject_1.npz` through `my_subject_4.npz`, generates the 0% / 25% / 100% embedding and uncertainty results, evaluates global cosine retrieval and LOSO classification, and saves `downstream_results/results_mydataset.json` plus CSV reports.

**For an unbiased comparison:** define the activity label set consistently before running, never mix test-subject windows into the classifier/calibration training folds, and report differences in sensor placement, label definitions and collection conditions. The pipeline's LOSO split controls subject-level classification leakage, but a new dataset may still differ materially from the Oxford training domain.

## Results and interpretation

### Metrics

| Metric | Where | Preferred direction | Interpretation |
| --- | --- | --- | --- |
| **HR MSE** | Upstream risk–coverage | Lower | Average squared error between standardized predicted and observed HR on retained windows |
| **Macro-R@1** | Downstream retrieval | Higher | Unweighted average of per-class correct top-1 cosine-neighbor retrieval rates |
| **Macro-F1** | Downstream classification | Higher | Unweighted average F1 over HAR classes for each LOSO test subject |
| **AURC** | Retrieval / classification | Lower | Average cumulative binary error rate over all retained coverage levels, ordered by increasing uncertainty |
| **`R_all`** | Retrieval / classification | Lower | Observed **full-coverage binary error rate** (the random-ranking AURC reference for that evaluation unit) |

AURC ranks **low uncertainty first**. A useful uncertainty score should tend to produce `AURC < R_all` for the same subject and prediction target. `R_all` is **not** `1 − Macro-F1`, and in class-balanced retrieval it is **not** `1 − Macro-R@1`; it uses the ordinary binary error frequency for the evaluated predictions.

The statistical exports use subjects as paired units, rather than treating thousands of overlapping or consecutive windows as statistically independent. Paired comparisons are performed **within a dataset**; results from different datasets should not be treated as matched observations.

### Practical interpretation of cross-task uncertainty

- **A vs B:** Does learning the relationship between upstream HR uncertainty and HAR errors improve risk ordering?
- **D vs E:** Does upstream HR uncertainty contain information complementary to the downstream classifier's entropy?
- **AURC vs `R_all`:** Does an uncertainty-based selection rule outperform random selection at the subject level?
- **Macro-R@1 vs Macro-F1:** How do changes in the learned representation relate to changes in downstream predictive performance?

Better upstream HR regression, better representation retrieval, and better downstream uncertainty ranking are **distinct empirical outcomes**; none implies the others automatically.

## Known issues and reproducibility notes

The observations below reflect the repository source inspected on **2026-11-02**. They are included so that commands in this README are not mistaken for a guarantee that a fresh checkout will run without study-specific files or small compatibility edits.

1. **Upstream 25%/100% argument mismatch.** In `upstream_hr.py`, `make_upstream_results(...)` is called with `batch_size=args.inference_batch_size`, but `benchmark_upstream_finetuned.py` currently defines `make_upstream_results(output_size, encoder_path)` **without a `batch_size` parameter**. For the smallest compatibility fix, replace:

   ```python
   frame = make_upstream_results(
       OUTPUT_SIZE[head], checkpoint, batch_size=args.inference_batch_size)
   ```

   with:

   ```python
   frame = make_upstream_results(OUTPUT_SIZE[head], checkpoint)
   ```

   This enables the current helper signature, **but the CLI `--inference-batch-size` then does not actually bound fine-tuned inference memory**. To use that flag as intended, extend the helper to process each subject's windows in batches.

2. **Retrieval plot metric-name mismatch.** Current `downstream_experiments.py` saves per-subject retrieval as **`Macro-R@1`**, while `plot_r1_vs_f1.py` reads **`R@1`**. In `plot_r1_vs_f1.py`, change the `_stats(...)` call inside `extract_summary()` from:

   ```python
   _stats(r1_by_id, "R@1", context)
   ```

   to:

   ```python
   _stats(r1_by_id, "Macro-R@1", context)
   ```

   where `context` denotes the **existing third argument**; change only the metric string in the actual source line. If you are plotting an older JSON, inspect which key it contains first.

3. **Legacy AUROC plotting script.** `plot_r_auroc_transfer.py` still reads `R-AUROC` / `c_auroc`. The current downstream evaluator stores **AURC**, so that legacy script is **not** a valid AURC plotting command without rewriting its input/metric logic.

4. **Static input paths and subject IDs.** Upstream file lists, `user_stats.json`, fine-tuned checkpoint paths, and downstream subject lists are not yet consolidated into a single configuration file. Adjust the relevant constants and lists to match your actual data.

5. **Checkpoints and dataset licenses.** The source code alone cannot regenerate model weights or private/third-party datasets. Provide authorized checkpoints and data locally, and respect each dataset's distribution terms. No model accuracy or statistical significance is promised for a new dataset without running its evaluation.

6. **Memory and runtime.** Retrieval is an exact, GPU-based nearest-neighbor search over the full embedding gallery, using `float32` and bounded matrix-multiplication blocks. Although this avoids allocating a full all-pairs matrix, very large datasets still require significant CPU RAM, GPU memory, and compute. Fine-tuned upstream inference currently loads a full subject's windows, and the full downstream pipeline repeats inference and nested LOSO fits across models and subjects. Start by validating input files and checkpoints before a full run.

7. **Reproducibility metadata.** For publication-grade replication, save the Git commit hash, CUDA/PyTorch versions, preprocessing and label mappings, subject lists, model checkpoint hashes, seed settings, and final JSON/CSV outputs alongside the analysis. The scripts fix several local seeds (for example, the HAR probe and random-shuffle helpers), but a fully deterministic GPU execution environment is not enforced globally.

## Acknowledgments

U-HARnet uses the Oxford / [OxWearables `ssl-wearables`](https://github.com/OxWearables/ssl-wearables) pretrained `harnet10` encoder and investigates cross-task uncertainty transfer from HR estimation to HAR. Please acknowledge the upstream encoder authors and cite the relevant datasets and any resulting U-HARnet publication when using the code in research.

**Repository:** https://github.com/DiegoSilvaFR/U-HARnet

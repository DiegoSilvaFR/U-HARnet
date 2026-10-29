"""Compare Insight/HR vs Oxford on Capture24 using pre-extracted features."""

import gc
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 4
CLASS_NAMES = [
    "sleep",
    "light",
    "moderate-vigorous",
    "sedentary",
]

FEATURE_DIRS = {
    "insight": Path("backbone_feature_chunks/insight_hr"),
 #   "oxford": Path("backbone_feature_chunks/oxford_ssl"),
}

TEST_CHUNK_TOKENS = [
    "P101_P110",
    "P111_P120",
    "P121_P130",
    "P131_P140",
    "P141_P150",
]

BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024
EPOCHS = 12
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
USE_CLASS_WEIGHTS = True

OUTPUT_CSV = Path("capture24_backbone_linear_probe_comparison_2.csv")


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Chunk handling
# ============================================================
def is_test_chunk(filename: str | Path) -> bool:
    return any(
        token in str(filename)
        for token in TEST_CHUNK_TOKENS
    )


def split_feature_chunks(feature_dir: str | Path):
    feature_dir = Path(feature_dir)

    if not feature_dir.exists():
        raise FileNotFoundError(
            f"Feature directory does not exist: {feature_dir}"
        )

    files = sorted(feature_dir.glob("*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No .npz files found in: {feature_dir}"
        )

    train_files = [
        file
        for file in files
        if not is_test_chunk(file.name)
    ]

    test_files = [
        file
        for file in files
        if is_test_chunk(file.name)
    ]

    if not train_files:
        raise RuntimeError(
            f"No training chunks found in: {feature_dir}"
        )

    if not test_files:
        raise RuntimeError(
            f"No test chunks found in: {feature_dir}"
        )

    return train_files, test_files


def infer_feature_dim(chunk_file: str | Path) -> int:
    with np.load(chunk_file) as data:
        X = data["X"]

        if X.ndim != 2:
            raise ValueError(
                f"Expected extracted features with shape (N, D), "
                f"but {chunk_file} has X.shape={X.shape}."
            )

        return int(X.shape[1])


def label_counts_from_chunks(
    chunk_files,
    num_classes: int = NUM_CLASSES,
):
    counts = np.zeros(
        num_classes,
        dtype=np.int64,
    )

    for chunk_file in chunk_files:
        with np.load(chunk_file) as data:
            y = np.asarray(
                data["y"],
                dtype=np.int64,
            )

        unique = np.unique(y)

        if np.any(unique < 0) or np.any(unique >= num_classes):
            raise ValueError(
                f"Unexpected labels in {chunk_file}: {unique}. "
                f"Expected 0..{num_classes - 1}."
            )

        counts += np.bincount(
            y,
            minlength=num_classes,
        )[:num_classes]

    return counts


def balanced_class_weights(counts):
    counts = np.asarray(
        counts,
        dtype=np.float64,
    )

    if np.any(counts == 0):
        raise ValueError(
            f"At least one class has zero training samples: {counts}"
        )

    n_samples = counts.sum()
    n_classes = len(counts)

    weights = n_samples / (
        n_classes * counts
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


# ============================================================
# Linear probe
# ============================================================
class LinearProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        self.classifier = nn.Linear(
            input_dim,
            num_classes,
        )

    def forward(self, X):
        return self.classifier(X)


def fit_linear_probe(
    feature_dir: str | Path,
):
    set_seed()

    train_files, _ = split_feature_chunks(
        feature_dir
    )

    feature_dim = infer_feature_dim(
        train_files[0]
    )

    class_counts = label_counts_from_chunks(
        train_files
    )

    class_weights = balanced_class_weights(
        class_counts
    )

    model = LinearProbe(
        input_dim=feature_dim,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=(
            class_weights
            if USE_CLASS_WEIGHTS
            else None
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        f"Feature dimension: {feature_dim}"
    )

    print(
        "Training class counts:",
        class_counts,
    )

    print(
        "Class weights:",
        class_weights.detach().cpu().numpy(),
    )

    for epoch in range(EPOCHS):
        model.train()

        files_this_epoch = list(
            train_files
        )

        random.shuffle(
            files_this_epoch
        )

        running_loss = 0.0
        n_samples = 0

        for chunk_file in files_this_epoch:
            # X here is ALREADY EXTRACTED FEATURES.
            # No backbone forward pass is performed.
            with np.load(chunk_file) as data:
                X_np = np.asarray(
                    data["X"],
                    dtype=np.float32,
                )

                y_np = np.asarray(
                    data["y"],
                    dtype=np.int64,
                )

            dataset = TensorDataset(
                torch.from_numpy(X_np),
                torch.from_numpy(y_np),
            )

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(
                    DEVICE,
                    non_blocking=True,
                )

                y_batch = y_batch.to(
                    DEVICE,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = model(
                    X_batch
                )

                loss = criterion(
                    logits,
                    y_batch,
                )

                loss.backward()
                optimizer.step()

                batch_n = len(y_batch)

                running_loss += (
                    loss.item() * batch_n
                )

                n_samples += batch_n

            del X_np
            del y_np
            del dataset
            del loader

            gc.collect()

        epoch_loss = (
            running_loss
            / max(n_samples, 1)
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"- loss={epoch_loss:.4f}"
        )

    return model


@torch.no_grad()
def evaluate_linear_probe(
    model,
    feature_dir: str | Path,
):
    _, test_files = split_feature_chunks(
        feature_dir
    )

    model.eval()

    y_true_all = []
    y_pred_all = []

    for chunk_file in test_files:
        with np.load(chunk_file) as data:
            X_np = np.asarray(
                data["X"],
                dtype=np.float32,
            )

            y_np = np.asarray(
                data["y"],
                dtype=np.int64,
            )

        dataset = TensorDataset(
            torch.from_numpy(X_np),
            torch.from_numpy(y_np),
        )

        loader = DataLoader(
            dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(
                X_batch
            )

            y_pred = torch.argmax(
                logits,
                dim=1,
            )

            y_true_all.append(
                y_batch.numpy()
            )

            y_pred_all.append(
                y_pred.cpu().numpy()
            )

        del X_np
        del y_np
        del dataset
        del loader

        gc.collect()

    y_true = np.concatenate(
        y_true_all
    )

    y_pred = np.concatenate(
        y_pred_all
    )

    return y_true, y_pred


# ============================================================
# Metrics
# ============================================================
def summarize_predictions(
    y_true,
    y_pred,
    experiment_name: str,
):
    print("\n" + "=" * 80)
    print(experiment_name)
    print("=" * 80)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        average=None,
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        normalize="true",
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"true_{name}"
            for name in CLASS_NAMES
        ],
        columns=[
            f"pred_{name}"
            for name in CLASS_NAMES
        ],
    )

    print(
        "Normalized confusion matrix:"
    )

    print(
        cm_df.round(4).to_string()
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "mvpa_f1": float(per_class_f1[2]),
        **{
            f"f1_{CLASS_NAMES[i]}": float(
                per_class_f1[i]
            )
            for i in range(NUM_CLASSES)
        },
    }


# ============================================================
# Main experiment
# ============================================================
def main():
    set_seed()

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Using PRE-EXTRACTED backbone features only."
    )

    print(
        "X.npy is not loaded anywhere in this script.\n"
    )

    results = {}

    for backbone_name in [
        "insight",
        "oxford",
    ]:
        print("\n" + "#" * 80)
        print(
            f"BACKBONE: {backbone_name.upper()}"
        )
        print("#" * 80)

        feature_dir = FEATURE_DIRS[
            backbone_name
        ]

        train_files, test_files = (
            split_feature_chunks(
                feature_dir
            )
        )

        print(
            f"Feature directory: {feature_dir}"
        )

        print(
            f"Train chunks: {len(train_files)}"
        )

        print(
            f"Test chunks : {len(test_files)}"
        )

        model = fit_linear_probe(
            feature_dir
        )

        y_true, y_pred = (
            evaluate_linear_probe(
                model,
                feature_dir,
            )
        )

        results[backbone_name] = (
            summarize_predictions(
                y_true,
                y_pred,
                experiment_name=(
                    f"{backbone_name.upper()} "
                    "- frozen linear probe"
                ),
            )
        )

        del model
        cleanup()

    comparison_df = pd.DataFrame(
        results
    ).T

    comparison_df.index.name = (
        "backbone"
    )

    comparison_df = (
        comparison_df
        .sort_values(
            "macro_f1",
            ascending=False,
        )
        .reset_index()
    )

    print("\n" + "=" * 80)
    print(
        "FINAL BACKBONE COMPARISON"
    )
    print("=" * 80)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    comparison_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"\nSaved results to: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()

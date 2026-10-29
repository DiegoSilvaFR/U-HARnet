import os
import argparse
import random
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score
from scipy.stats import t as scipy_student_t

from sslearning.models.accNet import Resnet, DropoutHead
from sslearning.pytorchtools import EarlyStopping,EarlyStoppingHead

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)

def compute_ENMO(X):
    
    X = np.asarray(X)

    # X shape expected: (n, 3, time)
    enmo_raw = np.sqrt(X[:, 0, :]**2 + X[:, 1, :]**2 + X[:, 2, :]**2).mean(axis=1)

    enmo = np.maximum(enmo_raw - 1, 0)

    return 1000 * enmo
# -----------------------------
# Dataset
# -----------------------------
def get_user_stats():
    with open("user_stats.json", "r") as file:
        return json.load(file)

class HRDataset(Dataset):
    
    """
    Each row in file_list.csv points to one subject-level .npz file with:
      - X: float32 array [n_windows, 3, T]
      - hr: float32 array [n_windows]

    We sample a fixed number of windows per subject at each __getitem__,
    following Oxford's subject-balanced batching idea.
    """

    def __init__(
        self,
        file_list_path,
        num_sample_per_subject,
        ratio2keep = 1.0,
        weighted_sample = False,
        hr_min = None,
        hr_max = None,
        is_hr_classification = False,
        hr_bins = None,
        expected_t = None,
        is_enmo_regression = False,
        is_z_hr = False
    ):
        df = pd.read_csv(file_list_path)

        if "file_list" in df.columns:
            self.file_list = df["file_list"].tolist()
        elif "path" in df.columns:
            self.file_list = df["path"].tolist()
        else:
            raise ValueError("file_list.csv must contain a 'file_list' or 'path' column.")

        self.num_sample_per_subject = num_sample_per_subject
        self.ratio2keep = ratio2keep
        self.weighted_sample = weighted_sample
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.is_hr_classification = is_hr_classification
        self.hr_bins = None if hr_bins is None else np.asarray(hr_bins, dtype=np.float32)
        self.expected_t = expected_t
        self.is_enmo_regression = is_enmo_regression
        self.is_z_hr = is_z_hr
        self.user_stats = get_user_stats() if self.is_z_hr is True else {} 

    def __len__(self):
        return len(self.file_list)

    def _sample_indices(self, X, hr, path= None):
        n = len(X)
        if n < self.num_sample_per_subject:
            print(path)
            raise ValueError(
                f"Subject has only {n} windows, but num_sample_per_subject={self.num_sample_per_subject}."
            )

        if not self.weighted_sample:
            return np.random.choice(n, size=self.num_sample_per_subject, replace=False)

     
        bins = np.quantile(hr, [0.25, 0.50, 0.75]).astype(np.float32)
        bins = np.unique(bins)  # avoids problems if quantiles are repeated
        
        hr_bin = np.digitize(hr, bins=bins, right=False)
        
        unique_bins, counts = np.unique(hr_bin, return_counts=True)
        count_map = dict(zip(unique_bins, counts))
        
        weights = np.array([1.0 / count_map[b] for b in hr_bin], dtype=np.float64)
        weights = np.maximum(weights, 1e-12)
        
        p = weights / weights.sum()
        
        return np.random.choice(n, size=self.num_sample_per_subject, replace=False, p=p)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        data = np.load(path, allow_pickle=True)

        X = data["X"].astype(np.float32)   # [N, 3, T]
        if self.is_enmo_regression is False:
            hr = data["hr"].astype(np.float32) # [N]
            if self.is_z_hr:
                
                user_id = path.split("user_")[1].replace(".npz","").strip()
                user_stats = self.user_stats[user_id]
                
                hr = (hr - user_stats['mean'])/user_stats['std']

        else:
            hr = compute_ENMO(X)

        if X.ndim != 3:
            raise ValueError(f"Expected X.ndim == 3 in {path}, got {X.shape}")

        # Safety: fix [N, T, 3] -> [N, 3, T] if needed
        if X.shape[1] != 3 and X.shape[2] == 3:
            X = np.transpose(X, (0, 2, 1))

        if X.shape[1] != 3:
            raise ValueError(f"Expected X.shape[1] == 3 in {path}, got {X.shape}")
            

        sel = self._sample_indices(X, hr, path)

        X = torch.from_numpy(X[sel]).float().contiguous()

        if self.is_hr_classification:
            if self.hr_bins is None:
                raise ValueError("For classification, hr_bins must be provided.")
            hr_classes = np.digitize(hr, self.hr_bins)
            hr = torch.from_numpy(hr_classes[sel]).long().contiguous()
        else:
            hr = torch.from_numpy(hr[sel]).float().contiguous()

        return X, hr


def hr_subject_collate(batch):
    X = torch.cat([item[0] for item in batch], dim=0).contiguous()   # [B*S, 3, T]
    hr = torch.cat([item[1] for item in batch], dim=0).contiguous()  # [B*S]
    return X, hr


# -----------------------------
# Probabilistic regression helpers
# -----------------------------

GAUSSIAN_LOG_VAR_MIN = -10.0
GAUSSIAN_LOG_VAR_MAX = 10.0
STUDENT_T_MIN_SCALE = 1e-3
STUDENT_T_MIN_DF = 2.01





@torch.no_grad()
def predict_mc_dropout(model, X, num_samples=50):
    """Return sample-wise MC mean and head epistemic variance/std.

    Works with 0%, partial (e.g., 25%), and full (100%) fine-tuned backbones.
    At inference the backbone runs once in eval mode; only the dropout head
    is sampled repeatedly. Accordingly, the estimated uncertainty reflects
    stochasticity of the head, not the backbone's parameter uncertainty.
    Variance uses 1/T (unbiased=False). Observation-noise variance is not added.

    The function leaves the model in eval mode, like evaluate().
    """
    if num_samples < 2:
        raise ValueError("MC inference requires at least 2 samples.")
    if not isinstance(model.classifier, DropoutHead):
        raise ValueError("predict_mc_dropout requires a DropoutHead classifier.")
    model.eval()
    feats = model.feature_extractor(X).reshape(X.shape[0], -1)
    # Online moments avoid retaining a [T, batch] tensor on the GPU.
    mean = None
    m2 = None
    for sample_index in range(num_samples):
        mu = model.classifier(feats, mc_dropout=True)[:, 0]
        if mean is None:
            mean = mu.clone()
            m2 = torch.zeros_like(mu)
        else:
            delta = mu - mean
            mean = mean + delta / (sample_index + 1)
            m2 = m2 + delta * (mu - mean)
    variance = (m2 / num_samples).clamp_min(0.0)
    return {
        "mu": mean,
        "epistemic_variance": variance,
        "epistemic_std": torch.sqrt(variance),
    }


def decode_regression_output(raw_output, regression_loss):
    """
    Convert the network output into interpretable distribution parameters.

    mse / mc_dropout:
        raw_output[:, 0] -> mu

    gaussian:
        raw_output[:, 0] -> mu
        raw_output[:, 1] -> log(sigma^2)

    student_t:
        raw_output[:, 0] -> mu
        raw_output[:, 1] -> unconstrained scale
        raw_output[:, 2] -> unconstrained degrees of freedom
    """
    if raw_output.ndim == 1:
        raw_output = raw_output.unsqueeze(1)

    if regression_loss in ("mse", "mc_dropout"):
        if raw_output.shape[1] != 1:
            raise ValueError(
                f"MSE regression expects 1 output, got shape {tuple(raw_output.shape)}"
            )
        mu = raw_output[:, 0]
        return {"mu": mu}

    if regression_loss == "gaussian":
        if raw_output.shape[1] != 2:
            raise ValueError(
                f"Gaussian regression expects 2 outputs [mu, log_var], "
                f"got shape {tuple(raw_output.shape)}"
            )

        mu = raw_output[:, 0]
        log_var = raw_output[:, 1].clamp(
            min=GAUSSIAN_LOG_VAR_MIN,
            max=GAUSSIAN_LOG_VAR_MAX,
        )
        sigma = torch.exp(0.5 * log_var)

        return {
            "mu": mu,
            "log_var": log_var,
            "sigma": sigma,
        }

    if regression_loss == "student_t":
        if raw_output.shape[1] != 3:
            raise ValueError(
                f"Student-t regression expects 3 outputs [mu, raw_scale, raw_df], "
                f"got shape {tuple(raw_output.shape)}"
            )

        mu = raw_output[:, 0]

        # Positive scale.
        scale = F.softplus(raw_output[:, 1]) + STUDENT_T_MIN_SCALE

        # df > 2 gives a finite predictive variance, which is useful when
        # interpreting/calibrating uncertainty.
        df = STUDENT_T_MIN_DF + F.softplus(raw_output[:, 2])

        return {
            "mu": mu,
            "scale": scale,
            "df": df,
        }

    raise ValueError(
        f"Unsupported regression_loss={regression_loss}. "
        "Use 'mse', 'gaussian', 'student_t', or 'mc_dropout'."
    )


def regression_loss_fn(
    raw_output,
    target,
    regression_loss,
):
    """
    Return (loss, decoded_parameters).

    Gaussian NLL:
        0.5 * [log(2*pi) + log_var + (y-mu)^2 / exp(log_var)]

    Student-t NLL:
        -log StudentT(y | df, mu, scale)
    """
    params = decode_regression_output(raw_output, regression_loss)
    mu = params["mu"]

    if regression_loss in ("mse", "mc_dropout"):
        loss = F.mse_loss(mu, target)

    elif regression_loss == "gaussian":
        log_var = params["log_var"]
        sq_error = (target - mu).pow(2)

        nll = 0.5 * (
            math.log(2.0 * math.pi)
            + log_var
            + sq_error * torch.exp(-log_var)
        )
        loss = nll.mean()

    elif regression_loss == "student_t":
        dist = torch.distributions.StudentT(
            df=params["df"],
            loc=mu,
            scale=params["scale"],
            validate_args=False,
        )
        loss = -dist.log_prob(target).mean()

    else:
        raise ValueError(f"Unsupported regression_loss={regression_loss}")

    return loss, params


def _inverse_softplus(value):
    """
    Numerically stable inverse of softplus for positive scalar values.
    """
    value = max(float(value), 1e-6)

    if value > 20.0:
        return value

    return math.log(math.expm1(value))


def initialize_uncertainty_head(
    model,
    num_outputs,
    regression_loss,
    is_z_hr,
):
    """
    Initialize uncertainty channels conservatively.

    Why:
    - With raw HR targets, an initial Gaussian sigma near 1 BPM makes the
      first NLL enormous when residuals are tens of BPM.
    - Starting with a reasonable constant uncertainty makes optimization much
      more stable while still allowing the model to learn window-specific
      uncertainty immediately.

    The mean channel is left with the model's normal initialization.
    """
    if regression_loss in ("mse", "mc_dropout"):
        return

    output_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.Linear) and module.out_features == num_outputs
    ]

    if not output_layers:
        print(
            "Warning: could not find the final nn.Linear layer for uncertainty "
            "initialization. Continuing with the model's default initialization."
        )
        return

    head = output_layers[-1]

    # HR z-score is naturally O(1). Raw HR residuals at initialization can be
    # tens of BPM, so use a wider initial scale in that case.
    initial_scale = 1.0 if is_z_hr else 10.0

    with torch.no_grad():
        
        if regression_loss == "gaussian":
            # output[1] is directly log variance.
            head.weight[1].zero_()
            if head.bias is not None:
                head.bias[1] = math.log(initial_scale ** 2)

        elif regression_loss == "student_t":
            # output[1] is transformed with softplus into scale.
            head.weight[1].zero_()
            if head.bias is not None:
                desired_pre_softplus = max(
                    initial_scale - STUDENT_T_MIN_SCALE,
                    1e-6,
                )
                head.bias[1] = _inverse_softplus(desired_pre_softplus)

            # Start with moderately heavy tails: nu ~ 5.
            head.weight[2].zero_()
            if head.bias is not None:
                initial_df = 5.0
                desired_pre_softplus = max(
                    initial_df - STUDENT_T_MIN_DF,
                    1e-6,
                )
                head.bias[2] = _inverse_softplus(desired_pre_softplus)


def prediction_interval_coverage(
    y_true,
    mu,
    regression_loss,
    sigma = None,
    scale = None,
    df = None,
):
    """
    Empirical coverage of central 50%, 80%, and 95% prediction intervals.
    """
    coverages = {}

    for nominal in (0.50, 0.80, 0.95):
        if regression_loss == "gaussian":
            # Standard-normal central interval quantiles.
            z = {
                0.50: 0.6744897501960817,
                0.80: 1.2815515655446004,
                0.95: 1.959963984540054,
            }[nominal]
            half_width = z * sigma

        elif regression_loss == "student_t":
            q = (1.0 + nominal) / 2.0
            half_width = scipy_student_t.ppf(q, df) * scale

        else:
            continue

        lower = mu - half_width
        upper = mu + half_width
        empirical = np.mean((y_true >= lower) & (y_true <= upper))
        coverages[f"coverage_{int(nominal * 100)}"] = float(empirical)

    return coverages


# -----------------------------
# Metrics / loops
# -----------------------------
@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    classification_loss_fn,
    is_hr_classification = False,
    regression_loss = "mse",
    mc_samples = 50,
):
    model.eval()

    losses = []
    y_true = []
    y_pred = []

    uncertainty_1 = []
    uncertainty_2 = []

    for X, hr in loader:
        X = X.to(device, dtype=torch.float32).contiguous()

        if is_hr_classification:
            hr = hr.to(device, dtype=torch.long)
            pred = model(X)
            loss = classification_loss_fn(pred, hr)
            pred_class = torch.argmax(pred, dim=1)

            losses.append(loss.item())
            y_true.append(hr.cpu())
            y_pred.append(pred_class.cpu())

        else:
            hr = hr.to(device, dtype=torch.float32)
            if regression_loss == "mc_dropout":
                params = predict_mc_dropout(model, X, num_samples=mc_samples)
                # Early stopping uses MSE of the MC predictive mean.
                loss = F.mse_loss(params["mu"], hr)
                uncertainty_1.append(params["epistemic_std"].cpu())
            else:
                raw_output = model(X)
                loss, params = regression_loss_fn(
                    raw_output=raw_output,
                    target=hr,
                    regression_loss=regression_loss,
                )

            losses.append(loss.item())
            y_true.append(hr.cpu())
            y_pred.append(params["mu"].cpu())

            if regression_loss == "gaussian":
                uncertainty_1.append(params["sigma"].cpu())
            elif regression_loss == "student_t":
                uncertainty_1.append(params["scale"].cpu())
                uncertainty_2.append(params["df"].cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    if is_hr_classification:
        acc = float((y_true == y_pred).mean())
        precision = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        recall = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        return {
            "loss": float(np.mean(losses)),
            "acc": acc,
            "precision": precision,
            "recall": recall,
        }

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    metrics = {
        "loss": float(np.mean(losses)),
        "mae": mae,
        "rmse": rmse,
    }

    if regression_loss == "mc_dropout":
        epistemic_std = torch.cat(uncertainty_1).numpy()
        metrics["mean_epistemic_std"] = float(np.mean(epistemic_std))
        metrics["median_epistemic_std"] = float(np.median(epistemic_std))
        metrics["mean_epistemic_variance"] = float(np.mean(epistemic_std ** 2))
        # MSE + dropout alone estimates head uncertainty, not observation
        # noise; do not report these as calibrated prediction intervals.

    if regression_loss == "gaussian":
        sigma = torch.cat(uncertainty_1).numpy()
        metrics["mean_sigma"] = float(np.mean(sigma))
        metrics["median_sigma"] = float(np.median(sigma))
        metrics.update(
            prediction_interval_coverage(
                y_true=y_true,
                mu=y_pred,
                regression_loss=regression_loss,
                sigma=sigma,
            )
        )

    elif regression_loss == "student_t":
        scale = torch.cat(uncertainty_1).numpy()
        df = torch.cat(uncertainty_2).numpy()

        metrics["mean_scale"] = float(np.mean(scale))
        metrics["median_scale"] = float(np.median(scale))
        metrics["mean_df"] = float(np.mean(df))
        metrics["median_df"] = float(np.median(df))
        metrics.update(
            prediction_interval_coverage(
                y_true=y_true,
                mu=y_pred,
                regression_loss=regression_loss,
                scale=scale,
                df=df,
            )
        )

    return metrics


def set_train_mode_respecting_frozen_backbone(model):
    """
    Put the model in training mode while keeping frozen backbone stages in
    eval mode. This is important for partial fine-tuning because otherwise
    BatchNorm running statistics in frozen Oxford stages would still change
    after model.train().
    """
    model.train()

    selected_blocks = getattr(model, "_trainable_feature_blocks", None)
    if selected_blocks is None:
        return

    selected_blocks = set(selected_blocks)

    if not selected_blocks:
        model.feature_extractor.eval()
        model.classifier.train()
        return

    for name, child in model.feature_extractor.named_children():
        if name in selected_blocks:
            child.train()
        else:
            child.eval()

    if hasattr(model, "classifier"):
        model.classifier.train()


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    classification_loss_fn,
    is_hr_classification,
    regression_loss = "mse",
):
    set_train_mode_respecting_frozen_backbone(model)

    losses = []
    y_true = []
    y_pred = []

    for X, hr in loader:
        X = X.to(device, dtype=torch.float32).contiguous()

        if is_hr_classification:
            hr = hr.to(device, dtype=torch.long)
            pred = model(X)
            loss = classification_loss_fn(pred, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_class = torch.argmax(pred, dim=1)
            losses.append(loss.item())
            y_true.append(hr.detach().cpu())
            y_pred.append(pred_class.detach().cpu())

        else:
            hr = hr.to(device, dtype=torch.float32)
            raw_output = model(X)

            loss, params = regression_loss_fn(
                raw_output=raw_output,
                target=hr,
                regression_loss=regression_loss,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            y_true.append(hr.detach().cpu())
            y_pred.append(params["mu"].detach().cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    if is_hr_classification:
        acc = float((y_true == y_pred).mean())
        precision = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        recall = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        return {
            "loss": float(np.mean(losses)),
            "acc": acc,
            "precision": precision,
            "recall": recall,
        }

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "loss": float(np.mean(losses)),
        "mae": mae,
        "rmse": rmse,
    }


# -----------------------------
# Model helpers
# -----------------------------

def infer_expected_t(epoch_len):
    mapping = {5: 150, 10: 300, 30: 900}
    if epoch_len not in mapping:
        raise ValueError(f"Unsupported epoch_len={epoch_len}. Use 5, 10 or 30.")
    return mapping[epoch_len]


def create_local_resnet(num_outputs, resnet_version, epoch_len, device):
    
    model = Resnet(
        output_size=num_outputs,
        resnet_version=resnet_version,
        epoch_len=epoch_len,
        is_mtl=False,
        is_eva=False,
    ).to(device)
#     model = CompactResidualTCN(
#     output_size=num_outputs,
#     in_channels=3,
#     embedding_size=512,
#     dropout_p=0.1,
# ).to(device)
    
    return model.float()


def load_oxford_backbone_from_hub(epoch_len, num_outputs, device):
    repo = "OxWearables/ssl-wearables"
    if epoch_len == 5:
        entry = "harnet5"
    elif epoch_len == 10:
        entry = "harnet10"
    elif epoch_len == 30:
        entry = "harnet30"
    else:
        raise ValueError(f"Unsupported epoch_len={epoch_len}. Use 5, 10 or 30.")

    model = torch.hub.load(
        repo,
        entry,
        pretrained=True,
        class_num=num_outputs,
        my_device=device,
    )
    return model.float().to(device)


def load_backbone_only_from_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state[k] = v

    backbone_state = {
        k: v for k, v in clean_state.items()
        if k.startswith("feature_extractor.")
    }

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    print(f"Loaded backbone from checkpoint: {ckpt_path}")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model


def build_model(args, num_outputs, device):
    if args.init_mode == "scratch":
        model = create_local_resnet(
            num_outputs=num_outputs,
            resnet_version=args.resnet_version,
            epoch_len=args.epoch_len,
            device=device,
        )
    elif args.init_mode == "oxford_hub":
        model = load_oxford_backbone_from_hub(
            epoch_len=args.epoch_len,
            num_outputs=num_outputs,
            device=device,
        )
    elif args.init_mode == "checkpoint":
        if args.pretrained_backbone_path is None:
            raise ValueError("--pretrained-backbone-path is required when --init-mode checkpoint")
        model = create_local_resnet(
            num_outputs=num_outputs,
            resnet_version=args.resnet_version,
            epoch_len=args.epoch_len,
            device=device,
        )
        model = load_backbone_only_from_checkpoint(model, args.pretrained_backbone_path, device)
    else:
        raise ValueError(f"Unsupported init_mode={args.init_mode}")

    if args.regression_loss == "mc_dropout":
        # Infer embedding width from the loaded head, rather than assuming
        # 1024: harnet5 and compact local ResNets can use different widths.
        input_layers = [m for m in model.classifier.modules() if isinstance(m, nn.Linear)]
        if not input_layers:
            raise ValueError("Cannot infer the classifier's input embedding width.")
        model.classifier = DropoutHead(
            input_size=input_layers[0].in_features,
            nn_size=args.mc_hidden_size,
            output_size=num_outputs,
            p=args.mc_dropout_p,
        ).float().to(device)

    return model


def get_head_parameters(model):
    if hasattr(model, "classifier"):
        return list(model.classifier.parameters())
    raise ValueError("Model has no attribute 'classifier'.")


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def configure_oxford_partial_finetuning(
    model,
    target_backbone_fraction = 0.25,
):
    """
    Freeze the Oxford feature extractor, then unfreeze the LAST complete
    parameterized feature-extractor stage(s) until their cumulative parameter
    count reaches approximately ``target_backbone_fraction`` of the original
    backbone.

    Why whole stages instead of individual tensors:
      * preserves residual blocks as coherent trainable units;
      * keeps earlier convolutional/BatchNorm stages fully frozen;
      * avoids training only half of a residual stage.

    The downstream classifier/regression head is always fully trainable.
    Flatten operations contain no parameters, so the trainable "flat/FC"
    part corresponds to the classifier head.
    """
    if not (0.0 <= target_backbone_fraction <= 1.0):
        raise ValueError(
            "target_backbone_fraction must be in [0, 1], "
            f"got {target_backbone_fraction}."
        )

    # Freeze the complete pretrained Oxford backbone first.
    for p in model.feature_extractor.parameters():
        p.requires_grad = False

    # Keep the new HR head trainable.
    for p in get_head_parameters(model):
        p.requires_grad = True

    parameterized_blocks = []
    for name, child in model.feature_extractor.named_children():
        n_params = count_parameters(child)
        if n_params > 0:
            parameterized_blocks.append((name, child, n_params))

    if not parameterized_blocks:
        raise RuntimeError("feature_extractor has no parameterized blocks.")

    total_backbone_params = sum(item[2] for item in parameterized_blocks)
    target_params = target_backbone_fraction * total_backbone_params

    selected = []
    selected_params = 0

    # Start from the deepest convolutional stage and move backwards only if
    # needed to get close to the requested trainable fraction.
    for name, child, n_params in reversed(parameterized_blocks):
        if target_backbone_fraction == 0.0:
            break
        selected.append((name, child, n_params))
        selected_params += n_params

        if selected_params >= target_params:
            break

    selected_names = [name for name, _, _ in selected]

    for _, child, _ in selected:
        for p in child.parameters():
            p.requires_grad = True

    # Used by set_train_mode_respecting_frozen_backbone() so frozen BatchNorm
    # statistics do not drift during training.
    model._trainable_feature_blocks = tuple(selected_names)

    head_params = sum(
        p.numel() for p in get_head_parameters(model) if p.requires_grad
    )
    total_model_params = count_parameters(model)
    trainable_model_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    actual_backbone_fraction = selected_params / total_backbone_params

    print("\nOxford partial fine-tuning configuration")
    print("-----------------------------------------")
    print(f"Requested trainable backbone fraction : {target_backbone_fraction:.1%}")
    print(f"Actual trainable backbone fraction    : {actual_backbone_fraction:.1%}")
    print(f"Backbone parameters                  : {total_backbone_params:,}")
    print(f"Trainable backbone parameters        : {selected_params:,}")
    print(f"Trainable HR-head parameters         : {head_params:,}")
    print(f"Total model parameters               : {total_model_params:,}")
    print(f"Total trainable model parameters     : {trainable_model_params:,}")
    print("Trainable deepest feature blocks     :")
    for name, _, n_params in reversed(selected):
        print(f"  - {name}: {n_params:,} parameters")

    return {
        "selected_blocks": selected_names,
        "backbone_total_params": total_backbone_params,
        "backbone_trainable_params": selected_params,
        "backbone_trainable_fraction": actual_backbone_fraction,
        "head_trainable_params": head_params,
        "total_model_params": total_model_params,
        "total_trainable_params": trainable_model_params,
    }


def configure_finetuning(args, model):
    """Set requires_grad flags before the optimizer is built."""
    # MC dropout uses the SAME backbone fine-tuning regime as MSE/Gaussian/Student-t.
    # In the 0% regime only the dropout regression head is trained;
    # in the 25%/100% regimes selected backbone blocks are trained as well.
    zero_backbone = (
        args.freeze_backbone
        or (args.init_mode == "oxford_hub" and args.oxford_trainable_fraction == 0.0)
    )
    if zero_backbone:
        for p in model.feature_extractor.parameters():
            p.requires_grad = False
        for p in get_head_parameters(model):
            p.requires_grad = True
        model._trainable_feature_blocks = tuple()
        model.feature_extractor.eval()
        return {
            "mode": "frozen_backbone",
            "selected_blocks": [],
            "backbone_total_params": count_parameters(model.feature_extractor),
            "backbone_trainable_params": 0,
            "backbone_trainable_fraction": 0.0,
            "head_trainable_params": sum(p.numel() for p in get_head_parameters(model)),
        }

    # Scratch training keeps the original behavior: train everything.
    if args.init_mode == "scratch":
        for p in model.parameters():
            p.requires_grad = True
        model._trainable_feature_blocks = None
        return {"mode": "scratch_full"}

    # Oxford default for this script: train only the deepest convolutional
    # stage(s), targeting ~25% of original backbone parameters, plus the head.
    if args.init_mode == "oxford_hub":
        info = configure_oxford_partial_finetuning(
            model=model,
            target_backbone_fraction=args.oxford_trainable_fraction,
        )
        info["mode"] = "oxford_partial"
        return info

    # Preserve prior checkpoint behavior unless --freeze-backbone was used.
    for p in model.parameters():
        p.requires_grad = True
    model._trainable_feature_blocks = None
    return {"mode": "checkpoint_full"}


def build_optimizer(
    model,
    lr,
    weight_decay,
    backbone_lr_mult,
):
    """
    Use a smaller LR for whichever backbone parameters are trainable and the
    base LR for the new HR classifier/regression head. Frozen parameters are
    excluded entirely.
    """
    backbone_params = [
        p for p in model.feature_extractor.parameters() if p.requires_grad
    ]
    head_params = [
        p for p in get_head_parameters(model) if p.requires_grad
    ]

    param_groups = []

    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": lr * backbone_lr_mult,
                "name": "backbone",
            }
        )

    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": lr,
                "name": "head",
            }
        )

    if not param_groups:
        raise RuntimeError("No trainable parameters were found.")

    return torch.optim.Adam(
        param_groups,
        weight_decay=weight_decay,
        amsgrad=True,
    )


# -----------------------------
# Main
# -----------------------------

def print_args(args):
    """Print all parsed command-line arguments."""
    print("\n" + "=" * 70)
    print("ARGUMENTS")
    print("=" * 70)

    for key, value in sorted(vars(args).items()):
        print(f"{key:30s}: {value}")

    print("=" * 70 + "\n")


def validate_training_args(parser, args):
    if not 0.0 <= args.oxford_trainable_fraction <= 1.0:
        parser.error("--oxford-trainable-fraction must be in [0, 1].")
    if args.regression_loss == "mc_dropout":
        if args.hr_classification:
            parser.error("MC dropout mode is for HR/ENMO regression only.")
        if args.init_mode == "scratch":
            parser.error("MC dropout needs --init-mode oxford_hub or checkpoint.")
        if args.mc_samples < 2:
            parser.error("--mc-samples must be at least 2.")
        if args.mc_hidden_size < 1:
            parser.error("--mc-hidden-size must be positive.")
        if not 0.0 < args.mc_dropout_p < 1.0:
            parser.error("--mc-dropout-p must be strictly between 0 and 1.")


def main():
    """Train HR/ENMO regression or HR classification.

nohup python -u /home/jovyan/ssl_wearables/train_hr_uncertainty_oxford_partial25.py --train-file-list /home/jovyan/ssl_wearables/data/ssl/ssl_ai4wellness/data/train/file_list.csv --dev-file-list /home/jovyan/ssl_wearables/data/ssl/ssl_ai4wellness/data/dev/file_list.csv --test-file-list /home/jovyan/ssl_wearables/data/ssl/ssl_ai4wellness/data/test/file_list.csv --save-dir /home/jovyan/ssl_wearables/output_hr --gpu 0 --epochs 50 --lr 1e-4 --batch-subject-num 4 --num-sample-per-subject 1500 --num-sample-test 250 --regression-loss mc_dropout --init-mode oxford_hub --oxford-trainable-fraction 0.25 --version 25 --z-hr > log_dropout_25.txt 2>&1 &

    """
 
    parser = argparse.ArgumentParser() 

    parser.add_argument("--train-file-list", type=str, required=True)
    parser.add_argument("--dev-file-list", type=str, required=True)
    parser.add_argument("--test-file-list", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--batch-subject-num", type=int, default=4)
    parser.add_argument("--num-sample-per-subject", type=int, default=250)
    parser.add_argument("--num-sample-test", type=int, default=100)
    parser.add_argument("--ratio2keep", type=float, default=1.0)
    parser.add_argument("--weighted-sample", action="store_true")
    parser.add_argument("--enmo-regression", action="store_true")
    parser.add_argument("--z-hr", action = "store_true")
    parser.add_argument(
        "--regression-loss",
        type=str,
        default="mse",
        choices=["mse", "gaussian", "student_t", "mc_dropout"],
        help=(
            "Regression objective. "
            "mse -> 1 output (mu); "
            "gaussian -> 2 outputs (mu, log_var); "
            "student_t -> 3 outputs (mu, raw_scale, raw_df); "
            "mc_dropout -> nonlinear dropout head, MSE training, MC mean and "
            "head epistemic uncertainty at evaluation (0%, 25%, or 100% fine-tuning)."
        ),
    )
    parser.add_argument("--mc-dropout-p", type=float, default=0.5)
    parser.add_argument("--mc-hidden-size", type=int, default=512)
    parser.add_argument(
        "--mc-samples", type=int, default=50,
        help="Stochastic head passes per dev/test batch in mc_dropout mode.",
    )

    parser.add_argument("--resnet-version", type=int, default=1)
    parser.add_argument("--epoch-len", type=int, default=10, choices=[5, 10, 30])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--version", type=int, default=1)

    parser.add_argument("--hr-min", type=float, default=35.0)
    parser.add_argument("--hr-max", type=float, default=220.0)

    # task mode
    parser.add_argument("--hr-classification", action="store_true")
    parser.add_argument(
        "--hr-bins",
        type=str,
        default="100",
        help="Comma-separated thresholds for HR classification, e.g. '100' or '80,100' or '50,65,80,95,110'",
    )

    # initialization / fine-tuning mode
    parser.add_argument(
        "--init-mode",
        type=str,
        default="scratch",
        choices=["scratch", "oxford_hub", "checkpoint"],
        help="scratch = random init local Resnet; oxford_hub = load Oxford pretrained backbone from torch.hub; checkpoint = load backbone from local checkpoint.",
    )
    parser.add_argument("--pretrained-backbone-path", type=str, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument(
        "--oxford-trainable-fraction",
        type=float,
        default=0.25,
        help=(
            "For --init-mode oxford_hub, unfreeze the deepest complete "
            "feature-extractor stage(s) until approximately this fraction "
            "of the original Oxford backbone parameters is trainable. "
            "0 freezes the whole backbone and trains only the head for any "
            "regression objective; 1 trains the full backbone. Default: 0.25."
        ),
    )

    args = parser.parse_args()
    validate_training_args(parser, args)
    print_args(args)
    
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    is_hr_classification = args.hr_classification
    is_enmo_regression = args.enmo_regression
    main_target = 'ENMO' if is_enmo_regression else 'HR'
    is_z_hr = args.z_hr 
    expected_t = infer_expected_t(args.epoch_len)

    if is_hr_classification:
        if args.regression_loss != "mse":
            raise ValueError(
                "--regression-loss applies only to regression. "
                "Do not combine it with --hr-classification."
            )

        hr_bins = np.array(
            [float(x) for x in args.hr_bins.split(",") if x.strip() != ""],
            dtype=np.float32,
        )
        classes = len(hr_bins) + 1
        suffix = "cls"

    else:
        hr_bins = None

        num_outputs_by_loss = {
            "mse": 1,
            "gaussian": 2,
            "student_t": 3,
            "mc_dropout": 1,
        }
        classes = num_outputs_by_loss[args.regression_loss]
        suffix = f"reg_{args.regression_loss}"

    train_ds = HRDataset(
        file_list_path=args.train_file_list,
        num_sample_per_subject=args.num_sample_per_subject,
        ratio2keep=args.ratio2keep,
        weighted_sample=args.weighted_sample,
        hr_min=args.hr_min,
        hr_max=args.hr_max,
        is_hr_classification=is_hr_classification,
        hr_bins=hr_bins,
        expected_t=expected_t,
        is_enmo_regression= is_enmo_regression,
        is_z_hr = is_z_hr
    )

    dev_ds = HRDataset(
        file_list_path=args.dev_file_list,
        num_sample_per_subject=args.num_sample_test,
        ratio2keep=1.0,
        weighted_sample=False,
        hr_min=args.hr_min,
        hr_max=args.hr_max,
        is_hr_classification=is_hr_classification,
        hr_bins=hr_bins,
        expected_t=expected_t,
        is_enmo_regression= is_enmo_regression,
        is_z_hr = is_z_hr
    )

    test_ds = HRDataset(
        file_list_path=args.test_file_list,
        num_sample_per_subject=args.num_sample_test,
        ratio2keep=1.0,
        weighted_sample=False,
        hr_min=args.hr_min,
        hr_max=args.hr_max,
        is_hr_classification=is_hr_classification,
        hr_bins=hr_bins,
        expected_t=expected_t,
        is_enmo_regression= is_enmo_regression,
        is_z_hr = is_z_hr
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_subject_num,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=hr_subject_collate,
        worker_init_fn=worker_init_fn,
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_subject_num,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=hr_subject_collate,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_subject_num,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=hr_subject_collate,
        worker_init_fn=worker_init_fn,
    )

    model = build_model(args=args, num_outputs=classes, device=device)

    if not is_hr_classification:
        initialize_uncertainty_head(
            model=model,
            num_outputs=classes,
            regression_loss=args.regression_loss,
            is_z_hr=is_z_hr,
        )

    finetune_info = configure_finetuning(
        args=args,
        model=model,
    )

    if args.regression_loss == "mc_dropout" and not isinstance(model.classifier, DropoutHead):
        raise RuntimeError("MC dropout requires a DropoutHead classifier.")

    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr_mult=args.backbone_lr_mult,
    )

    classification_loss_fn = nn.CrossEntropyLoss() if is_hr_classification else None

    init_suffix = {
        "scratch": "scratch",
        "oxford_hub": "oxfordhub",
        "checkpoint": "ckpt",
    }[args.init_mode]

    # Keep prior 0%-fine-tuning filenames unchanged. Give new MC-dropout
    # 25% and 100% runs explicit suffixes to prevent checkpoint collisions
    # when --version is reused for multiple fine-tuning regimes.
    if finetune_info["mode"] == "frozen_backbone":
        init_suffix += "_ft0"
    elif args.regression_loss == "mc_dropout":
        if args.init_mode == "oxford_hub":
            requested_percent = 100 * args.oxford_trainable_fraction
        else:
            requested_percent = 100  # non-frozen checkpoint backbones train fully
        init_suffix += f"_ft{requested_percent:g}"

    best_path = os.path.join(args.save_dir, f"{main_target}_{suffix}_{init_suffix}_{args.version}_final.mdl")
    hist_path = os.path.join(args.save_dir, f"{main_target}_evolution_parameters_{args.version}_{suffix}_{init_suffix}_final.json")

    config_path = best_path.replace("_final.mdl", "_config.json")
    with open(config_path, "w") as file:
        json.dump({
            "args": vars(args),
            "finetuning": finetune_info,
            "num_outputs": classes,
            "head_class": type(model.classifier).__name__,
            "mc_uncertainty": (
                "head_epistemic_only; no observation-noise variance"
                if args.regression_loss == "mc_dropout" else None
            ),
        }, file, indent=4)

    if finetune_info["mode"] == "frozen_backbone":

        early_stopping = EarlyStoppingHead(
            patience=args.patience,
            path=best_path,
            verbose=True,
        )

    else:

        early_stopping = EarlyStopping(
            patience=args.patience,
            path=best_path,
            verbose=True,
        )

    if finetune_info["mode"] == "scratch_full":
        print("Training mode: scratch + full training")
    elif finetune_info["mode"] == "frozen_backbone":
        print(f"Training mode: {args.init_mode} + frozen backbone + trainable HR head")
        print("Trainable backbone parameters: 0 (0.0%); backbone BatchNorm frozen")
    elif finetune_info["mode"] == "oxford_partial":
        print(
            "Training mode: Oxford pretrained backbone + deepest convolutional "
            f"stage(s) (~{finetune_info['backbone_trainable_fraction']:.1%} of backbone) "
            "+ trainable HR head"
        )
    else:
        print(f"Training mode: {args.init_mode} + full fine-tuning")

    if args.regression_loss == "mc_dropout":
        print(
            f"MC dropout: hidden={args.mc_hidden_size}, p={args.mc_dropout_p}, "
            f"passes={args.mc_samples}; stochastic head only at inference; "
            f"backbone fine-tuning mode={finetune_info['mode']}"
        )

    evolution_parameters = {}

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            classification_loss_fn=classification_loss_fn,
            is_hr_classification=is_hr_classification,
            regression_loss=args.regression_loss,
        )
        dev_metrics = evaluate(
            model=model,
            loader=dev_loader,
            device=device,
            classification_loss_fn=classification_loss_fn,
            is_hr_classification=is_hr_classification,
            regression_loss=args.regression_loss,
            mc_samples=args.mc_samples,
        )
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            classification_loss_fn=classification_loss_fn,
            is_hr_classification=is_hr_classification,
            regression_loss=args.regression_loss,
            mc_samples=args.mc_samples,
        )

        if is_hr_classification:
            msg = (
                f"[{epoch+1:03d}/{args.epochs:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.4f} "
                f"train_precision={train_metrics['precision']:.4f} "
                f"train_recall={train_metrics['recall']:.4f} | "
                f"dev_loss={dev_metrics['loss']:.4f} "
                f"dev_acc={dev_metrics['acc']:.4f} "
                f"dev_precision={dev_metrics['precision']:.4f} "
                f"dev_recall={dev_metrics['recall']:.4f} | "
                f"test_loss={test_metrics['loss']:.4f} "
                f"test_acc={test_metrics['acc']:.4f} "
                f"test_precision={test_metrics['precision']:.4f} "
                f"test_recall={test_metrics['recall']:.4f}"
            )
            evolution_parameters[epoch] = {
                "train_loss": round(train_metrics["loss"], 4),
                "train_acc": round(train_metrics["acc"], 4),
                "train_precision": round(train_metrics["precision"], 4),
                "train_recall": round(train_metrics["recall"], 4),
                "dev_loss": round(dev_metrics["loss"], 4),
                "dev_acc": round(dev_metrics["acc"], 4),
                "dev_precision": round(dev_metrics["precision"], 4),
                "dev_recall": round(dev_metrics["recall"], 4),
                "test_loss": round(test_metrics["loss"], 4),
                "test_acc": round(test_metrics["acc"], 4),
                "test_precision": round(test_metrics["precision"], 4),
                "test_recall": round(test_metrics["recall"], 4),
            }
        else:
            msg = (
                f"[{epoch+1:03d}/{args.epochs:03d}] "
                f"loss={args.regression_loss} | "
                f"train_nll/loss={train_metrics['loss']:.4f} "
                f"train_mae={train_metrics['mae']:.4f} "
                f"train_rmse={train_metrics['rmse']:.4f} | "
                f"dev_nll/loss={dev_metrics['loss']:.4f} "
                f"dev_mae={dev_metrics['mae']:.4f} "
                f"dev_rmse={dev_metrics['rmse']:.4f} | "
                f"test_nll/loss={test_metrics['loss']:.4f} "
                f"test_mae={test_metrics['mae']:.4f} "
                f"test_rmse={test_metrics['rmse']:.4f}"
            )

            if args.regression_loss == "mc_dropout":
                msg += (
                    f" dev_epistemic_std={dev_metrics['mean_epistemic_std']:.4f} | "
                    f"test_epistemic_std={test_metrics['mean_epistemic_std']:.4f}"
                )

            elif args.regression_loss == "gaussian":
                msg += (
                    f" dev_sigma={dev_metrics['mean_sigma']:.4f} "
                    f"dev_cov50={dev_metrics['coverage_50']:.3f} "
                    f"dev_cov80={dev_metrics['coverage_80']:.3f} "
                    f"dev_cov95={dev_metrics['coverage_95']:.3f} | "
                    f"test_sigma={test_metrics['mean_sigma']:.4f} "
                    f"test_cov50={test_metrics['coverage_50']:.3f} "
                    f"test_cov80={test_metrics['coverage_80']:.3f} "
                    f"test_cov95={test_metrics['coverage_95']:.3f}"
                )

            elif args.regression_loss == "student_t":
                msg += (
                    f" dev_scale={dev_metrics['mean_scale']:.4f} "
                    f"dev_df={dev_metrics['mean_df']:.2f} "
                    f"dev_cov50={dev_metrics['coverage_50']:.3f} "
                    f"dev_cov80={dev_metrics['coverage_80']:.3f} "
                    f"dev_cov95={dev_metrics['coverage_95']:.3f} | "
                    f"test_scale={test_metrics['mean_scale']:.4f} "
                    f"test_df={test_metrics['mean_df']:.2f} "
                    f"test_cov50={test_metrics['coverage_50']:.3f} "
                    f"test_cov80={test_metrics['coverage_80']:.3f} "
                    f"test_cov95={test_metrics['coverage_95']:.3f}"
                )

            evolution_parameters[epoch] = {
                "regression_loss": args.regression_loss,
                "train_loss": round(train_metrics["loss"], 4),
                "train_mae": round(train_metrics["mae"], 4),
                "train_rmse": round(train_metrics["rmse"], 4),
                "dev_loss": round(dev_metrics["loss"], 4),
                "dev_mae": round(dev_metrics["mae"], 4),
                "dev_rmse": round(dev_metrics["rmse"], 4),
                "test_loss": round(test_metrics["loss"], 4),
                "test_mae": round(test_metrics["mae"], 4),
                "test_rmse": round(test_metrics["rmse"], 4),
            }

            for split_name, split_metrics in (
                ("dev", dev_metrics),
                ("test", test_metrics),
            ):
                for key in (
                    "mean_sigma",
                    "median_sigma",
                    "mean_scale",
                    "median_scale",
                    "mean_df",
                    "median_df",
                    "mean_epistemic_std",
                    "median_epistemic_std",
                    "mean_epistemic_variance",
                    "coverage_50",
                    "coverage_80",
                    "coverage_95",
                ):
                    if key in split_metrics:
                        evolution_parameters[epoch][f"{split_name}_{key}"] = round(
                            split_metrics[key], 4
                        )

        print(msg)

        early_stopping(dev_metrics["loss"], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(f"Best checkpoint saved to: {best_path}")

    with open(hist_path, "w") as file:
        json.dump(evolution_parameters, file, indent=4)


if __name__ == "__main__":
    main()

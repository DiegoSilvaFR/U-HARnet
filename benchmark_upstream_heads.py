import os
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json
from ssl_wearables.sslearning.models.accNet import DropoutHead, EvaClassifier
import warnings

warnings.filterwarnings('ignore')

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def risk_coverage_curve_regression(
    y_true,
    y_pred,
    uncertainty,
    original,
    coverages=None,
    n_shuffles=10,
    random_state=42,
):
    """
    Compute a regression risk-coverage curve using MSE.

    Samples are sorted from LOWEST to HIGHEST uncertainty.
    Therefore, as coverage decreases, the highest-uncertainty
    samples are progressively rejected.

    For each coverage level, the mean and standard deviation
    are computed from the `original` column for the samples
    that remain after applying the uncertainty threshold.

    The Spearman correlation between uncertainty and squared
    error is also computed separately at each coverage level
    using only the retained samples.

    A shuffled uncertainty baseline is computed by randomly
    permuting uncertainty n_shuffles times.

    Parameters
    ----------
    y_true : array-like
        Ground-truth regression target used to compute MSE.

    y_pred : array-like
        Predicted regression values.

    uncertainty : array-like
        Sample-wise uncertainty score.

    original : array-like
        Original HR values. The mean and standard deviation
        reported at each coverage are computed from this array.

    coverages : array-like or None
        Fractions of samples to retain.

        If None:
            [1.0, 0.9, ..., 0.1]

    n_shuffles : int
        Number of uncertainty shuffles.

    random_state : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns:
            coverage
            n_retained
            n_rejected
            mse
            mse_shuffled
            mse_shuffled_std
            original_mean
            original_std
            uncertainty_threshold
            spearman_corr
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    uncertainty = np.asarray(
        uncertainty,
        dtype=float,
    )

    original = np.asarray(
        original,
        dtype=float,
    )

    # -------------------------------------------------
    # Check lengths
    # -------------------------------------------------

    if not (
        len(y_true)
        == len(y_pred)
        == len(uncertainty)
        == len(original)
    ):
        raise ValueError(
            "y_true, y_pred, uncertainty, and original "
            "must have the same length."
        )

    # -------------------------------------------------
    # Remove invalid samples
    # -------------------------------------------------

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(uncertainty)
        & np.isfinite(original)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    uncertainty = uncertainty[valid]
    original = original[valid]

    n = len(y_true)

    if n == 0:
        raise ValueError(
            "No valid samples remain."
        )

    # -------------------------------------------------
    # Coverage levels
    # -------------------------------------------------

    if coverages is None:
        coverages = np.arange(
            1.0,
            0.0,
            -0.1,
        )

    coverages = np.asarray(
        coverages,
        dtype=float,
    )

    if np.any(coverages <= 0) or np.any(coverages > 1):
        raise ValueError(
            "Coverage values must be in (0, 1]."
        )

    coverages = np.sort(
        coverages
    )[::-1]

    # -------------------------------------------------
    # Squared error
    # -------------------------------------------------

    squared_error = (
        y_pred - y_true
    ) ** 2

    # -------------------------------------------------
    # Sort by uncertainty
    # -------------------------------------------------
    #
    # Lowest uncertainty first.
    # As coverage decreases, highest uncertainty
    # is removed.
    # -------------------------------------------------

    order = np.argsort(
        uncertainty
    )

    uncertainty_sorted = uncertainty[
        order
    ]

    squared_error_sorted = squared_error[
        order
    ]

    original_sorted = original[
        order
    ]

    # -------------------------------------------------
    # Shuffled baseline
    # -------------------------------------------------

    rng = np.random.default_rng(
        random_state
    )

    shuffled_mse = {
        coverage: []
        for coverage in coverages
    }

    for _ in range(n_shuffles):

        shuffled_uncertainty = rng.permutation(
            uncertainty
        )

        shuffled_order = np.argsort(
            shuffled_uncertainty
        )

        shuffled_squared_error = squared_error[
            shuffled_order
        ]

        for coverage in coverages:

            n_retained = max(
                1,
                int(
                    np.ceil(
                        coverage * n
                    )
                ),
            )

            mse_shuffle = np.mean(
                shuffled_squared_error[
                    :n_retained
                ]
            )

            shuffled_mse[
                coverage
            ].append(
                mse_shuffle
            )

    # -------------------------------------------------
    # Real risk-coverage curve
    # -------------------------------------------------

    results = []

    for coverage in coverages:

        n_retained = max(
            1,
            int(
                np.ceil(
                    coverage * n
                )
            ),
        )

        # -------------------------------------------------
        # Remaining samples
        # -------------------------------------------------

        retained_squared_error = squared_error_sorted[
            :n_retained
        ]

        retained_uncertainty = uncertainty_sorted[
            :n_retained
        ]

        retained_original = original_sorted[
            :n_retained
        ]

        # -------------------------------------------------
        # Regression error
        # -------------------------------------------------

        mse = np.mean(
            retained_squared_error
        )

        # -------------------------------------------------
        # Shuffled baseline
        # -------------------------------------------------

        mse_shuffled = np.mean(
            shuffled_mse[
                coverage
            ]
        )

        mse_shuffled_std = np.std(
            shuffled_mse[
                coverage
            ]
        )

        # -------------------------------------------------
        # ORIGINAL HR distribution among remaining points
        # -------------------------------------------------

        original_mean = np.mean(
            retained_original
        )

        original_std = np.std(
            retained_original
        )

        # -------------------------------------------------
        # Uncertainty threshold
        # -------------------------------------------------

        uncertainty_threshold = uncertainty_sorted[
            n_retained - 1
        ]

        # -------------------------------------------------
        # Spearman correlation
        #
        # Correlation between uncertainty and squared error
        # among the samples retained at this coverage level.
        # -------------------------------------------------

        if (
            n_retained >= 2
            and np.std(retained_uncertainty) > 0
            and np.std(retained_squared_error) > 0
        ):
            spearman_corr = pd.Series(
                retained_uncertainty
            ).corr(
                pd.Series(
                    retained_squared_error
                ),
                method="spearman",
            )
        else:
            spearman_corr = np.nan

        # -------------------------------------------------
        # Store
        # -------------------------------------------------

        results.append({
            "coverage":
                n_retained / n,

            "n_retained":
                n_retained,

            "n_rejected":
                n - n_retained,

            "mse":
                mse,

            "mse_shuffled":
                mse_shuffled,

            "mse_shuffled_std":
                mse_shuffled_std,

            "original_mean":
                original_mean,

            "original_std":
                original_std,
            "uncertainty_threshold":
                uncertainty_threshold,

            "spearman_corr":
                spearman_corr,
        })

    return pd.DataFrame(
        results
    )

class EvalHeads:

    def __init__(
        self,
        heads_path,
        mc_samples=50,
        mc_dropout_p=0.5,
    ):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.mc_samples = mc_samples
        self.mc_dropout_p = mc_dropout_p

        self.load_encoder()
        self.heads_path = heads_path
        self.heads = self.load_heads()

    # ============================================================
    # Oxford encoder
    # ============================================================

    def load_encoder(self):

        model = torch.hub.load(
            "OxWearables/ssl-wearables",
            "harnet10",
            pretrained=True,
            class_num=1,
            my_device=self.device,
        )

        model = model.float().to(self.device)

        for p in model.feature_extractor.parameters():
            p.requires_grad = False

        model.feature_extractor.eval()

        self.feature_extractor = model.feature_extractor

        print("Oxford backbone loaded.")

    # ============================================================

    @torch.no_grad()
    def extract_batch_features(self, X):

        self.feature_extractor.eval()

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        X = X.float().to(self.device).contiguous()

        feats = self.feature_extractor(X)

        feats = feats.reshape(
            feats.shape[0],
            -1,
        )

        return feats

    # ============================================================
    # Checkpoint loading
    # ============================================================

    # ============================================================
    
    def load_heads(self):

        heads = {}

        # --------------------------------------------------------
        # Option 1:
        #
        # {
        #     "gaussian": "/path/model.mdl",
        #     "student_t": "/path/model.mdl",
        #     "mc_dropout": "/path/model.mdl",
        # }
        # --------------------------------------------------------

        if isinstance(self.heads_path, dict):

            head_files = self.heads_path

        # --------------------------------------------------------
        # Option 2:
        # directory containing all three heads
        # --------------------------------------------------------

        elif os.path.isdir(self.heads_path):

            head_files = {}

            for filename in os.listdir(self.heads_path):

                if not filename.endswith(
                    (".mdl", ".pt", ".pth")
                ):
                    continue

                name_lower = filename.lower()

                if (
                    "gaussian" in name_lower
                    or "student_t" in name_lower
                    or "student-t" in name_lower
                    or "mc_dropout" in name_lower
                ):

                    name = os.path.splitext(filename)[0]

                    head_files[name] = os.path.join(
                        self.heads_path,
                        filename,
                    )

        else:

            raise ValueError(
                "heads_path must be either a dictionary "
                "or a directory."
            )

        if len(head_files) == 0:
            raise ValueError(
                "No head checkpoints were found."
            )

        for name, path in head_files.items():

            heads[name] = self.load_head(
                name,
                path,
            )

        return heads
    
    def _clean_state_dict(self, checkpoint):

        # ---------------------------------------------------------
        # Case 1: entire nn.Module was saved
        # torch.save(model.classifier, path)
        # ---------------------------------------------------------
        if isinstance(checkpoint, torch.nn.Module):
            checkpoint = checkpoint.state_dict()

        # ---------------------------------------------------------
        # Case 2: checkpoint wrapper
        # ---------------------------------------------------------
        if isinstance(checkpoint, dict):

            possible_keys = [
                "state_dict",
                "model_state_dict",
                "head_state_dict",
                "classifier_state_dict",
                "head",
                "classifier",
                "model",
            ]

            for key in possible_keys:

                if key in checkpoint:

                    candidate = checkpoint[key]

                    if isinstance(candidate, torch.nn.Module):
                        checkpoint = candidate.state_dict()
                        break

                    if isinstance(candidate, dict):
                        checkpoint = candidate
                        break

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Checkpoint does not contain a valid state_dict."
            )

        # ---------------------------------------------------------
        # Keep only tensor parameters
        # ---------------------------------------------------------
        clean_state = {}

        for key, value in checkpoint.items():

            if not torch.is_tensor(value):
                continue

            # DataParallel
            if key.startswith("module."):
                key = key[len("module."):]

            # Possible full-model prefixes
            prefixes = [
                "model.classifier.",
                "model.head.",
                "classifier.",
                "head.",
            ]

            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break

            clean_state[key] = value

        return clean_state


    def load_head(self, name, path):

        name_lower = name.lower().strip()

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        state = self._clean_state_dict(checkpoint)

        # ---------------------------------------------------------
        # Debug information if checkpoint format is unexpected
        # ---------------------------------------------------------
        if "linear1.weight" not in state:

            print(f"\nCheckpoint keys for {name}:")
            print(list(state.keys()))

            raise ValueError(
                f"Could not find 'linear1.weight' in {path}. "
                f"Available keys: {list(state.keys())}"
            )

        # Infer dimensions directly from saved head
        input_size = state["linear1.weight"].shape[1]
        hidden_size = state["linear1.weight"].shape[0]

        # =========================================================
        # Gaussian
        # =========================================================

        if "gaussian" in name_lower:

            head = EvaClassifier(
                input_size=input_size,
                nn_size=hidden_size,
                output_size=2,
            )

        # =========================================================
        # Student-t
        # =========================================================

        elif (
            "student_t" in name_lower
            or "student-t" in name_lower
        ):

            head = EvaClassifier(
                input_size=input_size,
                nn_size=hidden_size,
                output_size=3,
            )

        # =========================================================
        # MC Dropout
        # =========================================================

        elif (
            "mc_dropout" in name_lower
            or "dropout" in name_lower
        ):

            head = DropoutHead(
                input_size=input_size,
                nn_size=hidden_size,
                output_size=1,
            )

        else:

            raise ValueError(
                f"Cannot identify head type from: {name}"
            )

        head = head.float().to(self.device)

        head.load_state_dict(
            state,
            strict=True,
        )

        head.eval()

        return head

    # ============================================================
    # Gaussian
    # ============================================================

    @torch.no_grad()
    def gaussian_head_inference(self, model, feats):

        model.eval()

        output = model(feats)

        mu = output[:, 0]

        log_var = output[:, 1].clamp(
            min=-10.0,
            max=10.0,
        )

        var = torch.exp(log_var)

        return mu, var

    # ============================================================
    # Student-t
    # ============================================================

    @torch.no_grad()
    def student_t_head_inference(self, model, feats):

        model.eval()

        output = model(feats)

        mu = output[:, 0]

        scale = (
            F.softplus(output[:, 1])
            + 1e-3
        )

        df = (
            2.01
            + F.softplus(output[:, 2])
        )

        # Student-t predictive variance
        #
        # Var = scale² * df / (df - 2)

        var = (
            scale.pow(2)
            * df
            / (df - 2.0)
        )

        return mu, var

    # ============================================================
    # MC Dropout
    # ============================================================

    @torch.no_grad()
    def mc_dropout_head_inference(self, model, feats):

        if self.mc_samples < 2:
            raise ValueError(
                "mc_samples must be >= 2."
            )

        # Everything stays in eval mode.
        model.eval()

        predictions = []

        for _ in range(self.mc_samples):

            # Explicitly activate dropout exactly as in training evaluation
            output = model(
                feats,
                mc_dropout=True,
            )

            mu = output[:, 0]

            predictions.append(mu)

        predictions = torch.stack(
            predictions,
            dim=0,
        )

        mu = predictions.mean(
            dim=0,
        )

        var = predictions.var(
            dim=0,
            unbiased=False,
        )

        return mu, var

    # ============================================================
    # Generic head prediction
    # ============================================================

    def model_pred(self, name, model, feats):

        name_lower = name.lower().strip()

        if "gaussian" in name_lower:

            return self.gaussian_head_inference(
                model,
                feats,
            )

        elif "student_t" in name_lower or "student-t" in name_lower:

            return self.student_t_head_inference(
                model,
                feats,
            )

        elif "mc_dropout" in name_lower or "dropout" in name_lower:

            return self.mc_dropout_head_inference(
                model,
                feats,
            )

        else:

            raise ValueError(
                f"Unknown regression head: {name}"
            )

    # ============================================================
    # Final prediction
    # ============================================================

    def predict(self, X, return_feats = False):

    
        # Backbone evaluated only once
        feats = self.extract_batch_features(X)
        batch_size = feats.shape[0]
        
        results = {} if return_feats else {"head_name": [],"mu": [],"var": []}

        for name, model in self.heads.items():

            mu, var = self.model_pred(
                name,
                model,
                feats,
            )

            mu = (
                mu.detach().cpu().numpy()
            )

            var = (
                var.detach()
                .cpu()
                .numpy()
            )
            
            if return_feats:
                
                results[name] = {'var': var.tolist()}
                
            else:    
                
                results["head_name"] += [name] * batch_size
                results["mu"] += mu.tolist()
                results["var"] += var.tolist()
        
        if return_feats:
            
            df = pd.DataFrame(feats.detach().cpu().numpy())
            
            for name, stats in results.items():
                
                df[name] = stats['var']
            
            return df
        
        return pd.DataFrame(results)
    

def take_id(path:str) -> str:
        patten = '/home/jovyan/'
        return path.replace(patten, '')


def benchmark_heads():
    
    heads = {'gaussian': 'ssl_wearables/output_hr/HR_reg_gaussian_oxfordhub_ft0_1_final.mdl',
        'student_t': 'ssl_wearables/output_hr/HR_reg_student_t_oxfordhub_ft0_1_final.mdl',
        'mc_dropout': 'ssl_wearables/output_hr/mc_dropout_head.mdl'}
    eval_heads = EvalHeads(heads)
    
    
    path = "ssl_wearables/data/ssl/ssl_ai4wellness/data/test/file_list.csv"
    df_list = pd.read_csv(path)

    

    users = df_list.file_list.apply(take_id).unique()
    


    dfs = []

    with open("user_stats.json", "r") as file:
        user_stats = json.load(file)


    for user_path in users:

        user_id = (
            user_path
            .replace(
                "ssl_wearables/data/ssl/ssl_ai4wellness/data/test/user_",
                "",
            )
            .replace(".npz", "")
        )

        data = np.load(user_path)

        user_stat = user_stats[user_id]

        y = data["hr"]
        X = data["X"]
        
        
        y_norm = (
            y - user_stat["mean"]
        ) / user_stat["std"]

        # 3 heads × N samples
        df_user = eval_heads.predict(X)

        # Repeat targets for each head
        df_user["y"] = np.tile(
            y_norm,
            len(eval_heads.heads),
        )

        df_user["user_id"] = user_id
        df_user["raw_hr"] = np.tile(
            y,
            len(eval_heads.heads),
        )
        
        dfs.append(df_user)


    df_results = pd.concat(
         dfs,
         ignore_index=True,
    )
    df_results.var = df_results['var']
    
    df_results['error'] = (df_results['y'] - df_results['mu'])**2
    
    return df_results
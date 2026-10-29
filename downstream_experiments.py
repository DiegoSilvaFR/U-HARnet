import pandas as pd
import numpy as np
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_upstream_heads import benchmark_heads, risk_coverage_curve_regression, EvalHeads
from benchmark_upstream_finetuned import make_upstream_results, mc_dropout_finetuned_inference, load_HR_encoder, gaussian_regression_inference, student_t_head_inference
from linear_probe import train_test_linear_probe

def _load_user(dataset_root_path, user):
    
    user_path = dataset_root_path + str(user)+'.npz'
    data = np.load(user_path)

    y = data["y"]
    X = data["X"]
    
    return X,y

def benchmark_downstream_heads(dataset_root_path, users):
    
    heads = {'gaussian': 'ssl_wearables/output_hr/HR_reg_gaussian_oxfordhub_ft0_1_final.mdl',
        'student_t': 'ssl_wearables/output_hr/HR_reg_student_t_oxfordhub_ft0_1_final.mdl',
        'mc_dropout': 'ssl_wearables/output_hr/mc_dropout_head.mdl'}
    
    eval_heads = EvalHeads(heads)

    dfs = []

    for user in users:
        
        X,y =  _load_user(dataset_root_path, user)
        
        # 3 heads × N samples
        df_user = eval_heads.predict(X, return_feats=True)
       
        # Repeat targets for each head
        df_user["y"] = y
           

        df_user["user_id"] = user
        
        dfs.append(df_user)


    df_results = pd.concat(
         dfs,
         ignore_index=True,
    )
    
    return df_results

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


def recall_auroc(embeddings, labels, uncertainty):
    """
    Calculate Recall@1 and R-AUROC.

    Parameters
    ----------
    embeddings : array-like, shape (N, d)
        Embeddings from the pretrained encoder.

    labels : array-like, shape (N,)
        Ground-truth downstream class labels.

    uncertainty : array-like, shape (N,)
        Uncertainty estimates, where larger means
        more uncertain.

    Returns
    -------
    dict
        R@1, R-AUROC, nearest-neighbor indices,
        correctness and cosine distances.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels).reshape(-1)
    uncertainty = np.asarray(
        uncertainty, dtype=float
    ).reshape(-1)

    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError(
            "embeddings must have shape (N, d), N >= 2."
        )

    n = embeddings.shape[0]

    if len(labels) != n or len(uncertainty) != n:
        raise ValueError("Inputs must have the same N.")

    if (
        not np.isfinite(embeddings).all()
        or not np.isfinite(uncertainty).all()
    ):
        raise ValueError("Inputs must be finite.")

    if np.any(np.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError("Embeddings must be nonzero.")

    # Step 1: Find nearest neighbors using cosine distance.
    nn = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(embeddings)

    # X=None automatically excludes self-matches.
    distances, indices = nn.kneighbors(
        X=None, n_neighbors=1
    )
    neighbor_idx = indices[:, 0]

    # Step 2: Check nearest-neighbor correctness.
    correct = labels == labels[neighbor_idx]

    # Step 3: Define the error indicator.
    error = (~correct).astype(int)

    # Step 4: Compute Recall@1.
    r_at_1 = float(correct.mean())

    # Step 5: Compute R-AUROC.
    # Requires both correct and incorrect matches.
    if np.unique(error).size == 2:
        auc = float(
            roc_auc_score(error, uncertainty)
        )
    else:
        auc = float("nan")

    return {
        "R@1": r_at_1,
        "R-AUROC": auc,
        "neighbor_idx": neighbor_idx,
        "correct": correct,
        "cosine_distance": distances[:, 0],
    }

        
def summarize_retrieval_by_subject(user_ids, correct, uncertainty):
    """R@1 and R-AUROC per subject, using the same global NN reference set."""
    user_ids = np.asarray(user_ids).reshape(-1)
    correct = np.asarray(correct, dtype=bool).reshape(-1)
    uncertainty = np.asarray(uncertainty, dtype=float).reshape(-1)

    if not (len(user_ids) == len(correct) == len(uncertainty)):
        raise ValueError("user_ids, correct and uncertainty must align")

    by_subject = {}
    for subject in pd.unique(user_ids):
        mask = user_ids == subject
        c = correct[mask]
        error = (~c).astype(int)
        u = uncertainty[mask]
        by_subject[str(subject)] = {
            "R@1": float(c.mean()),
            "R-AUROC": (
                float(roc_auc_score(error, u))
                if np.unique(error).size == 2 else float("nan")
            ),
        }
    return by_subject


def make_finetuned_embedding(output_size, dataset_root_path, users, encoder_path):
    """ 
    output_size: 1 = mc_dropout
                      2 = gaussian
                      3 = student_t
                      
    finetune_deepth: int
    encoder_paths: list[str]
    
    """
    
    is_mc_dropout = True if output_size ==1 else False 
    encoder = load_HR_encoder(output_size = output_size, ckpt_path = encoder_path, is_mc_dropout = is_mc_dropout)
    
    dfs = []
    
    for user in users:
        
        X,y =  _load_user(dataset_root_path, user)
        
        if output_size ==3:
            
            features, predicted_hr, hr_uncertainty = (
                    student_t_head_inference(
                        encoder,
                        X
                    )
                )
        elif output_size == 2:
            
            features, predicted_hr, hr_uncertainty = (
                    gaussian_regression_inference(
                        encoder,
                        X
                    )
                )
        else:
            
            features, predicted_hr, hr_uncertainty = mc_dropout_finetuned_inference(encoder, X)
             
 
        df_user = pd.DataFrame(features)

        # uncertainty learned during upstream HR task
        df_user["u"] = hr_uncertainty
        df_user['y'] = y
        df_user["user_id"] = user

        dfs.append(df_user)
        
    df_final = pd.concat(dfs,ignore_index=True
)
    return df_final

import torch.nn as nn

class OxfordMTLEntropy:

    def __init__(
        self,
        checkpoint_path="ssl_wearables/model_check_point/mtl_best.mdl",
        device=None,
        batch_size=2048,
    ):

        self.device = torch.device(
            device or (
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        )

        self.batch_size = batch_size

        # Only the three tasks used in the paper
        self.heads = nn.ModuleDict({
            "aot": nn.Linear(1024, 2),
            "permutation": nn.Linear(1024, 2),
            "time_warp": nn.Linear(1024, 2),
        })

        # Load original pretrained weights
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            # Only use this fallback with trusted checkpoints.
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
            )

        state = checkpoint
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

        # Handle original names, including "module." prefixes
        def get_weight(name):
            matches = [
                value for key, value in state.items()
                if key == name or key.endswith("." + name)
            ]

            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one checkpoint tensor "
                    f"for {name}, found {len(matches)}"
                )

            return matches[0]

        original_names = {
            "aot": "aot_h",
            "permutation": "permute_h",
            "time_warp": "time_w_h",
        }

        for name, original_name in original_names.items():

            weights = {
                "weight": get_weight(
                    f"{original_name}.linear1.weight"
                ),
                "bias": get_weight(
                    f"{original_name}.linear1.bias"
                ),
            }

            self.heads[name].load_state_dict(
                weights, strict=True
            )

        self.heads.to(self.device)
        self.heads.eval()

        print("Oxford MTL heads loaded successfully.")

    @staticmethod
    def entropy(logits):
        """
        Shannon entropy using natural logarithms.
        Maximum entropy = log(2).
        """

        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()

        return -(p * log_p).sum(dim=1)

    @torch.no_grad()
    def predict(self, features, return_components=False):
        """
        Parameters
        ----------
        features : numpy.ndarray or torch.Tensor
            Backbone features with shape (N, 1024)
            or (N, 1024, 1).

        return_components : bool
            If True, also return each task's
            normalized entropy.

        Returns
        -------
        pd.DataFrame
            Default: one column, 'mtl_entropy'.

            Optional: individual normalized entropies.
        """

        features = torch.as_tensor(
            features, dtype=torch.float32
        )

        if features.ndim == 3:
            features = features.flatten(start_dim=1)

        if features.ndim != 2 or features.shape[1] != 1024:
            raise ValueError(
                f"Expected features (N, 1024), "
                f"got {tuple(features.shape)}"
            )

        if not torch.isfinite(features).all():
            raise ValueError(
                "Features contain NaN or infinite values."
            )

        outputs = []

        for start in range(0, len(features), self.batch_size):

            batch = features[
                start:start + self.batch_size
            ].to(self.device)

            # Three original classification heads
            logits_aot = self.heads["aot"](batch)
            logits_perm = self.heads["permutation"](batch)
            logits_tw = self.heads["time_warp"](batch)

            # Individual Shannon entropies
            h_aot = self.entropy(logits_aot)
            h_perm = self.entropy(logits_perm)
            h_tw = self.entropy(logits_tw)

            # Global normalized entropy
            u_mtl = (
                h_aot + h_perm + h_tw
            ) / (3.0 * math.log(2.0))

            # Optional individual normalized values
            scores = torch.stack([
                u_mtl,
                h_aot / math.log(2.0),
                h_perm / math.log(2.0),
                h_tw / math.log(2.0),
            ], dim=1)

            outputs.append(scores.cpu().numpy())

        columns = [
            "mtl_entropy",
            "aot_entropy",
            "permutation_entropy",
            "time_warp_entropy",
        ]

        if not outputs:
            result = pd.DataFrame(columns=columns)
        else:
            result = pd.DataFrame(
                np.concatenate(outputs, axis=0),
                columns=columns,
            )

        if return_components:
            return result

        return result["mtl_entropy"].values





def recall_auroc_gpu(
    embeddings,
    labels,
    uncertainty,
    gpu_id=0,
    query_batch_size=1024,
    database_batch_size=200_000,
    precision="highest",
):
    """
    Exact GPU cosine nearest-neighbor search using PyTorch.

    Parameters
    ----------
    embeddings : array-like, shape (N, d)
        Encoder embeddings.

    labels : array-like, shape (N,)
        Ground-truth labels.

    uncertainty : array-like, shape (N,)
        Uncertainty estimates (higher = more uncertain).

    gpu_id : int
        CUDA device ID.

    query_batch_size : int
        Number of queries per batch.

    database_batch_size : int
        Number of database embeddings per batch.

    precision : str
        "highest": disable TF32 for better accuracy.
        "high": enable TF32 when supported.

    Returns
    -------
    dict
        R@1, R-AUROC, neighbor_idx,
        correct, cosine_distance.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if precision not in ("highest", "high"):
        raise ValueError(
            "precision must be 'highest' or 'high'."
        )

    if query_batch_size < 1 or database_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")

    device = torch.device("cuda:{}".format(gpu_id))

    def to_numpy(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        return np.asarray(v)

    labels = to_numpy(labels).reshape(-1)

    uncertainty = np.asarray(
        to_numpy(uncertainty),
        dtype=np.float64
    ).reshape(-1)

    # Transfer embeddings to GPU in float32.
    x = torch.as_tensor(
        embeddings,
        dtype=torch.float32,
        device=device
    ).detach()

    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("Invalid embedding shape.")

    n, d = x.shape

    if d < 1 or len(labels) != n or len(uncertainty) != n:
        raise ValueError("Input dimensions do not match.")

    if not torch.isfinite(x).all().item():
        raise ValueError("Embeddings contain NaN or Inf.")

    if not np.isfinite(uncertainty).all():
        raise ValueError("Uncertainty contains NaN or Inf.")

    # Normalize embeddings for cosine similarity.
    with torch.no_grad():

        norms = torch.norm(
            x,
            p=2,
            dim=1,
            keepdim=True
        )

        if (
            (norms == 0).any().item()
            or not torch.isfinite(norms).all().item()
        ):
            raise ValueError("Invalid embedding norm.")

        x = (x / norms).contiguous()

        del norms

    # Allocate CPU outputs.
    neighbor_idx = np.empty(n, dtype=np.int64)
    cosine_distance = np.empty(n, dtype=np.float32)

    # Handle TF32 compatibility with older PyTorch.
    matmul_backend = getattr(
        torch.backends.cuda, "matmul", None
    )

    has_tf32 = (
        matmul_backend is not None
        and hasattr(matmul_backend, "allow_tf32")
    )

    if has_tf32:
        previous_tf32 = matmul_backend.allow_tf32
        matmul_backend.allow_tf32 = (
            precision == "high"
        )

    try:
        with torch.no_grad():

            for q_start in range(0, n, query_batch_size):

                q_end = min(q_start + query_batch_size, n)

                queries = x[q_start:q_end]
                b = q_end - q_start

                best_sim = torch.full(
                    (b,),
                    -float("inf"),
                    device=device,
                    dtype=torch.float32
                )

                best_idx = torch.full(
                    (b,),
                    -1,
                    dtype=torch.long,
                    device=device
                )

                # Search entire database in chunks.
                for db_start in range(
                    0, n, database_batch_size
                ):

                    db_end = min(
                        db_start + database_batch_size,
                        n
                    )

                    database = x[db_start:db_end]

                    # GPU matrix multiplication.
                    similarities = torch.mm(
                        queries,
                        database.t()
                    )

                    # Exclude self-matches.
                    overlap_start = max(q_start, db_start)
                    overlap_end = min(q_end, db_end)

                    if overlap_start < overlap_end:

                        ids = torch.arange(
                            overlap_start,
                            overlap_end,
                            device=device,
                            dtype=torch.long
                        )

                        similarities[
                            ids - q_start,
                            ids - db_start
                        ] = -float("inf")

                    # Best neighbor within current chunk.
                    values, indices = torch.max(
                        similarities,
                        dim=1
                    )

                    # Update global nearest neighbor.
                    better = values > best_sim

                    best_sim = torch.where(
                        better,
                        values,
                        best_sim
                    )

                    best_idx = torch.where(
                        better,
                        indices + db_start,
                        best_idx
                    )

                    del similarities, values, indices

                if (best_idx < 0).any().item():
                    raise RuntimeError(
                        "No valid nearest neighbor found."
                    )

                # Copy final batch results to CPU.
                neighbor_idx[q_start:q_end] = (
                    best_idx.cpu().numpy()
                )

                cosine_distance[q_start:q_end] = (
                    (1.0 - best_sim)
                    .clamp(0.0, 2.0)
                    .cpu()
                    .numpy()
                )

    finally:
        if has_tf32:
            matmul_backend.allow_tf32 = previous_tf32

    # Nearest-neighbor correctness.
    correct = labels == labels[neighbor_idx]

    # Error indicator.
    error = (~correct).astype(np.int8)

    # Recall@1.
    r_at_1 = float(correct.mean())

    # R-AUROC.
    if np.unique(error).size == 2:
        auc = float(
            roc_auc_score(error, uncertainty)
        )
    else:
        auc = float("nan")

    return {
        "R@1": r_at_1,
        "R-AUROC": auc,
        "neighbor_idx": neighbor_idx,
        "correct": correct,
        "cosine_distance": cosine_distance,
    }


class DownstreamUncertainty:
    
    def __init__(self, dataset_root, users_ids, dataset_type= 1):
        
        """ dataset_type: int 
        0: CAPTURE-24
        1: other

        Both use LOSO. Dataset type preserves the original epoch settings.
        """
        
        self.dataset_type = dataset_type
        self.epochs = 100 if dataset_type == 0 else 100
        self.dataset_root = dataset_root
        self.users_ids = users_ids
        self.results = {'zero_shot':{}, 'classification':{}}
        self.mlt = OxfordMTLEntropy()
        
    
    def make_0_finetuned_results(self, run_linear_probe=True):
        df = benchmark_downstream_heads(self.dataset_root, self.users_ids)
        z = df[list(range(1024))].values
        y = df.y.values

        # Compute encoder baseline entropy and ONE global NN search: shared by
        # all uncertainty estimators for the frozen 0% encoder.
        df['u_ce'] = np.asarray(self.mlt.predict(z)).reshape(-1)
        results = recall_auroc_gpu(z, y, df['u_ce'].values)
        correct = results['correct']
        error = (~correct).astype(int)
        print("R@1 zero-finetuned done.")

        head_results = {}
        for name in ['CE', 'gaussian', 'student_t', 'mc_dropout']:
            col = 'u_ce' if name == 'CE' else name
            u = np.asarray(df[col].values, dtype=float).reshape(-1)
            auc = float(roc_auc_score(error, u)) if np.unique(error).size == 2 else float('nan')
            head_results[name] = {
                'R@1': float(results['R@1']),
                'R-AUROC': auc,
                'by_subject': summarize_retrieval_by_subject(df.user_id.values, correct, u),
            }
            print(f"Zero-finetuned {name} done.")

        self.results['zero_shot']['finetuned_0_results'] = head_results

        # Remap labels ONLY for linear probe, after zero-shot correctness was
        # determined with original activity labels.
        labels = df.y.unique()
        df['y'] = df.y.map({label: idx for idx, label in enumerate(labels)})
        if run_linear_probe:
            self.results['classification']['finetuned_0_results'] = (
                self._train_test_linear_prob(
                    df, ['gaussian', 'student_t', 'mc_dropout', 'u_ce'],
                    'original_encoder'
                )
            )


    def make_25_100_finetuned_results(self, paths, run_linear_probe=True):
        
        
        for finetune_deepth, encoder_paths in paths.items():
            
            result_per_deepth = {}
            results_LOSO ={}
            
            for encoder_name, encoder_path in encoder_paths.items():
                
                if 'gaussian' in encoder_name:
                    output_size = 2
                elif 'student_t' in encoder_name:
                    output_size = 3
                else:
                    output_size = 1
                    
                df = make_finetuned_embedding(output_size, self.dataset_root, self.users_ids, encoder_path)
                
                labels = df.y.unique()
                map_labels = {labels[i]: i for i in range(len(labels))}
                df['y'] = df.y.map(map_labels)
                u = df[['u']].values

                z = df[[i for i in range(1024)]].values
                y = df.y.values

                results = recall_auroc_gpu(z, y, u)
                print(f"R@1 {encoder_path} done.")
                
                result_per_deepth[encoder_name] = {
                    'R@1': float(results['R@1']),
                    'R-AUROC': float(results['R-AUROC']),
                    'by_subject': summarize_retrieval_by_subject(
                        df.user_id.values, results['correct'], u
                    ),
                }

                if run_linear_probe:
                    results_LOSO[encoder_name] = self._train_test_linear_prob(
                        df, ['u'], encoder_path
                    )

            self.results['zero_shot'][f'finetuned_{finetune_deepth}_results'] = result_per_deepth

            if run_linear_probe:
                self.results['classification'][f'finetuned_{finetune_deepth}_results'] = results_LOSO
            
    def _train_test_linear_prob(self, df, type_u, encoder_name=''):
        """Train/evaluate one linear probe per held-out subject (LOSO).

        Each test set consists exclusively of one subject; training uses all
        remaining subjects. This split is identical across datasets, including
        CAPTURE-24. Keys in the result are the actual subject IDs.
        """
        results_LOSO = {}
        users = df.user_id.unique()
        if len(users) < 2:
            raise ValueError('LOSO requires at least two distinct subjects.')

        feature_columns = list(range(1024))
        for fold_idx, user in enumerate(users, start=1):
            test_mask = df.user_id == user
            df_train, df_test = df.loc[~test_mask], df.loc[test_mask]

            X_train = df_train[feature_columns].values
            X_test = df_test[feature_columns].values
            y_train, y_test = df_train.y.values, df_test.y.values
            u_test = df_test[type_u]

            results_LOSO[user] = _clean_probe_output(
                train_test_linear_probe(
                    X_train, X_test, y_train, y_test, u_test,
                    epochs=self.epochs
                )
            )

            if fold_idx % 10 == 0 or fold_idx == len(users):
                print(f'LOSO {encoder_name}: {fold_idx}/{len(users)} subjects evaluated.')

        print(f'LOSO training for finetuned {encoder_name} done.')
        return results_LOSO

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, datetime


def save_json(data, filename="results.json"):
    """
    Recursively convert data to JSON-compatible types and save it.

    Handles:
    - Nested dictionaries and lists
    - NumPy scalars (float32, float64, int64, etc.)
    - NumPy arrays
    - Pandas DataFrames and Series
    - Tuples and sets
    - NaN, Inf, pd.NA -> None (JSON null)
    - Integer dictionary keys -> strings
    - Dates and timestamps -> ISO strings
    """

    def convert(obj):
        if obj is None or obj is pd.NA or obj is pd.NaT:
            return None

        if isinstance(obj, np.generic):
            return convert(obj.item())

        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None

        if isinstance(obj, (str, int, bool)):
            return obj

        if isinstance(obj, dict):
            return {
                str(convert(k)): convert(v)
                for k, v in obj.items()
            }

        if isinstance(obj, (list, tuple, set, np.ndarray)):
            return [convert(v) for v in obj]

        if isinstance(obj, pd.DataFrame):
            return convert(obj.to_dict(orient="records"))

        if isinstance(obj, pd.Series):
            return convert(obj.to_dict())

        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()

        if isinstance(obj, Path):
            return str(obj)

        raise TypeError(f"Unsupported type: {type(obj).__name__}")

    data = convert(data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
            allow_nan=False
        )

    print('results saved at: ', filename)


import numpy as np
import pandas as pd


# Only these metrics are retained in the statistical outputs:
#   - zero-shot: R@1 and R-AUROC
#   - classification: R-AUROC (when available) and every available F1 score


def _is_f1_metric(name):
    return str(name).lower() == 'f1' or 'f1' in str(name).lower()


def _clean_probe_metrics(metrics):
    """Keep all F1 scores and one consistently oriented AUROC only.

    Historical linear-probe outputs may contain c_auroc (correctness positive).
    Convert it to R-AUROC (retrieval/classification error positive).
    """
    if not isinstance(metrics, dict):
        return {}
    cleaned = {}
    for key, value in metrics.items():
        if _is_f1_metric(key) and np.isscalar(value):
            cleaned[key] = value
    if 'R-AUROC' in metrics:
        cleaned['R-AUROC'] = metrics['R-AUROC']
    elif 'c_auroc' in metrics:
        score = metrics['c_auroc']
        cleaned['R-AUROC'] = 1.0 - float(score) if score is not None else None
    return cleaned


def _clean_probe_output(result):
    """Clean the raw probe output, allowing multi-uncertainty and single heads."""
    if not isinstance(result, dict):
        return result
    if any(_is_f1_metric(key) for key in result):
        return _clean_probe_metrics(result)
    return {
        str(method): _clean_probe_metrics(values)
        for method, values in result.items() if isinstance(values, dict)
    }


def _clean_classification_structure(data):
    """Normalize legacy nested results without discarding F1 measurements."""
    if not data:
        return {}
    if all(str(k).isdigit() for k in data):
        # 0%: {test-unit: {uncertainty-method: {metric: value}}}
        return {
            str(unit): {
                str(method): _clean_probe_metrics(measurements)
                for method, measurements in method_data.items()
            }
            for unit, method_data in data.items()
        }
    # 25/100%: {method: {test-unit: {'u': {metric: value}}}}
    cleaned = {}
    for method, unit_data in data.items():
        cleaned[str(method)] = {}
        for unit, measurements in unit_data.items():
            if isinstance(measurements, dict) and set(measurements) == {'u'}:
                measurements = measurements['u']
            cleaned[str(method)][str(unit)] = _clean_probe_metrics(measurements)
    return cleaned


def _classification_method_units(classification_results):
    """Return {method: {subject-or-fold: metric_dict}} for both layouts."""
    cleaned = _clean_classification_structure(classification_results)
    if not cleaned:
        return {}
    if all(str(key).isdigit() for key in cleaned):
        methods = {}
        for unit, unit_methods in cleaned.items():
            for method, metrics in unit_methods.items():
                methods.setdefault(method, {})[str(unit)] = metrics
        return methods
    return cleaned


def validate_loso_classification_results(results, expected_subjects):
    """Reject legacy group-K-fold classification JSON used by CAPTURE-24.

    Reused/analyzed classification results must have one test unit for every
    subject, in every available finetuning regime and uncertainty method.
    """
    expected = {str(subject) for subject in expected_subjects}
    classification = results.get('classification', {})
    if not classification:
        raise ValueError('No saved classification results to validate.')

    for regime, regime_data in classification.items():
        method_units = _classification_method_units(regime_data)
        if not method_units:
            raise ValueError(f'No classification test units in {regime}.')
        for method, units in method_units.items():
            observed = set(map(str, units))
            if observed != expected:
                raise ValueError(
                    f'{regime}/{method} contains {len(observed)} test units; '
                    f'LOSO requires all {len(expected)} individual subjects. '
                    'These may be legacy grouped-fold results. Re-run the '
                    'CAPTURE-24 pipeline without --reuse-classification-json.'
                )


def per_unit_metrics_df(results):
    """One observation per subject (retrieval) or test fold/subject (probe)."""
    rows = []
    for regime, methods in results.get('zero_shot', {}).items():
        for method, values in methods.items():
            for subject, metrics in values.get('by_subject', {}).items():
                for key in ('R@1', 'R-AUROC'):
                    val = metrics.get(key)
                    if val is not None and np.isfinite(val):
                        rows.append(dict(study='zero_shot', finetuning=regime,
                                         method=method, unit=str(subject),
                                         metric=key, value=float(val)))

    for regime, data in results.get('classification', {}).items():
        for method, units in _classification_method_units(data).items():
            for unit, metrics in units.items():
                for key, value in metrics.items():
                    if (key == 'R-AUROC' or _is_f1_metric(key)) and value is not None:
                        try:
                            val = float(value)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(val):
                            rows.append(dict(study='classification', finetuning=regime,
                                             method=method, unit=str(unit),
                                             metric=key, value=val))
    return pd.DataFrame(rows, columns=[
        'study', 'finetuning', 'method', 'unit', 'metric', 'value'
    ])


def summarize_results_df(results):
    """Separate zero-shot and classification means + sample SD by experimental unit."""
    long = per_unit_metrics_df(results)
    output = {}
    for study in ('classification', 'zero_shot'):
        rows = []
        for regime, condition_results in results.get(study, {}).items():
            methods = (_classification_method_units(condition_results)
                       if study == 'classification' else condition_results)
            for method in methods:
                group = long[(long.study == study) &
                             (long.finetuning == regime) &
                             (long.method == method)]
                row = {'finetuning': regime, 'method': method}
                if study == 'zero_shot':
                    metric_names = ('R@1', 'R-AUROC')
                else:
                    metric_names = sorted(set(group.metric))
                row['n_units'] = int(group.unit.nunique())
                for metric in metric_names:
                    values = group.loc[group.metric == metric, 'value']
                    row[f'{metric}_mean'] = values.mean() if len(values) else np.nan
                    row[f'{metric}_std'] = (
                        values.std(ddof=1) if len(values) > 1 else np.nan
                    )
                rows.append(row)
        if rows:
            output[study] = (pd.DataFrame(rows)
                             .set_index(['finetuning', 'method'])
                             .sort_index())
        else:
            output[study] = pd.DataFrame()
    return output['classification'], output['zero_shot']


def _holm_adjust(p_values):
    """Holm family-wise correction, ignoring undefined p-values."""
    p = np.asarray(p_values, dtype=float)
    corrected = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid):
        order = valid[np.argsort(p[valid])]
        corrected[order] = np.maximum.accumulate(
            np.minimum(1, p[order] * np.arange(len(order), 0, -1))
        )
    return corrected


def compare_paired_results(results, paired_test='wilcoxon'):
    """Friedman omnibus and paired tests (Wilcoxon or paired t), Holm-corrected.

    Participants are paired by subject in all datasets, including CAPTURE-24.
    No sample-window observations are treated as independent replicates.
    """
    from itertools import combinations
    from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel
    if paired_test not in ('wilcoxon', 'paired_t'):
        raise ValueError('paired_test must be wilcoxon or paired_t')
    long = per_unit_metrics_df(results)
    omni, posthoc = [], []
    for (study, metric), sub in long.groupby(['study', 'metric']):
        for comparison, fixed_col, compare_col in (
            ('methods', 'finetuning', 'method'),
            ('finetuning', 'method', 'finetuning'),
        ):
            for fixed, block in sub.groupby(fixed_col):
                wide = block.pivot_table(index='unit', columns=compare_col,
                                         values='value', aggfunc='first')
                if wide.shape[1] < 2:
                    continue
                complete = wide.dropna()
                n, k = complete.shape
                if n < 3:
                    continue
                if k >= 3:
                    if np.all(complete.to_numpy() == complete.to_numpy()[:, :1]):
                        stat, p = 0.0, 1.0
                    else:
                        stat, p = friedmanchisquare(*[
                            complete[c].values for c in complete.columns
                        ])
                    omni.append(dict(study=study, metric=metric,
                                     comparison=comparison, fixed=fixed,
                                     n_units=n, statistic=float(stat), p=float(p)))
                for a, b in combinations(wide.columns, 2):
                    paired = wide[[a, b]].dropna()
                    if len(paired) < 3:
                        continue
                    x = paired[a].to_numpy(dtype=float)
                    y = paired[b].to_numpy(dtype=float)
                    delta = x - y
                    if np.all(delta == 0):
                        stat, p = 0., 1.
                    elif paired_test == 'paired_t':
                        res = ttest_rel(x, y)
                        stat, p = res.statistic, res.pvalue
                    else:
                        res = wilcoxon(delta, alternative='two-sided')
                        stat, p = res.statistic, res.pvalue
                    posthoc.append(dict(study=study, metric=metric,
                                        comparison=comparison, fixed=fixed,
                                        group_a=a, group_b=b, n_pairs=len(paired),
                                        mean_diff=float(delta.mean()), test=paired_test,
                                        statistic=float(stat), p=float(p)))
    omni_df = pd.DataFrame(omni)
    posthoc_df = pd.DataFrame(posthoc)
    for frame in (omni_df, posthoc_df):
        if not frame.empty:
            frame['p_holm'] = np.nan
            for _, ids in frame.groupby(['study', 'metric', 'comparison']).groups.items():
                frame.loc[ids, 'p_holm'] = _holm_adjust(frame.loc[ids, 'p'])
    return omni_df, posthoc_df


import argparse

def print_args(args):
    """Print all parsed command-line arguments."""
    print("\n" + "=" * 70)
    print("ARGUMENTS")
    print("=" * 70)

    for key, value in sorted(vars(args).items()):
        print(f"{key:30s}: {value}")

    print("=" * 70 + "\n")


def get_df_ids(df_type, dataset_name):
    if df_type ==0:
        return [i for i in range(1,151)]
    
    elif (df_type ==1)and('pamap' in dataset_name.lower()):
        
        return [1,2,5,6,8]
    
    elif (df_type ==1)and('realworld' in dataset_name.lower()):
        return [i for i in range(1,16)]
    
    elif (df_type ==1)and('usc' in dataset_name.lower()):
        return [i for i in range(1,15)]
    
    elif (df_type ==1)and('harth' in dataset_name.lower()):
        return [i for i in range(22)]
    
def export_statistical_results(results_dict, dataset_name):
    """Export only R@1, R-AUROC, and F1 results with paired statistical tests."""
    classification_df, zero_shot_df = summarize_results_df(results_dict)
    classification_df.to_csv(f'classification_df_{dataset_name}.csv')
    zero_shot_df.to_csv(f'zero_shot_df_{dataset_name}.csv')
    omnibus, posthoc = compare_paired_results(results_dict)
    omnibus.to_csv(f'omnibus_tests_{dataset_name}.csv', index=False)
    posthoc.to_csv(f'posthoc_holm_{dataset_name}.csv', index=False)


def run_downstream_pipeline():
    """Run full evaluation, refresh only zero-shot, or analyze saved JSON.

    Full runs:
        python downstream_experiments_statistical.py \
            --dataset-name pamap --dataset-root downstream_datasets/pamap2/pamap_subject_

    CAPTURE-24 is evaluated using LOSO (150 subject-level folds):
        python downstream_experiments.py \
            --dataset-name capture24 --dataset-root /path/to/capture24_subject_

    Analyze an existing JSON (per-subject zero-shot results are needed
    for mean/SD and paired tests):
        python downstream_experiments_statistical.py \
            --dataset-name pamap --analyze-json results_pamap.json

    Refresh zero-shot without retraining linear probes, reusing unchanged
    classification results from the previous run:
        python downstream_experiments_statistical.py \
            --dataset-name pamap --dataset-root downstream_datasets/pamap2/pamap_subject_ \
            --reuse-classification-json results_pamap.json
            
    nohup python -u /home/jovyan/downstream_experiments.py --dataset-name capture24 --dataset-root downstream_datasets/capture24/capture24_subject_ > downstream_log_capture24.txt 2>&1 &
    
    nohup python -u /home/jovyan/downstream_experiments.py --dataset-name usc --dataset-root downstream_datasets/usc/usc_subject_ > downstream_log_usc.txt 2>&1 &
    
    nohup python -u /home/jovyan/downstream_experiments.py --dataset-name harth --dataset-root downstream_datasets/harth/harth_subject_ > downstream_log_harth.txt 2>&1 &
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--dataset-root', type=str, default=None)
    parser.add_argument('--analyze-json', type=str, default=None)
    parser.add_argument('--reuse-classification-json', type=str, default=None)
    args = parser.parse_args()
    dataset_name = args.dataset_name

    if args.analyze_json:
        with open(args.analyze_json, encoding='utf-8') as handle:
            prior_results = json.load(handle)
        prior_results['classification'] = {
            regime: _clean_classification_structure(values)
            for regime, values in prior_results.get('classification', {}).items()
        }
        if 'capture24' in dataset_name.lower():
            validate_loso_classification_results(
                prior_results, get_df_ids(0, dataset_name)
            )
        export_statistical_results(prior_results, dataset_name)
        print('Statistical analyses saved from:', args.analyze_json)
        return

    if not args.dataset_root:
        parser.error('--dataset-root is required unless --analyze-json is specified')

    dataset_type = 0 if 'capture24' in dataset_name.lower() else 1
    
    users_ids = get_df_ids(dataset_type, dataset_name)
    
    print(f'total of {len(users_ids)} subjects.')
    prior_results = None
    if args.reuse_classification_json:
        with open(args.reuse_classification_json, encoding='utf-8') as handle:
            prior_results = json.load(handle)
        if not prior_results.get('classification'):
            raise ValueError('Saved JSON has no classification section to reuse')
        if dataset_type == 0:
            validate_loso_classification_results(prior_results, users_ids)
        print('Reusing classification results; verify subjects and checkpoints match.')

    results = DownstreamUncertainty(args.dataset_root, users_ids, dataset_type)
    should_train_probe = prior_results is None
    results.make_0_finetuned_results(run_linear_probe=should_train_probe)

    paths = {
        25: {
            'gaussian': 'ssl_wearables/output_hr/HR_reg_gaussian_oxfordhub_25_final.mdl',
            'student_t': 'ssl_wearables/output_hr/HR_reg_student_t_oxfordhub_25_final.mdl',
            'mc_dropout': 'ssl_wearables/output_hr/HR_reg_mc_dropout_oxfordhub_ft25_25_final.mdl',
        },
        100: {
            'gaussian': 'ssl_wearables/output_hr/HR_reg_gaussian_oxfordhub_100_final.mdl',
            'student_t': 'ssl_wearables/output_hr/HR_reg_student_t_oxfordhub_100_final.mdl',
            'mc_dropout': 'ssl_wearables/output_hr/HR_reg_mc_dropout_oxfordhub_ft100_100_final.mdl',
        },
    }
    results.make_25_100_finetuned_results(paths,
                                           run_linear_probe=should_train_probe)
    if prior_results is not None:
        results.results['classification'] = {
            regime: _clean_classification_structure(values)
            for regime, values in prior_results['classification'].items()
        }
    output_json = (f'results_{dataset_name}_statistical.json'
                   if prior_results is not None else f'results_{dataset_name}.json')
    save_json(results.results, output_json)
    export_statistical_results(results.results, dataset_name)


if __name__ == "__main__":
    
    run_downstream_pipeline()
"""HAR linear probe and leakage-safe cross-task uncertainty alignment.

Alignment is fitted *inside each outer LOSO training partition* from
subject-grouped out-of-fold predictions. The outer test subject is never used
for fitting the HAR probe, uncertainty standardization, or logistic models.

Scores (all higher = greater predicted error):
A  original HR uncertainty
B  logistic(HR uncertainty)
C  HAR predictive entropy
D  logistic(HR uncertainty, HAR entropy)
E  logistic(HAR entropy)

Use A vs B for direction correction, D vs E for complementary information.
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def _fit_probe(X, y, num_classes, epochs=100, seed=21):
    """Original full-batch class-weighted SGD training, with fixed label space."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(int(seed))
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    model = LinearClassifier(X.shape[1], num_classes).to(device)
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.train()
    for _ in range(epochs):
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _probe_predict(model, X, batch_size=32768):
    """Return class predictions and Shannon entropy, using bounded GPU memory."""
    device = next(model.parameters()).device
    predictions, entropies = [], []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size],
                                 dtype=torch.float32, device=device)
            logits = model(xb)
            log_p = torch.log_softmax(logits, dim=1)
            p = log_p.exp()
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            entropies.append(-(p * log_p).sum(dim=1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(entropies)


def compute_aurc(errors, uncertainty):
    """Area under risk-coverage curve: mean prefix error (lower is better).

    Retain lowest uncertainty first, at k=1,...,N. For equal uncertainty
    values, use the expected prefix error within each tied block instead of
    breaking ties using the input row order or test labels.
    """
    errors = np.asarray(errors, dtype=float).reshape(-1)
    uncertainty = np.asarray(uncertainty, dtype=float).reshape(-1)
    if len(errors) == 0 or len(uncertainty) != len(errors):
        raise ValueError('AURC requires nonempty arrays of equal length')
    if not np.isfinite(errors).all() or not np.isfinite(uncertainty).all():
        raise ValueError('AURC inputs must be finite')
    if np.any((errors != 0) & (errors != 1)):
        raise ValueError('AURC errors must be binary (0=correct, 1=incorrect)')

    order = np.argsort(uncertainty, kind='stable')
    sorted_errors = errors[order]
    sorted_scores = uncertainty[order]
    n = len(errors)

    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    group_errors = np.add.reduceat(sorted_errors, starts)
    group_counts = np.diff(np.r_[starts, n])
    before = np.repeat(np.r_[0., np.cumsum(group_errors)[:-1]], group_counts)
    within = np.arange(n) - np.repeat(starts, group_counts) + 1
    expected_prefix_errors = before + within * np.repeat(group_errors / group_counts, group_counts)
    return float(np.mean(expected_prefix_errors / np.arange(1, n + 1)))


def _selective_metrics(errors, scores, macro_f1=None):
    """Return AURC and its random-rejection reference (held-out error rate).

    R_all is metadata for interpreting AURC, not an extra hypothesis test.
    For one outer LOSO subject it is the same across all A/B/C/D/E scores.
    """
    out = {
        'AURC': compute_aurc(errors, scores),
        'R_all': float(np.asarray(errors, dtype=float).mean()),
    }
    if macro_f1 is not None:
        out['macro_f1'] = float(macro_f1)
    return out


def _fit_error_logistic(X, errors, label):
    """Train standardized logistic error predictor, training-data only."""
    X = np.asarray(X, dtype=np.float64)
    errors = np.asarray(errors).astype(int)
    if X.ndim != 2 or len(X) != len(errors) or not np.isfinite(X).all():
        raise ValueError('Invalid logistic calibration features')
    if np.unique(errors).size != 2:
        warnings.warn('Alignment {}: only one error class in inner OOF '
                      'predictions; using training prevalence (constant score).'
                      .format(label))
        return ('constant', float(errors.mean()))
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000, C=1.0,
                                             solver='lbfgs'))
    model.fit(X, errors)
    return ('model', model)


def _logistic_coefficients(fitted):
    """Return coefficients expressed in the ORIGINAL input units."""
    mode, payload = fitted
    if mode == 'constant':
        return {'status': 'single_class_fallback', 'coefficients': None,
                'intercept': None, 'training_error_rate': float(payload)}
    scaler = payload.named_steps['standardscaler']
    logistic = payload.named_steps['logisticregression']
    coef = logistic.coef_[0] / scaler.scale_
    intercept = logistic.intercept_[0] - np.dot(coef, scaler.mean_)
    return {'status': 'fitted',
            'coefficients': coef.astype(float).tolist(),
            'intercept': float(intercept)}


def _logistic_score(fitted, X):
    mode, payload = fitted
    if mode == 'constant':
        return np.full(len(X), payload, dtype=float)
    return payload.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]


def train_test_linear_probe_with_alignment(
    X_train, X_test, y_train, y_test, u_train, u_test, train_user_ids,
    epochs=100, inner_folds=2, seed=21, method_names=None,
    return_parameters=False,
):
    """Fit A/B/C/D/E per HR uncertainty source, on one outer LOSO split.

    Inner GroupKFold generates *out-of-subject* HAR probabilities and errors
    from outer training users only. All logistic coefficients and scalers are
    fitted on these inner OOF observations. Then a fresh final probe is trained
    on ALL outer-training subjects and scored on the held-out test subject.

    For each source, return keys:
      <source>                       A original HR uncertainty
      <source>/logistic_hr           B logistic HR only
      <source>/har_entropy           C entropy only
      <source>/fusion                D logistic HR + entropy
      <source>/logistic_har_entropy  E logistic entropy only
    Includes selective metrics on every score, and macro_f1 only on A (since
    the predicted class and macro-F1 are identical for the five scores).
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.int64).reshape(-1)
    groups = np.asarray(train_user_ids).reshape(-1)
    if len(groups) != len(X_train) or len(y_train) != len(X_train):
        raise ValueError('Training arrays and subject IDs must align')
    if len(y_test) != len(X_test) or not len(y_train):
        raise ValueError('Invalid test or training dimensions')
    if (y_train < 0).any():
        raise ValueError('HAR labels must be nonnegative integer IDs')
    num_classes = int(y_train.max()) + 1
    if num_classes < 2:
        raise ValueError('HAR training split requires at least two classes')
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError('At least two distinct training subjects required for OOF alignment')
    inner_folds = min(int(inner_folds), n_groups)
    if inner_folds < 2:
        raise ValueError('inner_folds must be >= 2')

    if not isinstance(u_train, pd.DataFrame) or not isinstance(u_test, pd.DataFrame):
        raise TypeError('u_train and u_test must be matching pandas DataFrames')
    if list(u_train.columns) != list(u_test.columns):
        raise ValueError('Training and test HR uncertainty columns must match')
    names = method_names or {c: ('CE' if c == 'u_ce' else str(c))
                             for c in u_train.columns}
    if set(names) != set(u_train.columns):
        raise ValueError('method_names must map every uncertainty column')

    # Inner out-of-fold predictions: no training subject predicts itself.
    oof_entropy = np.full(len(X_train), np.nan, dtype=np.float64)
    oof_error = np.full(len(X_train), -1, dtype=np.int8)
    splitter = GroupKFold(n_splits=inner_folds)
    for part, (fit_ids, val_ids) in enumerate(
            splitter.split(X_train, y_train, groups=groups)):
        inner_model = _fit_probe(X_train[fit_ids], y_train[fit_ids],
                                 num_classes=num_classes, epochs=epochs,
                                 seed=seed + part)
        inner_pred, inner_entropy = _probe_predict(inner_model, X_train[val_ids])
        oof_entropy[val_ids] = inner_entropy
        oof_error[val_ids] = (inner_pred != y_train[val_ids]).astype(np.int8)
        del inner_model
    if not np.isfinite(oof_entropy).all() or (oof_error < 0).any():
        raise RuntimeError('Incomplete subject-grouped OOF alignment predictions')

    # Final probe sees all (and only) outer-training subjects.
    final_model = _fit_probe(X_train, y_train, num_classes=num_classes,
                             epochs=epochs, seed=seed)
    test_pred, test_entropy = _probe_predict(final_model, X_test)
    del final_model
    test_error = (test_pred != y_test).astype(int)
    macro_f1 = f1_score(y_test, test_pred, average='macro', zero_division=0)

    entropy_calibrator = _fit_error_logistic(oof_entropy[:, None],
                                              oof_error, 'HAR entropy')
    aligned_entropy = _logistic_score(entropy_calibrator, test_entropy[:, None])

    output = {}
    parameters = {
        'oof_n_samples': int(len(oof_error)),
        'oof_error_rate': float(oof_error.mean()),
        'har_entropy': _logistic_coefficients(entropy_calibrator),
        'by_upstream_source': {},
    }
    for column in u_train.columns:
        source = str(names[column])
        u_cal = np.asarray(u_train[column], dtype=np.float64).reshape(-1)
        u_outer = np.asarray(u_test[column], dtype=np.float64).reshape(-1)
        if (len(u_cal) != len(X_train) or len(u_outer) != len(X_test)
                or not np.isfinite(u_cal).all() or not np.isfinite(u_outer).all()):
            raise ValueError('Uncertainty arrays must align and be finite: ' + source)
        hr_calibrator = _fit_error_logistic(u_cal[:, None], oof_error,
                                            source + '/HR')
        fusion_calibrator = _fit_error_logistic(
            np.column_stack([u_cal, oof_entropy]), oof_error, source + '/fusion')
        parameters['by_upstream_source'][source] = {
            'logistic_hr': _logistic_coefficients(hr_calibrator),
            'fusion': _logistic_coefficients(fusion_calibrator),
        }
        base = source + '/'
        output[source] = _selective_metrics(test_error, u_outer, macro_f1=macro_f1)
        output[base + 'logistic_hr'] = _selective_metrics(
            test_error, _logistic_score(hr_calibrator, u_outer[:, None]))
        output[base + 'har_entropy'] = _selective_metrics(test_error, test_entropy)
        output[base + 'fusion'] = _selective_metrics(
            test_error, _logistic_score(
                fusion_calibrator, np.column_stack([u_outer, test_entropy])))
        output[base + 'logistic_har_entropy'] = _selective_metrics(
            test_error, aligned_entropy)
    return (output, parameters) if return_parameters else output


def train_test_linear_probe(X_train, X_test, y_train, y_test, u_test, epochs=20):
    """Original non-aligned linear-probe interface, using macro-F1 and AURC."""
    y_train = np.asarray(y_train, dtype=np.int64)
    model = _fit_probe(X_train, y_train, int(y_train.max()) + 1, epochs)
    prediction, _ = _probe_predict(model, X_test)
    errors = (np.asarray(y_test) != prediction).astype(np.int8)
    score = f1_score(y_test, prediction, average='macro', zero_division=0)
    if isinstance(u_test, pd.DataFrame):
        items = list(u_test.items())
    else:
        items = [('u', np.asarray(u_test).reshape(-1))]
    res = {}
    for key, values in items:
        res[str(key).replace('u_ce', 'CE')] = {
            'macro_f1': float(score),
            'AURC': compute_aurc(errors, np.asarray(values).reshape(-1)),
            'R_all': float(np.asarray(errors, dtype=float).mean()),
        }
    return res if len(items) > 1 else next(iter(res.values()))

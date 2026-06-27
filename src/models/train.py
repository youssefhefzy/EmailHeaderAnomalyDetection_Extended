"""
train.py
--------
Train and evaluate individual classifiers on email-header features.

Models
------
- **RF**   — Random Forest (scikit-learn)
- **SVM**  — Support Vector Machine + Platt scaling calibration
- **MLP**  — Multi-Layer Perceptron
- **KNN**  — K-Nearest Neighbours
- **OC-SVM** — One-Class SVM (anomaly detection; trained on ham only)

Each model is evaluated via 10-fold stratified cross-validation (or a held-out
test split if requested).  Results include accuracy, precision, recall, F1,
AUC, Brier score, ECE, training time, and inference time.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM

from src.models.confidence import (
    calibrate_svm,
    compute_calibration_metrics,
    get_confidence_scores,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_name: str, cfg: Dict[str, Any]) -> Any:
    """
    Instantiate a fresh (unfitted) sklearn estimator.

    Parameters
    ----------
    model_name:
        One of ``'RF'``, ``'SVM'``, ``'MLP'``, ``'KNN'``, ``'OC_SVM'``.
    cfg:
        Hyperparameter config dict (from ``config.yaml``).

    Returns
    -------
    Unfitted sklearn estimator.
    """
    hp = cfg.get("hyperparameters", {})
    rs = cfg.get("random_state", 42)

    if model_name == "RF":
        p = hp.get("RF", {})
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth"),
            n_jobs=1,
            random_state=rs,
        )

    if model_name == "SVM":
        p = hp.get("SVM", {})
        # probability=False; calibration applied separately
        return SVC(
            C=p.get("C", 1.0),
            kernel=p.get("kernel", "rbf"),
            probability=False,
            random_state=rs,
        )

    if model_name == "MLP":
        p = hp.get("MLP", {})
        layers = tuple(p.get("hidden_layer_sizes", [100, 50]))
        return MLPClassifier(
            hidden_layer_sizes=layers,
            max_iter=p.get("max_iter", 300),
            early_stopping=p.get("early_stopping", True),
            random_state=rs,
        )

    if model_name == "KNN":
        p = hp.get("KNN", {})
        return KNeighborsClassifier(
            n_neighbors=p.get("n_neighbors", 5),
            n_jobs=1,
        )

    if model_name == "OC_SVM":
        p = hp.get("OC_SVM", {})
        return OneClassSVM(
            nu=p.get("nu", 0.1),
            kernel=p.get("kernel", "rbf"),
        )

    raise ValueError(f"Unknown model name: '{model_name}'")


# ---------------------------------------------------------------------------
# Training + evaluation (cross-validation)
# ---------------------------------------------------------------------------

def train_evaluate_cv(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
    feature_group: str = "G6",
    cv_folds: int = 10,
    scale_features: bool = True,
) -> Dict[str, Any]:
    """
    Train and evaluate a model using stratified k-fold cross-validation.

    For OC-SVM, only ham samples (y == 0) are used for training; the full set
    is used for evaluation.

    Parameters
    ----------
    model_name:
        Model identifier.
    X:
        Feature DataFrame.
    y:
        Label Series.
    cfg:
        Configuration dict.
    feature_group:
        Feature group label (for logging / results tagging).
    cv_folds:
        Number of folds.
    scale_features:
        Whether to apply ``StandardScaler`` within each fold.

    Returns
    -------
    dict with keys:
        ``model_name``, ``feature_group``, ``accuracy``, ``precision``,
        ``recall``, ``f1``, ``auc``, ``brier_score``, ``ece``,
        ``train_time_s``, ``inference_ms_per_1k``, and fold-level lists.
    """
    rs = cfg.get("random_state", 42)
    hp_svm = cfg.get("hyperparameters", {}).get("SVM", {})

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rs)

    fold_metrics: List[Dict[str, float]] = []
    all_y_true, all_y_prob = [], []
    train_times, inf_times = [], []

    X_arr = X.values
    y_arr = y.values

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        # Scale
        if scale_features:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        model = build_model(model_name, cfg)

        # ── OC-SVM: train on ham only ────────────────────────────────────
        if model_name == "OC_SVM":
            ham_mask = y_tr == 0
            t0 = time.perf_counter()
            model.fit(X_tr[ham_mask])
            train_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            raw_pred = model.predict(X_te)  # +1 = inlier, -1 = outlier
            inf_times.append((time.perf_counter() - t1) / max(len(X_te), 1) * 1000)

            # Map: +1 (inlier/ham) → 0, -1 (outlier) → 1
            y_pred = np.where(raw_pred == 1, 0, 1)
            y_prob = get_confidence_scores(model, X_te, model_type="ocsvm")

        # ── SVM: calibrate with Platt scaling ────────────────────────────
        elif model_name == "SVM":
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            calib_method = hp_svm.get("calibration_method", "sigmoid")
            calib_cv = hp_svm.get("calibration_cv", 5)
            calibrated = calibrate_svm(
                model, X_tr, y_tr, method=calib_method, cv=calib_cv
            )
            train_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            y_pred = calibrated.predict(X_te)
            inf_times.append((time.perf_counter() - t1) / max(len(X_te), 1) * 1000)
            y_prob = get_confidence_scores(calibrated, X_te)

        # ── All other models ──────────────────────────────────────────────
        else:
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            train_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            y_pred = model.predict(X_te)
            inf_times.append((time.perf_counter() - t1) / max(len(X_te), 1) * 1000)
            y_prob = get_confidence_scores(model, X_te)

        # Metrics
        fm = _fold_metrics(y_te, y_pred, y_prob)
        fold_metrics.append(fm)
        all_y_true.append(y_te)
        all_y_prob.append(y_prob)

        logger.debug(
            "Fold %d/%d — acc=%.4f f1=%.4f brier=%.4f ece=%.4f",
            fold_idx + 1, cv_folds,
            fm["accuracy"], fm["f1"], fm["brier_score"], fm["ece"]
        )

    # ── Aggregate across folds ────────────────────────────────────────────
    result = _aggregate_metrics(
        fold_metrics=fold_metrics,
        train_times=train_times,
        inf_times=inf_times,
        model_name=model_name,
        feature_group=feature_group,
    )
    return result


def _fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute per-fold metrics."""
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        auc = float("nan")
    else:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float("nan")

    calib = compute_calibration_metrics(y_true, y_prob)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0, average="binary"),
        "recall": recall_score(y_true, y_pred, zero_division=0, average="binary"),
        "f1": f1_score(y_true, y_pred, zero_division=0, average="binary"),
        "auc": auc,
        "brier_score": calib["brier_score"],
        "ece": calib["ece"],
    }


def _aggregate_metrics(
    fold_metrics: List[Dict[str, float]],
    train_times: List[float],
    inf_times: List[float],
    model_name: str,
    feature_group: str,
) -> Dict[str, Any]:
    """Average fold metrics and attach timing information."""
    keys = list(fold_metrics[0].keys())
    result: Dict[str, Any] = {
        "model_name": model_name,
        "feature_group": feature_group,
    }
    for k in keys:
        vals = [fm[k] for fm in fold_metrics if not np.isnan(fm[k])]
        result[k] = float(np.mean(vals)) if vals else float("nan")
        result[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")

    result["train_time_s"] = float(np.mean(train_times))
    result["inference_ms_per_1k"] = float(np.mean(inf_times))
    return result


# ---------------------------------------------------------------------------
# Train / evaluate on a single split
# ---------------------------------------------------------------------------

def train_evaluate_split(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: Dict[str, Any],
    feature_group: str = "G6",
    scale_features: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train on (X_train, y_train) and evaluate on (X_test, y_test).

    Returns
    -------
    (fitted_model, metrics_dict)
    """
    rs = cfg.get("random_state", 42)
    hp_svm = cfg.get("hyperparameters", {}).get("SVM", {})

    X_tr = X_train.values
    y_tr = y_train.values
    X_te = X_test.values
    y_te = y_test.values

    if scale_features:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    model = build_model(model_name, cfg)

    if model_name == "OC_SVM":
        ham_mask = y_tr == 0
        t0 = time.perf_counter()
        model.fit(X_tr[ham_mask])
        train_t = time.perf_counter() - t0
        t1 = time.perf_counter()
        raw_pred = model.predict(X_te)
        inf_t = (time.perf_counter() - t1) / max(len(X_te), 1) * 1000
        y_pred = np.where(raw_pred == 1, 0, 1)
        y_prob = get_confidence_scores(model, X_te, model_type="ocsvm")

    elif model_name == "SVM":
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        calibrated = calibrate_svm(
            model, X_tr, y_tr,
            method=hp_svm.get("calibration_method", "sigmoid"),
            cv=hp_svm.get("calibration_cv", 5),
        )
        train_t = time.perf_counter() - t0
        t1 = time.perf_counter()
        y_pred = calibrated.predict(X_te)
        inf_t = (time.perf_counter() - t1) / max(len(X_te), 1) * 1000
        y_prob = get_confidence_scores(calibrated, X_te)
        model = calibrated

    else:
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_t = time.perf_counter() - t0
        t1 = time.perf_counter()
        y_pred = model.predict(X_te)
        inf_t = (time.perf_counter() - t1) / max(len(X_te), 1) * 1000
        y_prob = get_confidence_scores(model, X_te)

    metrics = _fold_metrics(y_te, y_pred, y_prob)
    metrics["train_time_s"] = train_t
    metrics["inference_ms_per_1k"] = inf_t
    metrics["model_name"] = model_name
    metrics["feature_group"] = feature_group

    return model, metrics


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, path: Path | str) -> None:
    """Serialise a fitted model to disk with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)


def load_model(path: Path | str) -> Any:
    """Load a previously serialised model."""
    return joblib.load(path)

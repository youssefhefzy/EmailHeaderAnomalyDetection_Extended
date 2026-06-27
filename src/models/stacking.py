"""
stacking.py
-----------
Stacked ensemble classifier (RF + MLP + KNN base learners, Logistic Regression
meta-classifier).

The implementation follows the stacking strategy described in Beaman & Isah
(2021) and extends it with:

- Confidence-score output from the meta-classifier.
- Optional Platt calibration of the meta-classifier.
- Cross-validated training of base learners (``passthrough=False``).

Usage
-----
    from src.models.stacking import build_stacking_model, train_evaluate_stacking_cv
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.models.confidence import compute_calibration_metrics, get_confidence_scores
from src.models.train import _aggregate_metrics, _fold_metrics, build_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build stacking model
# ---------------------------------------------------------------------------

def build_stacking_model(
    cfg: Dict[str, Any],
    passthrough: bool = False,
) -> StackingClassifier:
    """
    Construct a :class:`~sklearn.ensemble.StackingClassifier`.

    Base estimators are RF, MLP, and KNN.
    Meta-classifier is Logistic Regression (``predict_proba`` is available,
    so confidence scores come naturally).

    Parameters
    ----------
    cfg:
        Configuration dict (from ``config.yaml``).
    passthrough:
        If ``True``, the original features are concatenated with the
        base-estimator predictions before passing to the meta-classifier.

    Returns
    -------
    StackingClassifier
        Unfitted stacking classifier.
    """
    rs = cfg.get("random_state", 42)
    hp = cfg.get("hyperparameters", {})

    # ── Base estimators ──────────────────────────────────────────────────
    rf_p = hp.get("RF", {})
    rf = RandomForestClassifier(
        n_estimators=rf_p.get("n_estimators", 100),
        max_depth=rf_p.get("max_depth"),
        n_jobs=1,
        random_state=rs,
    )

    mlp_p = hp.get("MLP", {})
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(mlp_p.get("hidden_layer_sizes", [100, 50])),
        max_iter=mlp_p.get("max_iter", 300),
        early_stopping=mlp_p.get("early_stopping", True),
        random_state=rs,
    )

    knn_p = hp.get("KNN", {})
    knn = KNeighborsClassifier(
        n_neighbors=knn_p.get("n_neighbors", 5),
        n_jobs=1,
    )

    estimators = [("rf", rf), ("mlp", mlp), ("knn", knn)]

    # ── Meta-classifier ──────────────────────────────────────────────────
    meta = LogisticRegression(max_iter=500, random_state=rs, n_jobs=1)

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        passthrough=passthrough,
        cv=5,
        n_jobs=1,
    )
    return stack


# ---------------------------------------------------------------------------
# Train + evaluate (cross-validation)
# ---------------------------------------------------------------------------

def train_evaluate_stacking_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
    feature_group: str = "G6",
    cv_folds: int = 10,
    scale_features: bool = True,
) -> Dict[str, Any]:
    """
    Train and evaluate the stacking ensemble via stratified k-fold CV.

    Parameters
    ----------
    X:
        Feature DataFrame.
    y:
        Label Series (binary: 0 / 1).
    cfg:
        Config dict.
    feature_group:
        Feature group label (used for tagging results).
    cv_folds:
        Number of CV folds.
    scale_features:
        Apply StandardScaler within each fold.

    Returns
    -------
    dict
        Aggregated metrics (same format as :func:`~src.models.train.train_evaluate_cv`).
    """
    rs = cfg.get("random_state", 42)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rs)

    fold_metrics: List[Dict[str, float]] = []
    train_times: List[float] = []
    inf_times: List[float] = []

    X_arr = X.values
    y_arr = y.values

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        if scale_features:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        stack = build_stacking_model(cfg)

        t0 = time.perf_counter()
        stack.fit(X_tr, y_tr)
        train_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        y_pred = stack.predict(X_te)
        inf_times.append((time.perf_counter() - t1) / max(len(X_te), 1) * 1000)

        y_prob = get_confidence_scores(stack, X_te)
        fm = _fold_metrics(y_te, y_pred, y_prob)
        fold_metrics.append(fm)

        logger.debug(
            "Stacking Fold %d/%d — acc=%.4f f1=%.4f",
            fold_idx + 1, cv_folds, fm["accuracy"], fm["f1"]
        )

    result = _aggregate_metrics(
        fold_metrics=fold_metrics,
        train_times=train_times,
        inf_times=inf_times,
        model_name="Stacking",
        feature_group=feature_group,
    )
    return result


# ---------------------------------------------------------------------------
# Single split (for reproducibility)
# ---------------------------------------------------------------------------

def train_evaluate_stacking_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: Dict[str, Any],
    feature_group: str = "G6",
    scale_features: bool = True,
) -> Tuple[StackingClassifier, Dict[str, Any]]:
    """
    Fit the stacking ensemble on a single train/test split.

    Returns
    -------
    (fitted_stack, metrics_dict)
    """
    X_tr = X_train.values
    y_tr = y_train.values
    X_te = X_test.values
    y_te = y_test.values

    if scale_features:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    stack = build_stacking_model(cfg)

    t0 = time.perf_counter()
    stack.fit(X_tr, y_tr)
    train_t = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = stack.predict(X_te)
    inf_t = (time.perf_counter() - t1) / max(len(X_te), 1) * 1000

    y_prob = get_confidence_scores(stack, X_te)
    metrics = _fold_metrics(y_te, y_pred, y_prob)
    metrics["train_time_s"] = train_t
    metrics["inference_ms_per_1k"] = inf_t
    metrics["model_name"] = "Stacking"
    metrics["feature_group"] = feature_group

    return stack, metrics

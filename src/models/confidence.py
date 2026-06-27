"""
confidence.py
-------------
Confidence score utilities for all model types.

Supports
--------
- ``predict_proba`` for RF, MLP, KNN (native).
- Platt scaling (``CalibratedClassifierCV``) for SVM.
- Sigmoid-normalised ``decision_function`` for One-Class SVM.
- Stacked-ensemble confidence via meta-classifier ``predict_proba``.

Calibration metrics
-------------------
- **Brier Score** — proper scoring rule for probabilistic classifiers.
- **Expected Calibration Error (ECE)** — mean absolute difference between
  predicted confidence and observed accuracy in equal-width bins.
- **Reliability diagrams** — plots of observed accuracy vs mean confidence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.svm import SVC, OneClassSVM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence extraction
# ---------------------------------------------------------------------------

def get_confidence_scores(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    model_type: str = "auto",
) -> np.ndarray:
    """
    Return a 1-D array of confidence scores (probability of positive class).

    Dispatches to the appropriate method based on *model_type*.

    Parameters
    ----------
    model:
        Fitted sklearn estimator.
    X:
        Feature matrix (n_samples × n_features).
    model_type:
        One of ``'proba'``, ``'ocsvm'``, ``'auto'``.

        - ``'proba'``:  calls ``model.predict_proba(X)[:, 1]``
        - ``'ocsvm'``:  applies sigmoid to ``model.decision_function(X)``
        - ``'auto'``:   tries ``predict_proba`` first, falls back to ``decision_function``

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Confidence scores in [0, 1].
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    if model_type == "ocsvm" or isinstance(model, OneClassSVM):
        return _ocsvm_confidence(model, X)

    if model_type == "proba" or hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            # Binary: column 1 = positive class probability
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.ravel()
        except Exception as exc:
            logger.debug("predict_proba failed (%s); trying decision_function.", exc)

    if hasattr(model, "decision_function"):
        return _sigmoid(model.decision_function(X))

    raise ValueError(
        f"Cannot extract confidence scores from model type {type(model).__name__}. "
        "It has neither predict_proba nor decision_function."
    )


def _ocsvm_confidence(model: OneClassSVM, X: np.ndarray) -> np.ndarray:
    """
    Convert OC-SVM decision function to [0,1] confidence.

    ``decision_function`` returns a signed distance from the decision boundary:
    - Positive → inlier (ham).
    - Negative → outlier (spam / phishing).

    We negate so that higher = more anomalous, then apply sigmoid.
    """
    scores = model.decision_function(X)
    return _sigmoid(-scores)  # negate: higher confidence = more anomalous


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply element-wise sigmoid to a 1-D array."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ---------------------------------------------------------------------------
# SVM Calibration wrapper
# ---------------------------------------------------------------------------

def calibrate_svm(
    svm_model: SVC,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "sigmoid",
    cv: int | str = 5,
) -> CalibratedClassifierCV:
    """
    Wrap a fitted :class:`~sklearn.svm.SVC` with Platt scaling.

    If ``cv='prefit'`` the model is assumed to be already fitted and only the
    calibrator is trained on (X_train, y_train).

    Parameters
    ----------
    svm_model:
        Fitted SVC (``probability=False`` is fine; we wrap it here).
    X_train:
        Calibration training data.
    y_train:
        Calibration training labels.
    method:
        Calibration method: ``'sigmoid'`` (Platt) or ``'isotonic'``.
    cv:
        Cross-validation strategy or ``'prefit'`` to use the model as-is.

    Returns
    -------
    CalibratedClassifierCV
        Fitted calibrated classifier.
    """
    logger.info("Calibrating SVM with method='%s', cv='%s' …", method, cv)
    calibrated = CalibratedClassifierCV(
        estimator=svm_model, method=method, cv=cv
    )
    calibrated.fit(X_train, y_train)
    return calibrated


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def brier_score(
    y_true: np.ndarray, y_prob: np.ndarray, pos_label: int = 1
) -> float:
    """
    Compute the Brier score (lower is better, 0 = perfect).

    Parameters
    ----------
    y_true:
        True binary labels.
    y_prob:
        Predicted probability of the positive class.
    pos_label:
        The label to consider as positive.

    Returns
    -------
    float
    """
    return float(brier_score_loss(y_true, y_prob, pos_label=pos_label))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the average gap between predicted confidence and empirical
    accuracy across equal-width confidence bins.

    Parameters
    ----------
    y_true:
        True binary labels (0 / 1).
    y_prob:
        Predicted probabilities for the positive class.
    n_bins:
        Number of equal-width bins.

    Returns
    -------
    float
        ECE ∈ [0, 1].  0 = perfectly calibrated.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        bin_count = mask.sum()
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (bin_count / n) * abs(bin_conf - bin_acc)

    return float(ece)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute a full suite of calibration metrics.

    Parameters
    ----------
    y_true:
        True binary labels.
    y_prob:
        Predicted positive-class probabilities.
    n_bins:
        Bins for ECE.

    Returns
    -------
    dict with keys:
        ``brier_score``, ``ece``, ``mean_confidence``, ``mean_accuracy``.
    """
    bs = brier_score(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    return {
        "brier_score": bs,
        "ece": ece,
        "mean_confidence": float(y_prob.mean()),
        "mean_accuracy": float(y_true.mean()),
    }


# ---------------------------------------------------------------------------
# Per-email confidence CSV
# ---------------------------------------------------------------------------

def save_confidence_csv(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    feature_group: str,
    output_path: Path | str,
) -> None:
    """
    Save per-email confidence scores to a CSV file.

    Parameters
    ----------
    X:
        Feature DataFrame (used only for index / metadata).
    y_true:
        True labels.
    y_pred:
        Predicted labels.
    y_prob:
        Confidence scores (probability of positive class).
    model_name:
        Model identifier string.
    feature_group:
        Feature group identifier (e.g., 'G6').
    output_path:
        Output CSV path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame({
        "email_idx": np.arange(len(y_true)),
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": y_prob,
        "correct": (y_true == y_pred).astype(int),
        "model": model_name,
        "feature_group": feature_group,
    })
    df_out.to_csv(output_path, index=False)
    logger.info("Confidence scores saved → %s (%d rows)", output_path, len(df_out))

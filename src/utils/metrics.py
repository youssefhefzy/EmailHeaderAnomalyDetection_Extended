"""
metrics.py
----------
Metric-collection and formatting utilities used across the pipeline.

Functions
---------
- :func:`compute_all_metrics` — full metric suite for a single (y_true, y_pred, y_prob).
- :func:`metrics_to_dataframe` — convert a list of metric dicts to a tidy DataFrame.
- :func:`print_metrics_table` — pretty-print a metrics DataFrame.
- :func:`timing_stats` — compute mean / std of a list of timing measurements.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.confidence import compute_calibration_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Full metric suite
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    n_bins: int = 10,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted labels.
    y_prob:
        Predicted probabilities for the positive class.  Required for AUC,
        Brier score, and ECE.
    n_bins:
        Number of bins for ECE.
    prefix:
        Optional string to prepend to every metric key.

    Returns
    -------
    dict
        Keys (without prefix): ``accuracy``, ``precision``, ``recall``,
        ``f1``, ``auc``, ``average_precision``, ``brier_score``, ``ece``.
    """
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0, average="binary"),
        "recall": recall_score(y_true, y_pred, zero_division=0, average="binary"),
        "f1": f1_score(y_true, y_pred, zero_division=0, average="binary"),
    }

    if y_prob is not None:
        unique_classes = np.unique(y_true)
        if len(unique_classes) >= 2:
            try:
                metrics["auc"] = roc_auc_score(y_true, y_prob)
                metrics["average_precision"] = average_precision_score(y_true, y_prob)
            except Exception:
                metrics["auc"] = float("nan")
                metrics["average_precision"] = float("nan")
        else:
            metrics["auc"] = float("nan")
            metrics["average_precision"] = float("nan")

        calib = compute_calibration_metrics(y_true, y_prob, n_bins=n_bins)
        metrics.update(calib)
    else:
        metrics["auc"] = float("nan")
        metrics["average_precision"] = float("nan")
        metrics["brier_score"] = float("nan")
        metrics["ece"] = float("nan")

    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

    return metrics


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_cv_metrics(
    fold_results: List[Dict[str, float]],
    model_name: str = "",
    feature_group: str = "",
    dataset: str = "",
) -> Dict[str, Any]:
    """
    Average metric dicts across cross-validation folds.

    Parameters
    ----------
    fold_results:
        List of per-fold metric dicts.
    model_name:
        Tag to attach.
    feature_group:
        Tag to attach.
    dataset:
        Tag to attach (e.g. ``'A'``).

    Returns
    -------
    dict
        Mean and std of each metric, plus the tags.
    """
    all_keys = set(k for d in fold_results for k in d)
    agg: Dict[str, Any] = {
        "model_name": model_name,
        "feature_group": feature_group,
        "dataset": dataset,
        "n_folds": len(fold_results),
    }
    for k in sorted(all_keys):
        vals = [d[k] for d in fold_results if k in d and not np.isnan(d[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")
    return agg


# ---------------------------------------------------------------------------
# DataFrame / display helpers
# ---------------------------------------------------------------------------

def metrics_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of metric dicts to a tidy pandas DataFrame.

    Parameters
    ----------
    results:
        Each dict corresponds to one experiment row.

    Returns
    -------
    pd.DataFrame
    """
    return pd.DataFrame(results)


def print_metrics_table(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    float_fmt: str = ".4f",
) -> None:
    """
    Pretty-print a metrics DataFrame.

    Parameters
    ----------
    df:
        DataFrame of results.
    cols:
        Subset of columns to display.  Defaults to key metric columns.
    float_fmt:
        Format string for floating-point values.
    """
    if cols is None:
        cols = [c for c in [
            "model_name", "feature_group", "dataset",
            "accuracy", "f1", "auc", "brier_score", "ece",
            "train_time_s", "inference_ms_per_1k",
        ] if c in df.columns]
    display_df = df[cols].copy()
    num_cols = display_df.select_dtypes(include=[float, int]).columns
    fmt = {c: f"{{:{float_fmt}}}".format for c in num_cols}
    print(display_df.to_string(index=False, formatters=fmt))


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def timing_stats(times: List[float]) -> Dict[str, float]:
    """
    Compute mean, std, min, max for a list of timing measurements.

    Parameters
    ----------
    times:
        List of timing values (e.g. seconds per training run).

    Returns
    -------
    dict with keys: ``mean``, ``std``, ``min``, ``max``.
    """
    arr = np.array(times)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }

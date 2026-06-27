"""
plotting.py
-----------
Visualisation utilities for the comparison report.

Functions
---------
- :func:`plot_calibration_curve` — reliability diagram for one model.
- :func:`plot_all_calibration_curves` — multi-panel reliability diagrams.
- :func:`plot_accuracy_vs_time` — scatter plot: accuracy vs inference speed.
- :func:`plot_metric_heatmap` — heatmap of a metric across groups × models.
- :func:`plot_feature_group_comparison` — grouped bar chart of a metric.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
_FIG_DPI = 150


# ---------------------------------------------------------------------------
# Calibration / reliability diagram
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "",
    feature_group: str = "",
    n_bins: int = 10,
    save_path: Optional[Path | str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot a reliability diagram (calibration curve).

    Parameters
    ----------
    y_true:
        True binary labels.
    y_prob:
        Predicted positive-class probabilities.
    model_name:
        Used in the plot title.
    feature_group:
        Used in the plot title.
    n_bins:
        Number of calibration bins.
    save_path:
        If given, save the figure to this path.
    ax:
        Existing Axes to draw on (creates a new figure if None).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_predicted, fraction_of_positives, "s-", label=model_name or "Model")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Positives")
    title = "Reliability Diagram"
    if model_name:
        title += f" — {model_name}"
    if feature_group:
        title += f" ({feature_group})"
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if own_fig:
        plt.tight_layout()
        if save_path:
            _save(fig, save_path)
    return fig


def plot_all_calibration_curves(
    records: List[Dict],
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """
    Multi-panel reliability diagrams for multiple (model, feature_group) combinations.

    Parameters
    ----------
    records:
        List of dicts with keys: ``y_true``, ``y_prob``, ``model_name``,
        ``feature_group``.
    save_path:
        Output figure path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(records)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).ravel()

    for i, rec in enumerate(records):
        plot_calibration_curve(
            y_true=rec["y_true"],
            y_prob=rec["y_prob"],
            model_name=rec.get("model_name", ""),
            feature_group=rec.get("feature_group", ""),
            ax=axes[i],
        )

    # Hide spare axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Reliability Diagrams", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Accuracy vs inference time
# ---------------------------------------------------------------------------

def plot_accuracy_vs_time(
    df: pd.DataFrame,
    accuracy_col: str = "accuracy",
    time_col: str = "inference_ms_per_1k",
    hue_col: str = "model_name",
    style_col: str = "feature_group",
    dataset_label: str = "",
    baseline_accuracy: Optional[float] = None,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """
    Scatter plot of accuracy vs inference time (one point per model × feature group).

    Parameters
    ----------
    df:
        Results DataFrame.
    accuracy_col:
        Column for y-axis (accuracy).
    time_col:
        Column for x-axis (inference time in ms/1k).
    hue_col:
        Column to colour-code points.
    style_col:
        Column to shape-code points.
    dataset_label:
        Subtitle / label.
    baseline_accuracy:
        If given, draw a horizontal dashed line for the original paper's best.
    save_path:
        Output path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = sns.color_palette("tab10", n_colors=df[hue_col].nunique())
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    unique_styles = df[style_col].unique().tolist()
    marker_map = {s: markers[i % len(markers)] for i, s in enumerate(unique_styles)}

    for model in df[hue_col].unique():
        for group in df[style_col].unique():
            sub = df[(df[hue_col] == model) & (df[style_col] == group)]
            if sub.empty:
                continue
            ax.scatter(
                sub[time_col],
                sub[accuracy_col],
                label=f"{model} / {group}",
                marker=marker_map.get(group, "o"),
                s=80,
                alpha=0.8,
            )
            # Annotate
            for _, row in sub.iterrows():
                ax.annotate(
                    f"{group}",
                    (row[time_col], row[accuracy_col]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    alpha=0.7,
                )

    if baseline_accuracy is not None:
        ax.axhline(
            baseline_accuracy, color="red", linestyle="--", lw=1.5,
            label=f"Original baseline ({baseline_accuracy:.3f})"
        )

    ax.set_xlabel("Inference Time (ms / 1 000 emails)")
    ax.set_ylabel("Accuracy")
    title = "Accuracy vs Inference Time"
    if dataset_label:
        title += f" — Dataset {dataset_label}"
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
    plt.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Metric heatmap
# ---------------------------------------------------------------------------

def plot_metric_heatmap(
    df: pd.DataFrame,
    metric_col: str = "f1",
    row_col: str = "feature_group",
    col_col: str = "model_name",
    title: str = "",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """
    Heatmap of *metric_col* with feature groups on the y-axis and models on the x-axis.

    Parameters
    ----------
    df:
        Results DataFrame.
    metric_col:
        Metric to display.
    row_col:
        DataFrame column for rows (default: feature group).
    col_col:
        DataFrame column for columns (default: model).
    title:
        Plot title.
    save_path:
        Output path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pivot = df.pivot_table(index=row_col, columns=col_col, values=metric_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.5), max(5, pivot.shape[0] * 0.8)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=ax,
        linewidths=0.5,
        vmin=pivot.values.min() - 0.01,
        vmax=min(1.0, pivot.values.max() + 0.01),
    )
    ax.set_title(title or f"{metric_col.upper()} Heatmap")
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature Group")
    plt.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Grouped bar chart
# ---------------------------------------------------------------------------

def plot_feature_group_comparison(
    df: pd.DataFrame,
    metric_col: str = "f1",
    group_col: str = "feature_group",
    hue_col: str = "model_name",
    dataset_label: str = "",
    baseline_value: Optional[float] = None,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing a metric across feature groups, coloured by model.

    Parameters
    ----------
    df:
        Results DataFrame.
    metric_col:
        Metric to display on the y-axis.
    group_col:
        Column for x-axis groups.
    hue_col:
        Column for bar colours.
    dataset_label:
        Added to plot title.
    baseline_value:
        If given, draw a horizontal reference line.
    save_path:
        Output path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x=group_col, y=metric_col, hue=hue_col, ax=ax)

    if baseline_value is not None:
        ax.axhline(
            baseline_value, color="red", linestyle="--", lw=1.5,
            label=f"Original baseline ({baseline_value:.3f})"
        )
        ax.legend()

    title = f"{metric_col.upper()} by Feature Group and Model"
    if dataset_label:
        title += f" — Dataset {dataset_label}"
    ax.set_title(title)
    ax.set_xlabel("Feature Group")
    ax.set_ylabel(metric_col.upper())
    plt.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path | str) -> None:
    """Save a figure to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
    logger.info("Figure saved → %s", path)
    plt.close(fig)

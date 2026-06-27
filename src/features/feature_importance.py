"""
feature_importance.py
---------------------
Utilities for computing, visualising, and persisting feature importance.

Methods
-------
- **Permutation Importance** (model-agnostic, used by Beaman & Isah 2021).
- **RFECV** (Recursive Feature Elimination with Cross-Validation) — finds
  the minimum feature subset that retains peak cross-validated accuracy.
- **Mean Decrease in Impurity** (MDI) — fast but biased; available for tree
  ensembles only.

Usage
-----
    from src.features.feature_importance import (
        compute_permutation_importance,
        run_rfecv,
        plot_importance,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permutation Importance
# ---------------------------------------------------------------------------

def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Compute permutation feature importance for any fitted sklearn estimator.

    The model must already be fitted.  Feature importances are computed on the
    provided (X, y) — typically the held-out test set.

    Parameters
    ----------
    model:
        A fitted sklearn estimator.
    X:
        Feature DataFrame (test set recommended).
    y:
        True labels.
    n_repeats:
        Number of times to permute each feature.
    random_state:
        Random seed.
    n_jobs:
        Parallel jobs.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``importance_mean``, ``importance_std``,
        sorted by ``importance_mean`` descending.
    """
    logger.info("Computing permutation importance (n_repeats=%d) …", n_repeats)
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring="accuracy",
    )
    df_imp = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    logger.info("Top-10 features by permutation importance:\n%s", df_imp.head(10).to_string())
    return df_imp


# ---------------------------------------------------------------------------
# MDI (tree-based)
# ---------------------------------------------------------------------------

def compute_mdi_importance(
    model: RandomForestClassifier,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Return Mean Decrease in Impurity (MDI) importance for a fitted RF/tree.

    Parameters
    ----------
    model:
        Fitted :class:`~sklearn.ensemble.RandomForestClassifier`.
    feature_names:
        List of feature names corresponding to model.feature_importances_.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``importance``, sorted descending.
    """
    importances = model.feature_importances_
    std = np.std([t.feature_importances_ for t in model.estimators_], axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "importance_std": std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# RFECV
# ---------------------------------------------------------------------------

def run_rfecv(
    X: pd.DataFrame,
    y: pd.Series,
    estimator=None,
    cv: int = 5,
    scoring: str = "f1_weighted",
    step: int = 1,
    min_features_to_select: int = 1,
    random_state: int = 42,
) -> Tuple[RFECV, List[str]]:
    """
    Run Recursive Feature Elimination with Cross-Validation (RFECV).

    Parameters
    ----------
    X:
        Full feature DataFrame.
    y:
        Labels.
    estimator:
        Base estimator for RFE.  Defaults to a small Random Forest.
    cv:
        Number of cross-validation folds.
    scoring:
        Metric to optimise.
    step:
        Number of features to remove at each step.
    min_features_to_select:
        Minimum number of features to keep.
    random_state:
        Seed.

    Returns
    -------
    (RFECV, selected_feature_names)
        Fitted RFECV object and the list of selected feature names.
    """
    if estimator is None:
        estimator = RandomForestClassifier(
            n_estimators=50, n_jobs=1, random_state=random_state
        )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    logger.info(
        "Running RFECV (cv=%d, scoring=%s, step=%d) on %d features …",
        cv, scoring, step, X.shape[1]
    )
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=skf,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=1,
    )
    selector.fit(X, y)

    selected = [c for c, s in zip(X.columns, selector.support_) if s]
    logger.info(
        "RFECV selected %d features (optimal): %s",
        len(selected), selected
    )
    return selector, selected


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_importance(df_imp: pd.DataFrame, path: Path | str) -> None:
    """Save a feature importance DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_imp.to_csv(path, index=False)
    logger.info("Feature importance saved to %s", path)


def load_importance(path: Path | str) -> pd.DataFrame:
    """Load a previously saved feature importance CSV."""
    return pd.read_csv(path)


def importance_to_dict(df_imp: pd.DataFrame, col: str = "importance_mean") -> Dict[str, float]:
    """
    Convert an importance DataFrame to a ``{feature: score}`` dict.

    Parameters
    ----------
    df_imp:
        DataFrame with a ``feature`` column and an importance column.
    col:
        Name of the importance column to use.

    Returns
    -------
    dict
    """
    return dict(zip(df_imp["feature"], df_imp[col]))


# ---------------------------------------------------------------------------
# Plotting (optional — requires matplotlib)
# ---------------------------------------------------------------------------

def plot_importance(
    df_imp: pd.DataFrame,
    top_n: int = 30,
    title: str = "Feature Importance",
    save_path: Optional[Path | str] = None,
) -> None:
    """
    Horizontal bar chart of the top-N feature importances.

    Parameters
    ----------
    df_imp:
        DataFrame with ``feature`` and ``importance_mean`` columns.
    top_n:
        How many features to show.
    title:
        Plot title.
    save_path:
        If given, save the figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed; skipping plot.")
        return

    df_plot = df_imp.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, max(6, top_n // 3)))
    sns.barplot(
        data=df_plot,
        y="feature",
        x="importance_mean",
        xerr=df_plot.get("importance_std"),
        ax=ax,
        palette="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("Mean Permutation Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Importance plot saved to %s", save_path)
    else:
        plt.show()
    plt.close(fig)

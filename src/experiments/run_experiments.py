"""
run_experiments.py
------------------
Main experimental pipeline.

1. Load Dataset A and/or Dataset B.
2. For each feature group (G1–G8) × each model (RF, SVM, MLP, KNN, Stacking, OC-SVM):
   a. Extract the feature subset.
   b. Train and evaluate via 10-fold cross-validation.
   c. Record accuracy, F1, AUC, Brier score, ECE, training time, inference time.
3. Save all results to ``results/feature_combinations/comparison.csv``.

Usage
-----
    # Full run (uses config.yaml)
    python src/experiments/run_experiments.py

    # Quick test with small datasets
    python src/experiments/run_experiments.py \
        --datasetA data/processed/datasetA_processed.csv \
        --datasetB data/processed/datasetB_processed.csv \
        --output   results/feature_combinations/comparison.csv \
        --groups G1 G2 G6 \
        --models RF SVM
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# Allow running from project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.feature_groups import get_all_group_names, get_feature_subset
from src.features.feature_importance import (
    compute_permutation_importance,
    importance_to_dict,
    save_importance,
)
from src.models.stacking import train_evaluate_stacking_cv
from src.models.train import build_model, train_evaluate_cv
from src.utils.metrics import metrics_to_dataframe, print_metrics_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_all_experiments(
    dataset_paths: Dict[str, Path],
    cfg: dict,
    groups: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    output_csv: Path = Path("results/feature_combinations/comparison.csv"),
) -> pd.DataFrame:
    """
    Run the full experimental matrix.

    Parameters
    ----------
    dataset_paths:
        Dict mapping dataset label (e.g. ``'A'``) to the processed CSV path.
    cfg:
        Loaded config.yaml dict.
    groups:
        Feature groups to evaluate.  Defaults to all (G1–G8).
    models:
        Model names to evaluate.  Defaults to all configured models.
    output_csv:
        Path for the results CSV.

    Returns
    -------
    pd.DataFrame
        Full results table.
    """
    groups = groups or get_all_group_names()
    models = models or cfg.get("models", ["RF", "SVM", "MLP", "KNN", "Stacking", "OC_SVM"])
    cv_folds = cfg.get("cv_folds", 10)
    rs = cfg.get("random_state", 42)

    all_results: List[dict] = []

    for dataset_label, csv_path in dataset_paths.items():
        if not csv_path.exists():
            logger.warning("Dataset %s not found at %s — skipping.", dataset_label, csv_path)
            continue

        logger.info("=" * 60)
        logger.info("Loading Dataset %s from %s …", dataset_label, csv_path)
        df = pd.read_csv(csv_path)

        if "label" not in df.columns:
            logger.error("No 'label' column in %s — skipping.", csv_path)
            continue

        y = df["label"]
        label_counts = y.value_counts().to_dict()
        logger.info("Label distribution: %s", label_counts)

        # ── Compute permutation importance (RF on G6) once per dataset ─────
        importance_scores = _compute_importance_for_dataset(df, y, cfg, dataset_label)

        for group in groups:
            logger.info("-" * 40)
            logger.info("Feature group: %s (Dataset %s)", group, dataset_label)

            feature_cols = get_feature_subset(df, group, importance_scores)
            if not feature_cols:
                logger.warning("Group %s has no valid features for Dataset %s — skipping.", group, dataset_label)
                continue

            X = df[feature_cols]
            logger.info("  Features: %d columns", X.shape[1])

            for model_name in models:
                logger.info("  Training %s …", model_name)
                try:
                    result = _run_single_experiment(
                        model_name=model_name,
                        X=X,
                        y=y,
                        cfg=cfg,
                        feature_group=group,
                        cv_folds=cv_folds,
                    )
                    result["dataset"] = dataset_label
                    result["n_features"] = X.shape[1]
                    all_results.append(result)

                    logger.info(
                        "  → acc=%.4f f1=%.4f brier=%.4f ece=%.4f "
                        "train=%.2fs inf=%.3fms/1k",
                        result.get("accuracy", float("nan")),
                        result.get("f1", float("nan")),
                        result.get("brier_score", float("nan")),
                        result.get("ece", float("nan")),
                        result.get("train_time_s", float("nan")),
                        result.get("inference_ms_per_1k", float("nan")),
                    )
                except Exception as exc:
                    logger.error(
                        "  FAILED: %s / %s / Dataset %s — %s",
                        model_name, group, dataset_label, exc, exc_info=True
                    )

    df_results = metrics_to_dataframe(all_results)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    logger.info("All results saved → %s (%d rows)", output_csv, len(df_results))

    print("\n" + "=" * 60)
    print("SUMMARY (key metrics)")
    print("=" * 60)
    print_metrics_table(df_results)

    return df_results


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def _run_single_experiment(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    feature_group: str,
    cv_folds: int,
) -> dict:
    """Train and evaluate a single (model, feature group) combination."""
    if model_name == "Stacking":
        return train_evaluate_stacking_cv(
            X=X, y=y, cfg=cfg,
            feature_group=feature_group,
            cv_folds=cv_folds,
        )
    else:
        return train_evaluate_cv(
            model_name=model_name,
            X=X, y=y,
            cfg=cfg,
            feature_group=feature_group,
            cv_folds=cv_folds,
        )


# ---------------------------------------------------------------------------
# Permutation importance (computed once per dataset)
# ---------------------------------------------------------------------------

def _compute_importance_for_dataset(
    df: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    dataset_label: str,
) -> Optional[Dict[str, float]]:
    """
    Fit a Random Forest on all features and compute permutation importance.

    Importance scores are used by G6 and G8 to select the top features.
    If computation fails, returns None (groups fall back to hard-coded lists).
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        label_col = "label"
        feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols]
        rs = cfg.get("random_state", 42)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=rs, stratify=y
        )

        hp_rf = cfg.get("hyperparameters", {}).get("RF", {})
        rf = RandomForestClassifier(
            n_estimators=hp_rf.get("n_estimators", 100),
            n_jobs=1,
            random_state=rs,
        )
        logger.info("Computing permutation importance for Dataset %s …", dataset_label)
        rf.fit(X_tr, y_tr)

        df_imp = compute_permutation_importance(
            rf, X_te, y_te,
            n_repeats=3,       # reduced from 5 for speed
            random_state=rs,
            n_jobs=1,          # single job to avoid parallelism hangs
        )

        imp_path = ROOT / "results" / "original_baseline" / f"importance_dataset_{dataset_label}.csv"
        save_importance(df_imp, imp_path)

        return importance_to_dict(df_imp)
    except Exception as exc:
        logger.warning("Could not compute permutation importance: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Baseline reproduction (original paper — G6, all models)
# ---------------------------------------------------------------------------

def reproduce_baseline(
    dataset_paths: Dict[str, Path],
    cfg: dict,
    output_csv: Path = Path("results/original_baseline/baseline.csv"),
) -> pd.DataFrame:
    """
    Reproduce the original paper's results using the G6 feature group.

    Parameters
    ----------
    dataset_paths:
        Mapping of dataset label → processed CSV path.
    cfg:
        Config dict.
    output_csv:
        Where to save the baseline results.

    Returns
    -------
    pd.DataFrame
        Baseline results table.
    """
    logger.info("Reproducing original paper baseline (G6, all models) …")
    return run_all_experiments(
        dataset_paths=dataset_paths,
        cfg=cfg,
        groups=["G6"],
        models=cfg.get("models", ["RF", "SVM", "MLP", "KNN", "Stacking", "OC_SVM"]),
        output_csv=output_csv,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run email-header anomaly detection experiments."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config.yaml",
        help="Path to config.yaml."
    )
    parser.add_argument(
        "--datasetA", type=Path, default=None,
        help="Path to processed Dataset A CSV."
    )
    parser.add_argument(
        "--datasetB", type=Path, default=None,
        help="Path to processed Dataset B CSV."
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "feature_combinations" / "comparison.csv",
        help="Output CSV path for experiment results."
    )
    parser.add_argument(
        "--groups", nargs="+", default=None,
        help="Feature groups to run (default: all G1–G8)."
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models to train (default: all configured models)."
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only reproduce the original paper baseline (G6)."
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Dataset paths
    paths_cfg = cfg.get("paths", {})
    processed_dir = ROOT / paths_cfg.get("data_processed", "data/processed")

    dataset_paths: Dict[str, Path] = {}
    if args.datasetA:
        dataset_paths["A"] = args.datasetA
    elif (processed_dir / "datasetA_processed.csv").exists():
        dataset_paths["A"] = processed_dir / "datasetA_processed.csv"

    if args.datasetB:
        dataset_paths["B"] = args.datasetB
    elif (processed_dir / "datasetB_processed.csv").exists():
        dataset_paths["B"] = processed_dir / "datasetB_processed.csv"

    if not dataset_paths:
        logger.error(
            "No datasets found.  Run src/data/make_dataset.py first, "
            "or pass --datasetA / --datasetB."
        )
        sys.exit(1)

    if args.baseline_only:
        reproduce_baseline(
            dataset_paths=dataset_paths,
            cfg=cfg,
            output_csv=ROOT / "results" / "original_baseline" / "baseline.csv",
        )
    else:
        # First reproduce baseline (G6 only), then run full matrix
        reproduce_baseline(
            dataset_paths=dataset_paths,
            cfg=cfg,
            output_csv=ROOT / "results" / "original_baseline" / "baseline.csv",
        )
        run_all_experiments(
            dataset_paths=dataset_paths,
            cfg=cfg,
            groups=args.groups,
            models=args.models,
            output_csv=args.output,
        )


if __name__ == "__main__":
    main()

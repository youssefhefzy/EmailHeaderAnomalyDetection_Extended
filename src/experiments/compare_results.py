"""
compare_results.py
------------------
Load experiment results and the original paper baseline, then:

1. Generate a comparison table (extended Table VII).
2. Plot accuracy vs inference time scatter.
3. Plot metric heatmaps (F1, accuracy).
4. Plot calibration curves for the best models.
5. Write a markdown comparison report.

Usage
-----
    python src/experiments/compare_results.py \
        --results  results/feature_combinations/comparison.csv \
        --baseline results/original_baseline/baseline.csv \
        --out_dir  results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# Allow running from project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.plotting import (
    plot_accuracy_vs_time,
    plot_feature_group_comparison,
    plot_metric_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare_results(
    results_csv: Path,
    baseline_csv: Optional[Path] = None,
    out_dir: Path = Path("results"),
    cfg: Optional[dict] = None,
) -> None:
    """
    Load experiment results, produce all plots and the markdown report.

    Parameters
    ----------
    results_csv:
        Full results CSV from run_experiments.py.
    baseline_csv:
        Baseline (G6 only) results CSV.  If None, extracted from results_csv.
    out_dir:
        Root output directory.
    cfg:
        Optional config dict (for paper baselines).
    """
    df = pd.read_csv(results_csv)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load or derive baseline
    if baseline_csv and baseline_csv.exists():
        df_base = pd.read_csv(baseline_csv)
    else:
        df_base = df[df["feature_group"] == "G6"].copy()

    logger.info("Loaded %d experiment results rows.", len(df))
    logger.info("Baseline rows: %d", len(df_base))

    # ── 1. Extended comparison table ─────────────────────────────────────
    table_path = out_dir / "feature_combinations" / "extended_comparison_table.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    _make_comparison_table(df, df_base, table_path, cfg)

    # ── 2. Accuracy vs time scatter ──────────────────────────────────────
    for dataset in df["dataset"].unique() if "dataset" in df.columns else [""]:
        sub = df[df["dataset"] == dataset] if dataset else df
        best_baseline = _best_baseline_accuracy(df_base, dataset)
        plot_accuracy_vs_time(
            df=sub,
            baseline_accuracy=best_baseline,
            dataset_label=dataset,
            save_path=figures_dir / f"accuracy_vs_time_{dataset}.png",
        )

    # ── 3. F1 heatmap ────────────────────────────────────────────────────
    for dataset in df["dataset"].unique() if "dataset" in df.columns else [""]:
        sub = df[df["dataset"] == dataset] if dataset else df
        plot_metric_heatmap(
            df=sub,
            metric_col="f1",
            title=f"F1 Score — Dataset {dataset}",
            save_path=figures_dir / f"f1_heatmap_{dataset}.png",
        )
        plot_metric_heatmap(
            df=sub,
            metric_col="accuracy",
            title=f"Accuracy — Dataset {dataset}",
            save_path=figures_dir / f"accuracy_heatmap_{dataset}.png",
        )

    # ── 4. Feature-group comparison bar charts ───────────────────────────
    for dataset in df["dataset"].unique() if "dataset" in df.columns else [""]:
        sub = df[df["dataset"] == dataset] if dataset else df
        best_baseline = _best_baseline_accuracy(df_base, dataset)
        plot_feature_group_comparison(
            df=sub,
            metric_col="f1",
            dataset_label=dataset,
            baseline_value=best_baseline,
            save_path=figures_dir / f"f1_by_group_{dataset}.png",
        )

    # ── 5. Markdown report ───────────────────────────────────────────────
    report_path = out_dir / "comparison_report.md"
    _write_markdown_report(df, df_base, report_path, cfg)
    logger.info("Comparison report written → %s", report_path)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _make_comparison_table(
    df: pd.DataFrame,
    df_base: pd.DataFrame,
    output_path: Path,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """Build and save the extended comparison table."""
    key_cols = [
        "dataset", "model_name", "feature_group", "n_features",
        "accuracy", "precision", "recall", "f1", "auc",
        "brier_score", "ece",
        "train_time_s", "inference_ms_per_1k",
    ]
    available = [c for c in key_cols if c in df.columns]
    table = df[available].copy()

    # Round floats
    float_cols = table.select_dtypes(include=float).columns
    table[float_cols] = table[float_cols].round(4)

    # Annotate best per (dataset, model)
    if "dataset" in table.columns and "model_name" in table.columns:
        table["is_best_accuracy"] = table.groupby(["dataset", "model_name"])["accuracy"].transform(
            lambda x: x == x.max()
        )

    table.to_csv(output_path, index=False)
    logger.info("Extended comparison table saved → %s (%d rows)", output_path, len(table))
    return table


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_markdown_report(
    df: pd.DataFrame,
    df_base: pd.DataFrame,
    output_path: Path,
    cfg: Optional[dict] = None,
) -> None:
    """Generate the markdown comparison report."""
    lines: List[str] = []

    lines.append("# Email Header Anomaly Detection — Comparison Report\n")
    lines.append("Generated automatically by `compare_results.py`.\n")

    # ── Paper baselines ──────────────────────────────────────────────────
    paper_orig = (cfg or {}).get("original_results", {})
    if paper_orig:
        lines.append("## 1. Original Paper Baselines (Beaman & Isah, 2021)\n")
        lines.append(_paper_baselines_table(paper_orig))

    # ── Reproduced baseline ──────────────────────────────────────────────
    lines.append("## 2. Reproduced Baseline (G6 Features)\n")
    base_g6 = df_base[df_base["feature_group"] == "G6"] if "feature_group" in df_base.columns else df_base
    if not base_g6.empty:
        lines.append(_df_to_markdown(base_g6, cols=[
            "dataset", "model_name", "accuracy", "f1", "auc",
            "brier_score", "ece", "train_time_s", "inference_ms_per_1k",
        ]))

    # ── Best accuracy ────────────────────────────────────────────────────
    lines.append("## 3. Best Accuracy per Dataset\n")
    lines.append(_best_accuracy_section(df, paper_orig))

    # ── Fastest inference ────────────────────────────────────────────────
    lines.append("## 4. Fastest Inference (≤1% accuracy drop vs best)\n")
    lines.append(_fastest_inference_section(df))

    # ── Calibration summary ──────────────────────────────────────────────
    lines.append("## 5. Calibration Summary\n")
    lines.append(_calibration_summary_section(df))

    # ── Full results table ───────────────────────────────────────────────
    lines.append("## 6. Full Results Table\n")
    lines.append(_df_to_markdown(df, cols=[
        "dataset", "model_name", "feature_group", "n_features",
        "accuracy", "f1", "auc", "brier_score", "ece",
        "train_time_s", "inference_ms_per_1k",
    ]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _paper_baselines_table(paper_orig: dict) -> str:
    rows = []
    for dataset, models in paper_orig.items():
        for model, metrics in models.items():
            rows.append({
                "Dataset": dataset,
                "Model": model,
                "Accuracy": metrics.get("accuracy", "—"),
                "F1": metrics.get("f1", "—"),
            })
    df = pd.DataFrame(rows)
    return _df_to_markdown(df) + "\n"


def _best_accuracy_section(df: pd.DataFrame, paper_orig: dict) -> str:
    lines = []
    if "dataset" not in df.columns:
        return "N/A\n"

    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            continue
        best_row = sub.loc[sub["accuracy"].idxmax()]
        model = best_row.get("model_name", "?")
        group = best_row.get("feature_group", "?")
        acc = best_row.get("accuracy", float("nan"))

        # Compare to paper
        paper_best = max(
            (v.get("accuracy", 0) for v in paper_orig.get(f"dataset_{dataset.lower()}", {}).values()),
            default=None
        )
        delta = f" (+{acc - paper_best:.4f} vs original)" if paper_best else ""

        lines.append(f"**Dataset {dataset}**: `{model}` on `{group}` — accuracy = {acc:.4f}{delta}\n")

    return "\n".join(lines) + "\n"


def _fastest_inference_section(df: pd.DataFrame) -> str:
    lines = []
    if "dataset" not in df.columns or "inference_ms_per_1k" not in df.columns:
        return "N/A\n"

    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            continue
        best_acc = sub["accuracy"].max()
        threshold = best_acc - 0.01
        fast_candidates = sub[sub["accuracy"] >= threshold]
        if fast_candidates.empty:
            continue
        fastest = fast_candidates.loc[fast_candidates["inference_ms_per_1k"].idxmin()]
        model = fastest.get("model_name", "?")
        group = fastest.get("feature_group", "?")
        inf_t = fastest.get("inference_ms_per_1k", float("nan"))
        acc = fastest.get("accuracy", float("nan"))
        lines.append(
            f"**Dataset {dataset}**: `{model}` on `{group}` — "
            f"inference = {inf_t:.3f} ms/1k, accuracy = {acc:.4f} "
            f"(≤1% drop from best {best_acc:.4f})\n"
        )

    return "\n".join(lines) + "\n"


def _calibration_summary_section(df: pd.DataFrame) -> str:
    lines = []
    for col in ("brier_score", "ece"):
        if col not in df.columns:
            continue
        best_row = df.loc[df[col].dropna().idxmin()] if not df[col].dropna().empty else None
        worst_row = df.loc[df[col].dropna().idxmax()] if not df[col].dropna().empty else None
        if best_row is not None:
            lines.append(
                f"**Best {col}**: {best_row.get(col, '?'):.4f} — "
                f"`{best_row.get('model_name', '?')}` / `{best_row.get('feature_group', '?')}` "
                f"/ Dataset {best_row.get('dataset', '?')}"
            )
        if worst_row is not None:
            lines.append(
                f"**Worst {col}**: {worst_row.get(col, '?'):.4f} — "
                f"`{worst_row.get('model_name', '?')}` / `{worst_row.get('feature_group', '?')}` "
                f"/ Dataset {worst_row.get('dataset', '?')}"
            )
    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_baseline_accuracy(df_base: pd.DataFrame, dataset: str) -> Optional[float]:
    """Return the best accuracy from the baseline DataFrame for a given dataset."""
    if df_base.empty or "accuracy" not in df_base.columns:
        return None
    sub = df_base[df_base.get("dataset", "") == dataset] if "dataset" in df_base.columns else df_base
    if sub.empty:
        return None
    return float(sub["accuracy"].max())


def _df_to_markdown(df: pd.DataFrame, cols: Optional[List[str]] = None) -> str:
    """Convert a DataFrame to a Markdown table string."""
    if cols:
        available = [c for c in cols if c in df.columns]
        df = df[available]
    return df.to_markdown(index=False, floatfmt=".4f") + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare experiment results with the original paper baseline."
    )
    parser.add_argument(
        "--results", type=Path,
        default=ROOT / "results" / "feature_combinations" / "comparison.csv",
        help="Full results CSV from run_experiments.py.",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "results" / "original_baseline" / "baseline.csv",
        help="Baseline (G6) results CSV.",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=ROOT / "results",
        help="Output directory for report and figures.",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config.yaml",
        help="Config file (for paper baselines).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    cfg = {}
    if args.config.exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    if not args.results.exists():
        logger.error("Results file not found: %s", args.results)
        sys.exit(1)

    compare_results(
        results_csv=args.results,
        baseline_csv=args.baseline if args.baseline.exists() else None,
        out_dir=args.out_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()

"""
run_all.py  —  Single-entry-point runner (cross-platform: Windows / Linux / macOS)
===================================================================================
Replaces ALL bash/PowerShell multi-line commands with one clean Python script.

Usage
-----
  # Quick smoke-test (synthetic data, cv=3, 3 fast models)
  python run_all.py --fast

  # Full run with synthetic data (cv=5, all 6 models, all 8 groups)
  python run_all.py

  # Full run with real SpamAssassin + phishing emails
  python run_all.py --real-data

  # Full run pointing at custom raw-email directories
  python run_all.py --real-data \\
      --datasetA data/raw/spamassassin \\
      --datasetB data/raw/phishing

Steps
-----
  1. Prepare datasets  (synthetic or real)
  2. Run G6 baseline   (reproduces original paper)
  3. Run full matrix   (all feature groups G1-G8 x all models)
  4. Generate report + figures
  5. Print summary table

Getting the Real Datasets
--------------------------
  Dataset A (SpamAssassin / TREC 2007):
      https://spamassassin.apache.org/old/publiccorpus/
      Place emails in:  data/raw/spamassassin/ham/   and   data/raw/spamassassin/spam/

  Dataset B (Phishing corpus):
      https://monkey.org/~jose/phishing/
      Place emails in:  data/raw/phishing/ham/   and   data/raw/phishing/phishing/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Directories ────────────────────────────────────────────────────────────────
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR   = ROOT / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "original_baseline").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "feature_combinations").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
log_file = RESULTS_DIR / "run_all.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
# Silence noisy sub-loggers
for _noisy in ("sklearn", "src.features", "src.models", "src.experiments"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

RNG = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Synthetic dataset generation
# ══════════════════════════════════════════════════════════════════════════════

def _bern(labels: np.ndarray, p_ham: float, p_spam: float) -> np.ndarray:
    probs = np.where(labels == 0, p_ham, p_spam)
    return RNG.binomial(1, probs).astype(int)


def _norm(labels: np.ndarray, mu_ham: float, mu_spam: float, sd: float = 1.0) -> np.ndarray:
    mu = np.where(labels == 0, mu_ham, mu_spam)
    return np.clip(RNG.normal(mu, sd), 0, None)


def generate_synthetic(n_ham: int, n_spam: int, dataset: str = "A") -> pd.DataFrame:
    """
    Create a synthetic email-header feature DataFrame.

    All feature names exactly match those produced by extract_headers.py so
    the same feature_groups.py definitions (G1-G8) apply to both real and
    synthetic data.
    """
    n = n_ham + n_spam
    L = np.array([0] * n_ham + [1] * n_spam)
    b = lambda ph, ps: _bern(L, ph, ps)     # noqa: E731
    m = lambda mh, ms, sd=1.0: _norm(L, mh, ms, sd)   # noqa: E731

    rows: dict = {
        "label": L,
        # G1 — header presence
        "has_received":                    b(0.99, 0.90),
        "has_message_id":                  b(0.99, 0.75),
        "has_date":                        b(1.00, 0.95),
        "has_from":                        b(1.00, 0.99),
        "has_to":                          b(0.98, 0.85),
        "has_reply_to":                    b(0.30, 0.70),
        "has_return_path":                 b(0.90, 0.60),
        "has_mime_version":                b(0.95, 0.80),
        "has_content_type":                b(0.99, 0.95),
        "has_content_transfer_encoding":   b(0.85, 0.78),
        "has_subject":                     b(0.99, 0.97),
        "has_x_mailer":                    b(0.50, 0.35),
        "has_x_originating_ip":            b(0.30, 0.55),
        "has_x_spam_status":               b(0.60, 0.40),
        "has_dkim_signature":              b(0.70, 0.30),
        "has_domainkey_signature":         b(0.20, 0.10),
        "has_authentication_results":      b(0.75, 0.40),
        "has_arc_authentication_results":  b(0.40, 0.20),
        "has_delivered_to":                b(0.70, 0.45),
        "has_envelope_to":                 b(0.30, 0.25),
        "has_errors_to":                   b(0.15, 0.30),
        "has_list_unsubscribe":            b(0.40, 0.55),
        "has_precedence":                  b(0.35, 0.50),
        "has_x_priority":                  b(0.20, 0.55),
        "has_spf":                         b(0.85, 0.50),
        "has_dkim":                        b(0.70, 0.30),
        "has_dmarc":                       b(0.65, 0.25),
        # G2 — domain matching
        "from_domain_len":                  m(12, 18, 4),
        "from_eq_reply_to_domain":          b(0.85, 0.20),
        "from_eq_return_path_domain":       b(0.90, 0.25),
        "from_eq_mid_domain":               b(0.88, 0.30),
        "from_eq_first_received_domain":    b(0.80, 0.35),
        "reply_to_present_and_differs":     b(0.10, 0.65),
        "return_path_present_and_differs":  b(0.08, 0.60),
        "to_domain_eq_from_domain":         b(0.30, 0.20),
        "mid_domain_eq_from_domain":        b(0.88, 0.30),
        "envelope_to_eq_from_domain":       b(0.50, 0.35),
        "reply_to_subdomain_of_from":       b(0.15, 0.05),
        # G3 — routing
        "n_received_hops":               m(3, 6, 2).astype(int),
        "missing_received":              b(0.01, 0.12),
        "malformed_received_hops":       m(0, 1.5, 1).astype(int),
        "n_unique_hop_ips":              m(2, 4, 1.5).astype(int),
        "n_private_ips_in_received":     m(0.5, 1.5, 1).astype(int),
        "tz_mismatch_in_received":       b(0.05, 0.40),
        "n_unique_tz_offsets":           m(1, 2.5, 1).astype(int),
        "has_unusual_tz":                b(0.03, 0.35),
        "n_hop_domains":                 m(3, 5, 2).astype(int),
        "date_vs_received_anomaly":      b(0.02, 0.25),
        "excessive_hops":                b(0.01, 0.15),
        # G4 — structural
        "duplicate_headers":             m(0, 1.5, 1).astype(int),
        "subject_non_ascii":             b(0.05, 0.30),
        "from_non_ascii":                b(0.02, 0.25),
        "malformed_date":                b(0.01, 0.20),
        "subject_len":                   m(35, 55, 20),
        "subject_has_re":                b(0.25, 0.10),
        "subject_all_caps":              b(0.02, 0.20),
        "subject_exclaim_count":         m(0.1, 2.0, 1),
        "subject_question_count":        m(0.1, 0.8, 0.5),
        "is_base64_encoded":             b(0.30, 0.55),
        "is_quoted_printable":           b(0.45, 0.30),
        "mid_malformed":                 b(0.02, 0.40),
        "mid_len":                       m(50, 35, 20),
        "is_multipart":                  b(0.70, 0.55),
        "is_html_only":                  b(0.15, 0.55),
        "total_header_count":            m(18, 12, 5),
        # G5 — authentication
        "spf_result":   np.where(L == 0,
                            RNG.choice([1, 0, -1], n, p=[0.80, 0.15, 0.05]),
                            RNG.choice([1, 0, -1], n, p=[0.20, 0.30, 0.50])),
        "dkim_result":  np.where(L == 0,
                            RNG.choice([1, 0, -1], n, p=[0.75, 0.20, 0.05]),
                            RNG.choice([1, 0, -1], n, p=[0.15, 0.35, 0.50])),
        "dmarc_result": np.where(L == 0,
                            RNG.choice([1, 0, -1], n, p=[0.70, 0.25, 0.05]),
                            RNG.choice([1, 0, -1], n, p=[0.15, 0.40, 0.45])),
        "dkim_signature_present":        b(0.70, 0.30),
        "dkim_signature_valid_format":   b(0.68, 0.20),
        "auth_results_present":          b(0.75, 0.40),
        "all_auth_pass":                 b(0.65, 0.10),
        "any_auth_fail":                 b(0.05, 0.55),
        # Misc
        "x_mailer_len":       m(20, 10, 10),
        "x_priority_value":   np.where(L == 0,
                                  RNG.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.70, 0.10, 0.05]),
                                  RNG.choice([1, 2, 3, 4, 5], n, p=[0.30, 0.20, 0.30, 0.10, 0.10])),
        "is_bulk_precedence": b(0.15, 0.55),
        "list_unsubscribe_len": m(40, 80, 30),
        "n_to_addresses":     m(1, 3, 2).astype(int),
        "n_cc_addresses":     m(0.5, 0.3, 1).astype(int),
        "n_bcc_addresses":    m(0.1, 0.5, 0.5).astype(int),
        "x_spam_score":       m(-1.5, 6.0, 3.0),
    }

    df = pd.DataFrame(rows)
    if dataset == "B":
        for col in ["n_received_hops", "tz_mismatch_in_received",
                    "has_unusual_tz", "n_unique_tz_offsets"]:
            if col in df.columns:
                df[col] = df[col] * RNG.binomial(1, 0.4, n)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Dataset preparation
# ══════════════════════════════════════════════════════════════════════════════

def prepare_datasets(args, n_ham: int, n_spam: int) -> dict[str, Path]:
    """Return {dataset_label: csv_path} for whatever data is available."""
    paths: dict[str, Path] = {}

    if args.real_data:
        logger.info("Attempting to build datasets from REAL email files …")
        from src.data.make_dataset import build_dataset_A, build_dataset_B

        da_dir = Path(args.datasetA)
        db_dir = Path(args.datasetB)

        if da_dir.exists():
            try:
                out_A = PROCESSED_DIR / "datasetA_processed.csv"
                build_dataset_A(da_dir, out_A, max_per_class=args.max_per_class)
                paths["A"] = out_A
                logger.info("Real Dataset A built → %s", out_A)
            except Exception as e:
                logger.error("Dataset A failed (%s) — using synthetic fallback.", e)

        if db_dir.exists():
            try:
                out_B = PROCESSED_DIR / "datasetB_processed.csv"
                build_dataset_B(db_dir, out_B, max_per_class=args.max_per_class)
                paths["B"] = out_B
                logger.info("Real Dataset B built → %s", out_B)
            except Exception as e:
                logger.error("Dataset B failed (%s) — using synthetic fallback.", e)

    # Fill any missing datasets with synthetic data
    for ds, letter in [("A", "A"), ("B", "B")]:
        if ds not in paths:
            nh = n_ham if letter == "A" else n_ham // 2
            ns = n_spam if letter == "A" else n_spam // 2
            logger.info("Generating SYNTHETIC Dataset %s (%d ham, %d spam) …", ds, nh, ns)
            df = generate_synthetic(nh, ns, dataset=letter)
            out = PROCESSED_DIR / f"synthetic_{letter}.csv"
            df.to_csv(out, index=False)
            paths[ds] = out
            logger.info("Synthetic Dataset %s → %s  shape=%s", ds, out, df.shape)

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2–3 — Run experiments
# ══════════════════════════════════════════════════════════════════════════════

def run_experiments(dataset_paths, cfg, groups, models, output_csv: Path) -> pd.DataFrame:
    """
    Run all (group × model × dataset) experiments.

    Iterates manually (no multiprocessing) to avoid Windows/Linux fork issues
    that caused the original scripts to hang.
    """
    from src.experiments.run_experiments import (
        _run_single_experiment,
        _compute_importance_for_dataset,
    )
    from src.features.feature_groups import get_feature_subset

    all_results: list[dict] = []

    for ds_label, csv_path in sorted(dataset_paths.items()):
        if not csv_path.exists():
            logger.warning("Dataset %s not found: %s — skipping.", ds_label, csv_path)
            continue

        df = pd.read_csv(csv_path)
        if "label" not in df.columns:
            logger.error("No 'label' column in %s — skipping.", csv_path)
            continue

        y = df["label"]
        logger.info(
            "Dataset %s: %d rows, label dist = %s",
            ds_label, len(df), dict(y.value_counts())
        )

        logger.info("Computing permutation importance for Dataset %s …", ds_label)
        imp = _compute_importance_for_dataset(df, y, cfg, ds_label)

        for group in groups:
            feat_cols = get_feature_subset(df, group, imp)
            if not feat_cols:
                continue
            X = df[feat_cols]

            for model_name in models:
                t0 = time.perf_counter()
                try:
                    r = _run_single_experiment(
                        model_name, X, y, cfg, group, cfg.get("cv_folds", 5)
                    )
                    r["dataset"]    = ds_label
                    r["n_features"] = len(feat_cols)
                    all_results.append(r)
                    logger.info(
                        "  OK  %s / %s / Dataset %s | "
                        "acc=%.4f  f1=%.4f  auc=%.4f  brier=%.4f  ece=%.4f  "
                        "train=%.2fs  inf=%.3fms/1k  (%.1fs elapsed)",
                        group, model_name, ds_label,
                        r.get("accuracy", float("nan")),
                        r.get("f1",       float("nan")),
                        r.get("auc",      float("nan")),
                        r.get("brier_score", float("nan")),
                        r.get("ece",      float("nan")),
                        r.get("train_time_s", float("nan")),
                        r.get("inference_ms_per_1k", float("nan")),
                        time.perf_counter() - t0,
                    )
                except Exception as exc:
                    logger.error(
                        "  ERR %s / %s / Dataset %s: %s",
                        group, model_name, ds_label, exc, exc_info=False
                    )

    df_out = pd.DataFrame(all_results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    logger.info("Results saved → %s  (%d rows)", output_csv, len(df_out))
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Summary printer
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df_all: pd.DataFrame, df_baseline: pd.DataFrame, cfg: dict) -> None:
    """Print a clean comparison table vs the original paper."""
    paper = cfg.get("original_results", {})

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY — Comparison with Beaman & Isah (2021)")
    print("=" * 80)

    datasets = sorted(df_all["dataset"].unique()) if "dataset" in df_all.columns else [""]

    for ds in datasets:
        sub     = df_all[df_all["dataset"] == ds] if ds else df_all
        sub_bas = df_baseline[df_baseline["dataset"] == ds] if (
            ds and "dataset" in df_baseline.columns) else df_baseline

        paper_ds = paper.get(f"dataset_{ds.lower()}", {})

        print(f"\n{'─'*80}")
        print(f"  Dataset {ds}")
        print(f"{'─'*80}")

        # Best overall
        if not sub.empty:
            best = sub.loc[sub["accuracy"].idxmax()]
            print(f"  Best accuracy: {best['accuracy']:.4f}  "
                  f"({best.get('model_name','?')} / {best.get('feature_group','?')})")
            print(f"  Best F1:       {sub['f1'].max():.4f}")
            print(f"  Best AUC:      {sub['auc'].max():.4f}")
            print()

        # G6 row-by-row comparison with paper
        g6 = sub_bas if sub_bas.empty else (
            sub_bas[sub_bas["feature_group"] == "G6"] if "feature_group" in sub_bas.columns
            else sub_bas
        )
        if not g6.empty:
            print(f"  {'Model':<12} {'Our acc':>9}  {'Paper acc':>10}  "
                  f"{'Δ acc':>8}  {'Our F1':>8}  {'Paper F1':>9}  "
                  f"{'Brier':>7}  {'ECE':>7}")
            print("  " + "-" * 78)
            for _, row in g6.sort_values("accuracy", ascending=False).iterrows():
                mdl = row.get("model_name", "?")
                p   = paper_ds.get(mdl, {})
                pa  = p.get("accuracy", None)
                pf  = p.get("f1", None)
                delta = f"{row['accuracy'] - pa:+.4f}" if pa is not None else "  N/A  "
                print(
                    f"  {mdl:<12} {row['accuracy']:>9.4f}  "
                    f"{pa if pa else 'N/A':>10}  "
                    f"{delta:>8}  "
                    f"{row.get('f1', float('nan')):>8.4f}  "
                    f"{pf if pf else 'N/A':>9}  "
                    f"{row.get('brier_score', float('nan')):>7.4f}  "
                    f"{row.get('ece', float('nan')):>7.4f}"
                )

    print("\n" + "=" * 80)
    print(f"Full results : {RESULTS_DIR / 'feature_combinations' / 'comparison.csv'}")
    print(f"Baseline     : {RESULTS_DIR / 'original_baseline' / 'baseline.csv'}")
    print(f"Report       : {RESULTS_DIR / 'comparison_report.md'}")
    print(f"Figures      : {RESULTS_DIR / 'figures' / '*.png'}")
    print(f"Log          : {log_file}")
    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EmailHeaderAnomalyDetection_Extended — full pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--fast", action="store_true",
                   help="Quick test: cv=3, groups G2/G5/G6/G7, models RF/MLP/KNN, 300 samples")
    p.add_argument("--real-data", action="store_true",
                   help="Use real email files (falls back to synthetic if dirs missing)")
    p.add_argument("--datasetA", default=str(ROOT / "data" / "raw" / "spamassassin"),
                   help="Path to SpamAssassin root (needs ham/ and spam/ sub-dirs)")
    p.add_argument("--datasetB", default=str(ROOT / "data" / "raw" / "phishing"),
                   help="Path to phishing root (needs ham/ and phishing/ sub-dirs)")
    p.add_argument("--max-per-class", type=int, default=None,
                   help="Cap emails per class when parsing real data (for quick tests)")
    p.add_argument("--n-ham",  type=int, default=1000, help="Synthetic ham count per dataset")
    p.add_argument("--n-spam", type=int, default=1000, help="Synthetic spam count per dataset")
    p.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds (default 5)")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["cv_folds"] = args.cv_folds

    # ── Fast mode overrides ───────────────────────────────────────────────────
    if args.fast:
        cfg["cv_folds"] = 3
        groups = ["G2", "G5", "G6", "G7"]
        models = ["RF", "MLP", "KNN"]
        n_ham = n_spam = 300
        logger.info("FAST MODE: cv=3, groups=%s, models=%s, 300 samples/class", groups, models)
    else:
        groups = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
        models = ["RF", "SVM", "MLP", "KNN", "Stacking", "OC_SVM"]
        n_ham  = args.n_ham
        n_spam = args.n_spam

    t_pipeline = time.perf_counter()
    logger.info("=" * 60)
    logger.info("EmailHeaderAnomalyDetection_Extended — Pipeline Start")
    logger.info("cv_folds=%d  groups=%s  models=%s", cfg["cv_folds"], groups, models)
    logger.info("=" * 60)

    # ── Step 1: Prepare datasets ──────────────────────────────────────────────
    logger.info("\n[Step 1/4] Preparing datasets …")
    dataset_paths = prepare_datasets(args, n_ham, n_spam)
    if not dataset_paths:
        logger.error("No datasets available — aborting.")
        sys.exit(1)

    # ── Step 2: G6 baseline (paper reproduction) ──────────────────────────────
    logger.info("\n[Step 2/4] Reproducing G6 baseline (original paper settings) …")
    baseline_csv = RESULTS_DIR / "original_baseline" / "baseline.csv"
    df_baseline = run_experiments(
        dataset_paths=dataset_paths,
        cfg=cfg,
        groups=["G6"],
        models=models,
        output_csv=baseline_csv,
    )

    # ── Step 3: Full matrix ───────────────────────────────────────────────────
    logger.info("\n[Step 3/4] Running full experiment matrix (%d groups × %d models) …",
                len(groups), len(models))
    comparison_csv = RESULTS_DIR / "feature_combinations" / "comparison.csv"
    df_all = run_experiments(
        dataset_paths=dataset_paths,
        cfg=cfg,
        groups=groups,
        models=models,
        output_csv=comparison_csv,
    )

    # ── Step 4: Report + figures ──────────────────────────────────────────────
    logger.info("\n[Step 4/4] Generating comparison report and figures …")
    try:
        from src.experiments.compare_results import compare_results
        compare_results(
            results_csv=comparison_csv,
            baseline_csv=baseline_csv,
            out_dir=RESULTS_DIR,
            cfg=cfg,
        )
        logger.info("Report and figures generated.")
    except Exception as e:
        logger.warning("compare_results failed (%s) — skipping plots.", e)

    # ── Step 5: Print summary ─────────────────────────────────────────────────
    print_summary(df_all, df_baseline, cfg)

    logger.info(
        "\n[DONE] Total pipeline time: %.1f minutes",
        (time.perf_counter() - t_pipeline) / 60
    )


if __name__ == "__main__":
    main()

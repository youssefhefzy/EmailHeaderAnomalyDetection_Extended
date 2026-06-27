"""
generate_synthetic_demo.py
--------------------------
Generate a small synthetic email-header dataset and run the full pipeline
WITHOUT needing real email files.  Useful for:

  * CI / smoke-testing the codebase.
  * Demonstrating the pipeline end-to-end.
  * Validating that all modules import and execute correctly.

Usage
-----
    python scripts/generate_synthetic_demo.py          # full demo
    python scripts/generate_synthetic_demo.py --fast   # 3-fold CV, fast models only

Outputs
-------
    data/processed/synthetic_A.csv
    data/processed/synthetic_B.csv
    results/feature_combinations/synthetic_comparison.csv
    results/original_baseline/baseline_synthetic.csv
    results/comparison_report.md
    results/figures/*.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from src.experiments.run_experiments import run_all_experiments
from src.experiments.compare_results import compare_results

# ── Config ────────────────────────────────────────────────────────────────────
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

RNG = np.random.default_rng(42)


# ── Synthetic feature generator ───────────────────────────────────────────────

def _make_dataset(n_ham: int, n_spam: int, dataset: str = "A") -> pd.DataFrame:
    """
    Synthesise a DataFrame with realistic feature distributions for ham vs spam/phishing.

    Each feature group maps directly to the groups used in the experiments (G1–G8).
    Distributions are calibrated to reflect patterns observed in TREC 2007 and
    the phishing corpus used by Beaman & Isah (2021).
    """
    n = n_ham + n_spam
    labels = np.array([0] * n_ham + [1] * n_spam)

    def bern(p_ham: float, p_spam: float) -> np.ndarray:
        probs = np.where(labels == 0, p_ham, p_spam)
        return RNG.binomial(1, probs).astype(int)

    def norm(mu_ham: float, mu_spam: float, sd: float = 1.0) -> np.ndarray:
        mu = np.where(labels == 0, mu_ham, mu_spam)
        return np.clip(RNG.normal(mu, sd), 0, None)

    rows = {
        "label": labels,

        # ── G1: header presence ──────────────────────────────────────
        "has_received":                    bern(0.99, 0.90),
        "has_message_id":                  bern(0.99, 0.75),
        "has_date":                        bern(1.00, 0.95),
        "has_from":                        bern(1.00, 0.99),
        "has_to":                          bern(0.98, 0.85),
        "has_reply_to":                    bern(0.30, 0.70),
        "has_return_path":                 bern(0.90, 0.60),
        "has_mime_version":                bern(0.95, 0.80),
        "has_content_type":                bern(0.99, 0.95),
        "has_content_transfer_encoding":   bern(0.85, 0.78),
        "has_subject":                     bern(0.99, 0.97),
        "has_x_mailer":                    bern(0.50, 0.35),
        "has_x_originating_ip":            bern(0.30, 0.55),
        "has_x_spam_status":               bern(0.60, 0.40),
        "has_dkim_signature":              bern(0.70, 0.30),
        "has_domainkey_signature":         bern(0.20, 0.10),
        "has_authentication_results":      bern(0.75, 0.40),
        "has_arc_authentication_results":  bern(0.40, 0.20),
        "has_delivered_to":                bern(0.70, 0.45),
        "has_envelope_to":                 bern(0.30, 0.25),
        "has_errors_to":                   bern(0.15, 0.30),
        "has_list_unsubscribe":            bern(0.40, 0.55),
        "has_precedence":                  bern(0.35, 0.50),
        "has_x_priority":                  bern(0.20, 0.55),
        "has_spf":                         bern(0.85, 0.50),
        "has_dkim":                        bern(0.70, 0.30),
        "has_dmarc":                       bern(0.65, 0.25),

        # ── G2: domain matching ──────────────────────────────────────
        "from_domain_len":                  norm(12, 18, 4),
        "from_eq_reply_to_domain":          bern(0.85, 0.20),
        "from_eq_return_path_domain":       bern(0.90, 0.25),
        "from_eq_mid_domain":               bern(0.88, 0.30),
        "from_eq_first_received_domain":    bern(0.80, 0.35),
        "reply_to_present_and_differs":     bern(0.10, 0.65),
        "return_path_present_and_differs":  bern(0.08, 0.60),
        "to_domain_eq_from_domain":         bern(0.30, 0.20),
        "mid_domain_eq_from_domain":        bern(0.88, 0.30),
        "envelope_to_eq_from_domain":       bern(0.50, 0.35),
        "reply_to_subdomain_of_from":       bern(0.15, 0.05),

        # ── G3: routing ──────────────────────────────────────────────
        "n_received_hops":               norm(3, 6, 2).astype(int),
        "missing_received":              bern(0.01, 0.12),
        "malformed_received_hops":       norm(0, 1.5, 1).astype(int),
        "n_unique_hop_ips":              norm(2, 4, 1.5).astype(int),
        "n_private_ips_in_received":     norm(0.5, 1.5, 1).astype(int),
        "tz_mismatch_in_received":       bern(0.05, 0.40),
        "n_unique_tz_offsets":           norm(1, 2.5, 1).astype(int),
        "has_unusual_tz":                bern(0.03, 0.35),
        "n_hop_domains":                 norm(3, 5, 2).astype(int),
        "date_vs_received_anomaly":      bern(0.02, 0.25),
        "excessive_hops":                bern(0.01, 0.15),

        # ── G4: structural ───────────────────────────────────────────
        "duplicate_headers":             norm(0, 1.5, 1).astype(int),
        "subject_non_ascii":             bern(0.05, 0.30),
        "from_non_ascii":                bern(0.02, 0.25),
        "malformed_date":                bern(0.01, 0.20),
        "subject_len":                   norm(35, 55, 20),
        "subject_has_re":                bern(0.25, 0.10),
        "subject_all_caps":              bern(0.02, 0.20),
        "subject_exclaim_count":         norm(0.1, 2.0, 1),
        "subject_question_count":        norm(0.1, 0.8, 0.5),
        "is_base64_encoded":             bern(0.30, 0.55),
        "is_quoted_printable":           bern(0.45, 0.30),
        "mid_malformed":                 bern(0.02, 0.40),
        "mid_len":                       norm(50, 35, 20),
        "is_multipart":                  bern(0.70, 0.55),
        "is_html_only":                  bern(0.15, 0.55),
        "total_header_count":            norm(18, 12, 5),

        # ── G5: authentication ───────────────────────────────────────
        "spf_result":   np.where(labels == 0,
                            RNG.choice([1, 0, -1], n, p=[0.80, 0.15, 0.05]),
                            RNG.choice([1, 0, -1], n, p=[0.20, 0.30, 0.50])),
        "dkim_result":  np.where(labels == 0,
                            RNG.choice([1, 0, -1], n, p=[0.75, 0.20, 0.05]),
                            RNG.choice([1, 0, -1], n, p=[0.15, 0.35, 0.50])),
        "dmarc_result": np.where(labels == 0,
                            RNG.choice([1, 0, -1], n, p=[0.70, 0.25, 0.05]),
                            RNG.choice([1, 0, -1], n, p=[0.15, 0.40, 0.45])),
        "dkim_signature_present":        bern(0.70, 0.30),
        "dkim_signature_valid_format":   bern(0.68, 0.20),
        "auth_results_present":          bern(0.75, 0.40),
        "all_auth_pass":                 bern(0.65, 0.10),
        "any_auth_fail":                 bern(0.05, 0.55),

        # ── Misc ─────────────────────────────────────────────────────
        "x_mailer_len":       norm(20, 10, 10),
        "x_priority_value":   np.where(labels == 0,
                                  RNG.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.70, 0.10, 0.05]),
                                  RNG.choice([1, 2, 3, 4, 5], n, p=[0.30, 0.20, 0.30, 0.10, 0.10])),
        "is_bulk_precedence": bern(0.15, 0.55),
        "list_unsubscribe_len": norm(40, 80, 30),
        "n_to_addresses":     norm(1, 3, 2).astype(int),
        "n_cc_addresses":     norm(0.5, 0.3, 1).astype(int),
        "n_bcc_addresses":    norm(0.1, 0.5, 0.5).astype(int),
        "x_spam_score":       norm(-1.5, 6.0, 3.0),
    }

    df = pd.DataFrame(rows)

    # Dataset B: zero-out most routing features (less available in phishing corpus)
    if dataset == "B":
        for col in ["n_received_hops", "tz_mismatch_in_received",
                    "has_unusual_tz", "n_unique_tz_offsets"]:
            if col in df.columns:
                df[col] = df[col] * RNG.binomial(1, 0.4, n)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Quick run: cv=3, only RF/KNN/MLP, groups G2/G5/G6/G7")
    args = parser.parse_args()

    cfg = dict(CFG)
    if args.fast:
        cfg["cv_folds"] = 3
        demo_groups = ["G2", "G5", "G6", "G7"]
        demo_models = ["RF", "KNN", "MLP"]
        n_ham = n_spam = 300
    else:
        cfg["cv_folds"] = 5
        demo_groups = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
        demo_models = ["RF", "SVM", "MLP", "KNN", "Stacking", "OC_SVM"]
        n_ham = n_spam = 500

    print("=" * 60)
    print("Synthetic Demo — EmailHeaderAnomalyDetection_Extended")
    print("=" * 60)

    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate datasets ─────────────────────────────────────────
    print(f"\n[1/4] Generating synthetic datasets ({n_ham} ham, {n_spam} spam each) …")
    df_A = _make_dataset(n_ham, n_spam, dataset="A")
    path_A = processed_dir / "synthetic_A.csv"
    df_A.to_csv(path_A, index=False)
    print(f"      Dataset A → {path_A}  shape={df_A.shape}")

    df_B = _make_dataset(n_ham // 2, n_spam // 2, dataset="B")
    path_B = processed_dir / "synthetic_B.csv"
    df_B.to_csv(path_B, index=False)
    print(f"      Dataset B → {path_B}  shape={df_B.shape}")

    # ── Step 2: Run all experiments ───────────────────────────────────────
    print(f"\n[2/4] Running experiments (cv={cfg['cv_folds']}, groups={demo_groups}, models={demo_models}) …")
    out_csv = ROOT / "results" / "feature_combinations" / "synthetic_comparison.csv"
    df_results = run_all_experiments(
        dataset_paths={"A": path_A, "B": path_B},
        cfg=cfg,
        groups=demo_groups,
        models=demo_models,
        output_csv=out_csv,
    )
    print(f"      Results → {out_csv}  ({len(df_results)} rows)")

    # ── Step 3: Reproduce G6 baseline ────────────────────────────────────
    print("\n[3/4] Reproducing G6 baseline …")
    baseline_csv = ROOT / "results" / "original_baseline" / "baseline_synthetic.csv"
    run_all_experiments(
        dataset_paths={"A": path_A, "B": path_B},
        cfg=cfg,
        groups=["G6"],
        models=demo_models,
        output_csv=baseline_csv,
    )
    print(f"      Baseline → {baseline_csv}")

    # ── Step 4: Compare and report ────────────────────────────────────────
    print("\n[4/4] Generating comparison report and figures …")
    compare_results(
        results_csv=out_csv,
        baseline_csv=baseline_csv,
        out_dir=ROOT / "results",
        cfg=cfg,
    )

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"Results CSV  : {out_csv}")
    print(f"Report       : {ROOT / 'results' / 'comparison_report.md'}")
    print(f"Figures      : {ROOT / 'results' / 'figures' / '*.png'}")


if __name__ == "__main__":
    main()

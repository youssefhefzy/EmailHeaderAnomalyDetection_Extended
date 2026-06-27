# EmailHeaderAnomalyDetection_Extended

Reproducible Python implementation of:

> Beaman, C. & Isah, H. (2021). *Anomaly Detection in Emails using Machine
> Learning and Header Information.* Canadian Institute for Cybersecurity,
> University of New Brunswick.

Extended with: Stacking ensemble, One-Class SVM, confidence/calibration
metrics (Brier score, ECE), permutation importance, and a cross-platform
single-entry-point runner (`run_all.py`).

---

## Quick Start (no real data required)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Fast smoke-test (synthetic data, ~2 min)
python run_all.py --fast

# 2b. Full run — synthetic data, all 8 groups, 6 models, cv=5 (~20–40 min)
python run_all.py

# 2c. Full run — real emails (see "Getting the Data" below)
python run_all.py --real-data
```

All outputs land in `results/`:

| File | Description |
|------|-------------|
| `results/original_baseline/baseline.csv` | G6 baseline (paper reproduction) |
| `results/feature_combinations/comparison.csv` | Full G1-G8 × all-model matrix |
| `results/comparison_report.md` | Markdown report vs paper benchmarks |
| `results/figures/*.png` | Heatmaps, scatter plots, bar charts |
| `results/run_all.log` | Full execution log |

---

## Getting the Data (optional — real emails)

### Dataset A — SpamAssassin / TREC 2007

Download the **SpamAssassin Public Corpus** from:
<https://spamassassin.apache.org/old/publiccorpus/>

Place the emails in:
```
data/raw/spamassassin/
    ham/        ← legitimate emails (easy_ham_*, hard_ham_*)
    spam/       ← spam emails (spam_*, spam_2*)
```

### Dataset B — Phishing corpus (2017–2020)

Download from: <https://monkey.org/~jose/phishing/>

Place the emails in:
```
data/raw/phishing/
    ham/        ← legitimate emails (same ham set used for dataset A)
    phishing/   ← phishing emails
```

Then run:
```bash
python run_all.py --real-data
```

> **Tip:** use `--max-per-class 2000` to cap at 2 000 emails per class while
> testing, then remove the flag for the full run.

You can also build the processed CSVs separately:

```bash
# Windows (PowerShell) — use double-dash args on one line
python src/data/make_dataset.py --dataset A --input data/raw/spamassassin --output data/processed/datasetA_processed.csv
python src/data/make_dataset.py --dataset B --input data/raw/phishing --output data/processed/datasetB_processed.csv
```

---

## Running the Experiments

### Option A — `run_all.py` (recommended, works on Windows + Linux)

```bash
# Quickest test
python run_all.py --fast

# Full synthetic run
python run_all.py --n-ham 1000 --n-spam 1000 --cv-folds 5

# Full real-data run
python run_all.py --real-data --datasetA data/raw/spamassassin --datasetB data/raw/phishing
```

### Option B — Step-by-step (advanced)

```bash
# 1. Build CSVs from raw emails (skip if using synthetic)
python src/data/make_dataset.py --dataset A --input data/raw/spamassassin --output data/processed/datasetA_processed.csv
python src/data/make_dataset.py --dataset B --input data/raw/phishing      --output data/processed/datasetB_processed.csv

# 2. Run experiments (single command, no multi-line needed)
python src/experiments/run_experiments.py --datasetA data/processed/datasetA_processed.csv --datasetB data/processed/datasetB_processed.csv --output results/feature_combinations/comparison.csv

# 3. Compare with paper baseline
python src/experiments/compare_results.py --results results/feature_combinations/comparison.csv --baseline results/original_baseline/baseline.csv --out_dir results/

# 4. Synthetic demo (generates data + runs pipeline in one go)
python scripts/generate_synthetic_demo.py
python scripts/generate_synthetic_demo.py --fast   # quick version
```

---

## Project Structure

```
EmailHeaderAnomalyDetection_Extended/
├── run_all.py                       ← MAIN ENTRY POINT (cross-platform)
├── config.yaml                      ← Hyperparameters, paths, paper baselines
├── requirements.txt
├── README.md
│
├── src/
│   ├── data/
│   │   ├── extract_headers.py       ← RFC-2822 parser → 94 header features
│   │   └── make_dataset.py          ← Build processed CSVs from raw emails
│   │
│   ├── features/
│   │   ├── feature_groups.py        ← G1–G8 feature group definitions
│   │   └── feature_importance.py   ← Permutation importance, RFECV, MDI
│   │
│   ├── models/
│   │   ├── train.py                 ← RF, SVM, MLP, KNN, OC-SVM (CV + split)
│   │   ├── stacking.py              ← Stacked ensemble (RF+MLP+KNN → LR)
│   │   └── confidence.py           ← Confidence scores, Brier, ECE, reliability
│   │
│   ├── experiments/
│   │   ├── run_experiments.py       ← Full G1-G8 × model experiment matrix
│   │   └── compare_results.py      ← Plots, markdown report, paper comparison
│   │
│   └── utils/
│       ├── compat.py               ← Optional-dep shims (tqdm, xgboost, lgb)
│       ├── metrics.py               ← Metric helpers, aggregation, pretty print
│       └── plotting.py             ← Heatmaps, scatter, bar charts (Agg backend)
│
├── scripts/
│   └── generate_synthetic_demo.py  ← Full demo without real email files
│
├── data/
│   ├── raw/                         ← Place real emails here (not in repo)
│   │   ├── spamassassin/ham/
│   │   ├── spamassassin/spam/
│   │   ├── phishing/ham/
│   │   └── phishing/phishing/
│   └── processed/                   ← Auto-generated CSVs
│
└── results/
    ├── original_baseline/           ← G6 baseline CSV + importance CSVs
    ├── feature_combinations/        ← Full matrix comparison CSV
    ├── figures/                     ← PNG plots
    └── comparison_report.md         ← Auto-generated markdown report
```

---

## Feature Groups

| Group | Name | Features | Description |
|-------|------|----------|-------------|
| G1 | Header Presence | 27 | Binary flags: is each key header present? |
| G2 | Domain Matching | 11 | From ↔ Reply-To ↔ Return-Path ↔ Message-ID domain checks |
| G3 | Routing Anomalies | 11 | Received-chain hops, timezone mismatches, IP diversity |
| G4 | Structural | 16 | Duplicates, non-ASCII, malformed Date, Subject features |
| G5 | Authentication | 8 | SPF / DKIM / DMARC pass/fail/neutral (encoded as +1/0/−1) |
| G6 | Top-30 (paper) | 30 | Permutation-importance top-30 (original paper baseline) |
| G7 | G2+G3+G5 | 30 | Combined routing + auth + domain matching |
| G8 | Top-5 | 5 | Fastest-inference subset from G6 |

---

## Models

| ID | Model | Notes |
|----|-------|-------|
| RF | Random Forest | 100 trees, `n_jobs=1` |
| SVM | SVM (RBF) | Platt-calibrated via `CalibratedClassifierCV` |
| MLP | Neural Network | 100→50, ReLU, early stopping |
| KNN | K-Nearest Neighbours | k=5, Euclidean |
| Stacking | Stacked Ensemble | RF+MLP+KNN base → LR meta |
| OC_SVM | One-Class SVM | Trained on ham only; anomaly detection mode |

---

## Metrics

All models are evaluated via **stratified k-fold cross-validation** (default k=5, configurable):

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct-classification rate |
| F1 | Harmonic mean of precision and recall |
| AUC | Area under the ROC curve |
| Brier Score | Probabilistic accuracy (lower = better) |
| ECE | Expected Calibration Error (lower = better) |
| Train time | Mean per-fold training time (seconds) |
| Inference | Inference speed (ms per 1 000 emails) |

---

## Paper Baselines (Table VII — G6, Dataset A)

| Model | Paper Acc | Paper F1 |
|-------|-----------|----------|
| RF | 0.998 | 0.998 |
| SVM | 0.991 | 0.991 |
| MLP | 0.989 | 0.988 |
| KNN | 0.985 | 0.984 |
| Stacking | 0.997 | 0.997 |

> **Note on synthetic data:** Our default mode uses synthetic datasets
> generated to match the statistical distributions of TREC 2007 and the
> monkey.org phishing corpus. High accuracy on synthetic data (often ≥99%)
> is expected because the generative distributions are learnable by design.
> To reproduce the paper's exact numbers, download the real datasets.

---

## Known Issues Fixed in This Version

| Issue | Fix |
|-------|-----|
| `'label' column not found` when running `make_dataset.py` on empty/flat directories | Proper error messages + `FileNotFoundError` with download instructions |
| `bash` not found on Windows PowerShell | All scripts are pure Python; use `python run_all.py` instead |
| Multi-line PowerShell `\` commands fail | `run_all.py` replaces all multi-line CLI commands |
| `tqdm`, `xgboost`, `lightgbm`, `mlxtend` import errors | `src/utils/compat.py` shim; removed from `requirements.txt` |
| `n_jobs=-1` hangs / deadlocks in some environments | Changed to `n_jobs=1` throughout (config + source files) |
| Duplicate `has_mime_version` / `has_x_mailer` columns in synthetic generator | Removed duplicates from `generate_synthetic_demo.py` |
| `run_experiments.py` used `n_jobs=-1` for permutation importance (hangs) | Reduced to `n_repeats=3, n_jobs=1` |

---

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
pyyaml>=5.4.0
joblib>=1.0.0
tabulate>=0.8.9   # for DataFrame.to_markdown()
```

Install:
```bash
pip install -r requirements.txt
```

Optional (install separately if desired):
```bash
pip install tqdm xgboost lightgbm mlxtend imbalanced-learn
```

---

## Citation

```bibtex
@inproceedings{beaman2021anomaly,
  title     = {Anomaly Detection in Emails using Machine Learning and Header Information},
  author    = {Beaman, Craig and Isah, Haruna},
  year      = {2021},
  institution = {Canadian Institute for Cybersecurity, University of New Brunswick}
}
```

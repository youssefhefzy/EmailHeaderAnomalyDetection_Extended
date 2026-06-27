"""
make_dataset.py
---------------
Build processed CSV files from raw email directories.

Usage
-----
    python src/data/make_dataset.py \
        --dataset A \
        --input   data/raw/spamassassin \
        --output  data/processed/datasetA_processed.csv

    python src/data/make_dataset.py \
        --dataset B \
        --input   data/raw/phishing \
        --output  data/processed/datasetB_processed.csv

Dataset A layout expected::

    data/raw/spamassassin/
        ham/        # legitimate emails
        spam/       # spam emails

Dataset B layout expected::

    data/raw/phishing/
        ham/        # legitimate emails
        phishing/   # phishing emails
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.extract_headers import EmailHeaderExtractor, extract_features_from_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset-specific builders
# ---------------------------------------------------------------------------

def build_dataset_A(
    input_dir: Path,
    output_path: Path,
    max_per_class: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build Dataset A (SpamAssassin) feature CSV.

    Labels: 0 = ham, 1 = spam.

    Parameters
    ----------
    input_dir:
        Root directory with ``ham/`` and ``spam/`` sub-directories.
        Also accepts a flat directory of mixed emails (all treated as ham).
    output_path:
        Where to write the processed CSV.
    max_per_class:
        Optional cap on emails per class (for quick testing).

    Returns
    -------
    pd.DataFrame
        Processed feature DataFrame.
    """
    logger.info("Building Dataset A from %s", input_dir)
    extractor = EmailHeaderExtractor()

    rows = []

    ham_dir = input_dir / "ham"
    spam_dir = input_dir / "spam"

    if not ham_dir.exists() and not spam_dir.exists():
        # Flat structure — treat all files as ham
        logger.warning(
            "No 'ham/' or 'spam/' sub-dirs found in %s. "
            "Treating all files as ham. "
            "Expected layout: input_dir/ham/ and input_dir/spam/",
            input_dir,
        )
        ham_dir = input_dir

    if ham_dir.exists():
        logger.info("Extracting ham emails …")
        rows += extract_features_from_directory(
            ham_dir, label=0, extractor=extractor, max_emails=max_per_class
        )
    else:
        logger.warning("ham/ directory not found in %s", input_dir)

    if spam_dir.exists():
        logger.info("Extracting spam emails …")
        rows += extract_features_from_directory(
            spam_dir, label=1, extractor=extractor, max_emails=max_per_class
        )
    else:
        logger.warning("spam/ directory not found in %s", input_dir)

    if not rows:
        raise FileNotFoundError(
            f"No email files found in {input_dir}. "
            "Please download the SpamAssassin dataset and place emails in "
            f"{input_dir}/ham/ and {input_dir}/spam/. "
            "See README.md § 'Getting the Data' for download instructions. "
            "Alternatively run: python scripts/generate_synthetic_demo.py"
        )

    df = pd.DataFrame(rows)
    if "label" not in df.columns:
        raise ValueError("'label' column not found in feature DataFrame.")
    df = _clean_and_encode(df, dataset="A")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Dataset A saved to %s — shape %s", output_path, df.shape)
    return df


def build_dataset_B(
    input_dir: Path,
    output_path: Path,
    max_per_class: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build Dataset B (phishing corpus) feature CSV.

    Labels: 0 = ham, 1 = phishing.

    For Dataset B, only domain-matching features (G2) are used to avoid
    overfitting to Dataset A's timezone / routing peculiarities — as per
    Beaman & Isah (2021).

    Parameters
    ----------
    input_dir:
        Root directory with ``ham/`` and ``phishing/`` sub-directories.
    output_path:
        Where to write the processed CSV.
    max_per_class:
        Optional cap on emails per class.

    Returns
    -------
    pd.DataFrame
        Processed feature DataFrame.
    """
    logger.info("Building Dataset B from %s", input_dir)
    extractor = EmailHeaderExtractor()

    rows = []

    ham_dir = input_dir / "ham"
    phishing_dir = input_dir / "phishing"

    if ham_dir.exists():
        logger.info("Extracting ham emails for Dataset B …")
        rows += extract_features_from_directory(
            ham_dir, label=0, extractor=extractor, max_emails=max_per_class
        )
    else:
        logger.warning("ham/ directory not found in %s", input_dir)

    if phishing_dir.exists():
        logger.info("Extracting phishing emails …")
        rows += extract_features_from_directory(
            phishing_dir, label=1, extractor=extractor, max_emails=max_per_class
        )
    else:
        logger.warning("phishing/ directory not found in %s", input_dir)

    if not rows:
        raise FileNotFoundError(
            f"No email files found in {input_dir}. "
            "Please download the phishing dataset and place emails in "
            f"{input_dir}/ham/ and {input_dir}/phishing/. "
            "See README.md § 'Getting the Data' for download instructions. "
            "Alternatively run: python scripts/generate_synthetic_demo.py"
        )

    df = pd.DataFrame(rows)
    if "label" not in df.columns:
        raise ValueError("'label' column not found in feature DataFrame.")
    df = _clean_and_encode(df, dataset="B")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Dataset B saved to %s — shape %s", output_path, df.shape)
    return df


# ---------------------------------------------------------------------------
# Cleaning / encoding
# ---------------------------------------------------------------------------

def _clean_and_encode(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Clean the raw feature DataFrame:

    1. Drop metadata columns (prefixed with ``_``).
    2. Impute missing values:
       - Numerical: median.
       - Boolean/binary: 0 (absent by default).
    3. Clip extreme outliers (cap at 99th percentile for count features).
    4. Ensure all features are numeric (encode any stray categoricals).

    Parameters
    ----------
    df:
        Raw DataFrame from :func:`extract_features_from_directory`.
    dataset:
        ``'A'`` or ``'B'``.  For B, authentication features that are
        almost always missing are replaced with 0.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for modelling.
    """
    # Drop metadata columns
    meta_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=meta_cols, errors="ignore")

    # Separate label
    label_col = "label"
    if label_col not in df.columns:
        raise ValueError("'label' column not found in feature DataFrame.")

    y = df[label_col].copy()
    X = df.drop(columns=[label_col])

    # Identify column types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Encode categorical columns (should be rare)
    le = LabelEncoder()
    for c in cat_cols:
        try:
            X[c] = le.fit_transform(X[c].astype(str))
        except Exception:
            X[c] = 0

    # Impute numerics
    imputer = SimpleImputer(strategy="median")
    X[num_cols] = imputer.fit_transform(X[num_cols])

    # Clip count features at 99th percentile
    count_cols = [
        c for c in num_cols
        if any(kw in c for kw in ("count", "n_", "len", "hops", "hop"))
    ]
    for c in count_cols:
        cap = X[c].quantile(0.99)
        X[c] = X[c].clip(upper=cap)

    # Reassemble
    df_clean = X.copy()
    df_clean[label_col] = y.values

    # Drop constant columns
    nunique = df_clean.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        logger.info("Dropping %d constant columns: %s", len(const_cols), const_cols)
        df_clean = df_clean.drop(columns=const_cols)

    # Dataset-B: zero-fill auth features that are mostly absent
    if dataset == "B":
        auth_cols = [c for c in df_clean.columns if any(
            kw in c for kw in ("spf", "dkim", "dmarc", "auth")
        )]
        df_clean[auth_cols] = df_clean[auth_cols].fillna(0)

    logger.info(
        "After cleaning — shape: %s, label distribution:\n%s",
        df_clean.shape, df_clean[label_col].value_counts().to_string()
    )
    return df_clean


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build processed email-header CSV datasets."
    )
    parser.add_argument(
        "--dataset", choices=["A", "B"], required=True,
        help="Which dataset to build (A = SpamAssassin, B = Phishing)."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to the raw email root directory."
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output CSV file path."
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Cap the number of emails per class (for quick tests)."
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "A":
        build_dataset_A(
            input_dir=args.input,
            output_path=args.output,
            max_per_class=args.max_per_class,
        )
    else:
        build_dataset_B(
            input_dir=args.input,
            output_path=args.output,
            max_per_class=args.max_per_class,
        )


if __name__ == "__main__":
    main()

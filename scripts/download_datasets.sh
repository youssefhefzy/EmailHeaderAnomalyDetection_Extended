#!/usr/bin/env bash
# ===========================================================================
# download_datasets.sh
# Download and extract the SpamAssassin and phishing email corpora.
#
# Usage:
#   bash scripts/download_datasets.sh
#
# Requires: wget, bzip2, tar
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
DATA_RAW="$ROOT/data/raw"

echo "=== Email Header Anomaly Detection — Dataset Downloader ==="
echo "Root: $ROOT"
echo "Target: $DATA_RAW"
echo ""

# ---------------------------------------------------------------------------
# Dataset A — SpamAssassin Public Corpus
# ---------------------------------------------------------------------------

SA_DIR="$DATA_RAW/spamassassin"
mkdir -p "$SA_DIR/ham" "$SA_DIR/spam"

SA_BASE="https://spamassassin.apache.org/old/publiccorpus"

declare -A SA_FILES=(
    ["20030228_easy_ham.tar.bz2"]="ham"
    ["20030228_easy_ham_2.tar.bz2"]="ham"
    ["20030228_hard_ham.tar.bz2"]="ham"
    ["20030228_spam.tar.bz2"]="spam"
    ["20050311_spam_2.tar.bz2"]="spam"
)

echo "--- Downloading SpamAssassin corpus ---"
for fname in "${!SA_FILES[@]}"; do
    dest_class="${SA_FILES[$fname]}"
    url="$SA_BASE/$fname"
    archive_path="$SA_DIR/$fname"

    if [ -f "$archive_path" ]; then
        echo "  Already downloaded: $fname"
    else
        echo "  Downloading: $url"
        wget -q --show-progress -O "$archive_path" "$url" || {
            echo "  WARNING: Failed to download $fname. Skipping."
            continue
        }
    fi

    echo "  Extracting $fname → $SA_DIR/$dest_class/ …"
    # Each archive extracts to a folder named after the category
    tar -xjf "$archive_path" -C "$SA_DIR/$dest_class/" --strip-components=1 2>/dev/null || true
done

echo ""
echo "SpamAssassin emails extracted:"
ham_count=$(find "$SA_DIR/ham" -type f | wc -l)
spam_count=$(find "$SA_DIR/spam" -type f | wc -l)
echo "  Ham:  $ham_count"
echo "  Spam: $spam_count"

# ---------------------------------------------------------------------------
# Dataset B — Phishing Email Corpus
# ---------------------------------------------------------------------------

PH_DIR="$DATA_RAW/phishing"
mkdir -p "$PH_DIR/ham" "$PH_DIR/phishing"

echo ""
echo "--- Phishing dataset ---"
echo "  The phishing email corpus used in the original paper is not freely"
echo "  redistributable via a single wget link.  Please obtain it from one"
echo "  of the following sources and place the emails in:"
echo ""
echo "    $PH_DIR/ham/        ← legitimate emails"
echo "    $PH_DIR/phishing/   ← phishing emails"
echo ""
echo "  Recommended sources:"
echo "    • CLAIR Collection (CMU):  https://www.cs.cmu.edu/~enron/"
echo "    • Jose Marcio's Phishing:  https://monkey.org/~jose/phishing/"
echo "    • Nazario Phishing Corpus: https://github.com/nickcowner/nazario-phishing"
echo "    • PhishTank samples:       https://www.phishtank.com/developer_info.php"
echo ""

# Try to download Nazario corpus from GitHub if available
NAZARIO_URL="https://github.com/nickcowner/nazario-phishing/archive/refs/heads/main.zip"
NAZARIO_ZIP="$PH_DIR/nazario.zip"

echo "  Attempting to download Nazario phishing corpus from GitHub …"
if wget -q --spider "$NAZARIO_URL" 2>/dev/null; then
    wget -q --show-progress -O "$NAZARIO_ZIP" "$NAZARIO_URL" && \
        unzip -q "$NAZARIO_ZIP" -d "$PH_DIR/phishing_raw/" 2>/dev/null && \
        find "$PH_DIR/phishing_raw/" -name "*.eml" -exec mv {} "$PH_DIR/phishing/" \; && \
        echo "  Nazario corpus downloaded and extracted." || \
        echo "  Download succeeded but extraction had issues.  Check $NAZARIO_ZIP manually."
else
    echo "  Nazario corpus not accessible. Please download manually (see above)."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "=== Download Summary ==="
echo "Dataset A (SpamAssassin):"
echo "  Ham files:      $(find "$SA_DIR/ham" -type f | wc -l)"
echo "  Spam files:     $(find "$SA_DIR/spam" -type f | wc -l)"
echo ""
echo "Dataset B (Phishing):"
echo "  Ham files:      $(find "$PH_DIR/ham" -type f | wc -l)"
echo "  Phishing files: $(find "$PH_DIR/phishing" -type f | wc -l)"
echo ""
echo "Next step — build processed CSVs:"
echo "  python src/data/make_dataset.py --dataset A --input data/raw/spamassassin --output data/processed/datasetA_processed.csv"
echo "  python src/data/make_dataset.py --dataset B --input data/raw/phishing    --output data/processed/datasetB_processed.csv"

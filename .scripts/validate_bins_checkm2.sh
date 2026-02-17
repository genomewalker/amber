#!/bin/bash
set -eo pipefail

# Validate bins using CheckM2 only (no GUNC)
# Usage: validate_bins_checkm2.sh <bin_directory> [output_dir] [threads]

set +u
source /maps/projects/fernandezguerra/apps/opt/conda/etc/profile.d/conda.sh
conda activate assembly
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
set -u

if [ $# -lt 1 ]; then
    echo "Usage: $0 <bin_directory> [output_dir] [threads]"
    exit 1
fi

BIN_DIR="$1"
OUTPUT_DIR="${2:-$(dirname "$BIN_DIR")}"
THREADS="${3:-16}"

CHECKM2_BIN="/maps/projects/fernandezguerra/apps/opt/conda/envs/assembly/bin/checkm2"

if [ ! -d "$BIN_DIR" ]; then
    echo "Error: Bin directory not found: $BIN_DIR"
    exit 1
fi

NUM_BINS=$(ls "$BIN_DIR"/*.fa 2>/dev/null | wc -l)
if [ "$NUM_BINS" -eq 0 ]; then
    echo "Error: No *.fa files found in $BIN_DIR"
    exit 1
fi

echo "========================================="
echo "  CheckM2 VALIDATION"
echo "========================================="
echo "Bins: $NUM_BINS in $BIN_DIR"
echo ""

CHECKM2_DIR="$OUTPUT_DIR/checkm2"
rm -rf "$CHECKM2_DIR"

$CHECKM2_BIN predict \
    --input "$BIN_DIR" \
    --output-directory "$CHECKM2_DIR" \
    --extension fa \
    --threads "$THREADS" \
    --force

CHECKM2_REPORT="$CHECKM2_DIR/quality_report.tsv"

if [ ! -f "$CHECKM2_REPORT" ]; then
    echo "Error: CheckM2 report not found"
    exit 1
fi

echo ""
echo "========================================="
echo "  RESULTS"
echo "========================================="

HQ_COUNT=$(awk -F'\t' 'NR>1 && $2>=90 && $3<5 {count++} END {print count+0}' "$CHECKM2_REPORT")
MQ_COUNT=$(awk -F'\t' 'NR>1 && $2>=50 && $2<90 && $3<10 {count++} END {print count+0}' "$CHECKM2_REPORT")

echo ""
echo "High-quality bins (>=90% complete, <5% contam): $HQ_COUNT"
echo "Medium-quality bins (>=50% complete, <10% contam): $MQ_COUNT"
echo ""

if [ "$HQ_COUNT" -gt 0 ]; then
    echo "HQ bins:"
    awk -F'\t' 'NR>1 && $2>=90 && $3<5 {printf "  %s: %.1f%% complete, %.1f%% contam\n", $1, $2, $3}' \
        "$CHECKM2_REPORT" | sort -t: -k2 -rn
    echo ""
fi

echo "Report: $CHECKM2_REPORT"

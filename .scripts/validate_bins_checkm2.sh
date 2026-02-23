#!/bin/bash
set -eo pipefail

# Validate bins using CheckM2 only (no GUNC)
# Usage: validate_bins_checkm2.sh <bin_directory> [output_dir] [threads]

ASSEMBLY_ENV=/maps/projects/fernandezguerra/apps/opt/conda/envs/assembly
export PATH="$ASSEMBLY_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ASSEMBLY_ENV/lib:${LD_LIBRARY_PATH:-}"

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

NUM_FA=$(find "$BIN_DIR" -maxdepth 1 -name "*.fa" | wc -l)
NUM_GZ=$(find "$BIN_DIR" -maxdepth 1 -name "*.fa.gz" | wc -l)
NUM_BINS=$((NUM_FA + NUM_GZ))
if [ "$NUM_BINS" -eq 0 ]; then
    echo "Error: No *.fa or *.fa.gz files found in $BIN_DIR"
    exit 1
fi

echo "========================================="
echo "  CheckM2 VALIDATION"
echo "========================================="
echo "Bins: $NUM_BINS in $BIN_DIR"
echo ""

CHECKM2_DIR="$OUTPUT_DIR/checkm2"
rm -rf "$CHECKM2_DIR"

# If any bins are gzipped, decompress all into a temp dir so CheckM2 sees only .fa
if [ "$NUM_GZ" -gt 0 ]; then
    TMPBINS=$(mktemp -d)
    trap "rm -rf $TMPBINS" EXIT
    for fa in "$BIN_DIR"/*.fa; do
        [ -e "$fa" ] && cp "$fa" "$TMPBINS/"
    done
    for gz in "$BIN_DIR"/*.fa.gz; do
        gunzip -c "$gz" > "$TMPBINS/$(basename "${gz%.gz}")"
    done
    INPUT_DIR="$TMPBINS"
else
    INPUT_DIR="$BIN_DIR"
fi

$CHECKM2_BIN predict \
    --input "$INPUT_DIR" \
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

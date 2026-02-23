#!/bin/bash
# AMBER Paper Workflow entry point
# Usage: ./run_paper.sh [target] [extra snakemake args...]
#   target: qc_only (default), qc_full, all, phylo, beast2_run
# Example: ./run_paper.sh qc_only --dryrun

set -euo pipefail

SCRIPT_DIR=$(dirname $(realpath $0))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TARGET=${1:-qc_only}
shift || true  # remaining args passed to snakemake

source /maps/projects/fernandezguerra/apps/opt/conda/etc/profile.d/conda.sh
conda activate snakemake9

cd "$SCRIPT_DIR"

# Create log directory for SLURM outputs (profile needs it before jobs run)
OUTDIR=$(python3 -c "
import yaml, sys
c = yaml.safe_load(open('config.yaml'))
ts = '${TIMESTAMP}'
od = c['outdir']
print(f'{od}/{ts}')
")
mkdir -p "$OUTDIR/logs"

echo "=== AMBER Paper Workflow ==="
echo "Target  : $TARGET"
echo "Outdir  : $OUTDIR"
echo "Timestamp: $TIMESTAMP"
echo ""

snakemake \
    --snakefile "$SCRIPT_DIR/Snakefile" \
    --configfile "$SCRIPT_DIR/config.yaml" \
    --config timestamp="$TIMESTAMP" \
    --profile "$SCRIPT_DIR/profile" \
    --rerun-incomplete \
    --printshellcmds \
    "$TARGET" \
    "$@"

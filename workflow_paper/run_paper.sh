#!/bin/bash
# AMBER Paper Workflow entry point
# Usage: ./run_paper.sh [target] [extra snakemake args...]
#   target: qc_only (default), qc_full, all, phylo, beast2_run
# Flags: --local  run with local executor instead of SLURM (no GPU rules)
# Example: ./run_paper.sh build_snp_matrix --local --forcerun build_snp_matrix

set -euo pipefail

SCRIPT_DIR=$(dirname $(realpath $0))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TARGET=${1:-qc_only}
shift || true

LOCAL=0
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        LOCAL=1
    else
        EXTRA_ARGS+=("$arg")
    fi
done

source /maps/projects/fernandezguerra/apps/opt/conda/etc/profile.d/conda.sh
conda activate snakemake9

cd "$SCRIPT_DIR"

OUTDIR=$(python3 -c "
import yaml
c = yaml.safe_load(open('config.yaml'))
ts = '${TIMESTAMP}'
od = c['outdir']
print(f'{od}/{ts}' if ts != 'default' else od)
")
mkdir -p "$OUTDIR/logs"

echo "=== AMBER Paper Workflow ==="
echo "Target  : $TARGET"
echo "Outdir  : $OUTDIR"
echo "Executor: $([ $LOCAL -eq 1 ] && echo local || echo slurm)"
echo ""

if [[ $LOCAL -eq 1 ]]; then
    snakemake \
        --snakefile "$SCRIPT_DIR/Snakefile" \
        --configfile "$SCRIPT_DIR/config.yaml" \
        --config timestamp="$TIMESTAMP" \
        --executor local \
        --cores 16 \
        --rerun-incomplete \
        --printshellcmds \
        "$TARGET" \
        "${EXTRA_ARGS[@]}"
else
    snakemake \
        --snakefile "$SCRIPT_DIR/Snakefile" \
        --configfile "$SCRIPT_DIR/config.yaml" \
        --config timestamp="$TIMESTAMP" \
        --profile "$SCRIPT_DIR/profile" \
        --rerun-incomplete \
        --printshellcmds \
        "$TARGET" \
        "${EXTRA_ARGS[@]}"
fi

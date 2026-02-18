#!/bin/bash
#SBATCH --job-name=amber_ablation
#SBATCH --output=/maps/projects/caeg/people/kbd606/scratch/kapk-assm/amber/results/logs/%x_%A_%a.out
#SBATCH --partition=compregular
#SBATCH --nodelist=dandygpun01fl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --array=1-15

# Ablation ladder: 5 configs × 3 reps
# Config 1: Baseline (standard)
# Config 2: + DamageInfoNCE
# Config 3: + QualityLeiden
# Config 4: Full (DamageInfoNCE + QualityLeiden)
# Config 5: + MapEquation (QualityLeiden + map equation Phase 1b)

set -e
AMBER_DIR=/maps/projects/caeg/people/kbd606/scratch/kapk-assm/amber
cd "$AMBER_DIR"

source /maps/projects/fernandezguerra/apps/opt/conda/etc/profile.d/conda.sh
conda activate assembly

export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc

# Decode config and rep from array task ID (1-indexed)
TASK=${SLURM_ARRAY_TASK_ID}
CONFIG_ID=$(( (TASK - 1) / 3 + 1 ))  # 1-5
REP_ID=$(( (TASK - 1) % 3 + 1 ))     # 1-3

# Config names and extra flags
case $CONFIG_ID in
    1) CONFIG_NAME="baseline"
       EXTRA_FLAGS="" ;;
    2) CONFIG_NAME="damage_infonce"
       EXTRA_FLAGS="--damage-infonce" ;;
    3) CONFIG_NAME="quality_leiden"
       EXTRA_FLAGS="--quality-leiden" ;;
    4) CONFIG_NAME="full_amber"
       EXTRA_FLAGS="--damage-infonce --quality-leiden" ;;
    5) CONFIG_NAME="map_equation"
       EXTRA_FLAGS="--map-equation" ;;
    *) echo "Unknown config $CONFIG_ID"; exit 1 ;;
esac

# Get git info
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(git diff --quiet 2>/dev/null && echo "" || echo "-dirty")

# Create unique output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$AMBER_DIR/results/ablation/${CONFIG_NAME}/${SLURM_ARRAY_JOB_ID}_rep${REP_ID}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Test data paths
CONTIGS="/projects/caeg/people/kbd606/scratch/kapk-assm/119_50/assembly/3e3ce8e9f7-derep-assm/final.contigs.fa"
BAM="/projects/caeg/people/kbd606/scratch/tmp/amber_cython_N1/map/mapped.reassigned.bam"

# Fixed parameters
ENCODER_SEED=42
LEIDEN_SEED=1006
RESOLUTION=5.0
BANDWIDTH=0.2
PGR=50
THREADS=16

# Save parameters
cat > "$OUTPUT_DIR/params.txt" << PARAMS
# AMBER Ablation Run Parameters
# Generated: $(date)
# Job: ${SLURM_ARRAY_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}
# Config: ${CONFIG_NAME} (ID=${CONFIG_ID}), Rep: ${REP_ID}

[git]
branch=$GIT_BRANCH
commit=$GIT_COMMIT$GIT_DIRTY

[config]
name=$CONFIG_NAME
id=$CONFIG_ID
rep=$REP_ID
extra_flags=$EXTRA_FLAGS

[parameters]
encoder_seed=$ENCODER_SEED
leiden_seed=$LEIDEN_SEED
resolution=$RESOLUTION
bandwidth=$BANDWIDTH
partgraph_ratio=$PGR
threads=$THREADS
PARAMS

echo "=========================================="
echo "AMBER Ablation Run"
echo "=========================================="
echo "Config: $CONFIG_NAME (ID=$CONFIG_ID), Rep=$REP_ID"
echo "Branch: $GIT_BRANCH"
echo "Commit: $GIT_COMMIT$GIT_DIRTY"
echo "Extra flags: $EXTRA_FLAGS"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# First task to grab the build lock does the build; others wait.
# Uses mkdir as atomic lock (succeeds only once).
BUILD_LOCK="$AMBER_DIR/.build_lock_${SLURM_ARRAY_JOB_ID}"
BUILD_MUTEX="$AMBER_DIR/.build_mutex_${SLURM_ARRAY_JOB_ID}"
BINARY="$AMBER_DIR/build/amber_gpu"

if mkdir "$BUILD_MUTEX" 2>/dev/null; then
    echo ""
    echo "=== Building AMBER (task ${SLURM_ARRAY_TASK_ID} is builder) ==="
    mkdir -p build && cd build
    TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
    cmake .. -DAMBER_USE_TORCH=ON -DAMBER_USE_LIBLEIDENALG=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PATH" 2>&1 | tail -5
    cmake --build . --parallel 16
    cd ..
    touch "$BUILD_LOCK"
else
    echo ""
    echo "=== Waiting for build lock ==="
    WAIT=0
    until [[ -f "$BUILD_LOCK" ]] || [[ $WAIT -ge 1200 ]]; do
        sleep 15
        WAIT=$((WAIT + 15))
        echo "  ... waiting ($WAIT s)"
    done
    if [[ ! -f "$BINARY" ]]; then
        echo "Error: Binary not found after ${WAIT}s wait"
        exit 1
    fi
    echo "  Build ready."
fi

echo ""
echo "=== Running bin2 ==="
"$BINARY" bin2 \
    --contigs "$CONTIGS" \
    --bam "$BAM" \
    --output "$OUTPUT_DIR/bins" \
    --hmm auxiliary/checkm_markers_only.hmm \
    --encoder-seed $ENCODER_SEED \
    --random-seed $LEIDEN_SEED \
    --resolution $RESOLUTION \
    --bandwidth $BANDWIDTH \
    --partgraph-ratio $PGR \
    --threads $THREADS \
    --no-sweep \
    $EXTRA_FLAGS 2>&1 | tee "$OUTPUT_DIR/amber.log"

# Extract results
CLUSTERS=$(grep "Leiden found\|QualityLeiden\|Found.*clusters" "$OUTPUT_DIR/amber.log" 2>/dev/null | grep -oP '\d+(?= clusters)' | tail -1 || echo "?")
BINS=$(ls "$OUTPUT_DIR/bins"/*.fa 2>/dev/null | wc -l)

echo ""
echo "=== CheckM2 Validation ==="
.scripts/validate_bins_checkm2.sh "$OUTPUT_DIR/bins" "$OUTPUT_DIR/bins" $THREADS

# Extract HQ/MQ counts
if [[ -f "$OUTPUT_DIR/bins/checkm2/quality_report.tsv" ]]; then
    HQ=$(awk -F'\t' 'NR>1 && $2>=90 && $3<5' "$OUTPUT_DIR/bins/checkm2/quality_report.tsv" | wc -l)
    MQ=$(awk -F'\t' 'NR>1 && $2>=50 && $3<10 && !($2>=90 && $3<5)' "$OUTPUT_DIR/bins/checkm2/quality_report.tsv" | wc -l)
else
    HQ="?"
    MQ="?"
fi

# Save summary
cat > "$OUTPUT_DIR/summary.txt" << SUMMARY
config: $CONFIG_NAME
config_id: $CONFIG_ID
rep: $REP_ID
branch: $GIT_BRANCH
commit: $GIT_COMMIT$GIT_DIRTY
extra_flags: $EXTRA_FLAGS
clusters: $CLUSTERS
bins: $BINS
hq: $HQ
mq: $MQ
SUMMARY

echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo "Config: $CONFIG_NAME | Rep: $REP_ID"
echo "Clusters: $CLUSTERS | Bins: $BINS"
echo "HQ: $HQ | MQ: $MQ"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

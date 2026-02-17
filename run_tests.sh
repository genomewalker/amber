#!/bin/bash
#SBATCH --job-name=amber_test
#SBATCH --output=/maps/projects/caeg/people/kbd606/scratch/kapk-assm/amber/results/logs/%x_%A_%a.out
#SBATCH --partition=compregular
#SBATCH --nodelist=dandygpun01fl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=1-3

set -e
AMBER_DIR=/maps/projects/caeg/people/kbd606/scratch/kapk-assm/amber
cd "$AMBER_DIR"

source /maps/projects/fernandezguerra/apps/opt/conda/etc/profile.d/conda.sh
conda activate assembly

export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc

# Get git info
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(git diff --quiet 2>/dev/null && echo "" || echo "-dirty")

# Create unique output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${GIT_BRANCH}_${GIT_COMMIT}${GIT_DIRTY}"
OUTPUT_DIR="$AMBER_DIR/results/${RUN_NAME}/${SLURM_ARRAY_JOB_ID}_run${SLURM_ARRAY_TASK_ID}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Test data paths
CONTIGS="/projects/caeg/people/kbd606/scratch/kapk-assm/119_50/assembly/3e3ce8e9f7-derep-assm/final.contigs.fa"
BAM="/projects/caeg/people/kbd606/scratch/tmp/amber_cython_N1/map/mapped.reassigned.bam"

# Parameters (11 HQ config)
ENCODER_SEED=42
LEIDEN_SEED=1006
RESOLUTION=5.0
BANDWIDTH=0.2
PGR=50
THREADS=16

# Save parameters
cat > "$OUTPUT_DIR/params.txt" << PARAMS
# AMBER Test Run Parameters
# Generated: $(date)
# Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

[git]
branch=$GIT_BRANCH
commit=$GIT_COMMIT$GIT_DIRTY

[paths]
contigs=$CONTIGS
bam=$BAM
hmm=auxiliary/bacar_marker.hmm

[parameters]
encoder_seed=$ENCODER_SEED
leiden_seed=$LEIDEN_SEED
resolution=$RESOLUTION
bandwidth=$BANDWIDTH
partgraph_ratio=$PGR
threads=$THREADS

[slurm]
job_id=$SLURM_ARRAY_JOB_ID
array_task=$SLURM_ARRAY_TASK_ID
node=$SLURMD_NODENAME
PARAMS

echo "=========================================="
echo "AMBER Test Run"
echo "=========================================="
echo "Branch: $GIT_BRANCH"
echo "Commit: $GIT_COMMIT$GIT_DIRTY"
echo "Run: ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT}"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Build if needed
if [[ ! -f build/amber_gpu ]] || [[ $(find src -newer build/amber_gpu 2>/dev/null | head -1) ]]; then
    echo ""
    echo "=== Building AMBER ==="
    mkdir -p build && cd build
    TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
    cmake .. -DAMBER_USE_TORCH=ON -DAMBER_USE_LIBLEIDENALG=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PATH" 2>&1 | tail -10
    make -j16
    cd ..
fi

echo ""
echo "=== Running bin2 ==="
./build/amber_gpu bin2 \
    --contigs "$CONTIGS" \
    --bam "$BAM" \
    --output "$OUTPUT_DIR/bins" \
    --hmm auxiliary/bacar_marker.hmm \
    --encoder-seed $ENCODER_SEED \
    --random-seed $LEIDEN_SEED \
    --resolution $RESOLUTION \
    --bandwidth $BANDWIDTH \
    --partgraph-ratio $PGR \
    --threads $THREADS 2>&1 | tee "$OUTPUT_DIR/amber.log"

# Extract results
CLUSTERS=$(grep "Leiden found" "$OUTPUT_DIR/amber.log" 2>/dev/null | grep -oP '\d+(?= clusters)' || echo "?")
BINS=$(ls "$OUTPUT_DIR/bins"/*.fa 2>/dev/null | wc -l)

echo ""
echo "=== Clustering Results ==="
echo "Clusters: $CLUSTERS"
echo "Bins: $BINS"

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
# AMBER Test Summary
# $(date)

branch: $GIT_BRANCH
commit: $GIT_COMMIT$GIT_DIRTY
run: ${SLURM_ARRAY_TASK_ID}
clusters: $CLUSTERS
bins: $BINS
hq: $HQ
mq: $MQ
SUMMARY

echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo "Branch: $GIT_BRANCH"
echo "Commit: $GIT_COMMIT$GIT_DIRTY"
echo "Run: ${SLURM_ARRAY_TASK_ID}"
echo "Clusters: $CLUSTERS"
echo "Bins: $BINS"
echo "HQ: $HQ"
echo "MQ: $MQ"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

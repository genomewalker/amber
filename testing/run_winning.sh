#!/bin/bash
#SBATCH --job-name=amber_win
#SBATCH --output=/maps/projects/caeg/people/kbd606/scratch/kapk-assm/amber/results/logs/%x_%A_%a.out
#SBATCH --partition=compregular
#SBATCH --nodelist=dandygpun01fl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --array=1-3

# Winning configuration: DamageInfoNCE + SCG hard negatives + Consensus kNN (3 encoder restarts) + Leiden seed sweep (25)
# Attacks both root causes:
#   - Completeness variance: 3 encoder restarts → averaged kNN → stable neighborhoods
#   - Contamination: SCG hard negatives push cross-genome pairs apart; Phase 4 decontamination

set -e
AMBER_DIR=/maps/projects/caeg/people/kbd606/scratch/kapk-assm/amber
cd "$AMBER_DIR"

source /maps/projects/fernandezguerra/apps/opt/conda/etc/profile.d/conda.sh
conda activate assembly

export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc

REP_ID=${SLURM_ARRAY_TASK_ID}

GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$AMBER_DIR/results/winning/${SLURM_ARRAY_JOB_ID}_rep${REP_ID}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

CONTIGS="/projects/caeg/people/kbd606/scratch/kapk-assm/119_50/assembly/3e3ce8e9f7-derep-assm/final.contigs.fa"
BAM="/projects/caeg/people/kbd606/scratch/tmp/amber_cython_N1/map/mapped.reassigned.bam"

BUILD_MUTEX="$AMBER_DIR/.build_mutex_${SLURM_ARRAY_JOB_ID}"
BUILD_LOCK="$AMBER_DIR/.build_lock_${SLURM_ARRAY_JOB_ID}"
BINARY="$AMBER_DIR/build/amber_gpu"

if mkdir "$BUILD_MUTEX" 2>/dev/null; then
    echo "=== Building AMBER (task ${SLURM_ARRAY_TASK_ID} is builder) ==="
    mkdir -p build && cd build
    TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
    cmake .. -DAMBER_USE_TORCH=ON -DAMBER_USE_LIBLEIDENALG=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PATH" 2>&1 | tail -5
    cmake --build . --parallel 16
    cd ..
    touch "$BUILD_LOCK"
else
    echo "=== Waiting for build ==="
    WAIT=0
    until [[ -f "$BUILD_LOCK" ]] || [[ $WAIT -ge 1200 ]]; do
        sleep 15; WAIT=$((WAIT + 15)); echo "  ... waiting ($WAIT s)"
    done
    [[ -f "$BINARY" ]] || { echo "Build failed"; exit 1; }
fi

echo "=========================================="
echo "AMBER Winning Config — Rep $REP_ID"
echo "Commit: $GIT_COMMIT"
echo "Flags: defaults (damage-infonce + scg-infonce on) --encoder-restarts 3 --leiden-restarts 25"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

"$BINARY" bin2 \
    --contigs "$CONTIGS" \
    --bam "$BAM" \
    --output "$OUTPUT_DIR/bins" \
    --hmm auxiliary/checkm_markers_only.hmm \
    --encoder-seed 42 \
    --random-seed 1006 \
    --resolution 5.0 \
    --bandwidth 0.2 \
    --partgraph-ratio 50 \
    --threads 16 \
    --encoder-restarts 3 \
    --leiden-restarts 25 \
    2>&1 | tee "$OUTPUT_DIR/amber.log"

BINS=$(ls "$OUTPUT_DIR/bins"/*.fa 2>/dev/null | wc -l)

echo "=== CheckM2 Validation ==="
.scripts/validate_bins_checkm2.sh "$OUTPUT_DIR/bins" "$OUTPUT_DIR/bins" 16

if [[ -f "$OUTPUT_DIR/bins/checkm2/quality_report.tsv" ]]; then
    HQ=$(awk -F'\t' 'NR>1 && $2>=90 && $3<5' "$OUTPUT_DIR/bins/checkm2/quality_report.tsv" | wc -l)
    MQ=$(awk -F'\t' 'NR>1 && $2>=50 && $3<10 && !($2>=90 && $3<5)' "$OUTPUT_DIR/bins/checkm2/quality_report.tsv" | wc -l)
    echo "Contaminated bins (>5%):"
    awk -F'\t' 'NR>1 && $3>5 {printf "  %s: %.1f%% comp, %.1f%% cont\n", $1, $2, $3}' \
        "$OUTPUT_DIR/bins/checkm2/quality_report.tsv"
else
    HQ="?"; MQ="?"
fi

echo ""
echo "=========================================="
echo "FINAL: Rep $REP_ID | Bins: $BINS | HQ: $HQ | MQ: $MQ"
echo "Commit: $GIT_COMMIT"
echo "=========================================="

# AMBER Project Instructions

AMBER (Ancient Metagenomic Binning with aDNA-aware Encoder and Refinement) is a metagenomic binner optimized for ancient DNA samples.

## Best Configuration (11 HQ bins)

| Parameter | Value |
|-----------|-------|
| Encoder seed | 42 |
| Leiden seed | 1006 |
| Resolution | 5.0 |
| Bandwidth | 0.2 |
| Partgraph ratio | 50 |
| Features | 148 dims (128 encoder + 20 aDNA) |

## Build

```bash
source /maps/projects/fernandezguerra/apps/opt/conda/etc/profile.d/conda.sh
conda activate assembly
mkdir build && cd build
cmake .. -DAMBER_USE_TORCH=ON -DAMBER_USE_LIBLEIDENALG=ON -DCMAKE_BUILD_TYPE=Release
make -j16
```

For GPU builds on SLURM:
```bash
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
```

## Key Paths

- **Test contigs**: `/projects/caeg/people/kbd606/scratch/kapk-assm/119_50/assembly/3e3ce8e9f7-derep-assm/final.contigs.fa`
- **Test BAM**: `/projects/caeg/people/kbd606/scratch/tmp/amber_cython_N1/map/mapped.reassigned.bam`

## Run

```bash
./build/amber_gpu bin2 \
    --contigs $CONTIGS \
    --bam $BAM \
    --hmm auxiliary/checkm_markers_only.hmm \
    --encoder-seed 42 \
    --random-seed 1006 \
    --resolution 5.0 \
    --bandwidth 0.2 \
    --partgraph-ratio 50 \
    --threads 16 \
    --output output_dir
```

## Validation

Use CheckM2 to validate bin quality:
```bash
conda activate checkm2
checkm2 predict -i output_dir -o output_dir/checkm2 -x fa --threads 16
awk -F'\t' 'NR>1 && $2>=90 && $3<5' output_dir/checkm2/quality_report.tsv | wc -l  # HQ count
```

## aDNA Features (20 dimensions)

- Damage profile (10): C→T rates at 5' positions 1-5, G→A rates at 3' positions 1-5
- Decay parameters (2): lambda5, lambda3
- Fragment length (2): mean, std
- Damage coverage (2): ratio, n_eff
- Mismatch spectrum (4): other mismatch types

## Notes

- 11 HQ is the reproducible best result
- GPU training is stochastic; results vary 7-11 HQ across runs
- Original 12 HQ result was a lucky outlier

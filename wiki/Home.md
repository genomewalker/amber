# AMBER — Ancient Metagenomic BinnER

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AMBER bins metagenomic contigs from **ancient DNA (aDNA)** samples using damage-aware contrastive learning and quality-guided Leiden clustering. It is optimised for paleogenomic assemblies where post-mortem DNA damage, short fragment lengths, and mixed ancient/modern populations make standard binners unreliable.

---

## The problem

Post-mortem deamination converts cytosines to uracil (read as T) at fragment termini. In standard binners this:

- **Shifts apparent tetranucleotide composition** of each contig → damaged and undamaged reads from the same genome look like different organisms
- **Biases coverage profiles** → ancient reads (50–150 bp) cover reference positions differently from modern reads (150–300 bp)
- **Mixes populations** → assemblies contain both ancient (damaged) and modern (undamaged) strains; binners conflate or fragment them

---

## How AMBER solves this

| Problem | AMBER solution |
|---------|---------------|
| Composition distortion | Damage-aware InfoNCE: downweights negatives with incompatible damage signatures |
| Coverage bias | 20-dimensional aDNA feature vector captures damage, fragment length, decay |
| Mixed populations | `amber deconvolve`: EM separation of ancient/modern reads before or after binning |
| Bin fragmentation | Quality-guided Leiden: SCG-penalised edges + contamination splitting + near-HQ rescue |
| Run-to-run stochasticity | `amber resolve`: co-binning consensus from multiple independent runs |

---

## Wiki pages

- [[Quick Start Tutorial]] — full walkthrough from BAM to HQ bins
- [[Methods and Model]] — damage model math, InfoNCE derivation, Leiden quality refinement, EM deconvolution
- [[aDNA Features]] — all 157 embedding dimensions explained
- [[Command Reference]] — every flag for every subcommand
- [[Output Formats]] — file schemas for bins, abin, damage TSV, deconvolution outputs
- [[Benchmarks]] — HQ bin counts vs baselines on KapK sediment data

---

## Quick links

```bash
# Build (no GPU)
cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . --parallel

# Bin (GPU/CPU)
amber bin --contigs contigs.fa --bam alignments.bam --hmm auxiliary/bacar_marker.hmm \
          --output bins/ --threads 16

# Aggregate runs
amber resolve --runs run1/run.abin run2/run.abin run3/run.abin --output consensus/

# Deconvolve ancient/modern
amber deconvolve --contigs ref.fa --bam alignments.bam --output deconvolve/
```

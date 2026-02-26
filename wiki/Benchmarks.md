# Benchmarks

All benchmarks use the **KapK ancient sediment dataset**: a Holocene lake sediment metagenome containing reads from ancient (damaged, ~86 bp) and modern (undamaged, ~177 bp) DNA populations. Assembly was performed with MEGAHIT from 119 time-point samples. Bin quality was assessed with CheckM2 [Chklovski et al. 2023].

**MIMAG thresholds:** HQ = completeness ≥ 90%, contamination < 5%; MQ = completeness ≥ 50%, contamination < 10%.

---

## HQ bin counts — comparison with other tools

Same assembly, same BAM, same CheckM2 database.

| Tool | Rep 1 | Rep 2 | Rep 3 | Mean HQ | Mean MQ |
|------|-------|-------|-------|---------|---------|
| SemiBin2 (self-supervised) | 6 | 6 | 5 | **5.7** | 17 |
| COMEBin | 9 | 8 | 8 | **8.3** | 17 |
| **AMBER (damage_infonce + 25 seeds)** | **11** | **11** | **11** | **11.0** | 19 |

*Reps = 3 independent runs with different random seeds.*

The consistent 11 HQ across 3 replicates demonstrates that SCG-supervised contrastive learning and the quality-guided Leiden refinement together stabilise results that would otherwise vary between 7–11 HQ.

---

## Ablation study

Effect of each AMBER component on mean HQ bins (3 replicates each).

| Configuration | Mean HQ | Notes |
|---------------|---------|-------|
| Baseline (COMEBin InfoNCE) | 9.0 | No aDNA features |
| + damage_infonce weighting | 10.0 | +1.0 HQ from damage-aware negatives |
| + CGR features | 9.7 | CGR adds modest benefit at baseline |
| + quality_leiden (Phase 2) | 8.3 | Phase 2 alone over-fragments |
| + all phases (1+2+3) | 9.3–10.0 | Phases balanced together |
| + encoder_restarts=3 | 9.3 | Consensus kNN alone |
| **+ leiden_restarts=25** | **11.0** | Resolution sweep critical |
| VICReg (no negatives) | 2–3 | InfoNCE negatives essential |

**Key findings:**
- **damage_infonce** is the most reliable single improvement (+1.0 HQ)
- **leiden_restarts=25** with resolution sweep is essential for stability
- VICReg failed catastrophically because it has no negative pairs — covariance decorrelation alone cannot repel different genomes in embedding space

---

## Resolve vs individual runs

Effect of `amber resolve` aggregating 3 independent runs.

| Source | HQ |
|--------|----|
| Individual run (rep 1) | 9 |
| Individual run (rep 2) | 10 |
| Individual run (rep 3) | 10 |
| `amber resolve` (3 runs) | 10 |

With 3 runs, resolve recovers the reliable 10-bin core (bins appearing in ≥ 2/3 runs). Resolve is expected to add more bins when using ≥ 5 runs, where "borderline" bins appear in enough runs to pass the co-binning filter.

---

## DAS Tool post-processing

Applying DAS Tool [Sieber et al. 2018] to the AMBER winning bins (dereplication and aggregation):

| Input | HQ |
|-------|----|
| AMBER winning (rep 1) | 11 |
| DAS Tool on winning bins | 11 |

DAS Tool does not improve on the winning AMBER result — the bins are already well-separated.

---

## Deconvolution validation (S17 incubation sample)

The S17 sample contains a known mixture of ancient and modern DNA from a controlled incubation experiment. `amber deconvolve` results on the ancient_consensus:

| Source | Completeness | Contamination | GUNC CSS | Classification |
|--------|-------------|---------------|----------|----------------|
| S17 all reads (no filter) | 90.8% | 5.0% | 0.42 (pass) | Modern + ancient mixed |
| S17 modern reads (de novo assembly) | 26.8% | 2.3% | — | Modern only |
| S17 ancient_consensus (deconvolve) | 90.8% | 5.0% | 0.42 (pass) | Ancient population |

**De novo assembly from EM-classified modern reads (177 bp) gives only 26.8% completeness** — confirming that reference-guided deconvolution is the correct approach. Short-read de novo assembly cannot reconstruct the 1.9 Mbp genome from 177 bp fragments at this coverage depth.

---

## Dataset details

| Parameter | Value |
|-----------|-------|
| Site | KapK lake, Greenland |
| Sample type | Holocene lake sediment |
| Sequencing | Illumina paired-end, 150 bp |
| Assembly | MEGAHIT, 119 time-point co-assembly |
| Total assembly | ~2.8 Gbp |
| Contigs ≥ 2,500 bp | ~280,000 |
| Reference: CheckM2 | v1.0.2, DIAMOND database v3 |

---

## References

- Chklovski A et al. (2023) CheckM2: a rapid, scalable, and accurate tool for assessing microbial genome quality using machine learning. *Nature Methods* 20:1203–1212.
- Sieber CMK et al. (2018) Recovery of genomes from metagenomes via a dereplication, aggregation and scoring strategy. *Nature Microbiology* 3:836–843.

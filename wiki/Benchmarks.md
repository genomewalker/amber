# Benchmarks

All benchmarks use the **KapK ancient sediment dataset**: a Holocene lake sediment metagenome from the Kap København Formation, Greenland, containing reads from ancient (damaged, ~86 bp modal length) and modern (undamaged, ~177 bp) DNA populations. Assembly was performed with MEGAHIT from 119 time-point samples co-assembled (~2.8 Gbp total, ~280,000 contigs ≥ 2,500 bp). Bin quality assessed with CheckM2 v1.0.2 [1].

**MIMAG thresholds [2]:** HQ = completeness ≥ 90%, contamination < 5%; MQ = completeness ≥ 50%, contamination < 10%.

---

## Tool comparison

All tools ran on the same assembly and BAM. AMBER result is the `amber resolve` consensus from 3 independent runs (3 encoder restarts × 25 Leiden seeds each). COMEBin and SemiBin2 results are across independent replicate runs; mean and range shown.

| Tool | HQ bins | MQ bins | Reps | Notes |
|------|---------|---------|------|-------|
| SemiBin2 [3] | 5.7 (range 5–6) | 17 | 3 | Self-supervised contrastive, no aDNA features |
| COMEBin [4] | 7.5 (range 6–9) | 16–17 | 8 | Standard self-supervised InfoNCE, no aDNA features |
| **AMBER (this work)** | **11** | **20** | 3 → resolve | Damage-aware InfoNCE + quality-guided Leiden + co-binning consensus |

AMBER recovers 2–6 additional HQ bins compared with competing methods. SemiBin2 is limited to 5–6 HQ and is stable across runs; COMEBin varies between 6 and 9 HQ across 8 runs due to Leiden stochasticity. AMBER is fully stable (11/11/11).

---

## AMBER — per-bin quality (resolve consensus)

11 HQ bins + 9 near-HQ / MQ bins shown. Genome sizes in Mbp.

| Bin | Completeness | Contamination | Size (Mbp) | Tier |
|-----|-------------|---------------|------------|------|
| bin_28 | 100.0% | 4.45% | 3.63 | **HQ** |
| bin_77 | 99.9% | 0.69% | 0.84 | **HQ** |
| bin_53 | 99.9% | 1.86% | 3.30 | **HQ** |
| bin_42 | 98.2% | 0.05% | 1.12 | **HQ** |
| bin_24 | 97.7% | 0.23% | 3.19 | **HQ** |
| bin_26 | 96.5% | 2.09% | 0.77 | **HQ** |
| bin_38 | 95.0% | 2.65% | 3.34 | **HQ** |
| bin_35 | 94.6% | 0.90% | 2.45 | **HQ** |
| bin_4 | 92.3% | 1.90% | 1.91 | **HQ** |
| bin_8 | 90.6% | 2.29% | 2.76 | **HQ** |
| bin_47 | 90.1% | 2.62% | 3.58 | **HQ** |
| bin_10 | 84.9% | 4.97% | 3.27 | near-HQ |
| bin_18 | 82.7% | 1.18% | 2.32 | near-HQ |
| bin_15 | 80.2% | 2.71% | 2.58 | near-HQ |
| bin_32 | 80.1% | 4.36% | 0.76 | near-HQ |
| bin_19 | 79.7% | 3.62% | 2.20 | near-HQ |
| bin_16 | 73.6% | 2.84% | 2.61 | MQ |
| bin_3 | 73.0% | 3.06% | 1.03 | MQ |
| bin_6 | 55.9% | 1.96% | 2.09 | MQ |
| bin_9 | 38.5% | 0.64% | 2.20 | partial |

*bin_31 (84.6%, 7.75% contamination) excluded from near-HQ — contamination above 5%.*

---

## SemiBin2 — per-bin quality (best replicate, 6 HQ)

Best of 3 SemiBin2 self-supervised runs. All 6 HQ bins shown; SemiBin2 produces 17 MQ bins per run.

| Bin | Completeness | Contamination | Size (Mbp) | Tier |
|-----|-------------|---------------|------------|------|
| SemiBin_13 | 100.0% | 4.36% | 3.69 | **HQ** |
| SemiBin_146 | 99.9% | 0.64% | 0.83 | **HQ** |
| SemiBin_37 | 99.1% | 2.07% | 3.44 | **HQ** |
| SemiBin_11 | 99.1% | 0.37% | 3.35 | **HQ** |
| SemiBin_21 | 97.7% | 0.04% | 1.10 | **HQ** |
| SemiBin_12 | 97.5% | 2.15% | 0.76 | **HQ** |

---

## COMEBin — per-bin quality (best replicate, 9 HQ)

Best of 8 COMEBin runs. All 9 HQ bins shown; COMEBin produces 16–17 MQ bins per run.

| Bin | Completeness | Contamination | Size (Mbp) | Tier |
|-----|-------------|---------------|------------|------|
| 27966 | 100.0% | 2.15% | 3.54 | **HQ** |
| 25795 | 100.0% | 4.31% | 3.56 | **HQ** |
| 28106 | 99.8% | 0.66% | 0.96 | **HQ** |
| 25284 | 99.2% | 0.41% | 3.33 | **HQ** |
| 27724 | 97.5% | 0.05% | 1.14 | **HQ** |
| 27394 | 93.3% | 1.67% | 3.53 | **HQ** |
| 26333 | 93.2% | 2.10% | 1.65 | **HQ** |
| 25942 | 91.9% | 0.94% | 2.72 | **HQ** |
| 23134 | 91.3% | 2.04% | 2.41 | **HQ** |

*Two COMEBin bins with completeness > 99% had contamination > 50% (likely chimeric) and are excluded.*

---

## Comparison highlights

**Top-quality bins (≥ 97% completeness, < 5% contamination):** All three methods recover 5 such bins. Genome sizes are consistent (~0.8–3.7 Mbp), suggesting a shared core of high-completeness genomes that any reasonable binner recovers. The differences emerge below 97% completeness.

**Bins unique to AMBER:** bin_8 (90.6% / 2.29%, 2.76 Mbp) and bin_47 (90.1% / 2.62%, 3.58 Mbp) cross the HQ threshold in AMBER but are not recovered as HQ by either COMEBin or SemiBin2. These are likely genomes where aDNA damage features provide signal to separate them from neighbouring bins.

**Contamination control:** AMBER's median contamination across 11 HQ bins is 2.09%; COMEBin best rep median is 1.94%; SemiBin2 best rep median is 1.36% — all comparable. Two COMEBin bins had > 50% contamination (excluded above); AMBER's worst HQ bin is bin_28 at 4.45%.

**Reproducibility:** AMBER produces exactly 11 HQ bins in every replicate run. SemiBin2 is stable (5–6 HQ) but recovers fewer genomes. COMEBin varies between 6 and 9 HQ across 8 independent runs on identical data, due to Leiden stochasticity without a resolution sweep.

---

## Deconvolution validation (S17 incubation sample)

The S17 sample contains a known mixture of ancient and modern DNA from a controlled incubation experiment. `amber deconvolve` results on bin_4 (1.9 Mbp, *Candidatus* sp.):

| Source | Completeness | Contamination | Classification |
|--------|-------------|---------------|----------------|
| All reads (no deconvolution) | 90.8% | 5.0% | Ancient + modern mixed |
| `amber deconvolve` ancient consensus | 90.8% | 5.0% | Ancient population, 4,080 damage corrections |
| De novo assembly from modern reads only | 26.8% | 2.3% | Modern only — insufficient depth |

De novo assembly from EM-classified modern reads (177 bp) achieves only 26.8% completeness — confirming that reference-guided deconvolution is required at this coverage. The `amber deconvolve` ancient consensus is fully polished (4,080 C→T and G→A corrections).

---

## Dataset details

| Parameter | Value |
|-----------|-------|
| Site | Kap København Formation, North Greenland |
| Sample type | Holocene lake sediment |
| Sequencing | Illumina paired-end, 150 bp |
| Assembly | MEGAHIT, 119 time-point co-assembly |
| Total assembly | ~2.8 Gbp |
| Contigs ≥ 2,500 bp | ~280,000 |
| CheckM2 version | v1.0.2 |
| CheckM2 database | DIAMOND v3 |

---

## References

1. Chklovski A et al. (2023) CheckM2: a rapid, scalable, and accurate tool for assessing microbial genome quality using machine learning. *Nature Methods* 20:1203–1212.
2. Bowers RM et al. (2017) Minimum information about a single amplified genome (MISAG) and a metagenome-assembled genome (MIMAG). *Nature Biotechnology* 35:725–731.
3. Pan S, Zhao X-M, Coelho LP (2023) SemiBin2: self-supervised contrastive learning leads to better MAGs for short- and long-read sequencing. *Bioinformatics* 39(Suppl 1):i21–i29.
4. Wang Z et al. (2024) COMEBin allows effective binning of metagenomic contigs using coverage multi-view encoder. *Nature Communications* 15:1119.

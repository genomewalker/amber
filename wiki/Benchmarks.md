# Benchmarks

All benchmarks use the **KapK ancient sediment dataset**: a Holocene lake sediment metagenome from the Kap København Formation, Greenland, containing reads from ancient (damaged, ~86 bp modal length) and modern (undamaged, ~177 bp) DNA populations. Assembly was performed with MEGAHIT from 119 time-point samples co-assembled (~2.8 Gbp total, ~280,000 contigs ≥ 2,500 bp). Bin quality assessed with CheckM2 v1.0.2 [1].

**MIMAG thresholds [2]:** HQ = completeness ≥ 90%, contamination < 5%; MQ = completeness ≥ 50%, contamination < 10%.

---

## Tool comparison

All tools ran on the same assembly and BAM. AMBER result is the `amber resolve` consensus from 3 independent runs (3 encoder restarts × 25 Leiden seeds each). COMEBin and SemiBin2 results are across independent replicate runs; mean and range shown.

| Tool | HQ bins | MQ bins | Reps | Notes |
|------|---------|---------|------|-------|
| SemiBin2 [3] | 5.7 (range 5–6) | 17 | 3 | Self-supervised contrastive, no aDNA features |
| COMEBin [4] | 8.3 (range 8–9) | 16–17 | 3 | Standard self-supervised InfoNCE, no aDNA features |
| **AMBER (this work)** | **11** | **20** | 3 → resolve | Damage-aware InfoNCE + quality-guided Leiden + co-binning consensus |

AMBER recovers 2–6 additional HQ bins compared with competing methods. SemiBin2 is limited to 5–6 HQ and is stable across runs; COMEBin varies between 8 and 9 HQ across 3 runs. AMBER is fully stable (11/11/11).

---

## AMBER — per-bin quality (resolve consensus)

All 11 HQ bins. Genome sizes in Mbp.

| Bin | Completeness | Contamination | Size (Mbp) |
|-----|-------------|---------------|------------|
| bin_28 | 100.0% | 4.45% | 3.63 |
| bin_77 | 99.9% | 0.69% | 0.84 |
| bin_53 | 99.9% | 1.86% | 3.30 |
| bin_42 | 98.2% | 0.05% | 1.12 |
| bin_24 | 97.7% | 0.23% | 3.19 |
| bin_26 | 96.5% | 2.09% | 0.77 |
| bin_38 | 95.0% | 2.65% | 3.34 |
| bin_35 | 94.6% | 0.90% | 2.45 |
| bin_4 | 92.3% | 1.90% | 1.91 |
| bin_8 | 90.6% | 2.29% | 2.76 |
| bin_47 | 90.1% | 2.62% | 3.58 |

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

## COMEBin — per-bin quality (best replicate, top 3 HQ)

Best of 3 COMEBin runs. Top 3 HQ bins shown; COMEBin recovers up to 9 HQ and 16–17 MQ bins per run.

| Bin | Completeness | Contamination | Size (Mbp) | Tier |
|-----|-------------|---------------|------------|------|
| 27966 | 100.0% | 2.15% | 3.54 | **HQ** |
| 25795 | 100.0% | 4.31% | 3.56 | **HQ** |
| 28106 | 99.8% | 0.66% | 0.96 | **HQ** |

*Two COMEBin bins with completeness > 99% had contamination > 50% (likely chimeric) and are excluded.*

---

## Comparison highlights

**Top-quality bins (≥ 97% completeness, < 5% contamination):** All three methods recover 5 such bins. Genome sizes are consistent (~0.8–3.7 Mbp), suggesting a shared core of high-completeness genomes that any reasonable binner recovers. The differences emerge below 97% completeness.

**Bins unique to AMBER:** bin_8 (90.6% / 2.29%, 2.76 Mbp) and bin_47 (90.1% / 2.62%, 3.58 Mbp) cross the HQ threshold in AMBER but are not recovered as HQ by either COMEBin or SemiBin2. These are likely genomes where aDNA damage features provide signal to separate them from neighbouring bins.

**Contamination control:** AMBER's median contamination across 11 HQ bins is 2.09%; COMEBin best rep median is 1.94%; SemiBin2 best rep median is 1.36% — all comparable. Two COMEBin bins had > 50% contamination (excluded above); AMBER's worst HQ bin is bin_28 at 4.45%.

**Reproducibility:** AMBER produces exactly 11 HQ bins in every replicate run. SemiBin2 is stable (5–6 HQ) but recovers fewer genomes. COMEBin varies between 8 and 9 HQ across 3 runs.

---

## References

1. Chklovski A et al. (2023) CheckM2: a rapid, scalable, and accurate tool for assessing microbial genome quality using machine learning. *Nature Methods* 20:1203–1212.
2. Bowers RM et al. (2017) Minimum information about a single amplified genome (MISAG) and a metagenome-assembled genome (MIMAG). *Nature Biotechnology* 35:725–731.
3. Pan S, Zhao X-M, Coelho LP (2023) SemiBin2: self-supervised contrastive learning leads to better MAGs for short- and long-read sequencing. *Bioinformatics* 39(Suppl 1):i21–i29.
4. Wang Z et al. (2024) COMEBin allows effective binning of metagenomic contigs using coverage multi-view encoder. *Nature Communications* 15:1119.

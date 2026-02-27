# Command Reference

## Global

```
amber <command> [options]
amber --version
amber --help
```

---

## amber bin

Bin contigs using damage-aware contrastive learning. **Requires LibTorch** (CPU or GPU build).

```
amber bin [options]
```

### Required

| Flag | Description |
|------|-------------|
| `--contigs FILE` | Input contigs FASTA |
| `--bam FILE` | BAM file (reads mapped to contigs, must be indexed) |
| `--output DIR` | Output directory |

### Training

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | 100 | Training epochs |
| `--batch-size N` | 2048 | Training batch size |
| `--lr F` | 1e-3 | Learning rate |
| `--temperature F` | 0.1 | InfoNCE temperature τ |
| `--encoder-seed N` | 42 | Random seed for encoder initialisation |
| `--encoder-restarts N` | 3 | Independent encoder restarts for consensus kNN |

### Clustering

| Flag | Default | Description |
|------|---------|-------------|
| `--random-seed N` | 1006 | Leiden random seed |
| `--resolution F` | 5.0 | Leiden resolution parameter |
| `--bandwidth F` | 0.2 | kNN edge kernel bandwidth |
| `--partgraph-ratio N` | 50 | Partition graph density ratio |
| `--leiden-restarts N` | 25 | Leiden seed restarts per resolution |
| `--knn N` | 100 | k nearest neighbours per contig |

### aDNA features

| Flag | Default | Description |
|------|---------|-------------|
| `--hmm FILE` | auto | HMM marker file (default: `checkm_markers_only.hmm`, auto-detected) |
| `--damage-positions N` | 15 | Terminal positions for damage estimation |
| `--damage-infonce` | on | Enable damage-aware InfoNCE weighting |
| `--min-length N` | 1001 | Minimum contig length (bp) |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--threads N` | 1 | CPU threads |
| `--verbose` | off | Verbose logging |

---

## amber resolve

Aggregate bins from multiple independent runs via co-binning consensus.

```
amber resolve --runs run1/run.abin run2/run.abin ... --output DIR [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--runs FILES` | — | Space-separated list of .abin files |
| `--output DIR` | — | Output directory |
| `--threads N` | 1 | CPU threads |
| `--resolution F` | auto | Override Leiden resolution (default: sweep) |
| `--leiden-restarts N` | 5 | Leiden seed restarts |

---

## amber deconvolve

Separate ancient and modern DNA populations via EM on damage signal and fragment length.

```
amber deconvolve --contigs FILE --bam FILE --output DIR [options]
```

### Required

| Flag | Description |
|------|-------------|
| `--contigs FILE` | Reference FASTA (ancient consensus or assembled contigs) |
| `--bam FILE` | BAM file (reads mapped to reference) |
| `--output DIR` | Output directory |

### EM parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--em-iterations N` | 10 | Maximum EM iterations |
| `--p-ancient-hard F` | 0.8 | Threshold: reads above this → ancient |
| `--p-modern-hard F` | 0.2 | Threshold: reads below this → modern |
| `--min-ancient-depth F` | 3.0 | Minimum weighted depth for ancient consensus |
| `--min-modern-depth F` | 2.0 | Minimum weighted depth for modern consensus |
| `--min-modern-fraction F` | 0.05 | Minimum modern fraction to output modern consensus |

### Output options

| Flag | Default | Description |
|------|---------|-------------|
| `--write-stats` | off | Write per-position deconvolution stats |
| `--write-modern-bam` | off | Write modern-classified reads to BAM |
| `--modern-bam-threshold F` | 0.2 | p_ancient threshold for modern BAM inclusion |
| `--threads N` | 1 | CPU threads |
| `--damage-positions N` | 15 | Terminal positions for damage likelihood |

### Outputs

- `ancient_consensus.fa` — consensus called from ancient-weighted reads
- `modern_consensus.fa` — consensus called from modern-weighted reads
- `deconv_stats.tsv` — per-contig EM statistics (π, n_ancient, n_modern, n_ambiguous)
- `deconv_uncertainty.tsv` — per-position posterior uncertainty (if `--write-stats`)
- `modern_reads.bam` — modern-classified reads (if `--write-modern-bam`)

---

## amber damage

Compute per-bin aDNA damage statistics from a BAM and a set of pre-assembled bins.

```
amber damage --bam FILE --bins DIR --output FILE [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--bam FILE` | — | Input BAM (must be indexed) |
| `--bins DIR` | — | Directory of bin FASTA files |
| `--output FILE` | `damage_per_bin.tsv` | Output TSV |
| `--threads N` | 1 | CPU threads |
| `--damage-positions N` | 15 | Terminal positions to analyse |
| `--min-insert N` | 0 | Skip paired reads with insert size < N (filters overlapping pairs) |

### Output columns

| Column | Description |
|--------|-------------|
| `bin` | Bin name |
| `n_contigs` | Number of contigs |
| `n_reads` | Total mapped reads |
| `ct_5p_1..5` | C→T rates at 5′ positions 1–5 |
| `ga_3p_1..5` | G→A rates at 3′ positions 1–5 |
| `lambda_5p` | Fitted exponential decay at 5′ |
| `lambda_3p` | Fitted exponential decay at 3′ |
| `p_ancient` | Fraction of reads classified as ancient |
| `frag_len_mean` | Mean fragment length |
| `frag_len_std` | Std dev fragment length |
| `damage_class` | `ancient` / `modern` / `mixed` |

---

## amber polish

Correct post-mortem damage in an assembled ancient FASTA. Fits a Bayesian damage model from the BAM and reverts T→C at 5′ terminal positions and A→G at 3′ terminal positions where the per-position credible interval confirms genuine damage rather than sequencing error. See [Assembly damage artifact](Methods-and-Model#assembly-damage-artifact) for background.

```
amber polish --contigs FILE --bam FILE --library-type ds|ss --output DIR [options]
```

### Required

| Flag | Description |
|------|-------------|
| `--contigs FILE` | Input FASTA to polish |
| `--bam FILE` | BAM file (reads mapped to contigs, must be indexed) |
| `--library-type STR` | `ds` (double-stranded) or `ss` (single-stranded) |
| `--output DIR` | Output directory |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--damage-positions N` | 15 | Terminal positions to examine for correction |
| `--min-mapq N` | 30 | Minimum mapping quality |
| `--min-baseq N` | 20 | Minimum base quality |
| `--min-length N` | 0 | Skip contigs shorter than N bp |
| `--threads N` | 1 | CPU threads |
| `--verbose` | off | Verbose logging |
| `--no-bayesian` | off | Use legacy log-LR mode instead of Bayesian caller |
| `--min-posterior F` | 0.7 | Minimum posterior probability for a correction call |
| `--max-second-posterior F` | 0.2 | Maximum posterior of the second-best base (ensures unambiguous calls) |
| `--write-calls` | off | Write per-position base call details |
| `--anvio` | off | Rename contigs for Anvi'o compatibility |
| `--prefix STR` | — | Contig name prefix (with `--anvio`) |

### Outputs

| File | Description |
|------|-------------|
| `polished.fa` | Damage-corrected FASTA |
| `damage_model.tsv` | Bayesian model: amplitude, λ, baseline, per-position rates with 95% credible intervals |
| `damage_profile.tsv` | Per-contig per-position raw C→T and G→A rates |
| `polish_stats.tsv` | Per-contig correction counts (C→T and G→A) |
| `polish_trace.log` | Full run log |

---

## amber seeds

Generate SCG marker seeds for kNN-guided binning initialisation.

```
amber seeds --contigs FILE --bam FILE --hmm FILE --output DIR [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--contigs FILE` | — | Input contigs FASTA |
| `--bam FILE` | — | Input BAM |
| `--hmm FILE` | — | HMM marker profile |
| `--output DIR` | — | Output directory |
| `--threads N` | 1 | CPU threads |

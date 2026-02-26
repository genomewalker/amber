# Output Formats

## `amber bin` outputs

### `bins/bin.*.fa`
One FASTA file per bin. Headers are the original contig names from the input FASTA.

```
>contig_001234
ACGTACGT...
>contig_005678
TTACGCGT...
```

### `bins/amber_summary.tsv`
Tab-separated summary of all bins.

| Column | Description |
|--------|-------------|
| `bin` | Bin name (e.g. `bin.001`) |
| `n_contigs` | Number of contigs |
| `total_length` | Total bin length (bp) |
| `n50` | N50 contig length |
| `n_markers` | Total SCG markers found |
| `n_unique_markers` | Unique SCG markers (≤ 1 copy each) |
| `n_duplicate_markers` | SCG markers present > 1 copy |
| `est_completeness` | Estimated completeness from SCG count (%) |
| `est_contamination` | Estimated contamination from SCG duplication (%) |
| `p_ancient_mean` | Mean p_ancient across all mapped reads |
| `ct_5p_1` | Mean C→T rate at 5′ position 1 |
| `ga_3p_1` | Mean G→A rate at 3′ position 1 |
| `frag_len_mean` | Mean fragment length (bp) |
| `damage_class` | `ancient` / `modern` / `mixed` |

### `bins/damage_per_bin.tsv`
Per-bin damage profile. Same columns as the `amber damage` output table below.

### `run.abin`

Binary archive produced by `amber bin` and consumed by `amber resolve`. It carries everything needed to reconstruct or aggregate a binning run without re-running the encoder.

**File layout:**

```
magic[8]       "AMBR\x01\x00\x00\n"
n_contigs[4]   uint32 — total contigs in this run
chunks...      [tag:4][size:4][data:size] repeated
END\0[4]       sentinel chunk (size = 0)
```

All multi-byte integers are **little-endian**. Unknown chunk tags are skipped by the reader, so the format is forward-compatible.

**Chunks:**

| Tag | Data layout | Contents |
|-----|-------------|----------|
| `NAME` | `offsets[n×uint32]` + `data_len[uint32]` + packed null-terminated strings | Contig name strings. Resolve using `packed_data + offsets[i]` |
| `LNGT` | `int32[n]` | Contig lengths in bp |
| `EMBD` | `dim[uint16]` + `float32[n×dim]` | L2-normalised encoder embeddings (128 dims) |
| `FEAT` | `dim[uint16]` + `float32[n×dim]` | Full 157-dim feature vector (128 encoder + 20 aDNA + 9 CGR) used for kNN |
| `DAMG` | `dim[uint16]` + `float32[n×dim]` | Raw damage profile (20 aDNA dims: C→T positions, G→A positions, λ, fragment length, coverage) |
| `BINL` | `int32[n]` | Bin label per contig; −1 = unbinned |
| `SCGM` | `total_markers[uint32]` + `n_hits[uint32]` + `hits[n_hits × (contig_idx:uint32 + marker_id:uint16)]` | Sparse SCG marker assignments. Each hit record maps one contig to one marker ID. `total_markers` is the size of the marker vocabulary (e.g. 107) |
| `END\0` | — | Sentinel; signals end of file |

**Why a custom binary format?** A single `.abin` file captures the full run state in one read-compatible object. `amber resolve` memory-maps or streams N files in parallel, builds the co-binning affinity matrix from `BINL` and `SCGM`, and uses `EMBD`/`FEAT` only if re-clustering is requested. JSON or TSV would require separate files for embeddings, markers, and labels, making resolve fragile to partial runs.

---

## `amber resolve` outputs

### `consensus_bins/bin.*.fa`
Consensus bins after co-binning aggregation. Same FASTA format as `amber bin`.

### `consensus_bins/resolve_summary.tsv`
Includes additional columns vs the `amber bin` summary:

| Extra column | Description |
|-------------|-------------|
| `n_runs_present` | Number of input runs in which at least one contig appears |
| `mean_p_cobin` | Mean co-binning probability across contig pairs in the bin |
| `jaccard_stability` | Mean Jaccard similarity of this bin across input runs |

---

## `amber deconvolve` outputs

### `ancient_consensus.fa` / `modern_consensus.fa`
Reference-guided consensus sequences called from ancient-weighted and modern-weighted reads respectively. Positions with weighted depth < `--min-ancient-depth` / `--min-modern-depth` are masked with N.

### `deconv_stats.tsv`
Per-contig EM statistics.

| Column | Description |
|--------|-------------|
| `contig` | Contig name |
| `n_reads_total` | Total mapped reads |
| `n_ancient` | Reads with p_ancient > 0.8 |
| `n_modern` | Reads with p_ancient < 0.2 |
| `n_ambiguous` | Reads in [0.2, 0.8] |
| `pi_ancient` | Estimated mixture fraction (ancient) |
| `ct_5p_1` | C→T rate at 5′ position 1 |
| `frag_len_ancient` | Mean fragment length of ancient reads |
| `frag_len_modern` | Mean fragment length of modern reads |

### `deconv_uncertainty.tsv` (with `--write-stats`)
Per-position posterior uncertainty.

| Column | Description |
|--------|-------------|
| `contig` | Contig name |
| `pos` | Reference position (0-based) |
| `depth_ancient` | Weighted depth from ancient reads |
| `depth_modern` | Weighted depth from modern reads |
| `posterior_var` | Variance of the p_ancient posterior at this position |
| `base_ancient` | Called base in ancient consensus |
| `base_modern` | Called base in modern consensus |
| `is_discordant` | 1 if ancient ≠ modern base call |

### `modern_reads.bam` (with `--write-modern-bam`)
BAM file containing all reads with p_ancient ≤ `--modern-bam-threshold`. Both mates of a read pair are included if either mate meets the threshold. Suitable for de novo assembly of the modern fraction.

---

## `amber damage` output

### `damage_per_bin.tsv`

| Column | Description |
|--------|-------------|
| `bin` | Bin name |
| `n_contigs` | Number of contigs |
| `n_reads` | Total mapped reads |
| `ct_5p_1` … `ct_5p_5` | C→T rates at 5′ positions 1–5 |
| `ga_3p_1` … `ga_3p_5` | G→A rates at 3′ positions 1–5 |
| `lambda_5p` | Fitted exponential decay constant at 5′ |
| `lambda_3p` | Fitted exponential decay constant at 3′ |
| `p_ancient` | Fraction of reads classified as ancient |
| `frag_len_mean` | Mean fragment length (bp) |
| `frag_len_std` | Std dev of fragment length (bp) |
| `damage_class` | `ancient` / `modern` / `mixed` |

---

## Validation with CheckM2

```bash
checkm2 predict -i bins/ -o bins/checkm2 -x fa --threads 16
```

Key output: `bins/checkm2/quality_report.tsv`

| Column | Description |
|--------|-------------|
| `Name` | Bin name |
| `Completeness` | Estimated completeness (%) |
| `Contamination` | Estimated contamination (%) |
| `Completeness_Model_Used` | Neural Network (General/Specific) |

**MIMAG thresholds:**
- **HQ (High Quality):** Completeness ≥ 90%, Contamination < 5%
- **MQ (Medium Quality):** Completeness ≥ 50%, Contamination < 10%
- **LQ (Low Quality):** below MQ

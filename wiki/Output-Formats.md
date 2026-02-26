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
Per-bin damage profile. Same columns as `amber damage` output — see [[Command Reference]].

### `run.abin`
Binary archive used by `amber resolve`. Contains:
- All contig names and their bin assignments
- Per-contig embeddings (157-dim float32)
- Per-contig SCG marker sets
- Per-contig damage profiles
- Run configuration (seeds, resolution, bandwidth)

**Format** (chunk-based):

| Chunk | Contents |
|-------|----------|
| `NAME` | Contig name strings |
| `LNGT` | Contig lengths (uint32) |
| `EMBD` | Embeddings (float32, 157 dims/contig) |
| `FEAT` | Feature vectors |
| `DAMG` | Damage profiles |
| `BINL` | Bin label assignments |
| `SCGM` | SCG marker bitsets |

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
See [[Command Reference]] for column descriptions.

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

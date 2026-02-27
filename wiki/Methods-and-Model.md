# Methods and Model

## Damage model

Post-mortem deamination hydrolytically removes the amine group from cytosine, converting it to uracil (sequenced as thymine). The reaction preferentially affects single-stranded overhangs at fragment termini, giving a characteristic gradient of C→T substitutions at 5′ ends and (in double-stranded libraries) complementary G→A substitutions at 3′ ends [Briggs et al. 2007]. The rate follows exponential decay from the terminus:

$$\delta(p) = d \cdot e^{-\lambda p}$$

where *d* is the damage amplitude at position 0 (≈ the C→T rate at the outermost base) and λ is the decay constant (typically 0.2–0.6 nats/position). AMBER estimates these parameters per contig by maximum likelihood from the observed C→T and G→A rates at each terminal position.

---

## Per-read ancient probability

For each mapped read *r*, AMBER computes a per-read posterior **p_ancient** using a log-likelihood ratio over terminal positions:

$$\text{LLR}(r) = \sum_{i \in \text{5′ terminal}} \log \frac{P(b_i \mid \text{ancient}, \delta(i))}{P(b_i \mid \text{modern})} + \sum_{j \in \text{3′ terminal}} \log \frac{P(b_j \mid \text{ancient}, \delta(j))}{P(b_j \mid \text{modern})}$$

At a cytosine position *i* with sequenced base *b*:

$$P(b = T \mid \text{ancient}, \delta(i)) = \delta(i) \cdot (1-\epsilon) + (1-\delta(i)) \cdot \frac{\epsilon}{3}$$
$$P(b = C \mid \text{ancient}, \delta(i)) = (1-\delta(i)) \cdot (1-\epsilon)$$

where ε is the base quality error probability. The modern likelihood assumes no damage (δ = 0).

The LLR is converted to a posterior by Bayesian update from sample-level prior π₀:

$$p_{\text{ancient}}(r) = \sigma\!\left(\text{logit}(\pi_0) + \text{LLR}(r)\right)$$

Reads with p_ancient > 0.8 are classified as **confident ancient**; < 0.2 as **confident modern**; 0.2–0.8 as **ambiguous**.

---

## aDNA feature vector (20 dimensions)

Beyond the 128-dimensional encoder output, each contig carries a 20-dimensional aDNA feature vector:

### Damage profile (10 dims)
Position-specific terminal damage rates estimated by pooling all reads mapped to the contig:

$$\hat{\delta}_{\text{5′},p} = \frac{\sum_r \mathbb{1}[b_{r,p} = T, \text{ref} = C]}{\sum_r \mathbb{1}[\text{ref} = C, \text{coverage at } p]}$$

Positions 1–5 from the 5′ end (C→T) and positions 1–5 from the 3′ end (G→A). Default: 0.5 (uninformative prior) for contigs with insufficient coverage.

Soft-clip offsets from the CIGAR string are applied before computing rates to exclude adapter bases.

### Decay parameters (2 dims)
λ₅ and λ₃ from fitting the per-position rates:

$$\hat{\lambda} = \arg\min_\lambda \sum_{p=1}^{P} \left(\hat{\delta}_p - d \cdot e^{-\lambda p}\right)^2$$

### Fragment length (2 dims)
Mean and standard deviation of aligned read lengths for reads mapped to the contig. Ancient reads are typically 50–100 bp; modern reads 150–300 bp.

### Damage-stratified coverage (2 dims)
Log-normalised coverage from **confident-ancient reads** (p_ancient > 0.6) and **confident-modern reads** (p_ancient < 0.4): `log1p(cov_ancient)/5` and `log1p(cov_modern)/5`. A truly ancient genome has high cov_ancient and low cov_modern; a modern genome the reverse. The ratio cov_ancient/(cov_ancient + cov_modern) is a useful summary but the two raw values are what enters the feature vector.

### Mismatch spectrum (4 dims)
Four terminal mismatch rates: T→C at 5′, other mismatches at 5′, C→T at 3′, other mismatches at 3′. These complement the primary C→T / G→A damage signal with broader coverage of substitution patterns at fragment termini.

---

## SCG-supervised damage-aware InfoNCE

### Positive pairs and SCG hard negatives

AMBER extends COMEBin's self-supervised InfoNCE with two modifications. Positive pairs are the **6 augmented views of the same contig** (3 coverage subsampling levels × 2 feature-noise intensities), identical to COMEBin. The two extensions concern the *denominator*:

**Modification 1 — SCG hard negatives.** Single-copy marker genes (SCGs) are typically present exactly once per genome. Two contigs both containing the same SCG are therefore likely from *different* genomes — high-confidence (though not guaranteed) negative pairs. AMBER amplifies these pairs in the denominator by a factor α (default 2.0), providing stronger repulsion for contigs that embeddings might otherwise conflate:

$$w_{ij}^{\text{SCG}} = \begin{cases} \alpha & \text{if } M(i) \cap M(j) \neq \emptyset \\ 1.0 & \text{otherwise} \end{cases}$$

SCG membership is determined by HMM profile search with HMMER3 against `checkm_markers_only.hmm` (206 CheckM universal bacterial and archaeal marker profiles).

### Six augmented views

Six views per contig are constructed:
- 3 coverage subsampling levels (33%, 66%, 100% of reads)
- 2 feature noise intensities (low and medium Gaussian noise on TNF and coverage)

All 6 views pass through the shared MLP encoder (138→512→256→128, BatchNorm+ReLU, L2-normalised).

### InfoNCE loss with combined weights

$$\mathcal{L} = -\frac{1}{BV} \sum_{i} \log \frac{\exp(\text{sim}(z_i, z_{i^+})/\tau)}{\sum_{k \neq i} w_{ik} \cdot \exp(\text{sim}(z_i, z_k)/\tau)}$$

where τ = 0.1 and *w_ik* = max(*w_ik*^SCG, *w_ik*^damage) — SCG hard negatives always win over damage down-weighting.

### Damage-aware negative weighting

Standard contrastive learning treats all negatives equally. For ancient DNA this is wrong: a contig pair where one member is ancient and the other is modern from the same environment should not be strongly repelled. AMBER downweights such negatives by *w_ij* ∈ [*w*_min, 1]:

$$w_{ij} = 1 - \lambda_{\text{att}} \cdot c_i c_j \cdot \left(1 - f_{\text{compat}}(i, j)\right)$$

*Confidence* — *c_i = n_{eff,i} / (n_{eff,i} + n_0)* where *n_eff* is the Kish effective read depth and *n_0* = 60 is the half-saturation constant. Contigs with few reads get low confidence and weights close to 1 (effectively standard negatives).

*Compatibility* — *f*_compat combines two signals:
1. *f_raw* = exp(−|δ_i − δ_j| / 0.2): similar terminal damage rates → compatible
2. *f_panc* = exp(−0.5 · z²) where z = |p_anc,i − p_anc,j| / √(var_i + var_j + τ²)

$$f_{\text{compat}} = \frac{f_{\text{raw}} + f_{\text{panc}}}{2}$$

When *w_ij* < 1, the gradient pushing *i* and *j* apart is reduced. The encoder cannot use damage state as a discriminative feature, so ancient and modern strains of the same genome stay together.

### Three encoder restarts

With `--encoder-restarts 3`, three independent encoders are trained with different random seeds. The consensus kNN graph averages their edge weights before Leiden clustering. Any single unlucky initialisation has less influence on the final partition.

---

## CGR features (9 dims)

Chaos Game Representation (CGR) plots 4-mer frequencies in a unit square. AMBER computes CGR at three k-mer resolutions (16×16, 32×32, 64×64) and extracts three metrics at each:

- **Entropy:** Shannon entropy of the normalised CGR image — captures global sequence complexity
- **Occupancy:** Fraction of non-zero cells — correlates with genomic diversity
- **Lacunarity:** Gap structure in the CGR image — sensitive to repetitive sequences and GC content

The slopes of these metrics across resolutions (16→32→64) capture how composition scales with k-mer length, giving a scale-invariant fingerprint of genomic composition.

---

## kNN graph and Leiden clustering

After encoding, AMBER builds a **kNN graph** using HNSW (Hierarchical Navigable Small World) approximate nearest neighbours [Malkov & Yashunin 2018]. For each contig *i*, the *k* most similar contigs by cosine distance in the 157-dim embedding space are connected with edge weight:

$$w_{ij} = \exp\!\left(-\frac{d_{ij}}{bw}\right)$$

where *d_ij* is the cosine distance and *bw* = 0.2 is the bandwidth. SCG-sharing contigs receive an additional penalty: *w'_ij = w_ij · exp(−3 · n_shared_markers)*. This stops single-copy marker genes from bridging contigs that belong to different genomes.

Leiden clustering [Traag et al. 2019] is run with **resolution sweep** over [0.5, 5.0] and 25 random seeds per resolution. The partition maximising the SCG quality score:

$$Q = 10^8 \cdot n_{\text{strict-HQ}} + 10^5 \cdot n_{\text{pre-HQ}} + 10^3 \cdot n_{\text{MQ}} + \text{mean completeness}$$

is selected, where strict-HQ = completeness ≥ 90% and contamination < 5%, pre-HQ ≥ 72%/5%, MQ ≥ 50%/10%.

**Completeness and contamination are estimated entirely in C++ from SCG counts** — no external CheckM2 call is made during clustering. See [Internal quality estimation](#internal-quality-estimation) below.

---

## Contamination splitting (Phase 2)

For each bin with *dup_excess* > 0 (more duplicate SCGs than expected):

1. Extract the subgraph of all contigs in the bin
2. Re-run Leiden at 3× the current resolution
3. If the split reduces total *dup_excess*, accept it

This iteratively peels off contaminating contigs without global re-clustering.

---

## Near-HQ rescue (Phase 3)

Bins at 75–90% estimated completeness recruit missing-marker contigs from kNN neighbours:

1. Identify SCGs absent from the target bin
2. Find all contigs carrying those markers via `MarkerIndex`
3. For each candidate contig: accept if none of its markers are already present in the target (strict no-duplication guarantee)
4. Reject candidates from large, viable bins (*n_contigs* > 5, *completeness* > 60%) unless they are themselves contaminated

---

## Partition consensus (encoder restarts)

With `--encoder-restarts N`, AMBER trains *N* independent encoders with different random seeds, builds *N* kNN graphs, and computes a **consensus kNN** before Leiden. The consensus edge weight between contigs *i* and *j* is:

$$w^{\text{cons}}_{ij} = \frac{1}{N} \sum_{r=1}^N w^{(r)}_{ij}$$

Any single unlucky initialisation has less influence on the final partition.

---

## `amber resolve` — co-binning consensus

Given *R* independent AMBER runs (each producing a `run.abin` archive), `amber resolve` builds a **co-binning affinity graph** and re-clusters it.

### Co-binning affinity

For each ordered pair of contigs (*i*, *j*), let:
- *E_ij* = number of runs in which *i* and *j* are assigned to the **same bin**
- *R_i* = number of runs in which contig *i* received **any** bin assignment

The per-pair affinity is:

$$p_{\text{cobin}}(i,j) = \frac{E_{ij}}{\min(R_i,\, R_j)}$$

The denominator *min(R_i, R_j)* rather than *R* matters: if a short contig is unbinned in some runs, dividing by *R* would suppress its affinity with its true bin-mates. Using min normalises by the runs in which the pair could actually have co-occurred.

### Consensus clustering

An affinity graph is built with edge weight *p_cobin(i,j)* for all pairs with *E_ij* > 0. Leiden is then run on this graph with an SCG-guided resolution sweep (same three-phase quality protocol as `amber bin`), and the partition maximising the SCG quality score is returned.

### Why resolve improves reproducibility

Each individual run is subject to stochastic variation in encoder training and Leiden initialisation. Borderline bins (those appearing in some but not all runs) will only cross the *p_cobin* threshold if they are internally consistent across enough runs. With 3 runs the reliable 10-bin core is recovered; with ≥ 5 runs, borderline bins accumulate enough co-binning evidence to appear in the consensus.

---

## EM deconvolution (`amber deconvolve`)

AMBER models the mapped BAM as a mixture of two read populations:

- **Ancient** (*a*): high terminal damage *δ*(p), short fragment length *l* ~ LogNormal(μ_a, σ_a)
- **Modern** (*m*): low damage, longer *l* ~ LogNormal(μ_m, σ_m)

**E-step:** For each read *r* compute soft assignment:

$$p_a(r) = \frac{\pi_a \cdot P_{\text{damage}}(r \mid a) \cdot P_{\text{length}}(l_r \mid a)}{\pi_a \cdot P(r \mid a) + \pi_m \cdot P(r \mid m)}$$

The damage likelihood marginalises over all terminal C/G positions using the current parameter estimates. The length likelihood uses LogNormal distributions fitted per iteration.

**M-step:** Update mixture fraction π_a and distribution parameters (d, λ, μ_a, σ_a, μ_m, σ_m) from weighted counts.

Convergence is declared when Δπ_a < 0.01 between iterations (typically 5–15 iterations).

**Consensus calling:** Reads with p_a > 0.8 contribute to the ancient consensus with weight p_a; reads with p_a < 0.2 contribute to the modern consensus. Positions with < *min_depth* weighted coverage are masked with N. A per-position uncertainty file records the posterior variance of the soft assignment.

**Damage correction of the ancient consensus (default: on):** After calling the ancient consensus, AMBER corrects assembly-baked damage before writing `ancient_consensus.fa`. See [Assembly damage artifact](#assembly-damage-artifact) below. Disable with `--no-polish`.

---

## Assembly damage artifact

### What happens during ancient assembly

When ancient reads are assembled by a standard assembler (MEGAHIT, metaSPAdes), the assembler builds each contig by majority vote across all reads covering each position. At positions near where many reads begin or end — particularly near contig termini — a large fraction of contributing reads show C→T substitutions at their 5′ ends from post-mortem deamination. If damage at a position is severe enough (>50% of reads show T), the assembler calls T instead of C and incorporates the damaged base into the consensus. This is *assembly-baked damage*: the assembled sequence has T at positions where the true genome has C.

### How it distorts downstream damage estimation

When reads are mapped back to an assembly containing baked-in T's and damage rates are computed:

- At baked-in-T positions: ancient reads (with T from damage) **match** the reference → their C→T mismatch is not counted. Modern reads (with true C) show a **T→C** mismatch — the anti-damage direction.
- The denominator of the C→T rate (reference-C positions) is **reduced**, because some C positions are now T in the reference.
- At positions where the assembler correctly called C, the damage signal is normal.

The net effect: apparent C→T damage rates are **suppressed** at the most heavily damaged positions (where the assembler baked in T), while T→C mismatches from undamaged reads appear at those same positions. Tools computing aggregate damage may report distorted profiles. The *true* damage in the sample is higher than what the assembled-reference analysis shows at those positions.

### How much suppression to expect

The severity scales with the fraction of ancient reads contributing to the assembly. At a terminal position with true C→T damage rate δ and ancient read fraction f:

- The assembler calls **T** (baking in damage) when f · δ > 0.5
- When baked in, the observed residual C→T rate at that position drops to **zero**
- Undamaged reads at that position create **T→C mismatches** at rate (1 − δ)

The table below illustrates the effect at 5′ position 1 (typically the most damaged position):

| Ancient fraction | True C→T at pos 1 | Observed C→T (residual) | T→C proxy (baked-in) | Notes |
|-----------------|-------------------|------------------------|----------------------|-------|
| 10% | 3% | 3% | ~0% | Assembler calls C; no suppression |
| 40% | 10% | 10% | ~0% | Still below baking threshold |
| 60% | 20% | 20% | ~0% | Marginal: 60% × 20% = 12%, below 50% |
| 80% | 30% | **~0%** | ~14% | 80% × 30% = 24% — baked in at this position |
| 95% | 40% | **~0%** | ~57% | Severe suppression; T→C dominates |

For the KapK dataset (~40% ancient reads, δ ≈ 3–5% at position 1), suppression is negligible — the assembler correctly calls C at most positions and the residual damage signal is reliable. The effect becomes significant in permafrost or high-enrichment ancient samples where f > 70%.

### Residual signal in the AMBER encoder features

AMBER's 20-dimensional aDNA feature vector includes both sides of the suppression:

- **`ct_rate_5p[0..4]`** — C→T mismatch rate at positions 1–5 (residual damage: reads show T where contig shows C). This is the direct ancient signal but is suppressed when damage is baked in.
- **`mm_5p_tc`** — T→C mismatch rate at 5′ end (reads show C where contig shows T). This is the complementary baked-in proxy: high when the assembler has incorporated C→T damage into the reference.

The encoder can learn to combine both: `ct_rate_5p + mm_5p_tc` approximates the total terminal mismatch rate, recovering the full damage signal regardless of whether it was baked in or not. Neither dimension alone is sufficient at high ancient fractions.

### The assembled sequence is wrong — not just the damage statistics

Assembly-baked damage is a sequence-level error. When the assembler calls T instead of C at a terminal position, that T is written into the FASTA file as the actual nucleotide. The genome sequence is incorrect at those positions. This matters for every downstream analysis:

- **Phylogenomics / SNP trees** — baked-in T's appear as derived mutations, creating false branches
- **Gene prediction / protein annotation** — coding sequences contain wrong codons near contig termini
- **Variant calling** — reads with true C are flagged as variants against the baked-in T reference
- **Re-mapping** — ancient reads mapping to the same region appear undamaged because they match the T

Running `amber deconvolve` (or `amber polish` for pre-existing assemblies) is therefore needed not only to characterise damage, but to obtain a genomically correct reference sequence.

### How AMBER addresses this

**`amber deconvolve` (default behaviour):** `deconvolve` corrects baked-in damage in three passes:

1. **Neutral consensus from interior positions** — before any classification, a reference is built from reads at positions far from read termini (outside the damage zone). Interior positions are damage-free regardless of whether reads are ancient or modern, so this neutral consensus is unbiased even in pure-ancient samples. All subsequent classification compares reads against this unbiased reference rather than the damaged assembled sequence.

2. **EM read classification and damage-aware consensus calling** — reads are soft-assigned to ancient/modern populations. The ancient consensus is built using the damage model at terminal positions (T from an ancient read at a 5′ C position is interpreted as C with probability proportional to the damage rate). The modern consensus falls back to the neutral consensus where modern read depth is insufficient.

3. **Damage harmonization** — the final pass compares ancient consensus vs neutral (modern) consensus at every position. Where ancient called T/A and neutral called C/G — the exact pattern expected from baked-in damage — the C/G base is adopted. This single step accounts for the bulk of corrections (4,080 out of 4,080 in the S17 experiment) because it directly identifies positions where the terminal damage pattern is inconsistent with what interior reads show.

**No modern reads:** `amber deconvolve` works correctly even when the BAM contains only ancient reads. The neutral consensus (pass 1) is built from interior positions of ancient reads, which are damage-free and show the true base. All reads are classified as ancient (p_ancient ≈ 1.0, correct), and the damage correction proceeds normally. Modern output is suppressed when the estimated modern fraction is below 5%. The corrected `ancient_consensus.fa` is produced in all cases.

In the S17 incubation experiment (bin_4, ~40% ancient, 1.9 Mbp), `amber deconvolve` made 4,080 corrections (2,002 C→T + 2,078 G→A). The correction counts appear in `deconvolution_stats.tsv` as `ancient_ct_corr` and `ancient_ga_corr`. Use `--no-polish` to skip correction and write the raw ancient consensus.

**`amber polish` (standalone):** A lighter-weight standalone polisher for pre-existing assemblies. It applies only the Bayesian Bayes-factor correction pass (no neutral consensus, no damage harmonization). On the same S17 bin_4 dataset, `amber polish` made 111 corrections vs 4,080 for `amber deconvolve` — roughly 37× fewer. Use `amber deconvolve` in preference when possible; `amber polish` is useful only when the original BAM is not available.

```bash
amber polish \
    --contigs assembly.fa \
    --bam mapped.bam \
    --library-type ds \
    --output polished/ \
    --threads 16
```

Outputs: `polished.fa`, `damage_model.tsv`, `polish_stats.tsv`.

### Practical guidance

| Situation | Recommended approach |
|-----------|----------------------|
| Any ancient assembly (with or without modern reads) | `amber deconvolve` — produces corrected `ancient_consensus.fa`; modern output suppressed if < 5% modern fraction |
| BAM not available, pre-existing assembly | `amber polish` — limited (37× fewer corrections), but better than nothing |
| Want to verify damage profile without correcting | `amber damage --bam reads.bam --bins bins/ --output damage.tsv` |

---

## Internal quality estimation

AMBER estimates bin completeness and contamination on-the-fly in C++ using the **206-marker CheckM universal single-copy gene set** (`checkm_markers_only.hmm`). No external tools (CheckM, CheckM2, BUSCO) are called during binning.

### Marker detection

HMM profiles are scanned against contig ORF translations using an embedded HMMER-compatible search. Each contig receives a list of detected marker IDs. AMBER builds a `MarkerIndex` (contig → marker set) and a reverse index (marker → contig list) for O(1) lookups during Leiden phases.

### Per-bin quality estimation

For each bin *B*, AMBER aggregates marker counts across all member contigs into a `ClusterMarkerProfile`:

$$\text{completeness}(B) = \frac{|\{m : \text{count}_B(m) \geq 1\}|}{206}$$

$$\text{contamination}(B) = \frac{\sum_m \max(0,\, \text{count}_B(m) - 1)}{\sum_m \text{count}_B(m)}$$

where the denominator counts total marker observations and the numerator counts excess copies above 1. This follows the CheckM formula and gives comparable results on standard benchmark datasets.

Quality thresholds used internally:

| Tier | Completeness | Contamination |
|------|-------------|---------------|
| Strict-HQ | ≥ 90% | < 5% |
| Pre-HQ | ≥ 72% | < 5% |
| MQ | ≥ 50% | < 10% |

These estimates guide all three Leiden phases (edge penalties, contamination splitting, near-HQ rescue) and the resolution sweep quality score. **CheckM2 is only used post-hoc for external validation**, not during the binning algorithm itself.

### Duplication excess

*dup_excess* = contamination × total_markers — the total number of excess marker copies in the bin. A bin with *dup_excess* > 0 is a candidate for Phase 2 splitting.

---

## References

- Briggs AW et al. (2007) Patterns of damage in genomic DNA sequences from a Neandertal. *PNAS* 104:14616–21.
- Malkov YA, Yashunin DA (2018) Efficient and robust approximate nearest neighbor search using HNSW. *IEEE TPAMI* 42:824–36.
- Oord A van den et al. (2018) Representation Learning with Contrastive Predictive Coding. *arXiv* 1807.03748.
- Traag VA et al. (2019) From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports* 9:5233.
- Wang Z et al. (2023) COMEBin allows effective binning of metagenomic contigs using coverage multi-view encoder. *Nature Communications* 15:1119.

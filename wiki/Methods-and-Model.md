# Methods and Model

## Damage model

Post-mortem deamination hydrolytically removes the amine group from cytosine, converting it to uracil (sequenced as thymine). This reaction preferentially affects single-stranded overhangs at fragment termini, producing a characteristic gradient of C→T substitutions at 5′ ends and (in double-stranded libraries) complementary G→A substitutions at 3′ ends [Briggs et al. 2007]. The rate follows exponential decay from the terminus:

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
Coverage from **confident-ancient reads** (p_ancient > 0.6) and **confident-modern reads** (p_ancient < 0.4), normalised by contig length. The ratio is a genome-specific signal: a truly ancient genome has high cov_ancient/cov_modern; a modern genome has low or zero.

### Mismatch spectrum (4 dims)
Terminal mismatch rates for: T→C at 5′ (reverse-complement of G→A damage on the complementary strand), A→G at 5′, C→T at 3′ (reverse complement), and all other mismatches. These complement the primary C→T / G→A signal.

---

## SCG-supervised damage-aware InfoNCE

### Positive pair definition

AMBER's central novelty in the encoder is replacing COMEBin's self-supervised positive pairs (random substrings of the same contig) with **SCG-supervised positive pairs** defined by single-copy marker gene co-membership.

For each contig *i* in the batch, the positive set is:

$$P(i) = \underbrace{\{v_1^{(i)}, \ldots, v_6^{(i)}\}}_{\text{6 augmented views}} \cup \underbrace{\{j : \text{SCG}(j) = \text{SCG}(i) \neq \emptyset\}}_{\text{SCG co-members}}$$

The SCG co-members are contigs identified by HMM profile search (HMMER, `bacar_marker.hmm`, 107 single-copy bacterial markers) as carrying the same single-copy marker gene. Since each marker should appear exactly once per genome, contigs sharing a marker belong to the same genome with high confidence — they are reliable positives without needing any assembly graph information.

Contigs sharing *any* marker with *i* are **excluded from the InfoNCE denominator** entirely (masked out). This avoids treating same-genome contigs that happen to share a different marker as false negatives.

### Six augmented views

Six views per contig are constructed:
- 3 coverage subsampling levels (33%, 66%, 100% of reads)
- 2 feature noise intensities (low and medium Gaussian noise on TNF and coverage)

All 6 views pass through the shared MLP encoder (138→512→256→128, BatchNorm+ReLU, L2-normalised) and all appear in the positive set for each other.

### SCG-SupCon loss

$$\mathcal{L}_{\text{SCG-SupCon}} = -\frac{1}{|B|} \sum_{i} \frac{1}{|P(i)|} \sum_{j \in P(i)} \log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \notin \text{excl}(i)} w_{ik} \cdot \exp(\text{sim}(z_i, z_k)/\tau)}$$

where excl(*i*) = {*k* : M(*i*) ∩ M(*k*) ≠ ∅} (any shared marker → excluded from denominator), and τ = 0.1 is the temperature.

### Damage-aware negative weighting

Standard contrastive learning treats all negatives equally. For ancient DNA, this is suboptimal: a negative pair where one contig is ancient and one is modern from the same environment should not be strongly repelled. AMBER downweights such negatives in the denominator by *w_ij* ∈ [*w*_min, 1]:

$$w_{ij} = 1 - \lambda_{\text{att}} \cdot c_i c_j \cdot \left(1 - f_{\text{compat}}(i, j)\right)$$

**Confidence:** *c_i = n_{eff,i} / (n_{eff,i} + n_0)* where *n_eff* is the Kish effective read depth and *n_0* = 60 is the half-saturation constant. Contigs with few reads get low confidence → weights close to 1 (standard negatives).

**Compatibility:** *f*_compat combines two signals:
1. *f_raw* = exp(−|δ_i − δ_j| / 0.2) — similar terminal damage rates → compatible
2. *f_panc* = exp(−0.5 · z²) where z = |p_anc,i − p_anc,j| / √(var_i + var_j + τ²)

$$f_{\text{compat}} = \frac{f_{\text{raw}} + f_{\text{panc}}}{2}$$

When *w_ij* < 1, the negative is downweighted in the SCG-SupCon denominator, reducing the gradient pushing *i* and *j* apart. This prevents the encoder from latching onto damage state as a discriminative feature — which would incorrectly separate ancient and modern strains of the same genome.

### Three encoder restarts

With `--encoder-restarts 3`, three independent encoders are trained with different random seeds. The consensus kNN graph averages their edge weights before Leiden clustering, reducing sensitivity to any single unlucky initialisation.

---

## CGR features (9 dims)

Chaos Game Representation (CGR) plots 4-mer frequencies in a unit square. AMBER computes CGR at three k-mer resolutions (16×16, 32×32, 64×64) and extracts three metrics at each:

- **Entropy:** Shannon entropy of the normalised CGR image — captures global sequence complexity
- **Occupancy:** Fraction of non-zero cells — correlates with genomic diversity
- **Lacunarity:** Gap structure in the CGR image — sensitive to repetitive sequences and GC content

The slopes of these metrics across resolutions (16→32→64) encode how composition scales with k-mer length, providing a scale-invariant fingerprint.

---

## kNN graph and Leiden clustering

After encoding, AMBER builds a **kNN graph** using HNSW (Hierarchical Navigable Small World) approximate nearest neighbours [Malkov & Yashunin 2018]. For each contig *i*, the *k* most similar contigs by cosine distance in the 157-dim embedding space are connected with edge weight:

$$w_{ij} = \exp\!\left(-\frac{d_{ij}}{bw}\right)$$

where *d_ij* is the cosine distance and *bw* = 0.2 is the bandwidth. SCG-sharing contigs receive an additional penalty: *w'_ij = w_ij · exp(−3 · n_shared_markers)*, preventing single-copy marker genes from bridging different genomes.

Leiden clustering [Traag et al. 2019] is run with **resolution sweep** over [0.5, 5.0] and 25 random seeds per resolution. The partition maximising the SCG quality score:

$$Q = 10^8 \cdot n_{\text{strict-HQ}} + 10^5 \cdot n_{\text{pre-HQ}} + 10^3 \cdot n_{\text{MQ}} + \text{mean completeness}$$

is selected, where strict-HQ = completeness ≥ 90% and contamination < 5%, pre-HQ ≥ 72%/5%, MQ ≥ 50%/10%.

**Completeness and contamination are estimated entirely in C++ from SCG counts** — no external CheckM2 call is made during clustering. See [[#internal-quality-estimation]] below.

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

This reduces the effect of any single unlucky encoder initialisation on the final clustering.

---

## `amber resolve` — co-binning consensus

Given *R* independent AMBER runs (each producing a `run.abin` archive):

1. For each pair of contigs (*i*, *j*), count in how many runs they appear in the same bin: *E_ij*
2. The fraction of runs in which they could co-occur: *D_ij = min(R_i, R_j)* where *R_i* is the number of runs in which contig *i* was assigned to any bin
3. Co-binning affinity: *p_cobin(i,j) = E_ij / D_ij*
4. Run Leiden on the affinity graph with an SCG-guided resolution sweep

The per-pair denominator ensures that contigs absent from some runs (too short, ambiguous) do not artificially suppress their affinity with their true bin-mates.

---

## EM deconvolution (`amber deconvolve`)

AMBER models the mapped BAM as a mixture of two read populations:

- **Ancient** (*a*): high terminal damage *δ*(p), short fragment length *l* ~ LogNormal(μ_a, σ_a)
- **Modern** (*m*): low damage, longer *l* ~ LogNormal(μ_m, σ_m)

**E-step:** For each read *r* compute soft assignment:

$$p_a(r) = \frac{\pi_a \cdot P_{\text{damage}}(r \mid a) \cdot P_{\text{length}}(l_r \mid a)}{\pi_a \cdot P(r \mid a) + \pi_m \cdot P(r \mid m)}$$

The damage likelihood marginalises over all terminal C/G positions using the current parameter estimates. The length likelihood is evaluated under LogNormal distributions fitted per iteration.

**M-step:** Update mixture fraction π_a and distribution parameters (d, λ, μ_a, σ_a, μ_m, σ_m) from weighted counts.

Convergence is declared when Δπ_a < 0.01 between iterations (typically 5–15 iterations).

**Consensus calling:** Reads with p_a > 0.8 contribute to the ancient consensus with weight p_a; reads with p_a < 0.2 contribute to the modern consensus. Positions with < *min_depth* weighted coverage are masked with N. A per-position uncertainty file records the posterior variance of the soft assignment.

---

## Internal quality estimation (C++ SCG engine) {#internal-quality-estimation}

AMBER estimates bin completeness and contamination on-the-fly in C++ using a **107-marker single-copy gene set** (`bacar_marker.hmm`, the same bacterial marker set used by COMEBin). No external tools (CheckM, CheckM2, BUSCO) are called during binning.

### Marker detection

HMM profiles are scanned against contig ORF translations using an embedded HMMER-compatible search. Each contig receives a list of detected marker IDs. AMBER builds a `MarkerIndex` (contig → marker set) and a reverse index (marker → contig list) for O(1) lookups during Leiden phases.

### Per-bin quality estimation

For each bin *B*, AMBER aggregates marker counts across all member contigs into a `ClusterMarkerProfile`:

$$\text{completeness}(B) = \frac{|\{m : \text{count}_B(m) \geq 1\}|}{107}$$

$$\text{contamination}(B) = \frac{\sum_m \max(0,\, \text{count}_B(m) - 1)}{\sum_m \text{count}_B(m)}$$

where the denominator counts total marker observations and the numerator counts the excess copies (duplicates) above 1. This mirrors the CheckM formula and gives comparable results on standard benchmark datasets.

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

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
Log-normalised coverage from **confident-ancient reads** (p_ancient > 0.6) and **confident-modern reads** (p_ancient < 0.4): `log1p(cov_ancient)/5` and `log1p(cov_modern)/5`. A truly ancient genome has high cov_ancient and low cov_modern; a modern genome the reverse.

### Mismatch spectrum (4 dims)
Four terminal mismatch rates: T→C at 5′, C→T at 3′, other mismatches at 5′, other mismatches at 3′. These complement the primary C→T / G→A damage signal with broader coverage of substitution patterns at fragment termini.

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

### Encoder restarts

With `--encoder-restarts N`, *N* independent encoders are trained with different random seeds. The consensus kNN graph averages their edge weights before Leiden clustering, reducing sensitivity to any single unlucky initialisation.

---

## CGR features (9 dims)

Chaos Game Representation (CGR) plots k-mer frequencies in a unit square. AMBER computes CGR at three resolutions (16×16, 32×32, 64×64) and extracts three metrics at each:

- **Entropy:** Shannon entropy of the normalised CGR image — captures global sequence complexity
- **Occupancy:** Fraction of non-zero cells — correlates with genomic diversity
- **Lacunarity:** Gap structure in the CGR image — sensitive to repetitive sequences and GC content

The slopes of these metrics across resolutions (16→32→64) capture how composition scales with k-mer length, giving a scale-invariant fingerprint of genomic composition.

---

## kNN graph and Leiden clustering

After encoding, AMBER builds a **kNN graph** using HNSW approximate nearest neighbours [Malkov & Yashunin 2018]. For each contig *i*, the *k* most similar contigs by cosine distance in the 157-dim embedding space are connected with edge weight:

$$w_{ij} = \exp\!\left(-\frac{d_{ij}}{bw}\right)$$

where *d_ij* is the cosine distance and *bw* = 0.2 is the bandwidth. SCG-sharing contigs receive an additional penalty: *w'_ij = w_ij · exp(−3 · n_shared_markers)*. This prevents single-copy marker genes from bridging contigs that belong to different genomes.

Leiden clustering [Traag et al. 2019] is run with **resolution sweep** over [0.5, 5.0] and 25 random seeds per resolution. The partition maximising the SCG quality score:

$$Q = 10^8 \cdot n_{\text{strict-HQ}} + 10^5 \cdot n_{\text{pre-HQ}} + 10^3 \cdot n_{\text{MQ}} + \text{mean completeness}$$

is selected, where strict-HQ = completeness ≥ 90% and contamination < 5%, pre-HQ ≥ 72%/5%, MQ ≥ 50%/10%.

**Completeness and contamination are estimated entirely in C++ from SCG counts** — no external CheckM2 call is made during clustering.

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

## `amber resolve` — co-binning consensus

Given *R* independent AMBER runs (each producing a `run.abin` archive), `amber resolve` builds a **co-binning affinity graph** and re-clusters it.

### Co-binning affinity

For each pair of contigs (*i*, *j*), let:
- *E_ij* = number of runs in which *i* and *j* are assigned to the **same bin**
- *R_i* = number of runs in which contig *i* received **any** bin assignment

The per-pair affinity is:

$$p_{\text{cobin}}(i,j) = \frac{E_{ij}}{\min(R_i,\, R_j)}$$

The denominator *min(R_i, R_j)* rather than *R* matters: if a short contig is unbinned in some runs, dividing by *R* would suppress its affinity with its true bin-mates.

### Consensus clustering

An affinity graph is built with edge weight *p_cobin(i,j)* for all pairs with *E_ij* > 0. Leiden is then run on this graph with the same SCG-guided quality protocol as `amber bin`, and the partition maximising the SCG quality score is returned.

With 3 runs the reliable bin core is recovered; with ≥ 5 runs, borderline bins accumulate enough co-binning evidence to appear in the consensus.

---

## EM deconvolution (`amber deconvolve`)

AMBER models the mapped BAM as a mixture of two read populations:

- **Ancient**: high terminal damage δ(p), short fragment length *l* ~ LogNormal(μ_a, σ_a)
- **Modern**: low damage, longer *l* ~ LogNormal(μ_m, σ_m)

**E-step:** For each read *r* compute soft assignment:

$$p_a(r) = \frac{\pi_a \cdot P_{\text{damage}}(r \mid a) \cdot P_{\text{length}}(l_r \mid a)}{\pi_a \cdot P(r \mid a) + \pi_m \cdot P(r \mid m)}$$

The damage likelihood marginalises over all terminal C/G positions using the current parameter estimates. The length likelihood uses LogNormal distributions fitted per iteration.

**M-step:** Update mixture fraction π_a and distribution parameters (d, λ, μ_a, σ_a, μ_m, σ_m) from weighted counts.

Convergence is declared when Δπ_a < 0.01 between iterations (typically 5–15 iterations).

**Consensus calling:** Reads with p_a > 0.8 contribute to the ancient consensus; reads with p_a < 0.2 contribute to the modern consensus. Positions with < *min_depth* weighted coverage are masked with N.

**Damage correction (default: on):** The ancient consensus is corrected for assembly-baked damage before writing `ancient_consensus.fa`. Disable with `--no-polish`.

---

## Assembly-baked damage

When ancient reads are assembled, the assembler calls each position by majority vote. At positions where many reads begin or end, a large fraction show C→T damage at their 5′ termini. If damage exceeds 50% of reads, the assembler calls T instead of C — incorporating the damaged base into the reference. This is **assembly-baked damage**.

**Consequences:**
- Ancient reads now *match* the baked-in T → their C→T mismatch goes uncounted
- Modern reads show T→C mismatches (the reverse direction)
- Apparent damage rates are **suppressed** at heavily damaged positions
- The assembled sequence is **wrong** — T where the true genome has C

**When it matters:** Baking occurs when (ancient_fraction × damage_rate) > 0.5. At 40% ancient reads with 10% damage, baking is negligible. At 80% ancient with 30% damage, most terminal positions are baked in.

**How AMBER handles it:** The mismatch spectrum features include both C→T (residual damage) and T→C (baked-in proxy). The encoder learns to combine them: `ct_rate + tc_rate ≈ total terminal mismatch`, recovering the full damage signal regardless of baking. The `amber deconvolve` command corrects baked-in damage when writing the ancient consensus.

---

## Internal quality estimation

AMBER estimates bin completeness and contamination on-the-fly using the **206-marker CheckM universal single-copy gene set** (`checkm_markers_only.hmm`). No external tools are called during binning.

For each bin *B*:

$$\text{completeness}(B) = \frac{|\{m : \text{count}_B(m) \geq 1\}|}{206}$$

$$\text{contamination}(B) = \frac{\sum_m \max(0,\, \text{count}_B(m) - 1)}{\sum_m \text{count}_B(m)}$$

Quality thresholds:

| Tier | Completeness | Contamination |
|------|-------------|---------------|
| Strict-HQ | ≥ 90% | < 5% |
| Pre-HQ | ≥ 72% | < 5% |
| MQ | ≥ 50% | < 10% |

**CheckM2 is only used post-hoc for external validation**, not during binning.

---

## References

- Briggs AW et al. (2007) Patterns of damage in genomic DNA sequences from a Neandertal. *PNAS* 104:14616–21.
- Malkov YA, Yashunin DA (2018) Efficient and robust approximate nearest neighbor search using HNSW. *IEEE TPAMI* 42:824–36.
- Oord A van den et al. (2018) Representation Learning with Contrastive Predictive Coding. *arXiv* 1807.03748.
- Traag VA et al. (2019) From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports* 9:5233.
- Wang Z et al. (2024) COMEBin allows effective binning of metagenomic contigs using coverage multi-view encoder. *Nature Communications* 15:1119.

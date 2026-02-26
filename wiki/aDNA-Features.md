# aDNA Feature Space

AMBER embeds each contig into a **157-dimensional** space that combines a 128-dim self-supervised encoder with a 29-dim hand-crafted aDNA feature vector. The aDNA features are critical for binning ancient samples: they capture signals that the encoder alone cannot learn from tetranucleotide and coverage information.

---

## Encoder (128 dims)

Trained with damage-aware InfoNCE contrastive learning on substring augmentations of each contig. The encoder learns:
- Tetranucleotide composition (genome-specific)
- Coverage depth patterns (abundance-specific)
- Co-variation between composition and coverage across the training set

The encoder is deliberately kept agnostic to damage state: damage-aware weighting prevents it from learning damage level as a separating feature.

---

## Damage profile (10 dims)

Per-contig terminal substitution rates estimated from all reads mapped to the contig.

**Dimensions 1–5:** C→T rate at 5′ positions 1, 2, 3, 4, 5

$$\text{dim}_{k} = \frac{\sum_{r} \mathbb{1}[\text{ref}_{k}=C,\; \text{read}_{k}=T]}{\sum_{r} \mathbb{1}[\text{ref}_{k}=C]}$$

**Dimensions 6–10:** G→A rate at 3′ positions 1, 2, 3, 4, 5 (analogous, measured from the 3′ end).

**Soft-clip correction:** Adapter bases identified by CIGAR soft-clip operations are excluded before computing positional rates. Without this correction, overlapping paired-end reads inflate terminal G→A rates artifactually.

**Default for low-coverage contigs:** 0.5 (uninformative prior, places them in the centre of the damage axis).

**Interpretation:**
- Ancient contig: C→T at pos 1 ≈ 10–40%, decaying to baseline (~1%) by pos 5
- Modern contig: C→T at pos 1 ≈ 1–3%, no gradient
- Mixed contig: intermediate values, less clear gradient

---

## Decay parameters (2 dims)

**Dim 11: λ₅′** — exponential decay constant fitted to the 5′ C→T profile
**Dim 12: λ₃′** — exponential decay constant fitted to the 3′ G→A profile

Fit by minimising:

$$\hat{\lambda} = \arg\min_\lambda \sum_{p=1}^{P} \left(\hat{\delta}_p - d \cdot e^{-\lambda p}\right)^2$$

- Ancient DNA: λ ≈ 0.3–0.7 (steep decay, damage concentrated at termini)
- Modern DNA: λ ≈ undefined (no gradient to fit; set to 0.5)

---

## Fragment length (2 dims)

**Dim 13:** Mean fragment length (bp) of reads mapped to the contig, normalised by 150
**Dim 14:** Std dev of fragment lengths, normalised by 50

Ancient DNA is typically fragmented to 50–150 bp; modern environmental DNA reads 150–300 bp. This dimension captures systematic differences in read length distribution between ancient and modern genomes.

For paired-end data, fragment length = abs(template insert size) from the BAM `TLEN` field. For single-end data, fragment length = aligned read length.

---

## Damage-stratified coverage (2 dims)

**Dim 15:** log1p(coverage from ancient reads) / 5
**Dim 16:** log1p(coverage from modern reads) / 5

"Ancient reads" = p_ancient > 0.6; "Modern reads" = p_ancient < 0.4.

Coverage is computed per contig by summing soft-weighted base contributions:

$$\text{cov}_{\text{ancient}} = \frac{\sum_{r: p_a(r) > 0.6} p_a(r) \cdot l_r}{\text{len(contig)}}$$

The **damage coverage ratio** cov_ancient / (cov_ancient + cov_modern) is a clean per-genome signal for ancient vs modern status, independent of absolute depth. Genuinely ancient genomes have ratio ≈ 0.7–0.9; modern genomes ≈ 0.0–0.2; mixed genomes ≈ 0.3–0.6.

---

## Mismatch spectrum (4 dims)

Terminal substitution rates beyond the primary C→T and G→A signals:

**Dim 17:** T→C rate at 5′ (reverse-complement of G→A damage on the complementary strand; present in single-stranded library preps)
**Dim 18:** A→G rate at 5′
**Dim 19:** C→T rate at 3′ (reverse-complement; complements the G→A at 3′)
**Dim 20:** All other mismatch types at 3′

These distinguish single-stranded from double-stranded library preparation protocols and provide additional discrimination between ancient and modern contigs.

---

## CGR entropy features (9 dims)

Chaos Game Representation (CGR) maps k-mer frequencies onto a 2D fractal image. AMBER computes CGR at three resolutions (k = 4, 5, 6 corresponding to 16×16, 32×32, 64×64 grids) and extracts three metrics at each:

**Entropy:** Shannon entropy of the normalised frequency image:

$$H = -\sum_{x,y} p_{xy} \log_2 p_{xy}$$

Measures global sequence complexity. Low entropy = repetitive sequence; high entropy = uniform k-mer usage.

**Occupancy:** Fraction of cells with frequency > 0:

$$\text{occ} = \frac{|\{(x,y) : p_{xy} > 0\}|}{n^2}$$

Increases with genome diversity and decreases with very AT-rich or GC-rich genomes.

**Lacunarity:** Gap structure measure from box-counting:

$$\Lambda = \frac{\text{Var}(\text{occupancy at scale } r)}{\text{Mean}(\text{occupancy at scale } r)^2}$$

Captures regularity of repetitive elements and genomic architecture.

The **slopes** of entropy, occupancy, and lacunarity across the three scales (16→32→64) encode how genome composition varies with k-mer length — a scale-invariant fingerprint that complements the 4-mer-based encoder.

**Dims 21–29:** [entropy_16, entropy_32, entropy_64, occ_16, occ_32, occ_64, lac_16, lac_32, lac_64] (or equivalently, the three metrics at each of three resolutions)

---

## Feature normalisation

All 157 dimensions are standardised (zero mean, unit variance) across contigs before kNN graph construction. Encoder dimensions are L2-normalised before standardisation (inherited from COMEBin).

---

## Summary table

| Range | Dims | Feature group |
|-------|------|---------------|
| 1–128 | 128 | InfoNCE encoder embeddings |
| 129–138 | 10 | Damage profile (C→T 5′ pos 1–5, G→A 3′ pos 1–5) |
| 139–140 | 2 | Decay parameters (λ₅, λ₃) |
| 141–142 | 2 | Fragment length (mean, std) |
| 143–144 | 2 | Damage-stratified coverage (ancient, modern) |
| 145–148 | 4 | Mismatch spectrum (T→C, A→G at 5′; C→T, other at 3′) |
| 149–157 | 9 | CGR entropy/occupancy/lacunarity at 3 scales |
| **Total** | **157** | |

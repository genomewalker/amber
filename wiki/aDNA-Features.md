# aDNA Feature Space

AMBER embeds each contig into a **157-dimensional** space: 128 dims from the InfoNCE encoder, 20 dims from hand-crafted aDNA damage features, and 9 dims from multi-scale CGR composition features. The aDNA and CGR features bypass the encoder and are concatenated after training.

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

**Dim 17:** T→C rate at 5′
**Dim 18:** C→T rate at 3′
**Dim 19:** All other mismatch types at 5′
**Dim 20:** All other mismatch types at 3′

These four dimensions provide broader coverage of terminal substitution patterns beyond the primary C→T and G→A damage signal.

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

The actual 9 values are **6 scale-transition slopes** (how each metric changes between consecutive resolutions) plus **3 absolute values at the middle resolution**:

**Dims 21–26:** ΔH(16→32), ΔH(32→64), ΔO(16→32), ΔO(32→64), ΔL(16→32), ΔL(32→64)
**Dims 27–29:** H₃₂, O₃₂, L₃₂ (entropy, occupancy, lacunarity at 32×32)

The slope dimensions capture how composition scales with k-mer length; the absolute values anchor the representation at the most informative resolution.

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
| 145–148 | 4 | Mismatch spectrum (T→C at 5′, C→T at 3′, other at 5′, other at 3′) |
| 149–157 | 9 | CGR: 6 scale-transition slopes (ΔH, ΔO, ΔL at 16→32 and 32→64) + H₃₂, O₃₂, L₃₂ |
| **Total** | **157** | |

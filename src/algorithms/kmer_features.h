#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace amber {

// Cache-friendly, fixed-size feature vector
// Fits in ~2-3 cache lines (512 bytes total)
struct ContigFeatures {
  // Contig metadata
  std::string contig_id;
  uint32_t length;

  // === K-MER FEATURES ===
  // Raw k-mer counts (will be CLR-transformed and PCA-compressed later)
  // Using only TNF (4-mers) - the gold standard for metagenomic binning
  // Trinuc (3-mers) redundant with TNF + CGR quadrant densities
  // Pentanuc (5-mers) redundant with fractal entropy scaling features
  std::array<float, 136> tnf_raw;      // 4-mers canonical (256/2 + 8 = 136)

  // K-mer variance (local heterogeneity, 5 values from sliding windows)
  std::array<float, 5> kmer_variance;

  // === COMPOSITIONAL FEATURES ===
  float gc_content;
  float gc_variance;  // Local GC variation
  float gc_skewness;  // (G-C)/(G+C)
  float at_skewness;  // (A-T)/(A+T)

  // === DINUCLEOTIDE RATIOS (observed/expected) ===
  // Strong phylogenetic signals - CpG suppression varies by methylation system
  float cpg_ratio;    // CpG obs/exp - key discriminator for merge decisions
  float tpa_ratio;    // TpA obs/exp - complements CpG
  float gpc_ratio;    // GpC obs/exp
  float apt_ratio;    // ApT obs/exp

  // === COVERAGE FEATURES (from BAM) ===
  std::array<float, 20> coverage_bins; // Binned coverage (raw, will PCA later)
  float mean_coverage;
  float var_coverage;
  float coverage_skewness;

  // === DAMAGE FEATURES (from bdamage/metaDMG) ===
  // Per-position damage frequencies (positions 1-N from each end)
  // Size determined by max position in damage file
  std::vector<float> ct_damage_5p_pos;  // C->T at positions 1-N from 5' end
  std::vector<float> ag_damage_5p_pos;  // A->G at positions 1-N from 5' end
  std::vector<float> ct_damage_3p_pos;  // C->T at positions 1-N from 3' end (reverse reads)
  std::vector<float> ag_damage_3p_pos;  // A->G at positions 1-N from 3' end (reverse reads)

  // Legacy summary statistics (kept for compatibility, but superseded by per-position)
  float ct_damage_5p;  // Average C->T at 5' end
  float ct_damage_3p;  // Average C->T at 3' end
  float ag_damage_5p;  // Average A->G at 5' end
  float ag_damage_3p;  // Average A->G at 3' end
  float damage_gradient; // Decay rate along sequence
  float damage_asymmetry; // |5'-3'| difference

  // === STRAND BIAS ===
  float strand_asymmetry; // G vs C, T vs A bias

  // === CODON BIAS FEATURES (for protein-coding regions) ===
  float enc;        // Effective Number of Codons (20-61, lower = more bias)
  float cai;        // Codon Adaptation Index (0-1, higher = better adapted)
  float gc1, gc2, gc3; // GC content at codon positions 1, 2, 3

  // === DNA PHYSICAL PROPERTIES (from dinucleotide composition) ===
  float dna_bendability;    // Mean dinucleotide bendability
  float dna_flexibility;    // Mean dinucleotide flexibility
  float dna_meltability;    // Mean dinucleotide meltability (stacking energy)

  // === FRACTAL & CHAOS-THEORETIC FEATURES ===
  // CGR (Chaos Game Representation) - geometric invariants
  float cgr_fractal_dimension;    // Box-counting fractal dimension
  float cgr_lacunarity;           // Texture heterogeneity
  std::array<float, 4> cgr_hu_moments;  // Rotation/scale invariant shape (moments 1-4 capture 95% variance)
  std::array<float, 10> cgr_radial_density;  // Multi-scale radial structure
  std::array<float, 4> cgr_quadrant_density; // ACGT corner biases
  float cgr_entropy;              // Shannon entropy of CGR grid
  float cgr_uniformity;           // Uniformity (sum of squared densities)
  float cgr_max_density;          // Maximum density in CGR grid

  // Multi-scale complexity
  std::array<float, 4> lempel_ziv_complexity; // Algorithmic complexity (4 scales)
  std::array<float, 3> hurst_exponent;        // Long-range correlations (GC, dinuc, kmer signals)
  float dfa_exponent;                          // Detrended fluctuation analysis
  std::array<float, 5> entropy_scaling;       // k-mer entropy at k=2,3,4,5,6
  float kmer_fractal_dim;                      // Fractal dim from entropy scaling

  // CGR-DFA fractal features (from CGR trajectory random walk)
  float cgr_hurst_x;              // Hurst exponent from CGR X trajectory
  float cgr_hurst_y;              // Hurst exponent from CGR Y trajectory
  float cgr_anisotropy;           // |Hurst_x - Hurst_y| directional bias
  float cgr_f_ratio;              // DFA fluctuation ratio (large/small scale)
  float cgr_local_hurst_var;      // Local Hurst variability (multifractal width proxy)

  // MFDFA quality metrics (for bin quality assessment)
  float mfdfa_hurst;              // Combined CGR Hurst exponent
  float mfdfa_width;              // Multifractal spectrum width
  float mfdfa_variance;           // Hurst variance across local windows

  // De Bruijn graph topology
  float graph_clustering_coef;     // Local connectivity
  float graph_avg_path_length;     // Global structure
  float graph_modularity;          // Community structure
  int   graph_num_components;      // Fragmentation
  float graph_degree_mean;         // Degree distribution
  float graph_degree_variance;
  float graph_degree_skewness;
  float graph_power_law_exp;       // Power-law exponent
  float graph_repeat_density;      // Palindrome/repeat structure

  // Coverage topology (if coverage available)
  float cov_hurst_exponent;              // Coverage long-range correlations
  float cov_fractal_dimension;           // Coverage surface roughness
  std::array<float, 8> cov_wavelet_coef; // Multi-scale wavelet decomposition
  float cov_variogram_sill;              // Spatial autocorrelation - maximum variance
  float cov_variogram_range;             // Correlation length
  // Note: variogram_nugget removed (unstable with only 20 coverage bins)

  // Constructor
  ContigFeatures() : length(0) {
    tnf_raw.fill(0.0f);
    kmer_variance.fill(0.0f);
    gc_content = gc_variance = gc_skewness = at_skewness = 0.0f;
    cpg_ratio = tpa_ratio = gpc_ratio = apt_ratio = 1.0f;  // Default = no bias
    coverage_bins.fill(0.0f);
    mean_coverage = var_coverage = coverage_skewness = 0.0f;
    ct_damage_5p = ct_damage_3p = ag_damage_5p = ag_damage_3p = 0.0f;
    damage_gradient = damage_asymmetry = 0.0f;
    strand_asymmetry = 0.0f;
    enc = cai = gc1 = gc2 = gc3 = 0.0f;

    // Initialize DNA physical properties
    dna_bendability = dna_flexibility = dna_meltability = 0.0f;

    // Initialize fractal features
    cgr_fractal_dimension = cgr_lacunarity = 0.0f;
    cgr_hu_moments.fill(0.0f);
    cgr_radial_density.fill(0.0f);
    cgr_quadrant_density.fill(0.0f);
    cgr_entropy = cgr_uniformity = cgr_max_density = 0.0f;
    lempel_ziv_complexity.fill(0.0f);
    hurst_exponent.fill(0.5f);  // Default = random walk
    dfa_exponent = kmer_fractal_dim = 0.0f;
    entropy_scaling.fill(0.0f);

    // Initialize CGR-DFA fractal features
    cgr_hurst_x = cgr_hurst_y = 0.5f;  // Default = random walk
    cgr_anisotropy = cgr_f_ratio = cgr_local_hurst_var = 0.0f;

    // Initialize MFDFA quality metrics
    mfdfa_hurst = 0.5f;
    mfdfa_width = mfdfa_variance = 0.0f;

    graph_clustering_coef = graph_avg_path_length = graph_modularity = 0.0f;
    graph_num_components = 0;
    graph_degree_mean = graph_degree_variance = graph_degree_skewness = 0.0f;
    graph_power_law_exp = graph_repeat_density = 0.0f;
    cov_hurst_exponent = cov_fractal_dimension = 0.0f;
    cov_wavelet_coef.fill(0.0f);
    cov_variogram_sill = cov_variogram_range = 0.0f;
  }
};

// Compressed features for clustering (after PCA)
struct CompressedFeatures {
  std::string contig_id;
  std::array<float, 30> features; // ~30 dimensions after PCA

  CompressedFeatures() { features.fill(0.0f); }
};

// Fast k-mer utilities using bit-packing
class KmerUtils {
public:
  // Convert base to 2-bit representation: A=00, C=01, G=10, T=11
  static inline uint8_t base_to_bits(char c) {
    switch (c) {
    case 'A':
    case 'a':
      return 0;
    case 'C':
    case 'c':
      return 1;
    case 'G':
    case 'g':
      return 2;
    case 'T':
    case 't':
      return 3;
    default:
      return 255; // Invalid/ambiguous base
    }
  }

  // Convert k-mer to integer index (for array lookup)
  // E.g., "ACG" (k=3) -> 001 010 -> 6
  static inline uint32_t kmer_to_index(const char *kmer, int k) {
    uint32_t idx = 0;
    for (int i = 0; i < k; ++i) {
      idx = (idx << 2) | base_to_bits(kmer[i]);
    }
    return idx;
  }

  // Reverse complement of a k-mer index
  static inline uint32_t reverse_complement_index(uint32_t idx, int k) {
    uint32_t rc = 0;
    for (int i = 0; i < k; ++i) {
      uint32_t base = idx & 0x3;            // Extract last 2 bits
      uint32_t comp = 3 - base;             // Complement: A<->T, C<->G
      rc = (rc << 2) | comp;
      idx >>= 2;
    }
    return rc;
  }

  // Canonical k-mer index (lexicographically smaller of kmer and its RC)
  static inline uint32_t canonical_index(uint32_t idx, int k) {
    uint32_t rc = reverse_complement_index(idx, k);
    return (idx < rc) ? idx : rc;
  }

  // Number of canonical k-mers
  static inline size_t num_canonical_kmers(int k) {
    size_t total = 1ULL << (2 * k); // 4^k
    return (total / 2) + (1ULL << (k - 1)); // Half + palindromes
  }
};

// CLR (Centered Log-Ratio) transformation for compositional data
class CLRTransform {
public:
  // Apply CLR transform: clr(x_i) = log(x_i / geom_mean(x))
  template <typename T, size_t N>
  static void apply(std::array<T, N> &data) {
    // Add pseudocount to avoid log(0)
    const T pseudocount = static_cast<T>(1e-6);

    // Compute geometric mean
    T log_sum = 0.0;
    size_t count = 0;
    for (const auto &val : data) {
      if (val + pseudocount > 0) {
        log_sum += std::log(val + pseudocount);
        count++;
      }
    }

    if (count == 0) return;

    T log_geom_mean = log_sum / count;

    // Apply transformation
    for (auto &val : data) {
      val = std::log(val + pseudocount) - log_geom_mean;
    }
  }
};

// Forward declarations for k-mer/composition feature extraction
void extract_kmer_features(const std::string &seq, ContigFeatures &features);
void extract_composition_features(const std::string &seq, ContigFeatures &features);

// ============================================================================
// BINNING-OPTIMIZED TNF EXTRACTION
// ============================================================================
// Based on distance metric evaluation (EXPERIMENT_METRICS.md):
//
// | Normalization | Distance        | AUC   | Use Case              |
// |---------------|-----------------|-------|------------------------|
// | CLR           | Euclidean       | 0.956 | BEST for binning       |
// | L2            | Euclidean       | 0.937 | Baseline compatible    |
// | Frequency     | Bray-Curtis     | 0.952 | Alternative            |
//
// CLR (Centered Log-Ratio) transform makes Euclidean distance equivalent
// to Aitchison distance, which is optimal for compositional data like TNF.

// Extract TNF with CLR transform applied (for Aitchison distance via Euclidean)
// This is the RECOMMENDED method for metagenomic binning
void extract_tnf_clr(const std::string &seq, std::array<float, 136> &tnf_clr);

// Extract TNF with L2 normalization (baseline-compatible)
// Use this for compatibility with existing binners that expect unit vectors
void extract_tnf_l2(const std::string &seq, std::array<float, 136> &tnf_l2);

// Extract RAW INTEGER COUNTS - matches COMEBin's gen_kmer.py output exactly
// No normalization - used for COMEBin compatibility verification
void extract_tnf_raw(const std::string &seq, std::array<uint32_t, 136> &tnf_raw);

// Extract TNF with COMEBin normalization: (raw + 1) / sum
// This is EXACTLY how COMEBin normalizes k-mers (get_augfeature.py lines 113-114):
//   compositMats = compositMats + 1
//   compositMats = compositMats / compositMats.sum(axis=1)[:, None]
void extract_tnf_comebin(const std::string &seq, std::array<float, 136> &tnf_comebin);

} // namespace amber

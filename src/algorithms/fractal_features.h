#pragma once

#include <array>
#include <string>
#include <vector>
#include <complex>
#include <unordered_map>
#include <unordered_set>

namespace amber {

// DNA Physical Properties (from dinucleotide composition)
struct DNAPhysicalProperties {
  double bendability;    // Mean dinucleotide bendability
  double flexibility;    // Mean dinucleotide flexibility
  double meltability;    // Mean dinucleotide meltability (stacking energy)

  DNAPhysicalProperties() : bendability(0.0), flexibility(0.0), meltability(0.0) {}
};

// Chaos Game Representation (CGR) features
// Maps DNA sequences to 2D fractal patterns
struct CGRFeatures {
  double fractal_dimension;     // Box-counting dimension of CGR pattern
  double lacunarity;            // "Gappiness" - measures texture heterogeneity

  // Hu's 7 moment invariants (rotation/scale/translation invariant)
  std::array<double, 7> hu_moments;

  // Radial density profile (multi-scale)
  std::array<double, 10> radial_density;

  // Quadrant occupancy (ACGT corner biases)
  std::array<double, 4> quadrant_density;

  // Additional CGR grid statistics
  double entropy;          // Shannon entropy of CGR grid
  double uniformity;       // Sum of squared densities
  double max_density;      // Maximum density in grid

  CGRFeatures() : fractal_dimension(0.0), lacunarity(0.0),
                  entropy(0.0), uniformity(0.0), max_density(0.0) {
    hu_moments.fill(0.0);
    radial_density.fill(0.0);
    quadrant_density.fill(0.0);
  }
};

// CGR-DFA fractal features (from CGR trajectory random walk analysis)
// These capture scale-dependent correlations in sequence structure
struct CGRDFAFeatures {
  double hurst_x;            // Hurst exponent from CGR X trajectory
  double hurst_y;            // Hurst exponent from CGR Y trajectory
  double anisotropy;         // |Hurst_x - Hurst_y| directional bias
  double f_ratio;            // DFA fluctuation ratio (large/small scale)
  double local_hurst_var;    // Local Hurst variability (multifractal width proxy)

  CGRDFAFeatures() : hurst_x(0.5), hurst_y(0.5), anisotropy(0.0),
                     f_ratio(0.5), local_hurst_var(0.0) {}
};

// MFDFA Quality Metrics (for bin quality assessment)
struct MFDFAQualityMetrics {
  double hurst;              // Combined CGR Hurst exponent
  double width;              // Multifractal spectrum width (local Hurst variability)
  double variance;           // Hurst variance across local windows

  // Quality indicators
  bool is_consistent() const {
    return hurst >= 0.4 && hurst <= 0.8 && variance < 0.05;
  }

  MFDFAQualityMetrics() : hurst(0.5), width(0.0), variance(0.0) {}
};

// Multi-scale complexity features
struct ComplexityFeatures {
  // Lempel-Ziv complexity at multiple scales
  std::array<double, 4> lempel_ziv_complexity;  // 100bp, 500bp, 2kb, full

  // Hurst exponent (long-range correlations, 0.5=random, >0.5=persistent, <0.5=antipersistent)
  std::array<double, 3> hurst_exponent;  // GC%, dinuc freq, k-mer diversity

  // Detrended Fluctuation Analysis (DFA) - power-law scaling exponent
  double dfa_exponent;

  // Shannon entropy at multiple k-mer scales
  std::array<double, 5> entropy_scaling;  // k=2,3,4,5,6

  // Fractal dimension from k-mer scaling (how diversity scales with k)
  double kmer_fractal_dim;

  ComplexityFeatures() : dfa_exponent(0.0), kmer_fractal_dim(0.0) {
    lempel_ziv_complexity.fill(0.0);
    hurst_exponent.fill(0.5);  // Default to random walk
    entropy_scaling.fill(0.0);
  }
};

// De Bruijn graph topology (graph-theoretic invariants)
struct GraphTopology {
  double clustering_coefficient;  // Local connectivity
  double avg_path_length;         // Global structure
  double modularity;              // Community structure strength
  int num_components;             // Graph fragmentation

  // Degree distribution moments
  double degree_mean;
  double degree_variance;
  double degree_skewness;

  // Power-law exponent (if degree dist follows power law)
  double power_law_exponent;

  // Repeat/palindrome density
  double repeat_density;

  GraphTopology() :
    clustering_coefficient(0.0), avg_path_length(0.0),
    modularity(0.0), num_components(0),
    degree_mean(0.0), degree_variance(0.0), degree_skewness(0.0),
    power_law_exponent(0.0), repeat_density(0.0) {}
};

// Coverage topology features (if coverage available)
struct CoverageTopology {
  double hurst_exponent;       // Long-range correlations in coverage
  double fractal_dimension;    // Surface roughness of coverage landscape

  // Wavelet decomposition (multi-scale)
  std::array<double, 8> wavelet_coefficients;

  // Variogram parameters (spatial autocorrelation)
  double variogram_sill;       // Maximum variance
  double variogram_range;      // Correlation length
  // Note: variogram_nugget removed (unstable with only 20 coverage bins)

  CoverageTopology() :
    hurst_exponent(0.5), fractal_dimension(0.0),
    variogram_sill(0.0), variogram_range(0.0) {
    wavelet_coefficients.fill(0.0);
  }
};

// Combined CGR-based features result (for optimized extraction)
struct CombinedCGRFeatures {
  CGRDFAFeatures cgr_dfa;
  MFDFAQualityMetrics mfdfa;
  bool has_cgr_dfa = false;
  bool has_mfdfa = false;
};

// ============================================================================
// CGR GEOMETRIC FEATURES (Optimized for Binning)
// ============================================================================
// Based on distance metric evaluation (EXPERIMENT_METRICS.md):
// - CGR with Bray-Curtis achieves AUC 0.880
// - Multi-scale quadrant densities capture ACGT bias at k=3,4,5,6 resolution
// - Radial profile captures overall structure
// - Hu moments (log-transformed) are rotation/scale invariant
// - L2 normalized for use with Euclidean distance
//
// This is the EXACT implementation from the baseline binner (mfdfa_controlled_binner.cpp)
// for reproducibility.

struct CGRGeometric {
  // Multi-scale quadrant densities (16 features)
  // Captures ACGT bias at scales equivalent to k=3,4,5,6
  std::array<double, 16> quadrant_multiscale;

  // Radial density profile (8 features)
  // Distance from center captures overall CGR structure
  std::array<double, 8> radial_profile;

  // Hu moments (4 features, log-transformed)
  // Rotation and scale invariant shape descriptors
  std::array<double, 4> hu_moments_log;

  // Scalar features (3 features)
  double entropy;
  double uniformity;
  double max_density;

  // Combined L2-normalized vector (31 features)
  // This is the format expected by the baseline binner
  std::array<double, 32> normalized_vector;

  CGRGeometric() : entropy(0.0), uniformity(0.0), max_density(0.0) {
    quadrant_multiscale.fill(0.0);
    radial_profile.fill(0.0);
    hu_moments_log.fill(0.0);
    normalized_vector.fill(0.0);
  }
};

// Master fractal feature extractor
class FractalFeatures {
public:
  // Extract DNA physical properties (bendability, flexibility, meltability)
  static DNAPhysicalProperties extract_dna_physical(const std::string &sequence);

  // Extract CGR-based features (original implementation)
  static CGRFeatures extract_cgr_features(const std::string &sequence);

  // ========================================================================
  // BINNING-OPTIMIZED CGR EXTRACTION
  // ========================================================================
  // Extract CGR geometric features using 64x64 grid (baseline-compatible)
  // Returns L2-normalized vector suitable for Euclidean distance or
  // raw features suitable for Bray-Curtis distance (AUC 0.880)
  static CGRGeometric extract_cgr_geometric(const std::string &sequence);

  // Extract CGR-DFA fractal features (Hurst from CGR trajectory)
  static CGRDFAFeatures extract_cgr_dfa_features(const std::string &sequence);

  // Extract MFDFA quality metrics (for bin quality assessment)
  static MFDFAQualityMetrics extract_mfdfa_quality(const std::string &sequence);

  // ========================================================================
  // OPTIMIZED COMBINED EXTRACTION (2-3x faster than separate calls)
  // ========================================================================

  // Extract both CGR-DFA and MFDFA features with shared CGR trajectory
  // This avoids computing the CGR trajectory twice and shares DFA results
  static CombinedCGRFeatures extract_cgr_features_combined(const std::string &sequence);

  // Extract complexity features
  static ComplexityFeatures extract_complexity_features(const std::string &sequence);

  // Extract graph topology
  static GraphTopology extract_graph_topology(const std::string &sequence, int k = 4);

  // Extract coverage topology (requires coverage array)
  static CoverageTopology extract_coverage_topology(
      const std::vector<float> &coverage, int contig_length);

private:
  // CGR helper functions
  static std::vector<std::pair<double, double>> compute_cgr_points(
      const std::string &seq, int max_points = 10000);

  static double compute_fractal_dimension(
      const std::vector<std::pair<double, double>> &points);

  static double compute_lacunarity(
      const std::vector<std::pair<double, double>> &points);

  static std::array<double, 7> compute_hu_moments(
      const std::vector<std::pair<double, double>> &points);

  // Complexity helper functions
  static double lempel_ziv_complexity(const std::string &seq);

  static double compute_hurst_exponent(const std::vector<double> &signal);

  static double compute_dfa_exponent(const std::vector<double> &signal);

  static std::array<double, 5> compute_entropy_scaling(const std::string &seq);

public:
  // DFA overlap stride factor (configurable). If <=0, use full overlap.
  // Effective stride = max(1, scale / dfa_stride_factor) on long series.
  // Made public for access by standalone helper functions.
  static int dfa_stride_factor;

private:

  // Graph helper functions
  static std::vector<std::pair<int, int>> build_debruijn_edges(
      const std::string &seq, int k);

  static double compute_clustering_coefficient(
      const std::vector<std::pair<int, int>> &edges, int num_nodes);

  static double compute_avg_path_length(
      const std::unordered_map<int, std::unordered_set<int>> &adj,
      const std::unordered_set<int> &nodes);

  static double compute_modularity(
      const std::vector<std::pair<int, int>> &edges,
      const std::unordered_map<int, std::unordered_set<int>> &adj);

  static double compute_power_law_exponent(
      const std::vector<int> &degree_sequence);

  // Coverage helper functions
  static std::array<double, 8> wavelet_decomposition(
      const std::vector<float> &signal);

  static void compute_variogram(
      const std::vector<float> &signal,
      double &sill, double &range);  // nugget removed (unstable with limited bins)
};

} // namespace amber

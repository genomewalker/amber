// MFDFA Binner - Constants
#pragma once

namespace amber {

// Feature dimensions
constexpr int TNF_DIM = 136;       // Tetranucleotide frequency
constexpr int CGR_DIM = 32;        // CGR geometric features
constexpr int DNA_DIM = 4;         // DNA physical properties
constexpr int FRACTAL_DIM = 5;     // CGR-DFA fractal features
constexpr int STABLE_DIM = 14;     // Stable features for resonance merger

// CGR feature indices
constexpr int CGR_IDX_FRACTAL_DIM = 0;
constexpr int CGR_IDX_LACUNARITY = 1;
constexpr int CGR_IDX_ENTROPY = 23;

// EM clustering log-weights (applied to log-probabilities)
// Higher = more influence on bin assignment
// Fixed weights (empirically tuned - MUST match original baseline for 8 HQ)
constexpr double TNF_LOG_WEIGHT = 2.0;
constexpr double CGR_LOG_WEIGHT = 1.5;           // K-mer density pattern
constexpr double FRACTAL_LOG_WEIGHT = 2.0;       // CGR-DFA Hurst, anisotropy
constexpr double DNA_PHYSICAL_LOG_WEIGHT = 2.0;  // Bendability, flexibility
constexpr double COVERAGE_LOG_WEIGHT = 5.0;      // Crucial for bin separation

// Coverage parameters
constexpr int MAX_EDGE_BASES = 75;               // Bases to exclude from edges
constexpr double MIN_PERCENT_IDENTITY = 90.0;    // Min alignment identity
constexpr int MIN_MAPQ = 0;                      // Min mapping quality

// K estimation bounds
constexpr int MIN_BINS = 5;
constexpr int MAX_BINS = 100;

// Quality thresholds
constexpr double HURST_OUTLIER_ZSCORE = 2.0;     // Z-score threshold for Hurst outliers
constexpr double MIN_BIN_HURST_STD = 0.10;       // Min std before enforcing Hurst QC

// Distance metric parameters
constexpr double GC_ATTENUATION = 5.0;           // Reduce composition distance when GC differs

// EM convergence thresholds
constexpr double LL_CONVERGENCE_THRESHOLD = 0.01;       // Log-likelihood convergence
constexpr double REJECTION_CONVERGENCE_THRESHOLD = 0.01; // Rejection rate convergence

// Contamination cleanup parameters
constexpr int MIN_BIN_SIZE_FOR_CLEANUP = 10;     // Min contigs before cleanup
constexpr int MIN_SUBCLUSTER_SIZE = 3;           // Min size for subclusters
constexpr double SPLIT_IMPROVEMENT_THRESHOLD = 0.2; // 20% improvement required

// Bayesian error model
constexpr double kSeqErrorDenom = 3.0;  // q_err / 3.0 pattern

// Merge thresholds
constexpr double kDefaultCovRatioMax = 2.0;
constexpr double kStrictCovRatioMax = 2.0;
constexpr double kBICMargin = 10.0;

// Statistical thresholds
constexpr double kKSPvalueThreshold = 0.05;
constexpr double kCGRZThreshold = 3.0;
constexpr double kFractalZThreshold = 1.5;
constexpr double kMaxRemovalFrac = 0.15;

// Cleanup thresholds
constexpr double kCovMADThreshold = 2.5;
constexpr double kDmgMADThreshold = 3.0;
constexpr double kImprovementFactor = 0.7;

}  // namespace amber

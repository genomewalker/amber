// MFDFA Binner - Cleanup Operations
// Contamination removal utilities for metagenomic bins
#pragma once

#include "constants.h"
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <algorithm>
#include <cctype>

namespace amber {

// =============================================================================
// Cleanup Parameters (constants for cleanup operations)
// =============================================================================

namespace cleanup {

// Coverage outlier cleanup parameters
constexpr double kDefaultCovDeviation = 0.40;    // Remove contigs outside 60-140% of median
constexpr double kDefaultMinCovCV = 0.25;        // Minimum CV to trigger cleanup
constexpr double kDefaultMinMeanCov = 2.0;       // Minimum mean coverage for reliable CV

// TNF contamination cleanup parameters
constexpr double kDefaultTnfSigmaThreshold = 2.0;   // MAD sigma threshold for outliers
constexpr double kDefaultTnfTopPercentile = 0.30;   // Only clean top 30% by contamination score

// Safety constraints
constexpr double kMinRetentionFraction = 0.80;   // Keep at least 80% of contigs when cleaning
constexpr int kMinContigsForCleanup = 10;        // Minimum contigs to apply cleanup

// Feature-based cleanup
constexpr size_t kMinBinContigsFeature = 50;     // Min contigs for CGR+fractal cleanup
constexpr size_t kMinBinBpFeature = 1000000;     // Min 1Mb for CGR+fractal cleanup

// CpG-based cleanup
constexpr size_t kMinBinContigsCpg = 10;
constexpr double kCpgZThreshold = 3.0;           // MAD-based z-score threshold
constexpr double kCpgMaxRemovalFrac = 0.20;      // Max 20% removal

// Karlin-based cleanup
constexpr size_t kMinBinContigsKarlin = 20;      // Need more contigs for reliable stats
constexpr double kKarlinZThreshold = 3.5;        // Higher threshold (second-pass)
constexpr double kKarlinMaxRemovalFrac = 0.15;   // More conservative (15%)

// Graph-based cleanup (hierarchical clustering)
constexpr size_t kMinBinContigsGraph = 15;
constexpr double kClusterThreshold = 0.045;      // Within-species clustering threshold
constexpr double kInterSpeciesDist = 0.06;       // Min avg dist to flag as contaminant
constexpr double kIsolationThreshold = 1.5;      // Minority must be 1.5x more distant
constexpr size_t kMinClusterSize = 2;
constexpr double kMaxMinorityFrac = 0.25;        // Don't remove clusters > 25% of bin

}  // namespace cleanup

// =============================================================================
// Statistical Utilities
// =============================================================================

// Compute median and MAD (median absolute deviation) for robust statistics
// Returns {median, scaled_MAD} where MAD is scaled by 1.4826 to match std dev
inline std::pair<double, double> compute_median_mad(std::vector<double>& vals) {
    if (vals.empty()) return {0.0, 1e-9};

    std::vector<double> sorted_vals = vals;
    std::sort(sorted_vals.begin(), sorted_vals.end());
    double median = sorted_vals[sorted_vals.size() / 2];

    std::vector<double> devs;
    devs.reserve(sorted_vals.size());
    for (double v : sorted_vals) {
        devs.push_back(std::abs(v - median));
    }
    std::sort(devs.begin(), devs.end());

    // Scale MAD to standard deviation equivalent (for normal distribution)
    double mad = devs[devs.size() / 2] * 1.4826;
    return {median, std::max(mad, 1e-9)};
}

// Overload that doesn't modify input
inline std::pair<double, double> compute_median_mad(const std::vector<double>& vals) {
    std::vector<double> copy = vals;
    return compute_median_mad(copy);
}

// =============================================================================
// Genomic Signature Utilities
// =============================================================================

// Compute CpG observed/expected ratio for a sequence
// CpG_oe = (CpG_count * length) / (C_count * G_count)
// Prokaryotes typically have CpG o/e < 1 due to methylation-driven deamination
// Different species have characteristic CpG depletion patterns
inline double compute_cpg_oe(const std::string& seq) {
    int c_count = 0, g_count = 0, cpg_count = 0;
    size_t len = seq.length();

    for (size_t i = 0; i < len; i++) {
        char base = std::toupper(seq[i]);
        if (base == 'C') {
            c_count++;
            if (i + 1 < len && std::toupper(seq[i + 1]) == 'G') {
                cpg_count++;
            }
        } else if (base == 'G') {
            g_count++;
        }
    }

    // Avoid division by zero
    if (c_count == 0 || g_count == 0 || len == 0) return 1.0;

    // CpG o/e = observed / expected = (cpg/len) / ((c/len) * (g/len))
    double expected = static_cast<double>(c_count * g_count) / (len * len);
    double observed = static_cast<double>(cpg_count) / len;
    return expected > 0 ? observed / expected : 1.0;
}

// Karlin distance: δ*(a,b) = (1/16) * Σ|ρ_XY(a) - ρ_XY(b)|
// Measures dinucleotide signature difference between sequences
// Within species: ~0.03-0.05
// Between species: >0.08
inline double karlin_distance(const std::array<double, 16>& a,
                              const std::array<double, 16>& b) {
    double sum = 0.0;
    for (int k = 0; k < 16; k++) {
        sum += std::abs(a[k] - b[k]);
    }
    return sum / 16.0;
}

// Compute centroid of Karlin signatures (dinucleotide relative abundances)
template<typename ContigPtr>
inline std::array<double, 16> compute_karlin_centroid(
    const std::vector<ContigPtr>& members) {

    std::array<double, 16> centroid = {};
    if (members.empty()) return centroid;

    for (const auto* c : members) {
        for (int k = 0; k < 16; k++) {
            centroid[k] += c->dinuc_rho[k];
        }
    }

    for (int k = 0; k < 16; k++) {
        centroid[k] /= members.size();
    }

    return centroid;
}

// =============================================================================
// TNF (Tetranucleotide Frequency) Utilities
// =============================================================================

// TNF_DIM is defined in constants.h (136 dimensions)

// Compute L2 distance from contig TNF to centroid
template<typename Contig>
inline double compute_tnf_distance(const Contig& c,
                                   const std::vector<double>& centroid) {
    if (c.tnf.size() != TNF_DIM || centroid.size() != TNF_DIM) {
        return -1.0;  // Invalid
    }

    double dist = 0.0;
    for (int i = 0; i < TNF_DIM; i++) {
        double d = c.tnf[i] - centroid[i];
        dist += d * d;
    }
    return std::sqrt(dist);
}

// Compute length-weighted TNF centroid
template<typename ContigPtr>
inline std::vector<double> compute_tnf_centroid(
    const std::vector<ContigPtr>& members) {

    std::vector<double> centroid(TNF_DIM, 0.0);
    double total_len = 0.0;

    for (const auto* c : members) {
        if (c->tnf.size() != TNF_DIM) continue;
        for (int i = 0; i < TNF_DIM; i++) {
            centroid[i] += c->tnf[i] * c->length;
        }
        total_len += c->length;
    }

    if (total_len < 1) return centroid;

    for (int i = 0; i < TNF_DIM; i++) {
        centroid[i] /= total_len;
    }

    return centroid;
}

// =============================================================================
// Cleanup Result Structures
// =============================================================================

// Result of a single cleanup operation
struct CleanupResult {
    int bins_cleaned = 0;
    int contigs_removed = 0;
    size_t bp_removed = 0;
};

// Aggregate result across all cleanup stages
struct CleanupSummary {
    CleanupResult coverage_aware;     // GMM-based coverage splitting
    CleanupResult tnf_based;          // TNF compositional outliers
    CleanupResult coverage_outlier;   // Coverage outliers (third pass)
    CleanupResult feature_based;      // CGR+fractal gated detection
    CleanupResult cpg_based;          // CpG ratio outliers
    CleanupResult karlin_based;       // Karlin signature outliers
    CleanupResult graph_based;        // Hierarchical clustering

    int total_contigs_removed() const {
        return coverage_aware.contigs_removed + tnf_based.contigs_removed +
               coverage_outlier.contigs_removed + feature_based.contigs_removed +
               cpg_based.contigs_removed + karlin_based.contigs_removed +
               graph_based.contigs_removed;
    }

    size_t total_bp_removed() const {
        return coverage_aware.bp_removed + tnf_based.bp_removed +
               coverage_outlier.bp_removed + feature_based.bp_removed +
               cpg_based.bp_removed + karlin_based.bp_removed +
               graph_based.bp_removed;
    }
};

// =============================================================================
// Documentation: Member functions remaining in MFDFABinner
// =============================================================================
//
// The following cleanup methods remain in binner.cpp as MFDFABinner members
// because they require access to class state (bins, log_, cleanup thresholds):
//
// 1. coverage_aware_contamination_cleanup()
//    - GMM-based coverage splitting for multi-modal coverage distributions
//    - Uses compute_contamination_score_denovo() for bin selection
//    - Creates new bins when coverage modes are detected
//
// 2. tnf_based_contamination_cleanup(sigma_threshold, top_percentile)
//    - Removes compositional outliers based on TNF distance from centroid
//    - Uses percentile-based threshold for dataset independence
//    - Applies MAD-based outlier detection
//
// 3. coverage_outlier_cleanup(cov_deviation, min_cov_cv, min_mean_cov)
//    - Removes contigs with anomalous coverage that escaped GMM splitting
//    - Uses coverage CV as trigger for dataset independence
//    - Applies median-based bounds (default 60-140% of median)
//
// 4. feature_based_cleanup()
//    - CGR+fractal gated detection (requires BOTH signals to flag outlier)
//    - Primary signal: CGR indices 21, 22, 23
//    - Secondary gate: fractal indices 1, 3
//    - Uses member variables: cgr_z_threshold, fractal_z_threshold, max_removal_frac
//
// 5. cpg_based_cleanup()
//    - Removes outliers based on CpG observed/expected ratio
//    - Uses MAD-based z-score threshold
//    - Flags removed contigs as flagged_contaminant
//
// 6. karlin_based_cleanup()
//    - Uses all 16 dinucleotides for more sensitive detection
//    - Higher threshold than CpG (second-pass cleanup)
//    - Flags removed contigs as flagged_contaminant
//
// 7. graph_based_cleanup() [Currently disabled]
//    - Hierarchical clustering with adaptive thresholds
//    - Detects minority clusters that are genomically distinct
//    - Uses Karlin distance with absolute species-level thresholds
//
// Pipeline order in run_binning():
//   1. coverage_aware_contamination_cleanup()
//   2. tnf_based_contamination_cleanup()
//   3. coverage_outlier_cleanup()
//   4. feature_based_cleanup()
//   5. cpg_based_cleanup()
//   6. karlin_based_cleanup()
//   7. graph_based_cleanup() -- disabled, too aggressive

}  // namespace amber

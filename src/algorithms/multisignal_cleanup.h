#pragma once
// ==============================================================================
// Multi-Signal Contamination Cleanup
//
// Unified cleanup using three orthogonal signals:
//   1. TNF core similarity - compositional structure
//   2. CGR core distance - geometric structure
//   3. Coverage z-score - abundance differences
//
// Replaces: tnf_based_cleanup, feature_based_cleanup, cpg_based_cleanup,
//           karlin_based_cleanup (all captured by these three signals)
//
// Algorithm:
//   1. Compute TNF and CGR features for all contigs in bin
//   2. Identify "core" contigs (top 50% by mean TNF similarity)
//   3. Score each contig: (1-TNF_core_sim) + α*CGR_norm + β*cov_zscore
//   4. Flag contigs with score above threshold (adaptive or percentile)
// ==============================================================================

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>

namespace amber {
namespace multisignal {

constexpr int TNF_DIM = 136;
constexpr int CGR_DIM = 32;

// =============================================================================
// Feature Structures (minimal, for cleanup only)
// =============================================================================

struct ContigSignals {
    size_t idx;
    std::array<float, TNF_DIM> tnf;
    std::array<double, CGR_DIM> cgr;
    double coverage;
    size_t length;
};

// =============================================================================
// Similarity/Distance Functions
// =============================================================================

inline double cosine_similarity(const std::array<float, TNF_DIM>& a,
                                const std::array<float, TNF_DIM>& b) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < TNF_DIM; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-10 || nb < 1e-10) return 0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

inline double euclidean_distance(const std::array<double, CGR_DIM>& a,
                                 const std::array<double, CGR_DIM>& b) {
    double sum = 0;
    for (int i = 0; i < CGR_DIM; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

// =============================================================================
// Multi-Signal Contamination Score
// =============================================================================

struct CleanupResult {
    std::vector<size_t> contigs_to_remove;  // Indices of contigs to remove
    std::vector<double> scores;              // Score for each contig
    double threshold_used;
    int n_flagged;

    // Per-signal statistics (for logging)
    double median_tnf_core_sim;
    double median_cgr_core_dist;
    double median_cov_zscore;
};

inline CleanupResult compute_contamination_scores(
    const std::vector<ContigSignals>& contigs,
    double alpha_cgr = 0.3,      // Weight for CGR signal
    double beta_cov = 0.3,       // Weight for coverage signal
    double percentile = 25.0,    // Flag top N% by score
    size_t min_bin_size = 10
) {
    CleanupResult result;
    size_t n = contigs.size();

    if (n < min_bin_size) {
        result.scores.resize(n, 0.0);
        result.threshold_used = 1.0;
        result.n_flagged = 0;
        return result;
    }

    // Step 1: Compute TNF similarity matrix
    std::vector<std::vector<double>> tnf_sim(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            double s = cosine_similarity(contigs[i].tnf, contigs[j].tnf);
            tnf_sim[i][j] = tnf_sim[j][i] = s;
        }
    }

    // Step 2: Compute CGR distance matrix
    std::vector<std::vector<double>> cgr_dist(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            double d = euclidean_distance(contigs[i].cgr, contigs[j].cgr);
            cgr_dist[i][j] = cgr_dist[j][i] = d;
        }
    }

    // Step 3: Identify core contigs (top 50% by mean TNF similarity)
    std::vector<double> mean_tnf_sim(n);
    for (size_t i = 0; i < n; ++i) {
        mean_tnf_sim[i] = std::accumulate(tnf_sim[i].begin(), tnf_sim[i].end(), 0.0) / n;
    }

    std::vector<size_t> sorted_by_sim(n);
    std::iota(sorted_by_sim.begin(), sorted_by_sim.end(), 0);
    std::sort(sorted_by_sim.begin(), sorted_by_sim.end(),
              [&](size_t a, size_t b) { return mean_tnf_sim[a] > mean_tnf_sim[b]; });

    size_t core_size = n / 2;
    if (core_size < 5) core_size = std::min(n, size_t(5));
    std::set<size_t> core_set(sorted_by_sim.begin(), sorted_by_sim.begin() + core_size);

    // Step 4: Compute core-based metrics
    std::vector<double> tnf_core_sim(n);
    std::vector<double> cgr_core_dist(n);

    for (size_t i = 0; i < n; ++i) {
        double tnf_sum = 0, cgr_sum = 0;
        for (size_t c : core_set) {
            tnf_sum += tnf_sim[i][c];
            cgr_sum += cgr_dist[i][c];
        }
        tnf_core_sim[i] = tnf_sum / core_size;
        cgr_core_dist[i] = cgr_sum / core_size;
    }

    // Step 5: Compute coverage z-scores
    std::vector<double> coverages(n);
    for (size_t i = 0; i < n; ++i) {
        coverages[i] = contigs[i].coverage;
    }
    std::vector<double> cov_sorted = coverages;
    std::sort(cov_sorted.begin(), cov_sorted.end());
    double cov_median = cov_sorted[n / 2];

    std::vector<double> cov_devs(n);
    for (size_t i = 0; i < n; ++i) {
        cov_devs[i] = std::abs(coverages[i] - cov_median);
    }
    std::sort(cov_devs.begin(), cov_devs.end());
    double cov_mad = cov_devs[n / 2] * 1.4826;
    if (cov_mad < 0.1) cov_mad = 0.1;  // Avoid division by zero

    std::vector<double> cov_zscore(n);
    for (size_t i = 0; i < n; ++i) {
        cov_zscore[i] = std::abs(coverages[i] - cov_median) / cov_mad;
    }

    // Step 6: Normalize CGR distances
    double cgr_max = *std::max_element(cgr_core_dist.begin(), cgr_core_dist.end());
    if (cgr_max < 0.001) cgr_max = 0.001;

    std::vector<double> cgr_norm(n);
    for (size_t i = 0; i < n; ++i) {
        cgr_norm[i] = cgr_core_dist[i] / cgr_max;
    }

    // Step 7: Compute combined scores
    result.scores.resize(n);
    for (size_t i = 0; i < n; ++i) {
        result.scores[i] = (1.0 - tnf_core_sim[i]) + alpha_cgr * cgr_norm[i] + beta_cov * cov_zscore[i];
    }

    // Store medians for logging
    std::vector<double> tnf_sorted = tnf_core_sim;
    std::sort(tnf_sorted.begin(), tnf_sorted.end());
    result.median_tnf_core_sim = tnf_sorted[n / 2];

    std::vector<double> cgr_sorted = cgr_core_dist;
    std::sort(cgr_sorted.begin(), cgr_sorted.end());
    result.median_cgr_core_dist = cgr_sorted[n / 2];

    std::vector<double> zscore_sorted = cov_zscore;
    std::sort(zscore_sorted.begin(), zscore_sorted.end());
    result.median_cov_zscore = zscore_sorted[n / 2];

    // Step 8: Determine threshold and flag contigs
    std::vector<double> scores_sorted = result.scores;
    std::sort(scores_sorted.begin(), scores_sorted.end());
    size_t threshold_idx = static_cast<size_t>(n * (100.0 - percentile) / 100.0);
    result.threshold_used = scores_sorted[threshold_idx];

    for (size_t i = 0; i < n; ++i) {
        if (result.scores[i] >= result.threshold_used) {
            result.contigs_to_remove.push_back(contigs[i].idx);
        }
    }

    result.n_flagged = result.contigs_to_remove.size();

    return result;
}

// =============================================================================
// Adaptive threshold version (using MAD)
// =============================================================================

inline CleanupResult compute_contamination_scores_adaptive(
    const std::vector<ContigSignals>& contigs,
    double alpha_cgr = 0.3,
    double beta_cov = 0.3,
    double mad_threshold = 2.5,  // Flag contigs > N MAD from median score
    size_t min_bin_size = 10,
    double max_removal_frac = 0.25  // Safety: don't remove more than 25%
) {
    // First compute scores with a dummy percentile
    auto result = compute_contamination_scores(contigs, alpha_cgr, beta_cov, 100.0, min_bin_size);

    if (contigs.size() < min_bin_size) {
        return result;
    }

    // Compute MAD-based threshold
    size_t n = contigs.size();
    std::vector<double> scores_sorted = result.scores;
    std::sort(scores_sorted.begin(), scores_sorted.end());
    double median = scores_sorted[n / 2];

    std::vector<double> devs(n);
    for (size_t i = 0; i < n; ++i) {
        devs[i] = std::abs(result.scores[i] - median);
    }
    std::sort(devs.begin(), devs.end());
    double mad = devs[n / 2] * 1.4826;
    if (mad < 0.001) mad = 0.001;

    double threshold = median + mad_threshold * mad;
    result.threshold_used = threshold;

    // Re-flag contigs with adaptive threshold
    result.contigs_to_remove.clear();
    for (size_t i = 0; i < n; ++i) {
        if (result.scores[i] >= threshold) {
            result.contigs_to_remove.push_back(contigs[i].idx);
        }
    }

    // Safety: limit removal fraction
    size_t max_remove = static_cast<size_t>(n * max_removal_frac);
    if (result.contigs_to_remove.size() > max_remove) {
        // Sort by score and keep only top max_remove
        std::vector<std::pair<double, size_t>> score_idx;
        for (size_t i = 0; i < n; ++i) {
            score_idx.push_back({result.scores[i], contigs[i].idx});
        }
        std::sort(score_idx.begin(), score_idx.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        result.contigs_to_remove.clear();
        for (size_t i = 0; i < max_remove; ++i) {
            result.contigs_to_remove.push_back(score_idx[i].second);
        }
        result.threshold_used = score_idx[max_remove - 1].first;
    }

    result.n_flagged = result.contigs_to_remove.size();
    return result;
}

}  // namespace multisignal
}  // namespace amber

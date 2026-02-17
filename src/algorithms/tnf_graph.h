#pragma once
// ==============================================================================
// TNF-based Intra-Bin Connectivity Graph
//
// Uses tetranucleotide frequency (TNF) similarity to build a connectivity graph
// within each bin. Contigs with low connectivity to the bin core are potential
// contaminants.
//
// This is more robust than exact k-mer sharing because:
// 1. Works on assembled contigs (no overlapping sequences needed)
// 2. Captures compositional similarity which is what binning uses
// 3. Computationally simpler (O(n²) pairwise comparisons)
//
// Algorithm:
//   1. Compute TNF for each contig in bin (already done in binning)
//   2. Build similarity matrix: cos_sim(TNF_i, TNF_j) for all pairs
//   3. Build graph: edge if similarity > threshold
//   4. Compute connectivity: degree, PageRank, or weighted centrality
//   5. Flag contigs with low connectivity relative to bin median
// ==============================================================================

#include <array>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

namespace amber {
namespace tnf_graph {

constexpr int TNF_DIM = 136;  // Canonical 4-mers

// =============================================================================
// TNF Similarity
// =============================================================================

// Cosine similarity between two TNF vectors
inline double cosine_similarity(const std::array<float, TNF_DIM>& a,
                                const std::array<float, TNF_DIM>& b) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < TNF_DIM; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a < 1e-10 || norm_b < 1e-10) return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// Euclidean distance between two TNF vectors (CLR-transformed)
inline double euclidean_distance(const std::array<float, TNF_DIM>& a,
                                 const std::array<float, TNF_DIM>& b) {
    double sum = 0.0;
    for (int i = 0; i < TNF_DIM; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

// =============================================================================
// Connectivity Graph
// =============================================================================

struct ContigTNFConnectivity {
    size_t contig_idx;
    int n_neighbors;              // Number of contigs with similarity > threshold
    double mean_similarity;       // Mean similarity to all other contigs
    double weighted_degree;       // Sum of similarities (edge weights)
    double core_similarity;       // Mean similarity to top 50% most connected
    double connectivity_score;    // Normalized score [0, 1]
    double dist_to_centroid;      // Distance to bin TNF centroid
};

// Build TNF connectivity graph for a set of contigs
inline std::vector<ContigTNFConnectivity> compute_tnf_connectivity(
    const std::vector<std::array<float, TNF_DIM>>& tnf_vectors,
    const std::vector<size_t>& contig_indices,
    double similarity_threshold = 0.95  // High threshold for "connected"
) {
    size_t n = tnf_vectors.size();
    std::vector<ContigTNFConnectivity> results(n);

    if (n < 2) {
        for (size_t i = 0; i < n; ++i) {
            results[i] = {contig_indices[i], 0, 1.0, 0.0, 1.0, 1.0, 0.0};
        }
        return results;
    }

    // Step 1: Compute pairwise similarities
    std::vector<std::vector<double>> sim_matrix(n, std::vector<double>(n, 0.0));

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double sim = cosine_similarity(tnf_vectors[i], tnf_vectors[j]);
            sim_matrix[i][j] = sim;
            sim_matrix[j][i] = sim;
        }
        sim_matrix[i][i] = 1.0;
    }

    // Step 2: Compute bin centroid
    std::array<float, TNF_DIM> centroid = {};
    for (size_t i = 0; i < n; ++i) {
        for (int k = 0; k < TNF_DIM; ++k) {
            centroid[k] += tnf_vectors[i][k];
        }
    }
    for (int k = 0; k < TNF_DIM; ++k) {
        centroid[k] /= n;
    }

    // Step 3: Compute connectivity metrics for each contig
    for (size_t i = 0; i < n; ++i) {
        results[i].contig_idx = contig_indices[i];

        // Count neighbors and compute weighted degree
        int neighbors = 0;
        double weighted_degree = 0.0;
        double total_sim = 0.0;

        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            double sim = sim_matrix[i][j];
            total_sim += sim;
            weighted_degree += sim;
            if (sim >= similarity_threshold) {
                neighbors++;
            }
        }

        results[i].n_neighbors = neighbors;
        results[i].weighted_degree = weighted_degree;
        results[i].mean_similarity = total_sim / (n - 1);
        results[i].dist_to_centroid = euclidean_distance(tnf_vectors[i], centroid);
    }

    // Step 4: Identify core contigs (top 50% by weighted degree)
    std::vector<size_t> sorted_by_degree(n);
    std::iota(sorted_by_degree.begin(), sorted_by_degree.end(), 0);
    std::sort(sorted_by_degree.begin(), sorted_by_degree.end(),
              [&](size_t a, size_t b) {
                  return results[a].weighted_degree > results[b].weighted_degree;
              });

    size_t core_size = n / 2;
    if (core_size < 5) core_size = std::min(n, size_t(5));

    std::vector<size_t> core_indices(sorted_by_degree.begin(),
                                      sorted_by_degree.begin() + core_size);

    // Step 5: Compute core similarity for each contig
    for (size_t i = 0; i < n; ++i) {
        double core_sim = 0.0;
        for (size_t j : core_indices) {
            if (i != j) {
                core_sim += sim_matrix[i][j];
            }
        }
        results[i].core_similarity = core_sim / core_size;
    }

    // Step 6: Compute final connectivity score
    // Score = mean similarity to core * (1 - normalized distance to centroid)
    double max_dist = 0.0;
    for (const auto& r : results) {
        max_dist = std::max(max_dist, r.dist_to_centroid);
    }

    for (auto& r : results) {
        double norm_dist = (max_dist > 0) ? r.dist_to_centroid / max_dist : 0.0;
        r.connectivity_score = r.core_similarity * (1.0 - 0.5 * norm_dist);
    }

    return results;
}

// =============================================================================
// Contamination Detection
// =============================================================================

struct TNFRefinementResult {
    std::vector<size_t> contigs_to_remove;
    std::vector<ContigTNFConnectivity> all_connectivity;
    int total_contigs;
    int flagged_contigs;
    double median_core_similarity;
    double threshold_used;
};

// Detect contigs with low connectivity (potential contaminants)
inline TNFRefinementResult refine_bin_tnf(
    const std::vector<std::array<float, TNF_DIM>>& tnf_vectors,
    const std::vector<size_t>& contig_indices,
    double similarity_threshold = 0.95,
    double outlier_factor = 0.7,  // Flag if core_similarity < median * factor
    size_t min_bin_size = 10
) {
    TNFRefinementResult result;
    result.total_contigs = static_cast<int>(tnf_vectors.size());
    result.flagged_contigs = 0;
    result.median_core_similarity = 1.0;
    result.threshold_used = 0.0;

    if (tnf_vectors.size() < min_bin_size) {
        return result;
    }

    // Compute connectivity
    result.all_connectivity = compute_tnf_connectivity(
        tnf_vectors, contig_indices, similarity_threshold
    );

    // Compute median core similarity
    std::vector<double> core_sims;
    for (const auto& c : result.all_connectivity) {
        core_sims.push_back(c.core_similarity);
    }
    std::sort(core_sims.begin(), core_sims.end());
    result.median_core_similarity = core_sims[core_sims.size() / 2];

    // Adaptive threshold based on median
    result.threshold_used = result.median_core_similarity * outlier_factor;

    // Flag outliers
    for (const auto& c : result.all_connectivity) {
        if (c.core_similarity < result.threshold_used) {
            result.contigs_to_remove.push_back(c.contig_idx);
        }
    }

    result.flagged_contigs = static_cast<int>(result.contigs_to_remove.size());
    return result;
}

}  // namespace tnf_graph
}  // namespace amber

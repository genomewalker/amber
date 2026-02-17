// Intrinsic Quality Metrics for Bin Assessment
//
// Functions for computing intrinsic quality metrics used in contamination
// detection and completeness estimation:
//   - CGR (Chaos Game Representation) occupancy
//   - De Bruijn graph modularity
//   - Jensen-Shannon divergence distance

#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cctype>
#include "../../algorithms/fractal_features.h"

namespace amber {

// Combined intrinsic quality assessment for a bin
struct IntrinsicQuality {
    // Primary metrics (validated on real ancient DNA bins vs CheckM2)
    double downsample_var = 0.0;      // var_80%: Pearson r=+0.528 with contamination
    double cgr_occupancy = 0.0;       // Pearson r=+0.368 with contamination

    // Legacy metrics (not used in decisions; kept for logging)
    double mean_jsd = 1.0;            // Poor correlation (r=-0.02); not used
    double debruijn_modularity = -1.0;

    // Estimated quality [0, 1]
    double completeness_est = 0.0;
    double contamination_est = 0.0;

    // Decision flags
    bool is_likely_clean = false;
    bool is_likely_complete = false;
    bool needs_splitting = false;
};

// Result struct for downsample variance computation
struct DownsampleVarResult {
    double var_80 = 0.0;           // Variance of centroid under 80% subsampling
    double mean_jsd = 1.0;         // Mean JSD (kept for comparison)
    int n_contigs = 0;
    bool valid = false;
};

// JSD distance for quality metrics (sqrt of Jensen-Shannon divergence)
inline double jsd_distance_static(const std::vector<double>& p, const std::vector<double>& q) {
    if (p.size() != q.size() || p.empty()) return 1.0;
    double h_m = 0, h_p = 0, h_q = 0;
    for (size_t i = 0; i < p.size(); i++) {
        double m = (p[i] + q[i]) / 2.0;
        if (m > 0) h_m -= m * std::log(m);
        if (p[i] > 0) h_p -= p[i] * std::log(p[i]);
        if (q[i] > 0) h_q -= q[i] * std::log(q[i]);
    }
    return std::sqrt(std::max(0.0, h_m - (h_p + h_q) / 2.0));
}

// CGR occupancy at k=10 for completeness estimation
// Higher occupancy indicates more complete genome coverage
inline double compute_cgr_occupancy(const std::string& sequence, int k = 10) {
    if (sequence.length() < 1000) return 0.0;

    // CGR grid resolution: 2^k x 2^k
    const int RESOLUTION = 1 << k;  // 1024 for k=10
    const size_t GRID_SIZE = (size_t)RESOLUTION * RESOLUTION;

    // Use bitset for memory efficiency
    std::vector<bool> occupied(GRID_SIZE, false);

    // Walk CGR path and mark occupied cells
    double x = 0.5, y = 0.5;
    for (char c : sequence) {
        switch (std::toupper(c)) {
            case 'A': x = x / 2.0;       y = y / 2.0;       break;  // Bottom-left
            case 'C': x = x / 2.0;       y = (y + 1.0) / 2.0; break;  // Top-left
            case 'G': x = (x + 1.0) / 2.0; y = (y + 1.0) / 2.0; break;  // Top-right
            case 'T': x = (x + 1.0) / 2.0; y = y / 2.0;       break;  // Bottom-right
            default: continue;
        }

        int cx = std::min((int)(x * RESOLUTION), RESOLUTION - 1);
        int cy = std::min((int)(y * RESOLUTION), RESOLUTION - 1);
        occupied[(size_t)cy * RESOLUTION + cx] = true;
    }

    // Count occupied cells
    size_t n_occupied = std::count(occupied.begin(), occupied.end(), true);
    return (double)n_occupied / GRID_SIZE;
}

// De Bruijn Graph Modularity for contamination detection
// Correlation with CheckM2 contamination: r=0.70 (STRONG)
// Clean bins: modularity ~ -1.0, Contaminated: modularity > -0.8
inline double compute_debruijn_modularity(const std::string& sequence, int k = 4) {
    if (sequence.length() < 1000) return -1.0;

    // Use FractalFeatures library for graph topology
    auto graph = FractalFeatures::extract_graph_topology(sequence, k);
    return graph.modularity;
}

}  // namespace amber

#pragma once
// ==============================================================================
// Strobemer-Lacunarity Features for Metagenomic Binning
//
// Novel fractal-based features that capture the spatial organization of k-mers,
// complementing frequency-based features (TNF) with structural information.
//
// Key insight: Different genomes have different "clumpiness" patterns in their
// k-mer spacing, even when k-mer frequencies are nearly identical (near-strains).
//
// Features computed:
//   1. Strobemer extraction (linked k-mer pairs at variable distances)
//   2. Multi-scale lacunarity of strobemer positions
//   3. Lacunarity slope (fractal dimension proxy)
//   4. Strobemer spacing statistics (entropy, variance)
//
// Total: 12 dimensions for binning, computed per-contig
// ==============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace amber {

// =============================================================================
// Strobemer Parameters
// =============================================================================

struct StrobemerParams {
    int k = 15;          // Strobe size (each k-mer in the pair)
    int w_min = 20;      // Minimum window offset for second strobe
    int w_max = 50;      // Maximum window offset for second strobe
    int order = 2;       // Order-2 strobemers (pairs)

    static constexpr int DEFAULT_K = 15;
    static constexpr int DEFAULT_W_MIN = 20;
    static constexpr int DEFAULT_W_MAX = 50;
};

// =============================================================================
// Strobemer-Lacunarity Feature Vector (12 dimensions)
// =============================================================================

struct StrobemerLacunarityFeatures {
    // Multi-scale lacunarity (5 scales: 100, 200, 500, 1000, 2000 bp)
    std::array<double, 5> lacunarity_scales = {};

    // Lacunarity slope (fractal dimension proxy)
    double lacunarity_slope = 0.0;
    double lacunarity_intercept = 0.0;

    // Strobemer spacing statistics
    double spacing_mean = 0.0;
    double spacing_cv = 0.0;      // Coefficient of variation
    double spacing_entropy = 0.0; // Shannon entropy of spacing distribution

    // Strobemer density
    double density = 0.0;         // Strobemers per kb

    // Lacunarity fit quality (R² of log-log regression) - used as reliability weight
    double fit_r_squared = 0.0;

    // Full vector (for cleanup compatibility)
    std::vector<double> to_vector() const {
        return {
            lacunarity_scales[0], lacunarity_scales[1], lacunarity_scales[2],
            lacunarity_scales[3], lacunarity_scales[4],
            lacunarity_slope, lacunarity_intercept,
            spacing_mean, spacing_cv, spacing_entropy,
            density, fit_r_squared
        };
    }

    // Reduced vector for EM clustering (6 dims, no redundancy)
    // Keeps: scale[0], scale[2], scale[4], slope, spacing_cv, spacing_entropy
    std::vector<double> to_em_vector() const {
        return {
            lacunarity_scales[0],  // 100bp scale
            lacunarity_scales[2],  // 500bp scale
            lacunarity_scales[4],  // 2000bp scale
            lacunarity_slope,      // fractal dimension proxy
            spacing_cv,            // spacing variability
            spacing_entropy        // spacing distribution entropy
        };
    }

    // Reliability weight for EM (based on R² and valid scales)
    double reliability() const {
        // Combine fit quality with scale validity
        int valid_scales = 0;
        for (double s : lacunarity_scales) {
            if (s > 0 && std::isfinite(s)) valid_scales++;
        }
        double scale_frac = valid_scales / 5.0;
        return std::max(0.0, std::min(1.0, fit_r_squared * scale_frac));
    }

    static constexpr int DIMS = 12;
    static constexpr int EM_DIMS = 6;
};

// =============================================================================
// Fast Hash Functions
// =============================================================================

inline uint64_t hash_kmer_fast(const char* seq, int k) {
    uint64_t h = 0;
    for (int i = 0; i < k; i++) {
        uint64_t base = 0;
        switch (seq[i]) {
            case 'A': case 'a': base = 0; break;
            case 'C': case 'c': base = 1; break;
            case 'G': case 'g': base = 2; break;
            case 'T': case 't': base = 3; break;
            default: return UINT64_MAX; // Invalid base
        }
        h = (h << 2) | base;
    }
    return h;
}

inline uint64_t canonical_kmer_hash(const char* seq, int k) {
    uint64_t fwd = hash_kmer_fast(seq, k);
    if (fwd == UINT64_MAX) return UINT64_MAX;

    // Reverse complement
    uint64_t rev = 0;
    for (int i = k - 1; i >= 0; i--) {
        uint64_t base = 0;
        switch (seq[i]) {
            case 'A': case 'a': base = 3; break; // T
            case 'C': case 'c': base = 2; break; // G
            case 'G': case 'g': base = 1; break; // C
            case 'T': case 't': base = 0; break; // A
            default: return UINT64_MAX;
        }
        rev = (rev << 2) | base;
    }
    return std::min(fwd, rev);
}

// Mix two hashes to form strobemer hash
inline uint64_t combine_hashes(uint64_t h1, uint64_t h2) {
    // Use a mixing function
    uint64_t h = h1 ^ (h2 * 0x9e3779b97f4a7c15ULL);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h;
}

// =============================================================================
// Strobemer Extraction (Randstrobes - order 2)
// =============================================================================

struct Strobemer {
    uint64_t hash;
    size_t position;  // Position of first strobe
};

inline std::vector<Strobemer> extract_strobemers(
    const std::string& seq,
    const StrobemerParams& params = {}) {

    std::vector<Strobemer> result;

    int k = params.k;
    int w_min = params.w_min;
    int w_max = params.w_max;

    if ((int)seq.length() < k + w_max + k) return result;

    size_t max_pos = seq.length() - k - w_max - k + 1;
    result.reserve(max_pos);

    for (size_t i = 0; i < max_pos; i++) {
        // First strobe
        uint64_t h1 = canonical_kmer_hash(seq.c_str() + i, k);
        if (h1 == UINT64_MAX) continue;

        // Find second strobe (minimize hash in window)
        uint64_t min_h2 = UINT64_MAX;
        for (int offset = w_min; offset <= w_max; offset++) {
            size_t pos2 = i + k + offset;
            if (pos2 + k > seq.length()) break;

            uint64_t h2 = canonical_kmer_hash(seq.c_str() + pos2, k);
            if (h2 < min_h2) {
                min_h2 = h2;
            }
        }

        if (min_h2 == UINT64_MAX) continue;

        // Combine to form strobemer
        uint64_t strobe_hash = combine_hashes(h1, min_h2);
        result.push_back({strobe_hash, i});
    }

    return result;
}

// =============================================================================
// Lacunarity Computation (Gliding Box Method)
// =============================================================================

// Compute lacunarity at a given box size r for a set of positions
inline double compute_lacunarity(const std::vector<size_t>& positions, size_t seq_len, size_t r) {
    if (positions.empty() || r == 0 || seq_len < r) return 1.0;

    // Count how many positions fall in each box
    size_t n_boxes = (seq_len - r) + 1;
    if (n_boxes == 0) return 1.0;

    // For efficiency, use a sliding window approach
    std::vector<int> counts(n_boxes, 0);

    // Sort positions
    std::vector<size_t> sorted_pos = positions;
    std::sort(sorted_pos.begin(), sorted_pos.end());

    // For each box starting position, count points inside
    size_t pos_idx = 0;
    for (size_t box_start = 0; box_start < n_boxes; box_start++) {
        size_t box_end = box_start + r;

        // Move pos_idx to first position >= box_start
        while (pos_idx < sorted_pos.size() && sorted_pos[pos_idx] < box_start) {
            pos_idx++;
        }

        // Count positions in [box_start, box_end)
        size_t temp_idx = pos_idx;
        while (temp_idx < sorted_pos.size() && sorted_pos[temp_idx] < box_end) {
            counts[box_start]++;
            temp_idx++;
        }
    }

    // Compute moments
    double sum = 0, sum_sq = 0;
    for (int c : counts) {
        sum += c;
        sum_sq += c * c;
    }

    double mean = sum / n_boxes;
    double mean_sq = sum_sq / n_boxes;

    if (mean < 1e-10) return 1.0;

    // Lacunarity = <n^2> / <n>^2
    return mean_sq / (mean * mean);
}

// =============================================================================
// Multi-scale Lacunarity Analysis
// =============================================================================

inline StrobemerLacunarityFeatures extract_strobemer_lacunarity(
    const std::string& seq,
    const StrobemerParams& params = {}) {

    StrobemerLacunarityFeatures features;

    if (seq.length() < 3000) return features; // Too short for meaningful analysis

    // Extract strobemers
    auto strobemers = extract_strobemers(seq, params);

    if (strobemers.size() < 20) return features; // Not enough strobemers

    // Get positions
    std::vector<size_t> positions;
    positions.reserve(strobemers.size());
    for (const auto& s : strobemers) {
        positions.push_back(s.position);
    }

    // Compute density
    features.density = 1000.0 * strobemers.size() / seq.length();

    // Multi-scale lacunarity (5 scales)
    std::array<size_t, 5> scales = {100, 200, 500, 1000, 2000};
    std::vector<double> log_r, log_lac;

    for (size_t i = 0; i < scales.size(); i++) {
        if (scales[i] < seq.length()) {
            features.lacunarity_scales[i] = compute_lacunarity(positions, seq.length(), scales[i]);
            log_r.push_back(std::log(scales[i]));
            log_lac.push_back(std::log(features.lacunarity_scales[i]));
        }
    }

    // Compute lacunarity slope (linear regression in log-log space) + R²
    if (log_r.size() >= 3) {
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0, sum_yy = 0;
        size_t n = log_r.size();

        for (size_t i = 0; i < n; i++) {
            sum_x += log_r[i];
            sum_y += log_lac[i];
            sum_xy += log_r[i] * log_lac[i];
            sum_xx += log_r[i] * log_r[i];
            sum_yy += log_lac[i] * log_lac[i];
        }

        double denom = n * sum_xx - sum_x * sum_x;
        if (std::abs(denom) > 1e-10) {
            features.lacunarity_slope = (n * sum_xy - sum_x * sum_y) / denom;
            features.lacunarity_intercept = (sum_y - features.lacunarity_slope * sum_x) / n;

            // Compute R² (coefficient of determination)
            double ss_tot = sum_yy - (sum_y * sum_y) / n;
            double ss_res = 0;
            double mean_y = sum_y / n;
            for (size_t i = 0; i < n; i++) {
                double y_pred = features.lacunarity_slope * log_r[i] + features.lacunarity_intercept;
                double residual = log_lac[i] - y_pred;
                ss_res += residual * residual;
            }
            if (ss_tot > 1e-10) {
                features.fit_r_squared = 1.0 - (ss_res / ss_tot);
                features.fit_r_squared = std::max(0.0, std::min(1.0, features.fit_r_squared));
            }
        }
    }

    // Compute spacing statistics
    std::sort(positions.begin(), positions.end());
    std::vector<double> spacings;
    spacings.reserve(positions.size() - 1);

    for (size_t i = 1; i < positions.size(); i++) {
        spacings.push_back(positions[i] - positions[i-1]);
    }

    if (!spacings.empty()) {
        // Mean spacing
        double sum = std::accumulate(spacings.begin(), spacings.end(), 0.0);
        features.spacing_mean = sum / spacings.size();

        // Variance and CV
        double sq_sum = 0;
        for (double s : spacings) {
            sq_sum += (s - features.spacing_mean) * (s - features.spacing_mean);
        }
        double variance = sq_sum / spacings.size();
        double std_dev = std::sqrt(variance);
        features.spacing_cv = (features.spacing_mean > 0) ? std_dev / features.spacing_mean : 0;

        // Entropy of spacing distribution (binned)
        const int N_BINS = 20;
        double max_spacing = *std::max_element(spacings.begin(), spacings.end());
        std::vector<int> bins(N_BINS, 0);

        for (double s : spacings) {
            int bin = std::min((int)(N_BINS * s / (max_spacing + 1)), N_BINS - 1);
            bins[bin]++;
        }

        features.spacing_entropy = 0;
        for (int count : bins) {
            if (count > 0) {
                double p = (double)count / spacings.size();
                features.spacing_entropy -= p * std::log2(p);
            }
        }
    }

    return features;
}

// =============================================================================
// Outlier Detection Using Strobemer-Lacunarity Features
// =============================================================================

struct StrobeLacunarityOutlierResult {
    std::vector<size_t> outlier_indices;
    std::vector<double> distances;  // Distance from centroid
    double median_distance = 0.0;
    double mad_distance = 0.0;
};

inline StrobeLacunarityOutlierResult find_strobelac_outliers(
    const std::vector<StrobemerLacunarityFeatures>& features,
    double z_threshold = 2.5,
    size_t min_contigs = 10) {

    StrobeLacunarityOutlierResult result;
    size_t n = features.size();

    if (n < min_contigs) return result;

    // Compute centroid
    std::vector<double> centroid(StrobemerLacunarityFeatures::DIMS, 0.0);
    int valid_count = 0;

    for (const auto& f : features) {
        auto vec = f.to_vector();
        bool valid = true;
        for (double v : vec) {
            if (!std::isfinite(v)) { valid = false; break; }
        }
        if (valid) {
            for (size_t d = 0; d < vec.size(); d++) {
                centroid[d] += vec[d];
            }
            valid_count++;
        }
    }

    if (valid_count < (int)min_contigs) return result;

    for (double& c : centroid) c /= valid_count;

    // Compute distances from centroid
    result.distances.resize(n, 0.0);

    for (size_t i = 0; i < n; i++) {
        auto vec = features[i].to_vector();
        double dist = 0;
        for (size_t d = 0; d < vec.size(); d++) {
            if (std::isfinite(vec[d]) && std::isfinite(centroid[d])) {
                double diff = vec[d] - centroid[d];
                dist += diff * diff;
            }
        }
        result.distances[i] = std::sqrt(dist);
    }

    // Compute median and MAD
    std::vector<double> sorted_dists = result.distances;
    std::sort(sorted_dists.begin(), sorted_dists.end());
    result.median_distance = sorted_dists[n / 2];

    std::vector<double> abs_devs;
    for (double d : result.distances) {
        abs_devs.push_back(std::abs(d - result.median_distance));
    }
    std::sort(abs_devs.begin(), abs_devs.end());
    result.mad_distance = abs_devs[n / 2];

    // Identify outliers
    if (result.mad_distance > 1e-6) {
        for (size_t i = 0; i < n; i++) {
            double z = (result.distances[i] - result.median_distance) / result.mad_distance;
            if (z > z_threshold) {
                result.outlier_indices.push_back(i);
            }
        }
    }

    return result;
}

}  // namespace amber

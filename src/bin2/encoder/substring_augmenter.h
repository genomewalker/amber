// AMBER bin2 - Substring Augmentation (COMEBin-compatible)
// Generates random subsequences and computes TNF features for contrastive learning
//
// COMEBin augmentation (data_aug/gen_seq.py):
// - n_views = 6 (default): original + 5 random substrings
// - Substring length: random in [min_length, contig_length]
// - TNF computed on each substring independently
// - Coverage computed from depth slice for each substring
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "../../algorithms/kmer_features.h"
#include "multiscale_cgr.h"

namespace amber::bin2 {

// COMEBin-style substring augmentation
class SubstringAugmenter {
public:
    struct Config {
        int n_views = 6;           // COMEBin default: 6 views (original + 5 augmented)
        int min_length = 1000;     // Minimum substring length (COMEBin default)
        unsigned int seed = 42;    // Random seed for reproducibility
        std::string mode = "random";  // "random" (COMEBin) or "fractal" (experimental)
        int fractal_candidates = 8;   // Number of random candidates scored per view
        int fractal_min_length = 500; // Min length for fractal feature extraction
        float fractal_dist_weight = 1.0f;
        float fractal_ratio_weight = 0.35f;
    };

    SubstringAugmenter() : config_() {}

    explicit SubstringAugmenter(const Config& config)
        : config_(config) {}

    // Generate augmented TNF features for a single contig
    // Returns n_views feature vectors (first is original, rest are random substrings)
    // Uses COMEBin normalization: (raw + 1) / sum
    std::vector<std::array<float, 136>> generate_views(const std::string& sequence) {
        auto all = generate_all_views(std::vector<std::string>{sequence});
        std::vector<std::array<float, 136>> views(config_.n_views);
        for (int v = 0; v < config_.n_views; ++v) {
            views[v] = all[v][0];
        }
        return views;
    }

    // Generate augmented features for all contigs
    // Returns: views[view_idx][contig_idx] = 136-dim TNF vector
    //
    // COMEBin compatibility: Uses sequential global RNG like Python's random module.
    // This means contig order affects augmentation (same as COMEBin).
    std::vector<std::vector<std::array<float, 136>>> generate_all_views(
        const std::vector<std::string>& sequences
    ) {
        const int N = static_cast<int>(sequences.size());

        // Initialize views: n_views x N x 136
        std::vector<std::vector<std::array<float, 136>>> views(config_.n_views);
        for (int v = 0; v < config_.n_views; ++v) {
            views[v].resize(N);
        }

        const auto& substring_params = get_or_compute_params(sequences);

        // Now compute TNF in parallel (order doesn't matter for TNF computation)
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N; ++i) {
            const std::string& seq = sequences[i];

            // View 0: Original with COMEBin normalization
            extract_tnf_comebin(seq, views[0][i]);

            // Views 1+: Use pre-computed substring parameters
            for (int v = 1; v < config_.n_views; ++v) {
                auto [start, sub_len] = substring_params[i][v-1];
                if (start < 0) {
                    views[v][i] = views[0][i];  // Copy original if too short
                } else {
                    std::string substring = seq.substr(start, sub_len);
                    extract_tnf_comebin(substring, views[v][i]);
                }
            }
        }

        return views;
    }

    // Generate augmentation info for coverage recomputation
    // Returns: substring_info[view_idx][contig_idx] = (start, end) or (-1, -1) for full contig
    struct SubstringInfo {
        int start;  // -1 means full contig
        int end;    // -1 means full contig
    };

    std::vector<std::vector<SubstringInfo>> generate_substring_info(
        const std::vector<std::string>& sequences
    ) {
        const int N = static_cast<int>(sequences.size());

        // Initialize info: n_views x N
        std::vector<std::vector<SubstringInfo>> info(config_.n_views);
        for (int v = 0; v < config_.n_views; ++v) {
            info[v].resize(N, {-1, -1});  // Default: full contig
        }

        const auto& substring_params = get_or_compute_params(sequences);
        for (int i = 0; i < N; ++i) {
            for (int v = 1; v < config_.n_views; ++v) {
                auto [start, sub_len] = substring_params[i][v - 1];
                if (start < 0) {
                    info[v][i] = {-1, -1};
                } else {
                    info[v][i] = {start, start + sub_len};
                }
            }
        }

        return info;
    }

private:
    bool use_fractal_mode() const {
        return config_.mode == "fractal";
    }

    std::pair<int, int> sample_random_substring(int seq_len, std::mt19937& rng) const {
        if (seq_len <= config_.min_length + 1) return {-1, -1};
        std::uniform_int_distribution<int> start_dist(0, seq_len - config_.min_length - 1);
        const int start = start_dist(rng);
        std::uniform_int_distribution<int> len_dist(config_.min_length, seq_len - start);
        const int sub_len = len_dist(rng);
        return {start, sub_len};
    }

    static float fractal_distance(
        const MultiScaleCGRFeatures& a,
        const MultiScaleCGRFeatures& b) {
        const auto va = a.to_vector();
        const auto vb = b.to_vector();
        float sum = 0.0f;
        for (size_t i = 0; i < va.size(); ++i) {
            sum += std::fabs(va[i] - vb[i]);
        }
        return sum;
    }

    std::pair<int, int> choose_fractal_substring(
        const std::string& seq,
        int view_idx,
        const MultiScaleCGRFeatures& full_sig,
        std::mt19937& rng) const {

        const int seq_len = static_cast<int>(seq.length());
        if (seq_len <= config_.min_length + 1) return {-1, -1};

        const int aug_views = std::max(1, config_.n_views - 1);
        const float t = (aug_views == 1) ? 0.0f :
            static_cast<float>(view_idx - 1) / static_cast<float>(aug_views - 1);
        const float min_ratio = std::max(
            static_cast<float>(config_.min_length) / std::max(1, seq_len),
            0.35f);
        const float target_ratio = 0.9f - t * (0.9f - min_ratio);

        int best_start = -1;
        int best_len = -1;
        float best_score = std::numeric_limits<float>::max();
        const int n_candidates = std::max(2, config_.fractal_candidates);

        for (int c = 0; c < n_candidates; ++c) {
            auto [start, sub_len] = sample_random_substring(seq_len, rng);
            if (start < 0 || sub_len <= 0) continue;

            const std::string sub = seq.substr(start, sub_len);
            auto sub_sig = MultiScaleCGRExtractor::extract(
                sub,
                std::min(config_.fractal_min_length, config_.min_length));

            const float d = fractal_distance(full_sig, sub_sig);
            const float ratio = static_cast<float>(sub_len) / std::max(1, seq_len);
            const float score =
                config_.fractal_dist_weight * d +
                config_.fractal_ratio_weight * std::fabs(ratio - target_ratio);

            if (score < best_score) {
                best_score = score;
                best_start = start;
                best_len = sub_len;
            }
        }

        if (best_start < 0 || best_len <= 0) {
            return sample_random_substring(seq_len, rng);
        }
        return {best_start, best_len};
    }

    size_t sequence_signature(const std::vector<std::string>& sequences) const {
        size_t total = 0;
        for (const auto& seq : sequences) total += seq.size();
        return total;
    }

    const std::vector<std::vector<std::pair<int, int>>>& get_or_compute_params(
        const std::vector<std::string>& sequences) {
        const size_t sig = sequence_signature(sequences);
        if (cached_valid_ &&
            cached_n_ == sequences.size() &&
            cached_total_len_ == sig) {
            return cached_params_;
        }

        const int N = static_cast<int>(sequences.size());
        cached_params_.assign(N, std::vector<std::pair<int, int>>(config_.n_views - 1, {-1, -1}));

        // COMEBin-compatible sequential RNG.
        std::mt19937 global_rng(config_.seed);

        for (int i = 0; i < N; ++i) {
            const std::string& seq = sequences[i];
            const int seq_len = static_cast<int>(seq.length());
            MultiScaleCGRFeatures full_sig;
            if (use_fractal_mode() && seq_len > config_.min_length + 1) {
                full_sig = MultiScaleCGRExtractor::extract(
                    seq,
                    std::min(config_.fractal_min_length, config_.min_length));
            }

            for (int v = 1; v < config_.n_views; ++v) {
                if (seq_len <= config_.min_length + 1) {
                    cached_params_[i][v - 1] = {-1, -1};
                    continue;
                }

                if (use_fractal_mode()) {
                    cached_params_[i][v - 1] = choose_fractal_substring(seq, v, full_sig, global_rng);
                } else {
                    cached_params_[i][v - 1] = sample_random_substring(seq_len, global_rng);
                }
            }
        }

        cached_n_ = sequences.size();
        cached_total_len_ = sig;
        cached_valid_ = true;
        return cached_params_;
    }

    Config config_;
    bool cached_valid_ = false;
    size_t cached_n_ = 0;
    size_t cached_total_len_ = 0;
    std::vector<std::vector<std::pair<int, int>>> cached_params_;
};

} // namespace amber::bin2

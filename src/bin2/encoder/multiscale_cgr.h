// AMBER bin2 - Multi-scale CGR Features
// Extract scale-transition features from CGR at multiple resolutions
// Based on OpenCode collaboration (Feb 2026)
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

namespace amber::bin2 {

// Multi-scale CGR features (9 dims by default)
// Captures scale-transition information not in single-scale TNF/CGR
struct MultiScaleCGRFeatures {
    // Scale transition slopes (how features change across resolutions)
    float dH_16_32 = 0.0f;   // Entropy slope: 16x16 -> 32x32
    float dH_32_64 = 0.0f;   // Entropy slope: 32x32 -> 64x64
    float dO_16_32 = 0.0f;   // Occupancy slope: 16x16 -> 32x32
    float dO_32_64 = 0.0f;   // Occupancy slope: 32x32 -> 64x64
    float dL_16_32 = 0.0f;   // Lacunarity slope: 16x16 -> 32x32
    float dL_32_64 = 0.0f;   // Lacunarity slope: 32x32 -> 64x64

    // Mid-scale absolute statistics (32x32)
    float H_32 = 0.0f;       // Entropy at 32x32
    float O_32 = 0.0f;       // Occupancy at 32x32
    float L_32 = 0.0f;       // Lacunarity at 32x32

    // Convert to feature vector
    std::vector<float> to_vector() const {
        return {dH_16_32, dH_32_64, dO_16_32, dO_32_64,
                dL_16_32, dL_32_64, H_32, O_32, L_32};
    }

    static constexpr int num_features() { return 9; }
};

// CGR statistics at a single resolution
struct CGRStats {
    float entropy = 0.0f;    // Shannon entropy of occupied cells
    float occupancy = 0.0f;  // Fraction of cells with >= 1 point
    float lacunarity = 0.0f; // Variance/mean^2 of cell counts (+ 1)

    static CGRStats compute(const std::vector<std::vector<int>>& grid) {
        CGRStats stats;
        int N = grid.size();
        if (N == 0) return stats;

        int total_cells = N * N;
        int occupied = 0;
        double sum = 0, sum_sq = 0;
        std::vector<int> counts;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int c = grid[i][j];
                if (c > 0) {
                    occupied++;
                    counts.push_back(c);
                }
                sum += c;
                sum_sq += c * c;
            }
        }

        // Occupancy
        stats.occupancy = (float)occupied / total_cells;

        // Entropy over occupied cells
        if (!counts.empty()) {
            double total = sum;
            double H = 0.0;
            for (int c : counts) {
                if (c > 0) {
                    double p = c / total;
                    H -= p * std::log2(p + 1e-12);
                }
            }
            // Normalize by log2(occupied) to get [0,1] range
            stats.entropy = (float)(H / std::max(1.0, std::log2((double)occupied)));
        }

        // Lacunarity
        double mean = sum / total_cells;
        double variance = (sum_sq / total_cells) - (mean * mean);
        if (mean > 1e-10) {
            stats.lacunarity = (float)(variance / (mean * mean));
        }

        return stats;
    }
};

// Extract multi-scale CGR features from a sequence
class MultiScaleCGRExtractor {
public:
    // CGR base coordinates
    static constexpr float CGR_A[2] = {0.0f, 0.0f};
    static constexpr float CGR_C[2] = {0.0f, 1.0f};
    static constexpr float CGR_G[2] = {1.0f, 1.0f};
    static constexpr float CGR_T[2] = {1.0f, 0.0f};

    // Get CGR corner for nucleotide
    static void get_corner(char c, float& x, float& y) {
        switch (c) {
            case 'A': case 'a': x = CGR_A[0]; y = CGR_A[1]; break;
            case 'C': case 'c': x = CGR_C[0]; y = CGR_C[1]; break;
            case 'G': case 'g': x = CGR_G[0]; y = CGR_G[1]; break;
            case 'T': case 't': x = CGR_T[0]; y = CGR_T[1]; break;
            default: x = 0.5f; y = 0.5f;  // Unknown -> center
        }
    }

    // Generate CGR trajectory
    static std::vector<std::pair<float, float>> cgr_trajectory(const std::string& seq) {
        std::vector<std::pair<float, float>> points;
        points.reserve(seq.length());

        float x = 0.5f, y = 0.5f;  // Start at center
        for (char c : seq) {
            float cx, cy;
            get_corner(c, cx, cy);
            x = (x + cx) / 2.0f;
            y = (y + cy) / 2.0f;
            points.push_back({x, y});
        }
        return points;
    }

    // Bin CGR points into grid of given resolution
    static std::vector<std::vector<int>> bin_to_grid(
        const std::vector<std::pair<float, float>>& points, int resolution) {

        std::vector<std::vector<int>> grid(resolution, std::vector<int>(resolution, 0));

        for (const auto& [x, y] : points) {
            int ix = std::min((int)(x * resolution), resolution - 1);
            int iy = std::min((int)(y * resolution), resolution - 1);
            grid[ix][iy]++;
        }
        return grid;
    }

    // Extract multi-scale features
    static MultiScaleCGRFeatures extract(const std::string& seq, int min_length = 500) {
        MultiScaleCGRFeatures features;

        if ((int)seq.length() < min_length) {
            return features;  // Return zeros for short sequences
        }

        // Generate CGR trajectory
        auto points = cgr_trajectory(seq);

        // Compute stats at three scales
        auto grid_16 = bin_to_grid(points, 16);
        auto grid_32 = bin_to_grid(points, 32);
        auto grid_64 = bin_to_grid(points, 64);

        auto stats_16 = CGRStats::compute(grid_16);
        auto stats_32 = CGRStats::compute(grid_32);
        auto stats_64 = CGRStats::compute(grid_64);

        // Scale transition slopes
        features.dH_16_32 = stats_32.entropy - stats_16.entropy;
        features.dH_32_64 = stats_64.entropy - stats_32.entropy;
        features.dO_16_32 = stats_32.occupancy - stats_16.occupancy;
        features.dO_32_64 = stats_64.occupancy - stats_32.occupancy;
        features.dL_16_32 = stats_32.lacunarity - stats_16.lacunarity;
        features.dL_32_64 = stats_64.lacunarity - stats_32.lacunarity;

        // Mid-scale absolute values
        features.H_32 = stats_32.entropy;
        features.O_32 = stats_32.occupancy;
        features.L_32 = stats_32.lacunarity;

        return features;
    }

    // Batch extract for all contigs
    static std::vector<MultiScaleCGRFeatures> extract_batch(
        const std::vector<std::pair<std::string, std::string>>& contigs,
        int threads = 1, int min_length = 500) {

        std::vector<MultiScaleCGRFeatures> results(contigs.size());

        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (size_t i = 0; i < contigs.size(); i++) {
            results[i] = extract(contigs[i].second, min_length);
        }

        return results;
    }

    // Sigmoid reliability weight based on contig length.
    // CGR stats become unreliable on short contigs (<3kb) because the 64x64
    // grid has too few points per cell. Weight attenuates CGR contribution
    // smoothly so short contigs fall back on encoder+aDNA features.
    // L_half=4000: 50% weight at 4kb (grounded in Bog-38 contamination analysis:
    // contaminating contigs avg 2.2kb, core archaeal contigs avg 4.8kb).
    static float length_weight(size_t seq_len, int L_half = 4000, float k = 4.0f) {
        float x = k * (std::log10(static_cast<float>(seq_len))
                     - std::log10(static_cast<float>(L_half)));
        return 1.0f / (1.0f + std::exp(-x));
    }
};

#ifdef USE_LIBTORCH

#include <torch/torch.h>

// Late fusion module for multi-scale CGR
// Adds small residual to base embedding, gated by reliability
struct MultiScaleCGRFusionImpl : torch::nn::Module {
    torch::nn::Linear branch1{nullptr}, branch2{nullptr};
    torch::nn::Linear gate{nullptr};
    float beta_;

    MultiScaleCGRFusionImpl(int mscgr_dim = 9, int hidden_dim = 16,
                            int embed_dim = 128, int gate_dim = 2,
                            float beta = 0.1f)
        : beta_(beta) {

        // msCGR branch: mscgr_dim -> hidden -> embed_dim
        branch1 = register_module("branch1",
            torch::nn::Linear(mscgr_dim, hidden_dim));
        branch2 = register_module("branch2",
            torch::nn::Linear(hidden_dim, embed_dim));

        // Reliability gate: [log_len, n_eff] -> scalar
        gate = register_module("gate",
            torch::nn::Linear(gate_dim, 1));
    }

    // z_base: [B, 128] base embedding from COMEBin encoder
    // mscgr: [B, 9] multi-scale CGR features
    // gate_feats: [B, 2] = [log(contig_len), n_eff]
    torch::Tensor forward(torch::Tensor z_base, torch::Tensor mscgr,
                          torch::Tensor gate_feats) {
        // Compute delta from msCGR branch
        auto h = torch::relu(branch1->forward(mscgr));
        h = torch::dropout(h, 0.1, this->is_training());
        auto delta = branch2->forward(h);

        // Reliability gate
        auto g = torch::sigmoid(gate->forward(gate_feats));

        // Fused embedding with gated residual
        auto z_fused = z_base + beta_ * g * delta;

        // L2 normalize
        z_fused = torch::nn::functional::normalize(z_fused,
            torch::nn::functional::NormalizeFuncOptions().dim(1));

        return z_fused;
    }
};
TORCH_MODULE(MultiScaleCGRFusion);

#endif  // USE_LIBTORCH

}  // namespace amber::bin2

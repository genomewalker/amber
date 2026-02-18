// AMBER - quality_leiden_backend.h
// Quality-weighted Leiden clustering that optimizes modularity + marker quality
// Design: Opus 4.6 (architecture) + GPT-5.3-codex (math)

#pragma once

#include "leiden_backend.h"
#include "marker_index.h"

#include <array>
#include <deque>
#include <atomic>
#include <omp.h>

namespace amber::bin2 {

static constexpr int MAX_MARKER_TYPES = 256;

struct CommQuality {
    std::array<int32_t, MAX_MARKER_TYPES> cnt{};
    int32_t present = 0;
    int32_t dup_excess = 0;
};

struct ThreadCtx {
    std::deque<int> q_marker;
    std::deque<int> q_plain;
    std::vector<double> acc;
    std::vector<uint32_t> seen;
    std::vector<int> touched;
    uint32_t epoch = 1;
    static constexpr int SMALL_DEGREE_THRESHOLD = 32;
};

// Configuration for quality-weighted Leiden
struct QualityLeidenConfig {
    // Quality weight (0 = pure modularity, 1 = balanced, >1 = quality-dominant)
    float alpha = 1.0f;

    // Contamination penalty multiplier (5.0 = domain standard)
    float contamination_penalty = 5.0f;

    // Empirical calibration settings
    int calibration_samples = 10000;   // Moves to sample for λ calibration
    float lambda_min = 0.01f;          // Minimum λ (prevent quality domination)
    float lambda_max = 100.0f;         // Maximum λ (prevent modularity domination)

    // Mode: scalarized vs constrained
    bool use_constrained_mode = false; // If true, use ε-constrained refinement
    float epsilon_budget = 0.05f;      // Fraction of modularity gain allowed as loss

    // Phase 1: edge weight penalty for pairs sharing the same SCG marker.
    // Reduces w by exp(-marker_edge_penalty * n_shared_markers).
    // 0 = disabled, 3.0 = ~95% reduction per shared marker.
    float marker_edge_penalty = 3.0f;

    int n_threads = 0;  // 0 = use hardware concurrency
};

// Quality-weighted Leiden backend
// Extends LeidenBackend with marker-based quality in the objective:
//   total_delta = delta_modularity + λ * delta_quality
// where delta_quality = delta_completeness - penalty * delta_contamination
class QualityLeidenBackend : public LeidenBackend {
public:
    QualityLeidenBackend() = default;

    // Set marker index (must be called before cluster())
    void set_marker_index(const MarkerIndex* index) { marker_index_ = index; }

    // Set quality configuration
    void set_quality_config(const QualityLeidenConfig& config) { qconfig_ = config; }

    // Override cluster to add quality-weighted optimization
    ClusteringResult cluster(const std::vector<WeightedEdge>& edges,
                          int n_nodes,
                          const LeidenConfig& config) override;

protected:
    // Override move_nodes_fast to add quality delta
    bool move_nodes_fast_quality(
        std::vector<int>& labels,
        std::vector<double>& comm_weights,
        std::vector<double>& comm_sizes,
        std::vector<std::unordered_map<int, int>>& comm_markers,
        float resolution,
        float lambda);

    // Override refinement to add quality delta
    void refine_partition_quality(
        std::vector<int>& labels,
        const std::vector<int>& aggregate_membership,
        std::vector<double>& comm_weights,
        std::vector<double>& comm_sizes,
        std::vector<std::unordered_map<int, int>>& comm_markers,
        float resolution,
        float lambda);

    // Calibrate λ empirically from observed move deltas
    float calibrate_lambda(
        const std::vector<int>& labels,
        const std::vector<double>& comm_weights,
        const std::vector<double>& comm_sizes,
        const std::vector<std::unordered_map<int, int>>& comm_markers,
        float resolution);

    // Initialize community marker counts from labels
    void init_comm_markers(
        const std::vector<int>& labels,
        int n_communities,
        std::vector<std::unordered_map<int, int>>& comm_markers);

    // Aggregate marker counts for aggregate graph
    std::vector<std::unordered_map<int, int>> aggregate_markers(
        const std::vector<int>& labels,
        const std::vector<std::unordered_map<int, int>>& comm_markers,
        int n_communities);

    // Update community marker counts when node moves
    void update_comm_markers(
        int node,
        int from_comm,
        int to_comm,
        std::vector<std::unordered_map<int, int>>& comm_markers);

    // Dense CommQuality versions
    void init_comm_quality(
        const std::vector<int>& labels,
        int n_communities,
        std::vector<CommQuality>& comm_q);

    void update_comm_quality(
        int node, int from, int to,
        std::vector<CommQuality>& comm_q);

    QualityDelta compute_quality_delta_fast(
        int node, int source_comm, int target_comm,
        const std::vector<CommQuality>& comm_q) const;

    // Multicore quality-weighted move kernel with work stealing
    bool move_nodes_fast_quality_mt(
        std::vector<int>& labels,
        std::vector<double>& comm_weights,
        std::vector<double>& comm_sizes,
        std::vector<CommQuality>& comm_q,
        float resolution,
        float lambda,
        int n_threads);

    // Multicore refinement wrapper - calls move_nodes_fast_quality_mt in a loop
    void refine_partition_quality_mt(
        std::vector<int>& labels,
        std::vector<double>& comm_weights,
        std::vector<double>& comm_sizes,
        std::vector<CommQuality>& comm_q,
        float resolution,
        float lambda,
        int n_threads);

    // Dense CommQuality version of calibrate_lambda
    float calibrate_lambda(
        const std::vector<int>& labels,
        const std::vector<double>& comm_weights,
        const std::vector<double>& comm_sizes,
        const std::vector<CommQuality>& comm_q,
        float resolution);

private:
    const MarkerIndex* marker_index_ = nullptr;
    QualityLeidenConfig qconfig_;
    std::vector<uint8_t> has_marker_;
};

// Factory function
std::unique_ptr<ILeidenBackend> create_quality_leiden_backend(
    const MarkerIndex* marker_index,
    const QualityLeidenConfig& qconfig);

}  // namespace amber::bin2

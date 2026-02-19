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

    // Phase 1b: marker-guided map equation refinement after libleidenalg.
    // Replaces modularity delta with the map equation delta (parameter-free,
    // MDL-based), combined with the marker quality penalty. This avoids the
    // resolution limit and over-merging issues of modularity-based refinement.
    bool use_map_equation = false;

    // Resolution calibration: try n_res_trials resolutions log-uniform over
    // [res_search_min, res_search_max], score each with the internal SCG metric
    // (Σ present - penalty*dup_excess), fit a quadratic in log-resolution space,
    // and use the argmax for the main pipeline run.
    bool calibrate_resolution = false;
    int n_res_trials = 8;
    float res_search_min = 0.5f;
    float res_search_max = 50.0f;

    int n_threads = 0;  // 0 = use hardware concurrency

    // Best-of-K bandwidth × seed restart search.
    // Resolution is pinned via calibrate_resolution() first; the bandit then
    // sweeps bandwidth (the kNN edge kernel scale), which is the remaining
    // free parameter that most affects partition quality.
    //
    // Edge re-weighting: given input edges with weights exp(-dist/bw_orig),
    // a new bandwidth bw_new yields w_new = pow(w_orig, bw_orig/bw_new).
    //
    // Stage 1: restart_stage1_bw log-uniform bandwidths × 1 seed → Phase 1 SCG score
    // Stage 2: top restart_stage2_topk arms, restart_stage2_extra runs via UCB bandit
    //          → Phase 2 score for candidates within margin of global best
    // Stage 3: best arm, up to restart_stage3_extra runs (always Phase 2), early stop
    int      n_leiden_restarts    = 1;        // Total budget (1 = off)
    float    original_bandwidth   = 0.2f;     // Bandwidth used to build input edges
    float    bw_search_min        = 0.05f;    // Log-uniform bandwidth search range
    float    bw_search_max        = 0.5f;
    int      restart_stage1_bw    = 7;        // Stage 1 bandwidth grid size
    int      restart_stage2_topk  = 3;        // Top arms to race in Stage 2
    int      restart_stage2_extra = 12;       // Extra runs across top arms in Stage 2
    int      restart_stage3_extra = 6;        // Max extra runs on best arm in Stage 3
    int      restart_patience     = 4;        // Stage 3 early stop after N non-improving runs
    float    restart_ucb_beta     = 1.0f;     // UCB: priority = mean + beta * std
    long long restart_min_viable_bp = 200000; // Only score clusters >= this size in bp
};

// Snapshot of one (bandwidth, seed) candidate from the restart search.
struct CandidateSnapshot {
    float resolution = 0.0f;
    float bandwidth  = 0.0f;
    int   seed = 0;

    ClusteringResult p1_result;         // output of libleidenalg Phase 1
    float score_p1 = -1e30f;           // SCG score after Phase 1

    bool             has_p2 = false;
    std::vector<int> labels_after_p2;
    int              num_clusters_after_p2 = 0;
    float            score_p2 = -1e30f; // SCG score after Phase 2 (valid if has_p2)

    float selection_score() const { return has_p2 ? score_p2 : score_p1; }
};

// One bandwidth arm in the adaptive racing search.
//
// Bandit statistics (mean_p1, m2_p1, pulls_p1) track Phase 1 scores only,
// making the reward process stationary. P2 scores are used only to update
// global_best — they do not enter arm statistics.
struct ResolutionArm {
    float  bandwidth  = 0.0f;

    // P1-only stats: used for UCB priority, Stage 2 arm ranking, Stage 3 selection.
    int    pulls_p1 = 0;
    double mean_p1  = 0.0;
    double m2_p1    = 0.0;   // Welford M2 for online variance
    float  best_p1  = -1e30f;

    CandidateSnapshot best;  // best candidate ever (P2 score if available, else P1)

    void add(const CandidateSnapshot& c) {
        // Update P1-only bandit stats (always available, stationary).
        pulls_p1++;
        double x = c.score_p1;
        double delta = x - mean_p1;
        mean_p1 += delta / pulls_p1;
        m2_p1   += delta * (x - mean_p1);
        if (c.score_p1 > best_p1) best_p1 = c.score_p1;

        // Update best-ever snapshot (may carry P2 result for global_best tracking).
        if (c.selection_score() > best.selection_score()) best = c;
    }

    // Sample std-dev of P1 scores, with prior_sigma when fewer than 2 pulls.
    double stddev_p1(double prior_sigma = 1.0) const {
        return pulls_p1 > 1 ? std::sqrt(m2_p1 / (pulls_p1 - 1)) : prior_sigma;
    }

    // UCB priority based on P1 scores only.
    double priority(double beta, double prior_sigma = 1.0) const {
        return mean_p1 + beta * stddev_p1(prior_sigma);
    }
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

    // Set CheckM estimator + node names for proper colocation-based scoring.
    // node_names[i] must match the node ordering used in cluster().
    // When set, scg_score_labels() returns CheckM HQ count and Phase 2 uses
    // CheckM contamination > 5% as the split criterion.
    void set_checkm_estimator(const CheckMQualityEstimator* est,
                              const std::vector<std::string>* node_names) {
        checkm_est_ = est;
        node_names_ = node_names;
    }

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

    // Map equation delta for moving node from curr to tgt (gain = -ΔL, positive = improvement).
    // comm_internal[c] = Σ_{v∈c} Σ_{u∈c,u~v} w_{uv} (each internal edge counted twice).
    // q_total = (total_edge_weight_ - Σ comm_internal[c]) / total_edge_weight_.
    double delta_map_equation(
        int node, int curr, int tgt,
        double edges_to_curr, double edges_to_tgt,
        const std::vector<double>& comm_weights,
        const std::vector<double>& comm_internal,
        double q_total) const;

    // Initialize comm_internal from current labels.
    void init_comm_internal(
        const std::vector<int>& labels,
        int n_communities,
        std::vector<double>& comm_internal) const;

    // Multicore quality-weighted move kernel with work stealing
    bool move_nodes_fast_quality_mt(
        std::vector<int>& labels,
        std::vector<double>& comm_weights,
        std::vector<double>& comm_sizes,
        std::vector<CommQuality>& comm_q,
        std::vector<double>& comm_internal,
        double& q_total,
        float resolution,
        float lambda,
        int n_threads);

    // Multicore refinement wrapper - calls move_nodes_fast_quality_mt in a loop
    void refine_partition_quality_mt(
        std::vector<int>& labels,
        std::vector<double>& comm_weights,
        std::vector<double>& comm_sizes,
        std::vector<CommQuality>& comm_q,
        std::vector<double>& comm_internal,
        double& q_total,
        float resolution,
        float lambda,
        int n_threads);

    // Resolution calibration: score a partition with the internal SCG metric.
    float scg_quality_score(const std::vector<CommQuality>& comm_q) const;

    // Try n_res_trials resolutions, fit quadratic in log-res space, return argmax.
    float calibrate_resolution(
        const std::vector<WeightedEdge>& edges,
        int n_nodes,
        const LeidenConfig& config) const;

    // SCG quality score for an arbitrary label vector (builds CommQuality internally).
    // Falls back to 0.0 if no markers hit any node.
    float scg_score_labels(const std::vector<int>& labels, int n_communities) const;

    // Run Phase 2 contamination splitting on a label vector.
    // Returns {updated_labels, updated_n_communities}.
    // Requires build_adjacency() to have been called.
    std::pair<std::vector<int>, int> run_phase2(
        std::vector<int> labels,
        int n_communities,
        const LeidenConfig& effective_config);

    // Evaluate one (bandwidth, seed) candidate.
    // raw_edges: edges with original_bandwidth applied (exp(-dist/bw_orig)).
    // Internally re-weights to bandwidth via pow(w, bw_orig/bw_new), applies
    // marker penalization, runs libleidenalg, and optionally scores Phase 2.
    CandidateSnapshot evaluate_candidate(
        const std::vector<WeightedEdge>& raw_edges,
        int n_nodes,
        const LeidenConfig& base_cfg,
        float bandwidth,
        int seed,
        bool run_phase2_for_score);

    // 3-stage adaptive racing over (bandwidth × seed) space at the calibrated resolution.
    // raw_edges: edges with original_bandwidth applied (no marker penalization yet).
    // Returns the globally best CandidateSnapshot.
    CandidateSnapshot run_adaptive_restarts(
        const std::vector<WeightedEdge>& raw_edges,
        int n_nodes,
        const LeidenConfig& base_cfg);

    // Dense CommQuality version of calibrate_lambda
    // comm_internal and q_total are only used when qconfig_.use_map_equation is true.
    float calibrate_lambda(
        const std::vector<int>& labels,
        const std::vector<double>& comm_weights,
        const std::vector<double>& comm_sizes,
        const std::vector<CommQuality>& comm_q,
        const std::vector<double>& comm_internal,
        double q_total,
        float resolution);

private:
    const MarkerIndex* marker_index_ = nullptr;
    QualityLeidenConfig qconfig_;
    std::vector<uint8_t> has_marker_;

    // Optional: proper CheckM estimator for colocation-based scoring
    const CheckMQualityEstimator* checkm_est_ = nullptr;
    const std::vector<std::string>* node_names_ = nullptr;
};

// Factory function
std::unique_ptr<ILeidenBackend> create_quality_leiden_backend(
    const MarkerIndex* marker_index,
    const QualityLeidenConfig& qconfig);

}  // namespace amber::bin2

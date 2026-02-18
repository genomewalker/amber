#pragma once

#include "leiden_backend.h"
#include <vector>
#include <memory>

namespace amber::bin2 {

struct EnsembleLeidenConfig {
    int n_runs = 10;              // Number of Leiden runs with different seeds
    float consensus_threshold = 0.5f;  // Min co-occurrence to keep edge (0.5 = majority)
    int base_seed = 42;           // Base seed, will use base_seed + i for run i
    bool verbose = false;

    // Resolution exploration (log-uniform over [res_min, res_max]).
    // When enabled, run i uses resolution exp(log(res_min) + i/(n_runs-1) * log(res_max/res_min)).
    // This covers the resolution space without needing a fixed grid, letting the
    // co-occurrence consensus vote on the right granularity across scales.
    bool vary_resolution = false;
    float res_min = 1.0f;
    float res_max = 20.0f;
};

// Ensemble Leiden: run Leiden N times, build consensus from co-occurrence
class EnsembleLeidenBackend : public ILeidenBackend {
public:
    explicit EnsembleLeidenBackend(std::unique_ptr<ILeidenBackend> base_backend);
    ~EnsembleLeidenBackend() override = default;

    ClusteringResult cluster(
        const std::vector<WeightedEdge>& edges,
        int n_nodes,
        const LeidenConfig& config) override;

    void set_seed(int seed) override { ensemble_config_.base_seed = seed; }

    void set_ensemble_config(const EnsembleLeidenConfig& cfg) { ensemble_config_ = cfg; }

private:
    // Build co-occurrence matrix from multiple runs
    // Returns edge list with weights = fraction of runs where pair was co-clustered
    std::vector<WeightedEdge> build_cooccurrence(
        const std::vector<std::vector<int>>& all_labels,
        int n_nodes) const;

    // Extract clusters from weighted co-occurrence graph
    // Uses connected components at threshold
    std::vector<int> consensus_clusters(
        const std::vector<WeightedEdge>& cooccurrence_edges,
        int n_nodes,
        float threshold) const;

    std::unique_ptr<ILeidenBackend> base_backend_;
    EnsembleLeidenConfig ensemble_config_;
};

// Factory function
std::unique_ptr<EnsembleLeidenBackend> create_ensemble_leiden(
    int n_runs = 10,
    float consensus_threshold = 0.5f);

}  // namespace amber::bin2

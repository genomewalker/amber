#pragma once

#include "clustering_types.h"
#include <memory>
#include <vector>
#include <random>

namespace amber::bin2 {

// Forward declaration for aggregate graph
struct AggregateGraph;

class ILeidenBackend {
public:
    virtual ~ILeidenBackend() = default;

    virtual ClusteringResult cluster(
        const std::vector<WeightedEdge>& edges,
        int n_nodes,
        const LeidenConfig& config) = 0;

    virtual void set_seed(int seed) = 0;
};

class LouvainBackend : public ILeidenBackend {
public:
    LouvainBackend();
    ~LouvainBackend() override = default;

    ClusteringResult cluster(
        const std::vector<WeightedEdge>& edges,
        int n_nodes,
        const LeidenConfig& config) override;

    void set_seed(int seed) override { rng_.seed(seed); }

private:
    void build_adjacency(
        const std::vector<WeightedEdge>& edges,
        int n_nodes);

    double modularity(
        const std::vector<int>& labels,
        float resolution) const;

    double delta_modularity(
        int node,
        int target_comm,
        const std::vector<int>& labels,
        float resolution) const;

    int best_community(
        int node,
        const std::vector<int>& labels,
        float resolution) const;

    bool local_move_phase(
        std::vector<int>& labels,
        float resolution);

    int compact_labels(std::vector<int>& labels) const;

    std::vector<std::vector<std::pair<int, float>>> adj_;
    std::vector<float> node_weights_;
    double total_edge_weight_ = 0.0;
    int n_nodes_ = 0;

    std::mt19937 rng_;
};

class LeidenBackend : public ILeidenBackend {
public:
    LeidenBackend();
    ~LeidenBackend() override = default;

    ClusteringResult cluster(
        const std::vector<WeightedEdge>& edges,
        int n_nodes,
        const LeidenConfig& config) override;

    void set_seed(int seed) override { rng_.seed(seed); }

private:
    void build_adjacency(
        const std::vector<WeightedEdge>& edges,
        int n_nodes,
        const LeidenConfig& config);

    // Fast delta modularity using precomputed edge weights
    double delta_modularity_fast(
        int node,
        int current_comm,
        int target_comm,
        double edges_to_current,
        double edges_to_target,
        const std::vector<double>& comm_weights,
        const std::vector<double>& comm_sizes,
        float resolution) const;

    // Single pass of local moving
    bool move_nodes_pass(
        std::vector<int>& labels,
        std::vector<double>& comm_weights,
        std::vector<double>& comm_sizes,
        const std::vector<int>& order,
        float resolution);

    // Local moving phase - iterates until convergence
    bool move_nodes_fast(
        std::vector<int>& labels,
        std::vector<double>& comm_weights,
        std::vector<double>& comm_sizes,
        float resolution);

    // Graph aggregation
    AggregateGraph aggregate_graph(
        const std::vector<int>& labels,
        int n_communities) const;

    // Cluster aggregate graph
    std::vector<int> cluster_aggregate(
        const AggregateGraph& agg,
        float resolution);

    // Refinement within aggregate communities
    void refine_partition(
        std::vector<int>& labels,
        const std::vector<int>& aggregate_membership,
        std::vector<double>& comm_weights,
        std::vector<double>& comm_sizes,
        float resolution);

    // Modularity computation
    double modularity_rber(
        const std::vector<int>& labels,
        float resolution) const;

    double modularity_config(
        const std::vector<int>& labels,
        float resolution) const;

    int compact_labels(std::vector<int>& labels) const;

    // Graph representation (using double for precision)
    std::vector<std::vector<std::pair<int, double>>> adj_;
    std::vector<double> node_weights_;      // Degree-based weights (for config model)
    std::vector<double> node_sizes_;        // Node sizes (for RBER model)
    double total_edge_weight_ = 0.0;
    double total_node_size_ = 0.0;          // Sum of all node sizes
    int n_nodes_ = 0;
    NullModel null_model_ = NullModel::Configuration;
    std::vector<bool> is_fixed_;            // Per-node fixed membership flag

    std::mt19937 rng_;
};

#ifdef USE_LIBLEIDENALG
// Backend using original libleidenalg library for exact COMEBin matching
class LibLeidenalgBackend : public ILeidenBackend {
public:
    LibLeidenalgBackend();
    ~LibLeidenalgBackend() override;

    ClusteringResult cluster(
        const std::vector<WeightedEdge>& edges,
        int n_nodes,
        const LeidenConfig& config) override;

    void set_seed(int seed) override { seed_ = seed; }

private:
    int seed_ = 42;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

std::unique_ptr<ILeidenBackend> create_leiden_backend(bool use_leiden = true);

}  // namespace amber::bin2

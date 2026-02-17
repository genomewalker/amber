#pragma once
// Leiden clustering using original libleidenalg library (exact COMEBin match)

#include <vector>
#include <string>
#include <memory>

#include <igraph/igraph.h>
#include <libleidenalg/GraphHelper.h>
#include <libleidenalg/Optimiser.h>
#include <libleidenalg/RBERVertexPartition.h>

namespace amber::bin2 {

struct LeidenIgraphConfig {
    float resolution = 5.0f;
    float bandwidth = 0.3f;
    int percentile_cutoff = 100;  // Keep edges within this percentile
    int k_neighbors = 100;
    int n_iterations = -1;  // -1 = until convergence
};

struct LeidenIgraphResult {
    std::vector<int> labels;
    int num_clusters;
    double quality;
};

class LeidenIgraph {
public:
    LeidenIgraph() = default;
    ~LeidenIgraph() = default;

    // Build graph from kNN results
    // distances should be SQUARED L2 distances
    void build_graph(
        int n_nodes,
        const std::vector<std::vector<int>>& knn_indices,
        const std::vector<std::vector<float>>& knn_distances,
        const std::vector<size_t>& node_sizes,
        const LeidenIgraphConfig& config);

    // Run Leiden with optional fixed membership
    // initial_membership: singleton (0,1,2,...,n-1) or pre-assigned
    // is_fixed: which nodes have fixed membership (e.g., seed contigs)
    LeidenIgraphResult run(
        const std::vector<size_t>& initial_membership,
        const std::vector<bool>& is_fixed,
        const LeidenIgraphConfig& config);

    // Convenience: run without fixed membership
    LeidenIgraphResult run(const LeidenIgraphConfig& config);

private:
    igraph_t graph_;
    std::vector<double> weights_;
    std::vector<double> node_sizes_d_;  // double version for Graph API
    int n_nodes_ = 0;
    bool graph_built_ = false;

    void cleanup();
};

// Implementation inline for header-only usage

inline void LeidenIgraph::cleanup() {
    if (graph_built_) {
        igraph_destroy(&graph_);
        graph_built_ = false;
    }
}

inline void LeidenIgraph::build_graph(
    int n_nodes,
    const std::vector<std::vector<int>>& knn_indices,
    const std::vector<std::vector<float>>& knn_distances,
    const std::vector<size_t>& node_sizes,
    const LeidenIgraphConfig& config) {

    cleanup();
    n_nodes_ = n_nodes;

    // Collect all distances for percentile calculation
    std::vector<float> all_dists;
    int k = config.k_neighbors;
    for (int i = 0; i < n_nodes; ++i) {
        for (int j = 1; j <= k && j < (int)knn_distances[i].size(); ++j) {
            all_dists.push_back(knn_distances[i][j]);
        }
    }
    std::sort(all_dists.begin(), all_dists.end());
    float dist_cutoff = all_dists[all_dists.size() * config.percentile_cutoff / 100 - 1];

    // Build edges: keep only i > j (deduplicate) and within cutoff
    std::vector<int> sources, targets;
    weights_.clear();

    for (int i = 0; i < n_nodes; ++i) {
        for (int j = 1; j <= k && j < (int)knn_indices[i].size(); ++j) {
            int target = knn_indices[i][j];
            float dist = knn_distances[i][j];
            if (dist <= dist_cutoff && i > target) {
                sources.push_back(i);
                targets.push_back(target);
                weights_.push_back(std::exp(-dist / config.bandwidth));
            }
        }
    }

    // Create igraph
    igraph_vector_int_t edge_vec;
    igraph_vector_int_init(&edge_vec, sources.size() * 2);
    for (size_t i = 0; i < sources.size(); ++i) {
        VECTOR(edge_vec)[2*i] = sources[i];
        VECTOR(edge_vec)[2*i + 1] = targets[i];
    }
    igraph_create(&graph_, &edge_vec, n_nodes, IGRAPH_UNDIRECTED);
    igraph_vector_int_destroy(&edge_vec);

    // Convert node_sizes to double
    node_sizes_d_.resize(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        node_sizes_d_[i] = static_cast<double>(node_sizes[i]);
    }

    graph_built_ = true;
}

inline LeidenIgraphResult LeidenIgraph::run(
    const std::vector<size_t>& initial_membership,
    const std::vector<bool>& is_fixed,
    const LeidenIgraphConfig& config) {

    if (!graph_built_) {
        throw std::runtime_error("Graph not built");
    }

    // Create Graph wrapper (libleidenalg's Graph class)
    Graph graph(&graph_, weights_, node_sizes_d_);

    // Create RBER partition
    RBERVertexPartition partition(&graph, initial_membership, config.resolution);

    // Run optimizer
    Optimiser optimiser;
    optimiser.set_rng_seed(42);  // For reproducibility

    // Run with or without fixed membership
    if (is_fixed.empty()) {
        optimiser.optimise_partition(&partition);
    } else {
        optimiser.optimise_partition(&partition, is_fixed);
    }

    // Extract results
    LeidenIgraphResult result;
    result.num_clusters = partition.n_communities();
    result.quality = partition.quality(config.resolution);
    result.labels.resize(n_nodes_);
    for (int i = 0; i < n_nodes_; ++i) {
        result.labels[i] = partition.membership(i);
    }

    return result;
}

inline LeidenIgraphResult LeidenIgraph::run(const LeidenIgraphConfig& config) {
    // Singleton initialization, no fixed membership
    std::vector<size_t> initial(n_nodes_);
    for (int i = 0; i < n_nodes_; ++i) {
        initial[i] = i;
    }
    return run(initial, {}, config);
}

}  // namespace amber::bin2

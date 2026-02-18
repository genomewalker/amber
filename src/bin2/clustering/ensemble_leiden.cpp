#include "ensemble_leiden.h"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>

namespace amber::bin2 {

EnsembleLeidenBackend::EnsembleLeidenBackend(std::unique_ptr<ILeidenBackend> base_backend)
    : base_backend_(std::move(base_backend)) {}

ClusteringResult EnsembleLeidenBackend::cluster(
    const std::vector<WeightedEdge>& edges,
    int n_nodes,
    const LeidenConfig& config) {

    if (ensemble_config_.verbose) {
        std::cerr << "[EnsembleLeiden] Running " << ensemble_config_.n_runs
                  << " Leiden iterations with different seeds\n";
    }

    // Collect labels from all runs
    std::vector<std::vector<int>> all_labels;
    all_labels.reserve(ensemble_config_.n_runs);

    double sum_modularity = 0.0;
    int sum_clusters = 0;

    const int n_runs = ensemble_config_.n_runs;
    const bool vary_res = ensemble_config_.vary_resolution && n_runs > 1;
    const float log_res_min = vary_res ? std::log(ensemble_config_.res_min) : 0.0f;
    const float log_res_max = vary_res ? std::log(ensemble_config_.res_max) : 0.0f;

    if (vary_res && ensemble_config_.verbose) {
        std::cerr << "[EnsembleLeiden] Resolution range: [" << ensemble_config_.res_min
                  << ", " << ensemble_config_.res_max << "] log-uniform\n";
    }

    for (int run = 0; run < n_runs; run++) {
        LeidenConfig run_config = config;
        run_config.random_seed = ensemble_config_.base_seed + run;

        if (vary_res) {
            float t = static_cast<float>(run) / (n_runs - 1);  // 0..1
            run_config.resolution = std::exp(log_res_min + t * (log_res_max - log_res_min));
        }

        base_backend_->set_seed(run_config.random_seed);
        auto result = base_backend_->cluster(edges, n_nodes, run_config);

        all_labels.push_back(std::move(result.labels));
        sum_modularity += result.modularity;
        sum_clusters += result.num_clusters;

        if (ensemble_config_.verbose) {
            std::cerr << "[EnsembleLeiden] Run " << (run + 1) << "/" << n_runs
                      << " - " << result.num_clusters << " clusters, Q=" << result.modularity;
            if (vary_res) std::cerr << ", res=" << run_config.resolution;
            std::cerr << "\n";
        }
    }

    // Build co-occurrence matrix
    if (ensemble_config_.verbose) {
        std::cerr << "[EnsembleLeiden] Building co-occurrence matrix...\n";
    }

    auto cooccurrence_edges = build_cooccurrence(all_labels, n_nodes);

    if (ensemble_config_.verbose) {
        std::cerr << "[EnsembleLeiden] " << cooccurrence_edges.size()
                  << " co-occurrence edges (threshold=" << ensemble_config_.consensus_threshold << ")\n";
    }

    // Extract consensus clusters
    auto consensus_labels = consensus_clusters(cooccurrence_edges, n_nodes,
                                               ensemble_config_.consensus_threshold);

    // Count clusters
    std::unordered_set<int> unique_labels(consensus_labels.begin(), consensus_labels.end());
    int n_clusters = static_cast<int>(unique_labels.size());

    if (ensemble_config_.verbose) {
        std::cerr << "[EnsembleLeiden] Consensus: " << n_clusters << " clusters\n";
    }

    ClusteringResult result;
    result.labels = std::move(consensus_labels);
    result.num_clusters = n_clusters;
    result.modularity = sum_modularity / ensemble_config_.n_runs;  // Average modularity

    return result;
}

std::vector<WeightedEdge> EnsembleLeidenBackend::build_cooccurrence(
    const std::vector<std::vector<int>>& all_labels,
    int n_nodes) const {

    int n_runs = static_cast<int>(all_labels.size());

    // For each pair of nodes, count how many runs they were co-clustered
    // Using map: (min_node, max_node) -> count
    std::unordered_map<uint64_t, int> cooccurrence_count;

    for (const auto& labels : all_labels) {
        // Group nodes by cluster
        std::unordered_map<int, std::vector<int>> cluster_members;
        for (int i = 0; i < n_nodes; i++) {
            cluster_members[labels[i]].push_back(i);
        }

        // For each cluster, count co-occurrences
        for (const auto& [cluster_id, members] : cluster_members) {
            // Add co-occurrence for all pairs in this cluster
            for (size_t i = 0; i < members.size(); i++) {
                for (size_t j = i + 1; j < members.size(); j++) {
                    int u = members[i], v = members[j];
                    if (u > v) std::swap(u, v);
                    uint64_t key = (static_cast<uint64_t>(u) << 32) | v;
                    cooccurrence_count[key]++;
                }
            }
        }
    }

    // Convert to edge list with weights = fraction of runs
    std::vector<WeightedEdge> edges;
    edges.reserve(cooccurrence_count.size());

    for (const auto& [key, count] : cooccurrence_count) {
        int u = static_cast<int>(key >> 32);
        int v = static_cast<int>(key & 0xFFFFFFFF);
        float weight = static_cast<float>(count) / n_runs;
        edges.emplace_back(u, v, weight);
    }

    return edges;
}

std::vector<int> EnsembleLeidenBackend::consensus_clusters(
    const std::vector<WeightedEdge>& cooccurrence_edges,
    int n_nodes,
    float threshold) const {

    // Build adjacency list for edges above threshold
    std::vector<std::vector<int>> adj(n_nodes);
    for (const auto& e : cooccurrence_edges) {
        if (e.w >= threshold) {
            adj[e.u].push_back(e.v);
            adj[e.v].push_back(e.u);
        }
    }

    // Find connected components using BFS
    std::vector<int> labels(n_nodes, -1);
    int current_cluster = 0;

    for (int start = 0; start < n_nodes; start++) {
        if (labels[start] >= 0) continue;

        // BFS from this node
        std::vector<int> queue;
        queue.push_back(start);
        labels[start] = current_cluster;

        size_t head = 0;
        while (head < queue.size()) {
            int node = queue[head++];
            for (int neighbor : adj[node]) {
                if (labels[neighbor] < 0) {
                    labels[neighbor] = current_cluster;
                    queue.push_back(neighbor);
                }
            }
        }

        current_cluster++;
    }

    return labels;
}

std::unique_ptr<EnsembleLeidenBackend> create_ensemble_leiden(
    int n_runs,
    float consensus_threshold) {

    auto base = create_leiden_backend(true);  // Use Leiden, not Louvain
    auto ensemble = std::make_unique<EnsembleLeidenBackend>(std::move(base));

    EnsembleLeidenConfig cfg;
    cfg.n_runs = n_runs;
    cfg.consensus_threshold = consensus_threshold;
    ensemble->set_ensemble_config(cfg);

    return ensemble;
}

}  // namespace amber::bin2

#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace amber::bin2 {

struct KnnConfig {
    int k = 100;                  // COMEBin default: 100
    int ef_search = 1000;         // COMEBin: ef = max_edges * 10
    int M = 16;                   // COMEBin default: 16
    int ef_construction = 1000;   // COMEBin: ef_construction = ef
    int random_seed = 42;
};

struct GraphConfig {
    float bandwidth = 0.1f;
    int partgraph_ratio = 50;     // Percentile cutoff for edge pruning (50, 80, 100)
};

struct NeighborList {
    std::vector<std::vector<int>> ids;
    std::vector<std::vector<float>> dists;

    size_t size() const { return ids.size(); }

    void resize(size_t n, size_t k) {
        ids.resize(n);
        dists.resize(n);
        for (size_t i = 0; i < n; ++i) {
            ids[i].reserve(k);
            dists[i].reserve(k);
        }
    }

    void clear() {
        ids.clear();
        dists.clear();
    }
};

struct WeightedEdge {
    int u;
    int v;
    float w;

    WeightedEdge() : u(0), v(0), w(0.0f) {}
    WeightedEdge(int u_, int v_, float w_) : u(u_), v(v_), w(w_) {}

    bool operator<(const WeightedEdge& other) const {
        return w > other.w;
    }
};

struct DamageFeatures {
    float amp5 = 0.0f;
    float lambda5 = 0.0f;
    float amp3 = 0.0f;
    float lambda3 = 0.0f;
    float p_ancient = 0.0f;
    float confidence = 0.0f;

    DamageFeatures() = default;
    DamageFeatures(float a5, float l5, float a3, float l3, float pa, float conf)
        : amp5(a5), lambda5(l5), amp3(a3), lambda3(l3), p_ancient(pa), confidence(conf) {}

    bool is_valid() const {
        return confidence > 0.0f;
    }
};

struct ClusteringResult {
    std::vector<int> labels;
    int num_clusters = 0;
    float modularity = 0.0f;
    int num_iterations = 0;
};

enum class NullModel {
    Configuration,  // Standard Newman-Girvan modularity (degree-based null)
    ErdosRenyi      // Reichardt-Bornholdt RBER (density-based null)
};

struct LeidenConfig {
    float resolution = 1.0f;
    int max_iterations = 100;       // -1 for unlimited iterations
    float tolerance = 1e-6f;
    int random_seed = 42;
    NullModel null_model = NullModel::Configuration;

    // Initial membership from seeds
    bool use_initial_membership = false;
    std::vector<int> initial_membership;

    // Fixed membership (nodes that cannot change communities)
    bool use_fixed_membership = false;
    std::vector<bool> is_membership_fixed;  // Per-node: true = fixed, false = can move

    // Node sizes (e.g., contig lengths for RBER)
    bool use_node_sizes = false;
    std::vector<double> node_sizes;
};

}  // namespace amber::bin2

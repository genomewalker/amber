#pragma once
// ==============================================================================
// Graph-based Bin Refinement via Edge Pruning
//
// Algorithm inspired by derep-genomes binary_search_filter:
// Find minimum similarity threshold where removing edges disconnects
// contaminants from the core bin.
//
// Steps:
//   1. Build TNF similarity graph (complete graph with cosine similarity edges)
//   2. Binary search for threshold where graph splits into components
//   3. Largest connected component = core bin
//   4. Small components / isolated nodes = contaminants
//
// This is a principled approach that doesn't require tuning thresholds -
// the algorithm finds the natural separation point.
// ==============================================================================

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <unordered_map>

namespace amber {
namespace graph_prune {

constexpr int TNF_DIM = 136;

// =============================================================================
// Union-Find for connected components
// =============================================================================

class UnionFind {
public:
    std::vector<int> parent, rank;

    UnionFind(int n) : parent(n), rank(n, 0) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return;
        if (rank[px] < rank[py]) std::swap(px, py);
        parent[py] = px;
        if (rank[px] == rank[py]) rank[px]++;
    }

    int count_components() {
        std::unordered_set<int> roots;
        for (size_t i = 0; i < parent.size(); ++i) {
            roots.insert(find(i));
        }
        return roots.size();
    }

    std::vector<std::vector<int>> get_components() {
        std::unordered_map<int, std::vector<int>> comp_map;
        for (size_t i = 0; i < parent.size(); ++i) {
            comp_map[find(i)].push_back(i);
        }
        std::vector<std::vector<int>> result;
        for (auto& [root, members] : comp_map) {
            result.push_back(std::move(members));
        }
        // Sort by size (largest first)
        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.size() > b.size(); });
        return result;
    }
};

// =============================================================================
// Edge representation
// =============================================================================

struct Edge {
    int u, v;
    double weight;  // similarity (higher = more similar)

    bool operator<(const Edge& other) const {
        return weight < other.weight;  // sort ascending
    }
};

// =============================================================================
// Cosine similarity
// =============================================================================

inline double cosine_similarity(const std::array<float, TNF_DIM>& a,
                                const std::array<float, TNF_DIM>& b) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < TNF_DIM; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a < 1e-10 || norm_b < 1e-10) return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// =============================================================================
// Build graph and find connected components at a given threshold
// =============================================================================

inline int count_components_at_threshold(
    const std::vector<Edge>& edges,
    int n_nodes,
    double min_weight
) {
    UnionFind uf(n_nodes);
    for (const auto& e : edges) {
        if (e.weight >= min_weight) {
            uf.unite(e.u, e.v);
        }
    }
    return uf.count_components();
}

inline std::vector<std::vector<int>> get_components_at_threshold(
    const std::vector<Edge>& edges,
    int n_nodes,
    double min_weight
) {
    UnionFind uf(n_nodes);
    for (const auto& e : edges) {
        if (e.weight >= min_weight) {
            uf.unite(e.u, e.v);
        }
    }
    return uf.get_components();
}

// =============================================================================
// Binary search for critical threshold
// =============================================================================

// Find minimum threshold where graph has > 1 component
// Returns the threshold and resulting components
struct PruneResult {
    double threshold;
    std::vector<std::vector<int>> components;
    int n_components;
};

inline PruneResult binary_search_threshold(
    const std::vector<Edge>& sorted_edges,  // sorted by weight ascending
    int n_nodes,
    int target_components = 2  // stop when we have at least this many
) {
    if (sorted_edges.empty()) {
        return {1.0, {{0}}, 1};
    }

    // Get unique weights
    std::vector<double> weights;
    weights.push_back(sorted_edges[0].weight);
    for (const auto& e : sorted_edges) {
        if (e.weight != weights.back()) {
            weights.push_back(e.weight);
        }
    }

    int low = 0, high = weights.size() - 1;
    double best_threshold = weights[high];
    std::vector<std::vector<int>> best_components;

    while (low <= high) {
        int mid = (low + high) / 2;
        double thresh = weights[mid];

        int n_comp = count_components_at_threshold(sorted_edges, n_nodes, thresh);

        if (n_comp >= target_components) {
            // Found a threshold that splits, try lower
            best_threshold = thresh;
            best_components = get_components_at_threshold(sorted_edges, n_nodes, thresh);
            high = mid - 1;
        } else {
            // Still connected, need higher threshold
            low = mid + 1;
        }
    }

    // If no split found, get components at highest threshold
    if (best_components.empty()) {
        best_components = get_components_at_threshold(sorted_edges, n_nodes, best_threshold);
    }

    return {best_threshold, best_components, static_cast<int>(best_components.size())};
}

// =============================================================================
// Main refinement function
// =============================================================================

struct GraphRefinementResult {
    std::vector<size_t> core_contigs;       // Largest component
    std::vector<size_t> outlier_contigs;    // All other components
    double threshold_used;
    int n_components;
    std::vector<int> component_sizes;
};

inline GraphRefinementResult refine_bin_graph(
    const std::vector<std::array<float, TNF_DIM>>& tnf_vectors,
    const std::vector<size_t>& contig_indices,
    double min_edge_weight = 0.90,  // Don't consider edges below this
    size_t min_bin_size = 10
) {
    GraphRefinementResult result;
    size_t n = tnf_vectors.size();

    if (n < min_bin_size) {
        result.core_contigs = contig_indices;
        result.threshold_used = 1.0;
        result.n_components = 1;
        return result;
    }

    // Step 1: Build complete graph with TNF similarity edges
    std::vector<Edge> edges;
    edges.reserve(n * (n - 1) / 2);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double sim = cosine_similarity(tnf_vectors[i], tnf_vectors[j]);
            if (sim >= min_edge_weight) {
                edges.push_back({static_cast<int>(i), static_cast<int>(j), sim});
            }
        }
    }

    // Sort edges by weight (ascending for binary search)
    std::sort(edges.begin(), edges.end());

    // Step 2: Binary search for threshold
    auto prune_result = binary_search_threshold(edges, n, 2);

    result.threshold_used = prune_result.threshold;
    result.n_components = prune_result.n_components;

    // Step 3: Identify core (largest component) and outliers
    for (const auto& comp : prune_result.components) {
        result.component_sizes.push_back(comp.size());
    }

    if (!prune_result.components.empty()) {
        // Largest component is the core
        for (int idx : prune_result.components[0]) {
            result.core_contigs.push_back(contig_indices[idx]);
        }

        // All others are outliers
        for (size_t c = 1; c < prune_result.components.size(); ++c) {
            for (int idx : prune_result.components[c]) {
                result.outlier_contigs.push_back(contig_indices[idx]);
            }
        }
    }

    return result;
}

// =============================================================================
// Alternative: Iterative pruning until target contamination
// =============================================================================

// Iteratively increase threshold until we remove N% of contigs
// This is useful when we know roughly how much to remove
inline GraphRefinementResult iterative_prune(
    const std::vector<std::array<float, TNF_DIM>>& tnf_vectors,
    const std::vector<size_t>& contig_indices,
    double max_removal_fraction = 0.25,  // Remove at most 25%
    double step = 0.005,                 // Threshold increment
    double start_threshold = 0.95
) {
    GraphRefinementResult result;
    size_t n = tnf_vectors.size();

    if (n < 10) {
        result.core_contigs = contig_indices;
        return result;
    }

    // Build complete graph
    std::vector<Edge> edges;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double sim = cosine_similarity(tnf_vectors[i], tnf_vectors[j]);
            edges.push_back({static_cast<int>(i), static_cast<int>(j), sim});
        }
    }
    std::sort(edges.begin(), edges.end());

    // Iteratively increase threshold
    double thresh = start_threshold;
    size_t max_removals = static_cast<size_t>(n * max_removal_fraction);

    while (thresh < 1.0) {
        auto components = get_components_at_threshold(edges, n, thresh);

        if (components.empty()) break;

        size_t core_size = components[0].size();
        size_t removals = n - core_size;

        if (removals >= max_removals || components.size() > 1) {
            // Found good threshold
            result.threshold_used = thresh;
            result.n_components = components.size();

            for (int idx : components[0]) {
                result.core_contigs.push_back(contig_indices[idx]);
            }
            for (size_t c = 1; c < components.size(); ++c) {
                for (int idx : components[c]) {
                    result.outlier_contigs.push_back(contig_indices[idx]);
                }
                result.component_sizes.push_back(components[c].size());
            }
            return result;
        }

        thresh += step;
    }

    // No good split found, return all as core
    result.core_contigs = contig_indices;
    result.threshold_used = 1.0;
    result.n_components = 1;
    return result;
}

}  // namespace graph_prune
}  // namespace amber

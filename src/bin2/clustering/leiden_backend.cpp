// Leiden clustering backend implementation
// Proper implementation with graph aggregation for efficiency
// Matches leidenalg's RBER (RBERVertexPartition) behavior

#include "leiden_backend.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <queue>

namespace amber::bin2 {

// ===========================================================================
// Graph structure for aggregation
// ===========================================================================

struct AggregateGraph {
    int n_nodes;
    std::vector<std::vector<std::pair<int, double>>> adj;  // neighbor, weight
    std::vector<double> node_weights;   // sum of edge weights per node
    std::vector<double> node_sizes;     // for RBER null model
    double total_edge_weight;
    double total_node_size;

    // Mapping from aggregate node to original nodes
    std::vector<std::vector<int>> members;
};

// ===========================================================================
// Louvain Backend (simpler, no refinement)
// ===========================================================================

LouvainBackend::LouvainBackend() : rng_(42) {}

void LouvainBackend::build_adjacency(
    const std::vector<WeightedEdge>& edges,
    int n_nodes) {

    n_nodes_ = n_nodes;
    adj_.clear();
    adj_.resize(n_nodes);
    node_weights_.assign(n_nodes, 0.0f);
    total_edge_weight_ = 0.0;

    for (const auto& e : edges) {
        if (e.u < 0 || e.u >= n_nodes || e.v < 0 || e.v >= n_nodes) {
            continue;
        }

        adj_[e.u].emplace_back(e.v, e.w);
        if (e.u != e.v) {
            adj_[e.v].emplace_back(e.u, e.w);
        }

        node_weights_[e.u] += e.w;
        node_weights_[e.v] += e.w;
        total_edge_weight_ += e.w;
    }
}

double LouvainBackend::modularity(
    const std::vector<int>& labels,
    float resolution) const {

    if (total_edge_weight_ == 0.0) return 0.0;

    std::unordered_map<int, double> comm_internal;
    std::unordered_map<int, double> comm_total;

    for (int i = 0; i < n_nodes_; ++i) {
        int ci = labels[i];
        comm_total[ci] += node_weights_[i];

        for (const auto& [j, w] : adj_[i]) {
            if (labels[j] == ci && j >= i) {
                comm_internal[ci] += w;
            }
        }
    }

    double Q = 0.0;
    double m2 = total_edge_weight_;

    for (const auto& [c, internal] : comm_internal) {
        double total = comm_total[c];
        Q += internal / m2 - resolution * (total * total) / (4.0 * m2 * m2);
    }

    return Q;
}

double LouvainBackend::delta_modularity(
    int node,
    int target_comm,
    const std::vector<int>& labels,
    float resolution) const {

    if (total_edge_weight_ == 0.0) return 0.0;

    int current_comm = labels[node];
    if (current_comm == target_comm) return 0.0;

    double ki = node_weights_[node];
    double m2 = total_edge_weight_;

    double ki_current = 0.0;
    double ki_target = 0.0;
    double sum_current = 0.0;
    double sum_target = 0.0;

    for (int j = 0; j < n_nodes_; ++j) {
        if (j == node) continue;
        if (labels[j] == current_comm) {
            sum_current += node_weights_[j];
        } else if (labels[j] == target_comm) {
            sum_target += node_weights_[j];
        }
    }

    for (const auto& [j, w] : adj_[node]) {
        if (labels[j] == current_comm) {
            ki_current += w;
        } else if (labels[j] == target_comm) {
            ki_target += w;
        }
    }

    double delta_out = -ki_current / m2 + resolution * ki * sum_current / (2.0 * m2 * m2);
    double delta_in = ki_target / m2 - resolution * ki * (sum_target + ki) / (2.0 * m2 * m2);

    return delta_in + delta_out;
}

int LouvainBackend::best_community(
    int node,
    const std::vector<int>& labels,
    float resolution) const {

    std::unordered_set<int> neighbor_comms;
    neighbor_comms.insert(labels[node]);

    for (const auto& [j, w] : adj_[node]) {
        neighbor_comms.insert(labels[j]);
    }

    int best = labels[node];
    double best_delta = 0.0;

    for (int c : neighbor_comms) {
        double delta = delta_modularity(node, c, labels, resolution);
        if (delta > best_delta) {
            best_delta = delta;
            best = c;
        }
    }

    return best;
}

bool LouvainBackend::local_move_phase(
    std::vector<int>& labels,
    float resolution) {

    std::vector<int> order(n_nodes_);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng_);

    bool improved = false;

    for (int node : order) {
        int best = best_community(node, labels, resolution);
        if (best != labels[node]) {
            labels[node] = best;
            improved = true;
        }
    }

    return improved;
}

int LouvainBackend::compact_labels(std::vector<int>& labels) const {
    std::unordered_map<int, int> mapping;
    int next_id = 0;

    for (int& label : labels) {
        auto it = mapping.find(label);
        if (it == mapping.end()) {
            mapping[label] = next_id;
            label = next_id++;
        } else {
            label = it->second;
        }
    }

    return next_id;
}

ClusteringResult LouvainBackend::cluster(
    const std::vector<WeightedEdge>& edges,
    int n_nodes,
    const LeidenConfig& config) {

    rng_.seed(config.random_seed);
    build_adjacency(edges, n_nodes);

    ClusteringResult result;
    result.labels.resize(n_nodes);

    if (config.use_initial_membership &&
        config.initial_membership.size() == static_cast<size_t>(n_nodes)) {
        result.labels = config.initial_membership;
    } else {
        std::iota(result.labels.begin(), result.labels.end(), 0);
    }

    int max_iter = config.max_iterations;
    if (max_iter < 0) max_iter = std::numeric_limits<int>::max();

    int iteration = 0;
    bool improved = true;

    while (improved && iteration < max_iter) {
        improved = local_move_phase(result.labels, config.resolution);
        ++iteration;
    }

    result.num_clusters = compact_labels(result.labels);
    result.modularity = static_cast<float>(modularity(result.labels, config.resolution));
    result.num_iterations = iteration;

    return result;
}

// ===========================================================================
// Leiden Backend with proper graph aggregation
// ===========================================================================

LeidenBackend::LeidenBackend() : rng_(42) {}

void LeidenBackend::build_adjacency(
    const std::vector<WeightedEdge>& edges,
    int n_nodes,
    const LeidenConfig& config) {

    n_nodes_ = n_nodes;
    null_model_ = config.null_model;
    adj_.clear();
    adj_.resize(n_nodes);
    node_weights_.assign(n_nodes, 0.0);
    total_edge_weight_ = 0.0;

    // Initialize node sizes
    if (config.use_node_sizes && config.node_sizes.size() == static_cast<size_t>(n_nodes)) {
        node_sizes_ = config.node_sizes;
    } else {
        node_sizes_.assign(n_nodes, 1.0);
    }
    total_node_size_ = 0.0;
    for (double s : node_sizes_) {
        total_node_size_ += s;
    }

    // Initialize fixed membership flags
    if (config.use_fixed_membership && config.is_membership_fixed.size() == static_cast<size_t>(n_nodes)) {
        is_fixed_ = config.is_membership_fixed;
    } else {
        is_fixed_.assign(n_nodes, false);
    }

    for (const auto& e : edges) {
        if (e.u < 0 || e.u >= n_nodes || e.v < 0 || e.v >= n_nodes) {
            continue;
        }

        adj_[e.u].emplace_back(e.v, static_cast<double>(e.w));
        if (e.u != e.v) {
            adj_[e.v].emplace_back(e.u, static_cast<double>(e.w));
        }

        node_weights_[e.u] += e.w;
        node_weights_[e.v] += e.w;
        total_edge_weight_ += e.w;
    }
}

// Fast delta modularity using precomputed community statistics
// Port of leidenalg's RBERVertexPartition::diff_move()
double LeidenBackend::delta_modularity_fast(
    int node,
    int current_comm,
    int target_comm,
    double edges_to_current,
    double edges_to_target,
    const std::vector<double>& comm_weights,
    const std::vector<double>& comm_sizes,
    float resolution) const {

    if (current_comm == target_comm) return 0.0;

    if (null_model_ == NullModel::ErdosRenyi) {
        // RBER null model (Erdős-Rényi null model with resolution parameter)
        //
        // CRITICAL: Python leidenalg computes density using vertex COUNT, not sum of node_sizes!
        // Graph has default node_sizes (all 1s), so density = 2*w / (n * (n-1))
        // But csize uses custom node_sizes (contig lengths).
        //
        // diff_move formula from leidenalg RBERVertexPartition.cpp:
        //   diff = (w_to_new + w_from_new + sw) - (w_to_old + w_from_old - sw)
        //          - resolution * density * (possible_edge_diff_new - possible_edge_diff_old)
        //
        // where possible_edge_diff = nsize * (2*csize + nsize) for new comm
        //       possible_edge_diff = nsize * (2*csize - nsize) for old comm

        if (n_nodes_ <= 1) return 0.0;
        double m = total_edge_weight_ / 2.0;
        if (m == 0.0) return 0.0;

        // Density uses VERTEX COUNT (like Python leidenalg with default node_sizes)
        double density_denom = static_cast<double>(n_nodes_) * (n_nodes_ - 1.0);
        double density = (2.0 * m) / density_denom;

        // Community sizes use CUSTOM node_sizes (contig lengths)
        double s_i = node_sizes_[node];
        double csize_old = comm_sizes[current_comm];
        double csize_new = comm_sizes[target_comm];

        // possible_edge_difference (without self-loops)
        double ped_old = s_i * (2.0 * csize_old - s_i - 1.0);
        double ped_new = s_i * (2.0 * csize_new + s_i - 1.0);

        // Edge contribution (undirected: w_to + w_from = 2*edges_to)
        double diff_old = 2.0 * edges_to_current - resolution * density * ped_old;
        double diff_new = 2.0 * edges_to_target + resolution * density * ped_new;

        return diff_new - diff_old;
    } else {
        // Configuration model
        if (total_edge_weight_ == 0.0) return 0.0;
        double ki = node_weights_[node];
        double m2 = total_edge_weight_;

        double sum_current = comm_weights[current_comm] - ki;
        double sum_target = comm_weights[target_comm];

        return (edges_to_target - edges_to_current) / m2 +
               resolution * ki * (sum_current - sum_target - ki) / (2.0 * m2 * m2);
    }
}

// Single pass of local moving with early termination
bool LeidenBackend::move_nodes_pass(
    std::vector<int>& labels,
    std::vector<double>& comm_weights,
    std::vector<double>& comm_sizes,
    const std::vector<int>& order,
    float resolution) {

    bool any_moved = false;

    for (int node : order) {
        if (is_fixed_[node]) continue;

        int current = labels[node];

        // Compute edges to each neighbor community
        std::unordered_map<int, double> edges_to_comm;
        edges_to_comm[current] = 0.0;

        for (const auto& [j, w] : adj_[node]) {
            edges_to_comm[labels[j]] += w;
        }

        // Remove self-loops from current community count
        double edges_to_current = edges_to_comm[current];

        int best = current;
        double best_delta = 0.0;

        for (const auto& [c, edges_to_c] : edges_to_comm) {
            if (c == current) continue;

            double delta = delta_modularity_fast(
                node, current, c,
                edges_to_current, edges_to_c,
                comm_weights, comm_sizes, resolution);

            if (delta > best_delta) {
                best_delta = delta;
                best = c;
            }
        }

        if (best != current) {
            // Update community statistics
            comm_weights[current] -= node_weights_[node];
            comm_weights[best] += node_weights_[node];
            comm_sizes[current] -= node_sizes_[node];
            comm_sizes[best] += node_sizes_[node];
            labels[node] = best;
            any_moved = true;
        }
    }

    return any_moved;
}

// Local moving phase - iterate until convergence
bool LeidenBackend::move_nodes_fast(
    std::vector<int>& labels,
    std::vector<double>& comm_weights,
    std::vector<double>& comm_sizes,
    float resolution) {

    // leidenalg uses a dynamic queue: when a node moves, its neighbors get re-queued
    // This continues until no more improvements can be made

    // Track which nodes are in queue
    std::vector<bool> in_queue(n_nodes_, true);

    // Initial queue: all nodes in random order
    std::deque<int> queue;
    std::vector<int> order(n_nodes_);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng_);
    for (int node : order) {
        queue.push_back(node);
    }

    bool any_moved = false;

    while (!queue.empty()) {
        int node = queue.front();
        queue.pop_front();
        in_queue[node] = false;

        if (is_fixed_[node]) continue;

        int current = labels[node];

        // Compute edges to each neighbor community
        std::unordered_map<int, double> edges_to_comm;
        edges_to_comm[current] = 0.0;

        for (const auto& [j, w] : adj_[node]) {
            edges_to_comm[labels[j]] += w;
        }

        double edges_to_current = edges_to_comm[current];

        int best = current;
        double best_delta = 0.0;

        for (const auto& [c, edges_to_c] : edges_to_comm) {
            if (c == current) continue;

            double delta = delta_modularity_fast(
                node, current, c,
                edges_to_current, edges_to_c,
                comm_weights, comm_sizes, resolution);

            if (delta > best_delta) {
                best_delta = delta;
                best = c;
            }
        }

        if (best != current) {
            // Move the node
            comm_weights[current] -= node_weights_[node];
            comm_weights[best] += node_weights_[node];
            comm_sizes[current] -= node_sizes_[node];
            comm_sizes[best] += node_sizes_[node];
            labels[node] = best;
            any_moved = true;

            // Re-queue neighbors that aren't in the new community
            for (const auto& [j, w] : adj_[node]) {
                if (!in_queue[j] && labels[j] != best) {
                    queue.push_back(j);
                    in_queue[j] = true;
                }
            }
        }
    }

    return any_moved;
}

// Create aggregate graph from current partition
AggregateGraph LeidenBackend::aggregate_graph(
    const std::vector<int>& labels,
    int n_communities) const {

    AggregateGraph agg;
    agg.n_nodes = n_communities;
    agg.adj.resize(n_communities);
    agg.node_weights.assign(n_communities, 0.0);
    agg.node_sizes.assign(n_communities, 0.0);
    agg.members.resize(n_communities);
    agg.total_edge_weight = 0.0;
    agg.total_node_size = 0.0;

    // Build member lists and aggregate node sizes
    for (int i = 0; i < n_nodes_; ++i) {
        int c = labels[i];
        agg.members[c].push_back(i);
        agg.node_sizes[c] += node_sizes_[i];
    }
    for (double s : agg.node_sizes) {
        agg.total_node_size += s;
    }

    // Build aggregate edges using hash map for efficiency
    std::vector<std::unordered_map<int, double>> edge_map(n_communities);

    for (int i = 0; i < n_nodes_; ++i) {
        int ci = labels[i];
        for (const auto& [j, w] : adj_[i]) {
            int cj = labels[j];
            if (ci <= cj) {  // Count each edge once
                edge_map[ci][cj] += w;
            }
        }
    }

    // Convert to adjacency list
    for (int ci = 0; ci < n_communities; ++ci) {
        for (const auto& [cj, w] : edge_map[ci]) {
            if (ci == cj) {
                // Self-loop (internal edges)
                agg.adj[ci].emplace_back(ci, w);
                agg.node_weights[ci] += w;
                agg.total_edge_weight += w;
            } else {
                // Inter-community edge
                agg.adj[ci].emplace_back(cj, w);
                agg.adj[cj].emplace_back(ci, w);
                agg.node_weights[ci] += w;
                agg.node_weights[cj] += w;
                agg.total_edge_weight += 2.0 * w;
            }
        }
    }

    return agg;
}

// Run Leiden on aggregate graph
std::vector<int> LeidenBackend::cluster_aggregate(
    const AggregateGraph& agg,
    float resolution) {

    if (agg.n_nodes == 0) return {};

    // Initialize: each aggregate node in its own community
    std::vector<int> labels(agg.n_nodes);
    std::iota(labels.begin(), labels.end(), 0);

    std::vector<double> comm_weights = agg.node_weights;
    std::vector<double> comm_sizes = agg.node_sizes;

    // Save original graph state
    auto orig_adj = adj_;
    auto orig_node_weights = node_weights_;
    auto orig_node_sizes = node_sizes_;
    auto orig_is_fixed = is_fixed_;
    int orig_n_nodes = n_nodes_;
    double orig_total_edge_weight = total_edge_weight_;
    double orig_total_node_size = total_node_size_;

    // Temporarily set graph to aggregate
    n_nodes_ = agg.n_nodes;
    adj_.clear();
    adj_.resize(agg.n_nodes);
    for (int i = 0; i < agg.n_nodes; ++i) {
        for (const auto& [j, w] : agg.adj[i]) {
            adj_[i].emplace_back(j, w);
        }
    }
    node_weights_ = agg.node_weights;
    node_sizes_ = agg.node_sizes;
    is_fixed_.assign(agg.n_nodes, false);
    total_edge_weight_ = agg.total_edge_weight;
    total_node_size_ = agg.total_node_size;

    // Move nodes on aggregate graph - single pass like leidenalg
    std::vector<int> order(agg.n_nodes);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng_);
    move_nodes_pass(labels, comm_weights, comm_sizes, order, resolution);

    // Restore original graph
    adj_ = std::move(orig_adj);
    node_weights_ = std::move(orig_node_weights);
    node_sizes_ = std::move(orig_node_sizes);
    is_fixed_ = std::move(orig_is_fixed);
    n_nodes_ = orig_n_nodes;
    total_edge_weight_ = orig_total_edge_weight;
    total_node_size_ = orig_total_node_size;

    return labels;
}

// Refinement phase: allow nodes to move within their aggregate community
void LeidenBackend::refine_partition(
    std::vector<int>& labels,
    const std::vector<int>& aggregate_membership,
    std::vector<double>& comm_weights,
    std::vector<double>& comm_sizes,
    float resolution) {

    // Group nodes by their aggregate community
    int n_agg = *std::max_element(aggregate_membership.begin(), aggregate_membership.end()) + 1;
    std::vector<std::vector<int>> subgraphs(n_agg);
    for (int i = 0; i < n_nodes_; ++i) {
        subgraphs[aggregate_membership[i]].push_back(i);
    }

    // Build mapping: community label -> aggregate membership
    // (updated dynamically as nodes move)
    std::unordered_map<int, int> comm_to_agg;
    for (int i = 0; i < n_nodes_; ++i) {
        comm_to_agg[labels[i]] = aggregate_membership[i];
    }

    // Refine within each aggregate
    for (const auto& nodes : subgraphs) {
        if (nodes.size() <= 1) continue;

        int my_agg = aggregate_membership[nodes[0]];  // All nodes in subgraph have same agg

        bool moved = true;
        int pass = 0;
        const int max_passes = 10;  // Limited passes per subgraph

        while (moved && pass < max_passes) {
            moved = false;
            std::vector<int> local_order = nodes;
            std::shuffle(local_order.begin(), local_order.end(), rng_);

            for (int node : local_order) {
                if (is_fixed_[node]) continue;

                int current = labels[node];

                // Compute edges to ALL communities (for correct delta calculation)
                std::unordered_map<int, double> edges_to_comm;
                edges_to_comm[current] = 0.0;

                for (const auto& [j, w] : adj_[node]) {
                    edges_to_comm[labels[j]] += w;  // Count ALL edges
                }

                double edges_to_current = edges_to_comm[current];
                int best = current;
                double best_delta = 0.0;

                // Only consider communities within same aggregate as targets
                for (const auto& [c, edges_to_c] : edges_to_comm) {
                    if (c == current) continue;

                    // Check if target community is in same aggregate using O(1) lookup
                    auto it = comm_to_agg.find(c);
                    if (it == comm_to_agg.end() || it->second != my_agg) continue;

                    double delta = delta_modularity_fast(
                        node, current, c,
                        edges_to_current, edges_to_c,
                        comm_weights, comm_sizes, resolution);

                    if (delta > best_delta + 1e-12) {
                        best_delta = delta;
                        best = c;
                    }
                }

                if (best != current) {
                    comm_weights[current] -= node_weights_[node];
                    comm_weights[best] += node_weights_[node];
                    comm_sizes[current] -= node_sizes_[node];
                    comm_sizes[best] += node_sizes_[node];
                    labels[node] = best;
                    // Update comm_to_agg for the new community membership
                    comm_to_agg[best] = my_agg;
                    moved = true;
                }
            }
            ++pass;
        }
    }
}

int LeidenBackend::compact_labels(std::vector<int>& labels) const {
    std::unordered_map<int, int> mapping;
    int next_id = 0;

    for (int& label : labels) {
        auto it = mapping.find(label);
        if (it == mapping.end()) {
            mapping[label] = next_id;
            label = next_id++;
        } else {
            label = it->second;
        }
    }

    return next_id;
}

// Compute RBER modularity (matching Python leidenalg)
double LeidenBackend::modularity_rber(
    const std::vector<int>& labels,
    float resolution) const {

    if (total_edge_weight_ == 0.0) return 0.0;
    if (n_nodes_ <= 1) return 0.0;

    // Compute community sizes and internal edges
    int max_label = *std::max_element(labels.begin(), labels.end()) + 1;
    std::vector<double> comm_sizes(max_label, 0.0);
    std::vector<double> comm_internal(max_label, 0.0);

    for (int i = 0; i < n_nodes_; ++i) {
        comm_sizes[labels[i]] += node_sizes_[i];
        for (const auto& [j, w] : adj_[i]) {
            if (labels[j] == labels[i] && j > i) {
                comm_internal[labels[i]] += w;
            }
        }
    }

    // CRITICAL: Use vertex count for density, like Python leidenalg
    double m = total_edge_weight_ / 2.0;
    double density_denom = static_cast<double>(n_nodes_) * (n_nodes_ - 1.0);
    double density = (2.0 * m) / density_denom;

    // Quality = sum over communities of: w_c - resolution * density * possible_edges(n_c)
    // where possible_edges(n_c) = n_c * (n_c - 1) / 2 for undirected
    // and w_c is internal edge weight
    double Q = 0.0;
    for (int c = 0; c < max_label; ++c) {
        if (comm_sizes[c] > 0) {
            double n_c = comm_sizes[c];
            double possible = n_c * (n_c - 1.0) / 2.0;  // undirected possible edges
            Q += comm_internal[c] - resolution * density * possible;
        }
    }

    // Return normalized quality (factor of 2 for undirected)
    return 2.0 * Q;
}

// Compute configuration model modularity
double LeidenBackend::modularity_config(
    const std::vector<int>& labels,
    float resolution) const {

    if (total_edge_weight_ == 0.0) return 0.0;

    int max_label = *std::max_element(labels.begin(), labels.end()) + 1;
    std::vector<double> comm_total(max_label, 0.0);
    std::vector<double> comm_internal(max_label, 0.0);

    for (int i = 0; i < n_nodes_; ++i) {
        comm_total[labels[i]] += node_weights_[i];
        for (const auto& [j, w] : adj_[i]) {
            if (labels[j] == labels[i] && j > i) {
                comm_internal[labels[i]] += w;
            }
        }
    }

    double Q = 0.0;
    double m2 = total_edge_weight_;

    for (int c = 0; c < max_label; ++c) {
        double total = comm_total[c];
        Q += comm_internal[c] / m2 - resolution * (total * total) / (4.0 * m2 * m2);
    }

    return Q;
}

ClusteringResult LeidenBackend::cluster(
    const std::vector<WeightedEdge>& edges,
    int n_nodes,
    const LeidenConfig& config) {

    rng_.seed(config.random_seed);
    build_adjacency(edges, n_nodes, config);

    ClusteringResult result;
    result.labels.resize(n_nodes);

    // Initialize membership
    if (config.use_initial_membership &&
        config.initial_membership.size() == static_cast<size_t>(n_nodes)) {
        result.labels = config.initial_membership;
    } else {
        std::iota(result.labels.begin(), result.labels.end(), 0);
    }

    // Initialize community statistics
    int max_comm = n_nodes;
    std::vector<double> comm_weights(max_comm, 0.0);
    std::vector<double> comm_sizes(max_comm, 0.0);

    for (int i = 0; i < n_nodes; ++i) {
        int c = result.labels[i];
        comm_weights[c] += node_weights_[i];
        comm_sizes[c] += node_sizes_[i];
    }

    int max_iter = config.max_iterations;
    // COMEBin uses max_iter=-1 (unlimited), we cap at 1000 for safety
    max_iter = (max_iter < 0) ? 1000 : max_iter;

    int iteration = 0;
    bool improved = true;

    while (improved && iteration < max_iter) {
        // Phase 1: Local moving
        improved = move_nodes_fast(result.labels, comm_weights, comm_sizes, config.resolution);

        if (!improved) break;

        // Compact labels after local moving
        int n_communities = compact_labels(result.labels);

        // Phase 2: Aggregate graph
        AggregateGraph agg = aggregate_graph(result.labels, n_communities);

        // If no aggregation possible (each community is a single node), we're done
        if (agg.n_nodes == n_nodes) break;

        // Phase 3: Cluster aggregate graph
        std::vector<int> agg_labels = cluster_aggregate(agg, config.resolution);
        int n_agg_communities = compact_labels(agg_labels);

        // If aggregate didn't improve, we're done
        if (n_agg_communities == n_communities) break;

        // Phase 4: Map aggregate labels back to original nodes
        // This creates the "refined" partition
        std::vector<int> aggregate_membership(n_nodes);  // Which aggregate comm each node belongs to
        for (int i = 0; i < n_nodes; ++i) {
            aggregate_membership[i] = agg_labels[result.labels[i]];
        }

        // Split into singletons for refinement
        int next_label = n_agg_communities;
        for (int i = 0; i < n_nodes; ++i) {
            result.labels[i] = next_label++;
        }

        // Resize comm vectors
        comm_weights.assign(next_label, 0.0);
        comm_sizes.assign(next_label, 0.0);
        for (int i = 0; i < n_nodes; ++i) {
            comm_weights[result.labels[i]] = node_weights_[i];
            comm_sizes[result.labels[i]] = node_sizes_[i];
        }

        // Refine: nodes can only merge within their aggregate community
        refine_partition(result.labels, aggregate_membership, comm_weights, comm_sizes, config.resolution);

        // Compact and update stats
        n_communities = compact_labels(result.labels);
        comm_weights.assign(n_communities, 0.0);
        comm_sizes.assign(n_communities, 0.0);
        for (int i = 0; i < n_nodes; ++i) {
            comm_weights[result.labels[i]] += node_weights_[i];
            comm_sizes[result.labels[i]] += node_sizes_[i];
        }

        ++iteration;
    }

    result.num_clusters = compact_labels(result.labels);

    // Final modularity
    if (null_model_ == NullModel::ErdosRenyi) {
        result.modularity = static_cast<float>(modularity_rber(result.labels, config.resolution));
    } else {
        result.modularity = static_cast<float>(modularity_config(result.labels, config.resolution));
    }
    result.num_iterations = iteration;

    return result;
}

#ifdef USE_LIBLEIDENALG
// Implementation using libleidenalg's RBERVertexPartition
// This matches Python leidenalg's behavior (RBER null model, not CPM)

}  // temporarily close namespace amber::bin2

#include <igraph/igraph.h>
#include <GraphHelper.h>
#include <RBERVertexPartition.h>
#include <Optimiser.h>

namespace amber::bin2 {  // reopen namespace

struct LibLeidenalgBackend::Impl {
    igraph_t igraph;
    bool igraph_created = false;

    ~Impl() {
        if (igraph_created) {
            igraph_destroy(&igraph);
        }
    }
};

LibLeidenalgBackend::LibLeidenalgBackend() : impl_(std::make_unique<Impl>()) {}
LibLeidenalgBackend::~LibLeidenalgBackend() = default;

ClusteringResult LibLeidenalgBackend::cluster(
    const std::vector<WeightedEdge>& edges,
    int n_nodes,
    const LeidenConfig& config) {


    // Clean up previous graph
    if (impl_->igraph_created) {
        igraph_destroy(&impl_->igraph);
        impl_->igraph_created = false;
    }

    // Build igraph edge list
    igraph_vector_int_t edge_vec;
    igraph_vector_int_init(&edge_vec, edges.size() * 2);
    for (size_t i = 0; i < edges.size(); ++i) {
        VECTOR(edge_vec)[2*i] = edges[i].u;
        VECTOR(edge_vec)[2*i + 1] = edges[i].v;
    }
    igraph_create(&impl_->igraph, &edge_vec, n_nodes, IGRAPH_UNDIRECTED);
    igraph_vector_int_destroy(&edge_vec);
    impl_->igraph_created = true;

    // Build edge weights
    std::vector<double> edge_weights(edges.size());
    for (size_t i = 0; i < edges.size(); ++i) {
        edge_weights[i] = edges[i].w;
    }

    // Build node sizes (contig lengths)
    std::vector<size_t> node_sizes(n_nodes);
    if (!config.node_sizes.empty()) {
        for (int i = 0; i < n_nodes; ++i) {
            node_sizes[i] = static_cast<size_t>(config.node_sizes[i]);
        }
        std::cerr << "[libleidenalg RBER] Using node_sizes (contig lengths)" << std::endl;
    } else {
        std::fill(node_sizes.begin(), node_sizes.end(), 1);
    }

    // Create libleidenalg Graph with node_sizes
    // NOTE: With our patch to GraphHelper.cpp, density uses vertex count, not sum of node_sizes
    Graph graph(&impl_->igraph, edge_weights, node_sizes);

    // Initial membership: always start with singletons [0, 1, 2, ..., n-1]
    // Seed constraints are handled via is_fixed below, NOT via initial_membership.
    // Using config.initial_membership causes libleidenalg to not merge clusters properly
    // (regression identified: 26798 singletons instead of ~384 merged clusters).
    std::vector<size_t> initial_membership(n_nodes);
    std::iota(initial_membership.begin(), initial_membership.end(), 0);

    // Create RBERVertexPartition (Erdős-Rényi null model - same as COMEBin)
    RBERVertexPartition partition(&graph, initial_membership, config.resolution);

    // Set up fixed membership for seeds
    std::vector<bool> is_fixed(n_nodes, false);
    if (!config.is_membership_fixed.empty()) {
        is_fixed = config.is_membership_fixed;
        int n_fixed = std::count(is_fixed.begin(), is_fixed.end(), true);
        std::cerr << "[libleidenalg RBER] " << n_fixed << " fixed seeds" << std::endl;
    }

    // Run Leiden optimization
    Optimiser optimiser;
    optimiser.set_rng_seed(config.random_seed);

    int n_iter = config.max_iterations;
    if (n_iter < 0) n_iter = 100;  // default max iterations

    double improvement = 0.0;
    for (int iter = 0; iter < n_iter; ++iter) {
        double delta = optimiser.optimise_partition(&partition, is_fixed);
        improvement += delta;
        if (delta < 1e-10) break;  // converged
    }

    double quality = partition.quality(config.resolution);
    std::cerr << "[libleidenalg RBER] Found " << partition.n_communities()
              << " clusters, quality=" << quality << std::endl;

    // Extract results
    ClusteringResult result;
    result.labels.resize(n_nodes);
    auto membership = partition.membership();
    for (int i = 0; i < n_nodes; ++i) {
        result.labels[i] = static_cast<int>(membership[i]);
    }

    // Compact labels
    std::unordered_map<int, int> label_map;
    int next_label = 0;
    for (int& lbl : result.labels) {
        auto it = label_map.find(lbl);
        if (it == label_map.end()) {
            label_map[lbl] = next_label;
            lbl = next_label++;
        } else {
            lbl = it->second;
        }
    }

    result.num_clusters = next_label;
    result.modularity = static_cast<float>(quality);
    result.num_iterations = n_iter;

    return result;
}

#endif  // USE_LIBLEIDENALG

std::unique_ptr<ILeidenBackend> create_leiden_backend(bool use_leiden) {
#ifdef USE_LIBLEIDENALG
    // Use libleidenalg's RBERVertexPartition - achieves ARI=0.9242 vs COMEBin
    return std::make_unique<LibLeidenalgBackend>();
#else
    if (use_leiden) {
        return std::make_unique<LeidenBackend>();
    } else {
        return std::make_unique<LouvainBackend>();
    }
#endif
}

}  // namespace amber::bin2

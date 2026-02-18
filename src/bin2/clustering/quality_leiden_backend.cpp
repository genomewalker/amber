// AMBER - quality_leiden_backend.cpp
// Quality-weighted Leiden clustering implementation

#include "quality_leiden_backend.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <cmath>
#include <deque>
#include <thread>
#include <omp.h>

namespace amber::bin2 {

ClusteringResult QualityLeidenBackend::cluster(const std::vector<WeightedEdge>& edges,
                                             int n_nodes,
                                             const LeidenConfig& config) {
    if (!marker_index_) {
        std::cerr << "[QualityLeiden] WARNING: No marker index, falling back to pure modularity\n";
        return LeidenBackend::cluster(edges, n_nodes, config);
    }

    if (marker_index_->max_marker_id() >= MAX_MARKER_TYPES) {
        throw std::runtime_error("Too many marker types: " +
            std::to_string(marker_index_->max_marker_id()) +
            " > " + std::to_string(MAX_MARKER_TYPES));
    }

    int n_threads = qconfig_.n_threads;
    if (n_threads <= 0)
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads < 1) n_threads = 1;

    std::cerr << "[QualityLeiden-MT] Starting quality-weighted clustering (" << n_threads << " threads)\n";
    std::cerr << "[QualityLeiden-MT] alpha=" << qconfig_.alpha
              << ", penalty=" << qconfig_.contamination_penalty << "\n";

    // Step 1: Run libleidenalg for initial clustering.
    // Pre-process edges: penalize connections between contigs sharing SCG markers.
    // Contigs sharing the same single-copy gene belong to different genomes, so their
    // edge weights are scaled down by exp(-penalty * n_shared_markers). This biases
    // libleidenalg away from contamination-creating merges without modifying libleidenalg.
    std::cerr << "[QualityLeiden-MT] Phase 1: libleidenalg for initial clustering...\n";
    std::vector<WeightedEdge> phase1_edges;
    if (qconfig_.marker_edge_penalty > 0.0f) {
        phase1_edges.reserve(edges.size());
        int penalized = 0;
        for (const auto& e : edges) {
            const auto& mu = marker_index_->get_markers(e.u);
            const auto& mv = marker_index_->get_markers(e.v);
            int shared = 0;
            if (!mu.empty() && !mv.empty()) {
                for (const auto& a : mu) {
                    for (const auto& b : mv) {
                        if (a.id == b.id) { shared++; break; }
                    }
                }
            }
            float w = e.w;
            if (shared > 0) {
                w *= std::exp(-qconfig_.marker_edge_penalty * shared);
                penalized++;
            }
            phase1_edges.push_back({e.u, e.v, w});
        }
        if (penalized > 0) {
            std::cerr << "[QualityLeiden-MT] Penalized " << penalized
                      << " edges with shared SCG markers\n";
        }
    }
    const std::vector<WeightedEdge>& p1_edges = phase1_edges.empty() ? edges : phase1_edges;

#ifdef USE_LIBLEIDENALG
    LibLeidenalgBackend fast_backend;
    ClusteringResult initial_result = fast_backend.cluster(p1_edges, n_nodes, config);
#else
    ClusteringResult initial_result = LeidenBackend::cluster(p1_edges, n_nodes, config);
#endif
    std::cerr << "[QualityLeiden-MT] Initial clustering: " << initial_result.num_clusters
              << " clusters, modularity=" << initial_result.modularity << "\n";

    // Use the initial clustering labels
    std::vector<int> labels = initial_result.labels;
    int n_communities = initial_result.num_clusters;

    // Build adjacency for refinement
    build_adjacency(edges, n_nodes, config);

    // Build has_marker_ lookup from marker_index
    has_marker_.resize(n_nodes_);
    for (int i = 0; i < n_nodes_; ++i) {
        has_marker_[i] = marker_index_->has_markers(i) ? 1 : 0;
    }

    // Initialize community statistics
    std::vector<double> comm_weights(n_communities, 0.0);
    std::vector<double> comm_sizes(n_communities, 0.0);
    for (int i = 0; i < n_nodes_; ++i) {
        int c = labels[i];
        comm_weights[c] += node_weights_[i];
        comm_sizes[c] += node_sizes_[i];
    }

    // Initialize dense community quality counts
    std::vector<CommQuality> comm_q;
    init_comm_quality(labels, n_communities, comm_q);
    std::cerr << "[QualityLeiden-MT] " << marker_index_->num_marker_contigs()
              << " contigs have markers across " << n_communities << " initial clusters\n";

    (void)comm_weights;
    (void)comm_sizes;

    // Phase 2: Quality-guided contamination splitting.
    // For each cluster with duplicate markers (dup_excess > 0), re-run libleidenalg
    // on the cluster's subgraph at higher resolution. Accept splits that reduce
    // contamination (total dup_excess decreases). Uses libleidenalg to avoid the
    // objective mismatch that caused the old local-move Phase 2 to pathologically merge.
    {
        // Index nodes by cluster
        std::vector<std::vector<int>> cluster_nodes(n_communities);
        for (int i = 0; i < n_nodes; ++i) {
            cluster_nodes[labels[i]].push_back(i);
        }

        int n_contaminated = 0;
        int n_split = 0;

        for (int c = 0; c < n_communities; ++c) {
            if (comm_q[c].dup_excess == 0) continue;
            const auto& nodes = cluster_nodes[c];
            if ((int)nodes.size() <= 1) continue;
            n_contaminated++;

            // Build local ID map and extract subgraph edges
            std::unordered_map<int, int> local_id;
            local_id.reserve(nodes.size());
            for (int i = 0; i < (int)nodes.size(); ++i) {
                local_id[nodes[i]] = i;
            }

            std::vector<WeightedEdge> sub_edges;
            for (int v : nodes) {
                for (const auto& [u, w] : adj_[v]) {
                    auto it = local_id.find(u);
                    if (it != local_id.end() && u > v) {
                        sub_edges.push_back({local_id[v], it->second, static_cast<float>(w)});
                    }
                }
            }
            if (sub_edges.empty()) continue;

            // Re-cluster at 3x resolution to force finer splits
            LeidenConfig sub_config = config;
            sub_config.resolution = config.resolution * 3.0f;
            sub_config.use_initial_membership = false;
            sub_config.initial_membership.clear();
            sub_config.use_fixed_membership = false;
            sub_config.is_membership_fixed.clear();
            if (config.use_node_sizes) {
                sub_config.node_sizes.resize(nodes.size());
                for (int i = 0; i < (int)nodes.size(); ++i) {
                    sub_config.node_sizes[i] = node_sizes_[nodes[i]];
                }
            }

#ifdef USE_LIBLEIDENALG
            LibLeidenalgBackend sub_leiden;
            ClusteringResult sub_result = sub_leiden.cluster(sub_edges, nodes.size(), sub_config);
#else
            LeidenBackend sub_leiden;
            ClusteringResult sub_result = sub_leiden.cluster(sub_edges, nodes.size(), sub_config);
#endif
            if (sub_result.num_clusters <= 1) continue;

            // Compute dup_excess for each new sub-cluster
            std::vector<CommQuality> sub_q(sub_result.num_clusters);
            for (int i = 0; i < (int)nodes.size(); ++i) {
                int sc = sub_result.labels[i];
                for (const auto& entry : marker_index_->get_markers(nodes[i])) {
                    int m = entry.id;
                    int k = entry.copy_count;
                    if (sub_q[sc].cnt[m] == 0) sub_q[sc].present++;
                    sub_q[sc].cnt[m] += k;
                }
            }
            int new_dup_excess = 0;
            for (auto& cq : sub_q) {
                for (int m = 0; m < MAX_MARKER_TYPES; ++m) {
                    if (cq.cnt[m] > 1) new_dup_excess += cq.cnt[m] - 1;
                }
            }

            if (new_dup_excess >= comm_q[c].dup_excess) continue;  // no improvement

            // Accept split: sub-cluster 0 keeps label c, rest get new labels
            std::vector<int> sc_to_global(sub_result.num_clusters);
            sc_to_global[0] = c;
            for (int sc = 1; sc < sub_result.num_clusters; ++sc) {
                sc_to_global[sc] = n_communities + sc - 1;
            }
            for (int i = 0; i < (int)nodes.size(); ++i) {
                labels[nodes[i]] = sc_to_global[sub_result.labels[i]];
            }
            n_communities += sub_result.num_clusters - 1;
            n_split++;

            std::cerr << "[QualityLeiden] Split cluster " << c << " (" << nodes.size()
                      << " nodes, dup_excess " << comm_q[c].dup_excess
                      << " -> " << new_dup_excess << ") into "
                      << sub_result.num_clusters << " parts\n";
        }

        std::cerr << "[QualityLeiden-MT] Phase 2: " << n_contaminated
                  << " contaminated clusters, " << n_split << " split, "
                  << n_communities << " total\n";
    }

    // Compact labels and compute final modularity
    n_communities = compact_labels(labels);
    double modularity = initial_result.modularity;

    ClusteringResult result;
    result.labels = labels;
    result.modularity = modularity;
    result.num_clusters = n_communities;

    std::cerr << "[QualityLeiden-MT] Final: " << result.num_clusters
              << " clusters, modularity=" << modularity << "\n";

    return result;
}

bool QualityLeidenBackend::move_nodes_fast_quality(
    std::vector<int>& labels,
    std::vector<double>& comm_weights,
    std::vector<double>& comm_sizes,
    std::vector<std::unordered_map<int, int>>& comm_markers,
    float resolution,
    float lambda) {

    std::vector<bool> in_queue(n_nodes_, true);
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

            // Modularity delta
            double mod_delta = delta_modularity_fast(
                node, current, c,
                edges_to_current, edges_to_c,
                comm_weights, comm_sizes, resolution);

            // Quality delta
            auto q_delta = compute_quality_delta(node, current, c, *marker_index_, comm_markers);
            double quality_delta = q_delta.total(qconfig_.contamination_penalty);

            // Combined delta
            double total_delta = mod_delta + lambda * quality_delta;

            if (total_delta > best_delta) {
                best_delta = total_delta;
                best = c;
            }
        }

        if (best != current) {
            // Update community weights/sizes
            comm_weights[current] -= node_weights_[node];
            comm_weights[best] += node_weights_[node];
            comm_sizes[current] -= node_sizes_[node];
            comm_sizes[best] += node_sizes_[node];

            // Update community markers
            update_comm_markers(node, current, best, comm_markers);

            labels[node] = best;
            any_moved = true;

            // Re-queue neighbors
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

void QualityLeidenBackend::refine_partition_quality(
    std::vector<int>& labels,
    const std::vector<int>& aggregate_membership,
    std::vector<double>& comm_weights,
    std::vector<double>& comm_sizes,
    std::vector<std::unordered_map<int, int>>& comm_markers,
    float resolution,
    float lambda) {

    // Refine within each aggregate community
    int n_agg = *std::max_element(aggregate_membership.begin(), aggregate_membership.end()) + 1;

    for (int agg_c = 0; agg_c < n_agg; ++agg_c) {
        // Collect nodes in this aggregate community
        std::vector<int> nodes_in_agg;
        for (int i = 0; i < n_nodes_; ++i) {
            if (aggregate_membership[i] == agg_c) {
                nodes_in_agg.push_back(i);
            }
        }

        if (nodes_in_agg.size() <= 1) continue;

        // Local refinement within this aggregate
        std::shuffle(nodes_in_agg.begin(), nodes_in_agg.end(), rng_);

        for (int node : nodes_in_agg) {
            if (is_fixed_[node]) continue;

            int current = labels[node];

            // Only consider communities within same aggregate
            std::unordered_map<int, double> edges_to_comm;
            edges_to_comm[current] = 0.0;

            for (const auto& [j, w] : adj_[node]) {
                if (aggregate_membership[j] == agg_c) {
                    edges_to_comm[labels[j]] += w;
                }
            }

            double edges_to_current = edges_to_comm[current];

            int best = current;
            double best_delta = 0.0;

            for (const auto& [c, edges_to_c] : edges_to_comm) {
                if (c == current) continue;

                double mod_delta = delta_modularity_fast(
                    node, current, c,
                    edges_to_current, edges_to_c,
                    comm_weights, comm_sizes, resolution);

                auto q_delta = compute_quality_delta(node, current, c, *marker_index_, comm_markers);
                double quality_delta = q_delta.total(qconfig_.contamination_penalty);

                double total_delta = mod_delta + lambda * quality_delta;

                if (total_delta > best_delta) {
                    best_delta = total_delta;
                    best = c;
                }
            }

            if (best != current) {
                comm_weights[current] -= node_weights_[node];
                comm_weights[best] += node_weights_[node];
                comm_sizes[current] -= node_sizes_[node];
                comm_sizes[best] += node_sizes_[node];
                update_comm_markers(node, current, best, comm_markers);
                labels[node] = best;
            }
        }
    }
}

float QualityLeidenBackend::calibrate_lambda(
    const std::vector<int>& labels,
    const std::vector<double>& comm_weights,
    const std::vector<double>& comm_sizes,
    const std::vector<std::unordered_map<int, int>>& comm_markers,
    float resolution) {

    // Sample random candidate moves and collect delta magnitudes
    std::vector<double> mod_deltas;
    std::vector<double> quality_deltas;

    int n_samples = std::min(qconfig_.calibration_samples,
                             std::max(5000, 10 * n_nodes_));

    std::uniform_int_distribution<int> node_dist(0, n_nodes_ - 1);

    for (int s = 0; s < n_samples; ++s) {
        int node = node_dist(rng_);
        if (is_fixed_[node]) continue;

        int current = labels[node];

        // Pick a random neighbor community
        if (adj_[node].empty()) continue;

        std::uniform_int_distribution<size_t> neighbor_dist(0, adj_[node].size() - 1);
        int j = adj_[node][neighbor_dist(rng_)].first;
        int target = labels[j];

        if (target == current) continue;

        // Compute edges to current and target
        double edges_to_current = 0.0;
        double edges_to_target = 0.0;
        for (const auto& [k, w] : adj_[node]) {
            if (labels[k] == current) edges_to_current += w;
            else if (labels[k] == target) edges_to_target += w;
        }

        // Modularity delta
        double mod_delta = delta_modularity_fast(
            node, current, target,
            edges_to_current, edges_to_target,
            comm_weights, comm_sizes, resolution);

        // Quality delta
        auto q_delta = compute_quality_delta(node, current, target, *marker_index_, comm_markers);
        double quality_delta = q_delta.total(qconfig_.contamination_penalty);

        mod_deltas.push_back(std::abs(mod_delta));
        if (std::abs(quality_delta) > 1e-10) {
            quality_deltas.push_back(std::abs(quality_delta));
        }
    }

    // Compute medians
    if (mod_deltas.empty() || quality_deltas.empty()) {
        std::cerr << "[QualityLeiden] Not enough samples for calibration, using default lambda\n";
        return qconfig_.alpha;
    }

    std::sort(mod_deltas.begin(), mod_deltas.end());
    std::sort(quality_deltas.begin(), quality_deltas.end());

    double s_R = mod_deltas[mod_deltas.size() / 2];
    double s_M = quality_deltas[quality_deltas.size() / 2];

    if (s_M < 1e-10) {
        std::cerr << "[QualityLeiden] Quality deltas too small, using default lambda\n";
        return qconfig_.alpha;
    }

    float lambda = qconfig_.alpha * static_cast<float>(s_R / s_M);

    // Clamp to reasonable range
    lambda = std::max(qconfig_.lambda_min, std::min(qconfig_.lambda_max, lambda));

    std::cerr << "[QualityLeiden] Calibration: s_R=" << s_R << ", s_M=" << s_M
              << ", lambda=" << lambda << " (from " << quality_deltas.size() << " quality samples)\n";

    return lambda;
}

void QualityLeidenBackend::init_comm_markers(
    const std::vector<int>& labels,
    int n_communities,
    std::vector<std::unordered_map<int, int>>& comm_markers) {

    comm_markers.clear();
    comm_markers.resize(n_communities);

    for (int i = 0; i < n_nodes_; ++i) {
        int c = labels[i];
        const auto& markers = marker_index_->get_markers(i);
        for (const auto& entry : markers) {
            comm_markers[c][entry.id] += entry.copy_count;
        }
    }
}

std::vector<std::unordered_map<int, int>> QualityLeidenBackend::aggregate_markers(
    const std::vector<int>& labels,
    const std::vector<std::unordered_map<int, int>>& comm_markers,
    int n_communities) {

    // For aggregate graph, marker counts stay the same (community -> supernode)
    // Just return a copy since community indices map directly to aggregate nodes
    return comm_markers;
}

void QualityLeidenBackend::update_comm_markers(
    int node,
    int from_comm,
    int to_comm,
    std::vector<std::unordered_map<int, int>>& comm_markers) {

    const auto& markers = marker_index_->get_markers(node);

    for (const auto& entry : markers) {
        // Remove from source
        comm_markers[from_comm][entry.id] -= entry.copy_count;
        if (comm_markers[from_comm][entry.id] <= 0) {
            comm_markers[from_comm].erase(entry.id);
        }

        // Add to target
        comm_markers[to_comm][entry.id] += entry.copy_count;
    }
}

void QualityLeidenBackend::init_comm_quality(
    const std::vector<int>& labels,
    int n_communities,
    std::vector<CommQuality>& comm_q) {
    comm_q.clear();
    comm_q.resize(n_communities);
    for (int i = 0; i < n_nodes_; ++i) {
        int c = labels[i];
        const auto& markers = marker_index_->get_markers(i);
        for (const auto& entry : markers) {
            int m = entry.id;
            int k = entry.copy_count;
            if (comm_q[c].cnt[m] == 0) comm_q[c].present++;
            comm_q[c].cnt[m] += k;
        }
    }
    for (auto& cq : comm_q) {
        cq.dup_excess = 0;
        for (int m = 0; m < MAX_MARKER_TYPES; ++m) {
            if (cq.cnt[m] > 1) cq.dup_excess += cq.cnt[m] - 1;
        }
    }
}

void QualityLeidenBackend::update_comm_quality(
    int node, int from, int to,
    std::vector<CommQuality>& comm_q) {
    const auto& markers = marker_index_->get_markers(node);
    for (const auto& entry : markers) {
        int m = entry.id;
        int k = entry.copy_count;

        auto& src = comm_q[from];
        int old_src = src.cnt[m];
        src.cnt[m] -= k;
        int new_src = src.cnt[m];
        if (old_src > 0 && new_src <= 0) src.present--;
        src.dup_excess -= std::max(0, old_src - 1) - std::max(0, new_src - 1);

        auto& dst = comm_q[to];
        int old_dst = dst.cnt[m];
        dst.cnt[m] += k;
        int new_dst = dst.cnt[m];
        if (old_dst <= 0 && new_dst > 0) dst.present++;
        dst.dup_excess -= std::max(0, old_dst - 1) - std::max(0, new_dst - 1);
    }
}

QualityDelta QualityLeidenBackend::compute_quality_delta_fast(
    int node, int source_comm, int target_comm,
    const std::vector<CommQuality>& comm_q) const {

    QualityDelta delta;
    const auto& markers = marker_index_->get_markers(node);
    if (markers.empty()) return delta;

    const auto& src = comm_q[source_comm];
    const auto& dst = comm_q[target_comm];

    for (const auto& entry : markers) {
        int m = entry.id;
        float w = entry.weight;
        int k = entry.copy_count;
        int n_old = src.cnt[m];
        int n_new = dst.cnt[m];

        int ind_new_after = (n_new + k > 0) ? 1 : 0;
        int ind_new_before = (n_new > 0) ? 1 : 0;
        int ind_old_after = (n_old - k > 0) ? 1 : 0;
        int ind_old_before = (n_old > 0) ? 1 : 0;
        delta.delta_completeness += w * (ind_new_after - ind_new_before + ind_old_after - ind_old_before);

        int cont_new_after = std::max(n_new + k - 1, 0);
        int cont_new_before = std::max(n_new - 1, 0);
        int cont_old_after = std::max(n_old - k - 1, 0);
        int cont_old_before = std::max(n_old - 1, 0);
        delta.delta_contamination += w * (cont_new_after - cont_new_before + cont_old_after - cont_old_before);
    }
    return delta;
}

float QualityLeidenBackend::calibrate_lambda(
    const std::vector<int>& labels,
    const std::vector<double>& comm_weights,
    const std::vector<double>& comm_sizes,
    const std::vector<CommQuality>& comm_q,
    float resolution) {

    std::vector<double> mod_deltas;
    std::vector<double> quality_deltas;

    int n_samples = std::min(qconfig_.calibration_samples,
                             std::max(5000, 10 * n_nodes_));

    std::uniform_int_distribution<int> node_dist(0, n_nodes_ - 1);

    for (int s = 0; s < n_samples; ++s) {
        int node = node_dist(rng_);
        if (is_fixed_[node]) continue;

        int current = labels[node];

        if (adj_[node].empty()) continue;

        std::uniform_int_distribution<size_t> neighbor_dist(0, adj_[node].size() - 1);
        int j = adj_[node][neighbor_dist(rng_)].first;
        int target = labels[j];

        if (target == current) continue;

        double edges_to_current = 0.0;
        double edges_to_target = 0.0;
        for (const auto& [k, w] : adj_[node]) {
            if (labels[k] == current) edges_to_current += w;
            else if (labels[k] == target) edges_to_target += w;
        }

        double mod_delta = delta_modularity_fast(
            node, current, target,
            edges_to_current, edges_to_target,
            comm_weights, comm_sizes, resolution);

        auto q_delta = compute_quality_delta_fast(node, current, target, comm_q);
        double quality_delta = q_delta.total(qconfig_.contamination_penalty);

        mod_deltas.push_back(std::abs(mod_delta));
        if (std::abs(quality_delta) > 1e-10) {
            quality_deltas.push_back(std::abs(quality_delta));
        }
    }

    if (mod_deltas.empty() || quality_deltas.empty()) {
        std::cerr << "[QualityLeiden] Not enough samples for calibration, using default lambda\n";
        return qconfig_.alpha;
    }

    std::sort(mod_deltas.begin(), mod_deltas.end());
    std::sort(quality_deltas.begin(), quality_deltas.end());

    double s_R = mod_deltas[mod_deltas.size() / 2];
    double s_M = quality_deltas[quality_deltas.size() / 2];

    if (s_M < 1e-10) {
        std::cerr << "[QualityLeiden] Quality deltas too small, using default lambda\n";
        return qconfig_.alpha;
    }

    float lambda = qconfig_.alpha * static_cast<float>(s_R / s_M);

    lambda = std::max(qconfig_.lambda_min, std::min(qconfig_.lambda_max, lambda));

    std::cerr << "[QualityLeiden] Calibration (dense): s_R=" << s_R << ", s_M=" << s_M
              << ", lambda=" << lambda << " (from " << quality_deltas.size() << " quality samples)\n";

    return lambda;
}

bool QualityLeidenBackend::move_nodes_fast_quality_mt(
    std::vector<int>& labels,
    std::vector<double>& comm_weights,
    std::vector<double>& comm_sizes,
    std::vector<CommQuality>& comm_q,
    float resolution,
    float lambda,
    int n_threads) {

    const int n = n_nodes_;
    const int C = static_cast<int>(comm_q.size());

    // Thread contexts
    std::vector<ThreadCtx> ctx(n_threads);
    for (auto& c : ctx) {
        c.acc.resize(C, 0.0);
        c.seen.resize(C, 0);
        c.touched.reserve(128);
    }

    // In-queue tracking (atomic for thread safety)
    std::vector<std::atomic<uint8_t>> in_q(n);
    for (int i = 0; i < n; ++i) in_q[i].store(0, std::memory_order_relaxed);

    // Initial population: distribute nodes across threads
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng_);

    for (int i = 0; i < n; ++i) {
        int node = order[i];
        int tid = i % n_threads;
        if (has_marker_[node]) {
            ctx[tid].q_marker.push_back(node);
        } else {
            ctx[tid].q_plain.push_back(node);
        }
        in_q[node].store(1, std::memory_order_relaxed);
    }

    // Proposed moves buffer
    struct ProposedMove { int node; int best; double delta; };
    std::vector<std::vector<ProposedMove>> thread_proposals(n_threads);

    // Convergence safeguards
    constexpr double EPSILON = 1e-5;   // Min improvement — prevents oscillating micro-moves
    constexpr int MAX_BATCHES = 500;   // ~70 full sweeps of 28k nodes at 16 threads × 256

    // Count initial queue sizes
    int initial_marker = 0, initial_plain = 0;
    for (int t = 0; t < n_threads; ++t) {
        initial_marker += ctx[t].q_marker.size();
        initial_plain += ctx[t].q_plain.size();
    }
    std::cerr << "[QualityLeiden-MT] Move start: n=" << n << ", C=" << C
              << ", threads=" << n_threads
              << ", marker_q=" << initial_marker << ", plain_q=" << initial_plain << "\n";
    std::cerr.flush();

    bool global_improved = false;
    int batch_count = 0;
    int total_proposals = 0;
    int total_commits = 0;
    int skipped_epsilon = 0;

    while (true) {
        // Check if any work remains
        bool has_work = false;
        int total_queued = 0;
        for (int t = 0; t < n_threads; ++t) {
            total_queued += ctx[t].q_marker.size() + ctx[t].q_plain.size();
            if (!ctx[t].q_marker.empty() || !ctx[t].q_plain.empty()) {
                has_work = true;
            }
        }
        if (!has_work) break;
        if (batch_count >= MAX_BATCHES) {
            std::cerr << "[QualityLeiden-MT] Reached batch limit (" << MAX_BATCHES << "), stopping\n";
            break;
        }

        if (batch_count % 100 == 0) {
            std::cerr << "[QualityLeiden-MT] Batch " << batch_count
                      << ": " << total_queued << " nodes queued, "
                      << total_proposals << " proposals, "
                      << total_commits << " commits\n";
            std::cerr.flush();
        }
        batch_count++;

        // Clear proposals
        for (auto& p : thread_proposals) p.clear();

        // --- Phase 1: Parallel evaluation ---
        #pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            auto& my_ctx = ctx[tid];
            auto& proposals = thread_proposals[tid];

            constexpr int BATCH_SIZE = 256;
            int marker_burst = 0;

            for (int batch = 0; batch < BATCH_SIZE; ++batch) {
                // Dequeue with marker priority
                int node = -1;
                if (!my_ctx.q_marker.empty() && (marker_burst < 8 || my_ctx.q_plain.empty())) {
                    node = my_ctx.q_marker.front();
                    my_ctx.q_marker.pop_front();
                    marker_burst++;
                } else if (!my_ctx.q_plain.empty()) {
                    node = my_ctx.q_plain.front();
                    my_ctx.q_plain.pop_front();
                    marker_burst = 0;
                } else {
                    // Work stealing disabled: std::deque is not thread-safe
                    // For safe stealing, would need per-queue mutex
                    break;
                }
                if (node < 0) break;

                uint8_t expected = 1;
                if (!in_q[node].compare_exchange_strong(expected, 0)) continue;

                if (is_fixed_[node]) continue;
                int current = labels[node];

                // Epoch-based accumulator (avoids clearing the full array)
                my_ctx.epoch++;
                if (my_ctx.epoch == 0) {
                    std::fill(my_ctx.seen.begin(), my_ctx.seen.end(), 0);
                    my_ctx.epoch = 1;
                }
                my_ctx.touched.clear();

                // Accumulate edges to communities
                for (size_t ei = 0; ei < adj_[node].size(); ++ei) {
                    int nbr = adj_[node][ei].first;
                    double w = adj_[node][ei].second;
                    int c = labels[nbr];
                    if (my_ctx.seen[c] != my_ctx.epoch) {
                        my_ctx.seen[c] = my_ctx.epoch;
                        my_ctx.acc[c] = w;
                        my_ctx.touched.push_back(c);
                    } else {
                        my_ctx.acc[c] += w;
                    }
                }

                // Ensure current community is in touched
                if (my_ctx.seen[current] != my_ctx.epoch) {
                    my_ctx.seen[current] = my_ctx.epoch;
                    my_ctx.acc[current] = 0.0;
                    my_ctx.touched.push_back(current);
                }

                double edges_to_current = my_ctx.acc[current];
                int best = current;
                double best_delta = 0.0;

                for (int c : my_ctx.touched) {
                    if (c == current) continue;

                    double mod_delta = delta_modularity_fast(
                        node, current, c, edges_to_current, my_ctx.acc[c],
                        comm_weights, comm_sizes, resolution);

                    double qual_delta = 0.0;
                    if (has_marker_[node]) {
                        auto qd = compute_quality_delta_fast(node, current, c, comm_q);
                        qual_delta = qd.total(qconfig_.contamination_penalty);
                    }

                    double total = mod_delta + lambda * qual_delta;
                    if (total > best_delta) {
                        best_delta = total;
                        best = c;
                    }
                }

                if (best != current) {
                    proposals.push_back({node, best, best_delta});
                }
            }
        }

        // Count proposals from all threads
        int batch_proposals = 0;
        for (int tid = 0; tid < n_threads; ++tid) {
            batch_proposals += thread_proposals[tid].size();
        }
        total_proposals += batch_proposals;

        // --- Phase 2: Serial commit with revalidation ---
        for (int tid = 0; tid < n_threads; ++tid) {
            for (const auto& prop : thread_proposals[tid]) {
                int node = prop.node;
                int best = prop.best;
                int current = labels[node];

                if (current == best) continue;

                // Revalidate: recompute delta to handle stale proposals
                double edges_to_current = 0.0;
                double edges_to_best = 0.0;
                for (size_t ei = 0; ei < adj_[node].size(); ++ei) {
                    int nbr = adj_[node][ei].first;
                    double w = adj_[node][ei].second;
                    int c = labels[nbr];
                    if (c == current) edges_to_current += w;
                    else if (c == best) edges_to_best += w;
                }
                double mod_delta = delta_modularity_fast(
                    node, current, best, edges_to_current, edges_to_best,
                    comm_weights, comm_sizes, resolution);
                double qual_delta = 0.0;
                if (has_marker_[node]) {
                    auto qd = compute_quality_delta_fast(node, current, best, comm_q);
                    qual_delta = qd.total(qconfig_.contamination_penalty);
                }
                double revalidated_delta = mod_delta + lambda * qual_delta;
                if (revalidated_delta <= EPSILON) {
                    skipped_epsilon++;
                    continue;  // Skip tiny/stale proposals
                }

                // Commit the move
                comm_weights[current] -= node_weights_[node];
                comm_weights[best] += node_weights_[node];
                comm_sizes[current] -= node_sizes_[node];
                comm_sizes[best] += node_sizes_[node];

                if (has_marker_[node]) {
                    update_comm_quality(node, current, best, comm_q);
                }

                labels[node] = best;
                global_improved = true;
                total_commits++;

                // Re-queue neighbors not in destination community
                for (size_t ei = 0; ei < adj_[node].size(); ++ei) {
                    int nbr = adj_[node][ei].first;
                    if (labels[nbr] != best) {  // Key: only if not in dest community
                        uint8_t exp = 0;
                        if (in_q[nbr].compare_exchange_strong(exp, 1)) {
                            int owner = nbr % n_threads;
                            if (has_marker_[nbr]) {
                                ctx[owner].q_marker.push_back(nbr);
                            } else {
                                ctx[owner].q_plain.push_back(nbr);
                            }
                        }
                    }
                }
            }
        }
    }

    std::cerr << "[QualityLeiden-MT] Move phase complete: " << batch_count << " batches, "
              << total_proposals << " proposals, " << total_commits << " commits, "
              << "skipped=" << skipped_epsilon << " (epsilon), improved=" << (global_improved ? "yes" : "no") << "\n";
    std::cerr.flush();

    return global_improved;
}

void QualityLeidenBackend::refine_partition_quality_mt(
    std::vector<int>& labels,
    std::vector<double>& comm_weights,
    std::vector<double>& comm_sizes,
    std::vector<CommQuality>& comm_q,
    float resolution,
    float lambda,
    int n_threads) {

    constexpr int MAX_REFINEMENT_PASSES = 20;

    for (int pass = 0; pass < MAX_REFINEMENT_PASSES; ++pass) {
        std::cerr << "[QualityLeiden-MT] Starting refinement pass " << pass << "...\n";
        bool improved = move_nodes_fast_quality_mt(
            labels, comm_weights, comm_sizes, comm_q,
            resolution, lambda, n_threads);

        if (!improved) {
            std::cerr << "[QualityLeiden-MT] Refinement converged at pass " << pass << "\n";
            break;
        }
        std::cerr << "[QualityLeiden-MT] Pass " << pass << " improved, continuing...\n";
    }
}

std::unique_ptr<ILeidenBackend> create_quality_leiden_backend(
    const MarkerIndex* marker_index,
    const QualityLeidenConfig& qconfig) {

    auto backend = std::make_unique<QualityLeidenBackend>();
    backend->set_marker_index(marker_index);
    backend->set_quality_config(qconfig);
    return backend;
}

}  // namespace amber::bin2

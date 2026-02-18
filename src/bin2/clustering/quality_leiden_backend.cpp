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

// h(x) = x·ln(x) with h(0) = 0 (used in map equation codelength)
static inline double hlog(double x) {
    return (x > 1e-300) ? x * std::log(x) : 0.0;
}

// Map equation delta for moving node from curr to tgt.
// gain = -ΔL (positive = the move reduces description length = improvement).
//
// L' = h(q_total) - 2·Σ_c h(q_c) + Σ_c h(flow_c)  (constant node-visit term omitted)
// where:
//   q_c    = (W_c - comm_internal[c]) / m2   (exit rate, comm_internal counts each edge twice)
//   flow_c = (2*W_c - comm_internal[c]) / m2  (= q_c + p_c)
//   m2     = total_edge_weight_ (sum of all edge weights, each edge counted from both endpoints)
double QualityLeidenBackend::delta_map_equation(
    int node, int curr, int tgt,
    double edges_to_curr, double edges_to_tgt,
    const std::vector<double>& comm_weights,
    const std::vector<double>& comm_internal,
    double q_total) const {

    const double m2 = total_edge_weight_;
    if (m2 < 1e-300) return 0.0;

    const double kv = node_weights_[node];

    // Current values
    double q_a    = (comm_weights[curr] - comm_internal[curr]) / m2;
    double q_b    = (comm_weights[tgt]  - comm_internal[tgt])  / m2;
    double flow_a = (2.0*comm_weights[curr] - comm_internal[curr]) / m2;
    double flow_b = (2.0*comm_weights[tgt]  - comm_internal[tgt])  / m2;

    // Values after move
    // Δcomm_internal[curr] = -2*edges_to_curr, Δcomm_internal[tgt] = +2*edges_to_tgt
    double q_a_new    = q_a    + (-kv + 2.0*edges_to_curr) / m2;
    double q_b_new    = q_b    + ( kv - 2.0*edges_to_tgt)  / m2;
    double flow_a_new = flow_a + 2.0*(-kv + edges_to_curr) / m2;
    double flow_b_new = flow_b + 2.0*( kv - edges_to_tgt)  / m2;
    double q_total_new = q_total + 2.0*(edges_to_curr - edges_to_tgt) / m2;

    // ΔL' = Δh(q_total) - 2·Δh(q_a) - 2·Δh(q_b) + Δh(flow_a) + Δh(flow_b)
    double delta_L = (hlog(q_total_new) - hlog(q_total))
                   - 2.0*(hlog(q_a_new) - hlog(q_a))
                   - 2.0*(hlog(q_b_new) - hlog(q_b))
                   + (hlog(flow_a_new) - hlog(flow_a))
                   + (hlog(flow_b_new) - hlog(flow_b));

    return -delta_L;  // gain: positive = improvement
}

void QualityLeidenBackend::init_comm_internal(
    const std::vector<int>& labels,
    int n_communities,
    std::vector<double>& comm_internal) const {

    comm_internal.assign(n_communities, 0.0);
    for (int v = 0; v < n_nodes_; ++v) {
        int cv = labels[v];
        for (const auto& [u, w] : adj_[v]) {
            if (labels[u] == cv) {
                comm_internal[cv] += w;  // counts each internal edge from both endpoints
            }
        }
    }
}

// Internal SCG quality score: sum of (completeness - penalty * contamination) over all clusters.
// Completeness = markers present in this cluster (unique SCGs).
// Contamination = duplicate marker excess (dup_excess).
// Higher = better. No CheckM2 needed — uses the in-memory MarkerIndex directly.
float QualityLeidenBackend::scg_quality_score(const std::vector<CommQuality>& comm_q) const {
    float score = 0.0f;
    for (const auto& cq : comm_q) {
        score += static_cast<float>(cq.present)
               - qconfig_.contamination_penalty * static_cast<float>(cq.dup_excess);
    }
    return score;
}

// Resolution calibration via internal SCG metric.
//
// Fits a quadratic model in log-resolution space to the SCG quality scores
// from n_res_trials Leiden runs (log-uniform over [res_search_min, res_search_max]),
// then returns the argmax. This adapts the resolution to the dataset without
// any external validation — uses only the HMM marker information already loaded.
//
// Each trial run takes <1s (libleidenalg without GPU), so total overhead is small.
float QualityLeidenBackend::calibrate_resolution(
    const std::vector<WeightedEdge>& edges,
    int n_nodes,
    const LeidenConfig& config) const {

    const int n = qconfig_.n_res_trials;
    const float log_lo = std::log(qconfig_.res_search_min);
    const float log_hi = std::log(qconfig_.res_search_max);

    std::vector<float> log_res(n), scores(n);

    for (int i = 0; i < n; ++i) {
        float t = (n > 1) ? static_cast<float>(i) / (n - 1) : 0.5f;
        log_res[i] = log_lo + t * (log_hi - log_lo);
        float res = std::exp(log_res[i]);

        LeidenConfig trial = config;
        trial.resolution = res;

#ifdef USE_LIBLEIDENALG
        LibLeidenalgBackend trial_leiden;
        auto result = trial_leiden.cluster(edges, n_nodes, trial);
#else
        LeidenBackend trial_leiden;
        auto result = trial_leiden.cluster(edges, n_nodes, trial);
#endif

        // Score with internal SCG metric
        int nc = result.num_clusters;
        std::vector<CommQuality> cq;
        // Temporarily init_comm_quality using a const-cast-free approach:
        // build comm_q from scratch using the marker_index_
        cq.resize(nc);
        for (int v = 0; v < n_nodes; ++v) {
            int c = result.labels[v];
            for (const auto& entry : marker_index_->get_markers(v)) {
                int m = entry.id;
                int k = entry.copy_count;
                if (cq[c].cnt[m] == 0) cq[c].present++;
                cq[c].cnt[m] += k;
            }
        }
        for (auto& q : cq) {
            for (int m = 0; m < MAX_MARKER_TYPES; ++m) {
                if (q.cnt[m] > 1) q.dup_excess += q.cnt[m] - 1;
            }
        }
        scores[i] = scg_quality_score(cq);

        std::cerr << "[QualityLeiden] ResCalib trial " << (i+1) << "/" << n
                  << " res=" << res << " clusters=" << nc
                  << " score=" << scores[i] << "\n";
    }

    // Fit quadratic: score ≈ a*x² + b*x + c where x = log_res
    // Least-squares via normal equations on the 3-parameter model.
    // If n < 3, just return the best sample.
    if (n < 3) {
        int best_i = static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
        return std::exp(log_res[best_i]);
    }

    double sx = 0, sx2 = 0, sx3 = 0, sx4 = 0;
    double sy = 0, sxy = 0, sx2y = 0;
    for (int i = 0; i < n; ++i) {
        double x = log_res[i], y = scores[i];
        sx += x; sx2 += x*x; sx3 += x*x*x; sx4 += x*x*x*x;
        sy += y; sxy += x*y; sx2y += x*x*y;
    }
    // Normal equations: [n, sx, sx2; sx, sx2, sx3; sx2, sx3, sx4] * [c,b,a]^T = [sy,sxy,sx2y]^T
    // Solve with Cramer's rule for the 3×3 system.
    double det = n*(sx2*sx4 - sx3*sx3) - sx*(sx*sx4 - sx3*sx2) + sx2*(sx*sx3 - sx2*sx2);

    float best_log_res = log_res[static_cast<int>(
        std::max_element(scores.begin(), scores.end()) - scores.begin())];

    if (std::abs(det) > 1e-12) {
        double da = n*(sx2*sx2y - sx3*sxy) - sx*(sx*sx2y - sx3*sy) + sx2*(sx*sxy - sx2*sy);
        double a = da / det;

        if (a < 0.0) {
            // Concave down: argmax at -b/(2a)
            double db = n*(sx2y*sx4 - sx3*sx2y) - sy*(sx*sx4 - sx3*sx2) + sx2*(sx*sx2y - sx2*sy);
            // Actually recompute b properly:
            double mat[3][3] = {{(double)n,sx,sx2},{sx,sx2,sx3},{sx2,sx3,sx4}};
            double rhs[3] = {sy, sxy, sx2y};
            // b is coefficient of x: use col 1 replacement
            double det_b = (double)n*(sx2*sx2y - sx3*sxy) - sx*(sx*sx2y - sx3*sy) + sx2*(sx*sxy - sx2*sy);
            (void)det_b;  // reuse da for a, compute b separately below
            // Simpler: just find argmax of fitted quadratic at midpoint
            // b/a coefficient: fit at 3 evenly-spaced points for stability
            int i0 = 0, i1 = n/2, i2 = n-1;
            double x0=log_res[i0], x1=log_res[i1], x2=log_res[i2];
            double y0=scores[i0], y1=scores[i1], y2=scores[i2];
            double denom = (x0-x1)*(x0-x2)*(x1-x2);
            if (std::abs(denom) > 1e-12) {
                double aa = (x2*(y1-y0) + x1*(y0-y2) + x0*(y2-y1)) / denom;
                double bb = (x2*x2*(y0-y1) + x1*x1*(y2-y0) + x0*x0*(y1-y2)) / denom;
                if (aa < 0.0) {
                    double peak = -bb / (2.0*aa);
                    // Clamp to search range
                    peak = std::max(static_cast<double>(log_lo),
                                    std::min(static_cast<double>(log_hi), peak));
                    best_log_res = static_cast<float>(peak);
                }
            }
        }
    }

    float best_res = std::exp(best_log_res);
    std::cerr << "[QualityLeiden] ResCalib: best resolution = " << best_res
              << " (log=" << best_log_res << ")\n";
    return best_res;
}

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

    // Optional: calibrate resolution using internal SCG quality before the main run.
    LeidenConfig calibrated_config = config;
    if (qconfig_.calibrate_resolution && marker_index_->num_marker_contigs() > 0) {
        std::cerr << "[QualityLeiden-MT] Calibrating resolution over ["
                  << qconfig_.res_search_min << ", " << qconfig_.res_search_max << "] ...\n";
        calibrated_config.resolution = calibrate_resolution(edges, n_nodes, config);
        std::cerr << "[QualityLeiden-MT] Using calibrated resolution = "
                  << calibrated_config.resolution << "\n";
    }
    const LeidenConfig& effective_config = qconfig_.calibrate_resolution ? calibrated_config : config;

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
    ClusteringResult initial_result = fast_backend.cluster(p1_edges, n_nodes, effective_config);
#else
    ClusteringResult initial_result = LeidenBackend::cluster(p1_edges, n_nodes, effective_config);
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
        int original_n_communities = n_communities;  // freeze before splits grow n_communities

        for (int c = 0; c < original_n_communities; ++c) {
            if (comm_q[c].dup_excess == 0) continue;
            const auto& nodes = cluster_nodes[c];
            if ((int)nodes.size() <= 1) continue;

            // Only split if cluster is large enough to yield at least 2 viable bins
            // (> 2 × 200kb min_bin_size). Splitting small clusters just fragments good bins.
            long long total_bp = 0;
            for (int v : nodes) total_bp += static_cast<long long>(node_sizes_[v]);
            if (total_bp < 400000LL) continue;

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

            // Re-cluster with resolution scaled inversely with cluster size.
            // For small clusters (~50 nodes), use 3x to force fine splits.
            // For large clusters (~1000+ nodes), scale down to avoid shattering into singletons
            // (a 1836-node cluster at 3x resolution fragments into 320 parts, all too small to bin).
            LeidenConfig sub_config = effective_config;
            float size_scale = std::max(1.0f, static_cast<float>(nodes.size()) / 100.0f);
            float res_mult = std::min(3.0f, 3.0f / std::sqrt(size_scale));
            sub_config.resolution = effective_config.resolution * res_mult;
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

            // Reject if over-fragmented: a cluster with dup_excess=k needs at most k+2 parts.
            // Over-fragmented splits shatter viable bins into sub-threshold fragments.
            int max_acceptable = comm_q[c].dup_excess + 2;
            if (sub_result.num_clusters > max_acceptable) continue;

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

    // Phase 1b (optional): marker-guided map equation refinement.
    // Uses the map equation delta (parameter-free MDL objective) combined with
    // the marker quality penalty. Unlike modularity, the map equation has no
    // resolution limit, so it won't over-merge unlike the old Phase 1b.
    if (qconfig_.use_map_equation) {
        std::cerr << "[QualityLeiden-MT] Phase 1b: map-equation quality refinement...\n";

        // Re-compact labels first (Phase 2 may have grown n_communities)
        n_communities = compact_labels(labels);

        // Re-init community stats after Phase 2
        comm_weights.assign(n_communities, 0.0);
        comm_sizes.assign(n_communities, 0.0);
        for (int i = 0; i < n_nodes_; ++i) {
            comm_weights[labels[i]] += node_weights_[i];
            comm_sizes[labels[i]] += node_sizes_[i];
        }
        init_comm_quality(labels, n_communities, comm_q);

        std::vector<double> comm_internal;
        init_comm_internal(labels, n_communities, comm_internal);

        // q_total = fraction of flow crossing community boundaries
        double ci_sum = 0.0;
        for (double ci : comm_internal) ci_sum += ci;
        double q_total = (total_edge_weight_ > 0.0)
                       ? (total_edge_weight_ - ci_sum) / total_edge_weight_
                       : 0.0;

        float lambda = calibrate_lambda(labels, comm_weights, comm_sizes, comm_q,
                                        comm_internal, q_total, effective_config.resolution);
        std::cerr << "[QualityLeiden-MT] Phase 1b lambda=" << lambda
                  << ", q_total=" << q_total
                  << ", communities=" << n_communities << "\n";

        refine_partition_quality_mt(labels, comm_weights, comm_sizes, comm_q,
                                    comm_internal, q_total,
                                    config.resolution, lambda, n_threads);

        n_communities = compact_labels(labels);
        std::cerr << "[QualityLeiden-MT] Phase 1b done: " << n_communities << " clusters\n";
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
    const std::vector<double>& comm_internal,
    double q_total,
    float resolution) {

    std::vector<double> base_deltas;   // map equation or modularity, depending on mode
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

        double base_delta;
        if (qconfig_.use_map_equation) {
            base_delta = delta_map_equation(
                node, current, target,
                edges_to_current, edges_to_target,
                comm_weights, comm_internal, q_total);
        } else {
            base_delta = delta_modularity_fast(
                node, current, target,
                edges_to_current, edges_to_target,
                comm_weights, comm_sizes, resolution);
        }

        auto q_delta = compute_quality_delta_fast(node, current, target, comm_q);
        double quality_delta = q_delta.total(qconfig_.contamination_penalty);

        base_deltas.push_back(std::abs(base_delta));
        if (std::abs(quality_delta) > 1e-10) {
            quality_deltas.push_back(std::abs(quality_delta));
        }
    }

    if (base_deltas.empty() || quality_deltas.empty()) {
        std::cerr << "[QualityLeiden] Not enough samples for calibration, using default lambda\n";
        return qconfig_.alpha;
    }

    std::sort(base_deltas.begin(), base_deltas.end());
    std::sort(quality_deltas.begin(), quality_deltas.end());

    double s_R = base_deltas[base_deltas.size() / 2];
    double s_M = quality_deltas[quality_deltas.size() / 2];

    if (s_M < 1e-10) {
        std::cerr << "[QualityLeiden] Quality deltas too small, using default lambda\n";
        return qconfig_.alpha;
    }

    float lambda = qconfig_.alpha * static_cast<float>(s_R / s_M);

    lambda = std::max(qconfig_.lambda_min, std::min(qconfig_.lambda_max, lambda));

    std::cerr << "[QualityLeiden] Calibration (dense,"
              << (qconfig_.use_map_equation ? "map-eq" : "modularity")
              << "): s_R=" << s_R << ", s_M=" << s_M
              << ", lambda=" << lambda << " (from " << quality_deltas.size() << " quality samples)\n";

    return lambda;
}

bool QualityLeidenBackend::move_nodes_fast_quality_mt(
    std::vector<int>& labels,
    std::vector<double>& comm_weights,
    std::vector<double>& comm_sizes,
    std::vector<CommQuality>& comm_q,
    std::vector<double>& comm_internal,
    double& q_total,
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

                    double base_delta;
                    if (qconfig_.use_map_equation) {
                        base_delta = delta_map_equation(
                            node, current, c, edges_to_current, my_ctx.acc[c],
                            comm_weights, comm_internal, q_total);
                    } else {
                        base_delta = delta_modularity_fast(
                            node, current, c, edges_to_current, my_ctx.acc[c],
                            comm_weights, comm_sizes, resolution);
                    }

                    double qual_delta = 0.0;
                    if (has_marker_[node]) {
                        auto qd = compute_quality_delta_fast(node, current, c, comm_q);
                        qual_delta = qd.total(qconfig_.contamination_penalty);
                    }

                    double total = base_delta + lambda * qual_delta;
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
                double base_delta;
                if (qconfig_.use_map_equation) {
                    base_delta = delta_map_equation(
                        node, current, best, edges_to_current, edges_to_best,
                        comm_weights, comm_internal, q_total);
                } else {
                    base_delta = delta_modularity_fast(
                        node, current, best, edges_to_current, edges_to_best,
                        comm_weights, comm_sizes, resolution);
                }
                double qual_delta = 0.0;
                if (has_marker_[node]) {
                    auto qd = compute_quality_delta_fast(node, current, best, comm_q);
                    qual_delta = qd.total(qconfig_.contamination_penalty);
                }
                double revalidated_delta = base_delta + lambda * qual_delta;
                if (revalidated_delta <= EPSILON) {
                    skipped_epsilon++;
                    continue;  // Skip tiny/stale proposals
                }

                // Commit the move
                comm_weights[current] -= node_weights_[node];
                comm_weights[best] += node_weights_[node];
                comm_sizes[current] -= node_sizes_[node];
                comm_sizes[best] += node_sizes_[node];

                if (qconfig_.use_map_equation) {
                    const double m2 = total_edge_weight_;
                    q_total += (m2 > 0.0) ? 2.0*(edges_to_current - edges_to_best) / m2 : 0.0;
                    comm_internal[current] -= 2.0 * edges_to_current;
                    comm_internal[best]    += 2.0 * edges_to_best;
                }

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
    std::vector<double>& comm_internal,
    double& q_total,
    float resolution,
    float lambda,
    int n_threads) {

    constexpr int MAX_REFINEMENT_PASSES = 20;

    for (int pass = 0; pass < MAX_REFINEMENT_PASSES; ++pass) {
        std::cerr << "[QualityLeiden-MT] Starting refinement pass " << pass << "...\n";
        bool improved = move_nodes_fast_quality_mt(
            labels, comm_weights, comm_sizes, comm_q,
            comm_internal, q_total,
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

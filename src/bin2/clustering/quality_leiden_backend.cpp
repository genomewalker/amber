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
#include <unordered_set>
#include <omp.h>
#include <Eigen/Dense>

namespace amber::bin2 {

// Knuth multiplicative hash: maps (base, counter) → well-spread 31-bit seed.
// Avoids RNG correlation that occurs when using sequential base+i seeds.
static inline int spread_seed(int base, int counter) {
    uint32_t h = static_cast<uint32_t>(base) ^ (static_cast<uint32_t>(counter) * 2654435761u);
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    return static_cast<int>(h & 0x7fffffffu);
}

float QualityLeidenBackend::scg_score_labels(
    const std::vector<int>& labels, int n_communities) const {
    if (!marker_index_ || marker_index_->num_marker_contigs() == 0) return 0.0f;
    // labels must be compacted: all values in [0, n_communities).
    // node_names_ must have size >= n_nodes_ when CheckM path is taken.

    // Accumulate bp per cluster.
    std::vector<long long> comm_bp(n_communities, 0);
    for (int i = 0; i < n_nodes_; ++i)
        comm_bp[labels[i]] += static_cast<long long>(node_sizes_[i]);

    // --- CheckM path: lexicographic composite score ---
    // Priority: HQ (75/5) > MQ (50/10) > near-HQ (30/5) > avg soft quality.
    // Calibrated thresholds: internal completeness underestimates by ~10-15pp
    // due to aDNA fragmentation; 75/5 internal matches CheckM2 90/5 empirically.
    // Strict scale separation (1e6 / 1e4 / 1e2 / <1) ensures correctness even
    // when many clusters contribute to the soft term.
    if (checkm_est_ && node_names_ && checkm_est_->has_marker_sets()) {
        std::vector<std::vector<int>> cluster_nodes(n_communities);
        for (int i = 0; i < n_nodes_; ++i)
            cluster_nodes[labels[i]].push_back(i);

        int   num_strict_hq = 0, num_hq = 0, num_pre_hq = 0, num_mq = 0, num_nhq = 0;
        float total_comp = 0.0f, total_cont = 0.0f;
        int   n_viable = 0;

        for (int c = 0; c < n_communities; ++c) {
            if (comm_bp[c] < qconfig_.restart_min_viable_bp) continue;
            std::vector<std::string> names;
            names.reserve(cluster_nodes[c].size());
            for (int v : cluster_nodes[c]) names.push_back((*node_names_)[v]);
            auto q = checkm_est_->estimate_bin_quality(names);

            // Calibrated thresholds: internal CheckM colocation underestimates CheckM2
            // by ~10-15pp due to aDNA fragmentation; 75/5 internal ≈ CheckM2 90/5.
            // strict_hq (≥90/5 internal ≈ CheckM2 100%): very solid, rarely hit.
            // hq       (≥75/5 internal ≈ CheckM2 90%):   primary HQ tier.
            // pre_hq   (≥72/5 internal ≈ CheckM2 87%):   gradient toward HQ boundary.
            if (q.completeness >= 90.0f && q.contamination <=  5.0f) num_strict_hq++;
            if (q.completeness >= 75.0f && q.contamination <=  5.0f) num_hq++;
            if (q.completeness >= 72.0f && q.contamination <=  5.0f) num_pre_hq++;
            if (q.completeness >= 50.0f && q.contamination <= 10.0f) num_mq++;
            if (q.completeness >= 30.0f && q.contamination <=  5.0f) num_nhq++;
            total_comp += q.completeness;
            total_cont += q.contamination;
            n_viable++;
        }

        // Soft tiebreaker: SUM (not mean) of (comp - 5*cont) across viable bins.
        // Using the mean normalised by n_viable penalises having more viable bins
        // (fewer large clusters score better than more small ones even though more
        // clusters at the right resolution produces more HQ bins). The sum correctly
        // rewards having more viable clusters with good quality.
        // Divided by 1e4 to stay safely below the 1e2 * num_nhq step.
        float soft = (total_comp - 5.0f * total_cont) / 1e4f;

        return 1e8f * num_strict_hq + 1e6f * num_hq + 5e5f * num_pre_hq
             + 1e4f * num_mq + 1e2f * num_nhq + soft;
    }

    // --- Proxy path (fallback when CheckM estimator not available) ---
    // Soft score Σ_viable [present - penalty * dup_excess].
    // The min-viable-bp filter prevents the monotonicity trap (more clusters →
    // less dup_excess per cluster) that caused UCB to favour res=50.
    std::vector<CommQuality> cq(n_communities);
    for (int i = 0; i < n_nodes_; ++i) {
        int c = labels[i];
        for (const auto& entry : marker_index_->get_markers(i)) {
            int m = entry.id, k = entry.copy_count;
            if (cq[c].cnt[m] == 0) cq[c].present++;
            cq[c].cnt[m] += k;
        }
    }
    for (auto& q : cq)
        for (int m = 0; m < MAX_MARKER_TYPES; ++m)
            if (q.cnt[m] > 1) q.dup_excess += q.cnt[m] - 1;

    float score = 0.0f;
    for (int c = 0; c < n_communities; ++c) {
        if (comm_bp[c] < qconfig_.restart_min_viable_bp) continue;
        score += static_cast<float>(cq[c].present)
               - qconfig_.contamination_penalty * static_cast<float>(cq[c].dup_excess);
    }
    return score;
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

std::pair<std::vector<int>, int> QualityLeidenBackend::run_phase2(
    std::vector<int> labels,
    int n_communities,
    const LeidenConfig& effective_config,
    float min_completeness) {

    std::vector<CommQuality> comm_q;
    init_comm_quality(labels, n_communities, comm_q);

    std::vector<std::vector<int>> cluster_nodes(n_communities);
    for (int i = 0; i < n_nodes_; ++i)
        cluster_nodes[labels[i]].push_back(i);

    int n_contaminated = 0, n_split = 0;
    int original_n_communities = n_communities;

    for (int c = 0; c < original_n_communities; ++c) {
        const auto& nodes = cluster_nodes[c];
        if ((int)nodes.size() <= 1) continue;

        long long total_bp = 0;
        for (int v : nodes) total_bp += static_cast<long long>(node_sizes_[v]);
        if (total_bp < 400000LL) continue;

        // Decide whether to attempt a split, using the same objective family
        // for trigger, fragmentation guard, and acceptance.
        bool use_checkm = checkm_est_ && node_names_ && checkm_est_->has_marker_sets();
        CheckMQuality parent_q;
        if (use_checkm) {
            std::vector<std::string> cluster_names;
            cluster_names.reserve(nodes.size());
            for (int v : nodes) cluster_names.push_back((*node_names_)[v]);
            parent_q = checkm_est_->estimate_bin_quality(cluster_names);
            // +0.5 hysteresis to avoid splitting near-clean clusters under estimator noise
            if (parent_q.contamination <= 5.5f) continue;
            // Targeted mode: skip bins below the completeness floor
            if (min_completeness > 0.0f && parent_q.completeness < min_completeness) continue;
        } else {
            if (comm_q[c].dup_excess == 0) continue;
        }

        n_contaminated++;
        std::unordered_map<int, int> local_id;
        local_id.reserve(nodes.size());
        for (int i = 0; i < (int)nodes.size(); ++i) local_id[nodes[i]] = i;

        std::vector<WeightedEdge> sub_edges;
        for (int v : nodes) {
            for (const auto& [u, w] : adj_[v]) {
                auto it = local_id.find(u);
                if (it != local_id.end() && u > v)
                    sub_edges.push_back({local_id[v], it->second, static_cast<float>(w)});
            }
        }
        if (sub_edges.empty()) continue;

        LeidenConfig sub_config = effective_config;
        sub_config.use_initial_membership = false;
        sub_config.initial_membership.clear();
        sub_config.use_fixed_membership = false;
        sub_config.is_membership_fixed.clear();
        if (effective_config.use_node_sizes) {
            sub_config.node_sizes.resize(nodes.size());
            for (int i = 0; i < (int)nodes.size(); ++i)
                sub_config.node_sizes[i] = node_sizes_[nodes[i]];
        }

        // Resolution sweep: try from very low to high, stop at first attempt that
        // lands within the fragmentation guard. Large contaminated bins (weak internal
        // edges) need low resolution to find the 2-genome split; small bins need
        // higher resolution to avoid merging everything into one cluster.
        int max_children_pre = std::min(12, 2 + static_cast<int>(
            total_bp / static_cast<long long>(qconfig_.restart_min_viable_bp)));
        const float size_scale = std::max(1.0f, static_cast<float>(nodes.size()) / 100.0f);
        const float res_scales[] = {0.05f, 0.1f, 0.2f, 0.5f, 1.0f,
                                    std::min(3.0f, 3.0f / std::sqrt(size_scale)), 3.0f};

        ClusteringResult sub_result;
        bool found = false;
        for (float rs : res_scales) {
            sub_config.resolution = effective_config.resolution * rs;
#ifdef USE_LIBLEIDENALG
            LibLeidenalgBackend sub_leiden;
            sub_result = sub_leiden.cluster(sub_edges, nodes.size(), sub_config);
#else
            LeidenBackend sub_leiden;
            sub_result = sub_leiden.cluster(sub_edges, nodes.size(), sub_config);
#endif
            if (sub_result.num_clusters >= 2 && sub_result.num_clusters <= max_children_pre) {
                found = true;
                break;
            }
        }
        if (!found) continue;

        // --- Fragmentation guard (same objective family as trigger) ---
        if (use_checkm) {
            // Guard in bp-space: limit children to 2 + floor(parent_bp / min_viable_bp),
            // capped at 12. Also require >= 2 children above the size threshold.
            int max_children = max_children_pre;
            if (sub_result.num_clusters > max_children) continue;

            // Count viable children and spill bp.
            std::vector<long long> sub_bp(sub_result.num_clusters, 0);
            for (int i = 0; i < (int)nodes.size(); ++i)
                sub_bp[sub_result.labels[i]] += static_cast<long long>(node_sizes_[nodes[i]]);

            int viable_children = 0;
            long long spill_bp = 0;
            for (int sc = 0; sc < sub_result.num_clusters; ++sc) {
                if (sub_bp[sc] >= qconfig_.restart_min_viable_bp) viable_children++;
                else spill_bp += sub_bp[sc];
            }
            if (viable_children < 2) continue;
            if (static_cast<float>(spill_bp) / total_bp > 0.25f) continue;

            // --- CheckM acceptance ---
            // Two modes based on parent contamination level:
            //
            // Near-HQ preservation (cont < 15%): conservative — require HQ gain or
            //   contamination drops >=1% without completeness regression >2%.
            //
            // Chimeric purification (cont >= 15%): aggressive — accept if any viable
            //   child is clean (cont < 5%) OR weighted contamination drops by >=10%.
            //   Completeness loss is accepted since parent completeness is inflated by
            //   the union of multiple genomes' markers.
            int parent_hq = (parent_q.completeness >= 90.0f &&
                             parent_q.contamination <=  5.0f &&
                             total_bp >= qconfig_.restart_min_viable_bp) ? 1 : 0;

            float wcont = 0.0f, wcomp = 0.0f;
            int child_hq = 0;
            bool any_clean_child = false;
            for (int sc = 0; sc < sub_result.num_clusters; ++sc) {
                if (sub_bp[sc] < qconfig_.restart_min_viable_bp) continue;
                std::vector<std::string> sub_names;
                for (int i = 0; i < (int)nodes.size(); ++i)
                    if (sub_result.labels[i] == sc)
                        sub_names.push_back((*node_names_)[nodes[i]]);
                auto sq = checkm_est_->estimate_bin_quality(sub_names);
                float frac = static_cast<float>(sub_bp[sc]) / total_bp;
                wcont += frac * sq.contamination;
                wcomp += frac * sq.completeness;
                if (sq.completeness >= 90.0f && sq.contamination <= 5.0f) child_hq++;
                if (sq.contamination < 5.0f) any_clean_child = true;
            }

            bool accept;
            if (parent_q.contamination >= 15.0f) {
                // Chimeric purification: accept if we produce a clean child or
                // achieve a large absolute contamination reduction.
                accept = any_clean_child ||
                         (wcont <= parent_q.contamination - 10.0f);
            } else {
                // Near-HQ preservation: conservative.
                accept = (child_hq > parent_hq) ||
                         (wcont <= parent_q.contamination - 1.0f &&
                          wcomp >= parent_q.completeness - 2.0f);
            }
            if (!accept) continue;

            std::vector<int> sc_to_global(sub_result.num_clusters);
            sc_to_global[0] = c;
            for (int sc = 1; sc < sub_result.num_clusters; ++sc)
                sc_to_global[sc] = n_communities + sc - 1;
            for (int i = 0; i < (int)nodes.size(); ++i)
                labels[nodes[i]] = sc_to_global[sub_result.labels[i]];
            n_communities += sub_result.num_clusters - 1;
            n_split++;

            std::cerr << "[QualityLeiden] Split cluster " << c
                      << " (cont " << parent_q.contamination
                      << "% -> wcont " << wcont
                      << "%, hq " << parent_hq << "->" << child_hq
                      << ") into " << sub_result.num_clusters << " parts\n";
        } else {
            // --- Proxy acceptance: dup_excess must decrease ---
            if (sub_result.num_clusters > comm_q[c].dup_excess + 2) continue;

            std::vector<CommQuality> sub_q(sub_result.num_clusters);
            for (int i = 0; i < (int)nodes.size(); ++i) {
                int sc = sub_result.labels[i];
                for (const auto& entry : marker_index_->get_markers(nodes[i])) {
                    int m = entry.id, k = entry.copy_count;
                    if (sub_q[sc].cnt[m] == 0) sub_q[sc].present++;
                    sub_q[sc].cnt[m] += k;
                }
            }
            int new_dup = 0;
            for (auto& cq : sub_q)
                for (int m = 0; m < MAX_MARKER_TYPES; ++m)
                    if (cq.cnt[m] > 1) new_dup += cq.cnt[m] - 1;
            if (new_dup >= comm_q[c].dup_excess) continue;

            std::vector<int> sc_to_global(sub_result.num_clusters);
            sc_to_global[0] = c;
            for (int sc = 1; sc < sub_result.num_clusters; ++sc)
                sc_to_global[sc] = n_communities + sc - 1;
            for (int i = 0; i < (int)nodes.size(); ++i)
                labels[nodes[i]] = sc_to_global[sub_result.labels[i]];
            n_communities += sub_result.num_clusters - 1;
            n_split++;

            std::cerr << "[QualityLeiden] Split cluster " << c
                      << " (" << nodes.size() << " nodes, dup_excess "
                      << comm_q[c].dup_excess << " -> " << new_dup
                      << ") into " << sub_result.num_clusters << " parts\n";
        }
    }

    std::cerr << "[QualityLeiden] Phase 2: " << n_contaminated
              << " contaminated, " << n_split << " split, "
              << n_communities << " total\n";

    return {std::move(labels), n_communities};
}

void QualityLeidenBackend::run_phase3_rescue(
    std::vector<int>& labels, int n_communities) {

    if (!checkm_est_ || !node_names_ || !checkm_est_->has_marker_sets()) return;
    if (!marker_index_ || marker_index_->num_marker_contigs() == 0) return;

    // Build cluster membership and bp.
    std::vector<std::vector<int>> cluster_nodes(n_communities);
    std::vector<long long> cluster_bp(n_communities, 0);
    for (int i = 0; i < n_nodes_; ++i) {
        if (labels[i] < 0 || labels[i] >= n_communities) continue;
        cluster_nodes[labels[i]].push_back(i);
        cluster_bp[labels[i]] += static_cast<long long>(node_sizes_[i]);
    }

    // Build CommQuality per cluster.
    std::vector<CommQuality> comm_q(n_communities);
    init_comm_quality(labels, n_communities, comm_q);

    // Identify near-HQ rescue targets: 75-90% internal completeness, <5% cont.
    // 75% internal ≈ CheckM2 90% (empirically calibrated); these bins are just
    // below the threshold and could be pushed over by recruiting 1-2 contigs.
    std::vector<int> rescue_targets;
    for (int c = 0; c < n_communities; ++c) {
        if (cluster_bp[c] < qconfig_.restart_min_viable_bp) continue;
        std::vector<std::string> names;
        names.reserve(cluster_nodes[c].size());
        for (int v : cluster_nodes[c]) names.push_back((*node_names_)[v]);
        auto q = checkm_est_->estimate_bin_quality(names);
        if (q.completeness >= 75.0f && q.completeness < 90.0f && q.contamination < 5.0f)
            rescue_targets.push_back(c);
    }

    if (rescue_targets.empty()) {
        std::cerr << "[QualityLeiden] Phase 3: no near-HQ targets\n";
        return;
    }
    std::cerr << "[QualityLeiden] Phase 3: " << rescue_targets.size() << " near-HQ rescue targets\n";

    int n_moves = 0;
    for (int c : rescue_targets) {
        // Collect marker-bearing border nodes: not in c, adjacent to c via kNN.
        std::unordered_map<int, double> border_weight;
        for (int v : cluster_nodes[c]) {
            for (const auto& [u, w] : adj_[v]) {
                if (labels[u] != c && marker_index_->has_markers(u))
                    border_weight[u] += w;
            }
        }

        // Candidates: contigs whose net marker contribution is positive (more new
        // markers added than duplicates created). Pure-missing contigs are preferred
        // (tier 0); net-positive mixed contigs are also accepted (tier 1) provided
        // the resulting contamination estimate stays < 5%.
        std::vector<std::tuple<int, double, int>> candidates; // (tier, w, node)
        for (const auto& [node, w] : border_weight) {
            int n_missing = 0, n_dup = 0;
            for (const auto& me : marker_index_->get_markers(node)) {
                if (comm_q[c].cnt[me.id] == 0) ++n_missing;
                else                            ++n_dup;
            }
            if (n_missing == 0) continue;  // adds nothing new
            int tier = (n_dup == 0) ? 0 : 1;
            candidates.push_back({tier, w, node});
        }

        // Sort: pure-missing first (tier 0), then mixed (tier 1); within tier by weight.
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) {
                      if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
                      return std::get<1>(a) > std::get<1>(b);
                  });

        for (const auto& [tier, w_to_target, node] : candidates) {
            int old_label = labels[node];

            // For mixed-marker candidates (tier 1): build the hypothetical cluster
            // name list and run a full quality estimate to verify contamination
            // stays below 5% before committing the move.
            if (tier == 1) {
                std::vector<std::string> hyp_names;
                hyp_names.reserve(cluster_nodes[c].size() + 1);
                for (int v : cluster_nodes[c]) hyp_names.push_back((*node_names_)[v]);
                hyp_names.push_back((*node_names_)[node]);
                auto q = checkm_est_->estimate_bin_quality(hyp_names);
                if (q.contamination >= 5.0f) continue;
            }

            // Donor protection: don't steal from large viable bins unless:
            // (a) the contig is duplicated in the donor (removes contamination), OR
            // (b) the contig's kNN affinity to the target strongly exceeds its
            //     affinity to its current bin (it almost certainly belongs here).
            // Threshold lowered to 2× for near-HQ rescue (was 3×).
            if (cluster_bp[old_label] >= qconfig_.restart_min_viable_bp) {
                bool removes_donor_dup = false;
                for (const auto& me : marker_index_->get_markers(node)) {
                    if (comm_q[old_label].cnt[me.id] > 1) { removes_donor_dup = true; break; }
                }
                if (!removes_donor_dup) {
                    double w_to_current = 0.0;
                    for (const auto& [nbr, ew] : adj_[node])
                        if (labels[nbr] == old_label) w_to_current += ew;
                    if (w_to_target <= 2.0 * w_to_current) continue;
                }
            }

            // Accept the move.
            labels[node] = c;
            update_comm_quality(node, old_label, c, comm_q);
            // Keep cluster_nodes consistent for subsequent rescue iterations.
            auto& donor = cluster_nodes[old_label];
            donor.erase(std::find(donor.begin(), donor.end(), node));
            cluster_nodes[c].push_back(node);
            n_moves++;

            std::cerr << "[QualityLeiden] Phase 3: node " << node
                      << " cluster " << old_label << " -> " << c
                      << " (edge_w=" << w_to_target
                      << (tier == 1 ? ", mixed)\n" : ")\n");
        }
    }

    std::cerr << "[QualityLeiden] Phase 3 rescue: " << n_moves << " contigs moved\n";
}

CandidateSnapshot QualityLeidenBackend::evaluate_candidate(
    const std::vector<WeightedEdge>& raw_edges,
    int n_nodes,
    const LeidenConfig& base_cfg,
    float bandwidth,
    float resolution,
    int seed,
    bool run_phase2_for_score) {

    CandidateSnapshot snap;
    snap.resolution = resolution;
    snap.bandwidth  = bandwidth;
    snap.seed       = seed;

    // Re-weight edges from original_bandwidth to trial bandwidth.
    // Input weights: w = exp(-dist/bw_orig)  =>  w^(bw_orig/bw_new) = exp(-dist/bw_new)
    const float bw_ratio = qconfig_.original_bandwidth / bandwidth;
    std::vector<WeightedEdge> trial_edges;
    trial_edges.reserve(raw_edges.size());
    for (const auto& e : raw_edges) {
        float w = std::pow(e.w, bw_ratio);
        // Apply complementary edge bonus for marker-bearing pairs with no shared markers.
        if (qconfig_.scg_complement_bonus > 0.0f) {
            const auto& mu = marker_index_->get_markers(e.u);
            const auto& mv = marker_index_->get_markers(e.v);
            if (!mu.empty() && !mv.empty()) {
                int shared = 0;
                for (const auto& a : mu)
                    for (const auto& b : mv)
                        if (a.id == b.id) { shared++; break; }
                if (shared == 0)
                    w *= (1.0f + qconfig_.scg_complement_bonus);
            }
        }
        trial_edges.push_back({e.u, e.v, w});
    }

    LeidenConfig trial = base_cfg;
    trial.random_seed = seed;
    trial.resolution  = resolution;

#ifdef USE_LIBLEIDENALG
    LibLeidenalgBackend trial_leiden;
    snap.p1_result = trial_leiden.cluster(trial_edges, n_nodes, trial);
#else
    LeidenBackend trial_leiden;
    snap.p1_result = trial_leiden.cluster(trial_edges, n_nodes, trial);
#endif
    snap.score_p1 = scg_score_labels(snap.p1_result.labels, snap.p1_result.num_clusters);

    if (run_phase2_for_score && !qconfig_.skip_phase2) {
        auto [p2_labels, p2_nc] = run_phase2(snap.p1_result.labels, snap.p1_result.num_clusters, trial);
        snap.has_p2             = true;
        snap.labels_after_p2    = std::move(p2_labels);
        snap.num_clusters_after_p2 = p2_nc;
        snap.score_p2 = scg_score_labels(snap.labels_after_p2, snap.num_clusters_after_p2);
    }

    return snap;
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

    // Build adjacency from the original (unpenalized, original-bandwidth) edges.
    // Phase 2 contamination splitting uses this adjacency intentionally: bandwidth
    // trials only affect how Leiden forms initial clusters; the base similarity graph
    // is the correct substrate for splitting contaminated clusters.
    build_adjacency(edges, n_nodes, config);
    has_marker_.resize(n_nodes_);
    for (int i = 0; i < n_nodes_; ++i)
        has_marker_[i] = marker_index_->has_markers(i) ? 1 : 0;

    const bool do_restarts = (qconfig_.n_leiden_restarts > 1)
                           && (marker_index_->num_marker_contigs() > 0);

    const LeidenConfig& effective_config = config;
    const std::vector<WeightedEdge>& p1_edges = edges;

    // Initial clustering: single run, pure seed sweep, or bandwidth+seed search.
    std::vector<int> labels;
    int n_communities;
    bool phase2_already_done = false;

    if (do_restarts && qconfig_.skip_phase2) {
        // Seed sweep: Stage 1 runs 1 seed, Stage 2 allocates remaining budget.
        // Consensus is built from all seeds on the best-score partition.
        const int N     = qconfig_.n_leiden_restarts;

        const std::vector<float> res_grid = {effective_config.resolution};

        int seed_counter = 0;
        auto next_seed = [&]() { return spread_seed(effective_config.random_seed, seed_counter++); };

        // UCB1 arms — one per resolution
        struct ResArm {
            float res;
            std::vector<float> scores;
            std::vector<std::vector<int>> partitions;
            float mean() const {
                float s = 0; for (float x : scores) s += x; return s / scores.size();
            }
            // UCB1: mean + sqrt(2 ln(T) / n)
            float ucb(int total) const {
                int n = static_cast<int>(scores.size());
                if (n == 0) return 1e30f;
                return mean() + std::sqrt(2.0f * std::log(static_cast<float>(total)) / n);
            }
        };
        std::vector<ResArm> arms;
        for (float r : res_grid) arms.push_back({r, {}, {}});

        // Stage 1: one seed per arm (initialise UCB estimate)
        for (auto& arm : arms) {
            auto c = evaluate_candidate(edges, n_nodes, effective_config,
                                        qconfig_.original_bandwidth, arm.res, next_seed(), false);
            arm.scores.push_back(c.score_p1);
            arm.partitions.push_back(c.p1_result.labels);
        }

        // Stage 2: allocate remaining budget to the single arm
        const int N_stage2 = std::max(0, N - 1);
        int total_pulls = 1;
        for (int i = 0; i < N_stage2; ++i) {
            int best_arm = 0;
            float best_ucb = -1e30f;
            for (int a = 0; a < (int)arms.size(); ++a) {
                float u = arms[a].ucb(total_pulls);
                if (u > best_ucb) { best_ucb = u; best_arm = a; }
            }
            auto& arm = arms[best_arm];
            auto c = evaluate_candidate(edges, n_nodes, effective_config,
                                        qconfig_.original_bandwidth, arm.res, next_seed(), false);
            arm.scores.push_back(c.score_p1);
            arm.partitions.push_back(c.p1_result.labels);
            ++total_pulls;
        }

        // Select best arm by mean score; build best single-partition snapshot
        int best_arm_idx = 0;
        for (int a = 1; a < (int)arms.size(); ++a)
            if (arms[a].mean() > arms[best_arm_idx].mean()) best_arm_idx = a;
        auto& best_arm_ref = arms[best_arm_idx];

        CandidateSnapshot best;
        for (int i = 0; i < (int)best_arm_ref.scores.size(); ++i) {
            if (best_arm_ref.scores[i] > best.score_p1) {
                best.score_p1 = best_arm_ref.scores[i];
                best.resolution = best_arm_ref.res;
                best.p1_result.labels = best_arm_ref.partitions[i];
                best.p1_result.num_clusters =
                    *std::max_element(best_arm_ref.partitions[i].begin(),
                                      best_arm_ref.partitions[i].end()) + 1;
            }
        }

        LeidenConfig sweep_cfg = effective_config;
        sweep_cfg.resolution = best_arm_ref.res;

        const auto& all_partitions = best_arm_ref.partitions;

        std::cerr << "[QualityLeiden-MT] Seed sweep done: res=" << best_arm_ref.res
                  << " clusters=" << best.p1_result.num_clusters
                  << " score=" << best.score_p1 << "\n";

        // Consensus Leiden: modal partition as initial_membership on ORIGINAL edges.
        // Each node is assigned to its most-frequent cluster across N partitions.
        // Boundary contigs that oscillate between the primary bin and a garbage cluster
        // will have their modal cluster determined by majority vote — if more seeds
        // place them in the garbage cluster, they start there and Leiden keeps them
        // separated. Running on ORIGINAL (unscaled) edges preserves the modularity
        // landscape at the calibrated resolution=5.0, avoiding over-fragmentation.
        const int n_part = static_cast<int>(all_partitions.size());

        // Vote counting: for each node, count occurrences of each cluster ID.
        std::vector<std::unordered_map<int,int>> vote_counts(n_nodes);
        for (const auto& part : all_partitions)
            for (int v = 0; v < n_nodes; ++v)
                vote_counts[v][part[v]]++;

        // Pick modal (most-voted) cluster per node; break ties by lower cluster ID.
        std::vector<int> modal_labels(n_nodes);
        for (int v = 0; v < n_nodes; ++v) {
            int best_c = -1, best_cnt = 0;
            for (const auto& [c, cnt] : vote_counts[v]) {
                if (cnt > best_cnt || (cnt == best_cnt && c < best_c)) {
                    best_c = c; best_cnt = cnt;
                }
            }
            modal_labels[v] = best_c;
        }

        // Compact-relabel cluster IDs to 0,1,2,...
        {
            std::unordered_map<int,int> remap;
            int next_id = 0;
            for (int v = 0; v < n_nodes; ++v) {
                auto [it, inserted] = remap.emplace(modal_labels[v], next_id);
                if (inserted) ++next_id;
                modal_labels[v] = it->second;
            }
        }
        const int modal_clusters = *std::max_element(modal_labels.begin(), modal_labels.end()) + 1;
        std::cerr << "[QualityLeiden] Consensus modal init: " << n_part
                  << " partitions → " << modal_clusters << " modal clusters\n";

        LeidenConfig consensus_cfg = sweep_cfg;  // carries best_res
        consensus_cfg.use_initial_membership = true;
        consensus_cfg.initial_membership = modal_labels;
        for (int s = 0; s < 3; ++s) {
            consensus_cfg.random_seed = spread_seed(effective_config.random_seed, 2000 + s);
            CandidateSnapshot c;
            c.resolution = consensus_cfg.resolution;
            c.bandwidth  = qconfig_.original_bandwidth;
            c.seed       = consensus_cfg.random_seed;
#ifdef USE_LIBLEIDENALG
            LibLeidenalgBackend cl;
            cl.set_seed(consensus_cfg.random_seed);
            c.p1_result = cl.cluster(edges, n_nodes, consensus_cfg);
#else
            LeidenBackend cl;
            cl.set_seed(consensus_cfg.random_seed);
            c.p1_result = cl.cluster(edges, n_nodes, consensus_cfg);
#endif
            c.score_p1 = scg_score_labels(c.p1_result.labels, c.p1_result.num_clusters);
            std::cerr << "  [consensus s=" << s << "] clusters=" << c.p1_result.num_clusters
                      << " score=" << c.score_p1;
            if (c.score_p1 > best.score_p1) {
                best = std::move(c);
                std::cerr << " NEW BEST";
            }
            std::cerr << "\n";
        }

        labels        = std::move(best.p1_result.labels);
        n_communities = best.p1_result.num_clusters;
        std::cerr << "[QualityLeiden-MT] Final partition: " << n_communities
                  << " clusters, score=" << best.score_p1 << "\n";

    } else {
#ifdef USE_LIBLEIDENALG
        LibLeidenalgBackend fast_backend;
        ClusteringResult initial_result = fast_backend.cluster(p1_edges, n_nodes, effective_config);
#else
        ClusteringResult initial_result = LeidenBackend::cluster(p1_edges, n_nodes, effective_config);
#endif
        std::cerr << "[QualityLeiden-MT] Initial clustering: " << initial_result.num_clusters
                  << " clusters, modularity=" << initial_result.modularity << "\n";
        labels = std::move(initial_result.labels);
        n_communities = initial_result.num_clusters;
    }

    // Initialize community statistics
    std::vector<double> comm_weights(n_communities, 0.0);
    std::vector<double> comm_sizes(n_communities, 0.0);
    for (int i = 0; i < n_nodes_; ++i) {
        comm_weights[labels[i]] += node_weights_[i];
        comm_sizes[labels[i]]   += node_sizes_[i];
    }
    std::vector<CommQuality> comm_q;
    init_comm_quality(labels, n_communities, comm_q);
    std::cerr << "[QualityLeiden-MT] " << marker_index_->num_marker_contigs()
              << " contigs have markers across " << n_communities << " clusters\n";

    // Phase 2: contamination splitting (skip if already done by restart search,
    // or if skip_phase2 is set, e.g., plain bandwidth restarts without --quality-leiden).
    if (!phase2_already_done && !qconfig_.skip_phase2) {
        auto [p2_labels, p2_nc] = run_phase2(std::move(labels), n_communities, effective_config);
        labels        = std::move(p2_labels);
        n_communities = p2_nc;
        init_comm_quality(labels, n_communities, comm_q);
    }

    // Phase 3: targeted rescue for near-HQ bins.
    // Recruits missing-marker contigs from kNN neighbors to push borderline bins
    // over the completeness threshold. Only enabled when CheckM estimator is set.
    if (checkm_est_ && node_names_ && checkm_est_->has_marker_sets()) {
        n_communities = compact_labels(labels);
        run_phase3_rescue(labels, n_communities);
    }

    // Compact labels and compute final modularity
    n_communities = compact_labels(labels);
    double modularity = 0.0;  // restart path doesn't track a single modularity value

    ClusteringResult result;
    result.labels = labels;
    result.modularity = modularity;
    result.num_clusters = n_communities;

    std::cerr << "[QualityLeiden-MT] Final: " << result.num_clusters
              << " clusters, modularity=" << modularity << "\n";

    return result;
}

int QualityLeidenBackend::decontaminate(std::vector<int>& labels,
                                        const LeidenConfig& config,
                                        float min_completeness) {
    if (!checkm_est_ || !node_names_ || !checkm_est_->has_marker_sets())
        return compact_labels(labels);

    // Phase 4A: iterative graph-based Leiden splitting.
    int n = compact_labels(labels);
    for (int iter = 0; iter < 3; ++iter) {
        auto [new_labels, new_n] = run_phase2(std::move(labels), n, config, 0.0f);
        labels = std::move(new_labels);
        if (new_n == n) break;
        n = new_n;
    }

    // Phase 4E: extended-neighborhood Leiden re-clustering.
    // Runs Leiden on bin contigs + 1-hop kNN neighbors. External neighbors
    // anchor contaminating contigs to their true genome cluster, letting Leiden
    // find cuts that Phase 4A (isolated subgraph) misses. Multiple seeds tried.
    {
        auto [new_labels, new_n] = run_phase4_extended(std::move(labels), n, config);
        labels = std::move(new_labels);
        n = new_n;
    }

    // Phase 4M: marker-direct eviction.
    // For bins that survive 4A+4E, directly evict the lowest-connectivity contig
    // carrying each duplicated SCG marker. Validated with CheckM.
    if (marker_index_) {
        auto [new_labels, new_n] = run_phase4_marker(std::move(labels), n);
        labels = std::move(new_labels);
        n = new_n;
    }

    // Phase 4G: cross-cluster graph-boundary eviction.
    // Contaminating contigs have high out-of-bin kNN weight — they sit at the
    // partition boundary and could fall into a different cluster with a different
    // Leiden seed. This is the mechanism behind 12 HQ vs 11 HQ variance.
    {
        auto [new_labels, new_n] = run_phase4_graph(std::move(labels), n);
        labels = std::move(new_labels);
        n = new_n;
    }

    // Phase 4C: Gaussian curvature eviction.
    // Catches contaminating contigs that carry no duplicate SCG markers.
    // Uses local SVD of kNN embedding neighborhoods: contaminating contigs
    // sit at cluster boundaries → high off-tangent-plane curvature.
    {
        auto [new_labels, new_n] = run_phase4_curvature(std::move(labels), n);
        labels = std::move(new_labels);
        n = new_n;
    }

    return n;
}

std::pair<std::vector<int>, int> QualityLeidenBackend::run_phase4_marker(
    std::vector<int> labels,
    int n_communities) {

    if (!marker_index_ || !checkm_est_ || !node_names_ ||
        !checkm_est_->has_marker_sets())
        return {std::move(labels), n_communities};

    std::vector<std::vector<int>> cluster_nodes(n_communities);
    for (int i = 0; i < n_nodes_; ++i)
        if (labels[i] >= 0 && labels[i] < n_communities)
            cluster_nodes[labels[i]].push_back(i);

    int n_split = 0;

    for (int c = 0; c < n_communities; ++c) {
        const auto& nodes = cluster_nodes[c];
        if (nodes.empty()) continue;

        long long total_bp = 0;
        for (int v : nodes) total_bp += static_cast<long long>(node_sizes_[v]);
        if (total_bp < 400000LL) continue;

        // Trigger: CheckM contamination > 5.5%
        std::vector<std::string> cluster_names;
        cluster_names.reserve(nodes.size());
        for (int v : nodes) cluster_names.push_back((*node_names_)[v]);
        CheckMQuality parent_q = checkm_est_->estimate_bin_quality(cluster_names);
        if (parent_q.contamination <= 5.5f) continue;

        // Build marker → carrier contigs map for this bin.
        std::unordered_map<int, std::vector<int>> marker_to_nodes;
        for (int v : nodes)
            for (const auto& entry : marker_index_->get_markers(v))
                marker_to_nodes[entry.id].push_back(v);

        // Find duplicated markers (present in >= 2 contigs).
        std::vector<std::pair<int, std::vector<int>>> dup_markers;
        for (auto& [mid, carriers] : marker_to_nodes)
            if ((int)carriers.size() >= 2)
                dup_markers.push_back({mid, std::move(carriers)});

        if (dup_markers.empty()) continue;

        // Compute mean intra-bin edge weight per node (connectivity proxy).
        std::unordered_set<int> bin_set(nodes.begin(), nodes.end());
        std::unordered_map<int, double> intra_mean;
        for (int v : nodes) {
            double wsum = 0.0;
            int cnt = 0;
            for (const auto& [u, w] : adj_[v]) {
                if (bin_set.count(u)) { wsum += w; cnt++; }
            }
            intra_mean[v] = cnt > 0 ? wsum / cnt : 0.0;
        }

        // Sole carriers: contigs that are the ONLY carrier of some marker in this bin.
        // Never evict these — doing so would permanently lose those markers.
        std::unordered_set<int> sole_carriers;
        for (const auto& [mid, carriers] : marker_to_nodes)
            if (carriers.size() == 1) sole_carriers.insert(carriers[0]);

        // For each duplicated marker, vote the weakest-connected carrier for eviction.
        std::unordered_map<int, int> evict_votes;
        for (const auto& [mid, carriers] : dup_markers) {
            int weakest = carriers[0];
            for (int v : carriers)
                if (intra_mean[v] < intra_mean[weakest]) weakest = v;
            evict_votes[weakest]++;
        }

        // Only evict contigs that are not sole carriers of any marker.
        std::vector<int> candidates;
        for (const auto& [v, votes] : evict_votes)
            if (!sole_carriers.count(v)) candidates.push_back(v);
        if (candidates.empty()) continue;

        // Score the remaining bin without evicted contigs.
        std::unordered_set<int> evict_set(candidates.begin(), candidates.end());
        std::vector<std::string> remaining_names;
        long long remaining_bp = 0, evicted_bp = 0;
        for (int v : nodes) {
            if (evict_set.count(v)) evicted_bp  += static_cast<long long>(node_sizes_[v]);
            else                   { remaining_names.push_back((*node_names_)[v]);
                                     remaining_bp += static_cast<long long>(node_sizes_[v]); }
        }
        if (remaining_bp < qconfig_.restart_min_viable_bp) continue;

        auto remaining_q = checkm_est_->estimate_bin_quality(remaining_names);

        // Only accept if we actually achieve a clean bin (<5% cont) without
        // significant completeness loss (≤5% drop). This ensures Phase 4M
        // is net-positive: contaminated → clean, not clean → slightly less contaminated.
        bool accept = remaining_q.contamination < 5.0f &&
                      remaining_q.completeness >= parent_q.completeness - 5.0f;
        if (!accept) continue;

        int new_label = n_communities + n_split;
        for (int v : candidates) labels[v] = new_label;
        n_split++;

        std::cerr << "[QualityLeiden] Phase 4M: evicted " << candidates.size()
                  << " contigs (" << evicted_bp / 1000 << "kb) from cluster " << c
                  << " (cont " << parent_q.contamination << "% -> "
                  << remaining_q.contamination << "%, comp "
                  << parent_q.completeness << "% -> " << remaining_q.completeness << "%)\n";
    }

    std::cerr << "[QualityLeiden] Phase 4M: " << n_split << " bins cleaned, "
              << (n_communities + n_split) << " total\n";

    return {std::move(labels), n_communities + n_split};
}

std::pair<std::vector<int>, int> QualityLeidenBackend::run_phase4_extended(
    std::vector<int> labels,
    int n_communities,
    const LeidenConfig& /*base_config*/) {

    if (!checkm_est_ || !node_names_ || !checkm_est_->has_marker_sets())
        return {std::move(labels), n_communities};

    std::vector<std::vector<int>> cluster_nodes(n_communities);
    for (int i = 0; i < n_nodes_; ++i)
        if (labels[i] >= 0 && labels[i] < n_communities)
            cluster_nodes[labels[i]].push_back(i);

    int n_split = 0;

    for (int c = 0; c < n_communities; ++c) {
        const auto& nodes = cluster_nodes[c];
        if ((int)nodes.size() <= 1) continue;

        long long total_bp = 0;
        for (int v : nodes) total_bp += static_cast<long long>(node_sizes_[v]);
        if (total_bp < 400000LL) continue;

        std::vector<std::string> cluster_names;
        cluster_names.reserve(nodes.size());
        for (int v : nodes) cluster_names.push_back((*node_names_)[v]);
        CheckMQuality parent_q = checkm_est_->estimate_bin_quality(cluster_names);
        if (parent_q.contamination <= qconfig_.phase4e_entry_contamination) continue;

        // --- Belonging score per contig ---
        // belonging_score(v) = fraction of v's total kNN edge weight going to
        // same-cluster neighbors. A contaminating contig from genome B embedded in
        // cluster A has most of its kNN weight pointing towards B's contigs in
        // other clusters, so belonging_score << 1. Pure-assembly contigs score ≈ 1.
        std::vector<double> belonging(nodes.size(), 0.0);
        for (int i = 0; i < (int)nodes.size(); ++i) {
            int v = nodes[i];
            double total_w = 0.0, intra_w = 0.0;
            for (const auto& [u, w] : adj_[v]) {
                total_w += w;
                if (labels[u] == c) intra_w += w;
            }
            belonging[i] = (total_w > 1e-9) ? intra_w / total_w : 0.0;
        }

        double mean_b = 0.0;
        for (double b : belonging) mean_b += b;
        mean_b /= (double)nodes.size();
        double var_b = 0.0;
        for (double b : belonging) var_b += (b - mean_b) * (b - mean_b);
        double std_b = std::sqrt(var_b / (double)nodes.size());
        const double outlier_threshold = mean_b - qconfig_.phase4e_sigma_threshold * std_b;

        // --- Per-bin marker duplicate counts ---
        std::unordered_map<int, int> marker_cnt;
        if (marker_index_) {
            for (int v : nodes)
                for (const auto& me : marker_index_->get_markers(v))
                    marker_cnt[me.id]++;
        }

        // --- Sole-carrier set ---
        // Contigs that are the only carrier of some marker in this bin are never
        // evicted: removing them would reduce completeness with no contamination gain.
        std::unordered_set<int> sole_carriers;
        if (marker_index_) {
            std::unordered_map<int, std::vector<int>> marker_to_nodes;
            for (int v : nodes)
                for (const auto& me : marker_index_->get_markers(v))
                    marker_to_nodes[me.id].push_back(v);
            for (const auto& [mid, carriers] : marker_to_nodes)
                if (carriers.size() == 1) sole_carriers.insert(carriers[0]);
        }

        // --- Contamination-signal score per contig ---
        // score = (1 - belonging) × n_dup_markers
        // Candidates must be: (a) embedding outliers (belonging < threshold),
        //                      (b) carry at least one duplicated marker in this bin,
        //                      (c) not sole carriers.
        std::vector<std::pair<double, int>> candidates; // (score, node index in nodes[])
        for (int i = 0; i < (int)nodes.size(); ++i) {
            int v = nodes[i];
            if (sole_carriers.count(v)) continue;
            if (belonging[i] >= outlier_threshold) continue;
            int n_dup = 0;
            if (marker_index_)
                for (const auto& me : marker_index_->get_markers(v))
                    if (marker_cnt[me.id] > 1) ++n_dup;
            if (n_dup == 0) continue;
            candidates.push_back({(1.0 - belonging[i]) * n_dup, i});
        }

        if (candidates.empty()) continue;

        // Sort descending by contamination signal (most suspicious first).
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Try evicting progressively smaller subsets of candidates.
        // For each subset: build kept_names, validate with CheckM, accept first pass.
        // Subset sizes tried: 100%, 75%, 50%, 25% (minimum 1 contig).
        const float fracs[] = {1.0f, 0.75f, 0.5f, 0.25f};
        for (float frac : fracs) {
            int n_try = std::max(1, (int)std::round(frac * (float)candidates.size()));
            std::unordered_set<int> evict_set;
            for (int k = 0; k < n_try; ++k)
                evict_set.insert(nodes[candidates[k].second]);

            std::vector<std::string> kept_names;
            std::vector<int> evict_list;
            for (int v : nodes) {
                if (evict_set.count(v)) evict_list.push_back(v);
                else kept_names.push_back((*node_names_)[v]);
            }
            if (kept_names.empty()) continue;

            auto kept_q = checkm_est_->estimate_bin_quality(kept_names);
            if (kept_q.contamination < 5.0f &&
                kept_q.completeness >= parent_q.completeness - 5.0f) {

                int new_label = n_communities + n_split;
                long long evicted_bp = 0;
                for (int v : evict_list) {
                    labels[v] = new_label;
                    evicted_bp += static_cast<long long>(node_sizes_[v]);
                }
                n_split++;

                std::cerr << "[QualityLeiden] Phase 4E: evicted " << evict_list.size()
                          << " contigs (" << evicted_bp / 1000 << "kb) from cluster " << c
                          << " (cont " << parent_q.contamination << "% -> "
                          << kept_q.contamination << "%, comp "
                          << parent_q.completeness << "% -> " << kept_q.completeness
                          << "%, sigma=" << qconfig_.phase4e_sigma_threshold
                          << " candidates=" << candidates.size()
                          << " tried=" << n_try << ")\n";
                break;
            }
        }
    }

    std::cerr << "[QualityLeiden] Phase 4E: " << n_split << " bins cleaned, "
              << (n_communities + n_split) << " total\n";

    return {std::move(labels), n_communities + n_split};
}

std::pair<std::vector<int>, int> QualityLeidenBackend::run_phase4_graph(
    std::vector<int> labels,
    int n_communities) {

    if (!checkm_est_ || !node_names_ || !checkm_est_->has_marker_sets())
        return {std::move(labels), n_communities};

    std::vector<std::vector<int>> cluster_nodes(n_communities);
    for (int i = 0; i < n_nodes_; ++i)
        if (labels[i] >= 0 && labels[i] < n_communities)
            cluster_nodes[labels[i]].push_back(i);

    int n_split = 0;

    for (int c = 0; c < n_communities; ++c) {
        const auto& nodes = cluster_nodes[c];
        if (nodes.empty()) continue;

        long long total_bp = 0;
        for (int v : nodes) total_bp += static_cast<long long>(node_sizes_[v]);
        if (total_bp < 400000LL) continue;

        std::vector<std::string> cluster_names;
        cluster_names.reserve(nodes.size());
        for (int v : nodes) cluster_names.push_back((*node_names_)[v]);
        CheckMQuality parent_q = checkm_est_->estimate_bin_quality(cluster_names);
        if (parent_q.contamination <= 5.5f) continue;

        // Sole-carrier protection.
        std::unordered_set<int> sole_carriers;
        if (marker_index_) {
            std::unordered_map<int, std::vector<int>> marker_to_nodes;
            for (int v : nodes)
                for (const auto& entry : marker_index_->get_markers(v))
                    marker_to_nodes[entry.id].push_back(v);
            for (const auto& [mid, carriers] : marker_to_nodes)
                if (carriers.size() == 1) sole_carriers.insert(carriers[0]);
        }

        // Cross-cluster ratio: w_out / (w_in + w_out).
        // Contaminating contigs are kNN-connected to other clusters → high ratio.
        std::unordered_set<int> bin_set(nodes.begin(), nodes.end());
        std::vector<std::pair<float, int>> cross_scores;
        cross_scores.reserve(nodes.size());

        for (int ii = 0; ii < (int)nodes.size(); ++ii) {
            int v = nodes[ii];
            double w_in = 0.0, w_out = 0.0;
            for (const auto& [u, w] : adj_[v]) {
                if (bin_set.count(u)) w_in  += w;
                else                  w_out += w;
            }
            float total = static_cast<float>(w_in + w_out);
            float ratio = (total > 0.0f) ? static_cast<float>(w_out / total) : 0.0f;
            cross_scores.push_back({ratio, ii});
        }

        std::sort(cross_scores.begin(), cross_scores.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Greedy eviction: highest cross-ratio first until cont < 5%.
        std::vector<int> evicted;
        bool found = false;
        for (auto& [ratio, ii] : cross_scores) {
            if ((int)evicted.size() >= (int)nodes.size() / 3) break;
            int v = nodes[ii];
            if (sole_carriers.count(v)) continue;
            evicted.push_back(v);

            std::unordered_set<int> evict_set(evicted.begin(), evicted.end());
            std::vector<std::string> remaining_names;
            long long remaining_bp = 0;
            for (int vv : nodes) {
                if (!evict_set.count(vv)) {
                    remaining_names.push_back((*node_names_)[vv]);
                    remaining_bp += static_cast<long long>(node_sizes_[vv]);
                }
            }
            if (remaining_bp < qconfig_.restart_min_viable_bp) break;

            auto remaining_q = checkm_est_->estimate_bin_quality(remaining_names);
            if (remaining_q.contamination < 5.0f &&
                remaining_q.completeness >= parent_q.completeness - 5.0f) {
                found = true;
                long long evicted_bp = 0;
                for (int vv : evicted) evicted_bp += static_cast<long long>(node_sizes_[vv]);
                std::cerr << "[QualityLeiden] Phase 4G: evicted " << evicted.size()
                          << " contigs (" << evicted_bp / 1000 << "kb) from cluster " << c
                          << " (cont " << parent_q.contamination << "% -> "
                          << remaining_q.contamination << "%, comp "
                          << parent_q.completeness << "% -> " << remaining_q.completeness
                          << "%, top_cross=" << cross_scores[0].first << ")\n";
                break;
            }
        }

        if (!found) continue;

        int new_label = n_communities + n_split;
        for (int v : evicted) labels[v] = new_label;
        n_split++;
    }

    std::cerr << "[QualityLeiden] Phase 4G: " << n_split << " bins cleaned, "
              << (n_communities + n_split) << " total\n";

    return {std::move(labels), n_communities + n_split};
}

std::pair<std::vector<int>, int> QualityLeidenBackend::run_phase4_curvature(
    std::vector<int> labels,
    int n_communities) {

    if (!embeddings_ || !checkm_est_ || !node_names_ ||
        !checkm_est_->has_marker_sets())
        return {std::move(labels), n_communities};

    const auto& emb = *embeddings_;
    if (emb.empty() || (int)emb.size() != n_nodes_) return {std::move(labels), n_communities};
    const int D = static_cast<int>(emb[0].size());

    std::vector<std::vector<int>> cluster_nodes(n_communities);
    for (int i = 0; i < n_nodes_; ++i)
        if (labels[i] >= 0 && labels[i] < n_communities)
            cluster_nodes[labels[i]].push_back(i);

    int n_split = 0;

    for (int c = 0; c < n_communities; ++c) {
        const auto& nodes = cluster_nodes[c];
        int n = static_cast<int>(nodes.size());
        if (n < 5) continue;

        long long total_bp = 0;
        for (int v : nodes) total_bp += static_cast<long long>(node_sizes_[v]);
        if (total_bp < 400000LL) continue;

        // Only process contaminated bins.
        std::vector<std::string> cluster_names;
        cluster_names.reserve(n);
        for (int v : nodes) cluster_names.push_back((*node_names_)[v]);
        CheckMQuality parent_q = checkm_est_->estimate_bin_quality(cluster_names);
        if (parent_q.contamination <= 5.5f) continue;

        // Sole-carrier protection.
        std::unordered_set<int> sole_carriers;
        if (marker_index_) {
            std::unordered_map<int, std::vector<int>> marker_to_nodes;
            for (int v : nodes)
                for (const auto& entry : marker_index_->get_markers(v))
                    marker_to_nodes[entry.id].push_back(v);
            for (const auto& [mid, carriers] : marker_to_nodes)
                if (carriers.size() == 1) sole_carriers.insert(carriers[0]);
        }

        // Compute per-contig Gaussian curvature proxy:
        // off-tangent-plane energy = total deviation² − projection onto local tangent plane².
        // Contaminating contigs lie at the boundary between two genome manifolds
        // → their k-NN spans both clusters → local tangent plane is poorly defined
        // → high off-tangent residual = high curvature.
        const int k = std::max(3, std::min(10, n / 3));

        std::vector<float> curvature(n, 0.0f);

        for (int ii = 0; ii < n; ++ii) {
            int vi = nodes[ii];

            // Brute-force k-NN within bin.
            std::vector<std::pair<float, int>> dists(n);
            for (int jj = 0; jj < n; ++jj) {
                int vj = nodes[jj];
                float d2 = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float diff = emb[vi][d] - emb[vj][d];
                    d2 += diff * diff;
                }
                dists[jj] = {d2, jj};
            }
            // k+1 so index 0 (self) + k neighbors
            std::partial_sort(dists.begin(), dists.begin() + k + 1, dists.end());

            // Local centroid of k neighbors (exclude self at dists[0]).
            Eigen::VectorXf centroid = Eigen::VectorXf::Zero(D);
            for (int jj = 1; jj <= k; ++jj) {
                int vj = nodes[dists[jj].second];
                for (int d = 0; d < D; ++d) centroid(d) += emb[vj][d];
            }
            centroid /= static_cast<float>(k);

            // k×D matrix X of centered neighbor vectors.
            Eigen::MatrixXf X(k, D);
            for (int jj = 1; jj <= k; ++jj) {
                int vj = nodes[dists[jj].second];
                for (int d = 0; d < D; ++d)
                    X(jj - 1, d) = emb[vj][d] - centroid(d);
            }

            // Thin SVD → top-2 right singular vectors = local tangent plane basis.
            const int d_tang = std::min(2, k - 1);
            Eigen::BDCSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeThinV);
            Eigen::MatrixXf V_d = svd.matrixV().leftCols(d_tang);  // D × d_tang

            // Deviation of contig i from local centroid.
            Eigen::VectorXf dev(D);
            for (int d = 0; d < D; ++d) dev(d) = emb[vi][d] - centroid(d);

            // Off-tangent score = total energy − projected energy.
            float proj_sq = (V_d.transpose() * dev).squaredNorm();
            curvature[ii] = dev.squaredNorm() - proj_sq;
        }

        // Sort by curvature descending: highest = most likely contaminant.
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return curvature[a] > curvature[b]; });

        // Greedy eviction: add contigs in curvature order until cont < 5%.
        std::vector<int> evicted;
        bool found = false;
        for (int ii = 0; ii < n && (int)evicted.size() < n / 3; ++ii) {
            int v = nodes[order[ii]];
            if (sole_carriers.count(v)) continue;
            evicted.push_back(v);

            std::unordered_set<int> evict_set(evicted.begin(), evicted.end());
            std::vector<std::string> remaining_names;
            long long remaining_bp = 0;
            for (int vv : nodes) {
                if (!evict_set.count(vv)) {
                    remaining_names.push_back((*node_names_)[vv]);
                    remaining_bp += static_cast<long long>(node_sizes_[vv]);
                }
            }
            if (remaining_bp < qconfig_.restart_min_viable_bp) break;

            auto remaining_q = checkm_est_->estimate_bin_quality(remaining_names);
            if (remaining_q.contamination < 5.0f &&
                remaining_q.completeness >= parent_q.completeness - 5.0f) {
                found = true;
                long long evicted_bp = 0;
                for (int vv : evicted) evicted_bp += static_cast<long long>(node_sizes_[vv]);
                std::cerr << "[QualityLeiden] Phase 4C: evicted " << evicted.size()
                          << " contigs (" << evicted_bp / 1000 << "kb) from cluster " << c
                          << " (cont " << parent_q.contamination << "% -> "
                          << remaining_q.contamination << "%, comp "
                          << parent_q.completeness << "% -> " << remaining_q.completeness << "%)\n";
                break;
            }
        }

        if (!found) continue;

        int new_label = n_communities + n_split;
        for (int v : evicted) labels[v] = new_label;
        n_split++;
    }

    std::cerr << "[QualityLeiden] Phase 4C: " << n_split << " bins cleaned, "
              << (n_communities + n_split) << " total\n";

    return {std::move(labels), n_communities + n_split};
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

    std::vector<double> base_deltas;
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

        double base_delta = delta_modularity_fast(
            node, current, target,
            edges_to_current, edges_to_target,
            comm_weights, comm_sizes, resolution);

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

    std::cerr << "[QualityLeiden] Calibration (dense, modularity): s_R=" << s_R << ", s_M=" << s_M
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

                    double base_delta = delta_modularity_fast(
                        node, current, c, edges_to_current, my_ctx.acc[c],
                        comm_weights, comm_sizes, resolution);

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
                double base_delta = delta_modularity_fast(
                    node, current, best, edges_to_current, edges_to_best,
                    comm_weights, comm_sizes, resolution);
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

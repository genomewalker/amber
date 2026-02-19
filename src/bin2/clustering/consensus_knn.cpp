#include "consensus_knn.h"
#include "hnsw_knn_index.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace amber::bin2 {

ConsensusKnnBuilder::ConsensusKnnBuilder(const ConsensusKnnConfig& cfg)
    : cfg_(cfg) {}

void ConsensusKnnBuilder::add_embeddings(const std::vector<std::vector<float>>& embeddings) {
    all_embeddings_.push_back(embeddings);
}

std::vector<WeightedEdge> ConsensusKnnBuilder::build() const {
    size_t n_runs = all_embeddings_.size();
    if (n_runs == 0) return {};

    size_t n_contigs = all_embeddings_[0].size();

    struct EdgeKey {
        int lo, hi;
        bool operator==(const EdgeKey& other) const {
            return lo == other.lo && hi == other.hi;
        }
    };
    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& k) const {
            return std::hash<int64_t>()(
                (static_cast<int64_t>(k.lo) << 32) | static_cast<int64_t>(k.hi));
        }
    };

    struct EdgeAgg {
        float sum_weight = 0.0f;
        int count = 0;
    };

    std::unordered_map<EdgeKey, EdgeAgg, EdgeKeyHash> edge_map;

    for (size_t run = 0; run < n_runs; ++run) {
        const auto& embeddings = all_embeddings_[run];

        // Build kNN index with same params bin2.cpp uses
        KnnConfig knn_cfg;
        knn_cfg.k = cfg_.k;
        knn_cfg.ef_search = 200;
        knn_cfg.M = 16;
        knn_cfg.ef_construction = 200;
        knn_cfg.random_seed = 42 + static_cast<int>(run);

        HnswKnnIndex index(knn_cfg);
        index.build(embeddings);
        NeighborList neighbors = index.query_all();

        // Percentile distance cutoff (same logic as compute_edge_distances)
        std::vector<float> all_dists;
        for (size_t i = 0; i < neighbors.size(); ++i) {
            for (float d : neighbors.dists[i]) {
                all_dists.push_back(d);
            }
        }

        float dist_cutoff = std::numeric_limits<float>::max();
        if (cfg_.partgraph_ratio < 100 && !all_dists.empty()) {
            size_t idx = static_cast<size_t>(
                static_cast<double>(cfg_.partgraph_ratio) / 100.0 * all_dists.size());
            idx = std::min(idx, all_dists.size() - 1);
            std::nth_element(all_dists.begin(),
                             all_dists.begin() + static_cast<ptrdiff_t>(idx),
                             all_dists.end());
            dist_cutoff = all_dists[idx];
        }

        // Deduplicate within this run: for each undirected edge (lo,hi), take max weight
        std::unordered_map<EdgeKey, float, EdgeKeyHash> run_edges;
        for (size_t i = 0; i < neighbors.size(); ++i) {
            const auto& ids = neighbors.ids[i];
            const auto& dists = neighbors.dists[i];

            for (size_t k = 0; k < ids.size(); ++k) {
                int j = ids[k];
                float dist = dists[k];

                if (dist > dist_cutoff) continue;

                int lo = std::min(static_cast<int>(i), j);
                int hi = std::max(static_cast<int>(i), j);
                if (lo == hi) continue;

                float w = std::exp(-dist / cfg_.bandwidth);

                EdgeKey key{lo, hi};
                auto it = run_edges.find(key);
                if (it == run_edges.end()) {
                    run_edges[key] = w;
                } else {
                    it->second = std::max(it->second, w);
                }
            }
        }

        // Merge into global accumulator
        for (const auto& [key, w] : run_edges) {
            auto& agg = edge_map[key];
            agg.sum_weight += w;
            agg.count += 1;
        }
    }

    // Compute consensus weight for all edges and filter by frequency
    float n_runs_f = static_cast<float>(n_runs);

    struct ScoredEdge {
        int lo, hi;
        float w_cons;
    };

    std::vector<WeightedEdge> consensus_edges;
    std::vector<ScoredEdge> all_scored;
    consensus_edges.reserve(edge_map.size());
    all_scored.reserve(edge_map.size());

    for (const auto& [key, agg] : edge_map) {
        float freq = static_cast<float>(agg.count) / n_runs_f;
        float w_cons = (agg.sum_weight / static_cast<float>(agg.count)) * freq;

        all_scored.push_back({key.lo, key.hi, w_cons});

        if (freq >= cfg_.min_freq) {
            consensus_edges.emplace_back(key.lo, key.hi, w_cons);
        }
    }

    // Per-node fallback: nodes with degree < min_degree get their best edges added back
    std::vector<int> degree(n_contigs, 0);
    for (const auto& e : consensus_edges) {
        degree[e.u]++;
        degree[e.v]++;
    }

    bool any_fallback = false;
    std::vector<bool> needs_fallback(n_contigs, false);
    for (size_t i = 0; i < n_contigs; ++i) {
        if (degree[i] < cfg_.min_degree) {
            needs_fallback[i] = true;
            any_fallback = true;
        }
    }

    if (any_fallback) {
        // Sort all edges by w_cons descending for greedy selection
        std::sort(all_scored.begin(), all_scored.end(),
                  [](const ScoredEdge& a, const ScoredEdge& b) {
                      return a.w_cons > b.w_cons;
                  });

        std::unordered_map<EdgeKey, bool, EdgeKeyHash> in_consensus;
        for (const auto& e : consensus_edges) {
            in_consensus[{e.u, e.v}] = true;
        }

        for (const auto& se : all_scored) {
            if (!needs_fallback[se.lo] && !needs_fallback[se.hi]) continue;

            EdgeKey key{se.lo, se.hi};
            if (in_consensus.count(key)) continue;

            consensus_edges.emplace_back(se.lo, se.hi, se.w_cons);
            in_consensus[key] = true;
            degree[se.lo]++;
            degree[se.hi]++;

            if (degree[se.lo] >= cfg_.min_degree) needs_fallback[se.lo] = false;
            if (degree[se.hi] >= cfg_.min_degree) needs_fallback[se.hi] = false;

            any_fallback = false;
            for (size_t i = 0; i < n_contigs; ++i) {
                if (needs_fallback[i]) { any_fallback = true; break; }
            }
            if (!any_fallback) break;
        }
    }

    // Sort for determinism
    std::sort(consensus_edges.begin(), consensus_edges.end(),
              [](const WeightedEdge& a, const WeightedEdge& b) {
                  if (a.u != b.u) return a.u < b.u;
                  return a.v < b.v;
              });

    return consensus_edges;
}

}  // namespace amber::bin2

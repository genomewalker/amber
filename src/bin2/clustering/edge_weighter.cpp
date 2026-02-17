#include "edge_weighter.h"
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace amber::bin2 {

DamageAwareEdgeWeighter::DamageAwareEdgeWeighter(const GraphConfig& config)
    : bandwidth_(config.bandwidth)
    , damage_beta_(config.damage_beta) {}

float DamageAwareEdgeWeighter::weight(float emb_dist,
                                       const DamageFeatures& di,
                                       const DamageFeatures& dj) const {
    float base = std::exp(-emb_dist / bandwidth_);

    float damage_penalty = 1.0f;
    if (di.confidence > confidence_threshold_ && dj.confidence > confidence_threshold_) {
        float dp_diff = std::abs(di.p_ancient - dj.p_ancient);
        damage_penalty = std::exp(-damage_beta_ * dp_diff);
    }

    return base * damage_penalty;
}

std::vector<WeightedEdge> DamageAwareEdgeWeighter::compute_edges(
    const NeighborList& neighbors,
    const std::vector<DamageFeatures>& damage_features) const {

    std::vector<WeightedEdge> edges;
    size_t n = neighbors.size();

    bool has_damage = !damage_features.empty();
    if (has_damage && damage_features.size() != n) {
        throw std::invalid_argument("Damage features size mismatch");
    }

    size_t total_edges = 0;
    for (size_t i = 0; i < n; ++i) {
        total_edges += neighbors.ids[i].size();
    }
    edges.reserve(total_edges);

    DamageFeatures empty_damage;

    for (size_t i = 0; i < n; ++i) {
        const auto& ids = neighbors.ids[i];
        const auto& dists = neighbors.dists[i];
        const DamageFeatures& di = has_damage ? damage_features[i] : empty_damage;

        for (size_t k = 0; k < ids.size(); ++k) {
            int j = ids[k];
            float dist = dists[k];

            const DamageFeatures& dj = has_damage ? damage_features[j] : empty_damage;
            float w = weight(dist, di, dj);

            edges.emplace_back(static_cast<int>(i), j, w);
        }
    }

    return edges;
}

std::vector<WeightedEdge> DamageAwareEdgeWeighter::compute_symmetric_edges(
    const NeighborList& neighbors,
    const std::vector<DamageFeatures>& damage_features) const {

    auto directed_edges = compute_edges(neighbors, damage_features);

    struct EdgeKey {
        int lo, hi;
        bool operator==(const EdgeKey& other) const {
            return lo == other.lo && hi == other.hi;
        }
    };

    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& k) const {
            return std::hash<int64_t>()(
                (static_cast<int64_t>(k.lo) << 32) | k.hi);
        }
    };

    std::unordered_map<EdgeKey, float, EdgeKeyHash> edge_weights;

    for (const auto& e : directed_edges) {
        EdgeKey key{std::min(e.u, e.v), std::max(e.u, e.v)};
        auto it = edge_weights.find(key);
        if (it == edge_weights.end()) {
            edge_weights[key] = e.w;
        } else {
            it->second = std::max(it->second, e.w);
        }
    }

    std::vector<WeightedEdge> symmetric_edges;
    symmetric_edges.reserve(edge_weights.size());

    for (const auto& [key, w] : edge_weights) {
        symmetric_edges.emplace_back(key.lo, key.hi, w);
    }

    // Sort edges for determinism (unordered_map iteration order is non-deterministic)
    std::sort(symmetric_edges.begin(), symmetric_edges.end(),
              [](const WeightedEdge& a, const WeightedEdge& b) {
                  if (a.u != b.u) return a.u < b.u;
                  return a.v < b.v;
              });

    return symmetric_edges;
}

AdaptiveBandwidthWeighter::AdaptiveBandwidthWeighter(float percentile)
    : percentile_(percentile) {}

float AdaptiveBandwidthWeighter::compute_bandwidth(const NeighborList& neighbors) const {
    std::vector<float> all_dists;

    for (size_t i = 0; i < neighbors.size(); ++i) {
        for (float d : neighbors.dists[i]) {
            all_dists.push_back(d);
        }
    }

    if (all_dists.empty()) {
        return 1.0f;
    }

    size_t idx = static_cast<size_t>(percentile_ * all_dists.size());
    idx = std::min(idx, all_dists.size() - 1);

    std::nth_element(all_dists.begin(),
                     all_dists.begin() + idx,
                     all_dists.end());

    float bandwidth = all_dists[idx];
    return std::max(bandwidth, 1e-6f);
}

std::vector<WeightedEdge> AdaptiveBandwidthWeighter::compute_edges(
    const NeighborList& neighbors,
    const std::vector<DamageFeatures>& damage_features,
    float damage_beta) const {

    float bandwidth = compute_bandwidth(neighbors);

    GraphConfig config;
    config.bandwidth = bandwidth;
    config.damage_beta = damage_beta;

    DamageAwareEdgeWeighter weighter(config);
    return weighter.compute_symmetric_edges(neighbors, damage_features);
}

std::vector<WeightedEdge> DamageAwareEdgeWeighter::compute_comebin_edges(
    const NeighborList& neighbors,
    int partgraph_ratio) const {

    size_t n = neighbors.size();

    // Step 1: Flatten all distances for percentile calculation
    // COMEBin: wei = ann_distances[:,1:].flatten()
    std::vector<float> all_dists;
    for (size_t i = 0; i < n; ++i) {
        for (float d : neighbors.dists[i]) {
            all_dists.push_back(d);
        }
    }

    // Step 2: Compute percentile cutoff
    // COMEBin: dist_cutoff = np.percentile(wei, partgraph_ratio)
    float dist_cutoff = std::numeric_limits<float>::max();
    if (partgraph_ratio < 100 && !all_dists.empty()) {
        size_t idx = static_cast<size_t>(
            static_cast<double>(partgraph_ratio) / 100.0 * all_dists.size());
        idx = std::min(idx, all_dists.size() - 1);
        std::nth_element(all_dists.begin(),
                         all_dists.begin() + idx,
                         all_dists.end());
        dist_cutoff = all_dists[idx];
    }

    // Step 3-5: Build edges with cutoff, weight, and source > target filter
    // COMEBin:
    //   save_index = wei <= dist_cutoff
    //   wei = np.exp(-wei / bandwidth)   [for l2 mode]
    //   index = sources > targets
    std::vector<WeightedEdge> edges;
    edges.reserve(all_dists.size() / 2);

    for (size_t i = 0; i < n; ++i) {
        const auto& ids = neighbors.ids[i];
        const auto& dists = neighbors.dists[i];

        for (size_t k = 0; k < ids.size(); ++k) {
            int j = ids[k];
            float dist = dists[k];

            // Cutoff filter
            if (dist > dist_cutoff) continue;

            // Dedup: only keep source > target (COMEBin convention)
            if (static_cast<int>(i) <= j) continue;

            // Weight: exp(-squared_l2 / bandwidth)
            float w = std::exp(-dist / bandwidth_);

            edges.emplace_back(static_cast<int>(i), j, w);
        }
    }

    return edges;
}

std::vector<WeightedEdge> DamageAwareEdgeWeighter::compute_comebin_edges_distances(
    const NeighborList& neighbors,
    int partgraph_ratio) const {
    // Same as compute_comebin_edges but returns raw distances instead of weights
    // For use in parameter sweeps where we need to recompute weights with different bandwidths

    size_t n = neighbors.size();

    std::vector<float> all_dists;
    for (size_t i = 0; i < n; ++i) {
        for (float d : neighbors.dists[i]) {
            all_dists.push_back(d);
        }
    }

    float dist_cutoff = std::numeric_limits<float>::max();
    if (partgraph_ratio < 100 && !all_dists.empty()) {
        size_t idx = static_cast<size_t>(
            static_cast<double>(partgraph_ratio) / 100.0 * all_dists.size());
        idx = std::min(idx, all_dists.size() - 1);
        std::nth_element(all_dists.begin(),
                         all_dists.begin() + idx,
                         all_dists.end());
        dist_cutoff = all_dists[idx];
    }

    std::vector<WeightedEdge> edges;
    edges.reserve(all_dists.size() / 2);

    for (size_t i = 0; i < n; ++i) {
        const auto& ids = neighbors.ids[i];
        const auto& dists = neighbors.dists[i];

        for (size_t k = 0; k < ids.size(); ++k) {
            int j = ids[k];
            float dist = dists[k];

            if (dist > dist_cutoff) continue;
            if (static_cast<int>(i) <= j) continue;

            // Store raw distance (not weight) for sweep reweighting
            edges.emplace_back(static_cast<int>(i), j, dist);
        }
    }

    return edges;
}

}  // namespace amber::bin2

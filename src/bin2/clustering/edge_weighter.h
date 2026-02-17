#pragma once

#include "clustering_types.h"
#include <vector>
#include <cmath>

namespace amber::bin2 {

class DamageAwareEdgeWeighter {
public:
    explicit DamageAwareEdgeWeighter(const GraphConfig& config = GraphConfig());

    float weight(float emb_dist,
                 const DamageFeatures& di,
                 const DamageFeatures& dj) const;

    std::vector<WeightedEdge> compute_edges(
        const NeighborList& neighbors,
        const std::vector<DamageFeatures>& damage_features) const;

    std::vector<WeightedEdge> compute_symmetric_edges(
        const NeighborList& neighbors,
        const std::vector<DamageFeatures>& damage_features) const;

    // COMEBin-exact edge construction:
    // 1. Percentile distance cutoff
    // 2. exp(-dist²/bandwidth) weighting
    // 3. Deduplicate with source > target (not symmetric merge)
    std::vector<WeightedEdge> compute_comebin_edges(
        const NeighborList& neighbors,
        int partgraph_ratio) const;

    // Same as compute_comebin_edges but returns raw distances instead of weights
    // For use in parameter sweeps where we recompute weights with different bandwidths
    std::vector<WeightedEdge> compute_comebin_edges_distances(
        const NeighborList& neighbors,
        int partgraph_ratio) const;

    void set_bandwidth(float bandwidth) { bandwidth_ = bandwidth; }
    void set_damage_beta(float beta) { damage_beta_ = beta; }

    float bandwidth() const { return bandwidth_; }
    float damage_beta() const { return damage_beta_; }

private:
    float bandwidth_;
    float damage_beta_;
    float confidence_threshold_ = 0.5f;
};

class AdaptiveBandwidthWeighter {
public:
    explicit AdaptiveBandwidthWeighter(float percentile = 0.1f);

    float compute_bandwidth(const NeighborList& neighbors) const;

    std::vector<WeightedEdge> compute_edges(
        const NeighborList& neighbors,
        const std::vector<DamageFeatures>& damage_features,
        float damage_beta = 1.0f) const;

private:
    float percentile_;
};

}  // namespace amber::bin2

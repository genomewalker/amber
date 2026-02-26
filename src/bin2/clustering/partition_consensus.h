#pragma once
#include "leiden_backend.h"
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <numeric>

namespace amber::bin2 {

// Given R label vectors (each length n_contigs, -1=unbinned),
// build co-binning affinity graph and run Leiden to find consensus clusters.
// Edge weight = fraction of runs where pair (i,j) are co-binned.
// Edges below threshold are excluded. Leiden finds natural communities
// without transitive over-merging.
inline std::vector<int> partition_consensus(
    const std::vector<std::vector<int>>& partitions,
    int n_contigs,
    float threshold,
    float resolution = 1.0f)
{
    int R = (int)partitions.size();
    if (R == 0) return {};
    if (R == 1) return partitions[0];

    // Count co-binning occurrences per pair
    std::unordered_map<uint64_t, int> cobin;
    for (const auto& labels : partitions) {
        std::unordered_map<int, std::vector<int>> clusters;
        for (int i = 0; i < n_contigs; ++i) {
            if (labels[i] >= 0) clusters[labels[i]].push_back(i);
        }
        for (auto& [c, members] : clusters) {
            int sz = (int)members.size();
            for (int a = 0; a < sz; ++a)
                for (int b = a + 1; b < sz; ++b) {
                    int u = members[a], v = members[b];
                    if (u > v) std::swap(u, v);
                    uint64_t key = ((uint64_t)u << 32) | (uint64_t)v;
                    cobin[key]++;
                }
        }
    }

    // Build weighted edge list for Leiden (only pairs meeting threshold)
    std::vector<WeightedEdge> edges;
    edges.reserve(cobin.size());
    for (auto& [key, cnt] : cobin) {
        float w = (float)cnt / R;
        if (w >= threshold) {
            int u = (int)(key >> 32);
            int v = (int)(key & 0xFFFFFFFF);
            edges.push_back({u, v, w});
        }
    }

    if (edges.empty()) {
        // No pairs meet threshold — return all singletons as unbinned
        return std::vector<int>(n_contigs, -1);
    }

    // Track which contigs were binned in at least one run (active contigs)
    std::vector<bool> active(n_contigs, false);
    for (const auto& labels : partitions)
        for (int i = 0; i < n_contigs; ++i)
            if (labels[i] >= 0) active[i] = true;

    // Run Leiden on the co-binning affinity graph
    LeidenConfig cfg;
    cfg.resolution = resolution;
    cfg.max_iterations = 10;
    cfg.random_seed = 42;

    auto backend = create_leiden_backend(true);  // use libleidenalg
    auto result = backend->cluster(edges, n_contigs, cfg);

    // Mark inactive contigs as unbinned (-1)
    std::vector<int> labels = std::move(result.labels);
    for (int i = 0; i < n_contigs; ++i) {
        if (!active[i]) labels[i] = -1;
    }

    return labels;
}

} // namespace amber::bin2

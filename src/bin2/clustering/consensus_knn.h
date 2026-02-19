#pragma once

#include "clustering_types.h"
#include <vector>

namespace amber::bin2 {

struct ConsensusKnnConfig {
    int k = 100;
    float bandwidth = 0.2f;
    int partgraph_ratio = 50;
    float min_freq = 2.0f / 3.0f;
    int min_degree = 20;
};

class ConsensusKnnBuilder {
public:
    explicit ConsensusKnnBuilder(const ConsensusKnnConfig& cfg);

    // Add one set of embeddings (one encoder run). Call N times before build().
    void add_embeddings(const std::vector<std::vector<float>>& embeddings);

    // Build consensus edge list. Returns WeightedEdges for Leiden.
    std::vector<WeightedEdge> build() const;

private:
    ConsensusKnnConfig cfg_;
    std::vector<std::vector<std::vector<float>>> all_embeddings_;  // [run][contig][dim]
};

}  // namespace amber::bin2

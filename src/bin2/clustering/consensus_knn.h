#pragma once

#include "clustering_types.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace amber::bin2 {

enum class ConsensusWeightMode { VOTE, QUALITY };

struct ConsensusKnnConfig {
    int k = 100;
    float bandwidth = 0.2f;
    int partgraph_ratio = 50;
    float min_freq = 2.0f / 3.0f;
    int min_degree = 20;
    ConsensusWeightMode weight_mode = ConsensusWeightMode::QUALITY;
    float quality_threshold = 0.5f;
};

// Score an encoder by how well it separates SCG-sharing contig pairs in kNN space.
// Returns fraction of same-marker contig pairs that are NOT kNN neighbors (0=bad, 1=perfect).
float compute_scg_separation_score(
    const std::vector<std::vector<int>>& knn_ids,
    const std::vector<std::vector<int>>& contig_marker_ids);

class ConsensusKnnBuilder {
public:
    explicit ConsensusKnnBuilder(const ConsensusKnnConfig& cfg);

    // Add one set of embeddings (one encoder run). Call N times before build().
    void add_embeddings(const std::vector<std::vector<float>>& embeddings);

    // Set per-encoder SCG separation scores (call before build()).
    void set_encoder_scores(const std::vector<float>& scores);

    // Build consensus edge list. Returns WeightedEdges for Leiden.
    std::vector<WeightedEdge> build() const;

private:
    ConsensusKnnConfig cfg_;
    std::vector<std::vector<std::vector<float>>> all_embeddings_;  // [run][contig][dim]
    std::vector<float> encoder_scores_;
};

}  // namespace amber::bin2

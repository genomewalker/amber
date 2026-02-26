#pragma once

#include "clustering_types.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace amber::bin2 {

// Quality-Guided Hierarchical Clustering
// Deterministic, hyperparameter-free alternative to Leiden
// Directly optimizes for HQ bins instead of modularity

struct QualityHierarchicalConfig {
    int min_bin_size = 200000;           // Minimum bin size in bp
    float min_completeness = 0.5f;       // Minimum completeness to consider
    float max_contamination = 0.1f;      // Maximum contamination allowed
    float hq_completeness = 0.9f;        // HQ threshold
    float hq_contamination = 0.05f;      // HQ contamination threshold
    int num_markers = 107;               // Number of single-copy markers (bacteria)
    bool verbose = false;
};

// Marker profile for a cluster (tracks marker gene presence/duplication)
struct ClusterMarkerProfile {
    std::unordered_map<int, int> marker_counts;  // marker_id -> copy count
    int total_markers = 0;
    int unique_markers = 0;
    int duplicated_markers = 0;

    float completeness() const {
        return static_cast<float>(unique_markers) / 107.0f;  // bacteria marker set
    }

    float contamination() const {
        if (total_markers == 0) return 0.0f;
        return static_cast<float>(duplicated_markers) / static_cast<float>(total_markers);
    }

    bool is_hq(float comp_thresh = 0.9f, float cont_thresh = 0.05f) const {
        return completeness() >= comp_thresh && contamination() <= cont_thresh;
    }

    bool is_mq(float comp_thresh = 0.5f, float cont_thresh = 0.1f) const {
        return completeness() >= comp_thresh && contamination() <= cont_thresh;
    }

    // Merge another profile into this one
    void merge(const ClusterMarkerProfile& other) {
        for (const auto& [marker, count] : other.marker_counts) {
            marker_counts[marker] += count;
        }
        recompute_stats();
    }

    void recompute_stats() {
        total_markers = 0;
        unique_markers = 0;
        duplicated_markers = 0;
        for (const auto& [marker, count] : marker_counts) {
            if (count > 0) {
                unique_markers++;
                total_markers += count;
                if (count > 1) {
                    duplicated_markers += (count - 1);
                }
            }
        }
    }
};

// Hierarchical cluster node
struct HierarchicalNode {
    int id;
    int left_child = -1;   // -1 means leaf
    int right_child = -1;
    float merge_distance;
    std::vector<int> members;  // contig indices in this subtree
    ClusterMarkerProfile markers;
    int64_t total_length = 0;  // total bp in this cluster
};

// Result of hierarchical clustering
struct QualityHierarchicalResult {
    std::vector<int> labels;           // contig -> cluster assignment
    int num_clusters;
    int num_hq;
    int num_mq;
    std::vector<ClusterMarkerProfile> cluster_profiles;
};

class QualityHierarchicalClusterer {
public:
    explicit QualityHierarchicalClusterer(const QualityHierarchicalConfig& config);

    // Main clustering method (SLOW - computes full distance matrix)
    // embeddings: [N x D] contig embeddings
    // contig_lengths: [N] length in bp for each contig
    // contig_markers: [N] list of (marker_id, count) for each contig
    QualityHierarchicalResult cluster(
        const std::vector<std::vector<float>>& embeddings,
        const std::vector<int64_t>& contig_lengths,
        const std::vector<std::vector<std::pair<int, int>>>& contig_markers
    );

    // FAST clustering using pre-computed kNN graph (280x faster than full distance matrix)
    // knn_indices: [N x k] indices of k nearest neighbors per contig
    // knn_distances: [N x k] distances to k nearest neighbors
    QualityHierarchicalResult cluster_with_knn(
        const std::vector<std::vector<int>>& knn_indices,
        const std::vector<std::vector<float>>& knn_distances,
        const std::vector<int64_t>& contig_lengths,
        const std::vector<std::vector<std::pair<int, int>>>& contig_markers
    );

private:
    // Build distance matrix from embeddings (O(N²) - avoid if possible)
    std::vector<std::vector<float>> compute_distance_matrix(
        const std::vector<std::vector<float>>& embeddings
    );

    // Find best merge using sparse kNN distances (O(N*k))
    std::pair<int, int> find_best_merge_knn(
        const std::vector<int>& active_clusters,
        const std::unordered_map<int, std::unordered_map<int, float>>& sparse_dist,
        const std::vector<HierarchicalNode>& nodes,
        const std::unordered_map<int, int>& contig_to_cluster
    );

    // Find best merge that maximizes quality improvement (O(N²) - slow)
    std::pair<int, int> find_best_merge(
        const std::vector<int>& active_clusters,
        const std::vector<std::vector<float>>& distances,
        const std::vector<HierarchicalNode>& nodes
    );

    // Compute quality score for current clustering
    int count_hq_bins(const std::vector<HierarchicalNode>& nodes,
                      const std::vector<int>& active_clusters);

    // Check if merging two clusters would create marker conflicts
    bool merge_creates_conflict(const ClusterMarkerProfile& a,
                                const ClusterMarkerProfile& b);

    // Extract final cluster assignments
    QualityHierarchicalResult extract_result(
        const std::vector<HierarchicalNode>& nodes,
        const std::vector<int>& active_clusters,
        int n_contigs
    );

    QualityHierarchicalConfig config_;
};

// Factory function
std::unique_ptr<QualityHierarchicalClusterer> create_quality_hierarchical(
    const QualityHierarchicalConfig& config = QualityHierarchicalConfig()
);

}  // namespace amber::bin2

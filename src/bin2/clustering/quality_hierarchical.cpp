#include "quality_hierarchical.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <set>

namespace amber::bin2 {

// Priority queue entry for merge candidates
struct MergeCandidate {
    float score;
    int cluster_i;
    int cluster_j;

    bool operator>(const MergeCandidate& other) const {
        return score > other.score;  // Min-heap
    }
};

QualityHierarchicalClusterer::QualityHierarchicalClusterer(
    const QualityHierarchicalConfig& config)
    : config_(config) {}

std::vector<std::vector<float>> QualityHierarchicalClusterer::compute_distance_matrix(
    const std::vector<std::vector<float>>& embeddings
) {
    int N = static_cast<int>(embeddings.size());
    std::vector<std::vector<float>> dist(N, std::vector<float>(N, 0.0f));

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            float d = 0.0f;
            for (size_t k = 0; k < embeddings[i].size(); ++k) {
                float diff = embeddings[i][k] - embeddings[j][k];
                d += diff * diff;
            }
            d = std::sqrt(d);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    return dist;
}

bool QualityHierarchicalClusterer::merge_creates_conflict(
    const ClusterMarkerProfile& a,
    const ClusterMarkerProfile& b
) {
    // Check if merging would create too much contamination
    ClusterMarkerProfile merged;
    merged.marker_counts = a.marker_counts;
    for (const auto& [marker, count] : b.marker_counts) {
        merged.marker_counts[marker] += count;
    }
    merged.recompute_stats();

    // If merged cluster has >10% contamination, consider it a conflict
    return merged.contamination() > config_.max_contamination;
}

std::pair<int, int> QualityHierarchicalClusterer::find_best_merge(
    const std::vector<int>& active_clusters,
    const std::vector<std::vector<float>>& distances,
    const std::vector<HierarchicalNode>& nodes
) {
    float best_score = std::numeric_limits<float>::max();
    int best_i = -1, best_j = -1;

    // For each pair of active clusters, compute merge score
    for (size_t i = 0; i < active_clusters.size(); ++i) {
        for (size_t j = i + 1; j < active_clusters.size(); ++j) {
            int ci = active_clusters[i];
            int cj = active_clusters[j];

            // Skip if merge would create marker conflict
            if (merge_creates_conflict(nodes[ci].markers, nodes[cj].markers)) {
                continue;
            }

            // Compute average linkage distance between clusters
            float total_dist = 0.0f;
            int count = 0;
            for (int mi : nodes[ci].members) {
                for (int mj : nodes[cj].members) {
                    total_dist += distances[mi][mj];
                    count++;
                }
            }
            float avg_dist = (count > 0) ? total_dist / count : std::numeric_limits<float>::max();

            // Quality-aware scoring: prefer merges that increase completeness
            // without increasing contamination
            ClusterMarkerProfile merged;
            merged.marker_counts = nodes[ci].markers.marker_counts;
            for (const auto& [marker, cnt] : nodes[cj].markers.marker_counts) {
                merged.marker_counts[marker] += cnt;
            }
            merged.recompute_stats();

            // Score: distance - bonus for completeness gain
            float comp_before = std::max(nodes[ci].markers.completeness(),
                                         nodes[cj].markers.completeness());
            float comp_after = merged.completeness();
            float comp_gain = comp_after - comp_before;

            // Penalize contamination increase
            float cont_before = std::min(nodes[ci].markers.contamination(),
                                         nodes[cj].markers.contamination());
            float cont_after = merged.contamination();
            float cont_penalty = (cont_after - cont_before) * 10.0f;  // Heavy penalty

            float score = avg_dist - comp_gain * 2.0f + cont_penalty;

            if (score < best_score) {
                best_score = score;
                best_i = ci;
                best_j = cj;
            }
        }
    }

    return {best_i, best_j};
}

int QualityHierarchicalClusterer::count_hq_bins(
    const std::vector<HierarchicalNode>& nodes,
    const std::vector<int>& active_clusters
) {
    int hq_count = 0;
    for (int ci : active_clusters) {
        if (nodes[ci].total_length >= config_.min_bin_size &&
            nodes[ci].markers.is_hq(config_.hq_completeness, config_.hq_contamination)) {
            hq_count++;
        }
    }
    return hq_count;
}

QualityHierarchicalResult QualityHierarchicalClusterer::extract_result(
    const std::vector<HierarchicalNode>& nodes,
    const std::vector<int>& active_clusters,
    int n_contigs
) {
    QualityHierarchicalResult result;
    result.labels.resize(n_contigs, -1);
    result.num_hq = 0;
    result.num_mq = 0;

    int cluster_id = 0;
    for (int ci : active_clusters) {
        const auto& node = nodes[ci];

        // Skip clusters that are too small
        if (node.total_length < config_.min_bin_size) {
            continue;
        }

        // Assign all members to this cluster
        for (int contig_idx : node.members) {
            result.labels[contig_idx] = cluster_id;
        }

        result.cluster_profiles.push_back(node.markers);

        if (node.markers.is_hq(config_.hq_completeness, config_.hq_contamination)) {
            result.num_hq++;
        } else if (node.markers.is_mq(config_.min_completeness, config_.max_contamination)) {
            result.num_mq++;
        }

        cluster_id++;
    }

    result.num_clusters = cluster_id;
    return result;
}

QualityHierarchicalResult QualityHierarchicalClusterer::cluster(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<int64_t>& contig_lengths,
    const std::vector<std::vector<std::pair<int, int>>>& contig_markers
) {
    int N = static_cast<int>(embeddings.size());

    if (config_.verbose) {
        std::cerr << "[QualityHierarchical] Clustering " << N << " contigs\n";
    }

    // Compute distance matrix
    if (config_.verbose) {
        std::cerr << "[QualityHierarchical] Computing distance matrix...\n";
    }
    auto distances = compute_distance_matrix(embeddings);

    // Initialize leaf nodes (one per contig)
    std::vector<HierarchicalNode> nodes(N);
    for (int i = 0; i < N; ++i) {
        nodes[i].id = i;
        nodes[i].members = {i};
        nodes[i].total_length = contig_lengths[i];
        nodes[i].merge_distance = 0.0f;

        // Initialize marker profile
        for (const auto& [marker_id, count] : contig_markers[i]) {
            nodes[i].markers.marker_counts[marker_id] = count;
        }
        nodes[i].markers.recompute_stats();
    }

    // Track active clusters (initially all leaves)
    std::vector<int> active_clusters(N);
    std::iota(active_clusters.begin(), active_clusters.end(), 0);

    // Track best HQ count seen
    int best_hq = count_hq_bins(nodes, active_clusters);
    std::vector<int> best_active = active_clusters;

    if (config_.verbose) {
        std::cerr << "[QualityHierarchical] Initial HQ bins: " << best_hq << "\n";
    }

    // Agglomerative clustering with quality-guided merging
    int next_node_id = N;
    int iterations = 0;
    int max_iterations = N - 1;  // Maximum possible merges

    while (active_clusters.size() > 1 && iterations < max_iterations) {
        // Find best pair to merge
        auto [ci, cj] = find_best_merge(active_clusters, distances, nodes);

        // No valid merge found (all remaining pairs conflict)
        if (ci < 0 || cj < 0) {
            if (config_.verbose) {
                std::cerr << "[QualityHierarchical] No valid merges remaining\n";
            }
            break;
        }

        // Create merged node
        HierarchicalNode merged;
        merged.id = next_node_id++;
        merged.left_child = ci;
        merged.right_child = cj;
        merged.members = nodes[ci].members;
        merged.members.insert(merged.members.end(),
                              nodes[cj].members.begin(), nodes[cj].members.end());
        merged.total_length = nodes[ci].total_length + nodes[cj].total_length;
        merged.markers = nodes[ci].markers;
        merged.markers.merge(nodes[cj].markers);

        // Compute merge distance (average linkage)
        float total_dist = 0.0f;
        int count = 0;
        for (int mi : nodes[ci].members) {
            for (int mj : nodes[cj].members) {
                total_dist += distances[mi][mj];
                count++;
            }
        }
        merged.merge_distance = (count > 0) ? total_dist / count : 0.0f;

        nodes.push_back(merged);

        // Update active clusters
        active_clusters.erase(
            std::remove_if(active_clusters.begin(), active_clusters.end(),
                           [ci, cj](int x) { return x == ci || x == cj; }),
            active_clusters.end()
        );
        active_clusters.push_back(merged.id);

        // Check if this merge improved HQ count
        int current_hq = count_hq_bins(nodes, active_clusters);
        if (current_hq > best_hq) {
            best_hq = current_hq;
            best_active = active_clusters;
            if (config_.verbose) {
                std::cerr << "[QualityHierarchical] New best: " << best_hq
                          << " HQ bins at " << active_clusters.size() << " clusters\n";
            }
        }

        iterations++;

        // Early termination if we're degrading quality significantly
        if (current_hq < best_hq - 2 && iterations > N / 4) {
            if (config_.verbose) {
                std::cerr << "[QualityHierarchical] Quality degrading, stopping\n";
            }
            break;
        }
    }

    if (config_.verbose) {
        std::cerr << "[QualityHierarchical] Final: " << best_hq << " HQ bins, "
                  << best_active.size() << " clusters\n";
    }

    // Extract result from best configuration
    return extract_result(nodes, best_active, N);
}

// Fast kNN-based find_best_merge
std::pair<int, int> QualityHierarchicalClusterer::find_best_merge_knn(
    const std::vector<int>& active_clusters,
    const std::unordered_map<int, std::unordered_map<int, float>>& sparse_dist,
    const std::vector<HierarchicalNode>& nodes,
    const std::unordered_map<int, int>& contig_to_cluster
) {
    float best_score = std::numeric_limits<float>::max();
    int best_i = -1, best_j = -1;

    // Build set of active clusters for O(1) lookup
    std::set<int> active_set(active_clusters.begin(), active_clusters.end());

    // For each active cluster, check merges with neighboring clusters
    for (int ci : active_clusters) {
        // Find neighboring clusters through kNN edges
        std::unordered_map<int, float> neighbor_dists;

        for (int member : nodes[ci].members) {
            auto it = sparse_dist.find(member);
            if (it == sparse_dist.end()) continue;

            for (const auto& [neighbor, dist] : it->second) {
                auto cit = contig_to_cluster.find(neighbor);
                if (cit == contig_to_cluster.end()) continue;

                int cj = cit->second;
                if (cj == ci || active_set.find(cj) == active_set.end()) continue;

                // Use minimum distance as representative
                if (neighbor_dists.find(cj) == neighbor_dists.end() ||
                    dist < neighbor_dists[cj]) {
                    neighbor_dists[cj] = dist;
                }
            }
        }

        // Evaluate each neighboring cluster
        for (const auto& [cj, min_dist] : neighbor_dists) {
            // Skip if merge would create marker conflict
            if (merge_creates_conflict(nodes[ci].markers, nodes[cj].markers)) {
                continue;
            }

            // Quality-aware scoring
            ClusterMarkerProfile merged;
            merged.marker_counts = nodes[ci].markers.marker_counts;
            for (const auto& [marker, cnt] : nodes[cj].markers.marker_counts) {
                merged.marker_counts[marker] += cnt;
            }
            merged.recompute_stats();

            // Score: distance - bonus for completeness gain
            float comp_before = std::max(nodes[ci].markers.completeness(),
                                         nodes[cj].markers.completeness());
            float comp_after = merged.completeness();
            float comp_gain = comp_after - comp_before;

            // Penalize contamination increase
            float cont_before = std::min(nodes[ci].markers.contamination(),
                                         nodes[cj].markers.contamination());
            float cont_after = merged.contamination();
            float cont_penalty = (cont_after - cont_before) * 10.0f;

            float score = min_dist - comp_gain * 2.0f + cont_penalty;

            if (score < best_score) {
                best_score = score;
                best_i = ci;
                best_j = cj;
            }
        }
    }

    return {best_i, best_j};
}

// FAST clustering using kNN graph (280x faster than full distance matrix)
QualityHierarchicalResult QualityHierarchicalClusterer::cluster_with_knn(
    const std::vector<std::vector<int>>& knn_indices,
    const std::vector<std::vector<float>>& knn_distances,
    const std::vector<int64_t>& contig_lengths,
    const std::vector<std::vector<std::pair<int, int>>>& contig_markers
) {
    int N = static_cast<int>(knn_indices.size());

    if (config_.verbose) {
        std::cerr << "[QualityHierarchical] Fast kNN clustering " << N << " contigs\n";
    }

    // Build sparse distance map from kNN graph
    std::unordered_map<int, std::unordered_map<int, float>> sparse_dist;
    for (int i = 0; i < N; ++i) {
        for (size_t k = 0; k < knn_indices[i].size(); ++k) {
            int j = knn_indices[i][k];
            float d = knn_distances[i][k];
            sparse_dist[i][j] = d;
            sparse_dist[j][i] = d;  // Symmetric
        }
    }

    if (config_.verbose) {
        size_t total_edges = 0;
        for (const auto& [i, neighbors] : sparse_dist) {
            total_edges += neighbors.size();
        }
        std::cerr << "[QualityHierarchical] Using " << total_edges / 2
                  << " kNN edges (vs " << (int64_t)N * N / 2 << " full matrix)\n";
    }

    // Initialize leaf nodes
    std::vector<HierarchicalNode> nodes(N);
    std::unordered_map<int, int> contig_to_cluster;

    for (int i = 0; i < N; ++i) {
        nodes[i].id = i;
        nodes[i].members = {i};
        nodes[i].total_length = contig_lengths[i];
        nodes[i].merge_distance = 0.0f;
        contig_to_cluster[i] = i;

        for (const auto& [marker_id, count] : contig_markers[i]) {
            nodes[i].markers.marker_counts[marker_id] = count;
        }
        nodes[i].markers.recompute_stats();
    }

    // Track active clusters
    std::vector<int> active_clusters(N);
    std::iota(active_clusters.begin(), active_clusters.end(), 0);

    // Track best HQ count
    int best_hq = count_hq_bins(nodes, active_clusters);
    std::vector<int> best_active = active_clusters;

    if (config_.verbose) {
        std::cerr << "[QualityHierarchical] Initial HQ bins: " << best_hq << "\n";
    }

    // Agglomerative clustering
    int next_node_id = N;
    int iterations = 0;
    int max_iterations = N - 1;
    int stale_count = 0;

    while (active_clusters.size() > 1 && iterations < max_iterations) {
        // Find best merge using kNN graph
        auto [ci, cj] = find_best_merge_knn(active_clusters, sparse_dist, nodes, contig_to_cluster);

        if (ci < 0 || cj < 0) {
            if (config_.verbose) {
                std::cerr << "[QualityHierarchical] No valid kNN merges remaining\n";
            }
            break;
        }

        // Create merged node
        HierarchicalNode merged;
        merged.id = next_node_id++;
        merged.left_child = ci;
        merged.right_child = cj;
        merged.members = nodes[ci].members;
        merged.members.insert(merged.members.end(),
                              nodes[cj].members.begin(), nodes[cj].members.end());
        merged.total_length = nodes[ci].total_length + nodes[cj].total_length;
        merged.markers = nodes[ci].markers;
        merged.markers.merge(nodes[cj].markers);

        // Compute merge distance (min link from kNN)
        float min_dist = std::numeric_limits<float>::max();
        for (int mi : nodes[ci].members) {
            auto it = sparse_dist.find(mi);
            if (it == sparse_dist.end()) continue;
            for (int mj : nodes[cj].members) {
                auto dit = it->second.find(mj);
                if (dit != it->second.end() && dit->second < min_dist) {
                    min_dist = dit->second;
                }
            }
        }
        merged.merge_distance = (min_dist < std::numeric_limits<float>::max()) ? min_dist : 0.0f;

        nodes.push_back(merged);

        // Update contig_to_cluster mapping
        for (int member : merged.members) {
            contig_to_cluster[member] = merged.id;
        }

        // Update active clusters
        active_clusters.erase(
            std::remove_if(active_clusters.begin(), active_clusters.end(),
                           [ci, cj](int x) { return x == ci || x == cj; }),
            active_clusters.end()
        );
        active_clusters.push_back(merged.id);

        // Check if this merge improved HQ count
        int current_hq = count_hq_bins(nodes, active_clusters);
        if (current_hq > best_hq) {
            best_hq = current_hq;
            best_active = active_clusters;
            stale_count = 0;
            if (config_.verbose) {
                std::cerr << "[QualityHierarchical] New best: " << best_hq
                          << " HQ bins at " << active_clusters.size() << " clusters\n";
            }
        } else {
            stale_count++;
        }

        iterations++;

        // Progress logging every 1000 iterations
        if (config_.verbose && iterations % 1000 == 0) {
            std::cerr << "[QualityHierarchical] Iteration " << iterations
                      << "/" << max_iterations << " - " << active_clusters.size()
                      << " clusters, " << current_hq << " HQ\n";
        }

        // Early termination if quality is degrading
        if (current_hq < best_hq - 2 && stale_count > 500) {
            if (config_.verbose) {
                std::cerr << "[QualityHierarchical] Quality degrading, stopping\n";
            }
            break;
        }
    }

    if (config_.verbose) {
        std::cerr << "[QualityHierarchical] Final: " << best_hq << " HQ bins, "
                  << best_active.size() << " clusters after " << iterations << " merges\n";
    }

    return extract_result(nodes, best_active, N);
}

std::unique_ptr<QualityHierarchicalClusterer> create_quality_hierarchical(
    const QualityHierarchicalConfig& config
) {
    return std::make_unique<QualityHierarchicalClusterer>(config);
}

}  // namespace amber::bin2

// AMBER - marker_index.h
// Fast marker lookup for quality-weighted Leiden clustering
// Design: Opus 4.6 (architecture) + GPT-5.3-codex (math)

#pragma once

#include "../quality/checkm_markers.h"

#include <vector>
#include <string>
#include <unordered_map>

namespace amber::bin2 {

// Precomputed marker entry for O(1) quality delta computation
struct MarkerEntry {
    int id;           // Marker ID (for fast lookup in comm_markers)
    float weight;     // w_m = Σ_{s∋m} 100/(|sets|·|s|)
    int copy_count;   // Number of copies of this marker on this contig (k >= 1)
};

// Index mapping contigs to their markers with precomputed weights
// Enables O(markers-per-contig) quality delta computation
class MarkerIndex {
public:
    MarkerIndex() = default;

    // Build index from CheckM marker sets and HMM hits
    // contig_names must match the node ordering used in Leiden
    void build(const CheckMMarkerSets& marker_sets,
               const std::unordered_map<std::string, std::vector<std::string>>& hits,
               const std::vector<std::string>& contig_names);

    // Build index directly from HMM hits (creates uniform weights for each marker)
    // Use this when marker sets aren't available (e.g., bacar_marker HMM)
    void build_from_hits(const std::unordered_map<std::string, std::vector<std::string>>& hits,
                         const std::vector<std::string>& contig_names);

    // Check if contig has any markers
    bool has_markers(int node) const {
        return node < static_cast<int>(contig_markers_.size()) &&
               !contig_markers_[node].empty();
    }

    // Get markers for a contig (empty if none)
    const std::vector<MarkerEntry>& get_markers(int node) const {
        static const std::vector<MarkerEntry> empty;
        if (node < 0 || node >= static_cast<int>(contig_markers_.size())) {
            return empty;
        }
        return contig_markers_[node];
    }

    // Number of unique markers across all sets
    int num_markers() const { return num_markers_; }

    // Maximum marker ID (for bounds checking against fixed-size arrays)
    int max_marker_id() const { return num_markers_; }

    // Number of marker sets (for normalization)
    int num_sets() const { return num_sets_; }

    // Number of contigs with at least one marker
    int num_marker_contigs() const { return num_marker_contigs_; }

    // Marker name from ID (for debugging)
    const std::string& marker_name(int id) const {
        static const std::string unknown = "?";
        for (const auto& [name, mid] : marker_to_id_) {
            if (mid == id) return name;
        }
        return unknown;
    }

private:
    // Per-contig marker list (indexed by node)
    std::vector<std::vector<MarkerEntry>> contig_markers_;

    // Marker name to integer ID
    std::unordered_map<std::string, int> marker_to_id_;

    // Precomputed marker weights
    std::vector<float> marker_weights_;

    int num_markers_ = 0;
    int num_sets_ = 0;
    int num_marker_contigs_ = 0;
};

// Quality delta calculator for Leiden moves
// Implements the finite-difference formula validated by GPT-5.3-codex:
//   Δcomp = w_m * ([n_new + k > 0] - [n_new > 0] + [n_old - k > 0] - [n_old > 0])
//   Δcont = w_m * (max(n_new + k - 1, 0) - max(n_new - 1, 0)
//                + max(n_old - k - 1, 0) - max(n_old - 1, 0))
//   Δquality = Δcomp - penalty * Δcont
struct QualityDelta {
    float delta_completeness = 0.0f;
    float delta_contamination = 0.0f;

    // Combined delta with penalty
    float total(float penalty) const {
        return delta_completeness - penalty * delta_contamination;
    }
};

// Compute quality delta for moving node from source to target community
// comm_markers: per-community marker counts (comm_markers[comm][marker_id] = count)
// This handles k>1 copies correctly using the exact finite-difference formula
QualityDelta compute_quality_delta(
    int node,
    int source_comm,
    int target_comm,
    const MarkerIndex& marker_index,
    const std::vector<std::unordered_map<int, int>>& comm_markers);

}  // namespace amber::bin2

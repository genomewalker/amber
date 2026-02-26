// AMBER - marker_index.cpp
// Fast marker lookup for quality-weighted Leiden clustering

#include "marker_index.h"

#include <algorithm>
#include <iostream>
#include <set>

namespace amber::bin2 {

void MarkerIndex::build(const CheckMMarkerSets& marker_sets,
                        const std::unordered_map<std::string, std::vector<std::string>>& hits,
                        const std::vector<std::string>& contig_names) {
    // Clear previous state
    contig_markers_.clear();
    marker_to_id_.clear();
    marker_weights_.clear();
    num_markers_ = 0;
    num_sets_ = marker_sets.num_sets();
    num_marker_contigs_ = 0;

    if (num_sets_ == 0) {
        std::cerr << "[MarkerIndex] WARNING: No marker sets provided\n";
        contig_markers_.resize(contig_names.size());
        return;
    }

    // Step 1: Build marker_to_id mapping and compute weights
    // Weight formula: w_m = Σ_{s∋m} 100/(|sets|·|s|)
    std::unordered_map<std::string, float> marker_weight_map;

    for (const auto& ms : marker_sets.sets) {
        float set_contribution = 100.0f / (num_sets_ * ms.size());
        for (const auto& marker : ms) {
            marker_weight_map[marker] += set_contribution;

            if (marker_to_id_.find(marker) == marker_to_id_.end()) {
                marker_to_id_[marker] = num_markers_++;
            }
        }
    }

    // Convert to vector for O(1) lookup
    marker_weights_.resize(num_markers_, 0.0f);
    for (const auto& [marker, weight] : marker_weight_map) {
        marker_weights_[marker_to_id_[marker]] = weight;
    }

    // Step 2: Build contig -> markers mapping
    // Create name to index mapping
    std::unordered_map<std::string, int> name_to_idx;
    for (size_t i = 0; i < contig_names.size(); i++) {
        name_to_idx[contig_names[i]] = static_cast<int>(i);
    }

    contig_markers_.resize(contig_names.size());

    // Debug: show first few names from each source
    int debug_count = 0;
    std::cerr << "[MarkerIndex] DEBUG: First 5 hit contig names:\n";
    for (const auto& [contig_name, markers] : hits) {
        if (debug_count++ < 5) {
            std::cerr << "  hit: '" << contig_name << "'\n";
        }
    }
    debug_count = 0;
    std::cerr << "[MarkerIndex] DEBUG: First 5 contig_names:\n";
    for (const auto& name : contig_names) {
        if (debug_count++ < 5) {
            std::cerr << "  contig: '" << name << "'\n";
        }
    }

    // Debug: check a specific hit name
    int found_count = 0;
    int not_found_count = 0;
    std::string first_not_found;

    for (const auto& [contig_name, markers] : hits) {
        auto it = name_to_idx.find(contig_name);
        if (it == name_to_idx.end()) {
            not_found_count++;
            if (first_not_found.empty()) {
                first_not_found = contig_name;
            }
            continue;
        }
        found_count++;

        int node = it->second;

        // Count occurrences of each marker (handles k>1 copies)
        std::unordered_map<std::string, int> marker_counts;
        for (const auto& m : markers) {
            if (marker_to_id_.find(m) != marker_to_id_.end()) {
                marker_counts[m]++;
            }
        }

        // Build entry list
        for (const auto& [marker, count] : marker_counts) {
            MarkerEntry entry;
            entry.id = marker_to_id_[marker];
            entry.weight = marker_weights_[entry.id];
            entry.copy_count = count;
            contig_markers_[node].push_back(entry);
        }

        if (!contig_markers_[node].empty()) {
            num_marker_contigs_++;
        }
    }

    std::cerr << "[MarkerIndex] DEBUG: found=" << found_count << ", not_found=" << not_found_count << "\n";
    if (!first_not_found.empty()) {
        std::cerr << "[MarkerIndex] DEBUG: First not found: '" << first_not_found << "' (len=" << first_not_found.size() << ")\n";
        // Check if a similar name exists with different suffix
        for (const auto& name : contig_names) {
            if (name.find(first_not_found.substr(0, 10)) != std::string::npos) {
                std::cerr << "[MarkerIndex] DEBUG: Similar in contigs: '" << name << "'\n";
                break;
            }
        }
    }

    std::cerr << "[MarkerIndex] Built index: " << num_markers_ << " unique markers, "
              << num_sets_ << " sets, " << num_marker_contigs_ << "/" << contig_names.size()
              << " contigs have markers\n";
}

void MarkerIndex::build_from_hits(
    const std::unordered_map<std::string, std::vector<std::string>>& hits,
    const std::vector<std::string>& contig_names) {

    // Clear previous state
    contig_markers_.clear();
    marker_to_id_.clear();
    marker_weights_.clear();
    num_markers_ = 0;
    num_marker_contigs_ = 0;

    // Step 1: Collect unique marker names from hits
    std::set<std::string> unique_markers;
    for (const auto& [contig, markers] : hits) {
        for (const auto& m : markers) {
            unique_markers.insert(m);
        }
    }

    num_markers_ = static_cast<int>(unique_markers.size());
    num_sets_ = num_markers_;  // Each marker is its own set

    if (num_markers_ == 0) {
        std::cerr << "[MarkerIndex] WARNING: No markers found in hits\n";
        contig_markers_.resize(contig_names.size());
        return;
    }

    // Assign IDs and uniform weights (100/num_markers per marker)
    float uniform_weight = 100.0f / num_markers_;
    marker_weights_.resize(num_markers_, uniform_weight);

    int id = 0;
    for (const auto& marker : unique_markers) {
        marker_to_id_[marker] = id++;
    }

    // Step 2: Build contig -> markers mapping
    std::unordered_map<std::string, int> name_to_idx;
    for (size_t i = 0; i < contig_names.size(); i++) {
        name_to_idx[contig_names[i]] = static_cast<int>(i);
    }

    contig_markers_.resize(contig_names.size());

    std::cerr << "[MarkerIndex] Using " << num_markers_ << " unique markers from HMM hits "
              << "(uniform weight=" << uniform_weight << " per marker)\n";

    int found_count = 0;
    int not_found_count = 0;

    for (const auto& [contig_name, markers] : hits) {
        auto it = name_to_idx.find(contig_name);
        if (it == name_to_idx.end()) {
            not_found_count++;
            continue;
        }
        found_count++;

        int node = it->second;

        // Count occurrences of each marker
        std::unordered_map<std::string, int> marker_counts;
        for (const auto& m : markers) {
            marker_counts[m]++;
        }

        // Build entry list
        for (const auto& [marker, count] : marker_counts) {
            MarkerEntry entry;
            entry.id = marker_to_id_[marker];
            entry.weight = marker_weights_[entry.id];
            entry.copy_count = count;
            contig_markers_[node].push_back(entry);
        }

        if (!contig_markers_[node].empty()) {
            num_marker_contigs_++;
        }
    }

    std::cerr << "[MarkerIndex] Matched " << found_count << "/" << (found_count + not_found_count)
              << " hit contigs to assembly, " << num_marker_contigs_
              << " contigs have markers\n";
}

QualityDelta compute_quality_delta(
    int node,
    int source_comm,
    int target_comm,
    const MarkerIndex& marker_index,
    const std::vector<std::unordered_map<int, int>>& comm_markers) {

    QualityDelta delta;

    const auto& markers = marker_index.get_markers(node);
    if (markers.empty()) {
        return delta;  // No markers, no quality impact
    }

    for (const auto& entry : markers) {
        int m = entry.id;
        float w = entry.weight;
        int k = entry.copy_count;  // Copies on this contig

        // Get current counts in source and target communities
        int n_old = 0;
        int n_new = 0;

        if (source_comm >= 0 && source_comm < static_cast<int>(comm_markers.size())) {
            auto it = comm_markers[source_comm].find(m);
            if (it != comm_markers[source_comm].end()) {
                n_old = it->second;
            }
        }

        if (target_comm >= 0 && target_comm < static_cast<int>(comm_markers.size())) {
            auto it = comm_markers[target_comm].find(m);
            if (it != comm_markers[target_comm].end()) {
                n_new = it->second;
            }
        }

        // Exact finite-difference formula (GPT-5.3-codex validated):
        // Δcomp = w * ([n_new + k > 0] - [n_new > 0] + [n_old - k > 0] - [n_old > 0])
        // Δcont = w * (max(n_new + k - 1, 0) - max(n_new - 1, 0)
        //            + max(n_old - k - 1, 0) - max(n_old - 1, 0))

        // Completeness delta
        int ind_new_after = (n_new + k > 0) ? 1 : 0;
        int ind_new_before = (n_new > 0) ? 1 : 0;
        int ind_old_after = (n_old - k > 0) ? 1 : 0;
        int ind_old_before = (n_old > 0) ? 1 : 0;

        delta.delta_completeness += w * (ind_new_after - ind_new_before + ind_old_after - ind_old_before);

        // Contamination delta
        int cont_new_after = std::max(n_new + k - 1, 0);
        int cont_new_before = std::max(n_new - 1, 0);
        int cont_old_after = std::max(n_old - k - 1, 0);
        int cont_old_before = std::max(n_old - 1, 0);

        delta.delta_contamination += w * (cont_new_after - cont_new_before + cont_old_after - cont_old_before);
    }

    return delta;
}

}  // namespace amber::bin2

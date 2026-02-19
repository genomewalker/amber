// AMBER - checkm_markers.h
// CheckM-style marker set parser and quality estimator
// Ports CheckM's colocation-based completeness/contamination formula

#pragma once

#include "marker_quality.h"  // For BinQualityScore

#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <map>

namespace amber::bin2 {

// A marker set is a group of co-located markers that should appear together
using MarkerSet = std::set<std::string>;

// Collection of marker sets for a domain (bacteria or archaea)
struct CheckMMarkerSets {
    std::string uid;
    std::string lineage;
    int num_genomes = 0;
    std::vector<MarkerSet> sets;  // List of co-located marker sets

    // Get all marker genes across all sets
    std::unordered_set<std::string> get_all_markers() const {
        std::unordered_set<std::string> all;
        for (const auto& s : sets) {
            all.insert(s.begin(), s.end());
        }
        return all;
    }

    int num_markers() const {
        int count = 0;
        for (const auto& s : sets) count += s.size();
        return count;
    }

    int num_sets() const { return sets.size(); }
};

// Quality result from CheckM-style calculation
struct CheckMQuality {
    float completeness = 0.0f;   // 0-100
    float contamination = 0.0f;  // 0-100
    int markers_found = 0;
    int markers_expected = 0;
    int sets_found = 0;
    int sets_expected = 0;
};

// Parse CheckM marker set files (.ms format)
class CheckMMarkerParser {
public:
    // Parse a .ms file (CheckM taxonomic marker set format)
    // Returns the most specific (first) marker set
    static CheckMMarkerSets parse_ms_file(const std::string& path);

    // Load both bacteria and archaea marker sets
    bool load(const std::string& bacteria_ms_path,
              const std::string& archaea_ms_path = "");

    const CheckMMarkerSets& bacteria() const { return bacteria_; }
    const CheckMMarkerSets& archaea() const { return archaea_; }

private:
    CheckMMarkerSets bacteria_;
    CheckMMarkerSets archaea_;
};

// CheckM-style quality estimator using colocation sets
class CheckMQualityEstimator {
public:
    CheckMQualityEstimator() = default;

    // Load marker sets from .ms files
    bool load_marker_sets(const std::string& bacteria_ms,
                          const std::string& archaea_ms = "");

    // Load marker hits from hmmsearch domtblout
    // hits: marker_id -> list of contig_ids where it's found
    bool load_hits(const std::string& domtblout_path);

    // Set hits directly (from seed generator)
    void set_hits(const std::unordered_map<std::string, std::vector<std::string>>& hits);

    // True if marker sets have been loaded (bacteria at minimum)
    bool has_marker_sets() const { return !parser_.bacteria().sets.empty(); }

    // Calculate quality for a bin (list of contig names)
    // Uses CheckM's colocation formula
    CheckMQuality estimate_bin_quality(const std::vector<std::string>& contig_names) const;

    // Calculate using specific domain
    CheckMQuality estimate_domain_quality(const std::vector<std::string>& contig_names,
                                          bool archaea) const;

    // Count HQ bins (>90% comp, <5% cont)
    int count_hq_bins(const std::vector<int>& labels,
                      const std::vector<std::pair<std::string, std::string>>& contigs,
                      int min_bin_size = 200000) const;

    // 6-category quality scoring (uses shared BinQualityScore from marker_quality.h)
    BinQualityScore compute_bin_quality_score(
        const std::vector<int>& labels,
        const std::vector<std::pair<std::string, std::string>>& contigs,
        int min_bin_size = 200000) const;

private:
    CheckMMarkerParser parser_;

    // contig -> list of markers found on that contig
    std::unordered_map<std::string, std::vector<std::string>> contig_markers_;

    // Helper: calculate quality using marker sets
    CheckMQuality calculate_quality(
        const std::unordered_map<std::string, int>& marker_counts,
        const CheckMMarkerSets& marker_sets) const;
};

}  // namespace amber::bin2

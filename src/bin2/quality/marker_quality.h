// AMBER - marker_quality.h
// Fast marker-based bin quality estimation using Anvi'o SCGs
// Supports both Bacteria_71 and Archaea_76 marker sets
//
// Calibration: Anvi'o SCG completeness scaled by 1.44× to match CheckM2
// (correlation 0.893, calibrated on 65 COMEBin bins)

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>

namespace amber::bin2 {

enum class Domain { BACTERIA, ARCHAEA, UNKNOWN };

struct MarkerHit {
    std::string contig;
    std::string marker;
    float score;
    Domain domain = Domain::UNKNOWN;
};

struct BinQuality {
    float completeness;    // 0-100
    float contamination;   // 0-100
    int unique_markers;
    int total_markers;
    int duplicate_markers;
    Domain domain = Domain::UNKNOWN;

    bool is_high_quality() const {
        return completeness >= 90.0f && contamination < 5.0f;
    }

    bool is_medium_quality() const {
        return completeness >= 50.0f && contamination < 10.0f;
    }
};

struct SweepResult {
    int partgraph_ratio;
    float bandwidth;
    float resolution;
    std::vector<int> labels;
    int num_clusters;
    int num_hq_bins;
    int num_mq_bins;
    float modularity;
};

// COMEBin-style 6-category quality scoring
// Categories: (completeness > 50/70/90%) × (contamination < 10/5%)
struct COMEBinScore {
    int num_5010 = 0;  // comp > 50%, cont < 10%
    int num_7010 = 0;  // comp > 70%, cont < 10%
    int num_9010 = 0;  // comp > 90%, cont < 10%
    int num_505 = 0;   // comp > 50%, cont < 5%
    int num_705 = 0;   // comp > 70%, cont < 5%
    int num_905 = 0;   // comp > 90%, cont < 5%

    // Primary score: sum of all 6 categories
    int sum() const {
        return num_5010 + num_7010 + num_9010 + num_505 + num_705 + num_905;
    }

    // Tiebreaker: sum of <5% contamination bins only
    int sum_cont5() const {
        return num_505 + num_705 + num_905;
    }

    // For comparison: returns true if this score is better than other
    // COMEBin selection order:
    // 1) max(sum of all 6 categories)
    // 2) tiebreak by max(sum of <5% contamination categories)
    bool better_than(const COMEBinScore& other) const {
        // Priority 1: Higher total sum
        if (sum() != other.sum()) return sum() > other.sum();
        // Priority 2: More <5% contamination bins
        return sum_cont5() > other.sum_cont5();
    }
};

// Marker-based quality estimator using Anvi'o SCG HMMs
// Supports dual-domain: Bacteria_71 (71 markers) + Archaea_76 (76 markers)
class MarkerQualityEstimator {
public:
    // Default: Anvi'o Bacteria_71 (71 markers)
    // Set dual_domain=true for combined Bacteria+Archaea estimation
    MarkerQualityEstimator(bool dual_domain = false);

    // Load marker gene lists from Anvi'o genes.txt files
    bool load_marker_list(const std::string& bacteria_genes_path,
                          const std::string& archaea_genes_path = "");

    // Load marker hits from HMMER output (domtblout format)
    bool load_hits(const std::string& domtblout_path);

    // Set marker hits directly (from SeedGenerator)
    void set_hits(const std::vector<MarkerHit>& hits);

    // Estimate quality for a single bin (list of contig names)
    // Returns best quality across domains when dual_domain=true
    BinQuality estimate_bin_quality(const std::vector<std::string>& contig_names) const;

    // Estimate quality for all bins from clustering labels
    std::map<int, BinQuality> estimate_all_bins(
        const std::vector<int>& labels,
        const std::vector<std::pair<std::string, std::string>>& contigs) const;

    // Count HQ bins from clustering result
    int count_hq_bins(
        const std::vector<int>& labels,
        const std::vector<std::pair<std::string, std::string>>& contigs,
        int min_bin_size = 200000) const;

    // Count MQ bins from clustering result
    int count_mq_bins(
        const std::vector<int>& labels,
        const std::vector<std::pair<std::string, std::string>>& contigs,
        int min_bin_size = 200000) const;

    // Score function: weighted sum of HQ and MQ bins (legacy)
    // score = 3*HQ + MQ
    int compute_score(
        const std::vector<int>& labels,
        const std::vector<std::pair<std::string, std::string>>& contigs,
        int min_bin_size = 200000) const;

    // COMEBin-style 6-category scoring
    // Primary: max(sum of all 6 categories)
    // Tiebreaker: max(sum of <5% contamination bins)
    COMEBinScore compute_comebin_score(
        const std::vector<int>& labels,
        const std::vector<std::pair<std::string, std::string>>& contigs,
        int min_bin_size = 200000) const;

    // Get marker counts
    int bacteria_marker_count() const { return bacteria_markers_.size(); }
    int archaea_marker_count() const { return archaea_markers_.size(); }
    bool is_dual_domain() const { return dual_domain_; }

private:
    bool dual_domain_;

    // Domain-specific marker sets
    std::unordered_set<std::string> bacteria_markers_;  // 71 Anvi'o bacteria SCGs
    std::unordered_set<std::string> archaea_markers_;   // 76 Anvi'o archaea SCGs

    // contig -> list of markers found (with domain info)
    std::unordered_map<std::string, std::vector<std::pair<std::string, Domain>>> contig_markers_;

    // Helper: estimate quality for specific domain
    BinQuality estimate_domain_quality(
        const std::vector<std::string>& contig_names,
        Domain domain) const;
};

}  // namespace amber::bin2

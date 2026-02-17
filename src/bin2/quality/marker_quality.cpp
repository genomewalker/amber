// AMBER - marker_quality.cpp
// Fast marker-based bin quality estimation using Anvi'o SCGs

#include "marker_quality.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include <utility>

namespace amber::bin2 {

MarkerQualityEstimator::MarkerQualityEstimator(bool dual_domain)
    : dual_domain_(dual_domain) {}

bool MarkerQualityEstimator::load_marker_list(
    const std::string& bacteria_genes_path,
    const std::string& archaea_genes_path) {

    // Load bacteria markers
    std::ifstream bac_file(bacteria_genes_path);
    if (bac_file.is_open()) {
        std::string line;
        std::getline(bac_file, line);  // Skip header
        while (std::getline(bac_file, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string gene;
            iss >> gene;
            if (!gene.empty()) {
                bacteria_markers_.insert(gene);
            }
        }
    }

    // Load archaea markers if dual-domain
    if (dual_domain_ && !archaea_genes_path.empty()) {
        std::ifstream arc_file(archaea_genes_path);
        if (arc_file.is_open()) {
            std::string line;
            std::getline(arc_file, line);  // Skip header
            while (std::getline(arc_file, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                std::string gene;
                iss >> gene;
                if (!gene.empty()) {
                    archaea_markers_.insert(gene);
                }
            }
        }
    }

    return !bacteria_markers_.empty();
}

bool MarkerQualityEstimator::load_hits(const std::string& domtblout_path) {
    std::ifstream file(domtblout_path);
    if (!file.is_open()) return false;

    int total_hits = 0;
    int matched_bacteria = 0;
    int matched_archaea = 0;
    int unmatched = 0;

    // Track seen (contig, marker) pairs to deduplicate
    std::set<std::pair<std::string, std::string>> seen;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string target, _, marker;
        iss >> target >> _ >> _ >> marker;

        // Extract contig name from FragGeneScan target format
        // Format: contig_name_startpos_endpos_strand
        auto last_underscore = target.rfind('_');
        if (last_underscore != std::string::npos) {
            auto second_last = target.rfind('_', last_underscore - 1);
            if (second_last != std::string::npos) {
                auto third_last = target.rfind('_', second_last - 1);
                if (third_last != std::string::npos) {
                    target = target.substr(0, third_last);
                }
            }
        }

        total_hits++;

        // Deduplicate: same marker on same contig counts once
        auto key = std::make_pair(target, marker);
        if (seen.count(key) > 0) continue;
        seen.insert(key);

        // Determine domain based on marker membership
        Domain domain = Domain::UNKNOWN;
        if (bacteria_markers_.count(marker) > 0) {
            domain = Domain::BACTERIA;
            matched_bacteria++;
        } else if (archaea_markers_.count(marker) > 0) {
            domain = Domain::ARCHAEA;
            matched_archaea++;
        } else {
            unmatched++;
        }

        contig_markers_[target].emplace_back(marker, domain);
    }

    fprintf(stderr, "[MarkerQuality] Loaded %d raw hits, deduplicated to %zu unique (contig,marker) pairs: %d bacteria, %d archaea, %d unmatched, %zu contigs\n",
            total_hits, seen.size(), matched_bacteria, matched_archaea, unmatched, contig_markers_.size());

    return true;
}

void MarkerQualityEstimator::set_hits(const std::vector<MarkerHit>& hits) {
    contig_markers_.clear();
    for (const auto& hit : hits) {
        Domain domain = hit.domain;
        if (domain == Domain::UNKNOWN) {
            if (bacteria_markers_.count(hit.marker) > 0) {
                domain = Domain::BACTERIA;
            } else if (archaea_markers_.count(hit.marker) > 0) {
                domain = Domain::ARCHAEA;
            }
        }
        contig_markers_[hit.contig].emplace_back(hit.marker, domain);
    }
}

BinQuality MarkerQualityEstimator::estimate_domain_quality(
    const std::vector<std::string>& contig_names,
    Domain domain) const {

    BinQuality q = {0, 0, 0, 0, 0, domain};

    const auto& marker_set = (domain == Domain::ARCHAEA) ? archaea_markers_ : bacteria_markers_;
    int expected_markers = marker_set.size();
    if (expected_markers == 0) expected_markers = 71;  // Default bacteria

    // Count markers in this bin for specific domain
    std::map<std::string, int> marker_counts;
    for (const auto& contig : contig_names) {
        auto it = contig_markers_.find(contig);
        if (it != contig_markers_.end()) {
            for (const auto& [marker, hit_domain] : it->second) {
                // Only count markers belonging to this domain
                if (marker_set.count(marker) > 0) {
                    marker_counts[marker]++;
                }
            }
        }
    }

    // Calculate quality metrics
    q.unique_markers = marker_counts.size();
    for (const auto& [marker, count] : marker_counts) {
        q.total_markers += count;
        if (count > 1) {
            q.duplicate_markers += (count - 1);
        }
    }

    // Completeness = unique markers / expected single-copy markers
    // Apply 1.44× scaling factor to match CheckM2 (calibrated on COMEBin bins)
    // Anvi'o SCG underestimates by ~25% vs CheckM2 (correlation 0.893)
    constexpr float CHECKM2_SCALING = 1.44f;
    float raw_completeness = 100.0f * q.unique_markers / expected_markers;
    q.completeness = std::min(100.0f, raw_completeness * CHECKM2_SCALING);

    // Contamination = duplicate markers / unique markers (Anvi'o style)
    // This differs from COMEBin which uses duplicates/total
    if (q.unique_markers > 0) {
        q.contamination = 100.0f * q.duplicate_markers / q.unique_markers;
    }

    return q;
}

BinQuality MarkerQualityEstimator::estimate_bin_quality(
    const std::vector<std::string>& contig_names) const {

    // Estimate for bacteria
    auto bac_quality = estimate_domain_quality(contig_names, Domain::BACTERIA);

    if (!dual_domain_ || archaea_markers_.empty()) {
        return bac_quality;
    }

    // Estimate for archaea and pick best
    auto arc_quality = estimate_domain_quality(contig_names, Domain::ARCHAEA);

    // Return domain with higher completeness (or lower contamination as tiebreaker)
    if (arc_quality.completeness > bac_quality.completeness ||
        (arc_quality.completeness == bac_quality.completeness &&
         arc_quality.contamination < bac_quality.contamination)) {
        return arc_quality;
    }
    return bac_quality;
}

std::map<int, BinQuality> MarkerQualityEstimator::estimate_all_bins(
    const std::vector<int>& labels,
    const std::vector<std::pair<std::string, std::string>>& contigs) const {

    // Group contigs by cluster
    std::map<int, std::vector<std::string>> clusters;
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] >= 0) {  // Skip unbinned
            clusters[labels[i]].push_back(contigs[i].first);
        }
    }

    // Estimate quality for each cluster
    std::map<int, BinQuality> results;
    for (const auto& [cluster_id, contig_names] : clusters) {
        results[cluster_id] = estimate_bin_quality(contig_names);
    }

    return results;
}

int MarkerQualityEstimator::count_hq_bins(
    const std::vector<int>& labels,
    const std::vector<std::pair<std::string, std::string>>& contigs,
    int min_bin_size) const {

    // Group contigs by cluster with sizes
    std::map<int, std::vector<std::string>> cluster_contigs;
    std::map<int, size_t> cluster_sizes;
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] >= 0) {
            cluster_contigs[labels[i]].push_back(contigs[i].first);
            cluster_sizes[labels[i]] += contigs[i].second.length();
        }
    }

    int hq_count = 0;
    int bins_checked = 0;
    float best_comp = 0, best_cont = 100;
    for (const auto& [cluster_id, contig_names] : cluster_contigs) {
        // Skip small bins
        if (static_cast<int>(cluster_sizes[cluster_id]) < min_bin_size) continue;

        bins_checked++;
        auto q = estimate_bin_quality(contig_names);
        if (q.completeness > best_comp) {
            best_comp = q.completeness;
            best_cont = q.contamination;
        }
        if (q.is_high_quality()) {
            hq_count++;
        }
    }

    // Debug: print best quality found on first call per sweep
    static int call_count = 0;
    if (++call_count <= 3) {
        fprintf(stderr, "[MarkerQuality] count_hq: %d bins checked, best: %.1f%% comp, %.1f%% cont, %d HQ\n",
                bins_checked, best_comp, best_cont, hq_count);
    }

    return hq_count;
}

int MarkerQualityEstimator::count_mq_bins(
    const std::vector<int>& labels,
    const std::vector<std::pair<std::string, std::string>>& contigs,
    int min_bin_size) const {

    // Group contigs by cluster with sizes
    std::map<int, std::vector<std::string>> cluster_contigs;
    std::map<int, size_t> cluster_sizes;
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] >= 0) {
            cluster_contigs[labels[i]].push_back(contigs[i].first);
            cluster_sizes[labels[i]] += contigs[i].second.length();
        }
    }

    int mq_count = 0;
    for (const auto& [cluster_id, contig_names] : cluster_contigs) {
        // Skip small bins
        if (static_cast<int>(cluster_sizes[cluster_id]) < min_bin_size) continue;

        auto q = estimate_bin_quality(contig_names);
        // MQ = medium quality but not HQ
        if (q.is_medium_quality() && !q.is_high_quality()) {
            mq_count++;
        }
    }

    return mq_count;
}

int MarkerQualityEstimator::compute_score(
    const std::vector<int>& labels,
    const std::vector<std::pair<std::string, std::string>>& contigs,
    int min_bin_size) const {

    int hq = count_hq_bins(labels, contigs, min_bin_size);
    int mq = count_mq_bins(labels, contigs, min_bin_size);

    // Legacy scoring: prioritize HQ
    return 3 * hq + mq;
}

COMEBinScore MarkerQualityEstimator::compute_comebin_score(
    const std::vector<int>& labels,
    const std::vector<std::pair<std::string, std::string>>& contigs,
    int min_bin_size) const {

    // Group contigs by cluster with sizes
    std::map<int, std::vector<std::string>> cluster_contigs;
    std::map<int, size_t> cluster_sizes;
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] >= 0) {
            cluster_contigs[labels[i]].push_back(contigs[i].first);
            cluster_sizes[labels[i]] += contigs[i].second.length();
        }
    }

    COMEBinScore score;
    for (const auto& [cluster_id, contig_names] : cluster_contigs) {
        // Skip small bins
        if (static_cast<int>(cluster_sizes[cluster_id]) < min_bin_size) continue;

        auto q = estimate_bin_quality(contig_names);

        // Count into 6 categories based on completeness/contamination thresholds
        // Note: bins can fall into multiple categories (e.g., 95% comp goes into both 50/70/90)
        if (q.completeness > 50.0f && q.contamination < 10.0f) score.num_5010++;
        if (q.completeness > 70.0f && q.contamination < 10.0f) score.num_7010++;
        if (q.completeness > 90.0f && q.contamination < 10.0f) score.num_9010++;
        if (q.completeness > 50.0f && q.contamination < 5.0f) score.num_505++;
        if (q.completeness > 70.0f && q.contamination < 5.0f) score.num_705++;
        if (q.completeness > 90.0f && q.contamination < 5.0f) score.num_905++;
    }

    return score;
}

}  // namespace amber::bin2

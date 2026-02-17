// AMBER - checkm_markers.cpp
// CheckM-style marker set parser and quality estimator

#include "checkm_markers.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <regex>

namespace amber::bin2 {

static std::string normalize_marker_id(const std::string& marker) {
    // CheckM marker sets often include PFAM version suffixes (e.g. PF01000.21)
    // while HMM databases may use a different version. Match on stable ID.
    if (marker.rfind("PF", 0) == 0) {
        auto dot = marker.find('.');
        if (dot != std::string::npos) {
            return marker.substr(0, dot);
        }
    }
    return marker;
}

// Parse Python set representation: set(['PF01000.21', 'PF01196.14'])
static MarkerSet parse_python_set(const std::string& s) {
    MarkerSet result;
    // Match quoted strings inside set([...])
    std::regex marker_regex("'([^']+)'");
    auto begin = std::sregex_iterator(s.begin(), s.end(), marker_regex);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        result.insert(normalize_marker_id((*it)[1].str()));
    }
    return result;
}

CheckMMarkerSets CheckMMarkerParser::parse_ms_file(const std::string& path) {
    CheckMMarkerSets result;
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "[CheckM] Failed to open marker set file: %s\n", path.c_str());
        return result;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        // Format: lineage<TAB>uid<TAB>?<TAB>name<TAB>num_genomes<TAB>[set([...]), set([...]), ...]
        std::istringstream iss(line);
        std::string lineage, uid_str, field3, name, num_genomes_str;
        if (!std::getline(iss, lineage, '\t')) continue;
        if (!std::getline(iss, uid_str, '\t')) continue;
        if (!std::getline(iss, field3, '\t')) continue;
        if (!std::getline(iss, name, '\t')) continue;
        if (!std::getline(iss, num_genomes_str, '\t')) continue;

        result.uid = uid_str;
        result.lineage = lineage;
        result.num_genomes = std::stoi(num_genomes_str);

        // Rest of line contains the marker sets
        std::string sets_str;
        std::getline(iss, sets_str);

        // Parse the list of sets: [set([...]), set([...]), ...]
        // Find each set([...]) block
        std::regex set_regex("set\\(\\[([^\\]]+)\\]\\)");
        auto begin = std::sregex_iterator(sets_str.begin(), sets_str.end(), set_regex);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            std::string set_content = (*it)[0].str();
            MarkerSet ms = parse_python_set(set_content);
            if (!ms.empty()) {
                result.sets.push_back(std::move(ms));
            }
        }

        // Only parse first line (most specific marker set)
        break;
    }

    fprintf(stderr, "[CheckM] Loaded %s: %d marker sets, %d total markers\n",
            result.lineage.c_str(), result.num_sets(), result.num_markers());

    return result;
}

bool CheckMMarkerParser::load(const std::string& bacteria_ms_path,
                              const std::string& archaea_ms_path) {
    bacteria_ = parse_ms_file(bacteria_ms_path);
    if (!archaea_ms_path.empty()) {
        archaea_ = parse_ms_file(archaea_ms_path);
    }
    return !bacteria_.sets.empty();
}

bool CheckMQualityEstimator::load_marker_sets(const std::string& bacteria_ms,
                                               const std::string& archaea_ms) {
    return parser_.load(bacteria_ms, archaea_ms);
}

bool CheckMQualityEstimator::load_hits(const std::string& domtblout_path) {
    std::ifstream file(domtblout_path);
    if (!file.is_open()) return false;

    contig_markers_.clear();
    std::set<std::pair<std::string, std::string>> seen;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string target, target_acc, tlen, query_name, query_acc;
        if (!(iss >> target >> target_acc >> tlen >> query_name >> query_acc)) continue;
        std::string marker = (query_acc != "-" ? query_acc : query_name);
        marker = normalize_marker_id(marker);

        // Extract contig name from FragGeneScan format: contig_start_end_strand
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

        // Deduplicate (contig, marker) pairs
        auto key = std::make_pair(target, marker);
        if (seen.count(key) > 0) continue;
        seen.insert(key);

        contig_markers_[target].push_back(marker);
    }

    fprintf(stderr, "[CheckM] Loaded markers for %zu contigs\n", contig_markers_.size());
    return true;
}

void CheckMQualityEstimator::set_hits(
    const std::unordered_map<std::string, std::vector<std::string>>& hits) {
    contig_markers_ = hits;
}

CheckMQuality CheckMQualityEstimator::calculate_quality(
    const std::unordered_map<std::string, int>& marker_counts,
    const CheckMMarkerSets& marker_sets) const {

    CheckMQuality q;
    q.sets_expected = marker_sets.num_sets();
    q.markers_expected = marker_sets.num_markers();

    if (q.sets_expected == 0) return q;

    float comp = 0.0f;
    float cont = 0.0f;

    for (const auto& ms : marker_sets.sets) {
        int present = 0;
        int multi_copy = 0;

        for (const auto& marker : ms) {
            auto it = marker_counts.find(marker);
            if (it != marker_counts.end()) {
                int count = it->second;
                if (count >= 1) {
                    present++;
                    q.markers_found++;
                    if (count > 1) {
                        multi_copy += (count - 1);
                    }
                }
            }
        }

        if (present > 0) q.sets_found++;

        // CheckM formula: contribution per set
        comp += static_cast<float>(present) / ms.size();
        cont += static_cast<float>(multi_copy) / ms.size();
    }

    // Final percentages
    q.completeness = 100.0f * comp / q.sets_expected;
    q.contamination = 100.0f * cont / q.sets_expected;

    return q;
}

CheckMQuality CheckMQualityEstimator::estimate_domain_quality(
    const std::vector<std::string>& contig_names,
    bool archaea) const {

    const auto& marker_sets = archaea ? parser_.archaea() : parser_.bacteria();

    // Count markers in this bin
    std::unordered_map<std::string, int> marker_counts;
    for (const auto& contig : contig_names) {
        auto it = contig_markers_.find(contig);
        if (it != contig_markers_.end()) {
            for (const auto& marker : it->second) {
                marker_counts[marker]++;
            }
        }
    }

    return calculate_quality(marker_counts, marker_sets);
}

CheckMQuality CheckMQualityEstimator::estimate_bin_quality(
    const std::vector<std::string>& contig_names) const {

    auto bac_q = estimate_domain_quality(contig_names, false);

    if (parser_.archaea().sets.empty()) {
        return bac_q;
    }

    auto arc_q = estimate_domain_quality(contig_names, true);

    // Select domain with higher completeness (NOT completeness + contamination!)
    // Previous bug: rewarding contamination led to wrong domain selection
    if (arc_q.completeness > bac_q.completeness) {
        return arc_q;
    }
    return bac_q;
}

int CheckMQualityEstimator::count_hq_bins(
    const std::vector<int>& labels,
    const std::vector<std::pair<std::string, std::string>>& contigs,
    int min_bin_size) const {

    // Group contigs by cluster
    std::map<int, std::vector<std::string>> cluster_contigs;
    std::map<int, size_t> cluster_sizes;
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] >= 0) {
            cluster_contigs[labels[i]].push_back(contigs[i].first);
            cluster_sizes[labels[i]] += contigs[i].second.length();
        }
    }

    int hq_count = 0;
    for (const auto& [cluster_id, contig_names] : cluster_contigs) {
        if (static_cast<int>(cluster_sizes[cluster_id]) < min_bin_size) continue;

        auto q = estimate_bin_quality(contig_names);
        if (q.completeness >= 90.0f && q.contamination <= 5.0f) {
            hq_count++;
        }
    }

    return hq_count;
}

COMEBinScore CheckMQualityEstimator::compute_comebin_score(
    const std::vector<int>& labels,
    const std::vector<std::pair<std::string, std::string>>& contigs,
    int min_bin_size) const {

    // Group contigs by cluster
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
        if (static_cast<int>(cluster_sizes[cluster_id]) < min_bin_size) continue;

        auto q = estimate_bin_quality(contig_names);

        // Count into 6 categories (use >= and <= to match CheckM1 thresholds)
        if (q.completeness >= 50.0f && q.contamination <= 10.0f) score.num_5010++;
        if (q.completeness >= 70.0f && q.contamination <= 10.0f) score.num_7010++;
        if (q.completeness >= 90.0f && q.contamination <= 10.0f) score.num_9010++;
        if (q.completeness >= 50.0f && q.contamination <= 5.0f) score.num_505++;
        if (q.completeness >= 70.0f && q.contamination <= 5.0f) score.num_705++;
        if (q.completeness >= 90.0f && q.contamination <= 5.0f) score.num_905++;
    }

    return score;
}

}  // namespace amber::bin2

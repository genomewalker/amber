// AMBER - seed_generator.cpp
// SCG marker-based seed generation for COMEBin-style binning

#include "seed_generator.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>
#include <climits>

namespace amber {

SeedGenerator::SeedGenerator(int threads, const std::string& hmm_path, int min_length,
                             const std::string& hmmsearch_path, const std::string& fraggenescan_path)
    : threads_(threads), hmm_path_(hmm_path), min_length_(min_length),
      hmmsearch_path_(hmmsearch_path), fraggenescan_path_(fraggenescan_path) {}

std::vector<std::string> SeedGenerator::generate(const std::string& contigs_path,
                                                  const std::string& output_path) {
    // Create output directory if needed
    std::string output_dir = output_path;
    size_t last_slash = output_dir.find_last_of('/');
    if (last_slash != std::string::npos) {
        output_dir = output_dir.substr(0, last_slash);
        mkdir(output_dir.c_str(), 0755);
    }

    // Load contig lengths for filtering
    std::cerr << "[seeds] Loading contig lengths from " << contigs_path << "...\n";
    load_contig_lengths(contigs_path);
    std::cerr << "[seeds] Loaded " << contig_lengths_.size() << " contigs\n";

    // Use temp prefix for intermediate files
    std::string temp_prefix = output_path + ".frag";

    // Step 1: Run FragGeneScan
    std::cerr << "[seeds] Running FragGeneScan on " << contigs_path << "...\n";
    if (!run_fraggenescan(contigs_path, temp_prefix)) {
        std::cerr << "[seeds] ERROR: FragGeneScan failed\n";
        return {};
    }

    std::string faa_path = temp_prefix + ".faa";
    faa_path_ = faa_path;

    // Count ORFs
    std::ifstream faa_file(faa_path);
    std::string line;
    while (std::getline(faa_file, line)) {
        if (!line.empty() && line[0] == '>') {
            total_orfs_++;
        }
    }
    faa_file.close();
    std::cerr << "[seeds] Found " << total_orfs_ << " ORFs\n";

    // Step 2: Run HMMER
    std::string domtblout_path = output_path + ".hmmout";
    std::cerr << "[seeds] Running hmmsearch with " << hmm_path_ << "...\n";
    if (!run_hmmsearch(faa_path, domtblout_path)) {
        std::cerr << "[seeds] ERROR: hmmsearch failed\n";
        return {};
    }

    // Step 3: Parse results
    std::cerr << "[seeds] Parsing HMMER results...\n";
    // Load NAME->accession map from HMM file so downstream colocation scoring
    // (bacteria.ms uses PF/TIGRFAM accessions) matches hit names correctly.
    load_hmm_accessions(hmm_path_);
    auto hits = parse_domtblout(domtblout_path);
    total_hits_ = hits.size();
    std::cerr << "[seeds] Found " << total_hits_ << " raw hits\n";

    // Build hits_by_contig index, translating HMM profile NAMEs to normalized
    // accessions (e.g. "Ribosomal_L39" -> "PF00832") so they align with bacteria.ms.
    hits_by_contig_.clear();
    for (const auto& hit : hits) {
        auto it = hmm_name_to_acc_.find(hit.marker);
        const std::string& acc = (it != hmm_name_to_acc_.end()) ? it->second : hit.marker;
        hits_by_contig_[hit.contig].push_back(acc);
    }

    // Step 4: Select seeds
    auto seeds = select_seeds(hits, output_path);
    std::cerr << "[seeds] Selected " << seeds.size() << " seed contigs\n";

    return seeds;
}

std::unordered_map<std::string, std::vector<std::string>>
SeedGenerator::search_additional_hmm(const std::string& hmm_path,
                                      const std::string& output_prefix) {
    if (faa_path_.empty() || !std::filesystem::exists(faa_path_)) {
        std::cerr << "[seeds] search_additional_hmm: no FAA file (call generate() first)\n";
        return {};
    }

    std::string saved_hmm = hmm_path_;
    hmm_path_ = hmm_path;

    std::string domtblout = output_prefix + ".hmmout";
    std::cerr << "[seeds] Running additional hmmsearch: " << hmm_path << "\n";
    bool ok = run_hmmsearch(faa_path_, domtblout);

    hmm_path_ = saved_hmm;

    if (!ok) {
        std::cerr << "[seeds] Additional hmmsearch failed\n";
        return {};
    }

    auto hits = parse_domtblout(domtblout);
    std::unordered_map<std::string, std::vector<std::string>> result;
    for (const auto& hit : hits)
        result[hit.contig].push_back(hit.marker);
    return result;
}

bool SeedGenerator::run_fraggenescan(const std::string& contigs_path,
                                      const std::string& output_prefix) {
    // FragGeneScan requires the 'train' directory relative to cwd
    // Resolve full path if just a command name (from PATH)
    std::string fgs_path = fraggenescan_path_;
    if (fgs_path.find('/') == std::string::npos) {
        // Resolve via 'which'
        std::string which_cmd = "which " + fgs_path + " 2>/dev/null";
        FILE* pipe = popen(which_cmd.c_str(), "r");
        if (pipe) {
            char buffer[512];
            if (fgets(buffer, sizeof(buffer), pipe)) {
                fgs_path = buffer;
                // Remove trailing newline
                while (!fgs_path.empty() && (fgs_path.back() == '\n' || fgs_path.back() == '\r')) {
                    fgs_path.pop_back();
                }
            }
            pclose(pipe);
        }
    }

    // Get the directory containing the FragGeneScan binary
    std::string fgs_dir;
    size_t last_slash = fgs_path.find_last_of('/');
    if (last_slash != std::string::npos) {
        fgs_dir = fgs_path.substr(0, last_slash);
    } else {
        fgs_dir = ".";
    }

    std::cerr << "[seeds] FragGeneScan resolved to: " << fgs_path << "\n";
    std::cerr << "[seeds] Running from directory: " << fgs_dir << "\n";

    // Convert output_prefix to absolute path (we'll cd to fgs_dir)
    std::string abs_output = output_prefix;
    if (!output_prefix.empty() && output_prefix[0] != '/') {
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd))) {
            abs_output = std::string(cwd) + "/" + output_prefix;
        }
    }

    // Build command that runs from the FragGeneScan directory
    std::ostringstream cmd;
    cmd << "cd " << fgs_dir << " && "
        << fgs_path
        << " -s " << contigs_path
        << " -o " << abs_output
        << " -w 0"
        << " -t complete"
        << " -p " << threads_
        << " 2>&1";

    int ret = std::system(cmd.str().c_str());
    return ret == 0;
}

bool SeedGenerator::run_hmmsearch(const std::string& faa_path,
                                   const std::string& output_path) {
    std::ostringstream cmd;
    // Try --cut_tc first (works with CheckM HMMs that have TC thresholds defined)
    cmd << hmmsearch_path_
        << " --domtblout " << output_path
        << " --cut_tc"
        << " --cpu " << threads_
        << " " << hmm_path_
        << " " << faa_path
        << " > /dev/null 2>&1";

    int ret = std::system(cmd.str().c_str());

    // Check if output is empty (--cut_tc failed due to missing TC thresholds)
    std::ifstream check(output_path);
    std::string line;
    bool has_hits = false;
    while (std::getline(check, line)) {
        if (!line.empty() && line[0] != '#') {
            has_hits = true;
            break;
        }
    }
    check.close();

    if (!has_hits) {
        // Retry without --cut_tc, using relaxed E-value threshold
        // Fallback for HMM files without TC thresholds; use E-value 1.0 and filter by coverage
        std::cerr << "[seeds] No hits with --cut_tc, retrying with -E 1...\n";
        std::ostringstream cmd2;
        cmd2 << hmmsearch_path_
             << " --domtblout " << output_path
             << " -E 1"
             << " --cpu " << threads_
             << " " << hmm_path_
             << " " << faa_path
             << " > /dev/null 2>&1";
        ret = std::system(cmd2.str().c_str());
    }

    return ret == 0;
}

std::vector<HMMHit> SeedGenerator::parse_domtblout(const std::string& domtblout_path) {
    std::vector<HMMHit> hits;
    std::ifstream file(domtblout_path);
    std::string line;

    while (std::getline(file, line)) {
        // Skip comment lines
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Parse whitespace-separated columns
        std::istringstream iss(line);
        std::vector<std::string> cols;
        std::string col;
        while (iss >> col) {
            cols.push_back(col);
        }

        // Need at least 17 columns for ali_to
        if (cols.size() < 17) {
            continue;
        }

        HMMHit hit;
        hit.contig = extract_contig(cols[0]);  // target name
        if (hit.contig.empty()) {
            continue;
        }

        hit.marker = check_marker(cols[3]);    // query name (marker)
        hit.qlen = std::stoi(cols[5]);         // query length
        hit.ali_from = std::stoi(cols[15]);    // alignment start
        hit.ali_to = std::stoi(cols[16]);      // alignment end

        // Calculate coverage
        hit.coverage = static_cast<double>(hit.ali_to - hit.ali_from) /
                       static_cast<double>(hit.qlen);

        // Coverage filter: >= 0.4
        if (hit.coverage >= 0.4) {
            hits.push_back(hit);
            filtered_hits_++;
        }
    }

    return hits;
}

void SeedGenerator::load_hmm_accessions(const std::string& hmm_path) {
    hmm_name_to_acc_.clear();
    std::ifstream f(hmm_path);
    if (!f.is_open()) return;

    std::string line, current_name;
    while (std::getline(f, line)) {
        if (line.rfind("NAME", 0) == 0 && line.size() > 5) {
            current_name = line.substr(4);
            // trim leading whitespace
            auto p = current_name.find_first_not_of(" \t");
            current_name = (p == std::string::npos) ? "" : current_name.substr(p);
        } else if (line.rfind("ACC", 0) == 0 && line.size() > 4 && !current_name.empty()) {
            std::string acc = line.substr(3);
            auto p = acc.find_first_not_of(" \t");
            acc = (p == std::string::npos) ? "" : acc.substr(p);
            // Strip version suffix from Pfam accessions (PF00832.15 -> PF00832)
            if (acc.rfind("PF", 0) == 0) {
                auto dot = acc.find('.');
                if (dot != std::string::npos) acc = acc.substr(0, dot);
            }
            if (!acc.empty()) hmm_name_to_acc_[current_name] = acc;
            current_name.clear();
        }
    }
    std::cerr << "[seeds] HMM accession map: " << hmm_name_to_acc_.size() << " entries\n";
}

std::string SeedGenerator::check_marker(const std::string& marker) {
    // COMEBin marker remapping
    if (marker == "TIGR00388") return "TIGR00389";
    if (marker == "TIGR00471") return "TIGR00472";
    if (marker == "TIGR00408") return "TIGR00409";
    if (marker == "TIGR02386") return "TIGR02387";
    return marker;
}

std::string SeedGenerator::extract_contig(const std::string& target) {
    // FragGeneScan output format: contigname_start_end_strand
    // e.g., k141_123456_100_500_+
    // We need to extract "k141_123456"

    // Regex: capture everything except the last three underscore-separated parts
    static std::regex pattern(R"((.+)_\d+_\d+_[+-]$)");
    std::smatch match;

    if (std::regex_match(target, match, pattern)) {
        return match[1].str();
    }

    // Fallback: try simpler pattern for non-standard names
    // Find last 3 underscores from the end
    size_t count = 0;
    size_t pos = target.length();
    while (pos > 0 && count < 3) {
        pos--;
        if (target[pos] == '_') {
            count++;
        }
    }

    if (count == 3 && pos > 0) {
        return target.substr(0, pos);
    }

    return "";
}

std::vector<std::string> SeedGenerator::select_seeds(const std::vector<HMMHit>& hits,
                                                      const std::string& output_path) {
    // Group hits by marker, filtering by contig length
    // COMEBin logic: only consider contigs >= min_length
    std::unordered_map<std::string, std::unordered_set<std::string>> by_marker;
    size_t length_filtered = 0;

    for (const auto& hit : hits) {
        // Check contig length
        auto it = contig_lengths_.find(hit.contig);
        if (it == contig_lengths_.end()) {
            continue;  // Contig not found (shouldn't happen)
        }

        if (static_cast<int>(it->second) < min_length_) {
            length_filtered++;
            continue;  // Skip short contigs
        }

        // Add unique contig to marker's set
        by_marker[hit.marker].insert(hit.contig);
    }

    std::cerr << "[seeds] Filtered " << length_filtered << " hits from contigs < "
              << min_length_ << " bp\n";

    if (by_marker.empty()) {
        return {};
    }

    // Count unique contigs per marker, find median
    std::vector<size_t> counts;
    for (const auto& [marker, contigs] : by_marker) {
        counts.push_back(contigs.size());
    }
    std::sort(counts.begin(), counts.end());
    size_t median_count = counts[counts.size() / 2];
    std::cerr << "[seeds] Median marker hit count: " << median_count << "\n";

    // COMEBin seed selection: pick the single marker whose count == median_count
    // (smallest query length as tiebreaker). Since the marker is single-copy, each
    // of its N hits comes from a different genome → exactly N seeds, one per genome.
    // Multi-marker selection risks assigning multiple seeds to the same genome
    // (different SCGs on different contigs of the same organism), which causes Leiden
    // to fragment that genome into separate fixed clusters.
    std::unordered_map<std::string, int> marker_qlen;
    for (const auto& hit : hits) {
        if (!marker_qlen.count(hit.marker) || hit.qlen < marker_qlen[hit.marker]) {
            marker_qlen[hit.marker] = hit.qlen;
        }
    }

    std::string best_marker;
    int best_qlen = std::numeric_limits<int>::max();
    for (const auto& [marker, contigs] : by_marker) {
        if (contigs.size() != median_count) continue;
        int qlen = marker_qlen.count(marker) ? marker_qlen[marker] : INT_MAX;
        if (qlen < best_qlen) {
            best_qlen = qlen;
            best_marker = marker;
        }
    }

    // Fallback: if no marker has exactly median_count, take closest above
    if (best_marker.empty()) {
        for (const auto& [marker, contigs] : by_marker) {
            if (contigs.size() < median_count) continue;
            int qlen = marker_qlen.count(marker) ? marker_qlen[marker] : INT_MAX;
            if (qlen < best_qlen) {
                best_qlen = qlen;
                best_marker = marker;
            }
        }
    }

    std::unordered_set<std::string> seed_set;
    if (!best_marker.empty()) {
        for (const auto& contig : by_marker[best_marker]) {
            seed_set.insert(contig);
        }
        selected_markers_ = 1;
        std::cerr << "[seeds] Selected marker: " << best_marker
                  << " (qlen=" << best_qlen << ", hits=" << seed_set.size() << ")\n";
    } else {
        selected_markers_ = 0;
    }

    std::vector<std::string> seeds(seed_set.begin(), seed_set.end());

    // Write seeds to output file
    std::ofstream out(output_path);
    for (const auto& seed : seeds) {
        out << seed << "\n";
    }
    out.close();

    return seeds;
}

void SeedGenerator::load_contig_lengths(const std::string& fasta_path) {
    std::ifstream file(fasta_path);
    if (!file.is_open()) {
        std::cerr << "[seeds] ERROR: Cannot open " << fasta_path << "\n";
        return;
    }

    std::string line;
    std::string current_name;
    size_t current_length = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (line[0] == '>') {
            // Save previous contig
            if (!current_name.empty()) {
                contig_lengths_[current_name] = current_length;
            }

            // Parse new contig name (first word after >)
            size_t space_pos = line.find_first_of(" \t", 1);
            if (space_pos != std::string::npos) {
                current_name = line.substr(1, space_pos - 1);
            } else {
                current_name = line.substr(1);
            }
            current_length = 0;
        } else {
            current_length += line.length();
        }
    }

    // Don't forget the last contig
    if (!current_name.empty()) {
        contig_lengths_[current_name] = current_length;
    }
}

}  // namespace amber

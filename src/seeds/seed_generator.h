// AMBER - seed_generator.h
// SCG marker-based seed generation for COMEBin-style binning

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace amber {

struct HMMHit {
    std::string contig;
    std::string marker;
    int qlen;
    int ali_from;
    int ali_to;
    double coverage;
};

class SeedGenerator {
public:
    SeedGenerator(int threads = 4, const std::string& hmm_path = "auxiliary/bacar_marker.hmm",
                  int min_length = 1001,
                  const std::string& hmmsearch_path = "hmmsearch",
                  const std::string& fraggenescan_path = "FragGeneScan");

    // Generate seed contigs from assembly
    // Returns vector of seed contig names
    std::vector<std::string> generate(const std::string& contigs_path,
                                      const std::string& output_path);

    // Get statistics after generation
    size_t total_orfs() const { return total_orfs_; }
    size_t total_hits() const { return total_hits_; }
    size_t filtered_hits() const { return filtered_hits_; }
    size_t selected_markers() const { return selected_markers_; }

private:
    int threads_;
    std::string hmm_path_;
    int min_length_;
    std::string hmmsearch_path_;
    std::string fraggenescan_path_;

    size_t total_orfs_ = 0;
    size_t total_hits_ = 0;
    size_t filtered_hits_ = 0;
    size_t selected_markers_ = 0;

    // Contig lengths loaded from FASTA
    std::unordered_map<std::string, size_t> contig_lengths_;

    // Load contig lengths from FASTA file
    void load_contig_lengths(const std::string& fasta_path);

    // Run FragGeneScan to predict ORFs
    bool run_fraggenescan(const std::string& contigs_path,
                          const std::string& output_prefix);

    // Run HMMER to find marker hits
    bool run_hmmsearch(const std::string& faa_path,
                       const std::string& output_path);

    // Parse HMMER domtblout format
    std::vector<HMMHit> parse_domtblout(const std::string& domtblout_path);

    // Remap certain marker names (COMEBin-specific)
    static std::string check_marker(const std::string& marker);

    // Extract contig name from FragGeneScan target name
    static std::string extract_contig(const std::string& target);

    // Select seed contigs based on marker statistics
    std::vector<std::string> select_seeds(const std::vector<HMMHit>& hits,
                                          const std::string& output_path);
};

}  // namespace amber

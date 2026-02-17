#pragma once

#include <string>
#include <vector>
#include <random>
#include <cstdint>
#include <zlib.h>

/**
 * Utilities for sequence manipulation in synthetic data generation
 */

namespace amber {

// FASTA sequence structure
struct FASTASequence {
    std::string id;
    std::string sequence;
    std::string description;
};

// Fragment metadata
struct Fragment {
    std::string id;
    std::string sequence;
    std::string source_genome;
    std::string taxonomy;
    size_t source_start;
    size_t source_end;
    bool is_chimera;
    std::vector<std::string> chimera_sources; // For multi-genome chimeras
    std::vector<size_t> chimera_breakpoints;
};

/**
 * FASTA I/O with gzip support
 */
class FASTAReader {
public:
    explicit FASTAReader(const std::string& path);
    ~FASTAReader();

    bool next(FASTASequence& seq);
    void reset();

private:
    std::string path_;
    gzFile gz_file_;
    FILE* pipe_file_;
    bool is_gzipped_;
    bool use_pigz_;
    char buffer_[262144];  // 256KB for better performance
    std::string next_line_;  // Buffer for lookahead (needed for pigz)
    bool has_next_line_;
};

void write_fasta(const std::string& path, const std::vector<Fragment>& fragments);
void write_fasta_single(const std::string& path, const Fragment& fragment);

/**
 * Read all contigs from a genome file, sorted by length (longest first)
 * Filters out contigs shorter than min_contig_length to remove contamination
 * Optionally limits to top max_contigs (0 = all contigs)
 */
std::vector<FASTASequence> read_longest_contigs(
    const std::string& genome_file,
    size_t min_contig_length = 0,
    size_t max_contigs = 0);

/**
 * Quickly get the maximum contig length from a genome file
 * Much faster than read_longest_contigs for filtering purposes
 * Returns 0 if file cannot be read or has no contigs
 */
size_t get_max_contig_length(const std::string& genome_file);

// Stitch all contigs with an N-buffer; can optionally create multiple shuffled replicates
std::vector<FASTASequence> stitch_contigs_with_buffer(
    const std::string& genome_file,
    size_t buffer_ns,
    size_t reps,
    bool shuffle,
    uint64_t seed);

/**
 * Sequence manipulation
 */
class SequenceManipulator {
public:
    explicit SequenceManipulator(uint64_t seed = 42);

    // Fragmentation
    std::vector<Fragment> fragment_random(
        const FASTASequence& genome,
        const std::string& taxonomy,
        size_t fragment_length,         // Mean fragment length
        size_t num_fragments,
        size_t min_length = 0,
        double length_stddev = 0.0,     // Stddev (normal/lognormal)
        const std::string& dist_type = "normal",  // normal, lognormal, gamma, exponential, powerlaw
        double shape = 2.0,             // Shape parameter (gamma, powerlaw)
        const std::string& genome_accession = "");  // Optional: override source_genome (defaults to genome.id)

    std::vector<Fragment> fragment_sliding_window(
        const FASTASequence& genome,
        const std::string& taxonomy,
        size_t window_size,
        size_t step_size,
        const std::string& genome_accession = "");  // Optional: override source_genome

    std::vector<Fragment> fragment_coverage_weighted(
        const FASTASequence& genome,
        const std::string& taxonomy,
        size_t fragment_length,
        size_t num_fragments,
        const std::vector<double>& coverage_profile,
        const std::string& genome_accession = "");  // Optional: override source_genome

    // Mutation introduction
    std::string introduce_snps(
        const std::string& sequence,
        double snp_rate,
        double transition_transversion_ratio = 2.0);

    std::string introduce_indels(
        const std::string& sequence,
        double indel_rate,
        double insert_delete_ratio = 1.0,
        size_t max_indel_length = 10);

    std::string introduce_divergence(
        const std::string& sequence,
        double divergence,
        bool include_indels = true);

    // Chimera creation
    Fragment create_chimera(
        const std::vector<FASTASequence>& genomes,
        const std::vector<std::string>& taxonomies,
        const std::vector<double>& ratios,
        size_t total_length);

    // HGT simulation
    std::string insert_hgt_region(
        const std::string& recipient,
        const std::string& donor_region,
        size_t insert_position);

    // Utility functions
    char complement(char base);
    std::string reverse_complement(const std::string& seq);
    char mutate_base(char base, bool transition_bias = true);

private:
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_;

    size_t sample_indel_length();
    char random_base();
};

/**
 * Quality control
 */
bool is_valid_dna(const std::string& seq);
double gc_content(const std::string& seq);
size_t count_n_bases(const std::string& seq);

/**
 * Contig renaming for anvi'o compatibility
 * Anvi'o requires: alphanumeric + underscore only, no special chars
 */
struct RenameEntry {
    std::string original_name;
    std::string new_name;
    size_t length;
};

class ContigRenamer {
public:
    // Anvi'o format: PREFIX_XXXXXXXXXXXX (underscore + 12 zero-padded digits)
    explicit ContigRenamer(const std::string& prefix = "c", int zero_pad = 12);

    // Generate anvi'o-compliant name for a contig
    std::string rename(const std::string& original_name, size_t length);

    // Get all rename mappings
    const std::vector<RenameEntry>& get_mappings() const { return mappings_; }

    // Write rename report TSV
    void write_report(const std::string& path) const;

    // For split contigs (chimera splitting), use suffix like _1, _2
    std::string rename_split(const std::string& parent_new_name, int split_index);

private:
    std::string prefix_;
    int zero_pad_;
    int counter_;
    std::vector<RenameEntry> mappings_;
};

// Check if a name is anvi'o compliant (alphanumeric + underscore only)
bool is_anvio_compliant(const std::string& name);

// Check if all contigs in a file have anvi'o compliant names
bool check_all_anvio_compliant(const std::string& fasta_path);

} // namespace amber

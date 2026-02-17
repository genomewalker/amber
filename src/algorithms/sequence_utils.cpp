#include "sequence_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace amber {

// ============================================================================
// FASTA Reader Implementation
// ============================================================================

FASTAReader::FASTAReader(const std::string& path)
    : path_(path), gz_file_(nullptr), pipe_file_(nullptr), use_pigz_(false), has_next_line_(false) {

    // Check if gzipped
    is_gzipped_ = (path.size() > 3 && path.substr(path.size() - 3) == ".gz");

    if (is_gzipped_) {
        // Try pigz first for parallel decompression, fallback to gzip
        std::string cmd = "pigz -dc '" + path + "' 2>/dev/null || gzip -dc '" + path + "'";
        pipe_file_ = popen(cmd.c_str(), "r");
        if (pipe_file_) {
            use_pigz_ = true;
            // Set large buffer for pipe reading
            setvbuf(pipe_file_, nullptr, _IOFBF, 262144);  // 256KB
        } else {
            throw std::runtime_error("Failed to decompress FASTA file: " + path);
        }
    } else {
        // Regular file - use gzFile (works for uncompressed too)
        gz_file_ = gzopen(path.c_str(), "r");
        if (!gz_file_) {
            throw std::runtime_error("Failed to open FASTA file: " + path);
        }
        gzbuffer(gz_file_, 262144);  // 256KB
    }
}

FASTAReader::~FASTAReader() {
    if (pipe_file_) {
        pclose(pipe_file_);
    }
    if (gz_file_) {
        gzclose(gz_file_);
    }
}

bool FASTAReader::next(FASTASequence& seq) {
    seq.id.clear();
    seq.sequence.clear();
    seq.description.clear();

    bool in_sequence = false;
    char* line_result = nullptr;

    while (true) {
        // Check if we have a buffered line from previous call
        if (has_next_line_) {
            strncpy(buffer_, next_line_.c_str(), sizeof(buffer_) - 1);
            buffer_[sizeof(buffer_) - 1] = '\0';
            has_next_line_ = false;
            line_result = buffer_;
        } else {
            // Read from appropriate source
            if (use_pigz_) {
                line_result = fgets(buffer_, sizeof(buffer_), pipe_file_);
            } else {
                line_result = gzgets(gz_file_, buffer_, sizeof(buffer_));
            }
        }

        if (!line_result) break;

        size_t len = strlen(buffer_);

        // Remove newline
        if (len > 0 && buffer_[len - 1] == '\n') {
            buffer_[--len] = '\0';
        }
        if (len > 0 && buffer_[len - 1] == '\r') {
            buffer_[--len] = '\0';
        }

        if (len == 0) continue;

        if (buffer_[0] == '>') {
            // New sequence header
            if (in_sequence) {
                // Return previous sequence, save current line for next call
                if (use_pigz_) {
                    // For pigz, buffer this line for next call
                    next_line_ = std::string(buffer_, len);
                    has_next_line_ = true;
                } else {
                    // For gzFile, rewind to re-read this line
                    gzseek(gz_file_, -(len + 1), SEEK_CUR);
                }
                return true;
            }

            // Parse header
            in_sequence = true;
            std::string header(buffer_ + 1);

            size_t space_pos = header.find_first_of(" \t");
            if (space_pos != std::string::npos) {
                seq.id = header.substr(0, space_pos);
                seq.description = header.substr(space_pos + 1);
            } else {
                seq.id = header;
            }
        } else if (in_sequence) {
            // Sequence data
            seq.sequence.append(buffer_, len);
        }
    }

    return in_sequence && !seq.sequence.empty();
}

void FASTAReader::reset() {
    if (use_pigz_) {
        // Can't reset pipe - would need to recreate
        throw std::runtime_error("Cannot reset FASTAReader using pigz pipe");
    } else {
        gzrewind(gz_file_);
    }
}

// ============================================================================
// FASTA Writer
// ============================================================================

void write_fasta(const std::string& path, const std::vector<Fragment>& fragments) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    for (const auto& frag : fragments) {
        out << ">" << frag.id << "\n";

        // Write sequence in 80-character lines
        for (size_t i = 0; i < frag.sequence.size(); i += 80) {
            out << frag.sequence.substr(i, 80) << "\n";
        }
    }
}

void write_fasta_single(const std::string& path, const Fragment& fragment) {
    std::vector<Fragment> frags = {fragment};
    write_fasta(path, frags);
}

std::vector<FASTASequence> read_longest_contigs(
    const std::string& genome_file,
    size_t min_contig_length,
    size_t max_contigs) {

    std::vector<FASTASequence> contigs;

    // Read all contigs from genome file
    FASTAReader reader(genome_file);
    FASTASequence seq;

    while (reader.next(seq)) {
        // Filter by minimum length
        if (min_contig_length > 0 && seq.sequence.size() < min_contig_length) {
            continue;  // Skip short contigs (likely contamination)
        }
        contigs.push_back(seq);
    }

    // Sort by length (descending - longest first)
    std::sort(contigs.begin(), contigs.end(),
        [](const FASTASequence& a, const FASTASequence& b) {
            return a.sequence.size() > b.sequence.size();
        });

    // Limit to top N contigs if specified
    if (max_contigs > 0 && contigs.size() > max_contigs) {
        contigs.resize(max_contigs);
    }

    return contigs;
}

size_t get_max_contig_length(const std::string& genome_file) {
    size_t max_length = 0;

    try {
        FASTAReader reader(genome_file);
        FASTASequence seq;

        while (reader.next(seq)) {
            if (seq.sequence.size() > max_length) {
                max_length = seq.sequence.size();
            }
        }
    } catch (const std::exception&) {
        // File not found or cannot be read - return 0
        return 0;
    }

    return max_length;
}

std::vector<FASTASequence> stitch_contigs_with_buffer(
    const std::string& genome_file,
    size_t buffer_ns,
    size_t reps,
    bool shuffle,
    uint64_t seed) {

    std::vector<FASTASequence> contigs;
    try {
        FASTAReader reader(genome_file);
        FASTASequence seq;
        while (reader.next(seq)) {
            contigs.push_back(seq);
        }
    } catch (const std::exception&) {
        return {};  // failed to read
    }

    if (contigs.empty()) return {};

    std::vector<FASTASequence> stitched;
    reps = std::max<size_t>(1, reps);

    for (size_t rep = 0; rep < reps; ++rep) {
        std::vector<size_t> idx(contigs.size());
        std::iota(idx.begin(), idx.end(), 0);

        if (shuffle) {
            std::mt19937_64 rng(seed + rep);
            std::shuffle(idx.begin(), idx.end(), rng);
        }

        std::string buffer(buffer_ns, 'N');
        std::string combined;
        size_t total_bases = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            if (i > 0) combined += buffer;
            combined += contigs[idx[i]].sequence;
            total_bases += contigs[idx[i]].sequence.size();
        }

        FASTASequence stitched_seq;
        stitched_seq.id = reps > 1 ? (genome_file + ":pseudo_rep" + std::to_string(rep))
                                    : (genome_file + ":pseudo");
        stitched_seq.sequence = std::move(combined);

        stitched.push_back(std::move(stitched_seq));
    }

    return stitched;
}

// ============================================================================
// Sequence Manipulator Implementation
// ============================================================================

SequenceManipulator::SequenceManipulator(uint64_t seed)
    : rng_(seed), uniform_(0.0, 1.0) {
}

std::vector<Fragment> SequenceManipulator::fragment_random(
    const FASTASequence& genome,
    const std::string& taxonomy,
    size_t fragment_length,
    size_t num_fragments,
    size_t min_length,
    double length_stddev,
    const std::string& dist_type,
    double shape,
    const std::string& genome_accession) {

    std::vector<Fragment> fragments;

    // Silently skip contigs shorter than fragment length (common for multi-contig genomes!)
    if (genome.sequence.size() < fragment_length) {
        return fragments;  // Empty vector, no warning spam
    }

    size_t attempts = 0;
    const size_t max_attempts = num_fragments * 10;  // Avoid infinite loop
    // If genome is just long enough that unique start positions < requested fragments,
    // downscale the requested count to the number of possible starts.
    size_t min_unique_starts = 0;
    if (genome.sequence.size() >= fragment_length) {
        min_unique_starts = genome.sequence.size() - fragment_length + 1;
        if (num_fragments > min_unique_starts) {
            num_fragments = min_unique_starts;
        }
    }

    for (size_t i = 0; i < num_fragments && attempts < max_attempts; ++attempts) {
        // Sample fragment length from distribution
        size_t actual_length = fragment_length;

        if (length_stddev > 0.0 || dist_type != "normal") {
            double sampled_length;

            if (dist_type == "lognormal") {
                // Log-normal: most realistic for contig lengths
                // Convert from desired linear mean/stddev to lognormal parameters
                // For lognormal: if X ~ lognormal(μ, σ), then E[X] = exp(μ + σ²/2)
                // We want E[X] ≈ fragment_length, so solve for μ and σ

                // Use smaller σ for reasonable variance (σ on log scale should be ~0.5-1.5)
                // If user provides large length_stddev, scale it down appropriately
                double sigma_log;
                if (length_stddev > 10.0) {
                    // User likely provided linear stddev - convert to log scale
                    // CV (coefficient of variation) = stddev / mean
                    double cv = length_stddev / fragment_length;
                    // For lognormal: CV² = exp(σ²) - 1, so σ = sqrt(log(1 + CV²))
                    sigma_log = std::sqrt(std::log(1.0 + cv * cv));
                    // Cap at reasonable value
                    sigma_log = std::min(sigma_log, 1.5);
                } else {
                    // User provided σ directly (for log scale)
                    sigma_log = length_stddev;
                }

                // Calculate μ such that E[X] = fragment_length
                // E[X] = exp(μ + σ²/2), so μ = log(fragment_length) - σ²/2
                double mu_log = std::log(fragment_length) - (sigma_log * sigma_log) / 2.0;

                std::lognormal_distribution<double> dist(mu_log, sigma_log);
                sampled_length = dist(rng_);
            } else if (dist_type == "gamma") {
                // Gamma distribution: shape α, scale β = mean/α
                double scale = fragment_length / shape;
                std::gamma_distribution<double> dist(shape, scale);
                sampled_length = dist(rng_);
            } else if (dist_type == "exponential") {
                // Exponential: rate λ = 1/mean
                std::exponential_distribution<double> dist(1.0 / fragment_length);
                sampled_length = dist(rng_);
            } else if (dist_type == "powerlaw") {
                // Power-law: P(x) ∝ x^(-α), use inverse transform sampling
                std::uniform_real_distribution<double> unif(0, 1);
                double u = unif(rng_);
                // Pareto distribution with x_min = fragment_length/2, α = shape
                sampled_length = (fragment_length / 2.0) * std::pow(1.0 - u, -1.0 / shape);
            } else {
                // Normal distribution (default, for backward compatibility)
                std::normal_distribution<double> dist(fragment_length, length_stddev);
                sampled_length = dist(rng_);
            }

            // Ensure reasonable bounds: min_length (or half target) to genome size
            size_t effective_min = (min_length > 0) ? min_length : (fragment_length / 2);
            actual_length = static_cast<size_t>(std::max(static_cast<double>(effective_min), sampled_length));
            actual_length = std::min(actual_length, genome.sequence.size());
        }

        // CRITICAL: Skip if actual_length is 0 or too small
        // This prevents 0-length fragments from being written to metadata
        size_t absolute_min = std::max(size_t(100), fragment_length / 10);  // At least 100bp or 10% of target
        if (actual_length < absolute_min) {
            continue;  // Retry with different length
        }

        // Skip if actual_length exceeds genome
        if (actual_length > genome.sequence.size()) {
            continue;
        }

        // Sample random start position
        size_t max_start = genome.sequence.size() - actual_length;
        std::uniform_int_distribution<size_t> start_dist(0, max_start);
        size_t start = start_dist(rng_);

        Fragment frag;
        frag.id = genome.id + "_frag_" + std::to_string(i);
        frag.sequence = genome.sequence.substr(start, actual_length);
        frag.source_genome = genome_accession.empty() ? genome.id : genome_accession;
        frag.taxonomy = taxonomy;
        frag.source_start = start;
        frag.source_end = start + actual_length;
        frag.is_chimera = false;

        // Skip if too many N's
        if (count_n_bases(frag.sequence) > actual_length * 0.1) {
            continue;  // Retry (don't increment i)
        }

        fragments.push_back(frag);
        ++i;  // Only increment on success
    }

    return fragments;
}

std::vector<Fragment> SequenceManipulator::fragment_sliding_window(
    const FASTASequence& genome,
    const std::string& taxonomy,
    size_t window_size,
    size_t step_size,
    const std::string& genome_accession) {

    std::vector<Fragment> fragments;

    for (size_t start = 0; start + window_size <= genome.sequence.size(); start += step_size) {
        Fragment frag;
        frag.id = genome.id + "_win_" + std::to_string(start);
        frag.sequence = genome.sequence.substr(start, window_size);
        frag.source_genome = genome_accession.empty() ? genome.id : genome_accession;
        frag.taxonomy = taxonomy;
        frag.source_start = start;
        frag.source_end = start + window_size;
        frag.is_chimera = false;

        // Skip if too many N's
        if (count_n_bases(frag.sequence) < window_size * 0.1) {
            fragments.push_back(frag);
        }
    }

    return fragments;
}

std::vector<Fragment> SequenceManipulator::fragment_coverage_weighted(
    const FASTASequence& genome,
    const std::string& taxonomy,
    size_t fragment_length,
    size_t num_fragments,
    const std::vector<double>& coverage_profile,
    const std::string& genome_accession) {

    // Simple implementation: use coverage as sampling weights
    std::vector<Fragment> fragments;

    if (coverage_profile.empty() || genome.sequence.size() < fragment_length) {
        return fragment_random(genome, taxonomy, fragment_length, num_fragments);
    }

    // Create cumulative distribution
    std::vector<double> cumulative(coverage_profile.size());
    cumulative[0] = coverage_profile[0];
    for (size_t i = 1; i < coverage_profile.size(); ++i) {
        cumulative[i] = cumulative[i-1] + coverage_profile[i];
    }

    double total = cumulative.back();
    std::uniform_real_distribution<double> sample_dist(0.0, total);

    for (size_t i = 0; i < num_fragments; ++i) {
        double r = sample_dist(rng_);
        auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);
        size_t start = std::distance(cumulative.begin(), it);

        // Adjust to valid range
        if (start + fragment_length > genome.sequence.size()) {
            start = genome.sequence.size() - fragment_length;
        }

        Fragment frag;
        frag.id = genome.id + "_cov_frag_" + std::to_string(i);
        frag.sequence = genome.sequence.substr(start, fragment_length);
        frag.source_genome = genome_accession.empty() ? genome.id : genome_accession;
        frag.taxonomy = taxonomy;
        frag.source_start = start;
        frag.source_end = start + fragment_length;
        frag.is_chimera = false;

        if (count_n_bases(frag.sequence) < fragment_length * 0.1) {
            fragments.push_back(frag);
        }
    }

    return fragments;
}

// ============================================================================
// Mutation Functions
// ============================================================================

char SequenceManipulator::complement(char base) {
    switch (base) {
        case 'A': case 'a': return 'T';
        case 'T': case 't': return 'A';
        case 'G': case 'g': return 'C';
        case 'C': case 'c': return 'G';
        default: return 'N';
    }
}

std::string SequenceManipulator::reverse_complement(const std::string& seq) {
    std::string rc;
    rc.reserve(seq.size());
    for (auto it = seq.rbegin(); it != seq.rend(); ++it) {
        rc.push_back(complement(*it));
    }
    return rc;
}

char SequenceManipulator::mutate_base(char base, bool transition_bias) {
    static const char bases[] = {'A', 'C', 'G', 'T'};

    if (transition_bias && uniform_(rng_) < 0.67) {
        // Transition (A<->G, C<->T)
        switch (base) {
            case 'A': case 'a': return 'G';
            case 'G': case 'g': return 'A';
            case 'C': case 'c': return 'T';
            case 'T': case 't': return 'C';
            default: return base;
        }
    } else {
        // Transversion or no bias
        std::vector<char> alternatives;
        for (char b : bases) {
            if (toupper(b) != toupper(base)) {
                alternatives.push_back(b);
            }
        }
        if (alternatives.empty()) return base;

        std::uniform_int_distribution<size_t> alt_dist(0, alternatives.size() - 1);
        return alternatives[alt_dist(rng_)];
    }
}

char SequenceManipulator::random_base() {
    static const char bases[] = {'A', 'C', 'G', 'T'};
    std::uniform_int_distribution<int> base_dist(0, 3);
    return bases[base_dist(rng_)];
}

size_t SequenceManipulator::sample_indel_length() {
    // Geometric distribution with mean ~3bp
    std::geometric_distribution<size_t> geom(0.33);
    return std::min(geom(rng_) + 1, size_t(20));
}

std::string SequenceManipulator::introduce_snps(
    const std::string& sequence,
    double snp_rate,
    double transition_transversion_ratio) {

    std::string mutated = sequence;
    bool transition_bias = (transition_transversion_ratio > 1.0);

    for (size_t i = 0; i < mutated.size(); ++i) {
        if (uniform_(rng_) < snp_rate) {
            mutated[i] = mutate_base(mutated[i], transition_bias);
        }
    }

    return mutated;
}

std::string SequenceManipulator::introduce_indels(
    const std::string& sequence,
    double indel_rate,
    double insert_delete_ratio,
    size_t max_indel_length) {

    std::string result;
    result.reserve(sequence.size() * 1.2); // Assume some insertions

    for (size_t i = 0; i < sequence.size(); ++i) {
        if (uniform_(rng_) < indel_rate) {
            double ins_del = uniform_(rng_);
            size_t indel_len = sample_indel_length();
            indel_len = std::min(indel_len, max_indel_length);

            if (ins_del < insert_delete_ratio / (1.0 + insert_delete_ratio)) {
                // Insertion
                for (size_t j = 0; j < indel_len; ++j) {
                    result.push_back(random_base());
                }
            } else {
                // Deletion - skip bases
                i += indel_len - 1;
                if (i >= sequence.size()) break;
            }
        }

        if (i < sequence.size()) {
            result.push_back(sequence[i]);
        }
    }

    return result;
}

std::string SequenceManipulator::introduce_divergence(
    const std::string& sequence,
    double divergence,
    bool include_indels) {

    // Split divergence between SNPs and indels
    double snp_rate = include_indels ? divergence * 0.85 : divergence;
    double indel_rate = include_indels ? divergence * 0.15 / sequence.size() : 0.0;

    std::string result = introduce_snps(sequence, snp_rate, 2.0);

    if (include_indels) {
        result = introduce_indels(result, indel_rate, 1.0, 10);
    }

    return result;
}

// ============================================================================
// Chimera Creation
// ============================================================================

Fragment SequenceManipulator::create_chimera(
    const std::vector<FASTASequence>& genomes,
    const std::vector<std::string>& taxonomies,
    const std::vector<double>& ratios,
    size_t total_length) {

    Fragment chimera;
    chimera.is_chimera = true;
    chimera.id = "chimera";

    if (genomes.empty() || genomes.size() != ratios.size()) {
        return chimera;
    }

    // Calculate segment lengths
    std::vector<size_t> lengths;
    size_t remaining = total_length;

    for (size_t i = 0; i < ratios.size(); ++i) {
        size_t len = (i == ratios.size() - 1) ? remaining :
                     static_cast<size_t>(total_length * ratios[i]);
        lengths.push_back(len);
        remaining -= len;
    }

    // Extract segments and concatenate
    for (size_t i = 0; i < genomes.size(); ++i) {
        const auto& genome = genomes[i];
        size_t seg_len = lengths[i];

        if (genome.sequence.size() < seg_len) {
            seg_len = genome.sequence.size();
        }

        // Random start position
        std::uniform_int_distribution<size_t> pos_dist(0, genome.sequence.size() - seg_len);
        size_t start = pos_dist(rng_);

        std::string segment = genome.sequence.substr(start, seg_len);
        chimera.sequence += segment;
        chimera.chimera_sources.push_back(genome.id);
        chimera.chimera_breakpoints.push_back(chimera.sequence.size());

        // Use first genome as primary source
        if (i == 0) {
            chimera.source_genome = genome.id;
            chimera.taxonomy = taxonomies[i];
        }
    }

    return chimera;
}

// ============================================================================
// HGT Simulation
// ============================================================================

std::string SequenceManipulator::insert_hgt_region(
    const std::string& recipient,
    const std::string& donor_region,
    size_t insert_position) {

    if (insert_position > recipient.size()) {
        insert_position = recipient.size();
    }

    std::string result = recipient.substr(0, insert_position);
    result += donor_region;
    result += recipient.substr(insert_position);

    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

bool is_valid_dna(const std::string& seq) {
    for (char c : seq) {
        char upper = toupper(c);
        if (upper != 'A' && upper != 'C' && upper != 'G' && upper != 'T' && upper != 'N') {
            return false;
        }
    }
    return true;
}

double gc_content(const std::string& seq) {
    if (seq.empty()) return 0.0;

    size_t gc_count = 0;
    size_t total = 0;

    for (char c : seq) {
        char upper = toupper(c);
        if (upper == 'G' || upper == 'C') {
            ++gc_count;
            ++total;
        } else if (upper == 'A' || upper == 'T') {
            ++total;
        }
    }

    return total > 0 ? static_cast<double>(gc_count) / total : 0.0;
}

size_t count_n_bases(const std::string& seq) {
    return std::count_if(seq.begin(), seq.end(),
                        [](char c) { return toupper(c) == 'N'; });
}

// ============================================================================
// Anvi'o Contig Renaming
// ============================================================================

bool is_anvio_compliant(const std::string& name) {
    if (name.empty()) return false;

    for (char c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
            return false;
        }
    }
    return true;
}

bool check_all_anvio_compliant(const std::string& fasta_path) {
    FASTAReader reader(fasta_path);
    FASTASequence seq;

    while (reader.next(seq)) {
        if (!is_anvio_compliant(seq.id)) {
            return false;
        }
    }
    return true;
}

ContigRenamer::ContigRenamer(const std::string& prefix, int zero_pad)
    : prefix_(prefix), zero_pad_(zero_pad), counter_(0) {}

std::string ContigRenamer::rename(const std::string& original_name, size_t length) {
    ++counter_;

    // Anvi'o format: PREFIX_XXXXXXXXXXXX (underscore + 12 zero-padded digits)
    std::ostringstream oss;
    oss << prefix_ << "_" << std::setw(zero_pad_) << std::setfill('0') << counter_;
    std::string new_name = oss.str();

    RenameEntry entry;
    entry.original_name = original_name;
    entry.new_name = new_name;
    entry.length = length;
    mappings_.push_back(entry);

    return new_name;
}

std::string ContigRenamer::rename_split(const std::string& parent_new_name, int split_index) {
    std::ostringstream oss;
    oss << parent_new_name << "_" << split_index;
    return oss.str();
}

void ContigRenamer::write_report(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open rename report file: " + path);
    }

    out << "original_name\tnew_name\tlength\n";
    for (const auto& entry : mappings_) {
        out << entry.original_name << "\t" << entry.new_name << "\t" << entry.length << "\n";
    }
}

} // namespace amber

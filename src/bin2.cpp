// AMBER bin2 - COMEBin-style Contrastive Binner
// Uses self-supervised contrastive learning with substring augmentation
// Requires LibTorch for GPU acceleration

#include <amber/config.hpp>
#include "algorithms/kmer_features.h"
#include "algorithms/fractal_features.h"
#include "algorithms/sequence_utils.h"
#include "algorithms/damage_profile.h"
#include "algorithms/coverage_features.h"
#include "binner/internal/types.h"

#ifndef USE_LIBTORCH
#error "bin2 requires LibTorch. Build with -DAMBER_USE_TORCH=ON"
#endif

// COMEBin-style encoder
#include "bin2/encoder/encoder_types.h"
#include "bin2/encoder/comebin_encoder.h"
#include "bin2/encoder/substring_augmenter.h"
#include "bin2/encoder/damage_aware_infonce.h"
#include "bin2/encoder/multiscale_cgr.h"
#include "algorithms/damage_likelihood.h"

#include "bin2/clustering/clustering_types.h"
#include "bin2/clustering/hnsw_knn_index.h"
#include "bin2/clustering/edge_weighter.h"
#include "bin2/clustering/leiden_backend.h"
#include "bin2/clustering/ensemble_leiden.h"
#include "bin2/clustering/marker_index.h"
#include "bin2/clustering/quality_leiden_backend.h"
#include "seeds/seed_generator.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include <set>
#include <atomic>
#include <cctype>
#include <htslib/sam.h>
#include <htslib/faidx.h>

namespace amber {

struct Bin2Config {
    std::string contigs_path;
    std::string bam_path;
    std::string output_dir;
    std::string seed_file;         // SCG marker seed file
    std::string hmm_file;          // HMM file for auto seed generation
    std::string hmmsearch_path;    // Path to hmmsearch binary
    std::string fraggenescan_path; // Path to FragGeneScan binary
    std::string embeddings_file;   // Pre-computed embeddings TSV (skip training)
    int threads = 16;
    int min_length = 1001;         // COMEBin default (filters > 1000bp)
    int min_bin_size = 200000;     // 200kb minimum bin size
    int epochs = 100;
    int batch_size = 1024;         // COMEBin default
    int embedding_dim = 128;
    float learning_rate = 0.001f;
    float temperature = 0.1f;      // COMEBin default
    // Clustering parameters
    int k = 100;                   // kNN neighbors (COMEBin default)
    int partgraph_ratio = 50;      // Percentile cutoff for edge pruning
    float resolution = 1.0f;       // Leiden resolution
    float bandwidth = 0.1f;        // Edge weighting bandwidth
    // Flags
    bool l2_normalize = true;      // L2 normalize embeddings before clustering
    bool verbose = false;
    bool sweep = false;            // COMEBin-style parameter sweep
    bool use_python_leiden = false; // Use Python leidenalg (exact COMEBin match)
    bool force_cpu = false;        // Force CPU mode even if GPU is available
    int hidden_dim = 2048;         // Hidden layer size (COMEBin default: 2048)
    int n_layer = 3;               // Number of hidden layers (COMEBin default: 3)
    float dropout = 0.2f;          // Dropout rate (COMEBin default: 0.2)
    int random_seed = 42;          // Random seed for Leiden clustering
    int encoder_seed = 42;         // Random seed for encoder training (optimal)
    // Damage-aware features (experimental)
    bool use_damage_infonce = false;   // Use damage-weighted InfoNCE loss
    float damage_lambda = 0.5f;        // Damage attenuation strength (0.4-0.6)
    float damage_wmin = 0.5f;          // Minimum negative weight (keep graph connected)
    bool use_multiscale_cgr = false;   // Use multi-scale CGR late fusion
    bool use_ensemble_leiden = false;  // Use ensemble Leiden for reproducible clustering
    int n_ensemble_runs = 10;          // Number of Leiden runs for ensemble
    float ensemble_threshold = 0.5f;   // Co-occurrence threshold for consensus
    bool ensemble_vary_resolution = false;
    float ensemble_res_min = 1.0f;
    float ensemble_res_max = 20.0f;
    bool use_quality_leiden = false;   // Use marker-quality-guided Leiden refinement
    bool use_map_equation = false;     // Phase 1b: marker-guided map equation refinement
    float quality_alpha = 1.0f;        // Quality weight (0=modularity only, 1=balanced)
    int n_leiden_restarts = 1;         // Best-of-K joint (res × seed) restart search (1=off)
    // CheckM marker files for proper colocation-weighted quality in QualityLeiden
    std::string checkm_hmm_file;       // HMM for CheckM markers (default: auxiliary/checkm_markers_only.hmm)
    std::string bacteria_ms_file;      // CheckM bacteria marker sets (default: scripts/checkm_ms/bacteria.ms)
    std::string archaea_ms_file;       // CheckM archaea marker sets (default: scripts/checkm_ms/archaea.ms)
};

// Load seed marker contig names from TSV (contig\tcluster) or one-per-line format
std::vector<std::string> load_seed_markers(const std::string& path) {
    std::vector<std::string> seeds;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        while (!line.empty() && std::isspace(line.back())) line.pop_back();
        if (line.empty()) continue;
        // Handle TSV format: extract first column (contig name)
        auto tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            seeds.push_back(line.substr(0, tab_pos));
        } else {
            seeds.push_back(line);
        }
    }
    return seeds;
}

// Call Python leidenalg for exact COMEBin matching
bin2::ClusteringResult run_python_leiden(
    const std::vector<bin2::WeightedEdge>& edges,
    int n_nodes,
    const std::vector<double>& node_sizes) {

    // Create temp file for input
    std::string tmp_input = "/tmp/amber_leiden_input_" + std::to_string(getpid()) + ".txt";
    std::string tmp_output = "/tmp/amber_leiden_output_" + std::to_string(getpid()) + ".txt";

    {
        std::ofstream f(tmp_input);
        f << n_nodes << "\n";
        f << edges.size() << "\n";
        for (const auto& e : edges) {
            f << e.u << " " << e.v << " " << std::setprecision(15) << e.w << "\n";
        }
        for (int i = 0; i < n_nodes; i++) {
            f << static_cast<int64_t>(node_sizes[i]) << "\n";
        }
    }

    // Find the Python script (relative to executable or in .scripts/)
    std::string script_path = ".scripts/leiden_cluster.py";
    if (!std::filesystem::exists(script_path)) {
        // Try relative to this source file's location
        script_path = "/maps/projects/caeg/people/kbd606/scratch/kapk-assm/amber/.scripts/leiden_cluster.py";
    }

    // Call Python with conda environment
    std::string cmd = "source /maps/projects/fernandezguerra/apps/opt/conda/etc/profile.d/conda.sh && "
                      "conda activate comebin && "
                      "cat " + tmp_input + " | python3 " + script_path + " > " + tmp_output + " 2>&1";

    int ret = system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Warning: Python leiden failed with code " << ret << std::endl;
    }

    // Read results
    bin2::ClusteringResult result;
    result.labels.resize(n_nodes, -1);

    std::ifstream f(tmp_output);
    std::string line;
    int max_cluster = -1;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        int node_id, cluster_id;
        if (iss >> node_id >> cluster_id) {
            if (node_id >= 0 && node_id < n_nodes) {
                result.labels[node_id] = cluster_id;
                max_cluster = std::max(max_cluster, cluster_id);
            }
        }
    }

    result.num_clusters = max_cluster + 1;
    result.modularity = 0.0f;  // Not computed by wrapper
    result.num_iterations = 1;

    // Clean up temp files
    std::filesystem::remove(tmp_input);
    std::filesystem::remove(tmp_output);

    return result;
}

class Bin2Logger {
    bool verbose_;
    std::chrono::steady_clock::time_point start_;
public:
    Bin2Logger(bool v) : verbose_(v), start_(std::chrono::steady_clock::now()) {}

    void info(const std::string& msg) {
        auto now = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_).count();
        std::cerr << "[" << ms << "ms] " << msg << std::endl;
    }
};

std::vector<std::pair<std::string, std::string>> load_contigs_simple(
    const std::string& path, int min_length) {

    std::vector<std::pair<std::string, std::string>> contigs;
    std::ifstream file(path);
    std::string line, name, seq;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            if (!name.empty() && static_cast<int>(seq.length()) >= min_length) {
                contigs.emplace_back(name, seq);
            }
            name = line.substr(1);
            auto space = name.find(' ');
            if (space != std::string::npos) name = name.substr(0, space);
            seq.clear();
        } else {
            seq += line;
        }
    }
    if (!name.empty() && static_cast<int>(seq.length()) >= min_length) {
        contigs.emplace_back(name, seq);
    }

    return contigs;
}

// Coverage result with mean, variance, and optional per-position depth
struct CoverageWithVariance {
    double mean = 0.0;
    double variance = 0.0;  // Population variance (sqrt used for COMEBin features)
    std::vector<uint32_t> per_position_depth;  // Per-position depth for substring views
};

// COMEBin-style coverage extraction (like bedtools genomecov -bga)
// NO identity filter, NO edge exclusion - counts ALL aligned reads
// Much faster than MetaBAT2-style and matches COMEBin exactly
std::unordered_map<std::string, CoverageWithVariance> load_coverage_comebin_style(
    const std::string& bam_path,
    const std::vector<std::pair<std::string, std::string>>& contigs,
    int threads) {

    std::unordered_map<std::string, CoverageWithVariance> coverage;
    coverage.reserve(contigs.size());

    // Pre-allocate results
    std::vector<CoverageWithVariance> results(contigs.size());

    // Each thread needs its own BAM handle for thread safety
    #pragma omp parallel num_threads(threads)
    {
        htsFile* bam = hts_open(bam_path.c_str(), "r");
        bam_hdr_t* header = bam ? sam_hdr_read(bam) : nullptr;
        hts_idx_t* idx = bam ? sam_index_load(bam, bam_path.c_str()) : nullptr;
        bam1_t* aln = bam_init1();

        if (bam && header && idx) {
            #pragma omp for schedule(dynamic)
            for (size_t ci = 0; ci < contigs.size(); ci++) {
                const auto& [name, seq] = contigs[ci];
                uint32_t contig_length = static_cast<uint32_t>(seq.length());

                int tid = bam_name2id(header, name.c_str());
                if (tid < 0) continue;

                // Allocate full depth array (like bedtools genomecov)
                std::vector<uint32_t> depth(contig_length, 0);

                hts_itr_t* iter = sam_itr_queryi(idx, tid, 0, contig_length);
                if (!iter) continue;

                while (sam_itr_next(bam, iter, aln) >= 0) {
                    // bedtools genomecov INCLUDES secondary alignments!
                    // Only skip unmapped and supplementary (chimeric splits)
                    // NO identity filter, NO MAPQ filter - matches bedtools genomecov -bga
                    if (aln->core.flag & (BAM_FUNMAP | BAM_FSUPPLEMENTARY))
                        continue;

                    // Parse CIGAR, accumulate coverage
                    uint32_t* cigar = bam_get_cigar(aln);
                    uint32_t n_cigar = aln->core.n_cigar;
                    uint32_t ref_pos = aln->core.pos;

                    for (uint32_t i = 0; i < n_cigar; ++i) {
                        int op = bam_cigar_op(cigar[i]);
                        uint32_t len = bam_cigar_oplen(cigar[i]);

                        if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                            uint32_t cov_end = std::min(ref_pos + len, contig_length);
                            for (uint32_t pos = ref_pos; pos < cov_end; ++pos) {
                                depth[pos]++;
                            }
                            ref_pos += len;
                        } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
                            ref_pos += len;
                        }
                    }
                }
                hts_itr_destroy(iter);

                // Compute mean and variance (population variance, N divisor)
                double sum = 0.0, sum_sq = 0.0;
                for (uint32_t d : depth) {
                    sum += d;
                    sum_sq += static_cast<double>(d) * d;
                }
                double mean = sum / contig_length;
                double variance = (sum_sq / contig_length) - (mean * mean);

                results[ci].mean = mean;
                results[ci].variance = variance;
                results[ci].per_position_depth = std::move(depth);
            }
        }

        bam_destroy1(aln);
        if (idx) hts_idx_destroy(idx);
        if (header) bam_hdr_destroy(header);
        if (bam) hts_close(bam);
    }

    // Build map from results
    for (size_t ci = 0; ci < contigs.size(); ci++) {
        if (!results[ci].per_position_depth.empty()) {
            coverage[contigs[ci].first] = std::move(results[ci]);
        }
    }

    return coverage;
}

// aDNA feature accumulator for per-contig statistics
struct ADNAAccumulator {
    // Fragment length accumulator
    double frag_sum = 0.0;
    double frag_sum_sq = 0.0;
    std::vector<int> frag_lengths;  // For median

    // Damage-stratified coverage
    double cov_damaged = 0.0;      // Reads with p_anc > 0.6
    double cov_undamaged = 0.0;    // Reads with p_anc < 0.4

    // Mismatch spectrum counts at 5' end
    int mm_5p_tc = 0, mm_5p_ag = 0, mm_5p_other = 0, mm_5p_total = 0;
    // Mismatch spectrum counts at 3' end
    int mm_3p_ct = 0, mm_3p_tc = 0, mm_3p_other = 0, mm_3p_total = 0;

    // Read counts
    int n_reads = 0;
    int n_damaged = 0;
    int n_undamaged = 0;

    void add_read(int read_len, float p_damaged) {
        n_reads++;
        frag_sum += read_len;
        frag_sum_sq += (double)read_len * read_len;
        frag_lengths.push_back(read_len);

        if (p_damaged > 0.6f) {
            cov_damaged += read_len;
            n_damaged++;
        } else if (p_damaged < 0.4f) {
            cov_undamaged += read_len;
            n_undamaged++;
        }
    }

    void add_5p_mismatch(char ref, char read) {
        mm_5p_total++;
        if (ref == 'T' && read == 'C') mm_5p_tc++;
        else if (ref == 'A' && read == 'G') mm_5p_ag++;
        else if (ref != read) mm_5p_other++;
    }

    void add_3p_mismatch(char ref, char read) {
        mm_3p_total++;
        if (ref == 'C' && read == 'T') mm_3p_ct++;
        else if (ref == 'T' && read == 'C') mm_3p_tc++;
        else if (ref != read) mm_3p_other++;
    }

    void finalize(bin2::DamageProfile& profile, float contig_len) {
        if (n_reads > 0) {
            profile.frag_len_mean = (float)(frag_sum / n_reads);
            double var = (frag_sum_sq / n_reads) - (profile.frag_len_mean * profile.frag_len_mean);
            profile.frag_len_std = (float)std::sqrt(std::max(0.0, var));

            // Compute median
            if (!frag_lengths.empty()) {
                std::sort(frag_lengths.begin(), frag_lengths.end());
                profile.frag_len_median = (float)frag_lengths[frag_lengths.size() / 2];
            }
        }

        profile.cov_damaged_reads = (float)(cov_damaged / std::max(1.0f, contig_len));
        profile.cov_undamaged_reads = (float)(cov_undamaged / std::max(1.0f, contig_len));
        profile.damage_cov_ratio = (profile.cov_damaged_reads + 0.01f) / (profile.cov_undamaged_reads + 0.01f);

        // Mismatch spectrum (normalized rates)
        if (mm_5p_total > 10) {
            profile.mm_5p_tc = (float)mm_5p_tc / mm_5p_total;
            profile.mm_5p_ag = (float)mm_5p_ag / mm_5p_total;
            profile.mm_5p_other = (float)mm_5p_other / mm_5p_total;
        }
        if (mm_3p_total > 10) {
            profile.mm_3p_ct = (float)mm_3p_ct / mm_3p_total;
            profile.mm_3p_tc = (float)mm_3p_tc / mm_3p_total;
            profile.mm_3p_other = (float)mm_3p_other / mm_3p_total;
        }

        profile.n_reads = n_reads;
        profile.n_damaged_reads = n_damaged;
        profile.n_undamaged_reads = n_undamaged;
    }
};

// Fit exponential decay to damage rates: rate[i] = amp * exp(-lambda * i) + baseline
// Returns (lambda, amplitude) using simple linear regression on log(rate - baseline)
std::pair<float, float> fit_damage_decay(const float* rates, int n_positions, float baseline = 0.01f) {
    // Use first 5 positions for fitting
    int n = std::min(5, n_positions);
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    int valid = 0;

    for (int i = 0; i < n; i++) {
        float r = rates[i] - baseline;
        if (r > 0.001f) {
            double x = i;
            double y = std::log(r);
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
            valid++;
        }
    }

    if (valid < 2) return {0.5f, 0.1f};  // Default values

    double n_d = valid;
    double denom = n_d * sum_xx - sum_x * sum_x;
    if (std::fabs(denom) < 1e-10) return {0.5f, 0.1f};

    double slope = (n_d * sum_xy - sum_x * sum_y) / denom;
    double intercept = (sum_y - slope * sum_x) / n_d;

    float lambda = (float)(-slope);  // Negative because decay
    float amplitude = (float)std::exp(intercept);

    // Clamp to reasonable values
    lambda = std::clamp(lambda, 0.1f, 2.0f);
    amplitude = std::clamp(amplitude, 0.01f, 0.5f);

    return {lambda, amplitude};
}

// Extract comprehensive aDNA damage profiles from BAM
// AMBER-specific: includes fragment length, damage-stratified coverage, mismatch spectrum
std::unordered_map<std::string, bin2::DamageProfile> extract_damage_profiles(
    const std::string& bam_path,
    const std::vector<std::pair<std::string, std::string>>& contigs,
    int threads,
    int n_terminal = 5) {

    // Damage model parameters (calibrated for ancient DNA)
    const DamageModelParams params = {
        .amplitude_5p = 0.3,
        .amplitude_3p = 0.3,
        .lambda = 0.5,
        .baseline = 0.01
    };

    std::vector<bin2::DamageAccumulator> accumulators(contigs.size());
    std::vector<ADNAAccumulator> adna_accumulators(contigs.size());
    std::vector<float> contig_lengths(contigs.size());

    // Per-position damage rates (for smiley plot)
    // [position 0-9][0=C_total, 1=C_to_T, 2=G_total, 3=G_to_A]
    std::vector<std::array<std::atomic<int>, 40>> position_stats(contigs.size());
    for (auto& ps : position_stats) {
        for (auto& a : ps) a.store(0);
    }

    for (size_t i = 0; i < contigs.size(); i++) {
        contig_lengths[i] = (float)contigs[i].second.length();
    }

    // Build contig name -> index map
    std::unordered_map<std::string, size_t> name_to_idx;
    for (size_t i = 0; i < contigs.size(); i++) {
        name_to_idx[contigs[i].first] = i;
    }

    #pragma omp parallel num_threads(threads)
    {
        htsFile* bam = hts_open(bam_path.c_str(), "r");
        bam_hdr_t* header = bam ? sam_hdr_read(bam) : nullptr;
        hts_idx_t* idx = bam ? sam_index_load(bam, bam_path.c_str()) : nullptr;
        bam1_t* aln = bam_init1();

        if (bam && header && idx) {
            #pragma omp for schedule(dynamic)
            for (size_t ci = 0; ci < contigs.size(); ci++) {
                const auto& [name, seq] = contigs[ci];
                uint32_t contig_length = seq.length();

                int tid = bam_name2id(header, name.c_str());
                if (tid < 0) continue;

                hts_itr_t* iter = sam_itr_queryi(idx, tid, 0, contig_length);
                if (!iter) continue;

                bin2::DamageAccumulator& acc = accumulators[ci];
                ADNAAccumulator& adna = adna_accumulators[ci];
                auto& pos_stats = position_stats[ci];

                while (sam_itr_next(bam, iter, aln) >= 0) {
                    if (aln->core.flag & (BAM_FUNMAP | BAM_FSUPPLEMENTARY | BAM_FSECONDARY))
                        continue;

                    int read_len = aln->core.l_qseq;
                    if (read_len < 2 * n_terminal) continue;

                    uint8_t* bam_seq = bam_get_seq(aln);
                    uint8_t* bam_qual = bam_get_qual(aln);
                    uint32_t ref_pos = aln->core.pos;

                    // Compute per-read damage probability using likelihood model
                    double log_lik_ancient = 0.0;
                    double log_lik_modern = 0.0;
                    int informative_positions = 0;

                    // Check 5' terminal positions
                    for (int i = 0; i < std::min(n_terminal, read_len); i++) {
                        if (ref_pos + i >= contig_length) break;
                        char ref_base = std::toupper(seq[ref_pos + i]);
                        char read_base = seq_nt16_str[bam_seqi(bam_seq, i)];
                        double q_err = phred_to_error_prob(bam_qual[i]);
                        int dist = i + 1;

                        // Track per-position C→T stats
                        if (i < 10 && ref_base == 'C') {
                            pos_stats[i * 4 + 0].fetch_add(1);
                            if (read_base == 'T') pos_stats[i * 4 + 1].fetch_add(1);
                        }

                        // Track mismatch spectrum at position 0
                        if (i == 0 && ref_base != 'N' && read_base != 'N') {
                            #pragma omp critical
                            adna.add_5p_mismatch(ref_base, read_base);
                        }

                        if (ref_base == 'C') {
                            double d = damage_rate_at_distance(dist, params.amplitude_5p, params.lambda, params.baseline);
                            informative_positions++;

                            if (read_base == 'T') {
                                log_lik_ancient += std::log(d * (1.0 - q_err) + (1.0 - d) * q_err / 3.0);
                                log_lik_modern += std::log(q_err / 3.0);
                            } else if (read_base == 'C') {
                                log_lik_ancient += std::log((1.0 - d) * (1.0 - q_err));
                                log_lik_modern += std::log(1.0 - q_err);
                            }
                        }
                    }

                    // Check 3' terminal positions
                    for (int i = 0; i < std::min(n_terminal, read_len); i++) {
                        int read_idx = read_len - 1 - i;
                        int ref_idx = ref_pos + read_len - 1 - i;
                        if (ref_idx >= (int)contig_length || ref_idx < 0) continue;
                        char ref_base = std::toupper(seq[ref_idx]);
                        char read_base = seq_nt16_str[bam_seqi(bam_seq, read_idx)];
                        double q_err = phred_to_error_prob(bam_qual[read_idx]);
                        int dist = i + 1;

                        // Track per-position G→A stats
                        if (i < 10 && ref_base == 'G') {
                            pos_stats[i * 4 + 2].fetch_add(1);
                            if (read_base == 'A') pos_stats[i * 4 + 3].fetch_add(1);
                        }

                        // Track mismatch spectrum at terminal position
                        if (i == 0 && ref_base != 'N' && read_base != 'N') {
                            #pragma omp critical
                            adna.add_3p_mismatch(ref_base, read_base);
                        }

                        if (ref_base == 'G') {
                            double d = damage_rate_at_distance(dist, params.amplitude_3p, params.lambda, params.baseline);
                            informative_positions++;

                            if (read_base == 'A') {
                                log_lik_ancient += std::log(d * (1.0 - q_err) + (1.0 - d) * q_err / 3.0);
                                log_lik_modern += std::log(q_err / 3.0);
                            } else if (read_base == 'G') {
                                log_lik_ancient += std::log((1.0 - d) * (1.0 - q_err));
                                log_lik_modern += std::log(1.0 - q_err);
                            }
                        }
                    }

                    // Convert log likelihood ratio to probability
                    float p_damaged = 0.5f;
                    if (informative_positions > 0) {
                        double log_diff = log_lik_ancient - log_lik_modern;
                        log_diff = std::max(-20.0, std::min(20.0, log_diff));
                        p_damaged = static_cast<float>(1.0 / (1.0 + std::exp(-log_diff)));
                    }

                    // Accumulate damage statistics
                    float weight = std::sqrt(static_cast<float>(informative_positions));
                    acc.add_read(p_damaged, weight);

                    // Accumulate aDNA-specific features
                    #pragma omp critical
                    adna.add_read(read_len, p_damaged);
                }
                hts_itr_destroy(iter);
            }
        }

        bam_destroy1(aln);
        if (idx) hts_idx_destroy(idx);
        if (header) bam_hdr_destroy(header);
        if (bam) hts_close(bam);
    }

    // Finalize profiles with all aDNA features
    std::unordered_map<std::string, bin2::DamageProfile> profiles;
    profiles.reserve(contigs.size());

    for (size_t ci = 0; ci < contigs.size(); ci++) {
        auto profile = accumulators[ci].finalize(contig_lengths[ci]);

        // Compute per-position damage rates (smiley plot data)
        const auto& ps = position_stats[ci];
        for (int i = 0; i < 10; i++) {
            int c_total = ps[i * 4 + 0].load();
            int ct_count = ps[i * 4 + 1].load();
            int g_total = ps[i * 4 + 2].load();
            int ga_count = ps[i * 4 + 3].load();

            profile.ct_rate_5p[i] = (ct_count + 0.5f) / (c_total + 1.0f);
            profile.ga_rate_3p[i] = (ga_count + 0.5f) / (g_total + 1.0f);
        }

        // Fit exponential decay to estimate lambda
        auto [lambda_5p, amp_5p] = fit_damage_decay(profile.ct_rate_5p, 10);
        auto [lambda_3p, amp_3p] = fit_damage_decay(profile.ga_rate_3p, 10);
        profile.lambda_5p = lambda_5p;
        profile.lambda_3p = lambda_3p;
        profile.amplitude_5p = amp_5p;
        profile.amplitude_3p = amp_3p;

        // Add aDNA-specific features
        adna_accumulators[ci].finalize(profile, contig_lengths[ci]);

        profiles[contigs[ci].first] = profile;
    }

    return profiles;
}

// Compute mean and variance from depth array for a given range [start, end)
std::pair<double, double> compute_range_coverage(
    const std::vector<uint32_t>& depth, int start, int end) {

    if (start < 0 || end <= start || end > static_cast<int>(depth.size())) {
        // Invalid range, return zeros
        return {0.0, 0.0};
    }

    int n = end - start;
    if (n == 0) return {0.0, 0.0};

    // Compute mean
    double sum = 0.0;
    for (int i = start; i < end; i++) {
        sum += depth[i];
    }
    double mean = sum / n;

    // Compute population variance (N divisor, matches COMEBin/numpy)
    double var_sum = 0.0;
    for (int i = start; i < end; i++) {
        double diff = depth[i] - mean;
        var_sum += diff * diff;
    }
    double variance = var_sum / n;

    return {mean, variance};
}

// HDBSCAN-style clustering on embeddings
std::vector<int> cluster_hdbscan(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<double>& coverages,
    int min_cluster_size = 5,
    float coverage_ratio_max = 3.0f) {

    int N = embeddings.size();
    int D = embeddings[0].size();
    std::vector<int> labels(N, -1);

    // Compute pairwise distances
    std::vector<std::vector<float>> dist(N, std::vector<float>(N, 0));
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            float d = 0;
            for (int k = 0; k < D; k++) {
                float diff = embeddings[i][k] - embeddings[j][k];
                d += diff * diff;
            }
            d = std::sqrt(d);

            // Add coverage penalty
            double cov_ratio = std::max(coverages[i], coverages[j]) /
                              std::max(1e-10, std::min(coverages[i], coverages[j]));
            if (cov_ratio > coverage_ratio_max) {
                d += 10.0f;  // Large penalty
            }

            dist[i][j] = dist[j][i] = d;
        }
    }

    // Compute core distances (distance to k-th nearest neighbor)
    int k = std::min(min_cluster_size, N - 1);
    std::vector<float> core_dist(N);
    for (int i = 0; i < N; i++) {
        std::vector<float> dists = dist[i];
        std::nth_element(dists.begin(), dists.begin() + k, dists.end());
        core_dist[i] = dists[k];
    }

    // Compute mutual reachability distance
    std::vector<std::vector<float>> mrd(N, std::vector<float>(N, 0));
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            float mr = std::max({core_dist[i], core_dist[j], dist[i][j]});
            mrd[i][j] = mrd[j][i] = mr;
        }
    }

    // Single-linkage clustering on mutual reachability graph
    std::vector<int> parent(N);
    std::iota(parent.begin(), parent.end(), 0);

    auto find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    auto unite = [&](int x, int y) {
        int px = find(x), py = find(y);
        if (px != py) parent[px] = py;
    };

    // Sort edges by weight
    std::vector<std::tuple<float, int, int>> edges;
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            edges.emplace_back(mrd[i][j], i, j);
        }
    }
    std::sort(edges.begin(), edges.end());

    // Build MST with adaptive threshold
    // Use percentile of edge weights for threshold
    size_t threshold_idx = std::min(edges.size() - 1, edges.size() * 3 / 4);  // 75th percentile
    float threshold = std::get<0>(edges[threshold_idx]);
    threshold = std::max(0.3f, std::min(threshold, 1.5f));  // Clamp to [0.3, 1.5]

    for (auto& [w, i, j] : edges) {
        if (w > threshold) break;
        unite(i, j);
    }

    // Extract clusters
    std::map<int, std::vector<int>> clusters;
    for (int i = 0; i < N; i++) {
        clusters[find(i)].push_back(i);
    }

    // Assign labels
    int cluster_id = 0;
    for (auto& [root, members] : clusters) {
        if (static_cast<int>(members.size()) >= min_cluster_size) {
            for (int m : members) {
                labels[m] = cluster_id;
            }
            cluster_id++;
        }
    }

    return labels;
}

// Load pre-computed embeddings from TSV (COMEBin format: contig\tdim0\tdim1\t...)
std::pair<std::vector<std::string>, std::vector<std::vector<float>>>
load_embeddings_tsv(const std::string& path) {
    std::vector<std::string> names;
    std::vector<std::vector<float>> embeddings;

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open embeddings file: " + path);
    }

    std::string line;
    bool first_line = true;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // Skip header row
        if (first_line) {
            first_line = false;
            // Check if this looks like a header (starts with tab, digit, or "contig")
            if (line[0] == '\t' || std::isdigit(line[0]) ||
                line.substr(0, 6) == "contig" || line.substr(0, 6) == "Contig") {
                continue;  // Skip header
            }
        }

        std::istringstream iss(line);
        std::string name;
        iss >> name;

        // Skip if name is empty or a number (header row)
        if (name.empty() || (name[0] >= '0' && name[0] <= '9' && name.find('_') == std::string::npos)) {
            continue;
        }

        names.push_back(name);

        std::vector<float> emb;
        float val;
        while (iss >> val) {
            emb.push_back(val);
        }
        embeddings.push_back(emb);
    }

    return {names, embeddings};
}

// Load contig lengths from FASTA file using htslib faidx
std::unordered_map<std::string, int64_t> load_contig_lengths(const std::string& fasta_path) {
    std::unordered_map<std::string, int64_t> lengths;

    faidx_t* fai = fai_load(fasta_path.c_str());
    if (!fai) {
        throw std::runtime_error("Cannot open FASTA index for: " + fasta_path);
    }

    int n_seqs = faidx_nseq(fai);
    for (int i = 0; i < n_seqs; i++) {
        const char* name = faidx_iseq(fai, i);
        int64_t len = faidx_seq_len(fai, name);
        lengths[name] = len;
    }

    fai_destroy(fai);
    return lengths;
}

int run_bin2(const Bin2Config& config) {
    std::cerr << "[DEBUG] run_bin2 started\n" << std::flush;
    Bin2Logger log(config.verbose);
    std::cerr << "[DEBUG] logger created\n" << std::flush;

    // Device selection with --cpu override
    bool use_gpu = torch::cuda::is_available() && !config.force_cpu;
    auto device = use_gpu ? torch::kCUDA : torch::kCPU;

    log.info("AMBER bin2 v" + std::string(VERSION));
    if (config.force_cpu) {
        log.info("Forced CPU mode (--cpu)");
    } else if (use_gpu) {
        log.info("CUDA available, using GPU");
    } else {
        log.info("CUDA not available, using CPU");
    }

    std::filesystem::create_directories(config.output_dir);

    // Clean old bin files from previous runs
    for (const auto& entry : std::filesystem::directory_iterator(config.output_dir)) {
        if (entry.path().extension() == ".fa" || entry.path().extension() == ".fasta") {
            std::filesystem::remove(entry.path());
        }
    }

    // ========================================
    // CLUSTERING-ONLY MODE (pre-computed embeddings)
    // ========================================
    std::cerr << "[DEBUG] Checking embeddings_file: '" << config.embeddings_file << "'\n" << std::flush;
    if (!config.embeddings_file.empty()) {
        std::cerr << "[DEBUG] Embeddings mode\n" << std::flush;
        log.info("Loading pre-computed embeddings from " + config.embeddings_file);
        std::cerr << "[DEBUG] Before loading\n" << std::flush;
        auto [emb_names, embeddings] = load_embeddings_tsv(config.embeddings_file);
        std::cerr << "[DEBUG] Loaded " << emb_names.size() << " embeddings\n" << std::flush;
        if (embeddings.empty() || embeddings[0].empty()) {
            std::cerr << "Error: No embeddings loaded or empty dimensions\n";
            return 1;
        }
        log.info("Loaded " + std::to_string(emb_names.size()) + " embeddings with " +
                 std::to_string(embeddings[0].size()) + " dimensions");

        // Build name-to-index map
        std::unordered_map<std::string, size_t> name_to_idx;
        for (size_t i = 0; i < emb_names.size(); i++) {
            name_to_idx[emb_names[i]] = i;
        }

        // L2 normalize if requested
        if (config.l2_normalize) {
            log.info("L2 normalizing embeddings");
            for (auto& emb : embeddings) {
                float norm = 0.0f;
                for (float v : emb) norm += v * v;
                norm = std::sqrt(norm);
                if (norm > 1e-10f) {
                    for (float& v : emb) v /= norm;
                }
            }
        }

        // Load contig lengths from FASTA if provided (for COMEBin-style node weighting)
        std::vector<double> contig_lengths(emb_names.size(), 1.0);
        bool use_node_sizes = false;
        if (!config.contigs_path.empty() && std::filesystem::exists(config.contigs_path)) {
            auto lengths_map = load_contig_lengths(config.contigs_path);
            int found = 0;
            for (size_t i = 0; i < emb_names.size(); i++) {
                auto it = lengths_map.find(emb_names[i]);
                if (it != lengths_map.end()) {
                    contig_lengths[i] = static_cast<double>(it->second);
                    found++;
                }
            }
            log.info("Loaded lengths for " + std::to_string(found) + "/" + std::to_string(emb_names.size()) +
                     " contigs from " + config.contigs_path);
            use_node_sizes = (found > 0);
        }

        // Load or auto-generate seed markers
        std::unordered_set<std::string> seed_set;
        std::string seed_path = config.seed_file;
        if (seed_path.empty()) {
            // Auto-generate seeds using SCG markers
            seed_path = config.output_dir + "/auto_seeds.txt";
            std::string hmm_path = config.hmm_file.empty() ? "auxiliary/checkm_markers_only.hmm" : config.hmm_file;
            std::string hmmsearch = config.hmmsearch_path.empty() ? "hmmsearch" : config.hmmsearch_path;
            std::string fgs = config.fraggenescan_path.empty() ? "FragGeneScan" : config.fraggenescan_path;
            log.info("Auto-generating seeds using " + hmm_path);
            SeedGenerator gen(config.threads, hmm_path, 1001, hmmsearch, fgs);
            auto seeds = gen.generate(config.contigs_path, seed_path);
            if (seeds.empty()) {
                log.info("WARNING: Failed to auto-generate seeds, continuing without seeds");
            } else {
                seed_set.insert(seeds.begin(), seeds.end());
                log.info("Auto-generated " + std::to_string(seed_set.size()) + " seed marker contigs");
            }
        } else if (std::filesystem::exists(seed_path)) {
            auto seeds = load_seed_markers(seed_path);
            seed_set.insert(seeds.begin(), seeds.end());
            log.info("Loaded " + std::to_string(seed_set.size()) + " seed marker contigs");
        }

        // Build initial membership (COMEBin style: each node in own cluster, seeds are fixed)
        // COMEBin: initial_list = list(np.arange(len(namelist)))  # [0, 1, 2, ...]
        //          is_membership_fixed = [i in seed_idx for i in initial_list]
        std::vector<int> initial_membership(emb_names.size());
        std::vector<bool> is_membership_fixed(emb_names.size(), false);
        int n_seeds = 0;
        for (size_t i = 0; i < emb_names.size(); i++) {
            initial_membership[i] = static_cast<int>(i);  // Each node starts in own cluster
            if (seed_set.count(emb_names[i]) > 0) {
                is_membership_fixed[i] = true;
                n_seeds++;
            }
        }
        log.info("COMEBin-style init: " + std::to_string(emb_names.size()) + " nodes, " +
                 std::to_string(n_seeds) + " fixed seeds");

        // Build kNN index
        std::cerr << "[DEBUG] Building kNN index k=" << config.k << "\n" << std::flush;
        log.info("Building kNN index (k=" + std::to_string(config.k) + ")");
        bin2::KnnConfig knn_config;
        knn_config.k = std::min(config.k, static_cast<int>(embeddings.size()) - 1);  // k can't exceed n-1
        knn_config.ef_search = 1000;
        knn_config.ef_construction = 1000;
        std::cerr << "[DEBUG] k adjusted to " << knn_config.k << "\n" << std::flush;

        bin2::HnswKnnIndex knn_index(knn_config);
        std::cerr << "[DEBUG] Index created, building...\n" << std::flush;
        knn_index.build(embeddings);
        std::cerr << "[DEBUG] Built, querying...\n" << std::flush;
        auto neighbors = knn_index.query_all();
        std::cerr << "[DEBUG] Queried\n" << std::flush;
        log.info("Built index, queried neighbors");

        // Debug: Print first 10 nodes' kNN neighbors and distances for validation
        std::cerr << "\n=== kNN Distance Validation (first 10 nodes) ===\n";
        for (size_t i = 0; i < std::min(size_t(10), neighbors.size()); i++) {
            std::cerr << "Node " << i << " (" << emb_names[i] << "): neighbors=[";
            for (size_t j = 0; j < std::min(size_t(5), neighbors.ids[i].size()); j++) {
                if (j > 0) std::cerr << ", ";
                std::cerr << neighbors.ids[i][j];
            }
            std::cerr << "], dists=[";
            for (size_t j = 0; j < std::min(size_t(5), neighbors.dists[i].size()); j++) {
                if (j > 0) std::cerr << ", ";
                std::cerr << std::fixed << std::setprecision(6) << neighbors.dists[i][j];
            }
            std::cerr << "]\n";
        }
        std::cerr << "=== End kNN Validation ===\n\n" << std::flush;

        // Compute edges (COMEBin method)
        bin2::GraphConfig graph_config;
        graph_config.bandwidth = config.bandwidth;
        graph_config.partgraph_ratio = config.partgraph_ratio;
        bin2::DamageAwareEdgeWeighter weighter(graph_config);
        auto edges = weighter.compute_comebin_edges(neighbors, config.partgraph_ratio);
        log.info("Computed " + std::to_string(edges.size()) + " edges (partgraph_ratio=" +
                 std::to_string(config.partgraph_ratio) + ")");

        // Run Leiden clustering
        bin2::LeidenConfig leiden_config;
        leiden_config.resolution = config.resolution;
        leiden_config.max_iterations = -1;  // COMEBin: run until convergence
        leiden_config.random_seed = config.random_seed;
        leiden_config.null_model = bin2::NullModel::ErdosRenyi;
        leiden_config.use_node_sizes = use_node_sizes;
        if (use_node_sizes) {
            leiden_config.node_sizes = contig_lengths;
            log.info("Using contig lengths as node sizes for Leiden clustering");
        }

        // Always use initial membership (COMEBin style)
        leiden_config.use_initial_membership = true;
        leiden_config.initial_membership = initial_membership;
        if (n_seeds > 0) {
            leiden_config.use_fixed_membership = true;
            leiden_config.is_membership_fixed = is_membership_fixed;
        }

        bin2::ClusteringResult result;
        if (config.use_python_leiden) {
            log.info("Using Python leidenalg for exact COMEBin matching");
            result = run_python_leiden(edges, static_cast<int>(embeddings.size()), contig_lengths);
        } else {
            auto leiden = bin2::create_leiden_backend(true);  // Use libleidenalg if available
            result = leiden->cluster(edges, static_cast<int>(embeddings.size()), leiden_config);
        }
        log.info("Leiden found " + std::to_string(result.num_clusters) + " clusters, modularity=" +
                 std::to_string(result.modularity));

        // Write clustering result
        std::ofstream out(config.output_dir + "/clustering.tsv");
        out << "contig\tcluster\n";
        for (size_t i = 0; i < emb_names.size(); i++) {
            out << emb_names[i] << "\t" << result.labels[i] << "\n";
        }
        out.close();
        log.info("Wrote clustering to " + config.output_dir + "/clustering.tsv");

        // Group contigs by cluster and compute bin sizes
        std::map<int, std::vector<size_t>> cluster_members;
        std::map<int, int64_t> cluster_bp;
        for (size_t i = 0; i < result.labels.size(); i++) {
            int label = result.labels[i];
            if (label >= 0) {
                cluster_members[label].push_back(i);
                cluster_bp[label] += static_cast<int64_t>(contig_lengths[i]);
            }
        }

        // Load contig sequences for bin writing
        auto contigs_seqs = load_contigs_simple(config.contigs_path, 0);  // Load all
        std::unordered_map<std::string, std::string> name_to_seq;
        for (const auto& [name, seq] : contigs_seqs) {
            name_to_seq[name] = seq;
        }

        // Write bins meeting min_bin_size threshold
        int bin_idx = 0;
        size_t total_binned = 0;
        for (auto& [cid, members] : cluster_members) {
            int64_t bp = cluster_bp[cid];
            if (bp < config.min_bin_size) continue;

            std::string bin_name = "bin_" + std::to_string(bin_idx);
            std::ofstream fout(config.output_dir + "/" + bin_name + ".fa");
            for (size_t idx : members) {
                const std::string& name = emb_names[idx];
                auto it = name_to_seq.find(name);
                if (it != name_to_seq.end()) {
                    fout << ">" << name << "\n" << it->second << "\n";
                    total_binned++;
                }
            }
            fout.close();
            bin_idx++;
        }

        log.info("Wrote " + std::to_string(bin_idx) + " bins (" +
                 std::to_string(total_binned) + " contigs)");
        log.info("Unbinned: " + std::to_string(emb_names.size() - total_binned) + " contigs");

        return 0;
    }
    // ========================================
    // END CLUSTERING-ONLY MODE
    // ========================================

    // Load contigs
    auto contigs = load_contigs_simple(config.contigs_path, config.min_length);
    log.info("Loaded " + std::to_string(contigs.size()) + " contigs >= " +
             std::to_string(config.min_length) + " bp");

    if (contigs.empty()) {
        std::cerr << "Error: No contigs pass length filter\n";
        return 1;
    }

    // Extract features
    log.info("Extracting features (TNF + CGR + complexity)");

    std::vector<std::array<float, 136>> tnf_features(contigs.size());
    std::vector<CGRFeatures> cgr_features(contigs.size());
    std::vector<ComplexityFeatures> complexity_features(contigs.size());
    std::vector<double> coverages(contigs.size(), 1.0);
    std::vector<double> variances(contigs.size(), 0.0);  // COMEBin: coverage variance

    omp_set_num_threads(config.threads);

    #pragma omp parallel for
    for (size_t i = 0; i < contigs.size(); i++) {
        extract_tnf_l2(contigs[i].second, tnf_features[i]);

        if (contigs[i].second.length() >= 500) {
            cgr_features[i] = FractalFeatures::extract_cgr_features(contigs[i].second);
        }

        if (contigs[i].second.length() >= 1000) {
            complexity_features[i] = FractalFeatures::extract_complexity_features(contigs[i].second);
        }
    }

    // Load coverage and variance using COMEBin-style (no identity filter, like bedtools genomecov)
    std::unordered_map<std::string, CoverageWithVariance> cov_map;
    if (!config.bam_path.empty() && std::filesystem::exists(config.bam_path)) {
        log.info("Loading coverage from BAM (COMEBin-style: no identity filter, all reads)");
        cov_map = load_coverage_comebin_style(config.bam_path, contigs, config.threads);
        for (size_t i = 0; i < contigs.size(); i++) {
            auto it = cov_map.find(contigs[i].first);
            if (it != cov_map.end()) {
                coverages[i] = it->second.mean;
                variances[i] = it->second.variance;
            }
        }
    }

    // Extract aDNA damage profiles (AMBER-specific features)
    // Always extract for aDNA features; use_damage_infonce controls InfoNCE weighting
    std::vector<bin2::DamageProfile> damage_profiles;
    if (!config.bam_path.empty() && std::filesystem::exists(config.bam_path)) {
        log.info("Extracting aDNA damage profiles from BAM (C→T at 5', G→A at 3', fragment length, mismatch spectrum)");
        auto damage_map = extract_damage_profiles(config.bam_path, contigs, config.threads);
        damage_profiles.resize(contigs.size());
        float total_neff = 0.0f;
        int n_informative = 0;
        float min_panc = 1.0f, max_panc = 0.0f, sum_panc = 0.0f, sum_panc_sq = 0.0f;
        std::array<int, 10> hist_bins = {0};  // p_anc histogram [0-0.1), [0.1-0.2), ..., [0.9-1.0]

        for (size_t i = 0; i < contigs.size(); i++) {
            auto it = damage_map.find(contigs[i].first);
            if (it != damage_map.end()) {
                damage_profiles[i] = it->second;
                total_neff += it->second.n_eff;
                if (it->second.n_eff > 20.0f) n_informative++;

                float p = it->second.p_anc;
                min_panc = std::min(min_panc, p);
                max_panc = std::max(max_panc, p);
                sum_panc += p;
                sum_panc_sq += p * p;
                int bin = std::min(9, static_cast<int>(p * 10));
                hist_bins[bin]++;
            }
        }

        float mean_panc = sum_panc / contigs.size();
        float var_panc = (sum_panc_sq / contigs.size()) - (mean_panc * mean_panc);
        float std_panc = std::sqrt(var_panc);

        log.info("Extracted damage profiles: " + std::to_string(n_informative) + "/" +
                 std::to_string(contigs.size()) + " contigs with n_eff > 20, " +
                 "mean n_eff = " + std::to_string(total_neff / contigs.size()));
        log.info("Damage p_anc distribution: min=" + std::to_string(min_panc) +
                 ", max=" + std::to_string(max_panc) + ", mean=" + std::to_string(mean_panc) +
                 ", std=" + std::to_string(std_panc));

        // Log histogram
        std::string hist_str = "p_anc histogram: ";
        for (int b = 0; b < 10; b++) {
            hist_str += "[" + std::to_string(b*10) + "-" + std::to_string((b+1)*10) + "%]=" +
                       std::to_string(hist_bins[b]) + " ";
        }
        log.info(hist_str);

        // Dump damage profiles to file with per-position rates
        std::ofstream dmg_out(config.output_dir + "/damage_profiles.tsv");
        dmg_out << "contig\tp_anc\tvar_anc\tn_eff\tcov_ancient\tcov_modern\tlog_ratio\tmean_5p\tmean_3p\n";
        for (size_t i = 0; i < contigs.size(); i++) {
            const auto& d = damage_profiles[i];
            dmg_out << contigs[i].first << "\t" << d.p_anc << "\t" << d.var_anc << "\t"
                    << d.n_eff << "\t" << d.cov_ancient << "\t" << d.cov_modern << "\t"
                    << d.log_ratio << "\t" << d.mean_5p_damage() << "\t" << d.mean_3p_damage() << "\n";
        }
        dmg_out.close();

        // Dump smiley plot data (position-specific damage rates)
        std::ofstream smiley_out(config.output_dir + "/smiley_plot_rates.tsv");
        smiley_out << "contig";
        for (int p = 0; p < 10; p++) smiley_out << "\tct_5p_" << p;
        for (int p = 0; p < 10; p++) smiley_out << "\tga_3p_" << p;
        smiley_out << "\n";
        for (size_t i = 0; i < contigs.size(); i++) {
            const auto& d = damage_profiles[i];
            smiley_out << contigs[i].first;
            for (int p = 0; p < 10; p++) smiley_out << "\t" << d.ct_rate_5p[p];
            for (int p = 0; p < 10; p++) smiley_out << "\t" << d.ga_rate_3p[p];
            smiley_out << "\n";
        }
        smiley_out.close();

        // Log summary of position-specific damage rates (average across all contigs)
        std::array<double, 10> avg_ct = {0}, avg_ga = {0};
        for (const auto& d : damage_profiles) {
            for (int p = 0; p < 10; p++) {
                avg_ct[p] += d.ct_rate_5p[p];
                avg_ga[p] += d.ga_rate_3p[p];
            }
        }
        for (int p = 0; p < 10; p++) {
            avg_ct[p] /= damage_profiles.size();
            avg_ga[p] /= damage_profiles.size();
        }
        std::ostringstream ct_str, ga_str;
        ct_str << std::fixed << std::setprecision(3);
        ga_str << std::fixed << std::setprecision(3);
        for (int p = 0; p < 5; p++) ct_str << avg_ct[p] << " ";
        for (int p = 0; p < 5; p++) ga_str << avg_ga[p] << " ";
        log.info("Avg C→T at 5' (pos 1-5): " + ct_str.str());
        log.info("Avg G→A at 3' (pos 1-5): " + ga_str.str());

        log.info("Saved damage profiles to " + config.output_dir + "/damage_profiles.tsv");
        log.info("Saved smiley plot data to " + config.output_dir + "/smiley_plot_rates.tsv");
    }

    // Build feature vectors
    // TNF (136) + CGR (26) + Complexity (14) + log_cov (1) + aDNA (20) = 197 dims
    // aDNA features: damage profile (10) + decay params (2) + frag len (2) + damage cov (2) + mismatch (4)
    bool use_adna_features = !damage_profiles.empty();
    int feature_dim = use_adna_features ? 197 : 177;
    log.info("Building feature vectors (" + std::to_string(feature_dim) + " dims" +
             (use_adna_features ? ", including 20 aDNA-specific features)" : ")"));

    std::vector<std::vector<float>> features(contigs.size());

    for (size_t i = 0; i < contigs.size(); i++) {
        features[i].resize(feature_dim);
        int idx = 0;

        // TNF features (136)
        for (int j = 0; j < 136; j++) features[i][idx++] = tnf_features[i][j];

        // CGR features (26)
        const auto& cgr = cgr_features[i];
        features[i][idx++] = static_cast<float>(cgr.fractal_dimension);
        features[i][idx++] = static_cast<float>(cgr.lacunarity);
        for (int j = 0; j < 7; j++) features[i][idx++] = static_cast<float>(cgr.hu_moments[j]);
        for (int j = 0; j < 10; j++) features[i][idx++] = static_cast<float>(cgr.radial_density[j]);
        for (int j = 0; j < 4; j++) features[i][idx++] = static_cast<float>(cgr.quadrant_density[j]);
        features[i][idx++] = static_cast<float>(cgr.entropy);
        features[i][idx++] = static_cast<float>(cgr.uniformity);
        features[i][idx++] = static_cast<float>(cgr.max_density);

        // Complexity features (14)
        const auto& cplx = complexity_features[i];
        for (int j = 0; j < 4; j++) features[i][idx++] = static_cast<float>(cplx.lempel_ziv_complexity[j]);
        for (int j = 0; j < 3; j++) features[i][idx++] = static_cast<float>(cplx.hurst_exponent[j]);
        features[i][idx++] = static_cast<float>(cplx.dfa_exponent);
        for (int j = 0; j < 5; j++) features[i][idx++] = static_cast<float>(cplx.entropy_scaling[j]);
        features[i][idx++] = static_cast<float>(cplx.kmer_fractal_dim);

        // Coverage (1)
        features[i][idx++] = static_cast<float>(std::log1p(coverages[i]));

        // aDNA-specific features (20) - AMBER novel
        if (use_adna_features && i < damage_profiles.size()) {
            const auto& dmg = damage_profiles[i];

            // Damage profile (10 dims) - first 5 positions from each end
            for (int j = 0; j < 5; j++) features[i][idx++] = dmg.ct_rate_5p[j];
            for (int j = 0; j < 5; j++) features[i][idx++] = dmg.ga_rate_3p[j];

            // Damage decay parameters (2 dims) - normalized
            features[i][idx++] = dmg.lambda_5p / 2.0f;  // Normalize ~[0,1]
            features[i][idx++] = dmg.lambda_3p / 2.0f;

            // Fragment length distribution (2 dims) - normalized
            features[i][idx++] = dmg.frag_len_mean / 150.0f;
            features[i][idx++] = dmg.frag_len_std / 50.0f;

            // Damage-stratified coverage (2 dims) - log-normalized
            features[i][idx++] = std::log1p(dmg.cov_damaged_reads) / 5.0f;
            features[i][idx++] = std::log1p(dmg.cov_undamaged_reads) / 5.0f;

            // Mismatch spectrum (4 dims) - already normalized rates
            features[i][idx++] = dmg.mm_5p_tc;
            features[i][idx++] = dmg.mm_3p_ct;
            features[i][idx++] = dmg.mm_5p_other;
            features[i][idx++] = dmg.mm_3p_other;
        }
    }

    // Log aDNA feature statistics
    if (use_adna_features && !damage_profiles.empty()) {
        float sum_lambda = 0, sum_frag = 0, sum_ratio = 0;
        for (const auto& d : damage_profiles) {
            sum_lambda += (d.lambda_5p + d.lambda_3p) / 2.0f;
            sum_frag += d.frag_len_mean;
            sum_ratio += d.damage_cov_ratio;
        }
        int n = damage_profiles.size();
        log.info("aDNA features: mean_lambda=" + std::to_string(sum_lambda/n) +
                 ", mean_frag_len=" + std::to_string(sum_frag/n) +
                 ", mean_damage_cov_ratio=" + std::to_string(sum_ratio/n));
    }

    // COMEBin-style encoding
    std::vector<std::vector<float>> embeddings;
    auto train_start = std::chrono::steady_clock::now();

    // Compute N50 for temperature selection (COMEBin: 0.07 if N50>10kb, else 0.15)
    std::vector<int64_t> contig_lengths(contigs.size());
    for (size_t i = 0; i < contigs.size(); i++) {
        contig_lengths[i] = static_cast<int64_t>(contigs[i].second.length());
    }
    std::sort(contig_lengths.begin(), contig_lengths.end(), std::greater<int64_t>());
    int64_t total_len = std::accumulate(contig_lengths.begin(), contig_lengths.end(), 0LL);
    int64_t cumsum = 0;
    int64_t n50 = contig_lengths.back();
    for (int64_t len : contig_lengths) {
        cumsum += len;
        if (cumsum >= total_len / 2) {
            n50 = len;
            break;
        }
    }
    log.info("Assembly N50: " + std::to_string(n50) + " bp");

    // COMEBin-style coverage normalization: (cov + 1e-5) / max(cov + 1e-5)
    // Also compute variance max for sqrt(var) normalization
    double cov_max = 1e-5;
    double var_max = 1e-5;
    for (size_t i = 0; i < coverages.size(); i++) {
        cov_max = std::max(cov_max, coverages[i] + 1e-5);
        var_max = std::max(var_max, std::sqrt(variances[i] + 1e-5));
    }
    log.info("Coverage normalization: cov_max=" + std::to_string(cov_max) +
             ", var_max=" + std::to_string(var_max));

    // COMEBin-compatible encoder with substring augmentation
    log.info("Training COMEBin-compatible encoder with substring augmentation");

        // Temperature: COMEBin default is 0.1 (NOT N50-dependent!)
        // Previous code used N50-dependent (0.07/>10kb, 0.15/<=10kb) but that was wrong
        float comebin_temperature = 0.1f;
        log.info("  temperature=" + std::to_string(comebin_temperature) + " (COMEBin default)");
        log.info("  epochs=" + std::to_string(config.epochs) +
                 ", hidden=" + std::to_string(config.hidden_dim) +
                 ", layers=" + std::to_string(config.n_layer));

        // COMEBin default: 6 views (original + 5 augmented substrings)
        const int n_views = 6;
        log.info("Generating " + std::to_string(n_views) + " views with substring augmentation...");

        // Extract sequences for augmentation
        std::vector<std::string> sequences(contigs.size());
        for (size_t i = 0; i < contigs.size(); i++) {
            sequences[i] = contigs[i].second;
        }

        // Generate multi-view TNF features using substring augmentation
        // This uses COMEBin normalization: (raw + 1) / sum
        bin2::SubstringAugmenter::Config aug_config;
        aug_config.n_views = n_views;
        aug_config.min_length = 1000;  // COMEBin default
        aug_config.seed = config.random_seed;
        bin2::SubstringAugmenter augmenter(aug_config);

        auto tnf_views = augmenter.generate_all_views(sequences);
        log.info("Generated " + std::to_string(tnf_views.size()) + " TNF views");

        // Get substring boundaries for each view (same RNG as TNF generation)
        auto substring_info = augmenter.generate_substring_info(sequences);

        int N = contigs.size();

        // Per-position depth already loaded during coverage extraction (single BAM pass)
        log.info("Computing per-view coverage from pre-loaded depth arrays...");

        // Compute per-view coverage/variance for all contigs
        // view_coverages[v][i] = coverage for contig i in view v
        std::vector<std::vector<double>> view_coverages(n_views);
        std::vector<std::vector<double>> view_variances(n_views);
        for (int v = 0; v < n_views; v++) {
            view_coverages[v].resize(N);
            view_variances[v].resize(N);
        }

        // View 0: Use full-contig coverage (same as original)
        for (int i = 0; i < N; i++) {
            view_coverages[0][i] = coverages[i];
            view_variances[0][i] = variances[i];
        }

        // Views 1+: Compute from substring region using pre-loaded per-position depth
        for (int v = 1; v < n_views; v++) {
            for (int i = 0; i < N; i++) {
                const auto& info = substring_info[v][i];
                if (info.start < 0) {
                    // Full contig (sequence too short for substring)
                    view_coverages[v][i] = coverages[i];
                    view_variances[v][i] = variances[i];
                } else {
                    // Compute coverage from substring region using cov_map (loaded earlier)
                    auto it = cov_map.find(contigs[i].first);
                    if (it != cov_map.end() && !it->second.per_position_depth.empty()) {
                        auto [mean, var] = compute_range_coverage(it->second.per_position_depth, info.start, info.end);
                        view_coverages[v][i] = mean;
                        view_variances[v][i] = var;
                    } else {
                        // Fallback if no depth data
                        view_coverages[v][i] = coverages[i];
                        view_variances[v][i] = variances[i];
                    }
                }
            }
        }

        // Compute GLOBAL normalization across ALL views (COMEBin stacks all views first)
        // CRITICAL FIX: COMEBin stacks all n_views*N samples, then normalizes by SINGLE global max
        // Previous code was wrong - used per-view max which breaks contrastive learning signal
        double global_cov_max = 1e-10;
        double global_var_max = 1e-10;
        for (int v = 0; v < n_views; v++) {
            for (int i = 0; i < N; i++) {
                global_cov_max = std::max(global_cov_max, view_coverages[v][i] + 1e-5);
                global_var_max = std::max(global_var_max, std::sqrt(view_variances[v][i]) + 1e-5);
            }
        }

        log.info("Global coverage normalization (all views) - cov_max=" + std::to_string(global_cov_max) +
                 ", var_max=" + std::to_string(global_var_max));

        // Convert to tensors - ONLY view-specific features for contrastive learning
        // COMEBin feature order: [sqrt(var), cov, TNF] = 138 dims
        // NOTE: aDNA features are contig-level (same across views) so they break contrastive learning
        // They will be added to embeddings AFTER encoding, before clustering
        std::vector<torch::Tensor> view_tensors;
        bool has_adna = !damage_profiles.empty() && damage_profiles.size() == static_cast<size_t>(N);
        int D = 138;  // View-specific features only

        if (has_adna) {
            log.info("aDNA features will be concatenated to embeddings after encoding (not used in contrastive training)");
        }

        for (int v = 0; v < n_views; v++) {
            torch::Tensor view_data = torch::zeros({N, D});
            for (int i = 0; i < N; i++) {
                // sqrt(variance) FIRST (index 0) - GLOBAL normalization
                float sqrt_var = static_cast<float>((std::sqrt(view_variances[v][i]) + 1e-5) / global_var_max);
                view_data[i][0] = sqrt_var;
                // Coverage (index 1) - GLOBAL normalization
                float norm_cov = static_cast<float>((view_coverages[v][i] + 1e-5) / global_cov_max);
                view_data[i][1] = norm_cov;
                // TNF features (indices 2-137) - varies per view (substring augmentation)
                for (int j = 0; j < 136; j++) {
                    view_data[i][j + 2] = tnf_views[v][i][j];
                }
            }
            view_tensors.push_back(view_data);
        }

        // Dump view 0 features for comparison with COMEBin
        {
            std::string feature_file = config.output_dir + "/amber_features_view0.tsv";
            std::ofstream fout(feature_file);
            fout << "contig";
            for (int d = 0; d < D; d++) {
                fout << "\tf" << d;
            }
            fout << "\n";
            auto view0_acc = view_tensors[0].accessor<float, 2>();
            for (int i = 0; i < N; i++) {
                fout << contigs[i].first;
                for (int d = 0; d < D; d++) {
                    fout << "\t" << view0_acc[i][d];
                }
                fout << "\n";
            }
            fout.close();
            log.info("Dumped view 0 features to " + feature_file);
        }

        // Create COMEBin encoder
        bin2::COMEBinEncoder comebin_encoder(
            D,                        // input: var(1) + cov(1) + TNF(136) = 138 (COMEBin order)
            config.embedding_dim,     // output: 128
            config.hidden_dim,        // hidden: 2048
            config.n_layer,           // layers: 3
            config.dropout,           // dropout: 0.2
            true                      // use BatchNorm
        );

        // Train with multi-view COMEBin trainer
        bin2::COMEBinTrainer::Config trainer_config;
        trainer_config.epochs = config.epochs;
        trainer_config.batch_size = config.batch_size;
        trainer_config.lr = config.learning_rate;
        trainer_config.temperature = comebin_temperature;  // N50-dependent
        trainer_config.hidden_sz = config.hidden_dim;
        trainer_config.n_layer = config.n_layer;
        trainer_config.out_dim = config.embedding_dim;
        trainer_config.dropout = config.dropout;
        trainer_config.n_views = n_views;
        trainer_config.use_substring_aug = true;
        trainer_config.earlystop = false;      // Disabled - early stopping at epoch 57 caused regression
        trainer_config.use_scheduler = true;   // COMEBin default (cosine annealing)
        trainer_config.warmup_epochs = 10;     // COMEBin default
        // Damage-aware InfoNCE settings
        trainer_config.use_damage_infonce = config.use_damage_infonce;
        trainer_config.damage_lambda = config.damage_lambda;
        trainer_config.damage_wmin = config.damage_wmin;
        // Encoder seed for reproducibility
        trainer_config.seed = config.encoder_seed;

        bin2::COMEBinTrainer trainer(trainer_config);
        // Use damage-aware training if enabled and profiles available
        if (config.use_damage_infonce && !damage_profiles.empty()) {
            log.info("Using damage-aware InfoNCE training");
            trainer.train_multiview(comebin_encoder, view_tensors, damage_profiles);
        } else {
            trainer.train_multiview(comebin_encoder, view_tensors);
        }

        // Encode using original features (view 0) - 138 dims
        std::vector<std::vector<float>> encode_features(N);
        for (int i = 0; i < N; i++) {
            encode_features[i].resize(D);  // D = 138
            // sqrt(variance) FIRST (index 0) - MetaBAT2-style
            encode_features[i][0] = static_cast<float>((std::sqrt(variances[i]) + 1e-5) / var_max);
            // Coverage (index 1) - MetaBAT2-style
            encode_features[i][1] = static_cast<float>((coverages[i] + 1e-5) / cov_max);
            // TNF features (indices 2-137)
            for (int j = 0; j < 136; j++) {
                encode_features[i][j + 2] = tnf_views[0][i][j];
            }
        }
        embeddings = comebin_encoder->encode(encode_features);

        // AMBER-specific: Concatenate aDNA features to embeddings AFTER encoding
        // This preserves contrastive learning while adding aDNA signal for clustering
        if (has_adna) {
            log.info("Concatenating 20 aDNA features to 128-dim embeddings (total 148 dims for clustering)");
            for (int i = 0; i < N; i++) {
                const auto& dmg = damage_profiles[i];
                // Append 20 aDNA features to embedding
                // Damage profile (10 dims)
                for (int j = 0; j < 5; j++) embeddings[i].push_back(dmg.ct_rate_5p[j]);
                for (int j = 0; j < 5; j++) embeddings[i].push_back(dmg.ga_rate_3p[j]);
                // Damage decay (2 dims)
                embeddings[i].push_back(dmg.lambda_5p / 2.0f);
                embeddings[i].push_back(dmg.lambda_3p / 2.0f);
                // Fragment length (2 dims)
                embeddings[i].push_back(dmg.frag_len_mean / 150.0f);
                embeddings[i].push_back(dmg.frag_len_std / 50.0f);
                // Damage-stratified coverage (2 dims)
                embeddings[i].push_back(std::log1p(dmg.cov_damaged_reads) / 5.0f);
                embeddings[i].push_back(std::log1p(dmg.cov_undamaged_reads) / 5.0f);
                // Mismatch spectrum (4 dims)
                embeddings[i].push_back(dmg.mm_5p_tc);
                embeddings[i].push_back(dmg.mm_3p_ct);
                embeddings[i].push_back(dmg.mm_5p_other);
                embeddings[i].push_back(dmg.mm_3p_other);
            }
        }

        // Append 9 multi-scale CGR features (scale-transition entropy/occupancy/lacunarity)
        // Captures sequence complexity at multiple scales, orthogonal to damage features
        {
            int total_dims = (int)embeddings[0].size() + bin2::MultiScaleCGRFeatures::num_features();
            log.info("Concatenating 9 CGR features to embeddings (total " +
                     std::to_string(total_dims) + " dims for clustering)");
            std::vector<bin2::MultiScaleCGRFeatures> cgr_features(N);
            #pragma omp parallel for num_threads(config.threads) schedule(dynamic)
            for (int i = 0; i < N; i++) {
                cgr_features[i] = bin2::MultiScaleCGRExtractor::extract(sequences[i], 500);
            }
            for (int i = 0; i < N; i++) {
                for (float f : cgr_features[i].to_vector()) {
                    embeddings[i].push_back(f);
                }
            }
        }

    auto train_end = std::chrono::steady_clock::now();
    auto train_s = std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start).count();
    log.info("Training completed in " + std::to_string(train_s) + " s");
    log.info("Encoded " + std::to_string(embeddings.size()) + " contigs to " +
             std::to_string(config.embedding_dim) + " dimensions");

    // L2 normalize embeddings (COMEBin default)
    if (config.l2_normalize) {
        log.info("L2 normalizing embeddings");
        for (auto& emb : embeddings) {
            float norm = 0.0f;
            for (float v : emb) norm += v * v;
            norm = std::sqrt(norm);
            if (norm > 1e-10f) {
                for (float& v : emb) v /= norm;
            }
        }
    }

    // contig_lengths already computed above (line ~1323) for N50 calc
    // Used below for Leiden node_sizes (RBER null model)

    // Load or auto-generate seed markers
    std::unordered_set<std::string> seed_set;
    std::unordered_map<std::string, std::vector<std::string>> marker_hits_by_contig;
    std::string seed_path = config.seed_file;
    if (seed_path.empty()) {
        // Auto-generate seeds using SCG markers
        seed_path = config.output_dir + "/auto_seeds.txt";
        std::string hmm_path = config.hmm_file.empty() ? "auxiliary/checkm_markers_only.hmm" : config.hmm_file;
        std::string hmmsearch = config.hmmsearch_path.empty() ? "hmmsearch" : config.hmmsearch_path;
        std::string fgs = config.fraggenescan_path.empty() ? "FragGeneScan" : config.fraggenescan_path;
        log.info("Auto-generating seeds using " + hmm_path);
        SeedGenerator gen(config.threads, hmm_path, 1001, hmmsearch, fgs);
        auto seeds = gen.generate(config.contigs_path, seed_path);
        if (seeds.empty()) {
            log.info("WARNING: Failed to auto-generate seeds, continuing without seeds");
        } else {
            seed_set.insert(seeds.begin(), seeds.end());
            log.info("Auto-generated " + std::to_string(seed_set.size()) + " seed marker contigs");
            marker_hits_by_contig = gen.get_hits_by_contig();
        }
    } else if (std::filesystem::exists(seed_path)) {
        auto seeds = load_seed_markers(seed_path);
        seed_set.insert(seeds.begin(), seeds.end());
        log.info("Loaded " + std::to_string(seed_set.size()) + " seed marker contigs");
    }

    // Build initial membership from seeds (contigs containing marker genes get their own cluster)
    std::vector<int> initial_membership(contigs.size(), -1);
    std::vector<bool> is_membership_fixed(contigs.size(), false);
    int seed_cluster_id = 0;
    for (size_t i = 0; i < contigs.size(); i++) {
        if (seed_set.count(contigs[i].first) > 0) {
            initial_membership[i] = seed_cluster_id++;
            is_membership_fixed[i] = true;
        }
    }
    if (seed_cluster_id > 0) {
        log.info("Assigned " + std::to_string(seed_cluster_id) + " seed contigs to fixed clusters");
    }

    // Colocation-guided marker contig pre-assignment.
    //
    // Theoretical max HQ bins = seed_cluster_id (= median marker copy count, i.e.
    // the number of detectable genomes).  We close the gap by pre-assigning ALL
    // non-seed marker-bearing contigs before Leiden runs:
    //
    //   Colocation (hard filter)  — a seed is INCOMPATIBLE if it already holds a
    //     copy of the same single-copy marker.  Assigning there would create
    //     contamination before Leiden even starts.
    //
    //   Embedding distance (ranking) — among compatible seeds, pick the one whose
    //     embedding (TNF + coverage + aDNA + CGR) is closest in L2 space.
    //
    // Processing order: most-constrained first (fewest compatible seeds), so
    // unambiguous assignments anchor the space before uncertain ones.
    //
    // Result: all ~269 marker-bearing contigs are fixed; Leiden only optimises the
    // ~89k non-marker contigs that cannot cause marker-level contamination.
    if (seed_cluster_id > 0 && !marker_hits_by_contig.empty() && !embeddings.empty()) {
        std::string bacteria_ms = config.bacteria_ms_file.empty()
                                ? "scripts/checkm_ms/bacteria.ms"
                                : config.bacteria_ms_file;
        if (std::filesystem::exists(bacteria_ms)) {
            // marker_hits_by_contig is keyed by contig name → list of markers
            // (the variable name is misleading; the seed generator returns this orientation)
            const auto& contig_to_markers = marker_hits_by_contig;

            // For each seed cluster: track how many copies of each marker it holds.
            // Initially populated from the seed contigs themselves.
            // Layout: seed_marker_counts[cluster_id][marker_name] = copy_count
            std::vector<std::unordered_map<std::string, int>> seed_marker_counts(seed_cluster_id);
            // Also record the embedding index for each seed cluster
            std::vector<int> seed_emb_idx(seed_cluster_id, -1);
            for (size_t i = 0; i < contigs.size(); i++) {
                if (!is_membership_fixed[i]) continue;
                int sc = initial_membership[i];
                seed_emb_idx[sc] = static_cast<int>(i);
                auto sit = contig_to_markers.find(contigs[i].first);
                if (sit != contig_to_markers.end())
                    for (const auto& m : sit->second)
                        seed_marker_counts[sc][m]++;
            }

            // Collect non-seed marker contigs and sort most-constrained first
            // (ascending compatible-seed count → deterministic, reproducible)
            struct Candidate { int idx; int n_compatible; };
            std::vector<Candidate> candidates;
            for (size_t i = 0; i < contigs.size(); i++) {
                if (is_membership_fixed[i]) continue;
                if (contigs[i].second.size() < 1001) continue;  // same length filter as seed generator
                const auto it = contig_to_markers.find(contigs[i].first);
                if (it == contig_to_markers.end()) continue;
                const auto& cmarkers = it->second;
                int n_compat = 0;
                for (int sc = 0; sc < seed_cluster_id; sc++) {
                    bool ok = true;
                    for (const auto& m : cmarkers) {
                        auto jt = seed_marker_counts[sc].find(m);
                        if (jt != seed_marker_counts[sc].end() && jt->second > 0) { ok = false; break; }
                    }
                    if (ok) n_compat++;
                }
                if (n_compat > 0)
                    candidates.push_back({static_cast<int>(i), n_compat});
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate& a, const Candidate& b) {
                          return a.n_compatible < b.n_compatible; });

            // Greedily assign each candidate to its closest compatible seed
            int n_preassigned = 0;
            const int D = static_cast<int>(embeddings[0].size());
            for (const auto& cand : candidates) {
                int ci = cand.idx;
                const auto& cmarkers = contig_to_markers.at(contigs[ci].first);

                int best_sc = -1;
                float best_dist = std::numeric_limits<float>::max();
                for (int sc = 0; sc < seed_cluster_id; sc++) {
                    bool ok = true;
                    for (const auto& m : cmarkers) {
                        auto jt = seed_marker_counts[sc].find(m);
                        if (jt != seed_marker_counts[sc].end() && jt->second > 0) { ok = false; break; }
                    }
                    if (!ok) continue;
                    int si = seed_emb_idx[sc];
                    if (si < 0) continue;
                    float dist = 0.0f;
                    for (int d = 0; d < D; d++) {
                        float diff = embeddings[ci][d] - embeddings[si][d];
                        dist += diff * diff;
                    }
                    if (dist < best_dist) { best_dist = dist; best_sc = sc; }
                }
                if (best_sc < 0) continue;  // no compatible seed — leave free

                // Assign to best seed cluster (not fixed — Leiden can still move it,
                // but it starts co-located with its genome anchor rather than isolated)
                initial_membership[ci] = best_sc;
                for (const auto& m : cmarkers)
                    seed_marker_counts[best_sc][m]++;
                n_preassigned++;
            }
            log.info("Colocation pre-assignment: fixed " + std::to_string(n_preassigned) +
                     " non-seed marker contigs (" +
                     std::to_string(seed_cluster_id + n_preassigned) +
                     " total fixed, theoretical max HQ=" + std::to_string(seed_cluster_id) + ")");
        }
    }

    // Use new modular clustering pipeline
    log.info("Building HNSW kNN index (k=" + std::to_string(config.k) + ")");
    bin2::KnnConfig knn_config;
    knn_config.k = config.k;
    knn_config.ef_search = 200;
    knn_config.M = 16;

    bin2::HnswKnnIndex knn_index(knn_config);
    knn_index.build(embeddings);

    auto neighbors = knn_index.query_all();
    log.info("Built kNN graph with " + std::to_string(neighbors.size()) + " nodes");

    // COMEBin-style parameter sweep (3 x 5 x 8 = 120 configurations)
    if (config.sweep) {
        log.info("Running COMEBin-exact parameter sweep");

        // COMEBin default sweep values
        std::vector<int> partgraph_ratios = {50, 100, 80};  // COMEBin pgr values
        std::vector<float> bandwidths = {0.05f, 0.1f, 0.15f, 0.2f, 0.3f};
        std::vector<float> resolutions = {1.0f, 5.0f, 10.0f, 30.0f, 50.0f, 70.0f, 90.0f, 110.0f};

        std::ofstream sweep_summary(config.output_dir + "/sweep_summary.tsv");
        sweep_summary << "partgraph_ratio\tbandwidth\tresolution\tnum_clusters\tmodularity\tnum_bins\ttotal_binned_bp\n";

        int total_runs = partgraph_ratios.size() * bandwidths.size() * resolutions.size();
        int run_idx = 0;

        for (int pgr : partgraph_ratios) {
            // Compute edges once per partgraph_ratio (COMEBin pattern)
            bin2::GraphConfig gc;
            gc.partgraph_ratio = pgr;
            gc.damage_beta = 0.0f;  // Damage features disabled in COMEBin encoder
            bin2::DamageAwareEdgeWeighter weighter(gc);
            // Use raw distances: compute_comebin_edges() already applies exp(-dist/bw),
            // so reusing its output in the bandwidth loop would double-transform.
            auto pgr_edges = weighter.compute_comebin_edges_distances(neighbors, pgr);
            log.info("partgraph_ratio=" + std::to_string(pgr) + " -> " +
                     std::to_string(pgr_edges.size()) + " edges");

            for (float bw : bandwidths) {
                // Apply COMEBin kernel: exp(-dist/bw)
                std::vector<bin2::WeightedEdge> weighted_edges;
                weighted_edges.reserve(pgr_edges.size());
                for (const auto& e : pgr_edges) {
                    float w = std::exp(-e.w / bw);
                    weighted_edges.emplace_back(e.u, e.v, w);
                }

                for (float res : resolutions) {
                    run_idx++;
                    std::ostringstream subdir_name;
                    subdir_name << "sweep_pgr" << pgr
                                << "_bw" << std::fixed << std::setprecision(2) << bw
                                << "_res" << std::setprecision(0) << res;
                    std::string subdir = config.output_dir + "/" + subdir_name.str();
                    std::filesystem::create_directories(subdir);

                    log.info("[" + std::to_string(run_idx) + "/" + std::to_string(total_runs) +
                             "] pgr=" + std::to_string(pgr) + " bw=" + std::to_string(bw) +
                             " res=" + std::to_string(res));

                    // Run Leiden with COMEBin-exact settings
                    bin2::LeidenConfig lc;
                    lc.resolution = res;
                    lc.max_iterations = -1;  // COMEBin: unlimited iterations
                    lc.random_seed = config.random_seed;
                    lc.null_model = bin2::NullModel::ErdosRenyi;  // COMEBin: RBER model
                    lc.use_node_sizes = true;
                    lc.node_sizes.resize(contigs.size());
                    for (size_t i = 0; i < contigs.size(); i++) {
                        lc.node_sizes[i] = static_cast<double>(contigs[i].second.length());
                    }

                    // Apply seed constraints if available
                    if (seed_cluster_id > 0) {
                        lc.use_initial_membership = true;
                        lc.initial_membership = initial_membership;
                        lc.use_fixed_membership = true;
                        lc.is_membership_fixed = is_membership_fixed;
                    }

                    auto backend = bin2::create_leiden_backend(true);
                    auto result = backend->cluster(weighted_edges, static_cast<int>(contigs.size()), lc);

                    // Collect clusters and filter by min_bin_size
                    std::map<int, std::vector<size_t>> clusters;
                    for (size_t i = 0; i < result.labels.size(); i++) {
                        clusters[result.labels[i]].push_back(i);
                    }

                    int num_bins = 0;
                    size_t total_binned_bp = 0;

                    for (auto& [cluster_id, members] : clusters) {
                        size_t total_bp = 0;
                        for (size_t idx : members) total_bp += contigs[idx].second.length();
                        if (static_cast<int>(total_bp) < config.min_bin_size) continue;

                        // Write bin
                        std::string bin_name = "bin_" + std::to_string(num_bins);
                        std::ofstream out(subdir + "/" + bin_name + ".fa");
                        for (size_t idx : members) {
                            out << ">" << contigs[idx].first << "\n";
                            const auto& seq = contigs[idx].second;
                            for (size_t i = 0; i < seq.length(); i += 80) {
                                out << seq.substr(i, 80) << "\n";
                            }
                        }
                        num_bins++;
                        total_binned_bp += total_bp;
                    }

                    sweep_summary << pgr << "\t"
                                  << std::fixed << std::setprecision(2) << bw << "\t"
                                  << std::setprecision(0) << res << "\t"
                                  << result.num_clusters << "\t"
                                  << std::setprecision(4) << result.modularity << "\t"
                                  << num_bins << "\t" << total_binned_bp << "\n";

                    log.info("  -> " + std::to_string(result.num_clusters) + " clusters, " +
                             std::to_string(num_bins) + " bins (>=" +
                             std::to_string(config.min_bin_size/1000) + "kb)");
                }
            }
        }

        sweep_summary.close();
        log.info("Sweep complete (120 configurations). Summary: " + config.output_dir + "/sweep_summary.tsv");
        log.info("Run CheckM2 on each subdirectory to find best parameters");
        return 0;
    }

    // Compute edges with COMEBin-exact method (partgraph_ratio=50 default)
    bin2::GraphConfig graph_config;
    graph_config.bandwidth = config.bandwidth;
    graph_config.damage_beta = 0.0f;  // Damage features disabled in COMEBin encoder
    graph_config.partgraph_ratio = 50;  // COMEBin default

    bin2::DamageAwareEdgeWeighter weighter(graph_config);
    // compute_comebin_edges already applies exp(-dist/bandwidth) internally
    auto edges = weighter.compute_comebin_edges(neighbors, 50);
    log.info("Computed " + std::to_string(edges.size()) + " edges (pgr=50)");

    // Run Leiden clustering with COMEBin-exact settings
    bin2::LeidenConfig leiden_config;
    leiden_config.resolution = config.resolution;
    leiden_config.max_iterations = -1;  // COMEBin: unlimited iterations
    leiden_config.random_seed = config.random_seed;
    leiden_config.null_model = bin2::NullModel::ErdosRenyi;  // COMEBin: RBER model
    leiden_config.use_node_sizes = true;
    leiden_config.node_sizes.resize(contigs.size());
    for (size_t i = 0; i < contigs.size(); i++) {
        leiden_config.node_sizes[i] = static_cast<double>(contigs[i].second.length());
    }

    // Apply seed constraints if available.
    // Replace -1 placeholders (unassigned contigs) with unique singleton IDs
    // so libleidenalg gets a valid initial partition where seeds + colocation
    // pre-assigned contigs start in their genome clusters (0..seed_cluster_id-1)
    // and all other contigs start as isolated singletons.
    if (seed_cluster_id > 0) {
        int next_singleton = seed_cluster_id;
        for (size_t i = 0; i < contigs.size(); i++)
            if (initial_membership[i] < 0)
                initial_membership[i] = next_singleton++;

        leiden_config.use_initial_membership = true;
        leiden_config.initial_membership = initial_membership;
        leiden_config.use_fixed_membership = true;
        leiden_config.is_membership_fixed = is_membership_fixed;
    }

    // Build marker index and CheckM estimator if quality Leiden is requested.
    // Both must outlive the backend, so they are declared at this scope.
    bin2::MarkerIndex marker_index;
    std::vector<std::string> bin2_contig_names;
    bin2::CheckMQualityEstimator bin2_checkm_est;

    if (config.use_quality_leiden && !marker_hits_by_contig.empty()) {
        bin2_contig_names.reserve(contigs.size());
        for (const auto& [name, seq] : contigs)
            bin2_contig_names.push_back(name);

        std::string bacteria_ms = config.bacteria_ms_file.empty() ? "scripts/checkm_ms/bacteria.ms" : config.bacteria_ms_file;
        std::string archaea_ms  = config.archaea_ms_file.empty()  ? "scripts/checkm_ms/archaea.ms"  : config.archaea_ms_file;

        if (std::filesystem::exists(bacteria_ms)) {
            bin2::CheckMMarkerParser ms_parser;
            ms_parser.load(bacteria_ms, archaea_ms);
            marker_index.build(ms_parser.bacteria(), marker_hits_by_contig, bin2_contig_names);
            log.info("MarkerIndex (CheckM colocation weights): " +
                     std::to_string(marker_index.num_marker_contigs()) +
                     " contigs, " + std::to_string(marker_index.num_markers()) +
                     " markers, " + std::to_string(marker_index.num_sets()) + " sets");

            // Build CheckM estimator for restart scoring and Phase 2 split criterion.
            // Uses the same colocation sets so quality scores are consistent.
            bin2_checkm_est.load_marker_sets(bacteria_ms, archaea_ms);
            bin2_checkm_est.set_hits(marker_hits_by_contig);
            log.info("CheckMQualityEstimator ready for colocation-based scoring");
        } else {
            log.info("CheckM marker sets not found at " + bacteria_ms + ", using uniform weights");
            marker_index.build_from_hits(marker_hits_by_contig, bin2_contig_names);
            log.info("MarkerIndex (uniform weights): " +
                     std::to_string(marker_index.num_marker_contigs()) +
                     " contigs, " + std::to_string(marker_index.num_markers()) + " markers");
        }
    }

    std::unique_ptr<bin2::ILeidenBackend> backend;
    if (config.use_quality_leiden && marker_index.num_marker_contigs() > 0) {
        log.info("Using QualityLeiden (alpha=" + std::to_string(config.quality_alpha) + ")");
        auto qb = std::make_unique<bin2::QualityLeidenBackend>();
        qb->set_marker_index(&marker_index);
        if (bin2_checkm_est.has_marker_sets()) {
            qb->set_checkm_estimator(&bin2_checkm_est, &bin2_contig_names);
            log.info("QualityLeiden: CheckM colocation scoring enabled");
        }
        bin2::QualityLeidenConfig qcfg;
        qcfg.alpha = config.quality_alpha;
        qcfg.use_map_equation = config.use_map_equation;
        qcfg.n_leiden_restarts = config.n_leiden_restarts;
        qcfg.n_threads = config.threads;
        qb->set_quality_config(qcfg);
        backend = std::move(qb);
    } else if (config.use_ensemble_leiden) {
        log.info("Using EnsembleLeiden (" + std::to_string(config.n_ensemble_runs) +
                 " runs, threshold=" + std::to_string(config.ensemble_threshold) +
                 (config.ensemble_vary_resolution ?
                  ", res=[" + std::to_string(config.ensemble_res_min) + "," +
                  std::to_string(config.ensemble_res_max) + "]" : "") + ")");
        auto ensemble = bin2::create_ensemble_leiden(config.n_ensemble_runs, config.ensemble_threshold);
        ensemble->set_seed(leiden_config.random_seed);
        if (config.ensemble_vary_resolution) {
            bin2::EnsembleLeidenConfig ecfg;
            ecfg.n_runs = config.n_ensemble_runs;
            ecfg.consensus_threshold = config.ensemble_threshold;
            ecfg.base_seed = leiden_config.random_seed;
            ecfg.vary_resolution = true;
            ecfg.res_min = config.ensemble_res_min;
            ecfg.res_max = config.ensemble_res_max;
            ecfg.verbose = true;
            ensemble->set_ensemble_config(ecfg);
        }
        backend = std::move(ensemble);
    } else {
        backend = bin2::create_leiden_backend(true);  // Use Leiden (not Louvain)
    }
    auto result = backend->cluster(edges, static_cast<int>(contigs.size()), leiden_config);

    std::vector<int> labels = result.labels;
    log.info("Leiden found " + std::to_string(result.num_clusters) + " clusters (modularity=" +
             std::to_string(result.modularity) + ")");

    // Count clusters
    std::map<int, std::vector<size_t>> clusters;
    for (size_t i = 0; i < labels.size(); i++) {
        clusters[labels[i]].push_back(i);
    }
    log.info("Found " + std::to_string(clusters.size()) + " clusters");

    // Pre-calculate bin sizes to filter small bins
    std::vector<std::pair<int, size_t>> bin_sizes;  // cluster_id, total_bp
    for (auto& [cluster_id, members] : clusters) {
        size_t total_bp = 0;
        for (size_t idx : members) {
            total_bp += contigs[idx].second.length();
        }
        bin_sizes.emplace_back(cluster_id, total_bp);
    }

    // Mark contigs from small bins as unbinned
    for (auto& [cluster_id, total_bp] : bin_sizes) {
        if (static_cast<int>(total_bp) < config.min_bin_size) {
            for (size_t idx : clusters[cluster_id]) {
                labels[idx] = -1;  // Mark as unbinned
            }
        }
    }

    // Write bins (only those meeting min_bin_size)
    int bin_idx = 0;
    size_t total_binned = 0;
    std::ofstream stats(config.output_dir + "/bin_stats.tsv");
    stats << "bin\tcontigs\ttotal_bp\tmean_coverage\tmean_gc\n";

    for (auto& [cluster_id, members] : clusters) {
        // Calculate total_bp
        size_t total_bp = 0;
        for (size_t idx : members) {
            total_bp += contigs[idx].second.length();
        }

        // Skip small bins
        if (static_cast<int>(total_bp) < config.min_bin_size) {
            continue;
        }

        std::string bin_name = "bin_" + std::to_string(bin_idx);
        std::ofstream out(config.output_dir + "/" + bin_name + ".fa");

        double sum_cov = 0, sum_gc = 0;

        for (size_t idx : members) {
            out << ">" << contigs[idx].first << "\n";
            const auto& seq = contigs[idx].second;
            for (size_t i = 0; i < seq.length(); i += 80) {
                out << seq.substr(i, 80) << "\n";
            }
            sum_cov += coverages[idx];

            int gc = 0;
            for (char c : seq) {
                if (c == 'G' || c == 'g' || c == 'C' || c == 'c') gc++;
            }
            sum_gc += static_cast<double>(gc) / seq.length();
        }

        stats << bin_name << "\t" << members.size() << "\t" << total_bp << "\t"
              << std::fixed << std::setprecision(2) << sum_cov / members.size() << "\t"
              << std::setprecision(4) << sum_gc / members.size() << "\n";

        total_binned += members.size();
        bin_idx++;
    }

    // Write unbinned (including contigs from small bins)
    std::ofstream unbinned(config.output_dir + "/unbinned.fa");
    size_t n_unbinned = 0;
    for (size_t i = 0; i < contigs.size(); i++) {
        if (labels[i] < 0) {
            unbinned << ">" << contigs[i].first << "\n";
            const auto& seq = contigs[i].second;
            for (size_t j = 0; j < seq.length(); j += 80) {
                unbinned << seq.substr(j, 80) << "\n";
            }
            n_unbinned++;
        }
    }

    log.info("Wrote " + std::to_string(bin_idx) + " bins (" +
             std::to_string(total_binned) + " contigs)");
    log.info("Unbinned: " + std::to_string(n_unbinned) + " contigs");

    // Write embeddings (pure: contig + dims only, no bin column).
    // The reader (load_embeddings_tsv) parses all post-name tokens as floats,
    // so a bin column would corrupt the embedding if fed back via --embeddings.
    // Bin assignments are already in the output .fa files.
    std::ofstream emb_out(config.output_dir + "/embeddings.tsv");
    emb_out << "contig";
    for (int d = 0; d < config.embedding_dim; d++) {
        emb_out << "\tdim_" << d;
    }
    emb_out << "\n";

    for (size_t i = 0; i < contigs.size(); i++) {
        emb_out << contigs[i].first;
        for (int d = 0; d < config.embedding_dim; d++) {
            emb_out << "\t" << std::fixed << std::setprecision(6) << embeddings[i][d];
        }
        emb_out << "\n";
    }

    return 0;
}

}  // namespace amber

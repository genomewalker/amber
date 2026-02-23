// AMBER - polish.cpp
// Damage-aware polishing module
// Corrects ancient DNA damage using Bayesian model with read evidence

#include "algorithms/damage_likelihood.h"
#include "algorithms/damage_profile.h"
#include "algorithms/polish_engine.h"
#include "algorithms/strain_deconvolution.h"
#include "algorithms/sequence_utils.h"
#include "util/logger.h"
#include <amber/config.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <omp.h>
#include <htslib/hts.h>
#include <htslib/sam.h>
#include <htslib/faidx.h>

namespace amber {

struct PolishStats {
    size_t total_contigs = 0;
    size_t processed_contigs = 0;
    size_t total_edits = 0;
    size_t ct_corrections = 0;
    size_t ga_corrections = 0;
    size_t positions_examined = 0;
    size_t positions_with_evidence = 0;
    size_t skipped_low_coverage = 0;
    size_t skipped_ambiguous = 0;
};

// Single read observation at a pileup position
struct ReadObservation {
    char base;           // Observed base (A, C, G, T)
    uint8_t baseq;       // Base quality score
    int dist_from_5p;    // Distance from read's 5' end (1-based)
    int dist_from_3p;    // Distance from read's 3' end (1-based)
    int read_len;        // Read length
    bool is_reverse;     // True if read is on reverse strand
};

// Pileup data at a single position with per-read information
struct PileupData {
    std::vector<ReadObservation> observations;
    int n_A = 0, n_C = 0, n_G = 0, n_T = 0;
    int total() const { return n_A + n_C + n_G + n_T; }
};

// Get pileup data at a specific contig position with per-read details
// Returns observations from reads passing quality filters, including 5' distance
static PileupData get_pileup_at_position(
    samFile *sf, bam_hdr_t *hdr, hts_idx_t *idx,
    const std::string &contig_name, int pos,  // 0-based position
    int min_mapq, int min_baseq) {

    PileupData data;

    int tid = bam_name2id(hdr, contig_name.c_str());
    if (tid < 0) return data;

    hts_itr_t *iter = sam_itr_queryi(idx, tid, pos, pos + 1);
    if (!iter) return data;

    bam1_t *b = bam_init1();

    while (sam_itr_next(sf, iter, b) >= 0) {
        if (b->core.qual < min_mapq) continue;
        if (b->core.flag & (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY | BAM_FDUP)) continue;

        uint32_t *cigar = bam_get_cigar(b);
        int n_cigar = b->core.n_cigar;
        int ref_pos = b->core.pos;
        int read_pos = 0;
        bool is_reverse = (b->core.flag & BAM_FREVERSE) != 0;
        int read_len = b->core.l_qseq;

        bool found = false;
        for (int i = 0; i < n_cigar && !found; i++) {
            int op = bam_cigar_op(cigar[i]);
            int len = bam_cigar_oplen(cigar[i]);

            if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                if (ref_pos <= pos && pos < ref_pos + len) {
                    int offset = pos - ref_pos;
                    int qpos = read_pos + offset;

                    uint8_t *qual = bam_get_qual(b);
                    if (qual[qpos] < min_baseq) break;

                    uint8_t *seq = bam_get_seq(b);
                    int base_code = bam_seqi(seq, qpos);

                    ReadObservation obs;
                    obs.baseq = qual[qpos];
                    obs.is_reverse = is_reverse;
                    obs.read_len = read_len;

                    // Calculate distance from 5' and 3' ends of the read
                    // For forward reads: 5' is at position 0, 3' is at end
                    // For reverse reads: 5' is at end, 3' is at position 0
                    if (is_reverse) {
                        obs.dist_from_5p = read_len - qpos;
                        obs.dist_from_3p = qpos + 1;
                    } else {
                        obs.dist_from_5p = qpos + 1;
                        obs.dist_from_3p = read_len - qpos;
                    }

                    switch (base_code) {
                        case 1: obs.base = 'A'; data.n_A++; break;
                        case 2: obs.base = 'C'; data.n_C++; break;
                        case 4: obs.base = 'G'; data.n_G++; break;
                        case 8: obs.base = 'T'; data.n_T++; break;
                        default: continue;
                    }

                    data.observations.push_back(obs);
                    found = true;
                }
                ref_pos += len;
                read_pos += len;
            } else if (op == BAM_CINS || op == BAM_CSOFT_CLIP) {
                read_pos += len;
            } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
                ref_pos += len;
            }
        }
    }

    bam_destroy1(b);
    hts_itr_destroy(iter);

    return data;
}

// Wrappers for shared damage likelihood template functions (see algorithms/damage_likelihood.h)
static double compute_ct_log_likelihood_ratio_perread(
    const std::vector<ReadObservation>& observations,
    double amplitude, double lambda, double baseline) {
    return compute_ct_log_lr(observations, amplitude, lambda, baseline);
}

static double compute_ga_log_likelihood_ratio_perread(
    const std::vector<ReadObservation>& observations,
    double amplitude, double lambda, double baseline) {
    return compute_ga_log_lr(observations, amplitude, lambda, baseline);
}

// Write damage profile to TSV file
static void write_damage_profile(const std::string &filename, const DamageAnalysis &analysis) {
    std::ofstream out(filename);
    out << "contig\tposition\tc_to_t\tg_to_a\ttotal_obs\n";
    for (const auto &[contig, entries] : analysis.profile.per_contig) {
        for (const auto &e : entries) {
            out << contig << "\t" << e.pos << "\t"
                << std::fixed << std::setprecision(4) << e.c_to_t << "\t"
                << e.g_to_a << "\t" << e.total << "\n";
        }
    }
}

// Write damage model summary
static void write_model_summary(const std::string &filename, const DamageAnalysis &analysis) {
    std::ofstream out(filename);
    const auto &gm = analysis.global_model;

    out << "# Bayesian damage model (AMBER)\n";
    out << "library_type\t" << analysis.library_type << "\n";
    out << "p_ancient\t" << std::fixed << std::setprecision(4) << gm.p_ancient << "\n";
    out << "log_bayes_factor\t" << gm.log_bf << "\n";
    out << "evidence_strength\t" << gm.evidence_strength << "\n";
    out << "is_ancient\t" << (gm.is_significant ? "yes" : "no") << "\n";
    out << "classification\t" << gm.significance_reason << "\n";
    out << "total_observations\t" << gm.total_obs << "\n";
    out << "effective_coverage\t" << gm.effective_coverage << "\n";

    out << "\n# Calibrated sequencing error rate (from interior positions)\n";
    out << "error_rate\t" << analysis.calibrated_error_rate << "\n";
    out << "error_rate_ci_lower\t" << (analysis.calibrated_error_rate - 1.96 * analysis.error_rate_se) << "\n";
    out << "error_rate_ci_upper\t" << (analysis.calibrated_error_rate + 1.96 * analysis.error_rate_se) << "\n";

    out << "\n# 5' end curve parameters (C->T deamination)\n";
    out << "amplitude_5p\t" << gm.curve_5p.amplitude << "\n";
    out << "amplitude_5p_ci_lower\t" << std::max(0.0, gm.curve_5p.amplitude - 1.96 * gm.curve_5p.amplitude_se) << "\n";
    out << "amplitude_5p_ci_upper\t" << std::min(1.0, gm.curve_5p.amplitude + 1.96 * gm.curve_5p.amplitude_se) << "\n";
    out << "lambda_5p\t" << gm.curve_5p.lambda << "\n";
    out << "baseline_5p\t" << gm.curve_5p.baseline << "\n";
    out << "concentration_5p\t" << gm.curve_5p.concentration << "\n";

    out << "\n# 3' end curve parameters (G->A deamination)\n";
    out << "amplitude_3p\t" << gm.curve_3p.amplitude << "\n";
    out << "amplitude_3p_ci_lower\t" << std::max(0.0, gm.curve_3p.amplitude - 1.96 * gm.curve_3p.amplitude_se) << "\n";
    out << "amplitude_3p_ci_upper\t" << std::min(1.0, gm.curve_3p.amplitude + 1.96 * gm.curve_3p.amplitude_se) << "\n";
    out << "lambda_3p\t" << gm.curve_3p.lambda << "\n";
    out << "baseline_3p\t" << gm.curve_3p.baseline << "\n";
    out << "concentration_3p\t" << gm.curve_3p.concentration << "\n";

    // Per-position statistics with credible intervals
    out << "\n# Per-position posterior statistics (5' end C->T)\n";
    out << "position\tn_opportunities\tn_mismatches\trate\tci_lower\tci_upper\n";
    for (const auto &s : gm.stats_5p) {
        out << s.position << "\t" << s.n_opportunities << "\t" << s.n_mismatches << "\t"
            << std::fixed << std::setprecision(6) << s.rate << "\t" << s.ci_lower << "\t" << s.ci_upper << "\n";
    }

    out << "\n# Per-position posterior statistics (3' end G->A)\n";
    out << "position\tn_opportunities\tn_mismatches\trate\tci_lower\tci_upper\n";
    for (const auto &s : gm.stats_3p) {
        out << s.position << "\t" << s.n_opportunities << "\t" << s.n_mismatches << "\t"
            << std::fixed << std::setprecision(6) << s.rate << "\t" << s.ci_lower << "\t" << s.ci_upper << "\n";
    }

    out << "\n# Per-contig Bayesian model\n";
    out << "contig\tp_ancient\tlog_bf\tevidence\tamplitude_5p\tamplitude_3p\tbaseline\tis_ancient\n";
    for (const auto &[contig, model] : analysis.per_contig_model) {
        out << contig << "\t"
            << std::fixed << std::setprecision(4)
            << model.p_ancient << "\t"
            << model.log_bf << "\t"
            << model.evidence_strength << "\t"
            << model.curve_5p.amplitude << "\t"
            << model.curve_3p.amplitude << "\t"
            << model.curve_5p.baseline << "\t"
            << (model.is_significant ? "yes" : "no") << "\n";
    }
}

int run_polisher(int argc, char** argv) {
    PolishConfig config;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--contigs" && i + 1 < argc) {
            config.contigs_file = argv[++i];
        } else if (arg == "--bam" && i + 1 < argc) {
            config.bam_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--library-type" && i + 1 < argc) {
            config.library_type = argv[++i];
        } else if (arg == "--min-length" && i + 1 < argc) {
            config.min_length = std::stoi(argv[++i]);
        } else if (arg == "--min-mapq" && i + 1 < argc) {
            config.min_mapq = std::stoi(argv[++i]);
        } else if (arg == "--min-baseq" && i + 1 < argc) {
            config.min_baseq = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.threads = std::stoi(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--anvio") {
            config.anvio = true;
        } else if (arg == "--prefix" && i + 1 < argc) {
            config.anvio_prefix = argv[++i];
        } else if (arg == "--damage-positions" && i + 1 < argc) {
            config.damage_positions = std::stoi(argv[++i]);
        } else if (arg == "--min-posterior" && i + 1 < argc) {
            config.min_posterior = std::stod(argv[++i]);
        } else if (arg == "--max-second-posterior" && i + 1 < argc) {
            config.max_second_posterior = std::stod(argv[++i]);
        } else if (arg == "--no-bayesian") {
            config.use_bayesian = false;
        } else if (arg == "--write-calls") {
            config.write_calls = true;
        }
    }

    // Validate inputs
    if (config.contigs_file.empty() || config.bam_file.empty()) {
        std::cerr << "Error: --contigs and --bam are required\n";
        return 1;
    }

    // Validate library type
    if (config.library_type != "ds" && config.library_type != "ss") {
        std::cerr << "Error: --library-type must be 'ds' or 'ss'\n";
        std::cerr << "  ds = double-stranded library\n";
        std::cerr << "  ss = single-stranded library\n";
        std::cerr << "Use --damage-positions to set how many positions to correct (default: 15)\n";
        return 1;
    }

    // Create output directory
    std::filesystem::create_directories(config.output_dir);
    omp_set_num_threads(config.threads);

    // Set up logging
    Logger log("polish", VERSION);
    log.console_level = config.verbose ? Verbosity::Verbose : Verbosity::Normal;
    log.open_trace(config.output_dir + "/polish_trace.log");
    log.info("AMBER polish v" + std::string(VERSION) + " starting");

    // Load contigs FIRST to build filter set
    log.info("Loading contigs: " + config.contigs_file);
    std::vector<FASTASequence> contigs;
    FASTAReader reader(config.contigs_file);
    FASTASequence seq;
    std::unordered_set<std::string> contig_filter;

    while (reader.next(seq)) {
        if ((int)seq.sequence.length() >= config.min_length) {
            contigs.push_back(seq);
            contig_filter.insert(seq.id);
        }
    }

    PolishStats stats;
    stats.total_contigs = contigs.size();
    log.info("Loaded " + std::to_string(stats.total_contigs) + " contigs for polishing");

    // Compute damage profile from BAM - ONLY for contigs being polished
    DamageAnalysis analysis;
    std::string err;

    log.info("Computing damage profile from " + std::to_string(contig_filter.size()) + " contigs in BAM");
    DamageComputeConfig dc_config;
    dc_config.min_mapq = config.min_mapq;
    dc_config.min_baseq = config.min_baseq;
    dc_config.threads = config.threads;

    if (!compute_damage_analysis(config.bam_file, analysis, err, dc_config, &contig_filter)) {
        log.error("Failed to compute damage profile: " + err);
        return 1;
    }

    // Write computed profile
    write_damage_profile(config.output_dir + "/damage_profile.tsv", analysis);
    write_model_summary(config.output_dir + "/damage_model.tsv", analysis);

    const auto &gm = analysis.global_model;
    {
        std::ostringstream ss;
        ss << "Bayesian damage model:\n";
        ss << "  P(ancient|data)=" << std::fixed << std::setprecision(3) << gm.p_ancient << "\n";
        ss << "  log_BF=" << std::setprecision(2) << gm.log_bf;
        if (gm.evidence_strength >= 4.0) ss << " (decisive)";
        else if (gm.evidence_strength >= 3.0) ss << " (strong)";
        else if (gm.evidence_strength >= 2.0) ss << " (moderate)";
        else if (gm.evidence_strength >= 1.0) ss << " (weak)";
        else ss << " (none)";
        ss << "\n";
        ss << "  5' C->T: rate[1]=" << std::setprecision(4)
           << (gm.stats_5p.empty() ? 0.0 : gm.stats_5p[0].rate)
           << " [" << (gm.stats_5p.empty() ? 0.0 : gm.stats_5p[0].ci_lower)
           << ", " << (gm.stats_5p.empty() ? 1.0 : gm.stats_5p[0].ci_upper) << "]\n";
        ss << "  3' G->A: rate[1]=" << std::setprecision(4)
           << (gm.stats_3p.empty() ? 0.0 : gm.stats_3p[0].rate)
           << " [" << (gm.stats_3p.empty() ? 0.0 : gm.stats_3p[0].ci_lower)
           << ", " << (gm.stats_3p.empty() ? 1.0 : gm.stats_3p[0].ci_upper) << "]\n";
        ss << "  baseline=" << std::setprecision(4) << analysis.calibrated_error_rate
           << " [" << std::setprecision(4) << (analysis.calibrated_error_rate - 1.96*analysis.error_rate_se)
           << ", " << (analysis.calibrated_error_rate + 1.96*analysis.error_rate_se) << "]";
        log.info(ss.str());
    }

    log.info("Profile: " + std::to_string(analysis.profile.total_positions()) +
             " positions across " + std::to_string(analysis.profile.per_contig.size()) + " contigs");
    log.info("Library type: " + config.library_type);

    // Warn if auto-detected library type differs from user-specified
    if (!analysis.library_type.empty() && analysis.library_type != config.library_type) {
        log.warn("Auto-detected library type is '" + analysis.library_type +
                 "' but you specified '" + config.library_type + "'. Check your library preparation method.");
    }

    // Check if damage is significant enough to warrant correction
    if (!analysis.global_model.is_significant) {
        log.info(analysis.global_model.significance_reason);
        log.info("No corrections will be applied - sequences appear modern.");
    } else {
        log.info(analysis.global_model.significance_reason);
    }

    // Contigs already loaded above (before damage computation)

    // Determine correction positions based on library type
    int damage_positions = config.damage_positions;

    // Use per-position observed rates and credible intervals for correction decisions
    // This is more robust than fitted curve parameters
    double baseline = analysis.calibrated_error_rate;

    // Correction rule: Bayesian-inspired decision
    // Correct if: (1) observed rate significantly exceeds baseline AND
    //             (2) credible interval lower bound > 2x baseline
    // This ensures we only correct positions with strong evidence of damage

    std::vector<bool> correct_5p(damage_positions + 1, false);
    std::vector<bool> correct_3p(damage_positions + 1, false);
    std::vector<double> rate_5p(damage_positions + 1, 0.0);
    std::vector<double> rate_3p(damage_positions + 1, 0.0);
    int last_5p_pos = 0, last_3p_pos = 0;

    {
        std::ostringstream ss;
        ss << "Correction approach: credible interval-based\n";
        ss << "  Baseline (sequencing error): " << std::fixed << std::setprecision(4) << baseline;
        log.info(ss.str());
    }
    log.section("Per-position analysis");

    for (int z = 1; z <= std::min(damage_positions, (int)gm.stats_5p.size()); z++) {
        const auto& s5 = gm.stats_5p[z-1];
        const auto& s3 = gm.stats_3p[z-1];

        rate_5p[z] = s5.rate;
        rate_3p[z] = s3.rate;

        // Correct if CI lower bound exceeds baseline's upper CI bound
        // This is a data-driven threshold: position rate is significantly above background
        double baseline_upper = baseline + 1.96 * analysis.error_rate_se;
        bool sig_5p = (s5.ci_lower > baseline_upper);
        bool sig_3p = (s3.ci_lower > baseline_upper);

        if (sig_5p) {
            correct_5p[z] = true;
            last_5p_pos = z;
        }
        if (sig_3p) {
            correct_3p[z] = true;
            last_3p_pos = z;
        }

        if (z <= 5) {
            std::ostringstream ss;
            ss << "pos " << z << ": 5' rate=" << std::setprecision(4) << s5.rate
               << " [" << s5.ci_lower << "," << s5.ci_upper << "]"
               << (sig_5p ? " CORRECT" : "");
            log.detail(ss.str());
            ss.str("");
            ss << "       3' rate=" << s3.rate
               << " [" << s3.ci_lower << "," << s3.ci_upper << "]"
               << (sig_3p ? " CORRECT" : "");
            log.detail(ss.str());
        }
    }

    log.info("Correcting 5' positions: 1-" + std::to_string(last_5p_pos));
    log.info("Correcting 3' positions: 1-" + std::to_string(last_3p_pos));
    {
        std::ostringstream ss;
        ss << "Min log-LR threshold: " << config.min_log_lr
           << " (evidence ratio: " << std::setprecision(1) << std::exp(config.min_log_lr) << "x)";
        log.info(ss.str());
    }

    // Open BAM file with index for pileup queries
    samFile *sf = sam_open(config.bam_file.c_str(), "r");
    if (!sf) {
        log.error("Cannot open BAM file for pileup: " + config.bam_file);
        return 1;
    }

    bam_hdr_t *hdr = sam_hdr_read(sf);
    if (!hdr) {
        log.error("Cannot read BAM header");
        sam_close(sf);
        return 1;
    }

    hts_idx_t *idx = sam_index_load(sf, config.bam_file.c_str());
    if (!idx) {
        log.error("Cannot load BAM index. Run 'samtools index " + config.bam_file + "'");
        bam_hdr_destroy(hdr);
        sam_close(sf);
        return 1;
    }

    log.info("Bayesian correction using read pileup evidence...");
    if (config.use_bayesian) {
        log.info("Bayesian caller: min_posterior=" + std::to_string(config.min_posterior) +
                 ", max_second=" + std::to_string(config.max_second_posterior));
    } else {
        log.info("Using legacy LR mode (--no-bayesian)");
    }
    log.info("Parallel processing with " + std::to_string(config.threads) + " threads");

    // Per-contig results for parallel processing
    struct ContigResult {
        std::string corrected;
        std::string output_name;
        size_t ct_corr = 0;
        size_t ga_corr = 0;
        size_t positions_examined = 0;
        size_t positions_with_evidence = 0;
        size_t skipped_low_coverage = 0;
        size_t skipped_ambiguous = 0;
    };
    std::vector<ContigResult> results(contigs.size());

    // Initialize renamer if anvio mode is enabled
    // Pre-compute output names BEFORE parallel region (renamer is not thread-safe)
    ContigRenamer renamer(config.anvio_prefix);
    if (config.anvio) {
        log.info("Anvi'o mode enabled: renaming contigs with prefix '" + config.anvio_prefix + "'");
    }

    // Pre-compute output names and initialize results (sequential, thread-safe)
    for (size_t ci = 0; ci < contigs.size(); ci++) {
        results[ci].corrected = contigs[ci].sequence;
        results[ci].output_name = config.anvio ?
            renamer.rename(contigs[ci].id, contigs[ci].sequence.length()) : contigs[ci].id;
    }

    // Close shared BAM handles - each thread will open its own
    hts_idx_destroy(idx);
    bam_hdr_destroy(hdr);
    sam_close(sf);

    // Parallel processing: each thread opens its own BAM file handle
    #pragma omp parallel num_threads(config.threads)
    {
        // Thread-local BAM handles
        samFile *t_sf = sam_open(config.bam_file.c_str(), "r");
        bam_hdr_t *t_hdr = sam_hdr_read(t_sf);
        hts_idx_t *t_idx = sam_index_load(t_sf, config.bam_file.c_str());

        // Configure the shared polish engine
        PolishEngineConfig engine_config;
        engine_config.max_passes = 1;
        engine_config.min_log_lr = config.min_log_lr;
        engine_config.damage_positions = damage_positions;
        engine_config.min_mapq = config.min_mapq;
        engine_config.min_baseq = config.min_baseq;
        engine_config.use_interior_consensus = true;   // Use interior consensus (legacy mode)
        engine_config.use_dynamic_zone = true;         // Use dynamic damage zone

        // Bayesian genotype caller settings
        engine_config.use_bayesian_caller = config.use_bayesian;
        engine_config.min_posterior = config.min_posterior;
        engine_config.max_second_posterior = config.max_second_posterior;
        engine_config.write_detailed_calls = config.write_calls;

        #pragma omp for schedule(dynamic, 100)
        for (size_t ci = 0; ci < contigs.size(); ci++) {
            const auto& contig = contigs[ci];
            ContigResult& res = results[ci];

            if (!analysis.global_model.is_significant) continue;

            // Use the shared polish engine with multi-pass + consensus
            PolishResult pr = polish_contig(
                res.corrected,
                contig.id,
                t_sf, t_hdr, t_idx,
                gm,
                engine_config
            );

            res.ct_corr = pr.ct_corrections;
            res.ga_corr = pr.ga_corrections;
            res.positions_examined = pr.positions_examined;
            res.positions_with_evidence = pr.positions_with_evidence;
            res.skipped_low_coverage = pr.skipped_low_coverage;
            res.skipped_ambiguous = pr.skipped_ambiguous;
        }

        // Cleanup thread-local handles
        hts_idx_destroy(t_idx);
        bam_hdr_destroy(t_hdr);
        sam_close(t_sf);
    }

    // Write results sequentially (preserves order)
    std::ofstream polished_out(config.output_dir + "/polished.fa");
    std::ofstream stats_out(config.output_dir + "/polish_stats.tsv");
    stats_out << "contig_name\tlength\tct_corrections\tga_corrections\ttotal_edits\n";

    for (size_t ci = 0; ci < contigs.size(); ci++) {
        const auto& res = results[ci];
        stats.ct_corrections += res.ct_corr;
        stats.ga_corrections += res.ga_corr;
        stats.total_edits += res.ct_corr + res.ga_corr;
        stats.positions_examined += res.positions_examined;
        stats.positions_with_evidence += res.positions_with_evidence;
        stats.skipped_low_coverage += res.skipped_low_coverage;
        stats.skipped_ambiguous += res.skipped_ambiguous;

        polished_out << ">" << res.output_name << "\n";
        for (size_t i = 0; i < res.corrected.length(); i += 80) {
            polished_out << res.corrected.substr(i, 80) << "\n";
        }

        stats_out << res.output_name << "\t" << contigs[ci].sequence.length() << "\t"
                  << res.ct_corr << "\t" << res.ga_corr << "\t" << (res.ct_corr + res.ga_corr) << "\n";
    }
    stats.processed_contigs = contigs.size();

    // Write summary
    log.section("Summary");
    log.metric("Total contigs", (int)stats.total_contigs);
    log.metric("Processed contigs", (int)stats.processed_contigs);
    log.metric("Positions examined", (int)stats.positions_examined);
    log.metric("Positions with pileup evidence", (int)stats.positions_with_evidence);
    log.metric("Skipped (low coverage <3)", (int)stats.skipped_low_coverage);
    log.metric("Skipped (ambiguous LR)", (int)stats.skipped_ambiguous);
    log.metric("Total corrections", (int)stats.total_edits);
    log.metric("C->T corrections (5' end)", (int)stats.ct_corrections);
    log.metric("G->A corrections (3' end)", (int)stats.ga_corrections);

    // Write rename report if anvio mode is enabled
    if (config.anvio) {
        std::string report_path = config.output_dir + "/rename_report.tsv";
        renamer.write_report(report_path);
        log.info("Wrote rename report: " + report_path);
    }

    log.info("Output: " + config.output_dir + "/");

    return 0;
}

// =============================================================================
// DECONVOLVE COMMAND
// Separates ancient and modern DNA populations
// =============================================================================

struct DeconvolveConfig {
    std::string contigs_file;
    std::string bam_file;
    std::string output_dir = ".";
    std::string library_type = "ds";
    int min_length = 0;
    int min_mapq = 30;
    int min_baseq = 20;
    int damage_positions = DEFAULT_DAMAGE_POSITIONS;
    int threads = 1;
    int em_iterations = 10;
    double min_ancient_depth = 3.0;
    double min_modern_depth = 2.0;
    bool use_full_likelihood = true;
    bool polish_ancient = true;
    bool verbose = false;

    // Length model options
    bool use_length_model = true;  // Default: on (damage + length classification)
    double length_weight = 0.5;
    double mode_length_ancient = 42.0;
    double std_length_ancient = 15.0;
    double mode_length_modern = 103.0;
    double std_length_modern = 30.0;

    // Identity model options
    bool use_identity_model = false;
    double identity_weight = 1.0;
    double mode_identity_ancient = 100.0;
    double std_identity_ancient = 2.0;
    double mode_identity_modern = 99.0;
    double std_identity_modern = 0.5;

    // Auto-estimation
    bool auto_estimate_params = true;

    // Prior probability
    double prior_ancient = 0.5;  // Default: uninformative

    // Per-position uncertainty output
    bool write_position_stats = false;
};

int run_deconvolver(int argc, char** argv) {
    DeconvolveConfig config;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--contigs" && i + 1 < argc) {
            config.contigs_file = argv[++i];
        } else if (arg == "--bam" && i + 1 < argc) {
            config.bam_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--library-type" && i + 1 < argc) {
            config.library_type = argv[++i];
        } else if (arg == "--min-length" && i + 1 < argc) {
            config.min_length = std::stoi(argv[++i]);
        } else if (arg == "--min-mapq" && i + 1 < argc) {
            config.min_mapq = std::stoi(argv[++i]);
        } else if (arg == "--min-baseq" && i + 1 < argc) {
            config.min_baseq = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.threads = std::stoi(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--damage-positions" && i + 1 < argc) {
            config.damage_positions = std::stoi(argv[++i]);
        } else if (arg == "--em-iterations" && i + 1 < argc) {
            config.em_iterations = std::stoi(argv[++i]);
        } else if (arg == "--min-ancient-depth" && i + 1 < argc) {
            config.min_ancient_depth = std::stod(argv[++i]);
        } else if (arg == "--min-modern-depth" && i + 1 < argc) {
            config.min_modern_depth = std::stod(argv[++i]);
        } else if (arg == "--no-full-likelihood") {
            config.use_full_likelihood = false;
        } else if (arg == "--no-polish") {
            config.polish_ancient = false;
        } else if (arg == "--no-length-model") {
            config.use_length_model = false;
        } else if (arg == "--length-weight" && i + 1 < argc) {
            config.length_weight = std::stod(argv[++i]);
        } else if (arg == "--mode-length-ancient" && i + 1 < argc) {
            config.mode_length_ancient = std::stod(argv[++i]);
        } else if (arg == "--std-length-ancient" && i + 1 < argc) {
            config.std_length_ancient = std::stod(argv[++i]);
        } else if (arg == "--mode-length-modern" && i + 1 < argc) {
            config.mode_length_modern = std::stod(argv[++i]);
        } else if (arg == "--std-length-modern" && i + 1 < argc) {
            config.std_length_modern = std::stod(argv[++i]);
        } else if (arg == "--use-identity-model") {
            config.use_identity_model = true;
        } else if (arg == "--identity-weight" && i + 1 < argc) {
            config.identity_weight = std::stod(argv[++i]);
        } else if (arg == "--mode-identity-ancient" && i + 1 < argc) {
            config.mode_identity_ancient = std::stod(argv[++i]);
        } else if (arg == "--std-identity-ancient" && i + 1 < argc) {
            config.std_identity_ancient = std::stod(argv[++i]);
        } else if (arg == "--mode-identity-modern" && i + 1 < argc) {
            config.mode_identity_modern = std::stod(argv[++i]);
        } else if (arg == "--std-identity-modern" && i + 1 < argc) {
            config.std_identity_modern = std::stod(argv[++i]);
        } else if (arg == "--auto-estimate") {
            config.auto_estimate_params = true;
        } else if (arg == "--prior" && i + 1 < argc) {
            config.prior_ancient = std::stod(argv[++i]);
        } else if (arg == "--write-position-stats") {
            config.write_position_stats = true;
        }
    }

    // Validate inputs
    if (config.contigs_file.empty() || config.bam_file.empty()) {
        std::cerr << "Error: --contigs and --bam are required\n";
        return 1;
    }

    // Validate library type
    if (config.library_type != "ds" && config.library_type != "ss") {
        std::cerr << "Error: --library-type must be 'ds' or 'ss'\n";
        return 1;
    }

    // Create output directory
    std::filesystem::create_directories(config.output_dir);
    omp_set_num_threads(config.threads);

    // Set up logging
    Logger log("deconvolve", VERSION);
    log.console_level = config.verbose ? Verbosity::Verbose : Verbosity::Normal;
    log.open_trace(config.output_dir + "/deconvolve_trace.log");
    log.info("AMBER deconvolve v" + std::string(VERSION) + " starting");

    // Load contigs
    log.info("Loading contigs: " + config.contigs_file);
    std::vector<FASTASequence> contigs;
    FASTAReader reader(config.contigs_file);
    FASTASequence seq;
    std::unordered_set<std::string> contig_filter;

    while (reader.next(seq)) {
        if ((int)seq.sequence.length() >= config.min_length) {
            contigs.push_back(seq);
            contig_filter.insert(seq.id);
        }
    }
    log.info("Loaded " + std::to_string(contigs.size()) + " contigs");

    // Compute damage profile from BAM
    DamageAnalysis analysis;
    std::string err;

    log.info("Computing damage profile from BAM");
    DamageComputeConfig dc_config;
    dc_config.min_mapq = config.min_mapq;
    dc_config.min_baseq = config.min_baseq;
    dc_config.max_position = config.damage_positions;
    dc_config.threads = config.threads;

    if (!compute_damage_analysis(config.bam_file, analysis, err, dc_config, &contig_filter)) {
        log.error("Failed to compute damage profile: " + err);
        return 1;
    }

    // Fit global Bayesian model
    const auto& gm = analysis.global_model;

    // Determine damage positions to use
    int damage_positions = config.damage_positions;
    if (config.library_type == "ss") {
        damage_positions = std::min(damage_positions, 3);
    }

    // Log damage model
    log.info("Profile: " + std::to_string(analysis.profile.total_positions()) +
             " positions across " + std::to_string(analysis.profile.per_contig.size()) + " contigs");
    log.info("Library type: " + config.library_type);
    if (config.verbose) {
        log.info("Ancient: P=" + std::to_string(gm.p_ancient) +
                 ", amp=" + std::to_string(gm.curve_5p.amplitude * 100) +
                 "%, λ=" + std::to_string(gm.curve_5p.lambda) +
                 ", score=" + std::to_string(gm.log_bf));
    }

    // Write damage profile
    write_damage_profile(config.output_dir + "/damage_profile.tsv", analysis);
    write_model_summary(config.output_dir + "/damage_model.tsv", analysis);

    log.info("Damage-based deconvolution: separating damaged vs undamaged reads");
    log.info("  Full read likelihood: " + std::string(config.use_full_likelihood ? "yes" : "no"));
    log.info("  EM iterations: " + std::to_string(config.em_iterations));
    log.info("  Prior P(ancient): " + std::to_string(config.prior_ancient));
    log.info("  Polish ancient: " + std::string(config.polish_ancient ? "yes" : "no"));

    struct DeconvContigResult {
        DeconvolutionResult deconv;
        std::string contig_name;
    };
    std::vector<DeconvContigResult> results(contigs.size());

    // Initialize output names
    for (size_t ci = 0; ci < contigs.size(); ci++) {
        results[ci].contig_name = contigs[ci].id;
    }

    // Parallel deconvolution
    #pragma omp parallel num_threads(config.threads)
    {
        samFile *t_sf = sam_open(config.bam_file.c_str(), "r");
        bam_hdr_t *t_hdr = sam_hdr_read(t_sf);
        hts_idx_t *t_idx = sam_index_load(t_sf, config.bam_file.c_str());

        DeconvolutionConfig deconv_config;
        deconv_config.min_ancient_depth = config.min_ancient_depth;
        deconv_config.min_modern_depth = config.min_modern_depth;
        deconv_config.min_mapq = config.min_mapq;
        deconv_config.min_baseq = config.min_baseq;
        deconv_config.polish_ancient = config.polish_ancient;
        deconv_config.em_iterations = config.em_iterations;
        deconv_config.use_full_likelihood = config.use_full_likelihood;
        deconv_config.use_length_model = config.use_length_model;
        deconv_config.length_weight = config.length_weight;
        deconv_config.mode_length_ancient = config.mode_length_ancient;
        deconv_config.std_length_ancient = config.std_length_ancient;
        deconv_config.mode_length_modern = config.mode_length_modern;
        deconv_config.std_length_modern = config.std_length_modern;

        deconv_config.use_identity_model = config.use_identity_model;
        deconv_config.identity_weight = config.identity_weight;
        deconv_config.mode_identity_ancient = config.mode_identity_ancient;
        deconv_config.std_identity_ancient = config.std_identity_ancient;
        deconv_config.mode_identity_modern = config.mode_identity_modern;
        deconv_config.std_identity_modern = config.std_identity_modern;
        deconv_config.auto_estimate_params = config.auto_estimate_params;
        deconv_config.prior_ancient = config.prior_ancient;
        deconv_config.write_position_stats = config.write_position_stats;

        #pragma omp for schedule(dynamic, 50)
        for (size_t ci = 0; ci < contigs.size(); ci++) {
            results[ci].deconv = deconvolve_contig(
                contigs[ci].sequence,
                contigs[ci].id,
                t_sf, t_hdr, t_idx,
                gm,
                deconv_config
            );
        }

        hts_idx_destroy(t_idx);
        bam_hdr_destroy(t_hdr);
        sam_close(t_sf);
    }

    // Write results
    std::ofstream ancient_out(config.output_dir + "/ancient_consensus.fa");
    std::ofstream modern_out(config.output_dir + "/modern_consensus.fa");
    std::ofstream stats_out(config.output_dir + "/deconvolution_stats.tsv");

    std::ofstream pos_stats_out, mask_out;
    if (config.write_position_stats) {
        pos_stats_out.open(config.output_dir + "/position_stats.tsv");
        pos_stats_out << "contig\tpos\tancient_base\tmodern_base\t"
                         "post_anc\tpost_mod\teff_depth_anc\teff_depth_mod\t"
                         "p_diff\tflags\n";
        mask_out.open(config.output_dir + "/high_confidence_mask.bed");
    }

    stats_out << "contig\tlength\ttotal_reads\tancient_reads\tmodern_reads\t"
              << "ancient_fraction\tmean_p_ancient\tclassification_confidence\t"
              << "positions_both\tpositions_ancient_only\tpositions_modern_only\t"
              << "ancient_modern_diffs\tancient_ct_corr\tancient_ga_corr\t"
              << "f_undamaged\tf_damaged\tf_undamaged_se\tf_undamaged_ci_lower\tf_undamaged_ci_upper\t"
              << "f_expected\tf_observed\tf_undamaged_moment\n";

    size_t total_ancient = 0, total_modern = 0, total_diffs = 0;
    size_t total_ct_corr = 0, total_ga_corr = 0;

    // Smiley plot data accumulators (ancient p>=0.8, modern p<=0.2, ambiguous in between)
    std::array<int64_t, DeconvolutionResult::SMILEY_MAX_POS> smiley_anc_t_5p = {};
    std::array<int64_t, DeconvolutionResult::SMILEY_MAX_POS> smiley_anc_tc_5p = {};
    std::array<int64_t, DeconvolutionResult::SMILEY_MAX_POS> smiley_anc_a_3p = {};
    std::array<int64_t, DeconvolutionResult::SMILEY_MAX_POS> smiley_anc_ag_3p = {};
    std::array<int64_t, DeconvolutionResult::SMILEY_MAX_POS> smiley_mod_t_5p = {};
    std::array<int64_t, DeconvolutionResult::SMILEY_MAX_POS> smiley_mod_tc_5p = {};
    std::array<int64_t, DeconvolutionResult::SMILEY_MAX_POS> smiley_mod_a_3p = {};
    std::array<int64_t, DeconvolutionResult::SMILEY_MAX_POS> smiley_mod_ag_3p = {};
    std::array<int64_t, DeconvolutionResult::SMILEY_MAX_POS> smiley_amb_t_5p = {};
    std::array<int64_t, DeconvolutionResult::SMILEY_MAX_POS> smiley_amb_tc_5p = {};
    std::array<int64_t, DeconvolutionResult::SMILEY_MAX_POS> smiley_amb_a_3p = {};
    std::array<int64_t, DeconvolutionResult::SMILEY_MAX_POS> smiley_amb_ag_3p = {};

    // Probability histogram accumulators
    std::array<int64_t, DeconvolutionResult::HIST_BINS> p_ancient_hist = {};   // Combined
    std::array<int64_t, DeconvolutionResult::HIST_BINS> p_damaged_hist = {};   // Damage-only
    std::array<int64_t, DeconvolutionResult::HIST_BINS> p_length_hist = {};    // Length-only

    // Read length histogram accumulators by classification
    std::array<int64_t, DeconvolutionResult::LENGTH_BINS> length_hist_ancient = {};
    std::array<int64_t, DeconvolutionResult::LENGTH_BINS> length_hist_modern = {};
    std::array<int64_t, DeconvolutionResult::LENGTH_BINS> length_hist_ambiguous = {};

    for (size_t ci = 0; ci < contigs.size(); ci++) {
        const auto& dr = results[ci].deconv;
        const auto& name = results[ci].contig_name;

        // Write ancient consensus
        ancient_out << ">" << name << "\n";
        for (size_t i = 0; i < dr.ancient_consensus.length(); i += 80) {
            ancient_out << dr.ancient_consensus.substr(i, 80) << "\n";
        }

        // Write modern consensus
        modern_out << ">" << name << "\n";
        for (size_t i = 0; i < dr.modern_consensus.length(); i += 80) {
            modern_out << dr.modern_consensus.substr(i, 80) << "\n";
        }

        // Write stats
        stats_out << name << "\t" << contigs[ci].sequence.length() << "\t"
                  << dr.total_reads << "\t" << dr.ancient_reads_hard << "\t"
                  << dr.modern_reads_hard << "\t"
                  << std::fixed << std::setprecision(4) << dr.ancient_fraction << "\t"
                  << dr.mean_p_ancient << "\t" << dr.classification_confidence << "\t"
                  << dr.positions_both_called << "\t" << dr.positions_ancient_only << "\t"
                  << dr.positions_modern_only << "\t" << dr.ancient_modern_differences << "\t"
                  << dr.ancient_ct_corrections << "\t" << dr.ancient_ga_corrections << "\t"
                  << dr.f_undamaged << "\t" << dr.f_damaged << "\t" << dr.f_undamaged_se << "\t"
                  << dr.f_undamaged_ci_lower << "\t" << dr.f_undamaged_ci_upper << "\t"
                  << dr.f_expected_detection << "\t" << dr.f_observed_detection << "\t"
                  << dr.f_undamaged_moment << "\n";

        total_ancient += dr.ancient_reads_hard;
        total_modern += dr.modern_reads_hard;
        total_diffs += dr.ancient_modern_differences;
        total_ct_corr += dr.ancient_ct_corrections;
        total_ga_corr += dr.ancient_ga_corrections;

        if (config.write_position_stats) {
            for (const auto& ps : dr.position_stats) {
                bool anc_fb = ps.flags & 1, mod_fb = (ps.flags >> 1) & 1;
                pos_stats_out << name << "\t" << (ps.pos + 1) << "\t"
                              << ps.ancient_base << "\t" << ps.modern_base << "\t"
                              << std::fixed << std::setprecision(4)
                              << ps.post_anc << "\t" << ps.post_mod << "\t"
                              << ps.eff_depth_anc << "\t" << ps.eff_depth_mod << "\t"
                              << ps.p_diff << "\t"
                              << (int)ps.flags << "\n";
                if (ps.ancient_base != ps.modern_base && (anc_fb || mod_fb || ps.p_diff < 0.5f)) {
                    mask_out << name << "\t" << ps.pos << "\t" << (ps.pos + 1) << "\n";
                }
            }
        }

        // Accumulate smiley data
        for (int d = 0; d < DeconvolutionResult::SMILEY_MAX_POS; d++) {
            smiley_anc_t_5p[d] += dr.anc_t_5p[d];
            smiley_anc_tc_5p[d] += dr.anc_tc_5p[d];
            smiley_anc_a_3p[d] += dr.anc_a_3p[d];
            smiley_anc_ag_3p[d] += dr.anc_ag_3p[d];
            smiley_mod_t_5p[d] += dr.mod_t_5p[d];
            smiley_mod_tc_5p[d] += dr.mod_tc_5p[d];
            smiley_mod_a_3p[d] += dr.mod_a_3p[d];
            smiley_mod_ag_3p[d] += dr.mod_ag_3p[d];
            smiley_amb_t_5p[d] += dr.amb_t_5p[d];
            smiley_amb_tc_5p[d] += dr.amb_tc_5p[d];
            smiley_amb_a_3p[d] += dr.amb_a_3p[d];
            smiley_amb_ag_3p[d] += dr.amb_ag_3p[d];
        }

        // Accumulate probability histograms
        for (int b = 0; b < DeconvolutionResult::HIST_BINS; b++) {
            p_ancient_hist[b] += dr.p_ancient_hist[b];
            p_damaged_hist[b] += dr.p_damaged_hist[b];
            p_length_hist[b] += dr.p_length_hist[b];
        }

        // Accumulate read length histograms
        for (int b = 0; b < DeconvolutionResult::LENGTH_BINS; b++) {
            length_hist_ancient[b] += dr.length_hist_ancient[b];
            length_hist_modern[b] += dr.length_hist_modern[b];
            length_hist_ambiguous[b] += dr.length_hist_ambiguous[b];
        }
    }

    // Compute weighted mean fragment lengths from per-class histograms
    auto hist_mean_std = [](const auto& hist, int n_bins, double bin_width) -> std::pair<double, double> {
        double sum = 0.0, sum_sq = 0.0, count = 0.0;
        for (int b = 0; b < n_bins; b++) {
            double len = (b + 0.5) * bin_width;
            double n = static_cast<double>(hist[b]);
            sum    += n * len;
            sum_sq += n * len * len;
            count  += n;
        }
        if (count < 1.0) return {0.0, 0.0};
        double mean = sum / count;
        double var  = sum_sq / count - mean * mean;
        return {mean, std::sqrt(std::max(0.0, var))};
    };
    auto [frag_mean_ancient, frag_std_ancient] =
        hist_mean_std(length_hist_ancient,   DeconvolutionResult::LENGTH_BINS, 5.0);
    auto [frag_mean_modern,  frag_std_modern] =
        hist_mean_std(length_hist_modern,    DeconvolutionResult::LENGTH_BINS, 5.0);
    auto [frag_mean_ambig,   frag_std_ambig] =
        hist_mean_std(length_hist_ambiguous, DeconvolutionResult::LENGTH_BINS, 5.0);

    // Write smiley plot data (ancient p>=0.8, modern p<=0.2, ambiguous 0.2<p<0.8)
    std::ofstream smiley_out(config.output_dir + "/smiley_data.tsv");
    smiley_out << "pos\tanc_5p\tanc_3p\tmod_5p\tmod_3p\tamb_5p\tamb_3p\t"
               << "anc_t_5p\tanc_tc_5p\tanc_a_3p\tanc_ag_3p\t"
               << "mod_t_5p\tmod_tc_5p\tmod_a_3p\tmod_ag_3p\t"
               << "amb_t_5p\tamb_tc_5p\tamb_a_3p\tamb_ag_3p\n";
    for (int d = 0; d < DeconvolutionResult::SMILEY_MAX_POS; d++) {
        auto ratio = [](int64_t num, int64_t denom) {
            return denom > 0 ? double(num) / denom : 0.5;
        };

        smiley_out << (d + 1) << "\t"
                   << ratio(smiley_anc_t_5p[d], smiley_anc_tc_5p[d]) << "\t"
                   << ratio(smiley_anc_a_3p[d], smiley_anc_ag_3p[d]) << "\t"
                   << ratio(smiley_mod_t_5p[d], smiley_mod_tc_5p[d]) << "\t"
                   << ratio(smiley_mod_a_3p[d], smiley_mod_ag_3p[d]) << "\t"
                   << ratio(smiley_amb_t_5p[d], smiley_amb_tc_5p[d]) << "\t"
                   << ratio(smiley_amb_a_3p[d], smiley_amb_ag_3p[d]) << "\t"
                   << smiley_anc_t_5p[d] << "\t" << smiley_anc_tc_5p[d] << "\t"
                   << smiley_anc_a_3p[d] << "\t" << smiley_anc_ag_3p[d] << "\t"
                   << smiley_mod_t_5p[d] << "\t" << smiley_mod_tc_5p[d] << "\t"
                   << smiley_mod_a_3p[d] << "\t" << smiley_mod_ag_3p[d] << "\t"
                   << smiley_amb_t_5p[d] << "\t" << smiley_amb_tc_5p[d] << "\t"
                   << smiley_amb_a_3p[d] << "\t" << smiley_amb_ag_3p[d] << "\n";
    }
    smiley_out.close();

    // Write probability histograms (combined, damage-only, length-only)
    std::ofstream hist_out(config.output_dir + "/p_ancient_histogram.tsv");
    hist_out << "bin_start\tbin_end\tp_combined\tp_damaged\tp_length\n";
    for (int b = 0; b < DeconvolutionResult::HIST_BINS; b++) {
        double bin_start = b * 0.05;
        double bin_end = (b + 1) * 0.05;
        hist_out << std::fixed << std::setprecision(2)
                 << bin_start << "\t" << bin_end << "\t"
                 << p_ancient_hist[b] << "\t"
                 << p_damaged_hist[b] << "\t"
                 << p_length_hist[b] << "\n";
    }
    hist_out.close();

    // Write read length histograms
    std::ofstream len_hist_out(config.output_dir + "/read_length_histogram.tsv");
    len_hist_out << "length_min\tlength_max\tancient\tmodern\tambiguous\n";
    for (int b = 0; b < DeconvolutionResult::LENGTH_BINS; b++) {
        int len_min = b * 5;
        int len_max = (b + 1) * 5;
        len_hist_out << len_min << "\t" << len_max << "\t"
                     << length_hist_ancient[b] << "\t"
                     << length_hist_modern[b] << "\t"
                     << length_hist_ambiguous[b] << "\n";
    }
    len_hist_out.close();

    // Compute dynamic thresholds from p_ancient histogram
    auto [dyn_modern, dyn_ancient] = compute_dynamic_thresholds(p_ancient_hist);

    // Count reads using dynamic thresholds
    int64_t dyn_ancient_count = 0, dyn_modern_count = 0, dyn_ambiguous_count = 0;
    for (int b = 0; b < DeconvolutionResult::HIST_BINS; b++) {
        double bin_mid = (b + 0.5) * 0.05;
        if (bin_mid >= dyn_ancient) {
            dyn_ancient_count += p_ancient_hist[b];
        } else if (bin_mid <= dyn_modern) {
            dyn_modern_count += p_ancient_hist[b];
        } else {
            dyn_ambiguous_count += p_ancient_hist[b];
        }
    }

    // Write dynamic thresholds to file
    std::ofstream thresh_out(config.output_dir + "/dynamic_thresholds.tsv");
    thresh_out << "parameter\tvalue\n";
    thresh_out << "modern_threshold\t" << std::fixed << std::setprecision(3) << dyn_modern << "\n";
    thresh_out << "ancient_threshold\t" << dyn_ancient << "\n";
    thresh_out << "modern_reads\t" << dyn_modern_count << "\n";
    thresh_out << "ancient_reads\t" << dyn_ancient_count << "\n";
    thresh_out << "ambiguous_reads\t" << dyn_ambiguous_count << "\n";
    thresh_out.close();

    // Summary
    log.section("Summary");
    log.metric("Total contigs", (int)contigs.size());
    log.metric("Ancient reads (p >= 0.8)", (int)total_ancient);
    log.metric("Modern reads (p <= 0.2)", (int)total_modern);
    log.info("Dynamic thresholds (from p_ancient distribution):");
    log.metric("  Modern threshold", dyn_modern);
    log.metric("  Ancient threshold", dyn_ancient);
    log.metric("  Ancient reads (dynamic)", (int)dyn_ancient_count);
    log.metric("  Modern reads (dynamic)", (int)dyn_modern_count);
    log.metric("  Ambiguous reads (dynamic)", (int)dyn_ambiguous_count);
    log.metric("Positions where ancient != modern", (int)total_diffs);
    log.info("Fragment length (mean ± std):");
    log.info("  Ancient: " + std::to_string(frag_mean_ancient).substr(0,5)
             + " ± " + std::to_string(frag_std_ancient).substr(0,5) + " bp");
    log.info("  Modern:  " + std::to_string(frag_mean_modern).substr(0,5)
             + " ± " + std::to_string(frag_std_modern).substr(0,5) + " bp");
    log.info("  Ambiguous: " + std::to_string(frag_mean_ambig).substr(0,5)
             + " ± " + std::to_string(frag_std_ambig).substr(0,5) + " bp");
    if (config.polish_ancient) {
        log.metric("Ancient C->T corrections", (int)total_ct_corr);
        log.metric("Ancient G->A corrections", (int)total_ga_corr);
    }
    log.info("Output: " + config.output_dir + "/ancient_consensus.fa");
    log.info("Output: " + config.output_dir + "/modern_consensus.fa");
    log.info("Output: " + config.output_dir + "/smiley_data.tsv");

    return 0;
}

}  // namespace amber

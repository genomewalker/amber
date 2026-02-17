// AMBER - chimera.cpp
// Chimera and misassembly detection module
// Uses multi-signal analysis with fractal corroboration

#include "algorithms/chimera_detector.h"
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
#include <chrono>
#include <omp.h>

namespace amber {

struct ContigChimeraResult {
    std::string contig_id;
    int contig_length;
    std::vector<ChimeraSignal> breakpoints;
    bool is_chimeric;
    double max_confidence;
    std::string primary_evidence;
};

void write_chimera_tsv(const std::vector<ContigChimeraResult>& results,
                       const std::string& output_path) {
    std::ofstream out(output_path);
    out << "contig_id\tposition\tconfidence\tprimary_evidence\t"
        << "gc_shift\tcoverage_jump\tas_drop\tnm_spike\tdamage_shift\t"
        << "hurst_jump\tlz_jump\tcgr_shift\n";

    for (const auto& result : results) {
        for (const auto& bp : result.breakpoints) {
            out << result.contig_id << "\t"
                << bp.position << "\t"
                << std::fixed << std::setprecision(4) << bp.confidence << "\t"
                << bp.primary_evidence << "\t"
                << std::setprecision(4) << bp.gc_shift << "\t"
                << bp.coverage_jump << "\t"
                << bp.alignment_score_drop << "\t"
                << bp.edit_distance_spike << "\t"
                << bp.damage_shift << "\t"
                << bp.hurst_discontinuity << "\t"
                << bp.lz_discontinuity << "\t"
                << bp.cgr_shift << "\n";
        }
    }
}

void write_contig_flags_tsv(const std::vector<ContigChimeraResult>& results,
                            const std::string& output_path) {
    std::ofstream out(output_path);
    out << "contig_id\tlength\tis_chimeric\tnum_breakpoints\tmax_confidence\tprimary_evidence\n";

    for (const auto& result : results) {
        out << result.contig_id << "\t"
            << result.contig_length << "\t"
            << (result.is_chimeric ? "true" : "false") << "\t"
            << result.breakpoints.size() << "\t"
            << std::fixed << std::setprecision(4) << result.max_confidence << "\t"
            << result.primary_evidence << "\n";
    }
}

// Write cleaned FASTA with chimeras handled according to action
void write_cleaned_fasta(const std::vector<FASTASequence>& contigs,
                         const std::vector<ContigChimeraResult>& results,
                         const std::string& output_path,
                         const std::string& action,  // split, remove, keep
                         int min_length,
                         Logger& log) {
    std::ofstream out(output_path);
    if (!out) {
        log.error("Failed to open output file: " + output_path);
        return;
    }

    size_t written_contigs = 0;
    size_t written_bp = 0;
    size_t split_contigs = 0;
    size_t removed_contigs = 0;

    for (size_t i = 0; i < contigs.size(); i++) {
        const auto& contig = contigs[i];
        const auto& result = results[i];

        if (action == "keep" || !result.is_chimeric) {
            // Write contig as-is
            out << ">" << contig.id << "\n";
            for (size_t j = 0; j < contig.sequence.length(); j += 80) {
                out << contig.sequence.substr(j, 80) << "\n";
            }
            written_contigs++;
            written_bp += contig.sequence.length();
        } else if (action == "remove") {
            // Skip chimeric contigs entirely
            removed_contigs++;
        } else if (action == "split") {
            // Split at breakpoints
            std::vector<size_t> positions;
            positions.push_back(0);
            for (const auto& bp : result.breakpoints) {
                positions.push_back(bp.position);
            }
            positions.push_back(contig.sequence.length());

            int part = 1;
            for (size_t p = 0; p + 1 < positions.size(); p++) {
                size_t start = positions[p];
                size_t end = positions[p + 1];
                size_t frag_len = end - start;

                // Only write fragments meeting minimum length
                if ((int)frag_len >= min_length) {
                    std::string frag_name = contig.id + "_" + std::to_string(part);
                    out << ">" << frag_name << "\n";
                    std::string frag_seq = contig.sequence.substr(start, frag_len);
                    for (size_t j = 0; j < frag_seq.length(); j += 80) {
                        out << frag_seq.substr(j, 80) << "\n";
                    }
                    written_contigs++;
                    written_bp += frag_len;
                    part++;
                }
            }
            if (part > 1) split_contigs++;
        }
    }

    log.info("Wrote " + std::to_string(written_contigs) + " contigs (" +
             std::to_string(written_bp / 1000000.0) + " Mbp) to " + output_path);
    if (removed_contigs > 0) {
        log.info("Removed " + std::to_string(removed_contigs) + " chimeric contigs");
    }
    if (split_contigs > 0) {
        log.info("Split " + std::to_string(split_contigs) + " chimeric contigs at breakpoints");
    }
}

void write_chimera_stats(const std::vector<ContigChimeraResult>& results,
                        const std::string& output_path,
                        double elapsed_seconds) {
    std::ofstream out(output_path);

    int total_contigs = results.size();
    int chimeric_contigs = 0;
    int total_breakpoints = 0;
    double sum_confidence = 0.0;

    // Count by evidence type
    int gc_primary = 0, cov_primary = 0, as_primary = 0, nm_primary = 0;
    int damage_primary = 0, hurst_primary = 0, lz_primary = 0, cgr_primary = 0;

    for (const auto& result : results) {
        if (result.is_chimeric) chimeric_contigs++;
        total_breakpoints += result.breakpoints.size();

        for (const auto& bp : result.breakpoints) {
            sum_confidence += bp.confidence;
            if (bp.primary_evidence == "gc_shift") gc_primary++;
            else if (bp.primary_evidence == "coverage_jump") cov_primary++;
            else if (bp.primary_evidence == "alignment_score") as_primary++;
            else if (bp.primary_evidence == "edit_distance") nm_primary++;
            else if (bp.primary_evidence == "damage_pattern") damage_primary++;
            else if (bp.primary_evidence == "hurst_discontinuity") hurst_primary++;
            else if (bp.primary_evidence == "lz_complexity") lz_primary++;
            else if (bp.primary_evidence == "cgr_shift") cgr_primary++;
        }
    }

    double avg_confidence = (total_breakpoints > 0) ? sum_confidence / total_breakpoints : 0.0;

    out << "metric\tvalue\n";
    out << "total_contigs\t" << total_contigs << "\n";
    out << "chimeric_contigs\t" << chimeric_contigs << "\n";
    out << "chimeric_rate\t" << std::fixed << std::setprecision(4)
        << (total_contigs > 0 ? (double)chimeric_contigs / total_contigs : 0.0) << "\n";
    out << "total_breakpoints\t" << total_breakpoints << "\n";
    out << "avg_confidence\t" << avg_confidence << "\n";
    out << "runtime_seconds\t" << std::setprecision(1) << elapsed_seconds << "\n";
    out << "\n";
    out << "primary_evidence_counts\n";
    out << "gc_shift\t" << gc_primary << "\n";
    out << "coverage_jump\t" << cov_primary << "\n";
    out << "alignment_score\t" << as_primary << "\n";
    out << "edit_distance\t" << nm_primary << "\n";
    out << "damage_pattern\t" << damage_primary << "\n";
    out << "hurst_discontinuity\t" << hurst_primary << "\n";
    out << "lz_complexity\t" << lz_primary << "\n";
    out << "cgr_shift\t" << cgr_primary << "\n";
}

int run_chimera_scanner(int argc, char** argv) {
    ChimeraConfig config;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--contigs" && i + 1 < argc) {
            config.contigs_file = argv[++i];
        } else if (arg == "--bam" && i + 1 < argc) {
            config.bam_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--min-length" && i + 1 < argc) {
            config.min_length = std::stoi(argv[++i]);
        } else if (arg == "--min-confidence" && i + 1 < argc) {
            config.min_confidence = std::stod(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.threads = std::stoi(argv[++i]);
        } else if (arg == "--library-type" && i + 1 < argc) {
            std::string lt = argv[++i];
            if (lt != "ds" && lt != "ss") {
                std::cerr << "Error: --library-type must be 'ds' or 'ss'\n";
                return 1;
            }
            config.library_type = lt;
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--chimera-action" && i + 1 < argc) {
            config.chimera_action = argv[++i];
        } else if (arg == "--no-fasta") {
            config.write_fasta = false;
        } else if (arg == "--damage-positions" && i + 1 < argc) {
            config.damage_positions = std::stoi(argv[++i]);
        }
    }

    // Validate inputs
    if (config.contigs_file.empty() || config.bam_file.empty()) {
        std::cerr << "Error: --contigs and --bam are required\n";
        return 1;
    }

    // Validate chimera action
    if (config.chimera_action != "split" && config.chimera_action != "remove" && config.chimera_action != "keep") {
        std::cerr << "Error: --chimera-action must be 'split', 'remove', or 'keep'\n";
        return 1;
    }

    // Create output directory
    std::filesystem::create_directories(config.output_dir);

    // Set threads
    omp_set_num_threads(config.threads);

    // Set up logging
    Logger log("chimera", VERSION);
    log.console_level = config.verbose ? Verbosity::Verbose : Verbosity::Normal;
    log.open_trace(config.output_dir + "/chimera_trace.log");
    log.info("AMBER chimera v" + std::string(VERSION) + " starting");

    log.info("Loading contigs from " + config.contigs_file);

    // Load contigs
    std::vector<FASTASequence> contigs;
    FASTAReader reader(config.contigs_file);
    FASTASequence seq;

    while (reader.next(seq)) {
        if ((int)seq.sequence.length() >= config.min_length) {
            contigs.push_back(seq);
        }
    }

    log.info("Loaded " + std::to_string(contigs.size()) + " contigs (>=" + std::to_string(config.min_length) + " bp)");
    log.info("Using " + std::to_string(config.threads) + " threads");

    // Check anvi'o compliance of input contig names
    bool all_anvio_compliant = true;
    int non_compliant_count = 0;
    for (const auto& c : contigs) {
        if (!is_anvio_compliant(c.id)) {
            all_anvio_compliant = false;
            non_compliant_count++;
        }
    }

    if (all_anvio_compliant) {
        log.info("All contig names are anvi'o compliant");
    } else {
        log.warn(std::to_string(non_compliant_count) + " contig names are NOT anvi'o compliant");
        log.warn("Use 'amber polish --anvio' first or 'anvi-script-reformat-fasta --simplify-names'");
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Open shared BAM state (one per thread for thread safety)
    std::vector<SharedBAMState> bam_states(config.threads);
    for (int t = 0; t < config.threads; t++) {
        if (!bam_states[t].open(config.bam_file)) {
            log.error("Failed to open BAM file for thread " + std::to_string(t));
            return 1;
        }
    }

    // Process contigs
    std::vector<ContigChimeraResult> results(contigs.size());

    log.info("Analyzing contigs for misassemblies...");

    int processed = 0;
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < contigs.size(); i++) {
        const auto& contig = contigs[i];

        ContigChimeraResult result;
        result.contig_id = contig.id;
        result.contig_length = contig.sequence.length();

        // Get thread-local BAM state
        int tid = omp_get_thread_num();
        SharedBAMState* bam_state = &bam_states[tid];

        // Run chimera detection with fractal analysis for biological specificity
        auto signals = ChimeraDetector::detect_chimeras_fast(
            contig.id, contig.sequence, bam_state, true, config.damage_positions);

        // Filter by confidence threshold
        for (const auto& sig : signals) {
            if (sig.confidence >= config.min_confidence) {
                result.breakpoints.push_back(sig);
            }
        }

        result.is_chimeric = !result.breakpoints.empty() &&
                            result.breakpoints[0].confidence >= config.min_confidence;
        result.max_confidence = result.breakpoints.empty() ? 0.0 :
                               result.breakpoints[0].confidence;
        result.primary_evidence = result.breakpoints.empty() ? "none" :
                                 result.breakpoints[0].primary_evidence;

        results[i] = result;

        #pragma omp atomic
        processed++;

        if (processed % 1000 == 0) {
            #pragma omp critical
            {
                log.progress("Contigs", processed, contigs.size());
            }
        }
    }

    log.progress("Contigs", contigs.size(), contigs.size());

    // Close BAM states
    for (auto& state : bam_states) {
        state.close();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double elapsed_seconds = duration.count() / 1000.0;

    // Count chimeric contigs
    int chimeric_count = 0;
    int total_breakpoints = 0;
    for (const auto& result : results) {
        if (result.is_chimeric) chimeric_count++;
        total_breakpoints += result.breakpoints.size();
    }

    {
        std::ostringstream ss;
        ss << "Found " << chimeric_count << " chimeric contigs ("
           << std::fixed << std::setprecision(1) << (100.0 * chimeric_count / contigs.size()) << "%)";
        log.info(ss.str());
    }
    log.info("Total breakpoints detected: " + std::to_string(total_breakpoints));

    // Write output files
    log.info("Writing results to " + config.output_dir + "/");

    write_chimera_tsv(results, config.output_dir + "/chimeras.tsv");
    write_contig_flags_tsv(results, config.output_dir + "/contig_flags.tsv");
    write_chimera_stats(results, config.output_dir + "/chimera_stats.tsv", elapsed_seconds);

    // Write cleaned FASTA if enabled
    if (config.write_fasta) {
        write_cleaned_fasta(contigs, results,
                           config.output_dir + "/cleaned.fa",
                           config.chimera_action,
                           config.min_length, log);
    }

    {
        std::ostringstream ss;
        ss << "Done in " << std::fixed << std::setprecision(1) << elapsed_seconds << " seconds";
        log.info(ss.str());
    }

    return 0;
}

}  // namespace amber

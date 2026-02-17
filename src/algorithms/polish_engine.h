#pragma once

// AMBER - Shared Polish Engine
// Damage-aware polishing using sweep-line algorithm with Bayesian correction
//
// Two modes:
// 1. Bayesian caller (default): Full 4-base posterior, requires 99% confidence
// 2. Legacy LR mode: Binary likelihood ratio test (historical behavior)

#include "bayesian_caller.h"
#include "damage_read_classifier.h"
#include "damage_likelihood.h"
#include "damage_profile.h"
#include <htslib/hts.h>
#include <htslib/sam.h>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <cstdint>
#include <unordered_map>

namespace amber {

// Compute effective damage zone from model parameters (Phase 3)
// Returns position where damage rate drops to 2x baseline
inline int effective_damage_zone(double amplitude, double lambda, double baseline) {
    if (amplitude <= baseline || lambda <= 0.0 || baseline <= 0.0) return 5;
    // D(z) = amplitude * exp(-lambda * (z-1)) + baseline = 2 * baseline
    // amplitude * exp(-lambda * (z-1)) = baseline
    // z = 1 + ln(amplitude / baseline) / lambda
    double z = 1.0 + std::log(amplitude / baseline) / lambda;
    int zone = static_cast<int>(std::ceil(z));
    return std::min(std::max(zone, 5), 30);  // Bounded [5, 30]
}

// Compute interior consensus log-likelihood ratio (Phase 2)
// Only uses reads where position is NOT in damage zone (ground truth signal)
template<typename ReadObs>
double compute_interior_consensus_lr(
    const std::vector<ReadObs>& observations,
    char test_base,   // 'C' for C>T correction, 'G' for G>A correction
    char alt_base,    // 'T' for C>T correction, 'A' for G>A correction
    int damage_positions) {

    double log_lik_test = 0.0, log_lik_alt = 0.0;
    int interior_count = 0;

    for (const auto& obs : observations) {
        // Only use reads where this position is interior (no damage zone)
        if (obs.dist_from_5p <= damage_positions || obs.dist_from_3p <= damage_positions)
            continue;

        double q_err = std::pow(10.0, -static_cast<double>(obs.baseq) / 10.0);
        q_err = std::max(0.001, std::min(0.5, q_err));

        if (obs.base == test_base) {
            log_lik_test += std::log(1.0 - q_err);
            log_lik_alt += std::log(q_err / 3.0);
        } else if (obs.base == alt_base) {
            log_lik_test += std::log(q_err / 3.0);
            log_lik_alt += std::log(1.0 - q_err);
        }
        interior_count++;
    }

    if (interior_count < 2) return 0.0;  // Not enough interior evidence
    return log_lik_test - log_lik_alt;
}

// Configuration for the polishing engine
struct PolishEngineConfig {
    int max_passes = 3;                  // Maximum number of iterative passes
    double min_log_lr = 3.0;             // Minimum log likelihood ratio for correction (legacy mode)
    int damage_positions = 15;           // Base positions (overridden by dynamic zone if enabled)
    int min_mapq = 30;                   // Minimum mapping quality
    int min_baseq = 20;                  // Minimum base quality
    double lr_decay_per_pass = 0.5;      // LR threshold decay per pass
    double convergence_fraction = 0.01;  // Stop when corrections < this fraction of previous
    bool use_interior_consensus = true;  // Phase 2: consensus-based voting (legacy mode)
    bool use_dynamic_zone = true;        // Phase 3: dynamic damage zone
    double consensus_weight = 0.5;       // Weight for interior consensus in combined score (legacy)
    int min_damage_reads = 3;            // Min reads in damage zone for correction
    int min_interior_reads = 2;          // Min interior reads for consensus

    // Bayesian caller options (NEW - replaces binary LR test)
    bool use_bayesian_caller = true;     // Use full 4-base Bayesian genotype caller
    double min_posterior = 0.99;         // Required posterior probability for correction
    double max_second_posterior = 0.05;  // Max allowed probability for second-best base
    bool write_detailed_calls = false;   // Write per-position call details

    // Two-population model settings
    bool use_two_population = true;      // Use ancient/modern read classification
    double min_ancient_fraction = 0.3;   // Skip correction if ancient fraction below this
    double disagreement_threshold = 0.5; // Flag positions where populations disagree strongly
    bool require_ancient_majority = true; // Require ancient fraction > 0.5 for correction
};

// Result from polishing a single contig
struct PolishResult {
    size_t ct_corrections = 0;           // Number of C->T corrections (T->C in sequence)
    size_t ga_corrections = 0;           // Number of G->A corrections (A->G in sequence)
    int passes_used = 0;                 // Number of passes performed
    std::vector<size_t> corrections_per_pass; // Corrections in each pass

    // Detailed statistics (optional)
    size_t positions_examined = 0;
    size_t positions_with_evidence = 0;
    size_t skipped_low_coverage = 0;
    size_t skipped_ambiguous = 0;

    // Bayesian caller statistics
    size_t bayesian_high_confidence = 0; // Positions with posterior >= min_posterior
    size_t bayesian_uncertain = 0;       // Positions with posterior < min_posterior
    size_t bayesian_mixed = 0;           // Positions where second_best > max_second_posterior

    // Two-population model statistics
    size_t positions_flagged_contamination = 0; // Skipped due to high contamination
    size_t population_disagreements = 0;        // Positions where ancient/modern disagree
    double estimated_ancient_fraction = 0.0;    // Bin-level ancient read fraction

    size_t total_corrections() const { return ct_corrections + ga_corrections; }
};

// Shared read observation type - alias to DamageReadObs for compatibility
using PolishReadObs = DamageReadObs;

// Core polishing function using sweep-line algorithm with multi-pass
// Modifies sequence in-place and returns statistics
inline PolishResult polish_contig(
    std::string& sequence,
    const std::string& contig_name,
    samFile* sf, bam_hdr_t* hdr, hts_idx_t* idx,
    const DamageModelResult& damage_model,
    const PolishEngineConfig& config = PolishEngineConfig()) {

    PolishResult result;

    int tid = bam_name2id(hdr, contig_name.c_str());
    if (tid < 0) return result;

    int len = static_cast<int>(sequence.length());

    // Compute dynamic damage zones (Phase 3)
    int zone_5p = config.use_dynamic_zone
        ? effective_damage_zone(damage_model.curve_5p.amplitude, damage_model.curve_5p.lambda, damage_model.curve_5p.baseline)
        : config.damage_positions;
    int zone_3p = config.use_dynamic_zone
        ? effective_damage_zone(damage_model.curve_3p.amplitude, damage_model.curve_3p.lambda, damage_model.curve_3p.baseline)
        : config.damage_positions;

    // Fetch all reads for this contig
    hts_itr_t* iter = sam_itr_queryi(idx, tid, 0, len);
    if (!iter) return result;

    std::vector<bam1_t*> reads;
    bam1_t* b = bam_init1();
    while (sam_itr_next(sf, iter, b) >= 0) {
        if (b->core.qual < config.min_mapq) continue;
        if (b->core.flag & (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY | BAM_FDUP)) continue;
        reads.push_back(bam_dup1(b));
    }
    bam_destroy1(b);
    hts_itr_destroy(iter);

    if (reads.empty()) return result;

    // =========================================================================
    // WHOLE-READ DAMAGE CLASSIFICATION (Pre-compute before position loop)
    // Key insight: Classify reads by their ENTIRE damage fingerprint, not
    // per-position. Ancient reads show C→T at 5' and G→A at 3' across the
    // whole read. Modern/contaminant reads don't show this pattern.
    // =========================================================================
    std::unordered_map<bam1_t*, ReadAncientScore> read_ancient_scores;

    if (config.use_two_population) {
        int damage_zone = std::max(zone_5p, zone_3p);

        ReadClassifierConfig rc_config;
        rc_config.amp_5p = damage_model.curve_5p.amplitude;
        rc_config.lam_5p = damage_model.curve_5p.lambda;
        rc_config.base_5p = damage_model.curve_5p.baseline;
        rc_config.amp_3p = damage_model.curve_3p.amplitude;
        rc_config.lam_3p = damage_model.curve_3p.lambda;
        rc_config.base_3p = damage_model.curve_3p.baseline;
        rc_config.damage_zone = damage_zone;
        rc_config.prior_ancient = 0.7;

        for (auto* r : reads) {
            // Extract ALL mismatches for this read vs current sequence
            std::vector<MismatchObs> mismatches;

            uint32_t* cigar = bam_get_cigar(r);
            int n_cigar = r->core.n_cigar;
            int ref_pos = r->core.pos;
            int qpos = 0;
            int read_len = r->core.l_qseq;
            bool is_rev = (r->core.flag & BAM_FREVERSE) != 0;

            uint8_t* seq = bam_get_seq(r);
            uint8_t* qual = bam_get_qual(r);

            for (int i = 0; i < n_cigar; i++) {
                int op = bam_cigar_op(cigar[i]);
                int oplen = bam_cigar_oplen(cigar[i]);

                if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                    for (int j = 0; j < oplen; j++) {
                        if (ref_pos + j >= 0 && ref_pos + j < len) {
                            char ref_base = sequence[ref_pos + j];
                            int base_code = bam_seqi(seq, qpos + j);
                            char read_base;
                            switch (base_code) {
                                case 1: read_base = 'A'; break;
                                case 2: read_base = 'C'; break;
                                case 4: read_base = 'G'; break;
                                case 8: read_base = 'T'; break;
                                default: read_base = 'N'; break;
                            }

                            // Record mismatch
                            if (read_base != ref_base && read_base != 'N' &&
                                ref_base != 'N' && qual[qpos + j] >= config.min_baseq) {
                                MismatchObs mm;
                                mm.ref_base = ref_base;
                                mm.read_base = read_base;
                                mm.baseq = qual[qpos + j];

                                if (is_rev) {
                                    mm.dist_from_5p = read_len - (qpos + j);
                                    mm.dist_from_3p = qpos + j + 1;
                                } else {
                                    mm.dist_from_5p = qpos + j + 1;
                                    mm.dist_from_3p = read_len - (qpos + j);
                                }
                                mismatches.push_back(mm);
                            }
                        }
                    }
                    ref_pos += oplen;
                    qpos += oplen;
                } else if (op == BAM_CINS || op == BAM_CSOFT_CLIP) {
                    qpos += oplen;
                } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
                    ref_pos += oplen;
                }
            }

            // Classify this read using whole-read damage fingerprint
            ReadAncientScore score = classify_read(mismatches, rc_config);
            read_ancient_scores[r] = score;
        }

        // Estimate bin-level ancient fraction
        std::vector<ReadAncientScore> all_scores;
        all_scores.reserve(read_ancient_scores.size());
        for (const auto& [r, score] : read_ancient_scores) {
            all_scores.push_back(score);
        }
        result.estimated_ancient_fraction = estimate_bin_ancient_fraction(all_scores);
    }

    // Mark candidate positions using dynamic zones
    std::vector<bool> is_5p_candidate(len, false);
    std::vector<bool> is_3p_candidate(len, false);

    for (auto* r : reads) {
        int read_start = r->core.pos;
        int read_end = bam_endpos(r);
        bool is_rev = (r->core.flag & BAM_FREVERSE) != 0;

        int pos_5p = is_rev ? (read_end - 1) : read_start;
        int pos_3p = is_rev ? read_start : (read_end - 1);

        for (int d = 0; d < zone_5p; d++) {
            int p5 = is_rev ? (pos_5p - d) : (pos_5p + d);
            if (p5 >= 0 && p5 < len) is_5p_candidate[p5] = true;
        }
        for (int d = 0; d < zone_3p; d++) {
            int p3 = is_rev ? (pos_3p + d) : (pos_3p - d);
            if (p3 >= 0 && p3 < len) is_3p_candidate[p3] = true;
        }
    }

    // Sort reads by start position for sweep line
    std::sort(reads.begin(), reads.end(), [](bam1_t* a, bam1_t* b) {
        return a->core.pos < b->core.pos;
    });

    // Multi-pass loop (Phase 1)
    size_t prev_corrections = 0;
    double current_lr_threshold = config.min_log_lr;

    for (int pass = 0; pass < config.max_passes; pass++) {
        size_t pass_ct = 0, pass_ga = 0;

        // Sweep through positions with active read set
        std::set<std::pair<int, bam1_t*>> active_by_end;
        size_t read_idx = 0;

        for (int pos = 0; pos < len; pos++) {
        // Add reads starting at or before this position
        while (read_idx < reads.size() && static_cast<int>(reads[read_idx]->core.pos) <= pos) {
            int end = bam_endpos(reads[read_idx]);
            active_by_end.insert({end, reads[read_idx]});
            read_idx++;
        }

        // Remove reads that ended before this position
        while (!active_by_end.empty() && active_by_end.begin()->first <= pos) {
            active_by_end.erase(active_by_end.begin());
        }

        char base = sequence[pos];
        bool check_5p = (base == 'T' && is_5p_candidate[pos]);
        bool check_3p = (base == 'A' && is_3p_candidate[pos]);
        if (!check_5p && !check_3p) continue;

        result.positions_examined++;

        // Build observations from active reads (track read pointers for two-pop lookup)
        std::vector<PolishReadObs> observations;
        std::vector<bam1_t*> obs_reads;  // Parallel vector: which read each obs came from
        int n_5p_damage = 0, n_3p_damage = 0;

        for (const auto& [end, r] : active_by_end) {
            uint32_t* cigar = bam_get_cigar(r);
            int n_cigar = r->core.n_cigar;
            int ref_pos = r->core.pos;
            int qpos = 0;
            bool found = false;

            for (int i = 0; i < n_cigar && !found; i++) {
                int op = bam_cigar_op(cigar[i]);
                int oplen = bam_cigar_oplen(cigar[i]);

                if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                    if (ref_pos <= pos && pos < ref_pos + oplen) {
                        qpos += (pos - ref_pos);
                        found = true;
                    } else {
                        ref_pos += oplen;
                        qpos += oplen;
                    }
                } else if (op == BAM_CINS || op == BAM_CSOFT_CLIP) {
                    qpos += oplen;
                } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
                    ref_pos += oplen;
                }
            }

            if (!found) continue;

            uint8_t* qual = bam_get_qual(r);
            if (qual[qpos] < config.min_baseq) continue;

            uint8_t* seq = bam_get_seq(r);
            int base_code = bam_seqi(seq, qpos);

            PolishReadObs obs;
            obs.baseq = qual[qpos];
            obs.is_reverse = (r->core.flag & BAM_FREVERSE) != 0;
            obs.read_len = r->core.l_qseq;

            if (obs.is_reverse) {
                obs.dist_from_5p = obs.read_len - qpos;
                obs.dist_from_3p = qpos + 1;
            } else {
                obs.dist_from_5p = qpos + 1;
                obs.dist_from_3p = obs.read_len - qpos;
            }

            switch (base_code) {
                case 1: obs.base = 'A'; break;
                case 2: obs.base = 'C'; break;
                case 4: obs.base = 'G'; break;
                case 8: obs.base = 'T'; break;
                default: continue;
            }
            observations.push_back(obs);
            obs_reads.push_back(r);  // Track which read this observation came from

            if (obs.dist_from_5p <= config.damage_positions) n_5p_damage++;
            if (obs.dist_from_3p <= config.damage_positions) n_3p_damage++;
        }

        if (observations.size() < 3) {
            result.skipped_low_coverage++;
            continue;
        }

        // =====================================================================
        // Bayesian Genotype Caller with Two-Population Model
        // Classifies reads as ancient (damaged) vs modern (contaminant)
        // Uses soft weights for robust calling in contaminated samples
        // =====================================================================
        if (config.use_bayesian_caller && (check_5p || check_3p)) {
            result.positions_with_evidence++;

            // Configure Bayesian caller with damage model parameters
            BayesianCallerConfig bc_config;
            bc_config.min_posterior = config.min_posterior;
            bc_config.max_second_posterior = config.max_second_posterior;
            bc_config.min_reads = config.min_damage_reads;
            bc_config.amp_5p = damage_model.curve_5p.amplitude;
            bc_config.lam_5p = damage_model.curve_5p.lambda;
            bc_config.base_5p = damage_model.curve_5p.baseline;
            bc_config.amp_3p = damage_model.curve_3p.amplitude;
            bc_config.lam_3p = damage_model.curve_3p.lambda;
            bc_config.base_3p = damage_model.curve_3p.baseline;

            int damage_zone = std::max(zone_5p, zone_3p);
            int current_base_idx = base_to_idx(base);
            GenotypeCall call;
            bool skip_due_to_contamination = false;

            // Two-population model: use pre-computed WHOLE-READ classifications
            // Key insight: reads are classified by their entire damage fingerprint,
            // not by what they show at this single position (avoids circular bias)
            if (config.use_two_population && !read_ancient_scores.empty()) {
                // Look up pre-computed whole-read scores for each observation
                std::vector<ReadAncientScore> read_scores;
                read_scores.reserve(observations.size());

                for (size_t i = 0; i < observations.size(); i++) {
                    bam1_t* r = obs_reads[i];
                    auto it = read_ancient_scores.find(r);
                    if (it != read_ancient_scores.end()) {
                        read_scores.push_back(it->second);
                    } else {
                        // Fallback: use prior if read not pre-classified
                        ReadAncientScore fallback;
                        fallback.p_ancient = 0.7;
                        fallback.llr = 0.85;
                        read_scores.push_back(fallback);
                    }
                }

                // Configure two-population caller
                TwoPopulationConfig tp_config;
                tp_config.min_ancient_fraction = config.min_ancient_fraction;
                tp_config.disagreement_threshold = config.disagreement_threshold;
                tp_config.require_ancient_majority = config.require_ancient_majority;
                tp_config.damage_zone = damage_zone;
                tp_config.enable_two_pop = true;

                // Call with two-population model
                TwoPopulationResult tp_result = call_genotype_two_population(
                    observations, read_scores, current_base_idx, bc_config, tp_config);

                // Track two-population statistics
                if (tp_result.high_contamination) {
                    result.positions_flagged_contamination++;
                    skip_due_to_contamination = true;
                }
                if (tp_result.populations_disagree) {
                    result.population_disagreements++;
                }

                // Use recommended call (ancient population if two-pop, else combined)
                call = tp_result.recommended_call();

                // Additional skip if should_skip flag is set
                if (tp_result.should_skip) {
                    skip_due_to_contamination = true;
                }
            } else {
                // Fallback to two-stage caller without read classification
                call = call_genotype_two_stage(observations, damage_zone, current_base_idx, bc_config);
            }

            // Track Bayesian caller statistics
            if (call.posterior_best >= config.min_posterior) {
                result.bayesian_high_confidence++;
            } else {
                result.bayesian_uncertain++;
            }
            if (call.posterior_second > config.max_second_posterior) {
                result.bayesian_mixed++;
            }

            // Apply correction if confident AND not flagged for contamination
            if (!skip_due_to_contamination && call.should_correct && call.best_base != base_to_idx(base)) {
                char new_base = idx_to_base(call.best_base);

                // Track correction type
                if (base == 'T' && new_base == 'C') {
                    result.ct_corrections++;
                    pass_ct++;
                } else if (base == 'A' && new_base == 'G') {
                    result.ga_corrections++;
                    pass_ga++;
                }

                sequence[pos] = new_base;
            } else if (!call.should_correct || skip_due_to_contamination) {
                result.skipped_ambiguous++;
            }
        }
        // =====================================================================
        // Legacy LR Mode (for comparison/fallback)
        // =====================================================================
        else {
            // C->T correction using damage model + consensus (Phase 2)
            if (check_5p && n_5p_damage >= config.min_damage_reads) {
                result.positions_with_evidence++;
                double log_lr_damage = compute_ct_log_lr(
                    observations,
                    damage_model.curve_5p.amplitude,
                    damage_model.curve_5p.lambda,
                    damage_model.curve_5p.baseline);

                double combined_lr = log_lr_damage;
                if (config.use_interior_consensus) {
                    double interior_lr = compute_interior_consensus_lr(observations, 'C', 'T', zone_5p);
                    combined_lr = log_lr_damage + config.consensus_weight * interior_lr;
                }

                if (combined_lr > current_lr_threshold) {
                    sequence[pos] = 'C';
                    result.ct_corrections++;
                    pass_ct++;
                } else if (combined_lr >= -current_lr_threshold) {
                    result.skipped_ambiguous++;
                }
            }

            // G->A correction using damage model + consensus (Phase 2)
            if (check_3p && n_3p_damage >= config.min_damage_reads) {
                result.positions_with_evidence++;
                double log_lr_damage = compute_ga_log_lr(
                    observations,
                    damage_model.curve_3p.amplitude,
                    damage_model.curve_3p.lambda,
                    damage_model.curve_3p.baseline);

                double combined_lr = log_lr_damage;
                if (config.use_interior_consensus) {
                    double interior_lr = compute_interior_consensus_lr(observations, 'G', 'A', zone_3p);
                    combined_lr = log_lr_damage + config.consensus_weight * interior_lr;
                }

                if (combined_lr > current_lr_threshold) {
                    sequence[pos] = 'G';
                    result.ga_corrections++;
                    pass_ga++;
                } else if (combined_lr >= -current_lr_threshold) {
                    result.skipped_ambiguous++;
                }
            }
        }
        }  // end position loop

        // Track per-pass stats
        size_t pass_total = pass_ct + pass_ga;
        result.corrections_per_pass.push_back(pass_total);
        result.passes_used = pass + 1;

        // Check convergence
        if (pass > 0 && prev_corrections > 0) {
            double conv_ratio = static_cast<double>(pass_total) / static_cast<double>(prev_corrections);
            if (conv_ratio < config.convergence_fraction) {
                break;  // Converged
            }
        }

        if (pass_total == 0) break;  // No more corrections possible

        prev_corrections = pass_total;

        // Relax threshold for next pass
        current_lr_threshold *= config.lr_decay_per_pass;
        current_lr_threshold = std::max(current_lr_threshold, 1.5);  // Floor at 1.5

    }  // end multi-pass loop

    // Cleanup reads
    for (auto* r : reads) bam_destroy1(r);

    return result;
}

}  // namespace amber

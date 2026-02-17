#pragma once

// AMBER - Damage-aware Bayesian Genotype Caller
// Full 4-base posterior computation with damage transition matrix
//
// Key insight: Binary LR tests (H1:C vs H0:T) over-correct at ambiguous positions.
// This caller computes P(true=g | all reads) for all 4 bases and requires:
//   1. P(best) > 0.99 (high confidence)
//   2. P(second_best) < 0.05 (no close alternatives)
// Only then do we correct, preventing over-correction at mixed positions.

#include "damage_likelihood.h"
#include "damage_read_classifier.h"
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace amber {

// Base indices for matrix operations
constexpr int BASE_A = 0;
constexpr int BASE_C = 1;
constexpr int BASE_G = 2;
constexpr int BASE_T = 3;

inline int base_to_idx(char b) {
    switch (b) {
        case 'A': case 'a': return BASE_A;
        case 'C': case 'c': return BASE_C;
        case 'G': case 'g': return BASE_G;
        case 'T': case 't': return BASE_T;
        default: return -1;
    }
}

inline char idx_to_base(int i) {
    constexpr char bases[] = {'A', 'C', 'G', 'T'};
    return (i >= 0 && i < 4) ? bases[i] : 'N';
}

// Result of genotype calling at a single position
struct GenotypeCall {
    int best_base = -1;           // Index of most likely true base
    int second_best = -1;         // Index of second most likely
    double posterior_best = 0.0;  // P(best | reads)
    double posterior_second = 0.0;// P(second_best | reads)
    std::array<double, 4> posteriors = {0.25, 0.25, 0.25, 0.25};  // Full posterior
    bool should_correct = false;  // True if confident enough to change sequence
    int reads_used = 0;           // Number of reads contributing to call

    // Helper to check if correction would change base
    bool would_change(char current) const {
        int current_idx = base_to_idx(current);
        return should_correct && best_base != current_idx && best_base >= 0;
    }
};

// Configuration for Bayesian caller
struct BayesianCallerConfig {
    double min_posterior = 0.99;       // Required confidence in best call
    double max_second_posterior = 0.05; // Max allowed probability for second-best
    int min_reads = 3;                 // Minimum reads to make a call
    bool use_gc_prior = false;         // Use GC-content-aware prior (not uniform)
    double gc_content = 0.5;           // GC content for prior (if enabled)

    // Damage model parameters (filled from DamageModelResult)
    double amp_5p = 0.3;    // C→T at 5' amplitude
    double lam_5p = 0.3;    // C→T decay rate
    double base_5p = 0.01;  // C→T baseline
    double amp_3p = 0.3;    // G→A at 3' amplitude
    double lam_3p = 0.3;    // G→A decay rate
    double base_3p = 0.01;  // G→A baseline
};

// 4x4 damage transition matrix: T[true_base][observed_base] = P(obs | true)
// Incorporates damage rates at specific read positions and sequencing error
using TransitionMatrix = std::array<std::array<double, 4>, 4>;

// Compute damage transition matrix for a single read observation
// Parameters:
//   d5: damage rate from 5' end (for C→T)
//   d3: damage rate from 3' end (for G→A)
//   q_err: sequencing error probability from Phred score
inline TransitionMatrix compute_damage_transition(double d5, double d3, double q_err) {
    TransitionMatrix T;

    // Clamp probabilities to valid range
    d5 = std::max(0.001, std::min(0.999, d5));
    d3 = std::max(0.001, std::min(0.999, d3));
    q_err = std::max(0.001, std::min(0.5, q_err));

    // Error probability for specific base substitution (uniform across 3 alternatives)
    double q_sub = q_err / 3.0;

    // Row 0: True = A
    // No damage channel for A, only sequencing error
    T[BASE_A][BASE_A] = 1.0 - q_err;
    T[BASE_A][BASE_C] = q_sub;
    T[BASE_A][BASE_G] = q_sub;
    T[BASE_A][BASE_T] = q_sub;

    // Row 1: True = C
    // C→T damage at 5' end, plus sequencing error
    // P(obs=T | true=C) = d5*(1-q_err) + (1-d5)*q_sub  [damage correct OR no damage + error]
    // P(obs=C | true=C) = (1-d5)*(1-q_err) + d5*q_sub  [no damage correct OR damage + error back]
    T[BASE_C][BASE_A] = q_sub;                                        // Error only
    T[BASE_C][BASE_C] = (1.0 - d5) * (1.0 - q_err) + d5 * q_sub;     // Survive or damage+error
    T[BASE_C][BASE_G] = q_sub;                                        // Error only
    T[BASE_C][BASE_T] = d5 * (1.0 - q_err) + (1.0 - d5) * q_sub;     // Damage or error

    // Row 2: True = G
    // G→A damage at 3' end, plus sequencing error
    T[BASE_G][BASE_A] = d3 * (1.0 - q_err) + (1.0 - d3) * q_sub;     // Damage or error
    T[BASE_G][BASE_C] = q_sub;                                        // Error only
    T[BASE_G][BASE_G] = (1.0 - d3) * (1.0 - q_err) + d3 * q_sub;     // Survive or damage+error
    T[BASE_G][BASE_T] = q_sub;                                        // Error only

    // Row 3: True = T
    // No damage channel for T, only sequencing error
    T[BASE_T][BASE_A] = q_sub;
    T[BASE_T][BASE_C] = q_sub;
    T[BASE_T][BASE_G] = q_sub;
    T[BASE_T][BASE_T] = 1.0 - q_err;

    return T;
}

// Log-sum-exp for numerical stability: log(sum(exp(x_i)))
inline double log_sum_exp(const std::array<double, 4>& log_vals) {
    double max_val = *std::max_element(log_vals.begin(), log_vals.end());
    if (max_val == -std::numeric_limits<double>::infinity()) {
        return -std::numeric_limits<double>::infinity();
    }
    double sum = 0.0;
    for (double lv : log_vals) {
        sum += std::exp(lv - max_val);
    }
    return max_val + std::log(sum);
}

// Call genotype at a single position using Bayesian inference
// P(true=g | reads) ∝ P(g) × ∏_i P(read_i | g)
template<typename ReadObs>
GenotypeCall call_genotype(
    const std::vector<ReadObs>& observations,
    int ref_base_idx,  // Reference/current base index (for prior adjustment)
    const BayesianCallerConfig& config) {

    GenotypeCall result;
    result.reads_used = 0;

    // Count usable observations
    for (const auto& obs : observations) {
        if (base_to_idx(obs.base) >= 0) result.reads_used++;
    }

    if (result.reads_used < config.min_reads) {
        // Not enough evidence - return uniform posterior, no correction
        return result;
    }

    // Initialize log prior: P(g)
    // Haploid microbial genome - uniform prior or GC-aware
    std::array<double, 4> log_prior;
    if (config.use_gc_prior) {
        double gc = config.gc_content;
        double at = 1.0 - gc;
        log_prior[BASE_A] = std::log(at / 2.0);
        log_prior[BASE_T] = std::log(at / 2.0);
        log_prior[BASE_C] = std::log(gc / 2.0);
        log_prior[BASE_G] = std::log(gc / 2.0);
    } else {
        log_prior = {std::log(0.25), std::log(0.25), std::log(0.25), std::log(0.25)};
    }

    // Initialize log likelihood for each true genotype: log P(reads | g)
    std::array<double, 4> log_lik = {0.0, 0.0, 0.0, 0.0};

    // Accumulate evidence from each read
    for (const auto& obs : observations) {
        int obs_idx = base_to_idx(obs.base);
        if (obs_idx < 0) continue;

        // Compute per-read damage rates based on position
        double q_err = phred_to_error_prob(obs.baseq);
        double d5 = damage_rate_at_distance(obs.dist_from_5p, config.amp_5p, config.lam_5p, config.base_5p);
        double d3 = damage_rate_at_distance(obs.dist_from_3p, config.amp_3p, config.lam_3p, config.base_3p);

        // Build transition matrix for this read
        TransitionMatrix T = compute_damage_transition(d5, d3, q_err);

        // Add log P(obs | true=g) for each possible true base
        for (int g = 0; g < 4; g++) {
            double p = T[g][obs_idx];
            log_lik[g] += std::log(std::max(1e-10, p));
        }
    }

    // Compute log posterior (unnormalized): log P(g) + log P(reads | g)
    std::array<double, 4> log_posterior;
    for (int g = 0; g < 4; g++) {
        log_posterior[g] = log_prior[g] + log_lik[g];
    }

    // Normalize using log-sum-exp
    double log_norm = log_sum_exp(log_posterior);

    for (int g = 0; g < 4; g++) {
        result.posteriors[g] = std::exp(log_posterior[g] - log_norm);
    }

    // Find best and second-best
    std::array<std::pair<double, int>, 4> sorted_post;
    for (int g = 0; g < 4; g++) {
        sorted_post[g] = {result.posteriors[g], g};
    }
    std::sort(sorted_post.begin(), sorted_post.end(), std::greater<>());

    result.best_base = sorted_post[0].second;
    result.posterior_best = sorted_post[0].first;
    result.second_best = sorted_post[1].second;
    result.posterior_second = sorted_post[1].first;

    // Decide if we should correct
    // Require: (1) high confidence in best, (2) low probability for alternatives
    result.should_correct =
        result.posterior_best >= config.min_posterior &&
        result.posterior_second <= config.max_second_posterior;

    return result;
}

// Two-Stage Bayesian Genotype Caller
// Stage 1: Use interior reads (outside damage zone) with simple error model
// Stage 2: Only if interior is ambiguous, incorporate damage-zone reads
//
// KEY PRINCIPLE: Interior reads are GROUND TRUTH (no damage confound).
// If interior supports current base, NEVER correct regardless of damage zone.
// Only correct when interior reads indicate a DIFFERENT base than current.
template<typename ReadObs>
GenotypeCall call_genotype_two_stage(
    const std::vector<ReadObs>& observations,
    int damage_zone,  // Positions within this distance are in damage zone
    int current_base_idx,  // The current sequence base at this position
    const BayesianCallerConfig& config) {

    GenotypeCall result;

    // Separate reads into interior (clean) and damage-zone (potentially damaged)
    std::vector<ReadObs> interior_obs, damage_obs;
    for (const auto& obs : observations) {
        if (obs.dist_from_5p > damage_zone && obs.dist_from_3p > damage_zone) {
            interior_obs.push_back(obs);
        } else {
            damage_obs.push_back(obs);
        }
    }

    result.reads_used = interior_obs.size() + damage_obs.size();

    // Stage 1: Call from interior reads using simple error model (no damage)
    // These reads give us ground truth about the actual sequence
    if (interior_obs.size() >= 2) {
        std::array<double, 4> log_lik_interior = {0.0, 0.0, 0.0, 0.0};

        for (const auto& obs : interior_obs) {
            int obs_idx = base_to_idx(obs.base);
            if (obs_idx < 0) continue;

            double q_err = phred_to_error_prob(obs.baseq);
            q_err = std::max(0.001, std::min(0.5, q_err));

            for (int g = 0; g < 4; g++) {
                double p = (g == obs_idx) ? (1.0 - q_err) : (q_err / 3.0);
                log_lik_interior[g] += std::log(std::max(1e-10, p));
            }
        }

        // Compute interior posterior
        std::array<double, 4> log_post_interior;
        for (int g = 0; g < 4; g++) {
            log_post_interior[g] = std::log(0.25) + log_lik_interior[g];
        }
        double log_norm = log_sum_exp(log_post_interior);

        std::array<double, 4> post_interior;
        for (int g = 0; g < 4; g++) {
            post_interior[g] = std::exp(log_post_interior[g] - log_norm);
        }

        // Find best from interior
        int best_interior = 0;
        for (int g = 1; g < 4; g++) {
            if (post_interior[g] > post_interior[best_interior]) {
                best_interior = g;
            }
        }

        // If interior is confident, use it directly (no need for damage-zone)
        if (post_interior[best_interior] >= config.min_posterior) {
            result.posteriors = post_interior;
            result.best_base = best_interior;
            result.posterior_best = post_interior[best_interior];

            // Find second best
            int second = (best_interior == 0) ? 1 : 0;
            for (int g = 0; g < 4; g++) {
                if (g != best_interior && post_interior[g] > post_interior[second]) {
                    second = g;
                }
            }
            result.second_best = second;
            result.posterior_second = post_interior[second];

            result.should_correct =
                result.posterior_best >= config.min_posterior &&
                result.posterior_second <= config.max_second_posterior;
            return result;
        }

        // Find second best for interior
        int second_interior = (best_interior == 0) ? 1 : 0;
        for (int g = 0; g < 4; g++) {
            if (g != best_interior && post_interior[g] > post_interior[second_interior]) {
                second_interior = g;
            }
        }

        result.posteriors = post_interior;
        result.best_base = best_interior;
        result.posterior_best = post_interior[best_interior];
        result.second_best = second_interior;
        result.posterior_second = post_interior[second_interior];

        // Count interior reads supporting each base
        std::array<int, 4> interior_counts = {0, 0, 0, 0};
        for (const auto& obs : interior_obs) {
            int idx = base_to_idx(obs.base);
            if (idx >= 0) interior_counts[idx]++;
        }
        int total_interior = static_cast<int>(interior_obs.size());

        // STRICT INTERIOR REQUIREMENTS:
        // 1. Need at least 5 interior reads total for reliable evidence
        // 2. Correction base must have at least 4 interior reads
        // 3. Current base must have 0 interior reads (complete disagreement)
        // 4. Standard posterior thresholds must be met

        if (current_base_idx >= 0 && current_base_idx < 4) {
            int current_count = interior_counts[current_base_idx];
            int best_count = interior_counts[best_interior];

            // Strict: need many interior reads AND zero support for current
            bool enough_interior = (total_interior >= 5 && best_count >= 4);
            bool current_rejected = (current_count == 0);

            if (!enough_interior || !current_rejected) {
                result.should_correct = false;
                return result;
            }
        }

        // All strict criteria met - allow correction
        result.should_correct = (post_interior[best_interior] >= config.min_posterior &&
                                 post_interior[second_interior] <= config.max_second_posterior &&
                                 best_interior != current_base_idx);
        return result;
    }

    // Insufficient interior reads (<2) - DO NOT CORRECT
    // Without interior validation, we cannot distinguish damage from true variants
    // This is the key insight: interior reads are ground truth, damage zone is confounded
    result.should_correct = false;
    return result;
}

// ============================================================================
// Two-Population Model: Ancient vs Modern (Contaminant) Reads
// ============================================================================
// Key insight: When we have soft read classifications (p_ancient from damage
// fingerprints), we can call separate genotypes for each population and detect
// positions where contamination is driving the consensus.

// Configuration for two-population genotype calling
struct TwoPopulationConfig {
    // Minimum ancient fraction to trust correction (skip if contamination too high)
    double min_ancient_fraction = 0.3;

    // Disagreement detection: flag positions where populations differ
    // Strength = |P(best|ancient) - P(best|modern)|
    double disagreement_threshold = 0.5;

    // Minimum total weight in each population to make a call
    double min_population_weight = 2.0;

    // Require ancient reads to be majority (fraction > 0.5) for correction
    bool require_ancient_majority = true;

    // When true, use two-population model; when false, fall back to two-stage
    bool enable_two_pop = true;

    // Damage zone for fallback to two-stage model
    int damage_zone = 15;
};

// Result of two-population genotype calling
struct TwoPopulationResult {
    GenotypeCall ancient_call;    // Genotype call from ancient population
    GenotypeCall modern_call;     // Genotype call from modern/contaminant population
    GenotypeCall combined_call;   // Fallback single-population call

    double fraction_ancient;      // Estimated fraction of ancient reads at this position
    double disagreement_strength; // How much populations disagree (0 = agree, 1 = opposite)
    bool populations_disagree;    // True if ancient != modern best base

    // Contamination flags
    bool high_contamination;      // fraction_ancient < min_ancient_fraction
    bool modern_drives_consensus; // Modern reads would flip the call

    // Decision flags
    bool should_skip;             // True if should not correct this position
    bool used_two_pop;            // True if two-pop was used, false if fell back

    // Helper: get the recommended genotype (ancient if two-pop, else combined)
    const GenotypeCall& recommended_call() const {
        if (used_two_pop && !high_contamination) {
            return ancient_call;
        }
        return combined_call;
    }
};

// Call genotype using two-population model with soft read weights
// Each read contributes to both populations weighted by p_ancient / (1-p_ancient)
template<typename ReadObs>
TwoPopulationResult call_genotype_two_population(
    const std::vector<ReadObs>& observations,
    const std::vector<ReadAncientScore>& read_scores,
    int current_base_idx,
    const BayesianCallerConfig& caller_config,
    const TwoPopulationConfig& two_pop_config) {

    TwoPopulationResult result;
    result.used_two_pop = false;
    result.high_contamination = false;
    result.modern_drives_consensus = false;
    result.populations_disagree = false;
    result.should_skip = false;
    result.disagreement_strength = 0.0;
    result.fraction_ancient = 0.5;

    // Validate inputs
    if (observations.empty()) {
        return result;
    }

    // If read scores not available or size mismatch, fall back to two-stage
    bool can_use_two_pop = two_pop_config.enable_two_pop &&
                           !read_scores.empty() &&
                           observations.size() == read_scores.size();

    if (!can_use_two_pop) {
        result.combined_call = call_genotype_two_stage(
            observations, two_pop_config.damage_zone, current_base_idx, caller_config);
        return result;
    }

    // Compute weighted likelihoods for each population
    std::array<double, 4> log_lik_ancient = {0.0, 0.0, 0.0, 0.0};
    std::array<double, 4> log_lik_modern = {0.0, 0.0, 0.0, 0.0};

    double total_ancient_weight = 0.0;
    double total_modern_weight = 0.0;

    for (size_t i = 0; i < observations.size(); i++) {
        const auto& obs = observations[i];
        const auto& score = read_scores[i];

        int obs_idx = base_to_idx(obs.base);
        if (obs_idx < 0) continue;

        double q_err = phred_to_error_prob(obs.baseq);
        q_err = std::max(0.001, std::min(0.5, q_err));

        // Compute damage transition matrix for ancient reads
        double d5 = damage_rate_at_distance(obs.dist_from_5p,
            caller_config.amp_5p, caller_config.lam_5p, caller_config.base_5p);
        double d3 = damage_rate_at_distance(obs.dist_from_3p,
            caller_config.amp_3p, caller_config.lam_3p, caller_config.base_3p);
        TransitionMatrix T_ancient = compute_damage_transition(d5, d3, q_err);

        // Modern reads: simple error model (no damage)
        // P(obs|true) = (1-q_err) if match, q_err/3 if mismatch

        double w_anc = score.p_ancient;
        double w_mod = 1.0 - score.p_ancient;

        total_ancient_weight += w_anc;
        total_modern_weight += w_mod;

        for (int g = 0; g < 4; g++) {
            // Ancient: use damage-aware transition
            double p_anc = T_ancient[g][obs_idx];
            log_lik_ancient[g] += w_anc * std::log(std::max(1e-10, p_anc));

            // Modern: simple error model
            double p_mod = (g == obs_idx) ? (1.0 - q_err) : (q_err / 3.0);
            log_lik_modern[g] += w_mod * std::log(std::max(1e-10, p_mod));
        }
    }

    // Compute ancient fraction
    double total_weight = total_ancient_weight + total_modern_weight;
    result.fraction_ancient = total_ancient_weight / std::max(1e-10, total_weight);

    // Check for high contamination
    result.high_contamination = (result.fraction_ancient < two_pop_config.min_ancient_fraction);

    // Check if we have enough weight in each population
    bool enough_ancient = (total_ancient_weight >= two_pop_config.min_population_weight);
    bool enough_modern = (total_modern_weight >= two_pop_config.min_population_weight);

    // If insufficient population data, fall back
    if (!enough_ancient && !enough_modern) {
        result.combined_call = call_genotype_two_stage(
            observations, two_pop_config.damage_zone, current_base_idx, caller_config);
        return result;
    }

    result.used_two_pop = true;

    // Helper to compute call from log likelihoods
    auto compute_call = [&caller_config](const std::array<double, 4>& log_lik) -> GenotypeCall {
        GenotypeCall call;

        std::array<double, 4> log_post;
        for (int g = 0; g < 4; g++) {
            log_post[g] = std::log(0.25) + log_lik[g];
        }
        double log_norm = log_sum_exp(log_post);

        for (int g = 0; g < 4; g++) {
            call.posteriors[g] = std::exp(log_post[g] - log_norm);
        }

        // Find best and second-best
        std::array<std::pair<double, int>, 4> sorted;
        for (int g = 0; g < 4; g++) {
            sorted[g] = {call.posteriors[g], g};
        }
        std::sort(sorted.begin(), sorted.end(), std::greater<>());

        call.best_base = sorted[0].second;
        call.posterior_best = sorted[0].first;
        call.second_best = sorted[1].second;
        call.posterior_second = sorted[1].first;

        call.should_correct = (call.posterior_best >= caller_config.min_posterior &&
                               call.posterior_second <= caller_config.max_second_posterior);

        return call;
    };

    // Compute calls for each population
    if (enough_ancient) {
        result.ancient_call = compute_call(log_lik_ancient);
        result.ancient_call.reads_used = static_cast<int>(observations.size());
    }

    if (enough_modern) {
        result.modern_call = compute_call(log_lik_modern);
        result.modern_call.reads_used = static_cast<int>(observations.size());
    }

    // Compute combined call (weighted by both populations)
    std::array<double, 4> log_lik_combined;
    for (int g = 0; g < 4; g++) {
        log_lik_combined[g] = log_lik_ancient[g] + log_lik_modern[g];
    }
    result.combined_call = compute_call(log_lik_combined);
    result.combined_call.reads_used = static_cast<int>(observations.size());

    // Check for population disagreement
    if (enough_ancient && enough_modern) {
        result.populations_disagree = (result.ancient_call.best_base != result.modern_call.best_base);

        // Disagreement strength: how different are the posterior distributions?
        // Use max absolute difference in posteriors
        double max_diff = 0.0;
        for (int g = 0; g < 4; g++) {
            double diff = std::abs(result.ancient_call.posteriors[g] -
                                   result.modern_call.posteriors[g]);
            max_diff = std::max(max_diff, diff);
        }
        result.disagreement_strength = max_diff;

        // Check if modern reads would flip the consensus away from ancient
        if (result.populations_disagree &&
            result.combined_call.best_base == result.modern_call.best_base &&
            result.ancient_call.should_correct) {
            result.modern_drives_consensus = true;
        }
    }

    // Determine if we should skip correction at this position
    // Skip if: high contamination, strong disagreement, or ancient not majority when required
    bool ancient_minority = two_pop_config.require_ancient_majority &&
                            result.fraction_ancient < 0.5;
    bool strong_disagreement = result.disagreement_strength > two_pop_config.disagreement_threshold;

    result.should_skip = result.high_contamination || strong_disagreement || ancient_minority;

    // For positions we should skip, don't allow correction
    if (result.should_skip) {
        result.ancient_call.should_correct = false;
    }

    return result;
}

// Convenience function: get corrected base using two-population model
template<typename ReadObs>
char get_corrected_base_two_pop(
    const std::vector<ReadObs>& observations,
    const std::vector<ReadAncientScore>& read_scores,
    char current_base,
    const BayesianCallerConfig& caller_config,
    const TwoPopulationConfig& two_pop_config) {

    int current_idx = base_to_idx(current_base);
    if (current_idx < 0) return current_base;

    TwoPopulationResult result = call_genotype_two_population(
        observations, read_scores, current_idx, caller_config, two_pop_config);

    const GenotypeCall& call = result.recommended_call();
    if (call.would_change(current_base)) {
        return idx_to_base(call.best_base);
    }
    return current_base;
}

// Wrapper function for polish_engine integration
// Takes observations + read classifications, returns full result with flags
template<typename ReadObs>
TwoPopulationResult call_genotype_with_classification(
    const std::vector<ReadObs>& observations,
    const std::vector<ReadAncientScore>& read_scores,
    char current_base,
    const BayesianCallerConfig& caller_config,
    const TwoPopulationConfig& two_pop_config = TwoPopulationConfig()) {

    int current_idx = base_to_idx(current_base);

    TwoPopulationResult result = call_genotype_two_population(
        observations, read_scores, current_idx, caller_config, two_pop_config);

    return result;
}

// Convenience function for damage-aware correction check at C→T or G→A positions
// Returns true if confident the true base differs from observed
template<typename ReadObs>
bool should_correct_damage(
    const std::vector<ReadObs>& observations,
    char current_base,
    const BayesianCallerConfig& config) {

    int current_idx = base_to_idx(current_base);
    if (current_idx < 0) return false;

    GenotypeCall call = call_genotype(observations, current_idx, config);
    return call.would_change(current_base);
}

// Get the corrected base if confident, otherwise return original
template<typename ReadObs>
char get_corrected_base(
    const std::vector<ReadObs>& observations,
    char current_base,
    const BayesianCallerConfig& config) {

    int current_idx = base_to_idx(current_base);
    if (current_idx < 0) return current_base;

    GenotypeCall call = call_genotype(observations, current_idx, config);
    if (call.would_change(current_base)) {
        return idx_to_base(call.best_base);
    }
    return current_base;
}

}  // namespace amber

#pragma once

// AMBER - Damage-aware Read Classifier
// Soft classification of reads as ancient (damaged) vs modern (contaminant)
//
// Design principles:
// 1. Soft weights (p_ancient), not hard classification
// 2. Reads with no damage-zone positions stay near prior
// 3. Base quality incorporated in likelihoods
// 4. Per-bin priors supported (hierarchical)

#include "damage_likelihood.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace amber {

// Local base conversion (duplicated from bayesian_caller.h to avoid circular dep)
namespace detail {
inline int base_to_idx_local(char b) {
    switch (b) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}
}  // namespace detail

// Result of read-level damage classification
// Separates damage-based and length-based evidence for proper Bayesian analysis
struct ReadAncientScore {
    // Primary outputs - separated by evidence source
    double p_damaged;           // Posterior P(damaged | read), from damage patterns only [0,1]
    double p_length_ancient;    // Posterior P(ancient | length), from fragment length only [0,1]
    double p_ancient;           // Combined posterior P(ancient | read), joint evidence [0,1]

    // Log-likelihood ratios (for debugging and analysis)
    double llr_damage;          // log(P(obs|damaged) / P(obs|undamaged)) from substitution patterns
    double llr_length;          // log(P(length|ancient) / P(length|modern)) from fragment length

    // Diagnostic counts
    int damage_consistent;      // Count: C→T at 5', G→A at 3' (expected damage)
    int damage_inconsistent;    // Count: C→T at 3', G→A at 5', other unexpected patterns
    int damage_zone_positions;  // How many positions were in the damage zone
    int total_mismatches;       // Total mismatches observed in read

    // Backward compatibility alias
    double llr;                 // = llr_damage (for existing code)
};

// Configuration for read classifier
struct ReadClassifierConfig {
    // Damage model parameters (from DamageModelResult)
    double amp_5p = 0.3;    // C→T amplitude at 5' end
    double lam_5p = 0.3;    // C→T decay rate
    double base_5p = 0.01;  // C→T baseline
    double amp_3p = 0.3;    // G→A amplitude at 3' end
    double lam_3p = 0.3;    // G→A decay rate
    double base_3p = 0.01;  // G→A baseline

    // Classification parameters
    double prior_ancient = 0.7;  // Prior P(ancient), can be set per-bin
    int damage_zone = 15;        // Positions within this distance are damage zone
    double min_baseq = 10.0;     // Minimum base quality to consider

    // Modern read error model
    double modern_error_rate = 0.01;  // Uniform error rate for modern reads

    // Read length model parameters
    // Ancient DNA typically has shorter fragments due to degradation
    bool use_length_model = false;        // Enable length-based classification
    double length_weight = 1.0;           // Weight for length contribution
    double mode_length_ancient = 42.0;    // Mode fragment length for ancient (bp)
    double std_length_ancient = 15.0;     // Std dev for ancient length
    double mode_length_modern = 103.0;    // Mode fragment length for modern (bp)
    double std_length_modern = 30.0;      // Std dev for modern length

    // Read identity model parameters
    // Ancient reads map at 100% to ancient assembly, modern at ~99%
    bool use_identity_model = false;      // Enable identity-based classification
    double identity_weight = 1.0;         // Weight for identity contribution
    double mode_identity_ancient = 100.0; // Mode identity for ancient (%)
    double std_identity_ancient = 2.0;    // Std dev for ancient identity
    double mode_identity_modern = 99.0;   // Mode identity for modern (%)
    double std_identity_modern = 0.5;     // Std dev for modern identity
};

// Observation for a single mismatch within a read
struct MismatchObs {
    char ref_base;     // Reference/consensus base
    char read_base;    // Observed base in read
    uint8_t baseq;     // Base quality
    int dist_from_5p;  // Distance from 5' end of read
    int dist_from_3p;  // Distance from 3' end of read
};

// Full aligned position data for a single position in a read
struct AlignedPosition {
    char ref_base;     // Reference/consensus base
    char read_base;    // Observed base in read
    uint8_t baseq;     // Base quality
    int16_t dist_5p;   // Distance from 5' end
    int16_t dist_3p;   // Distance from 3' end
};

// Full read data for complete likelihood computation
// Unlike MismatchObs which only captures mismatches, this captures ALL aligned positions
struct FullReadData {
    std::vector<AlignedPosition> positions;  // All aligned positions
    int read_length = 0;
    double identity = 100.0;  // Alignment identity (%), computed from NM tag
};

// Compute log-likelihood of read length under a normal distribution
// Returns log P(length | mode, std)
inline double length_log_likelihood(int length, double mode, double std_dev) {
    if (length <= 0 || std_dev <= 0) return -100.0;

    double x = static_cast<double>(length);
    double z = (x - mode) / std_dev;

    // Log of normal PDF: -0.5 * z^2 - log(std) - 0.5*log(2*pi)
    // Skip constant term since it cancels in likelihood ratio
    return -0.5 * z * z - std::log(std_dev);
}

// Compute log-likelihood for identity (continuous, no int cast)
inline double identity_log_likelihood(double identity, double mode, double std_dev) {
    if (std_dev <= 0) return 0.0;
    double z = (identity - mode) / std_dev;
    return -0.5 * z * z - std::log(std_dev);
}

// Check if a mismatch is damage-consistent
// C→T at 5' end OR G→A at 3' end
inline bool is_damage_consistent(const MismatchObs& obs, int damage_zone) {
    bool in_5p_zone = (obs.dist_from_5p <= damage_zone);
    bool in_3p_zone = (obs.dist_from_3p <= damage_zone);

    // C→T in 5' damage zone
    if (in_5p_zone && obs.ref_base == 'C' && obs.read_base == 'T') {
        return true;
    }
    // G→A in 3' damage zone
    if (in_3p_zone && obs.ref_base == 'G' && obs.read_base == 'A') {
        return true;
    }
    return false;
}

// Check if a mismatch is damage-inconsistent
// C→T at 3' end, G→A at 5' end, or any other mismatch pattern
inline bool is_damage_inconsistent(const MismatchObs& obs, int damage_zone) {
    bool in_5p_zone = (obs.dist_from_5p <= damage_zone);
    bool in_3p_zone = (obs.dist_from_3p <= damage_zone);

    // C→T in 3' zone (wrong end for damage)
    if (in_3p_zone && !in_5p_zone && obs.ref_base == 'C' && obs.read_base == 'T') {
        return true;
    }
    // G→A in 5' zone (wrong end for damage)
    if (in_5p_zone && !in_3p_zone && obs.ref_base == 'G' && obs.read_base == 'A') {
        return true;
    }

    // Any mismatch that isn't C→T or G→A is inconsistent with damage model
    if (!((obs.ref_base == 'C' && obs.read_base == 'T') ||
          (obs.ref_base == 'G' && obs.read_base == 'A'))) {
        return true;
    }

    return false;
}

// Classify a single read as ancient or modern based on its mismatch pattern
// Takes all mismatches observed in the read and computes P(ancient | mismatches)
inline ReadAncientScore classify_read(
    const std::vector<MismatchObs>& mismatches,
    const ReadClassifierConfig& config) {

    ReadAncientScore result;
    result.p_damaged = config.prior_ancient;  // Default to prior
    result.p_length_ancient = 0.5;            // Uninformative without length data
    result.p_ancient = config.prior_ancient;  // Default to prior
    result.llr_damage = 0.0;
    result.llr_length = 0.0;
    result.llr = 0.0;
    result.damage_consistent = 0;
    result.damage_inconsistent = 0;
    result.damage_zone_positions = 0;
    result.total_mismatches = 0;

    // Handle edge case: no mismatches
    // A read with no mismatches provides no damage signal - stay at prior
    if (mismatches.empty()) {
        return result;
    }

    double log_lik_ancient = 0.0;
    double log_lik_modern = 0.0;

    for (const auto& obs : mismatches) {
        // Skip low-quality bases
        if (obs.baseq < config.min_baseq) continue;

        result.total_mismatches++;

        double q_err = phred_to_error_prob(obs.baseq);
        q_err = std::max(0.001, std::min(0.5, q_err));

        // Track if this position is in damage zone
        bool in_5p_zone = (obs.dist_from_5p <= config.damage_zone);
        bool in_3p_zone = (obs.dist_from_3p <= config.damage_zone);
        if (in_5p_zone || in_3p_zone) {
            result.damage_zone_positions++;
        }

        // Count damage consistency
        if (is_damage_consistent(obs, config.damage_zone)) {
            result.damage_consistent++;
        } else if (is_damage_inconsistent(obs, config.damage_zone)) {
            result.damage_inconsistent++;
        }

        // Compute P(mismatch | ancient)
        // For ancient reads, C→T at 5' and G→A at 3' have elevated rates
        double p_mismatch_ancient = 0.0;

        if (obs.ref_base == 'C' && obs.read_base == 'T') {
            // C→T: use 5' damage rate
            double d5 = damage_rate_at_distance(obs.dist_from_5p,
                config.amp_5p, config.lam_5p, config.base_5p);
            // P(C→T | ancient) = d5*(1-q_err) + (1-d5)*(q_err/3)
            p_mismatch_ancient = d5 * (1.0 - q_err) + (1.0 - d5) * (q_err / 3.0);
        } else if (obs.ref_base == 'G' && obs.read_base == 'A') {
            // G→A: use 3' damage rate
            double d3 = damage_rate_at_distance(obs.dist_from_3p,
                config.amp_3p, config.lam_3p, config.base_3p);
            // P(G→A | ancient) = d3*(1-q_err) + (1-d3)*(q_err/3)
            p_mismatch_ancient = d3 * (1.0 - q_err) + (1.0 - d3) * (q_err / 3.0);
        } else {
            // Other mismatches: just sequencing error in ancient
            p_mismatch_ancient = q_err / 3.0;
        }

        // Compute P(mismatch | modern)
        // Modern reads have same error model but no damage
        double p_mismatch_modern = q_err / 3.0;

        // Clamp probabilities
        p_mismatch_ancient = std::max(1e-10, std::min(0.999, p_mismatch_ancient));
        p_mismatch_modern = std::max(1e-10, std::min(0.999, p_mismatch_modern));

        // Accumulate log likelihoods (quality already in error model)
        log_lik_ancient += std::log(p_mismatch_ancient);
        log_lik_modern += std::log(p_mismatch_modern);
    }

    // If no high-quality mismatches were processed, stay at prior
    if (result.total_mismatches == 0) {
        return result;
    }

    // Compute log-likelihood ratio from damage patterns
    result.llr_damage = log_lik_ancient - log_lik_modern;
    result.llr = result.llr_damage;  // Backward compatibility

    // Apply Bayes rule to get damage-based posterior
    // P(damaged | obs) = P(obs | damaged) * P(damaged) / P(obs)
    double log_prior_ancient = std::log(config.prior_ancient);
    double log_prior_modern = std::log(1.0 - config.prior_ancient);

    double log_joint_ancient = log_lik_ancient + log_prior_ancient;
    double log_joint_modern = log_lik_modern + log_prior_modern;

    // Log-sum-exp for normalization
    double max_log = std::max(log_joint_ancient, log_joint_modern);
    double log_evidence = max_log + std::log(
        std::exp(log_joint_ancient - max_log) +
        std::exp(log_joint_modern - max_log));

    result.p_damaged = std::exp(log_joint_ancient - log_evidence);
    result.p_damaged = std::max(1e-6, std::min(1.0 - 1e-6, result.p_damaged));

    // For mismatch-only classification, p_ancient = p_damaged (no length info)
    result.p_ancient = result.p_damaged;

    return result;
}

// Classify a read using FULL likelihood from ALL aligned positions
// Unlike classify_read() which only considers mismatches, this considers:
// - Matches: P(correct_obs | ancient) vs P(correct_obs | modern)
// - Mismatches: as before, with damage-aware likelihoods
//
// Key insight: Matching positions provide WEAK evidence for modern reads
// because ancient reads have damage that sometimes causes mismatches.
// A read with many perfect matches is slightly more likely to be modern.
inline ReadAncientScore classify_read_full(
    const FullReadData& read_data,
    const ReadClassifierConfig& config) {

    ReadAncientScore result;
    result.p_damaged = config.prior_ancient;
    result.p_length_ancient = 0.5;  // Uninformative default
    result.p_ancient = config.prior_ancient;
    result.llr_damage = 0.0;
    result.llr_length = 0.0;
    result.llr = 0.0;
    result.damage_consistent = 0;
    result.damage_inconsistent = 0;
    result.damage_zone_positions = 0;
    result.total_mismatches = 0;

    if (read_data.positions.empty()) {
        return result;
    }

    double log_lik_ancient = 0.0;
    double log_lik_modern = 0.0;

    for (const auto& pos : read_data.positions) {
        // Skip invalid bases
        int ref_idx = detail::base_to_idx_local(pos.ref_base);
        int obs_idx = detail::base_to_idx_local(pos.read_base);
        if (ref_idx < 0 || obs_idx < 0) continue;

        // Skip low-quality bases
        if (pos.baseq < config.min_baseq) continue;

        double q_err = phred_to_error_prob(pos.baseq);
        q_err = std::max(0.001, std::min(0.5, q_err));

        // Track damage zone
        bool in_5p_zone = (pos.dist_5p <= config.damage_zone);
        bool in_3p_zone = (pos.dist_3p <= config.damage_zone);
        if (in_5p_zone || in_3p_zone) {
            result.damage_zone_positions++;
        }

        bool is_match = (pos.ref_base == pos.read_base);

        if (!is_match) {
            result.total_mismatches++;

            // Track damage consistency
            MismatchObs mm{pos.ref_base, pos.read_base, pos.baseq, pos.dist_5p, pos.dist_3p};
            if (is_damage_consistent(mm, config.damage_zone)) {
                result.damage_consistent++;
            } else if (is_damage_inconsistent(mm, config.damage_zone)) {
                result.damage_inconsistent++;
            }
        }

        // Compute damage rates at this position
        double d5 = damage_rate_at_distance(pos.dist_5p,
            config.amp_5p, config.lam_5p, config.base_5p);
        double d3 = damage_rate_at_distance(pos.dist_3p,
            config.amp_3p, config.lam_3p, config.base_3p);

        // Compute P(observed | reference, ancient/modern)
        double p_obs_ancient = 0.0;
        double p_obs_modern = 0.0;

        if (is_match) {
            // Matching position: ref == obs
            // For C: P(C|C, ancient) = (1-d5)*(1-q_err) + d5*(q_err/3)
            // For G: P(G|G, ancient) = (1-d3)*(1-q_err) + d3*(q_err/3)
            // For A,T: P(X|X, ancient) = (1-q_err)
            if (pos.ref_base == 'C') {
                p_obs_ancient = (1.0 - d5) * (1.0 - q_err) + d5 * (q_err / 3.0);
            } else if (pos.ref_base == 'G') {
                p_obs_ancient = (1.0 - d3) * (1.0 - q_err) + d3 * (q_err / 3.0);
            } else {
                p_obs_ancient = 1.0 - q_err;
            }
            p_obs_modern = 1.0 - q_err;
        } else {
            // Mismatch position
            if (pos.ref_base == 'C' && pos.read_base == 'T') {
                // C→T: damage + sequencing or error alone
                p_obs_ancient = d5 * (1.0 - q_err) + (1.0 - d5) * (q_err / 3.0);
            } else if (pos.ref_base == 'G' && pos.read_base == 'A') {
                // G→A: damage + sequencing or error alone
                p_obs_ancient = d3 * (1.0 - q_err) + (1.0 - d3) * (q_err / 3.0);
            } else {
                // Other mismatches: just sequencing error
                p_obs_ancient = q_err / 3.0;
            }
            p_obs_modern = q_err / 3.0;  // Modern: same error model, no damage
        }

        // Clamp and accumulate (quality already in error model)
        p_obs_ancient = std::max(1e-10, std::min(0.999, p_obs_ancient));
        p_obs_modern = std::max(1e-10, std::min(0.999, p_obs_modern));

        log_lik_ancient += std::log(p_obs_ancient);
        log_lik_modern += std::log(p_obs_modern);
    }

    // Store damage-only LLR
    result.llr_damage = log_lik_ancient - log_lik_modern;
    result.llr = result.llr_damage;  // Backward compatibility

    // Compute P(damaged | damage_patterns) - DAMAGE ONLY
    {
        double log_prior = std::log(config.prior_ancient);
        double log_prior_mod = std::log(1.0 - config.prior_ancient);
        double log_joint_anc = log_lik_ancient + log_prior;
        double log_joint_mod = log_lik_modern + log_prior_mod;
        double max_log = std::max(log_joint_anc, log_joint_mod);
        double log_evidence = max_log + std::log(
            std::exp(log_joint_anc - max_log) + std::exp(log_joint_mod - max_log));
        result.p_damaged = std::exp(log_joint_anc - log_evidence);
        result.p_damaged = std::max(1e-6, std::min(1.0 - 1e-6, result.p_damaged));
    }

    // Compute P(ancient | length) - LENGTH ONLY
    double log_lik_len_ancient = 0.0;
    double log_lik_len_modern = 0.0;
    if (config.use_length_model && read_data.read_length > 0) {
        log_lik_len_ancient = length_log_likelihood(
            read_data.read_length,
            config.mode_length_ancient,
            config.std_length_ancient);
        log_lik_len_modern = length_log_likelihood(
            read_data.read_length,
            config.mode_length_modern,
            config.std_length_modern);
        result.llr_length = log_lik_len_ancient - log_lik_len_modern;

        // P(ancient | length) with flat prior (0.5)
        double log_joint_anc = log_lik_len_ancient + std::log(0.5);
        double log_joint_mod = log_lik_len_modern + std::log(0.5);
        double max_log = std::max(log_joint_anc, log_joint_mod);
        double log_evidence = max_log + std::log(
            std::exp(log_joint_anc - max_log) + std::exp(log_joint_mod - max_log));
        result.p_length_ancient = std::exp(log_joint_anc - log_evidence);
        result.p_length_ancient = std::max(1e-6, std::min(1.0 - 1e-6, result.p_length_ancient));
    } else {
        result.p_length_ancient = 0.5;  // Uninformative
        result.llr_length = 0.0;
    }

    // Add identity likelihood if enabled (added to damage likelihood)
    double log_lik_combined_ancient = log_lik_ancient;
    double log_lik_combined_modern = log_lik_modern;

    if (config.use_identity_model && read_data.identity > 0) {
        double log_lik_identity_ancient = identity_log_likelihood(
            read_data.identity,
            config.mode_identity_ancient,
            config.std_identity_ancient);
        double log_lik_identity_modern = identity_log_likelihood(
            read_data.identity,
            config.mode_identity_modern,
            config.std_identity_modern);
        log_lik_combined_ancient += config.identity_weight * log_lik_identity_ancient;
        log_lik_combined_modern += config.identity_weight * log_lik_identity_modern;
    }

    // Add length to combined if enabled
    if (config.use_length_model && read_data.read_length > 0) {
        log_lik_combined_ancient += config.length_weight * log_lik_len_ancient;
        log_lik_combined_modern += config.length_weight * log_lik_len_modern;
    }

    // Compute COMBINED posterior P(ancient | damage, length, identity)
    {
        double log_prior = std::log(config.prior_ancient);
        double log_prior_mod = std::log(1.0 - config.prior_ancient);
        double log_joint_anc = log_lik_combined_ancient + log_prior;
        double log_joint_mod = log_lik_combined_modern + log_prior_mod;
        double max_log = std::max(log_joint_anc, log_joint_mod);
        double log_evidence = max_log + std::log(
            std::exp(log_joint_anc - max_log) + std::exp(log_joint_mod - max_log));
        result.p_ancient = std::exp(log_joint_anc - log_evidence);
        result.p_ancient = std::max(1e-6, std::min(1.0 - 1e-6, result.p_ancient));
    }

    return result;
}

// Classify a read from DamageReadObs observations (position-level data)
// This variant takes per-position observations and a reference sequence
template<typename ReadObs>
ReadAncientScore classify_read_from_pileup(
    const std::vector<ReadObs>& observations,
    const std::vector<char>& ref_bases,  // Reference bases at each position
    const ReadClassifierConfig& config) {

    // Build mismatch list from observations vs reference
    std::vector<MismatchObs> mismatches;

    size_t n = std::min(observations.size(), ref_bases.size());
    for (size_t i = 0; i < n; i++) {
        const auto& obs = observations[i];
        char ref = ref_bases[i];

        // Skip if bases match
        if (obs.base == ref) continue;

        // Skip N or invalid bases
        if (detail::base_to_idx_local(obs.base) < 0 || detail::base_to_idx_local(ref) < 0) continue;

        MismatchObs mm;
        mm.ref_base = ref;
        mm.read_base = obs.base;
        mm.baseq = obs.baseq;
        mm.dist_from_5p = obs.dist_from_5p;
        mm.dist_from_3p = obs.dist_from_3p;

        mismatches.push_back(mm);
    }

    return classify_read(mismatches, config);
}

// Batch classify multiple reads with the same config
inline std::vector<ReadAncientScore> classify_reads_batch(
    const std::vector<std::vector<MismatchObs>>& reads_mismatches,
    const ReadClassifierConfig& config) {

    std::vector<ReadAncientScore> results;
    results.reserve(reads_mismatches.size());

    for (const auto& mismatches : reads_mismatches) {
        results.push_back(classify_read(mismatches, config));
    }

    return results;
}

// Estimate per-bin ancient fraction from read classifications
// Uses EM-style update: new_prior = mean(p_ancient) across all reads
inline double estimate_bin_ancient_fraction(
    const std::vector<ReadAncientScore>& read_scores,
    double min_fraction = 0.1,
    double max_fraction = 0.95) {

    if (read_scores.empty()) {
        return 0.5;  // Default if no reads
    }

    double sum_p_ancient = 0.0;
    double sum_weight = 0.0;

    for (const auto& score : read_scores) {
        // Weight by damage zone coverage (more informative reads count more)
        double weight = 1.0;
        if (score.damage_zone_positions > 0) {
            weight = 1.0 + 0.1 * score.damage_zone_positions;
        }
        sum_p_ancient += weight * score.p_ancient;
        sum_weight += weight;
    }

    double fraction = sum_p_ancient / std::max(1.0, sum_weight);
    return std::max(min_fraction, std::min(max_fraction, fraction));
}

}  // namespace amber

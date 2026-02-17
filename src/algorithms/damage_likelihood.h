#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <htslib/sam.h>  // For bam_seqi, seq_nt16_str
#include "../binner/internal/constants.h"  // For kSeqErrorDenom

namespace amber {

// Shared read observation struct for damage likelihood calculations
struct DamageReadObs {
    char base;
    uint8_t baseq;
    int dist_from_5p;
    int dist_from_3p;
    int read_len;
    bool is_reverse;
};

// Pre-computed Phred to error probability lookup table
// Avoids expensive std::pow() calls in hot loops
namespace detail {
    struct PhredLUT {
        double table[256];
        PhredLUT() {
            for (int q = 0; q < 256; q++) {
                table[q] = std::pow(10.0, -static_cast<double>(q) / 10.0);
            }
        }
    };
    inline const PhredLUT& get_phred_lut() {
        static const PhredLUT lut;
        return lut;
    }
}

// Convert Phred quality score to error probability: P(error) = 10^(-Q/10)
inline double phred_to_error_prob(uint8_t baseq) {
    return detail::get_phred_lut().table[baseq];
}

// Damage rate at distance from read end using exponential decay model
// D(z) = amplitude * exp(-lambda * (z - 1)) + baseline
inline double damage_rate_at_distance(int dist_from_end, double amplitude, double lambda, double baseline) {
    double rate = amplitude * std::exp(-lambda * (dist_from_end - 1)) + baseline;
    return std::max(0.001, std::min(0.999, rate));
}

// Compute log likelihood ratio for C→T correction using per-read evidence
// Uses each read's distance from 5' end for position-specific damage rate
// H1: True base is C (was damaged to T, needs correction)
// H0: True base is T (no correction needed)
template<typename ReadObs>
double compute_ct_log_lr(
    const std::vector<ReadObs>& observations,
    double amplitude, double lambda, double baseline) {

    double log_lik_C = 0.0;
    double log_lik_T = 0.0;

    for (const auto& obs : observations) {
        if (obs.base != 'C' && obs.base != 'T') continue;

        double q_err = std::max(0.001, std::min(0.999, phred_to_error_prob(obs.baseq)));
        double d = damage_rate_at_distance(obs.dist_from_5p, amplitude, lambda, baseline);

        if (obs.base == 'T') {
            // H1 (true=C): T from damage OR sequencing error (C→T specifically)
            double p_T_given_C = d * (1.0 - q_err) + (1.0 - d) * (q_err / kSeqErrorDenom);
            log_lik_C += std::log(std::max(0.001, p_T_given_C));

            // H0 (true=T): T observed correctly
            log_lik_T += std::log(1.0 - q_err);
        } else {
            // H1 (true=C): C observed correctly, or damaged then error back (T→C)
            double p_C_given_C = (1.0 - d) * (1.0 - q_err) + d * (q_err / kSeqErrorDenom);
            log_lik_C += std::log(std::max(0.001, p_C_given_C));

            // H0 (true=T): C from sequencing error (T→C specifically)
            log_lik_T += std::log(q_err / kSeqErrorDenom);
        }
    }

    return log_lik_C - log_lik_T;
}

// Compute log likelihood ratio for G→A correction using per-read evidence
// Uses each read's distance from 3' end for position-specific damage rate
// H1: True base is G (was damaged to A, needs correction)
// H0: True base is A (no correction needed)
template<typename ReadObs>
double compute_ga_log_lr(
    const std::vector<ReadObs>& observations,
    double amplitude, double lambda, double baseline) {

    double log_lik_G = 0.0;
    double log_lik_A = 0.0;

    for (const auto& obs : observations) {
        if (obs.base != 'G' && obs.base != 'A') continue;

        double q_err = std::max(0.001, std::min(0.999, phred_to_error_prob(obs.baseq)));
        double d = damage_rate_at_distance(obs.dist_from_3p, amplitude, lambda, baseline);

        if (obs.base == 'A') {
            // H1 (true=G): A from damage OR sequencing error (G→A specifically)
            double p_A_given_G = d * (1.0 - q_err) + (1.0 - d) * (q_err / kSeqErrorDenom);
            log_lik_G += std::log(std::max(0.001, p_A_given_G));

            // H0 (true=A): A observed correctly
            log_lik_A += std::log(1.0 - q_err);
        } else {
            // H1 (true=G): G observed correctly, or damaged then error back (A→G)
            double p_G_given_G = (1.0 - d) * (1.0 - q_err) + d * (q_err / kSeqErrorDenom);
            log_lik_G += std::log(std::max(0.001, p_G_given_G));

            // H0 (true=A): G from sequencing error (A→G specifically)
            log_lik_A += std::log(q_err / kSeqErrorDenom);
        }
    }

    return log_lik_G - log_lik_A;
}

// Default damage model parameters (calibrated for ancient DNA)
struct DamageModelParams {
    double amplitude_5p = 0.3;   // C→T damage amplitude at 5' end
    double amplitude_3p = 0.3;   // G→A damage amplitude at 3' end
    double lambda = 0.5;         // Exponential decay rate
    double baseline = 0.01;      // Background damage rate
};

// Compute per-read probability of being damaged (ancient)
// Uses terminal mismatches as evidence
// Returns p(ancient | observations) using Bayes with flat prior
inline double compute_read_damage_probability(
    const uint8_t* seq,        // BAM sequence (4-bit encoded)
    const uint8_t* qual,       // Base qualities
    const uint8_t* ref_seq,    // Reference sequence at this position
    int read_len,
    int n_terminal = 5,        // Number of terminal positions to check
    const DamageModelParams& params = {}) {

    if (read_len < 2 * n_terminal) {
        return 0.5;  // Too short, uninformative
    }

    double log_lik_ancient = 0.0;
    double log_lik_modern = 0.0;

    // Check 5' terminal positions for C→T
    for (int i = 0; i < std::min(n_terminal, read_len); i++) {
        uint8_t base = bam_seqi(seq, i);
        char base_char = seq_nt16_str[base];
        uint8_t ref_base = ref_seq[i];

        if (ref_base == 'C' || ref_base == 'c') {
            double q_err = phred_to_error_prob(qual[i]);
            double d = damage_rate_at_distance(i + 1, params.amplitude_5p, params.lambda, params.baseline);

            if (base_char == 'T') {
                // C→T observed: consistent with damage
                log_lik_ancient += std::log(d * (1.0 - q_err) + (1.0 - d) * q_err / 3.0);
                log_lik_modern += std::log(q_err / 3.0);  // Only sequencing error
            } else if (base_char == 'C') {
                // C→C observed: no damage at this position
                log_lik_ancient += std::log((1.0 - d) * (1.0 - q_err));
                log_lik_modern += std::log(1.0 - q_err);
            }
        }
    }

    // Check 3' terminal positions for G→A
    for (int i = 0; i < std::min(n_terminal, read_len); i++) {
        int pos = read_len - 1 - i;
        uint8_t base = bam_seqi(seq, pos);
        char base_char = seq_nt16_str[base];
        uint8_t ref_base = ref_seq[pos];

        if (ref_base == 'G' || ref_base == 'g') {
            double q_err = phred_to_error_prob(qual[pos]);
            double d = damage_rate_at_distance(i + 1, params.amplitude_3p, params.lambda, params.baseline);

            if (base_char == 'A') {
                // G→A observed: consistent with damage
                log_lik_ancient += std::log(d * (1.0 - q_err) + (1.0 - d) * q_err / 3.0);
                log_lik_modern += std::log(q_err / 3.0);
            } else if (base_char == 'G') {
                // G→G observed: no damage
                log_lik_ancient += std::log((1.0 - d) * (1.0 - q_err));
                log_lik_modern += std::log(1.0 - q_err);
            }
        }
    }

    // Convert to probability with flat prior
    double log_diff = log_lik_ancient - log_lik_modern;
    // Clip to avoid overflow
    log_diff = std::max(-20.0, std::min(20.0, log_diff));
    double p_ancient = 1.0 / (1.0 + std::exp(-log_diff));

    return p_ancient;
}

}  // namespace amber

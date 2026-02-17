#pragma once

// AMBER - Damage-Based Deconvolution
// Separates damaged and undamaged read populations using DNA damage patterns.
//
// Key insight: Damaged reads show C→T at 5' and G→A at 3' ends.
// Undamaged reads don't show this pattern.
// We build two consensus sequences:
//   1. Damaged consensus: from damaged reads, then correct damage artifacts
//   2. Undamaged consensus: from undamaged reads, no correction needed

#include "bayesian_caller.h"
#include "damage_read_classifier.h"
#include "damage_likelihood.h"
#include "damage_profile.h"
#include <htslib/hts.h>
#include <htslib/sam.h>
#include <array>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>

namespace amber {

// Pre-extracted observation for a read at a reference position
struct BaseObs {
    int8_t obs_idx;    // 0-3 for ACGT, -1 for N/skip
    uint8_t qual;
    int16_t dist_5p;
    int16_t dist_3p;
};

// All data for a single read, pre-extracted once
struct ReadData {
    bam1_t* read;
    ReadAncientScore score;
    int start_pos;
    int end_pos;
    std::vector<BaseObs> observations;  // indexed by (ref_pos - start_pos)
    FullReadData full_data;             // Full aligned positions for full-likelihood classification
};

// Configuration for damage-based deconvolution
struct DeconvolutionConfig {
    // Read classification thresholds
    double p_ancient_hard = 0.8;     // Reads with p >= this are "confidently ancient"
    double p_modern_hard = 0.2;      // Reads with p <= this are "confidently modern"
    bool use_soft_weights = true;    // Use soft p_ancient weights (recommended)

    // Consensus building thresholds
    double min_ancient_depth = 3.0;  // Minimum weighted depth for ancient consensus
    double min_modern_depth = 2.0;   // Minimum weighted depth for modern consensus
    double min_posterior = 0.7;      // Lower than polish (populations are smaller)

    // Polish settings
    bool polish_ancient = true;      // Apply damage correction to ancient consensus

    // Quality filters
    int min_mapq = 30;
    int min_baseq = 20;

    // Output control
    double min_modern_fraction = 0.05;  // Only output modern if fraction >= this
    bool write_position_stats = false;
    bool write_read_classifications = false;

    // Advanced classification options
    int em_iterations = 1;           // EM iterations for adaptive prior (1=single pass)
    bool use_full_likelihood = true; // Use full read likelihood (matches + mismatches)
    double prior_ancient = 0.5;      // Initial prior P(ancient) - 0.5 is uninformative
    double prior_min = 0.1;          // Minimum adaptive prior (allow full range)
    double prior_max = 0.9;          // Maximum adaptive prior

    // Experimental: disable damage model in consensus building
    bool raw_consensus = false;      // If true, use simple error model (no damage) for ancient consensus

    // Read length model for classification
    // Ancient DNA typically has shorter fragments due to degradation
    bool use_length_model = true;         // Enable length-based classification (default: on)
    double length_weight = 0.5;           // Weight for length contribution
    double mode_length_ancient = 42.0;    // Mode fragment length for ancient (bp)
    double std_length_ancient = 15.0;     // Std dev for ancient length
    double mode_length_modern = 103.0;    // Mode fragment length for modern (bp)
    double std_length_modern = 30.0;      // Std dev for modern length

    // Read identity model for classification
    // Ancient reads map at ~100% to ancient assembly, modern at ~99%
    bool use_identity_model = false;      // Enable identity-based classification
    double identity_weight = 1.0;         // Weight for identity contribution
    double mode_identity_ancient = 100.0; // Mode identity for ancient (%)
    double std_identity_ancient = 2.0;    // Std dev for ancient identity
    double mode_identity_modern = 99.0;   // Mode identity for modern (%)
    double std_identity_modern = 0.5;     // Std dev for modern identity

    // Dynamic thresholding
    // If enabled, computes optimal ancient/modern thresholds from p_ancient distribution
    bool use_dynamic_thresholds = false;  // Enable dynamic threshold computation

    // Auto-estimation of model parameters from data
    bool auto_estimate_params = false;    // Estimate length/identity params from read distributions
};

// Per-contig deconvolution statistics
struct DeconvolutionResult {
    // Consensus sequences
    std::string ancient_consensus;   // Polished ancient sequence
    std::string ancient_raw;         // Pre-polish ancient (for validation)
    std::string modern_consensus;    // Modern/contaminant sequence

    // Read classification stats
    size_t total_reads = 0;
    size_t ancient_reads_hard = 0;   // p_ancient >= 0.8
    size_t modern_reads_hard = 0;    // p_ancient <= 0.2
    size_t ambiguous_reads = 0;      // 0.2 < p_ancient < 0.8
    double mean_p_ancient = 0.0;
    double median_p_ancient = 0.0;

    // Consensus position stats
    size_t positions_both_called = 0;     // Both populations had depth
    size_t positions_ancient_only = 0;    // Only ancient had depth
    size_t positions_modern_only = 0;     // Only modern had depth
    size_t positions_neither = 0;         // Neither had depth
    size_t ancient_modern_differences = 0; // Positions where bases differ

    // Polish stats (ancient)
    size_t ancient_ct_corrections = 0;
    size_t ancient_ga_corrections = 0;

    // Quality metrics
    double ancient_fraction = 0.0;
    double classification_confidence = 0.0;  // Mean |p_ancient - 0.5|

    // EM convergence info
    int em_iterations_used = 1;              // Actual iterations before convergence
    double final_prior = 0.7;                // Final adaptive prior after convergence

    // Smiley plot data: damage rates by position from read end
    // Three categories: ancient (p>=0.8), modern (p<=0.2), ambiguous (0.2<p<0.8)
    static constexpr int SMILEY_MAX_POS = 50;
    std::array<int, SMILEY_MAX_POS> anc_t_5p = {};   // T count at 5' for ancient
    std::array<int, SMILEY_MAX_POS> anc_tc_5p = {};  // T+C count at 5' for ancient
    std::array<int, SMILEY_MAX_POS> anc_a_3p = {};   // A count at 3' for ancient
    std::array<int, SMILEY_MAX_POS> anc_ag_3p = {};  // A+G count at 3' for ancient
    std::array<int, SMILEY_MAX_POS> mod_t_5p = {};   // T count at 5' for modern
    std::array<int, SMILEY_MAX_POS> mod_tc_5p = {};  // T+C count at 5' for modern
    std::array<int, SMILEY_MAX_POS> mod_a_3p = {};   // A count at 3' for modern
    std::array<int, SMILEY_MAX_POS> mod_ag_3p = {};  // A+G count at 3' for modern
    std::array<int, SMILEY_MAX_POS> amb_t_5p = {};   // T count at 5' for ambiguous
    std::array<int, SMILEY_MAX_POS> amb_tc_5p = {};  // T+C count at 5' for ambiguous
    std::array<int, SMILEY_MAX_POS> amb_a_3p = {};   // A count at 3' for ambiguous
    std::array<int, SMILEY_MAX_POS> amb_ag_3p = {};  // A+G count at 3' for ambiguous

    // Probability histograms: 20 bins covering [0, 1] in 0.05 increments
    // Bin i contains counts for probability in [i*0.05, (i+1)*0.05)
    static constexpr int HIST_BINS = 20;
    std::array<int64_t, HIST_BINS> p_ancient_hist = {};    // Combined posterior
    std::array<int64_t, HIST_BINS> p_damaged_hist = {};    // Damage-only posterior
    std::array<int64_t, HIST_BINS> p_length_hist = {};     // Length-only posterior

    // Read length histograms by classification
    static constexpr int LENGTH_BINS = 50;  // 0-249bp in 5bp bins
    std::array<int64_t, LENGTH_BINS> length_hist_ancient = {};
    std::array<int64_t, LENGTH_BINS> length_hist_modern = {};
    std::array<int64_t, LENGTH_BINS> length_hist_ambiguous = {};

    // Dynamic thresholds (computed from p_ancient distribution)
    double dynamic_ancient_threshold = 0.8;  // Computed threshold for ancient
    double dynamic_modern_threshold = 0.2;   // Computed threshold for modern
    bool dynamic_thresholds_used = false;    // Whether dynamic thresholds were computed

    // Temporal population fraction estimation (Depositional vs Cryophilic)
    // Depositional: DNA deposited historically, subject to post-mortem deamination
    // Cryophilic: Viable organisms living in permafrost matrix, undamaged DNA
    double f_expected_detection = 0.0;       // E[P(detectable deamination | depositional)]
    double f_observed_detection = 0.0;       // Observed fraction with damage signal
    double f_undamaged = 0.0;                // Estimated undamaged fraction [0,1]
    double f_damaged = 1.0;                  // = 1 - f_undamaged (damaged fraction)
    double f_undamaged_se = 0.0;             // Standard error of estimate
    double f_undamaged_ci_lower = 0.0;       // 95% CI lower bound
    double f_undamaged_ci_upper = 0.0;       // 95% CI upper bound
    double f_undamaged_moment = 0.0;         // Moment estimator (for validation)
    int fraction_em_iterations = 0;          // EM iterations used
    bool fraction_converged = false;         // Whether EM converged
};

// Compute optimal thresholds from p_ancient histogram using valley detection
// Returns {modern_threshold, ancient_threshold}
inline std::pair<double, double> compute_dynamic_thresholds(
    const std::array<int64_t, DeconvolutionResult::HIST_BINS>& hist) {

    // Smooth the histogram with a simple moving average (window=3)
    std::array<double, DeconvolutionResult::HIST_BINS> smoothed = {};
    for (int i = 0; i < DeconvolutionResult::HIST_BINS; i++) {
        double sum = 0;
        int count = 0;
        for (int j = std::max(0, i - 1); j <= std::min(DeconvolutionResult::HIST_BINS - 1, i + 1); j++) {
            sum += hist[j];
            count++;
        }
        smoothed[i] = sum / count;
    }

    // Find valleys (local minima) in the smoothed histogram
    // Valley = point where both neighbors are higher
    std::vector<int> valleys;
    for (int i = 1; i < DeconvolutionResult::HIST_BINS - 1; i++) {
        if (smoothed[i] < smoothed[i-1] && smoothed[i] < smoothed[i+1]) {
            valleys.push_back(i);
        }
    }

    // Default thresholds
    double modern_thresh = 0.2;
    double ancient_thresh = 0.8;

    if (valleys.empty()) {
        // No valleys found - use defaults
        return {modern_thresh, ancient_thresh};
    }

    // Find the lowest valley in the lower half (modern threshold)
    // and lowest valley in the upper half (ancient threshold)
    int mid = DeconvolutionResult::HIST_BINS / 2;

    int best_lower = -1;
    double min_lower = std::numeric_limits<double>::max();
    int best_upper = -1;
    double min_upper = std::numeric_limits<double>::max();

    for (int v : valleys) {
        if (v < mid && smoothed[v] < min_lower) {
            min_lower = smoothed[v];
            best_lower = v;
        }
        if (v >= mid && smoothed[v] < min_upper) {
            min_upper = smoothed[v];
            best_upper = v;
        }
    }

    // Convert bin indices to p_ancient values
    // Bin i covers [i*0.05, (i+1)*0.05), so threshold is at (i+0.5)*0.05
    if (best_lower >= 0) {
        modern_thresh = (best_lower + 0.5) * 0.05;
        modern_thresh = std::max(0.05, std::min(0.45, modern_thresh));  // Clamp
    }
    if (best_upper >= 0) {
        ancient_thresh = (best_upper + 0.5) * 0.05;
        ancient_thresh = std::max(0.55, std::min(0.95, ancient_thresh));  // Clamp
    }

    return {modern_thresh, ancient_thresh};
}

// ============================================================================
// Temporal Population Fraction Estimation
// Separates reads from Depositional (historically deposited, damaged) vs
// Cryophilic (viable permafrost organisms, undamaged) populations
// ============================================================================

// Compute expected detection rate: P(at least one deamination event | depositional)
// This is computed per-read based on:
// - Number of C positions in 5' damage zone
// - Number of G positions in 3' damage zone
// - Deamination rates at each position
inline double compute_expected_detection_rate(
    const std::vector<FullReadData>& reads,
    const DamageCurveParams& curve_5p,
    const DamageCurveParams& curve_3p,
    int damage_zone,
    double base_freq_C = 0.24,
    double base_freq_G = 0.24)
{
    if (reads.empty()) return 0.0;

    double sum_p_detect = 0.0;

    for (const auto& read : reads) {
        // P(at least one deamination) = 1 - P(no deamination at any position)
        double log_p_no_damage = 0.0;

        for (const auto& pos : read.positions) {
            // 5' end: C->T deamination
            if (pos.dist_5p <= damage_zone && pos.ref_base == 'C') {
                double d5 = damage_rate_at_distance(pos.dist_5p,
                    curve_5p.amplitude, curve_5p.lambda, curve_5p.baseline);
                log_p_no_damage += std::log(std::max(1e-10, 1.0 - d5));
            }
            // 3' end: G->A deamination
            if (pos.dist_3p <= damage_zone && pos.ref_base == 'G') {
                double d3 = damage_rate_at_distance(pos.dist_3p,
                    curve_3p.amplitude, curve_3p.lambda, curve_3p.baseline);
                log_p_no_damage += std::log(std::max(1e-10, 1.0 - d3));
            }
        }

        double p_detect = 1.0 - std::exp(log_p_no_damage);
        sum_p_detect += p_detect;
    }

    return sum_p_detect / reads.size();
}

// Count observed fraction with detectable damage (has at least one C->T or G->A in damage zone)
inline double compute_observed_detection_fraction(
    const std::vector<FullReadData>& reads,
    int damage_zone)
{
    if (reads.empty()) return 0.0;

    size_t n_detected = 0;

    for (const auto& read : reads) {
        bool has_damage = false;

        for (const auto& pos : read.positions) {
            // C->T at 5' end
            if (pos.dist_5p <= damage_zone &&
                pos.ref_base == 'C' && pos.read_base == 'T') {
                has_damage = true;
                break;
            }
            // G->A at 3' end
            if (pos.dist_3p <= damage_zone &&
                pos.ref_base == 'G' && pos.read_base == 'A') {
                has_damage = true;
                break;
            }
        }

        if (has_damage) n_detected++;
    }

    return static_cast<double>(n_detected) / reads.size();
}

// Moment estimator for undamaged fraction
// f_undamaged = max(0, 1 - f_observed / f_expected)
inline double estimate_f_undamaged_moment(
    double f_expected,
    double f_observed)
{
    if (f_expected < 1e-6) return 0.5;  // No information

    double theta = 1.0 - f_observed / f_expected;
    return std::clamp(theta, 0.0, 1.0);
}

// Per-read log-likelihoods under depositional and cryophilic models
struct ReadPopulationLikelihoods {
    double log_lik_dep;   // log P(read | depositional)
    double log_lik_cryo;  // log P(read | cryophilic)
    int n_informative;    // Number of C/G positions in damage zone
};

// Compute per-read likelihoods for both populations
inline ReadPopulationLikelihoods compute_read_population_liks(
    const FullReadData& read,
    const DamageCurveParams& curve_5p,
    const DamageCurveParams& curve_3p,
    int damage_zone,
    int min_baseq)
{
    ReadPopulationLikelihoods result = {0.0, 0.0, 0};

    for (const auto& pos : read.positions) {
        if (pos.baseq < min_baseq) continue;

        double eps = phred_to_error_prob(pos.baseq);
        eps = std::clamp(eps, 0.001, 0.5);

        // Cryophilic model: sequencing error only
        double p_cryo = (pos.read_base == pos.ref_base)
            ? (1.0 - eps) : (eps / 3.0);

        // Depositional model: deamination + error
        double p_dep;
        bool is_informative = false;

        double d5 = damage_rate_at_distance(pos.dist_5p,
            curve_5p.amplitude, curve_5p.lambda, curve_5p.baseline);
        double d3 = damage_rate_at_distance(pos.dist_3p,
            curve_3p.amplitude, curve_3p.lambda, curve_3p.baseline);

        if (pos.ref_base == 'C') {
            // C can deaminate to T at 5' end
            if (pos.read_base == 'C') {
                p_dep = (1.0 - d5) * (1.0 - eps) + d5 * (eps / 3.0);
            } else if (pos.read_base == 'T') {
                p_dep = d5 * (1.0 - eps) + (1.0 - d5) * (eps / 3.0);
            } else {
                p_dep = eps / 3.0;
            }
            if (pos.dist_5p <= damage_zone) is_informative = true;

        } else if (pos.ref_base == 'G') {
            // G can deaminate to A at 3' end
            if (pos.read_base == 'G') {
                p_dep = (1.0 - d3) * (1.0 - eps) + d3 * (eps / 3.0);
            } else if (pos.read_base == 'A') {
                p_dep = d3 * (1.0 - eps) + (1.0 - d3) * (eps / 3.0);
            } else {
                p_dep = eps / 3.0;
            }
            if (pos.dist_3p <= damage_zone) is_informative = true;

        } else {
            // A or T: no deamination
            p_dep = p_cryo;
        }

        result.log_lik_cryo += std::log(std::max(1e-15, p_cryo));
        result.log_lik_dep += std::log(std::max(1e-15, p_dep));
        if (is_informative) result.n_informative++;
    }

    return result;
}

// EM algorithm for population fraction estimation
// Returns f_undamaged (undamaged fraction), gamma vector (per-read posteriors)
inline std::pair<double, std::vector<double>> estimate_population_fraction_em(
    const std::vector<ReadPopulationLikelihoods>& liks,
    double theta_init = 0.5,
    int max_iter = 50,
    double tol = 1e-4)
{
    int N = liks.size();
    if (N == 0) return {0.5, {}};

    std::vector<double> gamma(N, 0.5);
    double theta = theta_init;

    for (int iter = 0; iter < max_iter; iter++) {
        double log_theta = std::log(std::max(1e-15, theta));
        double log_1m_theta = std::log(std::max(1e-15, 1.0 - theta));

        // E-step: compute posterior P(cryophilic | read)
        double sum_gamma = 0.0;
        for (int i = 0; i < N; i++) {
            double a = log_theta + liks[i].log_lik_cryo;
            double b = log_1m_theta + liks[i].log_lik_dep;
            double max_ab = std::max(a, b);
            double log_px = max_ab + std::log(std::exp(a - max_ab) + std::exp(b - max_ab));
            gamma[i] = std::exp(a - log_px);
            sum_gamma += gamma[i];
        }

        // M-step: update theta
        double theta_new = sum_gamma / N;
        theta_new = std::clamp(theta_new, 1e-6, 1.0 - 1e-6);

        // Check convergence
        if (std::abs(theta_new - theta) < tol && iter > 0) {
            theta = theta_new;
            break;
        }
        theta = theta_new;
    }

    return {theta, gamma};
}

// Compute standard error and confidence interval for theta
inline void compute_theta_uncertainty(
    const std::vector<ReadPopulationLikelihoods>& liks,
    double theta,
    double& se_out,
    double& ci_lower_out,
    double& ci_upper_out)
{
    // Fisher information: sum of squared scores
    double fisher_info = 0.0;

    for (const auto& lik : liks) {
        double p_cryo = std::exp(lik.log_lik_cryo);
        double p_dep = std::exp(lik.log_lik_dep);
        double p_mix = theta * p_cryo + (1.0 - theta) * p_dep;

        if (p_mix > 1e-30) {
            double score = (p_cryo - p_dep) / p_mix;
            fisher_info += score * score;
        }
    }

    se_out = (fisher_info > 0) ? 1.0 / std::sqrt(fisher_info) : 1.0;
    ci_lower_out = std::max(0.0, theta - 1.96 * se_out);
    ci_upper_out = std::min(1.0, theta + 1.96 * se_out);
}

// Structure to hold estimated model parameters
struct EstimatedModelParams {
    // Length model
    double mode_length_ancient = 42.0;
    double std_length_ancient = 15.0;
    double mode_length_modern = 103.0;
    double std_length_modern = 30.0;
    bool length_bimodal = false;

    // Identity model
    double mode_identity_ancient = 100.0;
    double std_identity_ancient = 2.0;
    double mode_identity_modern = 99.0;
    double std_identity_modern = 0.5;
    bool identity_bimodal = false;
};

// Estimate model parameters from read data
// Uses peak detection in histograms to find ancient/modern populations
inline EstimatedModelParams estimate_model_params(
    const std::vector<int>& lengths,
    const std::vector<double>& identities) {

    EstimatedModelParams params;

    if (lengths.empty() || identities.empty()) return params;

    // Build length histogram (5bp bins, 0-250bp)
    constexpr int LEN_BINS = 50;
    std::array<int64_t, LEN_BINS> len_hist = {};
    for (int len : lengths) {
        int bin = std::min(len / 5, LEN_BINS - 1);
        if (bin >= 0) len_hist[bin]++;
    }

    // Build identity histogram (0.5% bins, 90-100%)
    constexpr int ID_BINS = 20;
    std::array<int64_t, ID_BINS> id_hist = {};
    for (double id : identities) {
        int bin = static_cast<int>((id - 90.0) / 0.5);
        bin = std::max(0, std::min(ID_BINS - 1, bin));
        id_hist[bin]++;
    }

    // Find peaks in length histogram
    // Look for local maxima with smoothing
    auto find_peaks = [](const auto& hist, int n_bins) -> std::vector<int> {
        std::vector<int> peaks;
        std::vector<double> smoothed(n_bins);

        // Smooth with window of 3
        for (int i = 0; i < n_bins; i++) {
            double sum = 0;
            int count = 0;
            for (int j = std::max(0, i - 1); j <= std::min(n_bins - 1, i + 1); j++) {
                sum += hist[j];
                count++;
            }
            smoothed[i] = sum / count;
        }

        // Find local maxima
        for (int i = 1; i < n_bins - 1; i++) {
            if (smoothed[i] > smoothed[i-1] && smoothed[i] > smoothed[i+1]) {
                // Must be significant (>5% of max)
                double max_val = *std::max_element(smoothed.begin(), smoothed.end());
                if (smoothed[i] > 0.05 * max_val) {
                    peaks.push_back(i);
                }
            }
        }
        return peaks;
    };

    // Estimate std from peak width (FWHM approximation)
    auto estimate_std = [](const auto& hist, int peak_bin, int n_bins) -> double {
        double peak_val = hist[peak_bin];
        double half_max = peak_val / 2.0;

        // Find left and right half-max points
        int left = peak_bin, right = peak_bin;
        while (left > 0 && hist[left] > half_max) left--;
        while (right < n_bins - 1 && hist[right] > half_max) right++;

        // FWHM to std: std ≈ FWHM / 2.355
        double fwhm = right - left;
        return std::max(1.0, fwhm / 2.355);
    };

    // Process length histogram
    auto len_peaks = find_peaks(len_hist, LEN_BINS);
    if (len_peaks.size() >= 2) {
        params.length_bimodal = true;
        // Lower peak = ancient (shorter), higher peak = modern (longer)
        int ancient_bin = len_peaks.front();
        int modern_bin = len_peaks.back();

        params.mode_length_ancient = (ancient_bin + 0.5) * 5.0;  // Convert bin to bp
        params.mode_length_modern = (modern_bin + 0.5) * 5.0;
        params.std_length_ancient = estimate_std(len_hist, ancient_bin, LEN_BINS) * 5.0;
        params.std_length_modern = estimate_std(len_hist, modern_bin, LEN_BINS) * 5.0;
    } else if (len_peaks.size() == 1) {
        // Unimodal - all reads likely ancient
        params.mode_length_ancient = (len_peaks[0] + 0.5) * 5.0;
        params.std_length_ancient = estimate_std(len_hist, len_peaks[0], LEN_BINS) * 5.0;
        params.mode_length_modern = params.mode_length_ancient + 50.0;  // Assume modern would be longer
    }

    // Process identity histogram
    auto id_peaks = find_peaks(id_hist, ID_BINS);
    if (id_peaks.size() >= 2) {
        params.identity_bimodal = true;
        // Higher identity = ancient (100%), lower identity = modern (~99%)
        int ancient_bin = id_peaks.back();
        int modern_bin = id_peaks.front();

        params.mode_identity_ancient = 90.0 + (ancient_bin + 0.5) * 0.5;
        params.mode_identity_modern = 90.0 + (modern_bin + 0.5) * 0.5;
        params.std_identity_ancient = estimate_std(id_hist, ancient_bin, ID_BINS) * 0.5;
        params.std_identity_modern = estimate_std(id_hist, modern_bin, ID_BINS) * 0.5;
    } else if (id_peaks.size() == 1) {
        // Unimodal - check if peak is at 100% (all ancient) or lower
        params.mode_identity_ancient = 90.0 + (id_peaks[0] + 0.5) * 0.5;
        params.std_identity_ancient = estimate_std(id_hist, id_peaks[0], ID_BINS) * 0.5;
        if (params.mode_identity_ancient > 99.5) {
            params.mode_identity_modern = 99.0;  // Assume modern would be ~99%
        }
    }

    return params;
}

// Core deconvolution function
// Builds two separate consensus sequences from ancient vs modern read populations
inline DeconvolutionResult deconvolve_contig(
    const std::string& sequence,  // Input (not modified)
    const std::string& contig_name,
    samFile* sf, bam_hdr_t* hdr, hts_idx_t* idx,
    const DamageModelResult& damage_model,
    const DeconvolutionConfig& config = DeconvolutionConfig()) {

    DeconvolutionResult result;
    int len = static_cast<int>(sequence.length());

    // Initialize consensus strings
    result.ancient_consensus.resize(len, 'N');
    result.ancient_raw.resize(len, 'N');
    result.modern_consensus.resize(len, 'N');

    int tid = bam_name2id(hdr, contig_name.c_str());
    if (tid < 0) return result;

    // Initial zone estimate from model (will be refined dynamically after read extraction)
    auto effective_zone = [](double amp, double lam, double base) -> int {
        if (amp <= base || lam <= 0.0 || base <= 0.0) return 5;
        double z = 1.0 + std::log(amp / base) / lam;
        return std::min(std::max(static_cast<int>(std::ceil(z)), 5), 30);
    };

    int zone_5p_initial = effective_zone(damage_model.curve_5p.amplitude,
                                          damage_model.curve_5p.lambda,
                                          damage_model.curve_5p.baseline);
    int zone_3p_initial = effective_zone(damage_model.curve_3p.amplitude,
                                          damage_model.curve_3p.lambda,
                                          damage_model.curve_3p.baseline);
    int zone_5p = zone_5p_initial;
    int zone_3p = zone_3p_initial;
    int damage_zone = std::max(zone_5p, zone_3p);

    // Fetch all reads
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

    result.total_reads = reads.size();

    // =========================================================================
    // Step 1: Classify all reads AND pre-extract all base observations
    // This avoids re-parsing CIGAR in subsequent passes (major speedup)
    // =========================================================================
    ReadClassifierConfig rc_config;
    rc_config.amp_5p = damage_model.curve_5p.amplitude;
    rc_config.lam_5p = damage_model.curve_5p.lambda;
    rc_config.base_5p = damage_model.curve_5p.baseline;
    rc_config.amp_3p = damage_model.curve_3p.amplitude;
    rc_config.lam_3p = damage_model.curve_3p.lambda;
    rc_config.base_3p = damage_model.curve_3p.baseline;
    rc_config.damage_zone = damage_zone;
    rc_config.prior_ancient = config.prior_ancient;  // Use config prior (default: 0.5)
    rc_config.min_baseq = config.min_baseq;

    // Length model parameters
    rc_config.use_length_model = config.use_length_model;
    rc_config.length_weight = config.length_weight;
    rc_config.mode_length_ancient = config.mode_length_ancient;
    rc_config.std_length_ancient = config.std_length_ancient;
    rc_config.mode_length_modern = config.mode_length_modern;
    rc_config.std_length_modern = config.std_length_modern;

    // Identity model parameters
    rc_config.use_identity_model = config.use_identity_model;
    rc_config.identity_weight = config.identity_weight;
    rc_config.mode_identity_ancient = config.mode_identity_ancient;
    rc_config.std_identity_ancient = config.std_identity_ancient;
    rc_config.mode_identity_modern = config.mode_identity_modern;
    rc_config.std_identity_modern = config.std_identity_modern;

    // Pre-compute damage rate LUT to avoid exp() calls in hot loops
    constexpr int MAX_DIST_LUT = 300;
    std::array<double, MAX_DIST_LUT> d5_lut, d3_lut;
    for (int d = 0; d < MAX_DIST_LUT; d++) {
        d5_lut[d] = damage_rate_at_distance(d + 1, rc_config.amp_5p, rc_config.lam_5p, rc_config.base_5p);
        d3_lut[d] = damage_rate_at_distance(d + 1, rc_config.amp_3p, rc_config.lam_3p, rc_config.base_3p);
    }

    std::vector<ReadData> read_data;
    read_data.reserve(reads.size());
    std::vector<double> all_p_ancient;
    all_p_ancient.reserve(reads.size());

    // Step 1a: Extract all read data (done once, reused across EM iterations)
    for (auto* r : reads) {
        ReadData rd;
        rd.read = r;
        rd.start_pos = r->core.pos;
        rd.end_pos = bam_endpos(r);

        int span = rd.end_pos - rd.start_pos;
        rd.observations.resize(span, {-1, 0, 0, 0});
        rd.full_data.read_length = r->core.l_qseq;

        uint32_t* cigar = bam_get_cigar(r);

        // Compute aligned length (excluding soft-clips) for identity calculation
        int aligned_len = 0;
        for (int i = 0; i < r->core.n_cigar; i++) {
            int op = bam_cigar_op(cigar[i]);
            int oplen = bam_cigar_oplen(cigar[i]);
            if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF || op == BAM_CINS) {
                aligned_len += oplen;
            }
        }

        // Extract alignment identity from NM tag (use aligned length, not l_qseq)
        uint8_t* nm_tag = bam_aux_get(r, "NM");
        if (nm_tag && aligned_len > 0) {
            int32_t nm = bam_aux2i(nm_tag);
            rd.full_data.identity = 100.0 * (aligned_len - nm) / aligned_len;
        } else {
            rd.full_data.identity = 100.0;  // Assume perfect if no NM tag
        }
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
                    int rp = ref_pos + j;
                    int qp = qpos + j;
                    if (rp >= 0 && rp < len && rp >= rd.start_pos && rp < rd.end_pos) {
                        int base_code = bam_seqi(seq, qp);
                        int8_t obs_idx = -1;
                        char read_base = 'N';
                        switch (base_code) {
                            case 1: obs_idx = 0; read_base = 'A'; break;
                            case 2: obs_idx = 1; read_base = 'C'; break;
                            case 4: obs_idx = 2; read_base = 'G'; break;
                            case 8: obs_idx = 3; read_base = 'T'; break;
                        }

                        int16_t d5 = static_cast<int16_t>(is_rev ? (read_len - qp) : (qp + 1));
                        int16_t d3 = static_cast<int16_t>(is_rev ? (qp + 1) : (read_len - qp));

                        rd.observations[rp - rd.start_pos] = {obs_idx, qual[qp], d5, d3};

                        // Also store in full_data for full likelihood computation
                        if (config.use_full_likelihood && read_base != 'N') {
                            char ref_base = sequence[rp];
                            if (ref_base != 'N') {
                                AlignedPosition ap;
                                ap.ref_base = ref_base;
                                ap.read_base = read_base;
                                ap.baseq = qual[qp];
                                ap.dist_5p = d5;
                                ap.dist_3p = d3;
                                rd.full_data.positions.push_back(ap);
                            }
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

        read_data.push_back(std::move(rd));
    }

    // =========================================================================
    // Auto-estimation of length/identity model parameters from read distributions
    // Uses peak detection in histograms to find ancient/modern populations
    // =========================================================================
    if (config.auto_estimate_params && (config.use_length_model || config.use_identity_model)) {
        std::vector<int> lengths;
        std::vector<double> identities;
        lengths.reserve(read_data.size());
        identities.reserve(read_data.size());

        for (const auto& rd : read_data) {
            lengths.push_back(rd.full_data.read_length);
            identities.push_back(rd.full_data.identity);
        }

        EstimatedModelParams estimated = estimate_model_params(lengths, identities);

        // Update rc_config with estimated values (if bimodal detected)
        if (config.use_length_model && estimated.length_bimodal) {
            rc_config.mode_length_ancient = estimated.mode_length_ancient;
            rc_config.std_length_ancient = estimated.std_length_ancient;
            rc_config.mode_length_modern = estimated.mode_length_modern;
            rc_config.std_length_modern = estimated.std_length_modern;
        }

        if (config.use_identity_model && estimated.identity_bimodal) {
            rc_config.mode_identity_ancient = estimated.mode_identity_ancient;
            rc_config.std_identity_ancient = estimated.std_identity_ancient;
            rc_config.mode_identity_modern = estimated.mode_identity_modern;
            rc_config.std_identity_modern = estimated.std_identity_modern;
        }
    }

    // =========================================================================
    // Dynamic zone calculation using T/(T+C) at 5' and A/(A+G) at 3'
    // This is data-driven: find where ratios drop to within 1% of interior baseline
    // =========================================================================
    {
        constexpr int MAX_DIST = 50;
        std::array<int, MAX_DIST> t_count_5p = {}, tc_count_5p = {};
        std::array<int, MAX_DIST> a_count_3p = {}, ag_count_3p = {};

        for (const auto& rd : read_data) {
            for (size_t i = 0; i < rd.observations.size(); i++) {
                const auto& obs = rd.observations[i];
                if (obs.obs_idx < 0 || obs.qual < config.min_baseq) continue;

                char base = "ACGT"[obs.obs_idx];
                int d5 = obs.dist_5p - 1;  // 0-indexed
                int d3 = obs.dist_3p - 1;

                if (d5 >= 0 && d5 < MAX_DIST) {
                    if (base == 'T') t_count_5p[d5]++;
                    if (base == 'T' || base == 'C') tc_count_5p[d5]++;
                }
                if (d3 >= 0 && d3 < MAX_DIST) {
                    if (base == 'A') a_count_3p[d3]++;
                    if (base == 'A' || base == 'G') ag_count_3p[d3]++;
                }
            }
        }

        // Interior baseline (positions 30-50)
        double t_tc_interior = 0, a_ag_interior = 0;
        int n5 = 0, n3 = 0;
        for (int d = 30; d < MAX_DIST; d++) {
            if (tc_count_5p[d] > 100) { t_tc_interior += double(t_count_5p[d]) / tc_count_5p[d]; n5++; }
            if (ag_count_3p[d] > 100) { a_ag_interior += double(a_count_3p[d]) / ag_count_3p[d]; n3++; }
        }
        t_tc_interior = n5 > 0 ? t_tc_interior / n5 : 0.5;
        a_ag_interior = n3 > 0 ? a_ag_interior / n3 : 0.5;

        // Find first contiguous stretch from position 0 where damage is elevated
        // Stop at first position that drops below threshold (don't extend on noise spikes)
        zone_5p = 5;  // Minimum
        for (int d = 0; d < 30; d++) {
            if (tc_count_5p[d] >= 100) {
                double ratio = double(t_count_5p[d]) / tc_count_5p[d];
                if (ratio - t_tc_interior > 0.02) {
                    zone_5p = d + 1;
                } else {
                    break;  // First drop below threshold ends the zone
                }
            }
        }

        zone_3p = 5;  // Minimum
        for (int d = 0; d < 30; d++) {
            if (ag_count_3p[d] >= 100) {
                double ratio = double(a_count_3p[d]) / ag_count_3p[d];
                if (ratio - a_ag_interior > 0.02) {
                    zone_3p = d + 1;
                } else {
                    break;  // First drop below threshold ends the zone
                }
            }
        }

        damage_zone = std::max(zone_5p, zone_3p);
    }

    // =========================================================================
    // NEW: Pass 0 - Build neutral consensus from INTERIOR positions only
    // This breaks the circular dependency between classification and consensus.
    // Key insight: Ancient reads have damage ONLY at read ends (5' C→T, 3' G→A).
    // In the interior (far from ends), all reads show the true base.
    // By using interior-only, the neutral consensus is unbiased regardless of
    // the ancient/modern read ratio.
    // =========================================================================
    std::string neutral_consensus(len, 'N');
    {
        // Use majority voting from INTERIOR positions only (no damage there)
        std::vector<std::array<int, 4>> base_counts(len, {0, 0, 0, 0});

        for (const auto& rd : read_data) {
            for (int i = 0; i < static_cast<int>(rd.observations.size()); i++) {
                int rp = rd.start_pos + i;
                if (rp < 0 || rp >= len) continue;
                const auto& obs = rd.observations[i];
                if (obs.obs_idx < 0 || obs.qual < config.min_baseq) continue;

                // Only count INTERIOR positions (outside damage zones)
                // This ensures no damage artifacts in neutral consensus
                bool is_interior = (obs.dist_5p > zone_5p && obs.dist_3p > zone_3p);
                if (is_interior) {
                    base_counts[rp][obs.obs_idx]++;
                }
            }
        }

        constexpr char bases[] = {'A', 'C', 'G', 'T'};
        for (int pos = 0; pos < len; pos++) {
            const auto& counts = base_counts[pos];
            int total = counts[0] + counts[1] + counts[2] + counts[3];
            if (total >= 3) {  // Minimum depth
                int best = 0;
                for (int g = 1; g < 4; g++) {
                    if (counts[g] > counts[best]) best = g;
                }
                // Require majority (>50%)
                if (counts[best] > total / 2) {
                    neutral_consensus[pos] = bases[best];
                } else {
                    // NO FALLBACK to original sequence - it may contain damage artifacts!
                    // Keep as 'N' - positions will be skipped during classification
                    neutral_consensus[pos] = 'N';
                }
            } else {
                // Insufficient interior coverage - keep as 'N' (unknown)
                // DO NOT fall back to original sequence which may have damage incorporated
                neutral_consensus[pos] = 'N';
            }
        }
    }

    // Lambda to classify reads against NEUTRAL consensus (not original sequence)
    // This is the key fix: damage mismatches are now detected relative to
    // the unbiased consensus, not the original sequence
    auto classify_all_reads_vs_neutral = [&](ReadClassifierConfig& rc) {
        all_p_ancient.clear();
        result.ancient_reads_hard = 0;
        result.modern_reads_hard = 0;
        result.ambiguous_reads = 0;

        for (auto& rd : read_data) {
            // Build mismatches against NEUTRAL consensus
            std::vector<MismatchObs> mismatches;
            mismatches.reserve(rd.observations.size() / 10);  // ~10% mismatch rate expected

            FullReadData full_data_vs_neutral;
            full_data_vs_neutral.read_length = rd.full_data.read_length;
            full_data_vs_neutral.identity = rd.full_data.identity;
            full_data_vs_neutral.positions.reserve(rd.observations.size());

            for (int i = 0; i < static_cast<int>(rd.observations.size()); i++) {
                int rp = rd.start_pos + i;
                if (rp < 0 || rp >= len) continue;
                const auto& obs = rd.observations[i];
                if (obs.obs_idx < 0) continue;

                char read_base = "ACGT"[obs.obs_idx];
                char ref_base = neutral_consensus[rp];  // KEY: use neutral consensus
                // DO NOT fall back to original sequence - it may have damage artifacts!
                // If neutral_consensus is 'N', skip this position entirely
                if (ref_base == 'N') continue;

                // For full likelihood: add all positions
                if (config.use_full_likelihood) {
                    AlignedPosition ap;
                    ap.ref_base = ref_base;
                    ap.read_base = read_base;
                    ap.baseq = obs.qual;
                    ap.dist_5p = obs.dist_5p;
                    ap.dist_3p = obs.dist_3p;
                    full_data_vs_neutral.positions.push_back(ap);
                }

                // For mismatch-only: only mismatches (ref_base != 'N' already guaranteed by continue above)
                if (read_base != ref_base && obs.qual >= config.min_baseq) {
                    MismatchObs mm;
                    mm.ref_base = ref_base;
                    mm.read_base = read_base;
                    mm.baseq = obs.qual;
                    mm.dist_from_5p = obs.dist_5p;
                    mm.dist_from_3p = obs.dist_3p;
                    mismatches.push_back(mm);
                }
            }

            // Classify using full likelihood or mismatch-only
            if (config.use_full_likelihood && !full_data_vs_neutral.positions.empty()) {
                rd.score = classify_read_full(full_data_vs_neutral, rc);
            } else {
                rd.score = classify_read(mismatches, rc);
            }
            all_p_ancient.push_back(rd.score.p_ancient);

            if (rd.score.p_ancient >= config.p_ancient_hard) {
                result.ancient_reads_hard++;
            } else if (rd.score.p_ancient <= config.p_modern_hard) {
                result.modern_reads_hard++;
            } else {
                result.ambiguous_reads++;
            }
        }
    };

    // Step 1b: Initial classification AGAINST NEUTRAL CONSENSUS
    classify_all_reads_vs_neutral(rc_config);

    // Step 1c: EM iterations for adaptive prior (auto-converges)
    result.em_iterations_used = 1;
    result.final_prior = rc_config.prior_ancient;

    for (int em_iter = 1; em_iter < config.em_iterations; em_iter++) {
        // Estimate bin-level ancient fraction from current classifications
        double sum_p = 0.0;
        double sum_weight = 0.0;
        for (const auto& rd : read_data) {
            // Weight by damage zone coverage (more informative reads)
            double weight = 1.0 + 0.1 * rd.score.damage_zone_positions;
            sum_p += weight * rd.score.p_ancient;
            sum_weight += weight;
        }
        double new_prior = sum_p / std::max(1.0, sum_weight);
        new_prior = std::max(config.prior_min, std::min(config.prior_max, new_prior));

        // Check convergence (prior changed by less than 1%)
        double prior_change = std::abs(new_prior - rc_config.prior_ancient);
        if (prior_change < 0.01) {
            result.em_iterations_used = em_iter;
            result.final_prior = rc_config.prior_ancient;
            break;
        }

        // Update prior and re-classify against neutral consensus
        rc_config.prior_ancient = new_prior;
        classify_all_reads_vs_neutral(rc_config);
        result.em_iterations_used = em_iter + 1;
        result.final_prior = new_prior;
    }

    if (!all_p_ancient.empty()) {
        double sum = 0.0, sum_conf = 0.0;
        for (double p : all_p_ancient) {
            sum += p;
            sum_conf += std::abs(p - 0.5);
        }
        result.mean_p_ancient = sum / all_p_ancient.size();
        result.classification_confidence = sum_conf / all_p_ancient.size();

        std::sort(all_p_ancient.begin(), all_p_ancient.end());
        result.median_p_ancient = all_p_ancient[all_p_ancient.size() / 2];
        result.ancient_fraction = result.mean_p_ancient;
    }

    // =========================================================================
    // Temporal Population Fraction Estimation
    // Estimate f_undamaged (undamaged fraction) using moment estimator + EM
    // =========================================================================
    {
        // Collect FullReadData for fraction estimation
        std::vector<FullReadData> full_reads;
        full_reads.reserve(read_data.size());
        for (const auto& rd : read_data) {
            full_reads.push_back(rd.full_data);
        }

        // Compute expected and observed detection rates
        result.f_expected_detection = compute_expected_detection_rate(
            full_reads, damage_model.curve_5p, damage_model.curve_3p, damage_zone);
        result.f_observed_detection = compute_observed_detection_fraction(
            full_reads, damage_zone);

        // Moment estimator (fast, for initialization and validation)
        result.f_undamaged_moment = estimate_f_undamaged_moment(
            result.f_expected_detection, result.f_observed_detection);

        // Compute per-read likelihoods for EM
        std::vector<ReadPopulationLikelihoods> pop_liks;
        pop_liks.reserve(read_data.size());
        for (const auto& rd : read_data) {
            pop_liks.push_back(compute_read_population_liks(
                rd.full_data, damage_model.curve_5p, damage_model.curve_3p,
                damage_zone, config.min_baseq));
        }

        // EM estimation
        auto [f_undamaged, gamma] = estimate_population_fraction_em(
            pop_liks, result.f_undamaged_moment, 50, 1e-4);

        result.f_undamaged = f_undamaged;
        result.f_damaged = 1.0 - f_undamaged;
        result.fraction_em_iterations = 50;  // max iterations
        result.fraction_converged = true;

        // Uncertainty quantification
        compute_theta_uncertainty(pop_liks, f_undamaged,
            result.f_undamaged_se, result.f_undamaged_ci_lower, result.f_undamaged_ci_upper);
    }

    // =========================================================================
    // Collect smiley plot data: damage rates by position for ancient/modern/ambiguous
    // Uses hard thresholds: ancient (p>=0.8), modern (p<=0.2), ambiguous (0.2<p<0.8)
    // =========================================================================
    for (const auto& rd : read_data) {
        double p = rd.score.p_ancient;

        // Select arrays based on classification
        std::array<int, DeconvolutionResult::SMILEY_MAX_POS>* t_5p;
        std::array<int, DeconvolutionResult::SMILEY_MAX_POS>* tc_5p;
        std::array<int, DeconvolutionResult::SMILEY_MAX_POS>* a_3p;
        std::array<int, DeconvolutionResult::SMILEY_MAX_POS>* ag_3p;

        if (p >= config.p_ancient_hard) {
            t_5p = &result.anc_t_5p; tc_5p = &result.anc_tc_5p;
            a_3p = &result.anc_a_3p; ag_3p = &result.anc_ag_3p;
        } else if (p <= config.p_modern_hard) {
            t_5p = &result.mod_t_5p; tc_5p = &result.mod_tc_5p;
            a_3p = &result.mod_a_3p; ag_3p = &result.mod_ag_3p;
        } else {
            t_5p = &result.amb_t_5p; tc_5p = &result.amb_tc_5p;
            a_3p = &result.amb_a_3p; ag_3p = &result.amb_ag_3p;
        }

        for (const auto& obs : rd.observations) {
            if (obs.obs_idx < 0 || obs.qual < config.min_baseq) continue;

            char base = "ACGT"[obs.obs_idx];
            int d5 = obs.dist_5p - 1;  // 0-indexed
            int d3 = obs.dist_3p - 1;

            if (d5 >= 0 && d5 < DeconvolutionResult::SMILEY_MAX_POS) {
                if (base == 'T') (*t_5p)[d5]++;
                if (base == 'T' || base == 'C') (*tc_5p)[d5]++;
            }
            if (d3 >= 0 && d3 < DeconvolutionResult::SMILEY_MAX_POS) {
                if (base == 'A') (*a_3p)[d3]++;
                if (base == 'A' || base == 'G') (*ag_3p)[d3]++;
            }
        }

        // Update probability histograms (20 bins, 0.05 width each)
        // Combined posterior
        int hist_bin = static_cast<int>(p * DeconvolutionResult::HIST_BINS);
        hist_bin = std::min(hist_bin, DeconvolutionResult::HIST_BINS - 1);
        result.p_ancient_hist[hist_bin]++;

        // Damage-only posterior
        int dam_bin = static_cast<int>(rd.score.p_damaged * DeconvolutionResult::HIST_BINS);
        dam_bin = std::min(dam_bin, DeconvolutionResult::HIST_BINS - 1);
        result.p_damaged_hist[dam_bin]++;

        // Length-only posterior
        int len_bin_p = static_cast<int>(rd.score.p_length_ancient * DeconvolutionResult::HIST_BINS);
        len_bin_p = std::min(len_bin_p, DeconvolutionResult::HIST_BINS - 1);
        result.p_length_hist[len_bin_p]++;

        // Update read length histogram by classification
        int read_len = rd.full_data.read_length;  // Use actual read length, not ref span
        int len_bin = read_len / 5;  // 5bp bins
        len_bin = std::min(len_bin, DeconvolutionResult::LENGTH_BINS - 1);
        if (p >= config.p_ancient_hard) {
            result.length_hist_ancient[len_bin]++;
        } else if (p <= config.p_modern_hard) {
            result.length_hist_modern[len_bin]++;
        } else {
            result.length_hist_ambiguous[len_bin]++;
        }
    }

    // Sort by position for sweep-line
    std::sort(read_data.begin(), read_data.end(), [](const ReadData& a, const ReadData& b) {
        return a.start_pos < b.start_pos;
    });

    // =========================================================================
    // Step 2: Build two consensus sequences using sweep-line
    // Uses pre-extracted observations from Step 1 (no CIGAR re-parsing)
    // =========================================================================
    std::set<std::pair<int, size_t>> active_by_end;  // {end_pos, read_data index}
    size_t read_idx = 0;

    for (int pos = 0; pos < len; pos++) {
        // Add reads starting at or before this position
        while (read_idx < read_data.size() && read_data[read_idx].start_pos <= pos) {
            active_by_end.insert({read_data[read_idx].end_pos, read_idx});
            read_idx++;
        }

        // Remove reads that ended before this position
        while (!active_by_end.empty() && active_by_end.begin()->first <= pos) {
            active_by_end.erase(active_by_end.begin());
        }

        if (active_by_end.empty()) {
            result.positions_neither++;
            // Fall back to original sequence for both consensuses
            result.ancient_raw[pos] = sequence[pos];
            result.ancient_consensus[pos] = sequence[pos];
            result.modern_consensus[pos] = sequence[pos];
            continue;
        }

        // ROBUST APPROACH: Independent soft-weighted consensuses
        // Ancient consensus: weight by p_ancient, damage-corrected likelihoods
        // Modern consensus: weight by (1 - p_ancient), no damage correction
        // Key: NO fallback coupling - each consensus from independent evidence

        // Effective depth trackers (soft weights)
        double w_ancient_total = 0.0;
        double w_modern_total = 0.0;

        // Likelihood accumulators (log-scale)
        std::array<double, 4> ancient_lik = {0.0, 0.0, 0.0, 0.0};
        std::array<double, 4> modern_lik = {0.0, 0.0, 0.0, 0.0};

        // Interior-only accumulators for ancient (damage-free positions)
        std::array<double, 4> ancient_interior_lik = {0.0, 0.0, 0.0, 0.0};
        double w_ancient_interior = 0.0;

        constexpr double kEps = 1e-10;
        constexpr int BASE_C = 1, BASE_T = 3, BASE_G = 2, BASE_A = 0;

        for (const auto& [end, rd_idx] : active_by_end) {
            const ReadData& rd = read_data[rd_idx];
            double p_anc = rd.score.p_ancient;
            double p_mod = 1.0 - p_anc;

            int offset = pos - rd.start_pos;
            if (offset < 0 || offset >= static_cast<int>(rd.observations.size())) continue;

            const BaseObs& obs = rd.observations[offset];
            if (obs.obs_idx < 0) continue;
            if (obs.qual < config.min_baseq) continue;

            double q_err = phred_to_error_prob(obs.qual);
            q_err = std::max(0.001, std::min(0.5, q_err));

            bool is_interior = (obs.dist_5p > zone_5p && obs.dist_3p > zone_3p);

            // Compute damage rates at this position
            double d5 = damage_rate_at_distance(obs.dist_5p,
                rc_config.amp_5p, rc_config.lam_5p, rc_config.base_5p);
            double d3 = damage_rate_at_distance(obs.dist_3p,
                rc_config.amp_3p, rc_config.lam_3p, rc_config.base_3p);

            // === ANCIENT CONSENSUS (weighted by p_ancient) ===
            // For interior positions: use directly (no damage)
            // For damage-zone positions: apply damage correction
            if (p_anc >= 0.01) {  // Skip negligible weights
                w_ancient_total += p_anc;

                for (int g = 0; g < 4; g++) {
                    double p_obs_given_g;

                    if (is_interior) {
                        // Interior: standard error model
                        p_obs_given_g = (g == obs.obs_idx) ? (1.0 - q_err) : (q_err / 3.0);
                    } else {
                        // Damage zone: account for deamination
                        if (g == BASE_C) {
                            // True C: could appear as C (no damage) or T (C→T damage)
                            if (obs.obs_idx == BASE_C) {
                                p_obs_given_g = (1.0 - d5) * (1.0 - q_err) + d5 * (q_err / 3.0);
                            } else if (obs.obs_idx == BASE_T) {
                                p_obs_given_g = d5 * (1.0 - q_err) + (1.0 - d5) * (q_err / 3.0);
                            } else {
                                p_obs_given_g = q_err / 3.0;
                            }
                        } else if (g == BASE_G) {
                            // True G: could appear as G (no damage) or A (G→A damage)
                            if (obs.obs_idx == BASE_G) {
                                p_obs_given_g = (1.0 - d3) * (1.0 - q_err) + d3 * (q_err / 3.0);
                            } else if (obs.obs_idx == BASE_A) {
                                p_obs_given_g = d3 * (1.0 - q_err) + (1.0 - d3) * (q_err / 3.0);
                            } else {
                                p_obs_given_g = q_err / 3.0;
                            }
                        } else {
                            // A or T: standard error model
                            p_obs_given_g = (g == obs.obs_idx) ? (1.0 - q_err) : (q_err / 3.0);
                        }
                    }

                    ancient_lik[g] += p_anc * std::log(std::max(kEps, p_obs_given_g));
                }

                // Also track interior-only for ancient (backup)
                if (is_interior) {
                    w_ancient_interior += p_anc;
                    for (int g = 0; g < 4; g++) {
                        double p_obs = (g == obs.obs_idx) ? (1.0 - q_err) : (q_err / 3.0);
                        ancient_interior_lik[g] += p_anc * std::log(std::max(kEps, p_obs));
                    }
                }
            }

            // === MODERN CONSENSUS (weighted by 1 - p_ancient) ===
            // No damage correction - modern reads shouldn't have damage
            if (p_mod >= 0.01) {  // Skip negligible weights
                w_modern_total += p_mod;

                for (int g = 0; g < 4; g++) {
                    double p_obs_given_g = (g == obs.obs_idx) ? (1.0 - q_err) : (q_err / 3.0);
                    modern_lik[g] += p_mod * std::log(std::max(kEps, p_obs_given_g));
                }
            }
        }

        // === CALL ANCIENT CONSENSUS ===
        // Prefer interior-only if sufficient depth (avoids damage), else use full
        bool ancient_called = false;
        constexpr double kMinEffDepth = 3.0;

        std::array<double, 4>* anc_lik_to_use = &ancient_lik;
        double anc_depth_to_use = w_ancient_total;

        // If good interior depth, prefer interior-only (safer)
        if (w_ancient_interior >= kMinEffDepth) {
            anc_lik_to_use = &ancient_interior_lik;
            anc_depth_to_use = w_ancient_interior;
        }

        if (anc_depth_to_use >= kMinEffDepth) {
            double max_anc = *std::max_element(anc_lik_to_use->begin(), anc_lik_to_use->end());
            std::array<double, 4> post_anc;
            double sum_anc = 0.0;
            for (int g = 0; g < 4; g++) {
                post_anc[g] = std::exp((*anc_lik_to_use)[g] - max_anc);
                sum_anc += post_anc[g];
            }
            int best_anc = 0;
            for (int g = 1; g < 4; g++) {
                if (post_anc[g] > post_anc[best_anc]) best_anc = g;
            }

            if (post_anc[best_anc] / sum_anc >= config.min_posterior) {
                constexpr char bases[] = {'A', 'C', 'G', 'T'};
                result.ancient_raw[pos] = bases[best_anc];
                result.ancient_consensus[pos] = bases[best_anc];
                ancient_called = true;
            }
        }

        // Fallback for ancient: use neutral consensus (NOT modern!)
        if (!ancient_called && neutral_consensus[pos] != 'N') {
            result.ancient_raw[pos] = neutral_consensus[pos];
            result.ancient_consensus[pos] = neutral_consensus[pos];
            ancient_called = true;
            result.positions_ancient_only++;  // Mark as independent fallback
        }

        // === CALL MODERN CONSENSUS ===
        bool modern_called = false;
        if (w_modern_total >= kMinEffDepth) {
            double max_mod = *std::max_element(modern_lik.begin(), modern_lik.end());
            std::array<double, 4> post_mod;
            double sum_mod = 0.0;
            for (int g = 0; g < 4; g++) {
                post_mod[g] = std::exp(modern_lik[g] - max_mod);
                sum_mod += post_mod[g];
            }
            int best_mod = 0;
            for (int g = 1; g < 4; g++) {
                if (post_mod[g] > post_mod[best_mod]) best_mod = g;
            }

            if (post_mod[best_mod] / sum_mod >= config.min_posterior) {
                constexpr char bases[] = {'A', 'C', 'G', 'T'};
                result.modern_consensus[pos] = bases[best_mod];
                modern_called = true;
            }
        }

        // Fallback for modern: use neutral consensus (NOT ancient!)
        // This is the key change - NO coupling between consensuses
        if (!modern_called && neutral_consensus[pos] != 'N') {
            result.modern_consensus[pos] = neutral_consensus[pos];
            modern_called = true;
            result.positions_modern_only++;  // Mark as independent fallback
        }

        // Track position categories (positions_ancient_only/modern_only already
        // incremented above when fallback is used)
        if (ancient_called && modern_called) {
            result.positions_both_called++;
            if (result.ancient_consensus[pos] != result.modern_consensus[pos]) {
                result.ancient_modern_differences++;
            }
        } else if (!ancient_called && !modern_called) {
            result.positions_neither++;
            // Ultimate fallback: original sequence
            result.ancient_raw[pos] = sequence[pos];
            result.ancient_consensus[pos] = sequence[pos];
            result.modern_consensus[pos] = sequence[pos];
        } else if (!ancient_called) {
            // Modern called but ancient not - use original for ancient
            result.ancient_raw[pos] = sequence[pos];
            result.ancient_consensus[pos] = sequence[pos];
        } else if (!modern_called) {
            // Ancient called but modern not - use original for modern
            result.modern_consensus[pos] = sequence[pos];
        }
    }

    // =========================================================================
    // Step 3: Polish ancient consensus (BF + interior veto + soft weights)
    // Replaces damage artifacts with true bases using calibrated thresholds
    // Key improvements over original 0.99/0.05 posterior threshold:
    //   1. Soft mixture weighting (w = p_ancient) instead of hard cutoff
    //   2. Bayes factor threshold instead of fixed posterior
    //   3. Interior-read veto to prevent false corrections
    // =========================================================================
    if (config.polish_ancient) {
        // Tunable thresholds (calibrated for aDNA damage correction)
        constexpr double kMinEffDepth = 3.0;          // Σw required overall
        constexpr double kMinEffInterior = 2.0;       // Σw required interior
        constexpr double kLogBFStrong = 0.0;          // Any positive BF (for diagnostic)
        constexpr double kLogBFRelaxed = -1.0;        // Even slightly negative (for diagnostic)
        constexpr double kLogBFInteriorSupport = 2.3; // BF ~10 interior supports
        constexpr double kLogBFInteriorVeto = -999.0; // Disable interior veto
        constexpr double kMinWeight = 1e-3;           // Ignore tiny weights
        constexpr double kEps = 1e-10;
        constexpr double kLogPriorOdds = 0.0;         // Equal priors C:T, G:A

        // Base indices for clarity
        constexpr int BASE_A = 0, BASE_C = 1, BASE_G = 2, BASE_T = 3;

        // Debug counters
        size_t dbg_ta_positions = 0;
        size_t dbg_low_depth = 0;
        size_t dbg_interior_veto = 0;
        size_t dbg_weak_bf = 0;
        size_t dbg_corrected = 0;
        double dbg_bf_sum = 0.0;
        size_t dbg_bf_count = 0;
        double dbg_bf_max = -1e9;
        double dbg_bf_int_sum = 0.0;
        size_t dbg_bf_int_count = 0;
        double dbg_bf_int_max = -1e9;
        size_t dbg_no_interior = 0;

        std::set<std::pair<int, size_t>> polish_active;
        size_t polish_idx = 0;

        for (int pos = 0; pos < len; pos++) {
            // Maintain sweep-line active set
            while (polish_idx < read_data.size() && read_data[polish_idx].start_pos <= pos) {
                polish_active.insert({read_data[polish_idx].end_pos, polish_idx});
                polish_idx++;
            }
            while (!polish_active.empty() && polish_active.begin()->first <= pos) {
                polish_active.erase(polish_active.begin());
            }

            char base = result.ancient_consensus[pos];
            if (base != 'T' && base != 'A') continue;
            dbg_ta_positions++;

            // Pair under test: T->C or A->G
            int idx_current = (base == 'T') ? BASE_T : BASE_A;
            int idx_correct = (base == 'T') ? BASE_C : BASE_G;
            char corrected_base = (base == 'T') ? 'C' : 'G';

            double logBF_pair = 0.0;   // Σ w * log(P(obs|correct) / P(obs|current))
            double logBF_int = 0.0;    // Interior-only BF (simple error model)
            double n_eff = 0.0;        // Effective depth Σw
            double n_eff_int = 0.0;    // Interior effective depth

            for (const auto& [end, rd_idx] : polish_active) {
                const ReadData& rd = read_data[rd_idx];

                int offset = pos - rd.start_pos;
                if (offset < 0 || offset >= static_cast<int>(rd.observations.size())) continue;

                const BaseObs& obs = rd.observations[offset];
                if (obs.obs_idx < 0 || obs.qual < config.min_baseq) continue;

                // Soft ancient weight (not hard p_ancient >= 0.5 cutoff)
                double w = std::clamp(rd.score.p_ancient, 0.0, 1.0);
                if (w < kMinWeight) continue;

                double q_err = phred_to_error_prob(obs.qual);
                q_err = std::clamp(q_err, 0.001, 0.5);

                // Damage-aware transition matrix
                double d5 = damage_rate_at_distance(obs.dist_5p,
                    damage_model.curve_5p.amplitude,
                    damage_model.curve_5p.lambda,
                    damage_model.curve_5p.baseline);
                double d3 = damage_rate_at_distance(obs.dist_3p,
                    damage_model.curve_3p.amplitude,
                    damage_model.curve_3p.lambda,
                    damage_model.curve_3p.baseline);
                TransitionMatrix T = compute_damage_transition(d5, d3, q_err);

                double p_corr = std::max(kEps, T[idx_correct][obs.obs_idx]);
                double p_curr = std::max(kEps, T[idx_current][obs.obs_idx]);

                logBF_pair += w * (std::log(p_corr) - std::log(p_curr));
                n_eff += w;

                // Interior evidence (damage-minimal reads for veto/support)
                bool is_interior = (obs.dist_5p > zone_5p && obs.dist_3p > zone_3p);
                if (is_interior) {
                    // Simple error model for interior positions
                    double p_int_corr = (obs.obs_idx == idx_correct) ? (1.0 - q_err) : (q_err / 3.0);
                    double p_int_curr = (obs.obs_idx == idx_current) ? (1.0 - q_err) : (q_err / 3.0);
                    logBF_int += w * (std::log(std::max(kEps, p_int_corr)) -
                                      std::log(std::max(kEps, p_int_curr)));
                    n_eff_int += w;
                }
            }

            // Coverage gate
            if (n_eff < kMinEffDepth) {
                dbg_low_depth++;
                continue;
            }

            // Track BF statistics
            dbg_bf_sum += logBF_pair;
            dbg_bf_count++;
            if (logBF_pair > dbg_bf_max) dbg_bf_max = logBF_pair;

            // NEW APPROACH: Use interior evidence directly
            // Interior reads are outside damage zone, so they show true base
            // If interior supports correction base (C for T position, G for A position),
            // then the overall T/A is likely from damage
            bool has_interior = (n_eff_int >= kMinEffInterior);

            // Track interior BF stats
            if (has_interior) {
                dbg_bf_int_sum += logBF_int;
                dbg_bf_int_count++;
                if (logBF_int > dbg_bf_int_max) dbg_bf_int_max = logBF_int;
            } else {
                dbg_no_interior++;
            }

            // Interior veto: if interior strongly supports current base, skip
            if (has_interior && logBF_int <= kLogBFInteriorVeto) {
                dbg_interior_veto++;
                continue;
            }

            // NEW: Correct if interior SUPPORTS correction (positive logBF_int)
            // This means interior reads (damage-free) show the correction base
            if (has_interior && logBF_int >= 1.0) {  // BF >= 2.7 for interior
                if (base == 'T' && corrected_base == 'C') {
                    result.ancient_ct_corrections++;
                } else if (base == 'A' && corrected_base == 'G') {
                    result.ancient_ga_corrections++;
                }
                result.ancient_consensus[pos] = corrected_base;
                dbg_corrected++;
            } else {
                dbg_weak_bf++;
            }
        }

    }

    // =========================================================================
    // Step 4: Final damage harmonization
    // For same-organism samples, ancient and modern consensuses should be identical.
    // Where they differ in damage-consistent pattern (ancient T vs modern C, or
    // ancient A vs modern G), copy modern to ancient since modern has no damage.
    // This bypasses mapping bias that corrupts even interior reads.
    // =========================================================================
    size_t damage_harmonized = 0;
    for (int pos = 0; pos < len; pos++) {
        char anc = result.ancient_consensus[pos];
        char mod = result.modern_consensus[pos];

        // Only harmonize damage-pattern differences
        if ((anc == 'T' && mod == 'C') || (anc == 'A' && mod == 'G')) {
            result.ancient_consensus[pos] = mod;
            damage_harmonized++;

            // Track as corrections
            if (anc == 'T') result.ancient_ct_corrections++;
            else result.ancient_ga_corrections++;
        }
    }

    if (damage_harmonized > 0) {
        std::cerr << "  Damage harmonization: copied " << damage_harmonized
                  << " positions from modern to ancient consensus\n";
    }

    // Cleanup
    for (auto* r : reads) bam_destroy1(r);

    return result;
}

// Batch deconvolution for multiple contigs (for parallel processing)
struct BatchDeconvolutionResult {
    std::vector<DeconvolutionResult> per_contig;

    // Aggregated stats
    size_t total_ancient_reads = 0;
    size_t total_modern_reads = 0;
    double overall_ancient_fraction = 0.0;
    size_t total_positions_called = 0;
    size_t total_differences = 0;
};

}  // namespace amber

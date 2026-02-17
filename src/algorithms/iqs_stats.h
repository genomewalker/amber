// Intrinsic Quality Score (IQS) Statistics for Bins
// Uses influence-score based incremental estimation
//
// Key insight from GTDB validation:
// - var_80% correlates POSITIVELY with completeness (+0.20)
// - var_80% correlates NEGATIVELY with contamination (-0.09)
// - This OPPOSITE SIGN property enables both quality estimation AND contamination detection
//
// IQS = 0.60 * var_80% + 0.40 * nmi_weighted
//
// Contamination detection: contigs with high influence that INCREASE IQS when removed

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <map>

namespace mfdfa_binner {

// Jensen-Shannon divergence for k-mer distributions
inline double jsd(const std::vector<double>& p, const std::vector<double>& q) {
    double h_m = 0, h_p = 0, h_q = 0;
    for (size_t i = 0; i < p.size(); i++) {
        double m = (p[i] + q[i]) / 2.0;
        if (m > 0) h_m -= m * std::log(m);
        if (p[i] > 0) h_p -= p[i] * std::log(p[i]);
        if (q[i] > 0) h_q -= q[i] * std::log(q[i]);
    }
    return std::max(0.0, h_m - (h_p + h_q) / 2.0);
}

inline double jsd_distance(const std::vector<double>& p, const std::vector<double>& q) {
    return std::sqrt(jsd(p, q));
}

// Normalize k-mer counts to probability distribution
inline std::vector<double> normalize_kmer(const std::vector<double>& counts) {
    double sum = std::accumulate(counts.begin(), counts.end(), 0.0);
    std::vector<double> probs(counts.size());
    for (size_t i = 0; i < counts.size(); i++) {
        probs[i] = sum > 0 ? counts[i] / sum : 1.0 / counts.size();
    }
    return probs;
}

// IQS weights (from GTDB optimization: 60% var_80% + 40% nmi_weighted)
constexpr double IQS_WEIGHT_VAR80 = 0.60;
constexpr double IQS_WEIGHT_NMI = 0.40;

// Tolerance band parameters for merge decisions
constexpr double IQS_K_CONSERVATIVE = 1.5;  // 6.7% one-sided risk
constexpr double IQS_K_MODERATE = 1.0;      // 16% one-sided risk

// Quality budget parameters
constexpr double IQS_BUDGET_BETA = 0.15;           // Allow 15% IQS loss during recruitment
constexpr double IQS_CATASTROPHIC_LAMBDA = 0.25;  // Max 25% of budget per contig

// Contamination detection thresholds
constexpr double CONTAM_IQS_GAIN_THRESHOLD = 0.02;  // 2% IQS improvement when removed
constexpr double CONTAM_INFLUENCE_ZSCORE = 2.0;     // Z-score threshold for influence
constexpr double CONTAM_MAX_REMOVAL_FRAC = 0.15;    // Max 15% of bin length removed

struct ContigIQSInfo {
    int index = -1;                         // Index in parent container
    std::string name;                       // Contig name
    std::vector<double> kmer_probs;         // Normalized 6-mer distribution (4096 elements)
    double length = 0;
    double influence = 0;                   // I_i = w_i * d_i
    double iqs_gain_if_removed = 0;         // ΔIQS = IQS_without_i - IQS_with_i
    double contamination_score = 0;         // Combined contamination probability
    bool flagged_contaminant = false;       // Final contamination call
};

struct BinIQSStats {
    // Centroid (length-weighted mean k-mer distribution)
    std::vector<double> centroid;
    double total_length = 0;
    int n_contigs = 0;

    // Per-contig information for contamination detection
    std::vector<ContigIQSInfo*> contig_info;

    // Influence-score statistics for var_80% estimation
    double sum_influence_sq = 0;  // Σ I_i²

    // NMI-related statistics (simplified: track centroid compactness)
    double mean_jsd_to_centroid = 0;

    // Cached IQS components
    double var_80_est = 0;
    double nmi_weighted = 0;
    double iqs = 0;
    double sigma_iqs = 0;  // Uncertainty in IQS estimate

    // Quality budget
    double budget_initial = 0;
    double budget_remaining = 0;

    static constexpr int K = 6;
    static constexpr int N_CELLS = 1 << (2 * K);  // 4096

    BinIQSStats() : centroid(N_CELLS, 0) {}

    // Compute influence score for a contig
    double compute_influence(const std::vector<double>& contig_probs,
                            double length) const {
        if (n_contigs == 0 || total_length == 0) return 0;
        double w = length / total_length;
        double d = jsd_distance(contig_probs, centroid);
        return w * d;
    }

    // Build stats from a list of contigs (full computation)
    void build_from_contigs(std::vector<ContigIQSInfo*>& contigs) {
        contig_info = contigs;
        n_contigs = contigs.size();

        if (n_contigs == 0) {
            centroid.assign(N_CELLS, 0);
            total_length = 0;
            sum_influence_sq = 0;
            recompute_iqs();
            return;
        }

        // Compute total length and centroid
        total_length = 0;
        for (auto* c : contigs) {
            total_length += c->length;
        }

        centroid.assign(N_CELLS, 0);
        for (auto* c : contigs) {
            double w = c->length / total_length;
            for (int d = 0; d < N_CELLS; d++) {
                centroid[d] += w * c->kmer_probs[d];
            }
        }

        // Compute influences and sum
        sum_influence_sq = 0;
        double sum_jsd = 0;
        for (auto* c : contigs) {
            c->influence = compute_influence(c->kmer_probs, c->length);
            sum_influence_sq += c->influence * c->influence;
            sum_jsd += jsd_distance(c->kmer_probs, centroid);
        }
        mean_jsd_to_centroid = sum_jsd / n_contigs;

        recompute_iqs();
        initialize_budget();
    }

    // Add a contig to the bin (incremental update)
    void add_contig(ContigIQSInfo* contig) {
        if (contig->kmer_probs.size() != N_CELLS) return;

        double L_old = total_length;
        double L_new = L_old + contig->length;

        if (L_old > 0) {
            double alpha = contig->length / L_new;
            double beta = L_old / L_new;

            for (int d = 0; d < N_CELLS; d++) {
                centroid[d] = beta * centroid[d] + alpha * contig->kmer_probs[d];
            }
        } else {
            centroid = contig->kmer_probs;
        }

        total_length = L_new;
        n_contigs++;
        contig_info.push_back(contig);

        // Recompute influence for this contig with new centroid
        contig->influence = compute_influence(contig->kmer_probs, contig->length);
        sum_influence_sq += contig->influence * contig->influence;

        // Update mean JSD
        double new_jsd = jsd_distance(contig->kmer_probs, centroid);
        mean_jsd_to_centroid = ((mean_jsd_to_centroid * (n_contigs - 1)) + new_jsd) / n_contigs;

        recompute_iqs();
    }

    // Remove a contig from the bin (incremental update)
    void remove_contig(ContigIQSInfo* contig) {
        if (contig->kmer_probs.size() != N_CELLS || n_contigs <= 1) return;

        // Remove from influence sum
        sum_influence_sq -= contig->influence * contig->influence;
        sum_influence_sq = std::max(0.0, sum_influence_sq);

        double L_old = total_length;
        double L_new = L_old - contig->length;

        if (L_new > 0) {
            double alpha = contig->length / L_old;
            double beta = L_new / L_old;

            for (int d = 0; d < N_CELLS; d++) {
                centroid[d] = (centroid[d] - alpha * contig->kmer_probs[d]) / beta;
            }
        }

        total_length = L_new;
        n_contigs--;

        // Remove from contig_info
        contig_info.erase(
            std::remove(contig_info.begin(), contig_info.end(), contig),
            contig_info.end()
        );

        recompute_iqs();
    }

    // Compute IQS if we remove a specific contig (without modifying state)
    double compute_iqs_without(const ContigIQSInfo* exclude) const {
        if (n_contigs <= 1) return 0;

        // Compute new centroid without this contig
        double L_new = total_length - exclude->length;
        if (L_new <= 0) return iqs;

        std::vector<double> new_centroid(N_CELLS);
        double alpha = exclude->length / total_length;
        double beta = L_new / total_length;

        for (int d = 0; d < N_CELLS; d++) {
            new_centroid[d] = (centroid[d] - alpha * exclude->kmer_probs[d]) / beta;
        }

        // Recompute influence sum without this contig
        double new_sum_I2 = sum_influence_sq - exclude->influence * exclude->influence;
        new_sum_I2 = std::max(0.0, new_sum_I2);

        int new_n = n_contigs - 1;

        // Compute var_80 with new stats
        double p = 0.8, q = 0.2;
        double alpha_factor = p / (q * new_n);
        double new_var_80 = std::sqrt(alpha_factor * new_sum_I2);

        // Approximate NMI change (heuristic)
        double new_nmi = nmi_weighted;  // First approximation: unchanged

        return IQS_WEIGHT_VAR80 * new_var_80 + IQS_WEIGHT_NMI * new_nmi;
    }

    // Predict IQS change if we add a contig (without modifying state)
    double predict_iqs_after_add(const ContigIQSInfo* contig) const {
        if (contig->kmer_probs.size() != N_CELLS) return iqs;

        double L_new = total_length + contig->length;
        std::vector<double> new_centroid(N_CELLS);

        if (total_length > 0) {
            double alpha = contig->length / L_new;
            double beta = total_length / L_new;
            for (int d = 0; d < N_CELLS; d++) {
                new_centroid[d] = beta * centroid[d] + alpha * contig->kmer_probs[d];
            }
        } else {
            new_centroid = contig->kmer_probs;
        }

        double w_new = contig->length / L_new;
        double d_new = jsd_distance(contig->kmer_probs, new_centroid);
        double I_new = w_new * d_new;

        double new_sum_I2 = sum_influence_sq + I_new * I_new;
        int new_n = n_contigs + 1;

        double p = 0.8, q = 0.2;
        double alpha_factor = p / (q * new_n);
        double new_var_80 = std::sqrt(alpha_factor * new_sum_I2);

        return IQS_WEIGHT_VAR80 * new_var_80 + IQS_WEIGHT_NMI * nmi_weighted;
    }

    // =========================================================================
    // CONTAMINATION DETECTION
    // =========================================================================

    // Compute contamination scores for all contigs in the bin
    void compute_contamination_scores() {
        if (n_contigs < 5) return;  // Need enough contigs

        // Compute IQS gain for each contig if removed
        for (auto* c : contig_info) {
            double iqs_without = compute_iqs_without(c);
            c->iqs_gain_if_removed = iqs_without - iqs;
        }

        // Compute influence statistics for z-score
        double mean_I = 0, var_I = 0;
        for (auto* c : contig_info) {
            mean_I += c->influence;
        }
        mean_I /= n_contigs;

        for (auto* c : contig_info) {
            var_I += (c->influence - mean_I) * (c->influence - mean_I);
        }
        double std_I = std::sqrt(var_I / (n_contigs - 1));

        // Compute combined contamination score
        // C_i = normalized(IQS_gain) * normalized(influence)
        for (auto* c : contig_info) {
            double z_influence = (std_I > 1e-10) ? (c->influence - mean_I) / std_I : 0;

            // Contamination score: high IQS gain + high influence = likely contaminant
            // Scale IQS gain to [0, 1] range
            double iqs_gain_normalized = std::max(0.0, c->iqs_gain_if_removed) /
                                        std::max(0.01, iqs);

            // Combined score
            c->contamination_score = iqs_gain_normalized * (1.0 + std::max(0.0, z_influence));
        }
    }

    // Flag contaminating contigs based on IQS gain and influence
    // Returns number of contigs flagged
    int flag_contaminants(double iqs_gain_threshold = CONTAM_IQS_GAIN_THRESHOLD,
                          double influence_zscore = CONTAM_INFLUENCE_ZSCORE,
                          double max_removal_frac = CONTAM_MAX_REMOVAL_FRAC) {
        if (n_contigs < 5) return 0;

        compute_contamination_scores();

        // Sort by contamination score (descending)
        std::vector<ContigIQSInfo*> sorted = contig_info;
        std::sort(sorted.begin(), sorted.end(),
                  [](const ContigIQSInfo* a, const ContigIQSInfo* b) {
                      return a->contamination_score > b->contamination_score;
                  });

        // Compute influence statistics
        double mean_I = 0;
        for (auto* c : contig_info) mean_I += c->influence;
        mean_I /= n_contigs;

        double var_I = 0;
        for (auto* c : contig_info) {
            var_I += (c->influence - mean_I) * (c->influence - mean_I);
        }
        double std_I = std::sqrt(var_I / (n_contigs - 1));

        // Flag contaminants
        int flagged = 0;
        double removed_length = 0;

        for (auto* c : sorted) {
            // Check IQS gain threshold
            bool high_iqs_gain = c->iqs_gain_if_removed > iqs_gain_threshold * iqs;

            // Check influence z-score
            double z = (std_I > 1e-10) ? (c->influence - mean_I) / std_I : 0;
            bool high_influence = z > influence_zscore;

            // Flag if both conditions met AND we haven't exceeded removal limit
            if (high_iqs_gain && high_influence) {
                if ((removed_length + c->length) / total_length <= max_removal_frac) {
                    c->flagged_contaminant = true;
                    removed_length += c->length;
                    flagged++;
                }
            }
        }

        return flagged;
    }

    // Iteratively remove contaminants until IQS stabilizes
    // Returns vector of removed contig indices
    std::vector<int> iterative_contamination_removal(
            double iqs_gain_threshold = CONTAM_IQS_GAIN_THRESHOLD,
            double max_removal_frac = CONTAM_MAX_REMOVAL_FRAC,
            int max_iterations = 10) {

        std::vector<int> removed_indices;
        double initial_length = total_length;
        double removed_length = 0;

        for (int iter = 0; iter < max_iterations && n_contigs > 5; iter++) {
            compute_contamination_scores();

            // Find contig with highest IQS gain if removed
            ContigIQSInfo* worst = nullptr;
            double max_gain = 0;

            for (auto* c : contig_info) {
                if (c->iqs_gain_if_removed > max_gain) {
                    max_gain = c->iqs_gain_if_removed;
                    worst = c;
                }
            }

            // Stop if no contig improves IQS significantly
            if (max_gain < iqs_gain_threshold * iqs) break;

            // Stop if we'd exceed removal limit
            if ((removed_length + worst->length) / initial_length > max_removal_frac) break;

            // Remove the contaminant
            removed_indices.push_back(worst->index);
            worst->flagged_contaminant = true;
            removed_length += worst->length;
            remove_contig(worst);
        }

        return removed_indices;
    }

    // =========================================================================
    // MERGE DECISIONS
    // =========================================================================

    // Check if adding contig would pass tolerance band
    bool would_pass_tolerance(const ContigIQSInfo* contig,
                              double k = IQS_K_CONSERVATIVE) const {
        double iqs_after = predict_iqs_after_add(contig);
        double threshold = iqs - k * sigma_iqs;
        return iqs_after >= threshold;
    }

    // Check if adding contig would stay within quality budget
    bool would_pass_budget(const ContigIQSInfo* contig) const {
        double iqs_after = predict_iqs_after_add(contig);
        double loss = std::max(0.0, iqs - iqs_after);

        // Catastrophic contig check
        if (loss > IQS_CATASTROPHIC_LAMBDA * budget_initial) {
            return false;
        }

        return loss <= budget_remaining;
    }

    // Update budget after adding contig
    void update_budget_after_add(double iqs_before) {
        double loss = std::max(0.0, iqs_before - iqs);
        budget_remaining -= loss;
        budget_remaining = std::max(0.0, budget_remaining);

        // Regenerate budget slightly if IQS improved
        double gain = std::max(0.0, iqs - iqs_before);
        budget_remaining += 0.25 * gain;
    }

    // Initialize quality budget for a newly formed bin
    void initialize_budget() {
        budget_initial = IQS_BUDGET_BETA * iqs;
        budget_remaining = budget_initial;
    }

private:
    // Compute ACTUAL downsampling variance (not approximation)
    // This is the key metric from GTDB validation that shows opposite signs
    double compute_actual_var_80(int n_reps = 20, unsigned seed = 42) const {
        if (contig_info.size() < 5) return 0;

        std::mt19937 rng(seed);
        std::vector<double> downsample_jsds;
        downsample_jsds.reserve(n_reps);

        int n = contig_info.size();
        int keep = std::max(2, (int)(n * 0.8));

        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);

        for (int rep = 0; rep < n_reps; rep++) {
            std::shuffle(indices.begin(), indices.end(), rng);

            // Compute 80% subsample centroid (length-weighted)
            std::vector<double> sub_centroid(N_CELLS, 0);
            double sub_len = 0;

            for (int j = 0; j < keep; j++) {
                int idx = indices[j];
                const auto& probs = contig_info[idx]->kmer_probs;
                double len = contig_info[idx]->length;
                for (int c = 0; c < N_CELLS; c++) {
                    sub_centroid[c] += probs[c] * len;
                }
                sub_len += len;
            }

            if (sub_len > 0) {
                for (int c = 0; c < N_CELLS; c++) {
                    sub_centroid[c] /= sub_len;
                }
            }

            // JSD distance from full centroid
            double d = jsd_distance(sub_centroid, centroid);
            downsample_jsds.push_back(d);
        }

        // Compute variance of distances
        double mean = 0;
        for (double v : downsample_jsds) mean += v;
        mean /= downsample_jsds.size();

        double var = 0;
        for (double v : downsample_jsds) {
            var += (v - mean) * (v - mean);
        }
        var /= downsample_jsds.size();

        return std::sqrt(var);  // Return std dev
    }

    void recompute_iqs() {
        // Use ACTUAL downsampling variance (validated on GTDB)
        if (n_contigs >= 5 && !contig_info.empty()) {
            var_80_est = compute_actual_var_80();
        } else if (n_contigs > 0) {
            // Fallback for small bins: use approximation
            double p = 0.8, q = 0.2;
            double alpha = p / (q * n_contigs);
            var_80_est = std::sqrt(alpha * sum_influence_sq);
        } else {
            var_80_est = 0;
        }

        // NMI component: use mean JSD to centroid as compactness measure
        if (n_contigs > 1 && mean_jsd_to_centroid > 0) {
            nmi_weighted = std::max(0.0, 1.0 - mean_jsd_to_centroid);
        } else if (n_contigs > 1) {
            double mean_influence = std::sqrt(sum_influence_sq / n_contigs);
            nmi_weighted = 1.0 - mean_influence;
            nmi_weighted = std::max(0.0, std::min(1.0, nmi_weighted));
        } else {
            nmi_weighted = 1.0;
        }

        // Combine into IQS
        iqs = IQS_WEIGHT_VAR80 * var_80_est + IQS_WEIGHT_NMI * nmi_weighted;

        // Estimate uncertainty (σ_IQS) - based on sample variance
        double var_80_uncertainty = var_80_est / std::sqrt(std::max(1, n_contigs));
        sigma_iqs = IQS_WEIGHT_VAR80 * var_80_uncertainty;
    }
};

// Compute IQS for a merge candidate (combines two bins)
inline double compute_merge_iqs(const BinIQSStats& a, const BinIQSStats& b) {
    if (a.n_contigs == 0) return b.iqs;
    if (b.n_contigs == 0) return a.iqs;

    double L_total = a.total_length + b.total_length;
    std::vector<double> merged_centroid(BinIQSStats::N_CELLS);

    for (int d = 0; d < BinIQSStats::N_CELLS; d++) {
        merged_centroid[d] = (a.centroid[d] * a.total_length +
                              b.centroid[d] * b.total_length) / L_total;
    }

    // Cross-term for influence sum
    double cross_term = jsd_distance(a.centroid, b.centroid);
    double w_a = a.total_length / L_total;
    double w_b = b.total_length / L_total;

    double merged_sum_I2 = a.sum_influence_sq + b.sum_influence_sq +
                           w_a * w_b * cross_term * cross_term;

    int merged_n = a.n_contigs + b.n_contigs;

    double p = 0.8, q = 0.2;
    double alpha = p / (q * merged_n);
    double merged_var_80 = std::sqrt(alpha * merged_sum_I2);

    double merged_nmi = (a.nmi_weighted * a.n_contigs + b.nmi_weighted * b.n_contigs) / merged_n;

    return IQS_WEIGHT_VAR80 * merged_var_80 + IQS_WEIGHT_NMI * merged_nmi;
}

// Check if merge should be allowed based on IQS tolerance band
inline bool should_allow_merge(const BinIQSStats& a, const BinIQSStats& b,
                               double k = IQS_K_CONSERVATIVE) {
    double iqs_after = compute_merge_iqs(a, b);
    double iqs_min = std::min(a.iqs, b.iqs);
    double sigma_merged = std::sqrt(a.sigma_iqs * a.sigma_iqs + b.sigma_iqs * b.sigma_iqs);

    return iqs_after >= iqs_min - k * sigma_merged;
}

}  // namespace mfdfa_binner

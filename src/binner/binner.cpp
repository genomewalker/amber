// ==============================================================================
// AMBER - Ancient Metagenomic Binning for Environmental Reconstruction
//
// Ancient DNA metagenomic binner with multi-stage contamination cleanup.
//
// Pipeline:
//   FFT init → EM → Resonance Merger → Coverage-aware Cleanup →
//   TNF Cleanup → CGR+Fractal Cleanup → Post-merge Rescue → Short Assign
// ==============================================================================

#include <amber/config.hpp>
#include "internal/constants.h"
#include "internal/types.h"
#include "internal/quality_metrics.h"
#include "internal/probabilistic_models.h"
#include "internal/k_estimation.h"
#include "internal/cleanup_ops.h"
#include "../algorithms/damage_likelihood.h"
#include "../algorithms/damage_profile.h"
#include "../algorithms/kmer_features.h"
#include "../algorithms/fractal_features.h"
#include "../algorithms/coverage_features.h"
#include "../algorithms/sequence_utils.h"
#include "../util/logger.h"
#include "../algorithms/iqs_stats.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <filesystem>
#include <thread>
#include <mutex>
#include <queue>
#include <set>
#include <functional>
#include <omp.h>
#include <htslib/hts.h>
#include <htslib/sam.h>
#include <unistd.h>

namespace amber {

// Import mfdfa_binner namespace items
using mfdfa_binner::BinIQSStats;
using mfdfa_binner::ContigIQSInfo;
using mfdfa_binner::compute_merge_iqs;
using mfdfa_binner::should_allow_merge;
using mfdfa_binner::jsd_distance;
using mfdfa_binner::normalize_kmer;
using mfdfa_binner::IQS_WEIGHT_VAR80;
using mfdfa_binner::IQS_WEIGHT_NMI;
using mfdfa_binner::IQS_K_CONSERVATIVE;
using mfdfa_binner::IQS_K_MODERATE;
using mfdfa_binner::IQS_BUDGET_BETA;

// Namespace-level logger pointer for standalone functions
static Logger* g_log = nullptr;

// ==============================================================================
// Algorithm Configuration Constants
// ==============================================================================

// Contamination detection thresholds (validated on real bins vs CheckM2)
constexpr double kVar80Clean = 0.005;     // var_80 < 0.005 = very clean (low contamination)
constexpr double kVar80Warn = 0.015;      // var_80 > 0.015 = moderate contamination risk
constexpr double kVar80Bad = 0.025;       // var_80 > 0.025 = likely high contamination

// Cleanup parameters imported from cleanup_ops.h
using cleanup::kDefaultCovDeviation;
using cleanup::kDefaultMinCovCV;
using cleanup::kDefaultMinMeanCov;
using cleanup::kDefaultTnfSigmaThreshold;
using cleanup::kDefaultTnfTopPercentile;
using cleanup::kMinRetentionFraction;
using cleanup::kMinContigsForCleanup;

// Two-stage binning threshold
constexpr int kStage2LengthThreshold = 2500;     // Short contigs threshold (bp)

// Contig struct is now in internal/types.h

// Intrinsic Quality Metrics - see internal/quality_metrics.h
// (IntrinsicQuality, DownsampleVarResult, compute_cgr_occupancy,
//  compute_debruijn_modularity, jsd_distance_static)

// De novo contamination detection using CGR occupancy and downsampling variance

// Compute CGR occupancy from sequence (de novo)
double compute_cgr_occupancy_denovo(const std::string& seq, int k = 6) {
    int grid_size = 1 << k;
    int total_cells = grid_size * grid_size;
    std::vector<bool> visited(total_cells, false);

    double x = 0.5, y = 0.5;
    int steps = 0;
    for (char c : seq) {
        int base = -1;
        switch (std::toupper(c)) {
            case 'A': base = 0; break;
            case 'C': base = 1; break;
            case 'G': base = 2; break;
            case 'T': base = 3; break;
        }
        if (base < 0) continue;

        double tx = (base & 1) ? 1.0 : 0.0;
        double ty = (base & 2) ? 1.0 : 0.0;
        x = (x + tx) / 2.0;
        y = (y + ty) / 2.0;
        steps++;
        if (steps >= k) {
            int cx = std::min((int)(x * grid_size), grid_size - 1);
            int cy = std::min((int)(y * grid_size), grid_size - 1);
            visited[cy * grid_size + cx] = true;
        }
    }

    int occupied = 0;
    for (bool v : visited) if (v) occupied++;
    return (double)occupied / total_cells;
}

// Compute k-mer distribution from sequence (de novo)
std::vector<double> compute_kmer_distribution_denovo(const std::string& seq, int k = 6) {
    int grid_size = 1 << k;
    int total_cells = grid_size * grid_size;
    std::vector<int> counts(total_cells, 0);

    double x = 0.5, y = 0.5;
    int steps = 0;
    for (char c : seq) {
        int base = -1;
        switch (std::toupper(c)) {
            case 'A': base = 0; break;
            case 'C': base = 1; break;
            case 'G': base = 2; break;
            case 'T': base = 3; break;
        }
        if (base < 0) continue;

        double tx = (base & 1) ? 1.0 : 0.0;
        double ty = (base & 2) ? 1.0 : 0.0;
        x = (x + tx) / 2.0;
        y = (y + ty) / 2.0;
        steps++;
        if (steps >= k) {
            int cx = std::min((int)(x * grid_size), grid_size - 1);
            int cy = std::min((int)(y * grid_size), grid_size - 1);
            counts[cy * grid_size + cx]++;
        }
    }

    int total = 0;
    for (int cnt : counts) total += cnt;

    std::vector<double> probs(total_cells);
    for (int i = 0; i < total_cells; i++) {
        probs[i] = total > 0 ? (double)counts[i] / total : 0;
    }
    return probs;
}

// JS divergence (not sqrt) for contamination detection
double js_divergence_denovo(const std::vector<double>& p, const std::vector<double>& q) {
    double jsd = 0;
    for (size_t i = 0; i < p.size(); i++) {
        double m = (p[i] + q[i]) / 2;
        if (p[i] > 0 && m > 0) jsd += p[i] * std::log(p[i] / m);
        if (q[i] > 0 && m > 0) jsd += q[i] * std::log(q[i] / m);
    }
    return jsd / 2;
}

// Contamination score using composite metric
struct ContaminationScoreDenovo {
    double occ_range;       // Range of per-contig occupancy values
    double downsample_var;  // Variance of centroid JSD under 80% subsampling
    double combined;        // occ_range × downsample_var (threshold: 0.001)
};

// Compute contamination score de novo from bin members
ContaminationScoreDenovo compute_contamination_score_denovo(
    const std::vector<Contig*>& members, int num_samples = 20) {

    ContaminationScoreDenovo score = {0, 0, 0};
    if (members.size() < 5) return score;

    // Sort members by contig name for deterministic results
    // (iteration order can be non-deterministic otherwise)
    std::vector<Contig*> sorted_members = members;
    std::sort(sorted_members.begin(), sorted_members.end(),
              [](const Contig* a, const Contig* b) {
                  return a->name < b->name;
              });

    const int k = 6;
    const int grid_size = 1 << k;
    const int total_cells = grid_size * grid_size;

    // Compute per-contig occupancy and k-mer distributions
    std::vector<double> occupancies;
    std::vector<std::vector<double>> distributions;
    std::vector<size_t> lengths;

    for (auto* c : sorted_members) {
        occupancies.push_back(compute_cgr_occupancy_denovo(c->sequence, k));
        distributions.push_back(compute_kmer_distribution_denovo(c->sequence, k));
        lengths.push_back(c->length);
    }

    // Occupancy range
    double min_occ = *std::min_element(occupancies.begin(), occupancies.end());
    double max_occ = *std::max_element(occupancies.begin(), occupancies.end());
    score.occ_range = max_occ - min_occ;

    // Full centroid (length-weighted)
    std::vector<double> full_centroid(total_cells, 0);
    double total_len = 0;
    for (size_t i = 0; i < sorted_members.size(); i++) {
        for (int j = 0; j < total_cells; j++) {
            full_centroid[j] += distributions[i][j] * lengths[i];
        }
        total_len += lengths[i];
    }
    for (int j = 0; j < total_cells; j++) {
        full_centroid[j] /= total_len;
    }

    // Downsample variance (80% subsampling)
    std::mt19937 rng(42);
    std::vector<double> centroid_jsds;

    for (int s = 0; s < num_samples; s++) {
        std::vector<size_t> indices(sorted_members.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        size_t sample_size = sorted_members.size() * 80 / 100;
        if (sample_size < 3) sample_size = sorted_members.size();

        std::vector<double> sample_centroid(total_cells, 0);
        double sample_len = 0;

        for (size_t i = 0; i < sample_size; i++) {
            size_t idx = indices[i];
            for (int j = 0; j < total_cells; j++) {
                sample_centroid[j] += distributions[idx][j] * lengths[idx];
            }
            sample_len += lengths[idx];
        }
        for (int j = 0; j < total_cells; j++) {
            sample_centroid[j] /= sample_len;
        }

        double jsd = js_divergence_denovo(full_centroid, sample_centroid);
        centroid_jsds.push_back(jsd);
    }

    // Variance of centroid JSDs
    double mean = 0;
    for (double d : centroid_jsds) mean += d;
    mean /= centroid_jsds.size();

    double var = 0;
    for (double d : centroid_jsds) var += (d - mean) * (d - mean);
    var /= (centroid_jsds.size() - 1);

    score.downsample_var = var;
    score.combined = score.occ_range * score.downsample_var;

    return score;
}

// GMM component for coverage splitting
struct GMMComponent {
    double mean;
    double variance;
    double weight;
};

// Gaussian PDF
double gaussian_pdf(double x, double mean, double var) {
    if (var <= 0) return 0;
    double z = (x - mean) / std::sqrt(var);
    return std::exp(-0.5 * z * z) / std::sqrt(2 * M_PI * var);
}

// GMM E-step
void gmm_estep(const std::vector<double>& data,
                   const std::vector<GMMComponent>& components,
                   std::vector<std::vector<double>>& responsibilities) {
    size_t n = data.size();
    size_t k = components.size();

    for (size_t i = 0; i < n; i++) {
        double total = 0;
        for (size_t j = 0; j < k; j++) {
            responsibilities[i][j] = components[j].weight *
                gaussian_pdf(data[i], components[j].mean, components[j].variance);
            total += responsibilities[i][j];
        }
        if (total > 0) {
            for (size_t j = 0; j < k; j++) {
                responsibilities[i][j] /= total;
            }
        }
    }
}

// GMM M-step
void gmm_mstep(const std::vector<double>& data,
                   const std::vector<std::vector<double>>& responsibilities,
                   std::vector<GMMComponent>& components) {
    size_t n = data.size();
    size_t k = components.size();

    for (size_t j = 0; j < k; j++) {
        double nk = 0;
        double mean_sum = 0;

        for (size_t i = 0; i < n; i++) {
            nk += responsibilities[i][j];
            mean_sum += responsibilities[i][j] * data[i];
        }

        if (nk > 1e-10) {
            components[j].mean = mean_sum / nk;

            double var_sum = 0;
            for (size_t i = 0; i < n; i++) {
                double diff = data[i] - components[j].mean;
                var_sum += responsibilities[i][j] * diff * diff;
            }
            components[j].variance = std::max(var_sum / nk, 1.0);
            components[j].weight = nk / n;
        }
    }
}

// Log-likelihood for GMM
double compute_gmm_log_likelihood(const std::vector<double>& data,
                                   const std::vector<GMMComponent>& components) {
    double ll = 0;
    for (double x : data) {
        double p = 0;
        for (const auto& c : components) {
            p += c.weight * gaussian_pdf(x, c.mean, c.variance);
        }
        if (p > 0) ll += std::log(p);
    }
    return ll;
}

// Fit GMM with k components
double fit_gmm(const std::vector<double>& data, int k,
                   std::vector<GMMComponent>& components, int max_iter = 100) {
    size_t n = data.size();
    if (n < (size_t)k) return -1e30;

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    components.resize(k);
    for (int j = 0; j < k; j++) {
        size_t idx = (j + 1) * n / (k + 1);
        components[j].mean = sorted[idx];
        components[j].variance = 100.0;
        components[j].weight = 1.0 / k;
    }

    std::vector<std::vector<double>> responsibilities(n, std::vector<double>(k));

    double prev_ll = -1e30;
    for (int iter = 0; iter < max_iter; iter++) {
        gmm_estep(data, components, responsibilities);
        gmm_mstep(data, responsibilities, components);

        double ll = compute_gmm_log_likelihood(data, components);
        if (std::abs(ll - prev_ll) < 1e-6) break;
        prev_ll = ll;
    }

    return prev_ll;
}

// Find optimal k using BIC
int find_optimal_gmm_components(const std::vector<double>& data, int max_k = 3) {
    if (data.size() < 10) return 1;

    double best_bic = 1e30;
    int best_k = 1;

    for (int k = 1; k <= max_k; k++) {
        std::vector<GMMComponent> components;
        double ll = fit_gmm(data, k, components);
        int num_params = 3 * k - 1;
        double bic = -2 * ll + 2.0 * num_params * std::log(data.size());

        if (bic < best_bic) {
            best_bic = bic;
            best_k = k;
        }
    }

    return best_k;
}

// Compute downsample variance (var_80%) - PRIMARY contamination metric
// Validated: r=+0.528 with contamination on real bins (BEST metric!)
//
// Method: Repeatedly subsample 80% of contigs, compute centroid JSD to full centroid
// High variance = inconsistent composition = likely contaminated
DownsampleVarResult compute_downsample_var(const std::vector<Contig*>& members,
                                           int n_reps = 20, unsigned seed = 42) {
    DownsampleVarResult result;
    if (members.size() < 5) return result;

    const size_t N_CELLS = 4096;  // 6-mer

    // Filter to contigs with valid k-mer distributions
    std::vector<Contig*> valid_members;
    for (auto* c : members) {
        if (c->kmer6_probs.size() == N_CELLS) {
            valid_members.push_back(c);
        }
    }

    int n = valid_members.size();
    if (n < 5) return result;

    // Compute full centroid (length-weighted)
    std::vector<double> full_centroid(N_CELLS, 0.0);
    double total_len = 0;
    for (auto* c : valid_members) {
        for (size_t i = 0; i < N_CELLS; i++) {
            full_centroid[i] += c->kmer6_probs[i] * c->length;
        }
        total_len += c->length;
    }
    for (size_t i = 0; i < N_CELLS; i++) {
        full_centroid[i] /= total_len;
    }

    // Compute mean JSD to centroid (for comparison)
    double jsd_sum = 0;
    for (auto* c : valid_members) {
        jsd_sum += jsd_distance_static(c->kmer6_probs, full_centroid);
    }
    result.mean_jsd = jsd_sum / n;

    // Compute downsample variance
    std::mt19937 rng(seed);
    int keep = std::max(2, (int)(n * 0.8));  // 80% subsample

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<double> subsample_jsds;
    subsample_jsds.reserve(n_reps);

    for (int rep = 0; rep < n_reps; rep++) {
        std::shuffle(indices.begin(), indices.end(), rng);

        // Compute subsample centroid
        std::vector<double> sub_centroid(N_CELLS, 0.0);
        double sub_len = 0;

        for (int j = 0; j < keep; j++) {
            auto* c = valid_members[indices[j]];
            for (size_t i = 0; i < N_CELLS; i++) {
                sub_centroid[i] += c->kmer6_probs[i] * c->length;
            }
            sub_len += c->length;
        }

        if (sub_len > 0) {
            for (size_t i = 0; i < N_CELLS; i++) {
                sub_centroid[i] /= sub_len;
            }
        }

        // JSD distance from subsample centroid to full centroid
        double d = jsd_distance_static(sub_centroid, full_centroid);
        subsample_jsds.push_back(d);
    }

    // Compute variance of subsample distances
    double mean_d = 0;
    for (double d : subsample_jsds) mean_d += d;
    mean_d /= subsample_jsds.size();

    double var_d = 0;
    for (double d : subsample_jsds) {
        var_d += (d - mean_d) * (d - mean_d);
    }
    var_d /= subsample_jsds.size();

    result.var_80 = std::sqrt(var_d);  // Return std dev
    result.n_contigs = n;
    result.valid = true;
    return result;
}

IntrinsicQuality assess_intrinsic_quality(const std::vector<Contig*>& members) {
    IntrinsicQuality iq;
    if (members.empty()) return iq;

    // Concatenate sequences for CGR occupancy (validated r=+0.368)
    std::string combined;
    size_t total_bp = 0;
    for (auto* c : members) {
        combined += c->sequence;
        total_bp += c->length;
    }

    if (combined.length() < 10000) return iq;

    // Compute CGR occupancy - SECONDARY contamination metric (r=+0.368)
    iq.cgr_occupancy = compute_cgr_occupancy(combined);

    // Compute modularity (DEPRECATED - kept for logging only)
    iq.debruijn_modularity = compute_debruijn_modularity(combined);

    // Compute downsample variance - PRIMARY contamination metric (r=+0.528)
    auto var_result = compute_downsample_var(members);
    iq.downsample_var = var_result.valid ? var_result.var_80 : 0.0;
    iq.mean_jsd = var_result.valid ? var_result.mean_jsd : 1.0;  // Keep for logging

    // Contamination estimation from downsample_var (validated metric!)
    // Linear mapping based on observed correlation with CheckM2
    // var_80 < 0.005 → ~0% contamination
    // var_80 > 0.025 → ~50%+ contamination
    if (iq.downsample_var < kVar80Clean) {
        iq.contamination_est = 0.05 * (iq.downsample_var / kVar80Clean);
    } else if (iq.downsample_var < kVar80Bad) {
        iq.contamination_est = 0.05 + 0.45 * ((iq.downsample_var - kVar80Clean) / (kVar80Bad - kVar80Clean));
    } else {
        iq.contamination_est = 0.50 + 0.50 * std::min(1.0, (iq.downsample_var - kVar80Bad) / 0.025);
    }
    iq.contamination_est = std::max(0.0, std::min(1.0, iq.contamination_est));

    // Completeness estimation - use CGR occupancy as proxy
    // Higher CGR occupancy correlates with larger/more complete genomes
    iq.completeness_est = std::max(0.0, std::min(1.0, iq.cgr_occupancy));

    // Decision flags based on downsample_var
    iq.is_likely_clean = (iq.downsample_var < kVar80Clean);
    iq.is_likely_complete = (iq.cgr_occupancy > 0.5);  // Use CGR for completeness
    iq.needs_splitting = (iq.downsample_var > kVar80Bad);

    return iq;
}

// Probabilistic models (MultivariateNormal, LogNormal, CoverageModel) are in internal/probabilistic_models.h

// V50: ARD (Automatic Relevance Determination) precisions for adaptive feature weighting
// Each bin learns its own feature weights based on within-bin variance
// High precision = feature is reliable for this bin (tight distribution)
// Low precision = feature is noisy for this bin (wide distribution)
// Bin model with variance-aware coverage
struct Bin {
    MultivariateNormal tnf_dist;
    MultivariateNormal cgr_dist;
    MultivariateNormal dna_dist;
    MultivariateNormal fractal_dist;
    CoverageModel coverage_model;

    std::vector<Contig*> members;
    double weight = 0.0;

    BinIQSStats iqs_stats;

    double mean_contig_modularity = -1.0;
    double std_contig_modularity = 0.0;    // Std of member contig modularities
    double current_cgr_occupancy = 0.0;    // CGR occupancy of concatenated bin

    // Quantize value to 4 decimal places for robust distance calculations
    // Reduces sensitivity to small perturbations from polish
    static double quantize(double v) {
        return std::round(v * 10000.0) / 10000.0;
    }

    // Quantize a feature vector
    static std::vector<double> quantize_vec(const std::vector<double>& v) {
        std::vector<double> q(v.size());
        for (size_t i = 0; i < v.size(); i++) {
            q[i] = quantize(v[i]);
        }
        return q;
    }

    void fit() {
        if (members.empty()) return;

        std::vector<std::vector<double>> tnf_data, cgr_data, dna_data, fractal_data;
        std::vector<std::pair<double, double>> cov_var_data;

        for (auto* c : members) {
            if (!c->tnf.empty()) tnf_data.push_back(c->tnf);
            if (!c->cgr.empty()) cgr_data.push_back(c->cgr);
            if (!c->dna_physical.empty()) dna_data.push_back(c->dna_physical);
            if (!c->fractal.empty()) fractal_data.push_back(c->fractal);
            if (c->mean_coverage > 0) {
                cov_var_data.push_back({c->mean_coverage, c->var_coverage});
            }
        }

        if (tnf_data.size() >= 3) tnf_dist.fit(tnf_data);
        if (cgr_data.size() >= 3) cgr_dist.fit(cgr_data);
        if (dna_data.size() >= 3) dna_dist.fit(dna_data);
        if (fractal_data.size() >= 3) fractal_dist.fit(fractal_data);
        coverage_model.fit(cov_var_data);

        weight = members.size();

        // Compute mean modularity from pre-computed per-contig values
        std::vector<double> mods;
        std::string combined;
        for (auto* c : members) {
            if (c->contig_modularity > -2.0) {  // Valid modularity
                mods.push_back(c->contig_modularity);
            }
            combined += c->sequence;
        }
        if (mods.size() >= 3) {
            mean_contig_modularity = std::accumulate(mods.begin(), mods.end(), 0.0) / mods.size();
            double var = 0.0;
            for (double m : mods) var += (m - mean_contig_modularity) * (m - mean_contig_modularity);
            std_contig_modularity = std::sqrt(var / (mods.size() - 1) + 0.01);
        }
        // Compute CGR occupancy of combined bin (for completeness gate)
        if (combined.length() >= 10000) {
            current_cgr_occupancy = compute_cgr_occupancy(combined, 10);
        }

        // Compute IQS statistics for merge decisions
        // Build vector of contig IQS info pointers (need actual ContigIQSInfo objects)
        std::vector<ContigIQSInfo*> iqs_infos;
        for (size_t i = 0; i < members.size(); i++) {
            auto* c = members[i];
            if (!c->kmer6_probs.empty()) {
                // Create a temporary ContigIQSInfo (stored in the contig)
                // Note: This requires ContigIQSInfo storage, we'll use a simpler approach
                // by computing IQS directly from the k-mer distributions
            }
        }
        // For now, compute a simplified IQS estimate using existing features
        compute_iqs_from_members();
    }

    void compute_iqs_from_members() {
        if (members.empty()) return;

        // Reset IQS stats
        iqs_stats = BinIQSStats();
        iqs_stats.n_contigs = members.size();
        iqs_stats.total_length = 0;

        // Collect contigs with 6-mer distributions
        std::vector<std::pair<Contig*, std::vector<double>>> valid_contigs;
        for (auto* c : members) {
            iqs_stats.total_length += c->length;
            if (!c->kmer6_probs.empty() && c->kmer6_probs.size() == 4096) {
                valid_contigs.push_back({c, c->kmer6_probs});
            }
        }

        if (valid_contigs.empty()) return;

        // Compute length-weighted centroid
        iqs_stats.centroid.assign(4096, 0.0);
        double total_w = 0;
        for (auto& [c, probs] : valid_contigs) {
            double w = (double)c->length / iqs_stats.total_length;
            total_w += w;
            for (int d = 0; d < 4096; d++) {
                iqs_stats.centroid[d] += w * probs[d];
            }
        }
        if (total_w > 0) {
            for (int d = 0; d < 4096; d++) {
                iqs_stats.centroid[d] /= total_w;
            }
        }

        // Compute influence scores and sum
        iqs_stats.sum_influence_sq = 0;
        for (auto& [c, probs] : valid_contigs) {
            double w = (double)c->length / iqs_stats.total_length;
            double d = jsd_distance(probs, iqs_stats.centroid);
            c->iqs_influence = w * d;
            iqs_stats.sum_influence_sq += c->iqs_influence * c->iqs_influence;
        }

        // Compute var_80 estimate
        if (valid_contigs.size() > 0) {
            double p = 0.8, q = 0.2;
            double alpha = p / (q * valid_contigs.size());
            iqs_stats.var_80_est = std::sqrt(alpha * iqs_stats.sum_influence_sq);
        }

        // NMI estimate (simplified: use mean JSD)
        double mean_jsd = 0;
        for (auto& [c, probs] : valid_contigs) {
            mean_jsd += jsd_distance(probs, iqs_stats.centroid);
        }
        mean_jsd /= valid_contigs.size();
        iqs_stats.nmi_weighted = std::max(0.0, 1.0 - mean_jsd);

        // Compute final IQS
        iqs_stats.iqs = IQS_WEIGHT_VAR80 * iqs_stats.var_80_est +
                        IQS_WEIGHT_NMI * iqs_stats.nmi_weighted;

        // Estimate uncertainty
        double var_80_uncertainty = iqs_stats.var_80_est / std::sqrt(std::max(1, (int)valid_contigs.size()));
        iqs_stats.sigma_iqs = IQS_WEIGHT_VAR80 * var_80_uncertainty;

        // Initialize quality budget
        iqs_stats.budget_initial = IQS_BUDGET_BETA * iqs_stats.iqs;
        iqs_stats.budget_remaining = iqs_stats.budget_initial;
    }

    double log_probability(const Contig& c) const {
        double log_p = std::log(weight + 1e-10);

        // Baseline fixed weights (empirically tuned)

        // TNF (primary compositional signal)
        if (tnf_dist.valid && !c.tnf.empty()) {
            log_p += TNF_LOG_WEIGHT * tnf_dist.log_probability(c.tnf);
        }

        // CGR (k-mer density pattern)
        if (cgr_dist.valid && !c.cgr.empty()) {
            log_p += CGR_LOG_WEIGHT * cgr_dist.log_probability(c.cgr);
        }

        // Fractal features (CGR-DFA Hurst, anisotropy, scale ratios)
        if (fractal_dist.valid && !c.fractal.empty()) {
            log_p += FRACTAL_LOG_WEIGHT * fractal_dist.log_probability(c.fractal);
        }

        // DNA physical properties
        if (dna_dist.valid && !c.dna_physical.empty()) {
            log_p += DNA_PHYSICAL_LOG_WEIGHT * dna_dist.log_probability(c.dna_physical);
        }

        // Coverage (crucial for bin separation)
        if (coverage_model.valid && c.mean_coverage > 0) {
            log_p += COVERAGE_LOG_WEIGHT * coverage_model.log_probability(c.mean_coverage, c.var_coverage);
        }

        return log_p;
    }

    // Log probability without fractal DFA features (for short contigs)
    double log_probability_no_dfa(const Contig& c) const {
        double log_p = std::log(weight + 1e-10);

        // Baseline fixed weights

        // TNF (primary compositional signal)
        if (tnf_dist.valid && !c.tnf.empty()) {
            log_p += TNF_LOG_WEIGHT * tnf_dist.log_probability(c.tnf);
        }

        // CGR (k-mer density pattern) - stable on short contigs
        if (cgr_dist.valid && !c.cgr.empty()) {
            log_p += CGR_LOG_WEIGHT * cgr_dist.log_probability(c.cgr);
        }

        // Fractal features skipped for short contigs (unstable)

        // DNA physical properties - stable on short contigs
        if (dna_dist.valid && !c.dna_physical.empty()) {
            log_p += DNA_PHYSICAL_LOG_WEIGHT * dna_dist.log_probability(c.dna_physical);
        }

        // Coverage
        if (coverage_model.valid && c.mean_coverage > 0) {
            log_p += COVERAGE_LOG_WEIGHT * coverage_model.log_probability(c.mean_coverage, c.var_coverage);
        }

        return log_p;
    }

    bool should_accept(const Contig& c) const {
        if (members.size() < 3) return true;

        // Completeness gate - if bin looks complete (CGR occupancy > 0.58), stop adding
        // Based on calibration: 100% complete → 0.62 occupancy
        // Use 0.58 as threshold (~90% complete)
        if (current_cgr_occupancy > 0.58) {
            // Bin is already near-complete, be very selective
            // Only accept if coverage matches very closely
            if (c.mean_coverage > 0 && coverage_model.valid) {
                double log_cov = std::log(c.mean_coverage);
                double z = std::abs(log_cov - coverage_model.bin_mean_log_cov);
                double sigma = std::sqrt(coverage_model.bin_cov_variance + 0.01);
                if (z > 1.5 * sigma) {  // Tighter threshold for complete bins
                    return false;
                }
            }
        }

        // Contamination gate - check if contig modularity differs from bin mean
        // Uses PRE-COMPUTED per-contig modularity (O(1) check, not O(n) recomputation)
        if (mean_contig_modularity > -2.0 && c.contig_modularity > -2.0) {
            double diff = std::abs(c.contig_modularity - mean_contig_modularity);
            // Reject if contig modularity is more than 2.5σ from bin mean
            if (diff > 2.5 * std_contig_modularity + 0.1) {
                return false;
            }
        }

        return true;
    }
};

// Parallel coverage loading with per-thread BAM handles (htslib is not thread-safe for shared handles)
std::map<std::string, std::pair<double, double>> load_coverage_with_variance(const std::string& bam_path, int num_threads = 0) {
    std::map<std::string, std::pair<double, double>> coverage;

    // If num_threads not specified, use OpenMP default
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    // First, open BAM once to get contig list (header only)
    samFile* bam_fp = sam_open(bam_path.c_str(), "r");
    if (!bam_fp) {
        if (g_log) g_log->warn("Cannot open BAM file: " + bam_path);
        return coverage;
    }

    bam_hdr_t* header = sam_hdr_read(bam_fp);
    if (!header) {
        sam_close(bam_fp);
        if (g_log) g_log->warn("Cannot read BAM header");
        return coverage;
    }

    // Copy contig info for parallel processing
    int n_contigs = header->n_targets;
    std::vector<std::string> contig_names(n_contigs);
    std::vector<uint32_t> contig_lengths(n_contigs);

    for (int i = 0; i < n_contigs; ++i) {
        contig_names[i] = header->target_name[i];
        contig_lengths[i] = header->target_len[i];
    }

    bam_hdr_destroy(header);
    sam_close(bam_fp);

    // Prepare thread-local results
    std::vector<std::map<std::string, std::pair<double, double>>> thread_results(num_threads);

    // MetaBAT2-compatible depth parameters
    DepthParams params;
    params.max_edge_bases = 75;           // MetaBAT2 default
    params.min_percent_identity = 97.0f;  // MetaBAT2 default
    params.min_mapq = 0;                   // MetaBAT2 default

    auto start_time = std::chrono::high_resolution_clock::now();
    if (g_log) g_log->detail("Extracting coverage for " + std::to_string(n_contigs) + " contigs with " + std::to_string(num_threads) + " threads");

    // Parallel coverage extraction
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();

        // Each thread opens its own BAM file and index (htslib requirement)
        samFile* local_bam = sam_open(bam_path.c_str(), "r");
        if (!local_bam) {
            if (g_log) g_log->warn("Thread " + std::to_string(thread_id) + ": Cannot open BAM");
        } else {
            bam_hdr_t* local_header = sam_hdr_read(local_bam);
            hts_idx_t* local_idx = sam_index_load(local_bam, bam_path.c_str());

            if (!local_header || !local_idx) {
                if (g_log) g_log->warn("Thread " + std::to_string(thread_id) + ": Cannot load header/index");
            } else {
                // Process contigs assigned to this thread
                #pragma omp for schedule(dynamic, 100)
                for (int tid = 0; tid < n_contigs; ++tid) {
                    DepthResult result = CoverageExtractor::extract_depth(
                        local_bam, local_header, local_idx,
                        contig_names[tid], contig_lengths[tid], params);

                    thread_results[thread_id][contig_names[tid]] = {result.mean_depth, result.variance};
                }
            }

            if (local_idx) hts_idx_destroy(local_idx);
            if (local_header) bam_hdr_destroy(local_header);
            sam_close(local_bam);
        }
    }

    // Merge thread-local results
    for (int t = 0; t < num_threads; ++t) {
        for (const auto& [name, cov] : thread_results[t]) {
            coverage[name] = cov;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if (g_log) {
        std::ostringstream ss;
        ss << "Coverage extraction: " << elapsed << " ms (" << std::fixed << std::setprecision(1)
           << (n_contigs * 1000.0 / elapsed) << " contigs/sec)";
        g_log->detail(ss.str());
    }

    return coverage;
}

// K estimation functions: count_coverage_modes, count_composition_clusters, estimate_k_data_driven
// are in internal/k_estimation.h

// ========== Main Binner ==========
class MFDFABinner {
public:
    std::vector<Bin> bins;
    std::vector<Contig> contigs;
    int n_threads = 1;
    int min_length = 2500;
    int random_seed = 42;
    int ensemble_runs = 1;  // Number of ensemble runs (1 = disabled)
    bool use_fractal = true;
    bool use_merge = true;
    bool use_kmedoids = false;
    bool use_farthest_first = true;
    bool use_refinement = true;
    int refinement_rounds = 5;
    double cov_weight = 1.0;  // Default weight for coverage in distance
    double merge_posterior_threshold = 0.8;
    double damage_relaxed_threshold = 0.5;   // Relaxed threshold when damage matches (KS test)
    double ks_pvalue_threshold = 0.8;

    // CGR+fractal cleanup thresholds (achieves 8 HQ bins on reference dataset)
    double cgr_z_threshold = kCGRZThreshold;       // CGR outlier z-score threshold
    double fractal_z_threshold = kFractalZThreshold;   // Fractal confirmation z-score threshold
    double max_removal_frac = kMaxRemovalFrac;     // Max fraction of bin to remove (15%)

    Logger log_;

    // Helper to check if verbose output is enabled
    bool verbose() const { return log_.console_level >= Verbosity::Verbose; }

    // Internal damage analysis
    DamageAnalysis damage_analysis_;
    bool damage_computed_ = false;
    std::string library_type_override;  // Override auto-detected library type (ds/ss)
    bool export_smiley_data = false;    // Export data for smiley plots
    bool anvio_mode = false;            // Rename contigs for anvi'o compatibility
    std::string anvio_prefix = "c";     // Prefix for anvi'o contig names

    // Temporal chimera detection thresholds
    double temporal_minority_ratio = 0.1;   // Max minority/majority ratio for temporal_outliers
    double temporal_damage_std = 0.15;      // Max damage_std for temporal_outliers

    // Damage classification thresholds (for temporal analysis)
    double ancient_p_threshold = 0.8;       // Min p_ancient for ANCIENT classification
    double modern_p_threshold = 0.5;        // Max p_ancient for MODERN classification
    double ancient_amp_threshold = 0.01;    // Min amplitude for ANCIENT classification (1%)

    void load_contigs(const std::string& fasta_path) {
        log_.section("Contig loading");
        log_.metric("input_file", fasta_path);
        log_.metric("min_length_filter", min_length);

        std::ifstream file(fasta_path);
        std::string line, name, seq;
        size_t total_contigs = 0;
        size_t filtered_contigs = 0;
        size_t total_bp = 0;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (line[0] == '>') {
                if (!name.empty()) {
                    total_contigs++;
                    if (seq.length() >= (size_t)min_length) {
                        contigs.push_back(Contig());
                        contigs.back().name = name;
                        contigs.back().sequence = seq;
                        contigs.back().length = seq.length();
                        total_bp += seq.length();
                    } else {
                        filtered_contigs++;
                    }
                }
                name = line.substr(1);
                auto space = name.find(' ');
                if (space != std::string::npos) name = name.substr(0, space);
                seq.clear();
            } else {
                seq += line;
            }
        }
        if (!name.empty()) {
            total_contigs++;
            if (seq.length() >= (size_t)min_length) {
                contigs.push_back(Contig());
                contigs.back().name = name;
                contigs.back().sequence = seq;
                contigs.back().length = seq.length();
                total_bp += seq.length();
            } else {
                filtered_contigs++;
            }
        }

        log_.metric("total_contigs_in_file", (int)total_contigs);
        log_.metric("contigs_after_filter", (int)contigs.size());
        log_.metric("filtered_out", (int)filtered_contigs);
        log_.metric("total_bp", (int)total_bp);
        log_.info("Loaded " + std::to_string(contigs.size()) + " contigs >= " + std::to_string(min_length) + " bp");
    }

    void load_coverage_data(const std::string& bam_path) {
        log_.section("Coverage loading");
        log_.metric("bam_file", bam_path);
        log_.metric("threads", n_threads);

        auto cov = load_coverage_with_variance(bam_path, n_threads);
        int found = 0;
        double min_cov = 1e30, max_cov = 0, sum_cov = 0;
        double min_var = 1e30, max_var = 0, sum_var = 0;
        int var_count = 0;

        for (auto& c : contigs) {
            auto it = cov.find(c.name);
            if (it != cov.end()) {
                c.mean_coverage = it->second.first;
                c.var_coverage = it->second.second;
                found++;
                min_cov = std::min(min_cov, c.mean_coverage);
                max_cov = std::max(max_cov, c.mean_coverage);
                sum_cov += c.mean_coverage;
                if (c.var_coverage > 0) {
                    min_var = std::min(min_var, c.var_coverage);
                    max_var = std::max(max_var, c.var_coverage);
                    sum_var += c.var_coverage;
                    var_count++;
                }
            }
        }

        log_.metric("contigs_with_coverage", found);

        if (found > 0) {
            log_.subsection("Coverage Distribution");
            log_.metric("min_coverage", min_cov);
            log_.metric("mean_coverage", sum_cov / found);
            log_.metric("max_coverage", max_cov);
        }

        if (var_count > 0) {
            log_.subsection("Variance Distribution (V10)");
            log_.metric("min_variance", min_var);
            log_.metric("mean_variance", sum_var / var_count);
            log_.metric("max_variance", max_var);
            log_.metric("contigs_with_variance", var_count);
        }

        log_.info("Loaded coverage for " + std::to_string(found) + " contigs");
    }

    void populate_contig_damage(const DamageAnalysis& analysis) {
        int found = 0;
        for (auto& c : contigs) {
            auto profile_it = analysis.profile.per_contig.find(c.name);
            if (profile_it != analysis.profile.per_contig.end() && !profile_it->second.empty()) {
                double sum_ct = 0, sum_ga = 0;
                int count = 0;
                for (const auto& entry : profile_it->second) {
                    if (entry.pos >= 1 && entry.pos <= 10) {
                        sum_ct += entry.c_to_t;
                        sum_ga += entry.g_to_a;
                        count++;
                    }
                }
                if (count > 0) {
                    c.damage_rate_5p = sum_ct / count;
                    c.damage_rate_3p = sum_ga / count;
                    c.has_damage = true;
                    found++;
                }
            }

            // Populate Bayesian parameters from per-contig model if available
            auto model_it = analysis.per_contig_model.find(c.name);
            if (model_it != analysis.per_contig_model.end()) {
                const auto& model = model_it->second;
                c.damage_amplitude_5p = model.curve_5p.amplitude;
                c.damage_lambda_5p = model.curve_5p.lambda;
                c.damage_baseline_5p = model.curve_5p.baseline;
                c.damage_amplitude_3p = model.curve_3p.amplitude;
                c.damage_lambda_3p = model.curve_3p.lambda;
                c.damage_baseline_3p = model.curve_3p.baseline;
                c.damage_significant = model.is_significant;
                c.damage_p_ancient = model.p_ancient;
            } else if (c.has_damage) {
                // Fallback: use global model parameters
                const auto& gm = analysis.global_model;
                c.damage_amplitude_5p = gm.curve_5p.amplitude;
                c.damage_lambda_5p = gm.curve_5p.lambda;
                c.damage_baseline_5p = gm.curve_5p.baseline;
                c.damage_amplitude_3p = gm.curve_3p.amplitude;
                c.damage_lambda_3p = gm.curve_3p.lambda;
                c.damage_baseline_3p = gm.curve_3p.baseline;
                c.damage_significant = gm.is_significant;
                c.damage_p_ancient = gm.p_ancient;
            }
        }
        log_.info("Populated damage for " + std::to_string(found) + " contigs");
    }

    void compute_damage_from_bam(const std::string& bam_path) {
        log_.section("Computing damage profile from BAM");

        std::unordered_set<std::string> contig_filter;
        for (const auto& c : contigs) {
            contig_filter.insert(c.name);
        }

        DamageComputeConfig dc_config;
        dc_config.threads = n_threads;
        dc_config.min_mapq = 30;
        dc_config.min_baseq = 20;
        dc_config.max_position = 20;

        std::string err;
        auto start = std::chrono::steady_clock::now();

        if (!compute_damage_analysis(bam_path, damage_analysis_, err, dc_config, &contig_filter)) {
            log_.error("Failed to compute damage profile: " + err);
            throw std::runtime_error("Cannot compute damage profile from BAM: " + err);
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count();
        log_.info("Damage computation completed in " + std::to_string(elapsed) + "s");

        damage_computed_ = true;

        // Apply library type override if specified
        if (!library_type_override.empty()) {
            log_.info("Overriding detected library type '" + damage_analysis_.library_type +
                     "' with user-specified '" + library_type_override + "'");
            damage_analysis_.library_type = library_type_override;
        }

        const auto& gm = damage_analysis_.global_model;
        log_.subsection("Bayesian damage model");
        log_.metric("p_ancient", gm.p_ancient);
        log_.metric("log_bf", gm.log_bf);
        std::string evidence = gm.evidence_strength >= 4.0 ? "decisive" :
                               gm.evidence_strength >= 3.0 ? "strong" :
                               gm.evidence_strength >= 2.0 ? "moderate" :
                               gm.evidence_strength >= 1.0 ? "weak" : "none";
        log_.metric("evidence_strength", evidence);

        if (!gm.stats_5p.empty()) {
            log_.metric("ct_5p_pos1", gm.stats_5p[0].rate);
            log_.metric("ct_5p_ci", "[" + std::to_string(gm.stats_5p[0].ci_lower) + ", " +
                        std::to_string(gm.stats_5p[0].ci_upper) + "]");
        }
        if (!gm.stats_3p.empty()) {
            log_.metric("ga_3p_pos1", gm.stats_3p[0].rate);
        }
        log_.metric("baseline_error", damage_analysis_.calibrated_error_rate);
        log_.metric("library_type", damage_analysis_.library_type);

        if (!gm.is_significant) {
            log_.info("Damage not significant: " + gm.significance_reason);
        }

        populate_contig_damage(damage_analysis_);
    }

    void write_damage_outputs(const std::string& output_dir) {
        std::ofstream profile_out(output_dir + "/damage_profile.tsv");
        profile_out << "contig\tposition\tc_to_t\tg_to_a\n";
        for (const auto& [contig, entries] : damage_analysis_.profile.per_contig) {
            for (const auto& e : entries) {
                profile_out << contig << "\t" << e.pos << "\t"
                           << e.c_to_t << "\t" << e.g_to_a << "\n";
            }
        }
        profile_out.close();

        std::ofstream model_out(output_dir + "/damage_model.tsv");
        const auto& gm = damage_analysis_.global_model;
        model_out << "metric\tvalue\n";
        model_out << "p_ancient\t" << gm.p_ancient << "\n";
        model_out << "damage_class\t" << classification_to_string(gm.classification) << "\n";
        model_out << "log_bf\t" << gm.log_bf << "\n";
        model_out << "is_significant\t" << (gm.is_significant ? "true" : "false") << "\n";
        model_out << "significance_reason\t" << gm.significance_reason << "\n";
        model_out << "amplitude_5p\t" << gm.curve_5p.amplitude << "\n";
        model_out << "lambda_5p\t" << gm.curve_5p.lambda << "\n";
        model_out << "baseline_5p\t" << gm.curve_5p.baseline << "\n";
        model_out << "amplitude_3p\t" << gm.curve_3p.amplitude << "\n";
        model_out << "lambda_3p\t" << gm.curve_3p.lambda << "\n";
        model_out << "baseline_3p\t" << gm.curve_3p.baseline << "\n";
        model_out << "library_type\t" << damage_analysis_.library_type << "\n";
        model_out << "calibrated_error_rate\t" << damage_analysis_.calibrated_error_rate << "\n";
        model_out.close();

        log_.info("Wrote damage outputs to " + output_dir);
    }

    void write_smiley_plot_data(const std::string& output_dir) {
        const auto& gm = damage_analysis_.global_model;

        // Export global smiley data (5' and 3' combined)
        std::ofstream out_global(output_dir + "/smiley_global.tsv");
        out_global << "position\tobs_5p\tmodel_5p\tobs_3p\tmodel_3p\tn_opp_5p\tn_opp_3p\n";

        size_t max_pos = std::max(gm.stats_5p.size(), gm.stats_3p.size());
        for (size_t i = 0; i < max_pos; i++) {
            int pos = (int)i + 1;
            double obs_5p = 0.0, model_5p = 0.0, obs_3p = 0.0, model_3p = 0.0;
            uint64_t n_opp_5p = 0, n_opp_3p = 0;

            if (i < gm.stats_5p.size()) {
                obs_5p = gm.stats_5p[i].rate;
                model_5p = gm.curve_5p.amplitude * std::exp(-gm.curve_5p.lambda * i) + gm.curve_5p.baseline;
                n_opp_5p = gm.stats_5p[i].n_opportunities;
            }
            if (i < gm.stats_3p.size()) {
                obs_3p = gm.stats_3p[i].rate;
                model_3p = gm.curve_3p.amplitude * std::exp(-gm.curve_3p.lambda * i) + gm.curve_3p.baseline;
                n_opp_3p = gm.stats_3p[i].n_opportunities;
            }

            out_global << pos << "\t"
                       << std::setprecision(6) << obs_5p << "\t"
                       << model_5p << "\t"
                       << obs_3p << "\t"
                       << model_3p << "\t"
                       << n_opp_5p << "\t"
                       << n_opp_3p << "\n";
        }
        out_global.close();

        // Export per-contig smiley data
        std::ofstream out_contig(output_dir + "/smiley_per_contig.tsv");
        out_contig << "contig\tposition\tobs_5p\tmodel_5p\tobs_3p\tmodel_3p\tn_opp_5p\tn_opp_3p\n";

        for (const auto& [contig_name, model] : damage_analysis_.per_contig_model) {
            size_t cmax = std::max(model.stats_5p.size(), model.stats_3p.size());
            for (size_t i = 0; i < cmax; i++) {
                int pos = (int)i + 1;
                double obs_5p = 0.0, model_5p = 0.0, obs_3p = 0.0, model_3p = 0.0;
                uint64_t n_opp_5p = 0, n_opp_3p = 0;

                if (i < model.stats_5p.size()) {
                    obs_5p = model.stats_5p[i].rate;
                    model_5p = model.curve_5p.amplitude * std::exp(-model.curve_5p.lambda * i) + model.curve_5p.baseline;
                    n_opp_5p = model.stats_5p[i].n_opportunities;
                }
                if (i < model.stats_3p.size()) {
                    obs_3p = model.stats_3p[i].rate;
                    model_3p = model.curve_3p.amplitude * std::exp(-model.curve_3p.lambda * i) + model.curve_3p.baseline;
                    n_opp_3p = model.stats_3p[i].n_opportunities;
                }

                out_contig << contig_name << "\t"
                           << pos << "\t"
                           << std::setprecision(6) << obs_5p << "\t"
                           << model_5p << "\t"
                           << obs_3p << "\t"
                           << model_3p << "\t"
                           << n_opp_5p << "\t"
                           << n_opp_3p << "\n";
            }
        }
        out_contig.close();

        log_.info("Wrote smiley plot data: smiley_global.tsv, smiley_per_contig.tsv");
    }

    void write_smiley_per_bin(const std::string& output_dir,
                               const std::unordered_map<int, int>& internal_to_export_id) {
        // Aggregate damage stats per bin using EXPORTED bin IDs
        std::unordered_map<int, std::vector<uint64_t>> bin_n_5p, bin_k_5p, bin_n_3p, bin_k_3p;
        const int max_pos = 20;

        // Initialize bins (only for exported bins)
        for (const auto& c : contigs) {
            if (c.bin_id >= 0) {
                auto it = internal_to_export_id.find(c.bin_id);
                if (it == internal_to_export_id.end()) continue;
                int export_id = it->second;
                if (bin_n_5p.find(export_id) == bin_n_5p.end()) {
                    bin_n_5p[export_id].resize(max_pos, 0);
                    bin_k_5p[export_id].resize(max_pos, 0);
                    bin_n_3p[export_id].resize(max_pos, 0);
                    bin_k_3p[export_id].resize(max_pos, 0);
                }
            }
        }

        // Aggregate per-contig stats into bins
        for (const auto& c : contigs) {
            if (c.bin_id < 0) continue;
            auto map_it = internal_to_export_id.find(c.bin_id);
            if (map_it == internal_to_export_id.end()) continue;
            int export_id = map_it->second;

            auto it = damage_analysis_.per_contig_model.find(c.name);
            if (it == damage_analysis_.per_contig_model.end()) continue;

            const auto& model = it->second;
            for (size_t i = 0; i < model.stats_5p.size() && i < (size_t)max_pos; i++) {
                bin_n_5p[export_id][i] += model.stats_5p[i].n_opportunities;
                bin_k_5p[export_id][i] += model.stats_5p[i].n_mismatches;
            }
            for (size_t i = 0; i < model.stats_3p.size() && i < (size_t)max_pos; i++) {
                bin_n_3p[export_id][i] += model.stats_3p[i].n_opportunities;
                bin_k_3p[export_id][i] += model.stats_3p[i].n_mismatches;
            }
        }

        // Fit damage model for each bin using pooled counts
        std::unordered_map<int, DamageModelResult> bin_models;
        for (const auto& [export_id, n5p] : bin_n_5p) {
            bin_models[export_id] = fit_damage_model(
                n5p, bin_k_5p[export_id], bin_n_3p[export_id], bin_k_3p[export_id], 0.5);
        }

        // Write per-bin smiley data with model fit (sorted by bin ID)
        std::ofstream out(output_dir + "/smiley_per_bin.tsv");
        out << "bin\tposition\tobs_5p\tmodel_5p\tobs_3p\tmodel_3p\tn_opp_5p\tn_opp_3p\n";

        std::vector<int> sorted_ids;
        for (const auto& [export_id, _] : bin_n_5p) sorted_ids.push_back(export_id);
        std::sort(sorted_ids.begin(), sorted_ids.end());

        for (int export_id : sorted_ids) {
            const auto& n5p = bin_n_5p[export_id];
            const auto& model = bin_models[export_id];
            for (int i = 0; i < max_pos; i++) {
                double obs_5p = (n5p[i] > 0) ? (double)bin_k_5p[export_id][i] / n5p[i] : 0.0;
                double obs_3p = (bin_n_3p[export_id][i] > 0) ? (double)bin_k_3p[export_id][i] / bin_n_3p[export_id][i] : 0.0;
                double model_5p = model.curve_5p.amplitude * std::exp(-model.curve_5p.lambda * i) + model.curve_5p.baseline;
                double model_3p = model.curve_3p.amplitude * std::exp(-model.curve_3p.lambda * i) + model.curve_3p.baseline;

                out << "bin_" << export_id << "\t"
                    << (i + 1) << "\t"
                    << std::setprecision(6) << obs_5p << "\t"
                    << model_5p << "\t"
                    << obs_3p << "\t"
                    << model_3p << "\t"
                    << n5p[i] << "\t"
                    << bin_n_3p[export_id][i] << "\n";
            }
        }
        out.close();

        // Write per-bin damage model parameters (sorted by bin ID)
        std::ofstream model_out(output_dir + "/damage_per_bin.tsv");
        model_out << "bin\tp_ancient\tamplitude_5p\tlambda_5p\tbaseline_5p\tamplitude_3p\tlambda_3p\tbaseline_3p\n";
        for (int export_id : sorted_ids) {
            const auto& model = bin_models[export_id];
            model_out << "bin_" << export_id << "\t"
                      << std::setprecision(6) << model.p_ancient << "\t"
                      << model.curve_5p.amplitude << "\t"
                      << model.curve_5p.lambda << "\t"
                      << model.curve_5p.baseline << "\t"
                      << model.curve_3p.amplitude << "\t"
                      << model.curve_3p.lambda << "\t"
                      << model.curve_3p.baseline << "\n";
        }
        model_out.close();

        log_.info("Wrote per-bin smiley data: smiley_per_bin.tsv, damage_per_bin.tsv");
    }

    static std::vector<double> compute_kmer6_probs(const std::string& seq) {
        static const int K = 6;
        static const int N_CELLS = 1 << (2 * K);  // 4096
        std::vector<double> counts(N_CELLS, 0);
        std::map<char, int> base = {{'A',0},{'C',1},{'G',2},{'T',3}};

        for (size_t i = 0; i + K <= seq.size(); i++) {
            int idx = 0;
            bool valid = true;
            for (int j = 0; j < K; j++) {
                char c = std::toupper(seq[i+j]);
                auto it = base.find(c);
                if (it == base.end()) { valid = false; break; }
                idx = idx * 4 + it->second;
            }
            if (valid) counts[idx]++;
        }

        // Normalize to probability distribution
        double total = 0;
        for (double c : counts) total += c;
        if (total > 0) {
            for (double& c : counts) c /= total;
        } else {
            // Uniform distribution if no valid k-mers
            for (double& c : counts) c = 1.0 / N_CELLS;
        }
        return counts;
    }

    void extract_features() {
        log_.section("Feature extraction");
        log_.metric("use_fractal_features", use_fractal ? "yes" : "no");
        log_.metric("feature_dimensions", "");
        log_.metric("  TNF", TNF_DIM);
        log_.metric("  CGR", CGR_DIM);
        log_.metric("  DNA_physical", DNA_DIM);
        if (use_fractal) {
            log_.metric("  Fractal", FRACTAL_DIM);
        }

        auto start = std::chrono::steady_clock::now();

        log_.info("Extracting features" + std::string(use_fractal ? " (with DFA)" : ""));

        #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
        for (size_t i = 0; i < contigs.size(); i++) {
            auto& c = contigs[i];

            // TNF: L2 normalized (SAME as baseline)
            std::array<float, 136> tnf_arr;
            extract_tnf_l2(c.sequence, tnf_arr);
            c.tnf.assign(tnf_arr.begin(), tnf_arr.end());

            // CGR: 64x64 grid geometric features, L2 normalized (SAME as baseline)
            CGRGeometric cgr_geom = FractalFeatures::extract_cgr_geometric(c.sequence);
            c.cgr.assign(cgr_geom.normalized_vector.begin(), cgr_geom.normalized_vector.end());

            // DNA Physical: bendability, flexibility, meltability, GC (SAME as baseline)
            DNAPhysicalProperties dna_phys = FractalFeatures::extract_dna_physical(c.sequence);
            c.dna_physical.resize(DNA_DIM);
            c.dna_physical[0] = dna_phys.bendability;
            c.dna_physical[1] = dna_phys.flexibility;
            c.dna_physical[2] = dna_phys.meltability;

            // GC content
            int gc_count = 0, total_count = 0;
            for (char base : c.sequence) {
                char upper = std::toupper(base);
                if (upper == 'G' || upper == 'C') gc_count++;
                if (upper == 'A' || upper == 'C' || upper == 'G' || upper == 'T') total_count++;
            }
            c.gc = total_count > 0 ? (double)gc_count / total_count : 0.5;
            c.dna_physical[3] = c.gc;

            // Fractal DFA features (SAME as baseline)
            if (use_fractal) {
                CGRDFAFeatures cgr_dfa = FractalFeatures::extract_cgr_dfa_features(c.sequence);
                c.fractal.resize(FRACTAL_DIM);
                c.fractal[0] = cgr_dfa.hurst_x;
                c.fractal[1] = cgr_dfa.hurst_y;
                c.fractal[2] = cgr_dfa.anisotropy;
                c.fractal[3] = cgr_dfa.f_ratio;
                c.fractal[4] = cgr_dfa.local_hurst_var;
            }

            // Stable features for resonance merger (low CV, consistent within genome)
            // Ordered by CV (most stable first): 14 features total
            // bendability(0.0023), lz_complexity(0.0034), cgr_geom_variance(0.0053),
            // cgr_fractal_dim(0.0093), flexibility(0.0132), cgr_geom_mean(0.0144),
            // entropy_k6(0.0145), kmer_fractal_dim(0.0158), entropy_k2(0.0174),
            // entropy_k3(0.0180), entropy_k4(0.0189), cgr_entropy(0.0201),
            // entropy_k5(0.0213), meltability(0.0250)
            c.stable.resize(STABLE_DIM);

            // Complexity features (need for lz_complexity, kmer_fractal_dim, entropy)
            ComplexityFeatures complexity = FractalFeatures::extract_complexity_features(c.sequence);

            // CGR features (need for cgr_fractal_dim, cgr_entropy)
            CGRFeatures cgr_full = FractalFeatures::extract_cgr_features(c.sequence);

            // CGR geometric mean and variance from the 64x64 grid
            double cgr_geom_sum = 0.0;
            for (double v : cgr_geom.normalized_vector) cgr_geom_sum += v;
            double cgr_geom_mean_val = cgr_geom_sum / cgr_geom.normalized_vector.size();
            double cgr_var = 0.0;
            for (double v : cgr_geom.normalized_vector) {
                cgr_var += (v - cgr_geom_mean_val) * (v - cgr_geom_mean_val);
            }
            cgr_var /= cgr_geom.normalized_vector.size();

            // Assign stable features in CV order
            c.stable[0] = dna_phys.bendability;              // CV=0.0023
            c.stable[1] = complexity.lempel_ziv_complexity[3]; // CV=0.0034
            c.stable[2] = cgr_var;                           // CV=0.0053 cgr_geom_variance
            c.stable[3] = cgr_full.fractal_dimension;        // CV=0.0093
            c.stable[4] = dna_phys.flexibility;              // CV=0.0132
            c.stable[5] = cgr_geom_mean_val;                 // CV=0.0144 cgr_geom_mean
            c.stable[6] = complexity.entropy_scaling[4];     // CV=0.0145 entropy_k6
            c.stable[7] = complexity.kmer_fractal_dim;       // CV=0.0158
            c.stable[8] = complexity.entropy_scaling[0];     // CV=0.0174 entropy_k2
            c.stable[9] = complexity.entropy_scaling[1];     // CV=0.0180 entropy_k3
            c.stable[10] = complexity.entropy_scaling[2];    // CV=0.0189 entropy_k4
            c.stable[11] = cgr_full.entropy;                 // CV=0.0201
            c.stable[12] = complexity.entropy_scaling[3];    // CV=0.0213 entropy_k5
            c.stable[13] = dna_phys.meltability;             // CV=0.0250

            // Pre-compute per-contig modularity and CGR occupancy for quality gates
            if (c.sequence.length() >= 1000) {
                c.contig_modularity = compute_debruijn_modularity(c.sequence, 4);
                c.contig_cgr_occupancy = compute_cgr_occupancy(c.sequence, 10);
            }

            c.kmer6_probs = compute_kmer6_probs(c.sequence);

            // Compute dinucleotide relative abundances (Karlin signature)
            // ρ_XY = f_XY / (f_X * f_Y) where f = frequency, normalized by sequence length
            {
                std::array<int, 4> base_counts = {};  // A, C, G, T
                std::array<int, 16> dinuc_counts = {}; // AA, AC, AG, AT, CA, CC, ...

                auto base_idx = [](char b) -> int {
                    switch (std::toupper(b)) {
                        case 'A': return 0;
                        case 'C': return 1;
                        case 'G': return 2;
                        case 'T': return 3;
                        default: return -1;
                    }
                };

                for (size_t pos = 0; pos + 1 < c.sequence.length(); pos++) {
                    int b1 = base_idx(c.sequence[pos]);
                    int b2 = base_idx(c.sequence[pos + 1]);
                    if (b1 >= 0 && b2 >= 0) {
                        dinuc_counts[b1 * 4 + b2]++;
                    }
                }

                for (size_t pos = 0; pos < c.sequence.length(); pos++) {
                    int b = base_idx(c.sequence[pos]);
                    if (b >= 0) base_counts[b]++;
                }

                int total_bases = base_counts[0] + base_counts[1] + base_counts[2] + base_counts[3];
                int total_dinucs = total_bases > 0 ? total_bases - 1 : 0;

                // Compute relative abundances: ρ_XY = (f_XY / total_dinucs) / ((f_X / total_bases) * (f_Y / total_bases))
                // Simplifies to: ρ_XY = (dinuc_count * total_bases * total_bases) / (base_X * base_Y * total_dinucs)
                for (int x = 0; x < 4; x++) {
                    for (int y = 0; y < 4; y++) {
                        int idx = x * 4 + y;
                        if (base_counts[x] > 0 && base_counts[y] > 0 && total_dinucs > 0) {
                            double expected = (double)base_counts[x] * base_counts[y] / (total_bases * (double)total_dinucs);
                            double observed = (double)dinuc_counts[idx] / total_dinucs;
                            c.dinuc_rho[idx] = expected > 0 ? observed / expected : 1.0;
                        } else {
                            c.dinuc_rho[idx] = 1.0;  // Neutral if no data
                        }
                    }
                }
            }

            if ((i + 1) % 1000 == 0) {
                #pragma omp critical
                log_.detail(std::to_string(i + 1) + "/" + std::to_string(contigs.size()) + " contigs");
            }
        }

        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        log_.metric("extraction_time_seconds", elapsed);
        log_.metric("contigs_per_second", contigs.size() / elapsed);
        log_.metric("  Stable_for_merger", STABLE_DIM);

        log_.detail(std::to_string(contigs.size()) + "/" + std::to_string(contigs.size()) + " contigs done");
    }

    // Z-score normalize stable features across all contigs
    void normalize_stable_features() {
        std::vector<double> mean(STABLE_DIM, 0.0);
        std::vector<double> std_dev(STABLE_DIM, 0.0);
        size_t n = contigs.size();

        // Compute means
        for (const auto& c : contigs) {
            for (int j = 0; j < STABLE_DIM; j++) {
                mean[j] += c.stable[j];
            }
        }
        for (int j = 0; j < STABLE_DIM; j++) mean[j] /= n;

        // Compute stds
        for (const auto& c : contigs) {
            for (int j = 0; j < STABLE_DIM; j++) {
                double diff = c.stable[j] - mean[j];
                std_dev[j] += diff * diff;
            }
        }
        for (int j = 0; j < STABLE_DIM; j++) {
            std_dev[j] = std::sqrt(std_dev[j] / (n - 1));
            if (std_dev[j] < 1e-10) std_dev[j] = 1.0;
        }

        // Apply z-score
        for (auto& c : contigs) {
            for (int j = 0; j < STABLE_DIM; j++) {
                c.stable[j] = (c.stable[j] - mean[j]) / std_dev[j];
            }
        }
    }

    void run_kmeans_init(int k) {
        log_.section("K-means++ initialization");
        log_.metric("k", k);

        bins.clear();
        bins.resize(k);

        std::vector<Contig*> contig_ptrs;
        for (auto& c : contigs) contig_ptrs.push_back(&c);

        std::mt19937 rng;
        if (random_seed >= 0) {
            rng.seed(random_seed);
            log_.metric("random_seed", random_seed);
        } else {
            std::random_device rd;
            rng.seed(rd());
            log_.metric("random_seed", "random_device");
        }

        // Distance function (V10: Variance-weighted coverage)
        // Key insight from MetaBAT2: contigs with high coverage variance are less reliable
        // so their coverage differences should have less influence on distance
        auto distance = [this](const Contig& a, const Contig& b) -> double {
            double d = 0;
            // TNF
            if (!a.tnf.empty() && !b.tnf.empty()) {
                for (size_t i = 0; i < a.tnf.size(); i++) {
                    d += (a.tnf[i] - b.tnf[i]) * (a.tnf[i] - b.tnf[i]);
                }
            }
            // CGR
            if (!a.cgr.empty() && !b.cgr.empty()) {
                for (size_t i = 0; i < a.cgr.size(); i++) {
                    d += (a.cgr[i] - b.cgr[i]) * (a.cgr[i] - b.cgr[i]);
                }
            }
            // DNA Physical
            if (!a.dna_physical.empty() && !b.dna_physical.empty()) {
                for (size_t i = 0; i < a.dna_physical.size(); i++) {
                    d += (a.dna_physical[i] - b.dna_physical[i]) * (a.dna_physical[i] - b.dna_physical[i]);
                }
            }
            // Fractal
            if (use_fractal && !a.fractal.empty() && !b.fractal.empty()) {
                for (size_t i = 0; i < a.fractal.size(); i++) {
                    d += (a.fractal[i] - b.fractal[i]) * (a.fractal[i] - b.fractal[i]);
                }
            }
            // Coverage (V10: Variance-weighted log-transformed coverage)
            // Formula: weighted squared difference = (log_a - log_b)^2 / (var_a/mean_a^2 + var_b/mean_b^2 + epsilon)
            // This downweights coverage differences for contigs with high variance
            double log_cov_a = std::log(a.mean_coverage + 1);
            double log_cov_b = std::log(b.mean_coverage + 1);
            double cov_diff = log_cov_a - log_cov_b;
            
            // Convert variance to log-space: var(log(X)) ≈ var(X) / mean(X)^2
            double var_a_log = (a.var_coverage > 0 && a.mean_coverage > 0) ? 
                               a.var_coverage / (a.mean_coverage * a.mean_coverage) : 0.01;
            double var_b_log = (b.var_coverage > 0 && b.mean_coverage > 0) ? 
                               b.var_coverage / (b.mean_coverage * b.mean_coverage) : 0.01;
            
            // Combined uncertainty (pooled variance in log-space)
            // Add small epsilon to avoid division by zero
            double pooled_var = var_a_log + var_b_log + 0.01;
            
            // Mahalanobis-style: normalize by uncertainty
            // When both contigs have low variance, this behaves like regular Euclidean
            // When variance is high, the coverage distance contributes less
            d += cov_weight * (cov_diff * cov_diff) / pooled_var;
            
            return std::sqrt(d);
        };

        // K-means++ initialization
        std::uniform_int_distribution<int> first_dist(0, contig_ptrs.size() - 1);
        int first_center = first_dist(rng);
        bins[0].members.push_back(contig_ptrs[first_center]);
        contig_ptrs[first_center]->bin_id = 0;
        contig_ptrs[first_center]->is_seed = true;

        log_.metric("center_0", contig_ptrs[first_center]->name);

        std::vector<double> min_dist(contig_ptrs.size(), std::numeric_limits<double>::max());

        for (int c = 1; c < k; c++) {
            #pragma omp parallel for num_threads(n_threads)
            for (size_t i = 0; i < contig_ptrs.size(); i++) {
                if (contig_ptrs[i]->bin_id >= 0) continue;

                for (auto* seed : bins[c-1].members) {
                    double d = distance(*contig_ptrs[i], *seed);
                    min_dist[i] = std::min(min_dist[i], d);
                }
            }

            double total_dist = 0;
            for (size_t i = 0; i < contig_ptrs.size(); i++) {
                if (contig_ptrs[i]->bin_id < 0) {
                    total_dist += min_dist[i] * min_dist[i];
                }
            }

            std::uniform_real_distribution<double> sample_dist(0, total_dist);
            double r = sample_dist(rng);
            double cumsum = 0;
            int next_center = -1;

            for (size_t i = 0; i < contig_ptrs.size(); i++) {
                if (contig_ptrs[i]->bin_id < 0) {
                    cumsum += min_dist[i] * min_dist[i];
                    if (cumsum >= r) {
                        next_center = i;
                        break;
                    }
                }
            }

            if (next_center < 0) {
                for (size_t i = 0; i < contig_ptrs.size(); i++) {
                    if (contig_ptrs[i]->bin_id < 0) {
                        next_center = i;
                        break;
                    }
                }
            }

            if (next_center >= 0) {
                bins[c].members.push_back(contig_ptrs[next_center]);
                contig_ptrs[next_center]->bin_id = c;
                contig_ptrs[next_center]->is_seed = true;
                log_.metric("center_" + std::to_string(c), contig_ptrs[next_center]->name);
            }
        }

        // Initial assignment
        #pragma omp parallel for num_threads(n_threads)
        for (size_t i = 0; i < contig_ptrs.size(); i++) {
            if (contig_ptrs[i]->bin_id >= 0) continue;

            double best_dist = std::numeric_limits<double>::max();
            int best_bin = 0;

            for (int b = 0; b < k; b++) {
                if (!bins[b].members.empty()) {
                    double d = distance(*contig_ptrs[i], *bins[b].members[0]);
                    if (d < best_dist) {
                        best_dist = d;
                        best_bin = b;
                    }
                }
            }

            contig_ptrs[i]->bin_id = best_bin;
        }

        for (auto& bin : bins) bin.members.clear();
        for (auto* c : contig_ptrs) {
            if (c->bin_id >= 0 && c->bin_id < k) {
                bins[c->bin_id].members.push_back(c);
            }
        }

        for (auto& bin : bins) bin.fit();

        // STAGE TRACKING: Save K-means initial assignment
        for (auto* c : contig_ptrs) {
            c->kmeans_bin_id = c->bin_id;
        }
        log_.detail("Saved K-means initial assignments for " + std::to_string(contig_ptrs.size()) + " contigs");
    }

    // Coverage-Stratified Farthest-First Traversal
    // Stratify by coverage first, then run FFT on composition within each stratum
    // to ensure centers represent all abundance levels with compositional diversity
    void run_farthest_first_init(int k) {
        log_.section("Coverage-stratified initialization");
        log_.metric("k", k);

        bins.clear();
        bins.resize(k);

        // Only include contigs not marked as excluded
        std::vector<Contig*> contig_ptrs;
        for (auto& c : contigs) {
            if (c.bin_id != -2) contig_ptrs.push_back(&c);  // Skip excluded contigs
        }

        int n = contig_ptrs.size();
        if (n == 0) return;

        log_.detail("FFT init on " + std::to_string(n) + " contigs (excluded " +
                    std::to_string(contigs.size() - n) + " marked)");

        // Full distance (with variance-weighted coverage) for final assignment
        auto full_distance = [this](const Contig& a, const Contig& b) -> double {
            double d = 0;
            if (!a.tnf.empty() && !b.tnf.empty()) {
                for (size_t i = 0; i < a.tnf.size(); i++) {
                    d += (a.tnf[i] - b.tnf[i]) * (a.tnf[i] - b.tnf[i]);
                }
            }
            if (!a.cgr.empty() && !b.cgr.empty()) {
                for (size_t i = 0; i < a.cgr.size(); i++) {
                    d += (a.cgr[i] - b.cgr[i]) * (a.cgr[i] - b.cgr[i]);
                }
            }
            if (!a.dna_physical.empty() && !b.dna_physical.empty()) {
                for (size_t i = 0; i < a.dna_physical.size(); i++) {
                    d += (a.dna_physical[i] - b.dna_physical[i]) * (a.dna_physical[i] - b.dna_physical[i]);
                }
            }
            if (use_fractal && !a.fractal.empty() && !b.fractal.empty()) {
                for (size_t i = 0; i < a.fractal.size(); i++) {
                    d += (a.fractal[i] - b.fractal[i]) * (a.fractal[i] - b.fractal[i]);
                }
            }
            // Variance-weighted coverage
            double log_cov_a = std::log(a.mean_coverage + 1);
            double log_cov_b = std::log(b.mean_coverage + 1);
            double cov_diff = log_cov_a - log_cov_b;
            double var_a_log = (a.var_coverage > 0 && a.mean_coverage > 0) ?
                               a.var_coverage / (a.mean_coverage * a.mean_coverage) : 0.01;
            double var_b_log = (b.var_coverage > 0 && b.mean_coverage > 0) ?
                               b.var_coverage / (b.mean_coverage * b.mean_coverage) : 0.01;
            double pooled_var = var_a_log + var_b_log + 0.01;
            d += cov_weight * (cov_diff * cov_diff) / pooled_var;
            return std::sqrt(d);
        };

        // Step 1: V23 FIX - Compute stratum BOUNDARIES using only LONG contigs (>= 2500bp)
        // Short contigs have noisier coverage that shifts strata boundaries.
        // We compute boundaries from long contigs, then assign ALL contigs to those strata.
        const int MIN_LENGTH_FOR_STRATA = 2500;

        std::vector<std::pair<double, int>> long_cov_idx;  // Only long contigs for boundaries
        std::vector<std::pair<double, int>> all_cov_idx;   // All contigs for assignment

        for (int i = 0; i < n; i++) {
            double log_cov = std::log(contig_ptrs[i]->mean_coverage + 1);
            all_cov_idx.push_back({log_cov, i});
            if ((int)contig_ptrs[i]->length >= MIN_LENGTH_FOR_STRATA) {
                long_cov_idx.push_back({log_cov, i});
            }
        }
        std::sort(long_cov_idx.begin(), long_cov_idx.end());
        std::sort(all_cov_idx.begin(), all_cov_idx.end());

        // Fallback: if too few long contigs, use all
        bool use_long_for_boundaries = (long_cov_idx.size() >= 100);
        auto& boundary_cov_idx = use_long_for_boundaries ? long_cov_idx : all_cov_idx;

        // Step 2: Determine number of strata based on coverage range
        double min_log_cov = boundary_cov_idx.front().first;
        double max_log_cov = boundary_cov_idx.back().first;
        double cov_range = max_log_cov - min_log_cov;

        // Use sqrt(k) strata, but at least 3 and at most k/2
        int n_strata = std::max(3, std::min(k / 2, (int)std::sqrt(k) + 1));

        // If coverage range is small, use fewer strata
        if (cov_range < 1.0) n_strata = std::min(n_strata, 3);  // < 3-fold range
        if (cov_range < 0.5) n_strata = 1;  // < 1.6-fold range, single stratum

        log_.metric("n_strata", n_strata);
        log_.metric("log_coverage_range", cov_range);
        log_.metric("used_long_contigs_for_boundaries", use_long_for_boundaries ? 1 : 0);
        log_.metric("long_contigs_count", (int)long_cov_idx.size());

        // Step 3: Compute stratum boundaries from LONG contigs
        int n_boundary = boundary_cov_idx.size();
        int boundary_per_stratum = n_boundary / n_strata;
        std::vector<double> stratum_boundaries(n_strata + 1);
        stratum_boundaries[0] = -std::numeric_limits<double>::infinity();  // Min bound
        stratum_boundaries[n_strata] = std::numeric_limits<double>::infinity();  // Max bound

        for (int s = 1; s < n_strata; s++) {
            int boundary_idx = s * boundary_per_stratum;
            boundary_idx = std::min(boundary_idx, n_boundary - 1);
            // Boundary is midpoint between adjacent contigs
            stratum_boundaries[s] = (boundary_cov_idx[boundary_idx - 1].first +
                                      boundary_cov_idx[boundary_idx].first) / 2.0;
        }

        // Step 4: Assign ALL contigs to strata based on computed boundaries
        std::vector<std::vector<int>> strata(n_strata);
        for (const auto& [log_cov, idx] : all_cov_idx) {
            int stratum = 0;
            for (int s = 1; s <= n_strata; s++) {
                if (log_cov < stratum_boundaries[s]) {
                    stratum = s - 1;
                    break;
                }
            }
            strata[stratum].push_back(idx);
        }

        // Log stratum sizes and coverage ranges
        for (int s = 0; s < n_strata; s++) {
            if (!strata[s].empty()) {
                double min_c = contig_ptrs[strata[s].front()]->mean_coverage;
                double max_c = contig_ptrs[strata[s].back()]->mean_coverage;
                log_.metric("stratum_" + std::to_string(s) + "_size", (int)strata[s].size());
                log_.metric("stratum_" + std::to_string(s) + "_cov_range", 
                                  std::to_string(min_c) + "-" + std::to_string(max_c));
            }
        }

        // Step 4: Allocate centers to strata
        // Base allocation + distribute remainder to largest strata
        std::vector<int> centers_per_stratum(n_strata, k / n_strata);
        int remainder = k % n_strata;
        
        // Give extra centers to strata with more contigs (sorted by size descending)
        std::vector<std::pair<int, int>> stratum_sizes;
        for (int s = 0; s < n_strata; s++) {
            stratum_sizes.push_back({(int)strata[s].size(), s});
        }
        std::sort(stratum_sizes.rbegin(), stratum_sizes.rend());
        
        for (int r = 0; r < remainder; r++) {
            centers_per_stratum[stratum_sizes[r].second]++;
        }

        // Ensure each non-empty stratum gets at least 1 center
        for (int s = 0; s < n_strata; s++) {
            if (!strata[s].empty() && centers_per_stratum[s] == 0) {
                // Steal from stratum with most centers
                for (int t = 0; t < n_strata; t++) {
                    int donor = stratum_sizes[t].second;
                    if (centers_per_stratum[donor] > 1) {
                        centers_per_stratum[donor]--;
                        centers_per_stratum[s] = 1;
                        break;
                    }
                }
            }
        }

        // Step 5: Run FFT within each stratum using COMPOSITION-ONLY distance
        int center_idx = 0;
        std::vector<int> all_center_indices;

        for (int s = 0; s < n_strata; s++) {
            if (strata[s].empty() || centers_per_stratum[s] == 0) continue;

            std::vector<int>& stratum_indices = strata[s];
            int n_s = stratum_indices.size();
            int k_s = std::min(centers_per_stratum[s], n_s);

            log_.subsection("Stratum " + std::to_string(s) + " FFT");
            log_.metric("contigs", n_s);
            log_.metric("centers_to_select", k_s);

            // First center in stratum: LONGEST contig (deterministic, favors quality)
            int first_in_stratum = -1;
            size_t max_len = 0;
            for (int idx : stratum_indices) {
                if (contig_ptrs[idx]->length > max_len) {
                    max_len = contig_ptrs[idx]->length;
                    first_in_stratum = idx;
                }
            }

            if (first_in_stratum < 0) continue;

            // Mark as center
            bins[center_idx].members.push_back(contig_ptrs[first_in_stratum]);
            contig_ptrs[first_in_stratum]->bin_id = center_idx;
            contig_ptrs[first_in_stratum]->is_seed = true;
            contig_ptrs[first_in_stratum]->record_fate("fft_center", -1, center_idx,
                "longest_in_stratum_" + std::to_string(s), (double)max_len);
            all_center_indices.push_back(first_in_stratum);
            
            log_.metric("center_" + std::to_string(center_idx),
                              contig_ptrs[first_in_stratum]->name);
            log_.metric("center_" + std::to_string(center_idx) + "_coverage",
                              contig_ptrs[first_in_stratum]->mean_coverage);
            log_.metric("center_" + std::to_string(center_idx) + "_length",
                              (int)contig_ptrs[first_in_stratum]->length);
            log_.metric("center_" + std::to_string(center_idx) + "_stratum", s);
            log_.metric("center_" + std::to_string(center_idx) + "_reason", "longest_in_stratum");
            center_idx++;

            if (k_s == 1) continue;

            // Track min distance (composition + coverage) to selected centers within this stratum
            // Using full_distance to separate compositionally-similar genomes with different coverage
            std::vector<double> min_dist(n_s, std::numeric_limits<double>::max());

            // Initialize distances from first center
            for (int i = 0; i < n_s; i++) {
                int idx = stratum_indices[i];
                if (idx == first_in_stratum) {
                    min_dist[i] = 0;
                } else {
                    min_dist[i] = full_distance(*contig_ptrs[idx],
                                                *contig_ptrs[first_in_stratum]);
                }
            }

            // Select remaining centers using FFT on full distance (composition + coverage)
            std::vector<int> stratum_centers = {first_in_stratum};

            for (int c = 1; c < k_s && center_idx < k; c++) {
                // Find farthest point (in composition+coverage space) from all selected centers
                int farthest_local = -1;
                double max_min_dist = -1.0;
                size_t tie_break_len = 0;

                for (int i = 0; i < n_s; i++) {
                    int idx = stratum_indices[i];
                    if (contig_ptrs[idx]->bin_id >= 0) continue;  // Already a center

                    // Deterministic tie-breaker: longest contig
                    if (min_dist[i] > max_min_dist ||
                        (min_dist[i] == max_min_dist &&
                         contig_ptrs[idx]->length > tie_break_len)) {
                        max_min_dist = min_dist[i];
                        farthest_local = i;
                        tie_break_len = contig_ptrs[idx]->length;
                    }
                }

                if (farthest_local < 0) break;

                int next_center = stratum_indices[farthest_local];
                bins[center_idx].members.push_back(contig_ptrs[next_center]);
                contig_ptrs[next_center]->bin_id = center_idx;
                contig_ptrs[next_center]->is_seed = true;
                contig_ptrs[next_center]->record_fate("fft_center", -1, center_idx,
                    "fft_farthest_stratum_" + std::to_string(s), max_min_dist);
                all_center_indices.push_back(next_center);
                stratum_centers.push_back(next_center);

                log_.metric("center_" + std::to_string(center_idx),
                                  contig_ptrs[next_center]->name);
                log_.metric("center_" + std::to_string(center_idx) + "_coverage",
                                  contig_ptrs[next_center]->mean_coverage);
                log_.metric("center_" + std::to_string(center_idx) + "_length",
                                  (int)contig_ptrs[next_center]->length);
                log_.metric("center_" + std::to_string(center_idx) + "_stratum", s);
                log_.metric("center_" + std::to_string(center_idx) + "_comp_dist",
                                  max_min_dist);
                log_.metric("center_" + std::to_string(center_idx) + "_reason", "fft_farthest");
                center_idx++;

                // Update min distances using full_distance (composition + coverage)
                for (int i = 0; i < n_s; i++) {
                    int idx = stratum_indices[i];
                    if (contig_ptrs[idx]->bin_id >= 0) continue;
                    double d = full_distance(*contig_ptrs[idx], *contig_ptrs[next_center]);
                    min_dist[i] = std::min(min_dist[i], d);
                }
            }
        }

        log_.metric("total_centers_selected", center_idx);

        // Step 6: Initial assignment using FULL distance (with coverage)
        // This ensures contigs are assigned considering both composition AND coverage
        #pragma omp parallel for num_threads(n_threads)
        for (int i = 0; i < n; i++) {
            if (contig_ptrs[i]->bin_id >= 0) continue;  // Already a center

            double best_dist = std::numeric_limits<double>::max();
            int best_bin = 0;

            for (int b = 0; b < center_idx; b++) {
                if (!bins[b].members.empty()) {
                    double d = full_distance(*contig_ptrs[i], *bins[b].members[0]);
                    if (d < best_dist) {
                        best_dist = d;
                        best_bin = b;
                    }
                }
            }

            contig_ptrs[i]->bin_id = best_bin;
            contig_ptrs[i]->record_fate("fft_assign", -1, best_bin, "closest_center", best_dist);
        }

        // Rebuild bin membership
        for (auto& bin : bins) bin.members.clear();
        for (auto* c : contig_ptrs) {
            if (c->bin_id >= 0 && c->bin_id < center_idx) {
                bins[c->bin_id].members.push_back(c);
            }
        }

        // Remove empty bins
        bins.erase(std::remove_if(bins.begin(), bins.end(), 
                   [](const Bin& b) { return b.members.empty(); }), bins.end());

        for (auto& bin : bins) bin.fit();

        // STAGE TRACKING: Save initial assignment
        for (auto* c : contig_ptrs) {
            c->kmeans_bin_id = c->bin_id;
        }

        // TRACE: Log bin size distribution after FFT init
        log_.trace("");
        log_.trace("=== FFT INITIAL BIN SIZES (top 20) ===");
        std::vector<std::pair<size_t, int>> bin_sizes;
        for (size_t b = 0; b < bins.size(); b++) {
            size_t total_bp = 0;
            for (auto* c : bins[b].members) total_bp += c->length;
            bin_sizes.push_back({total_bp, (int)b});
        }
        std::sort(bin_sizes.rbegin(), bin_sizes.rend());
        for (size_t i = 0; i < std::min(size_t(20), bin_sizes.size()); i++) {
            std::ostringstream trace;
            trace << "  bin_" << bin_sizes[i].second << ": "
                  << bins[bin_sizes[i].second].members.size() << " contigs, "
                  << bin_sizes[i].first / 1000 << " kb";
            if (!bins[bin_sizes[i].second].members.empty()) {
                trace << ", seed=" << bins[bin_sizes[i].second].members[0]->name
                      << ", cov=" << std::fixed << std::setprecision(1)
                      << bins[bin_sizes[i].second].members[0]->mean_coverage << "x";
            }
            log_.trace(trace.str());
        }
        log_.trace("=== END FFT BIN SIZES ===");

        // COMPREHENSIVE CONTIG FATE LOG: Record every contig's initial assignment
        log_.trace("");
        log_.table_header("FFT_CONTIG_ASSIGNMENTS",
            {"contig", "bin_id", "length", "coverage", "gc", "is_seed", "distance_to_center"});
        for (auto* c : contig_ptrs) {
            if (c->bin_id < 0) continue;
            double dist_to_center = 0.0;
            if (!bins[c->bin_id].members.empty() && bins[c->bin_id].members[0] != c) {
                dist_to_center = full_distance(*c, *bins[c->bin_id].members[0]);
            }
            std::ostringstream row;
            row << c->name << "\t" << c->bin_id << "\t" << c->length << "\t"
                << std::fixed << std::setprecision(2) << c->mean_coverage << "\t"
                << std::setprecision(4) << c->gc << "\t"
                << (c->is_seed ? "Y" : "N") << "\t"
                << std::setprecision(4) << dist_to_center;
            log_.trace(row.str());
        }
        log_.trace("[END_TABLE:FFT_CONTIG_ASSIGNMENTS]");

        log_.detail("Coverage-Stratified FFT: " + std::to_string(n_strata) + " strata, "
                  + std::to_string(center_idx) + " centers, " + std::to_string(bins.size()) + " non-empty bins");
    }

    // K-medoids - uses actual data points as cluster centers (more robust to outliers)
    void run_kmedoids_init(int k) {
        log_.section("K-medoids initialization");
        log_.metric("k", k);

        bins.clear();
        bins.resize(k);

        std::vector<Contig*> contig_ptrs;
        for (auto& c : contigs) contig_ptrs.push_back(&c);

        std::mt19937 rng;
        if (random_seed >= 0) {
            rng.seed(random_seed);
            log_.metric("random_seed", random_seed);
        } else {
            std::random_device rd;
            rng.seed(rd());
            log_.metric("random_seed", "random_device");
        }

        // Distance function with variance-weighted coverage
        auto distance = [this](const Contig& a, const Contig& b) -> double {
            double d = 0;
            // TNF
            if (!a.tnf.empty() && !b.tnf.empty()) {
                for (size_t i = 0; i < a.tnf.size(); i++) {
                    d += (a.tnf[i] - b.tnf[i]) * (a.tnf[i] - b.tnf[i]);
                }
            }
            // CGR
            if (!a.cgr.empty() && !b.cgr.empty()) {
                for (size_t i = 0; i < a.cgr.size(); i++) {
                    d += (a.cgr[i] - b.cgr[i]) * (a.cgr[i] - b.cgr[i]);
                }
            }
            // DNA Physical
            if (!a.dna_physical.empty() && !b.dna_physical.empty()) {
                for (size_t i = 0; i < a.dna_physical.size(); i++) {
                    d += (a.dna_physical[i] - b.dna_physical[i]) * (a.dna_physical[i] - b.dna_physical[i]);
                }
            }
            // Fractal
            if (use_fractal && !a.fractal.empty() && !b.fractal.empty()) {
                for (size_t i = 0; i < a.fractal.size(); i++) {
                    d += (a.fractal[i] - b.fractal[i]) * (a.fractal[i] - b.fractal[i]);
                }
            }
            // Coverage (V10: Variance-weighted log-transformed coverage)
            double log_cov_a = std::log(a.mean_coverage + 1);
            double log_cov_b = std::log(b.mean_coverage + 1);
            double cov_diff = log_cov_a - log_cov_b;
            
            double var_a_log = (a.var_coverage > 0 && a.mean_coverage > 0) ? 
                               a.var_coverage / (a.mean_coverage * a.mean_coverage) : 0.01;
            double var_b_log = (b.var_coverage > 0 && b.mean_coverage > 0) ? 
                               b.var_coverage / (b.mean_coverage * b.mean_coverage) : 0.01;
            
            double pooled_var = var_a_log + var_b_log + 0.01;
            d += cov_weight * (cov_diff * cov_diff) / pooled_var;
            
            return std::sqrt(d);
        };

        int n = contig_ptrs.size();
        std::vector<int> medoid_indices(k, -1);
        std::vector<int> assignments(n, -1);

        // Initialize medoids using k-medoids++ (like k-means++ but for medoids)
        std::uniform_int_distribution<int> first_dist(0, n - 1);
        medoid_indices[0] = first_dist(rng);
        log_.metric("medoid_0", contig_ptrs[medoid_indices[0]]->name);

        // Select remaining medoids with probability proportional to squared distance
        for (int c = 1; c < k; c++) {
            std::vector<double> min_dist(n, std::numeric_limits<double>::max());

            #pragma omp parallel for num_threads(n_threads)
            for (int i = 0; i < n; i++) {
                for (int m = 0; m < c; m++) {
                    double d = distance(*contig_ptrs[i], *contig_ptrs[medoid_indices[m]]);
                    min_dist[i] = std::min(min_dist[i], d);
                }
            }

            double total_dist = 0;
            for (int i = 0; i < n; i++) {
                total_dist += min_dist[i] * min_dist[i];
            }

            std::uniform_real_distribution<double> sample_dist(0, total_dist);
            double r = sample_dist(rng);
            double cumsum = 0;

            for (int i = 0; i < n; i++) {
                cumsum += min_dist[i] * min_dist[i];
                if (cumsum >= r) {
                    medoid_indices[c] = i;
                    break;
                }
            }

            if (medoid_indices[c] < 0) medoid_indices[c] = 0;
            log_.metric("medoid_" + std::to_string(c), contig_ptrs[medoid_indices[c]]->name);
        }

        // PAM (Partitioning Around Medoids) algorithm
        int max_pam_iter = 10;
        for (int iter = 0; iter < max_pam_iter; iter++) {
            // Assign points to nearest medoid
            #pragma omp parallel for num_threads(n_threads)
            for (int i = 0; i < n; i++) {
                double best_dist = std::numeric_limits<double>::max();
                int best_medoid = 0;

                for (int m = 0; m < k; m++) {
                    double d = distance(*contig_ptrs[i], *contig_ptrs[medoid_indices[m]]);
                    if (d < best_dist) {
                        best_dist = d;
                        best_medoid = m;
                    }
                }
                assignments[i] = best_medoid;
            }

            // Try to find better medoids
            bool changed = false;
            for (int m = 0; m < k; m++) {
                // Get all points in this cluster
                std::vector<int> cluster_points;
                for (int i = 0; i < n; i++) {
                    if (assignments[i] == m) {
                        cluster_points.push_back(i);
                    }
                }

                if (cluster_points.empty()) continue;

                // Find the point that minimizes total distance to all other cluster points
                int best_medoid = medoid_indices[m];
                double best_cost = std::numeric_limits<double>::max();

                for (int candidate : cluster_points) {
                    double cost = 0;
                    for (int other : cluster_points) {
                        cost += distance(*contig_ptrs[candidate], *contig_ptrs[other]);
                    }
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_medoid = candidate;
                    }
                }

                if (best_medoid != medoid_indices[m]) {
                    medoid_indices[m] = best_medoid;
                    changed = true;
                }
            }

            log_.metric("pam_iteration_" + std::to_string(iter + 1) + "_changed", changed ? "yes" : "no");
            if (!changed) break;
        }

        // Final assignment and set up bins
        for (int i = 0; i < n; i++) {
            contig_ptrs[i]->bin_id = assignments[i];
        }

        for (auto& bin : bins) bin.members.clear();
        for (auto* c : contig_ptrs) {
            if (c->bin_id >= 0 && c->bin_id < k) {
                bins[c->bin_id].members.push_back(c);
            }
        }

        // Store medoid as first member (seed) of each bin
        for (int m = 0; m < k; m++) {
            contig_ptrs[medoid_indices[m]]->is_seed = true;
        }

        for (auto& bin : bins) bin.fit();

        // STAGE TRACKING: Save K-medoids initial assignment
        for (auto* c : contig_ptrs) {
            c->kmeans_bin_id = c->bin_id;  // Reuse kmeans_bin_id for consistency
        }
        log_.detail("Saved K-medoids initial assignments for " + std::to_string(contig_ptrs.size()) + " contigs");
    }

    void run_em(int max_iter = 20) {
        log_.section("EM clustering");
        log_.metric("max_iterations", max_iter);

        std::vector<Contig*> contig_ptrs;
        for (auto& c : contigs) contig_ptrs.push_back(&c);

        int k = bins.size();

        for (int iter = 0; iter < max_iter; iter++) {
            log_.subsection("Iteration " + std::to_string(iter + 1));

            int changes = 0;
            int rejections = 0;

            #pragma omp parallel for num_threads(n_threads) reduction(+:changes,rejections)
            for (size_t i = 0; i < contig_ptrs.size(); i++) {
                auto* c = contig_ptrs[i];
                int old_bin = c->bin_id;

                if (old_bin == -2) continue;  // Skip excluded contigs

                double best_prob = -std::numeric_limits<double>::max();
                int best_bin = old_bin;

                for (int b = 0; b < k; b++) {
                    if (bins[b].members.empty()) continue;

                    double prob = bins[b].log_probability(*c);

                    if (!bins[b].should_accept(*c)) {
                        rejections++;
                        continue;
                    }

                    if (prob > best_prob) {
                        best_prob = prob;
                        best_bin = b;
                    }
                }

                if (best_bin != old_bin) {
                    c->bin_id = best_bin;
                    c->bin_probability = best_prob;
                    changes++;
                }
            }

            for (auto& bin : bins) bin.members.clear();
            for (auto* c : contig_ptrs) {
                if (c->bin_id >= 0 && c->bin_id < k) {
                    bins[c->bin_id].members.push_back(c);
                }
            }

            for (auto& bin : bins) bin.fit();

            // V49: Coverage-coherence merge DISABLED - too aggressive, causes over-merging
            // The ARD feature weighting (V50) handles this more elegantly by
            // downweighting TNF when coverage is tight
            // int merges = merge_coverage_coherent_bins(0.8, 0.05, 5.0, 4000000);

            log_.metric("changes", changes);
            log_.metric("rejections", rejections);
            log_.detail("Iter " + std::to_string(iter + 1) + ": " + std::to_string(changes) + " changes");

            if (changes < (int)contig_ptrs.size() / 100) {
                log_.info("EM converged");
                break;
            }
        }

        for (auto* c : contig_ptrs) {
            c->em_bin_id = c->bin_id;
        }
    }

    // V49: Merge bins with overlapping coverage distributions
    // Prevents TNF from over-splitting organisms with uniform coverage
    // Parameters:
    //   min_cov_overlap: minimum IQR overlap fraction (0.8 = 80%)
    //   max_damage_diff: maximum damage amplitude difference (0.05 = 5%)
    //   min_size_ratio: only merge if larger/smaller > this ratio
    //   max_merged_bp: don't create bins larger than this
    int merge_coverage_coherent_bins(double min_cov_overlap = 0.8,
                                      double max_damage_diff = 0.05,
                                      double min_size_ratio = 5.0,
                                      size_t max_merged_bp = 4000000) {
        int merges = 0;

        // Compute coverage quartiles for each bin
        struct BinCovStats {
            double q1, median, q3;
            double damage_5p;
            size_t bp_total;
            bool valid;
        };
        std::vector<BinCovStats> stats(bins.size());

        for (size_t bi = 0; bi < bins.size(); bi++) {
            auto& bin = bins[bi];
            if (bin.members.size() < 5) {
                stats[bi].valid = false;
                continue;
            }

            std::vector<double> covs;
            double dmg_sum = 0;
            size_t bp = 0;
            int dmg_count = 0;

            for (auto* c : bin.members) {
                covs.push_back(c->mean_coverage);
                bp += c->length;
                if (c->has_damage && c->damage_rate_5p > 0) {
                    dmg_sum += c->damage_rate_5p;
                    dmg_count++;
                }
            }

            std::sort(covs.begin(), covs.end());
            size_t n = covs.size();
            stats[bi].q1 = covs[n / 4];
            stats[bi].median = covs[n / 2];
            stats[bi].q3 = covs[3 * n / 4];
            stats[bi].damage_5p = dmg_count > 0 ? dmg_sum / dmg_count : 0;
            stats[bi].bp_total = bp;
            stats[bi].valid = true;
        }

        // Find merge candidates
        std::vector<std::pair<int, int>> to_merge;  // (smaller, larger)

        for (size_t i = 0; i < bins.size(); i++) {
            if (!stats[i].valid) continue;

            for (size_t j = i + 1; j < bins.size(); j++) {
                if (!stats[j].valid) continue;

                // Check size ratio
                size_t larger = std::max(stats[i].bp_total, stats[j].bp_total);
                size_t smaller = std::min(stats[i].bp_total, stats[j].bp_total);
                if (smaller == 0) continue;
                double ratio = (double)larger / smaller;
                if (ratio < min_size_ratio) continue;

                // V49 fix: Don't create overly large bins
                if (stats[i].bp_total + stats[j].bp_total > max_merged_bp) continue;

                // Check coverage IQR overlap
                double overlap_start = std::max(stats[i].q1, stats[j].q1);
                double overlap_end = std::min(stats[i].q3, stats[j].q3);
                double overlap = std::max(0.0, overlap_end - overlap_start);

                double iqr_i = stats[i].q3 - stats[i].q1;
                double iqr_j = stats[j].q3 - stats[j].q1;
                double min_iqr = std::min(iqr_i, iqr_j);
                if (min_iqr < 1.0) min_iqr = 1.0;

                double overlap_frac = overlap / min_iqr;
                if (overlap_frac < min_cov_overlap) continue;

                // Check damage similarity
                double dmg_diff = std::abs(stats[i].damage_5p - stats[j].damage_5p);
                if (dmg_diff > max_damage_diff) continue;

                // This pair should merge
                int smaller_idx = stats[i].bp_total < stats[j].bp_total ? i : j;
                int larger_idx = stats[i].bp_total < stats[j].bp_total ? j : i;
                to_merge.push_back({smaller_idx, larger_idx});
            }
        }

        // Execute merges
        for (auto& [smaller_idx, larger_idx] : to_merge) {
            // Move all contigs from smaller to larger
            for (auto* c : bins[smaller_idx].members) {
                c->bin_id = larger_idx;
                bins[larger_idx].members.push_back(c);
            }
            bins[smaller_idx].members.clear();
            bins[larger_idx].fit();
            merges++;
        }

        return merges;
    }

    // Ensemble clustering: run FFT+EM multiple times with different seeds
    // and use co-occurrence consensus to get robust bin assignments
    void run_ensemble_clustering(int k, int n_runs) {
        log_.section("Ensemble clustering");
        log_.metric("k", k);
        log_.metric("n_runs", n_runs);

        std::vector<Contig*> contig_ptrs;
        for (auto& c : contigs) {
            if (c.bin_id != -2) contig_ptrs.push_back(&c);
        }
        int n = contig_ptrs.size();

        // Store co-occurrence counts: how many times each pair ended up in same bin
        // Use a map for sparse storage: (i*n + j) -> count
        std::map<size_t, int> cooccur;

        log_.detail("Running " + std::to_string(n_runs) + " ensemble iterations...");

        for (int run = 0; run < n_runs; run++) {
            // Reset bin assignments
            for (auto* c : contig_ptrs) {
                c->bin_id = -1;
            }
            bins.clear();
            bins.resize(k);

            // Use different seed for each run
            random_seed = 42 + run * 1000;

            // Run FFT initialization and EM
            run_farthest_first_init(k);
            run_em(20);

            // Record co-occurrences for this run
            // Build index map: bin_id -> list of contig indices
            std::map<int, std::vector<int>> bin_members;
            for (int i = 0; i < n; i++) {
                int bid = contig_ptrs[i]->bin_id;
                if (bid >= 0) {
                    bin_members[bid].push_back(i);
                }
            }

            // Count pairs in same bin
            for (auto& [bid, members] : bin_members) {
                for (size_t a = 0; a < members.size(); a++) {
                    for (size_t b = a + 1; b < members.size(); b++) {
                        int i = members[a], j = members[b];
                        if (i > j) std::swap(i, j);
                        size_t key = (size_t)i * n + j;
                        cooccur[key]++;
                    }
                }
            }

            log_.detail("  Run " + std::to_string(run + 1) + "/" + std::to_string(n_runs) + " complete");
        }

        // Build consensus assignments using Union-Find with threshold
        // Two contigs are in same bin if they co-occurred in >50% of runs
        int threshold = n_runs / 2;

        std::vector<int> parent(n);
        std::iota(parent.begin(), parent.end(), 0);

        std::function<int(int)> find = [&](int x) {
            if (parent[x] != x) parent[x] = find(parent[x]);
            return parent[x];
        };

        auto unite = [&](int x, int y) {
            int px = find(x), py = find(y);
            if (px != py) parent[px] = py;
        };

        // Unite contigs that co-occurred frequently
        for (auto& [key, count] : cooccur) {
            if (count > threshold) {
                int i = key / n;
                int j = key % n;
                unite(i, j);
            }
        }

        // Assign final bin IDs based on connected components
        std::map<int, int> root_to_bin;
        int next_bin = 0;
        for (int i = 0; i < n; i++) {
            int root = find(i);
            if (root_to_bin.find(root) == root_to_bin.end()) {
                root_to_bin[root] = next_bin++;
            }
            contig_ptrs[i]->bin_id = root_to_bin[root];
        }

        // Rebuild bins with consensus assignments
        bins.clear();
        bins.resize(next_bin);
        for (auto* c : contig_ptrs) {
            if (c->bin_id >= 0 && c->bin_id < next_bin) {
                bins[c->bin_id].members.push_back(c);
            }
        }
        for (auto& bin : bins) bin.fit();

        // Save k-means and EM IDs for consistency
        for (auto* c : contig_ptrs) {
            c->kmeans_bin_id = c->bin_id;
            c->em_bin_id = c->bin_id;
        }

        log_.metric("consensus_bins", next_bin);
        log_.info("Ensemble clustering: " + std::to_string(n_runs) + " runs -> " +
                  std::to_string(next_bin) + " consensus bins");
    }

    // Bayesian GMM-based resonance merger
    // Models pairwise distances as mixture of two Gaussians:
    // - Component 0: same-genome pairs (low μ, small σ)
    // - Component 1: different-genome pairs (higher μ, larger σ)
    // Merges bins when P(same-genome | distance) > threshold for both TNF and stable features
    // Size constraint is data-driven using 75th percentile
    // Transitive merging: uses Union-Find so A→B + B→C results in A,B,C merged together
    // Transitive threshold is more lenient since multiple weak links provide confidence
    void merge_bins_resonance(double posterior_threshold = 0.9, double cov_ratio_max = 2.0, double relaxed_threshold = 0.5) {
        log_.section("Bayesian bin merger");
        log_.metric("posterior_threshold", posterior_threshold);
        log_.metric("relaxed_threshold", relaxed_threshold);
        log_.metric("cov_ratio_max", cov_ratio_max);

        struct BinSig {
            std::vector<double> tnf_mean;     // Mean TNF (composition)
            std::vector<double> stable_mean;  // Mean stable features
            double cov_mean = 0.0;
            std::vector<double> damage_values; // All damage values for KS test
            int bin_idx = -1;
            size_t n_contigs = 0;
            size_t total_bp = 0;

            double cgr_occupancy = 0.0;
            std::vector<double> tnf_variances; // Per-dimension TNF variance for BIC
            double tnf_total_variance = 0.0;   // Sum of variances

            // Aggregated dinucleotide relative abundances (Karlin signature)
            std::array<double, 16> dinuc_rho_mean = {};  // Weighted mean by contig length
        };

        // Kolmogorov-Smirnov test for comparing two damage distributions
        // Returns p-value: high p-value means distributions are similar
        auto ks_test = [](std::vector<double> a, std::vector<double> b) -> double {
            if (a.empty() || b.empty()) return 0.0;

            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());

            int n1 = a.size(), n2 = b.size();
            int i = 0, j = 0;
            double d_max = 0.0;

            // Compute maximum difference between empirical CDFs
            while (i < n1 && j < n2) {
                double cdf1 = (double)(i + 1) / n1;
                double cdf2 = (double)(j + 1) / n2;

                if (a[i] <= b[j]) {
                    d_max = std::max(d_max, std::abs(cdf1 - (double)j / n2));
                    i++;
                } else {
                    d_max = std::max(d_max, std::abs((double)i / n1 - cdf2));
                    j++;
                }
            }
            // Handle remaining elements
            while (i < n1) {
                d_max = std::max(d_max, std::abs((double)(i + 1) / n1 - 1.0));
                i++;
            }
            while (j < n2) {
                d_max = std::max(d_max, std::abs(1.0 - (double)(j + 1) / n2));
                j++;
            }

            // Compute p-value using asymptotic approximation
            // For large samples: p ≈ 2 * exp(-2 * n_eff * D^2)
            // where n_eff = (n1 * n2) / (n1 + n2)
            double n_eff = (double)(n1 * n2) / (n1 + n2);
            double lambda = (std::sqrt(n_eff) + 0.12 + 0.11 / std::sqrt(n_eff)) * d_max;

            // Kolmogorov distribution approximation
            double p_value = 0.0;
            if (lambda > 0) {
                // Series expansion for Kolmogorov distribution
                for (int k = 1; k <= 100; k++) {
                    double term = std::pow(-1.0, k - 1) * std::exp(-2.0 * k * k * lambda * lambda);
                    p_value += term;
                    if (std::abs(term) < 1e-10) break;
                }
                p_value *= 2.0;
            } else {
                p_value = 1.0;
            }

            return std::max(0.0, std::min(1.0, p_value));
        };

        // Karlin distance (δ*) for comparing genomic signatures
        // δ*(f,g) = (1/16) * Σ |ρ_XY(f) - ρ_XY(g)|
        // Species-specific, stable across genome, ~0.03-0.05 within species, >0.08 between species
        auto karlin_distance = [](const std::array<double, 16>& rho_a,
                                  const std::array<double, 16>& rho_b) -> double {
            double sum = 0.0;
            for (int k = 0; k < 16; k++) {
                sum += std::abs(rho_a[k] - rho_b[k]);
            }
            return sum / 16.0;
        };

        // Thresholds for Karlin distance (from literature and OpenCode recommendation)
        const double KARLIN_HARD_VETO = 0.08;   // Different species - hard veto
        const double KARLIN_SOFT_PENALTY = 0.05; // Borderline - discourage merge

        // Build signatures for non-empty bins
        std::vector<BinSig> sigs;
        for (size_t bi = 0; bi < bins.size(); bi++) {
            if (bins[bi].members.empty()) continue;

            BinSig s;
            s.bin_idx = bi;
            s.n_contigs = bins[bi].members.size();

            s.tnf_mean.assign(TNF_DIM, 0.0);
            s.stable_mean.assign(STABLE_DIM, 0.0);
            int tnf_count = 0, stable_count = 0;

            for (auto* c : bins[bi].members) {
                s.total_bp += c->length;

                if (!c->tnf.empty()) {
                    tnf_count++;
                    for (int k = 0; k < TNF_DIM; k++) s.tnf_mean[k] += c->tnf[k];
                }
                if (!c->stable.empty()) {
                    stable_count++;
                    for (int k = 0; k < STABLE_DIM; k++) s.stable_mean[k] += c->stable[k];
                }
            }

            if (tnf_count > 0) for (double& v : s.tnf_mean) v /= tnf_count;
            if (stable_count > 0) for (double& v : s.stable_mean) v /= stable_count;

            // Mean coverage
            double cov_sum = 0.0;
            int cov_n = 0;
            for (auto* c : bins[bi].members) {
                if (c->mean_coverage > 0) {
                    cov_sum += c->mean_coverage;
                    cov_n++;
                }
            }
            s.cov_mean = cov_n > 0 ? cov_sum / cov_n : 0.0;

            // Collect all damage values for KS test
            for (auto* c : bins[bi].members) {
                if (c->has_damage && c->damage_rate_5p > 0) {
                    s.damage_values.push_back(c->damage_rate_5p);
                }
            }

            // Mean CGR occupancy for completeness gate
            double cgr_occ_sum = 0.0;
            int cgr_occ_count = 0;
            for (auto* c : bins[bi].members) {
                if (c->contig_cgr_occupancy > 0) {
                    cgr_occ_sum += c->contig_cgr_occupancy;
                    cgr_occ_count++;
                }
            }
            s.cgr_occupancy = cgr_occ_count > 0 ? cgr_occ_sum / cgr_occ_count : 0.0;

            // Per-dimension TNF variances for BIC
            s.tnf_variances.assign(TNF_DIM, 0.0);
            if (tnf_count >= 2) {
                for (int k = 0; k < TNF_DIM; k++) {
                    double ss = 0.0;
                    for (auto* c : bins[bi].members) {
                        if (!c->tnf.empty() && k < (int)c->tnf.size()) {
                            double diff = c->tnf[k] - s.tnf_mean[k];
                            ss += diff * diff;
                        }
                    }
                    s.tnf_variances[k] = ss / (tnf_count - 1);  // Bessel correction
                    s.tnf_total_variance += s.tnf_variances[k];
                }
            }

            // Aggregate dinucleotide relative abundances (length-weighted mean)
            // This is the Karlin genomic signature for the bin
            s.dinuc_rho_mean.fill(0.0);
            size_t dinuc_bp = 0;
            for (auto* c : bins[bi].members) {
                for (int k = 0; k < 16; k++) {
                    s.dinuc_rho_mean[k] += c->dinuc_rho[k] * c->length;
                }
                dinuc_bp += c->length;
            }
            if (dinuc_bp > 0) {
                for (int k = 0; k < 16; k++) {
                    s.dinuc_rho_mean[k] /= dinuc_bp;
                }
            }

            if (tnf_count > 0 && stable_count > 0 && s.cov_mean > 0) {
                sigs.push_back(std::move(s));
            }
        }

        int m = sigs.size();
        log_.metric("bins_with_signatures", m);

        // COMPREHENSIVE BIN SIGNATURE LOG
        log_.trace("");
        log_.table_header("BIN_SIGNATURES_PRE_MERGE",
            {"bin_id", "n_contigs", "total_bp", "mean_cov", "cgr_occ", "n_damage"});
        for (const auto& s : sigs) {
            std::ostringstream row;
            row << s.bin_idx << "\t" << s.n_contigs << "\t" << s.total_bp << "\t"
                << std::fixed << std::setprecision(2) << s.cov_mean << "\t"
                << std::setprecision(4) << s.cgr_occupancy << "\t"
                << s.damage_values.size();
            log_.trace(row.str());
        }
        log_.trace("[END_TABLE:BIN_SIGNATURES_PRE_MERGE]");

        if (m < 2) {
            log_.metric("bins_merged", 0);
            return;
        }

        // L2 distance helper
        auto l2_dist = [](const std::vector<double>& a, const std::vector<double>& b) {
            double d = 0;
            for (size_t i = 0; i < a.size() && i < b.size(); i++) {
                double diff = a[i] - b[i];
                d += diff * diff;
            }
            return std::sqrt(d);
        };

        // Compute ALL pairwise distances
        std::vector<double> all_tnf_dists;
        std::vector<double> all_stable_dists;
        std::vector<std::tuple<int, int, double, double>> pair_dists; // (i, j, tnf_dist, stable_dist)

        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                double d_tnf = l2_dist(sigs[i].tnf_mean, sigs[j].tnf_mean);
                double d_stable = l2_dist(sigs[i].stable_mean, sigs[j].stable_mean);

                all_tnf_dists.push_back(d_tnf);
                all_stable_dists.push_back(d_stable);
                pair_dists.push_back({i, j, d_tnf, d_stable});
            }
        }

        int n_pairs = all_tnf_dists.size();
        if (n_pairs < 10) {
            log_.metric("bins_merged", 0);
            log_.detail("Too few pairs for GMM fitting (" + std::to_string(n_pairs) + ")");
            return;
        }

        // ========== Gaussian Mixture Model with EM ==========
        // Fit 2-component GMM to distance distribution
        // Component 0: same-genome (lower distances)
        // Component 1: different-genome (higher distances)

        struct GMM {
            double pi0, pi1;     // Mixing weights
            double mu0, mu1;     // Means
            double sigma0, sigma1; // Standard deviations

            // Gaussian PDF
            static double gaussian_pdf(double x, double mu, double sigma) {
                if (sigma < 1e-10) sigma = 1e-10;
                double z = (x - mu) / sigma;
                return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2.0 * M_PI));
            }

            // Posterior P(component 0 | x)
            double posterior_same_genome(double x) const {
                double p0 = pi0 * gaussian_pdf(x, mu0, sigma0);
                double p1 = pi1 * gaussian_pdf(x, mu1, sigma1);
                double total = p0 + p1;
                if (total < 1e-300) return 0.5; // Avoid division by zero
                return p0 / total;
            }
        };

        // EM algorithm for GMM fitting
        auto fit_gmm = [&](const std::vector<double>& data, int max_iter = 100, double tol = 1e-6) -> GMM {
            GMM gmm;
            int n = data.size();

            // Initialize: use percentiles for initial estimates
            std::vector<double> sorted_data = data;
            std::sort(sorted_data.begin(), sorted_data.end());

            // Component 0: lower 20% (same-genome pairs are minority)
            // Component 1: upper 80% (different-genome pairs are majority)
            gmm.pi0 = 0.2;  // Prior: ~20% are same-genome pairs
            gmm.pi1 = 0.8;

            size_t p20_idx = static_cast<size_t>(0.2 * n);
            size_t p80_idx = static_cast<size_t>(0.8 * n);

            gmm.mu0 = sorted_data[p20_idx / 2];  // Median of lower 20%
            gmm.mu1 = sorted_data[(p80_idx + n) / 2];  // Median of upper 80%

            // Initial std devs
            double sum_sq0 = 0, sum_sq1 = 0;
            int cnt0 = 0, cnt1 = 0;
            for (double x : data) {
                if (x < gmm.mu0 + 0.5 * (gmm.mu1 - gmm.mu0)) {
                    sum_sq0 += (x - gmm.mu0) * (x - gmm.mu0);
                    cnt0++;
                } else {
                    sum_sq1 += (x - gmm.mu1) * (x - gmm.mu1);
                    cnt1++;
                }
            }
            gmm.sigma0 = cnt0 > 1 ? std::sqrt(sum_sq0 / cnt0) : 0.01;
            gmm.sigma1 = cnt1 > 1 ? std::sqrt(sum_sq1 / cnt1) : 0.1;

            // Ensure reasonable initial values
            if (gmm.sigma0 < 1e-6) gmm.sigma0 = 0.01;
            if (gmm.sigma1 < 1e-6) gmm.sigma1 = 0.1;

            // EM iterations
            std::vector<double> gamma(n);  // Responsibilities for component 0
            double prev_ll = -1e300;

            for (int iter = 0; iter < max_iter; iter++) {
                // E-step: compute responsibilities
                double ll = 0;
                for (int i = 0; i < n; i++) {
                    double p0 = gmm.pi0 * GMM::gaussian_pdf(data[i], gmm.mu0, gmm.sigma0);
                    double p1 = gmm.pi1 * GMM::gaussian_pdf(data[i], gmm.mu1, gmm.sigma1);
                    double total = p0 + p1;
                    if (total < 1e-300) total = 1e-300;
                    gamma[i] = p0 / total;
                    ll += std::log(total);
                }

                // Check convergence
                if (std::abs(ll - prev_ll) < tol) break;
                prev_ll = ll;

                // M-step: update parameters
                double sum_gamma = 0, sum_1_gamma = 0;
                double sum_gamma_x = 0, sum_1_gamma_x = 0;

                for (int i = 0; i < n; i++) {
                    sum_gamma += gamma[i];
                    sum_1_gamma += (1.0 - gamma[i]);
                    sum_gamma_x += gamma[i] * data[i];
                    sum_1_gamma_x += (1.0 - gamma[i]) * data[i];
                }

                // Update means
                if (sum_gamma > 1e-10) gmm.mu0 = sum_gamma_x / sum_gamma;
                if (sum_1_gamma > 1e-10) gmm.mu1 = sum_1_gamma_x / sum_1_gamma;

                // Update variances
                double sum_gamma_sq = 0, sum_1_gamma_sq = 0;
                for (int i = 0; i < n; i++) {
                    sum_gamma_sq += gamma[i] * (data[i] - gmm.mu0) * (data[i] - gmm.mu0);
                    sum_1_gamma_sq += (1.0 - gamma[i]) * (data[i] - gmm.mu1) * (data[i] - gmm.mu1);
                }

                if (sum_gamma > 1e-10) gmm.sigma0 = std::sqrt(sum_gamma_sq / sum_gamma);
                if (sum_1_gamma > 1e-10) gmm.sigma1 = std::sqrt(sum_1_gamma_sq / sum_1_gamma);

                // Ensure min variance
                if (gmm.sigma0 < 1e-6) gmm.sigma0 = 1e-6;
                if (gmm.sigma1 < 1e-6) gmm.sigma1 = 1e-6;

                // Update mixing weights
                gmm.pi0 = sum_gamma / n;
                gmm.pi1 = 1.0 - gmm.pi0;

                // Ensure component 0 has lower mean (same-genome)
                if (gmm.mu0 > gmm.mu1) {
                    std::swap(gmm.mu0, gmm.mu1);
                    std::swap(gmm.sigma0, gmm.sigma1);
                    std::swap(gmm.pi0, gmm.pi1);
                    for (int i = 0; i < n; i++) gamma[i] = 1.0 - gamma[i];
                }
            }

            return gmm;
        };

        // BIC (Bayesian Information Criterion) for merge decisions
        // Veto merge if BIC(merged) > BIC(separate) - data prefers two clusters
        auto compute_bic_for_merge = [](const BinSig& sig_i, const BinSig& sig_j,
                                        const std::vector<Contig*>& members_i,
                                        const std::vector<Contig*>& members_j) -> std::pair<double, double> {
            // Compute BIC for TNF features (the main composition signal)
            int n_i = members_i.size();
            int n_j = members_j.size();
            int n_merged = n_i + n_j;
            int d = TNF_DIM;  // Feature dimensions

            if (n_i < 3 || n_j < 3) return {0.0, 0.0};  // Not enough data

            // Compute separate log-likelihood for each cluster
            // Assuming diagonal covariance (independence between features)
            double ll_separate = 0.0;

            // Cluster i: sum of log-likelihoods under Gaussian(mean_i, var_i)
            for (int feat = 0; feat < d && feat < (int)sig_i.tnf_variances.size(); feat++) {
                double var_i = sig_i.tnf_variances[feat] + 1e-10;
                // LL = -n/2 * log(2*pi*var) - sum((x-mu)^2)/(2*var)
                // Since we use the sample mean/var, the second term simplifies
                ll_separate += -n_i * 0.5 * (std::log(2.0 * M_PI * var_i) + 1.0);
            }
            for (int feat = 0; feat < d && feat < (int)sig_j.tnf_variances.size(); feat++) {
                double var_j = sig_j.tnf_variances[feat] + 1e-10;
                ll_separate += -n_j * 0.5 * (std::log(2.0 * M_PI * var_j) + 1.0);
            }

            // Compute merged mean and variance
            std::vector<double> merged_mean(d, 0.0);
            for (int f = 0; f < d; f++) {
                merged_mean[f] = (sig_i.tnf_mean[f] * n_i + sig_j.tnf_mean[f] * n_j) / n_merged;
            }

            // Compute merged variance
            double ll_merged = 0.0;
            for (int feat = 0; feat < d; feat++) {
                // Pooled variance: use formula for combining sample variances
                double var_i = (feat < (int)sig_i.tnf_variances.size()) ? sig_i.tnf_variances[feat] : 0.01;
                double var_j = (feat < (int)sig_j.tnf_variances.size()) ? sig_j.tnf_variances[feat] : 0.01;
                double mean_i = sig_i.tnf_mean[feat];
                double mean_j = sig_j.tnf_mean[feat];

                // Combined variance includes both within-group and between-group variance
                double ss_i = (n_i - 1) * var_i + n_i * (mean_i - merged_mean[feat]) * (mean_i - merged_mean[feat]);
                double ss_j = (n_j - 1) * var_j + n_j * (mean_j - merged_mean[feat]) * (mean_j - merged_mean[feat]);
                double merged_var = (ss_i + ss_j) / (n_merged - 1) + 1e-10;

                ll_merged += -n_merged * 0.5 * (std::log(2.0 * M_PI * merged_var) + 1.0);
            }

            // BIC = k*ln(n) - 2*ln(L)
            // Separate model: 2*d params (2 means + 2 variances per feature, but we use diagonal)
            // Actually: 2 clusters * d features * 2 params (mean, var) = 4d, but diagonal so 2*d*2 = 4d
            // Simplified: k_separate = 2*d*2 = 4d (2 clusters, each with d means and d variances)
            // But for diagonal, it's actually 2*(d means + d variances) = 4d
            int k_separate = 4 * d;
            int k_merged = 2 * d;  // 1 cluster: d means + d variances

            double bic_separate = k_separate * std::log(n_merged) - 2.0 * ll_separate;
            double bic_merged = k_merged * std::log(n_merged) - 2.0 * ll_merged;

            return {bic_separate, bic_merged};
        };

        // Fit GMMs for TNF and stable distances
        GMM gmm_tnf = fit_gmm(all_tnf_dists);
        GMM gmm_stable = fit_gmm(all_stable_dists);

        log_.metric("gmm_tnf_mu0", gmm_tnf.mu0);
        log_.metric("gmm_tnf_mu1", gmm_tnf.mu1);
        log_.metric("gmm_tnf_sigma0", gmm_tnf.sigma0);
        log_.metric("gmm_tnf_sigma1", gmm_tnf.sigma1);
        log_.metric("gmm_tnf_pi0", gmm_tnf.pi0);

        log_.metric("gmm_stable_mu0", gmm_stable.mu0);
        log_.metric("gmm_stable_mu1", gmm_stable.mu1);
        log_.metric("gmm_stable_sigma0", gmm_stable.sigma0);
        log_.metric("gmm_stable_sigma1", gmm_stable.sigma1);
        log_.metric("gmm_stable_pi0", gmm_stable.pi0);

        if (verbose()) {
            std::ostringstream oss;
            oss << "GMM TNF: same-genome (μ=" << std::fixed << std::setprecision(4)
                << gmm_tnf.mu0 << ", σ=" << gmm_tnf.sigma0 << ", π=" << gmm_tnf.pi0
                << "), diff-genome (μ=" << gmm_tnf.mu1 << ", σ=" << gmm_tnf.sigma1 << ")";
            log_.detail(oss.str());
            oss.str("");
            oss << "GMM Stable: same-genome (μ=" << gmm_stable.mu0 << ", σ=" << gmm_stable.sigma0
                << ", π=" << gmm_stable.pi0 << "), diff-genome (μ=" << gmm_stable.mu1
                << ", σ=" << gmm_stable.sigma1 << ")";
            log_.detail(oss.str());
        }

        // ========== Data-driven size constraint ==========
        // Use 75th percentile as "large bin" threshold
        // Bins below this are considered incomplete fragments eligible for merging
        std::vector<size_t> bin_sizes;
        for (const auto& s : sigs) {
            bin_sizes.push_back(s.total_bp);
        }
        std::sort(bin_sizes.begin(), bin_sizes.end());
        size_t median_bin_size = bin_sizes[bin_sizes.size() / 2];
        size_t p75_bin_size = bin_sizes[bin_sizes.size() * 3 / 4];

        // "Small" bin threshold: use 75th percentile but ensure at least 500kb
        // This prevents merging two bins that are both likely complete genomes
        size_t small_bin_threshold = std::max(p75_bin_size, size_t(500000));

        log_.metric("median_bin_size_bp", (double)median_bin_size);
        log_.metric("p75_bin_size_bp", (double)p75_bin_size);
        log_.metric("small_bin_threshold_bp", (double)small_bin_threshold);

        // DIAGNOSTIC: Log all pairwise GMM posteriors to trace file (not stderr)
        // This helps understand why reference bins aren't merging
        log_.trace("");
        log_.trace("  === DIAGNOSTIC: Reference bin pairwise comparisons ===");
        log_.trace("  bin_i\tbin_j\tsize_i\tsize_j\tcov_i\tcov_j\tcov_ratio\td_tnf\td_stable\tP_tnf\tP_stable\tKS_p\tdmg_match");

        // Find indices by bin number in pair_dists
        for (auto& [i, j, d_tnf, d_stable] : pair_dists) {
            // Get original bin indices
            int bin_i = sigs[i].bin_idx;
            int bin_j = sigs[j].bin_idx;

            // Compute posteriors and other metrics
            double p_tnf = gmm_tnf.posterior_same_genome(d_tnf);
            double p_stable = gmm_stable.posterior_same_genome(d_stable);

            double cov_ratio = std::max(sigs[i].cov_mean, sigs[j].cov_mean) /
                              (std::min(sigs[i].cov_mean, sigs[j].cov_mean) + 1e-10);

            // KS test for damage
            double ks_pvalue = 0.0;
            bool damage_matches = false;
            if (sigs[i].damage_values.size() >= 3 && sigs[j].damage_values.size() >= 3) {
                ks_pvalue = ks_test(sigs[i].damage_values, sigs[j].damage_values);
                damage_matches = ks_pvalue > ks_pvalue_threshold;
            }

            // Output all pairs with P_tnf > 0.3 OR P_stable > 0.3 (potential merge candidates)
            if (p_tnf > 0.3 || p_stable > 0.3) {
                std::ostringstream diag;
                diag << "  " << bin_i << "\t" << bin_j
                     << "\t" << std::fixed << std::setprecision(0) << sigs[i].total_bp/1000.0 << "kb"
                     << "\t" << sigs[j].total_bp/1000.0 << "kb"
                     << "\t" << std::setprecision(1) << sigs[i].cov_mean
                     << "\t" << sigs[j].cov_mean
                     << "\t" << std::setprecision(2) << cov_ratio << "x"
                     << "\t" << std::setprecision(4) << d_tnf
                     << "\t" << d_stable
                     << "\t" << std::setprecision(3) << p_tnf
                     << "\t" << p_stable
                     << "\t" << std::setprecision(4) << ks_pvalue
                     << "\t" << (damage_matches ? "Y" : "N");
                log_.trace(diag.str());
            }
        }
        log_.trace("  === END DIAGNOSTIC ===");

        // Find merge candidates using Bayesian posteriors
        // Require: both TNF and stable posteriors > threshold, at least one bin small
        std::vector<std::pair<int, int>> merge_pairs;
        std::vector<std::tuple<int, int, double, double>> candidate_posteriors; // (i, j, p_tnf, p_stable)

        for (auto& [i, j, d_tnf, d_stable] : pair_dists) {
            // Compute posteriors P(same-genome | distance)
            double p_tnf = gmm_tnf.posterior_same_genome(d_tnf);
            double p_stable = gmm_stable.posterior_same_genome(d_stable);

            // Coverage ratio check
            double cov_ratio = std::max(sigs[i].cov_mean, sigs[j].cov_mean) /
                              (std::min(sigs[i].cov_mean, sigs[j].cov_mean) + 1e-10);

            // Use Kolmogorov-Smirnov test to compare damage distributions FIRST
            // This determines if we can relax BOTH coverage AND posterior thresholds
            bool damage_matches = false;
            double ks_pvalue = 0.0;

            // Need at least 3 damage values in each bin for meaningful KS test
            if (sigs[i].damage_values.size() >= 3 && sigs[j].damage_values.size() >= 3) {
                ks_pvalue = ks_test(sigs[i].damage_values, sigs[j].damage_values);
                // If p-value > ks_pvalue_threshold, distributions are statistically similar
                // This means we cannot reject the hypothesis that they come from same distribution
                damage_matches = ks_pvalue > kKSPvalueThreshold;
            }

            // Strict coverage constraint to prevent chimeric bins
            double effective_cov_ratio_max = kStrictCovRatioMax;
            double effective_posterior_threshold = posterior_threshold;

            // Note: damage match no longer relaxes thresholds - in ancient metagenomes,
            // all organisms have similar damage patterns
            (void)damage_matches;

            // Both posteriors must exceed effective threshold
            if (p_tnf < effective_posterior_threshold) continue;
            if (p_stable < effective_posterior_threshold) continue;

            // Size constraint: at least one bin must be small to prevent merging complete genomes
            bool i_small = sigs[i].total_bp < small_bin_threshold;
            bool j_small = sigs[j].total_bp < small_bin_threshold;
            if (!i_small && !j_small) continue;

            // Debug: Log why pairs are blocked for high-coverage bins
            if (cov_ratio > cov_ratio_max && p_tnf >= 0.5 && p_stable >= 0.5) {
                std::ostringstream dbg; dbg << "  DEBUG bin " << sigs[i].bin_idx << " <-> " << sigs[j].bin_idx
                          << ": cov_ratio=" << std::fixed << std::setprecision(2) << cov_ratio
                          << "x, n_dmg_i=" << sigs[i].damage_values.size()
                          << ", n_dmg_j=" << sigs[j].damage_values.size()
                          << ", KS_pval=" << std::setprecision(4) << ks_pvalue
                          << ", damage_matches=" << (damage_matches ? "Y" : "N")
                          << ", eff_cov_max=" << effective_cov_ratio_max
                          << ", eff_post=" << effective_posterior_threshold; log_.trace(dbg.str());
            }

            if (cov_ratio > effective_cov_ratio_max) continue;

            // CGR occupancy gate disabled - unreliable for contamination detection

            // BIC-based merge veto - block merge if data prefers two clusters
            // Compute BIC for separate vs merged models
            auto [bic_separate, bic_merged] = compute_bic_for_merge(
                sigs[i], sigs[j], bins[sigs[i].bin_idx].members, bins[sigs[j].bin_idx].members);

            // If BIC(merged) > BIC(separate), data prefers two clusters
            // Use a small margin to avoid blocking borderline cases
            if (bic_merged > bic_separate + kBICMargin && bic_separate > 0) {
                std::ostringstream dbg;
                dbg << "  BIC-VETO: bin " << sigs[i].bin_idx << " <-> " << sigs[j].bin_idx
                    << " (BIC_sep=" << std::fixed << std::setprecision(1) << bic_separate
                    << ", BIC_merged=" << bic_merged << ") - data prefers two clusters";
                log_.trace(dbg.str());
                continue;
            }

            // IQS tolerance band check - block merge if it degrades quality
            // IQS_after >= min(IQS_A, IQS_B) - k * σ_merged
            auto& bin_i = bins[sigs[i].bin_idx];
            auto& bin_j = bins[sigs[j].bin_idx];
            if (bin_i.iqs_stats.iqs > 0 && bin_j.iqs_stats.iqs > 0) {
                double iqs_merged = compute_merge_iqs(bin_i.iqs_stats, bin_j.iqs_stats);
                double iqs_min = std::min(bin_i.iqs_stats.iqs, bin_j.iqs_stats.iqs);
                double sigma_merged = std::sqrt(bin_i.iqs_stats.sigma_iqs * bin_i.iqs_stats.sigma_iqs +
                                                bin_j.iqs_stats.sigma_iqs * bin_j.iqs_stats.sigma_iqs);
                double k = IQS_K_CONSERVATIVE;  // 1.5σ tolerance

                if (iqs_merged < iqs_min - k * sigma_merged) {
                    std::ostringstream dbg;
                    dbg << "  V58 IQS-VETO: bin " << sigs[i].bin_idx << " <-> " << sigs[j].bin_idx
                        << " (IQS_i=" << std::fixed << std::setprecision(4) << bin_i.iqs_stats.iqs
                        << ", IQS_j=" << bin_j.iqs_stats.iqs
                        << ", IQS_merged=" << iqs_merged
                        << ", threshold=" << (iqs_min - k * sigma_merged) << ")";
                    log_.trace(dbg.str());
                    continue;
                }
            }

            // Karlin distance veto - block merge if genomic signatures differ significantly
            // δ* > 0.08 indicates different species (hard veto)
            // δ* > 0.05 indicates borderline (soft penalty via stricter posteriors)
            double karlin_dist = karlin_distance(sigs[i].dinuc_rho_mean, sigs[j].dinuc_rho_mean);

            if (karlin_dist > KARLIN_HARD_VETO) {
                std::ostringstream dbg;
                dbg << "  KARLIN-VETO: bin " << sigs[i].bin_idx << " <-> " << sigs[j].bin_idx
                    << " (δ*=" << std::fixed << std::setprecision(4) << karlin_dist
                    << " > " << KARLIN_HARD_VETO << ") - different genomic signatures";
                log_.trace(dbg.str());
                continue;
            }

            // Soft penalty: require higher posteriors if Karlin distance is borderline
            if (karlin_dist > KARLIN_SOFT_PENALTY) {
                double required_posterior = posterior_threshold + 0.05;  // Stricter threshold
                if (p_tnf < required_posterior || p_stable < required_posterior) {
                    std::ostringstream dbg;
                    dbg << "  KARLIN-SOFT: bin " << sigs[i].bin_idx << " <-> " << sigs[j].bin_idx
                        << " (δ*=" << std::fixed << std::setprecision(4) << karlin_dist
                        << ", P_tnf=" << p_tnf << ", P_stable=" << p_stable
                        << " < " << required_posterior << ") - borderline signatures, require higher confidence";
                    log_.trace(dbg.str());
                    continue;
                }
            }

            candidate_posteriors.push_back({i, j, p_tnf, p_stable});

            // Log damage-enabled merge (when KS test relaxed thresholds)
            if (damage_matches && (cov_ratio > cov_ratio_max || p_tnf < posterior_threshold || p_stable < posterior_threshold)) {
                std::ostringstream oss;
                oss << "KS-enabled merge: bin " << sigs[i].bin_idx << " <-> bin " << sigs[j].bin_idx
                    << " (KS_p=" << std::fixed << std::setprecision(4) << ks_pvalue
                    << ", cov=" << std::setprecision(2) << cov_ratio << "x"
                    << ", P_tnf=" << std::setprecision(3) << p_tnf
                    << ", P_stable=" << p_stable << ")";
                log_.detail(oss.str());
            }

            // Merge smaller into larger
            if (sigs[i].total_bp < sigs[j].total_bp) {
                merge_pairs.push_back({i, j});
            } else {
                merge_pairs.push_back({j, i});
            }
        }

        log_.metric("merge_candidates", (int)merge_pairs.size());

        // Log some candidates (verbose only)
        if (verbose()) {
            for (size_t k = 0; k < std::min(size_t(5), candidate_posteriors.size()); k++) {
                auto& [i, j, p_tnf, p_stable] = candidate_posteriors[k];
                size_t merged_kb = (sigs[i].total_bp + sigs[j].total_bp) / 1000;
                double cov_r = std::max(sigs[i].cov_mean, sigs[j].cov_mean) /
                              (std::min(sigs[i].cov_mean, sigs[j].cov_mean) + 1e-10);
                std::ostringstream oss;
                oss << "Candidate: bin " << sigs[i].bin_idx << " <-> bin " << sigs[j].bin_idx
                    << " (P_tnf=" << std::fixed << std::setprecision(3) << p_tnf
                    << ", P_stable=" << p_stable
                    << ", cov_ratio=" << cov_r << "x"
                    << ", merged=" << merged_kb << "kb)";
                log_.detail(oss.str());
            }
        }

        if (merge_pairs.empty()) {
            log_.metric("bins_merged", 0);
            return;
        }

        // Build direct remap (smaller -> larger, no transitivity)
        std::map<int, int> remap;
        std::set<int> already_merged;

        // Sort by combined posterior (highest confidence first)
        std::sort(merge_pairs.begin(), merge_pairs.end(),
            [&candidate_posteriors](const auto& a, const auto& b) {
                // Find posteriors for these pairs
                double conf_a = 0, conf_b = 0;
                for (auto& [i, j, p_tnf, p_stable] : candidate_posteriors) {
                    if ((i == a.first && j == a.second) || (j == a.first && i == a.second)) {
                        conf_a = p_tnf * p_stable;
                    }
                    if ((i == b.first && j == b.second) || (j == b.first && i == b.second)) {
                        conf_b = p_tnf * p_stable;
                    }
                }
                return conf_a > conf_b; // Highest confidence first
            });

        for (auto& [src_idx, dst_idx] : merge_pairs) {
            int src_bin = sigs[src_idx].bin_idx;
            int dst_bin = sigs[dst_idx].bin_idx;

            // Skip if source already merged elsewhere
            if (already_merged.count(src_bin)) continue;
            // Skip if destination was already merged into something else
            if (already_merged.count(dst_bin)) continue;

            remap[src_bin] = dst_bin;
            already_merged.insert(src_bin);
        }

        int merges = remap.size();
        log_.metric("bins_merged", merges);

        if (merges == 0) {
            std::cerr << "  No bins merged (no pairs passed threshold)\n";
            return;
        }

        // Reassign contigs
        for (auto& c : contigs) {
            auto it = remap.find(c.bin_id);
            if (it != remap.end()) {
                int old_bin = c.bin_id;
                c.bin_id = it->second;
                c.record_fate("merge", old_bin, it->second, "bin_merged");
            }
        }

        // Rebuild bin membership
        for (auto& b : bins) b.members.clear();
        for (auto& c : contigs) {
            if (c.bin_id >= 0 && c.bin_id < (int)bins.size()) {
                bins[c.bin_id].members.push_back(&c);
            }
        }
        for (auto& b : bins) b.fit();

        // COMPREHENSIVE POST-MERGE CONTIG FATE LOG
        log_.trace("");
        log_.table_header("POST_MERGE_CONTIG_ASSIGNMENTS",
            {"contig", "final_bin", "prev_bin", "was_remapped"});
        for (auto& c : contigs) {
            bool remapped = (remap.find(c.kmeans_bin_id) != remap.end());
            std::ostringstream row;
            row << c.name << "\t" << c.bin_id << "\t" << c.kmeans_bin_id << "\t"
                << (remapped ? "Y" : "N");
            log_.trace(row.str());
        }
        log_.trace("[END_TABLE:POST_MERGE_CONTIG_ASSIGNMENTS]");

        std::cerr << "  Merged " << merges << " bins using Bayesian GMM (P_threshold=" << posterior_threshold << ")\n";
    }

    // Targeted contamination refinement:
    // - Identify contaminated bins by De Bruijn modularity
    // - Relocate outlier contigs using TNF + coverage + variance scoring
    
    void rebuild_bins() {
        for (auto& b : bins) b.members.clear();
        for (auto& c : contigs) {
            if (c.bin_id >= 0 && c.bin_id < (int)bins.size()) {
                bins[c.bin_id].members.push_back(&c);
            }
        }
    }
    
    // Compute bin statistics for fast distance calculations
    struct BinStats {
        std::vector<double> tnf_centroid;
        double log_cov_mean = 0.0;
        double log_cov_std = 0.0;
        size_t total_bp = 0;
        int bin_idx = -1;
        double modularity = -1.0;
        double cgr_occupancy = 0.0;
        bool is_contaminated = false;
    };
    
    BinStats compute_bin_stats(int bin_idx) {
        BinStats stats;
        stats.bin_idx = bin_idx;
        auto& members = bins[bin_idx].members;
        
        if (members.size() < 2) return stats;
        
        // TNF centroid
        stats.tnf_centroid.assign(TNF_DIM, 0.0);
        int tnf_count = 0;
        for (auto* c : members) {
            if (!c->tnf.empty()) {
                for (int k = 0; k < TNF_DIM; k++) {
                    stats.tnf_centroid[k] += c->tnf[k];
                }
                tnf_count++;
            }
            stats.total_bp += c->length;
        }
        if (tnf_count > 0) {
            for (int k = 0; k < TNF_DIM; k++) {
                stats.tnf_centroid[k] /= tnf_count;
            }
        }
        
        // Coverage statistics (log-space)
        std::vector<double> log_covs;
        for (auto* c : members) {
            if (c->mean_coverage > 0) {
                log_covs.push_back(std::log(c->mean_coverage));
            }
        }
        if (!log_covs.empty()) {
            stats.log_cov_mean = std::accumulate(log_covs.begin(), log_covs.end(), 0.0) / log_covs.size();
            if (log_covs.size() > 1) {
                double var = 0.0;
                for (double lc : log_covs) {
                    var += (lc - stats.log_cov_mean) * (lc - stats.log_cov_mean);
                }
                stats.log_cov_std = std::sqrt(var / (log_covs.size() - 1));
            }
        }
        
        return stats;
    }
    
    // Compute intrinsic quality (CGR occupancy + modularity) - expensive but done once per bin
    void compute_bin_intrinsic_quality(BinStats& stats, const std::vector<Contig*>& members) {
        if (members.empty() || stats.total_bp < 100000) return;
        
        // Concatenate sequences
        std::string combined;
        combined.reserve(stats.total_bp);
        for (auto* c : members) {
            combined += c->sequence;
        }
        
        stats.cgr_occupancy = compute_cgr_occupancy(combined);
        stats.modularity = compute_debruijn_modularity(combined);
        stats.is_contaminated = (stats.modularity > -0.8);
    }
    
    // Compute distance from contig to bin centroid
    // Uses TNF + coverage, weighted by coverage variance (reliability)
    double compute_contig_to_bin_distance(const Contig* c, const BinStats& bin_stats, 
                                          double tnf_weight = 1.0, double cov_weight = 2.0) {
        if (bin_stats.tnf_centroid.empty()) return 1e10;
        
        // TNF distance (L2)
        double tnf_dist = 0.0;
        if (!c->tnf.empty()) {
            for (int k = 0; k < TNF_DIM; k++) {
                double diff = c->tnf[k] - bin_stats.tnf_centroid[k];
                tnf_dist += diff * diff;
            }
            tnf_dist = std::sqrt(tnf_dist);
        }
        
        // Coverage distance (log-space), weighted by reliability
        double cov_dist = 0.0;
        double reliability = 1.0;
        if (c->mean_coverage > 0) {
            double log_cov = std::log(c->mean_coverage);
            cov_dist = std::abs(log_cov - bin_stats.log_cov_mean);
            
            // Reliability: lower variance = higher reliability
            // reliability = 1 / (1 + var/mean²)
            if (c->var_coverage > 0 && c->mean_coverage > 0) {
                double cv_squared = c->var_coverage / (c->mean_coverage * c->mean_coverage);
                reliability = 1.0 / (1.0 + cv_squared);
            }
        }
        
        // Combined distance
        return tnf_weight * tnf_dist + cov_weight * cov_dist * reliability;
    }
    
    // Find best bin for a contig (excluding current bin)
    // Returns (best_bin_idx, distance_to_best, distance_to_current)
    std::tuple<int, double, double> find_best_bin_for_contig(
            const Contig* c, int current_bin_idx, 
            const std::vector<BinStats>& all_stats) {
        
        double current_dist = compute_contig_to_bin_distance(c, all_stats[current_bin_idx]);
        
        int best_bin = -1;
        double best_dist = 1e10;
        
        for (size_t i = 0; i < all_stats.size(); i++) {
            if ((int)i == current_bin_idx) continue;
            if (all_stats[i].tnf_centroid.empty()) continue;
            if (all_stats[i].total_bp < 100000) continue;  // Skip tiny bins
            
            double dist = compute_contig_to_bin_distance(c, all_stats[i]);
            if (dist < best_dist) {
                best_dist = dist;
                best_bin = i;
            }
        }
        
        return {best_bin, best_dist, current_dist};
    }
    
    // Main targeted refinement function
    void targeted_contamination_refinement(int max_rounds = 5) {
        log_.section("Targeted contamination refinement");
        
        std::cerr << "  Computing intrinsic quality for all bins...\n";
        
        // Step 1: Compute statistics and intrinsic quality for all bins
        std::vector<BinStats> all_stats(bins.size());
        int contaminated_count = 0;
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < bins.size(); i++) {
            if (bins[i].members.size() >= 2) {
                all_stats[i] = compute_bin_stats(i);
                compute_bin_intrinsic_quality(all_stats[i], bins[i].members);
            }
        }
        
        // Count contaminated bins
        for (const auto& stats : all_stats) {
            if (stats.is_contaminated) contaminated_count++;
        }
        
        std::cerr << "  Found " << contaminated_count << " potentially contaminated bins (modularity > -0.8)\n";
        log_.metric("contaminated_bins_initial", contaminated_count);
        
        if (contaminated_count == 0) {
            std::cerr << "  No contaminated bins found, skipping refinement\n";
            return;
        }
        
        // Step 2: Iterative refinement
        int total_relocated = 0;
        
        for (int round = 0; round < max_rounds; round++) {
            log_.detail("Refinement round " + std::to_string(round + 1));
            
            // Collect all contigs from contaminated bins with their outlier scores
            struct ContigRelocation {
                Contig* contig;
                int src_bin;
                int dst_bin;
                double src_dist;
                double dst_dist;
                double improvement;  // src_dist - dst_dist
            };
            std::vector<ContigRelocation> candidates;
            
            for (size_t bi = 0; bi < bins.size(); bi++) {
                if (!all_stats[bi].is_contaminated) continue;
                
                for (auto* c : bins[bi].members) {
                    auto [best_bin, best_dist, current_dist] = find_best_bin_for_contig(c, bi, all_stats);
                    
                    if (best_bin >= 0 && best_dist < current_dist * 0.8) {  // At least 20% improvement
                        ContigRelocation rel;
                        rel.contig = c;
                        rel.src_bin = bi;
                        rel.dst_bin = best_bin;
                        rel.src_dist = current_dist;
                        rel.dst_dist = best_dist;
                        rel.improvement = current_dist - best_dist;
                        candidates.push_back(rel);
                    }
                }
            }
            
            std::cerr << "  Found " << candidates.size() << " relocation candidates\n";
            
            if (candidates.empty()) {
                std::cerr << "  No more relocations possible, stopping\n";
                break;
            }
            
            // Sort by improvement (best first)
            std::sort(candidates.begin(), candidates.end(), 
                [](const ContigRelocation& a, const ContigRelocation& b) {
                    return a.improvement > b.improvement;
                });
            
            // Apply relocations (limit per round to avoid instability)
            int max_relocations = std::min((int)candidates.size(), 100);
            int relocated_this_round = 0;
            std::set<int> modified_bins;
            
            for (int i = 0; i < max_relocations; i++) {
                auto& rel = candidates[i];
                
                // Skip if source bin would become too small
                if (bins[rel.src_bin].members.size() <= 3) continue;
                
                // Perform relocation
                rel.contig->bin_id = rel.dst_bin;
                relocated_this_round++;
                modified_bins.insert(rel.src_bin);
                modified_bins.insert(rel.dst_bin);
                
                if (verbose() && relocated_this_round <= 5) {
                    std::cerr << "    Relocating " << rel.contig->name
                              << " from bin " << rel.src_bin << " to bin " << rel.dst_bin
                              << " (dist: " << std::fixed << std::setprecision(3)
                              << rel.src_dist << " -> " << rel.dst_dist << ")\n";
                }
            }

            if (verbose() && relocated_this_round > 5) {
                std::cerr << "    ... and " << (relocated_this_round - 5) << " more relocations\n";
            }

            total_relocated += relocated_this_round;
            
            // Rebuild bins and recompute stats for modified bins
            rebuild_bins();
            
            for (int bi : modified_bins) {
                if (bins[bi].members.size() >= 2) {
                    all_stats[bi] = compute_bin_stats(bi);
                    compute_bin_intrinsic_quality(all_stats[bi], bins[bi].members);
                } else {
                    all_stats[bi] = BinStats();
                }
            }
            
            // Recount contaminated bins
            contaminated_count = 0;
            for (const auto& stats : all_stats) {
                if (stats.is_contaminated) contaminated_count++;
            }
            if (verbose()) std::cerr << "  Contaminated bins remaining: " << contaminated_count << "\n";

            if (relocated_this_round == 0) break;
        }

        // Final rebuild and fit
        rebuild_bins();
        for (auto& b : bins) b.fit();

        log_.metric("total_contigs_relocated", total_relocated);
        log_.metric("contaminated_bins_final", contaminated_count);

        log_.info("Refinement: relocated " + std::to_string(total_relocated) + " contigs");
    }

    // Consensus-gated contamination removal: requires 2+ signals to agree before removal
    // Signals: coverage outlier, damage outlier, graph modularity outlier, composition outlier
    // - Review pile: flagged contigs can be re-recruited if they improve quality

    struct ContigOutlierFlags {
        Contig* contig = nullptr;
        bool coverage_outlier = false;
        bool damage_outlier = false;
        bool graph_outlier = false;
        bool composition_outlier = false;
        double coverage_dev = 0.0;
        double damage_dev = 0.0;
        double graph_dev = 0.0;
        double composition_dev = 0.0;
        int signal_count() const {
            return (coverage_outlier ? 1 : 0) + (damage_outlier ? 1 : 0) +
                   (graph_outlier ? 1 : 0) + (composition_outlier ? 1 : 0);
        }
    };

    // Helper: compute median of a vector
    double compute_median(std::vector<double>& v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }

    // Helper: compute MAD (Median Absolute Deviation)
    double compute_mad(const std::vector<double>& v, double median) {
        std::vector<double> abs_devs;
        for (double x : v) abs_devs.push_back(std::abs(x - median));
        std::sort(abs_devs.begin(), abs_devs.end());
        return abs_devs.empty() ? 0.0 : abs_devs[abs_devs.size() / 2] * 1.4826;  // Scale to std dev
    }

    // V48: Iterative MAD - removes outliers progressively to get clean statistics
    // This solves the "self-defeating MAD" problem where contaminants inflate MAD
    // and mask their own outlier status.
    // Returns: pair<median, mad> computed on clean subset
    std::pair<double, double> compute_iterative_mad(std::vector<double> values,
                                                     double initial_threshold = 3.5,
                                                     double final_threshold = 2.5,
                                                     int max_iterations = 5) {
        if (values.size() < 5) {
            double med = compute_median(values);
            return {med, compute_mad(values, med)};
        }

        double median = compute_median(values);
        double mad = compute_mad(values, median);
        if (mad < 1.0) mad = 1.0;

        // Iterative outlier removal with progressively tighter threshold
        for (int iter = 0; iter < max_iterations; iter++) {
            // Interpolate threshold: start loose, end tight
            double threshold = initial_threshold -
                (initial_threshold - final_threshold) * iter / (max_iterations - 1);

            // Find inliers
            std::vector<double> inliers;
            for (double v : values) {
                double z = std::abs(v - median) / mad;
                if (z <= threshold) {
                    inliers.push_back(v);
                }
            }

            // Need at least 50% retention to trust the statistics
            if (inliers.size() < values.size() / 2) break;
            if (inliers.size() == values.size()) break;  // No change

            // Update statistics with inliers only
            values = inliers;
            median = compute_median(values);
            mad = compute_mad(values, median);
            if (mad < 1.0) mad = 1.0;
        }

        return {median, mad};
    }

    // Helper: compute GC content
    double compute_gc(const std::string& seq) {
        int gc = 0;
        for (char c : seq) if (c == 'G' || c == 'C' || c == 'g' || c == 'c') gc++;
        return seq.empty() ? 0.5 : (double)gc / seq.length();
    }

    // V33: GC-corrected coverage - apply LOESS-like correction
    // Simple piecewise linear correction based on GC bins
    std::vector<double> gc_correct_coverage(const std::vector<Contig*>& members) {
        // Collect (GC, coverage) pairs
        std::vector<std::pair<double, double>> gc_cov;
        for (auto* c : members) {
            if (c->mean_coverage > 0) {
                gc_cov.push_back({c->gc, c->mean_coverage});
            }
        }
        if (gc_cov.size() < 10) {
            // Not enough data for correction, return raw
            std::vector<double> raw;
            for (auto* c : members) raw.push_back(c->mean_coverage);
            return raw;
        }

        // Bin by GC content (5 bins: 0-0.3, 0.3-0.4, 0.4-0.5, 0.5-0.6, 0.6+)
        std::vector<std::vector<double>> gc_bins(5);
        for (auto& [gc, cov] : gc_cov) {
            int bin_idx = std::min(4, std::max(0, (int)((gc - 0.2) / 0.1)));
            gc_bins[bin_idx].push_back(cov);
        }

        // Compute median coverage per GC bin
        std::vector<double> bin_medians(5, 0.0);
        double overall_median = 0.0;
        std::vector<double> all_covs;
        for (auto& [gc, cov] : gc_cov) all_covs.push_back(cov);
        overall_median = compute_median(all_covs);

        for (int i = 0; i < 5; i++) {
            if (!gc_bins[i].empty()) {
                bin_medians[i] = compute_median(gc_bins[i]);
            } else {
                bin_medians[i] = overall_median;
            }
        }

        // Apply correction: cov_corrected = cov * (overall_median / bin_median)
        std::vector<double> corrected;
        for (auto* c : members) {
            if (c->mean_coverage <= 0) {
                corrected.push_back(0.0);
                continue;
            }
            int bin_idx = std::min(4, std::max(0, (int)((c->gc - 0.2) / 0.1)));
            double correction = bin_medians[bin_idx] > 0 ? overall_median / bin_medians[bin_idx] : 1.0;
            corrected.push_back(c->mean_coverage * correction);
        }
        return corrected;
    }

    // V33: Length-weighted MAD threshold
    // Short contigs need higher deviation to be flagged (more variance expected)
    double length_weighted_threshold(double base_threshold, size_t contig_len, size_t median_len) {
        // Scale: threshold *= sqrt(median_len / contig_len)
        // Short contigs (< median): higher threshold (harder to flag)
        // Long contigs (> median): lower threshold (easier to flag)
        double ratio = (double)median_len / std::max((size_t)1000, contig_len);
        double scale = std::sqrt(std::max(0.5, std::min(2.0, ratio)));
        return base_threshold * scale;
    }

    // V33: Compute bin quality score for optimization
    // Q = cgr_occupancy - 5 * modularity_excess
    // Where modularity_excess = max(0, modularity + 0.9)
    double compute_quality_score(const std::vector<Contig*>& members) {
        if (members.size() < 3) return 0.0;

        // Concatenate sequences for intrinsic quality
        std::string combined;
        for (auto* c : members) combined += c->sequence;
        if (combined.length() < 50000) return 0.0;

        double cgr_occ = compute_cgr_occupancy(combined, 10);
        double modularity = compute_debruijn_modularity(combined, 4);

        // modularity_excess: how much worse than "clean" (-0.9)
        double mod_excess = std::max(0.0, modularity + 0.9);

        return cgr_occ - 5.0 * mod_excess;
    }

    void consensus_gated_contamination_removal(int max_iterations = 10) {
        log_.section("Consensus-gated contamination removal");

        // Thresholds
        const double GRAPH_MOD_OUTLIER = -0.7;   // Flagged if modularity > -0.7
        const double COMP_STD_THRESHOLD = 3.0;   // TNF distance in std devs
        const double CGR_OCC_FLOOR = 0.58;       // Static floor (~90% completeness)
        const double CGR_OCC_MAX_DROP = 0.03;    // V34: Ratchet - max allowed drop from initial
        const double TNF_GRAVITY_THRESHOLD = 0.15; // V34: Force-recruit if TNF dist < this
        const size_t LONG_CONTIG_THRESHOLD = 10000; // V34: Long contigs need 3+ signals

        int total_flagged = 0;
        int total_removed = 0;
        int total_recruited = 0;
        int total_tnf_recruited = 0;  // V34: Track TNF gravity recruits

        for (size_t bi = 0; bi < bins.size(); bi++) {
            if (bins[bi].members.size() < 5) continue;

            // === Phase 1: Compute bin-level statistics ===

            // GC-corrected coverage
            std::vector<double> gc_corrected_cov = gc_correct_coverage(bins[bi].members);

            // Coverage median and MAD - V48: Use iterative MAD to handle contaminated bins
            // This prevents contaminants from inflating MAD and masking their outlier status
            std::vector<double> coverages = gc_corrected_cov;
            auto [cov_median, cov_mad] = compute_iterative_mad(coverages, 3.5, 2.5, 5);
            if (cov_mad < 1.0) cov_mad = 1.0;

            // Damage median and MAD
            std::vector<double> damages;
            for (auto* c : bins[bi].members) {
                if (c->has_damage && c->damage_rate_5p > 0) {
                    damages.push_back(c->damage_rate_5p);
                }
            }
            double dmg_median = damages.size() >= 5 ? compute_median(damages) : 0.0;
            double dmg_mad = damages.size() >= 5 ? compute_mad(damages, dmg_median) : 0.0;
            if (dmg_mad < 0.001) dmg_mad = 0.001;

            // Graph modularity median and MAD
            std::vector<double> modularities;
            for (auto* c : bins[bi].members) {
                if (c->contig_modularity > -2.0) {
                    modularities.push_back(c->contig_modularity);
                }
            }
            double mod_median = modularities.size() >= 5 ? compute_median(modularities) : -1.0;

            // TNF centroid and std dev
            std::vector<double> tnf_centroid(TNF_DIM, 0.0);
            for (auto* c : bins[bi].members) {
                for (int d = 0; d < TNF_DIM && d < (int)c->tnf.size(); d++) {
                    tnf_centroid[d] += c->tnf[d];
                }
            }
            for (int d = 0; d < TNF_DIM; d++) tnf_centroid[d] /= bins[bi].members.size();

            std::vector<double> tnf_dists;
            for (auto* c : bins[bi].members) {
                double dist = 0.0;
                for (int d = 0; d < TNF_DIM && d < (int)c->tnf.size(); d++) {
                    double diff = c->tnf[d] - tnf_centroid[d];
                    dist += diff * diff;
                }
                tnf_dists.push_back(std::sqrt(dist));
            }
            double tnf_median = compute_median(tnf_dists);
            double tnf_mad = compute_mad(tnf_dists, tnf_median);
            if (tnf_mad < 0.01) tnf_mad = 0.01;

            // Median contig length for length-weighted threshold
            std::vector<double> lengths;
            for (auto* c : bins[bi].members) lengths.push_back(c->length);
            size_t median_len = (size_t)compute_median(lengths);

            // Initial CGR occupancy and quality score
            double initial_cgr_occ = bins[bi].current_cgr_occupancy;
            if (initial_cgr_occ < 0.1) {
                // Recompute if not available
                std::string combined;
                for (auto* c : bins[bi].members) combined += c->sequence;
                initial_cgr_occ = combined.length() >= 10000 ? compute_cgr_occupancy(combined, 10) : 0.0;
            }
            double initial_quality = compute_quality_score(bins[bi].members);

            // === Phase 2: Flag contigs with outlier signals ===
            std::vector<ContigOutlierFlags> all_flags;
            std::vector<Contig*> review_pile;  // Flagged but not removed

            for (size_t ci = 0; ci < bins[bi].members.size(); ci++) {
                auto* c = bins[bi].members[ci];
                ContigOutlierFlags flags;
                flags.contig = c;

                // Length-weighted threshold
                double len_cov_thresh = length_weighted_threshold(kCovMADThreshold, c->length, median_len);
                double len_dmg_thresh = length_weighted_threshold(kDmgMADThreshold, c->length, median_len);
                double len_comp_thresh = length_weighted_threshold(COMP_STD_THRESHOLD, c->length, median_len);

                // Coverage signal (GC-corrected)
                if (gc_corrected_cov[ci] > 0) {
                    flags.coverage_dev = std::abs(gc_corrected_cov[ci] - cov_median) / cov_mad;
                    flags.coverage_outlier = (flags.coverage_dev > len_cov_thresh);
                }

                // Damage signal
                if (c->has_damage && c->damage_rate_5p > 0 && damages.size() >= 5) {
                    flags.damage_dev = std::abs(c->damage_rate_5p - dmg_median) / dmg_mad;
                    flags.damage_outlier = (flags.damage_dev > len_dmg_thresh);
                }

                // Graph modularity signal
                // Only flag if contig modularity is significantly higher (worse) than bin
                if (c->contig_modularity > -2.0 && mod_median < -0.85) {
                    flags.graph_dev = c->contig_modularity - mod_median;
                    flags.graph_outlier = (c->contig_modularity > GRAPH_MOD_OUTLIER);
                }

                // Composition signal (TNF distance)
                flags.composition_dev = tnf_dists[ci] / tnf_mad;
                flags.composition_outlier = (flags.composition_dev > len_comp_thresh);

                all_flags.push_back(flags);
            }

            // === Phase 3: Iterative consensus-gated removal (V34 enhanced) ===
            int bin_removed = 0;
            int bin_flagged = 0;

            // V34: Store ORIGINAL initial_cgr_occ for ratchet (don't update during iteration)
            const double original_cgr_occ = initial_cgr_occ;

            // V34: Estimate completeness for HQ cliff hysteresis
            // cgr_occ of 0.58 ≈ 90%, 0.55 ≈ 88%, 0.61 ≈ 92%
            auto estimate_completeness = [](double cgr_occ) {
                return (cgr_occ - 0.33) / (0.62 - 0.33) * 100.0;  // Map to 0-100%
            };
            double est_completeness = estimate_completeness(initial_cgr_occ);

            // V34: Determine minimum signal count based on completeness band
            // HQ cliff: if 88-92% completeness, require 3+ signals
            int min_signals_required = 2;
            if (est_completeness >= 88.0 && est_completeness <= 92.0) {
                min_signals_required = 3;
                log_.trace("  V34 HQ-CLIFF: bin " + std::to_string(bi) +
                    " est_comp=" + std::to_string(est_completeness) + "% - requiring 3+ signals");
            }

            for (int iter = 0; iter < max_iterations; iter++) {
                if (bins[bi].members.size() < 3) break;

                // Find contig with highest signal count (must meet minimum)
                int worst_idx = -1;
                int worst_signal_count = min_signals_required - 1;  // Need at least min_signals
                double worst_total_dev = 0.0;

                for (size_t fi = 0; fi < all_flags.size(); fi++) {
                    auto& f = all_flags[fi];
                    if (f.contig == nullptr || f.contig->bin_id == -1) continue;

                    int sc = f.signal_count();
                    double total_dev = f.coverage_dev + f.damage_dev + f.graph_dev + f.composition_dev;

                    // V34: Long contig veto - contigs >10kb always need 3+ signals
                    int effective_min = min_signals_required;
                    if (f.contig->length >= LONG_CONTIG_THRESHOLD) {
                        effective_min = std::max(3, min_signals_required);
                    }

                    if (sc >= effective_min && (sc > worst_signal_count ||
                        (sc == worst_signal_count && total_dev > worst_total_dev))) {
                        worst_signal_count = sc;
                        worst_total_dev = total_dev;
                        worst_idx = fi;
                    }
                }

                if (worst_idx < 0) break;  // No consensus outliers meeting threshold

                auto* worst_contig = all_flags[worst_idx].contig;

                // === CGR Occupancy Guardrail (V34: Enhanced with ratchet) ===
                // Check if removing this contig would drop completeness below threshold
                std::vector<Contig*> test_members;
                for (auto* c : bins[bi].members) {
                    if (c != worst_contig && c->bin_id != -1) test_members.push_back(c);
                }
                double test_quality = compute_quality_score(test_members);

                // Also check CGR occupancy directly
                std::string test_combined;
                for (auto* c : test_members) test_combined += c->sequence;
                double test_cgr_occ = test_combined.length() >= 10000 ?
                    compute_cgr_occupancy(test_combined, 10) : initial_cgr_occ;

                // V34: Ratchet guardrail - don't drop more than MAX_DROP from ORIGINAL
                bool ratchet_blocked = (original_cgr_occ - test_cgr_occ > CGR_OCC_MAX_DROP);

                // Guardrail: don't remove if it would drop below floor OR violate ratchet
                if ((test_cgr_occ < CGR_OCC_FLOOR && initial_cgr_occ >= CGR_OCC_FLOOR) || ratchet_blocked) {
                    std::string reason = ratchet_blocked ? "RATCHET" : "FLOOR";
                    log_.trace("  V34 " + reason + ": bin " + std::to_string(bi) +
                        " would drop cgr_occ from " + std::to_string(original_cgr_occ) +
                        " to " + std::to_string(test_cgr_occ) + " - blocking removal");
                    // Move to review pile instead
                    review_pile.push_back(worst_contig);
                    all_flags[worst_idx].contig = nullptr;  // Mark as processed
                    bin_flagged++;
                    total_flagged++;
                    continue;
                }

                // Quality score optimization: don't remove if Q doesn't improve
                if (test_quality < initial_quality - 0.01) {
                    log_.trace("  V33 QUALITY-STOP: bin " + std::to_string(bi) +
                        " removal would decrease Q from " + std::to_string(initial_quality) +
                        " to " + std::to_string(test_quality));
                    review_pile.push_back(worst_contig);
                    all_flags[worst_idx].contig = nullptr;
                    bin_flagged++;
                    total_flagged++;
                    continue;
                }

                // Remove the contig
                worst_contig->bin_id = -1;
                bins[bi].members.erase(
                    std::remove(bins[bi].members.begin(), bins[bi].members.end(), worst_contig),
                    bins[bi].members.end());
                all_flags[worst_idx].contig = nullptr;

                bin_removed++;
                total_removed++;

                // Update initial quality for next iteration
                initial_cgr_occ = test_cgr_occ;
                initial_quality = test_quality;

                std::ostringstream log_msg;
                log_msg << "  V33 REMOVE: bin " << bi << " contig " << worst_contig->name
                        << " (signals=" << worst_signal_count
                        << ", cov=" << std::fixed << std::setprecision(2) << all_flags[worst_idx].coverage_dev
                        << ", dmg=" << all_flags[worst_idx].damage_dev
                        << ", graph=" << all_flags[worst_idx].graph_dev
                        << ", comp=" << all_flags[worst_idx].composition_dev << ")";
                log_.trace(log_msg.str());
            }

            // === Phase 4: Review Pile Re-recruitment ===
            // Try to add back flagged contigs if they improve quality
            int bin_recruited = 0;
            for (auto* c : review_pile) {
                if (c->bin_id != -1) continue;  // Already processed

                std::vector<Contig*> test_members = bins[bi].members;
                test_members.push_back(c);
                double test_quality = compute_quality_score(test_members);

                if (test_quality > initial_quality + 0.005) {
                    // Re-recruit
                    c->bin_id = bi;
                    bins[bi].members.push_back(c);
                    initial_quality = test_quality;
                    total_recruited++;
                    bin_recruited++;
                    log_.trace("  V34 RE-RECRUIT: bin " + std::to_string(bi) +
                        " contig " + c->name + " (improved Q to " + std::to_string(test_quality) + ")");
                }
            }

            // === Phase 5: V34 TNF Gravity Force-Recruitment ===
            // Look for unbinned contigs with very strong TNF match to this bin
            int bin_tnf_recruited = 0;
            for (auto& c : contigs) {
                if (c.bin_id != -1) continue;  // Already binned
                if (c.length < 2000) continue;  // Skip very short
                if (c.tnf.empty()) continue;

                // Compute TNF distance to bin centroid
                double tnf_dist = 0.0;
                for (int d = 0; d < TNF_DIM && d < (int)c.tnf.size(); d++) {
                    double diff = c.tnf[d] - tnf_centroid[d];
                    tnf_dist += diff * diff;
                }
                tnf_dist = std::sqrt(tnf_dist);

                // V34: If very close compositionally, force-recruit
                if (tnf_dist < TNF_GRAVITY_THRESHOLD) {
                    // Check coverage compatibility (within 3x of bin median)
                    if (c.mean_coverage > 0 && cov_median > 0) {
                        double cov_ratio = c.mean_coverage / cov_median;
                        if (cov_ratio < 0.33 || cov_ratio > 3.0) continue;  // Too far in coverage
                    }

                    // Check that it doesn't hurt quality
                    std::vector<Contig*> test_members = bins[bi].members;
                    test_members.push_back(&c);
                    double test_quality = compute_quality_score(test_members);

                    if (test_quality >= initial_quality - 0.01) {  // Allow small quality drop
                        c.bin_id = bi;
                        bins[bi].members.push_back(&c);
                        total_tnf_recruited++;
                        bin_tnf_recruited++;
                        log_.trace("  V34 TNF-RECRUIT: bin " + std::to_string(bi) +
                            " contig " + c.name + " (tnf_dist=" + std::to_string(tnf_dist) + ")");
                    }
                }
            }

            if (bin_removed > 0 || bin_flagged > 0 || bin_recruited > 0 || bin_tnf_recruited > 0) {
                std::cerr << "  bin " << bi << ": flagged=" << bin_flagged
                          << ", removed=" << bin_removed << ", recruited=" << bin_recruited
                          << ", tnf_recruited=" << bin_tnf_recruited
                          << " (final cgr_occ=" << std::fixed << std::setprecision(3) << initial_cgr_occ << ")\n";
            }
        }

        // Rebuild and refit all bins
        for (auto& b : bins) {
            if (b.members.size() >= 2) b.fit();
        }

        log_.metric("consensus_flagged", total_flagged);
        log_.metric("consensus_removed", total_removed);
        log_.metric("consensus_recruited", total_recruited);
        log_.metric("tnf_recruited", total_tnf_recruited);

        std::cerr << "\n  Summary: flagged=" << total_flagged
                  << ", removed=" << total_removed
                  << ", re-recruited=" << total_recruited
                  << ", tnf-recruited=" << total_tnf_recruited << "\n";
    }

    // Keep old function names as aliases for backward compatibility
    void iterative_mad_peeling(int max_iterations = 10) {
        // V33: Replaced by consensus_gated_contamination_removal
        // This is now a no-op, actual work done in consensus_gated_contamination_removal
        (void)max_iterations;
    }

    void damage_timestamping_check() {
        // No-op: damage is handled as one of the consensus signals
    }

    // Merge small high-purity bins into compatible larger bins
    void purity_constrained_micro_merge() {
        log_.section("Bin merging");

        const double PURITY_THRESHOLD = -0.85;  // Modularity threshold for "pure"
        const double TNF_DIST_THRESHOLD = 0.20;  // Max TNF centroid distance
        const double COV_RATIO_THRESHOLD = 2.5;  // Max coverage ratio
        const size_t MIN_BIN_SIZE = 100000;      // Minimum bp to consider for merge

        int merges_performed = 0;

        // Compute bin signatures for merge candidates
        struct MergeSig {
            int bin_idx;
            size_t total_bp;
            double modularity;
            double cgr_occ;
            double mean_cov;
            std::vector<double> tnf_centroid;
            bool is_pure;
        };

        std::vector<MergeSig> sigs;
        for (size_t bi = 0; bi < bins.size(); bi++) {
            if (bins[bi].members.size() < 3) continue;

            MergeSig sig;
            sig.bin_idx = bi;
            sig.total_bp = 0;
            for (auto* c : bins[bi].members) sig.total_bp += c->length;
            if (sig.total_bp < MIN_BIN_SIZE) continue;

            // Compute intrinsic quality
            std::string combined;
            for (auto* c : bins[bi].members) combined += c->sequence;
            sig.modularity = compute_debruijn_modularity(combined, 4);
            sig.cgr_occ = compute_cgr_occupancy(combined, 10);
            sig.is_pure = (sig.modularity < PURITY_THRESHOLD);

            // Compute mean coverage
            double cov_sum = 0.0;
            int cov_count = 0;
            for (auto* c : bins[bi].members) {
                if (c->mean_coverage > 0) {
                    cov_sum += c->mean_coverage;
                    cov_count++;
                }
            }
            sig.mean_cov = cov_count > 0 ? cov_sum / cov_count : 0.0;

            // Compute TNF centroid
            sig.tnf_centroid.assign(TNF_DIM, 0.0);
            for (auto* c : bins[bi].members) {
                for (int d = 0; d < TNF_DIM && d < (int)c->tnf.size(); d++) {
                    sig.tnf_centroid[d] += c->tnf[d];
                }
            }
            for (int d = 0; d < TNF_DIM; d++) {
                sig.tnf_centroid[d] /= bins[bi].members.size();
            }

            sigs.push_back(sig);
        }

        // Sort by size (largest first) - we merge small INTO large
        std::sort(sigs.begin(), sigs.end(), [](const MergeSig& a, const MergeSig& b) {
            return a.total_bp > b.total_bp;
        });

        // Track which bins have been merged (absorbed)
        std::set<int> absorbed;

        // Try to merge small pure bins into larger compatible bins
        for (size_t i = 0; i < sigs.size(); i++) {
            if (absorbed.count(sigs[i].bin_idx)) continue;
            if (!sigs[i].is_pure) continue;

            int target_bin = sigs[i].bin_idx;

            for (size_t j = i + 1; j < sigs.size(); j++) {
                if (absorbed.count(sigs[j].bin_idx)) continue;
                if (!sigs[j].is_pure) continue;

                int source_bin = sigs[j].bin_idx;

                // Check TNF centroid distance
                double tnf_dist = 0.0;
                for (int d = 0; d < TNF_DIM; d++) {
                    double diff = sigs[i].tnf_centroid[d] - sigs[j].tnf_centroid[d];
                    tnf_dist += diff * diff;
                }
                tnf_dist = std::sqrt(tnf_dist);

                if (tnf_dist > TNF_DIST_THRESHOLD) continue;

                // Check coverage ratio
                if (sigs[i].mean_cov > 0 && sigs[j].mean_cov > 0) {
                    double cov_ratio = std::max(sigs[i].mean_cov, sigs[j].mean_cov) /
                                       std::min(sigs[i].mean_cov, sigs[j].mean_cov);
                    if (cov_ratio > COV_RATIO_THRESHOLD) continue;
                }

                // Check that merge improves quality
                std::vector<Contig*> merged_members = bins[target_bin].members;
                for (auto* c : bins[source_bin].members) {
                    merged_members.push_back(c);
                }

                double merged_quality = compute_quality_score(merged_members);
                double target_quality = compute_quality_score(bins[target_bin].members);
                double source_quality = compute_quality_score(bins[source_bin].members);

                // Merge if combined quality is at least as good as the better individual
                if (merged_quality >= std::max(target_quality, source_quality) - 0.02) {
                    // Perform merge
                    for (auto* c : bins[source_bin].members) {
                        c->bin_id = target_bin;
                        bins[target_bin].members.push_back(c);
                    }
                    bins[source_bin].members.clear();
                    absorbed.insert(source_bin);
                    merges_performed++;

                    log_.trace("  V34 MICRO-MERGE: bin " + std::to_string(source_bin) +
                        " (" + std::to_string(sigs[j].total_bp / 1000) + "kb) -> bin " +
                        std::to_string(target_bin) + " (tnf_dist=" + std::to_string(tnf_dist) + ")");

                    std::cerr << "  Merged bin " << source_bin << " (" << sigs[j].total_bp / 1000
                              << "kb) into bin " << target_bin << " (tnf_dist="
                              << std::fixed << std::setprecision(3) << tnf_dist << ")\n";
                }
            }
        }

        // Rebuild and refit
        for (auto& b : bins) {
            if (b.members.size() >= 2) b.fit();
        }

        log_.metric("micro_merges", merges_performed);
        std::cerr << "\n  Performed " << merges_performed << " micro-merges\n";
    }

    // Compute contamination likelihood based on coverage distribution
    double compute_ploidy_score(const std::vector<Contig*>& members) {
        if (members.size() < 5) return 0.0;

        // Collect coverages
        std::vector<double> covs;
        for (auto* c : members) {
            if (c->mean_coverage > 0) covs.push_back(c->mean_coverage);
        }
        if (covs.size() < 5) return 0.0;

        // Compute median coverage
        std::vector<double> sorted_covs = covs;
        std::sort(sorted_covs.begin(), sorted_covs.end());
        double median_cov = sorted_covs[sorted_covs.size() / 2];
        if (median_cov < 1.0) return 0.0;

        // Normalize to "ploidy" (ratio to median)
        std::vector<double> ploidy;
        for (double c : covs) {
            ploidy.push_back(c / median_cov);
        }

        // Build histogram bins (0.5 to 2.0 in 0.1 steps)
        const int NBINS = 15;
        std::vector<int> hist(NBINS, 0);
        for (double p : ploidy) {
            int bin = (int)((p - 0.5) * 10);
            if (bin >= 0 && bin < NBINS) hist[bin]++;
        }

        // Find peaks (local maxima)
        int num_peaks = 0;
        for (int i = 1; i < NBINS - 1; i++) {
            if (hist[i] > hist[i-1] && hist[i] > hist[i+1] && hist[i] >= 3) {
                num_peaks++;
            }
        }

        // Compute coefficient of variation
        double mean_ploidy = std::accumulate(ploidy.begin(), ploidy.end(), 0.0) / ploidy.size();
        double var = 0.0;
        for (double p : ploidy) var += (p - mean_ploidy) * (p - mean_ploidy);
        double cv = std::sqrt(var / ploidy.size()) / mean_ploidy;

        // Ploidy score: 0 = clean, 1 = contaminated
        // High CV or multiple peaks = contamination
        double score = 0.0;
        if (num_peaks > 1) score += 0.5;
        if (cv > 0.3) score += 0.3;
        if (cv > 0.5) score += 0.2;

        return std::min(1.0, score);
    }

    // Split chimeric bins by coverage modes using bimodal detection
    void split_chimeric_bins() {
        log_.section("Coverage-based bin splitting");

        const double PLOIDY_THRESHOLD = 0.5;  // Split if ploidy score > this
        const double MIN_MODE_RATIO = 0.2;    // Each mode must have at least 20% of contigs

        int bins_split = 0;
        int new_bins_created = 0;

        // Identify bins to split
        std::vector<int> bins_to_split;
        for (int bi = 0; bi < (int)bins.size(); bi++) {
            auto& bin = bins[bi];
            if (bin.members.size() < 10) continue;

            size_t bp = 0;
            for (auto* c : bin.members) bp += c->length;
            if (bp < 500000) continue;  // Only split substantial bins

            double ploidy_score = compute_ploidy_score(bin.members);
            if (ploidy_score > PLOIDY_THRESHOLD) {
                bins_to_split.push_back(bi);
                std::cerr << "  bin " << bi << ": ploidy_score=" << std::fixed
                          << std::setprecision(2) << ploidy_score
                          << " (" << bin.members.size() << " contigs) -> SPLIT CANDIDATE\n";
            }
        }

        for (int bi : bins_to_split) {
            auto& bin = bins[bi];

            // Collect log-coverages
            std::vector<std::pair<double, Contig*>> log_covs;
            for (auto* c : bin.members) {
                if (c->mean_coverage > 0) {
                    log_covs.push_back({std::log(c->mean_coverage), c});
                }
            }

            if (log_covs.size() < 10) continue;

            // Sort by log-coverage
            std::sort(log_covs.begin(), log_covs.end());

            // Find the largest gap (simple bimodal split)
            double max_gap = 0.0;
            int split_idx = -1;
            for (int i = 1; i < (int)log_covs.size() - 1; i++) {
                double gap = log_covs[i].first - log_covs[i-1].first;
                // Ensure both sides have enough contigs
                int left_count = i;
                int right_count = log_covs.size() - i;
                double left_ratio = (double)left_count / log_covs.size();
                double right_ratio = (double)right_count / log_covs.size();

                if (gap > max_gap && left_ratio >= MIN_MODE_RATIO && right_ratio >= MIN_MODE_RATIO) {
                    max_gap = gap;
                    split_idx = i;
                }
            }

            // Only split if gap is significant (> 0.5 in log space = ~1.6x ratio)
            if (max_gap > 0.5 && split_idx > 0) {
                // Create new bin with lower-coverage contigs
                Bin new_bin;
                new_bin.members.reserve(split_idx);

                for (int i = 0; i < split_idx; i++) {
                    Contig* c = log_covs[i].second;
                    c->bin_id = bins.size();  // New bin ID
                    new_bin.members.push_back(c);
                }

                // Keep higher-coverage contigs in original bin
                bin.members.clear();
                for (int i = split_idx; i < (int)log_covs.size(); i++) {
                    bin.members.push_back(log_covs[i].second);
                }

                // Refit both bins
                if (bin.members.size() >= 2) bin.fit();
                if (new_bin.members.size() >= 2) new_bin.fit();

                bins.push_back(new_bin);
                bins_split++;
                new_bins_created++;

                std::cerr << "  Split bin " << bi << ": " << new_bin.members.size()
                          << " contigs -> new bin, " << bin.members.size()
                          << " contigs remain (gap=" << std::fixed << std::setprecision(2)
                          << max_gap << ")\n";
            }
        }

        log_.metric("chimera_bins_split", bins_split);
        log_.metric("chimera_new_bins", new_bins_created);

        std::cerr << "\n  Chimera splitting: " << bins_split << " bins split, "
                  << new_bins_created << " new bins created\n";
    }

    // Global Bin Refinement: Move contigs to their optimal bin
    // For each contig, if another bin is significantly closer, move it
    // No arbitrary thresholds - purely based on compositional fit
    void temporal_chimera_rescue() {
        log_.section("Global bin refinement (attraction-based)");

        auto stable_distance = [](const std::vector<double>& a, const std::vector<double>& b) -> double {
            if (a.empty() || b.empty()) return 1e9;
            double d = 0;
            for (size_t i = 0; i < a.size() && i < b.size(); i++) {
                double diff = a[i] - b[i];
                d += diff * diff;
            }
            return std::sqrt(d);
        };

        // Build models for ALL bins with sufficient members
        struct BinModel {
            int bin_idx = -1;
            std::vector<double> centroid;
            double median_dist = 0.0;  // Median distance to centroid (defines "inlier")
            double mean_coverage = 0.0;
            size_t total_bp = 0;
            bool valid = false;
        };

        std::vector<BinModel> bin_models(bins.size());

        for (size_t bi = 0; bi < bins.size(); bi++) {
            auto& bin = bins[bi];
            if (bin.members.size() < 5) continue;

            size_t bp = 0;
            for (auto* c : bin.members) bp += c->length;

            // Compute centroid
            std::vector<double> centroid(STABLE_DIM, 0.0);
            std::vector<double> coverages;
            int count = 0;
            for (auto* c : bin.members) {
                if (!c->stable.empty() && (int)c->stable.size() >= STABLE_DIM) {
                    for (int k = 0; k < STABLE_DIM; k++) centroid[k] += c->stable[k];
                    count++;
                }
                if (c->mean_coverage > 0) coverages.push_back(c->mean_coverage);
            }
            if (count < 5) continue;
            for (auto& v : centroid) v /= count;

            // Compute median distance to centroid
            std::vector<double> dists;
            for (auto* c : bin.members) {
                if (!c->stable.empty()) {
                    dists.push_back(stable_distance(c->stable, centroid));
                }
            }
            std::sort(dists.begin(), dists.end());
            double median_dist = dists[dists.size() / 2];

            double mean_cov = 0.0;
            if (!coverages.empty()) {
                for (double c : coverages) mean_cov += c;
                mean_cov /= coverages.size();
            }

            bin_models[bi] = {(int)bi, centroid, median_dist, mean_cov, bp, true};
        }

        int valid_bins = 0;
        for (auto& m : bin_models) if (m.valid) valid_bins++;
        log_.detail("Built models for " + std::to_string(valid_bins) + " bins");
        if (valid_bins < 2) return;

        // Evaluate ALL contigs for potential reassignment
        const double COVERAGE_RATIO_MAX = 3.0;  // Hard coverage gate

        int rescued = 0;
        int rejected_coverage = 0;
        int rejected_no_improvement = 0;
        int stayed_optimal = 0;

        for (auto& c : contigs) {
            if (c.stable.empty() || (int)c.stable.size() < STABLE_DIM) continue;
            if (c.length < 2500) continue;

            // Distance to current bin (if binned)
            double d_current = 1e9;
            double median_current = 1e9;
            if (c.bin_id >= 0 && c.bin_id < (int)bin_models.size() && bin_models[c.bin_id].valid) {
                d_current = stable_distance(c.stable, bin_models[c.bin_id].centroid);
                median_current = bin_models[c.bin_id].median_dist;
            }

            // Only consider contigs on the OUTSKIRTS of their bin (d > median)
            // Contigs well within their bin stay put
            bool is_on_outskirts = (c.bin_id < 0) || (d_current > median_current);
            if (!is_on_outskirts) {
                stayed_optimal++;
                continue;
            }

            // Find closest bin by stable distance
            int best_bin = -1;
            double d_best = 1e9;
            for (size_t bi = 0; bi < bin_models.size(); bi++) {
                if ((int)bi == c.bin_id) continue;
                if (!bin_models[bi].valid) continue;

                // Coverage gate
                if (c.mean_coverage > 0 && bin_models[bi].mean_coverage > 0) {
                    double cov_ratio = std::max(c.mean_coverage, bin_models[bi].mean_coverage) /
                                      std::min(c.mean_coverage, bin_models[bi].mean_coverage);
                    if (cov_ratio > COVERAGE_RATIO_MAX) continue;
                }

                double d = stable_distance(c.stable, bin_models[bi].centroid);
                if (d < d_best) {
                    d_best = d;
                    best_bin = bi;
                }
            }

            // Decision: move if target is significantly closer
            // AND the contig would be an "inlier" in the target (d < median_dist of target)
            // AND target bin is larger (we consolidate fragments into large bins, not vice versa)
            if (best_bin >= 0 && best_bin != c.bin_id) {
                bool is_improvement = (d_best < d_current * kImprovementFactor);
                bool is_inlier = (d_best <= bin_models[best_bin].median_dist);  // Strict inlier
                bool target_larger = (c.bin_id < 0) ||
                    (bin_models[best_bin].total_bp > bin_models[c.bin_id].total_bp);

                if (is_improvement && is_inlier && target_larger) {
                    int old_bin = c.bin_id;
                    c.bin_id = best_bin;
                    bins[best_bin].members.push_back(&c);

                    if (old_bin >= 0 && old_bin < (int)bins.size()) {
                        auto& old_members = bins[old_bin].members;
                        old_members.erase(std::remove(old_members.begin(), old_members.end(), &c), old_members.end());
                    }

                    rescued++;
                    if (rescued <= 10) {
                        std::ostringstream oss;
                        oss << "  Move: " << c.name << " d=" << std::fixed << std::setprecision(2)
                            << d_current << "->" << d_best << " bin_" << old_bin << "->bin_" << best_bin;
                        log_.detail(oss.str());
                    }
                } else {
                    rejected_no_improvement++;
                }
            } else if (best_bin < 0) {
                rejected_coverage++;
            } else {
                stayed_optimal++;
            }
        }

        if (rescued > 10) {
            log_.detail("  ... and " + std::to_string(rescued - 10) + " more moves");
        }

        for (auto& bin : bins) {
            if (bin.members.size() >= 2) bin.fit();
        }

        log_.metric("refinement_moved", rescued);
        log_.metric("refinement_stayed", stayed_optimal);
        log_.metric("refinement_coverage_blocked", rejected_coverage);
        log_.metric("refinement_no_improvement", rejected_no_improvement);

        std::cerr << "\n  Global refinement: " << rescued << " moved, "
                  << stayed_optimal << " stayed optimal, "
                  << rejected_coverage << " cov-blocked, "
                  << rejected_no_improvement << " no improvement\n";
    }

    // Recover unbinned contigs using two-pass TNF matching with decaying thresholds
    void lazarus_rescue_phase() {
        log_.section("Unbinned contig recovery");

        int total_rescued = 0;
        int total_rejected_ambiguous = 0;
        int total_rejected_signals = 0;

        // Run two passes with decaying thresholds
        for (int pass = 1; pass <= 2; pass++) {
            // V36: Decay thresholds for second pass
            double TNF_RESCUE_THRESHOLD = (pass == 1) ? 0.10 : 0.08;
            double COVERAGE_RATIO_MAX = kDefaultCovRatioMax;
            double DAMAGE_SIGMA_MAX = 2.0;
            int MIN_SIGNALS_FOR_RESCUE = (pass == 1) ? 3 : 4;

            std::cerr << "  --- Pass " << pass << " (TNF < " << TNF_RESCUE_THRESHOLD
                      << ", " << MIN_SIGNALS_FOR_RESCUE << "+ signals) ---\n";

        // Collect unbinned contigs (satellites)
        std::vector<Contig*> satellites;
        for (auto& c : contigs) {
            if (c.bin_id < 0 && c.length >= 2000) {
                satellites.push_back(&c);
            }
        }

        std::cerr << "  Found " << satellites.size() << " unbinned contigs (satellites)\n";

        if (satellites.empty()) {
            std::cerr << "  No satellites to rescue\n";
            return;
        }

        // Compute bin statistics for matching
        struct BinStats {
            double mean_coverage = 0.0;
            double var_coverage = 0.0;
            double mean_damage_5p = 0.0;
            double var_damage_5p = 0.0;
            double mean_damage_3p = 0.0;
            double var_damage_3p = 0.0;
            std::vector<double> tnf_centroid;
            size_t total_bp = 0;
            int bin_idx = -1;
        };

        std::vector<BinStats> bin_stats;
        for (int bi = 0; bi < (int)bins.size(); bi++) {
            auto& bin = bins[bi];
            if (bin.members.size() < 5) continue;  // Skip tiny bins

            size_t bp = 0;
            for (auto* c : bin.members) bp += c->length;
            if (bp < 200000) continue;  // Skip small bins

            BinStats bs;
            bs.bin_idx = bi;
            bs.total_bp = bp;
            bs.tnf_centroid.resize(TNF_DIM, 0.0);

            // Coverage stats
            std::vector<double> covs, d5ps, d3ps;
            for (auto* c : bin.members) {
                covs.push_back(c->mean_coverage);
                d5ps.push_back(c->damage_rate_5p);
                d3ps.push_back(c->damage_rate_3p);
                for (int d = 0; d < TNF_DIM && d < (int)c->tnf.size(); d++) {
                    bs.tnf_centroid[d] += c->tnf[d];
                }
            }

            // Normalize TNF centroid
            for (auto& v : bs.tnf_centroid) v /= bin.members.size();

            // Coverage mean and variance
            if (!covs.empty()) {
                bs.mean_coverage = std::accumulate(covs.begin(), covs.end(), 0.0) / covs.size();
                double var = 0.0;
                for (double x : covs) var += (x - bs.mean_coverage) * (x - bs.mean_coverage);
                bs.var_coverage = covs.size() > 1 ? var / (covs.size() - 1) : 0.0;
            }

            // Damage mean and variance (5')
            if (!d5ps.empty()) {
                bs.mean_damage_5p = std::accumulate(d5ps.begin(), d5ps.end(), 0.0) / d5ps.size();
                double var = 0.0;
                for (double x : d5ps) var += (x - bs.mean_damage_5p) * (x - bs.mean_damage_5p);
                bs.var_damage_5p = d5ps.size() > 1 ? var / (d5ps.size() - 1) : 0.0;
            }

            // Damage mean and variance (3')
            if (!d3ps.empty()) {
                bs.mean_damage_3p = std::accumulate(d3ps.begin(), d3ps.end(), 0.0) / d3ps.size();
                double var = 0.0;
                for (double x : d3ps) var += (x - bs.mean_damage_3p) * (x - bs.mean_damage_3p);
                bs.var_damage_3p = d3ps.size() > 1 ? var / (d3ps.size() - 1) : 0.0;
            }

            bin_stats.push_back(bs);
        }

        std::cerr << "  Analyzing " << bin_stats.size() << " core bins for rescue targets\n";

        int rescued = 0;
        int rejected_ambiguous = 0;
        int rejected_signals = 0;

        for (auto* sat : satellites) {
            // Count how many bins this satellite matches
            std::vector<std::pair<int, int>> matches;  // (bin_idx, signal_count)

            for (auto& bs : bin_stats) {
                int signals = 0;

                // Signal 1: Coverage within ratio
                if (bs.mean_coverage > 0) {
                    double ratio = sat->mean_coverage / bs.mean_coverage;
                    if (ratio < 1.0) ratio = 1.0 / ratio;
                    if (ratio <= COVERAGE_RATIO_MAX) signals++;
                }

                // Signal 2: TNF distance
                double tnf_dist = 0.0;
                for (int d = 0; d < TNF_DIM && d < (int)sat->tnf.size(); d++) {
                    double diff = sat->tnf[d] - bs.tnf_centroid[d];
                    tnf_dist += diff * diff;
                }
                tnf_dist = std::sqrt(tnf_dist);
                if (tnf_dist < TNF_RESCUE_THRESHOLD) signals++;

                // Signal 3: Damage 5' within 2σ
                double sigma_5p = std::sqrt(bs.var_damage_5p + 1e-6);
                double z_5p = std::abs(sat->damage_rate_5p - bs.mean_damage_5p) / sigma_5p;
                if (z_5p <= DAMAGE_SIGMA_MAX || !sat->has_damage) signals++;

                // Signal 4: Damage 3' within 2σ (bonus signal)
                double sigma_3p = std::sqrt(bs.var_damage_3p + 1e-6);
                double z_3p = std::abs(sat->damage_rate_3p - bs.mean_damage_3p) / sigma_3p;
                if (z_3p <= DAMAGE_SIGMA_MAX || !sat->has_damage) signals++;

                if (signals >= MIN_SIGNALS_FOR_RESCUE) {
                    matches.push_back({bs.bin_idx, signals});
                }
            }

            // Require EXACTLY ONE matching bin (no ambiguity)
            if (matches.size() == 1) {
                int target_bin = matches[0].first;
                sat->bin_id = target_bin;
                bins[target_bin].members.push_back(sat);
                rescued++;
            } else if (matches.size() > 1) {
                rejected_ambiguous++;
            } else {
                rejected_signals++;
            }
        }

        // Update bin statistics after this pass
        for (auto& bin : bins) {
            if (bin.members.size() >= 2) bin.fit();
        }

        std::cerr << "    Pass " << pass << " rescued: " << rescued << " contigs\n";
        total_rescued += rescued;
        total_rejected_ambiguous += rejected_ambiguous;
        total_rejected_signals += rejected_signals;

        } // End of pass loop

        log_.metric("recovery_rescued", total_rescued);
        log_.metric("recovery_rejected_ambiguous", total_rejected_ambiguous);
        log_.metric("recovery_rejected_signals", total_rejected_signals);

        std::cerr << "\n  Total Lazarus rescue results (2 passes):\n";
        std::cerr << "    Total rescued: " << total_rescued << " contigs\n";
        std::cerr << "    Total rejected (ambiguous): " << total_rejected_ambiguous << "\n";
        std::cerr << "    Total rejected (signals): " << total_rejected_signals << "\n";
    }

    // Log intrinsic quality metrics for all bins
    void log_intrinsic_quality() {
        log_.section("Intrinsic quality assessment");

        log_.info("Assessing intrinsic quality...");

        int bin_num = 0;
        int likely_clean = 0;
        int likely_complete = 0;
        int needs_split = 0;

        for (auto& bin : bins) {
            if (bin.members.size() < 2) continue;

            size_t bin_bp = 0;
            for (auto* c : bin.members) bin_bp += c->length;
            if (bin_bp < 200000) continue;

            IntrinsicQuality iq = assess_intrinsic_quality(bin.members);

            std::string flags;
            if (iq.is_likely_clean) { flags += "CLEAN "; likely_clean++; }
            if (iq.is_likely_complete) { flags += "COMPLETE "; likely_complete++; }
            if (iq.needs_splitting) { flags += "SPLIT? "; needs_split++; }

            std::ostringstream ss;
            ss << "bin_" << bin_num << ": " << bin.members.size() << " contigs, "
               << bin_bp / 1000 << " kb, JSD=" << std::fixed << std::setprecision(4) << iq.mean_jsd
               << ", comp=" << std::setprecision(1) << (iq.completeness_est * 100) << "%"
               << ", contam=" << (iq.contamination_est * 100) << "% " << flags;
            log_.detail(ss.str());

            bin_num++;
        }

        {
            std::ostringstream ss;
            ss << "Quality: " << likely_clean << " likely clean, "
               << likely_complete << " likely complete, "
               << needs_split << " may need splitting";
            log_.info(ss.str());
        }

        log_.metric("likely_clean_bins", likely_clean);
        log_.metric("likely_complete_bins", likely_complete);
        log_.metric("bins_needing_split", needs_split);
    }

    void score_based_contamination_cleanup(double score_threshold = 5.0) {
        log_.section("Contamination cleanup");

        // Collect bins
        std::map<int, std::vector<Contig*>> bins;
        for (auto& c : contigs) {
            if (c.bin_id >= 0) {
                bins[c.bin_id].push_back(&c);
            }
        }

        int total_removed = 0;
        size_t total_bp_removed = 0;

        for (auto& [bin_id, members] : bins) {
            if (members.size() < 10) continue;  // Skip tiny bins

            // Compute coverage statistics (robust: median + MAD)
            std::vector<double> coverages;
            for (auto* c : members) {
                coverages.push_back(c->mean_coverage);
            }
            std::sort(coverages.begin(), coverages.end());
            double cov_median = coverages[coverages.size() / 2];

            std::vector<double> abs_devs;
            for (double cov : coverages) {
                abs_devs.push_back(std::abs(cov - cov_median));
            }
            std::sort(abs_devs.begin(), abs_devs.end());
            double cov_mad = abs_devs[abs_devs.size() / 2];
            if (cov_mad < 1.0) cov_mad = 1.0;  // Floor to prevent division issues

            // Compute TNF centroid (length-weighted)
            std::vector<double> tnf_centroid(136, 0.0);
            size_t total_bp = 0;
            for (auto* c : members) {
                for (size_t i = 0; i < 136 && i < c->tnf.size(); i++) {
                    tnf_centroid[i] += c->tnf[i] * c->length;
                }
                total_bp += c->length;
            }
            for (auto& v : tnf_centroid) v /= total_bp;

            // Compute GC mean
            double gc_sum = 0.0;
            for (auto* c : members) gc_sum += c->gc * c->length;
            double gc_mean = gc_sum / total_bp;

            // Score each contig and identify outliers
            std::vector<std::pair<Contig*, double>> outliers;

            for (auto* c : members) {
                // Coverage z-score
                double cov_z = std::abs(c->mean_coverage - cov_median) / cov_mad;

                // TNF distance from centroid
                double tnf_dist = 0.0;
                for (size_t i = 0; i < 136 && i < c->tnf.size(); i++) {
                    double diff = c->tnf[i] - tnf_centroid[i];
                    tnf_dist += diff * diff;
                }
                tnf_dist = std::sqrt(tnf_dist);

                // GC distance
                double gc_dist = std::abs(c->gc - gc_mean) / 0.02;  // ~2% expected std

                // Combined score (higher = more likely contaminant)
                double score = cov_z + 5.0 * tnf_dist + gc_dist;

                if (score > score_threshold) {
                    outliers.push_back({c, score});
                }
            }

            // Sort by score descending
            std::sort(outliers.begin(), outliers.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

            // Calculate bin completeness estimate (based on size relative to typical genome ~3Mbp)
            double est_completeness = std::min(1.0, (double)total_bp / 3000000.0);

            // For near-HQ bins (>85% estimated completeness), allow more aggressive cleanup
            // Otherwise limit to 10% to avoid over-cleaning
            size_t max_remove;
            if (est_completeness > 0.85) {
                max_remove = std::max((size_t)1, members.size() / 5);  // Up to 20% for near-HQ
            } else {
                max_remove = std::max((size_t)1, members.size() / 10); // 10% for others
            }

            size_t removed_from_bin = 0;
            size_t bp_removed_from_bin = 0;

            for (auto& [c, score] : outliers) {
                if (removed_from_bin >= max_remove) break;

                // Don't remove large contigs (>20kb) - they're usually core genome
                if (c->length > 20000) continue;

                c->bin_id = -1;
                c->final_bin_id = -1;
                removed_from_bin++;
                bp_removed_from_bin += c->length;
            }

            if (removed_from_bin > 0) {
                std::cerr << "  bin_" << bin_id << ": removed " << removed_from_bin
                          << " outliers (" << bp_removed_from_bin << " bp)\n";
                total_removed += removed_from_bin;
                total_bp_removed += bp_removed_from_bin;
            }
        }

        log_.info("Contamination cleanup: removed " + std::to_string(total_removed) +
                  " contigs (" + std::to_string(total_bp_removed) + " bp)");

        log_.metric("cleanup_contigs_removed", total_removed);
        log_.metric("cleanup_bp_removed", (int)total_bp_removed);
    }

    void coverage_aware_contamination_cleanup(double contamination_threshold = 5e-9,
                                              int max_k = 5) {
        log_.section("Coverage-aware contamination cleanup");

        // Group contigs by bin_id
        std::map<int, std::vector<Contig*>> bin_members;
        for (auto& c : contigs) {
            if (c.bin_id >= 0) {
                bin_members[c.bin_id].push_back(&c);
            }
        }

        int bins_checked = 0;
        int bins_split = 0;
        int new_bins_created = 0;
        int next_bin_id = bins.size();

        std::vector<std::tuple<int, int, double, double, double, int>> split_results;

        for (auto& [bin_id, members] : bin_members) {
            if (members.size() < 10) continue;
            bins_checked++;

            // Compute contamination score de novo
            auto score = compute_contamination_score_denovo(members, 20);

            if (score.combined <= contamination_threshold) continue;

            if (verbose()) {
                std::cerr << "  bin_" << bin_id << ": occ_range=" << std::fixed << std::setprecision(4)
                          << score.occ_range << ", downsample_var=" << std::scientific
                          << score.downsample_var << ", combined=" << score.combined << "\n";
            }

            // Extract coverage values
            std::vector<double> coverages;
            for (auto* c : members) {
                coverages.push_back(c->mean_coverage);
            }

            // Find optimal k using BIC
            int optimal_k = find_optimal_gmm_components(coverages, max_k);

            if (optimal_k == 1) {
                if (verbose()) std::cerr << "    -> Single coverage mode, cannot split by coverage\n";
                continue;
            }

            // Fit GMM with optimal k
            std::vector<GMMComponent> components;
            fit_gmm(coverages, optimal_k, components);

            // Sort components by mean coverage
            std::sort(components.begin(), components.end(),
                      [](const GMMComponent& a, const GMMComponent& b) {
                          return a.mean < b.mean;
                      });

            // Assign contigs to clusters
            std::vector<std::vector<Contig*>> clusters(optimal_k);
            for (auto* c : members) {
                double best_prob = 0;
                int best_cluster = 0;
                for (int j = 0; j < optimal_k; j++) {
                    double prob = components[j].weight *
                        gaussian_pdf(c->mean_coverage, components[j].mean, components[j].variance);
                    if (prob > best_prob) {
                        best_prob = prob;
                        best_cluster = j;
                    }
                }
                clusters[best_cluster].push_back(c);
            }

            // Check cluster sizes - need at least 2 valid clusters
            int valid_clusters = 0;
            for (int k = 0; k < optimal_k; k++) {
                if (clusters[k].size() >= 5) valid_clusters++;
            }
            if (valid_clusters < 2) {
                if (verbose()) std::cerr << "    -> Only " << valid_clusters << " valid cluster(s), skipping split\n";
                continue;
            }

            // Split: keep largest cluster in original bin, create new bins for others
            int largest_cluster = 0;
            size_t largest_size = 0;
            for (int k = 0; k < optimal_k; k++) {
                if (clusters[k].size() > largest_size) {
                    largest_size = clusters[k].size();
                    largest_cluster = k;
                }
            }

            // Create new bins for non-largest clusters
            for (int k = 0; k < optimal_k; k++) {
                if (k == largest_cluster) continue;
                if (clusters[k].size() < 5) continue;

                // Assign contigs to new bin
                for (auto* c : clusters[k]) {
                    c->bin_id = next_bin_id;
                    c->final_bin_id = next_bin_id;
                }

                // Create new Bin structure
                Bin new_bin;
                new_bin.members = clusters[k];
                new_bin.fit();  // Compute distributions for new bin
                bins.push_back(new_bin);

                size_t cluster_bp = 0;
                for (auto* c : clusters[k]) cluster_bp += c->length;

                if (verbose()) {
                    std::cerr << "    -> Created bin_" << next_bin_id << " with "
                              << clusters[k].size() << " contigs ("
                              << (cluster_bp / 1000.0) << " kb), mean_cov="
                              << std::fixed << std::setprecision(1) << components[k].mean << "\n";
                }

                next_bin_id++;
                new_bins_created++;
            }

            // Update original bin with largest cluster
            if ((size_t)bin_id < bins.size()) {
                bins[bin_id].members = clusters[largest_cluster];
                bins[bin_id].fit();  // Recompute distributions for updated bin
            }

            bins_split++;
            split_results.push_back({bin_id, (int)members.size(), score.occ_range,
                                     score.downsample_var, score.combined, optimal_k});
        }

        log_.info("Coverage cleanup: " + std::to_string(bins_checked) + " bins checked, " +
                  std::to_string(bins_split) + " split, " +
                  std::to_string(new_bins_created) + " new bins created");

        log_.metric("cov_cleanup_bins_checked", bins_checked);
        log_.metric("cov_cleanup_bins_split", bins_split);
        log_.metric("cov_cleanup_new_bins", new_bins_created);
    }

    // TNF-based contamination cleanup (second pass after coverage split)
    // Removes compositional outliers that couldn't be separated by coverage
    // Uses percentile-based threshold for dataset-independence
    void tnf_based_contamination_cleanup(
            double sigma_threshold = kDefaultTnfSigmaThreshold,
            double top_percentile = kDefaultTnfTopPercentile) {
        log_.info("Running TNF-based contamination cleanup...");
        log_.detail("Removing compositional outliers (>" + std::to_string(sigma_threshold) + " sigma from TNF centroid)");
        log_.detail("Targeting bins in top " + std::to_string(int(top_percentile * 100)) + "% of contamination scores");

        // Compute contamination scores for all bins to determine threshold
        std::map<size_t, double> bin_score_map;  // Cache scores to avoid recomputation
        std::vector<std::pair<size_t, double>> bin_scores;
        for (size_t bin_idx = 0; bin_idx < bins.size(); bin_idx++) {
            auto& bin = bins[bin_idx];
            if (bin.members.size() < kMinContigsForCleanup) continue;
            auto score = compute_contamination_score_denovo(bin.members, 20);
            bin_scores.push_back({bin_idx, score.combined});
            bin_score_map[bin_idx] = score.combined;
        }

        if (bin_scores.empty()) {
            std::cerr << "  No bins with >=10 contigs to process\n";
            return;
        }

        // Warn if few bins for percentile calculation
        if (bin_scores.size() < 5) {
            std::cerr << "  Warning: Only " << bin_scores.size()
                      << " bins - percentile thresholds may be coarse\n";
        }

        // Sort by score and find percentile threshold
        std::vector<double> scores_only;
        for (auto& p : bin_scores) scores_only.push_back(p.second);
        std::sort(scores_only.begin(), scores_only.end());
        size_t threshold_idx = scores_only.size() * (1.0 - top_percentile);
        double score_threshold = scores_only[threshold_idx];
        std::cerr << "  Contamination score threshold (top " << (top_percentile * 100)
                  << "%): " << std::scientific << score_threshold << "\n";

        int bins_cleaned = 0;
        int contigs_removed = 0;
        size_t bp_removed = 0;

        for (size_t bin_idx = 0; bin_idx < bins.size(); bin_idx++) {
            auto& bin = bins[bin_idx];
            if (bin.members.size() < kMinContigsForCleanup) continue;

            // Use cached score for percentile-based threshold
            auto it = bin_score_map.find(bin_idx);
            if (it == bin_score_map.end() || it->second < score_threshold) continue;

            // Compute length-weighted TNF centroid
            const int TNF_DIM = 136;  // Tetranucleotide frequency dimension
            std::vector<double> centroid(TNF_DIM, 0.0);
            double total_len = 0.0;

            for (auto* c : bin.members) {
                if (c->tnf.size() != TNF_DIM) continue;
                for (int i = 0; i < TNF_DIM; i++) {
                    centroid[i] += c->tnf[i] * c->length;
                }
                total_len += c->length;
            }
            if (total_len < 1) continue;

            for (int i = 0; i < TNF_DIM; i++) {
                centroid[i] /= total_len;
            }

            // Compute L2 distance from each contig to centroid
            std::vector<double> distances;
            std::vector<Contig*> valid_contigs;
            for (auto* c : bin.members) {
                if (c->tnf.size() != TNF_DIM) continue;
                double dist = 0.0;
                for (int i = 0; i < TNF_DIM; i++) {
                    double d = c->tnf[i] - centroid[i];
                    dist += d * d;
                }
                distances.push_back(std::sqrt(dist));
                valid_contigs.push_back(c);
            }

            if (distances.empty()) continue;

            // Compute median and MAD (more robust than mean/std)
            std::vector<double> sorted_dist = distances;
            std::sort(sorted_dist.begin(), sorted_dist.end());
            double median = sorted_dist[sorted_dist.size() / 2];

            std::vector<double> abs_devs;
            for (double d : distances) {
                abs_devs.push_back(std::abs(d - median));
            }
            std::sort(abs_devs.begin(), abs_devs.end());
            double mad = abs_devs[abs_devs.size() / 2] * 1.4826;  // Scale to match std

            if (mad < 0.001) continue;  // All contigs very similar

            // Identify outliers
            double threshold = median + sigma_threshold * mad;
            std::vector<Contig*> kept_contigs;
            std::vector<Contig*> outlier_contigs;

            for (size_t i = 0; i < valid_contigs.size(); i++) {
                if (distances[i] > threshold) {
                    outlier_contigs.push_back(valid_contigs[i]);
                } else {
                    kept_contigs.push_back(valid_contigs[i]);
                }
            }

            // Only remove if we have outliers and keep at least kMinRetentionFraction of contigs
            if (outlier_contigs.empty()) continue;
            if (kept_contigs.size() < bin.members.size() * kMinRetentionFraction) continue;

            // Calculate bp to remove
            size_t outlier_bp = 0;
            for (auto* c : outlier_contigs) {
                outlier_bp += c->length;
                c->bin_id = -1;  // Mark as unbinned
            }

            size_t orig_size = bin.members.size();
            bin.members = kept_contigs;
            bin.fit();  // Recompute distributions

            if (verbose()) {
                std::cerr << "  bin_" << bin_idx << ": removed " << outlier_contigs.size()
                          << " TNF outliers (" << outlier_bp / 1000 << " kb), "
                          << orig_size << " -> " << kept_contigs.size() << " contigs\n";
            }

            bins_cleaned++;
            contigs_removed += outlier_contigs.size();
            bp_removed += outlier_bp;
        }

        log_.info("TNF cleanup: " + std::to_string(bins_cleaned) + " bins, " +
                  std::to_string(contigs_removed) + " contigs removed");
    }

    // Coverage outlier cleanup (third pass)
    // Removes contigs with anomalous coverage that escaped GMM-based splitting
    // GMM finds distinct modes but misses individual outliers
    // Uses coverage CV (coefficient of variation) as trigger for dataset-independence
    void coverage_outlier_cleanup(
            double cov_deviation = kDefaultCovDeviation,
            double min_cov_cv = kDefaultMinCovCV,
            double min_mean_cov = kDefaultMinMeanCov) {
        log_.info("Running coverage outlier cleanup...");
        {
            std::ostringstream ss;
            ss << "Triggers on bins with coverage CV > " << min_cov_cv << " and mean_cov >= " << min_mean_cov << "x";
            log_.detail(ss.str());
        }
        {
            std::ostringstream ss;
            ss << "Removes contigs with coverage outside " << (1-cov_deviation)*100 << "-" << (1+cov_deviation)*100 << "% of median";
            log_.detail(ss.str());
        }

        int bins_cleaned = 0;
        int contigs_removed = 0;
        size_t bp_removed = 0;

        for (size_t bin_idx = 0; bin_idx < bins.size(); bin_idx++) {
            auto& bin = bins[bin_idx];
            if (bin.members.size() < kMinContigsForCleanup) continue;

            // Compute coverage statistics
            std::vector<double> coverages;
            double cov_sum = 0.0;
            for (auto* c : bin.members) {
                coverages.push_back(c->mean_coverage);
                cov_sum += c->mean_coverage;
            }
            double mean_cov = cov_sum / coverages.size();

            // Skip bins with very low coverage (CV would be unreliable)
            if (mean_cov < min_mean_cov) continue;

            double cov_sq_sum = 0.0;
            for (double cov : coverages) {
                cov_sq_sum += (cov - mean_cov) * (cov - mean_cov);
            }
            double cov_std = std::sqrt(cov_sq_sum / coverages.size());
            double cov_cv = cov_std / mean_cov;  // Coefficient of variation

            // Skip bins with homogeneous coverage (CV < threshold)
            if (cov_cv < min_cov_cv) continue;

            // Compute median coverage
            std::sort(coverages.begin(), coverages.end());
            double median_cov = coverages[coverages.size() / 2];

            // Coverage bounds (default: 60-140% of median)
            double min_cov = median_cov * (1.0 - cov_deviation);
            double max_cov = median_cov * (1.0 + cov_deviation);

            // Identify outliers
            std::vector<Contig*> kept_contigs;
            std::vector<Contig*> outlier_contigs;

            for (auto* c : bin.members) {
                if (c->mean_coverage < min_cov || c->mean_coverage > max_cov) {
                    outlier_contigs.push_back(c);
                } else {
                    kept_contigs.push_back(c);
                }
            }

            if (outlier_contigs.empty()) continue;
            if (kept_contigs.size() < bin.members.size() * kMinRetentionFraction) continue;

            // Calculate bp to remove
            size_t outlier_bp = 0;
            for (auto* c : outlier_contigs) {
                outlier_bp += c->length;
                c->bin_id = -1;
            }

            size_t orig_size = bin.members.size();
            bin.members = kept_contigs;
            bin.fit();

            if (verbose()) {
                std::cerr << "  bin_" << bin_idx << ": CV=" << std::fixed << std::setprecision(2) << cov_cv
                          << ", removed " << outlier_contigs.size()
                          << " coverage outliers (" << outlier_bp / 1000 << " kb), "
                          << "cov range [" << std::setprecision(1)
                          << min_cov << "-" << max_cov << "x], "
                          << orig_size << " -> " << kept_contigs.size() << " contigs\n";
            }

            bins_cleaned++;
            contigs_removed += outlier_contigs.size();
            bp_removed += outlier_bp;
        }

        log_.info("Coverage cleanup: " + std::to_string(bins_cleaned) + " bins, " +
                  std::to_string(contigs_removed) + " contigs removed");
    }

    // Feature-based contamination cleanup using CGR + fractal gated detection
    // Primary signal: CGR outliers at indices 21, 22, 23 (empirically most discriminative)
    // Secondary gate: fractal outliers at indices 1, 3 (reduces false positives)
    // Only removes contigs that are outliers in BOTH CGR AND fractal space
    void feature_based_cleanup() {
        log_.info("Running CGR+fractal gated cleanup...");

        // Use constants from cleanup_ops.h
        const size_t MIN_BIN_CONTIGS = cleanup::kMinBinContigsFeature;
        const size_t MIN_BIN_BP = cleanup::kMinBinBpFeature;

        // CGR indices that are most discriminative (from empirical analysis)
        const std::vector<size_t> CGR_INDICES = {21, 22, 23};
        // Fractal indices for confirmation gate
        const std::vector<size_t> FRACTAL_INDICES = {1, 3};

        int bins_cleaned = 0;
        int contigs_removed = 0;
        size_t bp_removed = 0;

        for (size_t bi = 0; bi < bins.size(); bi++) {
            auto& bin = bins[bi];
            if (bin.members.size() < MIN_BIN_CONTIGS) continue;

            size_t total_bp = 0;
            for (auto* c : bin.members) total_bp += c->length;
            if (total_bp < MIN_BIN_BP) continue;

            // Collect CGR values for discriminative indices
            std::vector<std::vector<double>> cgr_vals(CGR_INDICES.size());
            for (auto* c : bin.members) {
                if (c->cgr.size() >= 24) { // Need at least 24 CGR dimensions
                    for (size_t i = 0; i < CGR_INDICES.size(); i++) {
                        cgr_vals[i].push_back(c->cgr[CGR_INDICES[i]]);
                    }
                }
            }

            // Collect fractal values
            std::vector<std::vector<double>> fractal_vals(FRACTAL_INDICES.size());
            for (auto* c : bin.members) {
                if (c->fractal.size() >= 4) {
                    for (size_t i = 0; i < FRACTAL_INDICES.size(); i++) {
                        fractal_vals[i].push_back(c->fractal[FRACTAL_INDICES[i]]);
                    }
                }
            }

            // Skip if not enough valid contigs
            if (cgr_vals[0].size() < MIN_BIN_CONTIGS || fractal_vals[0].size() < MIN_BIN_CONTIGS) continue;

            // Compute statistics for each CGR index
            std::vector<std::pair<double, double>> cgr_stats(CGR_INDICES.size());
            for (size_t i = 0; i < CGR_INDICES.size(); i++) {
                cgr_stats[i] = compute_median_mad(cgr_vals[i]);
            }

            // Compute statistics for each fractal index
            std::vector<std::pair<double, double>> fractal_stats(FRACTAL_INDICES.size());
            for (size_t i = 0; i < FRACTAL_INDICES.size(); i++) {
                fractal_stats[i] = compute_median_mad(fractal_vals[i]);
            }

            // Identify outliers using gated detection
            std::vector<Contig*> kept_contigs;
            std::vector<Contig*> outlier_contigs;
            size_t outlier_bp = 0;

            for (auto* c : bin.members) {
                if (c->cgr.size() < 24 || c->fractal.size() < 4) {
                    kept_contigs.push_back(c);
                    continue;
                }

                // Primary signal: CGR outlier (any of indices 21, 22, 23 with z > threshold)
                bool cgr_outlier = false;
                for (size_t i = 0; i < CGR_INDICES.size(); i++) {
                    double z = (c->cgr[CGR_INDICES[i]] - cgr_stats[i].first) / cgr_stats[i].second;
                    if (z > cgr_z_threshold) {
                        cgr_outlier = true;
                        break;
                    }
                }

                // Secondary gate: fractal outlier (any of indices 1, 3 with z > threshold)
                bool fractal_outlier = false;
                for (size_t i = 0; i < FRACTAL_INDICES.size(); i++) {
                    double z = (c->fractal[FRACTAL_INDICES[i]] - fractal_stats[i].first) / fractal_stats[i].second;
                    if (z > fractal_z_threshold) {
                        fractal_outlier = true;
                        break;
                    }
                }

                // Only flag as outlier if BOTH signals agree (reduces false positives)
                bool is_outlier = cgr_outlier && fractal_outlier;

                if (is_outlier) {
                    outlier_contigs.push_back(c);
                    outlier_bp += c->length;
                } else {
                    kept_contigs.push_back(c);
                }
            }

            // Safety: don't remove more than max_removal_frac
            if (outlier_contigs.empty()) continue;
            if (outlier_contigs.size() > bin.members.size() * max_removal_frac) {
                log_.trace("Bin " + std::to_string(bi) + ": skipping cleanup, too many outliers (" +
                           std::to_string(outlier_contigs.size()) + "/" + std::to_string(bin.members.size()) + ")");
                continue;
            }

            // Apply cleanup
            for (auto* c : outlier_contigs) {
                c->bin_id = -1;
            }
            bin.members = kept_contigs;

            if (verbose()) {
                std::cerr << "  bin_" << bi << ": removed " << outlier_contigs.size()
                          << " CGR+fractal outliers (" << outlier_bp / 1000 << " kb), "
                          << outlier_contigs.size() + kept_contigs.size() << " -> " << kept_contigs.size() << " contigs\n";
            }

            bins_cleaned++;
            contigs_removed += outlier_contigs.size();
            bp_removed += outlier_bp;
        }

        if (bins_cleaned > 0) {
            log_.info("CGR+fractal cleanup: " + std::to_string(bins_cleaned) + " bins, " +
                      std::to_string(contigs_removed) + " contigs removed");
        }
    }

    // CpG-based contamination cleanup
    // CpG observed/expected ratio is a strong genomic signature that differs between species
    // Prokaryotes typically have CpG o/e < 1 due to methylation-driven deamination
    // Different species have characteristic CpG depletion patterns
    void cpg_based_cleanup() {
        log_.info("Running CpG-based cleanup...");

        // Use constants from cleanup_ops.h
        const size_t MIN_BIN_CONTIGS = cleanup::kMinBinContigsCpg;
        const double Z_THRESHOLD = cleanup::kCpgZThreshold;
        const double MAX_REMOVAL_FRAC = cleanup::kCpgMaxRemovalFrac;

        int bins_cleaned = 0;
        int contigs_removed = 0;
        size_t bp_removed = 0;

        for (size_t bi = 0; bi < bins.size(); bi++) {
            auto& bin = bins[bi];
            if (bin.members.size() < MIN_BIN_CONTIGS) continue;

            // Compute CpG o/e for all contigs in bin (using utility from cleanup_ops.h)
            std::vector<double> cpg_values;
            std::vector<double> cpg_per_contig(bin.members.size());

            for (size_t i = 0; i < bin.members.size(); i++) {
                double cpg_oe = compute_cpg_oe(bin.members[i]->sequence);
                cpg_per_contig[i] = cpg_oe;
                cpg_values.push_back(cpg_oe);
            }

            // Compute median and MAD
            std::vector<double> sorted_cpg = cpg_values;
            std::sort(sorted_cpg.begin(), sorted_cpg.end());
            double median = sorted_cpg[sorted_cpg.size() / 2];

            std::vector<double> deviations;
            for (double v : cpg_values) {
                deviations.push_back(std::abs(v - median));
            }
            std::sort(deviations.begin(), deviations.end());
            double mad = deviations[deviations.size() / 2] * 1.4826;  // Scale to std dev
            if (mad < 0.001) mad = 0.001;  // Minimum MAD to avoid division issues

            // Find outliers
            std::vector<Contig*> outliers;
            std::vector<Contig*> kept;
            size_t outlier_bp = 0;

            for (size_t i = 0; i < bin.members.size(); i++) {
                double z = std::abs(cpg_per_contig[i] - median) / mad;
                if (z > Z_THRESHOLD) {
                    outliers.push_back(bin.members[i]);
                    outlier_bp += bin.members[i]->length;
                } else {
                    kept.push_back(bin.members[i]);
                }
            }

            if (outliers.empty()) continue;

            // Safety: don't remove too many
            if (outliers.size() > bin.members.size() * MAX_REMOVAL_FRAC) {
                if (verbose()) {
                    log_.trace("Bin " + std::to_string(bi) + ": skipping CpG cleanup, too many outliers (" +
                               std::to_string(outliers.size()) + "/" + std::to_string(bin.members.size()) + ")");
                }
                continue;
            }

            // Apply removal and flag as contaminant (prevents rescue)
            for (auto* c : outliers) {
                c->bin_id = -1;
                c->flagged_contaminant = true;
            }
            bin.members = kept;

            if (verbose()) {
                std::cerr << "  bin_" << bi << ": removed " << outliers.size()
                          << " CpG outliers (" << outlier_bp / 1000 << " kb), median CpG="
                          << std::fixed << std::setprecision(3) << median << "\n";
            }

            bins_cleaned++;
            contigs_removed += outliers.size();
            bp_removed += outlier_bp;
        }

        if (bins_cleaned > 0) {
            log_.info("CpG cleanup: " + std::to_string(bins_cleaned) + " bins, " +
                      std::to_string(contigs_removed) + " contigs removed");
        }
    }

    // Karlin distance-based cleanup (uses all 16 dinucleotides)
    // More sensitive than CpG alone for detecting class-level contamination
    // Uses higher threshold than CpG since this is a second-pass cleanup
    void karlin_based_cleanup() {
        log_.info("Running Karlin-based cleanup...");

        // Use constants from cleanup_ops.h
        const size_t MIN_BIN_CONTIGS = cleanup::kMinBinContigsKarlin;
        const double Z_THRESHOLD = cleanup::kKarlinZThreshold;
        const double MAX_REMOVAL_FRAC = cleanup::kKarlinMaxRemovalFrac;

        int bins_cleaned = 0;
        int contigs_removed = 0;
        size_t bp_removed = 0;

        for (size_t bi = 0; bi < bins.size(); bi++) {
            auto& bin = bins[bi];
            if (bin.members.size() < MIN_BIN_CONTIGS) continue;

            // Compute centroid using utility from cleanup_ops.h
            std::array<double, 16> centroid = compute_karlin_centroid(bin.members);

            // Compute distance from centroid for each contig (using karlin_distance from cleanup_ops.h)
            std::vector<double> distances(bin.members.size());
            for (size_t i = 0; i < bin.members.size(); i++) {
                distances[i] = karlin_distance(bin.members[i]->dinuc_rho, centroid);
            }

            // Compute median and MAD
            std::vector<double> sorted_dists = distances;
            std::sort(sorted_dists.begin(), sorted_dists.end());
            double median = sorted_dists[sorted_dists.size() / 2];

            std::vector<double> deviations;
            for (double d : distances) {
                deviations.push_back(std::abs(d - median));
            }
            std::sort(deviations.begin(), deviations.end());
            double mad = deviations[deviations.size() / 2] * 1.4826;
            if (mad < 0.001) mad = 0.001;

            // Find outliers
            std::vector<Contig*> outliers;
            std::vector<Contig*> kept;
            size_t outlier_bp = 0;

            for (size_t i = 0; i < bin.members.size(); i++) {
                double z = (distances[i] - median) / mad;
                if (z > Z_THRESHOLD) {
                    outliers.push_back(bin.members[i]);
                    outlier_bp += bin.members[i]->length;
                } else {
                    kept.push_back(bin.members[i]);
                }
            }

            if (outliers.empty()) continue;

            // Safety: don't remove too many
            if (outliers.size() > bin.members.size() * MAX_REMOVAL_FRAC) {
                if (verbose()) {
                    log_.trace("Bin " + std::to_string(bi) + ": skipping Karlin cleanup, too many outliers (" +
                               std::to_string(outliers.size()) + "/" + std::to_string(bin.members.size()) + ")");
                }
                continue;
            }

            // Apply removal and flag as contaminant
            for (auto* c : outliers) {
                c->bin_id = -1;
                c->flagged_contaminant = true;
            }
            bin.members = kept;

            if (verbose()) {
                std::cerr << "  bin_" << bi << ": removed " << outliers.size()
                          << " Karlin outliers (" << outlier_bp / 1000 << " kb)\n";
            }

            bins_cleaned++;
            contigs_removed += outliers.size();
            bp_removed += outlier_bp;
        }

        if (bins_cleaned > 0) {
            log_.info("Karlin cleanup: " + std::to_string(bins_cleaned) + " bins, " +
                      std::to_string(contigs_removed) + " contigs removed");
        }
    }

    // V49: Damage-based contamination cleanup ("cryo separation")
    // For ancient bins (high damage amplitude), identifies modern contaminant contigs
    // that have significantly lower damage than the bin average.
    //
    // This catches modern DNA that was incorrectly binned with ancient material
    // based on composition similarity, but has no damage signature.
    //
    // Parameters:
    //   min_bin_damage: Only process bins with 5' amplitude above this (ancient bins)
    //   max_contig_damage: Contigs with damage below this in ancient bins are suspicious
    //   min_damage_ratio: Remove if contig_damage < bin_damage * ratio
    void damage_based_contamination_cleanup(
            double min_bin_damage = 0.08,
            double max_contig_damage = 0.02,
            double min_damage_ratio = 0.25) {
        log_.info("Running damage-based contamination cleanup (cryo separation)...");

        const size_t MIN_BIN_CONTIGS = 10;
        const double MAX_REMOVAL_FRAC = 0.15;

        int bins_cleaned = 0;
        int contigs_removed = 0;
        size_t bp_removed = 0;

        for (size_t bi = 0; bi < bins.size(); bi++) {
            auto& bin = bins[bi];
            if (bin.members.size() < MIN_BIN_CONTIGS) continue;

            // Compute bin-level damage statistics
            std::vector<double> damages;
            for (auto* c : bin.members) {
                if (c->has_damage && c->damage_rate_5p > 0) {
                    damages.push_back(c->damage_rate_5p);
                }
            }

            if (damages.size() < 5) continue;

            // Compute median damage for the bin
            std::sort(damages.begin(), damages.end());
            double bin_damage = damages[damages.size() / 2];

            // Only process ancient bins (high damage)
            if (bin_damage < min_bin_damage) continue;

            // Find contigs with suspiciously low damage
            std::vector<Contig*> kept;
            std::vector<Contig*> outliers;
            size_t outlier_bp = 0;

            for (auto* c : bin.members) {
                bool is_outlier = false;

                if (c->has_damage && c->damage_rate_5p >= 0) {
                    // Modern contamination: very low damage in ancient bin
                    if (c->damage_rate_5p < max_contig_damage) {
                        is_outlier = true;
                    }
                    // Or damage ratio too low compared to bin
                    else if (c->damage_rate_5p < bin_damage * min_damage_ratio) {
                        is_outlier = true;
                    }
                }

                if (is_outlier) {
                    outliers.push_back(c);
                    outlier_bp += c->length;
                } else {
                    kept.push_back(c);
                }
            }

            // Safety: don't remove too many
            if (outliers.empty()) continue;
            if (outliers.size() > bin.members.size() * MAX_REMOVAL_FRAC) continue;
            if (kept.size() < MIN_BIN_CONTIGS) continue;

            // Apply removal
            for (auto* c : outliers) {
                c->bin_id = -1;
                c->flagged_contaminant = true;
            }
            bin.members = kept;
            bin.fit();

            if (verbose()) {
                std::cerr << "  bin_" << bi << ": removed " << outliers.size()
                          << " low-damage contigs (" << outlier_bp / 1000 << " kb), "
                          << "bin_damage=" << std::fixed << std::setprecision(3) << bin_damage << "\n";
            }

            bins_cleaned++;
            contigs_removed += outliers.size();
            bp_removed += outlier_bp;
        }

        if (bins_cleaned > 0) {
            log_.info("Damage cleanup: " + std::to_string(bins_cleaned) + " bins, " +
                      std::to_string(contigs_removed) + " contigs removed (" +
                      std::to_string(bp_removed / 1000) + " kb)");
        }
    }

    // Graph-based contamination cleanup using Karlin distance with adaptive clustering
    // Detects minority clusters that are genomically distinct from the main population
    // Uses absolute Karlin distance thresholds based on literature:
    //   - Within species: ~0.03-0.05
    //   - Between species: >0.08
    // Algorithm:
    //   1. Compute pairwise Karlin distances (dinucleotide signature)
    //   2. Use absolute threshold for clustering (within-species variation)
    //   3. Hierarchical clustering (average linkage)
    //   4. Flag minority clusters with dist_to_main > 0.06 (inter-species boundary)
    void graph_based_cleanup() {
        log_.info("Running graph-based Karlin cleanup...");

        // Use constants from cleanup_ops.h
        const size_t MIN_BIN_CONTIGS = cleanup::kMinBinContigsGraph;
        const double CLUSTER_THRESHOLD = cleanup::kClusterThreshold;
        const double INTER_SPECIES_DIST = cleanup::kInterSpeciesDist;
        const double ISOLATION_THRESHOLD = cleanup::kIsolationThreshold;
        const size_t MIN_CLUSTER_SIZE = cleanup::kMinClusterSize;
        const double MAX_MINORITY_FRAC = cleanup::kMaxMinorityFrac;

        int bins_cleaned = 0;
        int contigs_removed = 0;
        size_t bp_removed = 0;

        for (size_t bi = 0; bi < bins.size(); bi++) {
            auto& bin = bins[bi];
            size_t n = bin.members.size();
            if (n < MIN_BIN_CONTIGS) continue;

            // Build distance matrix (using karlin_distance from cleanup_ops.h)
            std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0.0));
            std::vector<double> all_dists;
            all_dists.reserve(n * (n - 1) / 2);

            for (size_t i = 0; i < n; i++) {
                for (size_t j = i + 1; j < n; j++) {
                    double d = karlin_distance(bin.members[i]->dinuc_rho, bin.members[j]->dinuc_rho);
                    dist[i][j] = dist[j][i] = d;
                    all_dists.push_back(d);
                }
            }

            if (all_dists.empty()) continue;

            // Use fixed within-species threshold for clustering
            double threshold = CLUSTER_THRESHOLD;

            // Hierarchical clustering (average linkage)
            // Each contig starts in its own cluster
            std::vector<int> cluster_id(n);
            std::iota(cluster_id.begin(), cluster_id.end(), 0);
            int next_cluster = n;

            // Cluster sizes and merge tracking
            std::vector<size_t> cluster_size(n, 1);
            std::vector<std::vector<size_t>> cluster_members(n);
            for (size_t i = 0; i < n; i++) cluster_members[i] = {i};

            // Average linkage: distance between clusters = mean of pairwise distances
            auto cluster_dist = [&](int c1, int c2) -> double {
                if (cluster_members[c1].empty() || cluster_members[c2].empty()) return 1e9;
                double sum = 0.0;
                for (size_t i : cluster_members[c1]) {
                    for (size_t j : cluster_members[c2]) {
                        sum += dist[i][j];
                    }
                }
                return sum / (cluster_members[c1].size() * cluster_members[c2].size());
            };

            // Merge clusters until no pair is within threshold
            while (true) {
                // Find active clusters
                std::set<int> active;
                for (int c : cluster_id) active.insert(c);
                if (active.size() <= 1) break;

                // Find closest pair of clusters
                double min_dist = 1e9;
                int merge_a = -1, merge_b = -1;
                std::vector<int> active_list(active.begin(), active.end());

                for (size_t i = 0; i < active_list.size(); i++) {
                    for (size_t j = i + 1; j < active_list.size(); j++) {
                        double d = cluster_dist(active_list[i], active_list[j]);
                        if (d < min_dist) {
                            min_dist = d;
                            merge_a = active_list[i];
                            merge_b = active_list[j];
                        }
                    }
                }

                if (min_dist > threshold || merge_a < 0) break;

                // Merge clusters
                int new_id = next_cluster++;
                cluster_members.resize(next_cluster);
                cluster_size.resize(next_cluster);

                cluster_members[new_id] = cluster_members[merge_a];
                cluster_members[new_id].insert(cluster_members[new_id].end(),
                                               cluster_members[merge_b].begin(),
                                               cluster_members[merge_b].end());
                cluster_size[new_id] = cluster_members[new_id].size();

                for (size_t i : cluster_members[merge_a]) cluster_id[i] = new_id;
                for (size_t i : cluster_members[merge_b]) cluster_id[i] = new_id;

                cluster_members[merge_a].clear();
                cluster_members[merge_b].clear();
            }

            // Identify final clusters
            std::map<int, std::vector<size_t>> final_clusters;
            for (size_t i = 0; i < n; i++) {
                final_clusters[cluster_id[i]].push_back(i);
            }

            if (final_clusters.size() <= 1) continue;  // No separation

            // Find main cluster (largest)
            int main_cluster_id = -1;
            size_t main_size = 0;
            for (auto& [cid, members] : final_clusters) {
                if (members.size() > main_size) {
                    main_size = members.size();
                    main_cluster_id = cid;
                }
            }

            // Check minority clusters for contamination
            std::vector<Contig*> outliers;
            size_t outlier_bp = 0;

            for (auto& [cid, members] : final_clusters) {
                if (cid == main_cluster_id) continue;
                if (members.size() < MIN_CLUSTER_SIZE) continue;
                if (members.size() > n * MAX_MINORITY_FRAC) continue;

                // Compute isolation ratio: avg_dist_to_main / avg_internal_dist
                double sum_to_main = 0.0;
                int count_to_main = 0;
                for (size_t i : members) {
                    for (size_t j : final_clusters[main_cluster_id]) {
                        sum_to_main += dist[i][j];
                        count_to_main++;
                    }
                }
                double avg_to_main = count_to_main > 0 ? sum_to_main / count_to_main : 0.0;

                double sum_internal = 0.0;
                int count_internal = 0;
                for (size_t i = 0; i < members.size(); i++) {
                    for (size_t j = i + 1; j < members.size(); j++) {
                        sum_internal += dist[members[i]][members[j]];
                        count_internal++;
                    }
                }
                double avg_internal = count_internal > 0 ? sum_internal / count_internal : 1e-9;

                double isolation = avg_to_main / avg_internal;

                // Require BOTH conditions:
                // 1. Distance to main exceeds inter-species boundary (0.06)
                // 2. Isolation ratio is high (cluster is distinct from main)
                if (avg_to_main > INTER_SPECIES_DIST && isolation > ISOLATION_THRESHOLD) {
                    for (size_t idx : members) {
                        outliers.push_back(bin.members[idx]);
                        outlier_bp += bin.members[idx]->length;
                    }
                    if (verbose()) {
                        log_.trace("Bin " + std::to_string(bi) + ": cluster of " +
                                   std::to_string(members.size()) + " contigs, dist_to_main=" +
                                   std::to_string(avg_to_main).substr(0, 5) + ", isolation=" +
                                   std::to_string(isolation).substr(0, 4) + " -> flagged");
                    }
                }
            }

            if (outliers.empty()) continue;

            // Apply removal
            std::set<Contig*> outlier_set(outliers.begin(), outliers.end());
            std::vector<Contig*> kept;
            for (auto* c : bin.members) {
                if (outlier_set.count(c)) {
                    c->bin_id = -1;
                } else {
                    kept.push_back(c);
                }
            }
            bin.members = kept;

            if (verbose()) {
                std::cerr << "  bin_" << bi << ": removed " << outliers.size()
                          << " graph outliers (" << outlier_bp / 1000 << " kb), "
                          << kept.size() + outliers.size() << " -> " << kept.size() << " contigs\n";
            }

            bins_cleaned++;
            contigs_removed += outliers.size();
            bp_removed += outlier_bp;
        }

        if (bins_cleaned > 0) {
            log_.info("Graph-based cleanup: " + std::to_string(bins_cleaned) + " bins, " +
                      std::to_string(contigs_removed) + " contigs removed");
        }
    }

    // Post-merge rescue: recover scattered contigs that are compositionally similar to large bins
    // This helps recover contigs that got assigned to wrong bins due to coverage outliers
    void post_merge_rescue() {
        log_.info("Running post-merge rescue...");

        const double TNF_DIST_THRESHOLD = 0.15;   // Max TNF distance for rescue (conservative)
        const double COV_FOLD_THRESHOLD = 2.0;    // Max coverage fold difference (conservative)
        const size_t MIN_BIN_SIZE = 1000000;      // Only rescue into bins >= 1Mb
        const size_t MIN_RESCUE_SOURCE = 3;       // Don't rescue from bins with >= 3 contigs (keep them)

        // Compute TNF centroid and coverage median for each bin
        struct BinStats {
            std::vector<double> tnf_centroid;
            double median_cov = 0;
            double log_median_cov = 0;
            size_t total_bp = 0;
        };

        std::vector<BinStats> bin_stats(bins.size());

        for (size_t i = 0; i < bins.size(); i++) {
            auto& bin = bins[i];
            if (bin.members.empty()) continue;

            // Compute total bp
            for (auto* c : bin.members) {
                bin_stats[i].total_bp += c->length;
            }

            // Compute TNF centroid
            size_t tnf_dim = bin.members[0]->tnf.size();
            bin_stats[i].tnf_centroid.resize(tnf_dim, 0.0);
            for (auto* c : bin.members) {
                for (size_t j = 0; j < tnf_dim; j++) {
                    bin_stats[i].tnf_centroid[j] += c->tnf[j];
                }
            }
            for (size_t j = 0; j < tnf_dim; j++) {
                bin_stats[i].tnf_centroid[j] /= bin.members.size();
            }

            // Compute median coverage
            std::vector<double> covs;
            for (auto* c : bin.members) covs.push_back(c->mean_coverage);
            std::sort(covs.begin(), covs.end());
            bin_stats[i].median_cov = covs[covs.size() / 2];
            bin_stats[i].log_median_cov = std::log(bin_stats[i].median_cov + 1);
        }

        // Find rescue candidates: contigs in small bins or unbinned
        // IMPORTANT: Skip flagged contaminants - they were removed intentionally
        std::vector<Contig*> rescue_candidates;
        for (auto& bin : bins) {
            if (bin.members.size() < MIN_RESCUE_SOURCE) {
                for (auto* c : bin.members) {
                    if (!c->flagged_contaminant) {
                        rescue_candidates.push_back(c);
                    }
                }
            }
        }

        // Also include unbinned contigs (except flagged contaminants)
        for (auto& c : contigs) {
            if (c.bin_id < 0 && c.length >= 2500 && !c.flagged_contaminant) {
                rescue_candidates.push_back(&c);
            }
        }

        log_.detail("  Rescue candidates: " + std::to_string(rescue_candidates.size()) + " contigs");

        // For each large bin, try to rescue compatible contigs
        int total_rescued = 0;
        size_t bp_rescued = 0;

        for (size_t target_idx = 0; target_idx < bins.size(); target_idx++) {
            if (bin_stats[target_idx].total_bp < MIN_BIN_SIZE) continue;
            if (bins[target_idx].members.size() < 10) continue;

            auto& target_bin = bins[target_idx];
            auto& target_stats = bin_stats[target_idx];

            std::vector<Contig*> to_rescue;

            for (auto* c : rescue_candidates) {
                // Skip if already in target bin
                if (c->bin_id == (int)target_idx) continue;

                // Check coverage compatibility (within COV_FOLD_THRESHOLD fold)
                double log_cov = std::log(c->mean_coverage + 1);
                double cov_ratio = std::exp(std::abs(log_cov - target_stats.log_median_cov));
                if (cov_ratio > COV_FOLD_THRESHOLD) continue;

                // Check TNF distance
                double tnf_dist = 0;
                for (size_t j = 0; j < c->tnf.size() && j < target_stats.tnf_centroid.size(); j++) {
                    double d = c->tnf[j] - target_stats.tnf_centroid[j];
                    tnf_dist += d * d;
                }
                tnf_dist = std::sqrt(tnf_dist);

                if (tnf_dist < TNF_DIST_THRESHOLD) {
                    to_rescue.push_back(c);
                }
            }

            // Rescue contigs
            for (auto* c : to_rescue) {
                // Remove from old bin
                int old_bin = c->bin_id;
                if (old_bin >= 0 && old_bin < (int)bins.size()) {
                    auto& old_members = bins[old_bin].members;
                    old_members.erase(
                        std::remove(old_members.begin(), old_members.end(), c),
                        old_members.end());
                }

                // Add to target bin
                c->bin_id = target_idx;
                target_bin.members.push_back(c);
                c->record_fate("post_merge_rescue", old_bin, target_idx, "rescued_tnf_compatible", 0);

                total_rescued++;
                bp_rescued += c->length;
            }

            if (verbose() && !to_rescue.empty()) {
                std::cerr << "  bin_" << target_idx << ": rescued " << to_rescue.size()
                          << " contigs (" << (to_rescue.size() > 0 ? bp_rescued / to_rescue.size() / 1000 : 0) << " kb avg)\n";
            }
        }

        log_.info("Post-merge rescue: " + std::to_string(total_rescued) + " contigs rescued");
    }

    // Write bin statistics with validated intrinsic quality metrics
    // V62: downsample_var is PRIMARY contamination metric (r=+0.528)
    void write_bin_stats(const std::string& output_dir) {
        std::string filename = output_dir + "/bin_stats.tsv";
        std::ofstream out(filename);

        // Header - includes damage model parameters per bin and temporal chimera detection
        // Damage classification thresholds (per-contig):
        //   ANCIENT:   p_ancient >= 0.8 AND amplitude >= 1%
        //   MODERN:    p_ancient < 0.5
        //   UNCERTAIN: everything else (0.5 <= p_ancient < 0.8, or low amplitude)
        out << "bin_id\tnum_contigs\ttotal_bp\tmean_coverage\tcov_std\tmean_gc\tgc_std"
            << "\tdownsample_var\tcgr_occupancy\tmean_jsd\test_completeness\test_contamination"
            << "\tis_likely_clean\tis_likely_complete\tneeds_splitting"
            << "\tp_ancient\tdamage_class\tamplitude_5p\tlambda_5p\tamplitude_3p\tlambda_3p"
            << "\tn_ancient\tbp_ancient\tn_modern\tbp_modern\tn_uncertain\tbp_uncertain"
            << "\tdamage_std\ttemporal_chimera\ttemporal_outliers\n";

        const int max_pos = 20;
        int bin_num = 0;
        for (auto& bin : bins) {
            if (bin.members.size() < 2) continue;

            size_t bin_bp = 0;
            for (auto* c : bin.members) bin_bp += c->length;
            if (bin_bp < 200000) continue;

            // Compute coverage stats
            double cov_sum = 0, cov_sq_sum = 0;
            double gc_sum = 0, gc_sq_sum = 0;
            for (auto* c : bin.members) {
                cov_sum += c->mean_coverage;
                cov_sq_sum += c->mean_coverage * c->mean_coverage;
                gc_sum += c->gc;
                gc_sq_sum += c->gc * c->gc;
            }
            double n = bin.members.size();
            double mean_cov = cov_sum / n;
            double cov_std = std::sqrt(cov_sq_sum / n - mean_cov * mean_cov);
            double mean_gc = gc_sum / n;
            double gc_std = std::sqrt(gc_sq_sum / n - mean_gc * mean_gc);

            // Compute intrinsic quality metrics
            IntrinsicQuality iq = assess_intrinsic_quality(bin.members);

            // Aggregate damage stats for this bin
            std::vector<uint64_t> bin_n_5p(max_pos, 0), bin_k_5p(max_pos, 0);
            std::vector<uint64_t> bin_n_3p(max_pos, 0), bin_k_3p(max_pos, 0);

            if (damage_computed_) {
                for (auto* c : bin.members) {
                    auto it = damage_analysis_.per_contig_model.find(c->name);
                    if (it == damage_analysis_.per_contig_model.end()) continue;
                    const auto& model = it->second;
                    for (size_t i = 0; i < model.stats_5p.size() && i < (size_t)max_pos; i++) {
                        bin_n_5p[i] += model.stats_5p[i].n_opportunities;
                        bin_k_5p[i] += model.stats_5p[i].n_mismatches;
                    }
                    for (size_t i = 0; i < model.stats_3p.size() && i < (size_t)max_pos; i++) {
                        bin_n_3p[i] += model.stats_3p[i].n_opportunities;
                        bin_k_3p[i] += model.stats_3p[i].n_mismatches;
                    }
                }
            }

            // Fit damage model to pooled bin data
            DamageModelResult bin_damage;
            if (damage_computed_ && bin_n_5p[0] > 0) {
                bin_damage = fit_damage_model(bin_n_5p, bin_k_5p, bin_n_3p, bin_k_3p, 0.5);
            }

            // Temporal chimera detection: count per-contig damage classifications and bp
            // Uses configurable thresholds: ancient_p_threshold, modern_p_threshold, ancient_amp_threshold
            int n_ancient = 0, n_modern = 0, n_uncertain = 0;
            size_t bp_ancient = 0, bp_modern = 0, bp_uncertain = 0;
            std::vector<double> p_ancient_values;
            if (damage_computed_) {
                for (auto* c : bin.members) {
                    auto it = damage_analysis_.per_contig_model.find(c->name);
                    if (it != damage_analysis_.per_contig_model.end()) {
                        const auto& model = it->second;
                        p_ancient_values.push_back(model.p_ancient);

                        // Classify using configurable thresholds
                        double amp = std::max(model.curve_5p.amplitude, model.curve_3p.amplitude);
                        if (model.p_ancient < modern_p_threshold) {
                            n_modern++;
                            bp_modern += c->length;
                        } else if (model.p_ancient >= ancient_p_threshold && amp >= ancient_amp_threshold) {
                            n_ancient++;
                            bp_ancient += c->length;
                        } else {
                            n_uncertain++;
                            bp_uncertain += c->length;
                        }
                    }
                }
            }
            // Compute standard deviation of p_ancient across contigs
            double damage_std = 0.0;
            if (p_ancient_values.size() > 1) {
                double sum = 0, sq_sum = 0;
                for (double p : p_ancient_values) {
                    sum += p;
                    sq_sum += p * p;
                }
                double mean = sum / p_ancient_values.size();
                damage_std = std::sqrt(sq_sum / p_ancient_values.size() - mean * mean);
            }
            // Flag as temporal chimera if bin has both ancient AND modern contigs
            // Note: A bin with mostly ancient + few modern could indicate:
            // 1. True chimera (different organisms mixed)
            // 2. Time travelers (same lineage, ancient + modern descendants)
            // High damage_std suggests mixing; low std with few outliers suggests time travelers
            bool temporal_chimera = (n_ancient > 0 && n_modern > 0);

            // Detect "temporal outliers" scenario: mostly one class with few outliers
            // This suggests the bin is NOT a true chimera, just has minor temporal contamination
            // Could be: modern descendants, misclassified low-coverage contigs, or sample contamination
            bool temporal_outliers_only = temporal_chimera &&
                (((double)std::min(n_ancient, n_modern) / std::max(n_ancient, n_modern)) < temporal_minority_ratio) &&
                (damage_std < temporal_damage_std);

            out << "bin_" << bin_num
                << "\t" << bin.members.size()
                << "\t" << bin_bp
                << "\t" << std::fixed << std::setprecision(2) << mean_cov
                << "\t" << std::setprecision(2) << cov_std
                << "\t" << std::setprecision(4) << mean_gc
                << "\t" << std::setprecision(4) << gc_std
                << "\t" << std::setprecision(5) << iq.downsample_var
                << "\t" << std::setprecision(4) << iq.cgr_occupancy
                << "\t" << std::setprecision(4) << iq.mean_jsd
                << "\t" << std::setprecision(1) << (iq.completeness_est * 100)
                << "\t" << std::setprecision(1) << (iq.contamination_est * 100)
                << "\t" << (iq.is_likely_clean ? "yes" : "no")
                << "\t" << (iq.is_likely_complete ? "yes" : "no")
                << "\t" << (iq.needs_splitting ? "yes" : "no")
                << "\t" << std::setprecision(4) << bin_damage.p_ancient
                << "\t" << classification_to_string(bin_damage.classification)
                << "\t" << std::setprecision(4) << bin_damage.curve_5p.amplitude
                << "\t" << std::setprecision(4) << bin_damage.curve_5p.lambda
                << "\t" << std::setprecision(4) << bin_damage.curve_3p.amplitude
                << "\t" << std::setprecision(4) << bin_damage.curve_3p.lambda
                << "\t" << n_ancient << "\t" << bp_ancient
                << "\t" << n_modern << "\t" << bp_modern
                << "\t" << n_uncertain << "\t" << bp_uncertain
                << "\t" << std::setprecision(4) << damage_std
                << "\t" << (temporal_chimera ? "yes" : "no")
                << "\t" << (temporal_outliers_only ? "yes" : "no")
                << "\n";

            bin_num++;
        }

        std::cerr << "  Wrote bin statistics with validated quality metrics to " << filename << "\n";
    }

    void write_bins(const std::string& output_dir) {
        write_bins_impl(output_dir, "");
    }

    void write_bins_impl(const std::string& output_dir, const std::string& bam_path) {
        log_.section("Output");

        std::filesystem::create_directories(output_dir);

        // Initialize renamer if anvio mode is enabled
        ContigRenamer renamer(anvio_prefix);
        std::unordered_map<std::string, std::string> name_map;  // original -> new name

        if (anvio_mode) {
            log_.info("Anvi'o mode: renaming contigs with prefix '" + anvio_prefix + "'");
            // Pre-compute all name mappings
            for (auto& c : contigs) {
                std::string new_name = renamer.rename(c.name, c.length);
                name_map[c.name] = new_name;
            }
        }

        int bin_num = 0;
        size_t total_binned = 0;
        size_t total_bp_binned = 0;

        // Build mapping from internal bin_id to exported bin number
        std::unordered_map<int, int> internal_to_export_id;

        for (size_t bin_idx = 0; bin_idx < bins.size(); bin_idx++) {
            auto& bin = bins[bin_idx];
            if (bin.members.size() < 2) continue;

            size_t bin_bp = 0;
            for (auto* c : bin.members) bin_bp += c->length;
            if (bin_bp < 200000) continue;

            // Map internal bin_id (which equals bin_idx for contigs in this bin) to export number
            internal_to_export_id[bin_idx] = bin_num;

            std::string filename = output_dir + "/bin_" + std::to_string(bin_num) + ".fa";
            std::ofstream out(filename);

            for (auto* c : bin.members) {
                std::string output_name = anvio_mode ? name_map[c->name] : c->name;

                out << ">" << output_name << "\n";
                for (size_t i = 0; i < c->sequence.length(); i += 80) {
                    out << c->sequence.substr(i, 80) << "\n";
                }
            }

            total_binned += bin.members.size();
            total_bp_binned += bin_bp;
            bin_num++;
        }

        // Write unbinned contigs to separate file
        // Include: (1) contigs with no bin assignment, (2) contigs in bins too small to export
        size_t unbinned_count = 0;
        size_t unbinned_bp = 0;
        {
            std::string filename = output_dir + "/unbinned.fa";
            std::ofstream out(filename);
            for (auto& c : contigs) {
                bool is_unbinned = (c.bin_id < 0) ||
                    (internal_to_export_id.find(c.bin_id) == internal_to_export_id.end());
                if (is_unbinned) {
                    std::string output_name = anvio_mode ? name_map[c.name] : c.name;
                    out << ">" << output_name << "\n";
                    for (size_t i = 0; i < c.sequence.length(); i += 80) {
                        out << c.sequence.substr(i, 80) << "\n";
                    }
                    unbinned_count++;
                    unbinned_bp += c.length;
                }
            }
        }

        log_.subsection("Summary");
        log_.metric("total_bins", bin_num);
        log_.metric("total_contigs_binned", (int)total_binned);
        log_.metric("total_bp_binned", (int)total_bp_binned);
        log_.metric("unbinned_contigs", (int)unbinned_count);
        log_.metric("unbinned_bp", (int)unbinned_bp);

        std::cerr << "\n  Wrote " << bin_num << " bins (" << total_binned << " contigs, "
                  << total_bp_binned / 1000000.0 << " Mbp)";
        if (unbinned_count > 0) {
            std::cerr << " + unbinned.fa (" << unbinned_count << " contigs, "
                      << unbinned_bp / 1000000.0 << " Mbp)";
        }
        std::cerr << "\n";

        // Write rename report if anvio mode is enabled
        if (anvio_mode) {
            std::string report_path = output_dir + "/rename_report.tsv";
            renamer.write_report(report_path);
            log_.info("Wrote rename report: " + report_path);
        }

        // Write bin statistics with intrinsic quality metrics
        write_bin_stats(output_dir);

        // Write per-contig assignments with features for diagnostic analysis
        write_contig_assignments(output_dir, anvio_mode ? &name_map : nullptr);

        // Write comprehensive contig fate history to trace log
        write_fate_history_trace();

        // Write per-bin smiley data (after binning is complete)
        if (export_smiley_data && damage_computed_) {
            write_smiley_per_bin(output_dir, internal_to_export_id);
        }
    }

    void write_contig_assignments(const std::string& output_dir,
                                   const std::unordered_map<std::string, std::string>* name_map = nullptr) {
        std::string filename = output_dir + "/contig_assignments.tsv";
        std::ofstream out(filename);

        // COMPREHENSIVE HEADER with all features and stage tracking
        out << "contig\tfinal_bin\tkmeans_bin\tem_bin\tmerger_bin";
        out << "\tlength\tcoverage\tdamage_5p\tdamage_3p\tgc_content";
        // All 136 TNF features
        for (int i = 0; i < 136; i++) {
            out << "\ttnf_" << i;
        }
        // CGR features (32 dims)
        for (int i = 0; i < 32; i++) {
            out << "\tcgr_" << i;
        }
        // Fractal features (5 dims)
        for (int i = 0; i < 5; i++) {
            out << "\tfractal_" << i;
        }
        // DNA physical features (4 dims)
        out << "\tdna_bendability\tdna_flexibility\tdna_meltability\tdna_gc";
        // All stable features for merger (14 dims)
        out << "\tstable_bendability\tstable_lz_complexity\tstable_cgr_geom_var\tstable_cgr_fractal";
        out << "\tstable_flexibility\tstable_cgr_geom_mean\tstable_entropy_k6\tstable_kmer_fractal";
        out << "\tstable_entropy_k2\tstable_entropy_k3\tstable_entropy_k4\tstable_cgr_entropy\tstable_entropy_k5\tstable_meltability";
        out << "\n";

        // Build bin name mapping (original bin_id -> output bin number)
        std::map<int, int> bin_name_map;
        int out_bin_num = 0;
        for (size_t i = 0; i < bins.size(); i++) {
            if (bins[i].members.size() < 2) continue;
            size_t bin_bp = 0;
            for (auto* c : bins[i].members) bin_bp += c->length;
            if (bin_bp < 200000) continue;
            bin_name_map[i] = out_bin_num++;
        }

        // Build a mapping for all bin_ids to track intermediate stages
        std::map<int, std::string> all_bin_names;
        for (size_t i = 0; i < bins.size(); i++) {
            auto it = bin_name_map.find(i);
            if (it != bin_name_map.end()) {
                all_bin_names[i] = "bin_" + std::to_string(it->second);
            } else {
                all_bin_names[i] = "unbinned_" + std::to_string(i);
            }
        }

        // Output each contig with comprehensive features
        for (const auto& c : contigs) {
            // Use renamed name if name_map is provided
            std::string output_name = c.name;
            if (name_map) {
                auto name_it = name_map->find(c.name);
                if (name_it != name_map->end()) {
                    output_name = name_it->second;
                }
            }
            out << output_name << "\t";

            // Final bin assignment
            auto it = bin_name_map.find(c.bin_id);
            if (it != bin_name_map.end()) {
                out << "bin_" << it->second;
            } else {
                out << "unbinned";
            }

            // Stage tracking: K-means, EM, and Merger assignments (using internal IDs)
            out << "\t" << c.kmeans_bin_id;
            out << "\t" << c.em_bin_id;
            out << "\t" << c.final_bin_id;

            // Core features
            out << "\t" << c.length;
            out << "\t" << std::fixed << std::setprecision(4) << c.mean_coverage;
            out << "\t" << std::setprecision(6) << c.damage_rate_5p;
            out << "\t" << std::setprecision(6) << c.damage_rate_3p;
            out << "\t" << std::setprecision(6) << c.gc;

            // All 136 TNF features
            for (int i = 0; i < 136; i++) {
                if (i < (int)c.tnf.size()) {
                    out << "\t" << std::setprecision(8) << c.tnf[i];
                } else {
                    out << "\tNA";
                }
            }

            // CGR features (32 dims)
            for (int i = 0; i < 32; i++) {
                if (i < (int)c.cgr.size()) {
                    out << "\t" << std::setprecision(8) << c.cgr[i];
                } else {
                    out << "\tNA";
                }
            }

            // Fractal features (5 dims: CGR-DFA)
            for (int i = 0; i < 5; i++) {
                if (i < (int)c.fractal.size()) {
                    out << "\t" << std::setprecision(8) << c.fractal[i];
                } else {
                    out << "\tNA";
                }
            }

            // DNA physical features (4 dims)
            for (int i = 0; i < 4; i++) {
                if (i < (int)c.dna_physical.size()) {
                    out << "\t" << std::setprecision(8) << c.dna_physical[i];
                } else {
                    out << "\tNA";
                }
            }

            // All stable features (used by merger, 14 dims)
            for (int i = 0; i < STABLE_DIM; i++) {
                if (i < (int)c.stable.size()) {
                    out << "\t" << std::setprecision(8) << c.stable[i];
                } else {
                    out << "\tNA";
                }
            }

            out << "\n";
        }

        std::cerr << "  Wrote comprehensive contig assignments with stage tracking to " << filename << "\n";
    }

    void write_fate_history_trace() {
        log_.section("Contig Fate History");

        // Summary statistics
        int total_contigs = contigs.size();
        int contigs_with_history = 0;
        int total_events = 0;
        std::map<std::string, int> stage_counts;

        for (const auto& c : contigs) {
            if (!c.fate_history.empty()) {
                contigs_with_history++;
                total_events += c.fate_history.size();
                for (const auto& e : c.fate_history) {
                    stage_counts[e.stage]++;
                }
            }
        }

        log_.metric("total_contigs", total_contigs);
        log_.metric("contigs_with_history", contigs_with_history);
        log_.metric("total_fate_events", total_events);

        for (const auto& [stage, count] : stage_counts) {
            log_.metric("events_" + stage, count);
        }

        // Write detailed fate history table
        log_.trace("");
        log_.table_header("CONTIG_FATE_HISTORY",
            {"contig", "length", "coverage", "gc", "final_bin", "n_events", "event_chain"});

        for (const auto& c : contigs) {
            std::ostringstream event_chain;
            for (size_t i = 0; i < c.fate_history.size(); i++) {
                const auto& e = c.fate_history[i];
                if (i > 0) event_chain << " -> ";
                event_chain << e.stage << ":" << e.to_bin;
            }

            std::ostringstream row;
            row << c.name << "\t" << c.length << "\t"
                << std::fixed << std::setprecision(2) << c.mean_coverage << "\t"
                << std::setprecision(4) << c.gc << "\t"
                << c.bin_id << "\t"
                << c.fate_history.size() << "\t"
                << event_chain.str();
            log_.trace(row.str());
        }
        log_.trace("[END_TABLE:CONTIG_FATE_HISTORY]");
    }

    void assign_short_contigs(std::vector<Contig*>& short_contigs, int stage2_min_length = 2500) {
        if (short_contigs.empty() || bins.empty()) return;

        log_.section("Short contig assignment");
        log_.detail("Short contigs: " + std::to_string(short_contigs.size()) +
                    ", Anchor bins: " + std::to_string(bins.size()));

        // Pre-compute bin centroids for each metric
        struct BinCentroid {
            std::vector<double> tnf_mean;
            std::vector<double> cgr_mean;
            double cov_mean = 0.0;
            int n_members = 0;
        };
        std::vector<BinCentroid> centroids(bins.size());

        for (size_t bi = 0; bi < bins.size(); bi++) {
            BinCentroid& c = centroids[bi];
            c.tnf_mean.assign(TNF_DIM, 0.0);
            c.cgr_mean.assign(CGR_DIM, 0.0);
            int tnf_n = 0, cgr_n = 0, cov_n = 0;
            double cov_sum = 0.0;

            for (auto* m : bins[bi].members) {
                if (m->length >= (size_t)stage2_min_length) {  // Only long contigs for centroids
                    if (!m->tnf.empty()) {
                        tnf_n++;
                        for (int k = 0; k < TNF_DIM; k++) c.tnf_mean[k] += m->tnf[k];
                    }
                    if (!m->cgr.empty()) {
                        cgr_n++;
                        for (int k = 0; k < CGR_DIM; k++) c.cgr_mean[k] += m->cgr[k];
                    }
                    if (m->mean_coverage > 0) {
                        cov_sum += m->mean_coverage;
                        cov_n++;
                    }
                }
            }
            if (tnf_n > 0) for (double& v : c.tnf_mean) v /= tnf_n;
            if (cgr_n > 0) for (double& v : c.cgr_mean) v /= cgr_n;
            c.cov_mean = cov_n > 0 ? cov_sum / cov_n : 0.0;
            c.n_members = tnf_n;
        }

        // L2 distance helper
        auto l2_dist = [](const std::vector<double>& a, const std::vector<double>& b) -> double {
            double d = 0;
            for (size_t i = 0; i < a.size() && i < b.size(); i++) {
                double diff = a[i] - b[i];
                d += diff * diff;
            }
            return std::sqrt(d);
        };

        int assigned = 0;
        int unbinned = 0;
        int consensus_2of3 = 0;
        int consensus_3of3 = 0;

        for (auto* c : short_contigs) {
            // Vote 1: TNF-based (composition)
            int tnf_vote = -1;
            double tnf_best_dist = std::numeric_limits<double>::max();
            if (!c->tnf.empty()) {
                for (size_t bi = 0; bi < bins.size(); bi++) {
                    if (centroids[bi].n_members < 2) continue;
                    double d = l2_dist(c->tnf, centroids[bi].tnf_mean);
                    if (d < tnf_best_dist) {
                        tnf_best_dist = d;
                        tnf_vote = bi;
                    }
                }
            }

            // Vote 2: CGR-based (fractal composition)
            int cgr_vote = -1;
            double cgr_best_dist = std::numeric_limits<double>::max();
            if (!c->cgr.empty()) {
                for (size_t bi = 0; bi < bins.size(); bi++) {
                    if (centroids[bi].n_members < 2) continue;
                    double d = l2_dist(c->cgr, centroids[bi].cgr_mean);
                    if (d < cgr_best_dist) {
                        cgr_best_dist = d;
                        cgr_vote = bi;
                    }
                }
            }

            // Vote 3: Coverage-based (abundance)
            int cov_vote = -1;
            double cov_best_ratio = std::numeric_limits<double>::max();
            if (c->mean_coverage > 0) {
                for (size_t bi = 0; bi < bins.size(); bi++) {
                    if (centroids[bi].n_members < 2 || centroids[bi].cov_mean <= 0) continue;
                    // Coverage ratio (fold change)
                    double ratio = std::max(c->mean_coverage, centroids[bi].cov_mean) /
                                   (std::min(c->mean_coverage, centroids[bi].cov_mean) + 1e-10);
                    if (ratio < cov_best_ratio) {
                        cov_best_ratio = ratio;
                        cov_vote = bi;
                    }
                }
            }

            // Tug-of-War consensus: require 2/3 agreement
            int final_bin = -1;
            int agreement_count = 0;

            // Check if any bin got at least 2 votes
            std::map<int, int> vote_counts;
            if (tnf_vote >= 0) vote_counts[tnf_vote]++;
            if (cgr_vote >= 0) vote_counts[cgr_vote]++;
            if (cov_vote >= 0) vote_counts[cov_vote]++;

            for (auto& [bin, count] : vote_counts) {
                if (count >= 2) {
                    final_bin = bin;
                    agreement_count = count;
                    break;
                }
            }

            // Additional gate: coverage must be within 3x even with consensus
            if (final_bin >= 0 && c->mean_coverage > 0 && centroids[final_bin].cov_mean > 0) {
                double cov_ratio = std::max(c->mean_coverage, centroids[final_bin].cov_mean) /
                                   (std::min(c->mean_coverage, centroids[final_bin].cov_mean) + 1e-10);
                if (cov_ratio > 3.0) {
                    final_bin = -1;  // Coverage too different, reject
                }
            }

            if (final_bin >= 0) {
                c->bin_id = final_bin;
                bins[final_bin].members.push_back(c);
                assigned++;
                if (agreement_count == 3) consensus_3of3++;
                else consensus_2of3++;
            } else {
                c->bin_id = -1;  // Leave unbinned
                unbinned++;
            }
        }

        std::cerr << "  Assigned " << assigned << " short contigs (3/3 consensus: "
                  << consensus_3of3 << ", 2/3 consensus: " << consensus_2of3 << ")\n";
        std::cerr << "  Left " << unbinned << " short contigs unbinned (no consensus)\n";
    }

    void run(const std::string& fasta_path, const std::string& bam_path,
             const std::string& output_dir) {
        std::filesystem::create_directories(output_dir);
        log_.open_trace(output_dir + "/binner_trace.log");
        log_.info("AMBER v" + std::string(VERSION) + " starting");

        load_contigs(fasta_path);
        load_coverage_data(bam_path);
        compute_damage_from_bam(bam_path);

        if (damage_computed_) {
            write_damage_outputs(output_dir);
            if (export_smiley_data) {
                write_smiley_plot_data(output_dir);
            }
        }

        extract_features();
        normalize_stable_features();  // Z-score normalize stable features for merger

        // Two-stage binning:
        // Stage 1: Bin LONG contigs (>= kStage2LengthThreshold) with full features including DFA
        // Stage 2: Assign SHORT contigs (min_length to < threshold) using only stable features
        std::vector<Contig*> long_contig_ptrs;
        std::vector<Contig*> short_contig_ptrs;
        for (auto& c : contigs) {
            if (c.length >= (size_t)kStage2LengthThreshold) {
                long_contig_ptrs.push_back(&c);
            } else {
                short_contig_ptrs.push_back(&c);
            }
        }
        log_.section("Two-stage binning");
        log_.info("Stage 1: " + std::to_string(long_contig_ptrs.size()) + " long contigs (>=" +
                  std::to_string(kStage2LengthThreshold) + "bp)");
        log_.detail("Stage 2: " + std::to_string(short_contig_ptrs.size()) + " short contigs");


        auto k_est = estimate_k_data_driven(long_contig_ptrs);
        log_.decision("K_ESTIMATION", "k=" + std::to_string(k_est.combined),
                      "cov_modes=" + std::to_string(k_est.coverage_modes) +
                      " comp_clusters=" + std::to_string(k_est.composition_clusters));
        log_.info("Estimated K=" + std::to_string(k_est.combined));

        std::vector<Contig*> original_contig_ptrs;
        for (auto& c : contigs) original_contig_ptrs.push_back(&c);

        for (auto* c : short_contig_ptrs) {
            c->bin_id = -2;
        }

        // Use ensemble clustering if enabled (more robust to feature perturbations)
        if (ensemble_runs > 1) {
            log_.info("Running ensemble clustering (" + std::to_string(ensemble_runs) + " runs)");
            run_ensemble_clustering(k_est.combined, ensemble_runs);
        } else {
            log_.info("Initializing clusters (Farthest-First)");
            run_farthest_first_init(k_est.combined);

            // Run EM on long contigs only (contigs with bin_id == -2 are skipped)
            run_em(20);
        }

        // Merge fragmented bins using stable features (optional)
        if (use_merge) {
            merge_bins_resonance(merge_posterior_threshold, 2.0, damage_relaxed_threshold);
        }

        // Temporal chimera rescue: rescue contigs with unusual damage using stable features
        // This handles same-genome contigs split across temporal cohorts (ancient/modern)
        temporal_chimera_rescue();

        // STAGE TRACKING: Save post-merger assignment for long contigs
        for (auto& c : contigs) {
            if (c.bin_id >= 0) {
                c.final_bin_id = c.bin_id;
            }
        }
        std::cerr << "  Saved post-merger assignments for " << long_contig_ptrs.size() << " long contigs\n";

        // Log intrinsic quality metrics (CGR occupancy + De Bruijn modularity)
        log_intrinsic_quality();

        // Targeted contamination refinement
        // Uses CGR occupancy + De Bruijn modularity to identify contaminated bins,
        // then relocates outlier contigs using TNF + coverage + variance
        if (use_refinement) {
            targeted_contamination_refinement(refinement_rounds);

            // Update final assignments after refinement
            for (auto& c : contigs) {
                if (c.bin_id >= 0) {
                    c.final_bin_id = c.bin_id;
                }
            }

            // Log final intrinsic quality after refinement
            std::cerr << "\n  === INTRINSIC QUALITY AFTER REFINEMENT (LONG CONTIGS ONLY) ===\n";
            log_intrinsic_quality();
        }

        // Iterative Lazarus Rescue Phase (2 passes)
        // Pass 1: TNF < 0.10, 3+ signals
        // Pass 2: TNF < 0.08, 4 signals (stricter)
        lazarus_rescue_phase();

        // Update assignments after cleaning
        for (auto& c : contigs) {
            if (c.bin_id >= 0) {
                c.final_bin_id = c.bin_id;
            }
        }

        // STAGE 2: Assign short contigs to existing bins (no DFA)
        log_.info("Stage 2: Assigning short contigs...");
        assign_short_contigs(short_contig_ptrs, kStage2LengthThreshold);

        // Update final assignments for short contigs
        for (auto* c : short_contig_ptrs) {
            c->final_bin_id = c->bin_id;
        }

        // Log final stats
        int assigned_shorts = 0, unbinned_shorts = 0;
        for (auto* c : short_contig_ptrs) {
            if (c->bin_id >= 0) assigned_shorts++;
            else unbinned_shorts++;
        }
        log_.section("Cleanup");
        coverage_aware_contamination_cleanup();
        tnf_based_contamination_cleanup();
        coverage_outlier_cleanup();
        feature_based_cleanup();
        cpg_based_cleanup();
        karlin_based_cleanup();
        damage_based_contamination_cleanup();  // V49: Cryo separation
        // graph_based_cleanup();  // Disabled - too aggressive
        post_merge_rescue();

        write_bins_impl(output_dir, bam_path);
    }
};

// ========== Main ==========
void print_usage([[maybe_unused]] const char* prog) {
    std::cerr << "AMBER bin - Ancient Metagenomic Binning Engine v" << VERSION << "\n\n";
    std::cerr << "Ancient DNA metagenomic binner with multi-stage contamination cleanup.\n\n";
    std::cerr << "Pipeline:\n";
    std::cerr << "  FFT init -> EM -> Resonance Merger -> Coverage-aware Cleanup ->\n";
    std::cerr << "  TNF Cleanup -> CGR+Fractal Cleanup -> Post-merge Rescue -> Short Assign\n\n";
    std::cerr << "Usage: amber bin [options]\n\n";
    std::cerr << "Required:\n";
    std::cerr << "  --contigs FILE     Input contigs FASTA\n";
    std::cerr << "  --bam FILE         BAM file for coverage\n";
    std::cerr << "  --output DIR       Output directory\n\n";
    std::cerr << "Optional:\n";
    std::cerr << "  --damage FILE      Damage profile (bdamage format)\n";
    std::cerr << "  --threads N        Number of threads (default: 1)\n";
    std::cerr << "  --min-length N     Minimum contig length (default: 2000)\n";
    std::cerr << "  --seed N           Random seed (default: 42)\n";
    std::cerr << "  --ensemble N       Ensemble runs for robust clustering (default: 1=off)\n";
    std::cerr << "  --cov-weight W     Coverage weight in distance (default: 1.0)\n\n";
    std::cerr << "Merger options:\n";
    std::cerr << "  --ks-threshold P   KS test p-value threshold for damage match (default: 0.8)\n";
    std::cerr << "  --merge-threshold P  Bayesian GMM posterior threshold (default: 0.8)\n";
    std::cerr << "  --relaxed-threshold P  Posterior threshold when damage matches (default: 0.5)\n";
    std::cerr << "  --no-merge         Disable resonance merger\n\n";
    std::cerr << "Refinement options:\n";
    std::cerr << "  --refinement-rounds N  Targeted refinement rounds (default: 5)\n";
    std::cerr << "  --no-refinement        Disable targeted contamination refinement\n\n";
    std::cerr << "Feature options:\n";
    std::cerr << "  --no-fractal       Disable fractal DFA features\n\n";
    std::cerr << "Cleanup thresholds:\n";
    std::cerr << "  --cgr-z Z          CGR outlier z-score threshold (default: 3.0)\n";
    std::cerr << "  --fractal-z Z      Fractal confirmation z-score threshold (default: 1.5)\n";
    std::cerr << "  --max-removal F    Max fraction of bin to remove (default: 0.15)\n\n";
    std::cerr << "Per-bin polishing (corrects damage using bin-level models):\n";
    std::cerr << "  --polish           Enable per-bin polishing in output FASTAs\n";
    std::cerr << "  --polish-bam FILE  BAM file for polishing (default: same as --bam)\n";
    std::cerr << "  --polish-min-lr N  Min log-likelihood ratio for correction (default: 3.0)\n";
    std::cerr << "  --polish-positions N  Damage positions to correct (default: 15)\n\n";
    std::cerr << "Anvi'o compatibility:\n";
    std::cerr << "  --anvio            Enable anvi'o-compatible contig naming\n";
    std::cerr << "  --prefix STR       Prefix for anvi'o contig names (default: 'c')\n\n";
    std::cerr << "Output options:\n";
    std::cerr << "  --verbose          Show detailed progress\n";
    std::cerr << "  --quiet            Minimal output (errors only)\n";
    std::cerr << "  --export-smiley    Export data for smiley damage plots\n";
    std::cerr << "  --help             Show this help\n";
}

int run_binner(int argc, char** argv) {
    MFDFABinner binner;

    // Set namespace-level logger for standalone functions
    g_log = &binner.log_;

    std::string fasta_path, bam_path, output_dir;
    bool no_fractal = false;
    bool no_merge = false;
    bool no_refinement = false;
    double merge_threshold = 0.8;
    double relaxed_threshold = 0.5;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--contigs" && i + 1 < argc) {
            fasta_path = argv[++i];
        } else if (arg == "--bam" && i + 1 < argc) {
            bam_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            binner.n_threads = std::stoi(argv[++i]);
        } else if (arg == "--min-length" && i + 1 < argc) {
            binner.min_length = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            binner.random_seed = std::stoi(argv[++i]);
        } else if (arg == "--ensemble" && i + 1 < argc) {
            binner.ensemble_runs = std::stoi(argv[++i]);
        } else if (arg == "--merge-threshold" && i + 1 < argc) {
            merge_threshold = std::stod(argv[++i]);
        } else if (arg == "--relaxed-threshold" && i + 1 < argc) {
            relaxed_threshold = std::stod(argv[++i]);
        } else if (arg == "--cov-weight" && i + 1 < argc) {
            binner.cov_weight = std::stod(argv[++i]);
        } else if (arg == "--ks-threshold" && i + 1 < argc) {
            binner.ks_pvalue_threshold = std::stod(argv[++i]);
        } else if (arg == "--no-fractal") {
            no_fractal = true;
        } else if (arg == "--no-merge") {
            no_merge = true;
        } else if (arg == "--no-refinement") {
            no_refinement = true;
        } else if (arg == "--refinement-rounds" && i + 1 < argc) {
            binner.refinement_rounds = std::stoi(argv[++i]);
        } else if (arg == "--library-type" && i + 1 < argc) {
            std::string lt = argv[++i];
            if (lt != "ds" && lt != "ss") {
                binner.log_.error("--library-type must be 'ds' or 'ss'");
                return 1;
            }
            binner.library_type_override = lt;
        } else if (arg == "--export-smiley") {
            binner.export_smiley_data = true;
        } else if (arg == "--anvio") {
            binner.anvio_mode = true;
        } else if (arg == "--prefix" && i + 1 < argc) {
            binner.anvio_prefix = argv[++i];
        } else if (arg == "--temporal-minority-ratio" && i + 1 < argc) {
            binner.temporal_minority_ratio = std::stod(argv[++i]);
        } else if (arg == "--temporal-damage-std" && i + 1 < argc) {
            binner.temporal_damage_std = std::stod(argv[++i]);
        } else if (arg == "--ancient-p-threshold" && i + 1 < argc) {
            binner.ancient_p_threshold = std::stod(argv[++i]);
        } else if (arg == "--modern-p-threshold" && i + 1 < argc) {
            binner.modern_p_threshold = std::stod(argv[++i]);
        } else if (arg == "--ancient-amp-threshold" && i + 1 < argc) {
            binner.ancient_amp_threshold = std::stod(argv[++i]);
        } else if (arg == "--cgr-z" && i + 1 < argc) {
            binner.cgr_z_threshold = std::stod(argv[++i]);
        } else if (arg == "--fractal-z" && i + 1 < argc) {
            binner.fractal_z_threshold = std::stod(argv[++i]);
        } else if (arg == "--max-removal" && i + 1 < argc) {
            binner.max_removal_frac = std::stod(argv[++i]);
        } else if (arg == "--verbose" || arg == "-v") {
            binner.log_.console_level = Verbosity::Verbose;
        } else if (arg == "--quiet" || arg == "-q") {
            binner.log_.console_level = Verbosity::Quiet;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (fasta_path.empty() || bam_path.empty() || output_dir.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Validate required input files exist
    {
        std::ifstream f(fasta_path);
        if (!f.good()) {
            binner.log_.error("Cannot open contigs file: " + fasta_path);
            return 1;
        }
    }
    {
        std::ifstream f(bam_path);
        if (!f.good()) {
            binner.log_.error("Cannot open BAM file: " + bam_path);
            return 1;
        }
    }

    binner.use_fractal = !no_fractal;
    binner.use_merge = !no_merge;
    binner.use_refinement = !no_refinement;  // V11
    binner.merge_posterior_threshold = merge_threshold;
    binner.damage_relaxed_threshold = relaxed_threshold;
    
    // V31: FFT-only (K-means/K-medoids removed)
    // use_farthest_first is always true now

    binner.run(fasta_path, bam_path, output_dir);

    return 0;
}

}  // namespace amber

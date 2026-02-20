// AMBER bin2 - Damage-Aware InfoNCE Loss
// Confidence-gated negative weighting based on damage compatibility
// Based on OpenCode collaboration (Feb 2026)
#pragma once

#ifdef USE_LIBTORCH

#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace amber::bin2 {

// Per-contig aDNA damage profile with comprehensive ancient DNA features
// This is AMBER-specific - not in COMEBin
struct DamageProfile {
    // === Basic damage statistics ===
    float p_anc = 0.5f;     // Posterior mean ancient fraction
    float var_anc = 0.05f;  // Posterior variance (inflated for soft labels)
    float n_eff = 0.0f;     // Effective informative read depth (Kish's)
    float cov_ancient = 0.0f;
    float cov_modern = 0.0f;
    float log_ratio = 0.0f; // log((cov_anc+eps)/(cov_mod+eps))

    // === Per-position damage rates (smiley plot) ===
    // Position 0 = terminal base, position 9 = 10th base from end
    float ct_rate_5p[10] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};  // C→T at 5'
    float ga_rate_3p[10] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};  // G→A at 3'

    // === aDNA-specific features (AMBER novel) ===

    // Damage decay parameters (fitted exponential: rate = amp * exp(-lambda * pos) + baseline)
    float lambda_5p = 0.5f;     // Decay rate at 5' end
    float lambda_3p = 0.5f;     // Decay rate at 3' end
    float amplitude_5p = 0.1f;  // Damage amplitude at 5'
    float amplitude_3p = 0.1f;  // Damage amplitude at 3'

    // Fragment length distribution
    float frag_len_mean = 50.0f;   // Mean read length
    float frag_len_std = 20.0f;    // Std dev of read lengths
    float frag_len_median = 50.0f; // Median read length

    // Damage-stratified coverage
    float cov_damaged_reads = 0.0f;    // Coverage from reads with p_anc > 0.6
    float cov_undamaged_reads = 0.0f;  // Coverage from reads with p_anc < 0.4
    float damage_cov_ratio = 1.0f;     // Ratio (genome-specific signature)

    // Terminal mismatch spectrum (beyond C→T and G→A)
    // 5' end mismatches (ref→read)
    float mm_5p_tc = 0.0f;  // T→C (reverse complement damage on opposite strand)
    float mm_5p_ag = 0.0f;  // A→G
    float mm_5p_other = 0.0f; // All other mismatches
    // 3' end mismatches
    float mm_3p_ct = 0.0f;  // C→T (reverse complement)
    float mm_3p_tc = 0.0f;  // T→C
    float mm_3p_other = 0.0f;

    // Read counts for quality assessment
    int n_reads = 0;
    int n_damaged_reads = 0;
    int n_undamaged_reads = 0;

    // Raw counts for bin-level damage aggregation (pooled then re-fitted)
    uint64_t n_opp_5p[10] = {0};  // C opportunities at 5' positions 0-9
    uint64_t n_ct_5p[10]  = {0};  // C→T mismatches at 5' positions 0-9
    uint64_t n_opp_3p[10] = {0};  // G opportunities at 3' positions 0-9
    uint64_t n_ga_3p[10]  = {0};  // G→A mismatches at 3' positions 0-9

    // === Helper methods ===
    float mean_5p_damage() const {
        return (ct_rate_5p[0] + ct_rate_5p[1] + ct_rate_5p[2]) / 3.0f;
    }
    float mean_3p_damage() const {
        return (ga_rate_3p[0] + ga_rate_3p[1] + ga_rate_3p[2]) / 3.0f;
    }
    float combined_damage() const {
        return (mean_5p_damage() + mean_3p_damage()) / 2.0f;
    }

    // Get aDNA feature vector (20 dims for encoder input)
    std::vector<float> get_adna_features() const {
        std::vector<float> feats;
        feats.reserve(20);

        // Damage profile (10 dims) - first 5 positions from each end
        for (int i = 0; i < 5; i++) feats.push_back(ct_rate_5p[i]);
        for (int i = 0; i < 5; i++) feats.push_back(ga_rate_3p[i]);

        // Damage decay (2 dims)
        feats.push_back(lambda_5p);
        feats.push_back(lambda_3p);

        // Fragment length (2 dims) - normalized
        feats.push_back(frag_len_mean / 150.0f);  // Normalize to ~1
        feats.push_back(frag_len_std / 50.0f);

        // Damage-stratified coverage (2 dims)
        feats.push_back(std::log1p(cov_damaged_reads) / 5.0f);
        feats.push_back(std::log1p(cov_undamaged_reads) / 5.0f);

        // Mismatch spectrum (4 dims) - key ratios
        feats.push_back(mm_5p_tc);  // Strand-specific damage indicator
        feats.push_back(mm_3p_ct);
        feats.push_back(mm_5p_other);
        feats.push_back(mm_3p_other);

        return feats;
    }
};

// Accumulator for per-read damage probabilities -> contig profile
struct DamageAccumulator {
    double sw = 0.0;        // sum weights
    double sw2 = 0.0;       // sum weights^2
    double s_anc = 0.0;     // sum w * p_damaged
    double s_mod = 0.0;     // sum w * (1-p_damaged)
    double s_unc = 0.0;     // sum w^2 * p*(1-p) (soft-label uncertainty)

    void add_read(float p_damaged, float w = 1.0f) {
        if (w <= 0.0f) return;
        const double p = std::clamp((double)p_damaged, 0.0, 1.0);
        sw += w;
        sw2 += w * w;
        s_anc += w * p;
        s_mod += w * (1.0 - p);
        s_unc += (w * w) * p * (1.0 - p);
    }

    DamageProfile finalize(float contig_len,
                           float alpha0 = 1.0f,
                           float beta0 = 1.0f) const {
        DamageProfile d{};
        if (sw <= 0.0) return d;

        const double n_eff = (sw * sw) / std::max(1e-12, sw2); // Kish effective N
        const double a = alpha0 + s_anc;
        const double b = beta0 + s_mod;
        const double ab = a + b;

        const double p = a / ab;
        const double var_beta = (a * b) / (ab * ab * (ab + 1.0)); // posterior var
        const double var_soft = s_unc / std::max(1e-12, sw * sw);  // extra from uncertain labels

        d.p_anc = (float)p;
        d.var_anc = (float)std::clamp(var_beta + var_soft, 1e-6, 0.25);
        d.n_eff = (float)n_eff;
        d.cov_ancient = (float)(s_anc / std::max(1.0f, contig_len));
        d.cov_modern  = (float)(s_mod / std::max(1.0f, contig_len));
        d.log_ratio   = std::log((d.cov_ancient + 1e-6f) / (d.cov_modern + 1e-6f));
        return d;
    }
};

// Parameters for damage-aware InfoNCE
struct DamageInfoNCEParams {
    float n0 = 60.0f;           // Confidence half-saturation
    float tau2 = 1e-3f;         // Variance floor in z denominator
    float lambda = 0.5f;        // Max attenuation strength (0.4-0.6 recommended)
    float wmin = 0.5f;          // Keep graph connected (minimum negative weight)
    float min_neff_hard = 20.0f; // Below this: treat as uninformative
    bool enabled = true;         // Easy toggle
};

// Compute confidence from n_eff (smooth [0,1])
inline float conf_from_neff(float n_eff, float n0) {
    if (n_eff <= 0.0f) return 0.0f;
    return n_eff / (n_eff + n0);
}

// Symmetric damage compatibility for negative pair weighting
// Uses position-specific damage rates for better discrimination
// Returns weight in [wmin, 1.0] - attenuation only
inline float compute_damage_compatibility(
    const DamageProfile& a,
    const DamageProfile& b,
    const DamageInfoNCEParams& p) {

    if (!p.enabled) return 1.0f;
    if (a.n_eff < p.min_neff_hard || b.n_eff < p.min_neff_hard) return 1.0f;

    const float ci = conf_from_neff(a.n_eff, p.n0);
    const float cj = conf_from_neff(b.n_eff, p.n0);
    const float cpair = ci * cj;

    // Use combined terminal damage signal (positions 0-2 from 5' and 3' ends)
    const float da = a.combined_damage();
    const float db = b.combined_damage();

    // Baseline damage rate (non-ancient DNA)
    const float baseline = 0.02f;

    // Classify as damaged if terminal damage > baseline + margin
    const float damage_threshold = 0.05f;
    const bool a_damaged = da > baseline + damage_threshold;
    const bool b_damaged = db > baseline + damage_threshold;

    // If both similar damage state (both damaged or both undamaged), they're compatible
    if (a_damaged == b_damaged) {
        // Similar damage signature - could be same genome
        // Also check if damage rates are similar for damaged contigs
        if (a_damaged && b_damaged) {
            float diff = std::fabs(da - db);
            if (diff < 0.1f) return 1.0f;  // Similar damage rates
        } else {
            return 1.0f;  // Both undamaged - compatible
        }
    }

    // Mixed damage state: one ancient, one modern - likely different genomes
    // Attenuate based on confidence and difference magnitude
    const float diff = std::fabs(da - db);
    const float f_raw = std::exp(-diff / 0.2f);  // Larger diff = smaller f_raw

    // Also incorporate the old p_anc based signal
    const float denom = std::sqrt(std::max(1e-12f, a.var_anc + b.var_anc + p.tau2));
    const float z = std::fabs(a.p_anc - b.p_anc) / denom;
    const float f_panc = std::exp(-0.5f * z * z);

    // Combine both signals (position-specific and aggregate)
    const float f_combined = 0.5f * (f_raw + f_panc);

    // Attenuation: 1 when compatible or low confidence, <1 when incompatible + confident
    float f = 1.0f - p.lambda * cpair * (1.0f - f_combined);
    return std::clamp(f, p.wmin, 1.0f);
}

// Damage-aware InfoNCE loss
// Weights negative pairs by damage compatibility, weights anchor loss by confidence
class DamageAwareInfoNCE {
public:
    DamageAwareInfoNCE(float temperature = 0.1f, const DamageInfoNCEParams& params = {})
        : temperature_(temperature), params_(params) {}

    // Compute bs×bs damage weight matrix on the target device.
    // w[ci][cj] = damage_compatibility(profiles[ci], profiles[cj]); diagonal = 1.
    // Returns a (bs, bs) float32 tensor on `device`.
    torch::Tensor compute_weights(
        const std::vector<DamageProfile>& damage_profiles,
        torch::Device device) const {

        int bs = static_cast<int>(damage_profiles.size());
        std::vector<float> cw(static_cast<size_t>(bs) * bs, 1.0f);
        if (params_.enabled) {
            #pragma omp parallel for schedule(static)
            for (int ci = 0; ci < bs; ci++) {
                for (int cj = 0; cj < bs; cj++) {
                    if (ci == cj) continue;
                    cw[ci * bs + cj] = compute_damage_compatibility(
                        damage_profiles[ci], damage_profiles[cj], params_);
                }
            }
        }
        return torch::from_blob(cw.data(), {bs, bs}, torch::kFloat32)
                   .clone().to(device);
    }

private:
    float temperature_;
    DamageInfoNCEParams params_;
};

}  // namespace amber::bin2

#endif  // USE_LIBTORCH

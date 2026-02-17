#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace amber {

// A minimal damage profile: per-contig vector of (position, rate)
struct DamageEntry {
  int pos = 0;
  // Forward strand damage (5' end): C->T and A->G
  double c_to_t = 0.0;
  double a_to_g = 0.0;
  // Reverse strand damage (3' end): G->A and T->C
  double g_to_a = 0.0;
  double t_to_c = 0.0;
  // raw counts (when available)
  long total = 0;
  long countA = 0, countC = 0, countG = 0, countT = 0;
};

struct DamageProfile {
  // per-contig list of damage entries
  std::unordered_map<std::string, std::vector<DamageEntry>> per_contig;
  size_t total_positions() const;
  // average C->T rate across all entries (naive mean)
  double average_rate() const;
  // maximum position across all contigs (for determining feature vector size)
  int max_position() const;
};

// Load a bdamage binary file (bgzf/gzip compressed). If `ref_bam` is
// provided, the BAM header is read (via htslib) and numeric record IDs in the
// bdamage file are translated into reference names from the BAM header.
// If `contig_filter` is provided, only load damage data for contigs in that set.
// Returns true on success; on failure fills err_out with a short message.
bool load_damage_profile(const std::string &fn, DamageProfile &dp,
                         std::string &err_out, const std::string &ref_bam = "",
                         const std::unordered_set<std::string> *contig_filter = nullptr);

// Configuration for damage computation from BAM
struct DamageComputeConfig {
  int max_position = 20;      // Maximum position from read ends to track
  int min_mapq = 30;          // Minimum mapping quality
  int min_baseq = 20;         // Minimum base quality
  int threads = 1;            // Number of threads
};

// Per-position damage statistics with confidence intervals
struct PositionDamageStats {
  int position = 0;                    // 1-based position from end
  uint64_t n_opportunities = 0;        // Total C/G opportunities
  uint64_t n_mismatches = 0;           // C→T or G→A mismatches
  double rate = 0.0;                   // Observed mismatch rate
  double ci_lower = 0.0;               // 95% credible interval lower bound
  double ci_upper = 0.0;               // 95% credible interval upper bound
};

// Damage curve parameters with uncertainty
struct DamageCurveParams {
  double amplitude = 0.0;              // D(1) - damage at position 1
  double amplitude_se = 0.0;           // Standard error of amplitude
  double lambda = 0.35;                // Decay rate (larger = faster decay)
  double lambda_se = 0.0;              // Standard error of lambda
  double baseline = 0.01;              // Background error rate (sequencing errors)
  double baseline_se = 0.0;            // Standard error of baseline
  double concentration = 100.0;        // Beta-Binomial concentration (overdispersion)
};

// Three-tier classification based on combined evidence
enum class DamageClassification {
  ANCIENT,    // Strong evidence: p >= 0.8 AND amplitude >= 1% AND (lambda > 0.3 OR amp < 5%)
  UNCERTAIN,  // Ambiguous: 0.5 <= p < 0.8 OR (p >= 0.8 AND amplitude < 1%)
  MODERN      // Evidence against: p < 0.5
};

// Convert classification enum to string
inline const char* classification_to_string(DamageClassification c) {
  switch (c) {
    case DamageClassification::ANCIENT: return "ancient";
    case DamageClassification::UNCERTAIN: return "uncertain";
    case DamageClassification::MODERN: return "modern";
    default: return "unknown";
  }
}

// Bayesian damage model result for a single contig or globally
struct DamageModelResult {
  double p_ancient = 0.0;              // Posterior P(ancient | data)
  double log_bf = 0.0;                 // Log Bayes factor log(P(data|ancient)/P(data|modern))
  DamageClassification classification = DamageClassification::UNCERTAIN;  // Three-tier classification

  DamageCurveParams curve_5p;          // 5' end C→T damage curve
  DamageCurveParams curve_3p;          // 3' end G→A damage curve

  // Per-position statistics
  std::vector<PositionDamageStats> stats_5p;  // 5' end per-position stats
  std::vector<PositionDamageStats> stats_3p;  // 3' end per-position stats

  bool is_significant = false;         // Whether damage exceeds noise threshold
  std::string significance_reason;     // Explanation of significance decision
  uint64_t total_obs = 0;              // Total observations

  // Quality metrics
  double effective_coverage = 0.0;     // Average coverage at damage positions
  double fit_r_squared = 0.0;          // Goodness of fit (R² of exponential curve)

  // Probabilistic evidence strength
  double evidence_strength = 0.0;      // Interpreted Bayes factor: 0=none, 1=weak, 2=moderate, 3=strong, 4=decisive
};

// Full damage analysis result
struct DamageAnalysis {
  DamageProfile profile;                                                // Per-position rates
  DamageModelResult global_model;                                       // Global model fit
  std::unordered_map<std::string, DamageModelResult> per_contig_model;  // Per-contig fits
  std::string library_type;                                             // Detected library type (ds/ss)

  // Global calibration from interior positions
  double calibrated_error_rate = 0.01;  // Sequencing error rate from interior
  double error_rate_se = 0.0;           // Standard error of error rate
};

// Compute damage profile directly from BAM file by parsing MD tags.
// Counts C→T mismatches at 5' end and G→A mismatches at 3' end.
// Returns true on success; on failure fills err_out with a short message.
bool compute_damage_from_bam(const std::string &bam_file, DamageProfile &dp,
                             std::string &err_out,
                             const DamageComputeConfig &config = {},
                             const std::unordered_set<std::string> *contig_filter = nullptr);

// Compute damage analysis with Bayesian model fitting.
// Estimates damage parameters, computes Bayes factor for ancient vs modern,
// and classifies each contig as having significant damage or not.
bool compute_damage_analysis(const std::string &bam_file, DamageAnalysis &analysis,
                             std::string &err_out,
                             const DamageComputeConfig &config = {},
                             const std::unordered_set<std::string> *contig_filter = nullptr);

// Fit Bayesian damage model to accumulated statistics
// Uses Beta-Binomial model with exponential decay: D(z) = A * exp(-λ*z) + baseline
DamageModelResult fit_damage_model(const std::vector<uint64_t> &n_5p,
                                   const std::vector<uint64_t> &k_5p,
                                   const std::vector<uint64_t> &n_3p,
                                   const std::vector<uint64_t> &k_3p,
                                   double prior_ancient = 0.5);

} // namespace amber

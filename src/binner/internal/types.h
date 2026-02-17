// MFDFA Binner - Core Types
#pragma once

#include <string>
#include <vector>
#include <array>

namespace amber {

// MFDFA quality metrics
struct MFDFAMetrics {
    double hurst = 0.5;
    double width = 0.0;
    double variance = 0.0;
};

// Contig data structure (complete version used by binner)
struct Contig {
    std::string name;
    std::string sequence;
    size_t length = 0;

    // Clustering features (from V6 - proven to work)
    std::vector<double> tnf;           // 136 dims, L2 normalized
    std::vector<double> cgr;           // 32 dims, L2 normalized
    std::vector<double> dna_physical;  // 4 dims
    std::vector<double> fractal;       // 5 dims (CGR-DFA)
    double gc = 0.0;
    double mean_coverage = 0.0;
    double var_coverage = 0.0;         // Within-contig coverage variance

    // Dinucleotide relative abundances for Karlin distance (genomic signature)
    std::array<double, 16> dinuc_rho = {};  // AA,AC,AG,AT,CA,CC,CG,CT,GA,GC,GG,GT,TA,TC,TG,TT

    // Stable features for resonance merger (low CV, consistent within genome)
    std::vector<double> stable;        // 14 dims

    double contig_modularity = -1.0;
    double contig_cgr_occupancy = 0.0;

    // Damage info
    double damage_rate_5p = 0.0;
    double damage_rate_3p = 0.0;
    double damage_gradient = 0.0;
    bool has_damage = false;

    // Bayesian damage model parameters
    double damage_amplitude_5p = 0.0;
    double damage_lambda_5p = 0.35;
    double damage_baseline_5p = 0.01;
    double damage_amplitude_3p = 0.0;
    double damage_lambda_3p = 0.35;
    double damage_baseline_3p = 0.01;
    bool damage_significant = false;
    double damage_p_ancient = 0.0;

    // Bin assignment
    int bin_id = -1;
    double bin_probability = 0.0;
    bool is_seed = false;

    int kmeans_bin_id = -1;
    int em_bin_id = -1;
    int final_bin_id = -1;

    // IQS metrics
    std::vector<double> kmer6_probs;
    double iqs_influence = 0.0;
    double iqs_gain_if_removed = 0.0;
    bool flagged_contaminant = false;

    double jsd_to_centroid = 0.0;
    double jsd_zscore = 0.0;

    // Comprehensive fate tracking
    struct FateEvent {
        std::string stage;
        int from_bin = -1;
        int to_bin = -1;
        std::string reason;
        double distance = 0.0;
    };
    std::vector<FateEvent> fate_history;

    void record_fate(const std::string& stage, int from, int to,
                     const std::string& reason, double dist = 0.0) {
        fate_history.push_back({stage, from, to, reason, dist});
    }
};

// Bin diagnostics
struct BinDiagnostics {
    size_t size = 0;
    size_t total_bp = 0;
    double coverage_mean = 0.0;
    double coverage_cv = 0.0;
    double gc_mean = 0.0;
    double gc_std = 0.0;
    double hurst_mean = 0.0;
    double hurst_std = 0.0;
    double lacunarity_mean = 0.0;
    double lacunarity_var = 0.0;
    int coverage_modes = 0;
    bool is_suspect = false;
};

// BinnerConfig is in include/amber/config.hpp

// K estimation result
struct KEstimate {
    int coverage_modes = 1;
    int composition_clusters = 1;
    int combined = 5;
    std::string details;
    std::string rationale;
};

}  // namespace amber

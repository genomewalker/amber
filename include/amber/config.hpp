// AMBER - Centralized Configuration Structures
// All module configs in one place for consistency
#ifndef AMBER_CONFIG_HPP
#define AMBER_CONFIG_HPP

#include <string>
#include "amber/version.h"

namespace amber {

// Global version for all AMBER tools (set by cmake from git describe)
constexpr const char* VERSION = AMBER_VERSION_STRING;

// Default damage positions for all modules
constexpr int DEFAULT_DAMAGE_POSITIONS = 15;

// Feature extraction configuration
struct EmbedConfig {
    std::string contigs_file;
    std::string bam_file;
    std::string damage_file;
    std::string output_dir;
    std::string library_type = "ds";
    std::string feature_set = "all";
    int min_length = 1000;
    int damage_positions = DEFAULT_DAMAGE_POSITIONS;
    int threads = 1;
    bool skip_coverage = false;
    bool skip_damage = false;
    bool verbose = false;
};

// Binning configuration
struct BinnerConfig {
    std::string contigs_path;
    std::string bam_path;
    std::string output_dir;
    std::string damage_path;

    int threads = 1;
    int min_length = 2500;
    int damage_positions = DEFAULT_DAMAGE_POSITIONS;
    int max_em_iterations = 50;
    int random_seed = -1;

    bool use_fractal = true;
    bool use_damage = true;
    bool verbose = false;
};

// Damage-aware polishing configuration
struct PolishConfig {
    std::string contigs_file;
    std::string bam_file;
    std::string output_dir = ".";
    std::string library_type = "auto";
    int min_length = 0;
    int min_mapq = 30;
    int min_baseq = 20;
    int damage_positions = DEFAULT_DAMAGE_POSITIONS;
    int threads = 1;
    double min_log_lr = 3.0;           // Legacy LR threshold
    bool verbose = false;
    bool anvio = false;
    std::string anvio_prefix = "c";

    // Bayesian genotype caller options (NEW)
    bool use_bayesian = true;          // Use Bayesian caller (default ON)
    double min_posterior = 0.99;       // Required confidence for correction
    double max_second_posterior = 0.05;// Max probability for second-best base
    bool write_calls = false;          // Write detailed call information

    // Strain deconvolution options (damage-based: ancient vs modern)
    bool deconvolve = false;           // Enable two-population strain deconvolution
    double deconv_min_ancient_depth = 3.0;  // Min weighted depth for ancient consensus
    double deconv_min_modern_depth = 2.0;   // Min weighted depth for modern consensus
    double deconv_min_modern_fraction = 0.05; // Min modern fraction to output modern consensus
    bool deconv_write_stats = false;   // Write per-position deconvolution stats
    int deconv_em_iterations = 10;     // Max EM iterations (auto-converges when prior changes <1%)
    bool deconv_full_likelihood = true; // Use full read likelihood (matches + mismatches)
};

// Chimera detection configuration
struct ChimeraConfig {
    std::string contigs_file;
    std::string bam_file;
    std::string output_dir = ".";
    int min_length = 1000;
    int damage_positions = DEFAULT_DAMAGE_POSITIONS;
    double min_confidence = 0.5;
    int threads = 1;
    bool verbose = false;
    std::string library_type;
    std::string chimera_action = "split";
    bool write_fasta = true;
};

}  // namespace amber

#endif  // AMBER_CONFIG_HPP

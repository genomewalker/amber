// AMBER - cmd_bin2.cpp
// CLI handler for the 'bin2' subcommand (self-contrastive binner)

#include "cli_common.h"
#include <iostream>
#include <string>
#include <cstring>

namespace amber {

// Forward declaration
struct Bin2Config {
    std::string contigs_path;
    std::string bam_path;
    std::string output_dir;
    std::string seed_file;         // SCG marker seed file
    std::string hmm_file;          // HMM file for auto seed generation
    std::string hmmsearch_path;    // Path to hmmsearch binary
    std::string fraggenescan_path; // Path to FragGeneScan binary
    std::string embeddings_file;   // Pre-computed embeddings TSV (skip training)
    int threads = 16;
    int min_length = 1001;         // COMEBin default (filters > 1000bp)
    int min_bin_size = 200000;     // 200kb minimum bin size
    int epochs = 100;
    int batch_size = 1024;         // COMEBin default
    int embedding_dim = 128;
    float learning_rate = 0.001f;
    float temperature = 0.1f;      // COMEBin default
    // Clustering parameters (tuned for 9 HQ bins)
    int k = 100;                   // kNN neighbors (COMEBin default)
    int partgraph_ratio = 50;      // Percentile cutoff for edge pruning
    float resolution = 5.0f;       // Leiden resolution (optimal)
    float bandwidth = 0.2f;        // Edge weighting bandwidth (optimal)
    // Flags
    bool l2_normalize = true;      // L2 normalize embeddings before clustering
    bool verbose = false;
    bool sweep = false;            // COMEBin-style parameter sweep
    bool use_python_leiden = false; // Use Python leidenalg (exact COMEBin match)
    bool force_cpu = false;        // Force CPU mode even if GPU is available
    int hidden_dim = 2048;         // Hidden layer size (COMEBin default: 2048)
    int n_layer = 3;               // Number of hidden layers (COMEBin default: 3)
    float dropout = 0.2f;          // Dropout rate (COMEBin default: 0.2)
    int random_seed = 1006;        // Random seed for Leiden clustering (optimal)
    int encoder_seed = 42;         // Random seed for encoder training (optimal)
    // Damage-aware features (experimental)
    bool use_damage_infonce = false;   // Use damage-weighted InfoNCE loss
    float damage_lambda = 0.5f;        // Damage attenuation strength (0.4-0.6)
    float damage_wmin = 0.5f;          // Minimum negative weight
    bool use_multiscale_cgr = false;   // Use multi-scale CGR late fusion
    // Ensemble clustering
    bool use_ensemble_leiden = false;  // Use ensemble Leiden for reproducible clustering
    int n_ensemble_runs = 10;          // Number of Leiden runs for ensemble
    float ensemble_threshold = 0.5f;   // Co-occurrence threshold for consensus
    // Quality-guided clustering
    bool use_quality_leiden = false;   // Use marker-quality-guided Leiden refinement
    float quality_alpha = 1.0f;        // Quality weight (0=modularity only, 1=balanced)
    std::string checkm_hmm_file;       // HMM for CheckM markers (default: auxiliary/checkm_markers_only.hmm)
    std::string bacteria_ms_file;      // CheckM bacteria marker sets (default: scripts/checkm_ms/bacteria.ms)
    std::string archaea_ms_file;       // CheckM archaea marker sets (default: scripts/checkm_ms/archaea.ms)
};

extern int run_bin2(const Bin2Config& config);

int cmd_bin2(int argc, char** argv) {
    Bin2Config config;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: amber bin2 [options]\n\n"
                      << "COMEBin-style contrastive binner (9 HQ bins on test dataset)\n\n"
                      << "Required:\n"
                      << "  --contigs FILE         Input contigs FASTA\n"
                      << "  --bam FILE             BAM file (reads mapped to contigs)\n"
                      << "  --output DIR           Output directory\n\n"
                      << "Training:\n"
                      << "  --epochs N             Training epochs (default: 100)\n"
                      << "  --batch-size N         Batch size (default: 1024)\n"
                      << "  --lr FLOAT             Learning rate (default: 0.001)\n"
                      << "  --temperature FLOAT    Temperature for InfoNCE (default: 0.1)\n\n"
                      << "Clustering:\n"
                      << "  --k N                  kNN neighbors (default: 100)\n"
                      << "  --partgraph-ratio N    Percentile cutoff for edge pruning (default: 50)\n"
                      << "  --resolution FLOAT     Leiden resolution (default: 5.0)\n"
                      << "  --bandwidth FLOAT      Edge weighting bandwidth (default: 0.2)\n"
                      << "  --seeds FILE           SCG marker seed file (auto-generated if not provided)\n"
                      << "  --random-seed N        Random seed for Leiden clustering (default: 1006)\n"
                      << "  --encoder-seed N       Random seed for encoder training (default: 1)\n"
                      << "  --no-l2norm            Disable L2 normalization of embeddings\n\n"
                      << "Encoder:\n"
                      << "  --hidden-dim N         Encoder hidden layer size (default: 2048)\n"
                      << "  --n-layer N            Number of encoder hidden layers (default: 3)\n"
                      << "  --dropout FLOAT        Encoder dropout rate (default: 0.2)\n\n"
                      << "Seed Generation:\n"
                      << "  --hmm FILE             HMM file for seed generation (default: auxiliary/bacar_marker.hmm)\n"
                      << "  --hmmsearch PATH       Path to hmmsearch binary (default: hmmsearch)\n"
                      << "  --fraggenescan PATH    Path to FragGeneScan binary (default: FragGeneScan)\n\n"
                      << "Other:\n"
                      << "  --min-length N         Minimum contig length (default: 1001)\n"
                      << "  --min-bin-size N       Minimum bin size in bp (default: 200000)\n"
                      << "  --threads N            Number of threads (default: 16)\n"
                      << "  --cpu                  Force CPU mode (auto-detects GPU by default)\n"
                      << "  -v, --verbose          Verbose output\n\n"
                      << "Clustering-only Mode:\n"
                      << "  --embeddings FILE      Load pre-computed embeddings TSV (skip training)\n"
                      << "                         Format: contig_name\\tdim0\\tdim1\\t...\\tdimN\n\n"
                      << "Damage-Aware (experimental - use for mixed ancient/modern samples):\n"
                      << "  --damage-infonce       Enable damage-weighted InfoNCE loss\n"
                      << "                         Best for datasets with MIXED ancient+modern DNA\n"
                      << "                         May regress on uniformly ancient samples\n"
                      << "  --damage-lambda FLOAT  Attenuation strength (default: 0.5)\n"
                      << "  --damage-wmin FLOAT    Minimum negative weight (default: 0.5)\n"
                      << "  --multiscale-cgr       Enable multi-scale CGR late fusion\n\n"
                      << "Ensemble Clustering:\n"
                      << "  --ensemble-leiden      Use ensemble Leiden for reproducible clustering\n"
                      << "                         Runs Leiden N times, takes co-occurrence consensus\n"
                      << "  --n-ensemble-runs INT  Number of Leiden runs (default: 10)\n"
                      << "  --ensemble-threshold F Co-occurrence threshold for consensus (default: 0.5)\n"
                      << "Quality-Guided Clustering:\n"
                      << "  --quality-leiden       Use marker-quality-guided Leiden refinement\n"
                      << "                         Optimizes modularity + SCG completeness/contamination\n"
                      << "  --quality-alpha F      Quality weight in combined objective (default: 1.0)\n"
                      << "                         0=pure modularity, 1=balanced, >1=quality-dominant\n"
                      << "  --checkm-hmm FILE      CheckM HMM for seed+quality markers\n"
                      << "                         (default: auxiliary/checkm_markers_only.hmm)\n"
                      << "  --bacteria-ms FILE     CheckM bacteria marker sets (default: scripts/checkm_ms/bacteria.ms)\n"
                      << "  --archaea-ms FILE      CheckM archaea marker sets (default: scripts/checkm_ms/archaea.ms)\n";
            return 0;
        }
        else if (arg == "--contigs" && i + 1 < argc) {
            config.contigs_path = argv[++i];
        }
        else if (arg == "--bam" && i + 1 < argc) {
            config.bam_path = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
        }
        else if (arg == "--threads" && i + 1 < argc) {
            config.threads = std::stoi(argv[++i]);
        }
        else if (arg == "--min-length" && i + 1 < argc) {
            config.min_length = std::stoi(argv[++i]);
        }
        else if (arg == "--min-bin-size" && i + 1 < argc) {
            config.min_bin_size = std::stoi(argv[++i]);
        }
        else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::stoi(argv[++i]);
        }
        else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        }
        else if (arg == "--lr" && i + 1 < argc) {
            config.learning_rate = std::stof(argv[++i]);
        }
        else if (arg == "--temperature" && i + 1 < argc) {
            config.temperature = std::stof(argv[++i]);
        }
        else if (arg == "--k" && i + 1 < argc) {
            config.k = std::stoi(argv[++i]);
        }
        else if (arg == "--resolution" && i + 1 < argc) {
            config.resolution = std::stof(argv[++i]);
        }
        else if (arg == "--bandwidth" && i + 1 < argc) {
            config.bandwidth = std::stof(argv[++i]);
        }
        else if (arg == "--partgraph-ratio" && i + 1 < argc) {
            config.partgraph_ratio = std::stoi(argv[++i]);
        }
        else if (arg == "--seeds" && i + 1 < argc) {
            config.seed_file = argv[++i];
        }
        else if (arg == "--hmm" && i + 1 < argc) {
            config.hmm_file = argv[++i];
        }
        else if (arg == "--hmmsearch" && i + 1 < argc) {
            config.hmmsearch_path = argv[++i];
        }
        else if (arg == "--fraggenescan" && i + 1 < argc) {
            config.fraggenescan_path = argv[++i];
        }
        else if (arg == "--random-seed" && i + 1 < argc) {
            config.random_seed = std::stoi(argv[++i]);
        }
        else if (arg == "--encoder-seed" && i + 1 < argc) {
            config.encoder_seed = std::stoi(argv[++i]);
        }
        else if (arg == "--no-l2norm") {
            config.l2_normalize = false;
        }
        else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        }
        else if (arg == "--sweep") {
            config.sweep = true;
        }
        else if (arg == "--use-python-leiden") {
            config.use_python_leiden = true;
        }
        else if (arg == "--cpu") {
            config.force_cpu = true;
        }
        else if (arg == "--hidden-dim" && i + 1 < argc) {
            config.hidden_dim = std::stoi(argv[++i]);
        }
        else if (arg == "--n-layer" && i + 1 < argc) {
            config.n_layer = std::stoi(argv[++i]);
        }
        else if (arg == "--dropout" && i + 1 < argc) {
            config.dropout = std::stof(argv[++i]);
        }
        else if (arg == "--embeddings" && i + 1 < argc) {
            config.embeddings_file = argv[++i];
        }
        else if (arg == "--damage-infonce") {
            config.use_damage_infonce = true;
        }
        else if (arg == "--damage-lambda" && i + 1 < argc) {
            config.damage_lambda = std::stof(argv[++i]);
        }
        else if (arg == "--damage-wmin" && i + 1 < argc) {
            config.damage_wmin = std::stof(argv[++i]);
        }
        else if (arg == "--multiscale-cgr") {
            config.use_multiscale_cgr = true;
        }
        else if (arg == "--ensemble-leiden") {
            config.use_ensemble_leiden = true;
        }
        else if (arg == "--n-ensemble-runs" && i + 1 < argc) {
            config.n_ensemble_runs = std::stoi(argv[++i]);
        }
        else if (arg == "--ensemble-threshold" && i + 1 < argc) {
            config.ensemble_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--quality-leiden") {
            config.use_quality_leiden = true;
        }
        else if (arg == "--quality-alpha" && i + 1 < argc) {
            config.quality_alpha = std::stof(argv[++i]);
        }
        else if (arg == "--checkm-hmm" && i + 1 < argc) {
            config.checkm_hmm_file = argv[++i];
        }
        else if (arg == "--bacteria-ms" && i + 1 < argc) {
            config.bacteria_ms_file = argv[++i];
        }
        else if (arg == "--archaea-ms" && i + 1 < argc) {
            config.archaea_ms_file = argv[++i];
        }
    }

    // Validate required arguments
    if (config.embeddings_file.empty()) {
        // Full mode: need contigs and BAM
        if (config.contigs_path.empty()) {
            std::cerr << "Error: --contigs is required (or use --embeddings for clustering-only mode)\n";
            return 1;
        }
        if (config.bam_path.empty()) {
            std::cerr << "Error: --bam is required (or use --embeddings for clustering-only mode)\n";
            return 1;
        }
    }
    if (config.output_dir.empty()) {
        std::cerr << "Error: --output is required\n";
        return 1;
    }

    return run_bin2(config);
}

}  // namespace amber

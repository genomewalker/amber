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
    int min_length = 1001;         // AMBER default (filters > 1000bp)
    int min_bin_size = 200000;     // 200kb minimum bin size
    int epochs = 100;
    int batch_size = 1024;         // AMBER default
    int embedding_dim = 128;
    float learning_rate = 0.001f;
    float temperature = 0.1f;      // AMBER default
    // Clustering parameters (tuned for 9 HQ bins)
    int k = 100;                   // kNN neighbors (AMBER default)
    int partgraph_ratio = 50;      // Percentile cutoff for edge pruning
    float resolution = 5.0f;       // Leiden resolution (optimal)
    float bandwidth = 0.2f;        // Edge weighting bandwidth (optimal)
    // Flags
    bool l2_normalize = true;      // L2 normalize embeddings before clustering
    bool verbose = false;
    bool force_cpu = false;        // Force CPU mode even if GPU is available
    int hidden_dim = 2048;         // Hidden layer size (AMBER default: 2048)
    int n_layer = 3;               // Number of hidden layers (AMBER default: 3)
    float dropout = 0.2f;          // Dropout rate (AMBER default: 0.2)
    int random_seed = 1006;        // Random seed for Leiden clustering (optimal)
    int encoder_seed = 42;         // Random seed for encoder training (optimal)
    // Damage-aware features (experimental)
    bool use_damage_infonce = false;   // Use damage-weighted InfoNCE loss
    float damage_lambda = 0.5f;        // Damage attenuation strength (0.4-0.6)
    float damage_wmin = 0.5f;          // Minimum negative weight
    bool use_multiscale_cgr = false;   // Use multi-scale CGR late fusion
    int n_leiden_restarts = 1;         // Seed restart search budget (1=off, 25=recommended)
    int restart_stage1_res = 1;        // Resolution grid points in Stage 1 (1 = pinned)
    int n_encoder_restarts = 1;        // Consensus kNN: train N encoders, aggregate edges (1=off)
    std::string checkm_hmm_file;       // HMM for CheckM markers (default: auxiliary/checkm_markers_only.hmm)
    std::string bacteria_ms_file;      // CheckM bacteria marker sets (default: scripts/checkm_ms/bacteria.ms)
    std::string archaea_ms_file;       // CheckM archaea marker sets (default: scripts/checkm_ms/archaea.ms)
    // SCG hard negative mining
    bool use_scg_infonce = false;      // Boost SCG-sharing contig pairs as InfoNCE hard negatives
    float scg_boost = 2.0f;            // Multiplicative boost for shared-marker pairs (default: 2.0)
    // Phase 4E tuning
    float phase4e_entry_contamination = 3.0f;
    float phase4e_sigma_threshold = 1.0f;
    // Gradient clipping and EMA
    float grad_clip = 1.0f;            // Global gradient norm clip (0 = disabled)
    bool use_ema = false;              // EMA of encoder weights for final embeddings
    float ema_decay = 0.999f;          // EMA decay rate
    // QC gate: retry poor encoder runs
    float encoder_qc_threshold = 0.8f; // Retry if score < threshold * best
    int encoder_qc_max_extra = 2;      // Max extra encoder restarts via QC gate
};

extern int run_bin2(const Bin2Config& config);

int cmd_bin2(int argc, char** argv) {
    Bin2Config config;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: amber bin2 [options]\n\n"
                      << "AMBER contrastive binner (ancient DNA-aware metagenomic binner)\n\n"
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
                      << "Damage-Aware:\n"
                      << "  --damage-infonce       Enable damage-weighted InfoNCE loss\n"
                      << "  --damage-lambda FLOAT  Attenuation strength (default: 0.5)\n"
                      << "  --damage-wmin FLOAT    Minimum negative weight (default: 0.5)\n"
                      << "  --multiscale-cgr       Enable multi-scale CGR late fusion\n\n"
                      << "Leiden Restarts:\n"
                      << "  --leiden-restarts N    Seed sweep: run N Leiden restarts, take best SCG score\n"
                      << "                         N=1 (default, off). Recommended: N=25.\n"
                      << "  --stage1-res N         Resolution grid size in Stage 1 (default: 1 = pinned).\n"
                      << "  --encoder-restarts N   Consensus kNN: train N encoders independently,\n"
                      << "                         aggregate kNN edges. N=1 (default, off). Recommended: N=3.\n"
                      << "  --checkm-hmm FILE      CheckM HMM for seed+quality markers\n"
                      << "                         (default: auxiliary/checkm_markers_only.hmm)\n"
                      << "  --bacteria-ms FILE     CheckM bacteria marker sets (default: scripts/checkm_ms/bacteria.ms)\n"
                      << "  --archaea-ms FILE      CheckM archaea marker sets (default: scripts/checkm_ms/archaea.ms)\n"
                      << "SCG Hard Negative Mining:\n"
                      << "  --scg-infonce          Enable SCG hard negative mining in InfoNCE\n"
                      << "                         Boosts pairs sharing a single-copy marker as hard negatives.\n"
                      << "                         Forces SCG contigs apart in embedding space before clustering.\n"
                      << "  --scg-boost FLOAT      Amplification factor for shared-marker pairs (default: 2.0)\n"
                      << "                         2.0 = safe (exp(10) vs exp(5) partition denominator).\n"
                      << "                         >=3.0 can dominate log_Z; not recommended.\n"
                      << "Phase 4E Tuning (embedding-outlier contamination eviction):\n"
                      << "  --phase4e-entry-cont F Internal contamination % threshold to attempt Phase 4E\n"
                      << "                         (default: 3.0; ~3% internal ≈ ~6% CheckM2)\n"
                      << "  --phase4e-sigma F      Outlier sigma: contigs with belonging_score <\n"
                      << "                         mean - sigma*std AND dup markers are candidates\n"
                      << "                         (default: 1.0)\n"
                      << "Encoder Regularization:\n"
                      << "  --grad-clip FLOAT      Global gradient norm clip (default: 1.0, 0=disabled)\n"
                      << "  --encoder-ema          Enable EMA of encoder weights for final embeddings\n"
                      << "  --ema-decay FLOAT      EMA decay rate (default: 0.999)\n"
                      << "Encoder QC Gate:\n"
                      << "  --encoder-qc-threshold F  Retry if score < threshold * best (default: 0.8)\n"
                      << "  --encoder-qc-max-extra N  Max extra encoder retries via QC gate (default: 2)\n";
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
        else if (arg == "--leiden-restarts" && i + 1 < argc) {
            config.n_leiden_restarts = std::stoi(argv[++i]);
        }
        else if (arg == "--stage1-res" && i + 1 < argc) {
            config.restart_stage1_res = std::stoi(argv[++i]);
        }
        else if (arg == "--encoder-restarts" && i + 1 < argc) {
            config.n_encoder_restarts = std::stoi(argv[++i]);
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
        else if (arg == "--scg-infonce") {
            config.use_scg_infonce = true;
        }
        else if (arg == "--scg-boost" && i + 1 < argc) {
            config.scg_boost = std::stof(argv[++i]);
        }
        else if (arg == "--phase4e-entry-cont" && i + 1 < argc) {
            config.phase4e_entry_contamination = std::stof(argv[++i]);
        }
        else if (arg == "--phase4e-sigma" && i + 1 < argc) {
            config.phase4e_sigma_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--grad-clip" && i + 1 < argc) {
            config.grad_clip = std::stof(argv[++i]);
        }
        else if (arg == "--encoder-ema") {
            config.use_ema = true;
        }
        else if (arg == "--ema-decay" && i + 1 < argc) {
            config.ema_decay = std::stof(argv[++i]);
        }
        else if (arg == "--encoder-qc-threshold" && i + 1 < argc) {
            config.encoder_qc_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--encoder-qc-max-extra" && i + 1 < argc) {
            config.encoder_qc_max_extra = std::stoi(argv[++i]);
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

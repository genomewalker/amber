// AMBER - cmd_bin2.cpp
// CLI handler for the 'bin2' subcommand (self-contrastive binner)

#include "cli_common.h"
#include "../bin2_config.h"
#include <iostream>
#include <string>
#include <cstring>

namespace amber {

extern int run_bin2(const Bin2Config& config);

int cmd_bin2(int argc, char** argv) {
    Bin2Config config;

    // Parse arguments
    try {
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
                      << "  --encoder-seed N       Random seed for encoder training (default: 42)\n"
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
                      << "  --encoder-qc-max-extra N  Max extra encoder retries via QC gate (default: 2)\n"
                      << "Damage Profile:\n"
                      << "  --damage-positions N   Terminal positions for smiley plot and likelihood (default: 15, max: 25)\n";
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
        else if (arg == "--damage-positions" && i + 1 < argc) {
            config.damage_positions = std::stoi(argv[++i]);
            if (config.damage_positions < 1 || config.damage_positions > 25) {
                std::cerr << "ERROR: --damage-positions must be 1..25\n";
                return 1;
            }
        }
        else if (arg == "--encoder-qc-threshold" && i + 1 < argc) {
            config.encoder_qc_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--encoder-qc-max-extra" && i + 1 < argc) {
            config.encoder_qc_max_extra = std::stoi(argv[++i]);
        }
        else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Error: unknown flag '" << arg << "'. Run with --help for usage.\n";
            return 1;
        }
    }
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: invalid numeric argument: " << e.what() << "\n";
        return 1;
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: numeric argument out of range: " << e.what() << "\n";
        return 1;
    }

    // Validate required arguments
    if (config.contigs_path.empty()) {
        std::cerr << "Error: --contigs is required\n";
        return 1;
    }
    if (config.embeddings_file.empty() && config.bam_path.empty()) {
        std::cerr << "Error: --bam is required (or use --embeddings to skip training)\n";
        return 1;
    }
    if (config.output_dir.empty()) {
        std::cerr << "Error: --output is required\n";
        return 1;
    }

    return run_bin2(config);
}

}  // namespace amber

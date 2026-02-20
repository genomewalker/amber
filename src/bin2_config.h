#pragma once
// AMBER bin2 configuration — single source of truth for Bin2Config.
// Included by both bin2.cpp (the implementation) and cmd_bin2.cpp (the CLI).

#include <string>

namespace amber {

struct Bin2Config {
    // I/O
    std::string contigs_path;
    std::string bam_path;
    std::string output_dir;
    std::string seed_file;
    std::string hmm_file;
    std::string hmmsearch_path;
    std::string fraggenescan_path;
    std::string embeddings_file;

    // Resources
    int threads = 16;

    // Contig filtering
    int min_length = 1001;
    int min_bin_size = 200000;

    // Encoder architecture
    int epochs = 100;
    int batch_size = 1024;
    int embedding_dim = 128;
    int hidden_dim = 2048;
    int n_layer = 3;
    float dropout = 0.2f;
    float learning_rate = 0.001f;
    float temperature = 0.1f;
    bool l2_normalize = true;

    // Clustering
    int k = 100;
    int partgraph_ratio = 50;
    float resolution = 5.0f;
    float bandwidth = 0.2f;

    // Seeds
    int random_seed = 1006;
    int encoder_seed = 42;

    // Misc flags
    bool verbose = false;
    bool force_cpu = false;

    // Damage-aware InfoNCE
    bool use_damage_infonce = true;
    float damage_lambda = 0.5f;
    float damage_wmin = 0.5f;

    // Multi-scale CGR features
    bool use_multiscale_cgr = false;

    // Leiden restarts (seed sweep)
    int n_leiden_restarts = 25;
    int restart_stage1_res = 1;

    // Consensus kNN (encoder restarts)
    int n_encoder_restarts = 3;

    // CheckM marker files
    std::string checkm_hmm_file;
    std::string bacteria_ms_file;
    std::string archaea_ms_file;

    // SCG hard negative mining
    bool use_scg_infonce = true;
    float scg_boost = 2.0f;

    // Phase 4E decontamination
    float phase4e_entry_contamination = 3.0f;
    float phase4e_sigma_threshold = 1.0f;

    // Encoder regularization
    float grad_clip = 1.0f;
    bool use_ema = false;
    float ema_decay = 0.999f;

    // Damage position tracking (smiley plot + likelihood, max 25)
    int damage_positions = 15;

    // Encoder QC gate
    float encoder_qc_threshold = 0.8f;
    int encoder_qc_max_extra = 2;
};

} // namespace amber

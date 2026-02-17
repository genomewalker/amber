// AMBER bin2 - Encoder Types
// Core data structures for damage-aware contrastive learning
#pragma once

#ifdef USE_LIBTORCH

#include <torch/torch.h>

namespace amber::bin2 {

// Configuration for the damage-aware encoder
struct EncoderConfig {
    int tnf_dim = 136;       // Tetranucleotide frequency dimensions
    int cov_dim = 1;         // Coverage dimensions
    int damage_dim = 5;      // Damage features: amp5, lambda5, amp3, lambda3, p_ancient
    int emb_dim = 128;       // Final embedding dimension
    int hidden_dim = 512;    // Hidden layer dimension
    float dropout = 0.2f;    // Dropout probability

    // Total input dimension for TNF+coverage branch
    int tc_input_dim() const { return tnf_dim + cov_dim; }

    // Total input dimension for damage branch (includes confidence)
    int dmg_input_dim() const { return damage_dim + 1; }
};

// Batch of features for encoder input
struct FeatureBatch {
    torch::Tensor tnf_cov;   // [B, cov_dim + 136] - COMEBin order: coverage(0) + TNF(1-136)
    torch::Tensor damage;    // [B, 5] - amp5, lambda5, amp3, lambda3, p_ancient
    torch::Tensor dmg_conf;  // [B, 1] - damage confidence weight (based on coverage)

    // Move batch to device
    FeatureBatch to(torch::Device device) const {
        return FeatureBatch{
            tnf_cov.to(device),
            damage.to(device),
            dmg_conf.to(device)
        };
    }

    // Clone the batch
    FeatureBatch clone() const {
        return FeatureBatch{
            tnf_cov.clone(),
            damage.clone(),
            dmg_conf.clone()
        };
    }

    // Get batch size
    int64_t batch_size() const {
        return tnf_cov.size(0);
    }
};

// Create a FeatureBatch from separate vectors
inline FeatureBatch create_feature_batch(
    const std::vector<std::vector<float>>& tnf_cov_data,
    const std::vector<std::vector<float>>& damage_data,
    const std::vector<float>& confidence_data) {

    int64_t N = static_cast<int64_t>(tnf_cov_data.size());
    int64_t tc_dim = static_cast<int64_t>(tnf_cov_data[0].size());
    int64_t dmg_dim = static_cast<int64_t>(damage_data[0].size());

    torch::Tensor tnf_cov = torch::zeros({N, tc_dim});
    torch::Tensor damage = torch::zeros({N, dmg_dim});
    torch::Tensor dmg_conf = torch::zeros({N, 1});

    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < tc_dim; j++) {
            tnf_cov[i][j] = tnf_cov_data[i][j];
        }
        for (int64_t j = 0; j < dmg_dim; j++) {
            damage[i][j] = damage_data[i][j];
        }
        dmg_conf[i][0] = confidence_data[i];
    }

    return FeatureBatch{tnf_cov, damage, dmg_conf};
}

}  // namespace amber::bin2

#endif  // USE_LIBTORCH

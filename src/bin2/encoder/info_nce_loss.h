// AMBER bin2 - InfoNCE Loss
// NT-Xent loss for SimCLR-style contrastive learning
#pragma once

#ifdef USE_LIBTORCH

#include <torch/torch.h>

namespace amber::bin2 {

// InfoNCE loss (NT-Xent) for contrastive learning
// features: [2*B, emb_dim] - concatenated positive pairs (z1 stacked on z2)
// temperature: softmax temperature (typically 0.07-0.5)
// Returns: scalar loss value
inline torch::Tensor info_nce_loss(torch::Tensor features, float temperature) {
    int64_t total_size = features.size(0);
    int64_t batch_size = total_size / 2;

    // Compute similarity matrix: sim[i,j] = dot(z_i, z_j) / temperature
    // features are assumed to be L2-normalized
    torch::Tensor sim = torch::mm(features, features.t()) / temperature;  // [2B, 2B]

    // Create labels: positive pair for sample i is sample (i + B) % 2B
    // For first half (0..B-1): positive is at i+B
    // For second half (B..2B-1): positive is at i-B
    torch::Tensor labels = torch::arange(batch_size, features.options().dtype(torch::kLong));
    labels = torch::cat({labels + batch_size, labels}, 0);  // [2B]

    // Mask out self-similarity (diagonal) with large negative value
    torch::Tensor mask = torch::eye(total_size, features.options()).to(torch::kBool);
    sim.masked_fill_(mask, -1e9);

    // Cross-entropy loss treats each row as logits over all possible negatives + 1 positive
    return torch::nn::functional::cross_entropy(sim, labels);
}

// Overload: compute from two separate embedding tensors
// z1, z2: [B, emb_dim] - corresponding positive pairs
inline torch::Tensor info_nce_loss(torch::Tensor z1, torch::Tensor z2, float temperature) {
    torch::Tensor features = torch::cat({z1, z2}, 0);  // [2B, emb_dim]
    return info_nce_loss(features, temperature);
}

// Weighted InfoNCE loss - weights per sample for curriculum learning
// weights: [B] - weight for each sample pair (higher = more important)
inline torch::Tensor weighted_info_nce_loss(
    torch::Tensor z1, torch::Tensor z2,
    torch::Tensor weights, float temperature) {

    int64_t batch_size = z1.size(0);
    int64_t total_size = 2 * batch_size;

    torch::Tensor features = torch::cat({z1, z2}, 0);  // [2B, emb_dim]
    torch::Tensor sim = torch::mm(features, features.t()) / temperature;

    // Labels
    torch::Tensor labels = torch::arange(batch_size, z1.options().dtype(torch::kLong));
    labels = torch::cat({labels + batch_size, labels}, 0);

    // Mask diagonal
    torch::Tensor mask = torch::eye(total_size, z1.options()).to(torch::kBool);
    sim.masked_fill_(mask, -1e9);

    // Compute per-sample cross entropy
    torch::Tensor log_softmax = torch::log_softmax(sim, 1);
    torch::Tensor nll = -log_softmax.gather(1, labels.unsqueeze(1)).squeeze(1);  // [2B]

    // Apply weights (same weight for both views of a sample)
    torch::Tensor sample_weights = torch::cat({weights, weights}, 0);  // [2B]
    return (nll * sample_weights).sum() / sample_weights.sum();
}

}  // namespace amber::bin2

#endif  // USE_LIBTORCH

// AMBER Contrastive Encoder
// Exact port of COMEBin's EmbeddingNet architecture
// Architecture: Input -> Linear -> LeakyReLU -> [BN -> Dropout -> Linear -> LeakyReLU] x (n_layer-1) -> Linear
#pragma once

#ifdef USE_LIBTORCH

#include <torch/torch.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <random>
#include "damage_aware_infonce.h"

namespace amber::bin2 {

// COMEBin's EmbeddingNet architecture (EXACT match to mlp.py)
// https://github.com/ziyewang/COMEBin/blob/main/COMEBin/models/mlp.py
//
// Key differences from wrong implementation:
// 1. LeakyReLU (not ReLU) - train_CLmodel.py line 65: actn = nn.LeakyReLU()
// 2. n_layer-1 hidden blocks (not n_layer) - mlp.py line 12: self.n_embs = len(emb_szs) - 1
// 3. NO L2 normalization at output - normalization happens inside loss function
// 4. Optional residual hidden blocks for experimental architecture search
// 5. Optional mixture-of-experts fusion branch for modality-adaptive encoding
struct ResidualBlockImpl : torch::nn::Module {
    bool use_bn_ = true;
    float dropout_ = 0.0f;
    torch::nn::BatchNorm1d bn{nullptr};
    torch::nn::Dropout drop{nullptr};
    torch::nn::Linear fc{nullptr};

    ResidualBlockImpl(int hidden_sz, bool use_bn, float dropout)
        : use_bn_(use_bn), dropout_(dropout) {
        if (use_bn_) {
            bn = register_module("bn", torch::nn::BatchNorm1d(hidden_sz));
        }
        if (dropout_ > 0.0f) {
            drop = register_module("drop", torch::nn::Dropout(dropout_));
        }
        fc = register_module("fc", torch::nn::Linear(hidden_sz, hidden_sz));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto y = x;
        if (use_bn_) y = bn->forward(y);
        if (dropout_ > 0.0f) y = drop->forward(y);
        y = fc->forward(y);
        y = torch::leaky_relu(y, 0.01);
        return x + y;
    }
};
TORCH_MODULE(ResidualBlock);

// FT-Transformer style block for tabular numeric features.
struct FeatureTransformerBlockImpl : torch::nn::Module {
    int n_heads_ = 8;
    int head_dim_ = 16;
    float dropout_ = 0.0f;
    torch::nn::LayerNorm ln1{nullptr};
    torch::nn::LayerNorm ln2{nullptr};
    torch::nn::Linear q_proj{nullptr};
    torch::nn::Linear k_proj{nullptr};
    torch::nn::Linear v_proj{nullptr};
    torch::nn::Linear o_proj{nullptr};
    torch::nn::Linear ffn1{nullptr};
    torch::nn::Linear ffn2{nullptr};
    torch::nn::Dropout drop{nullptr};

    FeatureTransformerBlockImpl(int dim, int n_heads, int ffn_dim, float dropout)
        : n_heads_(std::max(1, n_heads)),
          head_dim_(dim / std::max(1, n_heads)),
          dropout_(dropout) {
        ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
        ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
        q_proj = register_module("q_proj", torch::nn::Linear(dim, dim));
        k_proj = register_module("k_proj", torch::nn::Linear(dim, dim));
        v_proj = register_module("v_proj", torch::nn::Linear(dim, dim));
        o_proj = register_module("o_proj", torch::nn::Linear(dim, dim));
        ffn1 = register_module("ffn1", torch::nn::Linear(dim, ffn_dim));
        ffn2 = register_module("ffn2", torch::nn::Linear(ffn_dim, dim));
        if (dropout_ > 0.0f) {
            drop = register_module("drop", torch::nn::Dropout(dropout_));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [B, T, C]
        auto y = ln1->forward(x);
        auto q = q_proj->forward(y);
        auto k = k_proj->forward(y);
        auto v = v_proj->forward(y);

        const auto B = q.size(0);
        const auto T = q.size(1);
        const auto C = q.size(2);

        q = q.view({B, T, n_heads_, head_dim_}).permute({0, 2, 1, 3});  // [B,H,T,D]
        k = k.view({B, T, n_heads_, head_dim_}).permute({0, 2, 1, 3});
        v = v.view({B, T, n_heads_, head_dim_}).permute({0, 2, 1, 3});

        auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_dim_));
        auto attn = torch::softmax(scores, -1);
        if (dropout_ > 0.0f) attn = drop->forward(attn);
        auto ctx = torch::matmul(attn, v).permute({0, 2, 1, 3}).contiguous().view({B, T, C});
        auto z = o_proj->forward(ctx);
        if (dropout_ > 0.0f) z = drop->forward(z);
        x = x + z;

        auto h = ln2->forward(x);
        h = torch::gelu(ffn1->forward(h));
        if (dropout_ > 0.0f) h = drop->forward(h);
        h = ffn2->forward(h);
        if (dropout_ > 0.0f) h = drop->forward(h);
        return x + h;
    }
};
TORCH_MODULE(FeatureTransformerBlock);

struct AmberEncoderImpl : torch::nn::Module {
    torch::nn::Sequential fc{nullptr};
    torch::nn::Sequential expert_a{nullptr};
    torch::nn::Sequential expert_b{nullptr};
    torch::nn::Linear input{nullptr};
    torch::nn::Linear output{nullptr};
    torch::nn::Linear gate{nullptr};
    torch::nn::ModuleList transformer_blocks{nullptr};
    torch::nn::LayerNorm ft_norm{nullptr};
    torch::nn::Linear ft_out{nullptr};
    torch::nn::ModuleList residual_blocks{nullptr};
    torch::Tensor cls_token_;
    torch::Tensor feature_weight_;
    torch::Tensor feature_bias_;
    int in_sz_, out_sz_;
    bool use_residual_encoder_ = false;
    bool use_moe_fusion_ = false;
    bool use_ft_transformer_ = false;
    int ft_token_dim_ = 128;

    AmberEncoderImpl(int in_sz, int out_sz, int hidden_sz = 2048,
                       int n_layer = 3, float dropout = 0.2f, bool use_bn = true,
                       bool use_residual_encoder = false,
                       bool use_moe_fusion = false,
                       bool use_ft_transformer = false,
                       int ft_token_dim = 128,
                       int ft_layers = 4,
                       int ft_heads = 8)
        : in_sz_(in_sz), out_sz_(out_sz),
          use_residual_encoder_(use_residual_encoder),
          use_moe_fusion_(use_moe_fusion),
          use_ft_transformer_(use_ft_transformer),
          ft_token_dim_(std::max(32, ft_token_dim)) {

        if (use_ft_transformer_) {
            use_moe_fusion_ = false;
            use_residual_encoder_ = false;
        } else if (use_moe_fusion_ && use_residual_encoder_) {
            // Keep behavior deterministic if both toggles are set.
            use_residual_encoder_ = false;
        }

        if (use_ft_transformer_) {
            const int n_heads = std::max(1, ft_heads);
            const int dim = std::max(n_heads, ft_token_dim_);
            // Force divisibility for manual multi-head split.
            ft_token_dim_ = (dim / n_heads) * n_heads;
            if (ft_token_dim_ <= 0) ft_token_dim_ = n_heads;
            const int layers = std::max(1, ft_layers);
            const int ffn_dim = ft_token_dim_ * 4;

            feature_weight_ = register_parameter(
                "feature_weight", torch::randn({in_sz, ft_token_dim_}) * 0.02);
            feature_bias_ = register_parameter(
                "feature_bias", torch::zeros({in_sz, ft_token_dim_}));
            cls_token_ = register_parameter(
                "cls_token", torch::zeros({1, 1, ft_token_dim_}));

            transformer_blocks = register_module("transformer_blocks", torch::nn::ModuleList());
            for (int i = 0; i < layers; i++) {
                transformer_blocks->push_back(
                    FeatureTransformerBlock(ft_token_dim_, n_heads, ffn_dim, dropout));
            }
            ft_norm = register_module(
                "ft_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ft_token_dim_})));
            ft_out = register_module("ft_out", torch::nn::Linear(ft_token_dim_, out_sz));

        } else if (use_moe_fusion_) {
            auto build_expert = [&](const std::string& name) {
                torch::nn::Sequential layers;
                layers->push_back(torch::nn::Linear(in_sz, hidden_sz));
                layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.01)));
                int n_hidden = n_layer - 1;
                for (int i = 0; i < n_hidden; i++) {
                    if (use_bn) {
                        layers->push_back(torch::nn::BatchNorm1d(hidden_sz));
                    }
                    if (dropout > 0) {
                        layers->push_back(torch::nn::Dropout(dropout));
                    }
                    layers->push_back(torch::nn::Linear(hidden_sz, hidden_sz));
                    layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.01)));
                }
                return register_module(name, layers);
            };

            expert_a = build_expert("expert_a");
            expert_b = build_expert("expert_b");
            gate = register_module("gate", torch::nn::Linear(in_sz, hidden_sz));
            output = register_module("output", torch::nn::Linear(hidden_sz, out_sz));

        } else if (!use_residual_encoder_) {
            torch::nn::Sequential layers;

            // Input layer: Linear -> LeakyReLU
            // COMEBin: layers = [nn.Linear(self.in_sz, emb_szs[0]), actn]
            layers->push_back(torch::nn::Linear(in_sz, hidden_sz));
            layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.01)));

            // Hidden layers: [BN -> Dropout -> Linear -> LeakyReLU] x (n_layer - 1)
            // COMEBin: for i in range(self.n_embs):  # where n_embs = len(emb_szs) - 1
            // With emb_szs = [2048] * n_layer, n_embs = n_layer - 1
            int n_hidden = n_layer - 1;  // CRITICAL FIX: COMEBin uses n_layer-1 hidden blocks
            for (int i = 0; i < n_hidden; i++) {
                if (use_bn) {
                    layers->push_back(torch::nn::BatchNorm1d(hidden_sz));
                }
                if (dropout > 0) {
                    layers->push_back(torch::nn::Dropout(dropout));
                }
                layers->push_back(torch::nn::Linear(hidden_sz, hidden_sz));
                layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.01)));
            }

            // Output layer: Linear (no activation - matches COMEBin)
            // COMEBin: layers.append(nn.Linear(emb_szs[-1], self.out_sz))
            layers->push_back(torch::nn::Linear(hidden_sz, out_sz));

            fc = register_module("fc", layers);
        } else {
            // Experimental residual encoder:
            // Input -> LeakyReLU -> ResidualBlock x (n_layer-1) -> Output
            input = register_module("input", torch::nn::Linear(in_sz, hidden_sz));
            residual_blocks = register_module("residual_blocks", torch::nn::ModuleList());
            int n_hidden = n_layer - 1;
            for (int i = 0; i < n_hidden; ++i) {
                residual_blocks->push_back(ResidualBlock(hidden_sz, use_bn, dropout));
            }
            output = register_module("output", torch::nn::Linear(hidden_sz, out_sz));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        // No L2 normalization here - COMEBin normalizes inside the loss function
        if (use_ft_transformer_) {
            auto tokens = x.unsqueeze(-1) * feature_weight_.unsqueeze(0) + feature_bias_.unsqueeze(0);
            auto cls = cls_token_.expand({x.size(0), 1, ft_token_dim_});
            auto seq = torch::cat({cls, tokens}, 1);  // [B, D+1, C]
            for (auto& block : *transformer_blocks) {
                seq = block->as<FeatureTransformerBlockImpl>()->forward(seq);
            }
            seq = ft_norm->forward(seq);
            auto cls_repr = seq.index({torch::indexing::Slice(), 0, torch::indexing::Slice()});
            return ft_out->forward(cls_repr);
        }

        if (use_moe_fusion_) {
            auto h1 = expert_a->forward(x);
            auto h2 = expert_b->forward(x);
            auto g = torch::sigmoid(gate->forward(x));
            auto h = g * h1 + (1.0f - g) * h2;
            return output->forward(h);
        }

        if (!use_residual_encoder_) {
            return fc->forward(x);
        }

        auto h = torch::leaky_relu(input->forward(x), 0.01);
        for (auto& block : *residual_blocks) {
            h = block->as<ResidualBlockImpl>()->forward(h);
        }
        return output->forward(h);
    }

    // Encode a dense feature tensor (N x D). Output is moved to CPU.
    torch::Tensor encode_tensor(const torch::Tensor& features, bool l2_normalize_output = false) {
        this->eval();
        torch::NoGradGuard no_grad;

        torch::Device device(torch::kCPU);
        for (const auto& p : this->parameters()) {
            device = p.device();
            break;
        }

        auto out = this->forward(features.to(device));
        if (l2_normalize_output) {
            out = torch::nn::functional::normalize(
                out, torch::nn::functional::NormalizeFuncOptions().dim(1));
        }
        return out.to(torch::kCPU);
    }

    // Encode batch of features (for inference after training)
    // L2 normalization happens here for final embeddings output
    std::vector<std::vector<float>> encode(const std::vector<std::vector<float>>& features) {
        this->eval();
        torch::NoGradGuard no_grad;

        int N = features.size();
        int D = features[0].size();

        std::vector<float> flat_data(N * D);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                flat_data[i * D + j] = features[i][j];
            }
        }

        // FT-Transformer is much heavier at inference too, so keep chunk sizes conservative.
        const int chunk_size = use_ft_transformer_ ? 256 : 4096;

        std::vector<std::vector<float>> embeddings(N);
        for (int i = 0; i < N; i++) embeddings[i].resize(out_sz_);

        for (int start = 0; start < N; start += chunk_size) {
            int bs = std::min(chunk_size, N - start);
            auto batch = torch::from_blob(
                flat_data.data() + static_cast<size_t>(start) * D,
                {bs, D}, torch::kFloat32).clone();
            auto out = encode_tensor(batch, false);
            auto out_acc = out.accessor<float, 2>();
            for (int i = 0; i < bs; i++) {
                for (int j = 0; j < out_sz_; j++) {
                    embeddings[start + i][j] = out_acc[i][j];
                }
            }
        }

        return embeddings;
    }
};
TORCH_MODULE(AmberEncoder);

// Compute top-1 accuracy for early stopping (COMEBin uses this)
inline float compute_accuracy(torch::Tensor logits, torch::Tensor labels, int topk = 1) {
    auto [values, indices] = logits.topk(topk, 1, true, true);
    auto correct = indices.eq(labels.view({-1, 1}).expand_as(indices));
    return correct.to(torch::kFloat).sum().item<float>() / labels.size(0) * 100.0f;
}

// COMEBin-style contrastive training with substring augmentation
// Exact match to simclr.py and train_CLmodel.py
class AmberTrainer {
public:
    struct Config {
        int epochs = 200;
        int batch_size = 1024;
        float lr = 0.001f;
        float weight_decay = 1e-4f;
        float temperature = 0.1f;  // COMEBin default
        int hidden_sz = 2048;
        int n_layer = 3;
        int out_dim = 128;
        float dropout = 0.2f;
        int n_views = 6;           // COMEBin default: 6 views (original + 5 augmented)
        bool use_substring_aug = true;
        bool earlystop = false;    // COMEBin default is False (opt-in via --earlystop)
        bool use_scheduler = true; // COMEBin default (cosine annealing)
        int warmup_epochs = 10;    // COMEBin: scheduler doesn't step for first 10 epochs
        // Damage-aware InfoNCE parameters (experimental)
        bool use_damage_infonce = false;  // Use damage-weighted negative pairs
        float damage_lambda = 0.5f;       // Max attenuation strength (0.4-0.6)
        float damage_wmin = 0.5f;         // Minimum negative weight (graph connectivity)
        int seed = 42;                    // Random seed for reproducibility
        // Stochastic Weight Averaging: average model weights over last epochs
        bool use_swa = true;
        int swa_start_epoch = 70;         // Start accumulating from this epoch
    };

    AmberTrainer(const Config& config)
        : config_(config),
          device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
        // Set random seed for reproducibility
        torch::manual_seed(config.seed);
        if (torch::cuda::is_available()) {
            std::cerr << "[AmberTrainer] Using CUDA GPU (seed=" << config.seed << ")\n";
        } else {
            std::cerr << "[AmberTrainer] CUDA not available, using CPU (seed=" << config.seed << ")\n";
        }
    }

    // COMEBin-exact InfoNCE loss (simclr.py lines 34-65)
    std::pair<torch::Tensor, torch::Tensor> info_nce_loss(
        torch::Tensor features, int batch_size, int n_views) {

        // Labels: each sample matches with its corresponding sample in other views
        // COMEBin: labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        std::vector<torch::Tensor> label_parts;
        for (int v = 0; v < n_views; v++) {
            label_parts.push_back(torch::arange(batch_size, device_));
        }
        auto labels = torch::cat(label_parts, 0);

        // COMEBin: labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(torch::kFloat);

        // L2 normalize features (COMEBin: features = F.normalize(features, dim=1))
        features = torch::nn::functional::normalize(features,
            torch::nn::functional::NormalizeFuncOptions().dim(1));

        // Similarity matrix
        auto similarity_matrix = torch::mm(features, features.t());

        // Discard main diagonal from both labels and similarities
        auto mask = torch::eye(labels.size(0), features.options().dtype(torch::kBool));
        labels = labels.masked_select(~mask).view({labels.size(0), -1});
        similarity_matrix = similarity_matrix.masked_select(~mask).view({similarity_matrix.size(0), -1});

        // Select and combine positives
        auto positives = similarity_matrix.masked_select(labels.to(torch::kBool)).view({-1, 1});

        // Select negatives
        auto negatives = similarity_matrix.masked_select(~labels.to(torch::kBool))
            .view({similarity_matrix.size(0), -1});

        // Expand negatives for each positive
        // COMEBin: negatives = negatives[:, None].expand(-1, self.args.n_views - 1, -1).flatten(0, 1)
        negatives = negatives.index({torch::indexing::Slice(), torch::indexing::None})
            .expand({-1, n_views - 1, -1}).flatten(0, 1);

        // Logits: [positives, negatives]
        auto logits = torch::cat({positives, negatives}, 1) / config_.temperature;
        auto target = torch::zeros(logits.size(0), torch::TensorOptions().dtype(torch::kLong).device(device_));

        return {logits, target};
    }

    // Train encoder with multi-view features (substring augmentation)
    // views[view_idx] contains features for all contigs in that view
    // Returns: best loss achieved during training
    float train_multiview(AmberEncoder& encoder,
                          const std::vector<torch::Tensor>& views) {

        int n_views = views.size();
        int N = views[0].size(0);
        int D = views[0].size(1);

        std::cerr << "[AmberTrainer] Training with " << n_views << " views, "
                  << N << " contigs, " << D << " features\n";

        // Move encoder to device (GPU if available)
        encoder->to(device_);

        // Move views to device for fast GPU indexing
        std::vector<torch::Tensor> views_gpu;
        for (int v = 0; v < n_views; v++) {
            views_gpu.push_back(views[v].to(device_));
        }

        // Setup optimizer (AdamW with weight decay - matches COMEBin)
        torch::optim::AdamW optimizer(
            encoder->parameters(),
            torch::optim::AdamWOptions(config_.lr).weight_decay(config_.weight_decay)
        );

        // Cosine annealing scheduler (COMEBin: T_max = epochs, eta_min = 0)
        // Note: COMEBin only steps scheduler after warmup_epochs (default 10)

        // COMEBin uses drop_last=True, so we only use full batches
        int n_full_batches = N / config_.batch_size;
        if (n_full_batches == 0) {
            n_full_batches = 1;  // At least one batch
        }
        std::cerr << "[AmberTrainer] Training on " << n_full_batches
                  << " minibatches (drop_last=True), epochs=" << config_.epochs << "\n";

        float best_loss = std::numeric_limits<float>::max();
        int best_epoch = 0;
        int earlystop_epoch = 0;

        // SWA state
        std::vector<torch::Tensor> swa_sums;
        int swa_count = 0;
        if (config_.use_swa) {
            for (const auto& param : encoder->parameters())
                swa_sums.push_back(torch::zeros_like(param));
        }

        // Loss function
        auto criterion = torch::nn::CrossEntropyLoss();

        for (int epoch = 1; epoch <= config_.epochs; epoch++) {
            encoder->train();
            auto total_loss_gpu = torch::zeros({1}, torch::TensorOptions().device(device_));
            float last_accuracy = 0;

            // Shuffle indices
            std::vector<int> indices(N);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));

            for (int b = 0; b < n_full_batches; b++) {
                int start = b * config_.batch_size;
                int bs = config_.batch_size;

                if (start + bs > N) break;

                std::vector<long> batch_idx(bs);
                for (int i = 0; i < bs; i++) batch_idx[i] = indices[start + i];
                auto idx_tensor = torch::tensor(batch_idx).to(device_);

                std::vector<torch::Tensor> batch_views;
                for (int v = 0; v < n_views; v++)
                    batch_views.push_back(views_gpu[v].index_select(0, idx_tensor));
                auto all_features = torch::cat(batch_views, 0);

                auto embeddings = encoder->forward(all_features);
                auto [logits, target] = info_nce_loss(embeddings, bs, n_views);
                auto loss = criterion(logits, target);

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                total_loss_gpu += loss.detach();
                if (b == n_full_batches - 1)
                    last_accuracy = compute_accuracy(logits, target, 1);
            }

            float epoch_loss = total_loss_gpu.item<float>() / std::max(1, n_full_batches);

            if (epoch_loss < best_loss) {
                best_loss = epoch_loss;
                best_epoch = epoch;
            }

            // Cosine annealing scheduler step (only after warmup)
            // COMEBin: if epoch_counter >= 10: self.scheduler.step()
            if (config_.use_scheduler && epoch >= config_.warmup_epochs) {
                // Manual cosine annealing: lr = eta_min + 0.5 * (lr_init - eta_min) * (1 + cos(pi * t / T_max))
                int t = epoch - config_.warmup_epochs;
                int T_max = config_.epochs - config_.warmup_epochs;
                float lr_new = 0.5f * config_.lr * (1.0f + std::cos(M_PI * t / T_max));
                for (auto& param_group : optimizer.param_groups()) {
                    static_cast<torch::optim::AdamWOptions&>(param_group.options()).lr(lr_new);
                }
            }

            // SWA accumulation
            if (config_.use_swa && epoch >= config_.swa_start_epoch) {
                torch::NoGradGuard no_grad;
                int idx = 0;
                for (const auto& param : encoder->parameters()) {
                    swa_sums[idx].add_(param);
                    idx++;
                }
                swa_count++;
            }

            // Early stopping (COMEBin: if top1 > 99% for 3 consecutive epochs after epoch 10)
            if (config_.earlystop) {
                if (epoch >= config_.warmup_epochs && last_accuracy > 99.0f) {
                    earlystop_epoch++;
                } else {
                    earlystop_epoch = 0;
                }
                if (earlystop_epoch >= 3) {
                    std::cerr << "  Early stopping at epoch " << epoch
                              << " (accuracy > 99% for 3 epochs)\n";
                    break;
                }
            }

            if (epoch == 1 || epoch % 10 == 0 || epoch == config_.epochs) {
                std::cerr << "  Epoch " << epoch << "/" << config_.epochs
                          << " - Loss: " << std::fixed << std::setprecision(5) << epoch_loss
                          << " Acc: " << std::setprecision(1) << last_accuracy << "%"
                          << " (best: " << std::setprecision(5) << best_loss << " @ " << best_epoch << ")\n";
            }
        }

        // Apply SWA averaged weights
        if (config_.use_swa) apply_swa(encoder, swa_sums, swa_count);

        std::cerr << "[AmberTrainer] Training complete. Best loss: "
                  << best_loss << " at epoch " << best_epoch << "\n";
        return best_loss;
    }

    // Damage-aware training: weights negative pairs by damage compatibility
    // Uses DamageAwareInfoNCE instead of standard InfoNCE when damage_profiles provided
    // Returns: best loss achieved during training
    float train_multiview(AmberEncoder& encoder,
                          const std::vector<torch::Tensor>& views,
                          const std::vector<DamageProfile>& damage_profiles) {

        // If no damage profiles or damage not enabled, use standard training
        if (damage_profiles.empty() || !config_.use_damage_infonce) {
            return train_multiview(encoder, views);
        }

        int n_views = views.size();
        int N = views[0].size(0);
        int D = views[0].size(1);

        std::cerr << "[AmberTrainer] Damage-aware training with " << n_views << " views, "
                  << N << " contigs, " << D << " features\n";
        std::cerr << "[AmberTrainer] Damage params: lambda=" << config_.damage_lambda
                  << ", wmin=" << config_.damage_wmin << "\n";

        // Move encoder to device (GPU if available)
        encoder->to(device_);

        // Move views to device for fast GPU indexing
        std::vector<torch::Tensor> views_gpu;
        for (int v = 0; v < n_views; v++) {
            views_gpu.push_back(views[v].to(device_));
        }

        // Setup damage-aware InfoNCE
        DamageInfoNCEParams damage_params;
        damage_params.enabled = true;
        damage_params.lambda = config_.damage_lambda;
        damage_params.wmin = config_.damage_wmin;
        DamageAwareInfoNCE damage_infonce(config_.temperature, damage_params);

        // Setup optimizer (AdamW with weight decay - matches COMEBin)
        torch::optim::AdamW optimizer(
            encoder->parameters(),
            torch::optim::AdamWOptions(config_.lr).weight_decay(config_.weight_decay)
        );

        // COMEBin uses drop_last=True, so we only use full batches
        int n_full_batches = N / config_.batch_size;
        if (n_full_batches == 0) {
            n_full_batches = 1;
        }
        std::cerr << "[AmberTrainer] Training on " << n_full_batches
                  << " minibatches (drop_last=True), epochs=" << config_.epochs << "\n";

        float best_loss = std::numeric_limits<float>::max();
        int best_epoch = 0;
        int earlystop_epoch = 0;

        // SWA state
        std::vector<torch::Tensor> swa_sums;
        int swa_count = 0;
        if (config_.use_swa) {
            for (const auto& param : encoder->parameters())
                swa_sums.push_back(torch::zeros_like(param));
        }

        auto criterion = torch::nn::CrossEntropyLoss();

        for (int epoch = 1; epoch <= config_.epochs; epoch++) {
            encoder->train();
            // Accumulate loss on GPU — sync to CPU once per epoch, not per batch.
            auto total_loss_gpu = torch::zeros({1}, torch::TensorOptions().device(device_));
            float last_accuracy = 0;

            // Shuffle indices
            std::vector<int> indices(N);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));

            for (int b = 0; b < n_full_batches; b++) {
                int start = b * config_.batch_size;
                int bs = config_.batch_size;

                if (start + bs > N) break;

                // Get batch indices
                std::vector<long> batch_idx(bs);
                std::vector<DamageProfile> batch_damage(bs);
                for (int i = 0; i < bs; i++) {
                    batch_idx[i] = indices[start + i];
                    batch_damage[i] = damage_profiles[indices[start + i]];
                }
                auto idx_tensor = torch::tensor(batch_idx).to(device_);

                // Get batches from each view and concatenate
                std::vector<torch::Tensor> batch_views;
                for (int v = 0; v < n_views; v++) {
                    batch_views.push_back(views_gpu[v].index_select(0, idx_tensor));
                }
                auto all_features = torch::cat(batch_views, 0);

                // Forward pass through encoder
                auto embeddings = encoder->forward(all_features);

                // L2 normalize for InfoNCE
                embeddings = torch::nn::functional::normalize(embeddings,
                    torch::nn::functional::NormalizeFuncOptions().dim(1));

                // Compute damage-aware InfoNCE loss
                auto [logits, target] = damage_infonce.forward(embeddings, bs, n_views, batch_damage);
                auto loss = criterion(logits, target);

                // Backward
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                total_loss_gpu += loss.detach();
                // Only compute accuracy on last batch (used for early stopping)
                if (b == n_full_batches - 1)
                    last_accuracy = compute_accuracy(logits, target, 1);
            }

            // Single CPU-GPU sync per epoch
            float epoch_loss = total_loss_gpu.item<float>() / std::max(1, n_full_batches);

            if (epoch_loss < best_loss) {
                best_loss = epoch_loss;
                best_epoch = epoch;
            }

            // Cosine annealing scheduler step (only after warmup)
            if (config_.use_scheduler && epoch >= config_.warmup_epochs) {
                int t = epoch - config_.warmup_epochs;
                int T_max = config_.epochs - config_.warmup_epochs;
                float lr_new = 0.5f * config_.lr * (1.0f + std::cos(M_PI * t / T_max));
                for (auto& param_group : optimizer.param_groups()) {
                    static_cast<torch::optim::AdamWOptions&>(param_group.options()).lr(lr_new);
                }
            }

            // SWA accumulation
            if (config_.use_swa && epoch >= config_.swa_start_epoch) {
                torch::NoGradGuard no_grad;
                int idx = 0;
                for (const auto& param : encoder->parameters()) {
                    swa_sums[idx].add_(param);
                    idx++;
                }
                swa_count++;
            }

            // Early stopping parity with standard training path:
            // stop if top-1 >99% for 3 consecutive epochs after warmup.
            if (config_.earlystop) {
                if (epoch >= config_.warmup_epochs && last_accuracy > 99.0f) {
                    earlystop_epoch++;
                } else {
                    earlystop_epoch = 0;
                }
                if (earlystop_epoch >= 3) {
                    std::cerr << "  Early stopping at epoch " << epoch
                              << " (accuracy > 99% for 3 epochs)\n";
                    break;
                }
            }

            if (epoch == 1 || epoch % 10 == 0 || epoch == config_.epochs) {
                std::cerr << "  Epoch " << epoch << "/" << config_.epochs
                          << " - Loss: " << std::fixed << std::setprecision(5) << epoch_loss
                          << " Acc: " << std::setprecision(1) << last_accuracy << "%"
                          << " (best: " << std::setprecision(5) << best_loss << " @ " << best_epoch << ")\n";
            }
        }

        // Apply SWA averaged weights
        if (config_.use_swa) apply_swa(encoder, swa_sums, swa_count);

        std::cerr << "[AmberTrainer] Damage-aware training complete. Best loss: "
                  << best_loss << " at epoch " << best_epoch << "\n";
        return best_loss;
    }

    // Legacy: Train encoder on TNF features with noise augmentation (fallback)
    void train(AmberEncoder& encoder,
               const std::vector<std::vector<float>>& features) {

        int N = features.size();
        int D = features[0].size();

        // Convert to tensor
        torch::Tensor data = torch::zeros({N, D});
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                data[i][j] = features[i][j];
            }
        }

        // Setup optimizer (AdamW with weight decay)
        torch::optim::AdamW optimizer(
            encoder->parameters(),
            torch::optim::AdamWOptions(config_.lr).weight_decay(config_.weight_decay)
        );

        int n_batches = N / config_.batch_size;  // drop_last=True
        if (n_batches == 0) n_batches = 1;

        std::cerr << "[AmberTrainer] Training on " << n_batches
                  << " minibatches, epochs=" << config_.epochs << "\n";

        float best_loss = std::numeric_limits<float>::max();
        int best_epoch = 0;
        auto criterion = torch::nn::CrossEntropyLoss();

        // SWA state
        std::vector<torch::Tensor> swa_sums;
        int swa_count = 0;
        if (config_.use_swa) {
            for (const auto& param : encoder->parameters())
                swa_sums.push_back(torch::zeros_like(param));
        }

        for (int epoch = 1; epoch <= config_.epochs; epoch++) {
            encoder->train();
            auto total_loss_gpu = torch::zeros({1}, torch::TensorOptions().device(device_));

            // Shuffle indices
            std::vector<int> indices(N);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));

            for (int b = 0; b < n_batches; b++) {
                int start = b * config_.batch_size;
                int bs = config_.batch_size;

                if (start + bs > N) break;

                // Get batch
                std::vector<long> batch_idx(bs);
                for (int i = 0; i < bs; i++) {
                    batch_idx[i] = indices[start + i];
                }
                auto idx_tensor = torch::tensor(batch_idx);
                auto batch = data.index_select(0, idx_tensor);

                // Data augmentation (noise-based fallback)
                auto mask1 = torch::rand_like(batch) > 0.15f;
                auto aug1 = batch * mask1 + 0.02f * torch::randn_like(batch);
                auto mask2 = torch::rand_like(batch) > 0.15f;
                auto aug2 = batch * mask2 + 0.02f * torch::randn_like(batch);

                // Forward (2 views)
                auto z1 = encoder->forward(aug1);
                auto z2 = encoder->forward(aug2);
                auto embeddings = torch::cat({z1, z2}, 0);

                // Use InfoNCE loss
                auto [logits, target] = info_nce_loss(embeddings, bs, 2);
                auto loss = criterion(logits, target);

                // Backward
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                total_loss_gpu += loss.detach();
            }

            float epoch_loss = total_loss_gpu.item<float>() / std::max(1, n_batches);

            if (epoch_loss < best_loss) {
                best_loss = epoch_loss;
                best_epoch = epoch;
            }

            // SWA accumulation
            if (config_.use_swa && epoch >= config_.swa_start_epoch) {
                torch::NoGradGuard no_grad;
                int idx = 0;
                for (const auto& param : encoder->parameters()) {
                    swa_sums[idx].add_(param);
                    idx++;
                }
                swa_count++;
            }

            if (epoch == 1 || epoch % 10 == 0 || epoch == config_.epochs) {
                std::cerr << "  Epoch " << epoch << "/" << config_.epochs
                          << " - Loss: " << std::fixed << std::setprecision(5) << epoch_loss
                          << " (best: " << best_loss << " @ " << best_epoch << ")\n";
            }
        }

        // Apply SWA averaged weights
        if (config_.use_swa) apply_swa(encoder, swa_sums, swa_count);

        std::cerr << "[AmberTrainer] Training complete. Best loss: "
                  << best_loss << " at epoch " << best_epoch << "\n";
    }

    // Apply SWA: average accumulated parameter sums into the model
    void apply_swa(AmberEncoder& encoder,
                   const std::vector<torch::Tensor>& swa_sums,
                   int swa_count) {
        if (swa_count <= 0) return;
        torch::NoGradGuard no_grad;
        int idx = 0;
        for (auto& param : encoder->parameters()) {
            param.copy_(swa_sums[idx] / static_cast<float>(swa_count));
            idx++;
        }
        std::cerr << "[SWA] Averaged weights over " << swa_count
                  << " epochs (swa_start=" << config_.swa_start_epoch << ")\n";
    }

private:
    Config config_;
    torch::Device device_;
};

} // namespace amber::bin2

#endif // USE_LIBTORCH

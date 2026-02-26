// AMBER bin2 - Post-processing bin confidence scorer
// Fuses four evidence sources into a per-contig confidence score:
//   P_graph  — fraction of edge weight inside the assigned bin
//   P_embed  — Cauchy kernel distance to bin embedding centroid
//   P_dmg    — aDNA damage profile cosine similarity to bin centroid
//   P_scg    — single-copy gene marker consistency within the bin
// Combined as softmax over S = w_g*logit(Pg) + w_e*logit(Pe) + r_dmg*w_d*logit(Pd) + r_scg*w_s*logit(Ps)
#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iomanip>

#include "clustering_types.h"
#include "../encoder/damage_aware_infonce.h"
#include "../encoder/scg_hard_negatives.h"

namespace amber::bin2 {

struct ContigConfidence {
    float confidence   = 0.0f;  // P(assigned bin | contig)
    float margin       = 0.0f;  // P1 - P2 over candidate bins
    float entropy      = 0.0f;  // -sum(p * log p) over candidates
    bool ambiguous       = false;  // margin < ambiguous_threshold
    bool graph_boundary  = false;  // >30% edge weight crosses to other bins
    bool damage_conflict = false;  // P_dmg to own bin < damage_conflict_threshold
    bool scg_conflict    = false;  // has marker already present in assigned bin
};

class BinConfidenceScorer {
public:
    struct Config {
        float w_graph  = 1.0f;  // weight for graph evidence
        float w_embed  = 1.0f;  // weight for embedding evidence
        float w_dmg    = 0.5f;  // weight for damage evidence
        float w_scg    = 0.5f;  // weight for SCG evidence
        float n0       = 10.0f; // n_reads saturation: r_dmg = min(1, n_reads/n0)
        float temperature = 1.0f;              // softmax temperature (calibrated if anchors found)
        float ambiguous_threshold      = 0.1f; // margin < this → ambiguous flag
        float graph_boundary_ratio     = 0.3f; // out_w / (in_w + out_w) > this → boundary flag
        float damage_conflict_threshold = 0.5f; // P_dmg < this → damage_conflict flag
        int   min_anchor_reads         = 5;    // min n_reads for calibration anchors
    };

    BinConfidenceScorer() : cfg_(Config{}) {}
    explicit BinConfidenceScorer(const Config& cfg) : cfg_(cfg) {}

    // Compute confidence for all contigs.
    // Returns one ContigConfidence per contig (index matches labels/embeddings/etc.).
    // Unbinned contigs (labels[i] < 0) get zero-valued ContigConfidence.
    std::vector<ContigConfidence> score(
        const std::vector<int>& labels,
        const std::vector<std::vector<float>>& embeddings,
        const std::vector<DamageProfile>& damage_profiles,
        const ContigMarkerLookup& scg_lookup,
        const std::vector<WeightedEdge>& edges) const
    {
        int N = (int)labels.size();
        std::vector<ContigConfidence> result(N);

        // --- 1. Build adjacency list: adj[i] = [(j, weight), ...] ---
        std::vector<std::vector<std::pair<int, float>>> adj(N);
        for (const auto& e : edges) {
            adj[e.u].emplace_back(e.v, e.w);
            adj[e.v].emplace_back(e.u, e.w);
        }

        // --- 2. Cluster membership: cluster_id → sorted contig indices ---
        std::unordered_map<int, std::vector<int>> clusters;
        for (int i = 0; i < N; i++)
            if (labels[i] >= 0)
                clusters[labels[i]].push_back(i);

        // Sequential index into cluster_ids[] for stable array indexing
        std::vector<int> cluster_ids;
        cluster_ids.reserve(clusters.size());
        for (const auto& [k, _] : clusters)
            cluster_ids.push_back(k);
        std::sort(cluster_ids.begin(), cluster_ids.end());

        std::unordered_map<int, int> cluster_idx;  // label → sequential k
        for (int k = 0; k < (int)cluster_ids.size(); k++)
            cluster_idx[cluster_ids[k]] = k;

        int K = (int)cluster_ids.size();
        if (K == 0) return result;

        // --- 3. Embedding centroids ---
        int D = embeddings.empty() ? 0 : (int)embeddings[0].size();
        std::vector<std::vector<float>> centroids(K, std::vector<float>(D, 0.0f));
        std::vector<int> cluster_sizes(K, 0);
        for (int i = 0; i < N; i++) {
            if (labels[i] < 0) continue;
            int k = cluster_idx.at(labels[i]);
            cluster_sizes[k]++;
            for (int d = 0; d < D; d++)
                centroids[k][d] += embeddings[i][d];
        }
        for (int k = 0; k < K; k++)
            if (cluster_sizes[k] > 0)
                for (int d = 0; d < D; d++)
                    centroids[k][d] /= cluster_sizes[k];

        // --- 4. Per-contig total edge weight and per-bin intra weight ---
        // intra_w_by_label[i][label] = sum of edge weights from i to contigs in cluster `label`
        std::vector<float> total_w(N, 0.0f);
        std::vector<std::unordered_map<int, float>> intra_w(N);
        for (int i = 0; i < N; i++) {
            for (const auto& [j, w] : adj[i]) {
                total_w[i] += w;
                if (labels[j] >= 0)
                    intra_w[i][labels[j]] += w;
            }
        }

        // --- 5. Damage centroids (20-dim aDNA feature vector per bin) ---
        bool has_dmg = !damage_profiles.empty();
        std::vector<std::vector<float>> dmg_centroids(K, std::vector<float>(20, 0.0f));
        std::vector<int> dmg_counts(K, 0);
        if (has_dmg) {
            for (int i = 0; i < N; i++) {
                if (labels[i] < 0 || damage_profiles[i].n_reads == 0) continue;
                int k = cluster_idx.at(labels[i]);
                auto feats = damage_profiles[i].get_adna_features();
                for (int d = 0; d < 20; d++)
                    dmg_centroids[k][d] += feats[d];
                dmg_counts[k]++;
            }
            for (int k = 0; k < K; k++)
                if (dmg_counts[k] > 0)
                    for (int d = 0; d < 20; d++)
                        dmg_centroids[k][d] /= dmg_counts[k];
        }

        // --- 6. Per-bin marker counts ---
        bool has_scg = !scg_lookup.empty();
        int n_markers = 0;
        if (has_scg) {
            for (int i = 0; i < N; i++)
                for (int m : scg_lookup.contig_marker_ids[i])
                    n_markers = std::max(n_markers, m + 1);
        }
        std::vector<std::vector<int>> bin_marker_counts(K, std::vector<int>(n_markers, 0));
        if (has_scg) {
            for (int i = 0; i < N; i++) {
                if (labels[i] < 0) continue;
                int k = cluster_idx.at(labels[i]);
                for (int m : scg_lookup.contig_marker_ids[i])
                    bin_marker_counts[k][m]++;
            }
        }

        // --- 7. Helpers ---
        auto logit = [](float p) -> float {
            p = std::clamp(p, 1e-6f, 1.0f - 1e-6f);
            return std::log(p / (1.0f - p));
        };

        auto dot_norm = [](const std::vector<float>& a, const std::vector<float>& b) -> float {
            float da = 0, db = 0, ab = 0;
            for (int i = 0; i < (int)a.size(); i++) {
                da += a[i] * a[i]; db += b[i] * b[i]; ab += a[i] * b[i];
            }
            return (da < 1e-12f || db < 1e-12f) ? 0.0f : ab / std::sqrt(da * db);
        };

        auto sq_dist = [](const std::vector<float>& a, const std::vector<float>& b) -> float {
            float s = 0;
            for (int i = 0; i < (int)a.size(); i++) { float d = a[i] - b[i]; s += d * d; }
            return s;
        };

        // Median squared L2 norm of embeddings (P_embed normalization)
        float median_sq_norm = 1.0f;
        if (D > 0 && N > 0) {
            std::vector<float> norms(N, 0.0f);
            for (int i = 0; i < N; i++)
                for (float x : embeddings[i]) norms[i] += x * x;
            std::nth_element(norms.begin(), norms.begin() + N / 2, norms.end());
            median_sq_norm = std::max(norms[N / 2], 1e-6f);
        }

        // --- 8. Temperature calibration on SCG-anchor contigs ---
        // Anchor: SCG-bearing contig in a marker-clean bin with enough reads.
        float T = cfg_.temperature;
        {
            std::vector<float> anchor_raw_scores;
            for (int i = 0; i < N; i++) {
                if (labels[i] < 0) continue;
                if (!has_scg || scg_lookup.contig_marker_ids[i].empty()) continue;
                if (has_dmg && damage_profiles[i].n_reads < cfg_.min_anchor_reads) continue;
                int k = cluster_idx.at(labels[i]);
                bool clean = true;
                for (int m : scg_lookup.contig_marker_ids[i])
                    if (bin_marker_counts[k][m] > 1) { clean = false; break; }
                if (!clean) continue;

                float score = 0.0f;
                float pg = (total_w[i] > 0)
                    ? (intra_w[i].count(labels[i]) ? intra_w[i].at(labels[i]) / total_w[i] : 0.0f)
                    : 0.5f;
                score += cfg_.w_graph * logit(pg);
                if (D > 0) {
                    float d2 = sq_dist(embeddings[i], centroids[k]);
                    score += cfg_.w_embed * logit(1.0f / (1.0f + d2 / median_sq_norm));
                }
                anchor_raw_scores.push_back(score);
            }
            // Set T so the median anchor raw score maps to confidence ≈ 0.9.
            // With K candidates and uniform background: exp(s/T) / (exp(s/T) + (K-1)) = 0.9
            // → s/T = log(9*(K-1)) → T = s / log(9*(K-1))
            if ((int)anchor_raw_scores.size() >= 10) {
                std::sort(anchor_raw_scores.begin(), anchor_raw_scores.end());
                float median_s = anchor_raw_scores[anchor_raw_scores.size() / 2];
                float target   = std::log(9.0f * std::max(1, K - 1));
                if (target > 0 && median_s > 0)
                    T = median_s / target;
                T = std::clamp(T, 0.1f, 10.0f);
            }
        }

        // --- 9. Score each binned contig ---
        for (int i = 0; i < N; i++) {
            if (labels[i] < 0) continue;
            int assigned_k = cluster_idx.at(labels[i]);

            // Candidate bins: assigned + any bin reachable via edges
            std::unordered_map<int, float> raw_scores;  // sequential k → raw score
            raw_scores[assigned_k] = 0.0f;
            for (const auto& [j, _] : adj[i])
                if (labels[j] >= 0)
                    raw_scores[cluster_idx.at(labels[j])] = 0.0f;

            // Reliability weights
            float r_dmg = 0.0f;
            std::vector<float> adna_i;
            if (has_dmg) {
                r_dmg = std::min(1.0f, (float)damage_profiles[i].n_reads / cfg_.n0);
                adna_i = damage_profiles[i].get_adna_features();
            }
            float r_scg = (has_scg && !scg_lookup.contig_marker_ids[i].empty()) ? 1.0f : 0.0f;

            for (auto& [k, raw] : raw_scores) {
                int label_k = cluster_ids[k];  // actual cluster label for this candidate

                // P_graph: fraction of contig's edge weight pointing into bin k
                float pg = 0.5f;
                if (total_w[i] > 0) {
                    float iw = intra_w[i].count(label_k) ? intra_w[i].at(label_k) : 0.0f;
                    pg = iw / total_w[i];
                }
                raw += cfg_.w_graph * logit(pg);

                // P_embed: Cauchy kernel distance to bin centroid
                if (D > 0) {
                    float d2 = sq_dist(embeddings[i], centroids[k]);
                    float pe = 1.0f / (1.0f + d2 / median_sq_norm);
                    raw += cfg_.w_embed * logit(pe);
                }

                // P_dmg: aDNA cosine similarity to damage centroid
                if (has_dmg && r_dmg > 1e-6f) {
                    float cos_sim = dot_norm(adna_i, dmg_centroids[k]);
                    float pd = (cos_sim + 1.0f) / 2.0f;
                    raw += cfg_.w_dmg * r_dmg * logit(pd);
                }

                // P_scg: marker consistency (complementary = good, duplicate = bad)
                if (r_scg > 0.5f) {
                    const auto& markers_i = scg_lookup.contig_marker_ids[i];
                    int n_dup = 0, n_comp = 0;
                    for (int m : markers_i) {
                        int cnt = bin_marker_counts[k][m];
                        if (cnt > 1)      n_dup++;   // contig i is counted → already once
                        else if (cnt == 0) n_comp++;  // missing in bin → complement
                    }
                    int n_total = (int)markers_i.size();
                    float ps = (float)n_comp / std::max(1, n_total)
                             - (float)n_dup  / std::max(1, n_total);
                    raw += cfg_.w_scg * r_scg * logit(std::clamp(ps + 0.5f, 0.01f, 0.99f));
                }

                raw /= T;
            }

            // Softmax over candidates
            float max_raw = -1e30f;
            for (const auto& [k, r] : raw_scores) max_raw = std::max(max_raw, r);

            float sum_exp = 0.0f;
            std::vector<std::pair<float, int>> probs;  // (prob, sequential k)
            probs.reserve(raw_scores.size());
            for (const auto& [k, r] : raw_scores) {
                float p = std::exp(r - max_raw);
                sum_exp += p;
                probs.emplace_back(p, k);
            }
            for (auto& [p, k] : probs) p /= sum_exp;

            float p_assigned = 0.0f, p_best_other = 0.0f;
            for (const auto& [p, k] : probs) {
                if (k == assigned_k) p_assigned = p;
                else if (p > p_best_other) p_best_other = p;
            }

            float ent = 0.0f;
            for (const auto& [p, k] : probs)
                if (p > 1e-12f) ent -= p * std::log(p);

            auto& conf = result[i];
            conf.confidence = p_assigned;
            conf.margin     = p_assigned - p_best_other;
            conf.entropy    = ent;
            conf.ambiguous  = conf.margin < cfg_.ambiguous_threshold;

            // graph_boundary: more than threshold fraction of weight goes to other bins
            float in_w  = intra_w[i].count(labels[i]) ? intra_w[i].at(labels[i]) : 0.0f;
            float out_w = total_w[i] - in_w;
            conf.graph_boundary = (in_w + out_w > 1e-9f) &&
                                  (out_w / (in_w + out_w) > cfg_.graph_boundary_ratio);

            // damage_conflict: assigned-bin centroid is dissimilar in aDNA space
            if (has_dmg && r_dmg > 1e-6f) {
                float cos_sim = dot_norm(adna_i, dmg_centroids[assigned_k]);
                conf.damage_conflict = (cos_sim + 1.0f) / 2.0f < cfg_.damage_conflict_threshold;
            }

            // scg_conflict: any of contig i's markers are duplicated in its assigned bin
            if (r_scg > 0.5f) {
                for (int m : scg_lookup.contig_marker_ids[i]) {
                    if (bin_marker_counts[assigned_k][m] > 1) {
                        conf.scg_conflict = true;
                        break;
                    }
                }
            }
        }

        return result;
    }

    // Write TSV with one row per contig (including unbinned).
    // bin_names[i] is the display name for contig i's bin (e.g. "bin_3" or "unbinned").
    void write_tsv(
        const std::string& path,
        const std::vector<std::pair<std::string, std::string>>& contigs,
        const std::vector<int>& labels,
        const std::vector<ContigConfidence>& scores,
        const std::unordered_map<int, std::string>& label_to_name) const
    {
        std::ofstream out(path);
        out << "contig\tbin\tconfidence\tmargin\tentropy\tambiguous\tgraph_boundary\tdamage_conflict\tscg_conflict\n";
        out << std::fixed << std::setprecision(4);
        for (size_t i = 0; i < contigs.size(); i++) {
            int lbl = (i < labels.size()) ? labels[i] : -1;
            auto it = label_to_name.find(lbl);
            std::string bin_name = (it != label_to_name.end()) ? it->second : "unbinned";
            const auto& c = scores[i];
            out << contigs[i].first << "\t" << bin_name << "\t"
                << c.confidence << "\t" << c.margin << "\t" << c.entropy << "\t"
                << (c.ambiguous ? 1 : 0) << "\t"
                << (c.graph_boundary ? 1 : 0) << "\t"
                << (c.damage_conflict ? 1 : 0) << "\t"
                << (c.scg_conflict ? 1 : 0) << "\n";
        }
    }

private:
    Config cfg_;
};

}  // namespace amber::bin2

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace amber {

// Minimizer extraction (simplified 31-mer with w=10)
inline std::vector<uint64_t> extract_minimizers(const std::string& seq, int k = 31, int w = 10) {
    std::vector<uint64_t> minimizers;
    if (seq.length() < (size_t)(k + w - 1)) return minimizers;

    auto hash_kmer = [](const std::string& s, size_t pos, int k) -> uint64_t {
        uint64_t h = 0;
        for (int i = 0; i < k; i++) {
            char c = s[pos + i];
            uint64_t base = (c == 'A' || c == 'a') ? 0 :
                           (c == 'C' || c == 'c') ? 1 :
                           (c == 'G' || c == 'g') ? 2 :
                           (c == 'T' || c == 't') ? 3 : 0;
            h = (h << 2) | base;
        }
        return h;
    };

    uint64_t last_min = UINT64_MAX;
    for (size_t i = 0; i <= seq.length() - k - w + 1; i++) {
        uint64_t window_min = UINT64_MAX;
        for (int j = 0; j < w; j++) {
            uint64_t h = hash_kmer(seq, i + j, k);
            window_min = std::min(window_min, h);
        }
        if (window_min != last_min) {
            minimizers.push_back(window_min);
            last_min = window_min;
        }
    }
    return minimizers;
}

struct PPRCleanupResult {
    std::vector<size_t> contigs_to_remove;
    std::vector<double> ppr_scores;
    double conductance;
    int cut_size;
    bool accepted;
    std::string rejection_reason;
};

// Build IDF-weighted minimizer graph for a bin
// Returns adjacency list with weights
inline std::vector<std::vector<std::pair<size_t, double>>> build_idf_minimizer_graph(
    const std::vector<std::string>& sequences,
    int max_df = 5,  // Maximum document frequency (contigs containing minimizer)
    int k = 31,
    int w = 10
) {
    size_t n = sequences.size();

    // Extract minimizers for each contig
    std::vector<std::unordered_set<uint64_t>> contig_minimizers(n);
    std::unordered_map<uint64_t, int> minimizer_df;  // Document frequency

    for (size_t i = 0; i < n; i++) {
        auto mins = extract_minimizers(sequences[i], k, w);
        for (uint64_t m : mins) {
            contig_minimizers[i].insert(m);
        }
        for (uint64_t m : contig_minimizers[i]) {
            minimizer_df[m]++;
        }
    }

    // Compute IDF weights, filtering high-frequency minimizers
    std::unordered_map<uint64_t, double> minimizer_idf;
    for (auto& [m, df] : minimizer_df) {
        if (df <= max_df) {
            minimizer_idf[m] = std::log((double)(n + 1) / (df + 1));
        }
    }

    // Build weighted adjacency list
    std::vector<std::vector<std::pair<size_t, double>>> adj(n);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            double weight = 0.0;
            for (uint64_t m : contig_minimizers[i]) {
                if (contig_minimizers[j].count(m) && minimizer_idf.count(m)) {
                    weight += minimizer_idf[m];
                }
            }
            if (weight > 0) {
                adj[i].push_back({j, weight});
                adj[j].push_back({i, weight});
            }
        }
    }

    return adj;
}

// Compute weighted degree for each node
inline std::vector<double> compute_weighted_degrees(
    const std::vector<std::vector<std::pair<size_t, double>>>& adj
) {
    std::vector<double> degrees(adj.size(), 0.0);
    for (size_t i = 0; i < adj.size(); i++) {
        for (auto& [j, w] : adj[i]) {
            degrees[i] += w;
        }
    }
    return degrees;
}

// Select core seeds: top k by weighted degree, plus longest contigs
inline std::vector<size_t> select_core_seeds(
    const std::vector<double>& weighted_degrees,
    const std::vector<size_t>& lengths,
    double core_fraction = 0.3
) {
    size_t n = weighted_degrees.size();
    size_t core_size = std::max((size_t)3, (size_t)(n * core_fraction));

    // Score by weighted degree + length bonus
    std::vector<std::pair<double, size_t>> scores(n);
    double max_deg = *std::max_element(weighted_degrees.begin(), weighted_degrees.end());
    size_t max_len = *std::max_element(lengths.begin(), lengths.end());

    for (size_t i = 0; i < n; i++) {
        double deg_score = max_deg > 0 ? weighted_degrees[i] / max_deg : 0;
        double len_score = max_len > 0 ? (double)lengths[i] / max_len : 0;
        scores[i] = {deg_score + 0.5 * len_score, i};
    }

    std::sort(scores.begin(), scores.end(), std::greater<>());

    std::vector<size_t> core;
    for (size_t i = 0; i < core_size && i < n; i++) {
        core.push_back(scores[i].second);
    }
    return core;
}

// Personalized PageRank from seed set
inline std::vector<double> personalized_pagerank(
    const std::vector<std::vector<std::pair<size_t, double>>>& adj,
    const std::vector<size_t>& seeds,
    double alpha = 0.30,  // Restart probability
    int max_iter = 100,
    double tol = 1e-6
) {
    size_t n = adj.size();
    std::vector<double> ppr(n, 0.0);
    std::vector<double> restart(n, 0.0);

    // Uniform restart distribution over seeds
    for (size_t s : seeds) {
        restart[s] = 1.0 / seeds.size();
    }

    // Initialize PPR to restart distribution
    ppr = restart;

    // Compute out-weights for normalization
    std::vector<double> out_weight(n, 0.0);
    for (size_t i = 0; i < n; i++) {
        for (auto& [j, w] : adj[i]) {
            out_weight[i] += w;
        }
    }

    // Power iteration
    for (int iter = 0; iter < max_iter; iter++) {
        std::vector<double> new_ppr(n, 0.0);

        // Teleport
        for (size_t i = 0; i < n; i++) {
            new_ppr[i] = alpha * restart[i];
        }

        // Random walk
        for (size_t i = 0; i < n; i++) {
            if (out_weight[i] > 0) {
                for (auto& [j, w] : adj[i]) {
                    new_ppr[j] += (1 - alpha) * ppr[i] * (w / out_weight[i]);
                }
            } else {
                // Dangling node: distribute to seeds
                for (size_t s : seeds) {
                    new_ppr[s] += (1 - alpha) * ppr[i] / seeds.size();
                }
            }
        }

        // Check convergence
        double diff = 0.0;
        for (size_t i = 0; i < n; i++) {
            diff += std::abs(new_ppr[i] - ppr[i]);
        }
        ppr = new_ppr;

        if (diff < tol) break;
    }

    return ppr;
}

// Compute conductance of a cut (S vs V\S)
inline double compute_conductance(
    const std::vector<std::vector<std::pair<size_t, double>>>& adj,
    const std::vector<bool>& in_set
) {
    double cut_weight = 0.0;
    double vol_S = 0.0;
    double vol_not_S = 0.0;

    for (size_t i = 0; i < adj.size(); i++) {
        double node_vol = 0.0;
        for (auto& [j, w] : adj[i]) {
            node_vol += w;
            if (in_set[i] && !in_set[j]) {
                cut_weight += w;
            }
        }
        if (in_set[i]) {
            vol_S += node_vol;
        } else {
            vol_not_S += node_vol;
        }
    }

    double min_vol = std::min(vol_S, vol_not_S);
    if (min_vol <= 0) return 1.0;
    return cut_weight / min_vol;
}

// Sweep cut: find minimum conductance cut by PPR ordering
inline std::pair<std::vector<size_t>, double> sweep_cut(
    const std::vector<std::vector<std::pair<size_t, double>>>& adj,
    const std::vector<double>& ppr_scores,
    size_t min_cut_size = 10,
    size_t max_cut_size = 60
) {
    size_t n = adj.size();

    // Sort nodes by PPR score (ascending - low PPR = candidate for removal)
    std::vector<std::pair<double, size_t>> sorted_nodes(n);
    for (size_t i = 0; i < n; i++) {
        sorted_nodes[i] = {ppr_scores[i], i};
    }
    std::sort(sorted_nodes.begin(), sorted_nodes.end());

    // Sweep to find minimum conductance
    std::vector<bool> in_cut(n, false);
    double best_conductance = 1.0;
    size_t best_cut_size = 0;

    for (size_t k = 1; k <= std::min(max_cut_size, n - min_cut_size); k++) {
        in_cut[sorted_nodes[k-1].second] = true;

        if (k >= min_cut_size) {
            double cond = compute_conductance(adj, in_cut);
            if (cond < best_conductance) {
                best_conductance = cond;
                best_cut_size = k;
            }
        }
    }

    // Return the nodes in the best cut
    std::vector<size_t> cut_nodes;
    for (size_t k = 0; k < best_cut_size; k++) {
        cut_nodes.push_back(sorted_nodes[k].second);
    }

    return {cut_nodes, best_conductance};
}

// Main cleanup function
inline PPRCleanupResult ppr_minimizer_cleanup(
    const std::vector<std::string>& sequences,
    const std::vector<size_t>& lengths,
    int max_df = 5,
    double alpha = 0.30,
    double max_conductance = 0.20,
    size_t min_cut_size = 10,
    size_t max_cut_size = 60,
    double max_removal_frac = 0.30
) {
    PPRCleanupResult result;
    result.accepted = false;

    size_t n = sequences.size();
    if (n < 20) {
        result.rejection_reason = "bin too small";
        return result;
    }

    // Build IDF-weighted minimizer graph
    auto adj = build_idf_minimizer_graph(sequences, max_df);

    // Check graph connectivity
    size_t edges = 0;
    for (auto& neighbors : adj) edges += neighbors.size();
    edges /= 2;

    if (edges < n / 2) {
        result.rejection_reason = "graph too sparse";
        return result;
    }

    // Compute weighted degrees and select core seeds
    auto degrees = compute_weighted_degrees(adj);
    auto core = select_core_seeds(degrees, lengths);

    // Run PPR from core
    auto ppr = personalized_pagerank(adj, core, alpha);
    result.ppr_scores = ppr;

    // Sweep cut
    auto [cut_nodes, conductance] = sweep_cut(adj, ppr, min_cut_size, max_cut_size);
    result.conductance = conductance;
    result.cut_size = cut_nodes.size();

    // Check acceptance criteria
    if (conductance > max_conductance) {
        result.rejection_reason = "conductance too high: " + std::to_string(conductance);
        return result;
    }

    if (cut_nodes.size() > n * max_removal_frac) {
        result.rejection_reason = "cut too large: " + std::to_string(cut_nodes.size()) + "/" + std::to_string(n);
        return result;
    }

    if (cut_nodes.size() < min_cut_size) {
        result.rejection_reason = "cut too small";
        return result;
    }

    result.contigs_to_remove = cut_nodes;
    result.accepted = true;
    return result;
}

}  // namespace amber

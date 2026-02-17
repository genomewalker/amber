// K Estimation Functions for MFDFA Binner
// Estimates optimal number of bins from coverage and composition features
#pragma once

#include "types.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace amber {

// Count distinct coverage modes using kernel density estimation
// Returns estimated number of coverage groups in the data
inline int count_coverage_modes(const std::vector<double>& log_coverages) {
    if (log_coverages.size() < 10) return 1;

    double min_c = *std::min_element(log_coverages.begin(), log_coverages.end());
    double max_c = *std::max_element(log_coverages.begin(), log_coverages.end());
    double range = max_c - min_c;
    if (range < 0.1) return 1;

    double mean_c = std::accumulate(log_coverages.begin(), log_coverages.end(), 0.0) / log_coverages.size();
    double var = 0;
    for (double c : log_coverages) var += (c - mean_c) * (c - mean_c);
    double std_dev = std::sqrt(var / (log_coverages.size() - 1));
    double bandwidth = 1.06 * std_dev * std::pow(log_coverages.size(), -0.2);
    bandwidth = std::max(bandwidth, 0.1);

    int n_eval = 100;
    std::vector<double> density(n_eval);
    double step = range / (n_eval - 1);

    for (int i = 0; i < n_eval; i++) {
        double x = min_c + i * step;
        double sum = 0;
        for (double c : log_coverages) {
            double z = (x - c) / bandwidth;
            sum += std::exp(-0.5 * z * z);
        }
        density[i] = sum / (log_coverages.size() * bandwidth * std::sqrt(2 * M_PI));
    }

    int n_modes = 0;
    double mean_density = std::accumulate(density.begin(), density.end(), 0.0) / density.size();

    for (int i = 2; i < n_eval - 2; i++) {
        if (density[i] > density[i-1] && density[i] > density[i+1] &&
            density[i] > density[i-2] && density[i] > density[i+2]) {
            if (density[i] > mean_density * 0.5) {
                n_modes++;
            }
        }
    }

    return std::max(1, n_modes);
}

// Count distinct composition clusters using union-find at multiple thresholds
// Estimates number of distinct genomes based on TNF and CGR similarity
inline int count_composition_clusters(const std::vector<Contig*>& contigs, int max_clusters = 100) {
    if (contigs.size() < 10) return 1;

    int sample_size = std::min((int)contigs.size(), 1000);
    std::vector<int> sample_idx;

    std::vector<std::pair<double, int>> cov_idx;
    for (int i = 0; i < (int)contigs.size(); i++) {
        cov_idx.push_back({contigs[i]->mean_coverage, i});
    }
    std::sort(cov_idx.begin(), cov_idx.end());
    int stride = std::max(1, (int)contigs.size() / sample_size);
    for (int i = 0; i < (int)contigs.size() && (int)sample_idx.size() < sample_size; i += stride) {
        sample_idx.push_back(cov_idx[i].second);
    }

    int n = sample_idx.size();

    auto compute_clusters_at_threshold = [&](double tnf_thresh, double cgr_thresh) -> int {
        std::vector<int> parent(n), rank(n, 0);
        std::iota(parent.begin(), parent.end(), 0);

        std::function<int(int)> find = [&](int x) {
            if (parent[x] != x) parent[x] = find(parent[x]);
            return parent[x];
        };

        auto unite = [&](int x, int y) {
            x = find(x); y = find(y);
            if (x == y) return;
            if (rank[x] < rank[y]) std::swap(x, y);
            parent[y] = x;
            if (rank[x] == rank[y]) rank[x]++;
        };

        for (int i = 0; i < n; i++) {
            const auto& tnf_i = contigs[sample_idx[i]]->tnf;
            const auto& cgr_i = contigs[sample_idx[i]]->cgr;

            for (int j = i + 1; j < n; j++) {
                const auto& tnf_j = contigs[sample_idx[j]]->tnf;
                const auto& cgr_j = contigs[sample_idx[j]]->cgr;

                double d_tnf = 0;
                if (!tnf_i.empty() && !tnf_j.empty()) {
                    for (size_t k = 0; k < tnf_i.size(); k++) {
                        d_tnf += (tnf_i[k] - tnf_j[k]) * (tnf_i[k] - tnf_j[k]);
                    }
                    d_tnf = std::sqrt(d_tnf);
                }

                double d_cgr = 0;
                if (!cgr_i.empty() && !cgr_j.empty()) {
                    for (size_t k = 0; k < cgr_i.size(); k++) {
                        d_cgr += (cgr_i[k] - cgr_j[k]) * (cgr_i[k] - cgr_j[k]);
                    }
                    d_cgr = std::sqrt(d_cgr);
                }

                if (d_tnf < tnf_thresh && d_cgr < cgr_thresh) {
                    unite(i, j);
                }
            }
        }

        std::set<int> roots;
        for (int i = 0; i < n; i++) roots.insert(find(i));
        return roots.size();
    };

    std::vector<std::pair<double, double>> thresholds = {
        {0.15, 0.20}, {0.20, 0.25}, {0.25, 0.30}, {0.30, 0.35}, {0.35, 0.40}
    };

    std::vector<int> cluster_counts;
    for (auto [tnf_t, cgr_t] : thresholds) {
        cluster_counts.push_back(compute_clusters_at_threshold(tnf_t, cgr_t));
    }

    int result = cluster_counts[2];

    if (cluster_counts[0] > result * 2) {
        result = (cluster_counts[0] + cluster_counts[1]) / 2;
    }

    return std::min(std::max(result, 3), max_clusters);
}

// Estimate optimal K combining coverage modes and composition clusters
// Returns KEstimate with coverage_modes, composition_clusters, and combined recommendation
inline KEstimate estimate_k_data_driven(const std::vector<Contig*>& contigs, int min_length_for_mode = 2500) {
    KEstimate result = {1, 1, 5, "", ""};

    if (contigs.empty()) return result;

    // Use only long contigs (>= 2500bp) for coverage mode detection
    // Short contigs have noisier coverage estimates that can obscure legitimate modes
    std::vector<double> log_coverages;
    int long_contig_count = 0;
    for (auto* c : contigs) {
        if (c->mean_coverage > 0 && c->length >= static_cast<size_t>(min_length_for_mode)) {
            log_coverages.push_back(std::log(c->mean_coverage + 1));
            long_contig_count++;
        }
    }

    // Fallback: if too few long contigs (< 100), use all contigs
    if (long_contig_count < 100) {
        log_coverages.clear();
        for (auto* c : contigs) {
            if (c->mean_coverage > 0) {
                log_coverages.push_back(std::log(c->mean_coverage + 1));
            }
        }
    }

    if (!log_coverages.empty()) {
        result.coverage_modes = count_coverage_modes(log_coverages);
    }

    bool has_features = false;
    for (auto* c : contigs) {
        if (!c->cgr.empty() || !c->tnf.empty()) { has_features = true; break; }
    }
    if (has_features) {
        result.composition_clusters = count_composition_clusters(contigs);
    }

    if (result.coverage_modes > 3) {
        result.combined = std::max(result.composition_clusters,
                                    result.coverage_modes + result.composition_clusters / 2);
    } else {
        result.combined = result.composition_clusters;
    }

    result.combined = std::max(5, std::min(100, result.combined));

    result.details = "cov_modes=" + std::to_string(result.coverage_modes) +
                     " comp_clusters=" + std::to_string(result.composition_clusters);

    return result;
}

}  // namespace amber

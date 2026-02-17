#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace amber {

// ========== Probabilistic Models ==========

class MultivariateNormal {
public:
    std::vector<double> mean;
    std::vector<std::vector<double>> inv_cov;
    double log_det;
    int dim;
    bool valid = false;

    void fit(const std::vector<std::vector<double>>& data) {
        if (data.empty() || data[0].empty()) return;
        dim = data[0].size();
        size_t n = data.size();

        mean.assign(dim, 0.0);
        for (const auto& x : data) {
            for (int i = 0; i < dim; i++) mean[i] += x[i];
        }
        for (int i = 0; i < dim; i++) mean[i] /= n;

        std::vector<std::vector<double>> cov(dim, std::vector<double>(dim, 0.0));
        for (const auto& x : data) {
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    cov[i][j] += (x[i] - mean[i]) * (x[j] - mean[j]);
                }
            }
        }
        double reg = 1e-4;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                cov[i][j] = cov[i][j] / (n - 1) + (i == j ? reg : 0);
            }
        }

        inv_cov.assign(dim, std::vector<double>(dim, 0.0));
        log_det = 0;
        for (int i = 0; i < dim; i++) {
            if (cov[i][i] > 0) {
                inv_cov[i][i] = 1.0 / cov[i][i];
                log_det += std::log(cov[i][i]);
            }
        }
        valid = true;
    }

    double log_probability(const std::vector<double>& x) const {
        if (!valid || (int)x.size() != dim) return -1e30;

        double mahal = 0;
        for (int i = 0; i < dim; i++) {
            double diff = x[i] - mean[i];
            mahal += diff * diff * inv_cov[i][i];
        }

        return -0.5 * (dim * std::log(2 * M_PI) + log_det + mahal);
    }
};

class LogNormal {
public:
    double mu, sigma;
    bool valid = false;

    void fit(const std::vector<double>& data) {
        std::vector<double> log_data;
        for (double x : data) {
            if (x > 0) log_data.push_back(std::log(x));
        }
        if (log_data.size() < 2) return;

        mu = std::accumulate(log_data.begin(), log_data.end(), 0.0) / log_data.size();
        double var = 0;
        for (double lx : log_data) var += (lx - mu) * (lx - mu);
        sigma = std::sqrt(var / (log_data.size() - 1) + 1e-6);
        valid = true;
    }

    double log_probability(double x) const {
        if (!valid || x <= 0) return -1e30;
        double lx = std::log(x);
        double z = (lx - mu) / sigma;
        return -0.5 * z * z - std::log(sigma) - lx - 0.5 * std::log(2 * M_PI);
    }
};

// ========== Coverage Model (MetaBAT2-style with variance) ==========
// Models coverage as Gaussian in log-space with contig-specific variance
// This matches MetaBAT2's approach where variance indicates coverage reliability
class CoverageModel {
public:
    double bin_mean_log_cov = 0.0;      // Mean of log(coverage) across bin members
    double bin_mean_var = 0.0;          // Mean within-contig variance (pooled)
    double bin_cov_variance = 0.0;      // Variance of log(coverage) across bin members
    bool valid = false;

    void fit(const std::vector<std::pair<double, double>>& cov_var_data) {
        // Input: vector of (mean_coverage, var_coverage) pairs
        if (cov_var_data.size() < 2) return;

        std::vector<double> log_covs;
        double sum_var = 0.0;
        int var_count = 0;

        for (const auto& [mean_cov, var_cov] : cov_var_data) {
            if (mean_cov > 0) {
                log_covs.push_back(std::log(mean_cov));
                if (var_cov > 0) {
                    sum_var += var_cov;
                    var_count++;
                }
            }
        }

        if (log_covs.empty()) return;

        // Compute bin mean log coverage
        bin_mean_log_cov = std::accumulate(log_covs.begin(), log_covs.end(), 0.0) / log_covs.size();

        // Compute between-contig variance in log space
        double var_sum = 0.0;
        for (double lc : log_covs) {
            var_sum += (lc - bin_mean_log_cov) * (lc - bin_mean_log_cov);
        }
        bin_cov_variance = var_sum / (log_covs.size() - 1);

        // Pool within-contig variances
        bin_mean_var = var_count > 0 ? sum_var / var_count : 0.0;

        valid = true;
    }

    // MetaBAT2-style probability: uses both between-contig variance AND within-contig variance
    // Key insight: contigs with HIGH variance should have LESS influence on distance
    // This is implemented by widening the effective sigma for uncertain contigs
    double log_probability(double mean_cov, double var_cov) const {
        if (!valid || mean_cov <= 0) return -1e30;

        double log_cov = std::log(mean_cov);

        // Effective variance = between-contig variance + scaled within-contig variance
        // The within-contig variance indicates measurement uncertainty
        // We convert it to log-space roughly: var(log(X)) ≈ var(X) / mean(X)^2
        double within_log_var = 0.0;
        if (var_cov > 0 && mean_cov > 0) {
            within_log_var = var_cov / (mean_cov * mean_cov);
        }

        // Total effective variance (minimum floor to avoid numerical issues)
        double effective_var = std::max(bin_cov_variance + within_log_var + bin_mean_var / (mean_cov * mean_cov + 1e-10), 0.01);
        double sigma = std::sqrt(effective_var);

        double z = (log_cov - bin_mean_log_cov) / sigma;
        return -0.5 * z * z - std::log(sigma) - 0.5 * std::log(2 * M_PI);
    }
};

} // namespace amber

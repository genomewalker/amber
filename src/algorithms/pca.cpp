#include "pca.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <omp.h>

namespace amber {

std::vector<double> SimplePCA::power_iteration(
    const std::vector<std::vector<double>> &matrix,
    int max_iter, double tol) {

  int n = matrix.size();
  if (n == 0 || matrix[0].size() != static_cast<size_t>(n)) {
    return {}; // Invalid matrix
  }

  // Initialize random vector
  std::vector<double> v(n, 1.0);
  double norm = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
  for (auto &x : v) x /= norm;

  for (int iter = 0; iter < max_iter; ++iter) {
    // v_new = A * v (parallelized matrix-vector product)
    std::vector<double> v_new(n, 0.0);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += matrix[i][j] * v[j];
      }
      v_new[i] = sum;
    }

    // Normalize
    norm = std::sqrt(std::inner_product(v_new.begin(), v_new.end(), v_new.begin(), 0.0));
    if (norm < 1e-10) break;
    for (auto &x : v_new) x /= norm;

    // Check convergence
    double diff = 0.0;
    for (int i = 0; i < n; ++i) {
      diff += (v_new[i] - v[i]) * (v_new[i] - v[i]);
    }

    v = v_new;
    if (std::sqrt(diff) < tol) break;
  }

  return v;
}

int SimplePCA::select_components_variance(
    const std::vector<double> &eigenvalues,
    double variance_threshold,
    bool verbose) {

  int p = eigenvalues.size();
  if (p == 0) return 0;

  // Compute total variance
  double total_variance = 0.0;
  for (const auto &ev : eigenvalues) {
    total_variance += ev;
  }

  if (total_variance < 1e-10) {
    if (verbose) {
      std::cout << "\n[Variance Selection] WARNING: Total variance near zero, using 5 components\n";
    }
    return std::min(5, p);
  }

  // Compute cumulative variance explained
  double cumulative_variance = 0.0;
  int n_components = 0;

  if (verbose) {
    std::cout << "\n[Cumulative Variance] Selecting PCA components (threshold: "
              << (variance_threshold * 100) << "%):\n";
    std::cout << "  Total variance: " << total_variance << "\n";
    std::cout << "  PC    Variance%   Cumulative%   Keep?\n";
    std::cout << "  --    ---------   -----------   -----\n";
  }

  for (int j = 0; j < p; ++j) {  // Process all eigenvalues (no artificial cap)
    double var_explained = eigenvalues[j] / total_variance;

    // Keep this component if adding it doesn't exceed threshold
    // OR if it's the first component (always keep PC1)
    // OR if keeping it brings us to the threshold (inclusive)
    bool keep = (j == 0) || (cumulative_variance < variance_threshold);

    cumulative_variance += var_explained;

    if (verbose && j < 20) {  // Only print first 20 for readability
      printf("  %2d    %8.4f%%   %9.4f%%    %s\n",
             j + 1,
             var_explained * 100,
             cumulative_variance * 100,
             keep ? "YES" : "NO");
    }

    if (keep) {
      n_components = j + 1;
    } else {
      break;  // Stop once we've exceeded threshold
    }
  }

  if (verbose) {
    std::cout << "\n  Selected " << n_components << " components"
              << " (explaining " << (cumulative_variance * 100.0) << "% variance)\n";
  }

  // Ensure minimum of 3 components for robustness
  if (n_components < 3 && p >= 3) {
    if (verbose) {
      std::cout << "  Note: Using minimum of 3 components for robustness\n";
    }
    n_components = 3;
  }

  // No maximum cap - use as many components as needed for variance threshold

  return n_components;
}

std::vector<std::vector<double>> SimplePCA::center_data(
    const std::vector<std::vector<double>> &data,
    std::vector<double> &means) {

  if (data.empty()) return {};

  int n_samples = data.size();
  int n_features = data[0].size();

  // Compute means
  means.assign(n_features, 0.0);
  for (const auto &row : data) {
    for (int j = 0; j < n_features; ++j) {
      means[j] += row[j];
    }
  }
  for (auto &m : means) m /= n_samples;

  // Center data (parallelize over samples)
  std::vector<std::vector<double>> centered(n_samples, std::vector<double>(n_features));
  #pragma omp parallel for
  for (int i = 0; i < n_samples; ++i) {
    for (int j = 0; j < n_features; ++j) {
      centered[i][j] = data[i][j] - means[j];
    }
  }

  return centered;
}

std::vector<std::vector<double>> SimplePCA::compute_covariance(
    const std::vector<std::vector<double>> &centered_data) {

  if (centered_data.empty()) return {};

  int n_samples = centered_data.size();
  int n_features = centered_data[0].size();

  // Cov = X^T * X / (n-1)
  std::vector<std::vector<double>> cov(n_features, std::vector<double>(n_features, 0.0));

  // Parallelize over rows of covariance matrix (only compute upper triangle)
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n_features; ++i) {
    for (int j = i; j < n_features; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n_samples; ++k) {
        sum += centered_data[k][i] * centered_data[k][j];
      }
      cov[i][j] = sum / (n_samples - 1);
    }
  }

  // Fill symmetric part AFTER parallel loop (avoid race conditions)
  for (int i = 0; i < n_features; ++i) {
    for (int j = i + 1; j < n_features; ++j) {
      cov[j][i] = cov[i][j];
    }
  }

  return cov;
}

std::vector<std::vector<double>> SimplePCA::fit(
    const std::vector<std::vector<double>> &data,
    int n_components,
    std::vector<double> *out_means) {

  if (data.empty()) return {};

  std::vector<double> means;
  auto centered = center_data(data, means);

  // Store means if requested
  if (out_means) {
    *out_means = means;
  }

  auto cov = compute_covariance(centered);

  int n_features = data[0].size();
  n_components = std::min(n_components, n_features);

  std::vector<std::vector<double>> components;
  components.reserve(n_components);

  // Extract top n_components eigenvectors using power iteration
  // (deflation method: subtract contribution of each PC)
  auto cov_deflated = cov;

  for (int comp = 0; comp < n_components; ++comp) {
    auto eigenvec = power_iteration(cov_deflated);
    if (eigenvec.empty()) break;

    components.push_back(eigenvec);

    // Deflate: cov_new = cov - λ * v * v^T
    // Approximate eigenvalue: λ = v^T * cov * v
    std::vector<double> cov_v(n_features, 0.0);
    #pragma omp parallel for
    for (int i = 0; i < n_features; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n_features; ++j) {
        sum += cov_deflated[i][j] * eigenvec[j];
      }
      cov_v[i] = sum;
    }

    double eigenvalue = std::inner_product(eigenvec.begin(), eigenvec.end(),
                                          cov_v.begin(), 0.0);

    // Deflate (parallelize outer loop)
    #pragma omp parallel for
    for (int i = 0; i < n_features; ++i) {
      for (int j = 0; j < n_features; ++j) {
        cov_deflated[i][j] -= eigenvalue * eigenvec[i] * eigenvec[j];
      }
    }
  }

  return components;
}

std::vector<std::vector<double>> SimplePCA::transform(
    const std::vector<std::vector<double>> &data,
    const std::vector<std::vector<double>> &components) {

  if (data.empty() || components.empty()) return {};

  int n_samples = data.size();
  int n_components = components.size();

  // Compute means for centering
  std::vector<double> means;
  auto centered = center_data(data, means);

  // Project: X_pca = X_centered * components^T
  std::vector<std::vector<double>> transformed(n_samples,
                                               std::vector<double>(n_components, 0.0));

  #pragma omp parallel for
  for (int i = 0; i < n_samples; ++i) {
    for (int j = 0; j < n_components; ++j) {
      double dot = 0.0;
      for (size_t k = 0; k < centered[i].size(); ++k) {
        dot += centered[i][k] * components[j][k];
      }
      transformed[i][j] = dot;
    }
  }

  return transformed;
}

std::vector<std::vector<double>> SimplePCA::fit_transform(
    const std::vector<std::vector<double>> &data,
    int n_components) {

  if (data.empty()) return {};

  // Center the data ONCE
  std::vector<double> means;
  auto centered = center_data(data, means);

  // Compute covariance
  auto cov = compute_covariance(centered);

  int n_features = data[0].size();

  // First pass: extract ALL eigenvectors and eigenvalues to determine optimal n_components
  std::cout << "\n[PCA] Extracting eigenvalues for variance analysis...\n";

  std::vector<std::vector<double>> all_components;
  std::vector<double> all_eigenvalues;
  all_components.reserve(n_features);
  all_eigenvalues.reserve(n_features);

  auto cov_deflated = cov;
  int max_components = std::min(n_components * 3, n_features);  // Extract up to 3x requested, or all

  for (int comp = 0; comp < max_components; ++comp) {
    auto eigenvec = power_iteration(cov_deflated);
    if (eigenvec.empty()) break;

    // Compute eigenvalue: λ = v^T * cov * v
    std::vector<double> cov_v(n_features, 0.0);
    #pragma omp parallel for
    for (int i = 0; i < n_features; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n_features; ++j) {
        sum += cov_deflated[i][j] * eigenvec[j];
      }
      cov_v[i] = sum;
    }

    double eigenvalue = std::inner_product(eigenvec.begin(), eigenvec.end(),
                                          cov_v.begin(), 0.0);

    // Stop if eigenvalue becomes negligible
    if (eigenvalue < 1e-10) break;

    all_components.push_back(eigenvec);
    all_eigenvalues.push_back(eigenvalue);

    // Deflate (parallelize outer loop)
    #pragma omp parallel for
    for (int i = 0; i < n_features; ++i) {
      for (int j = 0; j < n_features; ++j) {
        cov_deflated[i][j] -= eigenvalue * eigenvec[i] * eigenvec[j];
      }
    }
  }

  // Use cumulative variance explained to determine optimal number of components
  // Default: keep 90% of variance (reduced from 95% to avoid curse of dimensionality)
  int optimal_components = select_components_variance(all_eigenvalues, 0.90, true);

  // Store original user request for logging
  int user_requested = n_components;

  // Use optimal components (what variance analysis recommends)
  // Cap only at what we actually extracted, not at user request
  n_components = std::min(optimal_components, static_cast<int>(all_components.size()));

  std::cout << "[PCA] Using " << n_components << " components (user requested: "
            << user_requested << ", variance optimal: " << optimal_components << ")\n";

  // Keep only the selected components
  std::vector<std::vector<double>> components(all_components.begin(),
                                               all_components.begin() + n_components);

  // Project the already-centered data onto components
  // This avoids double-centering!
  int n_samples = centered.size();
  std::vector<std::vector<double>> transformed(n_samples,
                                               std::vector<double>(n_components, 0.0));

  #pragma omp parallel for
  for (int i = 0; i < n_samples; ++i) {
    for (int j = 0; j < n_components; ++j) {
      double dot = 0.0;
      for (int k = 0; k < n_features; ++k) {
        dot += centered[i][k] * components[j][k];
      }
      transformed[i][j] = dot;
    }
  }

  // Debug: check transformed values
  if (!transformed.empty() && !transformed[0].empty()) {
    double t_min = transformed[0][0], t_max = transformed[0][0];
    for (int i = 0; i < std::min(1000, static_cast<int>(transformed.size())); ++i) {
      for (int j = 0; j < static_cast<int>(transformed[i].size()); ++j) {
        t_min = std::min(t_min, transformed[i][j]);
        t_max = std::max(t_max, transformed[i][j]);
      }
    }
    std::cout << "  [PCA Debug] Transformed values range: [" << t_min << ", " << t_max << "]\n";
  }

  return transformed;
}

} // namespace amber

#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <algorithm>

namespace amber {

// Lightweight PCA implementation using SVD (via power iteration)
// For online compression of k-mer features
class SimplePCA {
public:
  // Fit PCA on feature matrix (rows = samples, cols = features)
  // Returns principal components (top n_components)
  // Optionally returns the means used for centering
  static std::vector<std::vector<double>> fit(
      const std::vector<std::vector<double>> &data,
      int n_components,
      std::vector<double> *out_means = nullptr);

  // Transform data using fitted PCs
  static std::vector<std::vector<double>> transform(
      const std::vector<std::vector<double>> &data,
      const std::vector<std::vector<double>> &components);

  // Fit and transform in one step (avoids double-centering!)
  static std::vector<std::vector<double>> fit_transform(
      const std::vector<std::vector<double>> &data,
      int n_components);

private:
  // Center data (subtract mean)
  static std::vector<std::vector<double>> center_data(
      const std::vector<std::vector<double>> &data,
      std::vector<double> &means);

  // Compute covariance matrix
  static std::vector<std::vector<double>> compute_covariance(
      const std::vector<std::vector<double>> &centered_data);

  // Power iteration to find top eigenvector
  static std::vector<double> power_iteration(
      const std::vector<std::vector<double>> &matrix,
      int max_iter = 100,
      double tol = 1e-6);

  // Select number of components using cumulative variance explained
  // Keeps components until variance_threshold (e.g., 0.95 = 95%) is explained
  // Better for metagenomic data than broken stick
  static int select_components_variance(
      const std::vector<double> &eigenvalues,
      double variance_threshold = 0.95,
      bool verbose = true);
};

} // namespace amber

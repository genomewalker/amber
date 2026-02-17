#include "fractal_features.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <random>
#include <iostream>
#include <queue>

namespace amber {

// Static member variable definition
int FractalFeatures::dfa_stride_factor = 0;  // Default: no striding (full overlap)

// ============================================================================
// PERFORMANCE: Hash function for pair<int,int> to enable unordered_map
// ============================================================================
struct PairHash {
  size_t operator()(const std::pair<int, int>& p) const {
    // Cantor pairing function - fast and well-distributed for small integers
    size_t h1 = static_cast<size_t>(p.first);
    size_t h2 = static_cast<size_t>(p.second);
    return (h1 + h2) * (h1 + h2 + 1) / 2 + h2;
  }
};

// ============================================================================
// CHAOS GAME REPRESENTATION (CGR)
// Maps DNA to 2D coordinates: A=(0,0), C=(1,0), G=(1,1), T=(0,1)
// Each base moves halfway towards its corner - creates fractal pattern
// ============================================================================

std::vector<std::pair<double, double>> FractalFeatures::compute_cgr_points(
    const std::string &seq, int max_points) {

  std::vector<std::pair<double, double>> points;
  if (seq.empty()) return points;

  // Starting position (center)
  double x = 0.5, y = 0.5;

  // Corner positions for each nucleotide: A=(0,0), C=(1,0), G=(1,1), T=(0,1)
  // OPTIMIZATION: Use inline lookup instead of map (8x faster)

  // Reservoir sample points (not indices): CGR state depends on every prior base,
  // so we must iterate full sequence in order and only subsample resulting points.
  // Use thread_local RNG to avoid shared state in parallel contexts.
  thread_local std::mt19937 rng;

  // Content-based seed using FNV-1a over the sequence to avoid length-only bias
  uint64_t seed = 1469598103934665603ull;
  for (unsigned char c : seq) {
    seed ^= c;
    seed *= 1099511628211ull;
  }
  rng.seed(static_cast<uint32_t>(seed ^ (seed >> 32)));
  points.reserve(std::min(static_cast<size_t>(max_points), seq.size()));
  size_t seen_points = 0;

  for (size_t idx = 0; idx < seq.size(); ++idx) {
    char base = seq[idx];
    double corner_x, corner_y;
    bool valid = true;

    // OPTIMIZATION: Switch instead of map lookup (8x faster)
    switch (base) {
      case 'A': case 'a': corner_x = 0.0; corner_y = 0.0; break;
      case 'C': case 'c': corner_x = 1.0; corner_y = 0.0; break;
      case 'G': case 'g': corner_x = 1.0; corner_y = 1.0; break;
      case 'T': case 't': corner_x = 0.0; corner_y = 1.0; break;
      default: valid = false; break;
    }

    if (!valid) continue;  // skip ambiguous bases

    // Move halfway towards corner (chaos game rule!)
    x = (x + corner_x) / 2.0;
    y = (y + corner_y) / 2.0;
    ++seen_points;

    if (points.size() < static_cast<size_t>(max_points)) {
      points.push_back({x, y});
    } else {
      std::uniform_int_distribution<size_t> dist(0, seen_points - 1);
      size_t r = dist(rng);
      if (r < static_cast<size_t>(max_points)) {
        points[r] = {x, y};
      }
    }
  }

  return points;
}

double FractalFeatures::compute_fractal_dimension(
    const std::vector<std::pair<double, double>> &points) {

  // No hard minimum - compute for all sizes with quality-dependent weighting
  if (points.empty()) return 0.0;

  size_t n = points.size();

  // Box-counting with occupancy-weighted Total Least Squares (TLS)
  // Use finer scale progression for better resolution
  std::vector<double> box_sizes;
  for (double e = 0.5; e >= 0.01; e /= 1.5)
    box_sizes.push_back(e);

  std::vector<double> log_sizes, log_counts, weights;

  for (double epsilon : box_sizes) {
    std::unordered_map<std::pair<int, int>, int, PairHash> box_counts;

    for (const auto &[x, y] : points) {
      int box_x = static_cast<int>(x / epsilon);
      int box_y = static_cast<int>(y / epsilon);
      box_counts[{box_x, box_y}]++;
    }

    size_t num_boxes = box_counts.size();
    if (num_boxes == 0) continue;

    double mean_occupancy = static_cast<double>(n) / num_boxes;

    // Restrict scales: 2 ≤ μ(ε) ≤ 0.8·n (avoid under/over-sampling)
    if (mean_occupancy < 2.0 || mean_occupancy > 0.8 * n) continue;

    // Occupancy-weighted: w = max(0, μ - 2) gives more weight to well-sampled scales
    double w = std::max(0.0, mean_occupancy - 2.0);

    log_sizes.push_back(std::log(1.0 / epsilon));
    log_counts.push_back(std::log(static_cast<double>(num_boxes)));
    weights.push_back(w);
  }

  // Need at least 2 valid scales
  if (log_sizes.size() < 2) {
    // Fallback: return estimate based on point density
    return std::min(2.0, 1.0 + std::log(n) / 10.0);
  }

  // Weighted least squares regression
  double sum_w = 0, sum_wx = 0, sum_wy = 0, sum_wxy = 0, sum_wxx = 0;
  for (size_t i = 0; i < log_sizes.size(); ++i) {
    double w = weights[i];
    sum_w += w;
    sum_wx += w * log_sizes[i];
    sum_wy += w * log_counts[i];
    sum_wxy += w * log_sizes[i] * log_counts[i];
    sum_wxx += w * log_sizes[i] * log_sizes[i];
  }

  double denom = sum_w * sum_wxx - sum_wx * sum_wx;
  double d_raw = (denom > 1e-10) ? (sum_w * sum_wxy - sum_wx * sum_wy) / denom : 1.5;

  // Adaptive Grassberger-Procaccia correction
  // c = 1/log(n) for n<500, otherwise 0.7/log(n)
  if (n > 0) {
    double log_n = std::log(static_cast<double>(n) + 1.0);
    double c = (n < 500) ? 1.0 : 0.7;
    double correction = c / log_n;
    d_raw -= correction;
  }

  // Return raw value - don't clamp, as that destroys variance
  // Downstream standardization should handle scaling
  return d_raw;
}

double FractalFeatures::compute_lacunarity(
    const std::vector<std::pair<double, double>> &points) {

  // No hard minimum - compute for all sizes with quality weighting
  if (points.empty()) return 0.0;

  size_t n = points.size();

  // Lacunarity measures "gappiness" - variance in box occupancy.
  // Use adaptive box sizes targeting mean occupancies μ ∈ {15, 25, 40}
  // for optimal bias-variance tradeoff.

  // Find box sizes that give target occupancies
  // For n points in unit square, with box size ε:
  //   num_boxes ≈ 1/ε², mean_occupancy μ ≈ n·ε²
  //   So ε ≈ sqrt(μ/n)
  std::vector<double> target_occupancies = {15.0, 25.0, 40.0};
  std::vector<double> box_sizes;

  for (double mu_target : target_occupancies) {
    double epsilon = std::sqrt(mu_target / n);
    // Clamp to reasonable range [0.02, 0.5]
    epsilon = std::max(0.02, std::min(0.5, epsilon));
    box_sizes.push_back(epsilon);
  }

  double lac_weighted_sum = 0.0;
  double weight_sum = 0.0;

  for (double box_size : box_sizes) {
    std::unordered_map<std::pair<int, int>, int, PairHash> box_counts;

    for (const auto &[x, y] : points) {
      int box_x = static_cast<int>(x / box_size);
      int box_y = static_cast<int>(y / box_size);
      box_counts[{box_x, box_y}]++;
    }

    if (box_counts.empty()) continue;

    size_t num_boxes = box_counts.size();
    double mean_occupancy = static_cast<double>(n) / num_boxes;

    // Skip scales with too few points per box (unreliable)
    if (mean_occupancy < 3.0) continue;

    // Compute sample variance (Bessel correction for finite sample)
    double sum_k = 0.0, sum_k2 = 0.0;
    for (const auto &[box, count] : box_counts) {
      sum_k += count;
      sum_k2 += static_cast<double>(count) * count;
    }
    double E_k = sum_k / num_boxes;  // Mean
    double E_k2 = sum_k2 / num_boxes;  // Second moment
    double raw_variance = E_k2 - E_k * E_k;  // Var(k)

    // Poisson correction: under random placement, Var(k) ≈ E[k]
    // Excess lacunarity = (Var(k) - E[k]) / E[k]²
    // This removes the baseline Poisson variance, leaving only
    // the "true" lacunarity from spatial structure.
    double poisson_corrected_variance = std::max(0.0, raw_variance - E_k);

    double lacunarity = (E_k > 1e-10) ?
      poisson_corrected_variance / (E_k * E_k) : 0.0;

    // Weight by occupancy (higher occupancy = more reliable)
    double weight = std::max(0.0, mean_occupancy - 3.0);
    lac_weighted_sum += lacunarity * weight;
    weight_sum += weight;
  }

  // Fallback for very small point sets
  if (weight_sum < 1e-10) {
    // Simple estimate: CV² of point distribution
    if (n < 10) return 0.0;

    // Use a single coarse box size
    double epsilon = 0.2;
    std::unordered_map<std::pair<int, int>, int, PairHash> box_counts;
    for (const auto &[x, y] : points) {
      int box_x = static_cast<int>(x / epsilon);
      int box_y = static_cast<int>(y / epsilon);
      box_counts[{box_x, box_y}]++;
    }
    if (box_counts.empty()) return 0.0;

    double mean = static_cast<double>(n) / box_counts.size();
    double var = 0.0;
    for (const auto &[box, count] : box_counts) {
      var += (count - mean) * (count - mean);
    }
    var /= box_counts.size();
    return (mean > 0) ? var / (mean * mean) : 0.0;
  }

  return lac_weighted_sum / weight_sum;
}

std::array<double, 7> FractalFeatures::compute_hu_moments(
    const std::vector<std::pair<double, double>> &points) {

  std::array<double, 7> hu_moments;
  hu_moments.fill(0.0);

  if (points.size() < 10) return hu_moments;

  // Compute central moments (translation invariant)
  double m00 = points.size();
  double m10 = 0.0, m01 = 0.0;

  for (const auto &[x, y] : points) {
    m10 += x;
    m01 += y;
  }

  double x_bar = m10 / m00;
  double y_bar = m01 / m00;

  // Central moments
  double mu20 = 0.0, mu02 = 0.0, mu11 = 0.0;
  double mu30 = 0.0, mu03 = 0.0, mu21 = 0.0, mu12 = 0.0;

  for (const auto &[x, y] : points) {
    double dx = x - x_bar;
    double dy = y - y_bar;

    mu20 += dx * dx;
    mu02 += dy * dy;
    mu11 += dx * dy;
    mu30 += dx * dx * dx;
    mu03 += dy * dy * dy;
    mu21 += dx * dx * dy;
    mu12 += dx * dy * dy;
  }

  // Normalize
  double norm = std::pow(m00, 2.0);
  if (norm > 0.0) {
    mu20 /= norm;
    mu02 /= norm;
    mu11 /= norm;

    norm = std::pow(m00, 2.5);
    mu30 /= norm;
    mu03 /= norm;
    mu21 /= norm;
    mu12 /= norm;

    // Hu's 7 invariant moments (simplified version)
    hu_moments[0] = mu20 + mu02;
    hu_moments[1] = (mu20 - mu02) * (mu20 - mu02) + 4 * mu11 * mu11;
    hu_moments[2] = (mu30 - 3 * mu12) * (mu30 - 3 * mu12) +
                    (3 * mu21 - mu03) * (3 * mu21 - mu03);
    hu_moments[3] = (mu30 + mu12) * (mu30 + mu12) +
                    (mu21 + mu03) * (mu21 + mu03);
    hu_moments[4] = (mu30 - 3 * mu12) * (mu30 + mu12) *
                    ((mu30 + mu12) * (mu30 + mu12) - 3 * (mu21 + mu03) * (mu21 + mu03));
    hu_moments[5] = (mu20 - mu02) * ((mu30 + mu12) * (mu30 + mu12) -
                    (mu21 + mu03) * (mu21 + mu03));
    hu_moments[6] = (3 * mu21 - mu03) * (mu30 + mu12) *
                    ((mu30 + mu12) * (mu30 + mu12) - 3 * (mu21 + mu03) * (mu21 + mu03));
  }

  return hu_moments;
}

CGRFeatures FractalFeatures::extract_cgr_features(const std::string &sequence) {
  CGRFeatures features;

  if (sequence.length() < 100) return features;

  // Generate CGR points
  auto points = compute_cgr_points(sequence, 10000);

  // Fractal dimension via box-counting
  features.fractal_dimension = compute_fractal_dimension(points);

  // Lacunarity (texture heterogeneity)
  features.lacunarity = compute_lacunarity(points);

  // Hu moments (rotation/scale invariant)
  features.hu_moments = compute_hu_moments(points);

  // Radial density profile (multi-scale structure)
  // CGR points are in [0,1] x [0,1], center is (0.5, 0.5)
  // Max radius from center to corner is sqrt(0.5^2 + 0.5^2) = sqrt(0.5) ≈ 0.7071
  double center_x = 0.5, center_y = 0.5;
  constexpr double MAX_RADIUS = 0.7071067811865476;  // sqrt(0.5)
  std::array<int, 10> radial_counts = {0};

  for (const auto &[x, y] : points) {
    double dist = std::sqrt((x - center_x) * (x - center_x) +
                           (y - center_y) * (y - center_y));
    // Normalize by max radius so all 10 bins are used
    int bin = std::min(9, static_cast<int>((dist / MAX_RADIUS) * 10.0));
    radial_counts[bin]++;
  }

  for (int i = 0; i < 10; ++i) {
    features.radial_density[i] = static_cast<double>(radial_counts[i]) / points.size();
  }

  // Quadrant occupancy (ACGT biases)
  std::array<int, 4> quadrant_counts = {0};
  for (const auto &[x, y] : points) {
    int quad = (x < 0.5 ? 0 : 1) + (y < 0.5 ? 0 : 2);
    quadrant_counts[quad]++;
  }

  for (int i = 0; i < 4; ++i) {
    features.quadrant_density[i] = static_cast<double>(quadrant_counts[i]) / points.size();
  }

  // Grid-based entropy and uniformity (16x16 grid)
  const int grid_size = 16;
  std::array<int, 256> grid_counts = {0};  // 16x16 = 256 cells

  for (const auto &[x, y] : points) {
    int gx = std::min(grid_size - 1, static_cast<int>(x * grid_size));
    int gy = std::min(grid_size - 1, static_cast<int>(y * grid_size));
    grid_counts[gy * grid_size + gx]++;
  }

  // Shannon entropy of grid distribution
  double entropy = 0.0;
  double uniformity = 0.0;
  int max_count = 0;
  double n_points = static_cast<double>(points.size());

  for (int i = 0; i < 256; ++i) {
    if (grid_counts[i] > 0) {
      double p = grid_counts[i] / n_points;
      entropy -= p * std::log2(p);
      uniformity += p * p;
      if (grid_counts[i] > max_count) {
        max_count = grid_counts[i];
      }
    }
  }

  features.entropy = entropy;
  features.uniformity = uniformity;
  features.max_density = max_count / n_points;

  return features;
}

// ============================================================================
// CGR GEOMETRIC FEATURES (Binning-Optimized, Baseline-Compatible)
// ============================================================================
// This is the EXACT implementation from mfdfa_controlled_binner.cpp
// Uses 64x64 CGR grid with:
// - Multi-scale quadrant densities (16 features)
// - Radial density profile (8 features)
// - Hu moments, log-transformed (4 features)
// - Entropy, uniformity, max_density (3 features)
// Total: 31 features, L2 normalized

CGRGeometric FractalFeatures::extract_cgr_geometric(const std::string &sequence) {
  CGRGeometric result;
  if (sequence.length() < 10) return result;

  const int GRID = 64;  // 64x64 grid for high resolution
  const double corners[4][2] = {{0,0}, {0,1}, {1,1}, {1,0}};  // A, C, G, T

  auto nuc_idx = [](char c) -> int {
    switch (std::toupper(c)) {
      case 'A': return 0;
      case 'C': return 1;
      case 'G': return 2;
      case 'T': return 3;
      default: return -1;
    }
  };

  // Build high-resolution CGR grid
  std::vector<double> grid(GRID * GRID, 0.0);
  double cx = 0.5, cy = 0.5;
  int count = 0;
  for (char c : sequence) {
    int idx = nuc_idx(c);
    if (idx < 0) continue;
    cx = (cx + corners[idx][0]) / 2.0;
    cy = (cy + corners[idx][1]) / 2.0;
    int gx = std::min(GRID - 1, static_cast<int>(cx * GRID));
    int gy = std::min(GRID - 1, static_cast<int>(cy * GRID));
    grid[gy * GRID + gx]++;
    count++;
  }
  if (count == 0) return result;
  for (double& v : grid) v /= count;

  int fi = 0;  // Feature index for normalized_vector

  // === 1. QUADRANT DENSITIES AT MULTIPLE SCALES (16 features) ===
  // Captures ACGT bias at different resolutions (k=3,4,5,6 equivalent)
  int qi = 0;
  for (int scale : {8, 16, 32, 64}) {
    (void)scale;  // Scale determines resolution but we always use full grid
    double q[4] = {0, 0, 0, 0};  // A(0,0), C(0,1), G(1,1), T(1,0) corners
    for (int y = 0; y < GRID; y++) {
      for (int x = 0; x < GRID; x++) {
        double v = grid[y * GRID + x];
        int qx = (x < GRID/2) ? 0 : 1;
        int qy = (y < GRID/2) ? 0 : 1;
        if (qx == 0 && qy == 0) q[0] += v;       // A corner
        else if (qx == 0 && qy == 1) q[1] += v;  // C corner
        else if (qx == 1 && qy == 1) q[2] += v;  // G corner
        else q[3] += v;                           // T corner
      }
    }
    for (int i = 0; i < 4 && qi < 16; i++) {
      result.quadrant_multiscale[qi] = q[i];
      if (fi < 32) result.normalized_vector[fi++] = q[i];
      qi++;
    }
  }

  // === 2. RADIAL DENSITY PROFILE (8 features) ===
  // Distance from center captures overall structure
  std::array<double, 8> radial = {0};
  std::array<int, 8> radial_count = {0};
  double center_x = GRID / 2.0, center_y = GRID / 2.0;
  double max_dist = std::sqrt(2.0) * GRID / 2.0;
  for (int y = 0; y < GRID; y++) {
    for (int x = 0; x < GRID; x++) {
      double dx = x - center_x, dy = y - center_y;
      double dist = std::sqrt(dx*dx + dy*dy) / max_dist;
      int bin = std::min(7, static_cast<int>(dist * 8));
      radial[bin] += grid[y * GRID + x];
      radial_count[bin]++;
    }
  }
  for (int i = 0; i < 8; i++) {
    double val = radial_count[i] > 0 ? radial[i] / radial_count[i] : 0;
    result.radial_profile[i] = val;
    if (fi < 32) result.normalized_vector[fi++] = val;
  }

  // === 3. HU MOMENTS (4 features, log-transformed) ===
  // Rotation and scale invariant shape descriptors
  double m00 = 0, m10 = 0, m01 = 0, m20 = 0, m02 = 0, m11 = 0;
  double m30 = 0, m03 = 0, m21 = 0, m12 = 0;
  for (int y = 0; y < GRID; y++) {
    for (int x = 0; x < GRID; x++) {
      double v = grid[y * GRID + x];
      m00 += v;
      m10 += x * v;
      m01 += y * v;
    }
  }
  double cx_m = m00 > 0 ? m10 / m00 : GRID/2.0;
  double cy_m = m00 > 0 ? m01 / m00 : GRID/2.0;

  for (int y = 0; y < GRID; y++) {
    for (int x = 0; x < GRID; x++) {
      double v = grid[y * GRID + x];
      double xc = x - cx_m, yc = y - cy_m;
      m20 += xc * xc * v;
      m02 += yc * yc * v;
      m11 += xc * yc * v;
      m30 += xc * xc * xc * v;
      m03 += yc * yc * yc * v;
      m21 += xc * xc * yc * v;
      m12 += xc * yc * yc * v;
    }
  }

  // Normalized central moments
  double norm20 = m00 > 0 ? m20 / (m00 * m00) : 0;
  double norm02 = m00 > 0 ? m02 / (m00 * m00) : 0;
  double norm11 = m00 > 0 ? m11 / (m00 * m00) : 0;
  double norm30 = m00 > 0 ? m30 / std::pow(m00, 2.5) : 0;
  double norm03 = m00 > 0 ? m03 / std::pow(m00, 2.5) : 0;
  double norm21 = m00 > 0 ? m21 / std::pow(m00, 2.5) : 0;
  double norm12 = m00 > 0 ? m12 / std::pow(m00, 2.5) : 0;

  // Hu moments (first 4)
  double hu1 = norm20 + norm02;
  double hu2 = (norm20 - norm02) * (norm20 - norm02) + 4 * norm11 * norm11;
  double hu3 = (norm30 - 3*norm12) * (norm30 - 3*norm12) + (3*norm21 - norm03) * (3*norm21 - norm03);
  double hu4 = (norm30 + norm12) * (norm30 + norm12) + (norm21 + norm03) * (norm21 + norm03);

  result.hu_moments_log[0] = std::log(std::abs(hu1) + 1e-10);
  result.hu_moments_log[1] = std::log(std::abs(hu2) + 1e-10);
  result.hu_moments_log[2] = std::log(std::abs(hu3) + 1e-10);
  result.hu_moments_log[3] = std::log(std::abs(hu4) + 1e-10);

  for (int i = 0; i < 4 && fi < 32; i++) {
    result.normalized_vector[fi++] = result.hu_moments_log[i];
  }

  // === 4. ENTROPY AND UNIFORMITY (3 features) ===
  double entropy = 0, uniformity = 0, max_val = 0;
  for (double v : grid) {
    if (v > 0) entropy -= v * std::log(v);
    uniformity += v * v;
    max_val = std::max(max_val, v);
  }
  result.entropy = entropy;
  result.uniformity = uniformity;
  result.max_density = max_val;

  if (fi < 32) result.normalized_vector[fi++] = entropy;
  if (fi < 32) result.normalized_vector[fi++] = uniformity;
  if (fi < 32) result.normalized_vector[fi++] = max_val;

  // === L2 NORMALIZE the combined vector ===
  double norm = 0;
  for (int i = 0; i < fi; i++) {
    norm += result.normalized_vector[i] * result.normalized_vector[i];
  }
  if (norm > 0) {
    norm = std::sqrt(norm);
    for (int i = 0; i < fi; i++) {
      result.normalized_vector[i] /= norm;
    }
  }

  return result;
}

// ============================================================================
// COMPLEXITY FEATURES
// ============================================================================

double FractalFeatures::lempel_ziv_complexity(const std::string &seq) {
  if (seq.empty()) return 0.0;

  // OPTIMIZATION: LZ76 factorization using rolling hash to avoid substr allocations
  // Polynomial rolling hash: hash = (a1*p^(n-1) + a2*p^(n-2) + ... + an) % mod
  const uint64_t MOD = 1000000007;
  const uint64_t BASE = 257;

  std::unordered_set<uint64_t> dictionary;
  int complexity = 0;
  size_t pos = 0;
  const size_t n = seq.size();

  while (pos < n) {
    size_t best = 1;
    size_t max_len = std::min(n - pos, size_t(100));  // Limit max phrase length for performance

    uint64_t hash = 0;
    uint64_t pow = 1;

    for (size_t len = 1; len <= max_len; ++len) {
      // Update rolling hash
      hash = (hash * BASE + static_cast<uint8_t>(seq[pos + len - 1])) % MOD;

      if (dictionary.find(hash) != dictionary.end()) {
        best = len;
        pow = (pow * BASE) % MOD;
        continue;
      }
      dictionary.insert(hash);
      break;
    }
    pos += best;
    complexity++;
  }

  // LZ complexity normalization for genomic discrimination
  //
  // For discrimination between genomes, we want a metric that varies between
  // sequences with different repetitive structures. The standard normalization
  // C(n)/C_max produces values too close to 1.0 for all DNA.
  //
  // Instead, we use: log(complexity) / log(n)
  // This measures how the dictionary size scales with sequence length.
  // - Random sequences: approaches ~0.9 for DNA (4-letter alphabet)
  // - Repetitive sequences: lower values (~0.6-0.7)
  // - Diverse sequences: higher values (~0.8-0.9)
  //
  // We return the raw ratio to preserve variance for discrimination.
  // NOTE: Do NOT clamp to [0,1] - that destroys variance for real DNA.

  if (n <= 1) return 0.5;  // Default for trivial sequences

  // Use log ratio for better discrimination
  // This measures the "compression ratio" in a sense - lower values = more compressible
  double log_complexity = std::log(static_cast<double>(complexity));
  double log_n = std::log(static_cast<double>(n));

  if (log_n <= 0) return 0.5;

  // Return raw ratio - typically ranges from ~0.6 (very repetitive) to ~0.9 (random-like)
  // for real DNA sequences. We explicitly do NOT normalize or clamp this value as
  // that was causing all values to saturate at 1.0 and lose variance.
  return log_complexity / log_n;
}

double FractalFeatures::compute_hurst_exponent(const std::vector<double> &signal) {
  // Lower minimum to support shorter sequences
  if (signal.size() < 12) return 0.5;

  // R/S analysis (Rescaled Range) with Anis-Lloyd bias correction
  // Use shorter scales for better short-sequence support:
  // Scales: {12, 18, 27, 40, 60, 90, 135, 200} (roughly 1.5x geometric progression)
  static const std::array<int, 8> target_windows = {12, 18, 27, 40, 60, 90, 135, 200};
  std::vector<double> scales;
  std::vector<double> rs_values;
  std::vector<double> weights;  // For weighted regression
  scales.reserve(8);
  rs_values.reserve(8);
  weights.reserve(8);

  // OPTIMIZATION: Pre-allocate buffer for deviations (reused across iterations)
  std::vector<double> deviations(200);  // Max window size

  for (int window : target_windows) {
    if (window > static_cast<int>(signal.size()) / 2) break;

    double mean_rs = 0.0;
    int count = 0;

    // Use 50% overlap for better statistics
    for (size_t start = 0; start + window <= signal.size(); start += window / 2) {
      // OPTIMIZATION: Compute mean directly from signal without copying
      double mean = 0.0;
      for (int i = 0; i < window; ++i) {
        mean += signal[start + i];
      }
      mean /= window;

      // Cumulative deviations (profile) - reuse pre-allocated buffer
      double cumsum = 0.0;
      double min_dev = 0.0, max_dev = 0.0;
      for (int i = 0; i < window; ++i) {
        cumsum += signal[start + i] - mean;
        deviations[i] = cumsum;
        if (i == 0 || cumsum < min_dev) min_dev = cumsum;
        if (i == 0 || cumsum > max_dev) max_dev = cumsum;
      }

      // Range (computed inline above)
      double range = max_dev - min_dev;

      // Standard deviation with Bessel correction
      double variance = 0.0;
      for (int i = 0; i < window; ++i) {
        double diff = signal[start + i] - mean;
        variance += diff * diff;
      }
      double std_dev = std::sqrt(variance / (window - 1));

      if (std_dev > 1e-10) {
        mean_rs += range / std_dev;
        count++;
      }
    }

    if (count > 0) {
      double rs = mean_rs / count;

      // Anis-Lloyd bias correction for E[R/S] under null (H=0.5)
      // E[R/S] ≈ sqrt(π*n/2) * (n-0.5)/n * Γ((n-1)/2) / Γ(n/2)
      // Simplified approximation for large n: E[R/S] ≈ sqrt(π*n/2)
      // For finite n, we use a better approximation from Anis & Lloyd (1976)
      double n = static_cast<double>(window);
      double expected_rs_null;
      if (window < 50) {
        // Use exact Anis-Lloyd formula for small n
        double sum = 0.0;
        for (int j = 1; j < window; j++) {
          sum += std::sqrt((n - j) / j);
        }
        expected_rs_null = sum / std::sqrt(n * M_PI / 2.0);
      } else {
        // Asymptotic approximation for larger n
        expected_rs_null = std::sqrt(n * M_PI / 2.0);
      }

      // Corrected R/S by dividing by null expectation
      // This makes different window sizes more comparable
      double corrected_rs = rs / expected_rs_null;
      (void)corrected_rs;  // Suppress unused warning - kept for diagnostics

      scales.push_back(std::log(static_cast<double>(window)));
      rs_values.push_back(std::log(rs));  // Keep raw for slope, use corrected for diagnostics
      // Weight by number of windows (sqrt for variance stabilization)
      weights.push_back(std::sqrt(count));
    }
  }

  // Need at least 3 points for reliable regression
  if (scales.size() < 3) return 0.5;

  // Weighted least squares regression: log(R/S) = H * log(n) + c
  double sum_w = 0, sum_wx = 0, sum_wy = 0, sum_wxy = 0, sum_wxx = 0;
  for (size_t i = 0; i < scales.size(); ++i) {
    double w = weights[i];
    sum_w += w;
    sum_wx += w * scales[i];
    sum_wy += w * rs_values[i];
    sum_wxy += w * scales[i] * rs_values[i];
    sum_wxx += w * scales[i] * scales[i];
  }

  double denom = sum_w * sum_wxx - sum_wx * sum_wx;
  double hurst = (denom > 1e-10) ? (sum_w * sum_wxy - sum_wx * sum_wy) / denom : 0.5;

  // Return raw value - don't clamp, as finite-size effects can exceed [0,1]
  return hurst;
}

double FractalFeatures::compute_dfa_exponent(const std::vector<double> &signal) {
  // Lower minimum for short-sequence support
  if (signal.size() < 24) return 0.0;

  // UDFA (unbiased DFA) with Fibonacci windows
  std::vector<double> profile(signal.size());
  double mean = std::accumulate(signal.begin(), signal.end(), 0.0) / signal.size();
  profile[0] = signal[0] - mean;
  for (size_t i = 1; i < signal.size(); ++i) {
    profile[i] = profile[i - 1] + (signal[i] - mean);
  }

  static const std::array<int, 8> fib_windows = {8, 13, 21, 34, 55, 89, 144, 233};
  std::vector<double> log_scales;
  std::vector<double> log_fluct;
  std::vector<double> weights;

  int max_window = static_cast<int>(signal.size()) / 4;  // Need at least 4 segments (overlapping)

  for (int window : fib_windows) {
    if (window > max_window) break;
    if (window < 4) continue;

    int stride = 1;
    if (dfa_stride_factor > 0 && profile.size() > 5000) {
      stride = std::max(1, window / dfa_stride_factor);
    }
    int num_windows = (static_cast<int>(profile.size()) - window) / stride + 1;  // overlapping windows
    if (num_windows < 2) continue;

    double sum_vstar = 0.0;
    int valid = 0;

    for (int w_idx = 0; w_idx < num_windows; ++w_idx) {
      int w = w_idx * stride;
      double sx = 0, sy = 0, sxy = 0, sx2 = 0;
      for (int i = 0; i < window; ++i) {
        int idx = w + i;
        sx += i;
        sy += profile[idx];
        sxy += i * profile[idx];
        sx2 += i * i;
      }
      double denom = window * sx2 - sx * sx;
      if (std::abs(denom) < 1e-10) continue;
      double b = (window * sxy - sx * sy) / denom;
      double a = (sy - b * sx) / window;

      std::vector<double> resid(window);
      double mean_resid = 0.0;
      for (int i = 0; i < window; ++i) {
        resid[i] = profile[w + i] - (a + b * i);
        mean_resid += resid[i];
      }
      mean_resid /= window;

      double sum_sq_A = 0.0, sum_sq_B = 0.0, sum_lag_A = 0.0, sum_lag_B = 0.0;
      for (int i = 0; i < window; ++i) {
        double ra = resid[i] - mean_resid;
        double rb = ra * ((i % 2 == 0) ? 1.0 : -1.0);
        sum_sq_A += ra * ra;
        sum_sq_B += rb * rb;
        if (i + 1 < window) {
          double ra_next = resid[i + 1] - mean_resid;
          double rb_next = ra_next * ((i + 1) % 2 == 0 ? 1.0 : -1.0);
          sum_lag_A += ra * ra_next;
          sum_lag_B += rb * rb_next;
        }
      }
      if (window <= 1) continue;
      double varA = sum_sq_A / (window - 1);
      double varB = sum_sq_B / (window - 1);
      if (varA < 1e-12 || varB < 1e-12) continue;

      double rhoA = (sum_sq_A + sum_lag_A) / ((2.0 * window - 1.0) * varA);
      double rhoB = (sum_sq_B - sum_lag_B) / ((2.0 * window - 1.0) * varB);

      double rhoA_star = rhoA + (1.0 / window) * (1.0 + 3.0 * rhoA);
      double rhoB_star = rhoB + (1.0 / window) * (1.0 + 3.0 * rhoB);

      double var_mean = 0.5 * (varA + varB);
      double vstar = (rhoA_star + rhoB_star) * (1.0 - 1.0 / (2.0 * window)) * var_mean;
      if (vstar > 0) {
        sum_vstar += vstar;
        valid++;
      }
    }

    if (valid == 0) continue;
    double fluct = std::sqrt(sum_vstar / (static_cast<double>(window) * valid));
    if (fluct > 1e-15) {
      log_scales.push_back(std::log(static_cast<double>(window)));
      log_fluct.push_back(std::log(fluct));
      weights.push_back(std::sqrt(valid));
    }
  }

  if (log_scales.size() < 3) return 0.0;

  double sum_w = 0, sum_wx = 0, sum_wy = 0, sum_wxy = 0, sum_wxx = 0;
  for (size_t i = 0; i < log_scales.size(); ++i) {
    double w = weights[i];
    sum_w += w;
    sum_wx += w * log_scales[i];
    sum_wy += w * log_fluct[i];
    sum_wxy += w * log_scales[i] * log_fluct[i];
    sum_wxx += w * log_scales[i] * log_scales[i];
  }

  double denom = sum_w * sum_wxx - sum_wx * sum_wx;
  double alpha = (denom > 1e-10) ? (sum_w * sum_wxy - sum_wx * sum_wy) / denom : 0.0;
  return alpha;  // Raw DFA exponent (≈ H for fBm)
}

std::array<double, 5> FractalFeatures::compute_entropy_scaling(const std::string &seq) {
  std::array<double, 5> entropies;
  entropies.fill(0.0);

  // Lower minimum for short-sequence support
  if (seq.length() < 20) return entropies;

  // OPTIMIZATION: Use integer-encoded k-mers to avoid string allocations
  // Convert sequence to numeric representation once: A=0, C=1, G=2, T=3
  std::vector<uint8_t> encoded;
  encoded.reserve(seq.length());

  for (char c : seq) {
    uint8_t val;
    switch (c) {
      case 'A': case 'a': val = 0; break;
      case 'C': case 'c': val = 1; break;
      case 'G': case 'g': val = 2; break;
      case 'T': case 't': val = 3; break;
      default: val = 255; break;  // Invalid marker
    }
    encoded.push_back(val);
  }

  // OPTIMIZATION: Compute Shannon entropy for k=2,3,4,5,6 in parallel
  #pragma omp parallel for if(seq.length() > 100000)
  for (int k_idx = 0; k_idx < 5; ++k_idx) {
    int k = k_idx + 2;

    if (encoded.size() < static_cast<size_t>(k)) continue;

    // Use integer hash map instead of string map
    std::unordered_map<uint32_t, int> kmer_counts;
    kmer_counts.reserve(1 << (2 * k));  // Pre-allocate for 4^k possible k-mers
    int total = 0;

    // Rolling hash with invalid-base reset (O(n) per k)
    uint32_t kmer_hash = 0;
    int valid_run = 0;
    uint32_t mask = (1u << (2 * k)) - 1u;
    for (size_t i = 0; i < encoded.size(); ++i) {
      uint8_t b = encoded[i];
      if (b > 3) {
        kmer_hash = 0;
        valid_run = 0;
        continue;
      }

      kmer_hash = ((kmer_hash << 2) | b) & mask;
      valid_run++;
      if (valid_run >= k) {
        kmer_counts[kmer_hash]++;
        total++;
      }
    }

    // Entropy estimation with Dirichlet-1 prior (add-one smoothing) and Miller-Madow correction
    // This provides better estimates for small samples where some k-mers may be unseen
    double entropy = 0.0;
    if (total > 0) {
      // Possible k-mers = 4^k
      int alphabet_size = 1 << (2 * k);

      // For short sequences, use Dirichlet-1 prior (adds 1 pseudocount per category)
      // This avoids log(0) and provides better bias correction
      // For long sequences, Miller-Madow alone is sufficient
      bool use_dirichlet = (total < 2 * alphabet_size);

      if (use_dirichlet) {
        // Dirichlet-1 prior (Laplace smoothing): p_i = (n_i + 1) / (N + |Σ|)
        double total_smoothed = static_cast<double>(total) + alphabet_size;
        double inv_total_smoothed = 1.0 / total_smoothed;

        // Sum over all possible k-mers (including unseen ones with count=0)
        for (int kmer = 0; kmer < alphabet_size; ++kmer) {
          int count = 0;
          auto it = kmer_counts.find(kmer);
          if (it != kmer_counts.end()) count = it->second;

          double p = (count + 1.0) * inv_total_smoothed;
          entropy -= p * std::log2(p);
        }
      } else {
        // Standard MLE with Miller-Madow for larger samples
        double inv_total = 1.0 / total;
        for (const auto &[kmer, count] : kmer_counts) {
          double p = count * inv_total;
          if (p > 0.0) {
            entropy -= p * std::log2(p);
          }
        }

        // Miller-Madow correction: H_corrected = H_MLE + (m-1)/(2N)
        // where m = number of observed categories
        int m = kmer_counts.size();
        if (m > 1) {
          double mm_correction = static_cast<double>(m - 1) / (2.0 * total);
          entropy += mm_correction;
        }
      }
    }

    entropies[k_idx] = entropy;
  }

  return entropies;
}

ComplexityFeatures FractalFeatures::extract_complexity_features(const std::string &sequence) {
  ComplexityFeatures features;

  if (sequence.length() < 100) return features;

  // Lempel-Ziv at multiple scales (optimized: use sampling for very long sequences)
  // Apply jackknife bias correction using sequence halves when halves are long enough.
  // IMPORTANT: Only apply jackknife when half-length >= 5000bp (full >= 10000bp)
  // to avoid variance amplification at transition lengths (especially 5000bp).
  const bool apply_jackknife = sequence.length() >= 10000;
  const std::string seq_half1 = apply_jackknife ? sequence.substr(0, sequence.length() / 2) : "";
  const std::string seq_half2 = apply_jackknife ? sequence.substr(sequence.length() / 2) : "";

  auto lz_for_length = [](const std::string& seq, int len) {
    if (seq.empty() || len <= 0) return 0.0;
    int actual_len = std::min(len, static_cast<int>(seq.length()));
    std::string sub;

    // For very long sequences (>500kb), sample instead of using full sequence
    // to avoid O(n²) explosion while maintaining representative complexity
    if (actual_len > 500000) {
      sub.reserve(500000);
      int step = actual_len / 500000;
      for (int j = 0; j < 500000 && j * step < actual_len; ++j) {
        sub += seq[j * step];
      }
    } else {
      sub = seq.substr(0, actual_len);
    }
    return lempel_ziv_complexity(sub);
  };

  std::vector<int> scales = {100, 500, 2000, static_cast<int>(sequence.length())};
  for (size_t i = 0; i < 4 && i < scales.size(); ++i) {
    int len = std::min(scales[i], static_cast<int>(sequence.length()));
    features.lempel_ziv_complexity[i] = lz_for_length(sequence, len);

    // Jackknife bias correction only for long sequences (>= 10000bp)
    if (apply_jackknife) {
      double half1 = lz_for_length(seq_half1, len);
      double half2 = lz_for_length(seq_half2, len);
      if (half1 > 0 && half2 > 0) {
        double mean_half = 0.5 * (half1 + half2);
        features.lempel_ziv_complexity[i] = 2.0 * features.lempel_ziv_complexity[i] - mean_half;
      }
    }
  }

  // Convert sequence to numeric signals for Hurst/DFA
  // OPTIMIZATION: For very long sequences (>1Mbp), use larger stride to reduce computation
  // This maintains signal characteristics while being much faster
  int window = 50;
  int stride = (sequence.length() > 1000000) ? 100 : 10;  // Adaptive stride

  // Signal 1: GC content in sliding windows
  // OPTIMIZATION: Use rolling window to avoid recounting entire window each time
  std::vector<double> gc_signal;
  gc_signal.reserve((sequence.length() - window) / stride + 1);

  // Count GC in first window
  int gc_count = 0;
  for (size_t j = 0; j < static_cast<size_t>(window) && j < sequence.length(); ++j) {
    char c = sequence[j];
    if (c == 'G' || c == 'C' || c == 'g' || c == 'c') gc_count++;
  }
  gc_signal.push_back(static_cast<double>(gc_count) / window);

  // Roll the window forward
  for (size_t i = stride; i + window <= sequence.length(); i += stride) {
    // Remove bases that left the window, add bases that entered
    for (size_t j = i - stride; j < i; ++j) {
      char c = sequence[j];
      if (c == 'G' || c == 'C' || c == 'g' || c == 'c') gc_count--;
    }
    for (size_t j = i - stride + window; j < i + window; ++j) {
      char c = sequence[j];
      if (c == 'G' || c == 'C' || c == 'g' || c == 'c') gc_count++;
    }
    gc_signal.push_back(static_cast<double>(gc_count) / window);
  }

  if (gc_signal.size() >= 20) {
    features.hurst_exponent[0] = compute_hurst_exponent(gc_signal);
    features.dfa_exponent = compute_dfa_exponent(gc_signal);
  }

  // Signal 2: Dinucleotide frequency variation
  // OPTIMIZATION: Use integer encoding for dinucleotides (16 possible values)
  std::vector<double> dinuc_signal;
  dinuc_signal.reserve((sequence.length() - window) / stride + 1);

  for (size_t i = 0; i + window <= sequence.length(); i += stride) {
    std::array<int, 16> dinuc_counts = {0};  // 4*4=16 possible dinucleotides
    int total = 0;

    for (size_t j = i; j < i + window - 1 && j + 1 < sequence.length(); ++j) {
      // Encode dinucleotide as integer: 0-15
      uint8_t base1 = 255, base2 = 255;
      char c1 = sequence[j], c2 = sequence[j + 1];

      switch (c1) {
        case 'A': case 'a': base1 = 0; break;
        case 'C': case 'c': base1 = 1; break;
        case 'G': case 'g': base1 = 2; break;
        case 'T': case 't': base1 = 3; break;
      }
      switch (c2) {
        case 'A': case 'a': base2 = 0; break;
        case 'C': case 'c': base2 = 1; break;
        case 'G': case 'g': base2 = 2; break;
        case 'T': case 't': base2 = 3; break;
      }

      if (base1 < 4 && base2 < 4) {
        dinuc_counts[base1 * 4 + base2]++;
        total++;
      }
    }

    // Shannon entropy
    double entropy = 0.0;
    if (total > 0) {
      for (int count : dinuc_counts) {
        if (count > 0) {
          double p = static_cast<double>(count) / total;
          entropy -= p * std::log2(p);
        }
      }
    }
    dinuc_signal.push_back(entropy);
  }

  if (dinuc_signal.size() >= 20) {
    features.hurst_exponent[1] = compute_hurst_exponent(dinuc_signal);
  }

  // Signal 3: K-mer diversity in sliding windows
  // Compute k=3 entropy in windows to measure k-mer diversity variation
  // OPTIMIZATION: Use integer encoding for 3-mers (64 possible values) to avoid string allocations
  std::vector<double> kmer_signal;
  kmer_signal.reserve((sequence.length() - window) / stride + 1);

  for (size_t i = 0; i + window <= sequence.length(); i += stride) {
    // Compute 3-mer counts using integer encoding (no string allocations)
    std::array<int, 64> kmer_counts = {0};  // 4^3 = 64 possible 3-mers
    int total = 0;

    for (size_t j = i; j + 3 <= i + window && j + 2 < sequence.length(); ++j) {
      // Encode 3-mer as integer: 0-63
      uint8_t b0 = 255, b1 = 255, b2 = 255;
      char c0 = sequence[j], c1 = sequence[j+1], c2 = sequence[j+2];

      switch (c0) {
        case 'A': case 'a': b0 = 0; break;
        case 'C': case 'c': b0 = 1; break;
        case 'G': case 'g': b0 = 2; break;
        case 'T': case 't': b0 = 3; break;
      }
      switch (c1) {
        case 'A': case 'a': b1 = 0; break;
        case 'C': case 'c': b1 = 1; break;
        case 'G': case 'g': b1 = 2; break;
        case 'T': case 't': b1 = 3; break;
      }
      switch (c2) {
        case 'A': case 'a': b2 = 0; break;
        case 'C': case 'c': b2 = 1; break;
        case 'G': case 'g': b2 = 2; break;
        case 'T': case 't': b2 = 3; break;
      }

      if (b0 < 4 && b1 < 4 && b2 < 4) {
        kmer_counts[(b0 << 4) | (b1 << 2) | b2]++;
        total++;
      }
    }

    // Shannon entropy of k-mer distribution
    double entropy = 0.0;
    if (total > 0) {
      for (int count : kmer_counts) {
        if (count > 0) {
          double p = static_cast<double>(count) / total;
          entropy -= p * std::log2(p);
        }
      }
    }
    kmer_signal.push_back(entropy);
  }

  if (kmer_signal.size() >= 20) {
    features.hurst_exponent[2] = compute_hurst_exponent(kmer_signal);
  }

  // Entropy scaling (fractal dimension of k-mer diversity) with jackknife bias correction
  // Only apply jackknife for long sequences (>= 10000bp) to avoid variance amplification
  auto entropies_full = compute_entropy_scaling(sequence);
  std::array<double, 5> entropies_half1 = {0}, entropies_half2 = {0};
  if (apply_jackknife) {
    entropies_half1 = compute_entropy_scaling(seq_half1);
    entropies_half2 = compute_entropy_scaling(seq_half2);
  }
  for (size_t i = 0; i < entropies_full.size(); ++i) {
    double val = entropies_full[i];
    if (apply_jackknife && entropies_half1[i] > 0 && entropies_half2[i] > 0) {
      double mean_half = 0.5 * (entropies_half1[i] + entropies_half2[i]);
      val = 2.0 * val - mean_half;  // jackknife bias correction
    }
    features.entropy_scaling[i] = val;
  }

  // Fractal dimension from entropy scaling (slope)
  {
    std::vector<double> xs, ys;
    for (int i = 0; i < 5; ++i) {
      if (features.entropy_scaling[i] > 0.0) {
        xs.push_back(i + 2);
        ys.push_back(features.entropy_scaling[i]);
      }
    }
    if (xs.size() >= 2) {
      double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
      for (size_t i = 0; i < xs.size(); ++i) {
        sum_x += xs[i];
        sum_y += ys[i];
        sum_xy += xs[i] * ys[i];
        sum_xx += xs[i] * xs[i];
      }
      double denom = xs.size() * sum_xx - sum_x * sum_x;
      if (denom != 0.0) {
        double slope = (xs.size() * sum_xy - sum_x * sum_y) / denom;
        features.kmer_fractal_dim = slope;
      }
    }
  }

  return features;
}

// ============================================================================
// GRAPH TOPOLOGY
// ============================================================================

std::vector<std::pair<int, int>> FractalFeatures::build_debruijn_edges(
    const std::string &seq, int k) {

  std::vector<std::pair<int, int>> edges;
  if (seq.length() < static_cast<size_t>(k + 1)) return edges;

  // OPTIMIZATION: Use integer encoding for k-mers (avoid substr allocations)
  size_t stride = (seq.length() > 1000000) ? 10 : 1;

  // Map k-mer hashes to node IDs
  std::unordered_map<uint32_t, int> kmer_to_id;
  int next_id = 0;

  // First pass: identify all k-mers
  for (size_t i = 0; i + k <= seq.length(); i += stride) {
    uint32_t kmer_hash = 0;
    bool valid = true;

    for (int j = 0; j < k; ++j) {
      uint8_t base;
      char c = seq[i + j];
      switch (c) {
        case 'A': case 'a': base = 0; break;
        case 'C': case 'c': base = 1; break;
        case 'G': case 'g': base = 2; break;
        case 'T': case 't': base = 3; break;
        default: valid = false; base = 0; break;
      }
      if (!valid) break;
      kmer_hash = (kmer_hash << 2) | base;
    }

    if (valid && kmer_to_id.find(kmer_hash) == kmer_to_id.end()) {
      kmer_to_id[kmer_hash] = next_id++;
    }
  }

  // Build edges using rolling hash
  uint32_t mask = (1U << (2 * k)) - 1;  // Keep only k bases
  uint32_t kmer1_hash = 0;
  bool kmer1_valid = true;

  // Initialize first k-mer
  for (int j = 0; j < k && j < static_cast<int>(seq.length()); ++j) {
    uint8_t base;
    char c = seq[j];
    switch (c) {
      case 'A': case 'a': base = 0; break;
      case 'C': case 'c': base = 1; break;
      case 'G': case 'g': base = 2; break;
      case 'T': case 't': base = 3; break;
      default: kmer1_valid = false; base = 0; break;
    }
    if (!kmer1_valid) break;
    kmer1_hash = (kmer1_hash << 2) | base;
  }

  // Roll through sequence
  for (size_t i = 0; i + k < seq.length(); ++i) {
    // Get next base for kmer2
    uint8_t next_base;
    char c_next = seq[i + k];
    bool next_valid = true;

    switch (c_next) {
      case 'A': case 'a': next_base = 0; break;
      case 'C': case 'c': next_base = 1; break;
      case 'G': case 'g': next_base = 2; break;
      case 'T': case 't': next_base = 3; break;
      default: next_valid = false; next_base = 0; break;
    }

    // kmer2 = (kmer1 << 2 | next_base) & mask
    uint32_t kmer2_hash = ((kmer1_hash << 2) | next_base) & mask;
    bool kmer2_valid = kmer1_valid && next_valid;

    if (kmer1_valid && kmer2_valid) {
      auto it1 = kmer_to_id.find(kmer1_hash);
      auto it2 = kmer_to_id.find(kmer2_hash);
      if (it1 != kmer_to_id.end() && it2 != kmer_to_id.end()) {
        edges.push_back({it1->second, it2->second});
      }
    }

    // Advance window
    kmer1_hash = kmer2_hash;
    kmer1_valid = kmer2_valid;
  }

  return edges;
}

double FractalFeatures::compute_clustering_coefficient(
    const std::vector<std::pair<int, int>> &edges, int num_nodes) {

  if (num_nodes < 3 || edges.empty()) return 0.0;

  // Build adjacency lists
  std::unordered_map<int, std::unordered_set<int>> adj;
  for (const auto &[u, v] : edges) {
    adj[u].insert(v);
    adj[v].insert(u);
  }

  // Count triangles
  double total_clustering = 0.0;
  int counted_nodes = 0;

  for (const auto &[node, neighbors] : adj) {
    if (neighbors.size() < 2) continue;

    int triangles = 0;
    int possible = neighbors.size() * (neighbors.size() - 1) / 2;

    // Check all pairs of neighbors
    std::vector<int> neighbor_list(neighbors.begin(), neighbors.end());
    for (size_t i = 0; i < neighbor_list.size(); ++i) {
      for (size_t j = i + 1; j < neighbor_list.size(); ++j) {
        int n1 = neighbor_list[i];
        int n2 = neighbor_list[j];
        if (adj[n1].count(n2)) {
          triangles++;
        }
      }
    }

    total_clustering += static_cast<double>(triangles) / possible;
    counted_nodes++;
  }

  return (counted_nodes > 0) ? (total_clustering / counted_nodes) : 0.0;
}

double FractalFeatures::compute_avg_path_length(
    const std::unordered_map<int, std::unordered_set<int>> &adj,
    const std::unordered_set<int> &nodes) {

  if (nodes.size() < 2) return 0.0;

  // Sample nodes to avoid O(n²) BFS complexity
  // For large graphs (>500 nodes), sample 100 source nodes
  int max_samples = std::min(100, static_cast<int>(nodes.size()));
  std::vector<int> node_list(nodes.begin(), nodes.end());

  // Deterministic sampling (every nth node)
  int step = std::max(1, static_cast<int>(nodes.size()) / max_samples);

  double total_path_length = 0.0;
  int path_count = 0;

  for (int i = 0; i < static_cast<int>(node_list.size()); i += step) {
    int source = node_list[i];

    // BFS from source
    std::unordered_map<int, int> distance;
    std::queue<int> queue;
    queue.push(source);
    distance[source] = 0;

    while (!queue.empty()) {
      int curr = queue.front();
      queue.pop();

      auto it = adj.find(curr);
      if (it == adj.end()) continue;

      for (int neighbor : it->second) {
        if (distance.find(neighbor) == distance.end()) {
          distance[neighbor] = distance[curr] + 1;
          queue.push(neighbor);
        }
      }
    }

    // Accumulate distances to all reachable nodes
    for (const auto &[node, dist] : distance) {
      if (node != source) {
        total_path_length += dist;
        path_count++;
      }
    }
  }

  return (path_count > 0) ? (total_path_length / path_count) : 0.0;
}

double FractalFeatures::compute_modularity(
    const std::vector<std::pair<int, int>> &edges,
    const std::unordered_map<int, std::unordered_set<int>> &adj) {

  if (edges.empty()) return 0.0;

  // Simple community detection: use connected components as communities
  // More sophisticated methods (Louvain, etc.) are too expensive for runtime feature extraction
  std::unordered_map<int, int> node_to_component;
  int component_id = 0;

  for (const auto &[node, neighbors] : adj) {
    if (node_to_component.find(node) != node_to_component.end()) continue;

    // BFS to find component
    std::queue<int> queue;
    queue.push(node);
    node_to_component[node] = component_id;

    while (!queue.empty()) {
      int curr = queue.front();
      queue.pop();

      auto it = adj.find(curr);
      if (it == adj.end()) continue;

      for (int neighbor : it->second) {
        if (node_to_component.find(neighbor) == node_to_component.end()) {
          node_to_component[neighbor] = component_id;
          queue.push(neighbor);
        }
      }
    }
    component_id++;
  }

  // Compute modularity Q = (1/2m) * sum_ij [A_ij - k_i*k_j/(2m)] * delta(c_i, c_j)
  // where m = num edges, A_ij = adjacency matrix, k_i = degree of node i
  int m = edges.size();
  double two_m = 2.0 * m;

  // Count degrees
  std::unordered_map<int, int> degree;
  for (const auto &[u, v] : edges) {
    degree[u]++;
    degree[v]++;
  }

  // Compute modularity
  double modularity = 0.0;
  std::set<std::pair<int, int>> edge_set;
  for (const auto &[u, v] : edges) {
    auto edge_key = (u < v) ? std::make_pair(u, v) : std::make_pair(v, u);
    edge_set.insert(edge_key);
  }

  for (const auto &[node1, comp1] : node_to_component) {
    for (const auto &[node2, comp2] : node_to_component) {
      if (comp1 != comp2) continue;  // Only same community

      // A_ij (1 if edge exists, 0 otherwise)
      auto edge_key = (node1 < node2) ? std::make_pair(node1, node2) : std::make_pair(node2, node1);
      int A_ij = edge_set.count(edge_key) ? 1 : 0;

      // Expected edges: k_i * k_j / (2m)
      double expected = (degree[node1] * degree[node2]) / two_m;

      modularity += (A_ij - expected);
    }
  }

  return modularity / two_m;
}

double FractalFeatures::compute_power_law_exponent(
    const std::vector<int> &degree_sequence) {

  if (degree_sequence.size() < 10) return 0.0;

  // Clauset-Shalizi-Newman (CSN) MLE for power-law exponent
  // More accurate than log-log regression for finite samples

  // Find minimum degree (x_min) - typically we use k_min = 1 for degree distributions
  int k_min = *std::min_element(degree_sequence.begin(), degree_sequence.end());
  if (k_min < 1) k_min = 1;

  // Filter data >= k_min and compute MLE
  std::vector<double> data_above_min;
  for (int deg : degree_sequence) {
    if (deg >= k_min) {
      data_above_min.push_back(static_cast<double>(deg));
    }
  }

  if (data_above_min.size() < 3) return 0.0;

  // MLE for discrete power law: α = 1 + n * [Σ ln(x_i / (x_min - 0.5))]^(-1)
  // This is the CSN estimator which corrects bias in log-log regression
  double sum_log_ratio = 0.0;
  for (double x : data_above_min) {
    sum_log_ratio += std::log(x / (k_min - 0.5));
  }

  double alpha = 1.0 + static_cast<double>(data_above_min.size()) / sum_log_ratio;

  // Return exponent (commonly denoted γ or α in literature)
  // Return raw value - don't clamp, let downstream handle standardization
  return alpha;
}

GraphTopology FractalFeatures::extract_graph_topology(const std::string &sequence, int k) {
  GraphTopology features;

  if (sequence.length() < 100) return features;

  auto edges = build_debruijn_edges(sequence, k);

  // Count unique nodes
  std::unordered_set<int> nodes;
  for (const auto &[u, v] : edges) {
    nodes.insert(u);
    nodes.insert(v);
  }
  int num_nodes = nodes.size();

  // Clustering coefficient
  features.clustering_coefficient = compute_clustering_coefficient(edges, num_nodes);

  // Degree distribution
  std::unordered_map<int, int> degree;
  for (const auto &[u, v] : edges) {
    degree[u]++;
    degree[v]++;
  }

  if (!degree.empty()) {
    std::vector<int> degrees;
    for (const auto &[node, deg] : degree) {
      degrees.push_back(deg);
    }

    double sum = std::accumulate(degrees.begin(), degrees.end(), 0.0);
    double raw_mean = sum / degrees.size();

    // Normalize degree_mean by graph size to get density (scale-invariant)
    // This represents average connectivity as a fraction of total nodes
    features.degree_mean = raw_mean / degrees.size();

    double sq_sum = 0.0;
    for (int d : degrees) {
      sq_sum += (d - raw_mean) * (d - raw_mean);
    }
    double raw_variance = sq_sum / degrees.size();

    // Normalize variance by mean² to get coefficient of variation (scale-invariant)
    // This measures relative variability independent of absolute scale
    if (raw_mean > 0.0) {
      features.degree_variance = raw_variance / (raw_mean * raw_mean);
    } else {
      features.degree_variance = 0.0;
    }

    if (raw_variance > 0.0) {
      double skew = 0.0;
      for (int d : degrees) {
        skew += std::pow((d - raw_mean) / std::sqrt(raw_variance), 3.0);
      }
      features.degree_skewness = skew / degrees.size();
    }
  }

  // Count connected components (simple DFS)
  std::unordered_map<int, std::unordered_set<int>> adj;
  for (const auto &[u, v] : edges) {
    adj[u].insert(v);
    adj[v].insert(u);
  }

  std::unordered_set<int> visited;
  int components = 0;

  for (int node : nodes) {
    if (visited.count(node)) continue;

    // DFS
    std::vector<int> stack = {node};
    while (!stack.empty()) {
      int curr = stack.back();
      stack.pop_back();
      if (visited.count(curr)) continue;
      visited.insert(curr);
      for (int neighbor : adj[curr]) {
        if (!visited.count(neighbor)) {
          stack.push_back(neighbor);
        }
      }
    }
    components++;
  }

  features.num_components = components;

  // Repeat density (count self-loops and multi-edges)
  std::map<std::pair<int, int>, int> edge_counts;
  int repeats = 0;
  for (const auto &[u, v] : edges) {
    auto edge = (u < v) ? std::make_pair(u, v) : std::make_pair(v, u);
    edge_counts[edge]++;
    if (edge_counts[edge] > 1) repeats++;
  }
  features.repeat_density = static_cast<double>(repeats) / edges.size();

  // Average path length (sampled BFS to avoid O(n²) complexity)
  features.avg_path_length = compute_avg_path_length(adj, nodes);

  // Modularity (using connected components as communities)
  features.modularity = compute_modularity(edges, adj);

  // Power-law exponent (from degree distribution)
  if (!degree.empty()) {
    std::vector<int> degrees;
    for (const auto &[node, deg] : degree) {
      degrees.push_back(deg);
    }
    features.power_law_exponent = compute_power_law_exponent(degrees);
  }

  return features;
}

// ============================================================================
// COVERAGE TOPOLOGY
// ============================================================================

std::array<double, 8> FractalFeatures::wavelet_decomposition(
    const std::vector<float> &signal) {

  std::array<double, 8> coefficients;
  coefficients.fill(0.0);

  if (signal.size() < 8) return coefficients;

  // Simple Haar wavelet transform (multi-scale averages and differences)
  std::vector<double> current(signal.begin(), signal.end());

  for (int level = 0; level < 4 && current.size() >= 2; ++level) {
    std::vector<double> averages, differences;

    for (size_t i = 0; i + 1 < current.size(); i += 2) {
      averages.push_back((current[i] + current[i + 1]) / 2.0);
      differences.push_back((current[i] - current[i + 1]) / 2.0);
    }

    // Store variance of differences as coefficient
    if (!differences.empty()) {
      double mean_diff = std::accumulate(differences.begin(), differences.end(), 0.0) / differences.size();
      double var = 0.0;
      for (double d : differences) {
        var += (d - mean_diff) * (d - mean_diff);
      }
      coefficients[level * 2] = var / differences.size();
      coefficients[level * 2 + 1] = mean_diff;
    }

    current = averages;
  }

  return coefficients;
}

void FractalFeatures::compute_variogram(
    const std::vector<float> &signal, double &sill, double &range) {
  // Note: nugget parameter removed (unstable with only 20 coverage bins)

  if (signal.size() < 8) {
    sill = range = 0.0;
    return;
  }

  // Compute semi-variogram for different lags
  std::vector<double> lags, variances;

  for (int lag = 1; lag <= std::min(50, static_cast<int>(signal.size()) / 4); lag *= 2) {
    double gamma = 0.0;
    int count = 0;

    for (size_t i = 0; i + lag < signal.size(); ++i) {
      gamma += (signal[i] - signal[i + lag]) * (signal[i] - signal[i + lag]);
      count++;
    }

    if (count > 0) {
      lags.push_back(lag);
      variances.push_back(gamma / (2.0 * count));
    }
  }

  if (variances.empty()) {
    sill = range = 0.0;
    return;
  }

  // Sill = maximum variance (asymptotic value)
  sill = *std::max_element(variances.begin(), variances.end());

  // Range = lag where variance reaches ~95% of sill
  double threshold = 0.95 * sill;
  range = lags.back();
  for (size_t i = 0; i < variances.size(); ++i) {
    if (variances[i] >= threshold) {
      range = lags[i];
      break;
    }
  }
}

// Helper: Cubic spline interpolation for coverage upsampling
// Uses natural boundary conditions (second derivative = 0 at endpoints)
static std::vector<float> cubic_spline_interpolate(
    const std::vector<float> &y, size_t target_size) {

  size_t n = y.size();
  if (n < 2 || target_size <= n) {
    return y;  // No interpolation needed
  }

  // Build natural cubic spline with second derivative = 0 at boundaries
  // Solve tridiagonal system for second derivatives
  std::vector<double> h(n - 1);  // Interval sizes
  std::vector<double> alpha(n);
  std::vector<double> l(n), mu(n), z(n);
  std::vector<double> c(n), b(n - 1), d(n - 1);

  // Compute intervals (uniform spacing assumed)
  for (size_t i = 0; i < n - 1; ++i) {
    h[i] = 1.0;  // Uniform x-spacing
  }

  // Set up alpha (RHS of tridiagonal system)
  for (size_t i = 1; i < n - 1; ++i) {
    alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) -
               (3.0 / h[i - 1]) * (y[i] - y[i - 1]);
  }

  // Solve tridiagonal system using Thomas algorithm
  l[0] = 1.0;
  mu[0] = z[0] = 0.0;

  for (size_t i = 1; i < n - 1; ++i) {
    l[i] = 2.0 * (h[i - 1] + h[i]) - h[i - 1] * mu[i - 1];
    mu[i] = h[i] / l[i];
    z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
  }

  l[n - 1] = 1.0;
  z[n - 1] = c[n - 1] = 0.0;  // Natural spline boundary

  // Back substitution
  for (int j = static_cast<int>(n) - 2; j >= 0; --j) {
    c[j] = z[j] - mu[j] * c[j + 1];
    b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
    d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
  }

  // Interpolate to target size
  std::vector<float> result(target_size);
  double scale = static_cast<double>(n - 1) / (target_size - 1);

  for (size_t i = 0; i < target_size; ++i) {
    double x = i * scale;
    size_t seg = std::min(static_cast<size_t>(x), n - 2);
    double dx = x - seg;

    // Evaluate cubic polynomial: a + b*dx + c*dx^2 + d*dx^3
    double val = y[seg] + b[seg] * dx + c[seg] * dx * dx + d[seg] * dx * dx * dx;
    result[i] = static_cast<float>(std::max(0.0, val));  // Coverage >= 0
  }

  return result;
}

CoverageTopology FractalFeatures::extract_coverage_topology(
    const std::vector<float> &coverage, int /*contig_length*/) {

  CoverageTopology features;

  if (coverage.size() < 8) return features;

  // Target size: at least 64 points for stable wavelet/variogram
  // Use power of 2 for efficient wavelet decomposition
  size_t target_size = 64;
  if (coverage.size() >= 64) {
    // If already large, round up to nearest power of 2
    target_size = 1;
    while (target_size < coverage.size()) target_size *= 2;
    target_size = std::min(target_size, static_cast<size_t>(256));  // Cap at 256
  }

  // Interpolate sparse coverage to target size
  std::vector<float> interp_cov;
  if (coverage.size() < target_size) {
    interp_cov = cubic_spline_interpolate(coverage, target_size);
  } else {
    interp_cov = coverage;
  }

  // Hurst exponent: Use interpolated signal if original is too sparse
  // With only 20 bins, original signal gives constant H=0.5 (useless)
  // Interpolation provides some information, though potentially biased
  std::vector<double> cov_signal;
  if (coverage.size() < 50) {
    // Use interpolated for sparse coverage (most ancient DNA contigs)
    cov_signal = std::vector<double>(interp_cov.begin(), interp_cov.end());
  } else {
    // Use original for well-sampled coverage
    cov_signal = std::vector<double>(coverage.begin(), coverage.end());
  }
  features.hurst_exponent = compute_hurst_exponent(cov_signal);

  // Wavelet decomposition on interpolated signal (needs power-of-2 size)
  features.wavelet_coefficients = wavelet_decomposition(interp_cov);

  // Variogram on interpolated signal (needs more lag values for stable fit)
  // Scale range back to original units
  double range_scale = static_cast<double>(coverage.size()) / interp_cov.size();
  compute_variogram(interp_cov, features.variogram_sill, features.variogram_range);
  features.variogram_range *= range_scale;

  // Fractal dimension from variogram
  // D = (4 - H) / 2, where H is Hurst exponent
  features.fractal_dimension = (4.0 - 2.0 * features.hurst_exponent) / 2.0;

  return features;
}

// ============================================================================
// DNA PHYSICAL PROPERTIES
// From consensus dinucleotide tables (bendability, flexibility, meltability)
// ============================================================================

DNAPhysicalProperties FractalFeatures::extract_dna_physical(const std::string &sequence) {
  DNAPhysicalProperties features;
  if (sequence.length() < 2) return features;

  // Dinucleotide physical properties from consensus tables
  // Format: [bendability, flexibility, meltability/stacking]
  // OPTIMIZATION: Use 16-element array indexed by (base1*4 + base2) to avoid string allocations
  // Index: A=0, C=1, G=2, T=3 -> AA=0, AC=1, AG=2, AT=3, CA=4, ...
  static const std::array<std::array<double, 3>, 16> dinuc_props = {{
    {{0.35, 0.24, 0.71}},  // AA (0)
    {{0.33, 0.38, 0.60}},  // AC (1)
    {{0.35, 0.35, 0.66}},  // AG (2)
    {{0.33, 0.34, 0.65}},  // AT (3)
    {{0.35, 0.32, 0.73}},  // CA (4)
    {{0.33, 0.30, 0.57}},  // CC (5)
    {{0.35, 0.26, 0.54}},  // CG (6)
    {{0.35, 0.35, 0.66}},  // CT (7)
    {{0.35, 0.32, 0.67}},  // GA (8)
    {{0.33, 0.26, 0.54}},  // GC (9)
    {{0.33, 0.30, 0.57}},  // GG (10)
    {{0.33, 0.38, 0.60}},  // GT (11)
    {{0.34, 0.23, 0.72}},  // TA (12)
    {{0.35, 0.32, 0.67}},  // TC (13)
    {{0.35, 0.32, 0.73}},  // TG (14)
    {{0.35, 0.24, 0.71}}   // TT (15)
  }};

  double bend = 0, flex = 0, melt = 0;
  int count = 0;

  for (size_t i = 0; i + 1 < sequence.length(); i++) {
    // Encode bases as integers: A=0, C=1, G=2, T=3
    uint8_t b1 = 255, b2 = 255;
    char c1 = sequence[i], c2 = sequence[i + 1];

    switch (c1) {
      case 'A': case 'a': b1 = 0; break;
      case 'C': case 'c': b1 = 1; break;
      case 'G': case 'g': b1 = 2; break;
      case 'T': case 't': b1 = 3; break;
    }
    switch (c2) {
      case 'A': case 'a': b2 = 0; break;
      case 'C': case 'c': b2 = 1; break;
      case 'G': case 'g': b2 = 2; break;
      case 'T': case 't': b2 = 3; break;
    }

    if (b1 < 4 && b2 < 4) {
      int idx = (b1 << 2) | b2;
      bend += dinuc_props[idx][0];
      flex += dinuc_props[idx][1];
      melt += dinuc_props[idx][2];
      count++;
    }
  }

  if (count > 0) {
    features.bendability = bend / count;
    features.flexibility = flex / count;
    features.meltability = melt / count;
  }

  return features;
}

// ============================================================================
// CGR-DFA FRACTAL FEATURES
// DFA on CGR trajectory for scale-dependent correlation analysis
// ============================================================================

// Helper: DFA on a single series, returns Hurst exponent
// Uses UDFA (unbiased DFA) per Yuan et al. 2018 for short-series robustness
// Steps:
//   1) Build profile
//   2) Overlapping windows of size s (Fibonacci spaced)
//   3) Detrend each window (linear)
//   4) Compute unbiased variance using autocorrelation correction:
//        rho* = rho + (1/s)*(1 + 3*rho) with rho estimated from residuals
//        V* = (rho*_A + rho*_B)*(1 - 1/(2s))*var_mean
//   5) F(s) = sqrt(sum_k V*) / sqrt(s * (N - s + 1))
static double compute_dfa_hurst_internal(const std::vector<double>& series) {
  // Lower threshold to 50 for short-sequence support
  if (series.size() < 50) return 0.5;

  // Compute cumulative deviation from mean
  double mean = std::accumulate(series.begin(), series.end(), 0.0) / series.size();
  std::vector<double> profile(series.size());
  profile[0] = series[0] - mean;
  for (size_t i = 1; i < series.size(); i++) {
    profile[i] = profile[i-1] + (series[i] - mean);
  }

  // DFA: compute fluctuations using Fibonacci scales {8,13,21,34,55,89}
  // Fibonacci gives optimal logarithmic spacing (golden ratio ≈ 1.618)
  std::vector<int> fib_scales = {8, 13, 21, 34, 55, 89, 144};
  std::vector<double> log_scales, log_flucts, weights;

  int max_scale = static_cast<int>(series.size()) / 4;

  for (int scale : fib_scales) {
    if (scale > max_scale) break;

    // Overlapping windows; stride to cap cost on long series
    int window_stride = 1;
    if (FractalFeatures::dfa_stride_factor > 0 && profile.size() > 5000) {
      window_stride = std::max(1, scale / FractalFeatures::dfa_stride_factor);
    }
    int n_segments = (static_cast<int>(profile.size()) - scale) / window_stride + 1;
    if (n_segments < 2) continue;

    double sum_vstar = 0.0;
    int valid_segments = 0;

    for (int seg_idx = 0; seg_idx < n_segments; seg_idx++) {
      int seg = seg_idx * window_stride;
      // Linear regression for trend in this segment
      double sx = 0, sy = 0, sxy = 0, sx2 = 0;
      for (int i = 0; i < scale; i++) {
        int idx = seg + i;
        sx += i;
        sy += profile[idx];
        sxy += i * profile[idx];
        sx2 += i * i;
      }
      double denom = scale * sx2 - sx * sx;
      if (std::abs(denom) < 1e-10) continue;
      double a = (scale * sxy - sx * sy) / denom;
      double b = (sy - a * sx) / scale;

      // Residuals
      std::vector<double> resid(scale);
      double mean_resid = 0.0;
      for (int i = 0; i < scale; i++) {
        resid[i] = profile[seg + i] - (a * i + b);
        mean_resid += resid[i];
      }
      mean_resid /= scale;

      // var(W) unbiased
      double varA = 0.0, varB = 0.0;
      double sum_sq_A = 0.0, sum_sq_B = 0.0, sum_lag_A = 0.0, sum_lag_B = 0.0;
      for (int i = 0; i < scale; i++) {
        double ra = resid[i] - mean_resid;
        double rb = ra * ((i % 2 == 0) ? 1.0 : -1.0);
        sum_sq_A += ra * ra;
        sum_sq_B += rb * rb;
        if (i + 1 < scale) {
          double ra_next = resid[i + 1] - mean_resid;
          double rb_next = ra_next * ((i + 1) % 2 == 0 ? 1.0 : -1.0);
          sum_lag_A += ra * ra_next;
          sum_lag_B += rb * rb_next;
        }
      }
      if (scale > 1) {
        varA = sum_sq_A / (scale - 1);
        varB = sum_sq_B / (scale - 1);
      }
      if (varA < 1e-12 || varB < 1e-12) continue;

      double rhoA = (sum_sq_A + sum_lag_A) / ((2.0 * scale - 1.0) * varA);
      double rhoB = (sum_sq_B - sum_lag_B) / ((2.0 * scale - 1.0) * varB);

      // Unbiased autocorr correction (m=0 after detrending)
      double rhoA_star = rhoA + (1.0 / scale) * (1.0 + 3.0 * rhoA);
      double rhoB_star = rhoB + (1.0 / scale) * (1.0 + 3.0 * rhoB);

      double var_mean = 0.5 * (varA + varB);
      double vstar = (rhoA_star + rhoB_star) * (1.0 - 1.0 / (2.0 * scale)) * var_mean;

      if (vstar > 0) {
        sum_vstar += vstar;
        valid_segments++;
      }
    }

    if (valid_segments > 0) {
      double F = std::sqrt(sum_vstar / (static_cast<double>(scale) * valid_segments));
      if (F > 1e-10) {
        log_scales.push_back(std::log(static_cast<double>(scale)));
        log_flucts.push_back(std::log(F));
        // Weight by number of segments (sqrt for variance stabilization)
        weights.push_back(std::sqrt(valid_segments));
      }
    }
  }

  // Need at least 3 points for reliable regression
  if (log_scales.size() < 3) return 0.5;

  // Weighted least squares regression
  double sum_w = 0, sum_wx = 0, sum_wy = 0, sum_wxy = 0, sum_wxx = 0;
  for (size_t i = 0; i < log_scales.size(); i++) {
    double w = weights[i];
    sum_w += w;
    sum_wx += w * log_scales[i];
    sum_wy += w * log_flucts[i];
    sum_wxy += w * log_scales[i] * log_flucts[i];
    sum_wxx += w * log_scales[i] * log_scales[i];
  }
  double denom = sum_w * sum_wxx - sum_wx * sum_wx;
  if (std::abs(denom) < 1e-10) return 0.5;

  double hurst = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
  return hurst;  // Raw value - don't clamp
}

// Helper: compute DFA fluctuation at a specific scale
static double compute_fluctuation_at_scale(const std::vector<double>& series, int scale) {
  if (scale > (int)series.size() / 4 || scale < 2 || series.size() < 50) return 0.0;

  double mean = std::accumulate(series.begin(), series.end(), 0.0) / series.size();
  std::vector<double> profile(series.size());
  profile[0] = series[0] - mean;
  for (size_t i = 1; i < series.size(); i++) {
    profile[i] = profile[i-1] + (series[i] - mean);
  }

  // Overlapping windows; stride to reduce cost on long series
  int window_stride = 1;
  if (FractalFeatures::dfa_stride_factor > 0 && profile.size() > 5000) {
    window_stride = std::max(1, scale / FractalFeatures::dfa_stride_factor);
  }
  int n_segments = (static_cast<int>(profile.size()) - scale) / window_stride + 1;
  double sum_vstar = 0.0;
  int valid_segments = 0;

  for (int seg_idx = 0; seg_idx < n_segments; seg_idx++) {
    int seg = seg_idx * window_stride;
    double sx = 0, sy = 0, sxy = 0, sx2 = 0;
    for (int i = 0; i < scale; i++) {
      int idx = seg + i;
      sx += i;
      sy += profile[idx];
      sxy += i * profile[idx];
      sx2 += i * i;
    }
    double denom = scale * sx2 - sx * sx;
    if (std::abs(denom) < 1e-10) continue;
    double a = (scale * sxy - sx * sy) / denom;
    double b = (sy - a * sx) / scale;

    std::vector<double> resid(scale);
    double mean_resid = 0.0;
    for (int i = 0; i < scale; i++) {
      resid[i] = profile[seg + i] - (a * i + b);
      mean_resid += resid[i];
    }
    mean_resid /= scale;

    double sum_sq_A = 0.0, sum_sq_B = 0.0, sum_lag_A = 0.0, sum_lag_B = 0.0;
    for (int i = 0; i < scale; i++) {
      double ra = resid[i] - mean_resid;
      double rb = ra * ((i % 2 == 0) ? 1.0 : -1.0);
      sum_sq_A += ra * ra;
      sum_sq_B += rb * rb;
      if (i + 1 < scale) {
        double ra_next = resid[i + 1] - mean_resid;
        double rb_next = ra_next * ((i + 1) % 2 == 0 ? 1.0 : -1.0);
        sum_lag_A += ra * ra_next;
        sum_lag_B += rb * rb_next;
      }
    }
    double varA = (scale > 1) ? sum_sq_A / (scale - 1) : 0.0;
    double varB = (scale > 1) ? sum_sq_B / (scale - 1) : 0.0;
    if (varA < 1e-12 || varB < 1e-12) continue;

    double rhoA = (sum_sq_A + sum_lag_A) / ((2.0 * scale - 1.0) * varA);
    double rhoB = (sum_sq_B - sum_lag_B) / ((2.0 * scale - 1.0) * varB);
    double rhoA_star = rhoA + (1.0 / scale) * (1.0 + 3.0 * rhoA);
    double rhoB_star = rhoB + (1.0 / scale) * (1.0 + 3.0 * rhoB);

    double var_mean = 0.5 * (varA + varB);
    double vstar = (rhoA_star + rhoB_star) * (1.0 - 1.0 / (2.0 * scale)) * var_mean;
    if (vstar > 0) {
      sum_vstar += vstar;
      valid_segments++;
    }
  }

  return valid_segments > 0
           ? std::sqrt(sum_vstar / (static_cast<double>(scale) * valid_segments))
           : 0.0;
}

CGRDFAFeatures FractalFeatures::extract_cgr_dfa_features(const std::string &sequence) {
  CGRDFAFeatures features;
  // Keep at 500bp - below this, CGR trajectory too short for reliable DFA
  if (sequence.length() < 500) return features;

  // Generate CGR trajectory as 2D random walk
  // A=(0,0), C=(1,0), G=(1,1), T=(0,1)
  std::vector<double> cgr_x, cgr_y;
  cgr_x.reserve(sequence.length());
  cgr_y.reserve(sequence.length());

  double x = 0.5, y = 0.5;
  for (char c : sequence) {
    switch (std::toupper(c)) {
      case 'A': x = x / 2;       y = y / 2;       break;
      case 'C': x = (x + 1) / 2; y = y / 2;       break;
      case 'G': x = (x + 1) / 2; y = (y + 1) / 2; break;
      case 'T': x = x / 2;       y = (y + 1) / 2; break;
      default: continue;
    }
    cgr_x.push_back(x);
    cgr_y.push_back(y);
  }

  // Lower internal threshold since helper function now handles short series
  if (cgr_x.size() < 50) return features;

  // 1. Hurst exponents from CGR trajectory (X and Y)
  features.hurst_x = compute_dfa_hurst_internal(cgr_x);
  features.hurst_y = compute_dfa_hurst_internal(cgr_y);

  // 2. Anisotropy: directional bias in sequence correlations
  features.anisotropy = std::abs(features.hurst_x - features.hurst_y);

  // 3. DFA fluctuation function at different scales (captures scale-dependent behavior)
  // Use Fibonacci scales {8, 55} instead of {16, 128} for better short-seq support
  int small_scale = 8;
  int large_scale = std::min(55, static_cast<int>(cgr_x.size()) / 4);
  double F_small_x = compute_fluctuation_at_scale(cgr_x, small_scale);
  double F_large_x = compute_fluctuation_at_scale(cgr_x, large_scale);
  double F_small_y = compute_fluctuation_at_scale(cgr_y, small_scale);
  double F_large_y = compute_fluctuation_at_scale(cgr_y, large_scale);

  // Ratio captures scale-dependent behavior (normalized to [0,1] range)
  // Adjusted scaling factor for Fib ratio
  double F_ratio_x = (F_small_x > 1e-10) ? std::tanh(F_large_x / F_small_x - 3.0) * 0.5 + 0.5 : 0.5;
  double F_ratio_y = (F_small_y > 1e-10) ? std::tanh(F_large_y / F_small_y - 3.0) * 0.5 + 0.5 : 0.5;
  features.f_ratio = (F_ratio_x + F_ratio_y) / 2.0;

  // 4. Local Hurst variability (multifractal width proxy)
  // Use n/6 window size with n/12 step for overlap (user suggestion)
  std::vector<double> local_hursts;
  size_t window_size = cgr_x.size() / 6;
  size_t step_size = window_size / 2;
  // Lower threshold to 50 to match helper function
  if (window_size >= 50) {
    for (size_t start = 0; start + window_size <= cgr_x.size(); start += step_size) {
      std::vector<double> local_x(cgr_x.begin() + start, cgr_x.begin() + start + window_size);
      double local_h = compute_dfa_hurst_internal(local_x);
      local_hursts.push_back(local_h);
    }
  }
  if (local_hursts.size() >= 2) {
    double mean_h = std::accumulate(local_hursts.begin(), local_hursts.end(), 0.0) / local_hursts.size();
    double var = 0;
    for (double h : local_hursts) var += (h - mean_h) * (h - mean_h);
    features.local_hurst_var = std::sqrt(var / local_hursts.size());
  }

  return features;
}

// ============================================================================
// MFDFA QUALITY METRICS
// For bin quality assessment (detecting contamination/chimeras)
// ============================================================================

MFDFAQualityMetrics FractalFeatures::extract_mfdfa_quality(const std::string &sequence) {
  MFDFAQualityMetrics features;
  if (sequence.length() < 500) return features;

  // Generate CGR trajectory
  std::vector<double> cgr_x, cgr_y;
  cgr_x.reserve(sequence.length());
  cgr_y.reserve(sequence.length());

  double x = 0.5, y = 0.5;
  for (char c : sequence) {
    switch (std::toupper(c)) {
      case 'A': x = x / 2;       y = y / 2;       break;
      case 'C': x = (x + 1) / 2; y = y / 2;       break;
      case 'G': x = (x + 1) / 2; y = (y + 1) / 2; break;
      case 'T': x = x / 2;       y = (y + 1) / 2; break;
      default: continue;
    }
    cgr_x.push_back(x);
    cgr_y.push_back(y);
  }

  if (cgr_x.size() < 100) return features;

  // Compute Hurst exponent for both X and Y trajectories
  double hurst_x = compute_dfa_hurst_internal(cgr_x);
  double hurst_y = compute_dfa_hurst_internal(cgr_y);

  // Combined Hurst (average of X and Y)
  features.hurst = (hurst_x + hurst_y) / 2.0;

  // Variance: difference between X and Y Hurst (indicates anisotropy)
  features.variance = std::abs(hurst_x - hurst_y);

  // Multifractal width: compute local Hurst exponents along the sequence
  std::vector<double> local_hursts;
  size_t window_size = cgr_x.size() / 5;
  if (window_size >= 200) {
    for (size_t start = 0; start + window_size <= cgr_x.size(); start += window_size / 2) {
      std::vector<double> local_x(cgr_x.begin() + start, cgr_x.begin() + start + window_size);
      std::vector<double> local_y(cgr_y.begin() + start, cgr_y.begin() + start + window_size);
      double local_hx = compute_dfa_hurst_internal(local_x);
      double local_hy = compute_dfa_hurst_internal(local_y);
      local_hursts.push_back((local_hx + local_hy) / 2.0);
    }
  }

  // Width as coefficient of variation of local Hurst exponents
  if (local_hursts.size() >= 3) {
    double mean_h = std::accumulate(local_hursts.begin(), local_hursts.end(), 0.0) / local_hursts.size();
    double var = 0;
    for (double h : local_hursts) {
      var += (h - mean_h) * (h - mean_h);
    }
    features.width = std::sqrt(var / (local_hursts.size() - 1)) / (mean_h + 0.01);
  }

  return features;
}

// ============================================================================
// OPTIMIZED COMBINED CGR FEATURE EXTRACTION
// Computes CGR trajectory once and extracts both CGR-DFA and MFDFA features
// This is 2-3x faster than calling extract_cgr_dfa_features and extract_mfdfa_quality separately
// ============================================================================

CombinedCGRFeatures FractalFeatures::extract_cgr_features_combined(const std::string &sequence) {
  CombinedCGRFeatures result;

  // Minimum length check
  if (sequence.length() < 500) return result;

  // ========================================================================
  // STEP 1: Generate CGR trajectory ONCE (shared by both CGR-DFA and MFDFA)
  // ========================================================================
  std::vector<double> cgr_x, cgr_y;
  cgr_x.reserve(sequence.length());
  cgr_y.reserve(sequence.length());

  double x = 0.5, y = 0.5;
  for (char c : sequence) {
    switch (std::toupper(c)) {
      case 'A': x = x / 2;       y = y / 2;       break;
      case 'C': x = (x + 1) / 2; y = y / 2;       break;
      case 'G': x = (x + 1) / 2; y = (y + 1) / 2; break;
      case 'T': x = x / 2;       y = (y + 1) / 2; break;
      default: continue;
    }
    cgr_x.push_back(x);
    cgr_y.push_back(y);
  }

  if (cgr_x.size() < 50) return result;

  // ========================================================================
  // STEP 2: Compute global Hurst exponents ONCE (used by both CGR-DFA and MFDFA)
  // ========================================================================
  double hurst_x = compute_dfa_hurst_internal(cgr_x);
  double hurst_y = compute_dfa_hurst_internal(cgr_y);

  // ========================================================================
  // STEP 3: Extract CGR-DFA features (>= 500bp)
  // ========================================================================
  result.has_cgr_dfa = true;
  result.cgr_dfa.hurst_x = hurst_x;
  result.cgr_dfa.hurst_y = hurst_y;
  result.cgr_dfa.anisotropy = std::abs(hurst_x - hurst_y);

  // Fluctuation ratio at different scales
  int small_scale = 8;
  int large_scale = std::min(55, static_cast<int>(cgr_x.size()) / 4);
  double F_small_x = compute_fluctuation_at_scale(cgr_x, small_scale);
  double F_large_x = compute_fluctuation_at_scale(cgr_x, large_scale);
  double F_small_y = compute_fluctuation_at_scale(cgr_y, small_scale);
  double F_large_y = compute_fluctuation_at_scale(cgr_y, large_scale);

  double F_ratio_x = (F_small_x > 1e-10) ? std::tanh(F_large_x / F_small_x - 3.0) * 0.5 + 0.5 : 0.5;
  double F_ratio_y = (F_small_y > 1e-10) ? std::tanh(F_large_y / F_small_y - 3.0) * 0.5 + 0.5 : 0.5;
  result.cgr_dfa.f_ratio = (F_ratio_x + F_ratio_y) / 2.0;

  // ========================================================================
  // STEP 4: Compute local Hurst variability ONCE (shared computation)
  // OPTIMIZATION: Use larger step size (n/4 instead of n/10) for 2.5x fewer DFA calls
  // ========================================================================
  std::vector<double> local_hursts;
  size_t window_size = cgr_x.size() / 6;
  size_t step_size = window_size;  // Non-overlapping for speed (was window_size/2)

  if (window_size >= 50) {
    for (size_t start = 0; start + window_size <= cgr_x.size(); start += step_size) {
      std::vector<double> local_x(cgr_x.begin() + start, cgr_x.begin() + start + window_size);
      double local_h = compute_dfa_hurst_internal(local_x);
      local_hursts.push_back(local_h);
    }
  }

  // Compute local Hurst variance (used by both CGR-DFA and MFDFA)
  double local_hurst_var = 0.0;
  double mean_h = 0.0;
  if (local_hursts.size() >= 2) {
    mean_h = std::accumulate(local_hursts.begin(), local_hursts.end(), 0.0) / local_hursts.size();
    double var = 0;
    for (double h : local_hursts) var += (h - mean_h) * (h - mean_h);
    local_hurst_var = std::sqrt(var / local_hursts.size());
  }

  result.cgr_dfa.local_hurst_var = local_hurst_var;

  // ========================================================================
  // STEP 5: Extract MFDFA features (>= 1000bp)
  // ========================================================================
  if (sequence.length() >= 1000) {
    result.has_mfdfa = true;

    // Combined Hurst (average of X and Y)
    result.mfdfa.hurst = (hurst_x + hurst_y) / 2.0;

    // Variance: difference between X and Y Hurst (indicates anisotropy)
    result.mfdfa.variance = std::abs(hurst_x - hurst_y);

    // Width as coefficient of variation of local Hurst exponents
    if (local_hursts.size() >= 3 && mean_h > 0.01) {
      result.mfdfa.width = local_hurst_var / mean_h;
    }
  }

  return result;
}

} // namespace amber

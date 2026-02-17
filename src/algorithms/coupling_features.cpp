#include "coupling_features.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace amber {

// =============================================================================
// Static Member Definitions
// =============================================================================

std::vector<std::vector<double>> CouplingExtractor::pca_basis_;
std::vector<double> CouplingExtractor::pca_means_;
bool CouplingExtractor::basis_fitted_ = false;

// TNF dimension (136 canonical tetranucleotides)
constexpr int TNF_DIM = 136;

// =============================================================================
// PCA Basis Accessors
// =============================================================================

bool CouplingExtractor::is_basis_ready() {
    return basis_fitted_;
}

const std::vector<std::vector<double>>& CouplingExtractor::get_pca_basis() {
    return pca_basis_;
}

const std::vector<double>& CouplingExtractor::get_pca_means() {
    return pca_means_;
}

// =============================================================================
// Window Parameter Computation
// =============================================================================

WindowParams CouplingExtractor::compute_window_params(size_t contig_length) {
    WindowParams params;

    // Window size: clamp(length/18, 1200, 2500)
    int w = static_cast<int>(contig_length) / 18;
    params.w_bp = std::clamp(w, WindowParams::MIN_WINDOW_BP, WindowParams::MAX_WINDOW_BP);

    // Stride: 75% of window size (25% overlap)
    params.stride_bp = static_cast<int>(0.75 * params.w_bp);

    // Number of windows
    if (contig_length < static_cast<size_t>(params.w_bp)) {
        params.n_windows = 0;
    } else {
        params.n_windows = 1 + static_cast<int>((contig_length - params.w_bp) / params.stride_bp);
    }

    return params;
}

// =============================================================================
// Local TNF Computation (136-dim)
// =============================================================================

// Canonical k-mer mapping (handles reverse complement)
namespace {

// Nucleotide to 2-bit encoding: A=0, C=1, G=2, T=3
inline int base_to_int(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;  // Invalid
    }
}

// Reverse complement of a 4-mer encoded as int (0-255)
inline int reverse_complement_4mer(int kmer) {
    // Each base is 2 bits, reverse and complement
    int rc = 0;
    for (int i = 0; i < 4; i++) {
        int base = (kmer >> (2 * i)) & 3;
        int comp = 3 - base;  // A<->T, C<->G
        rc = (rc << 2) | comp;
    }
    return rc;
}

// Get canonical form (min of kmer and its reverse complement)
inline int canonical_4mer(int kmer) {
    int rc = reverse_complement_4mer(kmer);
    return std::min(kmer, rc);
}

// Build mapping from 256 4-mers to 136 canonical indices
// Returns -1 for redundant (non-canonical) k-mers
std::vector<int> build_canonical_mapping() {
    std::vector<int> mapping(256, -1);
    int idx = 0;
    for (int kmer = 0; kmer < 256; kmer++) {
        if (canonical_4mer(kmer) == kmer) {
            mapping[kmer] = idx++;
        }
    }
    return mapping;
}

// Singleton mapping
const std::vector<int>& get_canonical_mapping() {
    static std::vector<int> mapping = build_canonical_mapping();
    return mapping;
}

}  // anonymous namespace

std::vector<double> CouplingExtractor::compute_local_tnf(
    const std::string& sequence, size_t start, size_t end) {

    std::vector<double> tnf(TNF_DIM, 0.0);
    const auto& mapping = get_canonical_mapping();

    if (end <= start || end - start < 4) {
        return tnf;
    }

    int count = 0;
    for (size_t i = start; i + 3 < end; i++) {
        // Encode 4-mer
        int kmer = 0;
        bool valid = true;
        for (int j = 0; j < 4; j++) {
            int b = base_to_int(sequence[i + j]);
            if (b < 0) {
                valid = false;
                break;
            }
            kmer = (kmer << 2) | b;
        }
        if (!valid) continue;

        // Get canonical index
        int canonical = canonical_4mer(kmer);
        int idx = mapping[canonical];
        if (idx >= 0 && idx < TNF_DIM) {
            tnf[idx] += 1.0;
            count++;
        }
    }

    // Normalize to frequencies
    if (count > 0) {
        for (double& f : tnf) {
            f /= count;
        }
    }

    return tnf;
}

// =============================================================================
// PCA Basis Fitting (Power Iteration)
// =============================================================================

void CouplingExtractor::fit_pca_basis(
    const std::vector<std::vector<double>>& all_tnf,
    int n_components) {

    if (all_tnf.empty() || all_tnf[0].size() != TNF_DIM) {
        basis_fitted_ = false;
        return;
    }

    size_t n = all_tnf.size();

    // Step 1: Compute mean
    pca_means_.assign(TNF_DIM, 0.0);
    for (const auto& tnf : all_tnf) {
        for (int i = 0; i < TNF_DIM; i++) {
            pca_means_[i] += tnf[i];
        }
    }
    for (double& m : pca_means_) {
        m /= n;
    }

    // Step 2: Center data
    std::vector<std::vector<double>> centered(n, std::vector<double>(TNF_DIM));
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < TNF_DIM; j++) {
            centered[i][j] = all_tnf[i][j] - pca_means_[j];
        }
    }

    // Step 3: Power iteration for top n_components eigenvectors
    pca_basis_.clear();
    pca_basis_.reserve(n_components);

    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int comp = 0; comp < n_components; comp++) {
        // Initialize random vector
        std::vector<double> v(TNF_DIM);
        for (double& x : v) {
            x = dist(rng);
        }

        // Normalize
        double norm = 0.0;
        for (double x : v) norm += x * x;
        norm = std::sqrt(norm);
        for (double& x : v) x /= norm;

        // Power iteration (50 iterations is typically sufficient)
        for (int iter = 0; iter < 50; iter++) {
            // Compute X^T * X * v
            // First: u = X * v (n-dim)
            std::vector<double> u(n, 0.0);
            for (size_t i = 0; i < n; i++) {
                for (int j = 0; j < TNF_DIM; j++) {
                    u[i] += centered[i][j] * v[j];
                }
            }

            // Then: w = X^T * u (TNF_DIM-dim)
            std::vector<double> w(TNF_DIM, 0.0);
            for (size_t i = 0; i < n; i++) {
                for (int j = 0; j < TNF_DIM; j++) {
                    w[j] += centered[i][j] * u[i];
                }
            }

            // Deflate: remove projections onto previous components
            for (const auto& prev : pca_basis_) {
                double proj = 0.0;
                for (int j = 0; j < TNF_DIM; j++) {
                    proj += w[j] * prev[j];
                }
                for (int j = 0; j < TNF_DIM; j++) {
                    w[j] -= proj * prev[j];
                }
            }

            // Normalize
            norm = 0.0;
            for (double x : w) norm += x * x;
            norm = std::sqrt(norm);
            if (norm < 1e-10) break;
            for (double& x : w) x /= norm;

            v = w;
        }

        pca_basis_.push_back(v);
    }

    basis_fitted_ = true;
}

// =============================================================================
// Project Local TNF onto PCA Basis
// =============================================================================

std::array<double, 3> CouplingExtractor::project_local_tnf(
    const std::string& sequence, size_t start, size_t end) {

    std::array<double, 3> projection = {0.0, 0.0, 0.0};

    if (!basis_fitted_ || pca_basis_.size() < 3) {
        return projection;
    }

    // Compute local TNF
    auto tnf = compute_local_tnf(sequence, start, end);

    // Center using global means
    for (int i = 0; i < TNF_DIM; i++) {
        tnf[i] -= pca_means_[i];
    }

    // Project onto each PCA component
    for (int c = 0; c < 3 && c < static_cast<int>(pca_basis_.size()); c++) {
        double proj = 0.0;
        for (int i = 0; i < TNF_DIM; i++) {
            proj += tnf[i] * pca_basis_[c][i];
        }
        projection[c] = proj;
    }

    return projection;
}

// =============================================================================
// Compute Window Signals
// =============================================================================

std::vector<WindowSignal> CouplingExtractor::compute_window_signals(
    const std::string& sequence,
    const std::vector<uint32_t>& per_base_coverage,
    const WindowParams& params) {

    std::vector<WindowSignal> signals;

    if (params.n_windows < WindowParams::MIN_WINDOWS) {
        return signals;
    }

    signals.reserve(params.n_windows);

    for (int w = 0; w < params.n_windows; w++) {
        size_t start = w * params.stride_bp;
        size_t end = start + params.w_bp;
        if (end > sequence.size()) end = sequence.size();

        WindowSignal sig;

        // Compute local TNF and project
        sig.comp_pca = project_local_tnf(sequence, start, end);

        // Compute mean coverage in window
        if (!per_base_coverage.empty()) {
            size_t cov_start = std::min(start, per_base_coverage.size());
            size_t cov_end = std::min(end, per_base_coverage.size());
            if (cov_end > cov_start) {
                double sum = 0.0;
                for (size_t i = cov_start; i < cov_end; i++) {
                    sum += per_base_coverage[i];
                }
                sig.coverage = sum / (cov_end - cov_start);
            }
        }

        signals.push_back(sig);
    }

    return signals;
}

// =============================================================================
// Kernel Dependence (HSIC, dCor, MIC)
// =============================================================================

KernelDependence CouplingExtractor::compute_kernel_dependence(
    const std::vector<WindowSignal>& signals) {

    KernelDependence kd;
    size_t n = signals.size();

    if (n < WindowParams::MIN_WINDOWS) {
        return kd;
    }

    // Extract composition (3-dim) and coverage (scalar) from signals
    std::vector<std::array<double, 3>> comp(n);
    std::vector<double> cov(n);
    for (size_t i = 0; i < n; i++) {
        comp[i] = signals[i].comp_pca;
        cov[i] = signals[i].coverage;
    }

    // =========================================================================
    // Compute pairwise distance matrices
    // =========================================================================

    // Composition: Euclidean distance in 3-dim PCA space
    std::vector<std::vector<double>> dist_comp(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            double d = 0.0;
            for (int k = 0; k < 3; k++) {
                double diff = comp[i][k] - comp[j][k];
                d += diff * diff;
            }
            d = std::sqrt(d);
            dist_comp[i][j] = d;
            dist_comp[j][i] = d;
        }
    }

    // Coverage: absolute difference
    std::vector<std::vector<double>> dist_cov(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            double d = std::abs(cov[i] - cov[j]);
            dist_cov[i][j] = d;
            dist_cov[j][i] = d;
        }
    }

    // =========================================================================
    // HSIC: Hilbert-Schmidt Independence Criterion
    // =========================================================================

    // Median heuristic for bandwidth
    std::vector<double> all_dist_comp, all_dist_cov;
    all_dist_comp.reserve(n * (n - 1) / 2);
    all_dist_cov.reserve(n * (n - 1) / 2);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            all_dist_comp.push_back(dist_comp[i][j]);
            all_dist_cov.push_back(dist_cov[i][j]);
        }
    }
    std::sort(all_dist_comp.begin(), all_dist_comp.end());
    std::sort(all_dist_cov.begin(), all_dist_cov.end());

    double sigma_comp = all_dist_comp.empty() ? 1.0 : all_dist_comp[all_dist_comp.size() / 2];
    double sigma_cov = all_dist_cov.empty() ? 1.0 : all_dist_cov[all_dist_cov.size() / 2];
    sigma_comp = std::max(sigma_comp, 1e-9);
    sigma_cov = std::max(sigma_cov, 1e-9);

    // Build Gaussian kernel matrices
    std::vector<std::vector<double>> K_comp(n, std::vector<double>(n));
    std::vector<std::vector<double>> K_cov(n, std::vector<double>(n));
    for (size_t i = 0; i < n; i++) {
        K_comp[i][i] = 1.0;
        K_cov[i][i] = 1.0;
        for (size_t j = i + 1; j < n; j++) {
            double k_c = std::exp(-dist_comp[i][j] * dist_comp[i][j] / (2.0 * sigma_comp * sigma_comp));
            double k_v = std::exp(-dist_cov[i][j] * dist_cov[i][j] / (2.0 * sigma_cov * sigma_cov));
            K_comp[i][j] = k_c;
            K_comp[j][i] = k_c;
            K_cov[i][j] = k_v;
            K_cov[j][i] = k_v;
        }
    }

    // Center the kernels: K_centered = H * K * H where H = I - 1/n * ones
    auto center_kernel = [n](std::vector<std::vector<double>>& K) {
        std::vector<double> row_mean(n, 0.0);
        double grand_mean = 0.0;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                row_mean[i] += K[i][j];
            }
            row_mean[i] /= n;
            grand_mean += row_mean[i];
        }
        grand_mean /= n;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                K[i][j] = K[i][j] - row_mean[i] - row_mean[j] + grand_mean;
            }
        }
    };

    center_kernel(K_comp);
    center_kernel(K_cov);

    // HSIC = trace(K_comp * K_cov) / (n-1)^2
    double hsic_xy = 0.0, hsic_xx = 0.0, hsic_yy = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            hsic_xy += K_comp[i][j] * K_cov[j][i];
            hsic_xx += K_comp[i][j] * K_comp[j][i];
            hsic_yy += K_cov[i][j] * K_cov[j][i];
        }
    }
    double denom = (n - 1) * (n - 1);
    hsic_xy /= denom;
    hsic_xx /= denom;
    hsic_yy /= denom;

    double norm_factor = std::sqrt(hsic_xx * hsic_yy);
    kd.hsic_normalized = (norm_factor > 1e-9) ? hsic_xy / norm_factor : 0.0;

    // =========================================================================
    // Distance Correlation (dCor)
    // =========================================================================

    // Double-center distance matrices
    auto double_center = [n](const std::vector<std::vector<double>>& D) {
        std::vector<std::vector<double>> A(n, std::vector<double>(n));
        std::vector<double> row_mean(n, 0.0);
        double grand_mean = 0.0;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                row_mean[i] += D[i][j];
            }
            row_mean[i] /= n;
            grand_mean += row_mean[i];
        }
        grand_mean /= n;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                A[i][j] = D[i][j] - row_mean[i] - row_mean[j] + grand_mean;
            }
        }
        return A;
    };

    auto A = double_center(dist_comp);
    auto B = double_center(dist_cov);

    // dCov^2 = (1/n^2) * sum(A_ij * B_ij)
    double dcov2 = 0.0, dvar_a = 0.0, dvar_b = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            dcov2 += A[i][j] * B[i][j];
            dvar_a += A[i][j] * A[i][j];
            dvar_b += B[i][j] * B[i][j];
        }
    }
    double n2 = n * n;
    dcov2 /= n2;
    dvar_a /= n2;
    dvar_b /= n2;

    double dvar_prod = dvar_a * dvar_b;
    kd.dcor = (dvar_prod > 1e-18) ? std::sqrt(std::max(0.0, dcov2) / std::sqrt(dvar_prod)) : 0.0;

    // =========================================================================
    // MIC Approximation (binned mutual information)
    // =========================================================================

    // Use 2D histogram with ~sqrt(n) bins per dimension
    int n_bins = std::max(3, static_cast<int>(std::sqrt(static_cast<double>(n))));

    // Find ranges
    double cov_min = cov[0], cov_max = cov[0];
    std::array<double, 3> comp_min = comp[0], comp_max = comp[0];
    for (size_t i = 1; i < n; i++) {
        cov_min = std::min(cov_min, cov[i]);
        cov_max = std::max(cov_max, cov[i]);
        for (int k = 0; k < 3; k++) {
            comp_min[k] = std::min(comp_min[k], comp[i][k]);
            comp_max[k] = std::max(comp_max[k], comp[i][k]);
        }
    }

    // Use first PCA component for MI estimation (dominant variance direction)
    double comp_range = comp_max[0] - comp_min[0] + 1e-9;
    double cov_range = cov_max - cov_min + 1e-9;

    // Build 2D histogram
    std::vector<std::vector<int>> hist(n_bins, std::vector<int>(n_bins, 0));
    std::vector<int> hist_comp(n_bins, 0), hist_cov(n_bins, 0);

    for (size_t i = 0; i < n; i++) {
        int bin_comp = std::min(n_bins - 1, static_cast<int>((comp[i][0] - comp_min[0]) / comp_range * n_bins));
        int bin_cov = std::min(n_bins - 1, static_cast<int>((cov[i] - cov_min) / cov_range * n_bins));
        hist[bin_comp][bin_cov]++;
        hist_comp[bin_comp]++;
        hist_cov[bin_cov]++;
    }

    // Compute MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    double mi = 0.0;
    double h_comp = 0.0, h_cov = 0.0;
    for (int i = 0; i < n_bins; i++) {
        if (hist_comp[i] > 0) {
            double p = static_cast<double>(hist_comp[i]) / n;
            h_comp -= p * std::log(p);
        }
        if (hist_cov[i] > 0) {
            double p = static_cast<double>(hist_cov[i]) / n;
            h_cov -= p * std::log(p);
        }
        for (int j = 0; j < n_bins; j++) {
            if (hist[i][j] > 0 && hist_comp[i] > 0 && hist_cov[j] > 0) {
                double p_xy = static_cast<double>(hist[i][j]) / n;
                double p_x = static_cast<double>(hist_comp[i]) / n;
                double p_y = static_cast<double>(hist_cov[j]) / n;
                mi += p_xy * std::log(p_xy / (p_x * p_y));
            }
        }
    }

    // Normalize MI to [0,1]: MIC_approx = MI / min(H_X, H_Y)
    double h_min = std::min(h_comp, h_cov);
    kd.mic_approx = (h_min > 1e-9) ? std::min(1.0, mi / h_min) : 0.0;

    // CMI residual: placeholder
    kd.cmi_residual = 0.0;

    return kd;
}

// =============================================================================
// Coverage Transport
// =============================================================================

CoverageTransport CouplingExtractor::compute_coverage_transport(
    const std::vector<WindowSignal>& signals) {

    CoverageTransport ct;
    size_t n = signals.size();

    if (n < 6) {
        return ct;
    }

    // Step 1: Create sorted indices by coverage
    std::vector<size_t> sorted_idx(n);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
        [&signals](size_t a, size_t b) {
            return signals[a].coverage < signals[b].coverage;
        });

    // Step 2: Partition into 3 strata (terciles)
    size_t tercile = n / 3;
    size_t low_end = tercile;
    size_t mid_end = 2 * tercile;

    // Check minimum stratum size
    bool use_binary = (low_end < 2 || (mid_end - low_end) < 2 || (n - mid_end) < 2);

    if (use_binary) {
        // Binary split: low = first half, high = second half
        low_end = n / 2;
        mid_end = low_end;  // No mid stratum
    }

    // Collect composition vectors for each stratum
    std::vector<std::array<double, 3>> low_comp, mid_comp, high_comp;
    low_comp.reserve(low_end);
    high_comp.reserve(n - mid_end);

    for (size_t i = 0; i < low_end; i++) {
        low_comp.push_back(signals[sorted_idx[i]].comp_pca);
    }
    for (size_t i = mid_end; i < n; i++) {
        high_comp.push_back(signals[sorted_idx[i]].comp_pca);
    }
    if (!use_binary) {
        mid_comp.reserve(mid_end - low_end);
        for (size_t i = low_end; i < mid_end; i++) {
            mid_comp.push_back(signals[sorted_idx[i]].comp_pca);
        }
    }

    // Step 3: Compute 1-Wasserstein (per-coordinate sorted sum)
    auto compute_w1 = [](const std::vector<std::array<double, 3>>& a,
                         const std::vector<std::array<double, 3>>& b) -> double {
        if (a.empty() || b.empty()) return 0.0;

        double w1 = 0.0;
        for (int dim = 0; dim < 3; dim++) {
            std::vector<double> va, vb;
            va.reserve(a.size());
            vb.reserve(b.size());
            for (const auto& x : a) va.push_back(x[dim]);
            for (const auto& x : b) vb.push_back(x[dim]);
            std::sort(va.begin(), va.end());
            std::sort(vb.begin(), vb.end());

            // Interpolate to same size (use larger)
            size_t m = std::max(va.size(), vb.size());
            double sum = 0.0;
            for (size_t i = 0; i < m; i++) {
                double fa = static_cast<double>(i) / m * va.size();
                double fb = static_cast<double>(i) / m * vb.size();
                size_t ia = std::min(static_cast<size_t>(fa), va.size() - 1);
                size_t ib = std::min(static_cast<size_t>(fb), vb.size() - 1);
                sum += std::abs(va[ia] - vb[ib]);
            }
            w1 += sum / m;
        }
        return w1;
    };

    ct.wasserstein_low_high = compute_w1(low_comp, high_comp);
    ct.wasserstein_mid_high = use_binary ? 0.0 : compute_w1(mid_comp, high_comp);

    // Step 4: Centroid shift (L2 norm of centroid difference)
    auto compute_centroid = [](const std::vector<std::array<double, 3>>& comp)
        -> std::array<double, 3> {
        std::array<double, 3> c = {0.0, 0.0, 0.0};
        if (comp.empty()) return c;
        for (const auto& x : comp) {
            for (int d = 0; d < 3; d++) c[d] += x[d];
        }
        for (int d = 0; d < 3; d++) c[d] /= comp.size();
        return c;
    };

    auto centroid_low = compute_centroid(low_comp);
    auto centroid_high = compute_centroid(high_comp);

    double shift_sq = 0.0;
    for (int d = 0; d < 3; d++) {
        double diff = centroid_high[d] - centroid_low[d];
        shift_sq += diff * diff;
    }
    ct.centroid_shift_norm = std::sqrt(shift_sq);

    // Step 5: Dispersion ratio (trace of covariance)
    auto compute_trace_cov = [](const std::vector<std::array<double, 3>>& comp,
                                const std::array<double, 3>& centroid) -> double {
        if (comp.size() < 2) return 0.0;
        double trace = 0.0;
        for (int d = 0; d < 3; d++) {
            double var = 0.0;
            for (const auto& x : comp) {
                double diff = x[d] - centroid[d];
                var += diff * diff;
            }
            trace += var / (comp.size() - 1);
        }
        return trace;
    };

    double trace_low = compute_trace_cov(low_comp, centroid_low);
    double trace_high = compute_trace_cov(high_comp, centroid_high);
    ct.dispersion_ratio = trace_high / (trace_low + 1e-9);

    // Step 6: Stratum entropy
    if (use_binary) {
        double p_low = static_cast<double>(low_end) / n;
        double p_high = static_cast<double>(n - low_end) / n;
        ct.stratum_entropy = -(p_low * std::log(p_low + 1e-9) +
                               p_high * std::log(p_high + 1e-9));
    } else {
        double p_low = static_cast<double>(low_end) / n;
        double p_mid = static_cast<double>(mid_end - low_end) / n;
        double p_high = static_cast<double>(n - mid_end) / n;
        ct.stratum_entropy = -(p_low * std::log(p_low + 1e-9) +
                               p_mid * std::log(p_mid + 1e-9) +
                               p_high * std::log(p_high + 1e-9));
    }

    return ct;
}

// =============================================================================
// Cross-Wavelet Coherence
// =============================================================================

namespace {

// Haar wavelet detail coefficients at a given scale
// scale=0: pairs (2 windows), scale=1: groups of 4, scale=2: groups of 8
std::vector<double> haar_detail_coeffs(const std::vector<double>& signal, int scale) {
    std::vector<double> details;
    int block_size = 1 << (scale + 1);  // 2, 4, 8 for scales 0, 1, 2
    int half = block_size / 2;

    size_t n = signal.size();
    if (n < static_cast<size_t>(block_size)) {
        return details;
    }

    // Compute detail coefficients: difference between first and second half averages
    for (size_t i = 0; i + block_size <= n; i += block_size) {
        double avg_first = 0.0, avg_second = 0.0;
        for (int j = 0; j < half; j++) {
            avg_first += signal[i + j];
            avg_second += signal[i + half + j];
        }
        avg_first /= half;
        avg_second /= half;
        details.push_back(avg_first - avg_second);
    }

    return details;
}

// Detrend signal by subtracting mean
std::vector<double> detrend_mean(const std::vector<double>& signal) {
    if (signal.empty()) return signal;

    double mean = 0.0;
    for (double v : signal) mean += v;
    mean /= signal.size();

    std::vector<double> detrended(signal.size());
    for (size_t i = 0; i < signal.size(); i++) {
        detrended[i] = signal[i] - mean;
    }
    return detrended;
}

}  // anonymous namespace

CrossWaveletCoherence CouplingExtractor::compute_cross_wavelet(
    const std::vector<WindowSignal>& signals) {

    CrossWaveletCoherence cwc;

    size_t n = signals.size();
    if (n < WindowParams::MIN_WINDOWS) {
        return cwc;
    }

    // Step 1: Create 1-D signals
    // comp_signal = L2 norm of comp_pca (scalar measure of composition deviation)
    // cov_signal = coverage
    std::vector<double> comp_signal(n), cov_signal(n);

    for (size_t i = 0; i < n; i++) {
        const auto& pca = signals[i].comp_pca;
        comp_signal[i] = std::sqrt(pca[0]*pca[0] + pca[1]*pca[1] + pca[2]*pca[2]);
        cov_signal[i] = signals[i].coverage;
    }

    // Step 2: Detrend both signals (subtract mean)
    comp_signal = detrend_mean(comp_signal);
    cov_signal = detrend_mean(cov_signal);

    // Step 3: Compute cross-wavelet coherence at 3 scales
    // Scale 0: pairs (2 windows) - requires n >= 2
    // Scale 1: groups of 4 - requires n >= 4
    // Scale 2: groups of 8 - requires n >= 8

    for (int scale = 0; scale < 3; scale++) {
        int block_size = 1 << (scale + 1);  // 2, 4, 8

        if (n < static_cast<size_t>(block_size)) {
            // Not enough windows for this scale - leave as 0
            continue;
        }

        // Get detail coefficients at this scale
        auto d_comp = haar_detail_coeffs(comp_signal, scale);
        auto d_cov = haar_detail_coeffs(cov_signal, scale);

        if (d_comp.empty() || d_cov.empty()) {
            continue;
        }

        size_t m = std::min(d_comp.size(), d_cov.size());

        // Compute cross-power and auto-powers
        double P_xy = 0.0;  // Cross-power
        double P_xx = 0.0;  // Auto-power of composition
        double P_yy = 0.0;  // Auto-power of coverage

        for (size_t i = 0; i < m; i++) {
            P_xy += d_comp[i] * d_cov[i];
            P_xx += d_comp[i] * d_comp[i];
            P_yy += d_cov[i] * d_cov[i];
        }
        P_xy /= m;
        P_xx /= m;
        P_yy /= m;

        // Coherence: squared correlation in wavelet domain
        // coherence = P_xy^2 / (P_xx * P_yy)
        double denom = P_xx * P_yy + 1e-9;
        cwc.coherence[scale] = (P_xy * P_xy) / denom;

        // Phase: for real signals, approximate using sign of cross-product
        // Normalized to [-1, 1] (positive = in-phase, negative = anti-phase)
        double sign = (P_xy >= 0) ? 1.0 : -1.0;
        cwc.phase[scale] = sign * std::sqrt(cwc.coherence[scale]);

        // Gain: amplitude transfer from composition to coverage
        // gain = |P_xy| / sqrt(P_xx)
        cwc.gain[scale] = std::abs(P_xy) / (std::sqrt(P_xx) + 1e-9);
    }

    return cwc;
}

// =============================================================================
// Main Feature Extraction
// =============================================================================

CouplingFeatures CouplingExtractor::extract_coupling_features(
    const std::string& sequence,
    const std::vector<uint32_t>& per_base_coverage,
    double damage_rate_5p) {

    CouplingFeatures features;
    (void)damage_rate_5p;  // Reserved for future use

    // Check minimum contig length
    if (sequence.size() < static_cast<size_t>(WindowParams::MIN_CONTIG_BP)) {
        return features;
    }

    // Check if PCA basis is ready
    if (!is_basis_ready()) {
        return features;
    }

    // Compute window parameters
    WindowParams params = compute_window_params(sequence.size());

    if (params.n_windows < WindowParams::MIN_WINDOWS) {
        return features;
    }

    // Compute window signals
    auto signals = compute_window_signals(sequence, per_base_coverage, params);

    if (signals.size() < static_cast<size_t>(WindowParams::MIN_WINDOWS)) {
        return features;
    }

    // Compute feature groups
    features.kernel = compute_kernel_dependence(signals);
    features.transport = compute_coverage_transport(signals);
    features.wavelet = compute_cross_wavelet(signals);

    return features;
}

}  // namespace amber

#pragma once
// ==============================================================================
// Coupling Features: Composition-Coverage Dependence Analysis
//
// These features capture the statistical dependence between local composition
// (TNF projected to 3-dim PCA) and coverage along a contig. Strong coupling
// suggests the contig is homogeneous (single genome), while weak or anomalous
// coupling may indicate chimerism or contamination.
//
// Features:
//   1. Kernel Dependence (4 dims): HSIC, dCor, MIC approx, CMI residual
//   2. Coverage Transport (5 dims): Wasserstein, centroid shift, dispersion, entropy
//   3. Cross-Wavelet Coherence (9 dims): coherence, phase, gain at 3 scales
//
// Total: 18 dimensions
// ==============================================================================

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace amber {

// =============================================================================
// Window Parameters
// =============================================================================

struct WindowParams {
    int w_bp;           // Window size: clamp(contig_length/18, 1200, 2500)
    int stride_bp;      // stride = 0.75 * w_bp
    int n_windows;      // Number of windows that fit

    static constexpr int MIN_CONTIG_BP = 8000;
    static constexpr int MIN_WINDOWS = 6;
    static constexpr int MIN_WINDOW_BP = 1200;
    static constexpr int MAX_WINDOW_BP = 2500;
};

// =============================================================================
// Window Signal (per-window composition + coverage)
// =============================================================================

struct WindowSignal {
    std::array<double, 3> comp_pca;  // Local TNF projected to 3-dim PCA
    double coverage;                  // Mean per-base coverage in window

    WindowSignal() : coverage(0.0) {
        comp_pca.fill(0.0);
    }
};

// =============================================================================
// Kernel Dependence Features (4 dimensions)
// =============================================================================

struct KernelDependence {
    double hsic_normalized = 0.0;  // HSIC / sqrt(HSIC_XX * HSIC_YY)
    double dcor = 0.0;             // Distance correlation
    double mic_approx = 0.0;       // MIC approximation
    double cmi_residual = 0.0;     // Conditional MI residual

    std::vector<double> to_vector() const {
        return {hsic_normalized, dcor, mic_approx, cmi_residual};
    }

    static constexpr int DIMS = 4;
};

// =============================================================================
// Coverage Transport Features (5 dimensions)
// =============================================================================

struct CoverageTransport {
    double wasserstein_low_high = 0.0;   // W1 between low/high coverage strata
    double wasserstein_mid_high = 0.0;   // W1 between mid/high coverage strata
    double centroid_shift_norm = 0.0;    // L2 norm of centroid shift
    double dispersion_ratio = 0.0;       // Dispersion high/low ratio
    double stratum_entropy = 0.0;        // Entropy of stratum sizes

    std::vector<double> to_vector() const {
        return {wasserstein_low_high, wasserstein_mid_high,
                centroid_shift_norm, dispersion_ratio, stratum_entropy};
    }

    static constexpr int DIMS = 5;
};

// =============================================================================
// Cross-Wavelet Coherence Features (9 dimensions)
// =============================================================================

struct CrossWaveletCoherence {
    std::array<double, 3> coherence = {};  // Coherence at 3 scales
    std::array<double, 3> phase = {};      // Phase at 3 scales
    std::array<double, 3> gain = {};       // Gain at 3 scales

    std::vector<double> to_vector() const {
        std::vector<double> v(9);
        for (int i = 0; i < 3; i++) v[i] = coherence[i];
        for (int i = 0; i < 3; i++) v[3 + i] = phase[i];
        for (int i = 0; i < 3; i++) v[6 + i] = gain[i];
        return v;
    }

    static constexpr int DIMS = 9;
};

// =============================================================================
// Combined Coupling Features (18 dimensions)
// =============================================================================

struct CouplingFeatures {
    KernelDependence kernel;
    CoverageTransport transport;
    CrossWaveletCoherence wavelet;

    std::vector<double> to_vector() const {
        std::vector<double> v;
        v.reserve(DIMS);
        auto k = kernel.to_vector();
        auto t = transport.to_vector();
        auto w = wavelet.to_vector();
        v.insert(v.end(), k.begin(), k.end());
        v.insert(v.end(), t.begin(), t.end());
        v.insert(v.end(), w.begin(), w.end());
        return v;
    }

    static constexpr int DIMS = 18;
};

// =============================================================================
// Coupling Feature Extractor
// =============================================================================

class CouplingExtractor {
public:
    // Fit PCA basis from all TNF vectors in the dataset
    // Call once before extract_coupling_features()
    static void fit_pca_basis(const std::vector<std::vector<double>>& all_tnf,
                              int n_components = 3);

    // Check if PCA basis has been fitted
    static bool is_basis_ready();

    // Extract coupling features for a single contig
    // Requires per-base coverage (one value per bp)
    static CouplingFeatures extract_coupling_features(
        const std::string& sequence,
        const std::vector<uint32_t>& per_base_coverage,
        double damage_rate_5p = 0.0);

    // Get the fitted PCA basis (for inspection/debugging)
    static const std::vector<std::vector<double>>& get_pca_basis();
    static const std::vector<double>& get_pca_means();

private:
    // PCA basis vectors: 3 x 136 (n_components x TNF_DIM)
    static std::vector<std::vector<double>> pca_basis_;
    static std::vector<double> pca_means_;  // 136-dim mean vector
    static bool basis_fitted_;

    // Window computation
    static WindowParams compute_window_params(size_t contig_length);

    // Compute window signals (local TNF + coverage)
    static std::vector<WindowSignal> compute_window_signals(
        const std::string& sequence,
        const std::vector<uint32_t>& per_base_coverage,
        const WindowParams& params);

    // Project local TNF onto 3-dim PCA basis
    static std::array<double, 3> project_local_tnf(
        const std::string& sequence, size_t start, size_t end);

    // Compute local 136-dim TNF for a subsequence
    static std::vector<double> compute_local_tnf(
        const std::string& sequence, size_t start, size_t end);

    // Kernel dependence computation (stub)
    static KernelDependence compute_kernel_dependence(
        const std::vector<WindowSignal>& signals);

    // Coverage transport computation (stub)
    static CoverageTransport compute_coverage_transport(
        const std::vector<WindowSignal>& signals);

    // Cross-wavelet coherence computation (stub)
    static CrossWaveletCoherence compute_cross_wavelet(
        const std::vector<WindowSignal>& signals);
};

}  // namespace amber

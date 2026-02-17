#pragma once

#include <string>
#include <vector>
#include <map>
#include "damage_profile.h"
#include "fractal_features.h"
#include <htslib/sam.h>

namespace amber {

// Shared BAM state to avoid opening/closing per contig
struct SharedBAMState {
  samFile* bam = nullptr;
  bam_hdr_t* header = nullptr;
  hts_idx_t* idx = nullptr;

  bool open(const std::string& bam_file);
  void close();
  ~SharedBAMState() { close(); }
};

// Comprehensive coverage and alignment data for a contig (single-end ancient DNA)
struct ContigAlignmentData {
  std::vector<int> coverage;              // Per-base coverage (CIGAR-aware)
  std::vector<double> alignment_score;    // Mean alignment score (AS tag) per window
  std::vector<double> edit_distance;      // Mean edit distance (NM tag) per window
  std::vector<double> damage_rate_5p;     // C→T damage rate per window (5' end)
  std::vector<double> damage_rate_3p;     // G→A damage rate per window (3' end)
  std::vector<int> damage_count_5p;       // Number of C+T observations at 5' (for filtering)
  std::vector<int> damage_count_3p;       // Number of G+A observations at 3' (for filtering)
  std::vector<int> read_count;            // Reads per window (for coverage-aware filtering)

  // Per-window raw counts for proper Bayesian model fitting (MD tag derived)
  std::vector<std::vector<uint64_t>> n_5p;  // [window][position] opportunities
  std::vector<std::vector<uint64_t>> k_5p;  // [window][position] mismatches
  std::vector<std::vector<uint64_t>> n_3p;  // [window][position] opportunities
  std::vector<std::vector<uint64_t>> k_3p;  // [window][position] mismatches
};

// Per-window fractal/complexity metrics for breakpoint detection
struct WindowFractalMetrics {
  double hurst;                // Hurst exponent (0.5=random, deviations indicate structure change)
  double lz_complexity;        // Lempel-Ziv complexity (normalized)
  double entropy;              // Shannon entropy of k-mer distribution
  double cgr_lacunarity;       // CGR lacunarity (texture heterogeneity)
};

// Chimera detection based on multiple signals (single-end ancient DNA optimized)
struct ChimeraSignal {
  int position;                // Breakpoint position

  // Original signals
  double gc_shift;             // GC content shift across breakpoint
  double coverage_jump;        // Coverage ratio (after/before)
  double alignment_score_drop; // Alignment score drop (AS tag)
  double edit_distance_spike;  // Edit distance spike (NM tag)
  double damage_shift;         // Damage pattern shift (ancient DNA specific!)

  // Model-based damage evidence (Bayesian)
  double p_ancient_before;     // P(ancient) for segment before breakpoint
  double p_ancient_after;      // P(ancient) for segment after breakpoint
  double amplitude_diff;       // |amplitude_before - amplitude_after|

  // Fractal/complexity signals (new)
  double hurst_discontinuity;  // Hurst exponent jump across breakpoint
  double lz_discontinuity;     // Lempel-Ziv complexity jump
  double entropy_discontinuity;// Entropy jump across breakpoint
  double cgr_shift;            // CGR feature space distance across breakpoint

  double confidence;           // Overall confidence score [0,1]

  // Evidence breakdown for interpretability
  std::string primary_evidence; // Which signal contributed most

  bool is_likely_chimera() const {
    return confidence > 0.7; // Threshold for chimera classification
  }

  ChimeraSignal() : position(0), gc_shift(0), coverage_jump(1.0),
                    alignment_score_drop(0), edit_distance_spike(1.0),
                    damage_shift(0), p_ancient_before(0), p_ancient_after(0),
                    amplitude_diff(0), hurst_discontinuity(0),
                    lz_discontinuity(0), entropy_discontinuity(0),
                    cgr_shift(0), confidence(0) {}
};

class ChimeraDetector {
public:
  // Detect chimeric regions in a sequence (deprecated - loads damage profile every time)
  static std::vector<ChimeraSignal> detect_chimeras(
      const std::string &contig_id,
      const std::string &sequence,
      const std::string &bam_file,
      const std::string &damage_file);

  // Detect chimeric regions with pre-loaded damage profile (MUCH faster!)
  static std::vector<ChimeraSignal> detect_chimeras(
      const std::string &contig_id,
      const std::string &sequence,
      const std::string &bam_file,
      const DamageProfile *damage_profile);

  // OPTIMIZED: Detect with shared BAM state and optional fractal features
  static std::vector<ChimeraSignal> detect_chimeras_fast(
      const std::string &contig_id,
      const std::string &sequence,
      SharedBAMState *bam_state,
      bool use_fractal_features = false,
      int damage_positions = 15);

private:
  // Extract comprehensive alignment data from BAM (CIGAR-aware, single pass)
  // Optimized for single-end ancient DNA data
  static ContigAlignmentData extract_alignment_data(
      const std::string &contig_id,
      const std::string &bam_file,
      int contig_length,
      int window_size = 100,
      int damage_positions = 15);

  // OPTIMIZED: Extract using shared BAM state
  static ContigAlignmentData extract_alignment_data_fast(
      const std::string &contig_id,
      SharedBAMState *bam_state,
      int contig_length,
      int window_size = 100,
      int damage_positions = 15);

  // Detect GC content shifts (sliding window)
  static std::vector<int> detect_gc_shifts(
      const std::string &sequence,
      int window_size = 500,
      double min_shift = 0.1);

  // Detect coverage jumps from pre-computed data
  static std::vector<int> detect_coverage_jumps(
      const std::vector<int> &coverage,
      int window_size = 500,
      double min_ratio = 2.0);

  // Detect alignment score drops (AS tag)
  static std::vector<int> detect_alignment_score_drops(
      const std::vector<double> &alignment_score,
      int window_size = 100,
      double min_drop_pct = 0.15);

  // Detect edit distance spikes (NM tag)
  static std::vector<int> detect_edit_distance_spikes(
      const std::vector<double> &edit_distance,
      int window_size = 100,
      double min_spike_ratio = 1.5);

  // Detect damage pattern shifts (ancient DNA specific!)
  static std::vector<int> detect_damage_shifts(
      const std::vector<double> &damage_5p,
      const std::vector<double> &damage_3p,
      int window_size = 100,
      double min_shift = 0.05);

  // Count-aware damage shift detection with smoothing (reduces false positives)
  // Uses 5-window smoothing for effective 500bp resolution at 100bp granularity
  // Requires minimum observations per window for reliable estimates
  static std::vector<int> detect_damage_shifts_robust(
      const std::vector<double> &damage_5p,
      const std::vector<double> &damage_3p,
      const std::vector<int> &count_5p,
      const std::vector<int> &count_3p,
      int window_size = 100,
      double min_shift = 0.08,
      int min_observations = 30);

  // Model-based damage shift detection using Bayesian model fitting
  // Fits exponential decay models to segments before/after breakpoints
  // Returns breakpoints where p_ancient or amplitude differs significantly
  static std::vector<int> detect_damage_model_shifts(
      const ContigAlignmentData &aln_data,
      int window_size = 100,
      double min_p_ancient_diff = 0.3,
      double min_amplitude_diff = 0.05);

  // Compute model-based damage evidence for a specific breakpoint
  // Pools damage counts from windows before/after and fits models
  static void compute_damage_model_evidence(
      const ContigAlignmentData &aln_data,
      int breakpoint_window,
      double &p_ancient_before,
      double &p_ancient_after,
      double &amplitude_diff);

  // Fractal-based breakpoint detection (NEW)
  // Compute per-window fractal metrics along the sequence
  static std::vector<WindowFractalMetrics> compute_window_fractals(
      const std::string &sequence,
      int window_size = 1000,
      int step_size = 500);

  // Detect Hurst exponent discontinuities
  static std::vector<int> detect_hurst_discontinuities(
      const std::vector<WindowFractalMetrics> &metrics,
      int window_size = 1000,
      double min_jump = 0.15);

  // Detect Lempel-Ziv complexity discontinuities
  static std::vector<int> detect_lz_discontinuities(
      const std::vector<WindowFractalMetrics> &metrics,
      int window_size = 1000,
      double min_jump = 0.10);

  // Detect entropy discontinuities
  static std::vector<int> detect_entropy_discontinuities(
      const std::vector<WindowFractalMetrics> &metrics,
      int window_size = 1000,
      double min_jump = 0.15);

  // Compute CGR feature distance between two sequence windows
  static double compute_cgr_distance(
      const std::string &seq1,
      const std::string &seq2);

  // Compute GC content in window
  static double compute_gc_window(
      const std::string &sequence,
      int start,
      int end);

  // Compute signal strength at position
  static double compute_signal_at_position(
      const std::vector<int> &data,
      int position,
      int window_size);

  // Compute signal strength for double vectors
  static double compute_signal_at_position(
      const std::vector<double> &data,
      int position,
      int window_size);

  // Determine primary evidence source from signal strengths
  static std::string determine_primary_evidence(const ChimeraSignal &sig);
};

} // namespace amber

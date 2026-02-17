#include "kmer_features.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace amber {

// COMEBin canonical 4-mer mapping lookup table
// Generated from gen_kmer.py's product('ATGC', repeat=4) ordering
// Maps AMBER's 256-index space (A=0,C=1,G=2,T=3) to COMEBin's 136 canonical indices
// Usage: comebin_idx = COMEBIN_4MER_LUT[amber_idx]
static constexpr uint8_t COMEBIN_4MER_LUT[256] = {
      0,   3,   2,   1,  12,  15,  14,  13,   8,  11,  10,   9,   4,   7,   6,   5,  // AAAA - AATT
     45,  47,  46,  24,  55,  57,  56,  38,  51,  54,  53,  52,  48,  50,  49,   9,  // ACAA - ACTT
     31,  33,  32,  28,  41,  44,  43,  42,  37,  40,  39,  38,  34,  36,  35,  13,  // AGAA - AGTT
     16,  19,  18,  17,  27,  30,  29,  28,  23,  26,  25,  24,  20,  22,  21,   1,  // ATAA - ATTT
     73, 110, 126,  21,  83, 116, 129,  35,  93, 122, 128,  49,  62, 102, 127,   6,  // CAAA - CATT
     76, 111, 131,  25,  86, 117, 133,  39,  96, 123, 135,  53,  65, 104, 128,  10,  // CCAA - CCTT
     79, 113, 132,  29,  89, 119, 134,  43,  98, 125, 133,  56,  68, 106, 129,  14,  // CGAA - CGTT
     71, 108, 130,  18,  81, 115, 132,  32,  91, 121, 131,  46,  59, 100, 126,   2,  // CTAA - CTTT
     74, 101, 100,  22,  84, 107, 106,  36,  94, 105, 104,  50,  63, 103, 102,   7,  // GAAA - GATT
     77, 112, 121,  26,  87, 118, 125,  40,  97, 124, 123,  54,  66, 105, 122,  11,  // GCAA - GCTT
     80, 114, 115,  30,  90, 120, 119,  44,  99, 118, 117,  57,  69, 107, 116,  15,  // GGAA - GGTT
     72, 109, 108,  19,  82, 114, 113,  33,  92, 112, 111,  47,  60, 101, 110,   3,  // GTAA - GTTT
     58,  60,  59,  20,  67,  69,  68,  34,  64,  66,  65,  48,  61,  63,  62,   4,  // TAAA - TATT
     75,  92,  91,  23,  85,  99,  98,  37,  95,  97,  96,  51,  64,  94,  93,   8,  // TCAA - TCTT
     78,  82,  81,  27,  88,  90,  89,  41,  85,  87,  86,  55,  67,  84,  83,  12,  // TGAA - TGTT
     70,  72,  71,  16,  78,  80,  79,  31,  75,  77,  76,  45,  58,  74,  73,   0   // TTAA - TTTT
};

// Fast k-mer counting with streaming (memory-efficient)
class KmerCounter {
public:
  // Count k-mers in sequence (canonical, to handle both strands)
  template <size_t N>
  static void count_kmers_canonical(const std::string &seq, int k,
                                     std::array<float, N> &counts) {
    if (seq.length() < static_cast<size_t>(k))
      return;

    counts.fill(0.0f);
    size_t valid_kmers = 0;

    // Sliding window k-mer extraction
    for (size_t i = 0; i <= seq.length() - k; ++i) {
      uint32_t idx = 0;
      bool valid = true;
      for (int j = 0; j < k; ++j) {
        uint8_t bits = KmerUtils::base_to_bits(seq[i + j]);
        if (bits > 3) {
          valid = false;
          break;
        }
        idx = (idx << 2) | bits;
      }
      if (!valid)
        continue;

      // Use COMEBin LUT for k=4, fall back to generic for other k
      size_t array_idx;
      if (k == 4) {
        array_idx = COMEBIN_4MER_LUT[idx];
      } else {
        uint32_t canon_idx = KmerUtils::canonical_index(idx, k);
        array_idx = canonical_to_array_index(canon_idx, k);
      }

      if (array_idx < N) {
        counts[array_idx] += 1.0f;
        valid_kmers++;
      }
    }

    // Normalize by number of valid k-mers (matches raw COMEBin semantics on ambiguous bases)
    if (valid_kmers > 0) {
      float total_kmers = static_cast<float>(valid_kmers);
      for (auto &c : counts) {
        c /= total_kmers;
      }
    }
  }

  // Count k-mers in a window region (no string copy!)
  template <size_t N>
  static void count_kmers_canonical_range(const std::string &seq, int k,
                                          size_t start, size_t end,
                                          std::array<float, N> &counts) {
    if (end - start < static_cast<size_t>(k))
      return;

    counts.fill(0.0f);
    size_t valid_kmers = 0;

    // Sliding window k-mer extraction in range
    for (size_t i = start; i + k <= end; ++i) {
      uint32_t idx = 0;
      bool valid = true;
      for (int j = 0; j < k; ++j) {
        uint8_t bits = KmerUtils::base_to_bits(seq[i + j]);
        if (bits > 3) {
          valid = false;
          break;
        }
        idx = (idx << 2) | bits;
      }
      if (!valid)
        continue;

      // Use COMEBin LUT for k=4, fall back to generic for other k
      size_t array_idx;
      if (k == 4) {
        array_idx = COMEBIN_4MER_LUT[idx];
      } else {
        uint32_t canon_idx = KmerUtils::canonical_index(idx, k);
        array_idx = canonical_to_array_index(canon_idx, k);
      }

      if (array_idx < N) {
        counts[array_idx] += 1.0f;
        valid_kmers++;
      }
    }

    // Normalize by number of valid k-mers in window
    if (valid_kmers > 0) {
      float total_kmers = static_cast<float>(valid_kmers);
      for (auto &c : counts) {
        c /= total_kmers;
      }
    }
  }

  // Count k-mers in a sliding window (for variance computation)
  static std::vector<float> sliding_window_kmer_variance(
      const std::string &seq, int k, size_t window_size, size_t step) {

    (void)k; // Currently hardcoded to 4-mers
    std::vector<float> variances;

    if (seq.length() < window_size)
      return variances;

    for (size_t i = 0; i + window_size <= seq.length(); i += step) {
      // Count 4-mers in this window (no string copy!)
      std::array<float, 136> kmers;
      count_kmers_canonical_range(seq, 4, i, i + window_size, kmers);

      // Compute variance of k-mer frequencies
      float mean = 0.0f;
      for (float val : kmers)
        mean += val;
      mean /= kmers.size();

      float variance = 0.0f;
      for (float val : kmers) {
        float diff = val - mean;
        variance += diff * diff;
      }
      variance /= kmers.size();

      variances.push_back(variance);
    }

    return variances;
  }

private:
  // Map canonical k-mer index to array index
  // For k=3: 32 canonical k-mers (out of 64 total)
  // For k=4: 136 canonical k-mers (out of 256 total)
  // For k=5: 512 canonical k-mers (out of 1024 total)
  static size_t canonical_to_array_index(uint32_t canon_idx, int k) {
    // Simple mapping: canonical indices are already in [0, num_canonical)
    // This works because canonical_index() returns the smaller of kmer and RC
    size_t max_idx = 1ULL << (2 * k);
    return (canon_idx < max_idx / 2) ? canon_idx : (max_idx - 1 - canon_idx);
  }
};

// Main feature extractor class
class FeatureExtractor {
public:
  // Extract all k-mer features from sequence
  static void extract_kmer_features(const std::string &seq,
                                     ContigFeatures &features) {
    features.length = seq.length();

    // Extract TNF (4-mers) - the gold standard for metagenomic binning
    // RAW counts - CLR transformation is done in the cluster command
    // NOTE: Trinuc (3-mers) removed - redundant with TNF + CGR quadrant densities
    //       Pentanuc (5-mers) removed - redundant with fractal entropy scaling features
    KmerCounter::count_kmers_canonical(seq, 4, features.tnf_raw);

    // Compute k-mer variance (local heterogeneity)
    size_t window_size = 500;
    size_t step = 250; // 50% overlap
    auto variances = KmerCounter::sliding_window_kmer_variance(
        seq, 4, window_size, step);

    // Store up to 5 variance values
    for (size_t i = 0; i < std::min(size_t(5), variances.size()); ++i) {
      features.kmer_variance[i] = variances[i];
    }
  }

  // Extract compositional features
  static void extract_composition_features(const std::string &seq,
                                            ContigFeatures &features) {
    if (seq.empty())
      return;

    // Count bases
    size_t A = 0, C = 0, G = 0, T = 0;
    for (char c : seq) {
      switch (c) {
      case 'A':
      case 'a':
        A++;
        break;
      case 'C':
      case 'c':
        C++;
        break;
      case 'G':
      case 'g':
        G++;
        break;
      case 'T':
      case 't':
        T++;
        break;
      }
    }

    size_t total = A + C + G + T;
    if (total == 0)
      return;

    // GC content
    features.gc_content = static_cast<float>(G + C) / total;

    // GC skewness: (G-C)/(G+C)
    // CRITICAL: Cast to signed BEFORE subtraction to avoid unsigned wraparound!
    if (G + C > 0) {
      features.gc_skewness = static_cast<float>(static_cast<int64_t>(G) - static_cast<int64_t>(C)) /
                              static_cast<float>(G + C);
    } else {
      features.gc_skewness = 0.0f;
    }

    // AT skewness: (A-T)/(A+T)
    // CRITICAL: Cast to signed BEFORE subtraction to avoid unsigned wraparound!
    if (A + T > 0) {
      features.at_skewness = static_cast<float>(static_cast<int64_t>(A) - static_cast<int64_t>(T)) /
                              static_cast<float>(A + T);
    } else {
      features.at_skewness = 0.0f;
    }

    // Debug: check for unreasonable values (should be in [-1, 1])
    if (std::abs(features.gc_skewness) > 1.1 || std::abs(features.at_skewness) > 1.1) {
      std::cerr << "WARNING: Unreasonable skewness values detected!\n";
      std::cerr << "  gc_skewness=" << features.gc_skewness
                << ", at_skewness=" << features.at_skewness << "\n";
      std::cerr << "  G=" << G << ", C=" << C << ", A=" << A << ", T=" << T << "\n";
    }

    // Strand asymmetry (cumulative G-C and T-A bias)
    features.strand_asymmetry =
        std::abs(features.gc_skewness) + std::abs(features.at_skewness);

    // GC variance (sliding window - no string copy!)
    size_t window_size = 500;
    std::vector<float> gc_values;

    for (size_t i = 0; i + window_size <= seq.length(); i += 250) {
      size_t win_G = 0, win_C = 0, win_total = 0;

      // Direct access to sequence range (no substring creation)
      for (size_t j = i; j < i + window_size; ++j) {
        char c = seq[j];
        if (c == 'G' || c == 'g')
          win_G++;
        else if (c == 'C' || c == 'c')
          win_C++;
        if (c == 'A' || c == 'a' || c == 'C' || c == 'c' || c == 'G' ||
            c == 'g' || c == 'T' || c == 't')
          win_total++;
      }

      if (win_total > 0) {
        gc_values.push_back(static_cast<float>(win_G + win_C) / win_total);
      }
    }

    // Compute variance of GC content across windows
    if (!gc_values.empty()) {
      float mean_gc = 0.0f;
      for (float val : gc_values)
        mean_gc += val;
      mean_gc /= gc_values.size();

      float variance = 0.0f;
      for (float val : gc_values) {
        float diff = val - mean_gc;
        variance += diff * diff;
      }
      variance /= gc_values.size();
      features.gc_variance = variance;
    }

    // === DINUCLEOTIDE RATIOS (observed/expected) ===
    // Strong phylogenetic signals - CpG suppression varies by methylation system
    // CpG_ratio = (CpG_count * N) / (C_count * G_count) where N = total dinucs

    // Count dinucleotides in single pass
    size_t di_CG = 0, di_TA = 0, di_GC = 0, di_AT = 0;
    size_t di_total = 0;

    for (size_t i = 0; i + 1 < seq.length(); ++i) {
      char a = seq[i], b = seq[i + 1];
      // Normalize to uppercase
      if (a >= 'a') a -= 32;
      if (b >= 'a') b -= 32;

      // Skip non-ACGT
      bool a_valid = (a == 'A' || a == 'C' || a == 'G' || a == 'T');
      bool b_valid = (b == 'A' || b == 'C' || b == 'G' || b == 'T');
      if (!a_valid || !b_valid) continue;

      di_total++;

      if (a == 'C' && b == 'G') di_CG++;
      else if (a == 'T' && b == 'A') di_TA++;
      else if (a == 'G' && b == 'C') di_GC++;
      else if (a == 'A' && b == 'T') di_AT++;
    }

    // Compute observed/expected ratios
    // Expected CpG = (C_freq * G_freq) * total_dinucs
    // Ratio = observed / expected = (CpG_count * N) / (C_count * G_count)
    if (di_total > 0 && C > 0 && G > 0) {
      features.cpg_ratio = (float)((double)di_CG * total / (C * G));
    }
    if (di_total > 0 && T > 0 && A > 0) {
      features.tpa_ratio = (float)((double)di_TA * total / (T * A));
    }
    if (di_total > 0 && G > 0 && C > 0) {
      features.gpc_ratio = (float)((double)di_GC * total / (G * C));
    }
    if (di_total > 0 && A > 0 && T > 0) {
      features.apt_ratio = (float)((double)di_AT * total / (A * T));
    }
  }
};

// Export functions for use in main pipeline
void extract_kmer_features(const std::string &seq, ContigFeatures &features) {
  FeatureExtractor::extract_kmer_features(seq, features);
}

void extract_composition_features(const std::string &seq,
                                   ContigFeatures &features) {
  FeatureExtractor::extract_composition_features(seq, features);
}

// ============================================================================
// BINNING-OPTIMIZED TNF EXTRACTION
// ============================================================================
// Based on EXPERIMENT_METRICS.md analysis:
// - Aitchison distance (CLR + Euclidean) achieves AUC 0.956 for TNF
// - L2 normalization achieves AUC 0.937
// - Raw Euclidean on frequencies is suboptimal

void extract_tnf_clr(const std::string &seq, std::array<float, 136> &tnf_clr) {
  // First extract raw frequencies
  ContigFeatures features;
  extract_kmer_features(seq, features);

  // Copy and apply CLR transform for Aitchison distance
  tnf_clr = features.tnf_raw;
  CLRTransform::apply(tnf_clr);
}

void extract_tnf_l2(const std::string &seq, std::array<float, 136> &tnf_l2) {
  // First extract raw frequencies
  ContigFeatures features;
  extract_kmer_features(seq, features);

  // Copy and apply L2 normalization (baseline-compatible)
  tnf_l2 = features.tnf_raw;

  float norm = 0.0f;
  for (float v : tnf_l2) norm += v * v;
  if (norm > 0) {
    norm = std::sqrt(norm);
    for (float& v : tnf_l2) v /= norm;
  }
}

void extract_tnf_raw(const std::string &seq, std::array<uint32_t, 136> &tnf_raw) {
  // RAW INTEGER COUNTS - matches COMEBin's gen_kmer.py output exactly
  // No normalization - used for COMEBin compatibility verification
  tnf_raw.fill(0);

  if (seq.length() < 4) return;

  for (size_t i = 0; i <= seq.length() - 4; ++i) {
    // Check for valid bases (skip k-mers with N/ambiguous)
    bool valid = true;
    for (int j = 0; j < 4; ++j) {
      char c = seq[i + j];
      if (c != 'A' && c != 'a' && c != 'C' && c != 'c' &&
          c != 'G' && c != 'g' && c != 'T' && c != 't') {
        valid = false;
        break;
      }
    }
    if (!valid) continue;

    uint32_t idx = KmerUtils::kmer_to_index(&seq[i], 4);
    uint8_t array_idx = COMEBIN_4MER_LUT[idx];
    tnf_raw[array_idx]++;
  }
}

void extract_tnf_comebin(const std::string &seq, std::array<float, 136> &tnf_comebin) {
  // COMEBin-style k-mer normalization (get_augfeature.py lines 113-114):
  //   compositMats = compositMats + 1   # Add pseudo-count
  //   compositMats = compositMats / compositMats.sum(axis=1)[:, None]  # Divide by row sum
  //
  // This is NOT L2 normalization - it produces a probability distribution

  // First get raw integer counts
  std::array<uint32_t, 136> tnf_raw;
  extract_tnf_raw(seq, tnf_raw);

  // Add 1 (pseudo-count) and compute sum
  float sum = 0.0f;
  for (size_t i = 0; i < 136; i++) {
    tnf_comebin[i] = static_cast<float>(tnf_raw[i]) + 1.0f;
    sum += tnf_comebin[i];
  }

  // Divide by sum (produces probability distribution)
  if (sum > 0) {
    for (float& v : tnf_comebin) {
      v /= sum;
    }
  }
}

} // namespace amber

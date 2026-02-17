#pragma once

#include "kmer_features.h"
#include <htslib/hts.h>
#include <htslib/sam.h>
#include <string>
#include <vector>

namespace amber {

// Depth computation parameters (MetaBAT2-style defaults)
struct DepthParams {
  uint32_t max_edge_bases = 75;    // Exclude this many bases from each edge (default: 75)
  float min_percent_identity = 97.0f;  // Minimum % identity to count read (default: 97%)
  uint8_t min_mapq = 0;           // Minimum MAPQ (default: 0)
  bool use_population_variance = false;  // If true, use N divisor (COMEBin); if false, use N-1 (MetaBAT2)
  bool emit_per_base_coverage = false;  // If true, populate DepthResult::coverage
};

// COMEBin-compatible coverage result (mean and population variance)
struct COMEBinCoverage {
  double mean = 0.0;
  double variance = 0.0;  // Population variance (N divisor) - matches np.var()
};

// Result of depth calculation
struct DepthResult {
  float mean_depth = 0.0f;
  float variance = 0.0f;  // Sample variance (N-1 divisor, Bessel's correction) - matches MetaBAT2
  std::vector<uint32_t> coverage;  // Per-base coverage vector (PERFORMANCE FIX: reuse in extract_coverage_features)
};

// Extract coverage features from BAM file
class CoverageExtractor {
public:
  // Extract binned coverage for a contig (opens/closes BAM file)
  static bool extract_coverage_features(const std::string &bam_path,
                                        const std::string &contig_id,
                                        uint32_t contig_length,
                                        ContigFeatures &features,
                                        uint32_t edge_exclusion_bp = 75);

  // Extract binned coverage using pre-opened BAM file (MUCH faster for batch processing!)
  static bool extract_coverage_features(samFile *bam_fp,
                                        bam_hdr_t *header,
                                        hts_idx_t *idx,
                                        const std::string &contig_id,
                                        uint32_t contig_length,
                                        ContigFeatures &features,
                                        uint32_t edge_exclusion_bp = 75);

  // Extract coverage from a specific coordinate range (for split contigs)
  // Reads from original_contig_id[range_start:range_end] in BAM, outputs as fragment_length
  static bool extract_coverage_features_range(samFile *bam_fp,
                                              bam_hdr_t *header,
                                              hts_idx_t *idx,
                                              const std::string &original_contig_id,
                                              uint32_t range_start,
                                              uint32_t range_end,
                                              uint32_t fragment_length,
                                              ContigFeatures &features,
                                              uint32_t edge_exclusion_bp = 75);

  // Extract depth and variance with configurable parameters
  // Default parameters match MetaBAT2's jgi_summarize_bam_contig_depths
  static DepthResult extract_depth(samFile *bam_fp,
                                   bam_hdr_t *header,
                                   hts_idx_t *idx,
                                   const std::string &contig_id,
                                   uint32_t contig_length,
                                   const DepthParams &params = DepthParams{});

  // COMEBin-compatible coverage extraction for a subsequence range
  // Returns mean and population variance (N divisor, matches np.var())
  // edge_within_range: bases to exclude from edges WITHIN the range (default 0 for COMEBin)
  static COMEBinCoverage extract_comebin_coverage(
      samFile *bam_fp,
      bam_hdr_t *header,
      hts_idx_t *idx,
      const std::string &contig_id,
      uint32_t range_start,
      uint32_t range_end,
      uint32_t edge_within_range = 0,
      float min_percent_identity = 0.0f);  // 0 = no identity filter (COMEBin default)

private:
  // Bin coverage into fixed number of bins
  static void bin_coverage(const std::vector<uint32_t> &per_base_cov,
                          std::array<float, 20> &binned,
                          float &mean, float &variance, float &skewness);
};

} // namespace amber

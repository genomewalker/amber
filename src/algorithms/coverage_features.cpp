#include "coverage_features.h"
#include <algorithm>
#include <cmath>
#include <htslib/hts.h>
#include <htslib/sam.h>

namespace amber {

void CoverageExtractor::bin_coverage(const std::vector<uint32_t> &per_base_cov,
                                     std::array<float, 20> &binned, float &mean,
                                     float &variance, float &skewness) {
  if (per_base_cov.empty()) {
    binned.fill(0.0f);
    mean = variance = skewness = 0.0f;
    return;
  }

  // Compute mean
  uint64_t sum = 0;
  for (uint32_t cov : per_base_cov)
    sum += cov;
  mean = static_cast<float>(sum) / per_base_cov.size();

  // Compute variance with Bessel correction for unbiased estimate
  double var_sum = 0.0;
  for (uint32_t cov : per_base_cov) {
    double diff = cov - mean;
    var_sum += diff * diff;
  }
  // Use (n-1) for unbiased variance (Bessel correction)
  size_t n = per_base_cov.size();
  variance = static_cast<float>(var_sum / (n > 1 ? n - 1 : 1));

  // Compute skewness
  double skew_sum = 0.0;
  double std_dev = std::sqrt(variance);
  if (std_dev > 0) {
    for (uint32_t cov : per_base_cov) {
      double z = (cov - mean) / std_dev;
      skew_sum += z * z * z;
    }
    skewness = static_cast<float>(skew_sum / per_base_cov.size());
  } else {
    skewness = 0.0f;
  }

  // Bin coverage (RAW values - log transform applied later in cluster.cpp)
  size_t bin_size = per_base_cov.size() / 20;
  if (bin_size == 0)
    bin_size = 1;

  for (size_t i = 0; i < 20; ++i) {
    size_t start = i * bin_size;
    size_t end = std::min(start + bin_size, per_base_cov.size());

    if (start >= per_base_cov.size())
      break;

    uint64_t bin_sum = 0;
    for (size_t j = start; j < end; ++j) {
      bin_sum += per_base_cov[j];
    }

    float bin_mean = static_cast<float>(bin_sum) / (end - start);
    // Store RAW coverage (not log-transformed)
    // cluster.cpp will apply log(cov + 1) transform consistently with other features
    binned[i] = bin_mean;
  }
}

bool CoverageExtractor::extract_coverage_features(const std::string &bam_path,
                                                  const std::string &contig_id,
                                                  uint32_t contig_length,
                                                  ContigFeatures &features,
                                                  uint32_t edge_exclusion_bp) {
  // Open BAM
  samFile *bam_fp = sam_open(bam_path.c_str(), "r");
  if (!bam_fp)
    return false;

  bam_hdr_t *header = sam_hdr_read(bam_fp);
  if (!header) {
    sam_close(bam_fp);
    return false;
  }

  hts_idx_t *idx = sam_index_load(bam_fp, bam_path.c_str());
  if (!idx) {
    bam_hdr_destroy(header);
    sam_close(bam_fp);
    return false;
  }

  // Use the optimized overload
  bool result = extract_coverage_features(bam_fp, header, idx, contig_id, contig_length, features, edge_exclusion_bp);

  hts_idx_destroy(idx);
  bam_hdr_destroy(header);
  sam_close(bam_fp);

  return result;
}

bool CoverageExtractor::extract_coverage_features(samFile *bam_fp,
                                                  bam_hdr_t *header,
                                                  hts_idx_t *idx,
                                                  const std::string &contig_id,
                                                  uint32_t contig_length,
                                                  ContigFeatures &features,
                                                  uint32_t edge_exclusion_bp) {
  // Use MetaBAT2-compatible depth extraction (with identity filtering)
  DepthParams params;
  params.max_edge_bases = edge_exclusion_bp;
  params.min_percent_identity = 97.0f;  // MetaBAT2 default
  params.min_mapq = 0;                   // MetaBAT2 default
  params.emit_per_base_coverage = true;  // Needed for skewness + binned profile

  DepthResult depth = extract_depth(bam_fp, header, idx, contig_id, contig_length, params);

  features.mean_coverage = depth.mean_depth;
  features.var_coverage = depth.variance;

  // PERFORMANCE FIX: Reuse coverage vector from extract_depth instead of re-scanning BAM
  // This eliminates redundant BAM iteration and per-base CIGAR parsing
  if (depth.coverage.empty()) {
    return false;  // No coverage data available
  }

  const std::vector<uint32_t>& coverage = depth.coverage;

  // PERFORMANCE FIX: All mean/variance already computed by extract_depth
  // No need to re-scan BAM or recalculate - just compute skewness and bins from coverage vector

  // Edge exclusion for skewness (same as used for mean/var)
  uint32_t edge_bp = edge_exclusion_bp;
  uint32_t start_pos = std::min(edge_bp, contig_length);
  uint32_t end_pos = contig_length > edge_bp ? contig_length - edge_bp : 0;

  if (start_pos >= end_pos) {
    start_pos = 0;
    end_pos = contig_length;
  }

  uint32_t count = end_pos - start_pos;

  // Calculate skewness (still uses excluded edges region)
  double skew_sum = 0.0;
  double std_dev = std::sqrt(features.var_coverage);
  if (std_dev > 0 && count > 0) {
    for (uint32_t i = start_pos; i < end_pos; ++i) {
      double z = (coverage[i] - features.mean_coverage) / std_dev;
      skew_sum += z * z * z;
    }
    features.coverage_skewness = static_cast<float>(skew_sum / count);
  } else {
    features.coverage_skewness = 0.0f;
  }

  // Bin the coverage into 20 bins (using FULL contig, not edge-trimmed)
  // This preserves the binned profile output while matching MetaBAT2 for mean/var
  size_t bin_size = contig_length / 20;
  if (bin_size == 0)
    bin_size = 1;

  for (size_t bin_idx = 0; bin_idx < 20; ++bin_idx) {
    size_t start = bin_idx * bin_size;
    size_t end = (bin_idx == 19) ? contig_length : (bin_idx + 1) * bin_size;
    if (start >= contig_length)
      break;
    if (end > contig_length)
      end = contig_length;

    uint64_t bin_sum = 0;
    for (size_t i = start; i < end; ++i) {
      bin_sum += coverage[i];
    }
    features.coverage_bins[bin_idx] =
        static_cast<float>(bin_sum) / (end - start);
  }

  return true;
}

bool CoverageExtractor::extract_coverage_features_range(samFile *bam_fp,
                                                        bam_hdr_t *header,
                                                        hts_idx_t *idx,
                                                        const std::string &original_contig_id,
                                                        uint32_t range_start,
                                                        uint32_t range_end,
                                                        uint32_t fragment_length,
                                                        ContigFeatures &features,
                                                        uint32_t edge_exclusion_bp) {
  // Find original contig tid in BAM
  int tid = bam_name2id(header, original_contig_id.c_str());
  if (tid < 0)
    return false;

  // Initialize coverage vector for fragment
  std::vector<uint32_t> coverage(fragment_length, 0);

  // Iterate alignments in the range
  bam1_t *aln = bam_init1();
  hts_itr_t *iter = sam_itr_queryi(idx, tid, range_start, range_end);
  if (!iter) {
    bam_destroy1(aln);
    return false;
  }

  while (sam_itr_next(bam_fp, iter, aln) >= 0) {
    // Skip unmapped, secondary, supplementary
    if (aln->core.flag &
        (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY))
      continue;

    uint32_t pos = aln->core.pos;
    uint32_t end_pos = bam_endpos(aln);

    // Only process reads overlapping our range
    if (end_pos <= range_start || pos >= range_end)
      continue;

    // Clamp to our range
    uint32_t overlap_start = std::max(pos, range_start);
    uint32_t overlap_end = std::min(end_pos, range_end);

    // Map to fragment coordinates (subtract range_start)
    for (uint32_t i = overlap_start; i < overlap_end; ++i) {
      uint32_t frag_pos = i - range_start;
      if (frag_pos < fragment_length) {
        coverage[frag_pos]++;
      }
    }
  }

  hts_itr_destroy(iter);
  bam_destroy1(aln);

  // Calculate mean and variance using same method as binner (matching MetaBAT2)
  // Exclude edge_exclusion_bp from edges and use population variance (N divisor)
  uint32_t edge_bp = edge_exclusion_bp;
  uint32_t start_pos = std::min(edge_bp, fragment_length);
  uint32_t end_pos = fragment_length > edge_bp ? fragment_length - edge_bp : 0;

  if (start_pos >= end_pos) {
    // Fragment too short for edge exclusion - use whole fragment
    start_pos = 0;
    end_pos = fragment_length;
  }

  uint32_t count = end_pos - start_pos;

  // Calculate mean excluding edges
  uint64_t sum = 0;
  for (uint32_t i = start_pos; i < end_pos; ++i) {
    sum += coverage[i];
  }
  features.mean_coverage = (count > 0) ? static_cast<float>(sum) / count : 0.0f;

  // Calculate variance with Bessel correction (N-1) - Consistent with extract_depth
  double var_sum = 0.0;
  for (uint32_t i = start_pos; i < end_pos; ++i) {
    double diff = coverage[i] - features.mean_coverage;
    var_sum += diff * diff;
  }
  
  if (count > 1) {
    features.var_coverage = static_cast<float>(var_sum / (count - 1));
  } else {
    features.var_coverage = 0.0f;
  }

  // Calculate skewness (still uses excluded edges region)
  double skew_sum = 0.0;
  double std_dev = std::sqrt(features.var_coverage);
  if (std_dev > 0 && count > 0) {
    for (uint32_t i = start_pos; i < end_pos; ++i) {
      double z = (coverage[i] - features.mean_coverage) / std_dev;
      skew_sum += z * z * z;
    }
    features.coverage_skewness = static_cast<float>(skew_sum / count);
  } else {
    features.coverage_skewness = 0.0f;
  }

  // Bin the coverage into 20 bins (using FULL fragment, not edge-trimmed)
  // This preserves the binned profile output while matching MetaBAT2 for mean/var
  size_t bin_size = fragment_length / 20;
  if (bin_size == 0)
    bin_size = 1;

  for (size_t bin_idx = 0; bin_idx < 20; ++bin_idx) {
    size_t start = bin_idx * bin_size;
    size_t end = (bin_idx == 19) ? fragment_length : (bin_idx + 1) * bin_size;
    if (start >= fragment_length)
      break;
    if (end > fragment_length)
      end = fragment_length;

    uint64_t bin_sum = 0;
    for (size_t i = start; i < end; ++i) {
      bin_sum += coverage[i];
    }
    features.coverage_bins[bin_idx] =
        static_cast<float>(bin_sum) / (end - start);
  }

  return true;
}

DepthResult CoverageExtractor::extract_depth(samFile *bam_fp,
                                             bam_hdr_t *header,
                                             hts_idx_t *idx,
                                             const std::string &contig_id,
                                             uint32_t contig_length,
                                             const DepthParams &params) {
  DepthResult result;

  // Find contig tid
  int tid = bam_name2id(header, contig_id.c_str());
  if (tid < 0)
    return result;

  // Edge exclusion bounds
  uint32_t edge_bp = params.max_edge_bases;
  uint32_t start_pos = std::min(edge_bp, contig_length);
  uint32_t end_pos = contig_length > edge_bp ? contig_length - edge_bp : 0;
  if (start_pos >= end_pos) {
    start_pos = 0;
    end_pos = contig_length;
  }

  if (params.emit_per_base_coverage) {
    result.coverage.assign(contig_length, 0);
  }

  // Streaming mean/variance accumulators
  uint64_t count = 0;
  long double sum = 0.0;
  long double sum_sq = 0.0;

  const uint32_t BLOCK = 500000;  // process coverage in blocks to keep memory bounded
  bam1_t *aln = bam_init1();

  for (uint32_t block_start = 0; block_start < contig_length; block_start += BLOCK) {
    uint32_t block_end = std::min(block_start + BLOCK, contig_length);
    // Skip blocks entirely outside the valid region
    if (block_end <= start_pos || block_start >= end_pos) continue;

    std::vector<uint32_t> cov_block(block_end - block_start, 0);

    hts_itr_t *iter = sam_itr_queryi(idx, tid, block_start, block_end);
    if (!iter) continue;

    while (sam_itr_next(bam_fp, iter, aln) >= 0) {
      // Skip unmapped, secondary, supplementary, duplicates, QC fail (MetaBAT2 filters)
      if (aln->core.flag &
          (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY | BAM_FDUP | BAM_FQCFAIL))
        continue;

      // MAPQ filter
      if (aln->core.qual < params.min_mapq)
        continue;

      // Identity filter (MetaBAT2-style)
      if (params.min_percent_identity > 0) {
        uint32_t *cigar = bam_get_cigar(aln);
        uint32_t n_cigar = aln->core.n_cigar;
        uint32_t aln_len = 0;
        for (uint32_t i = 0; i < n_cigar; ++i) {
          int op = bam_cigar_op(cigar[i]);
          if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF ||
              op == BAM_CINS) {
            aln_len += bam_cigar_oplen(cigar[i]);
          }
        }
        uint8_t *nm_tag = bam_aux_get(aln, "NM");
        if (nm_tag && aln_len > 0) {
          int32_t nm = bam_aux2i(nm_tag);
          if (nm < 0) nm = 0;
          int64_t matches = static_cast<int64_t>(aln_len) - static_cast<int64_t>(nm);
          if (matches < 0) matches = 0;
          float identity = 100.0f * static_cast<float>(matches) / static_cast<float>(aln_len);
          if (identity < params.min_percent_identity)
            continue;
        }
      }

      // Parse CIGAR, accumulate coverage within this block
      uint32_t *cigar = bam_get_cigar(aln);
      uint32_t n_cigar = aln->core.n_cigar;
      uint32_t ref_pos = aln->core.pos;

      for (uint32_t i = 0; i < n_cigar; ++i) {
        int op = bam_cigar_op(cigar[i]);
        uint32_t len = bam_cigar_oplen(cigar[i]);

        if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
          uint32_t cov_start = std::max(ref_pos, block_start);
          uint32_t cov_end = std::min(ref_pos + len, block_end);
          if (cov_start < cov_end) {
            for (uint32_t pos = cov_start; pos < cov_end; ++pos) {
              cov_block[pos - block_start]++;
            }
          }
          ref_pos += len;
        } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
          ref_pos += len;
        } else if (op == BAM_CINS || op == BAM_CSOFT_CLIP) {
          // consumes query only; no ref advance
        }
      }
    }

    hts_itr_destroy(iter);

    if (params.emit_per_base_coverage) {
      std::copy(cov_block.begin(), cov_block.end(), result.coverage.begin() + block_start);
    }

    // Accumulate stats for the in-bounds portion of this block
    uint32_t use_start = std::max(block_start, start_pos);
    uint32_t use_end = std::min(block_end, end_pos);
    for (uint32_t pos = use_start; pos < use_end; ++pos) {
      uint32_t cov = cov_block[pos - block_start];
      sum += cov;
      sum_sq += (long double)cov * (long double)cov;
      count++;
    }
  }

  bam_destroy1(aln);

  if (count == 0) return result;

  long double mean = sum / count;
  long double var = (sum_sq / count) - mean * mean;
  // Apply Bessel correction for sample variance unless population variance requested
  if (!params.use_population_variance && count > 1) {
    var *= (long double)count / (count - 1);
  }

  result.mean_depth = static_cast<float>(mean);
  result.variance = static_cast<float>(var > 0 ? var : 0.0);

  return result;
}

COMEBinCoverage CoverageExtractor::extract_comebin_coverage(
    samFile *bam_fp,
    bam_hdr_t *header,
    hts_idx_t *idx,
    const std::string &contig_id,
    uint32_t range_start,
    uint32_t range_end,
    uint32_t edge_within_range,
    float min_percent_identity) {
  COMEBinCoverage result;

  // Find contig tid
  int tid = bam_name2id(header, contig_id.c_str());
  if (tid < 0)
    return result;

  uint32_t range_len = range_end - range_start;
  if (range_len == 0)
    return result;

  // Per-base coverage for the range
  std::vector<uint32_t> coverage(range_len, 0);

  // Iterate alignments
  bam1_t *aln = bam_init1();
  hts_itr_t *iter = sam_itr_queryi(idx, tid, range_start, range_end);
  if (!iter) {
    bam_destroy1(aln);
    return result;
  }

  while (sam_itr_next(bam_fp, iter, aln) >= 0) {
    // Skip unmapped, secondary, supplementary, duplicates, QC fail
    if (aln->core.flag &
        (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY | BAM_FDUP | BAM_FQCFAIL))
      continue;

    // Optional identity filter
    if (min_percent_identity > 0) {
      uint32_t *cigar = bam_get_cigar(aln);
      uint32_t n_cigar = aln->core.n_cigar;
      uint32_t aln_len = 0;
      for (uint32_t i = 0; i < n_cigar; ++i) {
        int op = bam_cigar_op(cigar[i]);
        if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF || op == BAM_CINS) {
          aln_len += bam_cigar_oplen(cigar[i]);
        }
      }
      uint8_t *nm_tag = bam_aux_get(aln, "NM");
      if (nm_tag && aln_len > 0) {
        int32_t nm = bam_aux2i(nm_tag);
        if (nm < 0) nm = 0;
        int64_t matches = static_cast<int64_t>(aln_len) - static_cast<int64_t>(nm);
        if (matches < 0) matches = 0;
        float identity = 100.0f * static_cast<float>(matches) / static_cast<float>(aln_len);
        if (identity < min_percent_identity)
          continue;
      }
    }

    // Parse CIGAR, accumulate coverage
    uint32_t *cigar = bam_get_cigar(aln);
    uint32_t n_cigar = aln->core.n_cigar;
    uint32_t ref_pos = aln->core.pos;

    for (uint32_t i = 0; i < n_cigar; ++i) {
      int op = bam_cigar_op(cigar[i]);
      uint32_t len = bam_cigar_oplen(cigar[i]);

      if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
        uint32_t cov_start = std::max(ref_pos, range_start);
        uint32_t cov_end = std::min(ref_pos + len, range_end);
        if (cov_start < cov_end) {
          for (uint32_t pos = cov_start; pos < cov_end; ++pos) {
            coverage[pos - range_start]++;
          }
        }
        ref_pos += len;
      } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
        ref_pos += len;
      }
      // BAM_CINS, BAM_CSOFT_CLIP: no ref advance
    }
  }

  hts_itr_destroy(iter);
  bam_destroy1(aln);

  // Apply edge exclusion within the range (COMEBin: depth_value[start + edge:end + 1 - edge])
  uint32_t use_start = edge_within_range;
  uint32_t use_end = range_len > edge_within_range ? range_len - edge_within_range : 0;
  if (use_start >= use_end) {
    // Range too short for edge exclusion - use whole range
    use_start = 0;
    use_end = range_len;
  }

  uint64_t count = use_end - use_start;
  if (count == 0)
    return result;

  // Compute mean
  uint64_t sum = 0;
  for (uint32_t i = use_start; i < use_end; ++i) {
    sum += coverage[i];
  }
  double mean = static_cast<double>(sum) / count;

  // Compute population variance (N divisor, matches np.var())
  double var_sum = 0.0;
  for (uint32_t i = use_start; i < use_end; ++i) {
    double diff = coverage[i] - mean;
    var_sum += diff * diff;
  }
  double variance = var_sum / count;  // Population variance (N divisor)

  result.mean = mean;
  result.variance = variance;

  return result;
}

} // namespace amber

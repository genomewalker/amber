#include "chimera_detector.h"
#include "damage_profile.h"
#include <cmath>
#include <algorithm>
#include <htslib/sam.h>
#include <vector>
#include <set>
#include <iostream>
#include <numeric>

namespace amber {

// ============================================================================
// SHARED BAM STATE MANAGEMENT
// ============================================================================

bool SharedBAMState::open(const std::string& bam_file) {
  close();

  bam = sam_open(bam_file.c_str(), "r");
  if (!bam) return false;

  header = sam_hdr_read(bam);
  if (!header) {
    sam_close(bam);
    bam = nullptr;
    return false;
  }

  idx = sam_index_load(bam, bam_file.c_str());
  if (!idx) {
    bam_hdr_destroy(header);
    sam_close(bam);
    header = nullptr;
    bam = nullptr;
    return false;
  }

  return true;
}

void SharedBAMState::close() {
  if (idx) { hts_idx_destroy(idx); idx = nullptr; }
  if (header) { bam_hdr_destroy(header); header = nullptr; }
  if (bam) { sam_close(bam); bam = nullptr; }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

double ChimeraDetector::compute_gc_window(
    const std::string &sequence,
    int start,
    int end) {

  int gc_count = 0;
  int total = 0;

  for (int i = start; i < end && i < static_cast<int>(sequence.length()); ++i) {
    char c = sequence[i];
    if (c == 'G' || c == 'C' || c == 'g' || c == 'c') {
      gc_count++;
    }
    if (c == 'A' || c == 'T' || c == 'G' || c == 'C' ||
        c == 'a' || c == 't' || c == 'g' || c == 'c') {
      total++;
    }
  }

  return (total > 0) ? static_cast<double>(gc_count) / total : 0.0;
}

double ChimeraDetector::compute_signal_at_position(
    const std::vector<int> &data,
    int position,
    int window_size) {

  if (position < 0 || position >= static_cast<int>(data.size())) return 0.0;

  int start = std::max(0, position - window_size / 2);
  int end = std::min(static_cast<int>(data.size()), position + window_size / 2);

  int sum = 0;
  for (int i = start; i < end; ++i) {
    sum += data[i];
  }

  return static_cast<double>(sum) / (end - start);
}

double ChimeraDetector::compute_signal_at_position(
    const std::vector<double> &data,
    int position,
    int window_size) {

  if (position < 0 || position >= static_cast<int>(data.size())) return 0.0;

  int start = std::max(0, position - window_size / 2);
  int end = std::min(static_cast<int>(data.size()), position + window_size / 2);

  double sum = 0.0;
  for (int i = start; i < end; ++i) {
    sum += data[i];
  }

  return sum / (end - start);
}

// ============================================================================
// CIGAR-AWARE DATA EXTRACTION (SINGLE PASS!)
// ============================================================================

ContigAlignmentData ChimeraDetector::extract_alignment_data(
    const std::string &contig_id,
    const std::string &bam_file,
    int contig_length,
    int window_size,
    int damage_positions) {

  ContigAlignmentData data;

  // Initialize vectors
  data.coverage.resize(contig_length, 0);

  // Windowed data
  int num_windows = (contig_length + window_size - 1) / window_size;
  data.alignment_score.resize(num_windows, 0.0);
  data.edit_distance.resize(num_windows, 0.0);
  data.damage_rate_5p.resize(num_windows, 0.0);
  data.damage_rate_3p.resize(num_windows, 0.0);

  // Temporary accumulators for windowed statistics
  std::vector<double> as_sum(num_windows, 0.0);  // Alignment score
  std::vector<double> nm_sum(num_windows, 0.0);  // Edit distance
  std::vector<int> read_count(num_windows, 0);
  std::vector<int> damage_ct_5p(num_windows, 0);  // C in damaged 5' regions
  std::vector<int> damage_t_5p(num_windows, 0);   // T in damaged 5' regions
  std::vector<int> damage_ga_3p(num_windows, 0);  // G in damaged 3' regions
  std::vector<int> damage_a_3p(num_windows, 0);   // A in damaged 3' regions

  // Open BAM file
  samFile *bam = sam_open(bam_file.c_str(), "r");
  if (!bam) {
    std::cerr << "Warning: Cannot open BAM file for chimera detection\n";
    return data;
  }

  bam_hdr_t *header = sam_hdr_read(bam);
  if (!header) {
    sam_close(bam);
    return data;
  }

  hts_idx_t *idx = sam_index_load(bam, bam_file.c_str());
  if (!idx) {
    bam_hdr_destroy(header);
    sam_close(bam);
    return data;
  }

  int tid = bam_name2id(header, contig_id.c_str());
  if (tid < 0) {
    hts_idx_destroy(idx);
    bam_hdr_destroy(header);
    sam_close(bam);
    return data;
  }

  // Single pass through BAM
  hts_itr_t *iter = sam_itr_queryi(idx, tid, 0, contig_length);
  if (!iter) {
    hts_idx_destroy(idx);
    bam_hdr_destroy(header);
    sam_close(bam);
    return data;
  }

  bam1_t *aln = bam_init1();
  while (sam_itr_next(bam, iter, aln) >= 0) {
    // Skip unmapped, secondary, supplementary, QC fail
    if (aln->core.flag & (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY | BAM_FQCFAIL)) continue;

    uint32_t *cigar = bam_get_cigar(aln);
    int n_cigar = aln->core.n_cigar;
    int read_len = aln->core.l_qseq;
    bool is_reverse = (aln->core.flag & BAM_FREVERSE);
    uint8_t *seq_data = bam_get_seq(aln);

    // Extract alignment score (AS) and edit distance (NM) from aux tags
    int alignment_score = 0;
    int edit_dist = 0;
    uint8_t *as_tag = bam_aux_get(aln, "AS");
    uint8_t *nm_tag = bam_aux_get(aln, "NM");
    if (as_tag) alignment_score = bam_aux2i(as_tag);
    if (nm_tag) edit_dist = bam_aux2i(nm_tag);

    // Parse CIGAR to get accurate coverage
    int query_pos = 0;
    int ref_pos = aln->core.pos;
    int first_match_pos = -1;
    int last_match_pos = -1;

    for (int i = 0; i < n_cigar; ++i) {
      int op_len = bam_cigar_oplen(cigar[i]);
      int op = bam_cigar_op(cigar[i]);

      if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
        // Update coverage
        for (int j = 0; j < op_len; ++j) {
          int pos = ref_pos + j;
          if (pos >= 0 && pos < contig_length) {
            data.coverage[pos]++;

            if (first_match_pos < 0) first_match_pos = pos;
            last_match_pos = pos;

            // Accumulate damage statistics for this position's window
            int window_idx = pos / window_size;
            if (window_idx < num_windows) {
              // Get base
              int qpos = query_pos + j;
              if (qpos < read_len) {
                char base = seq_nt16_str[bam_seqi(seq_data, qpos)];

                // Check if in damaged region
                int dist_from_5p = qpos + 1;
                int dist_from_3p = read_len - qpos;

                // 5' end damage (C→T)
                if (!is_reverse && dist_from_5p <= damage_positions) {
                  if (base == 'C') damage_ct_5p[window_idx]++;
                  else if (base == 'T') damage_t_5p[window_idx]++;
                }

                // 3' end damage (G→A) - only for reverse reads in single-end
                if (is_reverse && dist_from_3p <= damage_positions) {
                  if (base == 'G') damage_ga_3p[window_idx]++;
                  else if (base == 'A') damage_a_3p[window_idx]++;
                }
              }
            }
          }
        }
        query_pos += op_len;
        ref_pos += op_len;

      } else if (op == BAM_CINS || op == BAM_CSOFT_CLIP) {
        query_pos += op_len;
      } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
        ref_pos += op_len;
      } else if (op == BAM_CHARD_CLIP) {
        // Hard clip: already removed from sequence
        continue;
      }
    }

    // Accumulate AS/NM for windows covered by this read
    if (first_match_pos >= 0 && last_match_pos >= 0) {
      int start_window = first_match_pos / window_size;
      int end_window = last_match_pos / window_size;
      for (int w = start_window; w <= end_window && w < num_windows; ++w) {
        as_sum[w] += alignment_score;
        nm_sum[w] += edit_dist;
        read_count[w]++;
      }
    }
  }

  bam_destroy1(aln);
  hts_itr_destroy(iter);
  hts_idx_destroy(idx);
  bam_hdr_destroy(header);
  sam_close(bam);

  // Compute averages for windowed statistics
  for (int w = 0; w < num_windows; ++w) {
    if (read_count[w] > 0) {
      data.alignment_score[w] = as_sum[w] / read_count[w];
      data.edit_distance[w] = nm_sum[w] / read_count[w];
    }

    // Compute damage rates
    int total_ct = damage_ct_5p[w] + damage_t_5p[w];
    if (total_ct > 0) {
      data.damage_rate_5p[w] = static_cast<double>(damage_t_5p[w]) / total_ct;
    }

    int total_ga = damage_ga_3p[w] + damage_a_3p[w];
    if (total_ga > 0) {
      data.damage_rate_3p[w] = static_cast<double>(damage_a_3p[w]) / total_ga;
    }
  }

  return data;
}

// ============================================================================
// OPTIMIZED ALIGNMENT DATA EXTRACTION (SHARED BAM STATE)
// ============================================================================

ContigAlignmentData ChimeraDetector::extract_alignment_data_fast(
    const std::string &contig_id,
    SharedBAMState *bam_state,
    int contig_length,
    int window_size,
    int damage_positions) {

  ContigAlignmentData data;
  data.coverage.resize(contig_length, 0);

  int num_windows = (contig_length + window_size - 1) / window_size;
  data.alignment_score.resize(num_windows, 0.0);
  data.edit_distance.resize(num_windows, 0.0);
  data.damage_rate_5p.resize(num_windows, 0.0);
  data.damage_rate_3p.resize(num_windows, 0.0);
  data.damage_count_5p.resize(num_windows, 0);
  data.damage_count_3p.resize(num_windows, 0);
  data.read_count.resize(num_windows, 0);

  // Per-window Bayesian damage model data (MD tag derived)
  const int max_damage_pos = damage_positions + 5;  // Extra buffer for model fitting
  data.n_5p.resize(num_windows, std::vector<uint64_t>(max_damage_pos, 0));
  data.k_5p.resize(num_windows, std::vector<uint64_t>(max_damage_pos, 0));
  data.n_3p.resize(num_windows, std::vector<uint64_t>(max_damage_pos, 0));
  data.k_3p.resize(num_windows, std::vector<uint64_t>(max_damage_pos, 0));

  if (!bam_state || !bam_state->bam || !bam_state->header || !bam_state->idx) {
    return data;
  }

  std::vector<double> as_sum(num_windows, 0.0);
  std::vector<double> nm_sum(num_windows, 0.0);
  std::vector<int> damage_ct_5p(num_windows, 0);
  std::vector<int> damage_t_5p(num_windows, 0);
  std::vector<int> damage_ga_3p(num_windows, 0);
  std::vector<int> damage_a_3p(num_windows, 0);

  int tid = bam_name2id(bam_state->header, contig_id.c_str());
  if (tid < 0) return data;

  hts_itr_t *iter = sam_itr_queryi(bam_state->idx, tid, 0, contig_length);
  if (!iter) return data;

  bam1_t *aln = bam_init1();
  while (sam_itr_next(bam_state->bam, iter, aln) >= 0) {
    if (aln->core.flag & (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY | BAM_FQCFAIL)) continue;

    uint32_t *cigar = bam_get_cigar(aln);
    int n_cigar = aln->core.n_cigar;
    int read_len = aln->core.l_qseq;
    bool is_reverse = (aln->core.flag & BAM_FREVERSE);
    uint8_t *seq_data = bam_get_seq(aln);

    int alignment_score = 0;
    int edit_dist = 0;
    uint8_t *as_tag = bam_aux_get(aln, "AS");
    uint8_t *nm_tag = bam_aux_get(aln, "NM");
    if (as_tag) alignment_score = bam_aux2i(as_tag);
    if (nm_tag) edit_dist = bam_aux2i(nm_tag);

    int query_pos = 0;
    int ref_pos = aln->core.pos;
    int first_match_pos = -1;
    int last_match_pos = -1;

    for (int i = 0; i < n_cigar; ++i) {
      int op_len = bam_cigar_oplen(cigar[i]);
      int op = bam_cigar_op(cigar[i]);

      if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
        for (int j = 0; j < op_len; ++j) {
          int pos = ref_pos + j;
          if (pos >= 0 && pos < contig_length) {
            data.coverage[pos]++;
            if (first_match_pos < 0) first_match_pos = pos;
            last_match_pos = pos;

            int window_idx = pos / window_size;
            if (window_idx < num_windows) {
              int qpos = query_pos + j;
              if (qpos < read_len) {
                char base = seq_nt16_str[bam_seqi(seq_data, qpos)];
                int dist_from_5p = qpos + 1;
                int dist_from_3p = read_len - qpos;

                if (!is_reverse && dist_from_5p <= damage_positions) {
                  if (base == 'C') damage_ct_5p[window_idx]++;
                  else if (base == 'T') damage_t_5p[window_idx]++;
                }
                if (is_reverse && dist_from_3p <= damage_positions) {
                  if (base == 'G') damage_ga_3p[window_idx]++;
                  else if (base == 'A') damage_a_3p[window_idx]++;
                }
              }
            }
          }
        }
        query_pos += op_len;
        ref_pos += op_len;
      } else if (op == BAM_CINS || op == BAM_CSOFT_CLIP) {
        query_pos += op_len;
      } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
        ref_pos += op_len;
      }
    }

    if (first_match_pos >= 0 && last_match_pos >= 0) {
      int start_window = first_match_pos / window_size;
      int end_window = last_match_pos / window_size;
      for (int w = start_window; w <= end_window && w < num_windows; ++w) {
        as_sum[w] += alignment_score;
        nm_sum[w] += edit_dist;
        data.read_count[w]++;
      }
    }

    // Parse MD tag for proper damage model accumulation
    uint8_t *md_tag = bam_aux_get(aln, "MD");
    if (md_tag && first_match_pos >= 0) {
      const char *md_str = bam_aux2Z(md_tag);
      if (md_str) {
        int read_pos = 0;  // Position along the read
        const char *p = md_str;

        while (*p) {
          if (isdigit(*p)) {
            // Number of matches
            int matches = 0;
            while (isdigit(*p)) {
              matches = matches * 10 + (*p - '0');
              p++;
            }
            read_pos += matches;
          } else if (*p == '^') {
            // Deletion from reference - skip the deleted bases
            p++;
            while (isalpha(*p)) p++;
          } else if (isalpha(*p)) {
            // Mismatch: *p is the reference base, read has something else
            char ref_base = toupper(*p);
            char read_base = (read_pos < read_len) ?
                toupper(seq_nt16_str[bam_seqi(seq_data, read_pos)]) : 'N';

            // Determine position from 5' and 3' ends (1-based)
            int dist_5p = is_reverse ? (read_len - read_pos) : (read_pos + 1);
            int dist_3p = is_reverse ? (read_pos + 1) : (read_len - read_pos);

            // Determine which window this position maps to
            // Use the center of the read for window assignment
            int center_ref_pos = first_match_pos + (last_match_pos - first_match_pos) / 2;
            int window_idx = center_ref_pos / window_size;

            if (window_idx >= 0 && window_idx < num_windows && dist_5p <= max_damage_pos) {
              int pos_idx = dist_5p - 1;  // Convert to 0-based index

              // 5' end: C→T damage (forward reads) or complement on reverse
              if (!is_reverse) {
                if (ref_base == 'C') {
                  data.n_5p[window_idx][pos_idx]++;
                  if (read_base == 'T') {
                    data.k_5p[window_idx][pos_idx]++;
                  }
                }
              } else {
                // Reverse strand: G in reference appears as C→T on the read
                if (ref_base == 'G') {
                  data.n_5p[window_idx][pos_idx]++;
                  if (read_base == 'A') {
                    data.k_5p[window_idx][pos_idx]++;
                  }
                }
              }
            }

            if (window_idx >= 0 && window_idx < num_windows && dist_3p <= max_damage_pos) {
              int pos_idx = dist_3p - 1;

              // 3' end: G→A damage (forward reads) or complement on reverse
              if (!is_reverse) {
                if (ref_base == 'G') {
                  data.n_3p[window_idx][pos_idx]++;
                  if (read_base == 'A') {
                    data.k_3p[window_idx][pos_idx]++;
                  }
                }
              } else {
                // Reverse strand: C in reference appears as G→A on the read
                if (ref_base == 'C') {
                  data.n_3p[window_idx][pos_idx]++;
                  if (read_base == 'T') {
                    data.k_3p[window_idx][pos_idx]++;
                  }
                }
              }
            }

            read_pos++;
            p++;
          } else {
            p++;
          }
        }

        // Also count opportunities for positions without mismatches
        // Parse CIGAR again to count all C positions at 5' and G positions at 3'
        int qpos = 0;
        for (int ci = 0; ci < n_cigar && qpos < read_len; ++ci) {
          int op_len = bam_cigar_oplen(cigar[ci]);
          int op = bam_cigar_op(cigar[ci]);

          if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
            for (int j = 0; j < op_len && qpos < read_len; ++j) {
              char base = toupper(seq_nt16_str[bam_seqi(seq_data, qpos)]);

              int dist_5p = is_reverse ? (read_len - qpos) : (qpos + 1);
              int dist_3p = is_reverse ? (qpos + 1) : (read_len - qpos);

              int center_ref_pos = first_match_pos + (last_match_pos - first_match_pos) / 2;
              int window_idx = center_ref_pos / window_size;

              if (window_idx >= 0 && window_idx < num_windows) {
                // 5' opportunities (only C bases can show C→T damage)
                if (dist_5p <= max_damage_pos) {
                  int pos_idx = dist_5p - 1;
                  if (!is_reverse && base == 'C') {
                    data.n_5p[window_idx][pos_idx]++;
                  } else if (is_reverse && base == 'G') {
                    data.n_5p[window_idx][pos_idx]++;
                  }
                }

                // 3' opportunities (only G bases can show G→A damage)
                if (dist_3p <= max_damage_pos) {
                  int pos_idx = dist_3p - 1;
                  if (!is_reverse && base == 'G') {
                    data.n_3p[window_idx][pos_idx]++;
                  } else if (is_reverse && base == 'C') {
                    data.n_3p[window_idx][pos_idx]++;
                  }
                }
              }

              qpos++;
            }
          } else if (op == BAM_CINS || op == BAM_CSOFT_CLIP) {
            qpos += op_len;
          }
        }
      }
    }
  }

  bam_destroy1(aln);
  hts_itr_destroy(iter);

  for (int w = 0; w < num_windows; ++w) {
    if (data.read_count[w] > 0) {
      data.alignment_score[w] = as_sum[w] / data.read_count[w];
      data.edit_distance[w] = nm_sum[w] / data.read_count[w];
    }
    int total_ct = damage_ct_5p[w] + damage_t_5p[w];
    data.damage_count_5p[w] = total_ct;
    if (total_ct > 0) {
      data.damage_rate_5p[w] = static_cast<double>(damage_t_5p[w]) / total_ct;
    }
    int total_ga = damage_ga_3p[w] + damage_a_3p[w];
    data.damage_count_3p[w] = total_ga;
    if (total_ga > 0) {
      data.damage_rate_3p[w] = static_cast<double>(damage_a_3p[w]) / total_ga;
    }
  }

  return data;
}

// ============================================================================
// SIGNAL DETECTION METHODS
// ============================================================================

std::vector<int> ChimeraDetector::detect_gc_shifts(
    const std::string &sequence,
    int window_size,
    double min_shift) {

  std::vector<int> breakpoints;
  int len = sequence.length();

  // Slide window and detect sharp transitions
  for (int i = window_size; i < len - window_size; i += window_size / 2) {
    double gc_before = compute_gc_window(sequence, i - window_size, i);
    double gc_after = compute_gc_window(sequence, i, i + window_size);

    double shift = std::abs(gc_after - gc_before);

    if (shift >= min_shift) {
      breakpoints.push_back(i);
    }
  }

  return breakpoints;
}

std::vector<int> ChimeraDetector::detect_coverage_jumps(
    const std::vector<int> &coverage,
    int window_size,
    double min_ratio) {

  std::vector<int> breakpoints;
  int len = coverage.size();

  // Scan for coverage jumps
  for (int i = window_size; i < len - window_size; i += window_size / 2) {
    // Compute average coverage in windows
    double cov_before = 0.0;
    for (int j = i - window_size; j < i; ++j) {
      if (j >= 0) cov_before += coverage[j];
    }
    cov_before /= window_size;

    double cov_after = 0.0;
    for (int j = i; j < i + window_size; ++j) {
      if (j < len) cov_after += coverage[j];
    }
    cov_after /= window_size;

    // Check for significant jump
    if (cov_before > 0 && cov_after > 0) {
      double ratio = std::max(cov_after / cov_before, cov_before / cov_after);
      if (ratio >= min_ratio) {
        breakpoints.push_back(i);
      }
    }
  }

  return breakpoints;
}

std::vector<int> ChimeraDetector::detect_alignment_score_drops(
    const std::vector<double> &alignment_score,
    int window_size,
    double min_drop_pct) {

  std::vector<int> breakpoints;
  int num_windows = alignment_score.size();

  for (int w = 1; w < num_windows - 1; ++w) {
    double as_before = alignment_score[w - 1];
    double as_after = alignment_score[w + 1];
    double as_current = alignment_score[w];

    // Check for percentage drop
    double avg_neighbors = (as_before + as_after) / 2.0;
    if (avg_neighbors > 0) {
      double drop_pct = (avg_neighbors - as_current) / avg_neighbors;
      if (drop_pct >= min_drop_pct) {
        breakpoints.push_back(w * window_size);
      }
    }
  }

  return breakpoints;
}

std::vector<int> ChimeraDetector::detect_edit_distance_spikes(
    const std::vector<double> &edit_distance,
    int window_size,
    double min_spike_ratio) {

  std::vector<int> breakpoints;
  int num_windows = edit_distance.size();

  for (int w = 1; w < num_windows - 1; ++w) {
    double nm_before = edit_distance[w - 1];
    double nm_after = edit_distance[w + 1];
    double nm_current = edit_distance[w];

    // Check for spike
    double avg_neighbors = (nm_before + nm_after) / 2.0;
    if (avg_neighbors > 0 && nm_current > 0) {
      double spike_ratio = nm_current / avg_neighbors;
      if (spike_ratio >= min_spike_ratio) {
        breakpoints.push_back(w * window_size);
      }
    }
  }

  return breakpoints;
}

std::vector<int> ChimeraDetector::detect_damage_shifts(
    const std::vector<double> &damage_5p,
    const std::vector<double> &damage_3p,
    int window_size,
    double min_shift) {

  std::vector<int> breakpoints;
  int num_windows = damage_5p.size();

  for (int w = 1; w < num_windows - 1; ++w) {
    // Check 5' damage shift
    double damage_before = damage_5p[w - 1];
    double damage_after = damage_5p[w + 1];
    double shift_5p = std::abs(damage_after - damage_before);

    // Check 3' damage shift
    double damage3_before = damage_3p[w - 1];
    double damage3_after = damage_3p[w + 1];
    double shift_3p = std::abs(damage3_after - damage3_before);

    // Either 5' or 3' shift indicates potential chimera
    if (shift_5p >= min_shift || shift_3p >= min_shift) {
      breakpoints.push_back(w * window_size);
    }
  }

  return breakpoints;
}

std::vector<int> ChimeraDetector::detect_damage_shifts_robust(
    const std::vector<double> &damage_5p,
    const std::vector<double> &damage_3p,
    const std::vector<int> &count_5p,
    const std::vector<int> &count_3p,
    int window_size,
    double min_shift,
    int min_observations) {

  std::vector<int> breakpoints;
  int num_windows = damage_5p.size();

  // Need at least 7 windows for 5-window smoothing with gap
  if (num_windows < 7) return breakpoints;

  // Smooth damage rates over 5-window spans (500bp effective at 100bp windows)
  // This provides more stable estimates for detecting real shifts
  auto smooth_damage = [&](const std::vector<double> &rates,
                           const std::vector<int> &counts,
                           int center, int half_span) -> std::pair<double, int> {
    int start = std::max(0, center - half_span);
    int end = std::min(num_windows, center + half_span + 1);

    double weighted_sum = 0.0;
    int total_count = 0;

    for (int i = start; i < end; ++i) {
      weighted_sum += rates[i] * counts[i];
      total_count += counts[i];
    }

    double smoothed_rate = (total_count > 0) ? (weighted_sum / total_count) : 0.0;
    return {smoothed_rate, total_count};
  };

  // Scan with 3-window gap between before/after regions for cleaner signal
  const int half_span = 2;  // 5-window span (center +/- 2)
  const int gap = 1;        // 1-window gap at potential breakpoint

  for (int w = half_span + gap; w < num_windows - half_span - gap; ++w) {
    // Smooth damage in regions before and after the breakpoint
    auto [damage_5p_before, count_5p_before] = smooth_damage(damage_5p, count_5p, w - gap - half_span, half_span);
    auto [damage_5p_after, count_5p_after] = smooth_damage(damage_5p, count_5p, w + gap + half_span, half_span);

    auto [damage_3p_before, count_3p_before] = smooth_damage(damage_3p, count_3p, w - gap - half_span, half_span);
    auto [damage_3p_after, count_3p_after] = smooth_damage(damage_3p, count_3p, w + gap + half_span, half_span);

    // Require minimum observations on BOTH sides for reliable shift detection
    bool has_5p_data = (count_5p_before >= min_observations && count_5p_after >= min_observations);
    bool has_3p_data = (count_3p_before >= min_observations && count_3p_after >= min_observations);

    double shift_5p = has_5p_data ? std::abs(damage_5p_after - damage_5p_before) : 0.0;
    double shift_3p = has_3p_data ? std::abs(damage_3p_after - damage_3p_before) : 0.0;

    if (shift_5p >= min_shift || shift_3p >= min_shift) {
      breakpoints.push_back(w * window_size);
    }
  }

  return breakpoints;
}

// ============================================================================
// MODEL-BASED DAMAGE SHIFT DETECTION (BAYESIAN)
// ============================================================================

std::vector<int> ChimeraDetector::detect_damage_model_shifts(
    const ContigAlignmentData &aln_data,
    int window_size,
    double min_p_ancient_diff,
    double min_amplitude_diff) {

  std::vector<int> breakpoints;
  int num_windows = aln_data.n_5p.size();

  // Need enough windows to pool segments
  if (num_windows < 10) return breakpoints;

  const int segment_size = 5;  // Pool 5 windows (500bp at 100bp windows) per segment
  const int step_size = 5;     // Only test every 5th window position (500bp steps) for efficiency

  // Scan potential breakpoints at coarse resolution
  for (int w = segment_size; w < num_windows - segment_size; w += step_size) {
    // Pool damage counts for segment BEFORE breakpoint
    std::vector<uint64_t> n_5p_before(20, 0), k_5p_before(20, 0);
    std::vector<uint64_t> n_3p_before(20, 0), k_3p_before(20, 0);
    for (int i = w - segment_size; i < w; ++i) {
      for (int p = 0; p < 20 && p < static_cast<int>(aln_data.n_5p[i].size()); ++p) {
        n_5p_before[p] += aln_data.n_5p[i][p];
        k_5p_before[p] += aln_data.k_5p[i][p];
        n_3p_before[p] += aln_data.n_3p[i][p];
        k_3p_before[p] += aln_data.k_3p[i][p];
      }
    }

    // Pool damage counts for segment AFTER breakpoint
    std::vector<uint64_t> n_5p_after(20, 0), k_5p_after(20, 0);
    std::vector<uint64_t> n_3p_after(20, 0), k_3p_after(20, 0);
    for (int i = w; i < w + segment_size && i < num_windows; ++i) {
      for (int p = 0; p < 20 && p < static_cast<int>(aln_data.n_5p[i].size()); ++p) {
        n_5p_after[p] += aln_data.n_5p[i][p];
        k_5p_after[p] += aln_data.k_5p[i][p];
        n_3p_after[p] += aln_data.n_3p[i][p];
        k_3p_after[p] += aln_data.k_3p[i][p];
      }
    }

    // Check if we have enough data on both sides
    uint64_t total_before = 0, total_after = 0;
    for (int p = 0; p < 10; ++p) {
      total_before += n_5p_before[p] + n_3p_before[p];
      total_after += n_5p_after[p] + n_3p_after[p];
    }

    if (total_before < 100 || total_after < 100) continue;

    // Fit damage models to both segments
    auto model_before = fit_damage_model(n_5p_before, k_5p_before, n_3p_before, k_3p_before, 0.5);
    auto model_after = fit_damage_model(n_5p_after, k_5p_after, n_3p_after, k_3p_after, 0.5);

    // Check for significant differences
    double p_ancient_diff = std::abs(model_after.p_ancient - model_before.p_ancient);
    double amplitude_5p_diff = std::abs(model_after.curve_5p.amplitude - model_before.curve_5p.amplitude);
    double amplitude_3p_diff = std::abs(model_after.curve_3p.amplitude - model_before.curve_3p.amplitude);
    double amplitude_diff = std::max(amplitude_5p_diff, amplitude_3p_diff);

    if (p_ancient_diff >= min_p_ancient_diff || amplitude_diff >= min_amplitude_diff) {
      breakpoints.push_back(w * window_size);
    }
  }

  return breakpoints;
}

void ChimeraDetector::compute_damage_model_evidence(
    const ContigAlignmentData &aln_data,
    int breakpoint_window,
    double &p_ancient_before,
    double &p_ancient_after,
    double &amplitude_diff) {

  int num_windows = aln_data.n_5p.size();
  const int segment_size = 5;

  p_ancient_before = 0.5;
  p_ancient_after = 0.5;
  amplitude_diff = 0.0;

  if (breakpoint_window < segment_size || breakpoint_window >= num_windows - segment_size) {
    return;
  }

  // Pool damage counts for segment BEFORE breakpoint
  std::vector<uint64_t> n_5p_before(20, 0), k_5p_before(20, 0);
  std::vector<uint64_t> n_3p_before(20, 0), k_3p_before(20, 0);
  for (int i = breakpoint_window - segment_size; i < breakpoint_window; ++i) {
    for (int p = 0; p < 20 && p < static_cast<int>(aln_data.n_5p[i].size()); ++p) {
      n_5p_before[p] += aln_data.n_5p[i][p];
      k_5p_before[p] += aln_data.k_5p[i][p];
      n_3p_before[p] += aln_data.n_3p[i][p];
      k_3p_before[p] += aln_data.k_3p[i][p];
    }
  }

  // Pool damage counts for segment AFTER breakpoint
  std::vector<uint64_t> n_5p_after(20, 0), k_5p_after(20, 0);
  std::vector<uint64_t> n_3p_after(20, 0), k_3p_after(20, 0);
  for (int i = breakpoint_window; i < breakpoint_window + segment_size && i < num_windows; ++i) {
    for (int p = 0; p < 20 && p < static_cast<int>(aln_data.n_5p[i].size()); ++p) {
      n_5p_after[p] += aln_data.n_5p[i][p];
      k_5p_after[p] += aln_data.k_5p[i][p];
      n_3p_after[p] += aln_data.n_3p[i][p];
      k_3p_after[p] += aln_data.k_3p[i][p];
    }
  }

  // Check if we have enough data
  uint64_t total_before = 0, total_after = 0;
  for (int p = 0; p < 10; ++p) {
    total_before += n_5p_before[p] + n_3p_before[p];
    total_after += n_5p_after[p] + n_3p_after[p];
  }

  if (total_before < 50 || total_after < 50) {
    return;
  }

  // Fit models
  auto model_before = fit_damage_model(n_5p_before, k_5p_before, n_3p_before, k_3p_before, 0.5);
  auto model_after = fit_damage_model(n_5p_after, k_5p_after, n_3p_after, k_3p_after, 0.5);

  p_ancient_before = model_before.p_ancient;
  p_ancient_after = model_after.p_ancient;

  double amplitude_5p_diff = std::abs(model_after.curve_5p.amplitude - model_before.curve_5p.amplitude);
  double amplitude_3p_diff = std::abs(model_after.curve_3p.amplitude - model_before.curve_3p.amplitude);
  amplitude_diff = std::max(amplitude_5p_diff, amplitude_3p_diff);
}

// ============================================================================
// FRACTAL-BASED BREAKPOINT DETECTION
// ============================================================================

std::vector<WindowFractalMetrics> ChimeraDetector::compute_window_fractals(
    const std::string &sequence,
    int window_size,
    int step_size) {

  std::vector<WindowFractalMetrics> metrics;
  int seq_len = sequence.length();

  if (seq_len < window_size) {
    return metrics;
  }

  for (int start = 0; start + window_size <= seq_len; start += step_size) {
    std::string window = sequence.substr(start, window_size);

    WindowFractalMetrics m;

    // Compute CGR-DFA features (Hurst exponent)
    auto cgr_dfa = FractalFeatures::extract_cgr_dfa_features(window);
    m.hurst = (cgr_dfa.hurst_x + cgr_dfa.hurst_y) / 2.0;

    // Compute complexity features (LZ and entropy)
    auto complexity = FractalFeatures::extract_complexity_features(window);
    m.lz_complexity = complexity.lempel_ziv_complexity[0];
    m.entropy = complexity.entropy_scaling[2];

    // Compute CGR features (lacunarity)
    auto cgr = FractalFeatures::extract_cgr_features(window);
    m.cgr_lacunarity = cgr.lacunarity;

    metrics.push_back(m);
  }

  return metrics;
}

std::vector<int> ChimeraDetector::detect_hurst_discontinuities(
    const std::vector<WindowFractalMetrics> &metrics,
    int window_size,
    double min_jump) {

  std::vector<int> breakpoints;
  int n = metrics.size();

  for (int i = 1; i < n - 1; ++i) {
    double hurst_before = metrics[i - 1].hurst;
    double hurst_after = metrics[i + 1].hurst;
    double jump = std::abs(hurst_after - hurst_before);

    if (jump >= min_jump) {
      breakpoints.push_back(i * (window_size / 2));
    }
  }

  return breakpoints;
}

std::vector<int> ChimeraDetector::detect_lz_discontinuities(
    const std::vector<WindowFractalMetrics> &metrics,
    int window_size,
    double min_jump) {

  std::vector<int> breakpoints;
  int n = metrics.size();

  for (int i = 1; i < n - 1; ++i) {
    double lz_before = metrics[i - 1].lz_complexity;
    double lz_after = metrics[i + 1].lz_complexity;
    double jump = std::abs(lz_after - lz_before);

    if (jump >= min_jump) {
      breakpoints.push_back(i * (window_size / 2));
    }
  }

  return breakpoints;
}

std::vector<int> ChimeraDetector::detect_entropy_discontinuities(
    const std::vector<WindowFractalMetrics> &metrics,
    int window_size,
    double min_jump) {

  std::vector<int> breakpoints;
  int n = metrics.size();

  for (int i = 1; i < n - 1; ++i) {
    double entropy_before = metrics[i - 1].entropy;
    double entropy_after = metrics[i + 1].entropy;
    double jump = std::abs(entropy_after - entropy_before);

    if (jump >= min_jump) {
      breakpoints.push_back(i * (window_size / 2));
    }
  }

  return breakpoints;
}

double ChimeraDetector::compute_cgr_distance(
    const std::string &seq1,
    const std::string &seq2) {

  if (seq1.empty() || seq2.empty()) return 0.0;

  auto cgr1 = FractalFeatures::extract_cgr_geometric(seq1);
  auto cgr2 = FractalFeatures::extract_cgr_geometric(seq2);

  // Euclidean distance in normalized CGR feature space
  double dist_sq = 0.0;
  for (size_t i = 0; i < cgr1.normalized_vector.size(); ++i) {
    double diff = cgr1.normalized_vector[i] - cgr2.normalized_vector[i];
    dist_sq += diff * diff;
  }

  return std::sqrt(dist_sq);
}

std::string ChimeraDetector::determine_primary_evidence(const ChimeraSignal &sig) {
  double max_contrib = 0.0;
  std::string primary = "unknown";

  // Weights match the updated confidence scoring
  double gc_contrib = std::min(1.0, sig.gc_shift / 0.15) * 0.20;
  double cov_contrib = std::min(1.0, (sig.coverage_jump - 1.0) / 2.0) * 0.25;
  double as_contrib = std::min(1.0, sig.alignment_score_drop) * 0.15;
  double nm_contrib = std::min(1.0, (sig.edit_distance_spike - 1.0) / 1.0) * 0.12;
  double damage_contrib = std::min(1.0, sig.damage_shift / 0.10) * 0.08;
  double hurst_contrib = std::min(1.0, sig.hurst_discontinuity / 0.15) * 0.03;
  double lz_contrib = std::min(1.0, sig.lz_discontinuity / 0.10) * 0.03;
  double cgr_contrib = std::min(1.0, sig.cgr_shift / 0.5) * 0.03;

  // Model-based damage evidence
  double p_ancient_diff = std::abs(sig.p_ancient_after - sig.p_ancient_before);
  double model_damage_contrib = std::min(1.0, p_ancient_diff / 0.5) * 0.10 +
                                std::min(1.0, sig.amplitude_diff / 0.10) * 0.10;

  if (gc_contrib > max_contrib) { max_contrib = gc_contrib; primary = "gc_shift"; }
  if (cov_contrib > max_contrib) { max_contrib = cov_contrib; primary = "coverage_jump"; }
  if (as_contrib > max_contrib) { max_contrib = as_contrib; primary = "alignment_score"; }
  if (nm_contrib > max_contrib) { max_contrib = nm_contrib; primary = "edit_distance"; }
  if (damage_contrib > max_contrib) { max_contrib = damage_contrib; primary = "damage_pattern"; }
  if (model_damage_contrib > max_contrib) { max_contrib = model_damage_contrib; primary = "damage_model"; }
  if (hurst_contrib > max_contrib) { max_contrib = hurst_contrib; primary = "hurst_discontinuity"; }
  if (lz_contrib > max_contrib) { max_contrib = lz_contrib; primary = "lz_complexity"; }
  if (cgr_contrib > max_contrib) { max_contrib = cgr_contrib; primary = "cgr_shift"; }

  return primary;
}

// ============================================================================
// MAIN DETECTION METHODS
// ============================================================================

std::vector<ChimeraSignal> ChimeraDetector::detect_chimeras(
    const std::string &contig_id,
    const std::string &sequence,
    const std::string &bam_file,
    const std::string &damage_file) {

  (void)damage_file; // Legacy parameter, not used in current implementation
  // Old version - loads damage profile (not needed for single-pass approach)
  return detect_chimeras(contig_id, sequence, bam_file, nullptr);
}

std::vector<ChimeraSignal> ChimeraDetector::detect_chimeras(
    const std::string &contig_id,
    const std::string &sequence,
    const std::string &bam_file,
    const DamageProfile *damage_profile) {

  (void)damage_profile; // Reserved for damage-aware chimera detection
  std::vector<ChimeraSignal> signals;
  int contig_length = sequence.length();

  // STEP 1: Extract all alignment data in single BAM pass (CIGAR-aware!)
  auto aln_data = extract_alignment_data(contig_id, bam_file, contig_length, 100);

  // STEP 2: Detect breakpoints using original signals
  auto gc_breakpoints = detect_gc_shifts(sequence, 500, 0.1);
  auto coverage_breakpoints = detect_coverage_jumps(aln_data.coverage, 500, 2.0);
  auto as_breakpoints = detect_alignment_score_drops(aln_data.alignment_score, 100, 0.15);
  auto nm_breakpoints = detect_edit_distance_spikes(aln_data.edit_distance, 100, 1.5);
  auto damage_breakpoints = detect_damage_shifts(
      aln_data.damage_rate_5p, aln_data.damage_rate_3p, 100, 0.05);

  // STEP 2b: Compute fractal features along sequence (for longer contigs)
  std::vector<WindowFractalMetrics> fractal_metrics;
  std::vector<int> hurst_breakpoints;
  std::vector<int> lz_breakpoints;
  std::vector<int> entropy_breakpoints;

  if (contig_length >= 3000) {
    fractal_metrics = compute_window_fractals(sequence, 1000, 500);

    if (fractal_metrics.size() >= 3) {
      hurst_breakpoints = detect_hurst_discontinuities(fractal_metrics, 1000, 0.15);
      lz_breakpoints = detect_lz_discontinuities(fractal_metrics, 1000, 0.10);
      entropy_breakpoints = detect_entropy_discontinuities(fractal_metrics, 1000, 0.15);
    }
  }

  // STEP 3: Merge all breakpoint candidates
  std::set<int> all_breakpoints;
  for (int pos : gc_breakpoints) all_breakpoints.insert(pos);
  for (int pos : coverage_breakpoints) all_breakpoints.insert(pos);
  for (int pos : as_breakpoints) all_breakpoints.insert(pos);
  for (int pos : nm_breakpoints) all_breakpoints.insert(pos);
  for (int pos : damage_breakpoints) all_breakpoints.insert(pos);
  for (int pos : hurst_breakpoints) all_breakpoints.insert(pos);
  for (int pos : lz_breakpoints) all_breakpoints.insert(pos);
  for (int pos : entropy_breakpoints) all_breakpoints.insert(pos);

  // STEP 4: Compute full signal for each breakpoint
  for (int pos : all_breakpoints) {
    if (pos < 500 || pos >= contig_length - 500) continue; // Skip ends

    ChimeraSignal sig;
    sig.position = pos;

    // GC shift
    sig.gc_shift = std::abs(
        compute_gc_window(sequence, pos - 500, pos) -
        compute_gc_window(sequence, pos, pos + 500));

    // Coverage jump
    double cov_before = compute_signal_at_position(aln_data.coverage, pos - 250, 500);
    double cov_after = compute_signal_at_position(aln_data.coverage, pos + 250, 500);
    sig.coverage_jump = (cov_before > 0 && cov_after > 0) ?
        std::max(cov_after / cov_before, cov_before / cov_after) : 1.0;

    // Alignment score drop
    int window_idx = pos / 100;
    if (window_idx > 0 && window_idx < static_cast<int>(aln_data.alignment_score.size()) - 1) {
      double as_avg = (aln_data.alignment_score[window_idx - 1] + aln_data.alignment_score[window_idx + 1]) / 2.0;
      double as_current = aln_data.alignment_score[window_idx];
      sig.alignment_score_drop = (as_avg > 0) ? std::max(0.0, (as_avg - as_current) / as_avg) : 0.0;
    } else {
      sig.alignment_score_drop = 0.0;
    }

    // Edit distance spike
    if (window_idx > 0 && window_idx < static_cast<int>(aln_data.edit_distance.size()) - 1) {
      double nm_avg = (aln_data.edit_distance[window_idx - 1] + aln_data.edit_distance[window_idx + 1]) / 2.0;
      double nm_current = aln_data.edit_distance[window_idx];
      sig.edit_distance_spike = (nm_avg > 0 && nm_current > 0) ? (nm_current / nm_avg) : 1.0;
    } else {
      sig.edit_distance_spike = 1.0;
    }

    // Damage shift (ancient DNA specific!)
    if (window_idx > 0 && window_idx < static_cast<int>(aln_data.damage_rate_5p.size()) - 1) {
      double damage_5p_before = aln_data.damage_rate_5p[window_idx - 1];
      double damage_5p_after = aln_data.damage_rate_5p[window_idx + 1];
      double damage_3p_before = aln_data.damage_rate_3p[window_idx - 1];
      double damage_3p_after = aln_data.damage_rate_3p[window_idx + 1];

      sig.damage_shift = std::max(
          std::abs(damage_5p_after - damage_5p_before),
          std::abs(damage_3p_after - damage_3p_before));
    } else {
      sig.damage_shift = 0.0;
    }

    // Fractal signals (for contigs with sufficient length)
    if (!fractal_metrics.empty()) {
      int fractal_idx = pos / 500;
      if (fractal_idx > 0 && fractal_idx < static_cast<int>(fractal_metrics.size()) - 1) {
        sig.hurst_discontinuity = std::abs(
            fractal_metrics[fractal_idx + 1].hurst -
            fractal_metrics[fractal_idx - 1].hurst);
        sig.lz_discontinuity = std::abs(
            fractal_metrics[fractal_idx + 1].lz_complexity -
            fractal_metrics[fractal_idx - 1].lz_complexity);
        sig.entropy_discontinuity = std::abs(
            fractal_metrics[fractal_idx + 1].entropy -
            fractal_metrics[fractal_idx - 1].entropy);
      }

      // CGR distance between flanking windows
      if (pos >= 1000 && pos + 1000 <= contig_length) {
        std::string window_before = sequence.substr(pos - 1000, 1000);
        std::string window_after = sequence.substr(pos, 1000);
        sig.cgr_shift = compute_cgr_distance(window_before, window_after);
      }
    }

    // Weighted confidence score (updated with fractal signals)
    // Weights sum to 1.0, rebalanced to include fractal features
    sig.confidence =
        0.15 * std::min(1.0, sig.gc_shift / 0.15) +                    // GC shift
        0.20 * std::min(1.0, (sig.coverage_jump - 1.0) / 2.0) +        // Coverage jump
        0.10 * std::min(1.0, sig.alignment_score_drop) +               // AS drop
        0.08 * std::min(1.0, (sig.edit_distance_spike - 1.0) / 1.0) +  // NM spike
        0.17 * std::min(1.0, sig.damage_shift / 0.10) +                // Damage shift
        0.10 * std::min(1.0, sig.hurst_discontinuity / 0.15) +         // Hurst jump
        0.10 * std::min(1.0, sig.lz_discontinuity / 0.10) +            // LZ complexity jump
        0.10 * std::min(1.0, sig.cgr_shift / 0.5);                     // CGR distance

    sig.confidence = std::min(1.0, sig.confidence);

    // Determine primary evidence
    sig.primary_evidence = determine_primary_evidence(sig);

    if (sig.confidence > 0.5) {
      signals.push_back(sig);
    }
  }

  // Sort by confidence
  std::sort(signals.begin(), signals.end(),
           [](const ChimeraSignal &a, const ChimeraSignal &b) {
             return a.confidence > b.confidence;
           });

  return signals;
}

// ============================================================================
// OPTIMIZED DETECTION (SHARED BAM STATE)
// ============================================================================

std::vector<ChimeraSignal> ChimeraDetector::detect_chimeras_fast(
    const std::string &contig_id,
    const std::string &sequence,
    SharedBAMState *bam_state,
    bool use_fractal_features,
    int damage_positions) {

  std::vector<ChimeraSignal> signals;
  int contig_length = sequence.length();

  // Use shared BAM state for faster extraction
  auto aln_data = extract_alignment_data_fast(contig_id, bam_state, contig_length, 100, damage_positions);

  // Detect breakpoints using alignment signals
  auto gc_breakpoints = detect_gc_shifts(sequence, 500, 0.1);
  auto coverage_breakpoints = detect_coverage_jumps(aln_data.coverage, 500, 2.0);
  auto as_breakpoints = detect_alignment_score_drops(aln_data.alignment_score, 100, 0.15);
  auto nm_breakpoints = detect_edit_distance_spikes(aln_data.edit_distance, 100, 1.5);

  // Use robust damage detection with smoothing and minimum count filtering
  auto damage_breakpoints = detect_damage_shifts_robust(
      aln_data.damage_rate_5p, aln_data.damage_rate_3p,
      aln_data.damage_count_5p, aln_data.damage_count_3p,
      100, 0.08, 30);

  // Model-based damage detection using Bayesian fitting
  auto model_damage_breakpoints = detect_damage_model_shifts(aln_data, 100, 0.3, 0.05);

  // Fractal features (optional, expensive)
  std::vector<WindowFractalMetrics> fractal_metrics;
  std::vector<int> hurst_breakpoints;
  std::vector<int> lz_breakpoints;
  std::vector<int> entropy_breakpoints;

  if (use_fractal_features && contig_length >= 3000) {
    fractal_metrics = compute_window_fractals(sequence, 1000, 500);
    if (fractal_metrics.size() >= 3) {
      hurst_breakpoints = detect_hurst_discontinuities(fractal_metrics, 1000, 0.15);
      lz_breakpoints = detect_lz_discontinuities(fractal_metrics, 1000, 0.10);
      entropy_breakpoints = detect_entropy_discontinuities(fractal_metrics, 1000, 0.15);
    }
  }

  // Merge breakpoint candidates
  std::set<int> all_breakpoints;
  for (int pos : gc_breakpoints) all_breakpoints.insert(pos);
  for (int pos : coverage_breakpoints) all_breakpoints.insert(pos);
  for (int pos : as_breakpoints) all_breakpoints.insert(pos);
  for (int pos : nm_breakpoints) all_breakpoints.insert(pos);
  for (int pos : damage_breakpoints) all_breakpoints.insert(pos);
  for (int pos : model_damage_breakpoints) all_breakpoints.insert(pos);
  for (int pos : hurst_breakpoints) all_breakpoints.insert(pos);
  for (int pos : lz_breakpoints) all_breakpoints.insert(pos);
  for (int pos : entropy_breakpoints) all_breakpoints.insert(pos);

  // Compute signals at each breakpoint
  for (int pos : all_breakpoints) {
    if (pos < 500 || pos >= contig_length - 500) continue;

    ChimeraSignal sig;
    sig.position = pos;

    sig.gc_shift = std::abs(
        compute_gc_window(sequence, pos - 500, pos) -
        compute_gc_window(sequence, pos, pos + 500));

    double cov_before = compute_signal_at_position(aln_data.coverage, pos - 250, 500);
    double cov_after = compute_signal_at_position(aln_data.coverage, pos + 250, 500);
    sig.coverage_jump = (cov_before > 0 && cov_after > 0) ?
        std::max(cov_after / cov_before, cov_before / cov_after) : 1.0;

    int window_idx = pos / 100;
    if (window_idx > 0 && window_idx < static_cast<int>(aln_data.alignment_score.size()) - 1) {
      double as_avg = (aln_data.alignment_score[window_idx - 1] + aln_data.alignment_score[window_idx + 1]) / 2.0;
      double as_current = aln_data.alignment_score[window_idx];
      sig.alignment_score_drop = (as_avg > 0) ? std::max(0.0, (as_avg - as_current) / as_avg) : 0.0;
    }

    if (window_idx > 0 && window_idx < static_cast<int>(aln_data.edit_distance.size()) - 1) {
      double nm_avg = (aln_data.edit_distance[window_idx - 1] + aln_data.edit_distance[window_idx + 1]) / 2.0;
      double nm_current = aln_data.edit_distance[window_idx];
      sig.edit_distance_spike = (nm_avg > 0 && nm_current > 0) ? (nm_current / nm_avg) : 1.0;
    }

    if (window_idx > 0 && window_idx < static_cast<int>(aln_data.damage_rate_5p.size()) - 1) {
      double damage_5p_before = aln_data.damage_rate_5p[window_idx - 1];
      double damage_5p_after = aln_data.damage_rate_5p[window_idx + 1];
      double damage_3p_before = aln_data.damage_rate_3p[window_idx - 1];
      double damage_3p_after = aln_data.damage_rate_3p[window_idx + 1];
      sig.damage_shift = std::max(
          std::abs(damage_5p_after - damage_5p_before),
          std::abs(damage_3p_after - damage_3p_before));
    }

    // Model-based damage evidence (Bayesian)
    compute_damage_model_evidence(aln_data, window_idx,
        sig.p_ancient_before, sig.p_ancient_after, sig.amplitude_diff);

    // Fractal signals (only if computed)
    if (!fractal_metrics.empty()) {
      int fractal_idx = pos / 500;
      if (fractal_idx > 0 && fractal_idx < static_cast<int>(fractal_metrics.size()) - 1) {
        sig.hurst_discontinuity = std::abs(
            fractal_metrics[fractal_idx + 1].hurst -
            fractal_metrics[fractal_idx - 1].hurst);
        sig.lz_discontinuity = std::abs(
            fractal_metrics[fractal_idx + 1].lz_complexity -
            fractal_metrics[fractal_idx - 1].lz_complexity);
        sig.entropy_discontinuity = std::abs(
            fractal_metrics[fractal_idx + 1].entropy -
            fractal_metrics[fractal_idx - 1].entropy);
      }
      if (pos >= 1000 && pos + 1000 <= contig_length) {
        std::string window_before = sequence.substr(pos - 1000, 1000);
        std::string window_after = sequence.substr(pos, 1000);
        sig.cgr_shift = compute_cgr_distance(window_before, window_after);
      }
    }

    // Multi-signal voting: count independent signal families that triggered
    int signal_families = 0;
    bool gc_triggered = (sig.gc_shift >= 0.08);
    bool coverage_triggered = (sig.coverage_jump >= 1.5);
    bool alignment_triggered = (sig.alignment_score_drop >= 0.10 || sig.edit_distance_spike >= 1.3);
    bool damage_triggered = (sig.damage_shift >= 0.06);
    bool model_damage_triggered = (std::abs(sig.p_ancient_after - sig.p_ancient_before) >= 0.2 ||
                                   sig.amplitude_diff >= 0.03);

    if (gc_triggered) signal_families++;
    if (coverage_triggered) signal_families++;
    if (alignment_triggered) signal_families++;
    if (damage_triggered || model_damage_triggered) signal_families++;

    // Confidence score - includes model-based damage evidence
    // Base signals (always computed)
    double gc_contrib = std::min(1.0, sig.gc_shift / 0.15) * 0.20;
    double cov_contrib = std::min(1.0, (sig.coverage_jump - 1.0) / 2.0) * 0.25;
    double as_contrib = std::min(1.0, sig.alignment_score_drop) * 0.15;
    double nm_contrib = std::min(1.0, (sig.edit_distance_spike - 1.0) / 1.0) * 0.12;
    double damage_contrib = std::min(1.0, sig.damage_shift / 0.10) * 0.08;

    // Model-based damage contribution (more reliable than heuristic damage_shift)
    double p_ancient_diff = std::abs(sig.p_ancient_after - sig.p_ancient_before);
    double model_damage_contrib = 0.0;
    model_damage_contrib += std::min(1.0, p_ancient_diff / 0.5) * 0.10;
    model_damage_contrib += std::min(1.0, sig.amplitude_diff / 0.10) * 0.10;

    double base_score = gc_contrib + cov_contrib + as_contrib + nm_contrib + damage_contrib + model_damage_contrib;

    // Fractal features: critical for distinguishing misassemblies from biological features
    // Prophages/genomic islands have coverage jumps but CONTINUOUS complexity
    // True chimeras have coverage jumps AND DISCONTINUOUS complexity
    bool fractal_discontinuity = false;
    double fractal_score = 0.0;

    if (use_fractal_features && !fractal_metrics.empty()) {
      fractal_score =
          0.10 * std::min(1.0, sig.hurst_discontinuity / 0.15) +
          0.10 * std::min(1.0, sig.lz_discontinuity / 0.10) +
          0.10 * std::min(1.0, sig.cgr_shift / 0.5);

      // Check if complexity metrics show discontinuity
      fractal_discontinuity = (sig.hurst_discontinuity >= 0.08 ||
                               sig.lz_discontinuity >= 0.06 ||
                               sig.cgr_shift >= 0.25);

      if (fractal_discontinuity) {
        signal_families++;
        // Fractal corroboration boosts confidence
        sig.confidence = base_score + fractal_score * 0.4;
      } else {
        // No fractal discontinuity - coverage signals are suspect
        // Biological features (prophages, islands) have coverage jumps but continuous complexity
        // Apply penalty to coverage contribution regardless of other signals
        if (coverage_triggered) {
          cov_contrib *= 0.4;  // 60% penalty for uncorroborated coverage
          base_score = gc_contrib + cov_contrib + as_contrib + nm_contrib + damage_contrib + model_damage_contrib;
        }
        sig.confidence = base_score;
      }
    } else {
      // Without fractal analysis, coverage signals are less reliable
      if (coverage_triggered) {
        cov_contrib *= 0.5;
        base_score = gc_contrib + cov_contrib + as_contrib + nm_contrib + damage_contrib + model_damage_contrib;
      }
      sig.confidence = base_score;
    }

    sig.confidence = std::min(1.0, sig.confidence);
    sig.primary_evidence = determine_primary_evidence(sig);

    // Conservative voting: require fractal corroboration OR strong multi-signal agreement
    // Without fractal support, need higher confidence threshold
    bool has_fractal_support = fractal_discontinuity;
    double confidence_threshold = has_fractal_support ? 0.50 : 0.60;
    bool passes_voting = (signal_families >= 2 && has_fractal_support) ||
                         (signal_families >= 3) ||
                         (sig.confidence > 0.70);

    if (sig.confidence > confidence_threshold && passes_voting) {
      signals.push_back(sig);
    }
  }

  std::sort(signals.begin(), signals.end(),
           [](const ChimeraSignal &a, const ChimeraSignal &b) {
             return a.confidence > b.confidence;
           });

  return signals;
}

} // namespace amber

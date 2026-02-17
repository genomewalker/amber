#include "damage_profile.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <htslib/hts.h>
#include <htslib/sam.h>
#include <sstream>
#include <Eigen/Dense>

namespace amber {

size_t DamageProfile::total_positions() const {
  size_t s = 0;
  for (auto &kv : per_contig)
    s += kv.second.size();
  return s;
}

double DamageProfile::average_rate() const {
  double sum = 0.0;
  size_t n = 0;
  for (auto &kv : per_contig) {
    for (auto &e : kv.second) {
      sum += e.c_to_t;
      ++n;
    }
  }
  return n ? (sum / n) : 0.0;
}

int DamageProfile::max_position() const {
  int max_pos = 0;
  for (const auto &kv : per_contig) {
    for (const auto &e : kv.second) {
      if (e.pos > max_pos) {
        max_pos = e.pos;
      }
    }
  }
  return max_pos;
}

static inline std::vector<std::string> split_tab(const std::string &s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == '\t' || c == ' ') {
      if (!cur.empty()) {
        out.push_back(cur);
        cur.clear();
      }
      if (c == '\t')
        continue;
    } else
      cur.push_back(c);
  }
  if (!cur.empty())
    out.push_back(cur);
  return out;
}

// helper: run a decompressor command and return FILE* (popen) or nullptr
static FILE *open_decompress_pipe(const std::string &fn) {
  // try bgzip, gzip, zcat in that order
  std::string cmd;
  // escape filename simply by surrounding with single quotes and escaping any
  // single quotes inside
  std::string esc = fn;
  size_t p = 0;
  while ((p = esc.find('\'', p)) != std::string::npos) {
    esc.replace(p, 1, "'\\''");
    p += 4;
  }
  esc = std::string("'") + esc + "'";
  cmd = "bgzip -d -c " + esc + " 2>/dev/null";
  FILE *f = popen(cmd.c_str(), "r");
  if (f)
    return f;
  cmd = "gzip -dc " + esc + " 2>/dev/null";
  f = popen(cmd.c_str(), "r");
  if (f)
    return f;
  cmd = "zcat " + esc + " 2>/dev/null";
  f = popen(cmd.c_str(), "r");
  if (f)
    return f;
  return nullptr;
}

// StreamReader wraps a FILE* and a small prefix buffer already read
struct StreamReader {
  FILE *f = nullptr;
  std::vector<char> prefix;
  size_t pos = 0;
  StreamReader(FILE *fp) : f(fp) {}
  ~StreamReader() {
    if (f)
      pclose(f);
  }
  // read bytes into dst, return bytes actually read
  size_t read(void *dst, size_t len) {
    size_t written = 0;
    // consume prefix first
    while (pos < prefix.size() && written < len) {
      ((char *)dst)[written++] = prefix[pos++];
    }
    if (written == len)
      return written;
    if (!f)
      return written;
    size_t r = fread(((char *)dst) + written, 1, len - written, f);
    return written + r;
  }
  bool eof() {
    if (pos < prefix.size())
      return false;
    if (!f)
      return true;
    return feof(f);
  }
};

// read little-endian 32-bit int
static bool read_int32(StreamReader &sr, int32_t &out) {
  uint8_t buf[4];
  size_t r = sr.read(buf, 4);
  if (r != 4)
    return false;
  out = (int32_t)(uint32_t(buf[0]) | (uint32_t(buf[1]) << 8) |
                  (uint32_t(buf[2]) << 16) | (uint32_t(buf[3]) << 24));
  return true;
}

// read little-endian float (IEEE 754)
static bool read_float32(StreamReader &sr, float &out) {
  uint8_t buf[4];
  size_t r = sr.read(buf, 4);
  if (r != 4)
    return false;
  uint32_t v = uint32_t(buf[0]) | (uint32_t(buf[1]) << 8) |
               (uint32_t(buf[2]) << 16) | (uint32_t(buf[3]) << 24);
  float fv;
  std::memcpy(&fv, &v, sizeof(float));
  out = fv;
  return true;
}

// Parse a bgzf/gzip-compressed bdamage stream and populate dp. This is a
// best-effort implementation that extracts per-cycle mismatch rates for the
// forward strand and maps the AA/AC/AG/... ordering to indices; C->T is index
// 7, A->G is index 2.
static bool
load_bdamage_stream(FILE *fpipe, DamageProfile &dp, std::string &err_out,
                    const std::vector<std::string> *refnames = nullptr,
                    const std::unordered_set<std::string> *contig_filter = nullptr) {
  StreamReader sr(fpipe);
  // read first 8 bytes to inspect magic if present
  sr.prefix.resize(8);
  size_t got = fread(sr.prefix.data(), 1, 8, fpipe);
  sr.prefix.resize(got);
  sr.pos = 0;
  // check for magic 'bdam1' at start
  bool has_magic = false;
  std::string magic;
  if (got >= 5) {
    magic.assign(sr.prefix.begin(), sr.prefix.begin() + 5);
    if (magic.find("bdam") != std::string::npos)
      has_magic = true;
  }
  (void)has_magic; // Detected for format validation, not currently used

  // If magic present, we proceed; otherwise assume stream begins with int
  // MAXLENGTH Ensure next reads include any prefix bytes already consumed via
  // StreamReader

  // read MAXLENGTH
  int32_t MAXL = 0;
  if (!read_int32(sr, MAXL)) {
    err_out = "bdamage: failed to read MAXLENGTH";
    return false;
  }
  if (MAXL <= 0) {
    err_out = "bdamage: invalid MAXLENGTH";
    return false;
  }

  // iterate over records until EOF
  while (!sr.eof()) {
    int32_t ID = 0;
    if (!read_int32(sr, ID))
      break; // EOF
    int32_t NREADS = 0;
    if (!read_int32(sr, NREADS)) {
      err_out = "bdamage: unexpected EOF reading NREADS";
      return false;
    }

    // read forward cycles: MAXL entries of 16 floats
    std::vector<std::array<float, 16>> fwd(MAXL);
    for (int i = 0; i < MAXL; ++i) {
      for (int j = 0; j < 16; ++j) {
        float v = 0.0f;
        if (!read_float32(sr, v)) {
          err_out = "bdamage: unexpected EOF reading forward floats";
          return false;
        }
        fwd[i][j] = v;
      }
    }
    // read reverse cycles (3' end damage)
    std::vector<std::array<float, 16>> rev(MAXL);
    for (int i = 0; i < MAXL; ++i) {
      for (int j = 0; j < 16; ++j) {
        float v = 0.0f;
        if (!read_float32(sr, v)) {
          err_out = "bdamage: unexpected EOF reading reverse floats";
          return false;
        }
        rev[i][j] = v;
      }
    }

    // populate entries for this ID
    std::string contig;
    if (refnames && ID >= 0 && (size_t)ID < refnames->size())
      contig = (*refnames)[ID];
    else
      contig = std::to_string(ID);

    // Skip this contig if it's not in the filter set
    if (contig_filter && contig_filter->find(contig) == contig_filter->end()) {
      continue;
    }

    auto &vec = dp.per_contig[contig];
    for (int i = 0; i < MAXL; ++i) {
      DamageEntry e;
      e.pos = i + 1;
      // ordering: AA,AC,AG,AT,CA,CC,CG,CT,GA,GC,GG,GT,TA,TC,TG,TT
      // Indices: 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15

      // Forward strand (5' end):
      // C->T is index 7 (CT), A->G is index 2 (AG)

      // Reverse strand (3' end):
      // G->A is index 8 (GA), T->C is index 13 (TC)

      // Calculate total observations at this position (sum of all 16 values)
      double total_obs_fwd = 0.0;
      double total_obs_rev = 0.0;
      for (int j = 0; j < 16; ++j) {
        total_obs_fwd += fwd[i][j];
        total_obs_rev += rev[i][j];
      }

      // Forward strand: C→T and A→G rates
      // C can transition to: CA(4), CC(5), CG(6), CT(7)
      double total_C_fwd = fwd[i][4] + fwd[i][5] + fwd[i][6] + fwd[i][7];
      double total_A_fwd = fwd[i][0] + fwd[i][1] + fwd[i][2] + fwd[i][3];

      // Reverse strand: G→A and T→C rates
      // G can transition to: GA(8), GC(9), GG(10), GT(11)
      double total_G_rev = rev[i][8] + rev[i][9] + rev[i][10] + rev[i][11];
      double total_T_rev = rev[i][12] + rev[i][13] + rev[i][14] + rev[i][15];

      // bdamage stores COUNTS, convert to rates (fractions)
      e.c_to_t = (total_C_fwd > 0) ? (double(fwd[i][7]) / total_C_fwd) : 0.0;
      e.a_to_g = (total_A_fwd > 0) ? (double(fwd[i][2]) / total_A_fwd) : 0.0;
      e.g_to_a = (total_G_rev > 0) ? (double(rev[i][8]) / total_G_rev) : 0.0;
      e.t_to_c = (total_T_rev > 0) ? (double(rev[i][13]) / total_T_rev) : 0.0;
      e.total = (long)(total_obs_fwd + total_obs_rev);
      vec.push_back(e);
    }
  }

  return true;
}

bool load_damage_profile(const std::string &fn, DamageProfile &dp, std::string &err_out) {
  dp.per_contig.clear();

  // No refname mapping provided in this signature; open and parse bdamage
  FILE *f = open_decompress_pipe(fn);
  if (!f) {
    err_out =
        std::string("load_damage_profile: cannot open/decompress bdamage file: ") +
        fn;
    return false;
  }
  bool ok = load_bdamage_stream(f, dp, err_out, nullptr);
  return ok;
}

// Read BAM header and extract target names in order
static bool load_bam_header_names(const std::string &bamfn,
                                  std::vector<std::string> &out_names,
                                  std::string &err_out) {
  out_names.clear();
  samFile *sf = sam_open(bamfn.c_str(), "r");
  if (!sf) {
    err_out = std::string("load_bam_header_names: cannot open BAM: ") + bamfn;
    return false;
  }
  bam_hdr_t *hdr = sam_hdr_read(sf);
  if (!hdr) {
    err_out = std::string("load_bam_header_names: cannot read BAM header: ") + bamfn;
    sam_close(sf);
    return false;
  }
  for (int i = 0; i < hdr->n_targets; ++i)
    out_names.emplace_back(hdr->target_name[i]);
  bam_hdr_destroy(hdr);
  sam_close(sf);
  return true;
}

// Backward compatible version without contig filter
bool load_damage_profile(const std::string &fn, DamageProfile &dp, std::string &err_out,
                         const std::string &ref_bam) {
  return load_damage_profile(fn, dp, err_out, ref_bam, nullptr);
}

// Full form: optionally provide a BAM whose header will be used to map numeric
// IDs in the bdamage file to reference names, and optionally filter to specific contigs.
bool load_damage_profile(const std::string &fn, DamageProfile &dp, std::string &err_out,
                         const std::string &ref_bam,
                         const std::unordered_set<std::string> *contig_filter) {
  dp.per_contig.clear();
  std::vector<std::string> refnames;
  if (!ref_bam.empty()) {
    if (!load_bam_header_names(ref_bam, refnames, err_out))
      return false;
  }

  FILE *f = open_decompress_pipe(fn);
  if (!f) {
    err_out =
        std::string("load_damage_profile: cannot open/decompress bdamage file: ") +
        fn;
    return false;
  }
  bool ok = load_bdamage_stream(f, dp, err_out,
                                refnames.empty() ? nullptr : &refnames,
                                contig_filter);
  return ok;
}

// Per-contig damage accumulator
struct DamageAccumulator {
  std::vector<uint64_t> n_5p;  // C opportunities at 5' end
  std::vector<uint64_t> k_5p;  // C→T mismatches at 5' end
  std::vector<uint64_t> n_3p;  // G opportunities at 3' end
  std::vector<uint64_t> k_3p;  // G→A mismatches at 3' end

  DamageAccumulator(int max_pos) : n_5p(max_pos, 0), k_5p(max_pos, 0),
                                    n_3p(max_pos, 0), k_3p(max_pos, 0) {}
};

// Parse MD tag and accumulate damage statistics
static void process_alignment_damage(bam1_t *b, bam_hdr_t *hdr,
                                     std::unordered_map<std::string, DamageAccumulator> &accumulators,
                                     const DamageComputeConfig &config,
                                     const std::unordered_set<std::string> *contig_filter) {
  if (b->core.tid < 0) return;
  if (b->core.qual < config.min_mapq) return;
  if (b->core.flag & (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY)) return;

  const char *contig_name = hdr->target_name[b->core.tid];
  if (contig_filter && contig_filter->find(contig_name) == contig_filter->end()) {
    return;
  }

  // Get or create accumulator for this contig
  std::string contig(contig_name);
  auto it = accumulators.find(contig);
  if (it == accumulators.end()) {
    it = accumulators.emplace(contig, DamageAccumulator(config.max_position)).first;
  }
  DamageAccumulator &acc = it->second;

  // Get MD tag
  uint8_t *md_ptr = bam_aux_get(b, "MD");
  if (!md_ptr) return;
  const char *md = bam_aux2Z(md_ptr);
  if (!md) return;

  // Get sequence and quality
  uint8_t *seq = bam_get_seq(b);
  uint8_t *qual = bam_get_qual(b);
  int seq_len = b->core.l_qseq;
  bool is_reverse = (b->core.flag & BAM_FREVERSE) != 0;

  // Build reference-offset -> read-offset map from CIGAR.
  // MD positions are in reference space, so this mapping is required when
  // insertions/deletions/clipping are present.
  uint32_t *cigar = bam_get_cigar(b);
  int n_cigar = b->core.n_cigar;
  int aln_ref_len = bam_cigar2rlen(n_cigar, cigar);
  std::vector<int> ref_to_read(static_cast<size_t>(std::max(0, aln_ref_len)), -1);
  {
    int ref_off = 0;
    int read_off = 0;
    for (int i = 0; i < n_cigar; i++) {
      int op = bam_cigar_op(cigar[i]);
      int len = bam_cigar_oplen(cigar[i]);
      if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
        for (int j = 0; j < len && ref_off + j < aln_ref_len; j++) {
          ref_to_read[ref_off + j] = read_off + j;
        }
        ref_off += len;
        read_off += len;
      } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
        ref_off += len;
      } else if (bam_cigar_type(op) & 1) {  // consumes query
        read_off += len;
      }
    }
  }

  // Parse MD tag and track mismatches using reference offsets.
  int ref_off_md = 0;
  int md_idx = 0;
  int md_len = strlen(md);

  while (md_idx < md_len && ref_off_md < aln_ref_len) {
    char c = md[md_idx];

    if (isdigit(c)) {
      // Match run
      int n = 0;
      while (md_idx < md_len && isdigit(md[md_idx])) {
        n = n * 10 + (md[md_idx] - '0');
        md_idx++;
      }
      ref_off_md += n;
    } else if (c == '^') {
      // Deletion in read - skip reference bases
      md_idx++;
      while (md_idx < md_len && isalpha(md[md_idx])) {
        ref_off_md++;
        md_idx++;
      }
    } else if (isalpha(c)) {
      // Mismatch: c is the reference base
      char ref_base = toupper(c);
      if (ref_off_md >= 0 && ref_off_md < aln_ref_len) {
        int read_pos = ref_to_read[ref_off_md];
        if (read_pos < 0 || read_pos >= seq_len) {
          ref_off_md++;
          md_idx++;
          continue;
        }

        // Get read base at this position
        int read_base_code = bam_seqi(seq, read_pos);
        char read_base = "=ACMGRSVTWYHKDBN"[read_base_code];
        int base_qual = qual[read_pos];

        if (base_qual >= config.min_baseq) {
          // Position from 5' end (1-based)
          int pos_5p = is_reverse ? (seq_len - read_pos) : (read_pos + 1);
          // Position from 3' end (1-based)
          int pos_3p = is_reverse ? (read_pos + 1) : (seq_len - read_pos);

          if (pos_5p >= 1 && pos_5p <= config.max_position) {
            int idx = pos_5p - 1;
            // For reverse reads, damage pattern is flipped:
            // - G→A at what appears as 5' end
            // For forward reads:
            // - C→T at 5' end
            if (!is_reverse && ref_base == 'C') {
              acc.n_5p[idx]++;
              if (read_base == 'T') acc.k_5p[idx]++;
            } else if (is_reverse && ref_base == 'G') {
              // Reverse complement: G→A on reverse strand = C→T on forward
              acc.n_5p[idx]++;
              if (read_base == 'A') acc.k_5p[idx]++;
            }
          }

          if (pos_3p >= 1 && pos_3p <= config.max_position) {
            int idx = pos_3p - 1;
            // For reverse reads, damage pattern is flipped
            if (!is_reverse && ref_base == 'G') {
              acc.n_3p[idx]++;
              if (read_base == 'A') acc.k_3p[idx]++;
            } else if (is_reverse && ref_base == 'C') {
              // Reverse complement: C→T on reverse strand = G→A on forward
              acc.n_3p[idx]++;
              if (read_base == 'T') acc.k_3p[idx]++;
            }
          }
        }
      }
      ref_off_md++;
      md_idx++;
    } else {
      md_idx++;
    }
  }

  // Also count opportunities from matches (C at 5' end, G at 3' end)
  int read_pos = 0;

  for (int i = 0; i < n_cigar; i++) {
    int op = bam_cigar_op(cigar[i]);
    int len = bam_cigar_oplen(cigar[i]);

    if (op == BAM_CMATCH || op == BAM_CEQUAL) {
      // Matches - count C and G opportunities at terminal positions
      for (int j = 0; j < len && read_pos + j < seq_len; j++) {
        int pos = read_pos + j;
        int base_qual = qual[pos];
        if (base_qual < config.min_baseq) continue;

        int read_base_code = bam_seqi(seq, pos);
        char read_base = "=ACMGRSVTWYHKDBN"[read_base_code];

        int pos_5p = is_reverse ? (seq_len - pos) : (pos + 1);
        int pos_3p = is_reverse ? (pos + 1) : (seq_len - pos);

        // For matches, the read base == ref base
        // Count C at 5' positions
        if (pos_5p >= 1 && pos_5p <= config.max_position) {
          int idx = pos_5p - 1;
          if (!is_reverse && read_base == 'C') {
            acc.n_5p[idx]++;
          } else if (is_reverse && read_base == 'G') {
            acc.n_5p[idx]++;
          }
        }

        // Count G at 3' positions
        if (pos_3p >= 1 && pos_3p <= config.max_position) {
          int idx = pos_3p - 1;
          if (!is_reverse && read_base == 'G') {
            acc.n_3p[idx]++;
          } else if (is_reverse && read_base == 'C') {
            acc.n_3p[idx]++;
          }
        }
      }
    }

    if (bam_cigar_type(op) & 1) {  // Consumes query
      read_pos += len;
    }
  }
}

bool compute_damage_from_bam(const std::string &bam_file, DamageProfile &dp,
                             std::string &err_out, const DamageComputeConfig &config,
                             const std::unordered_set<std::string> *contig_filter) {
  dp.per_contig.clear();

  samFile *sf = sam_open(bam_file.c_str(), "r");
  if (!sf) {
    err_out = "compute_damage_from_bam: cannot open BAM: " + bam_file;
    return false;
  }

  // Set thread pool for reading
  if (config.threads > 1) {
    hts_set_threads(sf, config.threads);
  }

  bam_hdr_t *hdr = sam_hdr_read(sf);
  if (!hdr) {
    err_out = "compute_damage_from_bam: cannot read BAM header: " + bam_file;
    sam_close(sf);
    return false;
  }

  // Accumulate statistics
  std::unordered_map<std::string, DamageAccumulator> accumulators;
  bam1_t *b = bam_init1();

  uint64_t total_reads = 0;
  while (sam_read1(sf, hdr, b) >= 0) {
    process_alignment_damage(b, hdr, accumulators, config, contig_filter);
    total_reads++;
  }

  bam_destroy1(b);
  bam_hdr_destroy(hdr);
  sam_close(sf);

  // Convert accumulators to DamageProfile
  for (auto &[contig, acc] : accumulators) {
    auto &vec = dp.per_contig[contig];
    for (int i = 0; i < config.max_position; i++) {
      DamageEntry e;
      e.pos = i + 1;
      e.c_to_t = (acc.n_5p[i] > 0) ? (double(acc.k_5p[i]) / acc.n_5p[i]) : 0.0;
      e.g_to_a = (acc.n_3p[i] > 0) ? (double(acc.k_3p[i]) / acc.n_3p[i]) : 0.0;
      e.total = acc.n_5p[i] + acc.n_3p[i];
      vec.push_back(e);
    }
  }

  return true;
}

// Log of Beta function: log(B(a,b)) = lgamma(a) + lgamma(b) - lgamma(a+b)
static double log_beta(double a, double b) {
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

// Standard normal quantile (inverse CDF) using Abramowitz & Stegun approximation
static double normal_quantile(double p) {
  if (p <= 0.0) return -6.0;
  if (p >= 1.0) return 6.0;
  if (std::fabs(p - 0.5) < 1e-10) return 0.0;

  double sign = (p < 0.5) ? -1.0 : 1.0;
  double t = std::sqrt(-2.0 * std::log(std::min(p, 1.0 - p)));

  // Rational approximation
  double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
  double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
  double z = t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t);
  return sign * z;
}

// Beta distribution quantile using normal approximation for concentrated distributions
// For Beta(alpha, beta) with alpha + beta >> 1, the distribution is approximately normal
static double beta_quantile(double p, double alpha, double beta) {
  if (p <= 0.0) return 0.0;
  if (p >= 1.0) return 1.0;

  double mean = alpha / (alpha + beta);
  double var = (alpha * beta) / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1));
  double sd = std::sqrt(var);

  // For concentrated distributions, normal approximation is accurate
  double z = normal_quantile(p);
  double x = mean + sd * z;

  return std::max(0.0, std::min(1.0, x));
}

// Compute 95% credible interval for rate using Beta posterior
// Prior: Beta(0.5, 0.5) (Jeffreys prior)
static void compute_credible_interval(uint64_t k, uint64_t n,
                                       double &rate, double &ci_lower, double &ci_upper) {
  if (n == 0) {
    rate = 0.0;
    ci_lower = 0.0;
    ci_upper = 1.0;
    return;
  }

  // Posterior: Beta(k + 0.5, n - k + 0.5)
  double alpha_post = (double)k + 0.5;
  double beta_post = (double)(n - k) + 0.5;

  rate = (double)k / n;
  ci_lower = beta_quantile(0.025, alpha_post, beta_post);
  ci_upper = beta_quantile(0.975, alpha_post, beta_post);
}

// Log probability mass function of Beta-Binomial distribution
static double log_beta_binomial_pmf(double k, double n, double alpha, double beta) {
  if (n < 0.01) return 0.0;
  if (k < 0) k = 0;
  if (k > n) k = n;

  double log_binom = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
  double log_beta_ratio = log_beta(k + alpha, n - k + beta) - log_beta(alpha, beta);
  return log_binom + log_beta_ratio;
}

// Log prior probability for amplitude using logistic-gated flat prior
// This is the optimal solution for ancient DNA damage:
// - FLAT (log_prior ≈ 0) for A > 3% (biologically meaningful ancient damage)
// - Smooth penalty for A < 2% (modern contamination level)
// - No hard thresholds, smooth sigmoid transition
//
// Formula: log_prior(A) = log(sigmoid((A - A0) / s))
// Where sigmoid(x) = 1 / (1 + exp(-x))
//
// ============================================================================
// 3-Model Bayesian Classifier
// ============================================================================
// Three competing models for DNA damage patterns:
//   M (Modern):  flat baseline, no damage
//   O (Other):   spike at terminal positions, artifact pattern
//   A (Ancient): sigmoidal decay from deamination
//
// Uses exact marginal likelihoods (conjugate) for M and O,
// Laplace approximation for A.

// ============================================================================
// ERALO: Evidence Ratio with Amplitude-Lambda Ordering
// A statistically principled scoring function for ancient DNA classification
// ============================================================================

// Effective decay length: L(λ) = (1 - λ^20) / (1 - λ)
// Measures how gradually damage decays - larger for slower decay (more ancient-like)
static double effective_decay_length(double lambda) {
  if (lambda < 1e-10) return 20.0;  // λ→0: all positions have damage
  if (lambda > 0.9999) return 1.0;  // λ→1: only position 0 has damage
  return (1.0 - std::pow(lambda, 20)) / (1.0 - lambda);
}

// ERALO scoring function
// Returns a score where higher = more likely ancient DNA
// Score is monotone increasing in amplitude and slow-decay (low λ)
struct ERALOResult {
  double score;           // Raw ERALO score
  double p_ancient;       // Calibrated probability
  double amplitude;       // Observed amplitude (terminal - baseline)
  double lambda;          // Fitted decay rate
  double baseline;        // Estimated baseline mismatch rate
  double effective_length; // L(λ) - effective decay length
};

// Compute three-tier classification based on p_ancient, amplitude, and lambda
// Rules:
//   ANCIENT:   p >= 0.8 AND amplitude >= 1% AND (lambda > 0.3 OR amplitude < 5%)
//   MODERN:    p < 0.5
//   UNCERTAIN: everything else
static DamageClassification compute_classification(double p_ancient,
                                                    double amplitude,
                                                    double lambda) {
  // Modern: clearly below threshold
  if (p_ancient < 0.5) {
    return DamageClassification::MODERN;
  }

  // Ancient requires strong evidence:
  // 1. High probability (p >= 0.8)
  // 2. Meaningful amplitude (>= 1%)
  // 3. Reasonable decay pattern (lambda > 0.3 for fast decay, or low amplitude)
  if (p_ancient >= 0.8 && amplitude >= 0.01) {
    // For high-amplitude signals, require lambda > 0.3 to avoid artifacts
    // For low-amplitude signals (<5%), allow any lambda since they're weak anyway
    if (amplitude < 0.05 || lambda > 0.3) {
      return DamageClassification::ANCIENT;
    }
  }

  // Everything else is uncertain
  return DamageClassification::UNCERTAIN;
}

// Compute log-likelihood under ancient model with exponential decay
static double compute_log_lik_ancient(const std::vector<uint64_t> &n_arr,
                                      const std::vector<uint64_t> &k_arr,
                                      double amplitude, double lambda,
                                      double baseline, double concentration) {
  double log_lik = 0.0;
  for (size_t z = 0; z < n_arr.size(); z++) {
    if (n_arr[z] < 1) continue;
    // D(z) = amplitude * exp(-lambda * z) + baseline
    double D_z = amplitude * std::exp(-lambda * z) + baseline;
    double mean_z = std::max(1e-10, std::min(1.0 - 1e-10, D_z));
    double alpha_z = mean_z * concentration;
    double beta_z = (1.0 - mean_z) * concentration;
    if (alpha_z < 0.01) alpha_z = 0.01;
    if (beta_z < 0.01) beta_z = 0.01;
    log_lik += log_beta_binomial_pmf(k_arr[z], n_arr[z], alpha_z, beta_z);
  }
  return log_lik;
}

// Compute ERALO from raw damage counts using G-test (log-likelihood ratio test)
// Focus on terminal positions where ancient damage signal is strongest.
// Uses G-test to compare observed rates to expected rates under each model.
static ERALOResult eralo_from_counts(const std::vector<uint64_t> &n_arr,
                                      const std::vector<uint64_t> &k_arr,
                                      double fitted_amplitude,
                                      double fitted_lambda) {
  ERALOResult result;
  result.amplitude = fitted_amplitude;
  result.lambda = fitted_lambda;
  result.effective_length = effective_decay_length(fitted_lambda);

  // Estimate baseline from interior positions (15-20)
  uint64_t interior_n = 0, interior_k = 0;
  for (size_t z = 14; z < std::min(n_arr.size(), (size_t)20); z++) {
    interior_n += n_arr[z];
    interior_k += k_arr[z];
  }
  result.baseline = (interior_n > 100) ? (double)interior_k / interior_n : 0.01;

  // Use G-test (log-likelihood ratio test) which properly handles large n
  // G = 2 * sum(O * log(O/E)) where O=observed, E=expected
  // Comparing: ancient model vs modern model
  double G_ancient = 0.0;  // G-statistic for ancient model
  double G_modern = 0.0;   // G-statistic for modern model
  size_t n_terminal = std::min((size_t)10, n_arr.size());

  for (size_t z = 0; z < n_terminal; z++) {
    if (n_arr[z] < 10) continue;

    double obs_k = (double)k_arr[z];
    double obs_nk = (double)(n_arr[z] - k_arr[z]);
    double n = (double)n_arr[z];

    // Expected under ancient model: D(z) = A * exp(-λz) + baseline
    double exp_rate_ancient = fitted_amplitude * std::exp(-fitted_lambda * z) + result.baseline;
    exp_rate_ancient = std::max(0.001, std::min(0.999, exp_rate_ancient));
    double exp_k_ancient = n * exp_rate_ancient;
    double exp_nk_ancient = n * (1.0 - exp_rate_ancient);

    // Expected under modern model: flat baseline
    double exp_rate_modern = result.baseline;
    exp_rate_modern = std::max(0.001, std::min(0.999, exp_rate_modern));
    double exp_k_modern = n * exp_rate_modern;
    double exp_nk_modern = n * (1.0 - exp_rate_modern);

    // G-statistic contribution: 2 * O * log(O/E)
    // Lower G means better fit
    if (obs_k > 0 && exp_k_ancient > 0) {
      G_ancient += obs_k * std::log(obs_k / exp_k_ancient);
    }
    if (obs_nk > 0 && exp_nk_ancient > 0) {
      G_ancient += obs_nk * std::log(obs_nk / exp_nk_ancient);
    }

    if (obs_k > 0 && exp_k_modern > 0) {
      G_modern += obs_k * std::log(obs_k / exp_k_modern);
    }
    if (obs_nk > 0 && exp_nk_modern > 0) {
      G_modern += obs_nk * std::log(obs_nk / exp_nk_modern);
    }
  }

  G_ancient *= 2.0;
  G_modern *= 2.0;

  // Delta G: positive means ancient model fits better
  // G_modern - G_ancient because lower G = better fit
  double delta_G = G_modern - G_ancient;

  // Count terminal observations for normalization
  uint64_t terminal_obs = 0;
  for (size_t z = 0; z < n_terminal; z++) {
    terminal_obs += n_arr[z];
  }

  // Normalize by sqrt(n) to get a z-score-like quantity
  // This is appropriate because G-statistic ~ chi-squared under null
  double score = (terminal_obs > 100) ? delta_G / std::sqrt((double)terminal_obs) : 0.0;

  result.score = score;

  // Convert to probability: score > 3 means very strong ancient evidence
  // Using a steeper sigmoid since G-test scores can be large
  result.p_ancient = 1.0 / (1.0 + std::exp(-score / 2.0));

  return result;
}

// Estimate baseline from interior positions (positions 16-20) with SE
static void estimate_baseline_with_se(const std::vector<uint64_t> &n_arr,
                                       const std::vector<uint64_t> &k_arr,
                                       double &baseline, double &baseline_se) {
  uint64_t n_int = 0, k_int = 0;
  for (size_t z = 15; z < std::min(n_arr.size(), (size_t)20); z++) {
    n_int += n_arr[z];
    k_int += k_arr[z];
  }
  if (n_int < 10) {
    baseline = 0.01;
    baseline_se = 0.01;
    return;
  }
  baseline = (double)k_int / n_int;
  // SE for binomial proportion: sqrt(p(1-p)/n)
  baseline_se = std::sqrt(baseline * (1.0 - baseline) / n_int);
}

// Fit exponential decay curve: D(z) = A * exp(-lambda * z) + baseline
// Uses grid search over lambda and weighted least squares for amplitude
// Returns: amplitude, amplitude_se, lambda, lambda_se, concentration, R²
struct CurveFitResult {
  double amplitude = 0.0;
  double amplitude_se = 0.0;
  double lambda = 0.35;
  double lambda_se = 0.0;
  double baseline = 0.01;
  double baseline_se = 0.0;
  double concentration = 100.0;
  double r_squared = 0.0;
  double log_lik = 0.0;
};

static CurveFitResult fit_exponential_curve(const std::vector<uint64_t> &n_arr,
                                             const std::vector<uint64_t> &k_arr,
                                             int n_positions = 15) {
  CurveFitResult result;
  size_t n_pos = std::min((size_t)n_positions, n_arr.size());

  // Estimate baseline from interior positions
  estimate_baseline_with_se(n_arr, k_arr, result.baseline, result.baseline_se);

  // Compute observed rates for curve fitting
  std::vector<double> rates(n_pos), weights(n_pos);
  double total_weight = 0.0;
  uint64_t total_n = 0;

  for (size_t z = 0; z < n_pos; z++) {
    if (n_arr[z] > 0) {
      rates[z] = (double)k_arr[z] / n_arr[z];
      weights[z] = std::sqrt((double)n_arr[z]);  // Weight by sqrt(n)
      total_weight += weights[z];
      total_n += n_arr[z];
    }
  }

  if (total_n < 100) {
    return result;
  }

  // Estimate amplitude directly from terminal position (position 1)
  // This is where ancient DNA damage signal is strongest
  double terminal_amplitude = 0.0;
  if (n_arr[0] >= 10) {
    terminal_amplitude = std::max(0.001, rates[0] - result.baseline);
  }

  // Grid search over lambda and amplitude values
  // Use terminal amplitude as starting point, but also search nearby values
  double best_log_lik = -1e30;
  double best_lambda = 0.35;
  double best_amplitude = 0.0;

  for (double lambda = 0.3; lambda <= 1.5; lambda += 0.05) {
    // Try multiple amplitude values around terminal estimate
    for (double amp_mult = 0.5; amp_mult <= 1.5; amp_mult += 0.1) {
      double amplitude = terminal_amplitude * amp_mult;
      if (amplitude < 0.001 || amplitude > 0.5) continue;

      // Compute log-likelihood with this configuration
      double log_lik = compute_log_lik_ancient(
        std::vector<uint64_t>(n_arr.begin(), n_arr.begin() + n_pos),
        std::vector<uint64_t>(k_arr.begin(), k_arr.begin() + n_pos),
        amplitude, lambda, result.baseline, 100.0);

      if (log_lik > best_log_lik) {
        best_log_lik = log_lik;
        best_lambda = lambda;
        best_amplitude = amplitude;
      }
    }
  }

  result.lambda = best_lambda;
  result.amplitude = best_amplitude;
  result.log_lik = best_log_lik;

  // Estimate concentration (overdispersion) via method of moments
  double sum_chi2 = 0.0;
  int df = 0;
  for (size_t z = 0; z < n_pos; z++) {
    if (n_arr[z] < 10) continue;
    double expected = result.amplitude * std::exp(-result.lambda * z) + result.baseline;
    double observed = rates[z];
    double var_binom = expected * (1.0 - expected) / n_arr[z];
    if (var_binom > 1e-10) {
      sum_chi2 += (observed - expected) * (observed - expected) / var_binom;
      df++;
    }
  }
  if (df > 2) {
    double overdispersion = sum_chi2 / df;
    result.concentration = std::max(10.0, std::min(1000.0, 100.0 / overdispersion));
  }

  // Compute R² (coefficient of determination)
  double ss_tot = 0.0, ss_res = 0.0;
  double mean_rate = 0.0;
  int n_valid = 0;
  for (size_t z = 0; z < n_pos; z++) {
    if (n_arr[z] > 0) {
      mean_rate += rates[z];
      n_valid++;
    }
  }
  if (n_valid > 0) mean_rate /= n_valid;

  for (size_t z = 0; z < n_pos; z++) {
    if (n_arr[z] < 10) continue;
    double expected = result.amplitude * std::exp(-result.lambda * z) + result.baseline;
    ss_tot += (rates[z] - mean_rate) * (rates[z] - mean_rate);
    ss_res += (rates[z] - expected) * (rates[z] - expected);
  }
  result.r_squared = (ss_tot > 1e-10) ? 1.0 - ss_res / ss_tot : 0.0;
  result.r_squared = std::max(0.0, result.r_squared);

  // Estimate standard errors using Fisher information approximation
  // SE(amplitude) ≈ amplitude / sqrt(sum_n * amplitude)
  if (total_n > 0 && result.amplitude > 0.001) {
    result.amplitude_se = std::sqrt(result.amplitude * (1.0 - result.amplitude) / total_n);
    result.lambda_se = 0.1;  // Approximate - would need Hessian for exact
  }

  return result;
}

DamageModelResult fit_damage_model(const std::vector<uint64_t> &n_5p,
                                   const std::vector<uint64_t> &k_5p,
                                   const std::vector<uint64_t> &n_3p,
                                   const std::vector<uint64_t> &k_3p,
                                   double prior_ancient) {
  DamageModelResult result;
  size_t n_positions = n_5p.size();

  // Count total observations
  for (size_t i = 0; i < n_positions; i++) {
    result.total_obs += n_5p[i] + n_3p[i];
  }

  // Compute per-position statistics with credible intervals
  result.stats_5p.resize(n_positions);
  result.stats_3p.resize(n_positions);

  for (size_t z = 0; z < n_positions; z++) {
    // 5' end statistics
    result.stats_5p[z].position = z + 1;
    result.stats_5p[z].n_opportunities = n_5p[z];
    result.stats_5p[z].n_mismatches = k_5p[z];
    compute_credible_interval(k_5p[z], n_5p[z],
                               result.stats_5p[z].rate,
                               result.stats_5p[z].ci_lower,
                               result.stats_5p[z].ci_upper);

    // 3' end statistics
    result.stats_3p[z].position = z + 1;
    result.stats_3p[z].n_opportunities = n_3p[z];
    result.stats_3p[z].n_mismatches = k_3p[z];
    compute_credible_interval(k_3p[z], n_3p[z],
                               result.stats_3p[z].rate,
                               result.stats_3p[z].ci_lower,
                               result.stats_3p[z].ci_upper);
  }

  // Compute effective coverage
  uint64_t total_coverage = 0;
  int n_terminal = std::min((int)n_positions, 5);
  for (int z = 0; z < n_terminal; z++) {
    total_coverage += n_5p[z] + n_3p[z];
  }
  result.effective_coverage = (n_terminal > 0) ? (double)total_coverage / (2 * n_terminal) : 0.0;

  if (result.total_obs < 100) {
    result.p_ancient = prior_ancient;
    result.significance_reason = "Insufficient data (<100 observations)";
    return result;
  }

  // Fit separate curves for 5' and 3' ends
  CurveFitResult fit_5p = fit_exponential_curve(n_5p, k_5p);
  CurveFitResult fit_3p = fit_exponential_curve(n_3p, k_3p);

  // Populate 5' curve parameters
  result.curve_5p.amplitude = fit_5p.amplitude;
  result.curve_5p.amplitude_se = fit_5p.amplitude_se;
  result.curve_5p.lambda = fit_5p.lambda;
  result.curve_5p.lambda_se = fit_5p.lambda_se;
  result.curve_5p.baseline = fit_5p.baseline;
  result.curve_5p.baseline_se = fit_5p.baseline_se;
  result.curve_5p.concentration = fit_5p.concentration;

  // Populate 3' curve parameters
  result.curve_3p.amplitude = fit_3p.amplitude;
  result.curve_3p.amplitude_se = fit_3p.amplitude_se;
  result.curve_3p.lambda = fit_3p.lambda;
  result.curve_3p.lambda_se = fit_3p.lambda_se;
  result.curve_3p.baseline = fit_3p.baseline;
  result.curve_3p.baseline_se = fit_3p.baseline_se;
  result.curve_3p.concentration = fit_3p.concentration;

  // Average R² across both ends
  result.fit_r_squared = (fit_5p.r_squared + fit_3p.r_squared) / 2.0;

  // Combine 5' and 3' counts for global likelihood computation
  std::vector<uint64_t> n_combined(n_positions), k_combined(n_positions);
  for (size_t i = 0; i < n_positions; i++) {
    n_combined[i] = n_5p[i] + n_3p[i];
    k_combined[i] = k_5p[i] + k_3p[i];
  }

  // ERALO scoring: statistically principled classification based on
  // amplitude and decay rate, with explicit penalties for suspicious patterns
  double avg_amplitude = (fit_5p.amplitude + fit_3p.amplitude) / 2.0;
  double avg_lambda = (fit_5p.lambda + fit_3p.lambda) / 2.0;
  ERALOResult eralo = eralo_from_counts(n_combined, k_combined, avg_amplitude, avg_lambda);

  result.p_ancient = eralo.p_ancient;
  result.log_bf = eralo.score;  // Use ERALO score as log odds ratio

  // Map ERALO score to evidence strength
  if (eralo.score < -1.0) {
    result.evidence_strength = 0.0;  // Modern/artifact
  } else if (eralo.score < 0.5) {
    result.evidence_strength = 1.0;  // Weak
  } else if (eralo.score < 1.5) {
    result.evidence_strength = 2.0;  // Moderate
  } else if (eralo.score < 2.5) {
    result.evidence_strength = 3.0;  // Strong
  } else {
    result.evidence_strength = 4.0;  // Decisive
  }

  // Compute three-tier classification
  result.classification = compute_classification(eralo.p_ancient, eralo.amplitude, eralo.lambda);

  // Classification interpretation
  std::ostringstream reason;
  switch (result.classification) {
    case DamageClassification::ANCIENT:
      result.is_significant = true;
      reason << "Ancient: P=" << std::fixed << std::setprecision(3) << eralo.p_ancient;
      reason << ", amp=" << std::setprecision(1) << (eralo.amplitude * 100.0) << "%";
      reason << ", λ=" << std::setprecision(2) << eralo.lambda;
      reason << ", score=" << std::setprecision(2) << eralo.score;
      break;
    case DamageClassification::MODERN:
      result.is_significant = false;
      if (eralo.amplitude < 0.03) {
        reason << "Modern (weak signal): amp=" << std::fixed << std::setprecision(1) << (eralo.amplitude * 100.0) << "%";
      } else {
        reason << "Modern: P=" << std::fixed << std::setprecision(3) << eralo.p_ancient;
      }
      reason << ", score=" << std::setprecision(2) << eralo.score;
      break;
    case DamageClassification::UNCERTAIN:
      result.is_significant = false;
      reason << "Uncertain: P=" << std::fixed << std::setprecision(3) << eralo.p_ancient;
      reason << ", amp=" << std::setprecision(1) << (eralo.amplitude * 100.0) << "%";
      reason << ", λ=" << std::setprecision(2) << eralo.lambda;
      if (eralo.p_ancient >= 0.8 && eralo.amplitude < 0.01) {
        reason << " (amplitude too low)";
      } else if (eralo.p_ancient >= 0.5 && eralo.p_ancient < 0.8) {
        reason << " (probability borderline)";
      }
      reason << ", score=" << std::setprecision(2) << eralo.score;
      break;
  }
  result.significance_reason = reason.str();

  return result;
}

// Detect library type from damage profile
static std::string detect_library_type_from_rates(const std::vector<uint64_t> &n_5p,
                                              const std::vector<uint64_t> &k_5p) {
  // DS libraries: damage concentrated at positions 1-3, rapid decay
  // SS libraries: damage extends to positions 10-15, slow decay
  if (n_5p.size() < 20) return "ds";

  // Estimate baseline from interior positions (16-20)
  double baseline = 0.0;
  int baseline_n = 0;
  for (size_t i = 15; i < std::min(n_5p.size(), (size_t)20); i++) {
    if (n_5p[i] > 100) {
      baseline += (double)k_5p[i] / n_5p[i];
      baseline_n++;
    }
  }
  if (baseline_n > 0) baseline /= baseline_n;

  // Calculate excess damage (above baseline) at different positions
  double damage_pos1 = (n_5p[0] > 0) ? (double)k_5p[0] / n_5p[0] - baseline : 0.0;
  double damage_pos3 = (n_5p[2] > 0) ? (double)k_5p[2] / n_5p[2] - baseline : 0.0;

  // For positions 10-15, calculate average excess damage
  double late_excess = 0.0;
  int late_n = 0;
  for (size_t i = 9; i < 15; i++) {
    if (n_5p[i] > 100) {
      double rate = (double)k_5p[i] / n_5p[i];
      late_excess += std::max(0.0, rate - baseline);
      late_n++;
    }
  }
  if (late_n > 0) late_excess /= late_n;

  // SS libraries: significant excess damage at positions 10-15
  // Threshold: late excess > 2% AND late excess > 20% of pos 3 excess
  if (damage_pos1 > 0.05 && late_excess > 0.02 && damage_pos3 > 0.01) {
    if (late_excess > damage_pos3 * 0.3) {
      return "ss";
    }
  }
  return "ds";
}

bool compute_damage_analysis(const std::string &bam_file, DamageAnalysis &analysis,
                             std::string &err_out, const DamageComputeConfig &config,
                             const std::unordered_set<std::string> *contig_filter) {
  auto t_start = std::chrono::high_resolution_clock::now();

  analysis.profile.per_contig.clear();
  analysis.per_contig_model.clear();

  samFile *sf = sam_open(bam_file.c_str(), "r");
  if (!sf) {
    err_out = "compute_damage_analysis: cannot open BAM: " + bam_file;
    return false;
  }

  if (config.threads > 1) {
    hts_set_threads(sf, config.threads);
  }

  bam_hdr_t *hdr = sam_hdr_read(sf);
  if (!hdr) {
    err_out = "compute_damage_analysis: cannot read BAM header: " + bam_file;
    sam_close(sf);
    return false;
  }

  // Accumulate statistics
  std::unordered_map<std::string, DamageAccumulator> accumulators;
  bam1_t *b = bam_init1();

  while (sam_read1(sf, hdr, b) >= 0) {
    process_alignment_damage(b, hdr, accumulators, config, contig_filter);
  }

  bam_destroy1(b);
  bam_hdr_destroy(hdr);
  sam_close(sf);

  auto t_bam = std::chrono::high_resolution_clock::now();
  auto bam_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_bam - t_start).count();

  // Global accumulators for overall model
  std::vector<uint64_t> global_n_5p(config.max_position, 0);
  std::vector<uint64_t> global_k_5p(config.max_position, 0);
  std::vector<uint64_t> global_n_3p(config.max_position, 0);
  std::vector<uint64_t> global_k_3p(config.max_position, 0);

  // Convert accumulators to DamageProfile and fit per-contig models
  for (auto &[contig, acc] : accumulators) {
    auto &vec = analysis.profile.per_contig[contig];
    for (int i = 0; i < config.max_position; i++) {
      DamageEntry e;
      e.pos = i + 1;
      e.c_to_t = (acc.n_5p[i] > 0) ? (double(acc.k_5p[i]) / acc.n_5p[i]) : 0.0;
      e.g_to_a = (acc.n_3p[i] > 0) ? (double(acc.k_3p[i]) / acc.n_3p[i]) : 0.0;
      e.total = acc.n_5p[i] + acc.n_3p[i];
      vec.push_back(e);

      // Accumulate into global
      global_n_5p[i] += acc.n_5p[i];
      global_k_5p[i] += acc.k_5p[i];
      global_n_3p[i] += acc.n_3p[i];
      global_k_3p[i] += acc.k_3p[i];
    }

    // Fit per-contig model
    analysis.per_contig_model[contig] = fit_damage_model(
      acc.n_5p, acc.k_5p, acc.n_3p, acc.k_3p, 0.5);
  }

  auto t_percontig = std::chrono::high_resolution_clock::now();
  auto percontig_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_percontig - t_bam).count();

  // Fit global model
  analysis.global_model = fit_damage_model(global_n_5p, global_k_5p,
                                            global_n_3p, global_k_3p, 0.5);

  auto t_global = std::chrono::high_resolution_clock::now();
  auto global_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_global - t_percontig).count();

  // Detect library type
  analysis.library_type = detect_library_type_from_rates(global_n_5p, global_k_5p);

  // Calibrate error rate from interior positions (damage-free zone)
  double baseline_5p = 0.0, baseline_se_5p = 0.0;
  double baseline_3p = 0.0, baseline_se_3p = 0.0;
  estimate_baseline_with_se(global_n_5p, global_k_5p, baseline_5p, baseline_se_5p);
  estimate_baseline_with_se(global_n_3p, global_k_3p, baseline_3p, baseline_se_3p);

  analysis.calibrated_error_rate = (baseline_5p + baseline_3p) / 2.0;
  analysis.error_rate_se = std::sqrt(baseline_se_5p * baseline_se_5p +
                                 baseline_se_3p * baseline_se_3p) / 2.0;

  auto t_end = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

  std::cerr << "  Damage timing: BAM scan " << bam_ms << "ms, "
       << "per-contig models (" << accumulators.size() << "): " << percontig_ms << "ms, "
       << "global model: " << global_ms << "ms, "
       << "total: " << total_ms << "ms" << std::endl;

  return true;
}

} // namespace amber

#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace amber {

// Simple, thread-safe reference/alignments reader scaffolding.
// Current implementation: BAM access via htslib indexed queries. This lets
// callers fetch alignments overlapping contig:[start,end] using the BAM index
// (.bai or .csi). The API is intentionally small so it can be extended later
// to provide a CSI-backed parallel fetch queue or other backends.

class RefReader {
public:
  RefReader();
  ~RefReader();

  // Open a FASTA file and its index (.fai). Returns true on success.
  bool open_fasta(const std::string &fasta_path, std::string &err_out);

  // Fetch the sequence for contig:[start,end] (1-based inclusive) from
  // FASTA/faidx. On success ok==true and returned string contains the bases;
  // otherwise ok==false and the returned string is empty.
  std::string fetch_region_fasta(const std::string &contig, int64_t start,
                                 int64_t end, bool &ok);

  // Open a BAM/CRAM file (must have an index .bai/.csi available).
  bool open_bam(const std::string &bam_path, std::string &err_out);

  // Fetch SAM-formatted records overlapping contig:[start,end] (1-based
  // inclusive). On success returns true and fills out_sam_lines with one SAM
  // line per record; on failure returns false and fills err_out.
  bool fetch_region_sam(const std::string &contig, int64_t start, int64_t end,
                        std::vector<std::string> &out_sam_lines,
                        std::string &err_out);

  // Close any open resources.
  void close();

private:
  struct Impl;
  Impl *impl_ = nullptr;
  // non-copyable
  RefReader(const RefReader &) = delete;
  RefReader &operator=(const RefReader &) = delete;
};

} // namespace amber

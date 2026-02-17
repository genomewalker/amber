#include "ref_reader.h"
#include <cstdlib>
#include <htslib/faidx.h>
#include <htslib/hts.h>
#include <htslib/sam.h>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace amber {

struct RefReader::Impl {
  faidx_t *fai = nullptr;
  samFile *sam = nullptr;
  bam_hdr_t *hdr = nullptr;
  hts_idx_t *idx = nullptr;
  std::string path;
  std::mutex mtx;
};

RefReader::RefReader() : impl_(new Impl()) {}

RefReader::~RefReader() {
  close();
  delete impl_;
}

bool RefReader::open_fasta(const std::string &fasta_path,
                           std::string &err_out) {
  std::lock_guard<std::mutex> lg(impl_->mtx);
  if (impl_->fai) {
    err_out = "RefReader: already open";
    return false;
  }
  impl_->fai = fai_load(fasta_path.c_str());
  if (!impl_->fai) {
    err_out = "RefReader: fai_load failed for " + fasta_path;
    return false;
  }
  impl_->path = fasta_path;
  return true;
}

std::string RefReader::fetch_region_fasta(const std::string &contig,
                                          int64_t start, int64_t end,
                                          bool &ok) {
  std::lock_guard<std::mutex> lg(impl_->mtx);
  ok = false;
  if (!impl_->fai)
    return std::string();
  int b = (int)(start - 1);
  int e = (int)(end - 1);
  int len = 0;
  char *res = faidx_fetch_seq(impl_->fai, contig.c_str(), b, e, &len);
  if (!res)
    return std::string();
  std::string out(res, len);
  free(res);
  ok = true;
  return out;
}

bool RefReader::open_bam(const std::string &bam_path, std::string &err_out) {
  std::lock_guard<std::mutex> lg(impl_->mtx);
  if (impl_->sam) {
    err_out = "RefReader: bam already open";
    return false;
  }
  impl_->sam = sam_open(bam_path.c_str(), "r");
  if (!impl_->sam) {
    err_out = "RefReader: sam_open failed for " + bam_path;
    return false;
  }
  impl_->hdr = sam_hdr_read(impl_->sam);
  if (!impl_->hdr) {
    err_out = "RefReader: sam_hdr_read failed for " + bam_path;
    sam_close(impl_->sam);
    impl_->sam = nullptr;
    return false;
  }
  impl_->idx = sam_index_load(impl_->sam, bam_path.c_str());
  if (!impl_->idx) {
    err_out = "RefReader: sam_index_load failed for " + bam_path +
              " (missing .bai/.csi?)";
    bam_hdr_destroy(impl_->hdr);
    impl_->hdr = nullptr;
    sam_close(impl_->sam);
    impl_->sam = nullptr;
    return false;
  }
  impl_->path = bam_path;
  return true;
}

bool RefReader::fetch_region_sam(const std::string &contig, int64_t start,
                                 int64_t end,
                                 std::vector<std::string> &out_sam_lines,
                                 std::string &err_out) {
  std::lock_guard<std::mutex> lg(impl_->mtx);
  out_sam_lines.clear();
  if (!impl_->sam || !impl_->hdr || !impl_->idx) {
    err_out = "RefReader: BAM not open with index";
    return false;
  }
  std::ostringstream reg;
  reg << contig << ":" << start << "-" << end;
  hts_itr_t *iter = sam_itr_querys(impl_->idx, impl_->hdr, reg.str().c_str());
  if (!iter) {
    err_out = "RefReader: sam_itr_querys failed for region " + reg.str();
    return false;
  }
  bam1_t *b = bam_init1();
  kstring_t ks = {0, 0, nullptr};
  while (sam_itr_next(impl_->sam, iter, b) > 0) {
    if (sam_format1(impl_->hdr, b, &ks) >= 0) {
      out_sam_lines.emplace_back(ks.s, ks.l);
    }
    if (ks.s) {
      free(ks.s);
      ks.s = nullptr;
      ks.l = ks.m = 0;
    }
  }
  if (ks.s)
    free(ks.s);
  bam_destroy1(b);
  hts_itr_destroy(iter);
  return true;
}

void RefReader::close() {
  std::lock_guard<std::mutex> lg(impl_->mtx);
  if (impl_->fai) {
    fai_destroy(impl_->fai);
    impl_->fai = nullptr;
  }
  if (impl_->idx) {
    hts_idx_destroy(impl_->idx);
    impl_->idx = nullptr;
  }
  if (impl_->hdr) {
    bam_hdr_destroy(impl_->hdr);
    impl_->hdr = nullptr;
  }
  if (impl_->sam) {
    sam_close(impl_->sam);
    impl_->sam = nullptr;
  }
  impl_->path.clear();
}

} // namespace amber

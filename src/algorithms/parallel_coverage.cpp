#include "parallel_coverage.h"
#include <omp.h>
#include <cmath>
#include <iostream>
#include <mutex>

namespace amber {

std::map<std::string, ContigCoverage> ParallelCoverageExtractor::extract_all(
    const std::string& bam_path,
    const std::vector<std::pair<std::string, uint32_t>>& contigs,
    int num_threads
) {
    std::map<std::string, ContigCoverage> results;

    if (contigs.empty()) return results;

    // First, open the BAM file to load index (needs valid BAM file pointer)
    samFile* tmp_fp = sam_open(bam_path.c_str(), "r");
    if (!tmp_fp) {
        std::cerr << "Warning: Cannot open BAM file " << bam_path << "\n";
        for (const auto& [name, len] : contigs) {
            results[name] = {1.0, 0.1};
        }
        return results;
    }

    // Load index using the open file handle
    hts_idx_t* shared_idx = sam_index_load(tmp_fp, bam_path.c_str());
    sam_close(tmp_fp);  // Close temp handle, we'll open new ones per thread

    if (!shared_idx) {
        std::cerr << "Warning: Cannot load BAM index for " << bam_path << "\n";
        // Return uniform coverage
        for (const auto& [name, len] : contigs) {
            results[name] = {1.0, 0.1};
        }
        return results;
    }

    // Pre-allocate results with mutex for thread-safe insertion
    std::mutex results_mutex;

    // Process contigs in parallel - each thread opens its own BAM handle
    #pragma omp parallel num_threads(num_threads)
    {
        // Each thread opens its own BAM file handle
        samFile* bam_fp = sam_open(bam_path.c_str(), "r");
        bam_hdr_t* header = nullptr;

        if (bam_fp) {
            header = sam_hdr_read(bam_fp);
        }

        if (!bam_fp || !header) {
            if (bam_fp) sam_close(bam_fp);
            #pragma omp critical
            {
                std::cerr << "Thread " << omp_get_thread_num()
                         << " failed to open BAM\n";
            }
        } else {
            bam1_t* read = bam_init1();

            #pragma omp for schedule(dynamic, 10)
            for (size_t i = 0; i < contigs.size(); i++) {
                const auto& [contig_name, contig_length] = contigs[i];

                // Get tid for this contig
                int tid = bam_name2id(header, contig_name.c_str());

                ContigCoverage cov;

                if (tid < 0) {
                    cov = {1.0, 0.1};  // Default for missing contig
                } else {
                    // Query using shared index, thread-local file handle
                    hts_itr_t* iter = sam_itr_queryi(shared_idx, tid, 0, contig_length);

                    if (iter) {
                        // Compute depth
                        std::vector<uint32_t> depth(contig_length, 0);

                        while (sam_itr_next(bam_fp, iter, read) >= 0) {
                            // Match original: skip unmapped, secondary, supplementary
                            if (read->core.flag & (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY))
                                continue;

                            int start = read->core.pos;
                            int end = bam_endpos(read);

                            for (int p = start; p < end && p < (int)contig_length; p++) {
                                if (p >= 0) depth[p]++;
                            }
                        }

                        hts_itr_destroy(iter);

                        // Compute mean and variance (matching original with Bessel correction)
                        double sum = 0;
                        for (uint32_t d : depth) {
                            sum += d;
                        }

                        size_t n = depth.size();
                        cov.mean = sum / n;

                        // Compute variance with Bessel correction (n-1) to match original
                        double var_sum = 0.0;
                        for (uint32_t d : depth) {
                            double diff = d - cov.mean;
                            var_sum += diff * diff;
                        }
                        cov.variance = var_sum / (n > 1 ? n - 1 : 1);

                    } else {
                        cov = {1.0, 0.1};
                    }
                }

                // Thread-safe insertion
                #pragma omp critical
                {
                    results[contig_name] = cov;
                }
            }

            bam_destroy1(read);
            bam_hdr_destroy(header);
            sam_close(bam_fp);
        }
    }

    hts_idx_destroy(shared_idx);

    return results;
}

} // namespace amber

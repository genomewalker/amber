#pragma once

#include <string>
#include <vector>
#include <map>
#include <htslib/sam.h>
#include <htslib/hts.h>

namespace amber {

struct ContigCoverage {
    double mean = 0.0;
    double variance = 0.0;
};

// Parallel coverage extraction using shared index with multiple BAM handles
class ParallelCoverageExtractor {
public:
    // Extract coverage for all contigs in parallel
    // Returns map of contig_name -> coverage stats
    static std::map<std::string, ContigCoverage> extract_all(
        const std::string& bam_path,
        const std::vector<std::pair<std::string, uint32_t>>& contigs,  // name, length pairs
        int num_threads = 1
    );

private:
    // Worker function for a batch of contigs
    static void extract_batch(
        const std::string& bam_path,
        hts_idx_t* shared_idx,
        const std::vector<std::pair<std::string, uint32_t>>& batch,
        std::map<std::string, ContigCoverage>& results
    );
};

} // namespace amber

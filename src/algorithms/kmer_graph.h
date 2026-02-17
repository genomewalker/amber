#pragma once
// ==============================================================================
// Minimizer-based Intra-Bin Connectivity Graph
//
// Fast computation of k-mer sharing between contigs using minimizers.
// Used to detect contaminating contigs that don't share sequences with bin core.
//
// Algorithm:
//   1. Extract minimizers from each contig (sampled k-mers)
//   2. Build inverted index: minimizer -> contigs containing it
//   3. Compute connectivity: contigs sharing minimizers are connected
//   4. Flag contigs with low connectivity to bin core
//
// Complexity: O(n * L/w) where n=contigs, L=total length, w=window size
// ==============================================================================

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>

namespace amber {
namespace kmer_graph {

// =============================================================================
// Fast 64-bit Rolling Hash (ntHash-inspired)
// =============================================================================

// Nucleotide hash seeds (from ntHash paper)
constexpr uint64_t HASH_A = 0x3c8bfbb395c60474ULL;
constexpr uint64_t HASH_C = 0x3193c18562a02b4cULL;
constexpr uint64_t HASH_G = 0x20323ed082572324ULL;
constexpr uint64_t HASH_T = 0x295549f54be24456ULL;

inline uint64_t base_hash(char c) {
    switch (c) {
        case 'A': case 'a': return HASH_A;
        case 'C': case 'c': return HASH_C;
        case 'G': case 'g': return HASH_G;
        case 'T': case 't': return HASH_T;
        default: return 0;  // N or other
    }
}

// Rotate left
inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

// Compute hash for a k-mer starting at position i
inline uint64_t kmer_hash(const std::string& seq, size_t i, int k) {
    uint64_t h = 0;
    for (int j = 0; j < k; ++j) {
        h = rotl64(h, 1) ^ base_hash(seq[i + j]);
    }
    return h;
}

// Rolling hash: update hash when sliding window by 1
inline uint64_t roll_hash(uint64_t h, char out, char in, int k) {
    // Remove contribution of outgoing base (was rotated k times)
    h ^= rotl64(base_hash(out), k);
    // Rotate and add incoming base
    h = rotl64(h, 1) ^ base_hash(in);
    return h;
}

// =============================================================================
// Minimizer Extraction
// =============================================================================

// Extract minimizers from a sequence
// Returns vector of (minimizer_hash, position) pairs
inline std::vector<uint64_t> extract_minimizers(
    const std::string& seq,
    int k = 25,      // k-mer size
    int w = 10       // window size (keep 1 minimizer per w k-mers)
) {
    std::vector<uint64_t> minimizers;

    if (seq.length() < static_cast<size_t>(k + w - 1)) {
        return minimizers;
    }

    // Compute hashes for first window
    std::vector<std::pair<uint64_t, size_t>> window;  // (hash, position)
    window.reserve(w);

    uint64_t h = kmer_hash(seq, 0, k);
    for (int i = 0; i < w; ++i) {
        if (i > 0) {
            h = roll_hash(h, seq[i - 1], seq[i + k - 1], k);
        }
        window.push_back({h, static_cast<size_t>(i)});
    }

    // Find minimum in first window
    auto min_it = std::min_element(window.begin(), window.end());
    uint64_t last_min = min_it->first;
    minimizers.push_back(last_min);

    // Slide window
    size_t n_kmers = seq.length() - k + 1;
    for (size_t i = w; i < n_kmers; ++i) {
        // Update hash for new k-mer
        h = roll_hash(h, seq[i - 1], seq[i + k - 1], k);

        // Remove old k-mer from window, add new one
        window.erase(window.begin());
        window.push_back({h, i});

        // Find new minimum
        min_it = std::min_element(window.begin(), window.end());

        // Only add if different from last minimizer (avoid duplicates)
        if (min_it->first != last_min) {
            minimizers.push_back(min_it->first);
            last_min = min_it->first;
        }
    }

    return minimizers;
}

// =============================================================================
// Connectivity Graph
// =============================================================================

struct ContigConnectivity {
    size_t contig_idx;
    int n_minimizers;           // Total minimizers in this contig
    int n_shared_contigs;       // Number of other contigs sharing any minimizer
    int n_shared_minimizers;    // Total shared minimizer count (sum over all edges)
    double connectivity_score;  // Normalized score [0, 1]
};

// Build connectivity graph for a set of contigs
// Returns connectivity info for each contig
inline std::vector<ContigConnectivity> compute_connectivity(
    const std::vector<std::string>& sequences,
    const std::vector<size_t>& contig_indices,  // Original indices
    int k = 25,
    int w = 10,
    int min_shared = 2  // Minimum shared minimizers to count as connected
) {
    size_t n = sequences.size();
    std::vector<ContigConnectivity> results(n);

    if (n < 2) {
        for (size_t i = 0; i < n; ++i) {
            results[i] = {contig_indices[i], 0, 0, 0, 1.0};
        }
        return results;
    }

    // Step 1: Extract minimizers for each contig
    std::vector<std::unordered_set<uint64_t>> minimizer_sets(n);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        auto mins = extract_minimizers(sequences[i], k, w);
        minimizer_sets[i] = std::unordered_set<uint64_t>(mins.begin(), mins.end());
    }

    // Step 2: Build inverted index (minimizer -> contigs)
    std::unordered_map<uint64_t, std::vector<size_t>> inverted_index;
    for (size_t i = 0; i < n; ++i) {
        for (uint64_t m : minimizer_sets[i]) {
            inverted_index[m].push_back(i);
        }
    }

    // Step 3: Compute pairwise shared counts using inverted index
    // shared_counts[i][j] = number of shared minimizers between i and j
    std::vector<std::unordered_map<size_t, int>> shared_counts(n);

    for (const auto& [minimizer, contigs] : inverted_index) {
        // All contigs in this list share this minimizer
        for (size_t i = 0; i < contigs.size(); ++i) {
            for (size_t j = i + 1; j < contigs.size(); ++j) {
                size_t ci = contigs[i];
                size_t cj = contigs[j];
                shared_counts[ci][cj]++;
                shared_counts[cj][ci]++;
            }
        }
    }

    // Step 4: Compute connectivity scores
    for (size_t i = 0; i < n; ++i) {
        results[i].contig_idx = contig_indices[i];
        results[i].n_minimizers = static_cast<int>(minimizer_sets[i].size());

        int n_connected = 0;
        int total_shared = 0;

        for (const auto& [other, count] : shared_counts[i]) {
            if (count >= min_shared) {
                n_connected++;
                total_shared += count;
            }
        }

        results[i].n_shared_contigs = n_connected;
        results[i].n_shared_minimizers = total_shared;

        // Connectivity score: fraction of other contigs this contig connects to
        results[i].connectivity_score = static_cast<double>(n_connected) / (n - 1);
    }

    return results;
}

// =============================================================================
// Contamination Detection
// =============================================================================

// Identify contigs with low connectivity (potential contaminants)
// Returns indices of contigs to flag for removal
inline std::vector<size_t> detect_low_connectivity(
    const std::vector<ContigConnectivity>& connectivity,
    double connectivity_threshold = 0.1,  // Connect to at least 10% of bin
    int min_connections = 2               // Absolute minimum connections
) {
    std::vector<size_t> flagged;

    // Compute median connectivity for robust threshold
    std::vector<double> scores;
    for (const auto& c : connectivity) {
        scores.push_back(c.connectivity_score);
    }
    std::sort(scores.begin(), scores.end());
    double median = scores[scores.size() / 2];

    // Adaptive threshold: max of fixed threshold and fraction of median
    double adaptive_threshold = std::max(connectivity_threshold, median * 0.3);

    for (const auto& c : connectivity) {
        if (c.connectivity_score < adaptive_threshold &&
            c.n_shared_contigs < min_connections) {
            flagged.push_back(c.contig_idx);
        }
    }

    return flagged;
}

// =============================================================================
// Bin Refinement Entry Point
// =============================================================================

struct RefinementResult {
    std::vector<size_t> contigs_to_remove;
    std::vector<ContigConnectivity> all_connectivity;
    int total_contigs;
    int flagged_contigs;
    double median_connectivity;
};

// Main entry point: refine a bin by removing low-connectivity contigs
// sequences: DNA sequences of contigs in the bin
// contig_indices: original indices of these contigs
// min_bin_size: minimum contigs to apply refinement
inline RefinementResult refine_bin(
    const std::vector<std::string>& sequences,
    const std::vector<size_t>& contig_indices,
    int k = 25,
    int w = 10,
    int min_shared = 2,
    double connectivity_threshold = 0.1,
    int min_connections = 2,
    size_t min_bin_size = 10
) {
    RefinementResult result;
    result.total_contigs = static_cast<int>(sequences.size());
    result.flagged_contigs = 0;
    result.median_connectivity = 1.0;

    // Skip small bins
    if (sequences.size() < min_bin_size) {
        return result;
    }

    // Compute connectivity
    result.all_connectivity = compute_connectivity(
        sequences, contig_indices, k, w, min_shared
    );

    // Compute median connectivity
    std::vector<double> scores;
    for (const auto& c : result.all_connectivity) {
        scores.push_back(c.connectivity_score);
    }
    std::sort(scores.begin(), scores.end());
    result.median_connectivity = scores[scores.size() / 2];

    // Detect low-connectivity contigs
    result.contigs_to_remove = detect_low_connectivity(
        result.all_connectivity,
        connectivity_threshold,
        min_connections
    );

    result.flagged_contigs = static_cast<int>(result.contigs_to_remove.size());

    return result;
}

}  // namespace kmer_graph
}  // namespace amber

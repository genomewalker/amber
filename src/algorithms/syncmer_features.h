#pragma once
// ==============================================================================
// Syncmer-based Strain Detection
//
// Syncmers are a deterministic subset of k-mers that provide stable,
// position-independent sequence fingerprints. Unlike random minimizers,
// syncmers are selected based on the position of the smallest s-mer
// within each k-mer, giving consistent sampling across sequences.
//
// For strain detection:
// - Same species/strain: ~99% syncmer Jaccard similarity
// - Different strains: ~95-98% similarity (detectable difference)
// - Different species: <90% similarity
//
// Parameters:
//   k = 21 (k-mer size, captures species-level signal)
//   s = 11 (s-mer size for syncmer selection)
//   Open syncmers: k-mer is syncmer if min s-mer is at position 0 or k-s
// ==============================================================================

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace amber {

// Parameters for syncmer extraction
struct SyncmerParams {
    int k = 21;      // k-mer size
    int s = 11;      // s-mer size for selection
    bool open = true; // Open syncmers (min at ends) vs closed (min anywhere)

    static constexpr int DEFAULT_K = 21;
    static constexpr int DEFAULT_S = 11;
};

// Syncmer set for a contig (uses 64-bit hashes for memory efficiency)
struct SyncmerSet {
    std::unordered_set<uint64_t> syncmers;
    size_t total_kmers = 0;  // Total k-mers in sequence (for density calculation)

    double density() const {
        return total_kmers > 0 ? double(syncmers.size()) / total_kmers : 0.0;
    }
};

// MinHash signature for fast approximate Jaccard (much faster than full set comparison)
struct MinHashSignature {
    static constexpr size_t NUM_HASHES = 128;  // Number of hash functions
    std::array<uint64_t, NUM_HASHES> min_hashes;

    MinHashSignature() {
        min_hashes.fill(UINT64_MAX);
    }

    // Compute approximate Jaccard similarity (O(NUM_HASHES) instead of O(|set|))
    double jaccard(const MinHashSignature& other) const {
        size_t matches = 0;
        for (size_t i = 0; i < NUM_HASHES; i++) {
            if (min_hashes[i] == other.min_hashes[i]) {
                matches++;
            }
        }
        return double(matches) / NUM_HASHES;
    }
};

// Fast hash mixing function for MinHash
inline uint64_t mix_hash(uint64_t h, uint64_t seed) {
    h ^= seed;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

// Compute MinHash signature from syncmer set (very fast)
inline MinHashSignature compute_minhash(const SyncmerSet& syncmers) {
    MinHashSignature sig;

    // Use different seeds to simulate different hash functions
    static const std::array<uint64_t, MinHashSignature::NUM_HASHES> seeds = []() {
        std::array<uint64_t, MinHashSignature::NUM_HASHES> s;
        for (size_t i = 0; i < MinHashSignature::NUM_HASHES; i++) {
            s[i] = 0x9e3779b97f4a7c15ULL * (i + 1);  // Golden ratio based
        }
        return s;
    }();

    for (uint64_t h : syncmers.syncmers) {
        for (size_t i = 0; i < MinHashSignature::NUM_HASHES; i++) {
            uint64_t mixed = mix_hash(h, seeds[i]);
            if (mixed < sig.min_hashes[i]) {
                sig.min_hashes[i] = mixed;
            }
        }
    }

    return sig;
}

// =============================================================================
// Core Syncmer Functions
// =============================================================================

// Hash a k-mer to 64-bit (uses simple rolling hash)
inline uint64_t hash_kmer(const char* seq, int k) {
    uint64_t h = 0;
    uint64_t base_val[256] = {};
    base_val['A'] = base_val['a'] = 0;
    base_val['C'] = base_val['c'] = 1;
    base_val['G'] = base_val['g'] = 2;
    base_val['T'] = base_val['t'] = 3;

    for (int i = 0; i < k; i++) {
        h = h * 4 + base_val[(unsigned char)seq[i]];
    }
    return h;
}

// Get canonical (strand-independent) hash
inline uint64_t canonical_hash(const char* seq, int k) {
    // Forward hash
    uint64_t fwd = hash_kmer(seq, k);

    // Reverse complement hash
    uint64_t rev = 0;
    uint64_t rc_val[256] = {};
    rc_val['A'] = rc_val['a'] = 3; // T
    rc_val['C'] = rc_val['c'] = 2; // G
    rc_val['G'] = rc_val['g'] = 1; // C
    rc_val['T'] = rc_val['t'] = 0; // A

    for (int i = k - 1; i >= 0; i--) {
        rev = rev * 4 + rc_val[(unsigned char)seq[i]];
    }

    return std::min(fwd, rev);
}

// Check if a k-mer is a valid syncmer (open syncmer: min s-mer at position 0 or k-s)
inline bool is_open_syncmer(const char* kmer, int k, int s) {
    if (k < s) return false;

    // Find position of minimum s-mer
    uint64_t min_hash = UINT64_MAX;
    int min_pos = -1;

    for (int i = 0; i <= k - s; i++) {
        uint64_t h = canonical_hash(kmer + i, s);
        if (h < min_hash) {
            min_hash = h;
            min_pos = i;
        }
    }

    // Open syncmer: min s-mer at first or last position
    return (min_pos == 0 || min_pos == k - s);
}

// Extract all syncmers from a sequence
inline SyncmerSet extract_syncmers(const std::string& seq, const SyncmerParams& params = {}) {
    SyncmerSet result;

    int k = params.k;
    int s = params.s;

    if ((int)seq.length() < k) return result;

    result.total_kmers = seq.length() - k + 1;

    for (size_t i = 0; i <= seq.length() - k; i++) {
        const char* kmer = seq.c_str() + i;

        // Skip if contains N
        bool has_n = false;
        for (int j = 0; j < k; j++) {
            char c = kmer[j];
            if (c != 'A' && c != 'C' && c != 'G' && c != 'T' &&
                c != 'a' && c != 'c' && c != 'g' && c != 't') {
                has_n = true;
                break;
            }
        }
        if (has_n) continue;

        if (params.open && is_open_syncmer(kmer, k, s)) {
            uint64_t h = canonical_hash(kmer, k);
            result.syncmers.insert(h);
        }
    }

    return result;
}

// Compute Jaccard similarity between two syncmer sets
inline double syncmer_jaccard(const SyncmerSet& a, const SyncmerSet& b) {
    if (a.syncmers.empty() || b.syncmers.empty()) return 0.0;

    size_t intersection = 0;
    for (uint64_t h : a.syncmers) {
        if (b.syncmers.count(h)) {
            intersection++;
        }
    }

    size_t union_size = a.syncmers.size() + b.syncmers.size() - intersection;
    return union_size > 0 ? double(intersection) / union_size : 0.0;
}

// =============================================================================
// Bin-level Analysis
// =============================================================================

struct SyncmerOutlierResult {
    std::vector<size_t> outlier_indices;  // Indices of outlier contigs
    std::vector<double> similarities;     // Mean similarity to other contigs
    double median_similarity = 0.0;       // Median pairwise similarity in bin
    double mad_similarity = 0.0;          // MAD of similarities
};

// Identify outliers in a bin based on syncmer similarity (using MinHash for speed)
// Returns indices of contigs with significantly lower similarity to others
// Complexity: O(n * syncmers) for MinHash + O(n² * 128) for comparisons
inline SyncmerOutlierResult find_syncmer_outliers(
    const std::vector<SyncmerSet>& sets,
    double z_threshold = 2.5,
    size_t min_contigs = 10) {

    SyncmerOutlierResult result;
    size_t n = sets.size();

    if (n < min_contigs) return result;

    // Step 1: Compute MinHash signatures (fast, parallelizable)
    std::vector<MinHashSignature> signatures(n);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; i++) {
        signatures[i] = compute_minhash(sets[i]);
    }

    // Step 2: Compute mean similarity for each contig using MinHash (O(n² * 128))
    result.similarities.resize(n, 0.0);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            if (i != j) {
                sum += signatures[i].jaccard(signatures[j]);
            }
        }
        result.similarities[i] = sum / (n - 1);
    }

    // Step 3: Compute median and MAD
    std::vector<double> sorted_sims = result.similarities;
    std::sort(sorted_sims.begin(), sorted_sims.end());
    result.median_similarity = sorted_sims[n / 2];

    std::vector<double> abs_devs;
    for (double s : result.similarities) {
        abs_devs.push_back(std::abs(s - result.median_similarity));
    }
    std::sort(abs_devs.begin(), abs_devs.end());
    result.mad_similarity = abs_devs[n / 2];

    // Step 4: Identify outliers (low similarity = potential contaminant)
    if (result.mad_similarity > 0.001) {  // Avoid division by near-zero
        for (size_t i = 0; i < n; i++) {
            double z = (result.median_similarity - result.similarities[i]) / result.mad_similarity;
            if (z > z_threshold) {
                result.outlier_indices.push_back(i);
            }
        }
    }

    return result;
}

}  // namespace amber

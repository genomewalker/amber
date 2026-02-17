#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <omp.h>

namespace amber {

// Lookup tables - cache-line aligned
alignas(64) static const uint8_t BASE_TO_IDX[128] = {
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    4,0,4,1,4,4,4,2,4,4,4,4,4,4,4,4,4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,
    4,0,4,1,4,4,4,2,4,4,4,4,4,4,4,4,4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4
};

alignas(64) static const uint8_t CODON_TO_AA[64] = {
    'K','N','K','N','T','T','T','T','R','S','R','S','I','I','M','I',
    'Q','H','Q','H','P','P','P','P','R','R','R','R','L','L','L','L',
    'E','D','E','D','A','A','A','A','G','G','G','G','V','V','V','V',
    '*','Y','*','Y','S','S','S','S','*','C','W','C','L','F','L','F'
};

// AA to index for hashing (0-19 valid, 20+ invalid)
alignas(64) static const uint8_t AA_IDX[128] = {
    20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,
    20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,
    20, 0, 1, 2, 3, 4, 5, 6, 7, 8,20,10,11,12,13,20,14,15,16,17,20,20,18,19,20, 9,20,20,20,20,20,20,
    20, 0, 1, 2, 3, 4, 5, 6, 7, 8,20,10,11,12,13,20,14,15,16,17,20,20,18,19,20, 9,20,20,20,20,20,20
};

// Compact ORF representation
struct ORF {
    uint32_t start;      // Start position in contig (nucleotide)
    uint16_t length_aa;  // Length in amino acids
    uint16_t contig;     // Contig index
    uint8_t frame;       // Frame 0-5
    uint8_t pad[3];
};

// Protein k-mer hit for chaining
struct PKmer {
    uint64_t hash;
    uint32_t pos_aa;     // Position in ORF (amino acids)
    uint32_t orf_id;     // ORF index
};

// Chain match between two ORFs
struct ChainMatch {
    uint32_t orf1, orf2;
    uint16_t chain_len;      // Number of k-mers in chain
    uint16_t span_aa;        // Span in amino acids
    float density;           // chain_len / total_matches
};

// Result structure
struct ProteinConflictResult {
    std::vector<size_t> contigs_to_remove;
    int total_chains;
    int contigs_in_conflict;
    bool accepted;
    std::string rejection_reason;
};

// Conflict graph result - returns pairs of conflicting contigs
struct ConflictGraphResult {
    std::vector<std::pair<size_t, size_t>> conflict_pairs;  // Pairs of conflicting contig indices
    std::vector<int> conflict_count;                         // Number of conflicts per contig
    int total_chains;
    int contigs_in_conflict;
};

// Fast ORF extraction from one frame
// Returns ORFs as (start_nt, length_aa) pairs
inline void extract_orfs_frame(
    const uint8_t* seq, size_t seq_len,
    int frame, uint16_t contig_idx,
    int min_aa,
    std::vector<ORF>& orfs_out
) {
    if (seq_len < (size_t)(frame + 3)) return;

    int in_orf = 0;
    uint32_t orf_start = 0;
    uint16_t orf_len = 0;

    for (size_t i = frame; i + 2 < seq_len; i += 3) {
        uint8_t b1 = BASE_TO_IDX[seq[i] & 0x7F];
        uint8_t b2 = BASE_TO_IDX[seq[i+1] & 0x7F];
        uint8_t b3 = BASE_TO_IDX[seq[i+2] & 0x7F];

        if ((b1 | b2 | b3) & 4) {
            // Invalid base - end ORF
            if (in_orf && orf_len >= min_aa) {
                orfs_out.push_back({orf_start, orf_len, contig_idx, (uint8_t)frame, {}});
            }
            in_orf = 0;
            orf_len = 0;
            continue;
        }

        uint8_t aa = CODON_TO_AA[(b1 << 4) | (b2 << 2) | b3];

        if (aa == '*') {
            // Stop codon - end ORF
            if (in_orf && orf_len >= min_aa) {
                orfs_out.push_back({orf_start, orf_len, contig_idx, (uint8_t)frame, {}});
            }
            in_orf = 0;
            orf_len = 0;
        } else if (aa == 'M' && !in_orf) {
            // Start codon
            in_orf = 1;
            orf_start = (uint32_t)i;
            orf_len = 1;
        } else if (in_orf) {
            orf_len++;
        }
    }

    // Handle ORF at end of sequence
    if (in_orf && orf_len >= min_aa) {
        orfs_out.push_back({orf_start, orf_len, contig_idx, (uint8_t)frame, {}});
    }
}

// Translate ORF to protein and extract k-mers
inline void extract_orf_kmers(
    const uint8_t* seq,
    const ORF& orf,
    uint32_t orf_id,
    int k,
    std::vector<PKmer>& kmers_out
) {
    if (orf.length_aa < k) return;

    // Translate and hash in single pass
    uint64_t rolling_hash = 0;
    uint64_t pow_k = 1;
    for (int i = 0; i < k - 1; i++) pow_k *= 23;

    int valid_count = 0;

    for (uint16_t aa_pos = 0; aa_pos < orf.length_aa; aa_pos++) {
        size_t nt_pos = orf.start + aa_pos * 3;

        uint8_t b1 = BASE_TO_IDX[seq[nt_pos] & 0x7F];
        uint8_t b2 = BASE_TO_IDX[seq[nt_pos + 1] & 0x7F];
        uint8_t b3 = BASE_TO_IDX[seq[nt_pos + 2] & 0x7F];

        uint8_t aa_idx;
        if ((b1 | b2 | b3) & 4) {
            aa_idx = 20;  // Invalid
        } else {
            uint8_t aa = CODON_TO_AA[(b1 << 4) | (b2 << 2) | b3];
            aa_idx = AA_IDX[aa & 0x7F];
        }

        if (aa_idx >= 20) {
            // Invalid AA - reset
            rolling_hash = 0;
            valid_count = 0;
            continue;
        }

        // Update rolling hash
        if (valid_count >= k) {
            rolling_hash = (rolling_hash - (AA_IDX[seq[orf.start + (aa_pos - k) * 3] & 0x7F] * pow_k)) * 23 + aa_idx;
        } else {
            rolling_hash = rolling_hash * 23 + aa_idx;
        }
        valid_count++;

        if (valid_count >= k) {
            kmers_out.push_back({rolling_hash, (uint32_t)(aa_pos - k + 1), orf_id});
        }
    }
}

// Find best co-linear chain between two k-mer lists - O(m log m) diagonal binning
inline ChainMatch find_best_chain(
    const PKmer* k1, size_t n1,
    const PKmer* k2, size_t n2,
    uint32_t orf1_id, uint32_t orf2_id,
    int k, [[maybe_unused]] int max_gap
) {
    ChainMatch result = {orf1_id, orf2_id, 0, 0, 0.0f};

    // Build hash index for k2
    std::vector<std::pair<uint64_t, uint32_t>> k2_idx;
    k2_idx.reserve(n2);
    for (size_t i = 0; i < n2; i++) {
        k2_idx.push_back({k2[i].hash, k2[i].pos_aa});
    }
    std::sort(k2_idx.begin(), k2_idx.end());

    // Find all matches with diagonal
    std::vector<std::pair<int32_t, int32_t>> diag_pos;  // (diagonal, pos1)

    for (size_t i = 0; i < n1; i++) {
        auto it = std::lower_bound(k2_idx.begin(), k2_idx.end(),
                                   std::make_pair(k1[i].hash, (uint32_t)0));
        while (it != k2_idx.end() && it->first == k1[i].hash) {
            int32_t diag = (int32_t)k1[i].pos_aa - (int32_t)it->second;
            diag_pos.push_back({diag, (int32_t)k1[i].pos_aa});
            ++it;
        }
    }

    if (diag_pos.size() < 3) return result;

    // Sort by (diagonal, pos1) for grouping
    std::sort(diag_pos.begin(), diag_pos.end());

    // Find best diagonal band (allowing ±2 diagonal tolerance)
    size_t best_count = 0;
    int32_t best_min_pos = 0, best_max_pos = 0;
    size_t total_matches = diag_pos.size();

    size_t i = 0;
    while (i < diag_pos.size()) {
        int32_t base_diag = diag_pos[i].first;
        int32_t min_pos = diag_pos[i].second;
        int32_t max_pos = diag_pos[i].second;
        size_t count = 1;

        // Extend to include nearby diagonals (±2)
        size_t j = i + 1;
        while (j < diag_pos.size() && diag_pos[j].first <= base_diag + 4) {
            min_pos = std::min(min_pos, diag_pos[j].second);
            max_pos = std::max(max_pos, diag_pos[j].second);
            count++;
            j++;
        }

        if (count > best_count) {
            best_count = count;
            best_min_pos = min_pos;
            best_max_pos = max_pos;
        }

        // Move to next distinct diagonal group
        while (i < diag_pos.size() && diag_pos[i].first <= base_diag + 4) i++;
    }

    if (best_count < 3) return result;

    result.chain_len = (uint16_t)best_count;
    result.span_aa = (uint16_t)(best_max_pos - best_min_pos + k);
    result.density = (float)best_count / total_matches;

    return result;
}

// Main function
inline ProteinConflictResult find_contaminants_by_protein_chains(
    const std::vector<std::string>& sequences,
    int protein_k = 8,
    int min_orf_aa = 100,
    int min_chain_len = 15,
    int min_span_aa = 60,
    float min_density = 0.5f,
    int min_orf_matches = 2,
    int max_df = 3,
    size_t min_removal = 20,
    size_t max_removal = 50,
    double max_removal_frac = 0.25
) {
    ProteinConflictResult result;
    result.accepted = false;
    result.total_chains = 0;
    result.contigs_in_conflict = 0;

    const size_t n = sequences.size();
    if (n < 50 || n > 65535) {
        result.rejection_reason = n < 50 ? "bin too small" : "bin too large";
        return result;
    }

    // Phase 1: Extract ORFs from all contigs (parallel)
    std::vector<std::vector<ORF>> thread_orfs(omp_get_max_threads());

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_orfs = thread_orfs[tid];
        local_orfs.reserve(1000);

        std::vector<uint8_t> rc_buf;

        #pragma omp for schedule(dynamic, 4)
        for (size_t ci = 0; ci < n; ci++) {
            const std::string& seq = sequences[ci];
            const uint8_t* seq_ptr = (const uint8_t*)seq.data();
            size_t seq_len = seq.length();

            // Forward frames
            for (int frame = 0; frame < 3; frame++) {
                extract_orfs_frame(seq_ptr, seq_len, frame, (uint16_t)ci, min_orf_aa, local_orfs);
            }

            // Reverse complement
            rc_buf.resize(seq_len);
            for (size_t i = 0; i < seq_len; i++) {
                char c = seq[seq_len - 1 - i];
                rc_buf[i] = (c == 'A' || c == 'a') ? 'T' :
                            (c == 'T' || c == 't') ? 'A' :
                            (c == 'C' || c == 'c') ? 'G' :
                            (c == 'G' || c == 'g') ? 'C' : 'N';
            }

            // Reverse frames (stored with frame 3-5)
            for (int frame = 0; frame < 3; frame++) {
                size_t orf_start = local_orfs.size();
                extract_orfs_frame(rc_buf.data(), seq_len, frame, (uint16_t)ci, min_orf_aa, local_orfs);
                // Adjust frame and start for reverse ORFs
                for (size_t i = orf_start; i < local_orfs.size(); i++) {
                    local_orfs[i].frame = frame + 3;
                    // Convert start to forward strand position
                    local_orfs[i].start = seq_len - local_orfs[i].start - local_orfs[i].length_aa * 3;
                }
            }
        }
    }

    // Merge ORFs
    std::vector<ORF> all_orfs;
    size_t total_orfs = 0;
    for (auto& to : thread_orfs) total_orfs += to.size();
    all_orfs.reserve(total_orfs);
    for (auto& to : thread_orfs) {
        all_orfs.insert(all_orfs.end(), to.begin(), to.end());
        to.clear();
    }

    if (all_orfs.size() < 10) {
        result.rejection_reason = "too few ORFs: " + std::to_string(all_orfs.size());
        return result;
    }

    // Phase 2: Extract k-mers from ORFs (parallel)
    std::vector<std::vector<PKmer>> thread_kmers(omp_get_max_threads());
    std::vector<size_t> orf_kmer_start(all_orfs.size() + 1);

    // First pass: count k-mers per ORF
    std::vector<size_t> orf_kmer_count(all_orfs.size());

    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t oi = 0; oi < all_orfs.size(); oi++) {
        const ORF& orf = all_orfs[oi];
        orf_kmer_count[oi] = (orf.length_aa >= protein_k) ? (orf.length_aa - protein_k + 1) : 0;
    }

    // Compute prefix sum
    orf_kmer_start[0] = 0;
    for (size_t i = 0; i < all_orfs.size(); i++) {
        orf_kmer_start[i + 1] = orf_kmer_start[i] + orf_kmer_count[i];
    }

    // Allocate and fill
    std::vector<PKmer> all_kmers(orf_kmer_start.back());

    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t oi = 0; oi < all_orfs.size(); oi++) {
        const ORF& orf = all_orfs[oi];
        if (orf.length_aa < protein_k) continue;

        const std::string& seq = sequences[orf.contig];
        std::vector<PKmer> local_kmers;
        local_kmers.reserve(orf.length_aa);

        extract_orf_kmers((const uint8_t*)seq.data(), orf, (uint32_t)oi, protein_k, local_kmers);

        // Copy to output array
        size_t out_pos = orf_kmer_start[oi];
        for (size_t i = 0; i < local_kmers.size() && out_pos < orf_kmer_start[oi + 1]; i++) {
            all_kmers[out_pos++] = local_kmers[i];
        }
    }

    // Phase 3: Build k-mer index and filter by df
    // Sort k-mers by hash
    std::vector<size_t> kmer_order(all_kmers.size());
    for (size_t i = 0; i < kmer_order.size(); i++) kmer_order[i] = i;

    std::sort(kmer_order.begin(), kmer_order.end(), [&](size_t a, size_t b) {
        return all_kmers[a].hash < all_kmers[b].hash;
    });

    // Find k-mers with df 2-max_df and collect ORF pairs
    std::vector<std::pair<uint32_t, uint32_t>> candidate_orf_pairs;

    size_t i = 0;
    while (i < kmer_order.size()) {
        size_t j = i + 1;
        uint64_t h = all_kmers[kmer_order[i]].hash;
        while (j < kmer_order.size() && all_kmers[kmer_order[j]].hash == h) j++;

        size_t group_size = j - i;

        if (group_size >= 2 && group_size <= (size_t)max_df * 2) {
            // Collect unique ORFs
            std::vector<uint32_t> orfs_with_kmer;
            orfs_with_kmer.reserve(group_size);
            for (size_t k = i; k < j; k++) {
                uint32_t orf_id = all_kmers[kmer_order[k]].orf_id;
                if (orfs_with_kmer.empty() || orfs_with_kmer.back() != orf_id) {
                    orfs_with_kmer.push_back(orf_id);
                }
            }

            // Unique ORFs
            std::sort(orfs_with_kmer.begin(), orfs_with_kmer.end());
            orfs_with_kmer.erase(std::unique(orfs_with_kmer.begin(), orfs_with_kmer.end()), orfs_with_kmer.end());

            // Filter by df (unique contigs)
            std::vector<uint16_t> contigs;
            for (uint32_t orf_id : orfs_with_kmer) {
                contigs.push_back(all_orfs[orf_id].contig);
            }
            std::sort(contigs.begin(), contigs.end());
            contigs.erase(std::unique(contigs.begin(), contigs.end()), contigs.end());

            if (contigs.size() >= 2 && contigs.size() <= (size_t)max_df) {
                // Add all ORF pairs from different contigs
                for (size_t a = 0; a < orfs_with_kmer.size(); a++) {
                    for (size_t b = a + 1; b < orfs_with_kmer.size(); b++) {
                        if (all_orfs[orfs_with_kmer[a]].contig != all_orfs[orfs_with_kmer[b]].contig) {
                            candidate_orf_pairs.push_back({orfs_with_kmer[a], orfs_with_kmer[b]});
                        }
                    }
                }
            }
        }

        i = j;
    }

    if (candidate_orf_pairs.empty()) {
        result.rejection_reason = "no candidate ORF pairs";
        return result;
    }

    // Unique and sort pairs
    std::sort(candidate_orf_pairs.begin(), candidate_orf_pairs.end());
    candidate_orf_pairs.erase(std::unique(candidate_orf_pairs.begin(), candidate_orf_pairs.end()), candidate_orf_pairs.end());

    // Phase 4: Find chains for candidate pairs (parallel)
    std::vector<ChainMatch> good_matches;
    std::mutex match_mutex;

    #pragma omp parallel
    {
        std::vector<ChainMatch> local_matches;

        #pragma omp for schedule(dynamic, 64)
        for (size_t pi = 0; pi < candidate_orf_pairs.size(); pi++) {
            auto [orf1, orf2] = candidate_orf_pairs[pi];

            size_t k1_start = orf_kmer_start[orf1];
            size_t k1_end = orf_kmer_start[orf1 + 1];
            size_t k2_start = orf_kmer_start[orf2];
            size_t k2_end = orf_kmer_start[orf2 + 1];

            if (k1_end <= k1_start || k2_end <= k2_start) continue;

            ChainMatch match = find_best_chain(
                &all_kmers[k1_start], k1_end - k1_start,
                &all_kmers[k2_start], k2_end - k2_start,
                orf1, orf2, protein_k, 15
            );

            if (match.chain_len >= min_chain_len &&
                match.span_aa >= min_span_aa &&
                match.density >= min_density) {
                local_matches.push_back(match);
            }
        }

        std::lock_guard<std::mutex> lock(match_mutex);
        good_matches.insert(good_matches.end(), local_matches.begin(), local_matches.end());
    }

    if (good_matches.empty()) {
        result.rejection_reason = "no strong chain matches";
        return result;
    }

    // Phase 5: Count ORF matches per contig pair
    std::vector<std::vector<int>> contig_match_count(n, std::vector<int>(n, 0));

    for (const auto& m : good_matches) {
        uint16_t c1 = all_orfs[m.orf1].contig;
        uint16_t c2 = all_orfs[m.orf2].contig;
        contig_match_count[c1][c2]++;
        contig_match_count[c2][c1]++;
        result.total_chains++;
    }

    // Phase 6: Build conflict graph (require min_orf_matches)
    std::vector<std::vector<uint16_t>> conflicts(n);

    for (size_t c1 = 0; c1 < n; c1++) {
        for (size_t c2 = c1 + 1; c2 < n; c2++) {
            if (contig_match_count[c1][c2] >= min_orf_matches) {
                conflicts[c1].push_back((uint16_t)c2);
                conflicts[c2].push_back((uint16_t)c1);
            }
        }
    }

    // Count contigs in conflict
    for (size_t c = 0; c < n; c++) {
        if (!conflicts[c].empty()) {
            result.contigs_in_conflict++;
        }
    }

    if (result.contigs_in_conflict < 10) {
        result.rejection_reason = "insufficient conflicts: " + std::to_string(result.contigs_in_conflict);
        return result;
    }

    // Phase 7: Find minority component using simple community detection
    // Use connected components, then pick the smaller one
    std::vector<int> component(n, -1);
    int num_components = 0;

    for (size_t c = 0; c < n; c++) {
        if (component[c] >= 0 || conflicts[c].empty()) continue;

        // BFS
        std::vector<size_t> queue;
        queue.push_back(c);
        component[c] = num_components;

        for (size_t qi = 0; qi < queue.size(); qi++) {
            size_t node = queue[qi];
            for (uint16_t neighbor : conflicts[node]) {
                if (component[neighbor] < 0) {
                    component[neighbor] = num_components;
                    queue.push_back(neighbor);
                }
            }
        }

        num_components++;
    }

    // If only one component, use greedy removal
    if (num_components <= 1) {
        // Greedy removal by conflict count
        std::vector<bool> removed(n, false);
        size_t max_to_remove = std::min(max_removal, (size_t)(n * max_removal_frac));
        size_t num_removed = 0;

        while (num_removed < max_to_remove) {
            int best_count = 0;
            size_t best_node = SIZE_MAX;

            for (size_t c = 0; c < n; c++) {
                if (removed[c] || conflicts[c].empty()) continue;

                int count = 0;
                for (uint16_t neighbor : conflicts[c]) {
                    if (!removed[neighbor]) count++;
                }

                if (count > best_count) {
                    best_count = count;
                    best_node = c;
                }
            }

            if (best_node == SIZE_MAX || (best_count == 0 && num_removed >= min_removal)) break;

            removed[best_node] = true;
            num_removed++;
        }

        if (num_removed < min_removal) {
            result.rejection_reason = "removal set too small: " + std::to_string(num_removed);
            return result;
        }

        for (size_t c = 0; c < n; c++) {
            if (removed[c]) result.contigs_to_remove.push_back(c);
        }
    } else {
        // Pick smaller component(s) as contaminants
        std::vector<size_t> comp_size(num_components, 0);
        for (size_t c = 0; c < n; c++) {
            if (component[c] >= 0) comp_size[component[c]]++;
        }

        // Find largest component
        int largest = 0;
        for (int i = 1; i < num_components; i++) {
            if (comp_size[i] > comp_size[largest]) largest = i;
        }

        // Remove all non-largest components (if size is reasonable)
        size_t total_remove = 0;
        for (int i = 0; i < num_components; i++) {
            if (i != largest) total_remove += comp_size[i];
        }

        if (total_remove >= min_removal && total_remove <= max_removal) {
            for (size_t c = 0; c < n; c++) {
                if (component[c] >= 0 && component[c] != largest) {
                    result.contigs_to_remove.push_back(c);
                }
            }
        } else {
            result.rejection_reason = "component sizes not suitable: " + std::to_string(total_remove);
            return result;
        }
    }

    result.accepted = !result.contigs_to_remove.empty();
    return result;
}

// Simplified function that returns conflict graph without removal decisions
// Let the caller combine with TNF/CGR info to decide which contigs to remove
inline ConflictGraphResult find_protein_conflicts(
    const std::vector<std::string>& sequences,
    int protein_k = 6,
    int min_orf_aa = 60,
    int min_chain_len = 10,
    int min_orf_matches = 2,
    int max_df = 5
) {
    ConflictGraphResult result;
    result.total_chains = 0;
    result.contigs_in_conflict = 0;

    size_t n = sequences.size();
    if (n < 2) return result;

    result.conflict_count.resize(n, 0);

    // Phase 1: Extract ORFs from all contigs
    std::vector<ORF> all_orfs;
    for (size_t ci = 0; ci < n; ci++) {
        const auto& seq = sequences[ci];
        const uint8_t* data = reinterpret_cast<const uint8_t*>(seq.data());
        size_t len = seq.size();

        for (int frame = 0; frame < 6; frame++) {
            extract_orfs_frame(data, len, frame, (uint16_t)ci, min_orf_aa, all_orfs);
        }
    }

    if (all_orfs.size() < 10) return result;

    // Phase 2: Extract k-mers from ORFs
    std::vector<PKmer> all_kmers;
    all_kmers.reserve(all_orfs.size() * 50);

    for (size_t oi = 0; oi < all_orfs.size(); oi++) {
        const auto& orf = all_orfs[oi];
        const auto& seq = sequences[orf.contig];
        const uint8_t* data = reinterpret_cast<const uint8_t*>(seq.data());

        std::vector<PKmer> local_kmers;
        extract_orf_kmers(data, orf, (uint32_t)oi, protein_k, local_kmers);
        all_kmers.insert(all_kmers.end(), local_kmers.begin(), local_kmers.end());
    }

    if (all_kmers.size() < 10) return result;

    // Phase 3: Sort k-mers by hash
    std::sort(all_kmers.begin(), all_kmers.end(),
        [](const PKmer& a, const PKmer& b) { return a.hash < b.hash; });

    // Phase 4: Find ORF pairs sharing k-mers (respecting df filter)
    std::unordered_map<uint64_t, int> orf_pair_kmer_count;

    size_t i = 0;
    while (i < all_kmers.size()) {
        size_t j = i + 1;
        uint64_t h = all_kmers[i].hash;
        while (j < all_kmers.size() && all_kmers[j].hash == h) j++;

        size_t group_size = j - i;

        // Collect unique ORFs with this k-mer
        std::vector<uint32_t> orfs_with_kmer;
        for (size_t k = i; k < j; k++) {
            orfs_with_kmer.push_back(all_kmers[k].orf_id);
        }
        std::sort(orfs_with_kmer.begin(), orfs_with_kmer.end());
        orfs_with_kmer.erase(std::unique(orfs_with_kmer.begin(), orfs_with_kmer.end()), orfs_with_kmer.end());

        // Filter by df (unique contigs)
        std::vector<uint16_t> contigs;
        for (uint32_t orf_id : orfs_with_kmer) {
            contigs.push_back(all_orfs[orf_id].contig);
        }
        std::sort(contigs.begin(), contigs.end());
        contigs.erase(std::unique(contigs.begin(), contigs.end()), contigs.end());

        if (contigs.size() >= 2 && contigs.size() <= (size_t)max_df) {
            // Count k-mer hits for all ORF pairs from different contigs
            for (size_t a = 0; a < orfs_with_kmer.size(); a++) {
                for (size_t b = a + 1; b < orfs_with_kmer.size(); b++) {
                    uint32_t o1 = orfs_with_kmer[a];
                    uint32_t o2 = orfs_with_kmer[b];
                    if (all_orfs[o1].contig != all_orfs[o2].contig) {
                        uint64_t key = ((uint64_t)std::min(o1, o2) << 32) | std::max(o1, o2);
                        orf_pair_kmer_count[key]++;
                    }
                }
            }
        }
        i = j;
    }

    // Phase 5: Count strong ORF chains per contig pair
    std::unordered_map<uint32_t, int> contig_pair_chains;
    for (const auto& [key, kmer_count] : orf_pair_kmer_count) {
        if (kmer_count >= min_chain_len) {
            uint32_t o1 = key >> 32;
            uint32_t o2 = key & 0xFFFFFFFF;
            uint16_t c1 = all_orfs[o1].contig;
            uint16_t c2 = all_orfs[o2].contig;
            uint32_t cpkey = ((uint32_t)std::min(c1, c2) << 16) | std::max(c1, c2);
            contig_pair_chains[cpkey]++;
            result.total_chains++;
        }
    }

    // Phase 6: Build conflict pairs (require min_orf_matches)
    for (const auto& [cpkey, chain_count] : contig_pair_chains) {
        if (chain_count >= min_orf_matches) {
            size_t c1 = cpkey >> 16;
            size_t c2 = cpkey & 0xFFFF;
            result.conflict_pairs.emplace_back(c1, c2);
            result.conflict_count[c1]++;
            result.conflict_count[c2]++;
        }
    }

    // Count contigs in conflict
    for (size_t c = 0; c < n; c++) {
        if (result.conflict_count[c] > 0) {
            result.contigs_in_conflict++;
        }
    }

    return result;
}

}  // namespace amber

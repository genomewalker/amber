#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

namespace amber::bin2 {

// Contig-to-marker mapping for SCG hard negative mining.
// contig_marker_ids[i] is sorted for O(k) intersection queries.
struct ContigMarkerLookup {
    std::vector<std::vector<int>> contig_marker_ids;
    int n_contigs     = 0;
    int total_markers = 0;  // number of distinct marker profiles (e.g. 206)
    bool empty() const { return contig_marker_ids.empty(); }
};

// Build lookup from SeedGenerator hit map and ordered contig list.
// Assigns integer IDs to marker name strings, maps contig names to indices,
// and populates sorted marker-ID vectors per contig.
inline ContigMarkerLookup build_contig_marker_lookup(
    const std::unordered_map<std::string, std::vector<std::string>>& hits_by_contig,
    const std::vector<std::pair<std::string, std::string>>& contigs) {

    std::unordered_map<std::string, int> marker_id_map;
    int next_id = 0;
    for (const auto& [contig, markers] : hits_by_contig)
        for (const auto& m : markers)
            if (!marker_id_map.count(m))
                marker_id_map[m] = next_id++;

    std::unordered_map<std::string, int> contig_idx_map;
    contig_idx_map.reserve(contigs.size());
    for (int i = 0; i < (int)contigs.size(); i++)
        contig_idx_map[contigs[i].first] = i;

    ContigMarkerLookup lookup;
    lookup.n_contigs     = (int)contigs.size();
    lookup.total_markers = next_id;
    lookup.contig_marker_ids.resize(contigs.size());

    for (const auto& [contig, markers] : hits_by_contig) {
        auto it = contig_idx_map.find(contig);
        if (it == contig_idx_map.end()) continue;
        int idx = it->second;
        for (const auto& m : markers) {
            auto mid = marker_id_map.find(m);
            if (mid != marker_id_map.end())
                lookup.contig_marker_ids[idx].push_back(mid->second);
        }
        std::sort(lookup.contig_marker_ids[idx].begin(),
                  lookup.contig_marker_ids[idx].end());
    }

    return lookup;
}

}  // namespace amber::bin2

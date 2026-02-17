#pragma once

#include "damage_profile.h"
#include "kmer_features.h"
#include <string>

namespace amber {

// Extract damage features from DamageProfile
class DamageFeatureExtractor {
public:
  // Extract damage features for a contig
  // library_type: "ds" (double-stranded) or "ss" (single-stranded)
  // max_position: maximum position in damage file (for sizing per-position vectors)
  static void extract_damage_features(const std::string &contig_id,
                                      uint32_t contig_length,
                                      const DamageProfile &damage_profile,
                                      ContigFeatures &features,
                                      const std::string &library_type = "ds",
                                      int max_position = 25);

  // Extract damage features for a range within a contig (for split contigs)
  // Looks up damage by original_contig_id and filters to [range_start, range_end)
  // Coordinates are 0-based, range_end is exclusive
  static void extract_damage_features_range(const std::string &original_contig_id,
                                           uint32_t range_start,
                                           uint32_t range_end,
                                           uint32_t fragment_length,
                                           const DamageProfile &damage_profile,
                                           ContigFeatures &features,
                                           const std::string &library_type = "ds",
                                           int max_position = 25);

private:
  // Compute average damage rate in position range
  static float compute_damage_rate(const std::vector<DamageEntry> &entries,
                                  int start_pos, int end_pos,
                                  bool use_ct);

  // Compute damage decay rate (exponential fit)
  static float compute_decay_rate(const std::vector<DamageEntry> &entries,
                                 bool use_ct);
};

} // namespace amber

#include "damage_features.h"
#include <algorithm>
#include <cmath>

namespace amber {

float DamageFeatureExtractor::compute_damage_rate(
    const std::vector<DamageEntry> &entries, int start_pos, int end_pos,
    bool use_ct) {

  if (entries.empty())
    return 0.0f;

  double sum = 0.0;
  int count = 0;

  for (const auto &entry : entries) {
    if (entry.pos >= start_pos && entry.pos <= end_pos) {
      sum += use_ct ? entry.c_to_t : entry.a_to_g;
      count++;
    }
  }

  return (count > 0) ? static_cast<float>(sum / count) : 0.0f;
}

float DamageFeatureExtractor::compute_decay_rate(
    const std::vector<DamageEntry> &entries, bool use_ct) {

  if (entries.size() < 5)
    return 0.0f;

  // Fit exponential decay: damage(x) = a * exp(-b * x)
  // Using log-linear fit: log(damage) = log(a) - b*x

  std::vector<double> x_vals, y_vals;
  for (const auto &entry : entries) {
    double rate = use_ct ? entry.c_to_t : entry.a_to_g;
    if (rate > 1e-6) { // Avoid log(0)
      x_vals.push_back(static_cast<double>(entry.pos));
      y_vals.push_back(std::log(rate));
    }
  }

  if (x_vals.size() < 5)
    return 0.0f;

  // Simple linear regression
  double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
  int n = x_vals.size();

  for (size_t i = 0; i < x_vals.size(); ++i) {
    sum_x += x_vals[i];
    sum_y += y_vals[i];
    sum_xy += x_vals[i] * y_vals[i];
    sum_xx += x_vals[i] * x_vals[i];
  }

  // Check for zero denominator (happens when all x values are identical)
  double denom = n * sum_xx - sum_x * sum_x;
  if (std::abs(denom) < 1e-10) {
    return 0.0f;  // No variance in position → no decay to fit
  }

  double slope = (n * sum_xy - sum_x * sum_y) / denom;

  return static_cast<float>(-slope); // Return decay rate (positive value)
}

void DamageFeatureExtractor::extract_damage_features(
    const std::string &contig_id, uint32_t contig_length,
    const DamageProfile &damage_profile, ContigFeatures &features,
    const std::string &library_type, int max_position) {

  (void)contig_length; // Reserved for future length-aware features

  // Initialize per-position damage vectors (all zeros)
  features.ct_damage_5p_pos.assign(max_position, 0.0f);
  features.ag_damage_5p_pos.assign(max_position, 0.0f);
  features.ct_damage_3p_pos.assign(max_position, 0.0f);
  features.ag_damage_3p_pos.assign(max_position, 0.0f);

  // Find damage entries for this contig
  auto it = damage_profile.per_contig.find(contig_id);
  if (it == damage_profile.per_contig.end()) {
    // No damage data for this contig - all zeros already set
    features.ct_damage_5p = features.ct_damage_3p = 0.0f;
    features.ag_damage_5p = features.ag_damage_3p = 0.0f;
    features.damage_gradient = features.damage_asymmetry = 0.0f;
    return;
  }

  const auto &entries = it->second;

  // Fill per-position damage vectors
  for (const auto &entry : entries) {
    if (entry.pos >= 1 && entry.pos <= max_position) {
      int idx = entry.pos - 1;  // Convert to 0-based index
      features.ct_damage_5p_pos[idx] = static_cast<float>(entry.c_to_t);
      features.ag_damage_5p_pos[idx] = static_cast<float>(entry.a_to_g);
      // For 3' end, use reverse strand data (G->A, T->C)
      features.ag_damage_3p_pos[idx] = static_cast<float>(entry.g_to_a);
      features.ct_damage_3p_pos[idx] = static_cast<float>(entry.t_to_c);
    }
  }

  if (library_type == "ds") {
    // DOUBLE-STRANDED library (most ancient DNA)
    // Damage entries are stored by READ POSITION (1-max_position from 5' end of reads)
    // - Forward reads: C→T at their 5' end (entry.c_to_t at positions 1-max_position)
    // - Reverse reads: G→A at their 5' end (entry.g_to_a at positions 1-max_position)
    //   Note: Reverse read 5' end = molecule 3' end

    // 5' damage: C→T from forward reads
    features.ct_damage_5p = compute_damage_rate(entries, 1, max_position, true);

    // 3' damage: G→A from reverse reads at SAME positions
    // (reverse reads' 5' end damage appears at molecule 3' end)
    double sum_ga = 0.0;
    int count_ga = 0;
    for (const auto &entry : entries) {
      if (entry.pos >= 1 && entry.pos <= max_position) {
        sum_ga += entry.g_to_a;  // G→A from reverse reads
        count_ga++;
      }
    }
    features.ag_damage_3p = (count_ga > 0) ? static_cast<float>(sum_ga / count_ga) : 0.0f;

    // For ds, these are typically low (no damage at these positions)
    features.ag_damage_5p = 0.0f;  // G→A at 5' end (not expected in ds)
    features.ct_damage_3p = 0.0f;  // C→T at 3' end (not expected in ds)

  } else {
    // SINGLE-STRANDED library (less common, e.g., UDG-half treated)
    // Both C→T and G→A damage at BOTH molecule ends
    // Damage entries stored by READ POSITION (1-max_position from 5' end of reads)

    // 5' damage from forward reads
    features.ct_damage_5p = compute_damage_rate(entries, 1, max_position, true);  // C→T

    // 3' damage from reverse reads
    double sum_ga = 0.0;
    int count_ga = 0;
    for (const auto &entry : entries) {
      if (entry.pos >= 1 && entry.pos <= max_position) {
        sum_ga += entry.g_to_a;  // G→A
        count_ga++;
      }
    }
    features.ag_damage_3p = (count_ga > 0) ? static_cast<float>(sum_ga / count_ga) : 0.0f;

    // For ss, mirror patterns: both damage types at both ends
    features.ag_damage_5p = features.ag_damage_3p;  // G→A also at 5' end
    features.ct_damage_3p = features.ct_damage_5p;  // C→T also at 3' end
  }

  // Damage gradient (C→T decay rate from 5' end)
  features.damage_gradient = compute_decay_rate(entries, true);

  // Damage asymmetry (5' vs 3' difference)
  // For ds: high asymmetry (C→T at 5', G→A at 3')
  // For ss: low asymmetry (both patterns at both ends)
  features.damage_asymmetry = std::abs(features.ct_damage_5p - features.ag_damage_3p);
}

void DamageFeatureExtractor::extract_damage_features_range(
    const std::string &original_contig_id, uint32_t range_start,
    uint32_t range_end, uint32_t fragment_length,
    const DamageProfile &damage_profile, ContigFeatures &features,
    const std::string &library_type, int max_position) {

  // For split contigs: damage patterns are measured from READ ENDS, not contig coordinates
  // Therefore, the damage pattern for a fragment is the SAME as the original contig
  // We just need to look up by original_contig_id instead of fragment name
  // No coordinate transformation needed - damage is independent of split position

  (void)range_start;    // Unused - damage is read-end relative, not position-dependent
  (void)range_end;      // Unused - damage is read-end relative, not position-dependent

  // Simply extract damage features using the original contig ID
  // This gives us the damage pattern from all reads that mapped to the original contig
  extract_damage_features(original_contig_id, fragment_length, damage_profile,
                         features, library_type, max_position);
}

} // namespace amber

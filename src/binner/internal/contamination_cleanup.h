// MFDFA Binner - Contamination Cleanup
// Uses downsample variance (var_80%) as validated metric
#pragma once

namespace amber {

// Contamination detection thresholds (validated on real metagenomic bins)
// - downsample_var (var_80%): r=+0.528 with contamination on real bins
// - cgr_occupancy: r=+0.368 with contamination on real bins
// - lw_jsd: USELESS (r=-0.02 on real bins despite r=+0.446 on GUNC)
constexpr double CONTAM_VAR80_THRESHOLD = 0.02;   // High var_80% = likely contaminated
constexpr double CONTAM_CGR_OCC_THRESHOLD = 0.5;  // High CGR occupancy + high var = suspect
constexpr double CONTAM_MAX_REMOVAL = 0.15;       // Max 15% of bin length removed

}  // namespace amber

// AMBER - damage_extraction_stub.cpp
// Stub for extract_damage_profiles() when built without LibTorch.
// The real implementation lives in bin2.cpp (requires USE_LIBTORCH).

#include "../bin2/encoder/damage_aware_infonce.h"

#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>

namespace amber {

std::unordered_map<std::string, bin2::DamageProfile> extract_damage_profiles(
    const std::string& /*bam_path*/,
    const std::vector<std::pair<std::string, std::string>>& /*contigs*/,
    int /*threads*/,
    int /*n_terminal*/,
    int /*min_insert_size*/) {
    std::cerr << "amber damage requires LibTorch support.\n"
              << "Rebuild with: cmake -DAMBER_USE_TORCH=ON ..\n";
    return {};
}

}  // namespace amber

// AMBER - cmd_bin_stub.cpp
// Placeholder for amber bin when built without LibTorch support.

#include <iostream>

namespace amber {

int cmd_bin(int argc, char** argv) {
    (void)argc; (void)argv;
    std::cerr << "amber bin requires LibTorch support.\n"
              << "Rebuild with: cmake -DAMBER_USE_TORCH=ON ..\n"
              << "\n"
              << "CPU build (no GPU required):\n"
              << "  conda install pytorch cpuonly -c pytorch\n"
              << "  cmake -DAMBER_USE_TORCH=ON -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ..\n"
              << "\n"
              << "GPU build:\n"
              << "  conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia\n"
              << "  cmake -DAMBER_USE_TORCH=ON -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ..\n";
    return 1;
}

}  // namespace amber

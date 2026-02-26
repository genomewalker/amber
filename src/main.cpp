// AMBER - Ancient Metagenomic BinnER
// Main entry point with git-style subcommand dispatch

#include <amber/config.hpp>
#include <iostream>
#include <string>
#include <cstring>

// Forward declarations for subcommands
namespace amber {
    int cmd_bin(int argc, char** argv);
    int cmd_chimera(int argc, char** argv);
    int cmd_deconvolve(int argc, char** argv);
    int cmd_seeds(int argc, char** argv);
    int cmd_resolve(int argc, char** argv);
    int cmd_damage(int argc, char** argv);
}

constexpr const char* CODENAME = "Ancient Metagenomic BinnER";

static void print_version() {
    std::cout << "amber " << amber::VERSION << "\n";
    std::cout << CODENAME << "\n";
}

static void print_usage(const char* prog) {
    std::cerr << "AMBER - Ancient Metagenomic BinnER\n";
    std::cerr << "Version: " << amber::VERSION << "\n\n";
    std::cerr << "Usage: " << prog << " <command> [options]\n\n";
    std::cerr << "Commands:\n";
    std::cerr << "  bin              Bin contigs (damage-aware contrastive learning)\n";
    std::cerr << "  resolve          Aggregate bins from multiple independent runs\n";
    std::cerr << "  chimera          Detect chimeric contigs and misassemblies\n";
    std::cerr << "  deconvolve       Separate ancient and modern DNA populations\n";
    std::cerr << "  seeds            Generate SCG marker seeds for binning\n";
    std::cerr << "  damage           Compute per-bin aDNA damage statistics\n";
    std::cerr << "\n";
    std::cerr << "Options:\n";
    std::cerr << "  -h, --help     Show this help message\n";
    std::cerr << "  -v, --version  Show version information\n";
    std::cerr << "\n";
    std::cerr << "Examples:\n";
    std::cerr << "  amber bin --contigs contigs.fa --bam alignments.bam --output bins/\n";
    std::cerr << "  amber chimera --contigs contigs.fa --bam alignments.bam --output chimeras/\n";
    std::cerr << "\n";
    std::cerr << "For command-specific help, use: amber <command> --help\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "-h" || cmd == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    if (cmd == "-v" || cmd == "--version") {
        print_version();
        return 0;
    }

    if (cmd == "bin") {
        return amber::cmd_bin(argc, argv);
    } else if (cmd == "resolve") {
        return amber::cmd_resolve(argc - 1, argv + 1);
    } else if (cmd == "chimera") {
        return amber::cmd_chimera(argc - 1, argv + 1);
    } else if (cmd == "deconvolve") {
        return amber::cmd_deconvolve(argc - 1, argv + 1);
    } else if (cmd == "seeds") {
        return amber::cmd_seeds(argc - 1, argv + 1);
    } else if (cmd == "damage") {
        return amber::cmd_damage(argc - 1, argv + 1);
    } else {
        std::cerr << "Error: Unknown command '" << cmd << "'\n\n";
        print_usage(argv[0]);
        return 1;
    }
}

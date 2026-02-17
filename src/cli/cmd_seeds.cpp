// AMBER - cmd_seeds.cpp
// CLI handler for the 'seeds' subcommand

#include "cli_common.h"
#include "../seeds/seed_generator.h"
#include <iostream>

namespace amber {

CLICommand make_seeds_command() {
    CLICommand cmd;
    cmd.name = "seeds";
    cmd.description = "Generate SCG marker-based seed contigs for COMEBin-style binning.";
    cmd.description_extra = {
        "",
        "Uses FragGeneScan for ORF prediction and HMMER for marker gene detection.",
        "Seeds are contigs containing single-copy marker genes used to initialize clustering.",
    };

    cmd.options = {
        {"--contigs", "FILE", "Input contigs FASTA file", "", true},
        {"--output", "FILE", "Output seed contig list file", "", true},
        {"--threads", "N", "Number of threads", "4"},
        {"--hmm", "FILE", "HMM profile file for markers", "auxiliary/bacar_marker.hmm"},
        {"--min-length", "N", "Minimum contig length for seeds", "1001"},
        {"--hmmsearch", "PATH", "Path to hmmsearch binary", "hmmsearch"},
        {"--fraggenescan", "PATH", "Path to FragGeneScan binary", "FragGeneScan"},
        {"-h, --help", "", "Show this help message"},
    };

    cmd.outputs = {
        {"<output>", "List of seed contig names (one per line)"},
        {"<output>.hmmout", "HMMER domain table output"},
        {"<output>.frag.*", "FragGeneScan intermediate files"},
    };

    cmd.note = "Requires FragGeneScan and HMMER3 (hmmsearch) to be in PATH.";

    cmd.examples = {
        "amber seeds --contigs assembly.fa --output seeds.txt --threads 16",
        "amber seeds --contigs assembly.fa --output seeds.txt --hmm custom.hmm",
    };

    return cmd;
}

int cmd_seeds(int argc, char** argv) {
    CLICommand cmd = make_seeds_command();

    if (cmd.has_help_flag(argc, argv)) {
        cmd.print_help();
        return 0;
    }

    if (!cmd.validate_required(argc, argv)) {
        return 1;
    }

    // Parse arguments
    std::string contigs = cmd.get_option(argc, argv, "--contigs");
    std::string output = cmd.get_option(argc, argv, "--output");
    int threads = std::stoi(cmd.get_option(argc, argv, "--threads", "4"));
    std::string hmm = cmd.get_option(argc, argv, "--hmm", "auxiliary/bacar_marker.hmm");
    int min_length = std::stoi(cmd.get_option(argc, argv, "--min-length", "1001"));
    std::string hmmsearch = cmd.get_option(argc, argv, "--hmmsearch", "hmmsearch");
    std::string fraggenescan = cmd.get_option(argc, argv, "--fraggenescan", "FragGeneScan");

    std::cerr << "[seeds] Input: " << contigs << "\n";
    std::cerr << "[seeds] Output: " << output << "\n";
    std::cerr << "[seeds] Threads: " << threads << "\n";
    std::cerr << "[seeds] HMM: " << hmm << "\n";
    std::cerr << "[seeds] Min length: " << min_length << "\n";
    std::cerr << "[seeds] hmmsearch: " << hmmsearch << "\n";
    std::cerr << "[seeds] FragGeneScan: " << fraggenescan << "\n";

    // Run seed generation
    SeedGenerator gen(threads, hmm, min_length, hmmsearch, fraggenescan);
    auto seeds = gen.generate(contigs, output);

    if (seeds.empty()) {
        std::cerr << "[seeds] ERROR: No seeds generated\n";
        return 1;
    }

    std::cerr << "[seeds] Summary:\n";
    std::cerr << "[seeds]   Total ORFs: " << gen.total_orfs() << "\n";
    std::cerr << "[seeds]   Raw hits: " << gen.total_hits() << "\n";
    std::cerr << "[seeds]   Filtered hits (cov>=0.4): " << gen.filtered_hits() << "\n";
    std::cerr << "[seeds]   Selected markers: " << gen.selected_markers() << "\n";
    std::cerr << "[seeds]   Seed contigs: " << seeds.size() << "\n";
    std::cerr << "[seeds] Done. Seeds written to " << output << "\n";

    return 0;
}

}  // namespace amber

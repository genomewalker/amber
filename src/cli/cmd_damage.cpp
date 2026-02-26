// AMBER - cmd_damage.cpp
// Compute per-bin aDNA damage statistics from BAM + pre-assembled bins

#include "cli_common.h"
#include "../bin2/encoder/damage_aware_infonce.h"
#include "../algorithms/damage_profile.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <unordered_map>
#include <vector>
#include <string>

namespace amber {

// Defined in bin2.cpp
std::unordered_map<std::string, bin2::DamageProfile> extract_damage_profiles(
    const std::string& bam_path,
    const std::vector<std::pair<std::string, std::string>>& contigs,
    int threads,
    int n_terminal,
    int min_insert_size);

int cmd_damage(int argc, char** argv) {
    CLICommand cmd;
    cmd.name = "damage";
    cmd.description = "Compute per-bin aDNA damage statistics from BAM + pre-assembled bins.";
    cmd.description_extra = {
        "",
        "Scans terminal positions of aligned reads per contig, estimates C->T / G->A",
        "deamination rates, fits an exponential decay model, and classifies each bin",
        "as ancient, modern, or mixed.",
        "",
        "Outputs the same damage_per_bin.tsv as 'amber bin', but for any set of bins",
        "without re-running the full binning pipeline.",
    };
    cmd.options = {
        {"--bam",              "FILE", "Input BAM file (must be indexed)",                    "", true},
        {"--bins",             "DIR",  "Directory of bin FASTA files (*.fa / *.fasta)",       "", true},
        {"--output",           "FILE", "Output TSV file",                                     "damage_per_bin.tsv"},
        {"--threads",          "N",    "Number of threads",                                   "4"},
        {"--damage-positions", "N",    "Terminal positions to analyse (1-25)",                "15"},
        {"--min-insert",       "N",    "Skip paired reads with insert size < N (0 = off)",    "0"},
        {"-h, --help",         "",     "Show this help message"},
    };
    cmd.outputs = {
        {"<output>", "TSV: bin, p_ancient, lambda5, lambda3, amplitude_5p, amplitude_3p, ct_1p, ga_1p, damage_class"},
    };
    cmd.examples = {
        "amber damage --bam mapping.bam --bins bins/ --output damage_per_bin.tsv",
        "amber damage --bam mapping.bam --bins bins/ --threads 16 --damage-positions 20",
    };

    if (cmd.has_help_flag(argc, argv)) {
        cmd.print_help();
        return 0;
    }
    if (!cmd.validate_required(argc, argv)) {
        return 1;
    }

    const std::string bam_path  = cmd.get_option(argc, argv, "--bam");
    const std::string bins_dir  = cmd.get_option(argc, argv, "--bins");
    const std::string output    = cmd.get_option(argc, argv, "--output", "damage_per_bin.tsv");
    const int threads           = std::stoi(cmd.get_option(argc, argv, "--threads", "4"));
    const int n_terminal        = std::stoi(cmd.get_option(argc, argv, "--damage-positions", "15"));
    const int min_insert        = std::stoi(cmd.get_option(argc, argv, "--min-insert", "0"));

    if (n_terminal < 1 || n_terminal > 25) {
        std::cerr << "[damage] ERROR: --damage-positions must be between 1 and 25\n";
        return 1;
    }

    std::cerr << "[damage] BAM:              " << bam_path  << "\n";
    std::cerr << "[damage] Bins dir:         " << bins_dir  << "\n";
    std::cerr << "[damage] Output:           " << output    << "\n";
    std::cerr << "[damage] Threads:          " << threads   << "\n";
    std::cerr << "[damage] Damage positions: " << n_terminal << "\n";
    if (min_insert > 0)
        std::cerr << "[damage] Min insert size:  " << min_insert << " (skipping overlapping pairs)\n";

    // Read all bin FASTAs; build flat contig list and contig->bin map
    std::vector<std::pair<std::string, std::string>> contigs;
    std::unordered_map<std::string, std::string> contig_to_bin;
    int n_bins = 0;

    namespace fs = std::filesystem;
    for (const auto& entry : fs::directory_iterator(bins_dir)) {
        const std::string ext = entry.path().extension().string();
        if (ext != ".fa" && ext != ".fasta") continue;

        const std::string bin_name = entry.path().stem().string();
        std::ifstream f(entry.path());
        if (!f) {
            std::cerr << "[damage] WARNING: Cannot open " << entry.path() << "\n";
            continue;
        }

        std::string line, name, seq;
        auto flush = [&]() {
            if (!name.empty()) {
                contigs.push_back({name, seq});
                contig_to_bin[name] = bin_name;
            }
        };
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (line[0] == '>') {
                flush();
                name = line.substr(1);
                const auto sp = name.find(' ');
                if (sp != std::string::npos) name = name.substr(0, sp);
                seq.clear();
            } else {
                seq += line;
            }
        }
        flush();
        n_bins++;
    }

    if (contigs.empty()) {
        std::cerr << "[damage] ERROR: No contigs found in " << bins_dir << "\n";
        return 1;
    }
    std::cerr << "[damage] Loaded " << contigs.size() << " contigs from " << n_bins << " bins\n";

    // Per-contig damage profiles
    std::cerr << "[damage] Scanning BAM for damage signatures...\n";
    const auto damage_map = extract_damage_profiles(bam_path, contigs, threads, n_terminal, min_insert);

    // Accumulate per bin
    struct BinAccum {
        std::vector<uint64_t> n5, k5, n3, k3;
        double sum_panc   = 0.0;
        int    total_reads = 0;
        explicit BinAccum(int np) : n5(np, 0), k5(np, 0), n3(np, 0), k3(np, 0) {}
    };
    std::unordered_map<std::string, BinAccum> bin_accum;

    for (const auto& [cname, _] : contigs) {
        const auto bit = contig_to_bin.find(cname);
        if (bit == contig_to_bin.end()) continue;
        const auto dit = damage_map.find(cname);
        if (dit == damage_map.end()) continue;

        const std::string& bname = bit->second;
        const auto& dp = dit->second;

        if (bin_accum.find(bname) == bin_accum.end())
            bin_accum.emplace(bname, BinAccum(n_terminal));
        auto& ba = bin_accum.at(bname);

        ba.sum_panc    += static_cast<double>(dp.p_anc) * dp.n_reads;
        ba.total_reads += dp.n_reads;
        for (int p = 0; p < n_terminal; ++p) {
            ba.n5[p] += dp.n_opp_5p[p];
            ba.k5[p] += dp.n_ct_5p[p];
            ba.n3[p] += dp.n_opp_3p[p];
            ba.k3[p] += dp.n_ga_3p[p];
        }
    }

    // Write TSV
    std::ofstream out(output);
    if (!out) {
        std::cerr << "[damage] ERROR: Cannot open output file: " << output << "\n";
        return 1;
    }
    out << "bin\tp_ancient\tlambda5\tlambda3\tamplitude_5p\tamplitude_3p\tct_1p\tga_1p\tdamage_class\n";

    int written = 0;
    for (auto& [bname, ba] : bin_accum) {
        const float p_anc = (ba.total_reads > 0)
                          ? static_cast<float>(ba.sum_panc / ba.total_reads) : 0.5f;
        const auto model = fit_damage_model(ba.n5, ba.k5, ba.n3, ba.k3, 0.5);
        const float ct1  = (ba.n5[0] > 0) ? static_cast<float>(ba.k5[0]) / ba.n5[0] : 0.0f;
        const float ga1  = (ba.n3[0] > 0) ? static_cast<float>(ba.k3[0]) / ba.n3[0] : 0.0f;

        const char* cls = (p_anc > 0.5f && ct1 > 0.02f) ? "ancient"
                        : (p_anc < 0.2f)                 ? "modern"
                                                          : "mixed";

        out << bname << "\t"
            << std::fixed << std::setprecision(6)
            << p_anc << "\t"
            << model.curve_5p.lambda    << "\t" << model.curve_3p.lambda    << "\t"
            << model.curve_5p.amplitude << "\t" << model.curve_3p.amplitude << "\t"
            << ct1 << "\t" << ga1 << "\t" << cls << "\n";
        written++;
    }

    std::cerr << "[damage] Wrote " << written << " bins to " << output << "\n";
    return 0;
}

}  // namespace amber

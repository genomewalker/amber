// AMBER - cli_common.cpp
// Common CLI infrastructure implementation

#include "cli_common.h"
#include <iostream>
#include <algorithm>
#include <iomanip>

namespace amber {

void CLICommand::print_help() const {
    std::cerr << "Usage: amber " << name << " [options]\n\n";
    std::cerr << description << "\n";

    for (const auto& line : description_extra) {
        std::cerr << line << "\n";
    }
    std::cerr << "\n";

    // Collect required and optional options
    std::vector<const CLIOption*> required_opts;
    std::vector<const CLIOption*> optional_opts;

    for (const auto& opt : options) {
        if (opt.required) {
            required_opts.push_back(&opt);
        } else {
            optional_opts.push_back(&opt);
        }
    }

    // Calculate column width based on longest option/output
    const size_t MIN_COL = 21;
    size_t opt_col = MIN_COL;
    size_t out_col = MIN_COL;

    for (const auto& opt : options) {
        size_t len = 2 + opt.name.length();
        if (!opt.arg_name.empty()) len += 1 + opt.arg_name.length();
        if (len + 1 > opt_col) opt_col = len + 1;
    }
    for (const auto& out : outputs) {
        size_t len = 2 + out.filename.length();
        if (len + 1 > out_col) out_col = len + 1;
    }

    // Print required section
    if (!required_opts.empty()) {
        std::cerr << "Required:\n";
        for (const auto* opt : required_opts) {
            std::string opt_str = "  " + opt->name;
            if (!opt->arg_name.empty()) {
                opt_str += " " + opt->arg_name;
            }
            while (opt_str.length() < opt_col) opt_str += " ";
            std::cerr << opt_str << opt->description << "\n";
        }
        std::cerr << "\n";
    }

    // Print options section
    std::cerr << "Options:\n";
    for (const auto* opt : optional_opts) {
        std::string opt_str = "  " + opt->name;
        if (!opt->arg_name.empty()) {
            opt_str += " " + opt->arg_name;
        }
        while (opt_str.length() < opt_col) opt_str += " ";

        std::string desc = opt->description;
        if (!opt->default_value.empty()) {
            desc += " (default: " + opt->default_value + ")";
        }
        std::cerr << opt_str << desc << "\n";
    }
    std::cerr << "\n";

    // Print output section
    if (!outputs.empty()) {
        std::cerr << "Output:\n";
        for (const auto& out : outputs) {
            std::string out_str = "  " + out.filename;
            while (out_str.length() < out_col) out_str += " ";
            std::string desc = out.description;
            if (!out.condition.empty()) {
                desc += " " + out.condition;
            }
            std::cerr << out_str << desc << "\n";
        }
        std::cerr << "\n";
    }

    // Print note section
    if (!note.empty()) {
        std::cerr << "Note:\n  " << note << "\n\n";
    }

    // Print examples
    if (!examples.empty()) {
        std::cerr << "Example:\n";
        for (const auto& ex : examples) {
            std::cerr << "  " << ex << "\n";
        }
    }
}

bool CLICommand::has_help_flag(int argc, char** argv) const {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            return true;
        }
    }
    return false;
}

bool CLICommand::has_flag(int argc, char** argv, const std::string& flag) const {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == flag) {
            return true;
        }
    }
    return false;
}

std::string CLICommand::get_option(int argc, char** argv, const std::string& name,
                                    const std::string& default_val) const {
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return default_val;
}

std::vector<std::string> CLICommand::get_missing_required(int argc, char** argv) const {
    std::vector<std::string> missing;
    for (const auto& opt : options) {
        if (opt.required) {
            bool found = false;
            for (int i = 1; i < argc; ++i) {
                if (argv[i] == opt.name && i + 1 < argc) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                missing.push_back(opt.name);
            }
        }
    }
    return missing;
}

bool CLICommand::validate_required(int argc, char** argv) const {
    auto missing = get_missing_required(argc, argv);
    if (missing.empty()) {
        return true;
    }

    std::cerr << "Error: Missing required arguments.\n";
    std::cerr << "Required:";
    for (size_t i = 0; i < missing.size(); ++i) {
        if (i > 0) std::cerr << ",";
        std::cerr << " " << missing[i];
    }
    std::cerr << "\n\n";
    print_help();
    return false;
}

// Command definitions


CLICommand make_deconvolve_command() {
    CLICommand cmd;
    cmd.name = "deconvolve";
    cmd.description = "Separate damaged and undamaged DNA populations from mixed samples.";
    cmd.description_extra = {
        "",
        "Uses damage patterns (C→T at 5', G→A at 3') and fragment length to classify reads,",
        "then generates separate consensus sequences for each population.",
    };

    cmd.options = {
        {"--contigs", "FILE", "Input contigs FASTA file (bin or assembly)", "", true},
        {"--bam", "FILE", "BAM file (reads mapped to contigs)", "", true},
        {"--damage-positions", "N", "Positions from read ends for damage analysis", "15"},
        {"--em-iterations", "N", "Max EM iterations for adaptive prior (auto-converges)", "10"},
        {"--library-type", "TYPE", "Library prep type: ds (double-stranded) or ss (single-stranded)"},
        {"--min-ancient-depth", "N", "Min weighted depth for ancient consensus", "3.0"},
        {"--min-baseq", "N", "Minimum base quality", "20"},
        {"--min-length", "N", "Minimum contig length to process", "0"},
        {"--min-mapq", "N", "Minimum mapping quality", "30"},
        {"--min-modern-depth", "N", "Min weighted depth for modern consensus", "2.0"},
        {"--no-full-likelihood", "", "Use mismatch-only likelihood (faster, less accurate)"},
        {"--no-polish", "", "Skip damage correction on ancient consensus"},
        {"--no-length-model", "", "Disable read length in classification (default: on)"},
        {"--length-weight", "N", "Weight for length feature in likelihood", "0.5"},
        {"--mode-length-ancient", "N", "Mode read length for ancient DNA", "42.0"},
        {"--std-length-ancient", "N", "Std dev of ancient read length", "15.0"},
        {"--mode-length-modern", "N", "Mode read length for modern DNA", "103.0"},
        {"--std-length-modern", "N", "Std dev of modern read length", "30.0"},
        {"--use-identity-model", "", "Use alignment identity as classification feature"},
        {"--identity-weight", "N", "Weight for identity feature in likelihood", "1.0"},
        {"--mode-identity-ancient", "N", "Mode identity for ancient (%) ", "100.0"},
        {"--std-identity-ancient", "N", "Std dev of ancient identity", "2.0"},
        {"--mode-identity-modern", "N", "Mode identity for modern (%)", "99.0"},
        {"--std-identity-modern", "N", "Std dev of modern identity", "0.5"},
        {"--auto-estimate", "", "Auto-estimate length/identity params from read distributions"},
        {"--prior", "N", "Prior probability of ancient (0-1, default: 0.5 uninformative)", "0.5"},
        {"--output", "DIR", "Output directory", "current directory"},
        {"--threads", "N", "Number of threads", "1"},
        {"-v, --verbose", "", "Enable verbose output"},
        {"-h, --help", "", "Show this help message"},
    };

    cmd.outputs = {
        {"ancient_consensus.fa", "Ancient population consensus (damage-corrected by default)"},
        {"modern_consensus.fa", "Modern/contaminant population consensus"},
        {"deconvolution_stats.tsv", "Per-contig deconvolution statistics"},
        {"damage_profile.tsv", "Per-position damage rates"},
        {"damage_model.tsv", "Bayesian damage model parameters"},
    };

    cmd.note = "Ancient consensus is polished (damage-corrected) by default. Use --no-polish to disable.";

    cmd.examples = {
        "amber deconvolve --contigs bin_28.fa --bam reads.bam --library-type ds",
        "amber deconvolve --contigs bin_28.fa --bam reads.bam --output deconv/ --em-iterations 20",
    };

    return cmd;
}

CLICommand make_chimera_command() {
    CLICommand cmd;
    cmd.name = "chimera";
    cmd.description = "Detect chimeric contigs and misassemblies using multi-signal analysis.";
    cmd.description_extra = {
        "",
        "Combines coverage discontinuities, GC shifts, alignment quality, damage patterns,",
        "and fractal complexity to distinguish true chimeras from biological features.",
    };

    cmd.options = {
        {"--contigs", "FILE", "Input contigs FASTA file", "", true},
        {"--bam", "FILE", "BAM file (reads mapped to contigs)", "", true},
        {"--chimera-action", "X", "Handle chimeras: split (default), remove, or keep"},
        {"--damage-positions", "N", "Positions from read ends for damage analysis", "15"},
        {"--library-type", "TYPE", "Library prep type: ds (double-stranded) or ss (single-stranded)"},
        {"--min-confidence", "N", "Minimum confidence to report breakpoint", "0.5"},
        {"--min-length", "N", "Minimum contig length to analyze", "1000"},
        {"--no-fasta", "", "Skip writing cleaned FASTA output"},
        {"--output", "DIR", "Output directory", "current directory"},
        {"--threads", "N", "Number of threads", "1"},
        {"-v, --verbose", "", "Enable verbose output"},
        {"-h, --help", "", "Show this help message"},
    };

    cmd.outputs = {
        {"chimeras.tsv", "Detected breakpoints with evidence scores"},
        {"contig_flags.tsv", "Per-contig chimera classification"},
        {"chimera_stats.tsv", "Summary statistics"},
        {"cleaned.fa", "Contigs with chimeras handled per --chimera-action"},
    };

    cmd.examples = {
        "amber chimera --contigs bins/bin_0.fa --bam reads.bam --output results/",
        "amber chimera --contigs bins/bin_0.fa --bam reads.bam --chimera-action remove",
    };

    return cmd;
}

CLICommand make_embed_command() {
    CLICommand cmd;
    cmd.name = "embed";
    cmd.description = "Extract multi-scale sequence features from contigs for downstream analysis.";

    cmd.options = {
        {"--contigs", "FILE", "Input contigs FASTA file", "", true},
        {"--output", "DIR", "Output directory for feature files", "", true},
        {"--bam", "FILE", "BAM file for coverage features"},
        {"--damage", "FILE", "metaDMG/bdamage damage profile"},
        {"--damage-positions", "N", "Positions from read ends for damage features", "15"},
        {"--feature-set", "SET", "Feature set: all (default), optimal, minimal"},
        {"--library-type", "TYPE", "Library prep type: ds (default) or ss (single-stranded)"},
        {"--min-length", "N", "Minimum contig length", "1000"},
        {"--skip-coverage", "", "Skip coverage feature extraction"},
        {"--skip-damage", "", "Skip damage feature extraction"},
        {"--threads", "N", "Number of threads", "1"},
        {"-v, --verbose", "", "Enable verbose output"},
        {"-h, --help", "", "Show this help message"},
    };

    cmd.outputs = {
        {"compositional.tsv", "GC content and codon bias features"},
        {"cgr.tsv", "Chaos Game Representation features"},
        {"tnf.tsv", "Tetranucleotide frequencies"},
        {"complexity.tsv", "Fractal and complexity metrics"},
        {"coverage.tsv", "Coverage profile features", "(if BAM provided)"},
        {"damage.tsv", "Ancient DNA damage patterns", "(if profile provided)"},
    };

    cmd.examples = {
        "amber embed --contigs assembly.fa --output features/",
        "amber embed --contigs assembly.fa --bam reads.bam --output features/",
    };

    return cmd;
}

CLICommand make_bin_command() {
    CLICommand cmd;
    cmd.name = "bin";
    cmd.description = "Bin contigs into metagenome-assembled genomes (MAGs) using fractal-based";
    cmd.description_extra = {
        "probabilistic clustering optimized for ancient DNA.",
    };

    cmd.options = {
        {"--contigs", "FILE", "Input contigs FASTA file", "", true},
        {"--bam", "FILE", "BAM file (reads mapped to contigs)", "", true},
        {"--output", "DIR", "Output directory for bins", "", true},
        {"--min-length", "N", "Minimum contig length", "2000"},
        {"--threads", "N", "Number of threads", "1"},
        {"--seed", "N", "Random seed for reproducibility", "42"},
        {"--anvio", "", "Rename contigs for anvi'o (format: c_000000000001)"},
        {"--prefix", "PREFIX", "Contig name prefix for --anvio", "c"},
        {"--ancient-p-threshold", "P", "Min p_ancient for ANCIENT classification", "0.8"},
        {"--modern-p-threshold", "P", "Max p_ancient for MODERN classification", "0.5"},
        {"--ancient-amp-threshold", "A", "Min amplitude for ANCIENT classification", "0.01"},
        {"--temporal-minority-ratio", "R", "Max minority/majority for temporal_outliers", "0.1"},
        {"--temporal-damage-std", "S", "Max damage_std for temporal_outliers", "0.15"},
        {"--damage-positions", "N", "Positions from read ends for damage analysis", "15"},
        {"--library-type", "TYPE", "Library prep: ds (double-stranded) or ss (single-stranded)"},
        {"--export-smiley", "", "Export per-bin damage smiley plot data"},
        {"--cgr-z", "Z", "CGR outlier z-score threshold", "3.0"},
        {"--fractal-z", "Z", "Fractal outlier z-score threshold", "1.5"},
        {"--karlin-z", "Z", "Karlin signature z-score threshold", "2.0"},
        {"--max-removal", "F", "Max fraction of bin to remove in cleanup", "0.15"},
        {"-v, --verbose", "", "Enable verbose output"},
        {"-h, --help", "", "Show this help message"},
    };

    cmd.outputs = {
        {"bin_*.fa", "Bin FASTAs (renamed if --anvio)"},
        {"bin_stats.tsv", "Per-bin quality metrics and temporal analysis"},
        {"contig_assignments.tsv", "Per-contig bin assignments with all features"},
        {"damage_profile.tsv", "Per-position damage rates (C->T, G->A)"},
        {"damage_model.tsv", "Global Bayesian damage model parameters"},
        {"binner_trace.log", "Detailed trace log for debugging"},
        {"rename_report.tsv", "Original to anvi'o name mapping", "(with --anvio)"},
        {"smiley_*.tsv", "Per-bin damage smiley plot data", "(with --export-smiley)"},
    };

    cmd.note = "The --anvio flag renames contigs to anvi'o format (c_000000000001, c_000000000002, ...)\n"
               "  using underscore + 12 zero-padded digits. Original names are in rename_report.tsv.";

    cmd.examples = {
        "amber bin --contigs assembly.fa --bam reads.bam --output bins/ --threads 16",
        "amber bin --contigs assembly.fa --bam reads.bam --output bins/ --anvio",
    };

    return cmd;
}

}  // namespace amber

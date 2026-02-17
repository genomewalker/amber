// AMBER - cli_common.h
// Common CLI infrastructure for consistent command-line interface

#pragma once

#include <string>
#include <vector>
#include <functional>

namespace amber {

struct CLIOption {
    std::string name;           // e.g., "--contigs"
    std::string arg_name;       // e.g., "FILE", "N", "TYPE", "" for flags
    std::string description;
    std::string default_value;  // "" if required or no default
    bool required;

    CLIOption(const std::string& n, const std::string& arg, const std::string& desc,
              const std::string& def = "", bool req = false)
        : name(n), arg_name(arg), description(desc), default_value(def), required(req) {}
};

struct CLIOutput {
    std::string filename;
    std::string description;
    std::string condition;      // e.g., "(with --anvio)" or ""

    CLIOutput(const std::string& f, const std::string& d, const std::string& c = "")
        : filename(f), description(d), condition(c) {}
};

struct CLICommand {
    std::string name;
    std::string description;
    std::vector<std::string> description_extra;  // Additional description lines
    std::vector<CLIOption> options;
    std::vector<CLIOutput> outputs;
    std::string note;
    std::vector<std::string> examples;

    // Print formatted help message to stderr
    void print_help() const;

    // Check if help flag is present
    bool has_help_flag(int argc, char** argv) const;

    // Validate required arguments are present
    // Returns true if valid, false otherwise (prints error message)
    bool validate_required(int argc, char** argv) const;

    // Get option value (returns default if not found)
    std::string get_option(int argc, char** argv, const std::string& name,
                           const std::string& default_val = "") const;

    // Check if flag is present
    bool has_flag(int argc, char** argv, const std::string& flag) const;

    // Get list of missing required arguments
    std::vector<std::string> get_missing_required(int argc, char** argv) const;
};

// Command definitions
CLICommand make_polish_command();
CLICommand make_deconvolve_command();
CLICommand make_chimera_command();
CLICommand make_embed_command();
CLICommand make_bin_command();
CLICommand make_seeds_command();

}  // namespace amber

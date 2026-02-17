// AMBER - cmd_deconvolve.cpp
// CLI handler for the 'deconvolve' subcommand

#include "cli_common.h"

namespace amber {

extern int run_deconvolver(int argc, char** argv);

int cmd_deconvolve(int argc, char** argv) {
    CLICommand cmd = make_deconvolve_command();

    if (cmd.has_help_flag(argc, argv)) {
        cmd.print_help();
        return 0;
    }

    if (!cmd.validate_required(argc, argv)) {
        return 1;
    }

    return run_deconvolver(argc, argv);
}

}  // namespace amber

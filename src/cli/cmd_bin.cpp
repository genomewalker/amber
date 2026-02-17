// AMBER - cmd_bin.cpp
// CLI handler for the 'bin' subcommand

#include "cli_common.h"

namespace amber {

extern int run_binner(int argc, char** argv);

int cmd_bin(int argc, char** argv) {
    CLICommand cmd = make_bin_command();

    if (cmd.has_help_flag(argc, argv)) {
        cmd.print_help();
        return 0;
    }

    if (!cmd.validate_required(argc, argv)) {
        return 1;
    }

    return run_binner(argc, argv);
}

}  // namespace amber

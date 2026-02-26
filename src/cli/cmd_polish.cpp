// AMBER - cmd_polish.cpp
// CLI handler for the 'polish' subcommand

#include "cli_common.h"

namespace amber {

extern int run_polisher(int argc, char** argv);

int cmd_polish(int argc, char** argv) {
    CLICommand cmd = make_polish_command();

    if (cmd.has_help_flag(argc, argv)) {
        cmd.print_help();
        return 0;
    }

    if (!cmd.validate_required(argc, argv)) {
        return 1;
    }

    return run_polisher(argc, argv);
}

}  // namespace amber

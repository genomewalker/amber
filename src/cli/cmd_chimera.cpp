// AMBER - cmd_chimera.cpp
// CLI handler for the 'chimera' subcommand

#include "cli_common.h"

namespace amber {

extern int run_chimera_scanner(int argc, char** argv);

int cmd_chimera(int argc, char** argv) {
    CLICommand cmd = make_chimera_command();

    if (cmd.has_help_flag(argc, argv)) {
        cmd.print_help();
        return 0;
    }

    if (!cmd.validate_required(argc, argv)) {
        return 1;
    }

    return run_chimera_scanner(argc, argv);
}

}  // namespace amber

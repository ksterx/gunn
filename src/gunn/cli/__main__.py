"""Entry point for `python -m gunn` command."""

import sys


def main(args: list[str] | None = None) -> int:
    """Main entry point for the gunn CLI."""
    if args is None:
        args = sys.argv[1:]

    if not args or args[0] in ["-h", "--help"]:
        print_help()
        return 0

    command = args[0]

    if command == "version":
        print_version()
        return 0
    elif command == "replay":
        return run_replay(args[1:])
    else:
        print(f"Unknown command: {command}")
        print_help()
        return 1


def print_help() -> None:
    """Print CLI help message."""
    print(
        """gunn - Multi-agent simulation core

Usage:
    python -m gunn <command> [options]

Commands:
    version     Show version information
    replay      Replay event logs for debugging
    help        Show this help message

Options:
    -h, --help  Show help message

For more information, visit: https://github.com/your-org/gunn
"""
    )


def print_version() -> None:
    """Print version information."""
    try:
        from gunn import __version__

        print(f"gunn {__version__}")
    except ImportError:
        print("gunn (version unknown)")


def run_replay(args: list[str]) -> int:
    """Run the replay command."""
    from gunn.cli.replay import main as replay_main
    return replay_main(args)


if __name__ == "__main__":
    sys.exit(main())

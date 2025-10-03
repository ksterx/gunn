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
    elif command == "web":
        return run_web(args[1:])
    elif command == "config":
        return run_config(args[1:])
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
    web         Run web adapter server
    config      Configuration management
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
    import asyncio

    from gunn.cli.replay import run_replay_command

    return asyncio.run(run_replay_command(args))


def run_web(args: list[str]) -> int:
    """Run the web adapter command."""
    import sys

    from gunn.adapters.web.cli import cli

    # Set up sys.argv for click command routing
    original_argv = sys.argv
    try:
        sys.argv = ["gunn-web", *args]
        cli()
        return 0
    except SystemExit as e:
        return e.code or 0  # type: ignore
    except Exception as e:
        print(f"Web adapter error: {e}")
        return 1
    finally:
        sys.argv = original_argv


def run_config(args: list[str]) -> int:
    """Run the config command."""
    from gunn.cli.config import run_config_command

    return run_config_command(args)


if __name__ == "__main__":
    sys.exit(main())

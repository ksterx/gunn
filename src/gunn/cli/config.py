"""Configuration management CLI commands."""

import json
from pathlib import Path

import yaml

from gunn.config import (
    Config,
    ConfigError,
    load_config,
    load_config_from_env,
    validate_config,
)
from gunn.config.environment import get_config_file_path, get_environment
from gunn.utils.telemetry import get_logger

logger = get_logger(__name__)


def config_validate_command(args: list[str]) -> int:
    """Validate configuration file or environment variables.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    config_path = None
    if args and not args[0].startswith("-"):
        config_path = Path(args[0])

    try:
        if config_path:
            print(f"Validating configuration file: {config_path}")
            if not config_path.exists():
                print(f"Error: Configuration file not found: {config_path}")
                return 1
            config = load_config(config_path)
        else:
            print("Validating configuration from environment variables")
            config = load_config_from_env()

        validate_config(config)
        print("✓ Configuration is valid")
        return 0

    except ConfigError as e:
        print(f"✗ Configuration validation failed: {e}")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1


def config_show_command(args: list[str]) -> int:
    """Show current configuration.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    format_type = "yaml"
    config_path = None

    # Parse arguments
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ["--format", "-f"]:
            if i + 1 < len(args):
                format_type = args[i + 1]
                i += 2
            else:
                print("Error: --format requires a value")
                return 1
        elif arg.startswith("--format="):
            format_type = arg.split("=", 1)[1]
            i += 1
        elif not arg.startswith("-"):
            config_path = Path(arg)
            i += 1
        else:
            print(f"Unknown argument: {arg}")
            return 1

    if format_type not in ["yaml", "json"]:
        print(f"Error: Invalid format '{format_type}'. Use 'yaml' or 'json'")
        return 1

    try:
        config = load_config(config_path)
        config_dict = config.model_dump()

        # Convert OrchestratorConfig to dict for serialization
        if hasattr(config_dict["orchestrator"], "__dict__"):
            config_dict["orchestrator"] = config_dict["orchestrator"].__dict__

        # Convert FeatureFlags to dict
        if hasattr(config_dict["features"], "to_dict"):
            config_dict["features"] = config_dict["features"].to_dict()
        elif hasattr(config_dict["features"], "__dict__"):
            config_dict["features"] = config_dict["features"].__dict__

        if format_type == "json":
            print(json.dumps(config_dict, indent=2, default=str))
        else:
            print(yaml.dump(config_dict, default_flow_style=False, sort_keys=True))

        return 0

    except ConfigError as e:
        print(f"Error loading configuration: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


def config_init_command(args: list[str]) -> int:
    """Initialize configuration file with defaults.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    output_path = Path("gunn.yaml")
    force = False

    # Parse arguments
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ["--output", "-o"]:
            if i + 1 < len(args):
                output_path = Path(args[i + 1])
                i += 2
            else:
                print("Error: --output requires a value")
                return 1
        elif arg.startswith("--output="):
            output_path = Path(arg.split("=", 1)[1])
            i += 1
        elif arg in ["--force", "-f"]:
            force = True
            i += 1
        else:
            print(f"Unknown argument: {arg}")
            return 1

    if output_path.exists() and not force:
        print(f"Error: Configuration file already exists: {output_path}")
        print("Use --force to overwrite")
        return 1

    try:
        # Create default configuration
        config = Config()
        config_dict = config.model_dump()

        # Convert complex objects to dicts for YAML serialization
        if hasattr(config_dict["orchestrator"], "__dict__"):
            config_dict["orchestrator"] = config_dict["orchestrator"].__dict__

        if hasattr(config_dict["features"], "to_dict"):
            config_dict["features"] = config_dict["features"].to_dict()
        elif hasattr(config_dict["features"], "__dict__"):
            config_dict["features"] = config_dict["features"].__dict__

        # Write configuration file
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)

        print(f"✓ Configuration file created: {output_path}")
        return 0

    except Exception as e:
        print(f"Error creating configuration file: {e}")
        return 1


def config_env_command(args: list[str]) -> int:
    """Show environment variable configuration.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    show_all = "--all" in args or "-a" in args

    try:
        environment = get_environment()
        config_file = get_config_file_path()

        print(f"Environment: {environment.value}")
        print(f"Config file: {config_file or 'None found'}")
        print()

        # Show relevant environment variables
        import os

        env_vars = [
            "GUNN_ENVIRONMENT",
            "GUNN_DEBUG",
            "GUNN_FEATURES",
            "GUNN_LOG_LEVEL",
            "GUNN_LOG_FORMAT",
            "GUNN_LOG_FILE",
            "GUNN_METRICS_PORT",
            "GUNN_METRICS_PATH",
            "GUNN_DATABASE_URL",
            "GUNN_MAX_AGENTS",
            "GUNN_STALENESS_THRESHOLD",
            "GUNN_BACKPRESSURE_POLICY",
            "GUNN_DEBOUNCE_MS",
            "GUNN_DEADLINE_MS",
        ]

        print("Environment Variables:")
        for var in env_vars:
            value = os.getenv(var)
            if value or show_all:
                print(f"  {var}={value or '(not set)'}")

        if not show_all:
            print("\nUse --all to show all variables (including unset)")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def run_config_command(args: list[str]) -> int:
    """Run configuration management commands.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    if not args:
        print_config_help()
        return 0

    command = args[0]
    command_args = args[1:]

    if command == "validate":
        return config_validate_command(command_args)
    elif command == "show":
        return config_show_command(command_args)
    elif command == "init":
        return config_init_command(command_args)
    elif command == "env":
        return config_env_command(command_args)
    elif command in ["help", "-h", "--help"]:
        print_config_help()
        return 0
    else:
        print(f"Unknown config command: {command}")
        print_config_help()
        return 1


def print_config_help() -> None:
    """Print configuration command help."""
    print(
        """gunn config - Configuration management

Usage:
    gunn config <command> [options]

Commands:
    validate [file]     Validate configuration file or environment
    show [file]         Show current configuration
                        Options: --format=yaml|json
    init [options]      Create default configuration file
                        Options: --output=file, --force
    env                 Show environment variables
                        Options: --all
    help                Show this help message

Examples:
    gunn config validate
    gunn config validate config/production.yaml
    gunn config show --format=json
    gunn config init --output=config/staging.yaml
    gunn config env --all
"""
    )

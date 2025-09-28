"""Environment detection and configuration.

This module provides utilities for detecting the current environment
and loading environment-specific configurations.
"""

import os
from enum import Enum
from pathlib import Path


class Environment(Enum):
    """Supported deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


def get_environment() -> Environment:
    """Detect current environment from various sources.

    Environment is detected in the following order:
    1. GUNN_ENVIRONMENT environment variable
    2. NODE_ENV environment variable (for compatibility)
    3. Presence of specific files (.env.production, etc.)
    4. Default to development

    Returns:
        Detected environment
    """
    # Check GUNN_ENVIRONMENT first
    env_str = os.getenv("GUNN_ENVIRONMENT", "").lower()
    if env_str:
        try:
            return Environment(env_str)
        except ValueError:
            pass

    # Check NODE_ENV for compatibility
    node_env = os.getenv("NODE_ENV", "").lower()
    if node_env in {"production", "prod"}:
        return Environment.PRODUCTION
    elif node_env in {"staging", "stage"}:
        return Environment.STAGING
    elif node_env in {"test", "testing"}:
        return Environment.TESTING
    elif node_env in {"development", "dev"}:
        return Environment.DEVELOPMENT

    # Check for environment-specific files
    cwd = Path.cwd()
    if (cwd / ".env.production").exists():
        return Environment.PRODUCTION
    elif (cwd / ".env.staging").exists():
        return Environment.STAGING
    elif (cwd / ".env.testing").exists():
        return Environment.TESTING

    # Default to development
    return Environment.DEVELOPMENT


def get_config_file_path(environment: Environment | None = None) -> Path | None:
    """Get the configuration file path for the given environment.

    Args:
        environment: Environment to get config for (defaults to current)

    Returns:
        Path to configuration file, or None if not found
    """
    if environment is None:
        environment = get_environment()

    # Check for environment-specific config files
    config_paths = [
        Path(f"config/{environment.value}.yaml"),
        Path(f"config/{environment.value}.yml"),
        Path(f"gunn.{environment.value}.yaml"),
        Path(f"gunn.{environment.value}.yml"),
        Path("config/gunn.yaml"),
        Path("config/gunn.yml"),
        Path("gunn.yaml"),
        Path("gunn.yml"),
    ]

    for path in config_paths:
        if path.exists():
            return path

    return None


def is_production() -> bool:
    """Check if running in production environment."""
    return get_environment() == Environment.PRODUCTION


def is_development() -> bool:
    """Check if running in development environment."""
    return get_environment() == Environment.DEVELOPMENT


def is_testing() -> bool:
    """Check if running in testing environment."""
    return get_environment() == Environment.TESTING

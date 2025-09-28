"""Core configuration management for gunn.

This module provides the main configuration classes and loading functionality
with environment variable support and feature flags.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gunn.core.orchestrator import OrchestratorConfig


class ConfigError(Exception):
    """Configuration-related errors."""

    pass


@dataclass
class FeatureFlags:
    """Feature flags for enabling/disabling functionality.

    These can be controlled via the GUNN_FEATURES environment variable
    as a comma-separated list (e.g., "latency,backpressure,telemetry").
    """

    # Core features
    latency_simulation: bool = True
    backpressure_management: bool = True
    staleness_detection: bool = True
    cancellation_tokens: bool = True

    # Observability features
    telemetry: bool = True
    metrics_export: bool = True
    structured_logging: bool = True
    pii_redaction: bool = True

    # Performance features
    memory_management: bool = True
    log_compaction: bool = True
    view_caching: bool = True

    # Security features
    authentication: bool = False
    authorization: bool = False
    rate_limiting: bool = True

    # Experimental features
    distributed_mode: bool = False
    gpu_acceleration: bool = False

    @classmethod
    def from_env(cls, env_var: str = "GUNN_FEATURES") -> "FeatureFlags":
        """Load feature flags from environment variable.

        Args:
            env_var: Environment variable name (default: GUNN_FEATURES)

        Returns:
            FeatureFlags instance with features enabled based on env var
        """
        features_str = os.getenv(env_var, "")
        if not features_str:
            return cls()

        enabled_features = {f.strip().lower() for f in features_str.split(",")}

        # Map feature names to attributes
        feature_mapping = {
            "latency": "latency_simulation",
            "backpressure": "backpressure_management",
            "staleness": "staleness_detection",
            "cancellation": "cancellation_tokens",
            "telemetry": "telemetry",
            "metrics": "metrics_export",
            "logging": "structured_logging",
            "pii": "pii_redaction",
            "memory": "memory_management",
            "compaction": "log_compaction",
            "caching": "view_caching",
            "auth": "authentication",
            "authz": "authorization",
            "ratelimit": "rate_limiting",
            "distributed": "distributed_mode",
            "gpu": "gpu_acceleration",
        }

        kwargs = {}
        for feature_name, attr_name in feature_mapping.items():
            kwargs[attr_name] = feature_name in enabled_features

        return cls(**kwargs)

    def to_dict(self) -> dict[str, bool]:
        """Convert to dictionary for metrics export."""
        return {
            "latency_simulation": self.latency_simulation,
            "backpressure_management": self.backpressure_management,
            "staleness_detection": self.staleness_detection,
            "cancellation_tokens": self.cancellation_tokens,
            "telemetry": self.telemetry,
            "metrics_export": self.metrics_export,
            "structured_logging": self.structured_logging,
            "pii_redaction": self.pii_redaction,
            "memory_management": self.memory_management,
            "log_compaction": self.log_compaction,
            "view_caching": self.view_caching,
            "authentication": self.authentication,
            "authorization": self.authorization,
            "rate_limiting": self.rate_limiting,
            "distributed_mode": self.distributed_mode,
            "gpu_acceleration": self.gpu_acceleration,
        }


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "text"] = "json"
    enable_pii_redaction: bool = True
    log_file: str | None = None
    max_file_size_mb: int = 100
    backup_count: int = 5


class MetricsConfig(BaseModel):
    """Metrics configuration."""

    enabled: bool = True
    port: int = 8000
    path: str = "/metrics"
    export_interval_seconds: float = 15.0
    include_feature_flags: bool = True


class SecurityConfig(BaseModel):
    """Security configuration."""

    enable_auth: bool = False
    auth_token_header: str = "Authorization"
    rate_limit_per_minute: int = 1000
    max_request_size_mb: int = 10
    allowed_origins: list[str] = Field(default_factory=lambda: ["*"])


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = "sqlite:///gunn.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    echo: bool = False


class Config(BaseModel):
    """Main configuration class for gunn.

    This class combines all configuration sections and provides
    validation and environment variable loading.
    """

    # Core orchestrator configuration
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)

    # Feature flags
    features: FeatureFlags = Field(default_factory=FeatureFlags)

    # Subsystem configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    # Environment and deployment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("orchestrator", mode="before")
    @classmethod
    def validate_orchestrator_config(cls, v):
        """Validate orchestrator configuration."""
        if isinstance(v, dict):
            return OrchestratorConfig(**v)
        return v

    @field_validator("features", mode="before")
    @classmethod
    def validate_features(cls, v):
        """Validate feature flags."""
        if isinstance(v, dict):
            return FeatureFlags(**v)
        return v


def load_config_from_file(config_path: Path) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded configuration

    Raises:
        ConfigError: If configuration is invalid or file cannot be read
    """
    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            config_data = {}

        return Config(**config_data)

    except FileNotFoundError as e:
        raise ConfigError(f"Configuration file not found: {config_path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in configuration file: {e}") from e
    except ValidationError as e:
        raise ConfigError(f"Configuration validation failed: {e}") from e


def load_config_from_env() -> Config:
    """Load configuration from environment variables.

    Environment variables are mapped as follows:
    - GUNN_ENVIRONMENT: Environment name (development/staging/production)
    - GUNN_DEBUG: Enable debug mode (true/false)
    - GUNN_FEATURES: Comma-separated list of enabled features
    - GUNN_LOG_LEVEL: Logging level
    - GUNN_LOG_FORMAT: Logging format (json/text)
    - GUNN_METRICS_PORT: Metrics server port
    - GUNN_DATABASE_URL: Database connection URL
    - GUNN_MAX_AGENTS: Maximum number of agents
    - GUNN_STALENESS_THRESHOLD: Staleness detection threshold
    - GUNN_BACKPRESSURE_POLICY: Backpressure policy (defer/shed/drop)

    Returns:
        Configuration loaded from environment variables
    """
    config_data = {}

    # Environment and debug
    if env_val := os.getenv("GUNN_ENVIRONMENT"):
        config_data["environment"] = env_val
    if env_val := os.getenv("GUNN_DEBUG"):
        config_data["debug"] = env_val.lower() in ("true", "1", "yes", "on")

    # Feature flags
    config_data["features"] = FeatureFlags.from_env()

    # Logging configuration
    logging_config = {}
    if env_val := os.getenv("GUNN_LOG_LEVEL"):
        logging_config["level"] = env_val.upper()
    if env_val := os.getenv("GUNN_LOG_FORMAT"):
        logging_config["format"] = env_val.lower()
    if env_val := os.getenv("GUNN_LOG_FILE"):
        logging_config["log_file"] = env_val
    if logging_config:
        config_data["logging"] = logging_config

    # Metrics configuration
    metrics_config = {}
    if env_val := os.getenv("GUNN_METRICS_PORT"):
        try:
            metrics_config["port"] = int(env_val)
        except ValueError as e:
            raise ConfigError(f"Invalid GUNN_METRICS_PORT: {env_val}") from e
    if env_val := os.getenv("GUNN_METRICS_PATH"):
        metrics_config["path"] = env_val
    if metrics_config:
        config_data["metrics"] = metrics_config

    # Database configuration
    database_config = {}
    if env_val := os.getenv("GUNN_DATABASE_URL"):
        database_config["url"] = env_val
    if database_config:
        config_data["database"] = database_config

    # Orchestrator configuration
    orchestrator_config = {}
    if env_val := os.getenv("GUNN_MAX_AGENTS"):
        try:
            orchestrator_config["max_agents"] = int(env_val)
        except ValueError as e:
            raise ConfigError(f"Invalid GUNN_MAX_AGENTS: {env_val}") from e
    if env_val := os.getenv("GUNN_STALENESS_THRESHOLD"):
        try:
            orchestrator_config["staleness_threshold"] = int(env_val)
        except ValueError as e:
            raise ConfigError(f"Invalid GUNN_STALENESS_THRESHOLD: {env_val}") from e
    if env_val := os.getenv("GUNN_BACKPRESSURE_POLICY"):
        orchestrator_config["backpressure_policy"] = env_val
    if env_val := os.getenv("GUNN_DEBOUNCE_MS"):
        try:
            orchestrator_config["debounce_ms"] = float(env_val)
        except ValueError as e:
            raise ConfigError(f"Invalid GUNN_DEBOUNCE_MS: {env_val}") from e
    if env_val := os.getenv("GUNN_DEADLINE_MS"):
        try:
            orchestrator_config["deadline_ms"] = float(env_val)
        except ValueError as e:
            raise ConfigError(f"Invalid GUNN_DEADLINE_MS: {env_val}") from e
    if orchestrator_config:
        config_data["orchestrator"] = orchestrator_config

    try:
        return Config(**config_data)
    except ValidationError as e:
        raise ConfigError(f"Environment configuration validation failed: {e}") from e


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from file and environment variables.

    Configuration is loaded in the following order (later sources override earlier):
    1. Default values
    2. Configuration file (if provided)
    3. Environment variables

    Args:
        config_path: Optional path to configuration file

    Returns:
        Merged configuration
    """
    # Start with defaults
    config = Config()

    # Load from file if provided
    if config_path and config_path.exists():
        file_config = load_config_from_file(config_path)
        # Merge file config with defaults
        config = Config(
            **{**config.model_dump(), **file_config.model_dump(exclude_unset=True)}
        )

    # Load from environment (highest priority)
    env_config = load_config_from_env()
    config = Config(
        **{**config.model_dump(), **env_config.model_dump(exclude_unset=True)}
    )

    return config


def validate_config(config: Config) -> None:
    """Validate configuration for consistency and correctness.

    Args:
        config: Configuration to validate

    Raises:
        ConfigError: If configuration is invalid
    """
    # Validate orchestrator limits
    if config.orchestrator.max_agents <= 0:
        raise ConfigError("max_agents must be positive")

    if config.orchestrator.staleness_threshold < 0:
        raise ConfigError("staleness_threshold must be non-negative")

    if config.orchestrator.debounce_ms < 0:
        raise ConfigError("debounce_ms must be non-negative")

    if config.orchestrator.deadline_ms <= 0:
        raise ConfigError("deadline_ms must be positive")

    # Validate backpressure policy
    valid_policies = {"defer", "shed", "drop"}
    if config.orchestrator.backpressure_policy not in valid_policies:
        raise ConfigError(
            f"Invalid backpressure_policy: {config.orchestrator.backpressure_policy}. "
            f"Must be one of: {valid_policies}"
        )

    # Validate metrics configuration
    if config.metrics.port <= 0 or config.metrics.port > 65535:
        raise ConfigError("metrics.port must be between 1 and 65535")

    # Validate logging configuration
    if config.logging.max_file_size_mb <= 0:
        raise ConfigError("logging.max_file_size_mb must be positive")

    if config.logging.backup_count < 0:
        raise ConfigError("logging.backup_count must be non-negative")

    # Environment-specific validations
    if config.environment == "production":
        if config.debug:
            raise ConfigError("Debug mode should not be enabled in production")

        if config.logging.level == "DEBUG":
            raise ConfigError("DEBUG logging should not be used in production")

        if not config.features.pii_redaction:
            raise ConfigError("PII redaction should be enabled in production")

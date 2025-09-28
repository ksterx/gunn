"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from gunn.config import (
    Config,
    ConfigError,
    FeatureFlags,
    load_config,
    load_config_from_env,
    load_config_from_file,
    validate_config,
)
from gunn.config.environment import Environment, get_environment


class TestFeatureFlags:
    """Test feature flags functionality."""

    def test_default_features(self):
        """Test default feature flag values."""
        features = FeatureFlags()

        assert features.latency_simulation is True
        assert features.backpressure_management is True
        assert features.telemetry is True
        assert features.authentication is False
        assert features.distributed_mode is False

    def test_from_env_empty(self):
        """Test loading feature flags from empty environment."""
        features = FeatureFlags.from_env("NONEXISTENT_VAR")

        # Should use defaults
        assert features.latency_simulation is True
        assert features.telemetry is True

    def test_from_env_with_features(self, monkeypatch):
        """Test loading feature flags from environment variable."""
        monkeypatch.setenv("TEST_FEATURES", "latency,auth,distributed")

        features = FeatureFlags.from_env("TEST_FEATURES")

        assert features.latency_simulation is True
        assert features.authentication is True
        assert features.distributed_mode is True
        assert features.backpressure_management is False  # Not in list

    def test_to_dict(self):
        """Test converting feature flags to dictionary."""
        features = FeatureFlags(
            latency_simulation=True, authentication=False, distributed_mode=True
        )

        feature_dict = features.to_dict()

        assert feature_dict["latency_simulation"] is True
        assert feature_dict["authentication"] is False
        assert feature_dict["distributed_mode"] is True


class TestConfig:
    """Test configuration loading and validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        assert config.environment == "development"
        assert config.debug is False
        assert config.features.telemetry is True
        assert config.logging.level == "INFO"
        assert config.metrics.enabled is True

    def test_config_validation_success(self):
        """Test successful configuration validation."""
        config = Config()

        # Should not raise any exception
        validate_config(config)

    def test_config_validation_invalid_max_agents(self):
        """Test configuration validation with invalid max_agents."""
        config = Config()
        config.orchestrator.max_agents = -1

        with pytest.raises(ConfigError, match="max_agents must be positive"):
            validate_config(config)

    def test_config_validation_invalid_backpressure_policy(self):
        """Test configuration validation with invalid backpressure policy."""
        config = Config()
        config.orchestrator.backpressure_policy = "invalid"

        with pytest.raises(ConfigError, match="Invalid backpressure_policy"):
            validate_config(config)

    def test_config_validation_production_debug(self):
        """Test configuration validation for production with debug enabled."""
        config = Config()
        config.environment = "production"
        config.debug = True

        with pytest.raises(
            ConfigError, match="Debug mode should not be enabled in production"
        ):
            validate_config(config)


class TestConfigLoading:
    """Test configuration loading from various sources."""

    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "environment": "staging",
            "debug": True,
            "logging": {"level": "DEBUG", "format": "text"},
            "orchestrator": {"max_agents": 50, "staleness_threshold": 5},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = load_config_from_file(config_path)

            assert config.environment == "staging"
            assert config.debug is True
            assert config.logging.level == "DEBUG"
            assert config.logging.format == "text"
            assert config.orchestrator.max_agents == 50
            assert config.orchestrator.staleness_threshold == 5

        finally:
            config_path.unlink()

    def test_load_config_from_file_not_found(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(ConfigError, match="Configuration file not found"):
            load_config_from_file(Path("nonexistent.yaml"))

    def test_load_config_from_env(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("GUNN_ENVIRONMENT", "production")
        monkeypatch.setenv("GUNN_DEBUG", "false")
        monkeypatch.setenv("GUNN_LOG_LEVEL", "WARNING")
        monkeypatch.setenv("GUNN_MAX_AGENTS", "200")
        monkeypatch.setenv("GUNN_FEATURES", "telemetry,metrics,logging")

        config = load_config_from_env()

        assert config.environment == "production"
        assert config.debug is False
        assert config.logging.level == "WARNING"
        assert config.orchestrator.max_agents == 200
        assert config.features.telemetry is True
        assert config.features.metrics_export is True
        assert config.features.authentication is False  # Not in features list

    def test_load_config_from_env_invalid_values(self, monkeypatch):
        """Test loading configuration with invalid environment values."""
        monkeypatch.setenv("GUNN_MAX_AGENTS", "invalid")

        with pytest.raises(ConfigError, match="Invalid GUNN_MAX_AGENTS"):
            load_config_from_env()

    def test_load_config_priority(self, monkeypatch):
        """Test configuration loading priority (env > file > defaults)."""
        # Create config file
        config_data = {
            "environment": "staging",
            "debug": True,
            "orchestrator": {"max_agents": 50},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        # Set environment variables (should override file)
        monkeypatch.setenv("GUNN_ENVIRONMENT", "production")
        monkeypatch.setenv("GUNN_MAX_AGENTS", "100")

        try:
            config = load_config(config_path)

            # Environment should override file
            assert config.environment == "production"
            assert config.orchestrator.max_agents == 100

            # File should override defaults
            assert config.debug is True  # From file, not default False

        finally:
            config_path.unlink()


class TestEnvironmentDetection:
    """Test environment detection functionality."""

    def test_get_environment_from_gunn_env(self, monkeypatch):
        """Test environment detection from GUNN_ENVIRONMENT."""
        monkeypatch.setenv("GUNN_ENVIRONMENT", "production")

        env = get_environment()
        assert env == Environment.PRODUCTION

    def test_get_environment_from_node_env(self, monkeypatch):
        """Test environment detection from NODE_ENV."""
        monkeypatch.delenv("GUNN_ENVIRONMENT", raising=False)
        monkeypatch.setenv("NODE_ENV", "staging")

        env = get_environment()
        assert env == Environment.STAGING

    def test_get_environment_default(self, monkeypatch):
        """Test default environment detection."""
        monkeypatch.delenv("GUNN_ENVIRONMENT", raising=False)
        monkeypatch.delenv("NODE_ENV", raising=False)

        env = get_environment()
        assert env == Environment.DEVELOPMENT

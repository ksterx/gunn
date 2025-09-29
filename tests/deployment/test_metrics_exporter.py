"""Tests for metrics exporter functionality."""

import time

import pytest

from gunn.config import Config, FeatureFlags
from gunn.utils.metrics_exporter import (
    MetricsExporter,
    create_metrics_exporter,
    get_metrics_text,
)


class TestMetricsExporter:
    """Test metrics exporter functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            environment="development",
            debug=True,
            features=FeatureFlags(
                latency_simulation=True,
                authentication=False,
                telemetry=True,
            ),
        )

    @pytest.fixture
    def metrics_exporter(self, config):
        """Create metrics exporter instance."""
        return MetricsExporter(config)

    def test_export_feature_flags(self, metrics_exporter, config):
        """Test exporting feature flags as metrics."""
        metrics_exporter.export_feature_flags(config.features)

        # Check that metrics were set
        metrics_text = get_metrics_text()

        assert (
            'gunn_feature_flag_enabled{feature_name="latency_simulation"} 1.0'
            in metrics_text
        )
        assert (
            'gunn_feature_flag_enabled{feature_name="authentication"} 0.0'
            in metrics_text
        )
        assert 'gunn_feature_flag_enabled{feature_name="telemetry"} 1.0' in metrics_text

    def test_export_config_info(self, metrics_exporter, config):
        """Test exporting configuration information."""
        metrics_exporter.export_config_info(config)

        metrics_text = get_metrics_text()

        assert "gunn_config_info" in metrics_text
        assert 'environment="development"' in metrics_text
        assert 'debug="True"' in metrics_text
        assert 'log_level="INFO"' in metrics_text

    def test_export_system_info(self, metrics_exporter):
        """Test exporting system information."""
        metrics_exporter.export_system_info()

        metrics_text = get_metrics_text()

        assert "gunn_system_info" in metrics_text
        assert "python_version=" in metrics_text
        assert "platform=" in metrics_text
        assert 'gunn_version="0.1.0"' in metrics_text

    def test_record_startup_time(self, metrics_exporter):
        """Test recording startup time."""
        startup_duration = 2.5

        metrics_exporter.record_startup_time(startup_duration)

        metrics_text = get_metrics_text()
        assert "gunn_startup_time_seconds 2.5" in metrics_text

    def test_record_config_reload(self, metrics_exporter):
        """Test recording configuration reload events."""
        _ = get_metrics_text()

        metrics_exporter.record_config_reload()
        metrics_exporter.record_config_reload()

        final_metrics = get_metrics_text()

        # Should have incremented the counter
        assert "gunn_config_reloads_total" in final_metrics

    def test_record_config_validation_time(self, metrics_exporter):
        """Test recording configuration validation time."""
        validation_duration = 0.123

        metrics_exporter.record_config_validation_time(validation_duration)

        metrics_text = get_metrics_text()
        assert "gunn_config_validation_duration_seconds" in metrics_text

    def test_should_export_interval(self, config):
        """Test export interval checking."""
        # Set very short interval for testing
        config.metrics.export_interval_seconds = 0.1
        exporter = MetricsExporter(config)

        # First call should return True
        assert exporter.should_export() is True

        # Immediate second call should return False
        assert exporter.should_export() is False

        # After interval, should return True again
        time.sleep(0.11)
        assert exporter.should_export() is True

    def test_export_all(self, metrics_exporter, config):
        """Test exporting all metrics."""
        # Set interval to 0 to force export
        metrics_exporter._export_interval = 0

        metrics_exporter.export_all()

        metrics_text = get_metrics_text()

        # Should contain all metric types
        assert "gunn_feature_flag_enabled" in metrics_text
        assert "gunn_config_info" in metrics_text
        assert "gunn_system_info" in metrics_text


class TestMetricsExporterIntegration:
    """Test metrics exporter integration functionality."""

    def test_create_metrics_exporter(self):
        """Test creating and initializing metrics exporter."""
        config = Config(
            features=FeatureFlags(
                telemetry=True,
                authentication=True,
            )
        )

        exporter = create_metrics_exporter(config)

        assert isinstance(exporter, MetricsExporter)
        assert exporter.config == config

        # Should have exported initial metrics
        metrics_text = get_metrics_text()
        assert 'gunn_feature_flag_enabled{feature_name="telemetry"} 1.0' in metrics_text
        assert (
            'gunn_feature_flag_enabled{feature_name="authentication"} 1.0'
            in metrics_text
        )

    def test_get_metrics_text(self):
        """Test getting metrics in Prometheus text format."""
        config = Config()
        _ = create_metrics_exporter(config)

        metrics_text = get_metrics_text()

        assert isinstance(metrics_text, str)
        assert len(metrics_text) > 0
        assert "# HELP" in metrics_text  # Prometheus format
        assert "# TYPE" in metrics_text  # Prometheus format

    def test_feature_flag_changes(self):
        """Test updating feature flag metrics when flags change."""
        config = Config()
        exporter = MetricsExporter(config)

        # Export initial flags
        initial_features = FeatureFlags(telemetry=True, authentication=False)
        exporter.export_feature_flags(initial_features)

        initial_metrics = get_metrics_text()
        assert (
            'gunn_feature_flag_enabled{feature_name="telemetry"} 1.0' in initial_metrics
        )
        assert (
            'gunn_feature_flag_enabled{feature_name="authentication"} 0.0'
            in initial_metrics
        )

        # Update flags
        updated_features = FeatureFlags(telemetry=False, authentication=True)
        exporter.export_feature_flags(updated_features)

        updated_metrics = get_metrics_text()
        assert (
            'gunn_feature_flag_enabled{feature_name="telemetry"} 0.0' in updated_metrics
        )
        assert (
            'gunn_feature_flag_enabled{feature_name="authentication"} 1.0'
            in updated_metrics
        )

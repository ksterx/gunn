"""Metrics exporter with feature flag status.

This module provides functionality to export feature flag status
and other configuration metrics for operational visibility.
"""

import time

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from prometheus_client.core import REGISTRY

from gunn.config import Config, FeatureFlags
from gunn.utils.telemetry import get_logger

logger = get_logger(__name__)

# Feature flag metrics
feature_flag_gauge = Gauge(
    "gunn_feature_flag_enabled",
    "Feature flag status (1=enabled, 0=disabled)",
    ["feature_name"],
    registry=REGISTRY,
)

# Configuration metrics
config_info = Info("gunn_config", "Configuration information", registry=REGISTRY)

# System metrics
system_info = Info("gunn_system", "System information", registry=REGISTRY)

# Performance metrics
startup_time_gauge = Gauge(
    "gunn_startup_time_seconds", "Time taken to start the service", registry=REGISTRY
)

config_reload_counter = Counter(
    "gunn_config_reloads_total", "Number of configuration reloads", registry=REGISTRY
)

config_validation_histogram = Histogram(
    "gunn_config_validation_duration_seconds",
    "Time spent validating configuration",
    registry=REGISTRY,
)


class MetricsExporter:
    """Exports configuration and feature flag metrics."""

    def __init__(self, config: Config):
        self.config = config
        self._last_export_time = 0.0
        self._export_interval = config.metrics.export_interval_seconds

    def export_feature_flags(self, features: FeatureFlags) -> None:
        """Export feature flag status as metrics.

        Args:
            features: Feature flags to export
        """
        try:
            feature_dict = features.to_dict()

            for feature_name, enabled in feature_dict.items():
                feature_flag_gauge.labels(feature_name=feature_name).set(
                    1 if enabled else 0
                )

            logger.debug(f"Exported {len(feature_dict)} feature flags to metrics")

        except Exception as e:
            logger.error(f"Failed to export feature flags: {e}")

    def export_config_info(self, config: Config) -> None:
        """Export configuration information as metrics.

        Args:
            config: Configuration to export
        """
        try:
            config_data = {
                "environment": config.environment,
                "debug": str(config.debug),
                "log_level": config.logging.level,
                "log_format": config.logging.format,
                "metrics_enabled": str(config.metrics.enabled),
                "backpressure_policy": config.orchestrator.backpressure_policy,
                "max_agents": str(config.orchestrator.max_agents),
                "staleness_threshold": str(config.orchestrator.staleness_threshold),
            }

            config_info.info(config_data)
            logger.debug("Exported configuration info to metrics")

        except Exception as e:
            logger.error(f"Failed to export config info: {e}")

    def export_system_info(self) -> None:
        """Export system information as metrics."""
        try:
            import platform
            import sys

            system_data = {
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor() or "unknown",
                "gunn_version": "0.1.0",  # TODO: Get from package metadata
            }

            system_info.info(system_data)
            logger.debug("Exported system info to metrics")

        except Exception as e:
            logger.error(f"Failed to export system info: {e}")

    def record_startup_time(self, startup_duration: float) -> None:
        """Record service startup time.

        Args:
            startup_duration: Time taken to start the service in seconds
        """
        startup_time_gauge.set(startup_duration)
        logger.info(f"Recorded startup time: {startup_duration:.3f}s")

    def record_config_reload(self) -> None:
        """Record a configuration reload event."""
        config_reload_counter.inc()
        logger.debug("Recorded config reload event")

    def record_config_validation_time(self, validation_duration: float) -> None:
        """Record configuration validation time.

        Args:
            validation_duration: Time taken to validate config in seconds
        """
        config_validation_histogram.observe(validation_duration)
        logger.debug(f"Recorded config validation time: {validation_duration:.3f}s")

    def should_export(self) -> bool:
        """Check if metrics should be exported based on interval.

        Returns:
            True if enough time has passed since last export
        """
        current_time = time.time()
        if current_time - self._last_export_time >= self._export_interval:
            self._last_export_time = current_time
            return True
        return False

    def export_all(self) -> None:
        """Export all metrics if interval has passed."""
        if not self.should_export():
            return

        try:
            self.export_feature_flags(self.config.features)
            self.export_config_info(self.config)
            self.export_system_info()

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")


def create_metrics_exporter(config: Config) -> MetricsExporter:
    """Create and initialize metrics exporter.

    Args:
        config: Configuration to use for metrics export

    Returns:
        Configured metrics exporter
    """
    exporter = MetricsExporter(config)

    # Export initial metrics
    exporter.export_feature_flags(config.features)
    exporter.export_config_info(config)
    exporter.export_system_info()

    return exporter


def get_metrics_text() -> str:
    """Get metrics in Prometheus text format.

    Returns:
        Metrics as text string
    """
    return generate_latest(REGISTRY).decode("utf-8")

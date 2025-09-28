"""Tests for health check functionality."""

import asyncio
import time

import pytest

from gunn.config.deployment import (
    GracefulShutdownHandler,
    HealthCheckConfig,
    HealthChecker,
    HealthStatus,
)


class TestHealthChecker:
    """Test health check functionality."""

    @pytest.fixture
    def health_config(self):
        """Create test health check configuration."""
        return HealthCheckConfig(
            enabled=True,
            timeout_seconds=1.0,
            check_interval_seconds=5.0,
            max_memory_usage_percent=80.0,
            max_response_time_ms=500.0,
        )

    @pytest.fixture
    def health_checker(self, health_config):
        """Create health checker instance."""
        return HealthChecker(health_config)

    @pytest.mark.asyncio
    async def test_health_check_success(self, health_checker):
        """Test successful health check."""
        status = await health_checker.check_health()

        assert isinstance(status, HealthStatus)
        assert status.healthy is True
        assert status.ready is True
        assert status.timestamp > 0
        assert status.response_time_ms >= 0
        assert "database" in status.checks
        assert "event_log" in status.checks
        assert "orchestrator" in status.checks
        assert "memory" in status.checks
        assert "response_time" in status.checks

    @pytest.mark.asyncio
    async def test_health_check_response_time_threshold(self):
        """Test health check with response time threshold exceeded."""
        config = HealthCheckConfig(
            max_response_time_ms=0.1  # Very low threshold
        )
        health_checker = HealthChecker(config)

        status = await health_checker.check_health()

        # Should be unhealthy due to response time
        assert status.healthy is False
        assert status.checks["response_time"]["healthy"] is False
        assert status.response_time_ms > config.max_response_time_ms

    @pytest.mark.asyncio
    async def test_health_check_caching(self, health_checker):
        """Test health check result caching."""
        # First check
        status1 = await health_checker.check_health()
        cached_status = health_checker.get_last_check()

        assert cached_status is not None
        assert cached_status.timestamp == status1.timestamp

        # Second check should update cache
        await asyncio.sleep(0.01)  # Small delay
        status2 = await health_checker.check_health()
        cached_status2 = health_checker.get_last_check()

        assert cached_status2.timestamp == status2.timestamp
        assert cached_status2.timestamp > status1.timestamp

    def test_health_check_config_validation(self):
        """Test health check configuration validation."""
        config = HealthCheckConfig(
            timeout_seconds=5.0,
            max_memory_usage_percent=90.0,
            max_response_time_ms=1000.0,
        )

        assert config.timeout_seconds == 5.0
        assert config.max_memory_usage_percent == 90.0
        assert config.max_response_time_ms == 1000.0
        assert config.enabled is True


class TestGracefulShutdownHandler:
    """Test graceful shutdown functionality."""

    @pytest.fixture
    def shutdown_handler(self):
        """Create shutdown handler instance."""
        return GracefulShutdownHandler(timeout_seconds=1.0)

    @pytest.mark.asyncio
    async def test_graceful_shutdown_no_tasks(self, shutdown_handler):
        """Test graceful shutdown with no registered tasks."""
        start_time = time.time()

        await shutdown_handler.shutdown()

        end_time = time.time()
        duration = end_time - start_time

        # Should complete quickly with no tasks
        assert duration < 0.1
        assert shutdown_handler.is_shutting_down() is True

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_tasks(self, shutdown_handler):
        """Test graceful shutdown with registered tasks."""
        task_completed = False

        async def test_task():
            nonlocal task_completed
            await asyncio.sleep(0.1)
            task_completed = True

        # Register shutdown task
        shutdown_handler.register_shutdown_task(test_task())

        # Perform shutdown
        await shutdown_handler.shutdown()

        assert task_completed is True
        assert shutdown_handler.is_shutting_down() is True

    @pytest.mark.asyncio
    async def test_graceful_shutdown_timeout(self, shutdown_handler):
        """Test graceful shutdown with timeout."""

        async def slow_task():
            await asyncio.sleep(2.0)  # Longer than timeout

        # Register slow task
        shutdown_handler.register_shutdown_task(slow_task())

        start_time = time.time()
        await shutdown_handler.shutdown()
        end_time = time.time()

        duration = end_time - start_time

        # Should timeout after 1 second
        assert 0.9 < duration < 1.5
        assert shutdown_handler.is_shutting_down() is True

    @pytest.mark.asyncio
    async def test_shutdown_signal_waiting(self, shutdown_handler):
        """Test waiting for shutdown signal."""
        # Start waiting for shutdown in background
        wait_task = asyncio.create_task(shutdown_handler.wait_for_shutdown())

        # Give it a moment to start waiting
        await asyncio.sleep(0.01)
        assert not wait_task.done()

        # Trigger shutdown
        await shutdown_handler.shutdown()

        # Wait task should complete
        await wait_task
        assert wait_task.done()
        assert shutdown_handler.is_shutting_down() is True

"""Deployment configuration and health checks.

This module provides deployment-specific configuration and health check
functionality for production deployments.
"""

import asyncio
import time
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any

import psutil
from pydantic import BaseModel, Field

from gunn.utils.telemetry import get_logger

logger = get_logger(__name__)


class HealthCheckConfig(BaseModel):
    """Configuration for health checks."""

    enabled: bool = True
    endpoint: str = "/health"
    readiness_endpoint: str = "/ready"
    liveness_endpoint: str = "/live"
    timeout_seconds: float = 5.0
    check_interval_seconds: float = 30.0

    # Component-specific checks
    check_database: bool = True
    check_event_log: bool = True
    check_orchestrator: bool = True
    check_memory_usage: bool = True

    # Thresholds
    max_memory_usage_percent: float = 90.0
    max_response_time_ms: float = 1000.0
    max_queue_depth: int = 1000


@dataclass
class HealthStatus:
    """Health check status."""

    healthy: bool
    ready: bool
    checks: dict[str, Any]
    timestamp: float
    response_time_ms: float


class DeploymentConfig(BaseModel):
    """Deployment-specific configuration."""

    # Service configuration
    service_name: str = "gunn"
    service_version: str = "0.1.0"
    instance_id: str | None = None

    # Network configuration
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1

    # Health checks
    health: HealthCheckConfig = Field(default_factory=HealthCheckConfig)

    # Graceful shutdown
    shutdown_timeout_seconds: float = 30.0

    # Resource limits
    max_memory_mb: int | None = None
    max_cpu_percent: float | None = None

    # Monitoring
    enable_profiling: bool = False
    profiling_interval_seconds: float = 60.0


class HealthChecker:
    """Health check implementation."""

    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self._last_check: HealthStatus | None = None
        self._check_lock = asyncio.Lock()

    async def check_health(self) -> HealthStatus:
        """Perform comprehensive health check.

        Returns:
            Health status with detailed check results
        """
        start_time = time.time()

        async with self._check_lock:
            checks = {}
            healthy = True
            ready = True

            # Database connectivity check
            if self.config.check_database:
                try:
                    db_healthy = await self._check_database()
                    checks["database"] = {"healthy": db_healthy, "error": None}
                    if not db_healthy:
                        healthy = False
                        ready = False
                except Exception as e:
                    checks["database"] = {"healthy": False, "error": str(e)}
                    healthy = False
                    ready = False

            # Event log check
            if self.config.check_event_log:
                try:
                    log_healthy = await self._check_event_log()
                    checks["event_log"] = {"healthy": log_healthy, "error": None}
                    if not log_healthy:
                        healthy = False
                except Exception as e:
                    checks["event_log"] = {"healthy": False, "error": str(e)}
                    healthy = False

            # Orchestrator check
            if self.config.check_orchestrator:
                try:
                    orch_healthy, orch_ready = await self._check_orchestrator()
                    checks["orchestrator"] = {
                        "healthy": orch_healthy,
                        "ready": orch_ready,
                        "error": None,
                    }
                    if not orch_healthy:
                        healthy = False
                    if not orch_ready:
                        ready = False
                except Exception as e:
                    checks["orchestrator"] = {
                        "healthy": False,
                        "ready": False,
                        "error": str(e),
                    }
                    healthy = False
                    ready = False

            # Memory usage check
            if self.config.check_memory_usage:
                try:
                    memory_ok, memory_info = await self._check_memory()
                    checks["memory"] = {
                        "healthy": memory_ok,
                        "usage_percent": memory_info.get("usage_percent", 0),
                        "error": None,
                    }
                    if not memory_ok:
                        healthy = False
                except Exception as e:
                    checks["memory"] = {"healthy": False, "error": str(e)}
                    healthy = False

            response_time_ms = (time.time() - start_time) * 1000

            # Check response time threshold
            if response_time_ms > self.config.max_response_time_ms:
                healthy = False
                checks["response_time"] = {
                    "healthy": False,
                    "response_time_ms": response_time_ms,
                    "threshold_ms": self.config.max_response_time_ms,
                }
            else:
                checks["response_time"] = {
                    "healthy": True,
                    "response_time_ms": response_time_ms,
                }

            status = HealthStatus(
                healthy=healthy,
                ready=ready,
                checks=checks,
                timestamp=time.time(),
                response_time_ms=response_time_ms,
            )

            self._last_check = status
            return status

    async def _check_database(self) -> bool:
        """Check database connectivity."""
        # TODO: Implement actual database health check
        # For now, assume healthy
        await asyncio.sleep(0.001)  # Simulate check
        return True

    async def _check_event_log(self) -> bool:
        """Check event log health."""
        # TODO: Implement actual event log health check
        # Check log integrity, write permissions, etc.
        await asyncio.sleep(0.001)  # Simulate check
        return True

    async def _check_orchestrator(self) -> tuple[bool, bool]:
        """Check orchestrator health and readiness.

        Returns:
            Tuple of (healthy, ready)
        """
        # TODO: Implement actual orchestrator health check
        # Check if orchestrator is running, queue depths, etc.
        await asyncio.sleep(0.001)  # Simulate check
        return True, True

    async def _check_memory(self) -> tuple[bool, dict[str, Any]]:
        """Check memory usage.

        Returns:
            Tuple of (within_limits, memory_info)
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            within_limits = memory_percent < self.config.max_memory_usage_percent

            info = {
                "usage_percent": memory_percent,
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "threshold_percent": self.config.max_memory_usage_percent,
            }

            return within_limits, info

        except ImportError:
            # psutil not available, skip memory check
            return True, {"usage_percent": 0, "error": "psutil not available"}
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")
            return False, {"error": str(e)}

    def get_last_check(self) -> HealthStatus | None:
        """Get the last health check result."""
        return self._last_check


class GracefulShutdownHandler:
    """Handles graceful shutdown of the service."""

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds
        self._shutdown_event = asyncio.Event()
        self._shutdown_tasks: list[asyncio.Task] = []

    def register_shutdown_task(self, coro: Coroutine) -> None:
        """Register a coroutine to run during shutdown."""
        task = asyncio.create_task(coro)
        self._shutdown_tasks.append(task)

    async def shutdown(self) -> None:
        """Perform graceful shutdown."""
        logger.info("Starting graceful shutdown...")

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for shutdown tasks with timeout
        if self._shutdown_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._shutdown_tasks, return_exceptions=True),
                    timeout=self.timeout_seconds,
                )
                logger.info("All shutdown tasks completed")
            except TimeoutError:
                logger.warning(f"Shutdown timeout after {self.timeout_seconds}s")
                # Cancel remaining tasks
                for task in self._shutdown_tasks:
                    if not task.done():
                        task.cancel()

        logger.info("Graceful shutdown completed")

    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._shutdown_event.is_set()

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

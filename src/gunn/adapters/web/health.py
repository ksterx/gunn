"""Health check endpoints for web adapter."""

import time

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from gunn.config.deployment import HealthCheckConfig, HealthChecker
from gunn.utils.telemetry import get_logger

logger = get_logger(__name__)

# Global health checker instance
_health_checker: HealthChecker | None = None


def setup_health_checker(config: HealthCheckConfig) -> HealthChecker:
    """Setup global health checker instance."""
    global _health_checker
    _health_checker = HealthChecker(config)
    return _health_checker


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    if _health_checker is None:
        # Create default health checker if none exists
        return HealthChecker(HealthCheckConfig())
    return _health_checker


def create_health_router() -> APIRouter:
    """Create health check router."""
    router = APIRouter(tags=["health"])

    @router.get("/health")
    async def health_check() -> JSONResponse:
        """Comprehensive health check endpoint.

        Returns:
            JSON response with health status and detailed checks
        """
        try:
            health_checker = get_health_checker()
            status_result = await health_checker.check_health()

            response_data = {
                "status": "healthy" if status_result.healthy else "unhealthy",
                "timestamp": status_result.timestamp,
                "response_time_ms": status_result.response_time_ms,
                "checks": status_result.checks,
                "version": "0.1.0",  # TODO: Get from package metadata
                "environment": "production",  # TODO: Get from config
            }

            status_code = (
                status.HTTP_200_OK
                if status_result.healthy
                else status.HTTP_503_SERVICE_UNAVAILABLE
            )

            return JSONResponse(content=response_data, status_code=status_code)

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                content={
                    "status": "unhealthy",
                    "timestamp": time.time(),
                    "error": str(e),
                    "checks": {},
                },
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

    @router.get("/ready")
    async def readiness_check() -> JSONResponse:
        """Readiness check endpoint.

        Returns:
            JSON response indicating if service is ready to accept traffic
        """
        try:
            health_checker = get_health_checker()
            status_result = await health_checker.check_health()

            response_data = {
                "status": "ready" if status_result.ready else "not_ready",
                "timestamp": status_result.timestamp,
                "response_time_ms": status_result.response_time_ms,
            }

            status_code = (
                status.HTTP_200_OK
                if status_result.ready
                else status.HTTP_503_SERVICE_UNAVAILABLE
            )

            return JSONResponse(content=response_data, status_code=status_code)

        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return JSONResponse(
                content={
                    "status": "not_ready",
                    "timestamp": time.time(),
                    "error": str(e),
                },
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

    @router.get("/live")
    async def liveness_check() -> JSONResponse:
        """Liveness check endpoint.

        Returns:
            JSON response indicating if service is alive
        """
        try:
            # Simple liveness check - just verify the service is responding
            return JSONResponse(
                content={
                    "status": "alive",
                    "timestamp": time.time(),
                },
                status_code=status.HTTP_200_OK,
            )

        except Exception as e:
            logger.error(f"Liveness check failed: {e}")
            return JSONResponse(
                content={"status": "dead", "timestamp": time.time(), "error": str(e)},
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

    return router

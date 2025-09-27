"""Telemetry utilities for logging, metrics, and tracing.

This module provides centralized observability infrastructure including:
- Structured logging with PII redaction
- Prometheus metrics collection
- OpenTelemetry tracing setup
- Performance measurement utilities
"""

import asyncio
import re
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from prometheus_client import Counter, Histogram, start_http_server
from structlog.processors import JSONRenderer

# Prometheus metrics
OPERATION_COUNTER = Counter(
    "gunn_operations_total",
    "Total number of operations",
    ["operation", "status", "agent_id"],
)

OPERATION_LATENCY = Histogram(
    "gunn_operation_duration_seconds",
    "Operation latency in seconds",
    ["operation", "agent_id"],
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

QUEUE_DEPTH = Histogram(
    "gunn_queue_depth",
    "Queue depth by agent",
    ["agent_id"],
    buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

CANCEL_RATE = Counter(
    "gunn_cancellations_total", "Total number of cancellations", ["agent_id", "reason"]
)


# PII patterns for redaction
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b|\b[0-9]{3}-[0-9]{4}\b"
    ),
    "token": re.compile(r"\b[A-Za-z0-9]{20,}\b"),  # Generic token pattern
    "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
}


def redact_pii(text: str) -> str:
    """Redact personally identifiable information from text.

    Args:
        text: Input text that may contain PII

    Returns:
        Text with PII patterns replaced with [REDACTED_<type>]

    Example:
        >>> redact_pii("Contact john@example.com or call 555-123-4567")
        'Contact [REDACTED_EMAIL] or call [REDACTED_PHONE]'
    """
    result = text
    for pii_type, pattern in PII_PATTERNS.items():
        result = pattern.sub(f"[REDACTED_{pii_type.upper()}]", result)
    return result


def pii_redaction_processor(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor to redact PII from log events.

    Args:
        logger: Logger instance (unused)
        method_name: Log method name (unused)
        event_dict: Event dictionary to process

    Returns:
        Event dictionary with PII redacted from string values
    """

    def redact_value(value: Any) -> Any:
        if isinstance(value, str):
            return redact_pii(value)
        elif isinstance(value, dict):
            return {k: redact_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [redact_value(item) for item in value]
        return value

    return {key: redact_value(value) for key, value in event_dict.items()}


def setup_logging(log_level: str = "INFO", enable_pii_redaction: bool = True) -> None:
    """Initialize structured logging with PII redaction.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_pii_redaction: Whether to enable PII redaction processor
    """
    processors: list[Any] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if enable_pii_redaction:
        processors.append(pii_redaction_processor)

    processors.append(JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **context: Any) -> Any:
    """Get a structured logger with optional context.

    Args:
        name: Logger name
        **context: Additional context to bind to logger

    Returns:
        Bound logger with context
    """
    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger


class PerformanceTimer:
    """Context manager for measuring operation performance.

    Automatically records metrics and logs timing information.
    """

    def __init__(
        self,
        operation: str,
        agent_id: str | None = None,
        logger: structlog.BoundLogger | None = None,
        record_metrics: bool = True,
    ):
        self.operation = operation
        self.agent_id = agent_id or "unknown"
        self.logger = logger or get_logger("gunn.performance")
        self.record_metrics = record_metrics
        self.start_time: float | None = None
        self.end_time: float | None = None

    def __enter__(self) -> "PerformanceTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.perf_counter()
        duration = self.end_time - (self.start_time or 0)

        status = "error" if exc_type else "success"

        # Record metrics
        if self.record_metrics:
            OPERATION_COUNTER.labels(
                operation=self.operation, status=status, agent_id=self.agent_id
            ).inc()

            OPERATION_LATENCY.labels(
                operation=self.operation, agent_id=self.agent_id
            ).observe(duration)

        # Log timing
        self.logger.info(
            "Operation completed",
            operation=self.operation,
            agent_id=self.agent_id,
            duration_ms=duration * 1000,
            status=status,
        )

    @property
    def duration(self) -> float | None:
        """Get operation duration in seconds."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@asynccontextmanager
async def async_performance_timer(
    operation: str,
    agent_id: str | None = None,
    logger: structlog.BoundLogger | None = None,
    record_metrics: bool = True,
) -> AsyncGenerator[PerformanceTimer, None]:
    """Async context manager for measuring operation performance.

    Args:
        operation: Operation name for metrics/logging
        agent_id: Agent identifier (optional)
        logger: Logger instance (optional)
        record_metrics: Whether to record Prometheus metrics

    Yields:
        PerformanceTimer instance
    """
    timer = PerformanceTimer(operation, agent_id, logger, record_metrics)
    timer.start_time = time.perf_counter()

    try:
        yield timer
    except Exception as e:
        timer.end_time = time.perf_counter()
        duration = timer.end_time - timer.start_time

        # Record error metrics
        if record_metrics:
            OPERATION_COUNTER.labels(
                operation=operation, status="error", agent_id=agent_id or "unknown"
            ).inc()

            OPERATION_LATENCY.labels(
                operation=operation, agent_id=agent_id or "unknown"
            ).observe(duration)

        # Log error
        if logger:
            logger.error(
                "Operation failed",
                operation=operation,
                agent_id=agent_id,
                duration_ms=duration * 1000,
                error=str(e),
            )

        raise
    else:
        timer.end_time = time.perf_counter()
        duration = timer.end_time - timer.start_time

        # Record success metrics
        if record_metrics:
            OPERATION_COUNTER.labels(
                operation=operation, status="success", agent_id=agent_id or "unknown"
            ).inc()

            OPERATION_LATENCY.labels(
                operation=operation, agent_id=agent_id or "unknown"
            ).observe(duration)

        # Log success
        if logger:
            logger.info(
                "Operation completed",
                operation=operation,
                agent_id=agent_id,
                duration_ms=duration * 1000,
                status="success",
            )


def record_queue_depth(agent_id: str, depth: int) -> None:
    """Record queue depth metric for an agent.

    Args:
        agent_id: Agent identifier
        depth: Current queue depth
    """
    QUEUE_DEPTH.labels(agent_id=agent_id).observe(depth)


def record_cancellation(agent_id: str, reason: str) -> None:
    """Record a cancellation event.

    Args:
        agent_id: Agent identifier
        reason: Cancellation reason
    """
    CANCEL_RATE.labels(agent_id=agent_id, reason=reason).inc()


def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus metrics HTTP server.

    Args:
        port: Port to serve metrics on
    """
    start_http_server(port)


class MonotonicClock:
    """Monotonic clock for internal timing measurements.

    Uses asyncio event loop's monotonic time for consistent timing
    that's not affected by system clock adjustments.
    """

    @staticmethod
    def now() -> float:
        """Get current monotonic time in seconds.

        Returns:
            Current time from asyncio event loop's monotonic clock
        """
        try:
            loop = asyncio.get_running_loop()
            return loop.time()
        except RuntimeError:
            # No event loop running, fall back to time.monotonic()
            return time.monotonic()

    @staticmethod
    def wall_time() -> float:
        """Get current wall clock time for display purposes.

        Returns:
            Current wall clock time in seconds since epoch
        """
        return time.time()


def get_timing_context() -> dict[str, float]:
    """Get current timing context for logging.

    Returns:
        Dictionary with monotonic_time and wall_time
    """
    return {
        "monotonic_time": MonotonicClock.now(),
        "wall_time": MonotonicClock.wall_time(),
    }

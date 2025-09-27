"""Telemetry utilities for logging, metrics, and tracing.

This module provides centralized observability infrastructure including:
- Structured logging with PII redaction
- Prometheus metrics collection
- OpenTelemetry tracing setup
<<<<<<< HEAD
- Performance measurement utilities
=======
- Monotonic timing utilities
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7
"""

import asyncio
import re
import time
<<<<<<< HEAD
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

=======
from collections.abc import MutableMapping
from typing import Any

import structlog
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CollectorRegistry, Counter, Histogram
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7

# PII patterns for redaction
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(
<<<<<<< HEAD
        r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b|\b[0-9]{3}-[0-9]{4}\b"
    ),
    "token": re.compile(r"\b[A-Za-z0-9]{20,}\b"),  # Generic token pattern
    "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
}


def redact_pii(text: str) -> str:
=======
        r"\b\d{3}[-.]?\d{3,4}[-.]?\d{4}\b|\b\d{3}[-.]?\d{4}\b"
    ),  # Support both 10 and 7 digit formats
    "token": re.compile(r"\b[A-Za-z0-9]{20,}\b"),  # Generic token pattern
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
}

# Prometheus metrics - use separate registry for tests

# Create a separate registry for this module to avoid conflicts
_METRICS_REGISTRY = CollectorRegistry()

OPERATION_COUNTER = Counter(
    "gunn_operations_total",
    "Total number of operations",
    ["operation", "status", "agent_id"],
    registry=_METRICS_REGISTRY,
)

OPERATION_DURATION = Histogram(
    "gunn_operation_duration_seconds",
    "Duration of operations in seconds",
    ["operation", "agent_id"],
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    registry=_METRICS_REGISTRY,
)

QUEUE_DEPTH_GAUGE: dict[str, Any] = {}  # Will be populated with agent-specific gauges


def redact_pii(text: Any) -> Any:
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7
    """Redact personally identifiable information from text.

    Args:
        text: Input text that may contain PII

    Returns:
        Text with PII patterns replaced with [REDACTED_<type>]
<<<<<<< HEAD

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
=======
    """
    if not isinstance(text, str):
        return text

    result = text
    for pii_type, pattern in PII_PATTERNS.items():
        result = pattern.sub(f"[REDACTED_{pii_type.upper()}]", result)

    return result


def redact_dict(data: Any) -> Any:
    """Recursively redact PII from dictionary values.

    Args:
        data: Dictionary that may contain PII in values

    Returns:
        Dictionary with PII redacted from string values
    """
    if not isinstance(data, dict):
        return data

    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = redact_pii(value)
        elif isinstance(value, dict):
            result[key] = redact_dict(value)
        elif isinstance(value, list):
            result[key] = [
                redact_dict(item)
                if isinstance(item, dict)
                else redact_pii(item)
                if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value

    return result


class PIIRedactionProcessor:
    """Structlog processor for PII redaction."""

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: MutableMapping[str, Any],
    ) -> Any:
        """Process log event to redact PII.

        Args:
            logger: Logger instance
            method_name: Log method name
            event_dict: Event dictionary to process

        Returns:
            Event dictionary with PII redacted
        """
        # Redact the main event message
        if "event" in event_dict:
            event_dict["event"] = redact_pii(str(event_dict["event"]))

        # Redact other string fields
        for key, value in event_dict.items():
            if isinstance(value, str):
                event_dict[key] = redact_pii(value)
            elif isinstance(value, dict):
                event_dict[key] = redact_dict(value)

        return event_dict


def setup_logging(service_name: str = "gunn", log_level: str = "INFO") -> None:
    """Initialize structured logging with PII redaction.

    Args:
        service_name: Name of the service for log context
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    import logging

    # Map string levels to logging constants
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        PIIRedactionProcessor(),
    ]
    if log_level == "DEBUG":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            level_map.get(log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Add service context
    structlog.contextvars.bind_contextvars(service=service_name)


def setup_tracing(service_name: str = "gunn", otlp_endpoint: str | None = None) -> None:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service for tracing
        otlp_endpoint: OTLP endpoint URL (optional)
    """
    from opentelemetry.sdk.trace.export import SpanExporter

    span_exporter: SpanExporter
    if otlp_endpoint:
        span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    else:
        # Use console exporter for development
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        span_exporter = ConsoleSpanExporter()

    span_processor = BatchSpanProcessor(span_exporter)

    provider = TracerProvider()
    provider.add_span_processor(span_processor)
    trace.set_tracer_provider(provider)


def setup_metrics(service_name: str = "gunn", otlp_endpoint: str | None = None) -> None:
    """Initialize OpenTelemetry metrics.

    Args:
        service_name: Name of the service for metrics
        otlp_endpoint: OTLP endpoint URL (optional)
    """
    from opentelemetry.sdk.metrics.export import MetricExporter

    metric_exporter: MetricExporter
    if otlp_endpoint:
        metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
    else:
        # Use console exporter for development
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

        metric_exporter = ConsoleMetricExporter()

    metric_reader = PeriodicExportingMetricReader(
        exporter=metric_exporter,
        export_interval_millis=10000,  # Export every 10 seconds
    )

    provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(provider)


def start_prometheus_server(port: int = 8000) -> None:
    """Start Prometheus metrics HTTP server.

    Args:
        port: Port to serve metrics on
    """
    from prometheus_client import start_http_server

    start_http_server(port, registry=_METRICS_REGISTRY)


class OperationTimer:
    """Context manager for timing operations with automatic metrics collection."""

    def __init__(self, operation: str, agent_id: str = "system"):
        """Initialize operation timer.

        Args:
            operation: Name of the operation being timed
            agent_id: Agent ID for metrics labeling
        """
        self.operation = operation
        self.agent_id = agent_id
        self.start_time: float | None = None
        self.logger = structlog.get_logger()

    def __enter__(self) -> "OperationTimer":
        """Start timing the operation."""
        self.start_time = asyncio.get_running_loop().time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timing and record metrics."""
        if self.start_time is None:
            return

        duration = asyncio.get_running_loop().time() - self.start_time
        status = "error" if exc_type else "success"

        # Record Prometheus metrics
        OPERATION_COUNTER.labels(
            operation=self.operation, status=status, agent_id=self.agent_id
        ).inc()

        OPERATION_DURATION.labels(
            operation=self.operation, agent_id=self.agent_id
        ).observe(duration)

        # Log the operation
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7
        self.logger.info(
            "Operation completed",
            operation=self.operation,
            agent_id=self.agent_id,
            duration_ms=duration * 1000,
            status=status,
        )

<<<<<<< HEAD
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
=======

def get_monotonic_time() -> float:
    """Get monotonic time from the event loop.

    Returns:
        Monotonic time in seconds

    Raises:
        RuntimeError: If called outside an async context
    """
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        # Fallback to time.monotonic() if not in async context
        return time.monotonic()


def get_wall_time() -> float:
    """Get wall clock time.

    Returns:
        Wall clock time in seconds since epoch
    """
    return time.time()


def create_logger(name: str, **context: Any) -> Any:
    """Create a logger with bound context.

    Args:
        name: Logger name
        **context: Additional context to bind

    Returns:
        Structured logger with bound context
    """
    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger


# Global logger instance
logger = structlog.get_logger(__name__)
>>>>>>> 7b3ac25cdcb4c4caf4a86b7ef7f0b0ca54a26ef7

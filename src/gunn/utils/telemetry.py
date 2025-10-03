"""Telemetry utilities for logging, metrics, and tracing.

This module provides centralized observability infrastructure including:
- Structured logging with PII redaction
- Prometheus metrics collection
- OpenTelemetry tracing setup
- Performance measurement utilities
- Memory usage tracking and reporting
- Bandwidth/CPU measurement for patch operations
"""

import asyncio
import re
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import psutil
import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from prometheus_client import Counter, Gauge, Histogram, start_http_server
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

QUEUE_DEPTH_HIGH_WATERMARK = Histogram(
    "gunn_queue_depth_high_watermark",
    "High watermark for queue depths triggering backpressure",
    ["agent_id", "queue_type"],
    buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

BACKPRESSURE_EVENTS = Counter(
    "gunn_backpressure_events_total",
    "Total number of backpressure events",
    ["agent_id", "queue_type", "policy"],
)

CIRCUIT_BREAKER_STATE = Counter(
    "gunn_circuit_breaker_state_changes_total",
    "Circuit breaker state changes",
    ["component", "from_state", "to_state"],
)

ERROR_RECOVERY_ACTIONS = Counter(
    "gunn_error_recovery_actions_total",
    "Error recovery actions taken",
    ["error_type", "recovery_action", "agent_id"],
)

# Enhanced metrics for Task 15
INTENT_THROUGHPUT = Counter(
    "gunn_intents_processed_total",
    "Total number of intents processed",
    ["agent_id", "intent_kind", "status"],
)

CONFLICT_RATE = Counter(
    "gunn_conflicts_total",
    "Total number of intent conflicts",
    ["agent_id", "conflict_type"],
)

OBSERVATION_DELIVERY_LATENCY = Histogram(
    "gunn_observation_delivery_duration_seconds",
    "Time from effect creation to observation delivery",
    ["agent_id"],
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
)

PATCH_OPERATIONS_COUNT = Histogram(
    "gunn_patch_operations_count",
    "Number of JSON patch operations in observation deltas",
    ["agent_id"],
    buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

PATCH_FALLBACK_EVENTS = Counter(
    "gunn_patch_fallback_total",
    "Number of times patch operations exceeded max_patch_ops and fell back to full snapshot",
    ["agent_id"],
)

MEMORY_USAGE_BYTES = Gauge(
    "gunn_memory_usage_bytes",
    "Memory usage in bytes",
    ["component"],
)

CPU_USAGE_PERCENT = Gauge(
    "gunn_cpu_usage_percent",
    "CPU usage percentage",
    ["component"],
)

BANDWIDTH_BYTES = Counter(
    "gunn_bandwidth_bytes_total",
    "Total bytes transferred",
    ["direction", "component"],
)

GLOBAL_SEQ_GAUGE = Gauge(
    "gunn_global_seq_current",
    "Current global sequence number",
)

VIEW_SEQ_GAUGE = Gauge(
    "gunn_view_seq_current",
    "Current view sequence number by agent",
    ["agent_id"],
)

ACTIVE_AGENTS_GAUGE = Gauge(
    "gunn_active_agents_count",
    "Number of currently active agents",
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


def redact_pii(text: Any) -> Any:
    """Redact personally identifiable information from text.

    Args:
        text: Input text that may contain PII

    Returns:
        Text with PII patterns replaced with [REDACTED_<type>], or original input if not a string

    Example:
        >>> redact_pii("Contact john@example.com or call 555-123-4567")
        'Contact [REDACTED_EMAIL] or call [REDACTED_PHONE]'
    """
    if not isinstance(text, str):
        return text

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


def setup_tracing(
    service_name: str = "gunn",
    otlp_endpoint: str | None = None,
    enable_fastapi_instrumentation: bool = True,
) -> None:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service for tracing
        otlp_endpoint: OTLP endpoint URL (if None, uses console exporter)
        enable_fastapi_instrumentation: Whether to enable FastAPI auto-instrumentation
    """
    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.1.0",  # TODO: Get from package metadata
        }
    )

    # Set up tracer provider
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    exporter: OTLPSpanExporter | ConsoleSpanExporter
    # Configure exporter
    if otlp_endpoint:
        # Use OTLP exporter for production
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    else:
        # Use console exporter for development
        exporter = ConsoleSpanExporter()

    # Add span processor
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)

    # Enable FastAPI instrumentation if requested
    if enable_fastapi_instrumentation:
        try:
            FastAPIInstrumentor.instrument()
        except Exception as e:
            get_logger("gunn.telemetry").warning(
                "Failed to instrument FastAPI", error=str(e)
            )


def get_tracer(name: str) -> trace.Tracer:
    """Get OpenTelemetry tracer for a component.

    Args:
        name: Tracer name (typically module name)

    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


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


def log_operation(
    logger: Any,
    operation: str,
    status: str = "success",
    global_seq: int | None = None,
    view_seq: int | None = None,
    agent_id: str | None = None,
    req_id: str | None = None,
    latency_ms: float | None = None,
    **extra_context: Any,
) -> None:
    """Log an operation with standardized fields for observability.

    Args:
        logger: Structured logger instance
        operation: Operation name
        status: Operation status (success, error, warning)
        global_seq: Global sequence number
        view_seq: View sequence number
        agent_id: Agent identifier
        req_id: Request identifier
        latency_ms: Operation latency in milliseconds
        **extra_context: Additional context fields
    """
    log_data = {
        "operation": operation,
        "status": status,
        **extra_context,
    }

    # Add optional fields if provided
    if global_seq is not None:
        log_data["global_seq"] = global_seq
    if view_seq is not None:
        log_data["view_seq"] = view_seq
    if agent_id is not None:
        log_data["agent_id"] = agent_id
    if req_id is not None:
        log_data["req_id"] = req_id
    if latency_ms is not None:
        log_data["latency_ms"] = latency_ms

    # Add timing context
    log_data.update(get_timing_context())

    # Log at appropriate level based on status
    if status == "error":
        logger.error("Operation completed", **log_data)
    elif status == "warning":
        logger.warning("Operation completed", **log_data)
    else:
        logger.info("Operation completed", **log_data)


class PerformanceTimer:
    """Context manager for measuring operation performance.

    Automatically records metrics, logs timing information, and creates tracing spans.
    """

    def __init__(
        self,
        operation: str,
        agent_id: str | None = None,
        req_id: str | None = None,
        global_seq: int | None = None,
        view_seq: int | None = None,
        logger: structlog.BoundLogger | None = None,
        record_metrics: bool = True,
        create_span: bool = True,
        tracer_name: str = "gunn.performance",
    ):
        self.operation = operation
        self.agent_id = agent_id or "unknown"
        self.req_id = req_id
        self.global_seq = global_seq
        self.view_seq = view_seq
        self.logger = logger or get_logger("gunn.performance")
        self.record_metrics = record_metrics
        self.create_span = create_span
        self.tracer = get_tracer(tracer_name) if create_span else None
        self.span: trace.Span | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None

    def __enter__(self) -> "PerformanceTimer":
        self.start_time = time.perf_counter()

        # Create tracing span
        if self.create_span and self.tracer:
            self.span = self.tracer.start_span(self.operation)

            # Add attributes to span
            if self.agent_id:
                self.span.set_attribute("agent_id", self.agent_id)
            if self.req_id:
                self.span.set_attribute("req_id", self.req_id)
            if self.global_seq is not None:
                self.span.set_attribute("global_seq", self.global_seq)
            if self.view_seq is not None:
                self.span.set_attribute("view_seq", self.view_seq)

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

        # Update span
        if self.span:
            self.span.set_attribute("duration_seconds", duration)
            self.span.set_attribute("status", status)

            if exc_type:
                self.span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(trace.Status(trace.StatusCode.OK))

            self.span.end()

        # Enhanced logging with all context
        log_operation(
            self.logger,
            self.operation,
            status=status,
            global_seq=self.global_seq,
            view_seq=self.view_seq,
            agent_id=self.agent_id,
            req_id=self.req_id,
            latency_ms=duration * 1000,
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
    req_id: str | None = None,
    global_seq: int | None = None,
    view_seq: int | None = None,
    logger: structlog.BoundLogger | None = None,
    record_metrics: bool = True,
    create_span: bool = True,
    tracer_name: str = "gunn.performance",
) -> AsyncGenerator[PerformanceTimer, None]:
    """Async context manager for measuring operation performance.

    Args:
        operation: Operation name for metrics/logging
        agent_id: Agent identifier (optional)
        req_id: Request identifier (optional)
        global_seq: Global sequence number (optional)
        view_seq: View sequence number (optional)
        logger: Logger instance (optional)
        record_metrics: Whether to record Prometheus metrics
        create_span: Whether to create tracing span
        tracer_name: Tracer name for spans

    Yields:
        PerformanceTimer instance
    """
    timer = PerformanceTimer(
        operation=operation,
        agent_id=agent_id,
        req_id=req_id,
        global_seq=global_seq,
        view_seq=view_seq,
        logger=logger,
        record_metrics=record_metrics,
        create_span=create_span,
        tracer_name=tracer_name,
    )

    timer.start_time = time.perf_counter()

    # Create tracing span
    if create_span and timer.tracer:
        timer.span = timer.tracer.start_span(operation)

        # Add attributes to span
        if agent_id:
            timer.span.set_attribute("agent_id", agent_id)
        if req_id:
            timer.span.set_attribute("req_id", req_id)
        if global_seq is not None:
            timer.span.set_attribute("global_seq", global_seq)
        if view_seq is not None:
            timer.span.set_attribute("view_seq", view_seq)

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

        # Update span
        if timer.span:
            timer.span.set_attribute("duration_seconds", duration)
            timer.span.set_attribute("status", "error")
            timer.span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            timer.span.record_exception(e)
            timer.span.end()

        # Enhanced error logging
        log_operation(
            logger or get_logger("gunn.performance"),
            operation,
            status="error",
            global_seq=global_seq,
            view_seq=view_seq,
            agent_id=agent_id,
            req_id=req_id,
            latency_ms=duration * 1000,
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

        # Update span
        if timer.span:
            timer.span.set_attribute("duration_seconds", duration)
            timer.span.set_attribute("status", "success")
            timer.span.set_status(trace.Status(trace.StatusCode.OK))
            timer.span.end()

        # Enhanced success logging
        log_operation(
            logger or get_logger("gunn.performance"),
            operation,
            status="success",
            global_seq=global_seq,
            view_seq=view_seq,
            agent_id=agent_id,
            req_id=req_id,
            latency_ms=duration * 1000,
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


def record_queue_high_watermark(agent_id: str, queue_type: str, depth: int) -> None:
    """Record queue depth high watermark for backpressure monitoring.

    Args:
        agent_id: Agent identifier
        queue_type: Type of queue (agent_queue, system_queue, etc.)
        depth: Queue depth that triggered high watermark
    """
    QUEUE_DEPTH_HIGH_WATERMARK.labels(agent_id=agent_id, queue_type=queue_type).observe(
        depth
    )


def record_backpressure_event(agent_id: str, queue_type: str, policy: str) -> None:
    """Record a backpressure event.

    Args:
        agent_id: Agent identifier
        queue_type: Type of queue that triggered backpressure
        policy: Backpressure policy applied (defer, shed_oldest, drop_newest)
    """
    BACKPRESSURE_EVENTS.labels(
        agent_id=agent_id, queue_type=queue_type, policy=policy
    ).inc()


def record_circuit_breaker_state_change(
    component: str, from_state: str, to_state: str
) -> None:
    """Record circuit breaker state change.

    Args:
        component: Component name with circuit breaker
        from_state: Previous state (CLOSED, OPEN, HALF_OPEN)
        to_state: New state (CLOSED, OPEN, HALF_OPEN)
    """
    CIRCUIT_BREAKER_STATE.labels(
        component=component, from_state=from_state, to_state=to_state
    ).inc()


def record_error_recovery_action(
    error_type: str, recovery_action: str, agent_id: str = "unknown"
) -> None:
    """Record an error recovery action.

    Args:
        error_type: Type of error that occurred
        recovery_action: Recovery action taken
        agent_id: Agent identifier (optional)
    """
    ERROR_RECOVERY_ACTIONS.labels(
        error_type=error_type, recovery_action=recovery_action, agent_id=agent_id
    ).inc()


def record_intent_throughput(
    agent_id: str, intent_kind: str, status: str = "success"
) -> None:
    """Record intent processing throughput.

    Args:
        agent_id: Agent identifier
        intent_kind: Type of intent (Speak, Move, etc.)
        status: Processing status (success, error, conflict)
    """
    INTENT_THROUGHPUT.labels(
        agent_id=agent_id, intent_kind=intent_kind, status=status
    ).inc()


def record_conflict(agent_id: str, conflict_type: str) -> None:
    """Record an intent conflict.

    Args:
        agent_id: Agent identifier
        conflict_type: Type of conflict (staleness, validation, quota)
    """
    CONFLICT_RATE.labels(agent_id=agent_id, conflict_type=conflict_type).inc()


def record_observation_delivery_latency(agent_id: str, latency_seconds: float) -> None:
    """Record observation delivery latency.

    Args:
        agent_id: Agent identifier
        latency_seconds: Time from effect creation to observation delivery
    """
    OBSERVATION_DELIVERY_LATENCY.labels(agent_id=agent_id).observe(latency_seconds)


def update_global_seq(seq: int) -> None:
    """Update the current global sequence number metric.

    Args:
        seq: Current global sequence number
    """
    GLOBAL_SEQ_GAUGE.set(seq)


def update_view_seq(agent_id: str, seq: int) -> None:
    """Update the current view sequence number for an agent.

    Args:
        agent_id: Agent identifier
        seq: Current view sequence number
    """
    VIEW_SEQ_GAUGE.labels(agent_id=agent_id).set(seq)


def update_active_agents_count(count: int) -> None:
    """Update the number of active agents.

    Args:
        count: Number of currently active agents
    """
    ACTIVE_AGENTS_GAUGE.set(count)


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


class SystemMonitor:
    """Monitor system resources and record metrics."""

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._last_cpu_time = time.time()
        self._last_cpu_percent = 0.0

    def record_memory_usage(self, component: str = "system") -> float:
        """Record current memory usage and return bytes used.

        Args:
            component: Component name for metrics labeling

        Returns:
            Memory usage in bytes
        """
        try:
            memory_info = self._process.memory_info()
            memory_bytes: float = memory_info.rss  # Resident Set Size

            MEMORY_USAGE_BYTES.labels(component=component).set(memory_bytes)

            return memory_bytes

        except Exception as e:
            get_logger("gunn.telemetry.monitor").warning(
                "Failed to record memory usage", component=component, error=str(e)
            )
            return 0.0

    def record_cpu_usage(self, component: str = "system") -> float:
        """Record current CPU usage and return percentage.

        Args:
            component: Component name for metrics labeling

        Returns:
            CPU usage percentage
        """
        try:
            # Use interval to get accurate CPU percentage
            cpu_percent: float = self._process.cpu_percent(interval=None)

            # If this is the first call, cpu_percent returns 0.0
            # Use the cached value from previous call
            if cpu_percent == 0.0 and self._last_cpu_percent > 0.0:
                cpu_percent = self._last_cpu_percent
            else:
                self._last_cpu_percent = cpu_percent

            CPU_USAGE_PERCENT.labels(component=component).set(cpu_percent)

            return cpu_percent

        except Exception as e:
            get_logger("gunn.telemetry.monitor").warning(
                "Failed to record CPU usage", component=component, error=str(e)
            )
            return 0.0

    def get_system_stats(self) -> dict[str, Any]:
        """Get comprehensive system statistics.

        Returns:
            Dictionary with system resource information
        """
        try:
            memory_info = self._process.memory_info()
            cpu_times = self._process.cpu_times()

            # Get system-wide stats
            system_memory = psutil.virtual_memory()
            system_cpu = psutil.cpu_percent(interval=None)

            return {
                "process": {
                    "memory_rss_bytes": memory_info.rss,
                    "memory_vms_bytes": memory_info.vms,
                    "cpu_user_seconds": cpu_times.user,
                    "cpu_system_seconds": cpu_times.system,
                    "cpu_percent": self._process.cpu_percent(interval=None),
                    "num_threads": self._process.num_threads(),
                    "num_fds": self._process.num_fds()
                    if hasattr(self._process, "num_fds")
                    else 0,
                },
                "system": {
                    "memory_total_bytes": system_memory.total,
                    "memory_available_bytes": system_memory.available,
                    "memory_used_bytes": system_memory.used,
                    "memory_percent": system_memory.percent,
                    "cpu_percent": system_cpu,
                    "cpu_count": psutil.cpu_count(),
                },
            }
        except Exception as e:
            get_logger("gunn.telemetry.monitor").warning(
                "Failed to get system stats", error=str(e)
            )
            return {}


class BandwidthMonitor:
    """Monitor bandwidth usage for patch operations and other data transfers."""

    def __init__(self) -> None:
        self._logger = get_logger("gunn.telemetry.bandwidth")

    def record_patch_bandwidth(
        self,
        agent_id: str,
        patch_size_bytes: int,
        operation_count: int,
        is_fallback: bool = False,
    ) -> None:
        """Record bandwidth usage for patch operations.

        Args:
            agent_id: Agent identifier
            patch_size_bytes: Size of patch data in bytes
            operation_count: Number of patch operations
            is_fallback: Whether this was a fallback to full snapshot
        """
        # Record bandwidth
        BANDWIDTH_BYTES.labels(direction="outbound", component="observation_delta").inc(
            patch_size_bytes
        )

        # Record patch operation count
        PATCH_OPERATIONS_COUNT.labels(agent_id=agent_id).observe(operation_count)

        # Record fallback events
        if is_fallback:
            PATCH_FALLBACK_EVENTS.labels(agent_id=agent_id).inc()

        self._logger.debug(
            "Recorded patch bandwidth",
            agent_id=agent_id,
            patch_size_bytes=patch_size_bytes,
            operation_count=operation_count,
            is_fallback=is_fallback,
        )

    def record_data_transfer(
        self, direction: str, component: str, bytes_transferred: int
    ) -> None:
        """Record general data transfer.

        Args:
            direction: Transfer direction (inbound/outbound)
            component: Component name
            bytes_transferred: Number of bytes transferred
        """
        BANDWIDTH_BYTES.labels(direction=direction, component=component).inc(
            bytes_transferred
        )


# Global instances for convenience
system_monitor = SystemMonitor()
bandwidth_monitor = BandwidthMonitor()

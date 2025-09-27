"""Telemetry utilities for logging, metrics, and tracing.

This module provides centralized observability infrastructure including:
- Structured logging with PII redaction
- Prometheus metrics collection
- OpenTelemetry tracing setup
- Monotonic timing utilities
"""

import asyncio
import re
import time
from typing import Any, Dict, Optional

import structlog
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram, start_http_server


# PII patterns for redaction
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3,4}[-.]?\d{4}\b|\b\d{3}[-.]?\d{4}\b"),  # Support both 10 and 7 digit formats
    "token": re.compile(r"\b[A-Za-z0-9]{20,}\b"),  # Generic token pattern
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
}

# Prometheus metrics
OPERATION_COUNTER = Counter(
    "gunn_operations_total",
    "Total number of operations",
    ["operation", "status", "agent_id"],
)

OPERATION_DURATION = Histogram(
    "gunn_operation_duration_seconds",
    "Duration of operations in seconds",
    ["operation", "agent_id"],
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

QUEUE_DEPTH_GAUGE = {}  # Will be populated with agent-specific gauges


def redact_pii(text: str) -> str:
    """Redact personally identifiable information from text.
    
    Args:
        text: Input text that may contain PII
        
    Returns:
        Text with PII patterns replaced with [REDACTED_<type>]
    """
    if not isinstance(text, str):
        return text
        
    result = text
    for pii_type, pattern in PII_PATTERNS.items():
        result = pattern.sub(f"[REDACTED_{pii_type.upper()}]", result)
    
    return result


def redact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
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
            result[key] = [redact_dict(item) if isinstance(item, dict) else 
                          redact_pii(item) if isinstance(item, str) else item 
                          for item in value]
        else:
            result[key] = value
    
    return result


class PIIRedactionProcessor:
    """Structlog processor for PII redaction."""
    
    def __call__(self, logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
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
    
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            PIIRedactionProcessor(),
            structlog.dev.ConsoleRenderer() if log_level == "DEBUG" else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            level_map.get(log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Add service context
    structlog.contextvars.bind_contextvars(service=service_name)


def setup_tracing(service_name: str = "gunn", otlp_endpoint: Optional[str] = None) -> None:
    """Initialize OpenTelemetry tracing.
    
    Args:
        service_name: Name of the service for tracing
        otlp_endpoint: OTLP endpoint URL (optional)
    """
    if otlp_endpoint:
        span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        span_processor = BatchSpanProcessor(span_exporter)
    else:
        # Use console exporter for development
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        span_exporter = ConsoleSpanExporter()
        span_processor = BatchSpanProcessor(span_exporter)
    
    provider = TracerProvider()
    provider.add_span_processor(span_processor)
    trace.set_tracer_provider(provider)


def setup_metrics(service_name: str = "gunn", otlp_endpoint: Optional[str] = None) -> None:
    """Initialize OpenTelemetry metrics.
    
    Args:
        service_name: Name of the service for metrics
        otlp_endpoint: OTLP endpoint URL (optional)
    """
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
    start_http_server(port)


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
        self.start_time: Optional[float] = None
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
            operation=self.operation,
            status=status,
            agent_id=self.agent_id
        ).inc()
        
        OPERATION_DURATION.labels(
            operation=self.operation,
            agent_id=self.agent_id
        ).observe(duration)
        
        # Log the operation
        self.logger.info(
            "Operation completed",
            operation=self.operation,
            agent_id=self.agent_id,
            duration_ms=duration * 1000,
            status=status,
        )


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
# Technology Stack

## Build System & Package Management

- **Build system**: Hatchling (modern Python packaging)
- **Package manager**: uv (fast Python package management)
- **Python version**: 3.13+ (required minimum)

## Core Dependencies

- **pydantic v2**: Data validation and JSON schema generation
- **orjson**: High-performance JSON serialization with deterministic ordering
- **jsonpatch**: RFC6902 JSON Patch operations for observation deltas
- **structlog**: Structured logging with PII redaction
- **asyncio**: Async/await concurrency model throughout

## Monitoring & Observability

- **prometheus-client**: Metrics collection (queue depths, throughput, latency)
- **OpenTelemetry**: Distributed tracing across adapters
  - `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`
  - (Web adapter) `opentelemetry-instrumentation-fastapi`
- **structlog**: Structured logging with `global_seq`, `view_seq`, `agent_id`, `req_id`

## External Integration
- **FastAPI**: Web adapter with REST/WebSocket endpoints
- **websockets**: Real-time communication with game engines
- **SQLite**: Persistent storage (via `sqlite3` or `aiosqlite`) for deduplication and audit logs

## Development Tools

- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checking
- **pytest-asyncio**: Async testing framework
- **pre-commit**: Git hooks for code quality

## Common Commands

```bash
# Development setup
uv sync                   # Install dependencies
uv run pytest             # Run tests
uv run ruff format        # Format code
uv run mypy src/          # Type check

# Build and distribution
uv build                  # Build wheel/sdist
uv run python -m gunn     # Run CLI tools

# Testing specific scenarios
uv run pytest -k "test_cancellation"  # Test interruption behavior
uv run pytest -k "test_determinism"   # Test replay consistency
```

## Architecture Patterns

- **Event-driven**: All state changes flow through immutable event log
- **Two-phase commit**: Intent validation → Effect creation for consistency
- **Policy separation**: ObservationPolicy and EffectValidator are independent
- **Dependency injection**: Orchestrator accepts pluggable validators and policies
- **Circuit breaker**: Fault tolerance with configurable failure thresholds
- **Monotonic timing**: Internal timing via event loop’s monotonic clock; wall-clock for logs only

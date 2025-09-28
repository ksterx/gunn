# Technology Stack

## Build System & Package Management

- **Build system**: Hatchling (configured via `pyproject.toml`)
- **Package manager**: uv (fast resolver/runner used in CI and docs)
- **Python version**: 3.13+ (type-checked against 3.13, compatible with 3.12 during transition)

## Core Runtime Dependencies

- **pydantic v2** for data validation (`WorldState`, `View`, etc.) and schema generation.
- **orjson** for deterministic JSON serialization along the event log.
- **jsonpatch** to emit RFC6902 deltas per agent view.
- **structlog** for structured logging with optional PII redaction.
- **asyncio** primitives throughout the orchestrator, queues, and storage layers.

## Monitoring & Observability

- **prometheus-client** counters/histograms exported via helpers in `utils.telemetry`.
- **OpenTelemetry** packages vendored for future tracing; instrumentation hooks are scaffolded but not yet wired into adapters.
- **structlog** plus custom processors for PII redaction and performance timing.

## External Integration Stack (planned vs current)

- **FastAPI** and **websockets** are declared so the upcoming web adapter (Task 17) can expose REST/WebSocket APIs. Current code only contains placeholder modules.
- **aiosqlite** backs the deduplication store; in-memory mode is available for tests.
- Adapter packages (`src/gunn/adapters/{web,unity,llm}`) currently export stubs until their respective roadmap tasks land.

## Development Tooling

- **ruff** for linting + formatting (`uv run ruff format`, `uv run ruff check`).
- **mypy** with strict settings on library code and relaxed rules for tests.
- **pytest** / **pytest-asyncio** for unit and async integration tests.
- **pre-commit** hooks mirror CI checks.

## Common Commands

```bash
# Environment setup
uv sync

# Quality gates
uv run ruff format
uv run ruff check
uv run mypy src/
uv run pytest

# Packaging & utilities
uv build
uv run python -m gunn replay --help
```

## Architecture Patterns

- **Event-driven core**: Effects append to an immutable log with hash chaining and replay support.
- **Two-phase intent handling**: Idempotency → quota/backpressure → scheduling → validation → effect emission.
- **Policy separation**: Observation policies inject filtering + latency logic; validator interface allows domain rules.
- **Cancellation primitives**: Cancel tokens track generation state and reason codes.
- **Telemetry hub**: Centralised logging/metrics ensures consistent observability across components.

## Implementation Notes

- Effect validation currently uses `DefaultEffectValidator` (allow-all) until Task 6 introduces domain checks.
- OpenTelemetry exporters are present but not yet initialised; integrate once adapters emit spans.
- RL/message facades and external adapters will start consuming FastAPI/websockets dependencies as the roadmap progresses.

# Project Structure

## Repository Layout

```
gunn/
├── .github/
│   └── workflows/
│       └── ci.yml          # CI pipeline (lint, type checks, tests, contracts)
├── .kiro/                  # Kiro guidance and specs consumed by the AI agent
│   ├── steering/
│   └── specs/
├── docs/                   # Human-facing documentation (kept in sync with steering docs)
│   ├── product.md
│   ├── structure.md
│   └── tech.md
├── schemas/                # Contract definitions checked in CI
│   ├── README.md
│   ├── openapi.yaml
│   └── proto/
│       └── simulation.proto
├── src/gunn/               # Python package (see layout below)
├── tests/                  # Repository-level tests (integration, performance, contract)
│   ├── contract/
│   ├── integration/
│   ├── performance/
│   └── test_*.py
├── dist/                   # Build artifacts (created by `uv build`, may be absent locally)
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── SECURITY.md
├── TEST_STATUS.md
├── pyproject.toml          # Project + tool configuration
└── uv.lock                 # Dependency lock file
```

## Source Code Organization (current implementation)

```
src/gunn/
├── __init__.py             # Public exports (Orchestrator, AgentHandle, etc.)
├── __main__.py             # Allows `python -m gunn`
├── adapters/               # Adapter namespaces (placeholders until tasks 17–19)
│   ├── __init__.py
│   ├── llm/__init__.py
│   ├── unity/__init__.py
│   └── web/__init__.py
├── cli/
│   ├── __init__.py
│   ├── __main__.py
│   └── replay.py           # Deterministic replay CLI (implemented)
├── core/
│   ├── __init__.py
│   ├── event_log.py        # Append-only log with hash chain + concurrency tests
│   └── orchestrator.py     # Orchestrator, AgentHandle, default validator (stubbed logic)
├── facades/
│   └── __init__.py         # Placeholder package for upcoming RL/message facades
├── policies/
│   ├── __init__.py
│   └── observation.py      # Partial observation filtering + JSON Patch deltas
├── schemas/
│   ├── __init__.py
│   ├── messages.py         # Pydantic models (WorldState, View, etc.)
│   └── types.py            # TypedDicts (Intent, Effect, ObservationDelta, CancelToken)
├── storage/
│   ├── __init__.py
│   └── dedup_store.py      # SQLite + in-memory deduplication store
└── utils/
    ├── __init__.py
    ├── errors.py           # Structured error types (StaleContextError, etc.)
    ├── hashing.py          # Hash chain utilities
    ├── hashing_demo.py     # Example usage (dev aid)
    ├── scheduling.py       # Weighted round-robin scheduler
    ├── telemetry.py        # Logging, Prometheus metrics, timers
    └── timing.py           # TimedQueue / monotonic clock helpers (+ demos/tests)
```

- Unit tests for core utilities live alongside their modules (`test_*.py` inside `src/gunn/**`).
- Repository-level integration/performance/contract tests are in `tests/`.
- Snapshot storage, validation policies, and facade implementations are tracked in `tasks.md` and not yet present in the tree.

## Key Design Principles (still applicable)

- **Schema separation**: Contract files live in `schemas/`; runtime models live in `src/gunn/schemas/`.
- **Deterministic core**: `core/orchestrator.py` and `core/event_log.py` enforce ordering and log integrity.
- **Observation policies**: `policies/observation.py` encapsulates filtering/delta logic with configurable limits.
- **Two-phase processing**: Orchestrator handles idempotency, quotas, backpressure, fairness, then effect emission.
- **Telemetry**: `utils/telemetry.py` centralises logging + metrics; metrics server start is opt-in via `start_metrics_server`.

## Configuration & Tooling Files

- `pyproject.toml` configures Hatchling, uv integration, dependencies, Ruff, MyPy, and Pytest.
- `uv.lock` locks dependency versions.
- `.github/workflows/ci.yml` runs formatting, linting, typing, tests, and contract drift checks.
- CI/formatting assumes `uv` for environment management; local `.venv` directories are optional and not checked in.

## Development Conventions

- `src/` layout with explicit `py.typed` for type checkers.
- Strict typing in production code; tests relax some MyPy rules via overrides.
- Async-first design: orchestration, queues, and stores expose async APIs.
- Docs and steering guides mirror each other—update `docs/` and `.kiro/steering/` together.

## Import Patterns

```python
# Public API imports
from gunn import Orchestrator, AgentHandle, OrchestratorConfig

# Internal usage examples
from gunn.core.orchestrator import Orchestrator
from gunn.policies.observation import ObservationPolicy
from gunn.schemas.types import Intent, Effect
from gunn.storage.dedup_store import DedupStore
from gunn.utils.telemetry import setup_logging, PerformanceTimer
```

## Open Source Metadata

- Root documents (`LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`) stay in sync with GitHub publishing expectations.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**gunn (群)** is a multi-agent simulation core providing controlled agent-environment interaction with:
- Event-driven architecture with deterministic ordering
- Partial observation through configurable policies
- Concurrent intent processing without head-of-line blocking
- Intelligent interruption and staleness detection
- Both RL-style (`env.step()`) and message-oriented (`env.emit()`) APIs

## Essential Commands

### Development Workflow
```bash
# Install dependencies (use uv, not pip)
uv sync --all-extras

# Run all tests
uv run pytest

# Run specific test categories
uv run pytest -m "not slow"              # Skip slow tests
uv run pytest tests/integration/         # Integration tests only
uv run pytest -k "test_cancellation"     # Specific pattern

# Run single test file or function
uv run pytest tests/unit/test_event_log.py
uv run pytest tests/unit/test_event_log.py::test_deterministic_ordering

# Quality checks (run before committing)
uv run ruff check                        # Lint
uv run ruff format                       # Format
uv run mypy src/                         # Type check

# Fix linting issues automatically
uv run ruff check --fix

# Pre-commit hooks (runs automatically on commit)
uv run pre-commit install
uv run pre-commit run --all-files

# Build documentation
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html

# Run demo (requires demo group)
uv sync --group demo
python demo/backend/server.py            # Backend server
python demo/frontend/__main__.py         # Frontend (pygame)
```

### Important Testing Notes
- **Unit tests**: Located in `tests/unit/` organized by module (core, policies, utils, etc.)
- **Integration tests**: Located in `tests/integration/`
- **Performance tests**: Located in `tests/performance/`
- **Contract tests**: Located in `tests/contract/` for schema validation
- **Demo tests**: Located in `demo/tests/` for demo-specific functionality
- Always use `uv run pytest`, never bare `pytest`
- Async tests use `pytest-asyncio` with `asyncio_mode = "auto"`

## Architecture Deep Dive

### Event-Driven Core (src/gunn/core/)

The system operates through a central **Orchestrator** that coordinates all operations:

1. **EventLog** (`event_log.py`): Immutable, append-only log of all effects
   - Deterministic ordering: `(sim_time, priority, source_id, uuid)`
   - Thread-safe with `asyncio.Lock`
   - Supports compaction and memory management

2. **Orchestrator** (`orchestrator.py`): Central coordinator
   - Manages agent registration and lifecycle
   - Routes intents → validators → effects → observations
   - Enforces backpressure, quota limits, and staleness checks
   - Non-blocking per-agent operation queues

3. **Agent Handles**: Per-agent interfaces for observation and intent submission
   - `submit_intent(intent)`: Submit agent actions for validation
   - `next_observation()`: Async iterator for observations
   - `get_view()`: Current world state filtered by ObservationPolicy

4. **ConcurrentIntentProcessor** (`concurrent_processor.py`):
   - Processes multiple intents in parallel using configurable modes
   - Modes: `PARALLEL` (concurrent), `SEQUENTIAL` (ordered), `BATCH` (grouped)
   - Handles conflicts and rollback with transactional semantics

### Two-Phase Commit Pattern

**Critical**: All state changes follow this pattern:

1. **Intent Validation**: `EffectValidator.validate_intent(intent, world_state) → bool`
   - Check permissions, quotas, cooldowns, world state constraints
   - **Must be deterministic** based on current world state only

2. **Effect Creation**: `intent → Effect` (if validation passes)
   - Effect is appended to EventLog
   - World state updated via `apply_effect(effect, world_state)`
   - Observations generated and distributed to relevant agents

### Observation System (src/gunn/policies/)

**ObservationPolicy** controls what each agent observes:

- **Distance-based filtering**: Spatial constraints on observations
- **Relationship filtering**: Social graph-based visibility
- **Field visibility**: Per-field access control
- **Delta generation**: RFC6902 JSON Patch for efficient updates
- **Latency modeling**: Realistic observation delays

Observations are delivered as `ObservationDelta` objects containing:
- Previous view hash (for staleness detection)
- JSON Patch operations (or full snapshot if too many ops)
- Metadata (timestamp, source agent, priority)

### Staleness Detection (src/gunn/utils/scheduling.py, src/gunn/policies/staleness.py)

**Intent-specific staleness logic** prevents false positives:

- Track `context_seq` (agent's view version) per intent
- Compare against current view hash to detect stale context
- Raise `StaleContextError` for intents based on outdated observations
- Intelligent staleness: different thresholds per intent type
- Reduces unnecessary LLM regeneration costs

### Facades (src/gunn/facades/)

**Two API styles** for different use cases:

1. **RLFacade** (`rl.py`): Reinforcement learning interface
   - `env.observe(agent_id)`: Get current observation
   - `env.step(agent_id, intent)`: Submit action, get (effect, delta)
   - Synchronous-style API suitable for RL training loops

2. **MessageFacade** (`message.py`): Event-driven interface
   - `await env.emit(event)`: Publish events to system
   - Async streaming of observations
   - Suitable for real-time interactive applications

### External Adapters (src/gunn/adapters/)

**Integration points** for external systems:

- **UnityAdapter**: WebSocket bridge for Unity game engine
- **WebAdapter**: REST + WebSocket API server (FastAPI)
- **LLMAdapter**: Token-level streaming with sub-100ms cancellation

### Memory Management (src/gunn/utils/memory.py)

**MemoryManager** handles resource limits:

- Configurable memory limits per component
- Automatic event log compaction when thresholds exceeded
- Memory pressure callbacks for graceful degradation
- Metrics tracking (heap size, RSS, event counts)

### Performance SLOs

The system is designed to meet:
- **Observation delivery**: ≤20ms median latency
- **Cancellation response**: ≤100ms for LLM stream halting
- **Intent throughput**: ≥100 intents/sec per agent
- **No head-of-line blocking**: Independent agent operation queues

## Key Design Principles

1. **Deterministic ordering**: All events use `(sim_time, priority, source_id, uuid)` for total order
2. **Async by default**: All I/O uses `async/await`
3. **Policy separation**: ObservationPolicy and EffectValidator are pluggable
4. **Event sourcing**: Complete event log enables replay and debugging
5. **Type safety**: Complete type annotations, strict mypy checking
6. **Non-blocking**: Per-agent queues prevent cross-agent blocking

## Code Conventions

### Type Hints
- Use modern syntax: `str | None` instead of `Optional[str]`
- All public APIs must have complete type annotations
- Protocol classes for interfaces (see `EffectValidator` in orchestrator.py)

### Async Patterns
- Always use `async/await`, never callbacks
- Use `asyncio.Lock` for thread safety
- Use `asyncio.Queue` for async message passing
- Timeout-aware operations with `asyncio.wait_for()`

### Testing Patterns
- **Unit tests structure**: Mirror source structure in `tests/unit/`
  - `tests/unit/core/` for core module tests
  - `tests/unit/policies/` for policy tests
  - `tests/unit/utils/` for utility tests
  - etc.
- **Integration tests**: In `tests/integration/` for cross-module scenarios
- Use fixtures for common test setup
- Mock external dependencies (LLM APIs, databases)
- Test both success and failure paths
- Test determinism with replay scenarios

### Commit Message Format
Follow Conventional Commits:
- `feat:` new features
- `fix:` bug fixes
- `test:` test changes
- `refactor:` code restructuring
- `docs:` documentation
- `perf:` performance improvements

## Common Pitfalls

1. **Don't mix sync/async**: Use `async def` and `await` consistently
2. **Don't modify EventLog outside Orchestrator**: It's append-only for a reason
3. **Don't skip validation**: Always use two-phase commit pattern
4. **Don't assume ordering without priority**: Use explicit priorities for intent ordering
5. **Don't forget staleness checks**: Always validate `context_seq` before processing intents
6. **Don't use `pip`**: Always use `uv` for dependency management
7. **Don't put test files in src/**: All tests belong in `tests/` directory, not co-located with source

## Demo Applications

Located in `demo/`:
- `backend/server.py`: FastAPI server with WebSocket support
- `frontend/`: Pygame-based real-time visualization
- `backend/gunn_integration.py`: Example Orchestrator setup

Run with: `uv sync --group demo` then start backend and frontend separately.

## Schemas and Contracts

Located in `schemas/`: JSON schema definitions for intent/effect validation
- Contract tests in `tests/contract/` ensure schema compliance
- Use Pydantic models in `src/gunn/schemas/` for runtime validation

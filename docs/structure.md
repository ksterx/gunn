# Project Structure

## Repository Layout

```
gunn/
├── .github/
│   └── workflows/
│       └── ci.yml          # CI pipeline (lint, type, test, contract)
├── .kiro/                  # Kiro IDE configuration
│   ├── steering/          # AI assistant guidance documents
│   └── specs/             # Feature specifications and design docs
├── docs/                   # External documentation
│   ├── product.md         # Product overview
│   ├── tech.md            # Technology stack
│   └── structure.md       # Project structure
├── schemas/                # Contract definitions (golden files)
│   ├── openapi.yaml       # REST API contract
│   └── proto/             # Protocol buffer definitions
├── tests/                  # Integration, performance, and contract tests
│   ├── integration/       # Cross-component tests
│   ├── performance/       # Benchmark and SLO validation
│   └── contract/          # API contract validation
├── src/gunn/              # Main package source code
│   ├── __init__.py        # Package entry point
│   └── py.typed           # Type checking marker
├── dist/                  # Build artifacts (wheels, sdist)
├── .venv/                 # Virtual environment (uv managed)
├── pyproject.toml         # Project configuration and dependencies
├── uv.lock                # Dependency lock file
├── LICENSE                # Open source license
├── CONTRIBUTING.md        # Contribution guidelines
├── CODE_OF_CONDUCT.md     # Community standards
├── SECURITY.md            # Security policy
└── README.md              # Project documentation
```

## Source Code Organization

The `src/gunn/` directory contains the core implementation with clear separation of concerns:

```
src/gunn/
├── __init__.py             # Public API exports
├── py.typed                # Type checking marker
├── core/                   # Core simulation engine
│   ├── orchestrator.py     # Central coordinator
│   ├── event_log.py        # Immutable event storage
│   ├── world_state.py      # Current state projection
│   └── agent_handle.py     # Per-agent interface
├── policies/               # Observation and validation policies
│   ├── observation.py      # Partial observation filtering
│   └── validation.py       # Intent validation rules
├── facades/                # API interfaces
│   ├── rl_facade.py        # RL-style env.step() interface
│   └── message_facade.py   # Event-driven messaging
├── adapters/               # External system integration (subpackages)
│   ├── web/                # Web adapter components
│   │   ├── __init__.py     # Web adapter public interface
│   │   ├── app.py          # FastAPI application setup
│   │   ├── routes.py       # REST endpoint handlers
│   │   ├── ws.py           # WebSocket handlers
│   │   └── auth.py         # Authentication and authorization
│   ├── unity/              # Unity game engine integration
│   │   └── __init__.py     # Unity adapter interface
│   └── llm/                # LLM streaming integration
│       └── __init__.py     # LLM adapter interface
├── storage/                # Persistent storage layer
│   ├── dedup_store.py      # Deduplication storage (aiosqlite)
│   └── snapshots.py        # WorldState snapshot storage
├── schemas/                # Python schema definitions
│   ├── messages.py         # Pydantic models for messages
│   └── types.py            # TypedDict definitions and core types
├── utils/                  # Shared utilities
│   ├── timing.py           # TimedQueue and monotonic clock
│   ├── hashing.py          # Hash chain integrity
│   ├── errors.py           # Structured error types
│   └── telemetry.py        # Logging, metrics, and tracing aggregation
└── cli/                    # Command-line interface
    ├── __main__.py         # Entry point for `python -m gunn`
    └── replay.py           # Event log replay utility
```

## Key Design Principles

### Schema and Storage Separation
- **Contract definitions**: `schemas/` contains golden files (openapi.yaml, proto/) for CI drift detection
- **Python schemas**: `src/gunn/schemas/` contains Pydantic/TypedDict definitions
- **Storage layer**: `src/gunn/storage/` handles SQLite and persistent data separately

### Observability Centralization
- **Telemetry hub**: `utils/telemetry.py` aggregates logging, metrics, and tracing initialization
- **Common helpers**: Structured logging with PII redaction, Prometheus metrics, OpenTelemetry setup

### Adapter Subpackaging
- **Role separation**: Each adapter (web/, unity/, llm/) is a subpackage with clear responsibilities
- **Web adapter**: Separate routes.py, ws.py, auth.py for maintainability
- **Extensibility**: Easy to add new adapters without affecting core

### Two-Tier Testing Structure
- **Unit tests**: Co-located with source code using `test_*.py` pattern
- **Integration/Performance**: Centralized in `tests/` for CI pipeline efficiency
- **Contract tests**: `tests/contract/` validates API schema compliance

### CLI Entry Point
- **Module execution**: `cli/__main__.py` enables `python -m gunn` command
- **Extensible commands**: Easy to add replay, benchmark, and other utilities

## Configuration Files

- **pyproject.toml**: Project metadata, dependencies, and tool configuration
- **uv.lock**: Exact dependency versions for reproducible builds
- **.python-version**: Python version specification for uv
- **.gitignore**: Standard Python gitignore with build artifacts
- **.github/workflows/ci.yml**: CI pipeline with contract validation

## Development Conventions

- **Package structure**: Use `src/` layout for proper import testing
- **Type annotations**: All public APIs must have complete type hints
- **Async by default**: Use `async/await` for all I/O operations
- **Error handling**: Structured exceptions with recovery actions
- **Testing**: Unit tests co-located, integration tests in `tests/`
- **Documentation**: Docstrings for all public classes and methods
- **Contract validation**: CI checks for schema drift in golden files

## Import Patterns

```python
# Public API imports (from __init__.py)
from gunn import Orchestrator, AgentHandle, WorldState

# Internal imports (within package)
from gunn.core.orchestrator import Orchestrator
from gunn.policies.observation import ObservationPolicy
from gunn.utils.errors import StaleContextError
from gunn.schemas.types import Intent, Effect
from gunn.storage.dedup_store import DedupStore
from gunn.utils.telemetry import setup_logging, get_metrics
```

## File Naming Conventions

- **Modules**: `snake_case.py` (e.g., `event_log.py`)
- **Classes**: `PascalCase` (e.g., `EventLog`, `AgentHandle`)
- **Functions/methods**: `snake_case` (e.g., `submit_intent()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_PRIORITY`)
- **Private members**: Leading underscore (e.g., `_global_seq`)

## Open Source Metadata

For OSS compliance, include these files at repository root:
- **LICENSE**: Open source license (MIT, Apache 2.0, etc.)
- **CONTRIBUTING.md**: Contribution guidelines and development setup
- **CODE_OF_CONDUCT.md**: Community standards and behavior expectations
- **SECURITY.md**: Security policy and vulnerability reporting process

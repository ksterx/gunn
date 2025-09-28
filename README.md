# gunn (群)

**gunn** is a multi-agent simulation core that provides a controlled interface for agent-environment interaction, supporting both single and multi-agent settings with partial observation, concurrent execution, and intelligent interruption capabilities.

## Features

- **Partial Observation**: Agents see only what they should based on distance, relationships, and policies
- **Concurrent Execution**: Multiple agents can act simultaneously without blocking each other
- **Intelligent Interruption**: Agents can interrupt and regenerate responses when new relevant information arrives
- **Event-Driven Architecture**: Unified core with deterministic ordering and complete audit trails
- **Dual API Facades**: Both RL-style (`env.step()`) and message-oriented (`env.emit()`) interfaces
- **Real-time Streaming**: Token-level streaming with sub-100ms cancellation response
- **External Integration**: Unity, Unreal, and web API adapters for rich interactive experiences
- **Deterministic Replay**: Complete event logs enable debugging and analysis
- **Multi-tenant Security**: Proper isolation and access controls for production use

## Quick Start

### Installation

```bash
# Install with uv (recommended)
uv add gunn

# Or with pip
pip install gunn
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/gunn.git
cd gunn

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Run linting and type checking
uv run ruff check
uv run mypy src/
```

### Basic Usage

```python
import asyncio
from gunn import Orchestrator, ObservationPolicy

async def main():
    # Create orchestrator
    orchestrator = Orchestrator()

    # Register agents with observation policies
    policy = ObservationPolicy(distance_limit=10.0)
    agent_a = await orchestrator.register_agent("agent_a", policy)
    agent_b = await orchestrator.register_agent("agent_b", policy)

    # Submit intents and observe results
    intent = {
        "kind": "Speak",
        "payload": {"text": "Hello, world!"},
        "context_seq": 0,
        "req_id": "req_1",
        "agent_id": "agent_a",
        "priority": 1,
        "schema_version": "1.0.0"
    }

    await agent_a.submit_intent(intent)

    # Get observations
    observation = await agent_b.next_observation()
    print(f"Agent B observed: {observation}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture

The system is built around a central event-driven core with the following components:

- **Orchestrator**: Central coordinator managing all system operations
- **Event Log**: Immutable sequence of effects with deterministic ordering
- **World State**: Current state projection from the event log
- **Agent Handles**: Per-agent interfaces for observation and intent submission
- **Observation Policies**: Configurable filtering for partial observation
- **Effect Validators**: Intent validation and conflict resolution

## API Interfaces

### RL-Style Facade

```python
from gunn.facades import RLFacade

env = RLFacade(orchestrator)
observation = env.observe("agent_id")
effect, delta = env.step("agent_id", intent)
```

### Message-Oriented Facade

```python
from gunn.facades import MessageFacade

env = MessageFacade(orchestrator)
await env.emit(event)
```

## External Integrations

### Unity Integration

```python
from gunn.adapters.unity import UnityAdapter

adapter = UnityAdapter(orchestrator)
await adapter.connect("ws://localhost:8080")
```

### Web API

```python
from gunn.adapters.web import WebAdapter

app = WebAdapter(orchestrator).create_app()
# Serves REST and WebSocket endpoints
```

### LLM Streaming

```python
from gunn.adapters.llm import LLMAdapter

adapter = LLMAdapter(orchestrator)
async for token in adapter.stream_generate(prompt, cancel_token):
    print(token)
```

## Documentation

The documentation site is built with Sphinx and published from the `docs/_build/html` directory.
Build it locally with:

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

When targeting GitHub Pages, instruct Pages to serve from `docs/_build/html`.

## Configuration

The system supports extensive configuration through environment variables and config files:

```python
from gunn import OrchestratorConfig

config = OrchestratorConfig(
    max_agents=100,
    staleness_threshold=0,
    debounce_ms=50.0,
    deadline_ms=5000.0,
    token_budget=1000,
    backpressure_policy="defer",
    default_priority=0
)

orchestrator = Orchestrator(config)
```

## Performance

The system is designed to meet strict performance SLOs:

- **Observation Delivery**: ≤20ms median latency under nominal load
- **Cancellation Response**: ≤100ms cancel-to-halt latency for LLM streaming
- **Intent Processing**: ≥100 intents/sec per agent under benchmark scenarios
- **Non-blocking Operations**: No head-of-line blocking across agents

## Development

### Project Structure

```
gunn/
├── src/gunn/              # Main package source
│   ├── core/              # Core simulation engine
│   ├── policies/          # Observation and validation policies
│   ├── facades/           # API interfaces
│   ├── adapters/          # External system integration
│   ├── storage/           # Persistent storage layer
│   ├── schemas/           # Data models and types
│   ├── utils/             # Shared utilities
│   └── cli/               # Command-line interface
├── tests/                 # Test suites
├── schemas/               # Contract definitions
└── docs/                  # Documentation
```

### Common Commands

```bash
# Development
uv sync                    # Install dependencies
uv run pytest            # Run tests
uv run ruff check         # Lint code
uv run mypy src/          # Type check

# Build and distribution
uv build                  # Build wheel/sdist
uv run python -m gunn     # Run CLI tools

# Testing specific scenarios
uv run pytest -k "test_cancellation"  # Test interruption behavior
uv run pytest -k "test_determinism"   # Test replay consistency
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the full test suite
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/ksterx/gunn/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ksterx/gunn/discussions)

# Contributing to gunn

Thank you for your interest in contributing to gunn! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Getting Started

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/gunn.git
   cd gunn
   ```

2. **Install dependencies**
   ```bash
   uv sync --all-extras
   ```

3. **Install pre-commit hooks**
   ```bash
   uv run pre-commit install
   ```

4. **Verify setup**
   ```bash
   uv run pytest
   uv run ruff check
   uv run mypy src/
   ```

## Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following the project conventions
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**
   ```bash
   uv run ruff check --fix
   uv run ruff format
   uv run mypy src/
   uv run pytest
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Code Style

- **Formatting**: We use `ruff format` for code formatting
- **Linting**: We use `ruff check` for linting
- **Type checking**: We use `mypy` with strict settings
- **Import sorting**: Handled automatically by ruff
- **Line length**: 88 characters maximum

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions or modifications
- `refactor:` for code refactoring
- `perf:` for performance improvements
- `chore:` for maintenance tasks

### Testing

- **Unit tests**: Co-located with source code using `test_*.py` pattern
- **Integration tests**: Located in `tests/integration/`
- **Performance tests**: Located in `tests/performance/`
- **Contract tests**: Located in `tests/contract/`

#### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest -m "not slow"           # Skip slow tests
uv run pytest tests/integration/      # Integration tests only
uv run pytest -k "test_cancellation"  # Specific test pattern

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

### Documentation

- **Docstrings**: All public classes and methods must have docstrings
- **Type hints**: All public APIs must have complete type annotations
- **README**: Update README.md for user-facing changes
- **API docs**: Update relevant documentation in `docs/`

## Project Structure

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
├── schemas/               # Contract definitions (golden files)
├── docs/                  # Documentation
└── .github/               # GitHub workflows and templates
```

## Architecture Guidelines

### Core Principles

1. **Event-driven architecture**: All state changes flow through the event log
2. **Deterministic ordering**: Events are ordered using `(sim_time, priority, source_id, uuid)`
3. **Policy separation**: ObservationPolicy and EffectValidator are independent
4. **Async by default**: Use `async/await` for all I/O operations
5. **Type safety**: Complete type annotations for all public APIs

### Design Patterns

- **Two-phase commit**: Intent validation → Effect creation
- **Circuit breaker**: Fault tolerance with configurable thresholds
- **Dependency injection**: Orchestrator accepts pluggable components
- **Observer pattern**: Event distribution through observation policies

### Performance Considerations

- **SLO targets**: 20ms observation delivery, 100ms cancellation response
- **Non-blocking**: No head-of-line blocking across agents
- **Memory management**: Configurable limits and compaction strategies
- **Concurrency**: Agent-specific thread pools and async operations

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass locally
- [ ] Code is properly formatted and linted
- [ ] Type checking passes
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] Changes are covered by tests

### PR Description Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Code is commented, particularly in hard-to-understand areas
- [ ] Corresponding changes to documentation made
- [ ] Changes generate no new warnings
- [ ] Tests added that prove the fix is effective or feature works
- [ ] New and existing unit tests pass locally
```

## Release Process

Releases are managed by maintainers and follow semantic versioning:

- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

## Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/your-org/gunn/discussions)
- **Bugs**: Open a [GitHub Issue](https://github.com/your-org/gunn/issues)
- **Security**: See [SECURITY.md](SECURITY.md) for reporting security issues

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## License

By contributing to gunn, you agree that your contributions will be licensed under the same license as the project (MIT License).

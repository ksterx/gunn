# Repository Guidelines

## Project Structure & Module Organization
- `src/gunn/` contains the production package: `core/` for the orchestrator, `facades/` for RL/message APIs, `adapters/` for external bridges, `policies/` for visibility rules, and `storage/` for persistence layers.
- Tests live in `tests/`: fast suites at the root, scenario flows in `tests/integration/`, timing and load checks in `tests/performance/`, and contract fixtures in `tests/contract/`.
- Shared assets sit in `schemas/` (golden records), `docs/` (design notes), and `scripts/` (uv helpers); artifacts in `dist/` are generated and should remain untouched.

## Build, Test, and Development Commands
- Bootstrap with `uv sync --all-extras`; rerun after dependency changes.
- Install hooks via `uv run pre-commit install`; enforce locally using `uv run pre-commit run --all-files`.
- Execute `uv run pytest` for the default suite, refine to `uv run pytest tests/integration` for system flows, or append `-k pattern` to target a scenario.
- Keep quality gates green with `uv run ruff check`, `uv run ruff format`, and `uv run mypy src/`; validate packaging through `uv run python -m build`.

## Coding Style & Naming Conventions
- Target Python 3.13, four-space indents, and 88-character lines; rely on Ruff for formatting and import ordering.
- Adopt `snake_case` for modules/functions, `PascalCase` for classes, and `SCREAMING_SNAKE_CASE` for constants; align filenames with the primary exported symbol.
- Public entry points must carry precise type hints and concise docstrings; design async flows so orchestrator sequencing stays deterministic.

## Testing Guidelines
- `pytest` discovers `test_*` files automatically; keep fast checks near the code and move cross-component flows into `tests/integration/`.
- Mark expensive scenarios with `@pytest.mark.slow` to enable `uv run pytest -m "not slow"` for quick passes.
- Exercise timing-sensitive work inside `tests/performance/` and update golden payloads in `schemas/` whenever behaviour changes.
- Verify coverage with `uv run pytest --cov=src --cov-report=term-missing` and explain intentional gaps in your PR.

## Commit & Pull Request Guidelines
- Follow the Conventional Commit style (`feat:`, `fix:`, `docs:`, `chore:`); keep subjects under 72 characters and describe behavioural changes.
- Group related code, tests, and schema adjustments in a single commit to preserve replay reproducibility.
- PRs need a concise summary, linked issues (e.g., `Fixes #123`), and command output from linting and `uv run pytest`.
- Highlight breaking APIs, new configuration, or adapter-facing updates with screenshots or logs, and call out any follow-up work explicitly.

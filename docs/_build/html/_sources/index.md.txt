# gunn Documentation

`gunn` is an event-driven multi-agent orchestration core. This site provides a comprehensive overview of the product, technical details, integration patterns, and API reference.

## Getting Started

- For repository setup, refer to `README.md` and `CONTRIBUTING.md`.
- Navigate through the documentation using the menu below or the table of contents on the right side of each page.

## Build and Deploy

To publish to GitHub Pages, run the following commands from the root directory and configure Pages to serve from `docs/_build/html`:

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

```{toctree}
:caption: Guides
:maxdepth: 2
:hidden:

product
structure
tech
integration-patterns
web_adapter
errors
build-and-deploy
api/index
```

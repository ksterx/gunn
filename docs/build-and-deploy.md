# Documentation Build and Deployment

## Preparing Dependencies

Documentation dependencies are managed in the `docs` group. For initial setup, run:

```bash
uv sync --group docs
```

## Local Build

To generate HTML documentation, run the following from the project root:

```bash
uv run sphinx-build -b html docs docs/_build/html
```

You can also use the Makefile shortcut:

```bash
cd docs
make html
```

Build output is written to `docs/_build/html`, so you can open `index.html` in a browser to preview.

## Link Validation

To validate external links, use the `linkcheck` builder:

```bash
cd docs
make linkcheck
```

## Deploying to GitHub Pages

The repository includes a `Publish Docs` workflow (`.github/workflows/docs-pages.yml`) that builds Sphinx documentation and deploys it to GitHub Pages (`gh-pages` branch) on every push to `main`.

1. On first setup, go to **Settings → Pages** in GitHub and change the source to "GitHub Actions".
2. The workflow runs `uv sync --group docs` → `uv run sphinx-apidoc` → `uv run sphinx-build -b html` in sequence, then artifacts `docs/_build/html` and deploys to Pages.
3. A `.nojekyll` file is automatically generated and uploaded, ensuring static assets (`_static`, etc.) are served correctly. No additional configuration is required.

To manually verify the publishing workflow, run the same command sequence locally and use `docs/_build/html` content as-is for deployment—it will match the GitHub Pages output.

## Tips for Updating API Reference

- Use `sphinx-apidoc -o docs/api src/gunn` to auto-generate stubs.
- Convert generated files to MyST Markdown format as needed and add them to the toctree in `api/index.md`.
- When publishing new modules, ensure docstrings and type hints are complete to improve documentation readability.

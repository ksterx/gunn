# API Reference

Entry point for automatically generating `gunn`'s public API documentation using Sphinx's autodoc. Add pages for each submodule as needed.

```{toctree}
:maxdepth: 2
:hidden:

orchestrator
facades
adapters
gunn
```

## Auto-generation Workflow

1. Run `sphinx-apidoc -o docs/api src/gunn` from the project root.
2. Format the generated `*.rst`/`*.md` files into MyST format and add them to the toctree above.
3. Run `make html` to verify that intro pages and sample code render as expected.

> **Note**: Until auto-generation is performed, toctree placeholders (such as `orchestrator`) can remain empty and may be removed if unnecessary.

from __future__ import annotations

import os
import sys
from datetime import datetime
from importlib import metadata

# -- Path setup -----------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# -- Project information --------------------------------------------------------

project = "gunn"
author = "gunn contributors"
current_year = datetime.now().year

try:
    release = metadata.version("gunn")
except metadata.PackageNotFoundError:  # pragma: no cover
    release = "0.1.0"

# -- General configuration ------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "deflist",
    "fieldlist",
    "colon_fence",
    "linkify",
]

myst_heading_anchors = 3
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

language = "ja"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Options for HTML output ----------------------------------------------------

html_theme = "furo"
html_title = "gunn Documentation"
html_static_path = ["_static"]
html_show_sourcelink = True
html_last_updated_fmt = "%Y-%m-%d"

html_theme_options = {
    "announcement": "GitHub Pages で公開する最新版ドキュメントです。",
    "navigation_with_keys": True,
}

html_baseurl = "https://ksterx.github.io/gunn/"

copyright = f"{current_year}, {author}"

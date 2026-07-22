import os
from datetime import datetime

project = "QURI SDK"
author = "QunaSys"
copyright = f"{datetime.now():%Y}, QunaSys"

extensions = [
    "myst_nb",
    "sphinx_external_toc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "notfound.extension",
]

source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

external_toc_path = "_toc.yml"
external_toc_exclude_missing = False

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_image",
    "linkify",
    "substitution",
    "tasklist",
    "attrs_inline",
]
myst_heading_anchors = 4

nb_execution_mode = "off"
nb_merge_streams = True

html_theme = "furo"
html_title = "QURI SDK"
html_static_path = ["_static"]
html_css_files = ["landing-cards.css"]
html_favicon = "_static/favicon.ico"
# The quri-parts API reference is built separately (autodoc) and staged into
# _api_staged/api/ by CI; serve it verbatim under /api/. Absent on local builds,
# where the `has_api_ref` tag stays unset so reference.md shows a note instead of
# redirecting to a missing page.
_conf_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_conf_dir, "_api_staged")):
    html_extra_path = ["_api_staged"]
    tags.add("has_api_ref")  # noqa: F821  (injected by Sphinx into conf namespace)
else:
    html_extra_path = []
html_theme_options = {
    "light_logo": "logo.png",
    "dark_logo": "logo-dark.png",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/QunaSys/quri-sdk/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

language = "en"
locale_dirs = ["locale/"]
gettext_compact = False
gettext_uuid = True

intersphinx_mapping = {
    "quri-parts": ("https://quri-sdk.qunasys.com/api/", None),
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
intersphinx_disabled_reftypes = ["std:doc"]

exclude_patterns = [
    "**/.ipynb_checkpoints",
    # Per-version release notes are content sources, inlined into
    # release-notes/index.md; they are not standalone build pages.
    "release-notes/[0-9]*.md",
]

suppress_warnings = ["mystnb.unknown_mime_type"]

# Netlify serves the site at the root, so 404.html must reference assets from / .
notfound_urls_prefix = "/"

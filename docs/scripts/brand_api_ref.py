"""Stitch the separately-built QURI Parts API reference into the main docs.

The API reference is a standalone Furo/Sphinx build served under ``/api/``.
On its own it has no logo, no favicon, and no link back to the main site, so
crossing into it feels like leaving QURI SDK. Rather than patch the API build,
we post-process the *staged* HTML (after it is copied
into ``source/_api_staged/api/``) to graft on the shared identity:

* a "back to QURI SDK docs" link at the top of the sidebar (fixes the "trapped
  in a foreign site" feeling), and
* the main site's logo + favicon, using Furo's own logo markup so its built-in
  light/dark handling applies.

Assets are referenced root-relative from ``/_static/`` (the main build copies
``source/_static`` there and Netlify serves everything at the domain root), so
nothing needs to be copied into the API tree.

Idempotent: pages already carrying the marker are skipped, so re-running after
an incremental build is safe.

Usage: ``python scripts/brand_api_ref.py [STAGED_API_DIR]``
(default ``source/_api_staged/api``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER = "quri-brand-nav"

_BACK_LINK = '<a class="sidebar-back" href="/">← QURI SDK Docs</a>'

_LOGO = (
    '<div class="sidebar-logo-container">'
    '<img class="sidebar-logo only-light" src="/_static/logo.png" alt="QURI SDK"/>'
    '<img class="sidebar-logo only-dark" src="/_static/logo-dark.png" alt="QURI SDK"/>'
    "</div>"
)

_HEAD = (
    '<link rel="icon" href="/_static/favicon.ico"/>'
    f'<style id="{MARKER}">'
    ".sidebar-brand-text{display:none}"
    ".sidebar-back{display:block;padding:.375rem 0;margin-bottom:.75rem;"
    "font-size:var(--font-size--small);color:var(--color-foreground-muted);"
    "text-decoration:none;border-bottom:1px solid var(--color-background-border)}"
    ".sidebar-back:hover{color:var(--color-brand-content)}"
    "</style>"
)

# `<div class="sidebar-sticky">` is immediately followed by the brand anchor;
# insert the back link between them so it sits at the very top of the sidebar.
_STICKY_RE = re.compile(r'(<div class="sidebar-sticky">)(<a class="sidebar-brand")')

# The brand anchor's href varies with page depth (#, index.html, ../index.html);
# match the whole opening tag and inject the logo as its first child.
_BRAND_OPEN_RE = re.compile(r'(<a class="sidebar-brand"[^>]*>)')


def brand_page(html: str) -> str | None:
    """Return branded HTML, or None if the page can't/needn't be branded."""
    if MARKER in html:
        return None
    if "</head>" not in html or "sidebar-brand" not in html:
        return None

    html = html.replace("</head>", _HEAD + "</head>", 1)
    html = _STICKY_RE.sub(r"\1" + _BACK_LINK + r"\2", html, count=1)
    html = _BRAND_OPEN_RE.sub(r"\1" + _LOGO, html, count=1)
    return html


def main(staged_dir: Path) -> int:
    if not staged_dir.is_dir():
        print(f"brand_api_ref: {staged_dir} not found; skipping", file=sys.stderr)
        return 0

    branded = 0
    for page in staged_dir.rglob("*.html"):
        result = brand_page(page.read_text(encoding="utf-8"))
        if result is not None:
            page.write_text(result, encoding="utf-8")
            branded += 1
    print(f"brand_api_ref: branded {branded} page(s) under {staged_dir}")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("source/_api_staged/api")
    raise SystemExit(main(target))

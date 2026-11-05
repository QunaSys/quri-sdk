"""Sync GitHub releases into the documentation's release notes.

Run via CI cron. Opens a PR with new release notes for any versions not yet in
docs/reference/release-notes/. Existing per-version files are never overwritten.

Per-version files (docs/reference/release-notes/<slug>.md) are the content source;
they are excluded from the Sphinx build. The built page is a single index.md
with one heading per version (newest first), so the page-local TOC can jump
straight to any release. The _toc.yml entry for that page is static and is not
touched here.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "QunaSys/quri-sdk"
DOCS_DIR = Path(__file__).resolve().parents[1]
NOTES_DIR = DOCS_DIR / "source/docs/reference/release-notes"
INDEX_PATH = NOTES_DIR / "index.md"
TOC_PATH = DOCS_DIR / "source/_toc.yml"

_HEADING = re.compile(r"^(#{1,6})(\s+)(.*)$")
_FENCE = re.compile(r"^\s*(```+|~~~+)")


def slug_for(tag: str) -> str:
    """v0.20.0 -> 0-20-0"""
    return tag.lstrip("v").replace(".", "-")


def version_key(slug: str) -> list[tuple[int, object]]:
    """Sort key for a slug like 0-12-0 or 0-12-0-post1 (numeric, suffixes last)."""
    return [(0, int(tok)) if tok.isdigit() else (1, tok) for tok in slug.split("-")]


def fetch_releases() -> list[dict]:
    out = subprocess.check_output(
        ["gh", "release", "list", "--repo", REPO, "--limit", "100",
         "--json", "tagName,name,publishedAt,isPrerelease,isDraft"]
    )
    return json.loads(out)


def fetch_body(tag: str) -> str:
    out = subprocess.check_output(
        ["gh", "release", "view", tag, "--repo", REPO, "--json", "body"]
    )
    return json.loads(out).get("body", "")


def write_note(rel: dict) -> Path | None:
    slug = slug_for(rel["tagName"])
    path = NOTES_DIR / f"{slug}.md"
    if path.exists():
        return None
    title = rel.get("name") or rel["tagName"].lstrip("v")
    body = fetch_body(rel["tagName"]).strip()
    path.write_text(f"# {title}\n\n{body}\n")
    return path


def parse_note(slug: str) -> tuple[str, str]:
    """Return (title, body) for a per-version file, stripping its leading H1."""
    lines = (NOTES_DIR / f"{slug}.md").read_text().splitlines()
    title = slug
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip()
            start = i + 1
            break
    return title, "\n".join(lines[start:]).strip()


def _map_outside_code(body: str, fn) -> str:
    """Apply fn to each line that is not inside a fenced code block."""
    out = []
    fence = None
    for line in body.splitlines():
        m = _FENCE.match(line)
        if m:
            token = m.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            out.append(line)
        else:
            out.append(line if fence else fn(line))
    return "\n".join(out)


def headings_to_bold(body: str) -> str:
    """Turn headings into bold text so collapsed entries add no page-TOC noise."""
    def to_bold(line: str) -> str:
        m = _HEADING.match(line)
        return f"**{m.group(3).strip()}**" if m else line

    return _map_outside_code(body, to_bold)


def build_index(ordered_slugs: list[str]) -> None:
    """Write the single Release Notes page: one heading per version.

    Version headings are the only headings (bodies are bolded), so the
    page-local TOC lists exactly the versions and jumps to any of them.
    """
    if not ordered_slugs:
        return

    parts = ["# Release Notes", ""]
    for slug in ordered_slugs:
        t, b = parse_note(slug)
        parts += [f"## {t}", "", headings_to_bold(b), ""]

    INDEX_PATH.write_text("\n".join(parts).rstrip() + "\n")


def ordered_slugs_from_releases(releases: list[dict]) -> list[str]:
    """Released slugs newest-first, then any local-only files as orphans."""
    releases.sort(key=lambda r: r["publishedAt"], reverse=True)
    ordered = [
        slug_for(r["tagName"])
        for r in releases
        if not r.get("isPrerelease") and not r.get("isDraft")
    ]
    known = set(ordered)
    orphans = sorted(
        (p.stem for p in NOTES_DIR.glob("*.md") if p.stem not in known and p.name != "index.md"),
        key=version_key,
        reverse=True,
    )
    return ordered + orphans


def main() -> int:
    if not NOTES_DIR.is_dir() or not TOC_PATH.exists():
        print("docs source is missing (docs/reference/release-notes/ and _toc.yml expected)")
        return 1

    releases = fetch_releases()

    new_files = []
    for rel in releases:
        if rel.get("isPrerelease") or rel.get("isDraft"):
            continue
        path = write_note(rel)
        if path:
            new_files.append(path)

    for p in new_files:
        print(f"created {p}")
    if not new_files:
        print("no new release files")

    ordered = ordered_slugs_from_releases(releases)
    build_index(ordered)
    print(f"wrote {INDEX_PATH} ({len(ordered)} releases, latest in full)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

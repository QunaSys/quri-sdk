"""Alphabetize the module list in the QURI Parts API reference toctree.

The sidebar order of the API reference comes from the ``.. toctree::`` in
``quri-parts/docs/reference.rst``, which is hand-ordered (circuit, core,
backend, algo, ...). Rather than reorder the source file, sort the toctree
entries alphabetically at build time before the API build.

Entries look like ``    quri_parts/core/quri_parts.backend``; they are sorted by
the module name after the final dot (``backend``), so ``quri_parts.core`` and
``quri_parts.backend`` sort independently. Everything else in the file (title,
toctree options, blank lines) is left untouched, so it is safe to run after CI's
``sed`` step and is idempotent.

Usage: ``python scripts/sort_api_toctree.py path/to/reference.rst``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ENTRY_RE = re.compile(r"^(\s+)(quri_parts/\S*quri_parts\.(\w+))\s*$")


def sort_toctree(text: str) -> str:
    lines = text.splitlines(keepends=True)

    entries: list[tuple[int, str, str]] = []  # (line index, module name, indent)
    for i, line in enumerate(lines):
        m = _ENTRY_RE.match(line)
        if m:
            entries.append((i, m.group(3), m.group(1)))

    if len(entries) < 2:
        return text

    positions = [i for i, _, _ in entries]
    ordered = sorted(entries, key=lambda e: e[1].lower())

    # Snapshot the rewritten entries before touching `lines`; positions and
    # source indices overlap, so mutating in the same pass would corrupt reads.
    rewritten = []
    for src_idx, _, indent in ordered:
        src = lines[src_idx]
        path = _ENTRY_RE.match(src).group(2)
        newline = "\n" if src.endswith("\n") else ""
        rewritten.append(f"{indent}{path}{newline}")

    for pos, line in zip(positions, rewritten):
        lines[pos] = line

    return "".join(lines)


def main(path: Path) -> int:
    if not path.is_file():
        print(f"sort_api_toctree: {path} not found; skipping", file=sys.stderr)
        return 0
    original = path.read_text(encoding="utf-8")
    sorted_text = sort_toctree(original)
    if sorted_text != original:
        path.write_text(sorted_text, encoding="utf-8")
        print(f"sort_api_toctree: alphabetized toctree in {path}")
    else:
        print(f"sort_api_toctree: {path} already sorted")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: sort_api_toctree.py path/to/reference.rst", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))

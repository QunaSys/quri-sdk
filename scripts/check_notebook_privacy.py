#!/usr/bin/env python3
"""Reject notebooks that contain private user home-directory paths."""

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


PRIVATE_HOME_PATH = re.compile(
    r"/(?:Users|home)/[^/\\\s\"']+|(?i:[a-z]:\\users\\[^/\\\s\"']+)"
)


def _strings(value: Any, location: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield location, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{location}[{index}]")
    elif isinstance(value, dict):
        for index, (key, item) in enumerate(value.items()):
            key_location = f"{location}.<key {index}>"
            yield key_location, key
            safe_key = "<private-key>" if PRIVATE_HOME_PATH.search(key) else key
            yield from _strings(item, f"{location}.{safe_key}")


def _tracked_notebooks() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.ipynb"],
        check=True,
        capture_output=True,
    )
    return [Path(path) for path in result.stdout.decode().split("\0") if path]


def _private_path_locations(notebook: Path) -> list[str]:
    with notebook.open(encoding="utf-8") as file:
        content = json.load(file)
    return [
        location
        for location, value in _strings(content)
        if PRIVATE_HOME_PATH.search(value)
    ]


def main() -> int:
    """Check the given notebooks, or every tracked notebook, for private paths."""
    parser = argparse.ArgumentParser(
        description="Reject private user home-directory paths in Jupyter notebooks."
    )
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument(
        "--all",
        action="store_true",
        help="check every tracked Jupyter notebook",
    )
    args = parser.parse_args()

    if args.all and args.notebooks:
        parser.error("--all cannot be combined with notebook paths")
    if not args.all and not args.notebooks:
        parser.error("provide notebook paths or use --all")

    notebooks = _tracked_notebooks() if args.all else args.notebooks
    failed = False
    for notebook in notebooks:
        try:
            locations = _private_path_locations(notebook)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            print(f"{notebook}: could not inspect notebook: {error}", file=sys.stderr)
            failed = True
            continue

        for location in locations:
            print(
                f"{notebook}: private home-directory path at {location}",
                file=sys.stderr,
            )
            failed = True

    if failed:
        print(
            "Remove or replace private paths before committing notebook output.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

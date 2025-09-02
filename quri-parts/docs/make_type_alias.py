import argparse
import json
import os
from typing import Callable

from typing_extensions import TypeAlias

DIR = r"../packages/"
TYPE_ALIAS_RECORDER = {}
ALIAS_JSON = "qp_type_aliases.json"

Action: TypeAlias = Callable[[str], None]


def iterate_files(dir: str, action: Action) -> None:
    for root, _, files in os.walk(dir):
        if ("test" in root) or ("__pycache__" in root):
            continue
        for name in files:
            filepath = root + os.sep + name
            if filepath.endswith(".py"):
                action(filepath)


def iterate_line_to_find_type_alias(file_name: str) -> None:
    with open(file_name, "r") as f:
        module_name = get_module_name(file_name)
        for line in f:
            if "TypeAlias" in line and "import" not in line:
                alias = line.split(":")[0]
                TYPE_ALIAS_RECORDER[alias] = module_name + "." + alias


def get_module_name(file_name: str) -> str:
    split_name = file_name.split("/")
    if split_name[-1] == "__init__.py":
        split_name = split_name[:-1]
    else:
        split_name[-1] = split_name[-1].split(".")[0]
    return ".".join(split_name[4:])


def insert_future_annotation(file_name: str) -> None:
    """Insert "from __future__ import annotations"."""
    lines: list[str] = []
    import_text = "from __future__ import annotations   # isort: skip"

    with open(file_name, "r") as f:
        for line in f:
            lines.append(line)

    if import_text in [line.strip() for line in lines]:
        return

    lines.insert(0, import_text + "\n")

    new_line_text = "".join(lines)
    with open(file_name, "w") as f:
        f.write(new_line_text)


def remove_future_annotation(file_name: str) -> None:
    """Remove "from __future__ import annotations"."""
    lines = []
    import_text = "from __future__ import annotations   # isort: skip\n"

    with open(file_name, "r") as f:
        for line in f:
            if import_text in line:
                continue
            lines.append(line)

    new_line_text = "".join(lines)
    with open(file_name, "w") as f:
        f.write(new_line_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    args = parser.parse_args()
    mode = args.mode

    make_mode = "m"
    remove_mode = "r"

    if mode == make_mode:
        iterate_files(DIR, iterate_line_to_find_type_alias)
        with open(ALIAS_JSON, "w") as f:
            json.dump(TYPE_ALIAS_RECORDER, fp=f)

        iterate_files(DIR, insert_future_annotation)

    elif mode == remove_mode:
        iterate_files(DIR, remove_future_annotation)
        os.remove(ALIAS_JSON)

    else:
        pass

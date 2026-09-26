#!/usr/bin/env python3
"""Print metadata entry IDs touched between two Git revisions.

The LLM metadata workflow must validate both newly added entries and existing
entries updated through an edit issue.  A conventional diff only shows ``id``
lines for additions, so this utility maps each changed diff hunk back to the
closest preceding YAML entry ID in the revised file.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


DATA_FILES = ("data/models.yaml", "data/datasets.yaml", "data/tools.yaml")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
ENTRY_ID_RE = re.compile(r"^\s*-\s+id:\s*(?:['\"]([^'\"]+)['\"]|(\S+))\s*$")


def entry_id_at_line(lines: list[str], line_number: int) -> str | None:
    """Return the entry ID owning a one-based line number in a YAML list."""
    for line in reversed(lines[: max(0, min(line_number, len(lines)))]):
        match = ENTRY_ID_RE.match(line)
        if match:
            return match.group(1) or match.group(2)
    return None


def changed_entry_ids_from_diff(diff: str, head_files: dict[str, str]) -> list[str]:
    """Map zero-context diff hunks to entry IDs in their revised YAML files."""
    changed: set[str] = set()
    current_path: str | None = None

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_path = line.removeprefix("+++ b/")
            continue

        match = HUNK_RE.match(line)
        if not match or current_path not in head_files:
            continue

        start = int(match.group(1))
        count = int(match.group(2) or "1")
        # A deletion has no line in the new file; map it to the line just
        # before the deleted range, which remains in the same YAML entry.
        first_line = start if count else max(1, start - 1)
        last_line = first_line + max(count, 1) - 1
        file_lines = head_files[current_path].splitlines()
        for line_number in range(first_line, last_line + 1):
            entry_id = entry_id_at_line(file_lines, line_number)
            if entry_id:
                changed.add(entry_id)

    return sorted(changed)


def git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, capture_output=True
    ).stdout


def changed_entry_ids(root: Path, base: str, head: str) -> list[str]:
    diff = git_output(root, "diff", "--unified=0", base, head, "--", *DATA_FILES)
    head_files: dict[str, str] = {}
    for path in DATA_FILES:
        try:
            head_files[path] = git_output(root, "show", f"{head}:{path}")
        except subprocess.CalledProcessError:
            continue
    return changed_entry_ids_from_diff(diff, head_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print IDs of changed metadata entries")
    parser.add_argument("--base", required=True, help="Base Git revision")
    parser.add_argument("--head", required=True, help="Revised Git revision")
    args = parser.parse_args()
    print(",".join(changed_entry_ids(Path.cwd(), args.base, args.head)))


if __name__ == "__main__":
    main()

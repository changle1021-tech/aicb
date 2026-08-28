#!/usr/bin/env python3
"""Remove expert-parallel collectives from single-rank EP workloads."""

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence


_EP_SIZE_PATTERN = re.compile(r"(?:^|\s)ep:\s*(\d+)(?:\s|$)")
_COMM_TYPE_COLUMNS = (3, 6, 9)


def normalize_ep1_workload(path: Path) -> int:
    """Replace EP collectives with no-ops when the workload declares EP=1.

    Returns the number of communication entries changed.
    """

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        raise ValueError(f"empty workload file: {path}")

    match = _EP_SIZE_PATTERN.search(lines[0])
    if match is None:
        raise ValueError(f"workload header does not declare an EP size: {path}")
    if int(match.group(1)) != 1:
        return 0

    changed = 0
    for line_index in range(2, len(lines)):
        content = lines[line_index].rstrip("\r\n")
        line_ending = lines[line_index][len(content) :]
        fields = content.split("\t")

        for comm_type_column in _COMM_TYPE_COLUMNS:
            if (
                comm_type_column + 2 < len(fields)
                and fields[comm_type_column] == "ALLTOALL_EP"
            ):
                fields[comm_type_column] = "NONE"
                fields[comm_type_column + 1] = "0"
                fields[comm_type_column + 2] = "0"
                changed += 1

        lines[line_index] = "\t".join(fields) + line_ending

    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove ALLTOALL_EP entries from EP=1 SimAI workloads."
    )
    parser.add_argument("workloads", nargs="+", type=Path)
    args = parser.parse_args(argv)

    for path in args.workloads:
        changed = normalize_ep1_workload(path)
        if changed:
            print(f"Normalized {changed} EP communication entries in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

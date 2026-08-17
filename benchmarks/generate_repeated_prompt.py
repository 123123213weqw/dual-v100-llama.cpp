#!/usr/bin/env python3
"""Generate a deterministic, local long-context capacity-test prompt."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=255_000)
    parser.add_argument("--unit", default=" test")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")

    content = args.unit * args.repeats
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    print(f"output={args.output} bytes={len(content.encode('utf-8'))} repeats={args.repeats} sha256={digest}")


if __name__ == "__main__":
    main()

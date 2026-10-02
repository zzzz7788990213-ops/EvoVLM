"""Print a secret-safe, machine-readable evolution readiness record."""

from __future__ import annotations

import json

from .core import build_status


def main() -> int:
    print(json.dumps(build_status(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

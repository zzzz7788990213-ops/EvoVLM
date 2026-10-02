"""Packaged entry point for the canonical fail-closed submission checker."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve(strict=True).parent.parent
sys.path.insert(0, str(PROJECT))

from submission_checker import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

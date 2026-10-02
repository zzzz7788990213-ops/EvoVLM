#!/usr/bin/env python3
"""Compatibility entrypoint for immutable historical v3 CPU-only checks."""

from evaluation.verify_historical_v3 import main


if __name__ == "__main__":
    raise SystemExit(main())

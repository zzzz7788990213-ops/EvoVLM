"""Shared deterministic metric reductions for writer and verifier."""

from __future__ import annotations

import math
import statistics
from typing import Iterable


def percentile_type7(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def latency_statistics(values: Iterable[float]) -> dict[str, float]:
    sequence = list(values)
    if not sequence:
        raise ValueError("latency statistics require measurements")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in sequence
    ):
        raise ValueError("latencies must be finite positive numbers")
    measured = math.fsum(sequence)
    p50 = percentile_type7(sequence, 0.50)
    return {
        "measured_call_seconds": measured,
        "mean": measured / len(sequence),
        "median": statistics.median(sequence),
        "p50": p50,
        "p95": percentile_type7(sequence, 0.95),
        "min": min(sequence),
        "max": max(sequence),
        "throughput_queries_per_second": len(sequence) / measured,
    }

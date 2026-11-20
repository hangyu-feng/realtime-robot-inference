"""The one summary of raw samples that every benchmark reports.

A shared summary keeps the numbers of different benchmarks comparable: the same
statistics, computed the same way, under the same keys.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Summarize raw samples. Percentiles interpolate linearly between neighbouring samples."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("cannot summarize an empty set of samples")
    p50, p95, p99 = np.percentile(samples, (50, 95, 99))
    return {
        "count": samples.size,
        "mean": float(samples.mean()),
        "min": float(samples.min()),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(samples.max()),
    }

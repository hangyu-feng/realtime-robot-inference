"""The fixed log-scale histogram of durations that /metrics and robot telemetry carry.

Bucket i covers [RANGE_MIN_MS * 2**(i / 8), RANGE_MIN_MS * 2**((i + 1) / 8)): eight buckets
per octave, so a bucket is about 9% wide. Values below the range, including zero and the
negative values that a clock difference can produce, go to bucket 0, and values above it go to
the last bucket. The layout is fixed, so a histogram never grows and two snapshots subtract
bucket by bucket.
"""

from __future__ import annotations

import bisect
import itertools
import math

# The layout is part of the /metrics contract: a reader turns a bucket index back into
# durations with these values.
RANGE_MIN_MS = 0.01
RANGE_MAX_MS = 60_000.0
BUCKETS_PER_OCTAVE = 8
# The fewest buckets that reach RANGE_MAX_MS: 181, so the last upper edge is about 64 685 ms.
BUCKET_COUNT = math.ceil(BUCKETS_PER_OCTAVE * math.log2(RANGE_MAX_MS / RANGE_MIN_MS))

# Edge k is the lower edge of bucket k and the upper edge of bucket k - 1.
_EDGES_MS = tuple(
    RANGE_MIN_MS * 2 ** (boundary / BUCKETS_PER_OCTAVE) for boundary in range(BUCKET_COUNT + 1)
)
# A bucket key must be the index exactly as to_dict writes it: int() would also accept a sign,
# leading zeros, spaces, underscores and non-ASCII digits.
_BUCKET_INDEX_BY_KEY = {str(index): index for index in range(BUCKET_COUNT)}
_FIELDS = frozenset(("count", "sum", "min", "max", "buckets"))
# Counts stay within a signed 64-bit integer, like every counter in a robot's report, so that a
# reader of /metrics can hold them in one.
_MAX_COUNT = 2**63 - 1


class Histogram:
    """A cumulative histogram of durations in milliseconds.

    count, sum_ms, min_ms and max_ms come from the recorded values themselves, not from the
    buckets; min_ms and max_ms are None while the histogram is empty. Two histograms are equal
    when these four and every bucket are equal. Only __eq__ is defined, so Python leaves the
    class unhashable, which suits a mutable object.
    """

    __slots__ = ("_buckets", "_count", "_max_ms", "_min_ms", "_sum_ms")

    def __init__(self) -> None:
        # Preallocated: record only updates numbers in place and never grows a structure.
        self._buckets = [0] * BUCKET_COUNT
        self._count = 0
        self._sum_ms = 0.0
        self._min_ms: float | None = None
        self._max_ms: float | None = None

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum_ms(self) -> float:
        return self._sum_ms

    @property
    def min_ms(self) -> float | None:
        return self._min_ms

    @property
    def max_ms(self) -> float | None:
        return self._max_ms

    def record(self, value_ms: float) -> None:
        """Count one duration; ValueError for a value that would make the sum NaN or infinite.

        That refuses NaN, both infinities, and a finite value that would overflow the sum.
        """
        # The float annotation also admits an int or a bool, which to_dict would otherwise write
        # as a JSON int or true where the contract has a float.
        try:
            value_ms = float(value_ms)
        except OverflowError as error:
            raise ValueError(
                "recording an int beyond the float range would make the sum non-finite"
            ) from error
        sum_ms = self._sum_ms + value_ms
        # A histogram that a robot reports is republished in /metrics as strict JSON, which has
        # no NaN or infinity.
        if not math.isfinite(sum_ms):
            raise ValueError(f"recording {value_ms} ms would make the sum non-finite")
        if value_ms < RANGE_MIN_MS:
            index = 0
        else:
            # Clamping before the floor also sends the infinite ratio of a value near the largest
            # float to the last bucket.
            index = math.floor(
                min(BUCKET_COUNT - 1, BUCKETS_PER_OCTAVE * math.log2(value_ms / RANGE_MIN_MS))
            )
        self._buckets[index] += 1
        self._count += 1
        self._sum_ms = sum_ms
        if self._min_ms is None or value_ms < self._min_ms:
            self._min_ms = value_ms
        if self._max_ms is None or value_ms > self._max_ms:
            self._max_ms = value_ms

    def quantile_ms(self, q: float) -> float:
        """Return the upper edge of the bucket that holds rank ceil(q * count), rank 1 for q = 0.

        For values inside the range the result is at or above the true quantile and at most one
        bucket width, a factor of 2**(1/8), above it. A quantile that falls in the last bucket may
        be below the true value, because a value above the range is reported as the last bucket's
        upper edge. ValueError when the histogram is empty or q is outside [0, 1].
        """
        if self._count == 0:
            raise ValueError("an empty histogram has no quantile")
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be in [0, 1], got {q}")
        # q * count is rounded to a float, which for a count above 2**53 can exceed the count.
        rank = min(self._count, max(1, math.ceil(q * self._count)))
        index = bisect.bisect_left(list(itertools.accumulate(self._buckets)), rank)
        return _EDGES_MS[index + 1]

    def to_dict(self) -> dict[str, object]:
        """Return the /metrics form, which lists only the non-empty buckets."""
        # These keys are the /metrics contract. They carry no unit because the field that holds
        # the histogram has it in its name.
        return {
            "count": self._count,
            "sum": self._sum_ms,
            "min": self._min_ms,
            "max": self._max_ms,
            "buckets": {
                str(index): bucket_count
                for index, bucket_count in enumerate(self._buckets)
                if bucket_count
            },
        }

    @classmethod
    def from_dict(cls, data: object) -> Histogram:
        """Read the to_dict form from an untrusted peer or a file; ValueError for anything else."""
        # Every rule is checked here because json.loads returns nan and inf for NaN, Infinity and
        # 1e999, and ints of up to 4300 digits, and whatever this accepts is written back out in
        # /metrics as strict JSON. No message quotes the input: formatting an int of more than
        # 4300 digits raises, and a nested value can be of any size.
        if not isinstance(data, dict) or data.keys() != _FIELDS:
            raise ValueError(
                "a histogram is an object with exactly the keys count, sum, min, max and buckets"
            )
        count = data["count"]
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= _MAX_COUNT:
            raise ValueError("histogram count is not an int in [0, 2**63 - 1]")
        histogram = cls()
        histogram._buckets = _read_buckets(data["buckets"])
        if sum(histogram._buckets) != count:
            raise ValueError("histogram bucket counts do not add up to its count")
        histogram._count = count
        histogram._sum_ms = _finite_float(data["sum"], "sum")
        if count == 0:
            if data["min"] is not None or data["max"] is not None:
                raise ValueError("an empty histogram has a null min and max")
            return histogram
        histogram._min_ms = _finite_float(data["min"], "min")
        histogram._max_ms = _finite_float(data["max"], "max")
        if histogram._min_ms > histogram._max_ms:
            raise ValueError("histogram min is above its max")
        return histogram

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Histogram):
            return NotImplemented
        return (
            self._count == other._count
            and self._sum_ms == other._sum_ms
            and self._min_ms == other._min_ms
            and self._max_ms == other._max_ms
            and self._buckets == other._buckets
        )


def _read_buckets(table: object) -> list[int]:
    """Return the bucket list of a to_dict bucket table; ValueError for anything else."""
    if not isinstance(table, dict):
        raise ValueError("histogram buckets is not an object")
    buckets = [0] * BUCKET_COUNT
    for key, bucket_count in table.items():
        index = _BUCKET_INDEX_BY_KEY.get(key)
        if index is None:
            raise ValueError(
                f"histogram bucket key is not an index in [0, {BUCKET_COUNT}) "
                "without sign or leading zeros"
            )
        # to_dict lists only the non-empty buckets.
        if isinstance(bucket_count, bool) or not isinstance(bucket_count, int) or bucket_count < 1:
            raise ValueError("histogram bucket count is not an int of at least 1")
        buckets[index] = bucket_count
    return buckets


def _finite_float(value: object, field: str) -> float:
    """Return a JSON number as a finite float; ValueError for anything else."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"histogram {field} is not a number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"histogram {field} is not finite") from error
    if not math.isfinite(number):
        raise ValueError(f"histogram {field} is not finite")
    return number


def difference(later: Histogram, earlier: Histogram) -> Histogram:
    """Return the window between two cumulative snapshots: what later recorded after earlier.

    count and sum_ms are the exact differences. min_ms and max_ms are bounds, not exact values:
    the lower edge of the lowest non-empty bucket and the upper edge of the highest, both None
    when the window is empty. They bound only values between RANGE_MIN_MS and the last bucket's
    upper edge, because bucket 0 also holds smaller values and the last bucket larger ones.
    ValueError when a bucket would go negative: earlier is not a snapshot taken before later.
    """
    window = Histogram()
    window._buckets = [
        later_count - earlier_count
        for later_count, earlier_count in zip(later._buckets, earlier._buckets, strict=True)
    ]
    # In every histogram the count is the sum of the buckets, so a count that would go negative
    # shows as a negative bucket.
    if min(window._buckets) < 0:
        raise ValueError("earlier is not a snapshot taken before later: a bucket would go negative")
    window._count = later._count - earlier._count
    window._sum_ms = later._sum_ms - earlier._sum_ms
    occupied = [index for index, bucket_count in enumerate(window._buckets) if bucket_count]
    if occupied:
        window._min_ms = _EDGES_MS[occupied[0]]
        window._max_ms = _EDGES_MS[occupied[-1] + 1]
    return window

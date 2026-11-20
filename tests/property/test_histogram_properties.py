"""Property tests for the fixed log-scale histogram of durations."""

import contextlib
import json
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from common.histogram import (
    BUCKET_COUNT,
    BUCKETS_PER_OCTAVE,
    RANGE_MAX_MS,
    RANGE_MIN_MS,
    Histogram,
    difference,
)

# Durations inside and outside the range, negative ones included. The bound keeps the sum of any
# list that hypothesis draws finite.
VALUES = st.lists(st.floats(min_value=-1e300, max_value=1e300))
IN_RANGE_VALUES = st.lists(st.floats(min_value=RANGE_MIN_MS, max_value=RANGE_MAX_MS), min_size=1)
# json.loads yields ints of up to 4300 digits, and a Python caller can pass larger ones.
INTEGERS = st.integers() | st.builds(
    lambda digits, sign: sign * 10**digits,
    st.integers(min_value=19, max_value=5_000),
    st.sampled_from((1, -1)),
)
SCALARS = st.none() | st.booleans() | INTEGERS | st.floats() | st.text()
JSON_LIKE = st.recursive(
    SCALARS, lambda children: st.lists(children) | st.dictionaries(st.text(), children)
)
# Valid keys only, so that every non-empty table reaches the bucket-count check, which a table
# drawn by JSON_LIKE almost never does: its keys are arbitrary text.
BUCKET_TABLES = st.dictionaries(
    st.integers(min_value=0, max_value=BUCKET_COUNT - 1).map(str), INTEGERS | JSON_LIKE
)


def recorded(values: list[float]) -> Histogram:
    histogram = Histogram()
    for value_ms in values:
        histogram.record(value_ms)
    return histogram


@given(values=IN_RANGE_VALUES, q=st.floats(min_value=0.0, max_value=1.0))
def test_quantile_is_at_or_above_the_true_quantile_and_within_one_bucket_width(
    values: list[float], q: float
) -> None:
    true_quantile_ms = sorted(values)[max(1, math.ceil(q * len(values))) - 1]

    quantile_ms = recorded(values).quantile_ms(q)

    # The relative 1e-9 allows for rounding where a value lies on a bucket edge.
    assert true_quantile_ms * (1 - 1e-9) <= quantile_ms
    assert quantile_ms <= true_quantile_ms * 2 ** (1 / BUCKETS_PER_OCTAVE) * (1 + 1e-9)


@given(values=VALUES)
def test_count_sum_min_and_max_are_those_of_the_recorded_values(values: list[float]) -> None:
    histogram = recorded(values)

    # Each addition in record order rounds by at most 2**-53 times the sum of the magnitudes added
    # so far, so with mixed signs the tolerance scales with the sum of the magnitudes.
    magnitude_ms = math.fsum(abs(value_ms) for value_ms in values)
    assert (histogram.count, histogram.min_ms, histogram.max_ms) == (
        len(values),
        min(values, default=None),
        max(values, default=None),
    )
    assert histogram.sum_ms == pytest.approx(math.fsum(values), rel=1e-9, abs=1e-9 * magnitude_ms)


@given(values=VALUES)
def test_to_dict_then_from_dict_through_strict_json_gives_an_equal_histogram(
    values: list[float],
) -> None:
    histogram = recorded(values)

    text = json.dumps(histogram.to_dict(), allow_nan=False)

    assert Histogram.from_dict(json.loads(text)) == histogram


@given(data=JSON_LIKE)
def test_from_dict_raises_only_value_error_for_any_json_like_value(data: object) -> None:
    with contextlib.suppress(ValueError):
        Histogram.from_dict(data)


# Scalars are drawn on their own as well, because a recursive draw rarely stops at one, and NaN,
# infinity and ints too large for a float are scalars.
@given(
    values=VALUES,
    field=st.sampled_from(("count", "sum", "min", "max", "buckets")),
    value=SCALARS | JSON_LIKE,
)
def test_from_dict_raises_only_value_error_when_one_field_holds_any_json_like_value(
    values: list[float], field: str, value: object
) -> None:
    data = recorded(values).to_dict()
    data[field] = value

    with contextlib.suppress(ValueError):
        Histogram.from_dict(data)


@given(values=VALUES, table=BUCKET_TABLES)
def test_from_dict_raises_only_value_error_when_the_buckets_hold_any_json_like_counts(
    values: list[float], table: dict[str, object]
) -> None:
    data = recorded(values).to_dict()
    data["buckets"] = table

    with contextlib.suppress(ValueError):
        Histogram.from_dict(data)


@given(first=VALUES, second=VALUES)
def test_difference_of_two_snapshots_has_the_buckets_and_count_recorded_between_them(
    first: list[float], second: list[float]
) -> None:
    window = difference(recorded(first + second), recorded(first))

    expected = recorded(second).to_dict()
    assert (window.count, window.to_dict()["buckets"]) == (expected["count"], expected["buckets"])

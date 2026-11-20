"""Tests for the fixed log-scale histogram of durations."""

import math
import sys

import pytest

from common.histogram import (
    BUCKET_COUNT,
    BUCKETS_PER_OCTAVE,
    RANGE_MAX_MS,
    RANGE_MIN_MS,
    Histogram,
    difference,
)


def edge_ms(boundary: int) -> float:
    """Return the lower edge of bucket `boundary` by the documented layout."""
    return RANGE_MIN_MS * 2 ** (boundary / BUCKETS_PER_OCTAVE)


def recorded(values: list[float]) -> Histogram:
    histogram = Histogram()
    for value_ms in values:
        histogram.record(value_ms)
    return histogram


# The to_dict form of recorded([0.64, 0.64, 1.28]). Values of RANGE_MIN_MS * 2**k divide by
# RANGE_MIN_MS without rounding, so they land exactly at the start of bucket 8 * k, and their
# sums here are exact.
FILLED: dict[str, object] = {
    "count": 3,
    "sum": 2.56,
    "min": 0.64,
    "max": 1.28,
    "buckets": {"48": 2, "56": 1},
}
EMPTY: dict[str, object] = {"count": 0, "sum": 0.0, "min": None, "max": None, "buckets": {}}
# The to_dict form of recorded([1.0]) with every number written as a JSON int.
INTEGRAL_NUMBERS: dict[str, object] = {
    "count": 1,
    "sum": 1,
    "min": 1,
    "max": 1,
    "buckets": {"53": 1},
}
LARGEST_COUNT: dict[str, object] = {
    "count": 2**63 - 1,
    "sum": 0.64 * 2**63,
    "min": 0.64,
    "max": 0.64,
    "buckets": {"48": 2**63 - 1},
}


def test_range_takes_181_buckets() -> None:
    assert BUCKET_COUNT == 181


# Edges other than octave edges are tested from just below and just above: rounding can put a
# value equal to such an edge in either bucket.
@pytest.mark.parametrize(
    ("value_ms", "index"),
    [
        pytest.param(RANGE_MIN_MS, 0, id="range-minimum"),
        pytest.param(0.02, 8, id="octave-edge"),
        pytest.param(41_943.04, 176, id="last-octave-edge"),
        pytest.param(edge_ms(8) * (1 - 1e-9), 7, id="just-below-an-octave-edge"),
        pytest.param(edge_ms(1) * (1 - 1e-9), 0, id="just-below-edge-1"),
        pytest.param(edge_ms(1) * (1 + 1e-9), 1, id="just-above-edge-1"),
        pytest.param(edge_ms(57) * (1 - 1e-9), 56, id="just-below-edge-57"),
        pytest.param(edge_ms(57) * (1 + 1e-9), 57, id="just-above-edge-57"),
        pytest.param(edge_ms(180) * (1 - 1e-9), 179, id="just-below-the-last-bucket"),
        pytest.param(edge_ms(180) * (1 + 1e-9), 180, id="just-inside-the-last-bucket"),
        pytest.param(RANGE_MAX_MS, 180, id="range-maximum"),
        pytest.param(RANGE_MIN_MS * (1 - 1e-9), 0, id="just-below-the-range"),
        pytest.param(0.001, 0, id="below-the-range"),
        pytest.param(0.0, 0, id="zero"),
        pytest.param(-2.5, 0, id="negative"),
        pytest.param(100_000.0, 180, id="above-the-range"),
        pytest.param(sys.float_info.max, 180, id="largest-float"),
    ],
)
def test_value_is_counted_in_the_bucket_that_covers_it(value_ms: float, index: int) -> None:
    histogram = recorded([value_ms])

    assert histogram.to_dict()["buckets"] == {str(index): 1}


@pytest.mark.parametrize(
    ("before", "value_ms"),
    [
        pytest.param([], math.nan, id="nan"),
        pytest.param([], math.inf, id="infinity"),
        pytest.param([], -math.inf, id="negative-infinity"),
        pytest.param([sys.float_info.max], sys.float_info.max, id="sum-overflow"),
        pytest.param([], 10**400, id="int-beyond-the-float-range"),
    ],
)
def test_value_that_would_make_the_sum_non_finite_is_refused_and_changes_nothing(
    before: list[float], value_ms: float
) -> None:
    histogram = recorded(before)

    with pytest.raises(ValueError, match="non-finite"):
        histogram.record(value_ms)

    assert histogram == recorded(before)


@pytest.mark.parametrize(
    ("q", "index"),
    [
        pytest.param(0.0, 0, id="q-0-is-rank-1"),
        pytest.param(0.5, 8, id="median"),
        pytest.param(1.0, 24, id="q-1-is-the-last-rank"),
    ],
)
def test_quantile_is_the_upper_edge_of_the_bucket_that_holds_the_rank(q: float, index: int) -> None:
    # One value at the start of each of the buckets 0, 8, 16 and 24.
    histogram = recorded([0.01, 0.02, 0.04, 0.08])

    assert histogram.quantile_ms(q) == pytest.approx(edge_ms(index + 1))


def test_quantile_of_the_largest_count_is_read_from_its_bucket() -> None:
    histogram = Histogram.from_dict(LARGEST_COUNT)

    assert histogram.quantile_ms(1.0) == pytest.approx(edge_ms(49))


def test_quantile_of_an_empty_histogram_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        Histogram().quantile_ms(0.5)


@pytest.mark.parametrize("q", [-0.01, 1.01, math.nan])
def test_quantile_outside_zero_to_one_is_refused(q: float) -> None:
    with pytest.raises(ValueError, match="q must be in"):
        recorded([1.0]).quantile_ms(q)


def test_empty_histogram_has_no_buckets_and_null_min_and_max() -> None:
    assert Histogram().to_dict() == EMPTY


def test_to_dict_lists_only_the_non_empty_buckets_by_decimal_index() -> None:
    assert recorded([0.64, 0.64, 1.28]).to_dict() == FILLED


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(EMPTY, Histogram(), id="empty"),
        pytest.param(FILLED, recorded([0.64, 0.64, 1.28]), id="filled"),
        pytest.param(INTEGRAL_NUMBERS, recorded([1.0]), id="integral-numbers"),
    ],
)
def test_from_dict_reads_the_to_dict_form(data: dict[str, object], expected: Histogram) -> None:
    assert Histogram.from_dict(data) == expected


# Equal histograms write the same JSON. The types are compared because 1 == 1.0.
@pytest.mark.parametrize(
    "histogram",
    [
        pytest.param(recorded([1]), id="recorded-int"),
        pytest.param(Histogram.from_dict(INTEGRAL_NUMBERS), id="read-integral-numbers"),
    ],
)
def test_to_dict_writes_sum_min_and_max_as_floats(histogram: Histogram) -> None:
    data = histogram.to_dict()

    assert (type(data["sum"]), type(data["min"]), type(data["max"])) == (float, float, float)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        pytest.param([3, 2.56, 0.64, 1.28, {}], "exactly the keys", id="not-an-object"),
        pytest.param(
            {"count": 3, "sum": 2.56, "min": 0.64, "buckets": {"48": 2, "56": 1}},
            "exactly the keys",
            id="missing-key",
        ),
        pytest.param({**FILLED, "mean": 0.85}, "exactly the keys", id="extra-key"),
        pytest.param({**FILLED, "count": True}, "count is not an int", id="bool-count"),
        pytest.param({**FILLED, "count": 3.0}, "count is not an int", id="float-count"),
        pytest.param({**EMPTY, "count": -1}, "count is not an int", id="negative-count"),
        pytest.param(
            {**FILLED, "count": 2**63, "buckets": {"48": 2**63}},
            "count is not an int",
            id="count-above-the-signed-64-bit-range",
        ),
        pytest.param({**FILLED, "buckets": [2, 1]}, "buckets is not an object", id="buckets-list"),
        pytest.param({**FILLED, "buckets": {"007": 3}}, "bucket key", id="leading-zeros"),
        pytest.param({**FILLED, "buckets": {"-1": 3}}, "bucket key", id="negative-key"),
        pytest.param({**FILLED, "buckets": {"181": 3}}, "bucket key", id="key-past-the-last"),
        pytest.param({**FILLED, "buckets": {"x": 3}}, "bucket key", id="key-not-a-number"),
        pytest.param(
            {**FILLED, "buckets": {"48": 2, "56": 1, "57": 0}},
            "bucket count is not",
            id="empty-bucket-listed",
        ),
        pytest.param(
            {**FILLED, "buckets": {"48": 2, "56": True}}, "bucket count is not", id="bool-bucket"
        ),
        # The negative and the float bucket add up to the count, so only the bucket-count rule
        # can refuse them.
        pytest.param(
            {**FILLED, "buckets": {"48": 4, "56": -1}}, "bucket count is not", id="negative-bucket"
        ),
        pytest.param(
            {**FILLED, "buckets": {"48": 2.0, "56": 1}}, "bucket count is not", id="float-bucket"
        ),
        pytest.param(
            {**FILLED, "buckets": {"48": "2", "56": 1}}, "bucket count is not", id="string-bucket"
        ),
        pytest.param(
            {**FILLED, "buckets": {"48": None, "56": 1}}, "bucket count is not", id="null-bucket"
        ),
        pytest.param({**FILLED, "count": 4}, "do not add up", id="counts-not-adding-up"),
        pytest.param({**FILLED, "sum": math.nan}, "sum is not finite", id="nan-sum"),
        pytest.param({**FILLED, "sum": math.inf}, "sum is not finite", id="infinite-sum"),
        pytest.param({**FILLED, "sum": 10**400}, "sum is not finite", id="sum-beyond-floats"),
        pytest.param({**FILLED, "sum": False}, "sum is not a number", id="bool-sum"),
        pytest.param(
            {"count": 1, "sum": 0.64, "min": None, "max": 0.64, "buckets": {"48": 1}},
            "min is not a number",
            id="null-min-with-count-1",
        ),
        pytest.param(
            {"count": 1, "sum": 0.64, "min": 0.64, "max": None, "buckets": {"48": 1}},
            "max is not a number",
            id="null-max-with-count-1",
        ),
        pytest.param({**FILLED, "min": True}, "min is not a number", id="bool-min"),
        pytest.param({**FILLED, "max": math.inf}, "max is not finite", id="infinite-max"),
        pytest.param({**EMPTY, "min": 0.64}, "null min and max", id="min-set-when-empty"),
        pytest.param({**EMPTY, "max": 0.64}, "null min and max", id="max-set-when-empty"),
        pytest.param({**FILLED, "min": 1.5}, "min is above its max", id="min-above-max"),
    ],
)
def test_from_dict_refuses_every_value_outside_the_to_dict_form(data: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Histogram.from_dict(data)


def test_from_dict_accepts_the_largest_count() -> None:
    assert Histogram.from_dict(LARGEST_COUNT).to_dict() == LARGEST_COUNT


def test_difference_holds_the_window_with_bucket_edges_as_min_and_max() -> None:
    window = difference(recorded([0.64, 1.28, 5.12]), recorded([0.64]))

    assert window.to_dict() == {
        "count": 2,
        "sum": pytest.approx(6.4),
        "min": pytest.approx(edge_ms(56)),
        "max": pytest.approx(edge_ms(73)),
        "buckets": {"56": 1, "72": 1},
    }


def test_difference_of_a_snapshot_with_itself_is_empty() -> None:
    histogram = recorded([0.64, 1.28])

    assert difference(histogram, histogram) == Histogram()


@pytest.mark.parametrize(
    ("later_values", "earlier_values"),
    [
        pytest.param([0.64], [1.28], id="same-count-other-bucket"),
        pytest.param([0.64], [0.64, 0.64], id="earlier-holds-more"),
    ],
)
def test_difference_refuses_a_bucket_that_would_go_negative(
    later_values: list[float], earlier_values: list[float]
) -> None:
    with pytest.raises(ValueError, match="not a snapshot taken before later"):
        difference(recorded(later_values), recorded(earlier_values))


def test_histograms_that_recorded_the_same_values_are_equal() -> None:
    assert recorded([0.64, 1.28]) == recorded([1.28, 0.64])


# The count is the sum of the buckets, so it cannot differ alone.
@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"sum": 2.57}, id="sum"),
        pytest.param({"min": 0.65}, id="min"),
        pytest.param({"max": 1.29}, id="max"),
        pytest.param({"buckets": {"48": 1, "56": 2}}, id="buckets"),
        pytest.param({"count": 4, "buckets": {"48": 3, "56": 1}}, id="count-and-buckets"),
    ],
)
def test_histograms_that_differ_in_one_field_are_not_equal(changed: dict[str, object]) -> None:
    assert Histogram.from_dict({**FILLED, **changed}) != Histogram.from_dict(FILLED)


def test_histogram_is_not_equal_to_its_dict_form() -> None:
    assert Histogram() != Histogram().to_dict()


def test_histogram_is_not_hashable() -> None:
    with pytest.raises(TypeError, match="unhashable"):
        hash(Histogram())

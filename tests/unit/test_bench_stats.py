"""Tests for the summary of raw benchmark samples."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bench.stats import summarize

# Whole-number samples keep every sum exact, so the properties hold without a tolerance.
WHOLE_SAMPLES = st.lists(st.integers(min_value=0, max_value=10**9), min_size=1)


def test_summarize_reports_exact_statistics_of_the_samples() -> None:
    assert summarize([5.0, 1.0, 4.0, 2.0, 3.0]) == pytest.approx(
        {"count": 5, "mean": 3.0, "min": 1.0, "p50": 3.0, "p95": 4.8, "p99": 4.96, "max": 5.0}
    )


def test_summarize_of_one_sample_repeats_it_in_every_statistic() -> None:
    assert summarize([12.5]) == {
        "count": 1,
        "mean": 12.5,
        "min": 12.5,
        "p50": 12.5,
        "p95": 12.5,
        "p99": 12.5,
        "max": 12.5,
    }


def test_summarize_rejects_an_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize([])


@given(samples=WHOLE_SAMPLES)
def test_summary_count_min_and_max_are_those_of_the_samples(samples: list[int]) -> None:
    summary = summarize(samples)

    assert (summary["count"], summary["min"], summary["max"]) == (
        len(samples),
        min(samples),
        max(samples),
    )


@given(samples=WHOLE_SAMPLES)
def test_summary_statistics_are_ordered(samples: list[int]) -> None:
    summary = summarize(samples)

    assert summary["min"] <= summary["mean"] <= summary["max"]
    assert summary["min"] <= summary["p50"] <= summary["p95"] <= summary["p99"] <= summary["max"]


@given(samples=WHOLE_SAMPLES)
def test_summary_does_not_depend_on_the_order_of_the_samples(samples: list[int]) -> None:
    assert summarize(sorted(samples)) == summarize(samples)

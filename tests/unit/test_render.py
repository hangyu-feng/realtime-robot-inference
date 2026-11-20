"""Tests for rendering the documents: the format rule, placeholders, and absent facts."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.render import MissingFactsError, format_value, main, render_all
from bench.runfiles import Fact, Facts, ModelSet

CODE_DIGEST = "0" * 64


def fact(value: int | float | str | bool | None, unit: str = "ms") -> Fact:
    return Fact(
        value=value,
        unit=unit,
        runs=1,
        min=None,
        max=None,
        run_ids=("single-run-r1",),
        dirty_runs=0,
        failed_checks={},
        attempts=1,
        failed_attempts=0,
    )


def facts_of(values: dict[str, Fact]) -> Facts:
    return Facts(model_set=ModelSet.FAKE, code_digest=CODE_DIGEST, counted_run_ids=(), facts=values)


def templates(design: str, readme: str = "readme\n") -> dict[str, str]:
    return {"DESIGN.tmpl.md": design, "README.tmpl.md": readme}


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (None, "ms", "n/a"),
        (True, "", "yes"),
        (False, "", "no"),
        ("between 3 and 5 robots", "", "between 3 and 5 robots"),
        (7, "robots", "7"),
        (12345, "B", "12345"),
        (-4, "count", "-4"),
        (150.4, "ms", "150"),
        (150.5, "ms", "151"),
        (-250.5, "ms", "-251"),
        (100.0, "ms", "100"),
        (99.94, "ms", "99.9"),
        (99.96, "ms", "100"),
        (9.995, "ms", "10.0"),
        (12.34, "ms", "12.3"),
        (2.0, "ms", "2.00"),
        (50.0, "robots", "50.0"),
        (0.012345, "ms", "0.0123"),
        (1.5e-7, "ms", "0.000000150"),
        (1e20, "B", "100000000000000000000"),
        (0.0, "ms", "0"),
        (-0.0, "ms", "0"),
        (0.25, "ratio", "0.250"),
        (0.1235, "ratio", "0.124"),
        (0.001, "ratio", "0.001"),
        (0.0, "ratio", "0.000"),
        (-0.0, "ratio", "0.000"),
        (0.0004567, "ratio", "0.000457"),
        (0.00012, "ratio", "0.000120"),
    ],
)
def test_format_value_follows_the_format_rule(
    value: int | float | str | bool | None, unit: str, expected: str
) -> None:
    assert format_value(value, unit) == expected


def test_integer_valued_fact_renders_as_an_integer() -> None:
    rendered = render_all(
        templates("{{massive.cls.useful_robots}} robots\n"),
        facts_of({"massive.cls.useful_robots": fact(50, "robots")}),
        keep_missing=False,
    )
    assert rendered["DESIGN.md"] == "50 robots\n"


def test_null_fact_renders_as_not_available() -> None:
    rendered = render_all(
        templates("capacity {{sweep.cls.capacity_fps}} fps\n"),
        facts_of({"sweep.cls.capacity_fps": fact(None, "fps")}),
        keep_missing=False,
    )
    assert rendered["DESIGN.md"] == "capacity n/a fps\n"


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("p95 `{{steady.cls.p95}}` ms\n", "p95 2.50 ms\n"),
        ("p95 {{steady.cls.p95}} ms\n", "p95 2.50 ms\n"),
        ("`{{steady.cls.p95}} ms` at the start\n", "`2.50 ms` at the start\n"),
        ("at the end `p95 {{steady.cls.p95}}`\n", "at the end `p95 2.50`\n"),
    ],
)
def test_backticks_are_part_of_a_placeholder_only_as_a_pair(template: str, expected: str) -> None:
    rendered = render_all(
        templates(template), facts_of({"steady.cls.p95": fact(2.5)}), keep_missing=False
    )
    assert rendered["DESIGN.md"] == expected


def test_rendering_keeps_every_other_byte_of_the_template() -> None:
    template = "# Title\r\n\r\nNo placeholder, 30 fps, {braces} and `code`.\r\n"
    rendered = render_all(templates(template), facts_of({}), keep_missing=False)
    assert rendered["DESIGN.md"] == template


def test_absent_facts_are_named_for_every_template() -> None:
    with pytest.raises(MissingFactsError) as raised:
        render_all(
            templates("{{a.one}} {{a.present}} {{a.two}} {{a.one}}\n", "{{b.three}}\n"),
            facts_of({"a.present": fact(1)}),
            keep_missing=False,
        )
    assert raised.value.missing == {
        "DESIGN.tmpl.md": ("a.one", "a.two"),
        "README.tmpl.md": ("b.three",),
    }


def test_keep_missing_without_facts_leaves_placeholders_as_they_are() -> None:
    template = "a `{{a.one}}` and {{a.two}}\n"
    rendered = render_all(templates(template), None, keep_missing=True)
    assert rendered["DESIGN.md"] == template


def test_without_facts_and_without_keep_missing_every_key_is_named() -> None:
    with pytest.raises(MissingFactsError) as raised:
        render_all(templates("{{a.one}}\n", "{{b.two}}\n"), None, keep_missing=False)
    assert raised.value.missing == {"DESIGN.tmpl.md": ("a.one",), "README.tmpl.md": ("b.two",)}


def test_keep_missing_with_facts_still_fails_on_an_absent_key() -> None:
    with pytest.raises(MissingFactsError) as raised:
        render_all(
            templates("{{a.present}} {{a.absent}}\n"),
            facts_of({"a.present": fact(1)}),
            keep_missing=True,
        )
    assert raised.value.missing == {"DESIGN.tmpl.md": ("a.absent",)}


def write_templates(directory: Path, design: str, readme: str) -> None:
    directory.mkdir(parents=True)
    (directory / "DESIGN.tmpl.md").write_bytes(design.encode())
    (directory / "README.tmpl.md").write_bytes(readme.encode())


def write_facts(results: Path, facts: Facts) -> None:
    (results / "summary").mkdir(parents=True)
    (results / "summary" / "facts.json").write_bytes(facts.to_json().encode())


def test_cli_renders_both_templates_into_out_from_the_templates_given(tmp_path: Path) -> None:
    write_templates(tmp_path / "templates", "steady {{steady.cls.p95}}\n", "{{env.kernel}}\n")
    write_facts(
        tmp_path / "results",
        facts_of({"steady.cls.p95": fact(3.25), "env.kernel": fact("6.6.87", "")}),
    )
    status = main(
        [
            "--results",
            str(tmp_path / "results"),
            "--out",
            str(tmp_path / "rendered"),
            "--templates",
            str(tmp_path / "templates"),
        ]
    )
    assert status == 0
    assert (tmp_path / "rendered" / "DESIGN.md").read_bytes() == b"steady 3.25\n"
    assert (tmp_path / "rendered" / "README.md").read_bytes() == b"6.6.87\n"


def test_cli_names_absent_keys_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_templates(tmp_path / "templates", "{{a.absent}}\n", "readme\n")
    write_facts(tmp_path / "results", facts_of({}))
    status = main(
        [
            "--results",
            str(tmp_path / "results"),
            "--out",
            str(tmp_path / "rendered"),
            "--templates",
            str(tmp_path / "templates"),
            "--keep-missing",
        ]
    )
    assert status == 1
    assert "a.absent" in capsys.readouterr().err
    assert not (tmp_path / "rendered").exists()


def test_cli_with_keep_missing_and_no_facts_writes_the_placeholders(tmp_path: Path) -> None:
    write_templates(tmp_path / "templates", "`{{a.absent}}`\n", "readme\n")
    status = main(
        [
            "--results",
            str(tmp_path / "results"),
            "--out",
            str(tmp_path / "rendered"),
            "--templates",
            str(tmp_path / "templates"),
            "--keep-missing",
        ]
    )
    assert status == 0
    assert (tmp_path / "rendered" / "DESIGN.md").read_bytes() == b"`{{a.absent}}`\n"


def test_cli_names_a_missing_template(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "DESIGN.tmpl.md").write_bytes(b"design\n")
    status = main(
        ["--results", str(tmp_path / "results"), "--templates", str(tmp_path / "templates")]
    )
    assert status == 1
    assert "README.tmpl.md" in capsys.readouterr().err

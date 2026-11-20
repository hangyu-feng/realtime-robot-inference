"""Render the design document and the README from their templates and the facts.

A template holds every measured value as a placeholder (the grammar is bench.facts'), and
every other number as its literal. Rendering replaces each placeholder with its fact's value,
formatted by one rule, and leaves every other byte of the template as it is.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from bench.facts import PLACEHOLDER
from bench.runfiles import FACTS_FILE, Fact, Facts, RunFileError

# Each template and the file it renders into, both by name.
TEMPLATES: Mapping[str, str] = {"DESIGN.tmpl.md": "DESIGN.md", "README.tmpl.md": "README.md"}
# Where facts.json lies under the results root.
SUMMARY_DIRECTORY = "summary"

RATIO_UNIT = "ratio"
# A ratio below this, and not zero, would read as zero with three decimals; a stale ratio
# must not read as zero when its count is not.
SMALL_RATIO = Decimal("0.001")
RATIO_DECIMALS = Decimal("0.001")
# From this magnitude up, a float renders as an integer: a decimal adds no information there.
INTEGER_MAGNITUDE = 100
SIGNIFICANT_FIGURES = 3
NULL_TEXT = "n/a"


class RenderError(Exception):
    """A template or facts.json cannot be read, or a placeholder has no fact."""


def format_value(value: int | float | str | bool | None, unit: str) -> str:
    """Return a fact's value as the documents show it."""
    if value is None:
        return NULL_TEXT
    # A bool is an int to isinstance, so it is tested first.
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    # The shortest repr reads back as the same float, so rounding it rounds the digits a
    # reader would see, and half rounds up as a reader expects.
    number = Decimal(repr(value))
    if unit == RATIO_UNIT:
        if number.is_zero():
            return format(Decimal(0).quantize(RATIO_DECIMALS), "f")
        if abs(number) >= SMALL_RATIO:
            return format(number.quantize(RATIO_DECIMALS, ROUND_HALF_UP), "f")
        return _significant(number)
    if number.is_zero():
        # Zero has no significant figure to show.
        return "0"
    if abs(number) >= INTEGER_MAGNITUDE:
        return format(number.to_integral_value(ROUND_HALF_UP), "f")
    return _significant(number)


def _significant(number: Decimal) -> str:
    """Return a non-zero number to three significant figures, never in exponent notation."""
    leading = number.adjusted()
    rounded = number.quantize(Decimal(10) ** (leading - SIGNIFICANT_FIGURES + 1), ROUND_HALF_UP)
    if rounded.adjusted() > leading:
        # Rounding carried into a new leading digit (9.995 became 10.00): keep three figures.
        rounded = rounded.quantize(Decimal(10) ** (leading - SIGNIFICANT_FIGURES + 2))
    return format(rounded, "f")


class MissingFactsError(RenderError):
    """Placeholders whose keys facts.json does not hold, by template."""

    def __init__(self, missing: Mapping[str, Sequence[str]], has_facts: bool) -> None:
        self.missing = {name: tuple(keys) for name, keys in missing.items()}
        where = f"absent from {FACTS_FILE}" if has_facts else f"not rendered: no {FACTS_FILE}"
        listed = "; ".join(f"{name}: {', '.join(keys)}" for name, keys in self.missing.items())
        super().__init__(f"keys {where}: {listed}")


def _substitute(template: str, facts: Mapping[str, Fact]) -> tuple[str, list[str]]:
    """Return the template with every placeholder that has a fact replaced, and the others' keys.

    A placeholder whose key has no fact stays as it is in the text returned.
    """
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        key = match["key"]
        fact = facts.get(key)
        if fact is None:
            if key not in missing:
                missing.append(key)
            return match[0]
        return format_value(fact.value, fact.unit)

    return PLACEHOLDER.sub(replace, template), missing


def render_all(
    templates: Mapping[str, str], facts: Facts | None, keep_missing: bool
) -> dict[str, str]:
    """Return each rendered document by its file name, from the templates by theirs.

    keep_missing leaves the placeholders as they are, and takes effect only without facts:
    once facts.json exists, a key it lacks is an error with or without it. The error names
    every absent key of every template.
    """
    values: Mapping[str, Fact] = {} if facts is None else facts.facts
    rendered: dict[str, str] = {}
    missing: dict[str, Sequence[str]] = {}
    for name, template in templates.items():
        rendered[TEMPLATES[name]], absent = _substitute(template, values)
        if absent:
            missing[name] = absent
    if missing and not (keep_missing and facts is None):
        raise MissingFactsError(missing, has_facts=facts is not None)
    return rendered


def facts_path(results: Path) -> Path:
    return results / SUMMARY_DIRECTORY / FACTS_FILE


def read_facts(results: Path) -> Facts | None:
    """Return the facts of a results root; None when it holds no facts.json."""
    path = facts_path(results)
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RenderError(f"{path}: cannot be read: {error}") from error
    try:
        return Facts.from_json(content.decode("utf-8"))
    except (UnicodeDecodeError, RunFileError) as error:
        raise RenderError(f"{path}: {error}") from error


def read_templates(directory: Path) -> dict[str, str]:
    """Return the text of every template by its name, read as bytes so no newline changes."""
    templates: dict[str, str] = {}
    for name in TEMPLATES:
        path = directory / name
        try:
            templates[name] = path.read_bytes().decode("utf-8")
        except FileNotFoundError as error:
            raise RenderError(f"{path}: the template is missing") from error
        except (OSError, UnicodeDecodeError) as error:
            raise RenderError(f"{path}: cannot be read: {error}") from error
    return templates


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.render", description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"), help="results root")
    parser.add_argument("--out", type=Path, default=Path("."), help="where the documents go")
    parser.add_argument("--templates", type=Path, default=Path("docs"), help="the templates")
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help=f"while there is no {FACTS_FILE}, leave its placeholders as they are",
    )
    arguments = parser.parse_args(argv)
    try:
        rendered = render_all(
            read_templates(arguments.templates),
            read_facts(arguments.results),
            arguments.keep_missing,
        )
    except RenderError as error:
        print(f"render: {error}", file=sys.stderr)
        return 1
    arguments.out.mkdir(parents=True, exist_ok=True)
    for name, text in rendered.items():
        path = arguments.out / name
        # Bytes, so the newlines are the template's on every platform.
        path.write_bytes(text.encode("utf-8"))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

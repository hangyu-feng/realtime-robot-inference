"""The number check: every number in the documents is a fact of the evidence or declared.

It fails, and prints where, when a rendered document differs from rendering its template
now; when facts.json, the tables or the ledger differ from what aggregation and table
generation produce now from the same results root; when a template holds a character with a
numeric value that no rule exempts, or a character reference; or when a declared constant is
malformed. The scan of the templates and the checks of the declarations are pure functions.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from bench.facts import PLACEHOLDER
from bench.render import (
    TEMPLATES,
    RenderError,
    facts_path,
    read_facts,
    read_templates,
    render_all,
)
from bench.runfiles import ModelSet

CONSTANTS_FILE = "constants.json"
CATEGORIES = ("brief", "protocol", "design", "assumption", "derived")
# What aggregation writes under the results root besides the tables, and the tables' directory.
LEDGER_FILE = "ledger.jsonl"
TABLES_DIRECTORY = "tables"
# The modules that write the tables, each run with --results and --out.
TABLE_MODULES = ("bench.tables", "bench.metrics_table")
_TABLE_WRITERS = " and ".join(TABLE_MODULES)
RAW_DIRECTORY = "raw"
DIGEST_MODES = ("current", "recorded")
# Far above what a build or the tables take; it only keeps a hung command from hanging the
# check.
COMMAND_TIMEOUT_S = 900.0
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# A numeric literal: digits, a space before each further group of three, a decimal part.
LITERAL = re.compile(r"[0-9]+(?: [0-9]{3})*(?:\.[0-9]+)?")
# A word token, matched whole: what precedes it is no ASCII letter, digit or underscore.
WORD_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*")
SECTION_HEADING = re.compile(r"^## ([0-9]+)\.(?= |$)")
SUBSECTION_HEADING = re.compile(r"^### ([0-9]+)\.([0-9]+)(?= |$)")
LIST_MARKER = re.compile(r"^([ \t]*)([0-9]+)\. ")
_SECTION_NUMBER = r"[0-9]+(?:\.[0-9]+)?"
SECTION_NUMBER = re.compile(_SECTION_NUMBER)
# Each number may be followed by punctuation; the punctuation sits inside the repeated group,
# so that a comma that is part of the joining ", " can be given back to it.
SECTION_REFERENCE = re.compile(
    rf"(?<![A-Za-z0-9_])[Ss]ections? {_SECTION_NUMBER}"
    rf"(?:[^\w\s]*(?:, | and | or | to ){_SECTION_NUMBER})*"
)
CHARACTER_REFERENCE = "&#"
# Stands for each character of a placeholder in the scan: no rule reads it as a digit, a
# letter, a space or punctuation that joins numbers.
_MASK = "\0"


class NumbersError(Exception):
    """constants.json cannot be read, or is not the shape the check reads."""


@dataclass(frozen=True)
class Constant:
    """A number that stands in the templates as its literal, with what decides it."""

    text: str
    category: str
    source: str


@dataclass(frozen=True)
class Declarations:
    """docs/constants.json: the declared literals and the tokens with a digit that may stand."""

    constants: tuple[Constant, ...]
    identifiers: frozenset[str]


def _require[T](value: object, kind: type[T], path: str) -> T:
    if not isinstance(value, kind):
        raise NumbersError(f"{CONSTANTS_FILE}: {path} must be a {kind.__name__}")
    return value


def _object(value: object, path: str, keys: tuple[str, ...]) -> dict[str, object]:
    document: dict[str, object] = _require(value, dict, path)
    missing = [key for key in keys if key not in document]
    extra = [key for key in document if key not in keys]
    if missing or extra:
        raise NumbersError(f"{CONSTANTS_FILE}: {path} must have exactly the fields {keys}")
    return document


def parse_declarations(text: str) -> Declarations:
    """Read constants.json; its field values are checked by check_declarations."""
    try:
        value = json.loads(text)
    except ValueError as error:
        raise NumbersError(f"{CONSTANTS_FILE} is not JSON: {error}") from error
    document = _object(value, "the file", ("constants", "identifiers"))
    constants: list[Constant] = []
    for index, entry in enumerate(_require(document["constants"], list, "constants")):
        path = f"constants[{index}]"
        fields = _object(entry, path, ("text", "category", "source"))
        constants.append(
            Constant(
                text=_require(fields["text"], str, f"{path}.text"),
                category=_require(fields["category"], str, f"{path}.category"),
                source=_require(fields["source"], str, f"{path}.source"),
            )
        )
    identifiers = _require(document["identifiers"], list, "identifiers")
    return Declarations(
        constants=tuple(constants),
        identifiers=frozenset(
            _require(identifier, str, f"identifiers[{index}]")
            for index, identifier in enumerate(identifiers)
        ),
    )


def check_declarations(declarations: Declarations) -> list[str]:
    """Return a problem for every constant whose text or category is malformed (rule 4)."""
    problems: list[str] = []
    for index, constant in enumerate(declarations.constants):
        where = f"{CONSTANTS_FILE}: constants[{index}]"
        if LITERAL.fullmatch(constant.text) is None:
            problems.append(f"{where}: text {constant.text!r} is not one numeric literal")
        if constant.category not in CATEGORIES:
            problems.append(
                f"{where}: category {constant.category!r} is not one of {', '.join(CATEGORIES)}"
            )
    return problems


@dataclass(frozen=True)
class Finding:
    """Characters of a template that no rule allows, by line and column (both from 1)."""

    line: int
    column: int
    text: str
    reason: str

    def where(self, path: str) -> str:
        return f"{path}:{self.line}:{self.column}: {self.text!r}: {self.reason}"


@dataclass(frozen=True)
class _Span:
    """A part of a line that a rule reads; when it is not exempt, why its numbers fail."""

    start: int
    end: int
    exempt: bool
    reason: str


_NUMERIC_REASON = "a character with a numeric value that no rule allows"


def _heading_numbers(lines: Sequence[str]) -> frozenset[str]:
    numbers: set[str] = set()
    for line in lines:
        if section := SECTION_HEADING.match(line):
            numbers.add(section[1])
        elif subsection := SUBSECTION_HEADING.match(line):
            numbers.add(f"{subsection[1]}.{subsection[2]}")
    return frozenset(numbers)


class _Headings:
    """Whether each numbered heading is the next one: 1, 2, ... and, under N, N.1, N.2, ..."""

    def __init__(self) -> None:
        self._section = 0
        self._subsection = 0

    def span(self, line: str) -> _Span | None:
        if section := SECTION_HEADING.match(line):
            number = int(section[1])
            in_order = number == self._section + 1
            # The next heading is checked against this one, so a gap is reported once.
            self._section, self._subsection = number, 0
            return _Span(*section.span(1), in_order, "a heading number out of order")
        if subsection := SUBSECTION_HEADING.match(line):
            number = int(subsection[2])
            in_order = int(subsection[1]) == self._section and number == self._subsection + 1
            self._subsection = number
            return _Span(
                subsection.start(1), subsection.end(2), in_order, "a heading number out of order"
            )
        return None


class _Lists:
    """The position of each ordered-list marker in its list.

    A list is the markers at one indentation. Blank lines do not end it; any other line
    that is not indented past it does, as does a marker less indented than it.
    """

    def __init__(self) -> None:
        self._open: dict[int, int] = {}  # indentation: markers so far

    def span(self, line: str) -> _Span | None:
        if not line.strip():
            return None
        marker = LIST_MARKER.match(line)
        indentation = len(line) - len(line.lstrip(" \t"))
        nested = indentation if marker else indentation - 1
        self._open = {depth: count for depth, count in self._open.items() if depth <= nested}
        if marker is None:
            return None
        position = self._open.get(indentation, 0) + 1
        self._open[indentation] = position
        return _Span(
            *marker.span(2),
            int(marker[2]) == position,
            f"a list marker that is not its position, {position}, in its list",
        )


class _Scan:
    """The spans of each line that the rules read, in the order of the template's lines."""

    def __init__(self, declared: Declarations, heading_numbers: frozenset[str]) -> None:
        self._identifiers = declared.identifiers
        self._constant_texts = frozenset(constant.text for constant in declared.constants)
        self._heading_numbers = heading_numbers
        self._headings = _Headings()
        self._lists = _Lists()

    def spans(self, line: str) -> list[_Span]:
        """Return the spans of a line; earlier spans give the reason a number fails."""
        spans = [span for span in (self._headings.span(line), self._lists.span(line)) if span]
        for reference in SECTION_REFERENCE.finditer(line):
            numbers = list(SECTION_NUMBER.finditer(line, reference.start(), reference.end()))
            exists = all(number[0] in self._heading_numbers for number in numbers)
            spans.extend(
                _Span(*number.span(), exists, "a section reference to a heading that is not here")
                for number in numbers
            )
        tokens = list(WORD_TOKEN.finditer(line))
        spans.extend(
            _Span(*token.span(), token[0] in self._identifiers, "not a declared identifier")
            for token in tokens
        )
        spans.extend(
            _Span(*literal.span(), literal[0] in self._constant_texts, "not a declared constant")
            for literal in LITERAL.finditer(line)
            if not any(
                literal.start() < token.end() and token.start() < literal.end() for token in tokens
            )
        )
        return spans


def _failures(line: str, spans: Sequence[_Span]) -> Iterator[_Span]:
    """Yield what holds the numeric characters that no span exempts, each once.

    That is the first span that reads such a character, or, where no span reads it, the
    run of such characters it belongs to.
    """
    reported: set[_Span] = set()
    run: _Span | None = None
    for column, character in enumerate(line):
        if unicodedata.numeric(character, None) is None:
            continue
        covering = [span for span in spans if span.start <= column < span.end]
        if any(span.exempt for span in covering):
            continue
        if covering:
            if covering[0] not in reported:
                reported.add(covering[0])
                yield covering[0]
        elif run is not None and run.end == column:
            run = _Span(run.start, column + 1, False, _NUMERIC_REASON)
        else:
            if run is not None:
                yield run
            run = _Span(column, column + 1, False, _NUMERIC_REASON)
    if run is not None:
        yield run


def scan_template(template: str, declared: Declarations) -> list[Finding]:
    """Return every character with a numeric value that no rule allows, and every `&#`.

    Placeholders are left out. The rules that allow a number, any one of them enough: a
    heading's section number when it is the next one; a section reference whose every
    number is a heading's; an ordered-list marker at its position; a word token that is a
    declared identifier; a numeric literal outside every word token that is a declared
    constant's text.
    """
    lines = [
        PLACEHOLDER.sub(lambda match: _MASK * len(match[0]), line) for line in template.split("\n")
    ]
    scan = _Scan(declared, _heading_numbers(lines))
    findings: list[Finding] = []
    for number, line in enumerate(lines, start=1):
        spans = scan.spans(line)
        findings.extend(
            Finding(number, match.start() + 1, CHARACTER_REFERENCE, "a character reference")
            for match in re.finditer(re.escape(CHARACTER_REFERENCE), line)
        )
        findings.extend(
            Finding(number, failure.start + 1, line[failure.start : failure.end], failure.reason)
            for failure in _failures(line, spans)
        )
    return sorted(findings, key=lambda finding: (finding.line, finding.column))


def differing_lines(expected: str, actual: str) -> list[int]:
    """Return the lines of actual, from 1, where it differs from expected."""
    expected_lines = expected.split("\n")
    actual_lines = actual.split("\n")
    lines: list[int] = []
    matcher = difflib.SequenceMatcher(None, expected_lines, actual_lines, autojunk=False)
    for tag, _, _, start, end in matcher.get_opcodes():
        if tag == "equal":
            continue
        # A deletion has no line of its own in actual: name the line where it is missing.
        lines.extend(
            range(start + 1, end + 1) if end > start else [min(start + 1, len(actual_lines))]
        )
    return lines


def check_rendered(
    templates: Mapping[str, str], rendered_now: Mapping[str, str], out: Path
) -> list[str]:
    """Rule 1: each rendered document in out is what rendering its template gives now."""
    problems: list[str] = []
    for name in templates:
        document = TEMPLATES[name]
        path = out / document
        try:
            actual = path.read_bytes().decode("utf-8")
        except FileNotFoundError:
            problems.append(f"{path}: missing; render it from {name}")
            continue
        except (OSError, UnicodeDecodeError) as error:
            problems.append(f"{path}: cannot be read: {error}")
            continue
        actual_lines = actual.split("\n")
        problems.extend(
            f"{path}:{line}: differs from rendering {name} now: {actual_lines[line - 1]!r}"
            for line in differing_lines(rendered_now[document], actual)
        )
    return problems


def _run_module(arguments: Sequence[str]) -> str | None:
    """Run a module of this repository; return its output when it fails, None when it passes."""
    command = [sys.executable, "-m", *arguments]
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"python -m {' '.join(arguments)} did not finish within {COMMAND_TIMEOUT_S:g} s"
    if completed.returncode == 0:
        return None
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return f"python -m {' '.join(arguments)} exited {completed.returncode}: {output}"


def _files(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _compare_file(recorded: Path, rebuilt: Path, name: str) -> list[str]:
    recorded_bytes = recorded.read_bytes() if recorded.is_file() else None
    rebuilt_bytes = rebuilt.read_bytes() if rebuilt.is_file() else None
    if recorded_bytes == rebuilt_bytes:
        return []
    if rebuilt_bytes is None:
        return [f"{recorded}: not produced now from the same results ({name})"]
    if recorded_bytes is None:
        return [f"{recorded}: missing; {name} produces it now"]
    return [f"{recorded}: differs from what {name} produces now"]


def _compare_tables(recorded: Path, rebuilt: Path) -> list[str]:
    recorded_files, rebuilt_files = _files(recorded), _files(rebuilt)
    problems: list[str] = []
    for name in sorted(recorded_files.keys() | rebuilt_files.keys()):
        if recorded_files.get(name) == rebuilt_files.get(name):
            continue
        if name not in rebuilt_files:
            problems.append(f"{recorded / name}: not produced now by {_TABLE_WRITERS}")
        elif name not in recorded_files:
            problems.append(f"{recorded / name}: missing; {_TABLE_WRITERS} produce it now")
        else:
            problems.append(f"{recorded / name}: differs from what {_TABLE_WRITERS} produce now")
    return problems


def _holds_runs(results: Path) -> bool:
    # Every run directory, a hand run's included, starts with its config.json.
    return any((results / RAW_DIRECTORY).rglob("config.json"))


def check_rebuilt(results: Path, model_set: ModelSet | None, digest: str) -> list[str]:
    """Rule 2: facts.json, the ledger and the tables are what the tools produce now."""
    recorded_facts = facts_path(results)
    if model_set is None:
        if not _holds_runs(results):
            # No run directory and no facts.json: there is nothing to have changed.
            return []
        return [f"{recorded_facts}: missing, and no --models to build the facts of the runs with"]
    with tempfile.TemporaryDirectory(prefix="rri-numbers-") as directory:
        rebuilt = Path(directory)
        refused = _run_module(
            [
                "bench.aggregate",
                "build",
                "--results",
                str(results),
                "--models",
                model_set.value,
                "--out",
                str(rebuilt),
                "--digest",
                digest,
            ]
        )
        if refused is not None:
            return [f"the facts cannot be built now: {refused}"]
        rebuilt_facts = facts_path(rebuilt)
        problems = _compare_file(recorded_facts, rebuilt_facts, "bench.aggregate build")
        problems += _compare_file(
            results / LEDGER_FILE, rebuilt / LEDGER_FILE, "bench.aggregate build"
        )
        if not recorded_facts.is_file() and not rebuilt_facts.is_file():
            # No run directory: there are no facts to draw tables from.
            return problems
        for module in TABLE_MODULES:
            failed = _run_module([module, "--results", str(results), "--out", str(rebuilt)])
            if failed is not None:
                return [*problems, f"the tables cannot be produced now: {failed}"]
        return problems + _compare_tables(results / TABLES_DIRECTORY, rebuilt / TABLES_DIRECTORY)


def read_declarations(templates: Path) -> Declarations:
    path = templates / CONSTANTS_FILE
    try:
        return parse_declarations(path.read_bytes().decode("utf-8"))
    except FileNotFoundError as error:
        raise NumbersError(f"{path}: missing") from error
    except (OSError, UnicodeDecodeError) as error:
        raise NumbersError(f"{path}: cannot be read: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.numbers", description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"), help="results root")
    parser.add_argument("--out", type=Path, default=Path("."), help="the rendered documents")
    parser.add_argument(
        "--models",
        type=ModelSet,
        choices=list(ModelSet),
        help="the model set to build the facts for; default the one facts.json records",
    )
    parser.add_argument("--templates", type=Path, default=Path("docs"), help="the templates")
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="while there is no facts.json, leave its placeholders as they are",
    )
    parser.add_argument("--digest", choices=DIGEST_MODES, default="current")
    arguments = parser.parse_args(argv)
    results: Path = arguments.results.resolve()
    try:
        templates = read_templates(arguments.templates)
        declared = read_declarations(arguments.templates)
        facts = read_facts(results)
    except (RenderError, NumbersError) as error:
        print(f"numbers: {error}", file=sys.stderr)
        return 1
    try:
        problems = check_rendered(
            templates, render_all(templates, facts, arguments.keep_missing), arguments.out
        )
    except RenderError as error:
        problems = [str(error)]
    model_set = arguments.models or (None if facts is None else facts.model_set)
    problems += check_rebuilt(results, model_set, arguments.digest)
    for name, template in templates.items():
        path = str(arguments.templates / name)
        problems += [finding.where(path) for finding in scan_template(template, declared)]
    problems += check_declarations(declared)
    for problem in problems:
        print(problem)
    if problems:
        print(f"number check failed: {len(problems)} problems")
        return 1
    print("number check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

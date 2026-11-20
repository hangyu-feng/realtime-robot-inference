"""Tests for the number check: the template scan, the declarations, and the four rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import render
from bench.numbers import (
    Constant,
    Declarations,
    NumbersError,
    check_declarations,
    check_rebuilt,
    check_rendered,
    differing_lines,
    main,
    parse_declarations,
    scan_template,
)
from bench.render import render_all
from bench.runfiles import Fact, Facts, ModelSet, RunConfig

CODE_DIGEST = "0" * 64


def declared(*texts: str, identifiers: tuple[str, ...] = ()) -> Declarations:
    return Declarations(
        constants=tuple(Constant(text, "design", "a test") for text in texts),
        identifiers=frozenset(identifiers),
    )


def failing_texts(template: str, declarations: Declarations) -> list[str]:
    return [finding.text for finding in scan_template(template, declarations)]


@pytest.mark.parametrize(
    ("template", "declarations", "expected"),
    [
        ("It takes 7 ms.\n", declared(), ["7"]),
        ("| cls | 7 | 8 |\n", declared("8"), ["7"]),
        ("```bash\nrun --robots 7\n```\n", declared(), ["7"]),
        ("See http://localhost:8080/metrics.\n", declared("80"), ["8080"]),
        ("Frames of 640x480 pixels.\n", declared("640"), ["480"]),
        ("Frames of 640x480 pixels.\n", declared("480"), ["640"]),
        ("## 1. One\n\nAs section 2 says.\n", declared(), ["2"]),
        ("## 1. One\n### 1.1 A\n\nSections 1 and 1.2.\n", declared(), ["1", "1.2"]),
        ("1. first\n3. second\n", declared(), ["3"]),
        ("1. first\n\nA paragraph.\n\n2. again\n", declared(), ["2"]),
        ("The p95 of it.\n", declared(), ["p95"]),
        ("The p95 of it.\n", declared("95"), ["p95"]),
        ("The x86_64 host.\n", declared("86", "64"), ["x86_64"]),
        ("An area in m\N{SUPERSCRIPT TWO}.\n", declared(), ["\N{SUPERSCRIPT TWO}"]),
        (
            "Half, \N{VULGAR FRACTION ONE HALF}, of it.\n",
            declared(),
            ["\N{VULGAR FRACTION ONE HALF}"],
        ),
        ("A reference &#; in text.\n", declared(), ["&#"]),
        ("## 1. One\n## 3. Three\n", declared(), ["3"]),
        ("## 1. One\n### 1.2 B\n", declared(), ["1.2"]),
        ("## 1. One\n## 2. Two\n### 1.1 B\n", declared(), ["1.1"]),
        ("Sixty thousand is 60 000.\n", declared("60", "000"), ["60 000"]),
    ],
)
def test_scan_fails_on_a_number_that_no_rule_allows(
    template: str, declarations: Declarations, expected: list[str]
) -> None:
    assert failing_texts(template, declarations) == expected


@pytest.mark.parametrize(
    ("template", "declarations"),
    [
        ("## 1. One\n### 1.1 A\n### 1.2 B\n## 2. Two\n### 2.1 C\n", declared()),
        ("## 1. One\n### 1.1 A\n## 2. Two\n\nSection 2, and section 1.1.\n", declared()),
        (
            "## 1. One\n## 2. Two\n## 3. Three\n\nSections 1, 2 and 3; sections 1 to 3.\n",
            declared(),
        ),
        ("## 1. One\n\n(see section 1).\n", declared()),
        ("## 1. One\n\nSection 1 says so.\n", declared()),
        ("## 1. One\n## 2. Two\n## 3. Three\n\nSections 1, 2, and 3 or 1.\n", declared()),
        ("It takes 7 ms.\n", declared("7")),
        ("| cls | 7 |\n", declared("7")),
        ("```bash\nrun --robots 7\n```\n", declared("7")),
        ("See http://localhost:8080/metrics.\n", declared("8080")),
        ("Frames of 640x480 pixels.\n", declared("640", "480")),
        ("Sixty thousand is 60 000.\n", declared("60 000")),
        ("A batch of 0.5 ms.\n", declared("0.5")),
        ("The p95, A20 and x86_64.\n", declared(identifiers=("p95", "A20", "x86_64"))),
        ("1. first\n2. second\n\n3. third, after a blank line\n", declared()),
        ("1. first\n   a continuation\n   - a bullet\n2. second\n", declared()),
        ("1. outer\n   1. inner\n   2. inner\n2. outer\n", declared()),
        ("Paragraph.\n\n1. a new list\n", declared()),
        (
            "Median `{{sweep.n5.cls.e2e_ms.mean}}` and {{model_bench.depth.t1.b1.wall_ms}}.\n",
            declared(),
        ),
        ("No number at all.\n", declared()),
    ],
)
def test_scan_passes_on_each_exemption(template: str, declarations: Declarations) -> None:
    assert scan_template(template, declarations) == []


def test_scan_names_the_line_and_column() -> None:
    (finding,) = scan_template("first line\nit takes 7 ms\n", declared())
    assert (finding.line, finding.column, finding.text) == (2, 10, "7")
    assert finding.where("docs/DESIGN.tmpl.md").startswith("docs/DESIGN.tmpl.md:2:10: ")


@pytest.mark.parametrize(
    ("constant", "problems"),
    [
        (Constant("60 000", "design", "section 8.3"), 0),
        (Constant("0.5", "brief", "the brief"), 0),
        (Constant("1,5", "design", "a test"), 1),
        (Constant("1e3", "design", "a test"), 1),
        (Constant("", "design", "a test"), 1),
        (Constant("60  000", "design", "a test"), 1),
        (Constant("seven", "design", "a test"), 1),
        (Constant("7", "guess", "a test"), 1),
        (Constant("7 ms", "measured", "a test"), 2),
    ],
)
def test_declarations_check_each_text_and_category(constant: Constant, problems: int) -> None:
    assert len(check_declarations(Declarations((constant,), frozenset()))) == problems


def test_declarations_parse_constants_and_identifiers() -> None:
    text = json.dumps(
        {
            "constants": [{"text": "150", "category": "design", "source": "section 5.2"}],
            "identifiers": ["p95", "A20"],
        }
    )
    assert parse_declarations(text) == Declarations(
        (Constant("150", "design", "section 5.2"),), frozenset({"p95", "A20"})
    )


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        json.dumps({"constants": []}),
        json.dumps({"constants": [{"text": "1", "category": "design"}], "identifiers": []}),
        json.dumps({"constants": [{"text": 1, "category": "d", "source": "s"}], "identifiers": []}),
        json.dumps({"constants": [], "identifiers": [5]}),
        json.dumps({"constants": [], "identifiers": [], "extra": 1}),
    ],
)
def test_declarations_refuse_a_malformed_file(text: str) -> None:
    with pytest.raises(NumbersError):
        parse_declarations(text)


@pytest.mark.parametrize(
    ("expected", "actual", "lines"),
    [
        ("a\nb\nc\n", "a\nb\nc\n", []),
        ("a\nb\nc\n", "a\nB\nc\n", [2]),
        ("a\nb\nc\n", "a\nc\n", [2]),
        ("a\nb\n", "a\nb\nextra\n", [3]),
    ],
)
def test_differing_lines_name_the_lines_of_the_rendered_file(
    expected: str, actual: str, lines: list[int]
) -> None:
    assert differing_lines(expected, actual) == lines


def fact(value: int | float | str | bool | None) -> Fact:
    return Fact(value, "ms", 1, None, None, ("single-run-r1",), 0, {}, 1, 0)


TEMPLATES = {
    "DESIGN.tmpl.md": "# Design\nIt takes {{steady.cls.p95}} ms.\nEnd.\n",
    "README.tmpl.md": "Readme.\n",
}
FACTS = Facts(ModelSet.FAKE, CODE_DIGEST, (), {"steady.cls.p95": fact(2.5)})


def write_rendered(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, text in render_all(TEMPLATES, FACTS, keep_missing=False).items():
        (out / name).write_bytes(text.encode())


def test_rendered_check_passes_on_documents_rendered_now(tmp_path: Path) -> None:
    write_rendered(tmp_path)
    rendered_now = render_all(TEMPLATES, FACTS, keep_missing=False)
    assert check_rendered(TEMPLATES, rendered_now, tmp_path) == []


def test_rendered_check_names_a_hand_edited_line(tmp_path: Path) -> None:
    write_rendered(tmp_path)
    design = tmp_path / "DESIGN.md"
    design.write_bytes(design.read_bytes().replace(b"2.50", b"2.40"))
    (problem,) = check_rendered(TEMPLATES, render_all(TEMPLATES, FACTS, False), tmp_path)
    assert problem.startswith(f"{design}:2: ")


def test_rendered_check_names_a_missing_document(tmp_path: Path) -> None:
    write_rendered(tmp_path)
    (tmp_path / "README.md").unlink()
    (problem,) = check_rendered(TEMPLATES, render_all(TEMPLATES, FACTS, False), tmp_path)
    assert problem.startswith(f"{tmp_path / 'README.md'}: missing")


def test_rebuild_check_passes_with_no_run_and_no_facts(tmp_path: Path) -> None:
    assert check_rebuilt(tmp_path, None, "current") == []


def test_rebuild_check_fails_on_runs_without_facts_or_a_model_set(tmp_path: Path) -> None:
    write_hand_run(tmp_path)
    (problem,) = check_rebuilt(tmp_path, None, "current")
    assert "facts.json: missing" in problem


def write_hand_run(results: Path) -> None:
    run = results / "raw" / "steady" / "manual"
    run.mkdir(parents=True)
    config = RunConfig(spec=None, run_id=None, seed=0, extra_compose_path=None, fleet=None)
    (run / "config.json").write_bytes(config.to_json().encode())


def test_rebuild_check_fails_on_a_facts_file_no_build_produces(tmp_path: Path) -> None:
    (tmp_path / "summary").mkdir()
    (tmp_path / "summary" / "facts.json").write_bytes(FACTS.to_json().encode())
    problems = check_rebuilt(tmp_path, ModelSet.FAKE, "current")
    assert problems[0] == (
        f"{render.facts_path(tmp_path)}: not produced now from the same results"
        " (bench.aggregate build)"
    )


def test_rebuild_check_fails_with_the_message_of_a_refused_build(tmp_path: Path) -> None:
    write_hand_run(tmp_path)
    # The recorded digest, because the current one lists the tree with git, which the image
    # whose suite is the tests gate does not have.
    (problem,) = check_rebuilt(tmp_path, ModelSet.FAKE, "recorded")
    assert problem.startswith("the facts cannot be built now: python -m bench.aggregate build")
    assert " exited 2: " in problem


CONSTANTS = {
    "constants": [{"text": "2", "category": "design", "source": "a test"}],
    "identifiers": ["p95"],
}


def write_documents(root: Path, design: str) -> None:
    templates = root / "docs"
    templates.mkdir(parents=True, exist_ok=True)
    (templates / "DESIGN.tmpl.md").write_bytes(design.encode())
    (templates / "README.tmpl.md").write_bytes(b"Readme with p95.\n")
    (templates / "constants.json").write_text(json.dumps(CONSTANTS))


def run_check(root: Path) -> int:
    return main(
        [
            "--results",
            str(root / "results"),
            "--out",
            str(root / "rendered"),
            "--templates",
            str(root / "docs"),
            "--keep-missing",
            "--digest",
            "recorded",
        ]
    )


def test_check_passes_on_templates_rendered_without_facts(tmp_path: Path) -> None:
    write_documents(tmp_path, "## 1. One\n\nThe p95 of `{{steady.cls.p95}}` ms, 2 robots.\n")
    assert (
        render.main(
            [
                "--results",
                str(tmp_path / "results"),
                "--out",
                str(tmp_path / "rendered"),
                "--templates",
                str(tmp_path / "docs"),
                "--keep-missing",
            ]
        )
        == 0
    )
    assert run_check(tmp_path) == 0


@pytest.mark.parametrize(
    "design",
    [
        "## 1. One\n\nThe p95 of `{{steady.cls.p95}}` ms, 3 robots.\n",
        "## 1. One\n\nSee section 4.\n",
        "## 3. Three\n",
        "Robots at &#50;.\n",
    ],
)
def test_check_fails_on_a_template_number_that_no_rule_allows(
    tmp_path: Path, design: str, capsys: pytest.CaptureFixture[str]
) -> None:
    write_documents(tmp_path, design)
    render.main(
        [
            "--results",
            str(tmp_path / "results"),
            "--out",
            str(tmp_path / "rendered"),
            "--templates",
            str(tmp_path / "docs"),
            "--keep-missing",
        ]
    )
    assert run_check(tmp_path) == 1
    assert f"{tmp_path / 'docs' / 'DESIGN.tmpl.md'}:" in capsys.readouterr().out


def test_check_fails_on_a_hand_edited_rendered_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_documents(tmp_path, "## 1. One\n\n2 robots, `{{steady.cls.p95}}` ms.\n")
    render.main(
        [
            "--results",
            str(tmp_path / "results"),
            "--out",
            str(tmp_path / "rendered"),
            "--templates",
            str(tmp_path / "docs"),
            "--keep-missing",
        ]
    )
    design = tmp_path / "rendered" / "DESIGN.md"
    design.write_bytes(design.read_bytes().replace(b"`{{steady.cls.p95}}`", b"2"))
    assert run_check(tmp_path) == 1
    assert f"{design}:3: " in capsys.readouterr().out


def test_check_fails_on_a_malformed_constant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_documents(tmp_path, "Nothing.\n")
    bad = {**CONSTANTS, "constants": [{"text": "2,5", "category": "guess", "source": "x"}]}
    (tmp_path / "docs" / "constants.json").write_text(json.dumps(bad))
    render.main(
        [
            "--results",
            str(tmp_path / "results"),
            "--out",
            str(tmp_path / "rendered"),
            "--templates",
            str(tmp_path / "docs"),
            "--keep-missing",
        ]
    )
    assert run_check(tmp_path) == 1
    output = capsys.readouterr().out
    assert "constants[0]: text '2,5'" in output
    assert "constants[0]: category 'guess'" in output

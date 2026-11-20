"""The /metrics contract: its field names and scalar types, as DESIGN.md states them and as a
snapshot has them.

Every function keys its result by object path, with a placeholder for every key that names an
instance: "robots.<id>.models.<m>", "processes.<p>". A test compares the two sides, so a field
or its JSON type cannot change in /metrics without changing in DESIGN.md section 10, or the
other way round.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType

# The subsections of DESIGN.md section 10 that hold the field tables, and the one whose table
# is the config echo.
_SUBSECTION = re.compile(r"### 10\.(\d+) (.*)")
_FIRST_TABLE_SUBSECTION = 2
_CONFIG_SUBSECTION = 8
_NAME = re.compile(r"`([^`]+)`")
# A paragraph that names the object of the table below it: "`workers.<m>`:" or "`cgroup`, read".
_TABLE_LABEL = re.compile(r"`([^`]+)`[:,]")
_PARENTHESIZED = re.compile(r"\([^)]*\)")

# The snapshot objects that hold one entry per instance, with the placeholder of their keys.
_KEYED = MappingProxyType(
    {
        "robots": "<id>",
        "robots.<id>.models": "<m>",
        "robots.<id>.telemetry.models": "<m>",
        "models": "<m>",
        "stages": "<m>",
        "workers": "<m>",
        "processes": "<p>",
        "config.models": "<m>",
    }
)
# The other snapshot objects whose keys are fields, besides the groups of settings inside the
# config echo. Every other object is a value: a histogram, or a table keyed by a model, a
# reason or a size.
_OBJECTS = frozenset(("robots.<id>.telemetry", "connections", "cgroup", "loop", "config"))
# The scalar types of the Type column and the JSON values each allows. Python's equality
# cannot tell true from 1 or 3 from 3.0, so a value is checked by its exact type.
_SCALAR_TYPES: Mapping[str, type] = MappingProxyType(
    {"int": int, "float": float, "bool": bool, "string": str}
)
_OR_NULL = " or null"


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def contract_fields(design_text: str) -> dict[str, frozenset[str]]:
    """Read the field names of every object path from DESIGN.md sections 10.2 to 10.8.

    A table belongs to the object that its section heading names in backticks, or that the
    paragraph above it starts with, and to the top level without either. Its fields are the
    backticked names of its first column. A dotted name such as `models.<m>.commands_received`
    is a field of the object it sits in, and its first segment is a field of the table's own
    object.

    The config echo of section 10.8 has no type column, so there a row names an object exactly
    when its first column is one name and its second column, without its parenthesized parts
    and everything from its first colon, is a comma-separated list of single backticked names:
    those are the object's fields. Every other row names scalars, one per name of its first
    column; alternatives joined with "or" and prose are not fields.
    """
    fields: defaultdict[str, set[str]] = defaultdict(set)
    for subsection, path, names, second in _table_rows(design_text):
        for name in names:
            segments = name.split(".")
            fields[path].add(segments[0])
            if len(segments) > 1 and not segments[-1].startswith("<"):
                fields[_join(path, ".".join(segments[:-1]))].add(segments[-1])
        keys = [key.strip() for key in _PARENTHESIZED.sub("", second).split(":")[0].split(",")]
        if (
            subsection == _CONFIG_SUBSECTION
            and len(names) == 1
            and all(_NAME.fullmatch(key) for key in keys)
        ):
            fields[_join(path, names[0])].update(key.strip("`") for key in keys)
    return {path: frozenset(names) for path, names in fields.items()}


def contract_types(design_text: str) -> dict[str, dict[str, str]]:
    """Read the Type column of every scalar field from DESIGN.md sections 10.2 to 10.7.

    The result maps an object path to its scalar fields and their types: int, float, bool or
    string, each possibly followed by " or null". Objects, lists and histograms are left out,
    and so is the config echo, whose table has no Type column.
    """
    types: defaultdict[str, dict[str, str]] = defaultdict(dict)
    for subsection, path, names, second in _table_rows(design_text):
        if subsection == _CONFIG_SUBSECTION or second.removesuffix(_OR_NULL) not in _SCALAR_TYPES:
            continue
        for name in names:
            *object_segments, field = name.split(".")
            object_path = _join(path, ".".join(object_segments)) if object_segments else path
            types[object_path][field] = second
    return dict(types)


def _table_rows(design_text: str) -> Iterator[tuple[int, str, list[str], str]]:
    """Yield every table row of DESIGN.md sections 10.2 to 10.8.

    A row is its subsection's number, the path of the object its table belongs to, the
    backticked names of its first column, and the text of its second column.
    """
    # 0 outside the subsections that hold the tables.
    subsection = 0
    path = ""
    for line in design_text.splitlines():
        heading = _SUBSECTION.fullmatch(line)
        if heading is not None:
            number = int(heading[1])
            subsection = number if _FIRST_TABLE_SUBSECTION <= number <= _CONFIG_SUBSECTION else 0
            named = _NAME.search(heading[2])
            path = "" if named is None else named[1]
        elif line.startswith("## "):
            # Section 11 ends the contract.
            subsection = 0
        elif subsection and (label := _TABLE_LABEL.match(line)) is not None:
            path = label[1]
        elif subsection and line.startswith("|"):
            first, second = (cell.strip() for cell in line.strip().strip("|").split("|")[:2])
            yield subsection, path, _NAME.findall(first), second


def snapshot_fields(snapshot: Mapping[str, object]) -> dict[str, frozenset[str]]:
    """Read the field names of every object path from a rendered /metrics snapshot.

    The paths are those of contract_fields. The entries of every robot id, model and process
    are read under one path, which holds the union of their fields; an entry that is null (a
    worker without a live process) adds none, and a path with no entry at all is absent.
    ValueError when a value that the contract makes an object is not one.
    """
    fields: defaultdict[str, set[str]] = defaultdict(set)
    _visit("", snapshot, lambda path, entry: fields[path].update(entry))
    return {path: frozenset(names) for path, names in fields.items()}


def type_violations(
    snapshot: Mapping[str, object], types: Mapping[str, Mapping[str, str]]
) -> list[str]:
    """Return every scalar of the snapshot whose JSON type is not the one types gives it.

    Each is named by its path and field, with the value found. ValueError as for
    snapshot_fields.
    """
    violations: list[str] = []

    def check(path: str, entry: Mapping[str, object]) -> None:
        for name, stated in types.get(path, {}).items():
            value = entry.get(name)
            if not (
                (value is None and stated.endswith(_OR_NULL))
                or type(value) is _SCALAR_TYPES[stated.removesuffix(_OR_NULL)]
            ):
                violations.append(f"{_join(path, name)}: {value!r} is not {stated}")

    _visit("", snapshot, check)
    return violations


def _visit(
    path: str, entry: Mapping[str, object], visit: Callable[[str, Mapping[str, object]], None]
) -> None:
    """Call visit with the entry and with every object inside it, each under its path."""
    visit(path, entry)
    for name, value in entry.items():
        child = _join(path, name)
        if child in _KEYED:
            for instance in _as_object(child, value).values():
                if instance is not None:
                    instance_path = f"{child}.{_KEYED[child]}"
                    _visit(instance_path, _as_object(instance_path, instance), visit)
        # Every object inside the config echo is a group of settings.
        elif child in _OBJECTS or (path == "config" and isinstance(value, Mapping)):
            _visit(child, _as_object(child, value), visit)


def _as_object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not an object in this snapshot")
    return value

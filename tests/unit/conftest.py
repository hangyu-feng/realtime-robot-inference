"""Fixtures shared by the unit tests of more than one module."""

from pathlib import Path

import pytest

from tests.support.runs import write_evidence


@pytest.fixture(scope="session")
def evidence_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a results root as make report leaves it, built once: its readers only read it."""
    root = tmp_path_factory.mktemp("evidence")
    write_evidence(root)
    return root

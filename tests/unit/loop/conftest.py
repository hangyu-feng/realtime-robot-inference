"""Runs every async test in this directory on the virtual-time loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping

import pytest

from tests.support.virtual_time import VirtualTimeLoop


def pytest_asyncio_loop_factories(
    config: pytest.Config, item: pytest.Item
) -> Mapping[str, Callable[[], asyncio.AbstractEventLoop]]:
    return {"virtual": VirtualTimeLoop}

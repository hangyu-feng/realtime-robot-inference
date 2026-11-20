"""Runs every async test in this directory on uvloop, the loop the services use."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Mapping

import pytest

if sys.platform == "linux":
    import uvloop

    def pytest_asyncio_loop_factories(
        config: pytest.Config, item: pytest.Item
    ) -> Mapping[str, Callable[[], asyncio.AbstractEventLoop]]:
        return {"uvloop": uvloop.new_event_loop}

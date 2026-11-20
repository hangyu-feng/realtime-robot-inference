"""The gateway process: logging, configuration, signals, the listener, the banner, the exit code.

Run as `python -m cloud.main`. The settings come from the environment (cloud/config.py). It
prints two things to standard output: `listening on HOST:PORT` once the socket is bound, and a
banner whose first line starts with `ready:` once robots are accepted. It exits with 0 after
SIGTERM or SIGINT, with 1 when a worker fails to start, and with 2 for a bad configuration or
an address that cannot be bound.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import random
import signal
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

from cloud.config import ServerConfig, Topology, load_config
from cloud.gateway import Gateway
from cloud.metrics import read_system_stats
from cloud.scheduler import ModelPlan
from cloud.worker_link import WorkerStartError, start_worker
from common import aio
from common.cgroup import Limits, format_limits, read_limits
from common.clock import MonotonicClock
from common.errors import ConfigError
from common.transport_ws import ServerOptions, serve_ws

logger = logging.getLogger(__name__)

_MS_PER_S = 1_000
_NS_PER_MS = 1_000_000


def _banner(config: ServerConfig, port: int, limits: Limits, plans: Mapping[str, ModelPlan]) -> str:
    """Return the lines printed when the gateway starts accepting.

    Every value is the running service's: the port bound, the batch caps of the calibration
    and the limits of the cgroup.
    """
    caps = ", ".join(f"{model} cap {plan.scheduler.batch_cap}" for model, plan in plans.items())
    if config.topology is Topology.SPLIT:
        robots = (
            "docker compose -f docker-compose.split.yml run --rm robots python -m robot.fleet "
            f"--url ws://system:{port}/ws --robots 3 --duration-s 30"
        )
    else:
        robots = "docker compose exec system python -m robot.fleet --robots 3 --duration-s 30"
    return "\n".join(
        (
            f"ready: listening on {config.host}:{port}; {caps}; {format_limits(limits)}",
            f"  metrics  http://localhost:{port}/metrics",
            f"  health   http://localhost:{port}/healthz",
            f"  robots   {robots}",
            "  tests    docker compose run --rm system pytest -q tests/integration",
        )
    )


def _server_options(config: ServerConfig) -> ServerOptions:
    return ServerOptions(
        host=config.host,
        port=config.port,
        max_message_bytes=config.max_message_bytes,
        receive_queue_frames=config.receive_queue_frames,
        rcvbuf_bytes=config.socket_rcvbuf_bytes,
        sndbuf_bytes=config.socket_sndbuf_bytes,
        ping_interval_s=config.ping_interval_ms / _MS_PER_S,
        ping_timeout_s=config.ping_timeout_ms / _MS_PER_S,
        # The library's open timeout bounds the whole HTTP exchange of a connection, a
        # /metrics render included.
        open_timeout_s=config.handshake_timeout_ms / _MS_PER_S,
        close_timeout_s=config.close_timeout_ms / _MS_PER_S,
    )


async def _serve(config: ServerConfig, limits: Limits, version: str) -> int:
    """Run the gateway until a signal stops it, and return the process's exit code."""
    loop = asyncio.get_running_loop()
    # The statistics of a scrape are read in this executor, never on the loop. It has one
    # thread, so the reads happen one at a time.
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1, thread_name_prefix="stats"))
    stop = asyncio.Event()
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signal_number, stop.set)
    clock = MonotonicClock()
    gateway = Gateway(
        config,
        clock,
        lambda: time.time_ns() // _NS_PER_MS,
        limits,
        version,
        functools.partial(start_worker, config=config, clock=clock),
        read_system_stats,
        random.Random(),
    )
    try:
        listener = await serve_ws(gateway.handle, gateway.http, _server_options(config))
    except OSError as error:
        print(f"gateway: cannot listen on {config.host}:{config.port}: {error}", file=sys.stderr)
        return 2
    logger.info(
        "socket buffers granted: receive %d bytes, send %d bytes",
        listener.buffers.rcvbuf_bytes,
        listener.buffers.sndbuf_bytes,
    )
    # The port bound, which is the kernel's choice when the configured port is 0.
    print(f"listening on {config.host}:{listener.port}", flush=True)

    def announce(plans: Mapping[str, ModelPlan]) -> None:
        print(_banner(config, listener.port, limits, plans), flush=True)

    exit_code = 0
    try:
        await gateway.run(listener, stop, announce)
    except* WorkerStartError as failures:
        for failure in failures.exceptions:
            print(f"gateway: {failure}", file=sys.stderr)
        exit_code = 1
        # A failed start ends run() before its first shutdown stage.
        listener.stop_accepting()
    try:
        async with asyncio.timeout(config.close_timeout_ms / _MS_PER_S):
            await listener.wait_closed()
    except TimeoutError:
        logger.warning("the listener did not close within %g ms", config.close_timeout_ms)
    return exit_code


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # The library logs every connection and every plain HTTP reply at INFO, which would be a
    # line per scrape.
    logging.getLogger("websockets").setLevel(logging.WARNING)
    try:
        config = load_config(os.environ)
    except ConfigError as error:
        print(f"gateway: {error}", file=sys.stderr)
        return 2
    limits = read_limits()
    logger.info("%s", format_limits(limits))
    # The image sets the version from its build argument; a local run has none.
    version = os.environ.get("RRI_VERSION", "unknown")
    return aio.run(_serve(config, limits, version))


if __name__ == "__main__":
    sys.exit(main())

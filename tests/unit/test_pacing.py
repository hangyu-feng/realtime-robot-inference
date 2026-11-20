"""Tests for the feed-forward pacing share and its application to the paced scheduler."""

from dataclasses import replace
from typing import Any

import pytest

from cloud.config import ServerConfig
from cloud.dispatch import ModelDispatch
from cloud.mailbox import Frame, Mailbox
from cloud.pacing import Pacing
from cloud.registry import RobotRegistry
from cloud.scheduler import Dispatch, WaitUntil
from common.commands import MODEL_KEYS
from common.worker_protocol import WorkerReady

NS_PER_MS = 1_000_000
QUOTA_CORES = 2.0
# No frame I/O cost and half a core of headroom keep every share an exact binary fraction.
CONFIG = ServerConfig(io_cpu_ms_per_frame=0.0, pacing_headroom_cores=0.5)
JPEG = b"\xff\xd8"
# A robot declares 25 fps, so it offers cls 25 frames per second. At the 10 ms of CPU time of a
# cls batch of one that is a quarter core per robot, while the wall time of a batch of one,
# 20 ms, would make it half a core, and the CPU time per frame of a batch of eight, 6.5 ms, less.
FRAME_PERIOD_NS = 40 * NS_PER_MS
CLS_READY = WorkerReady(
    model="cls",
    pid=101,
    model_file=None,
    model_sha256=None,
    threads=1,
    t_b_ns={1: 20 * NS_PER_MS, 2: 30 * NS_PER_MS, 4: 50 * NS_PER_MS, 8: 90 * NS_PER_MS},
    t_b_cpu_ns={1: 10 * NS_PER_MS, 2: 16 * NS_PER_MS, 4: 28 * NS_PER_MS, 8: 52 * NS_PER_MS},
)
DEPTH_READY = WorkerReady(
    model="depth",
    pid=102,
    model_file=None,
    model_sha256=None,
    threads=1,
    t_b_ns={1: 100 * NS_PER_MS, 2: 190 * NS_PER_MS, 4: 370 * NS_PER_MS, 8: 730 * NS_PER_MS},
    t_b_cpu_ns={1: 100 * NS_PER_MS, 2: 190 * NS_PER_MS, 4: 370 * NS_PER_MS, 8: 730 * NS_PER_MS},
)


# The changes are typed Any: mypy checks the value type of a ** dict passed to
# dataclasses.replace against every field, and the callers change fields of different types.
def with_model(config: ServerConfig, model: str, **changes: Any) -> ServerConfig:
    """Return config with the settings of one model changed."""
    return replace(
        config, models={**config.models, model: replace(config.models[model], **changes)}
    )


class World:
    """The registry, the mailbox and both models' dispatches, with the pacing over them."""

    def __init__(
        self, config: ServerConfig = CONFIG, quota_cores: float | None = QUOTA_CORES
    ) -> None:
        self.registry = RobotRegistry(retention_ns=NS_PER_MS, max_entries=16)
        self.mailbox = Mailbox(config.queue_policy, config.fifo_capacity_frames)
        self.dispatches = {
            "cls": ModelDispatch("cls", config, CLS_READY, self.mailbox, 0),
            "depth": ModelDispatch("depth", config, DEPTH_READY, self.mailbox, 0),
        }
        self.pacing = Pacing(config, quota_cores, self.dispatches)

    def join(self, robot_id: str) -> None:
        record = self.registry.connect(
            robot_id,
            MODEL_KEYS,
            {"cls": 500 * NS_PER_MS, "depth": 150 * NS_PER_MS},
            {"cls": 1_000 * NS_PER_MS, "depth": 300 * NS_PER_MS},
            FRAME_PERIOD_NS,
            0,
        )
        self.mailbox.open(record)
        self.pacing.update(self.registry.records())

    def leave(self, robot_id: str) -> None:
        self.mailbox.close(robot_id)
        self.registry.disconnect(robot_id, 0)
        self.pacing.update(self.registry.records())

    def held_until(self, model: str, robot_id: str, cpu_ns: int) -> int | None:
        """Run one batch of the robot's frame and return when pacing then releases the worker.

        The batch is dispatched at 1000 ns with cpu_ns of CPU time, and its worker asks again at
        1200 ns with a new frame pending. None when pacing does not hold the worker.
        """
        scheduler = self.dispatches[model].scheduler
        self.mailbox.put(robot_id, Frame(1, 0, 900, 0, 0, JPEG))
        decision = scheduler.next(1_000)
        assert isinstance(decision, Dispatch)
        scheduler.complete(decision.batch, (), 100, cpu_ns, 1_100)
        self.mailbox.put(robot_id, Frame(2, 0, 1_150, 0, 0, JPEG))
        decision = scheduler.next(1_200)
        assert isinstance(decision, WaitUntil | Dispatch)
        return decision.deadline_ns if isinstance(decision, WaitUntil) else None


def test_nothing_is_paced_before_the_first_update() -> None:
    world = World()

    assert world.pacing.shares() == {"cls": None, "depth": None}


@pytest.mark.parametrize(
    ("config", "quota_cores"),
    [
        pytest.param(CONFIG, None, id="no-cpu-quota"),
        pytest.param(with_model(CONFIG, "depth", paced=False), QUOTA_CORES, id="no-paced-model"),
    ],
)
def test_nothing_is_paced_without_a_cpu_quota_or_a_paced_model(
    config: ServerConfig, quota_cores: float | None
) -> None:
    world = World(config, quota_cores)
    for robot_id in "abcdef":
        world.join(robot_id)

    shares = world.pacing.shares()

    assert shares == {"cls": None, "depth": None}
    assert world.held_until("depth", "a", 750) is None


@pytest.mark.parametrize(
    ("depth_threads", "robots", "share_cores"),
    [
        pytest.param(1, 1, None, id="remainder-above-the-threads"),
        pytest.param(1, 2, None, id="remainder-at-the-threads"),
        pytest.param(1, 3, 0.75, id="remainder-below-the-threads"),
        pytest.param(2, 0, 1.5, id="remainder-below-two-threads"),
        pytest.param(2, 3, 0.75, id="remainder-below-two-threads-with-robots"),
    ],
)
def test_the_paced_worker_gets_a_share_only_when_the_remainder_is_below_its_threads(
    depth_threads: int, robots: int, share_cores: float | None
) -> None:
    # The quota is 2 cores and the headroom half a core; each robot's cls frames take a quarter
    # core, up to the one thread cls has.
    world = World(with_model(CONFIG, "depth", threads=depth_threads))
    world.pacing.update(world.registry.records())
    for robot_id in "abc"[:robots]:
        world.join(robot_id)

    assert world.pacing.shares() == {"cls": None, "depth": share_cores}


def test_the_share_never_falls_below_the_configured_minimum() -> None:
    # Two cls threads let six robots take 1.5 cores, which leaves the paced worker nothing.
    world = World(with_model(CONFIG, "cls", threads=2))
    for robot_id in "abcdef":
        world.join(robot_id)

    assert world.pacing.shares()["depth"] == CONFIG.pacing_min_share_cores


def test_the_share_follows_robots_joining_and_leaving() -> None:
    world = World()
    world.pacing.update(world.registry.records())
    shares = [world.pacing.shares()["depth"]]
    for robot_id in "abcde":
        world.join(robot_id)
        shares.append(world.pacing.shares()["depth"])
    for robot_id in "edc":
        world.leave(robot_id)
        shares.append(world.pacing.shares()["depth"])

    assert shares == [None, None, None, 0.75, 0.5, 0.5, 0.5, 0.75, None]


def test_the_share_counts_cls_frames_at_the_calibrated_cpu_time_of_a_batch_of_one() -> None:
    world = World()
    for robot_id in "abc":
        world.join(robot_id)

    # Three robots at a quarter core each leave 2 - 0.5 - 0.75 cores. At the wall time of a
    # batch of one the share would be 0.5, and at the CPU time per frame of a batch of eight
    # depth would not be paced.
    assert world.pacing.shares()["depth"] == 0.75


@pytest.mark.parametrize(
    ("robots", "release_ns"),
    [
        pytest.param(2, None, id="unpaced"),
        # 750 ns of CPU time at 0.75 cores hold the worker for 1000 ns after its dispatch.
        pytest.param(3, 2_000, id="three-quarters-of-a-core"),
        pytest.param(4, 2_500, id="half-a-core"),
    ],
)
def test_the_share_is_set_on_the_paced_scheduler(robots: int, release_ns: int | None) -> None:
    world = World()
    for robot_id in "abcd"[:robots]:
        world.join(robot_id)

    assert world.held_until("depth", "a", 750) == release_ns


def test_an_unpaced_scheduler_is_never_held() -> None:
    world = World()
    for robot_id in "abcd":
        world.join(robot_id)

    assert world.held_until("cls", "a", 750) is None

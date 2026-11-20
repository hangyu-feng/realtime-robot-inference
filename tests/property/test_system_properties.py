"""Property tests of the whole system in one process: random scripts of robots and workers.

Each example runs a script of joins, leaves, aborts, replacements, worker crashes and time
advances against the gateway with fake workers on a virtual-time loop, and reads /metrics
after every step.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st

from common.commands import MODEL_KEYS
from tests.support.system import InProcessSystem, identity_violations
from tests.support.virtual_time import run_virtual

SECOND = 1_000_000_000
# Depth is saturated and classification is not, so frames are overwritten, skipped, pending
# and in flight at every step.
FRAME_MS = {"cls": 2, "depth": 40}
# Few ids, so that a join often finds its id connected and replaces that session.
ROBOT_IDS = ("a", "b", "c")
# Longer than any script runs: a robot leaves only by a step or at the end of the script.
STREAM_NS = 60 * SECOND
MAX_STEPS = 16
MAX_ADVANCE_MS = 600


@dataclass(frozen=True)
class Join:
    """A robot starts now; under the id of a connected robot it replaces that session."""

    robot_id: str

    async def apply(self, system: InProcessSystem) -> None:
        now_ns = system.clock.now_ns()
        system.add_robot(self.robot_id, now_ns, now_ns + STREAM_NS)


@dataclass(frozen=True)
class Leave:
    robot_id: str

    async def apply(self, system: InProcessSystem) -> None:
        system.leave(self.robot_id)


@dataclass(frozen=True)
class Abort:
    robot_id: str

    async def apply(self, system: InProcessSystem) -> None:
        system.abort(self.robot_id)


@dataclass(frozen=True)
class Crash:
    model: str

    async def apply(self, system: InProcessSystem) -> None:
        system.crash_worker(self.model)


@dataclass(frozen=True)
class Advance:
    ms: int

    async def apply(self, system: InProcessSystem) -> None:
        await system.advance(self.ms / 1000)


Step = Join | Leave | Abort | Crash | Advance


@st.composite
def scripts(draw: st.DrawFn) -> list[Step]:
    """Draw a script whose leaves and aborts name only robots that an earlier step added."""
    steps: list[Step] = []
    added: list[str] = []
    for _ in range(draw(st.integers(min_value=1, max_value=MAX_STEPS))):
        choices: list[st.SearchStrategy[Step]] = [
            st.builds(Join, st.sampled_from(ROBOT_IDS)),
            st.builds(Crash, st.sampled_from(MODEL_KEYS)),
            st.builds(Advance, st.integers(min_value=1, max_value=MAX_ADVANCE_MS)),
        ]
        if added:
            choices += [
                st.builds(Leave, st.sampled_from(added)),
                st.builds(Abort, st.sampled_from(added)),
            ]
        step = draw(st.one_of(choices))
        if isinstance(step, Join):
            added = [*added, step.robot_id]
        steps.append(step)
    return steps


@dataclass(frozen=True)
class Reading:
    """What the snapshot after one step showed."""

    step: Step
    violations: list[str]
    internal_errors: int


async def play(script: list[Step]) -> list[Reading]:
    readings: list[Reading] = []
    async with InProcessSystem(FRAME_MS) as system:
        for step in script:
            await step.apply(system)
            snapshot = await system.metrics()
            readings.append(
                Reading(
                    step,
                    identity_violations(snapshot),
                    snapshot["connections"]["closed"]["internal_error"],
                )
            )
    return readings


@settings(max_examples=100)
@given(scripts())
def test_every_step_keeps_the_identities_and_no_session_ends_in_error(script: list[Step]) -> None:
    readings = run_virtual(play(script))
    assert [reading for reading in readings if reading.violations] == []
    assert [reading for reading in readings if reading.internal_errors != 0] == []

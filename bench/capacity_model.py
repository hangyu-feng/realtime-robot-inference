"""The capacity model: predictions of the demos from the real scheduler code on a simulated clock.

python -m bench.capacity_model --t-b PATH --out PATH [--seed N] reads one JSON file of measured
costs: per model, thread count and batch size, the wall and CPU time of one whole batch in ms;
the handling times outside the workers; and the CPU time per frame of the frame I/O. It
simulates the demos of DESIGN.md 9 and a sweep over robot counts with bench.simulation, under
the default server configuration, and writes one JSON object: the input's label, the inputs and
scheduler settings used, the predict.* fact keys of DESIGN.md 9, and the curves that the sweep
plots draw (DESIGN.md 6.3 and 6.5).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from bench.scenarios import Scenario, disconnect, late_joiner, massive, steady, sweep
from bench.simulation import Overheads, PhaseMode, ServiceTable, SimulationResult, simulate
from bench.stats import summarize
from cloud.config import ModelConfig, ServerConfig
from cloud.scheduler import pacing_release_ns
from common.commands import MODEL_KEYS

_NS_PER_S = 1_000_000_000
_NS_PER_MS = 1_000_000
_MS_PER_S = 1_000
# A time in ms becomes integer nanoseconds, which meet readings of the monotonic clock: an i64.
_MAX_CLOCK_NS = 2**63 - 1

# ASSUMPTION: A19. The CPU limit is a quota of 2 CPUs of time per period: Q of DESIGN.md 6.3, the
# limit every demo runs under.
QUOTA_CORES = 2.0
# The curves run from one robot to the robots of the massive demo.
_CURVE_ROBOT_COUNTS = range(1, len(massive().robots) + 1)
# The robot counts of the sweep demo (DESIGN.md 9.3).
_DEMO_SWEEP_COUNTS = (1, 3, 5, 10)
# The keys of each model's curves; each holds one value per robot count.
_CURVE_FIELDS = (
    "commands_per_s_per_robot",
    "freshness_ms_p50",
    "freshness_ms_p95",
    "command_gap_ms_max",
    "command_gap_bound_ms",
    "batch_size_mean",
    "utilization",
)


class CapacityInputError(Exception):
    """The input file lacks a key, holds a value of the wrong type, or holds a bad number."""


@dataclass(frozen=True)
class CapacityInput:
    """The content of the input file, in the units the simulation takes."""

    tables: Mapping[str, ServiceTable]  # every model, at its configured thread count
    overheads: Overheads
    io_cpu_ms_per_frame: float
    label: str


@dataclass(frozen=True)
class _Run:
    """A simulated scenario and its result."""

    scenario: Scenario
    result: SimulationResult


def command_gap_bound_ns(
    robot_count: int,
    batch_cap: int,
    frame_period_ns: int,
    cycle_ns: int,
    skip_max_unserved_ns: int | None,
) -> int:
    """Return the longest command gap of a robot that the scheduler guarantees (DESIGN.md 6.3).

    skip_max_unserved_ns is given when the skip rule is on, and None when it is off.
    """
    # Derivation. A robot served at a dispatch has its next frame pending at most one frame
    # period later. From then on every dispatch either includes it or serves batch_cap robots
    # that were ahead of it in the order, each of which then falls behind it, and at most
    # robot_count - 1 are ahead, so it is dispatched within ceil(robot_count / batch_cap)
    # cycles. A command gap is measured between two pops, which happen at the completions of the
    # two batches, and that adds at most one more cycle. With the skip rule on, a robot whose
    # frames are doomed can be passed over until it has gone skip_max_unserved_ns unserved
    # (guard (b)); from then on it cannot be skipped and the same argument runs from that
    # moment, which is the added term. Because the registry starts the gap at the connect and a
    # robot never served goes first, the same bound covers the wait from connect to the first
    # command sent, when the robot's first frame reaches the gateway within one frame period of
    # the connect. In the simulation a robot's first frame arrives up to one frame period plus
    # transport_up_ns after its start, and the robot receives a command deliver_ns after its
    # pop, so the interval from a robot's start to the first command it receives is at most this
    # bound plus transport_up_ns plus deliver_ns.
    #
    # Preconditions. Every robot sends one frame per frame period. Every cycle is at most
    # cycle_ns, where a cycle is the time from a dispatch until next() can dispatch again while
    # frames are pending: the largest, over the sizes up to the cap, of
    # max(dispatch_request + wall(b) + dispatch_result, ceil(cpu(b) / share)), the second term
    # only with a pacing share (longest_cycle_ns). next() is called at every completion, pacing
    # release and arrival while Idle. No worker fails and no command is dropped from a downlink
    # queue. robot_count is the number of robots connected at some time during the run, not the
    # largest number connected at once: a robot that joins has never been served and goes ahead
    # of the others once (one that reconnects keeps its record, and with it its place, for the
    # retention time). Where the share changes as robots join or leave, the cycle is the
    # longest.
    bound_ns = frame_period_ns + (math.ceil(robot_count / batch_cap) + 1) * cycle_ns
    return bound_ns if skip_max_unserved_ns is None else bound_ns + skip_max_unserved_ns


def longest_cycle_ns(
    table: ServiceTable, overheads: Overheads, batch_cap: int, shares_cores: Collection[float]
) -> int:
    """Return the longest time from a dispatch until the scheduler can dispatch again.

    A batch is back dispatch_request_ns + wall + dispatch_result_ns after its dispatch, and
    under a pacing share the worker is also held until the batch's CPU time over the share has
    passed since its dispatch. Every size up to the cap and every share the run set count.
    """
    longest_ns = 0
    for size in range(1, batch_cap + 1):
        wall_ns, cpu_ns = table.at(size)
        back_ns = overheads.dispatch_request_ns + wall_ns + overheads.dispatch_result_ns
        held_ns = [pacing_release_ns(0, cpu_ns, share_cores) for share_cores in shares_cores]
        longest_ns = max(longest_ns, back_ns, *held_ns)
    return longest_ns


def predict(
    config: ServerConfig,
    tables: Mapping[str, ServiceTable],
    overheads: Overheads,
    quota_cores: float,
    seed: int,
    robot_counts: Sequence[int],
    label: str,
) -> dict[str, object]:
    """Simulate the demos and a sweep at each robot count, and build the predictions."""

    def run(scenario: Scenario) -> _Run:
        # Every run draws its robots' phases at random from the seed, as the demos run.
        result = simulate(config, scenario, tables, overheads, quota_cores, PhaseMode.RANDOM, seed)
        return _Run(scenario, result)

    sweeps = {count: run(sweep(count)) for count in sorted({*_DEMO_SWEEP_COUNTS, *robot_counts})}
    steady_run = run(steady())
    return {
        "label": label,
        "inputs": _inputs(config, tables, overheads, quota_cores, seed, robot_counts, steady_run),
        "facts": _facts(steady_run, run(late_joiner()), sweeps, run(disconnect()), run(massive())),
        "curves": _curves(config, tables, overheads, quota_cores, robot_counts, sweeps),
    }


def _inputs(
    config: ServerConfig,
    tables: Mapping[str, ServiceTable],
    overheads: Overheads,
    quota_cores: float,
    seed: int,
    robot_counts: Sequence[int],
    steady_run: _Run,
) -> dict[str, object]:
    """Return what the predictions were made from: the inputs, and the plan of each model."""
    plans = {model: steady_run.result.models[model].plan for model in MODEL_KEYS}
    return {
        "seed": seed,
        "quota_cores": quota_cores,
        "io_cpu_ms_per_frame": config.io_cpu_ms_per_frame,
        "overheads_ms": {
            "transport_up": overheads.transport_up_ns / _NS_PER_MS,
            "dispatch_request": overheads.dispatch_request_ns / _NS_PER_MS,
            "dispatch_result": overheads.dispatch_result_ns / _NS_PER_MS,
            "deliver": overheads.deliver_ns / _NS_PER_MS,
        },
        "tables": {
            model: {
                "threads": config.models[model].threads,
                "wall_ms": {
                    str(size): ns / _NS_PER_MS for size, ns in sorted(tables[model].wall_ns.items())
                },
                "cpu_ms": {
                    str(size): ns / _NS_PER_MS for size, ns in sorted(tables[model].cpu_ns.items())
                },
            }
            for model in MODEL_KEYS
        },
        "plans": {
            model: {
                "batch_cap": plan.scheduler.batch_cap,
                "service_ms": plan.scheduler.service_ns / _NS_PER_MS,
                "return_allowance_ms": plan.scheduler.return_allowance_ns / _NS_PER_MS,
                "skip_rule": plan.scheduler.skip_rule,
                "skip_max_unserved_ms": plan.scheduler.skip_max_unserved_ns / _NS_PER_MS,
                "budget_ms": plan.budget_ns / _NS_PER_MS,
                "margin_ms": plan.margin_ns / _NS_PER_MS,
                "feasible": plan.feasible,
            }
            for model, plan in plans.items()
        },
        "robot_counts": list(robot_counts),
    }


def _facts(
    steady_run: _Run,
    late_joiner_run: _Run,
    sweeps: Mapping[int, _Run],
    disconnect_run: _Run,
    massive_run: _Run,
) -> dict[str, float | None]:
    """Return the predict.* fact keys of DESIGN.md 9."""
    facts: dict[str, float | None] = {}
    for model in MODEL_KEYS:
        steady_robots = steady_run.result.windows["all"][model].values()
        freshness_ms = [ns / _NS_PER_MS for robot in steady_robots for ns in robot.freshness_ns]
        for name in ("p50", "p95", "p99", "max"):
            facts[f"predict.steady.{model}.freshness_ms.{name}"] = _statistic(freshness_ms, name)
        facts[f"predict.steady.{model}.commands_per_s"] = sum(
            robot.commands_per_s for robot in steady_robots
        )
        peak_ages_ms = [
            ns / _NS_PER_MS for robot in steady_robots for ns in robot.command_age_peak_ns
        ]
        facts[f"predict.steady.{model}.command_age_peak_ms.p95"] = _statistic(peak_ages_ms, "p95")
        after = late_joiner_run.result.windows["after"][model]
        facts[f"predict.late_joiner.after.a.{model}.commands_per_s"] = after["a"].commands_per_s
        for count in _DEMO_SWEEP_COUNTS:
            facts[f"predict.sweep.n{count}.{model}.commands_per_s_per_robot.mean"] = _statistic(
                list(_commands_per_s(sweeps[count], "all", model).values()), "mean"
            )
        facts[f"predict.massive.{model}.commands_per_s_per_robot.mean"] = _statistic(
            list(_commands_per_s(massive_run, "all", model).values()), "mean"
        )
    survivors = {plan.robot_id for plan in disconnect_run.scenario.robots if plan.stop_s is None}
    for window_name in ("before", "after"):
        commands_per_s = _commands_per_s(disconnect_run, window_name, "depth")
        survivors_per_s = [
            per_s for robot_id, per_s in commands_per_s.items() if robot_id in survivors
        ]
        facts[f"predict.disconnect.{window_name}.survivors.depth.commands_per_s_per_robot"] = (
            _statistic(survivors_per_s, "mean")
        )
    return facts


def _curves(
    config: ServerConfig,
    tables: Mapping[str, ServiceTable],
    overheads: Overheads,
    quota_cores: float,
    robot_counts: Sequence[int],
    sweeps: Mapping[int, _Run],
) -> dict[str, object]:
    """Return the curves over robot counts: one value per count in each list."""
    curves: dict[str, object] = {"robots": list(robot_counts)}
    for model in MODEL_KEYS:
        points = [
            _curve_point(config, tables[model], overheads, quota_cores, model, sweeps[count])
            for count in robot_counts
        ]
        curves[model] = {field: [point[field] for point in points] for field in _CURVE_FIELDS}
    curves["cpu_demand_cores"] = [sweeps[count].result.cpu_demand_cores for count in robot_counts]
    return curves


def _curve_point(
    config: ServerConfig,
    table: ServiceTable,
    overheads: Overheads,
    quota_cores: float,
    model: str,
    sweep_run: _Run,
) -> dict[str, float | None]:
    """Return one model's curve values at the robot count of one sweep run."""
    robot_count = len(sweep_run.scenario.robots)
    robots = sweep_run.result.windows["all"][model].values()
    model_run = sweep_run.result.models[model]
    scheduler = model_run.plan.scheduler
    freshness_ms = [ns / _NS_PER_MS for robot in robots for ns in robot.freshness_ns]
    shares_cores = [
        share.share_cores for share in model_run.shares if share.share_cores is not None
    ]
    # The period every simulated robot declares.
    frame_period_ns = round(_NS_PER_S / config.frame_rate_hz)
    bound_ns = command_gap_bound_ns(
        robot_count,
        scheduler.batch_cap,
        frame_period_ns,
        longest_cycle_ns(table, overheads, scheduler.batch_cap, shares_cores),
        scheduler.skip_max_unserved_ns if scheduler.skip_rule else None,
    )
    frames = sum(size * count for size, count in model_run.batch_size.items())
    batches = sum(model_run.batch_size.values())
    return {
        "commands_per_s_per_robot": _statistic(
            list(_commands_per_s(sweep_run, "all", model).values()), "mean"
        ),
        "freshness_ms_p50": _statistic(freshness_ms, "p50"),
        "freshness_ms_p95": _statistic(freshness_ms, "p95"),
        "command_gap_ms_max": max(robot.command_gap_max_ns for robot in robots) / _NS_PER_MS,
        "command_gap_bound_ms": bound_ns / _NS_PER_MS,
        "batch_size_mean": frames / batches if batches else None,
        "utilization": _utilization(config, quota_cores, model, sweep_run, frames),
    }


def _utilization(
    config: ServerConfig, quota_cores: float, model: str, sweep_run: _Run, frames: int
) -> float | None:
    """Return a model's utilization in a sweep run; None when it served no frame."""
    # Demand over capacity, as DESIGN.md 6.3 and 10.5 define it: N * f / K_m, with the capacity
    # K_m = 1000 * s_m / c_m frames per second. It is not the busy fraction of the worker: it
    # keeps rising with N after the model saturates, and per-robot throughput leaves the frame
    # rate where it passes 1.
    if frames == 0:
        return None
    model_run = sweep_run.result.models[model]
    # c_m: the CPU ms per served frame of this run.
    cpu_ms_per_frame = model_run.cpu_ns / frames / _NS_PER_MS
    # s_m: the pacing share with all N robots connected, when there is one. Every robot of a
    # sweep starts at 0 s and none stops, so that is the last share set. Otherwise, for a model
    # that is not paced or that gets no share at this N, the cores the worker can use: its
    # threads, or the quota when the runtime chooses its thread count (threads 0).
    share_cores = model_run.shares[-1].share_cores
    if share_cores is None:
        threads = config.models[model].threads
        share_cores = min(threads, quota_cores) if threads > 0 else quota_cores
    robot_count = len(sweep_run.scenario.robots)
    return robot_count * config.frame_rate_hz * cpu_ms_per_frame / (_MS_PER_S * share_cores)


def _commands_per_s(run: _Run, window_name: str, model: str) -> dict[str, float]:
    """Return, in robot order, the commands per second of each robot that ran all the window."""
    window = next(window for window in run.scenario.windows if window.name == window_name)
    return {
        plan.robot_id: run.result.windows[window_name][model][plan.robot_id].commands_per_s
        for plan in run.scenario.robots
        if plan.start_s <= window.start_s and (plan.stop_s is None or plan.stop_s >= window.end_s)
    }


def _statistic(samples: Sequence[float], name: str) -> float | None:
    """Return one statistic of bench.stats.summarize; None for no samples, which it refuses."""
    return summarize(samples)[name] if samples else None


def parse_input(data: bytes, models: Mapping[str, ModelConfig]) -> CapacityInput:
    """Read the input file's bytes for the configured models.

    Every model of MODEL_KEYS needs an entry for its configured thread count that holds every
    size of its batch_sizes. Other thread counts are not read, and other sizes are kept for the
    interpolation. CapacityInputError names the key path of anything missing or bad: nothing
    gets a default.
    """
    try:
        document = json.loads(data)
    except (ValueError, RecursionError) as error:
        # ValueError covers bad syntax, bytes that are not UTF-8, and an integer of more than
        # 4300 digits; RecursionError, brackets nested too deep.
        raise CapacityInputError(f"the input is not a JSON document: {error}") from error
    label = _lookup(document, ("label",))
    if not isinstance(label, str):
        raise CapacityInputError("label: must be a string")
    return CapacityInput(
        tables={model: _service_table(document, model, models[model]) for model in MODEL_KEYS},
        overheads=Overheads(
            transport_up_ns=_ns(document, ("overheads_ms", "transport_up")),
            dispatch_request_ns=_ns(document, ("overheads_ms", "dispatch_request")),
            dispatch_result_ns=_ns(document, ("overheads_ms", "dispatch_result")),
            deliver_ns=_ns(document, ("overheads_ms", "deliver")),
        ),
        io_cpu_ms_per_frame=_milliseconds(document, ("io_cpu_ms_per_frame",)),
        label=label,
    )


def _service_table(document: object, model: str, model_config: ModelConfig) -> ServiceTable:
    """Return one model's batch times at its configured thread count."""
    path = ("models", model, "threads", str(model_config.threads))
    entries = _lookup(document, path)
    if not isinstance(entries, dict):
        raise CapacityInputError(f"{_dotted(path)}: must be an object")
    wall_ns: dict[int, int] = {}
    cpu_ns: dict[int, int] = {}
    for key in entries:
        size = _batch_size(path, key)
        wall_ns[size] = _ns(document, (*path, key, "wall_ms"))
        cpu_ns[size] = _ns(document, (*path, key, "cpu_ms"))
    for size in model_config.batch_sizes:
        if size not in wall_ns:
            raise CapacityInputError(f"{_dotted((*path, str(size)))}: missing")
    return ServiceTable(wall_ns, cpu_ns)


def _batch_size(path: tuple[str, ...], key: str) -> int:
    """Return the batch size a key names, written in decimal without a sign or leading zeros."""
    message = f"{_dotted((*path, key))}: must name a batch size, a whole number from 1"
    # int() also refuses a run of more than 4300 digits.
    try:
        size = int(key)
    except ValueError as error:
        raise CapacityInputError(message) from error
    # One size has one key: int() would also read "01", "+1" and " 1".
    if size < 1 or str(size) != key:
        raise CapacityInputError(message)
    return size


def _ns(document: object, path: tuple[str, ...]) -> int:
    return round(_milliseconds(document, path) * _NS_PER_MS)


def _milliseconds(document: object, path: tuple[str, ...]) -> float:
    """Return the time in ms at the key path."""
    value = _lookup(document, path)
    # json.loads reads true and false as bools, which Python counts as ints.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CapacityInputError(f"{_dotted(path)}: must be a number")
    # json.loads reads NaN, Infinity and 1e999 as nan and inf, which fail this test, and the
    # time must stay a finite number of nanoseconds within the clock's range.
    if not 0 <= value * _NS_PER_MS <= _MAX_CLOCK_NS:
        raise CapacityInputError(
            f"{_dotted(path)}: must be finite, 0 or more, and at most {_MAX_CLOCK_NS} ns"
        )
    return float(value)


def _lookup(document: object, path: tuple[str, ...]) -> object:
    """Return the value at the key path, or name the first step that is missing or no object."""
    value = document
    for depth, key in enumerate(path):
        if not isinstance(value, dict):
            raise CapacityInputError(f"{_dotted(path[:depth])}: must be an object")
        if key not in value:
            raise CapacityInputError(f"{_dotted(path[: depth + 1])}: missing")
        value = value[key]
    return value


def _dotted(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "the input"


def main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m bench.capacity_model",
        description="Predict the demos from measured batch times and write one JSON file.",
    )
    parser.add_argument(
        "--t-b",
        type=Path,
        required=True,
        metavar="PATH",
        help="JSON file of the batch times, the overheads and the I/O CPU time per frame",
    )
    parser.add_argument(
        "--out", type=Path, required=True, metavar="PATH", help="path of the JSON predictions"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed of the robots' phases (default: %(default)s)"
    )
    arguments = parser.parse_args(argv)
    # Checked before the runs: a path that cannot take the predictions would otherwise surface
    # only once they are made.
    if arguments.out.is_dir() or not arguments.out.parent.is_dir():
        parser.error(f"argument --out: {arguments.out} is not a file in an existing directory")
    try:
        data = arguments.t_b.read_bytes()
    except OSError as error:
        parser.error(f"argument --t-b: {error}")
    default = ServerConfig()
    try:
        measured = parse_input(data, default.models)
    except CapacityInputError as error:
        parser.error(f"{arguments.t_b}: {error}")
    # ASSUMPTION: A1. The predictions are for the single topology, in which the robot fleet runs
    # under the same quota as the gateway, so the measured I/O cost per frame covers both. The
    # file is the one source of that cost, and the simulation reads it from the configuration only.
    config = replace(default, io_cpu_ms_per_frame=measured.io_cpu_ms_per_frame)
    predictions = predict(
        config,
        measured.tables,
        measured.overheads,
        QUOTA_CORES,
        arguments.seed,
        _CURVE_ROBOT_COUNTS,
        measured.label,
    )
    arguments.out.write_text(json.dumps(predictions, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])

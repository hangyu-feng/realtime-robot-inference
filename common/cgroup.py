"""Read the container's CPU and memory limits from its own cgroup files.

Supports cgroup v2 and falls back to v1. The limits are logged at startup and
checked by the benchmark gate, so every result records the limits it ran under.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")

# cgroup v1 reports "no limit" as a huge page-rounded byte count, not a keyword.
_V1_UNLIMITED_BYTES = 1 << 60


@dataclass(frozen=True)
class Limits:
    """Limits as seen from inside the container. None means unlimited or unknown."""

    version: int
    cpu_cores: float | None
    cpu_period_us: int | None
    memory_bytes: int | None
    swap_bytes: int | None  # swap allowed on top of memory; reported by v2 only


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def parse_cpu_max(text: str) -> tuple[float | None, int]:
    """Parse cgroup v2 ``cpu.max``: "<quota|max> <period>", both in microseconds."""
    quota, period = text.split()
    period_us = int(period)
    if quota == "max":
        return None, period_us
    return int(quota) / period_us, period_us


def parse_memory_max(text: str) -> int | None:
    """Parse a cgroup v2 memory limit: a byte count, or "max" for no limit."""
    return None if text == "max" else int(text)


def _read_v2(root: Path) -> Limits:
    cpu_cores: float | None = None
    cpu_period_us: int | None = None
    cpu_text = _read(root / "cpu.max")
    if cpu_text:
        cpu_cores, cpu_period_us = parse_cpu_max(cpu_text)
    memory_text = _read(root / "memory.max")
    swap_text = _read(root / "memory.swap.max")
    return Limits(
        version=2,
        cpu_cores=cpu_cores,
        cpu_period_us=cpu_period_us,
        memory_bytes=parse_memory_max(memory_text) if memory_text else None,
        swap_bytes=parse_memory_max(swap_text) if swap_text else None,
    )


def _read_v1(root: Path) -> Limits:
    quota_text = _read(root / "cpu" / "cpu.cfs_quota_us")
    period_text = _read(root / "cpu" / "cpu.cfs_period_us")
    cpu_period_us = int(period_text) if period_text else None
    cpu_cores: float | None = None
    if quota_text and cpu_period_us and int(quota_text) > 0:  # -1 means no quota
        cpu_cores = int(quota_text) / cpu_period_us
    memory_text = _read(root / "memory" / "memory.limit_in_bytes")
    memory_bytes = int(memory_text) if memory_text else None
    if memory_bytes is not None and memory_bytes >= _V1_UNLIMITED_BYTES:
        memory_bytes = None
    return Limits(
        version=1,
        cpu_cores=cpu_cores,
        cpu_period_us=cpu_period_us,
        memory_bytes=memory_bytes,
        swap_bytes=None,
    )


def read_limits(root: Path = CGROUP_ROOT) -> Limits:
    """Return the limits of the cgroup this process runs in."""
    if (root / "cgroup.controllers").exists():
        return _read_v2(root)
    return _read_v1(root)


def format_limits(limits: Limits) -> str:
    cpu = "unlimited" if limits.cpu_cores is None else f"{limits.cpu_cores:g} CPUs"
    memory = "unlimited" if limits.memory_bytes is None else f"{limits.memory_bytes / 2**30:g} GiB"
    return f"cgroup v{limits.version}: cpu limit {cpu}, memory limit {memory}"


def main() -> None:
    print(format_limits(read_limits()))


if __name__ == "__main__":
    main()

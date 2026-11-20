"""Read one process's CPU, memory, thread and descriptor counters from /proc.

The gateway reports these for itself and its workers in the metrics endpoint,
and the benchmark runner records the same fields for the robot fleet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROC_ROOT = Path("/proc")

_MS_PER_S = 1000

# proc(5) numbers the fields of /proc/<pid>/stat from 1, and the state is field 3. The
# fields are split from the state onwards, so each index is the proc(5) number minus 3.
_USER_TICKS_INDEX = 11
_SYSTEM_TICKS_INDEX = 12
_NICE_INDEX = 16
_THREADS_INDEX = 17
_VIRTUAL_MEMORY_BYTES_INDEX = 20
_RSS_PAGES_INDEX = 21


@dataclass(frozen=True)
class ProcessStats:
    """Counters of one live process. CPU times cover all its threads and advance by clock ticks."""

    pid: int
    cpu_user_ms: int
    cpu_system_ms: int
    # proc(5) calls this value inaccurate: the kernel gives up accuracy in the counter behind
    # it to scale across CPUs. The exact value is in smaps_rollup, which the kernel fills by
    # walking every mapping of the process.
    rss_bytes: int
    threads: int
    fds: int  # a process that reads itself also counts the descriptor used for the listing
    nice: int


class ProcessGoneError(Exception):
    """The process has exited: it is being torn down, is a zombie, or was reaped."""


def _read_stat_fields(pid: int, process_dir: Path) -> list[bytes]:
    """Read the fields after the command name. Raise ProcessGoneError for an exited process."""
    try:
        stat = (process_dir / "stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError) as error:
        raise ProcessGoneError(f"process {pid} does not exist") from error
    # The command name can itself contain spaces and parentheses, so the fields are counted
    # from its closing parenthesis, which is the last one in the file.
    fields = stat[stat.rindex(b")") + 1 :].split()
    # A dying process gives up its address space before it closes its descriptors, and its
    # state reads as zombie only after both. Its virtual size is zero from the first step
    # until it is reaped. So is its RSS, but that counter is approximate and can read zero
    # for a live process too. A kernel thread never has an address space, so it is reported
    # as gone as well: this module is for user processes.
    if int(fields[_VIRTUAL_MEMORY_BYTES_INDEX]) == 0:
        raise ProcessGoneError(f"process {pid} has exited")
    return fields


def read_process_stats(pid: int, proc_root: Path = PROC_ROOT) -> ProcessStats:
    """Read the counters of one process. Raise ProcessGoneError if it has exited."""
    process_dir = proc_root / str(pid)
    try:
        fds = len(os.listdir(process_dir / "fd"))
    except FileNotFoundError as error:
        raise ProcessGoneError(f"process {pid} does not exist") from error
    except PermissionError:
        # The fd directory of a process without an address space belongs to root. Reading
        # the stat file raises ProcessGoneError if that is why the listing was refused, and
        # a live process that the caller may not inspect stays a permission error.
        _read_stat_fields(pid, process_dir)
        raise
    # The stat file is read after the listing: a process that still has its address space
    # then had not begun to close its descriptors when they were counted.
    fields = _read_stat_fields(pid, process_dir)
    clock_ticks_per_s = os.sysconf("SC_CLK_TCK")
    return ProcessStats(
        pid=pid,
        cpu_user_ms=int(fields[_USER_TICKS_INDEX]) * _MS_PER_S // clock_ticks_per_s,
        cpu_system_ms=int(fields[_SYSTEM_TICKS_INDEX]) * _MS_PER_S // clock_ticks_per_s,
        rss_bytes=int(fields[_RSS_PAGES_INDEX]) * os.sysconf("SC_PAGE_SIZE"),
        threads=int(fields[_THREADS_INDEX]),
        fds=fds,
        nice=int(fields[_NICE_INDEX]),
    )

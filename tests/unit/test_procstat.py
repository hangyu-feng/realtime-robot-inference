"""Tests for reading per-process counters from a fake /proc tree."""

import errno
import os
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

from common.procstat import ProcessGoneError, ProcessStats, read_process_stats

PID = 4321
USER_MS = 7000
SYSTEM_MS = 2250
VIRTUAL_MEMORY_BYTES = 5988352
RSS_PAGES = 403
THREADS = 4
NICE = 10


def write_stat(
    proc_root: Path,
    command: bytes = b"python",
    state: bytes = b"S",
    nice: int = NICE,
    virtual_memory_bytes: int = VIRTUAL_MEMORY_BYTES,
    rss_pages: int = RSS_PAGES,
) -> None:
    """Write /proc/PID/stat. Every field the reader uses differs from its neighbours."""
    clock_ticks_per_s = os.sysconf("SC_CLK_TCK")
    user_ticks = USER_MS * clock_ticks_per_s // 1000
    system_ticks = SYSTEM_MS * clock_ticks_per_s // 1000
    (proc_root / str(PID) / "stat").write_bytes(
        b"%d (%b) %b 824 825 826 34816 827 4194304 92 1 2 3 %d %d 11 13 30 %d %d 0 3535 "
        b"%d %d 18446744073709551615 1 2 3 4 5 0 0 0 0 0 0 0 17 13 0 0 0 0 0 6 7 8 9 "
        b"10 11 12 0\n"
        % (
            PID,
            command,
            state,
            user_ticks,
            system_ticks,
            nice,
            THREADS,
            virtual_memory_bytes,
            rss_pages,
        )
    )


def write_stat_of_dying_process(proc_root: Path, state: bytes) -> None:
    """Write /proc/PID/stat as it reads once a process has given up its address space."""
    write_stat(proc_root, state=state, virtual_memory_bytes=0, rss_pages=0)


def write_process(
    proc_root: Path, command: bytes = b"python", nice: int = NICE, fd_count: int = 3
) -> None:
    """Write the fake /proc/PID of a live process."""
    fd_dir = proc_root / str(PID) / "fd"
    fd_dir.mkdir(parents=True)
    for fd in range(fd_count):
        (fd_dir / str(fd)).touch()
    write_stat(proc_root, command=command, nice=nice)


def refuse_listing(path: Path) -> NoReturn:
    """Answer like the kernel when the reader may not list the descriptors of a process."""
    raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))


def test_read_process_stats_converts_ticks_and_pages(tmp_path: Path) -> None:
    write_process(tmp_path)

    assert read_process_stats(PID, tmp_path) == ProcessStats(
        pid=PID,
        cpu_user_ms=USER_MS,
        cpu_system_ms=SYSTEM_MS,
        rss_bytes=RSS_PAGES * os.sysconf("SC_PAGE_SIZE"),
        threads=THREADS,
        fds=3,
        nice=NICE,
    )


@pytest.mark.parametrize(
    "command",
    [b"tmux: server", b"(sd-pam)", b"a) R (b", b") S 1 2 3 (", b"caf\xe9 \xff"],
    ids=["spaces", "parentheses", "fake-state", "fake-fields", "not-utf-8"],
)
def test_command_name_does_not_shift_the_fields(tmp_path: Path, command: bytes) -> None:
    write_process(tmp_path, command=command)

    stats = read_process_stats(PID, tmp_path)

    assert (stats.cpu_user_ms, stats.cpu_system_ms, stats.nice, stats.threads) == (
        USER_MS,
        SYSTEM_MS,
        NICE,
        THREADS,
    )


@pytest.mark.parametrize("fd_count", [0, 1, 64])
def test_fds_counts_the_entries_of_the_fd_directory(tmp_path: Path, fd_count: int) -> None:
    write_process(tmp_path, fd_count=fd_count)

    assert read_process_stats(PID, tmp_path).fds == fd_count


@pytest.mark.parametrize("nice", [-20, -5, 0, 19])
def test_nice_keeps_its_sign(tmp_path: Path, nice: int) -> None:
    write_process(tmp_path, nice=nice)

    assert read_process_stats(PID, tmp_path).nice == nice


def test_live_process_whose_rss_reads_zero_is_not_gone(tmp_path: Path) -> None:
    write_process(tmp_path)
    write_stat(tmp_path, rss_pages=0)

    assert read_process_stats(PID, tmp_path).rss_bytes == 0


def test_process_that_does_not_exist_raises_process_gone(tmp_path: Path) -> None:
    with pytest.raises(ProcessGoneError, match=f"{PID} does not exist") as raised:
        read_process_stats(PID, tmp_path)

    assert isinstance(raised.value.__cause__, FileNotFoundError)


def test_process_reaped_after_its_descriptors_were_listed_raises_process_gone(
    tmp_path: Path,
) -> None:
    write_process(tmp_path)
    (tmp_path / str(PID) / "stat").unlink()

    with pytest.raises(ProcessGoneError, match=f"{PID} does not exist") as raised:
        read_process_stats(PID, tmp_path)

    assert isinstance(raised.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("state", [b"R", b"Z"], ids=["exiting", "zombie"])
def test_process_without_an_address_space_raises_process_gone(tmp_path: Path, state: bytes) -> None:
    write_process(tmp_path)
    write_stat_of_dying_process(tmp_path, state)

    with pytest.raises(ProcessGoneError, match=f"{PID} has exited"):
        read_process_stats(PID, tmp_path)


@pytest.mark.parametrize("state", [b"R", b"Z"], ids=["exiting", "zombie"])
def test_process_killed_before_its_descriptors_are_listed_raises_process_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: bytes
) -> None:
    write_process(tmp_path)

    def refuse_listing_of_process_killed_just_now(path: Path) -> NoReturn:
        write_stat_of_dying_process(tmp_path, state)
        refuse_listing(path)

    monkeypatch.setattr(os, "listdir", refuse_listing_of_process_killed_just_now)

    with pytest.raises(ProcessGoneError, match=f"{PID} has exited"):
        read_process_stats(PID, tmp_path)


def test_process_reaped_after_its_listing_was_refused_raises_process_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "listdir", refuse_listing)

    with pytest.raises(ProcessGoneError, match=f"{PID} does not exist") as raised:
        read_process_stats(PID, tmp_path)

    assert isinstance(raised.value.__cause__, FileNotFoundError)


def test_live_process_whose_descriptors_may_not_be_listed_raises_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_process(tmp_path)
    monkeypatch.setattr(os, "listdir", refuse_listing)

    with pytest.raises(PermissionError):
        read_process_stats(PID, tmp_path)


@pytest.mark.skipif(sys.platform != "linux", reason="only Linux has /proc/<pid>/stat")
def test_read_process_stats_of_this_process_from_the_real_proc() -> None:
    stats = read_process_stats(os.getpid())

    assert stats.pid == os.getpid()
    assert stats.rss_bytes > 0
    assert stats.threads >= 1
    assert stats.fds == len(os.listdir("/proc/self/fd"))
    assert stats.nice == os.getpriority(os.PRIO_PROCESS, 0)


@pytest.mark.skipif(sys.platform != "linux", reason="only Linux has /proc/<pid>/stat")
def test_zombie_on_the_real_proc_raises_process_gone() -> None:
    with subprocess.Popen([sys.executable, "-c", "pass"]) as child:
        # Returns once the child has exited, and leaves it a zombie.
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)

        with pytest.raises(ProcessGoneError, match=f"{child.pid} has exited"):
            read_process_stats(child.pid)

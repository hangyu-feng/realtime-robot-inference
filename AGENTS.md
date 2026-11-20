# AGENTS.md

Guidance for coding agents working in this repository. The project brief is in [take-home-assignment.md](take-home-assignment.md).

## Code guidelines

Python 3.14, checked by `ruff`, `mypy --strict` and `pytest`. These rules cover what the tools cannot check.

### Structure

- ALWAYS keep scheduling, queueing, protocol and robot-control logic free of I/O: no sockets, no `asyncio`, no real clock. Pass time in as a value: the I/O layer reads its clock once per event and passes the reading (`now_ns`).
- ALWAYS keep the I/O layers thin. They move bytes, call the logic and record metrics.
- PREFER one job per module and per function. Reconsider a function that passes about 40 lines or takes a flag that switches its behaviour.
- ALWAYS validate untrusted input once, at the boundary where it enters, and trust the types after that.
- ALWAYS send typed messages between processes, encoded and decoded by the protocol module. NEVER pickle arbitrary objects across a process boundary.
- ALWAYS keep tunable values in config, with the unit in the name (`idle_timeout_ms`, `max_message_bytes`).
- ALWAYS mark an assumption about the brief with a comment that starts with `ASSUMPTION:`, at the place where it applies.

### Simplicity

- NEVER add a trivial helper or wrapper that has a single caller. Inline it.
- AVOID an abstraction before it has a second real use.
- NEVER add options, parameters or features that nothing uses yet.
- NEVER write defensive code for a state the types or validation already rule out. If such a state must be handled, raise. NEVER substitute a default.
- PREFER plain functions and dataclasses over class hierarchies.
- NEVER leave commented-out code, debug prints or unused imports.

### Naming and comments

- ALWAYS use full words in names (`version`, not `ver`), and put the unit in the name of a quantity (`freshness_ms`, `capture_ns`, `frame_bytes`).
- ALWAYS make a comment give a reason or a constraint that the code cannot show. NEVER restate the next line.
- NEVER add divider comments, change history ("now fixed", "previously") or anything about how the code came to be written.
- ALWAYS give a module a short docstring that states its responsibility. Give a function a docstring only when its name and types do not say enough: one summary line in the imperative, no repeated types.

### Errors

- ALWAYS catch the specific exceptions a call can raise, in the smallest `try` block that works. NEVER use a bare `except`.
- NEVER catch `Exception` except at an isolation boundary (one connection, one frame, one worker), where the failure is counted and logged.
- ALWAYS translate errors from libraries and peers into this project's own exception types at the boundary, with `raise ... from`.
- ALWAYS fail at startup, with a clear message, on a bad config or a missing model. NEVER continue on a guess.
- NEVER silence a lint or type error. If one must be suppressed, scope it to the line and give the reason.

### asyncio and processes

- ALWAYS give every task an owner: start it in a `TaskGroup`, or keep a reference and handle its result.
- NEVER swallow `CancelledError`. Clean up in `finally` and let it propagate.
- ALWAYS set a timeout at the call site, with `asyncio.timeout()`, on every await that depends on a peer or another process. An await that this process itself completes (an Event, a task it owns) takes none and carries a comment naming what completes it.
- NEVER block the event loop. Inference and image decoding run in the worker processes, and nothing on the frame path writes a file or sleeps.
- ALWAYS bound every queue and buffer, and say in a comment what happens when it is full.
- ALWAYS measure durations with the monotonic clock. NEVER subtract wall-clock times.
- NEVER log per frame. Count it in a metric, and log state changes only.
- ALWAYS shut down in stages: stop accepting, drain within a deadline, cancel what is left.

### Tests

- ALWAYS test logic directly, with time given as values or a fake clock, and no event loop. NEVER sleep in a test.
- PREFER property tests for invariants (conservation, fairness, ordering) and `pytest.mark.parametrize` for repeated cases. NEVER branch inside a test.
- ALWAYS name a test after the behaviour it checks, and assert at the public boundary of the unit.
- PREFER fakes with real behaviour over mocks.
- ALWAYS write the failing test before fixing a bug.

Run `make check` before handing work back. It must pass with no new warnings.

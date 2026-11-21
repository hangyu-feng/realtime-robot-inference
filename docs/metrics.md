# Metrics contract

This file is section 10 of the [design document](../DESIGN.md), kept in a file of its own for its length. Its headings keep the design document's numbering: a reference to a subsection of section 10 points into this file, a reference to section 11 points to the [run-output contract](run-output.md), and every other section reference, like "this document", means the design document.

## 10. Metrics contract

`GET /metrics` returns one JSON object. This section is the contract for it: a change to a field changes this section in the same commit, and a test compares a rendered snapshot with the tables below. When four scrapes are already being served, or the cgroup and process files cannot be read within one second, `/metrics` answers 503 with `{"status": "busy"}` or `{"status": "stats unavailable"}`. `GET /healthz` returns 200 with `{"status": "ok"}` while the gateway accepts connections and every worker is ready, and 503 with `{"status": "unavailable", "accepting": ..., "workers": {...}}` otherwise: during warm-up, during shutdown, and while a worker restarts, when connected robots keep their sessions and new robots are still accepted.

### 10.1 Conventions

- **Kinds.** A counter only grows, from gateway start. A gauge is the value now. A histogram is cumulative. The scraper reads `/metrics` once per second and computes rates and window values from differences.
- **Units** are in the field name: `_ms`, `_us`, `_ns`, `_s`, `_bytes`.
- **Model keys** are `cls` and `depth`. `<m>` stands for either.
- **Time histograms** (type H below) use fixed log-scale buckets, 8 per octave, from 0.01 ms to 60 000 ms. Bucket i covers values from 0.01 x 2^(i/8) up to 0.01 x 2^((i+1)/8). Values below the range go to bucket 0 and values above it to the last bucket. A histogram is serialized as `{"count": int, "sum": float, "min": float or null, "max": float or null, "buckets": {"<i>": int}}`, with only the non-empty buckets listed; `min` and `max` are null while the histogram is empty. A quantile read from it is the upper edge of the bucket that holds the sample of rank ceil(q x count), so for a value inside the range it is at or above that sample and less than one bucket width, about 9%, above it. The percentiles in this document's tables interpolate between neighbouring samples, so where the samples around the rank are far apart, as in a sparse tail, the two can differ by more. `count`, `sum`, `min` and `max` are exact.
- **Exact percentiles** in this document come from the robots' per-command samples, not from these histograms. The histograms are for live inspection and as a cross-check.
- **Size bound.** Entries under `robots` are kept for a retention time after a disconnect and are capped in number, oldest disconnected first.
- **Cost.** `/metrics` is served from the gateway's event loop. The snapshot is built from counters and histograms that are already aggregated, so a scrape does not stall the loop: the cgroup and process files are read in a separate thread, a robot's telemetry is kept as the object it is published as, and the robot entries are serialized in chunks with a yield to the loop between chunks.
- **Consistency.** Each robot entry is read in one step, so the identities of section 10.3 hold for every robot in every snapshot. Totals elsewhere in the snapshot may differ from the sum over robots by what happened while the robot entries were being serialized.

### 10.2 Top level

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `schema` | int | constant | Schema version, 1 |
| `ts_mono_ns` | int | gauge | Gateway monotonic clock at the snapshot |
| `ts_wall_ms` | int | gauge | Wall clock at the snapshot, UTC epoch |
| `uptime_s` | float or null | gauge | Time since the gateway began accepting connections; null during warm-up |
| `config` | object | constant | Config echo, section 10.8 |
| `robots` | object keyed by robot id | | Section 10.3 and 10.4 |
| `models` | object keyed by model | | Section 10.5 |
| `stages` | object keyed by model | | Section 10.6 |
| `connections` | object | | Section 10.7 |
| `workers` | object keyed by model | | Section 10.7 |
| `cgroup` | object | | Section 10.7 |
| `processes` | object keyed by process name | | Section 10.7 |
| `loop` | object | | Section 10.7 |

### 10.3 `robots.<id>`: server-side fields

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `session` | int | gauge | Session number, raised on every connect of this id |
| `connected` | bool | gauge | |
| `connected_s` | float | gauge | Length of the current or last session |
| `subscribed` | list of model keys | constant | From the handshake |
| `stale_ms` | object, model to float | constant | Thresholds declared in the handshake |
| `watchdog_ms` | object, model to float | constant | Watchdog limits declared in the handshake |
| `rtt_ms` | float or null | gauge | Last ping round trip, read when the robot's telemetry report arrives, in whole milliseconds (the resolution of the event loop's clock); null until the session's first one returns, and while the last one measured 0 at that resolution, which on one host is usual |
| `bytes_in`, `bytes_out` | int | counter | Message bytes received from and sent to this robot; a text message counts its characters, which are its bytes for valid telemetry |
| `frames_in` | int | counter | Binary messages received in its sessions, plus each message the WebSocket library refused for its size |
| `frames_rejected` | int | counter | Refused at ingress: malformed (dropped; the connection stays open), oversized or over the rate limit (both close the connection) |
| `frames_accepted` | int | counter | Placed in the slot |
| `models.<m>` | object | | Per-model counters, below |
| `telemetry` | object | | Robot-reported, section 10.4 |

`robots.<id>.models.<m>`:

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `frames_received` | int | counter | Frames accepted while subscribed to this model |
| `frames_served` | int | counter | Frames sent to the worker |
| `frames_overwritten` | int | counter | Replaced in the slot before this model served them |
| `frames_skipped` | int | counter | Passed over by the skip rule |
| `frames_dropped_disconnect` | int | counter | Still unserved when the robot disconnected |
| `frames_pending` | int | gauge | 1 if the slot holds a frame this model has not finished with, else 0 |
| `in_flight` | int | gauge | Frames at the worker now |
| `commands_queued` | int | gauge | Commands waiting in the robot's downlink queue now |
| `commands_sent` | int | counter | Commands handed to the robot's connection. A command whose write fails because the connection has just closed is still counted, at most one per session |
| `commands_dropped_downlink` | int | counter | Dropped because the downlink queue was full |
| `results_dropped_disconnect` | int | counter | Results that came back after the robot had gone, or were still queued for it when it went |
| `frames_failed` | int | counter | Served with no command: decode error, inference error, worker crash, inference timeout, or a result the gateway refused |
| `last_served_age_ms` | float or null | gauge | Time since this model last served this robot; null until it first does |
| `max_service_gap_ms` | float | gauge | Longest time this robot went without a command within a session: from the session's start or a command to the next command, or, for the last interval, to now while the robot is connected and to its disconnect after it |

Conservation identities. Tests assert them, and every run checks them at its end:

```
frames_in       = frames_rejected + frames_accepted
frames_received = frames_served + frames_overwritten + frames_skipped
                  + frames_dropped_disconnect + frames_pending
frames_served   = commands_sent + commands_queued + commands_dropped_downlink
                  + results_dropped_disconnect + frames_failed + in_flight
```

Under the FIFO control policy, `frames_pending` is the queue length and `frames_overwritten` counts frames dropped because the queue was full.

### 10.4 `robots.<id>.telemetry`: robot-reported fields

The robot sends these at 1 Hz, after a frame and only when nothing is unsent, and once more when it stops on purpose, after it has stopped taking commands, so the last report covers exactly the commands the session received. Counters and histograms restart with each session, except `reconnects`. The fields are absent until the session's first report.

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `ts_mono_ns` | int | gauge | Robot clock at the report |
| `frames_captured` | int | counter | Ticks that came due in the session, including ticks passed while the simulator woke late |
| `frames_sent` | int | counter | |
| `frames_skipped_send` | int | counter | Ticks not sent: the previous frame was still unsent, or the simulator woke after the next tick was due. `frames_captured = frames_sent + frames_skipped_send` |
| `fire_lag_ms` | H | histogram | Send start minus scheduled capture time. The first latency stage, and the validity check for the simulator |
| `reconnects` | int | counter | Sessions after the first in this robot process |
| `models.<m>.commands_received` | int | counter | |
| `models.<m>.commands_applied` | int | counter | |
| `models.<m>.commands_stale` | int | counter | Rejected: older than the stale threshold at receipt |
| `models.<m>.commands_out_of_order` | int | counter | Rejected: frame number not above the last applied |
| `models.<m>.freshness_ms` | H | histogram | Receive time minus capture time, all received commands |
| `models.<m>.command_gap_ms` | H | histogram | Time between consecutive received commands |
| `models.<m>.command_age_peak_ms` | H | histogram | Age of the command in effect at the moment it is replaced |
| `models.<m>.deliver_ms` | H | histogram | Gateway result time to robot receive time. Valid on a shared clock only |
| `models.<m>.watchdog_trips` | int | counter | Times the state changed to expired |
| `models.<m>.watchdog_expired_ms` | int | counter | Total time without an unexpired command: before the first applied command, or in the expired state |
| `models.<m>.watchdog_state` | string | gauge | `ok`, `expired`, or `none` before the first command |

`commands_received = commands_applied + commands_stale + commands_out_of_order`.

### 10.5 `models.<m>`: scheduling and batching

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `batches` | int | counter | Batches dispatched |
| `frames` | int | counter | Frames served |
| `batch_size` | object, size to int | counter | Number of batches of each size |
| `batch_ms` | H | histogram | Worker wall time per batch, from batch received to results sent |
| `busy_ms` | int | counter | Wall time the worker spent on batches |
| `paced_ms` | int | counter | Time a ready worker was held back by pacing |
| `idle_ms` | int | counter | Time a ready worker had no eligible frame. `paced_ms` and `idle_ms` are booked at the dispatch that ends the wait |
| `eligible` | int | gauge | Robots with an unserved frame now |

The measured CPU cost per frame of a model is its worker's CPU time (`processes.worker_<m>`) divided by `frames`. Measured capacity is the worker's CPU share divided by that cost. Utilization is demand divided by capacity.

### 10.6 `stages.<m>`: latency stages seen by the server

One histogram (type H) per stage, per model. Section 7 defines the stages. A command's stages are recorded when the command is queued for its robot, so the histograms describe the commands robots can receive, not frames whose robot had already gone. `fire_lag_ms` and `deliver_ms` are measured by the robot and are in section 10.4.

| Field | Meaning |
|---|---|
| `transport_up_ms` | Robot send start to frame received. Exact when robot and gateway share a clock (`config.shared_clock`); otherwise half the ping round trip |
| `slot_wait_ms` | Frame received to frame selected |
| `dispatch_rt_ms` | The two IPC legs: selected to batch at the worker, plus results sent to results at the gateway |
| `decode_ms` | JPEG decode of this frame |
| `preprocess_ms` | Resize, colour conversion, normalization, layout |
| `batch_peers_ms` | Worker time spent on the other frames of the same batch |
| `inference_ms` | Assembling the batch tensor and the model run, for the batch |
| `postprocess_ms` | Output to command |

### 10.7 Connections, workers, cgroup, processes, loop

`connections`:

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `current` | int | gauge | Open robot connections |
| `accepted` | int | counter | Handshakes completed |
| `closed` | object, reason to int | counter | Reasons: `client_close`, `abnormal`, `idle_timeout`, `slow_consumer`, `ingress_limit`, `protocol_error`, `replaced`, `shutdown`, `internal_error` |
| `rejected` | object, reason to int | counter | Reasons: `capacity`, `version`, `auth`, `bad_hello`, `unknown_model`, `handshake_timeout` |
| `frames_rejected` | object, reason to int | counter | Reasons: `malformed` (a bad header, a wrong session number, a frame number not above the last accepted one, or implausible timestamps), `oversize`, `rate_limited` |
| `bytes_in`, `bytes_out` | int | counter | All robot connections |

Every accepted session ends once, with one reason, so `accepted = current + sum(closed)`. `internal_error` counts sessions ended by a fault in the gateway's own code and is 0 in a correct run. A peer that leaves before its hello, or says hello after shutdown has begun, is neither accepted nor rejected. Every reason is present from the start, at 0.

`workers.<m>`:

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `state` | string | gauge | `warming` (the first start, until its calibration arrives), `ready`, `restarting` (after a failure, while a new process starts and warms up), `failed` (a restart attempt failed; the next one follows a capped backoff) |
| `pid` | int or null | gauge | Null while no worker process is live |
| `restarts` | int | counter | |
| `model_file`, `model_sha256` | string or null | constant | The file served; null for the fake and null model sets, which load no file, and before the first calibration |
| `threads` | int | constant | Intra-op threads |
| `batch_sizes` | list of int | constant | Sizes calibrated at warm-up |
| `t_b_ms` | object, size to float, or null | constant | Calibrated wall time per batch, from the first warm-up of the run (a restarted worker's calibration is logged, not adopted); null before it. When depth's published file is served at cap 1, size 1 is measured on that file and the larger sizes on the patched copy |
| `t_b_cpu_ms` | object, size to float, or null | constant | Calibrated CPU time per batch, from the same warm-up; null before it |
| `batch_cap` | int or null | constant | Cap fixed for the run; null before the first calibration |
| `pacing_share` | float or null | gauge | CPU share this worker is paced to now, in cores; recomputed when a robot joins or leaves; null when it is not paced |
| `margin_ms` | float or null | constant | The model's configured stale threshold minus one frame period, the service time at the batch cap and the overhead allowance; null before the first calibration |
| `feasible` | bool or null | constant | False when the service time alone exceeds the threshold; null before the first calibration |
| `timeouts` | int | counter | Inference timeouts |
| `errors` | object, kind to int | counter | Frames that failed: `decode` (the frame did not decode), `inference` (the batch's inference raised an error; the worker kept serving) |

`cgroup`, read from the container's own cgroup files (v2, with a v1 fallback). A value the kernel does not expose is null: cgroup v1 has no socket-memory counter, `memory.peak` needs Linux 5.19 or later, and a cgroup without the cpu or memory controller has none of that controller's counters.

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `version` | int | constant | 1 or 2 |
| `cpu_limit_cores` | float or null | constant | Quota divided by period; null when unlimited |
| `cpu_period_us` | int or null | constant | |
| `cpuset` | string or null | constant | Effective CPU list |
| `cpu_usage_us`, `cpu_user_us`, `cpu_system_us` | int or null | counter | |
| `nr_periods`, `nr_throttled` | int or null | counter | Quota periods elapsed, and periods in which the container was throttled |
| `throttled_us` | int or null | counter | Total time throttled |
| `memory_limit_bytes` | int or null | constant | |
| `memory_swap_limit_bytes` | int or null | constant | |
| `memory_current_bytes` | int or null | gauge | Includes page cache |
| `memory_peak_bytes` | int or null | gauge | |
| `memory_anon_bytes` | int or null | gauge | |
| `memory_file_bytes` | int or null | gauge | |
| `memory_sock_bytes` | int or null | gauge | |
| `oom_kills` | int or null | counter | |

`processes.<p>`, for `gateway`, `worker_cls` and `worker_depth`. A worker's entry is null while it has no live process; its counters restart with each new process, which `workers.<m>.restarts` counts:

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `pid` | int | gauge | |
| `cpu_user_ms`, `cpu_system_ms` | int | counter | |
| `rss_bytes` | int | gauge | |
| `threads` | int | gauge | |
| `fds` | int | gauge | Open file descriptors |
| `nice` | int | gauge | |

The robot fleet is not a child of the gateway. The scenario driver, which runs beside the fleet, records the same fields for every fleet process and for itself in each scrape line (section 11.3).

`loop`:

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `lag_ms` | H | histogram | Lateness of a periodic timer in the gateway's event loop, sampled only while at least one robot is connected, so an idle gateway runs no timer and idle samples do not dilute it. One stall that passes several periods gives one sample. Lateness below about 1 ms is the resolution of the loop's timers |
| `tasks` | int | gauge | Live asyncio tasks |

### 10.8 `config`: config echo

| Field | Meaning |
|---|---|
| `version` | Git commit of the running code |
| `protocol_version` | 1 |
| `listen` | `host`, `port` |
| `topology` | `single` or `split` |
| `shared_clock` | Whether robot and gateway timestamps are comparable |
| `queue_policy`, `fifo_capacity_frames` | `latest` or `fifo`, and the queue bound per robot and model under `fifo` |
| `frame_rate_hz`, `overhead_allowance_ms` | The frame rate and the handling allowance the batch-cap budget is computed with |
| `skip_rule`, `skip_max_unserved_ms` | Whether the skip rule is on, and how long it may pass over a robot |
| `models.<m>` | `runtime`, `threads`, `batch_sizes`, `forced_batch_cap` (null unless the batch-cap comparison sets it), `paced`, `stale_ms`, `watchdog_ms`, `fake_frame_ms` |
| `pacing` | `headroom_cores`, `min_share_cores`, `io_cpu_ms_per_frame` |
| `worker_nice` | Niceness of the worker processes |
| `max_connections` | Connection cap |
| `auth_required` | Whether a shared token is configured |
| `ingress` | `max_messages_per_s`, `max_bytes_per_s`, `burst_s`, `max_message_bytes`, `max_frame_age_ms` |
| `timeouts_ms` | `handshake`, `idle`, `inference`, `slow_consumer`, `warmup`, `shutdown_grace`, `close`, `ping`, `worker_stop` |
| `socket` | `rcvbuf_bytes`, `sndbuf_bytes` |
| `receive_queue_frames` | High-water mark of the WebSocket library's receive queue, in frames |
| `downlink_queue_commands` | Commands held per connection |
| `calibration_runs` | Timed runs per batch size at warm-up |
| `worker_restart_ms` | `min`, `max`: the bounds of the backoff between restart attempts |
| `ping_interval_ms` | Period of the WebSocket ping |
| `robot_retention_s`, `max_robot_entries` | Bounds on the `robots` table |
| `loop_probe_ms` | Period of the event-loop lag probe |

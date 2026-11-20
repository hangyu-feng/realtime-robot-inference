# Design: real-time inference for a robot fleet

> **Status: skeleton.** This document fixes the structure, the claims, the experiments and the metrics contract before the system is built. Nothing in it has been measured by a gated run yet. Every measured value is written as a fact key in double braces and will be filled from files under `results/` written by gated benchmark runs. A run is gated when five gates pass: the test suite passes inside the image the run uses, with no skipped test (the tests that need the host's docker CLI are not selected); the image was built in the same invocation and every container of the run has its digest; the container reports the 2-CPU and 4 GB limits, and no swap, from inside; the run starts in fresh containers that accepted no connection before it began; and the configuration the gateway reports in `/metrics` is the one the run was planned with. Every run records a digest of the code that decides what it does, its commit, and whether the working tree was dirty. A fact counts only gated runs of the planned spec and code digest, and states how many of them come from a dirty tree. An interrupted or failed run is kept as an attempt and counted, never used. Every statement about performance below is a hypothesis until its placeholder is filled.

Status labels used in this document:

- **planned**: the experiment is defined and has not run.
- **preliminary**: a number exists but does not come from a gated run.
- **measured**: the number comes from a gated run under `results/`.
- **not measured**: out of scope or not possible here. Section 13 lists these.

Contents:

1. Overview
2. Evidence plan
3. Assumptions
4. Tool choice
5. Frame management
6. Batching, scheduling and scaling
7. Latency breakdown at 5 robots
8. The 50-robot mechanism
9. The five demos
10. Metrics contract
11. Run-output contract
12. Tried and rejected
13. Limitations and what is not measured
14. Beyond one container
15. Reproduction

## 1. Overview

### 1.1 What is built

Robots stream camera frames to a cloud service. The service runs two small CPU models on the frames and sends each result back as a control command. Robots produce frames faster than the models can process them, so the system decides which frames to process, keeps command freshness bounded and shares capacity fairly.

Three components run in one container limited to 2 CPUs and 4 GB:

- **Robot fleet** (`robot/`). Simulated robots. Each streams 640x480 JPEG frames at 30 fps, and every frame carries its frame number and capture time. Each robot receives commands, rejects stale ones and counts them.
- **Cloud gateway** (`cloud/`). Accepts robot connections, keeps the newest frame of each robot, schedules frames to the model workers, sends commands back and serves `/metrics`.
- **Model workers** (`workers/`). One process per model, both on ONNX Runtime, CPU only: MobileNetV3-small for classification (224x224 input, model key `cls`) and MiDaS-small for depth (256x256 input, model key `depth`).

Stack: Python 3.14, asyncio with uvloop, WebSocket binary frames, ONNX Runtime, and a scheduler and batcher written for this system.

### 1.2 Processes and data path

```
 one container: 2 CPUs, 4 GB

 +---------------------+    frames (WebSocket, binary)    +--------------------------------------------+
 | robot fleet process | -------------------------------> | gateway process (asyncio + uvloop)         |
 |   N robot tasks     | <------------------------------- |   reader per connection -> slot per robot  |
 +---------------------+            commands              |   scheduler per model: pull, fair, batched |
                                                           |   /metrics and /healthz on the same port   |
                                                           +----------+----------------------^----------+
                                                  batch of JPEG bytes |                      | results, stage timings,
                                                    (Unix socket)     v                      | "ready for the next batch"
                                                           +---------------------+  +---------------------+
                                                           | worker cls          |  | worker depth        |
                                                           | decode, preprocess, |  | decode, preprocess, |
                                                           | MobileNetV3, command|  | MiDaS, command      |
                                                           +---------------------+  +---------------------+
```

| Process | Count | Threads doing work | Role |
|---|---|---|---|
| gateway | 1 | 1 (event loop) | connections, slots, scheduling, result routing, metrics |
| worker `cls` | 1 | 1 inference thread | decode, preprocess, MobileNetV3-small, command |
| worker `depth` | 1 | 1 inference thread (2 in one experiment) | decode, preprocess, MiDaS-small, command |
| robot fleet | 1, plus a single-robot process in the kill demo | 1 (event loop) | N simulated robots |

The path of one frame, with the latency stage each step produces (section 7 defines the stages):

1. Robot tick k is scheduled at t0 + k/30 s. The robot takes the next pre-encoded JPEG, stamps the frame number and the scheduled capture time, and sends it. If the previous frame is still unsent, it skips this one. Stage `fire_lag`.
2. The gateway reader receives the message, validates the header and overwrites that robot's slot. Stage `transport_up`.
3. When a worker reports ready, the scheduler for that model picks robots that have an unserved frame, least recently served first, up to the batch cap. Stage `slot_wait`.
4. The gateway sends the batch (headers and raw JPEG bytes) to the worker over a Unix socket. First leg of stage `dispatch_rt`.
5. The worker decodes, preprocesses, runs the model and turns each output into a command. Stages `decode`, `preprocess`, `batch_peers`, `inference`, `postprocess`.
6. The worker returns the results with their stage timings and, in the same message, asks for the next batch. Second leg of `dispatch_rt`.
7. The gateway routes each command to its robot's connection. Stage `deliver`.
8. The robot receives the command, rejects it if it is stale or out of order, and otherwise applies it. Freshness is receive time minus scheduled capture time, on the robot's clock.

### 1.3 Design rules

1. Frames wait in exactly one place: one slot per robot, and the newest frame wins. No other queue holds frames.
2. Late binding. A frame is chosen at the moment a worker is free, never earlier.
3. Workers pull. The gateway never pushes work to a busy worker.
4. Each model serves robots least recently served first. Sending faster buys no extra share.
5. A dropped frame is never decoded.
6. Every buffer has a bound.
7. All state is soft. Nothing is retried and nothing is persisted.
8. CPU is budgeted against the 2-CPU quota, so the kernel does not have to freeze the container.
9. Batch time is measured at warm-up and the batch cap is fixed for the run. At most one feedback loop runs at runtime, and by default none does.

### 1.4 Bounds and failure handling

- **Bounded everywhere.** One slot per robot. A small bounded downlink queue per connection that drops the oldest command. A maximum message size. A connection cap, and a cap on connections that have not yet said hello. At most a few `/metrics` responses at once. Handshake and idle timeouts. Fixed-size histograms. Per-robot metric entries are capped and expire, so reconnect churn cannot grow memory.
- **Input validation.** Bad header, wrong protocol version, unknown model, zero-byte, truncated or corrupt JPEG, wrong resolution. One bad frame fails alone. At the gateway, a frame whose header fails validation (malformed, a wrong session number, a frame number not above the last accepted one, implausible timestamps) is dropped and counted, and the connection stays open; a malformed text message closes the connection. In the worker, frames are decoded one by one, so a frame cannot fail its batch or crash a worker. The image size declared in the JPEG header is checked before decoding, so a frame cannot make a worker allocate more than one frame's worth of pixels.
- **Fault isolation.** Each connection has its own exception boundary: around its task, and around each command write that a model's dispatcher makes for it. Workers are supervised: a crashed worker is restarted and its in-flight batch is dropped, not retried; a restart that fails is tried again after a capped backoff for as long as the gateway runs. A batch whose inference raises an error fails alone and the worker keeps serving. Inference has a timeout. If one model is down, the other keeps serving.
- **Slow consumer.** A robot that stops reading cannot grow the server's write buffer: a command is written only when the transport holds nothing unsent, and the others wait in the bounded downlink queue. Once the kernel refuses a command, the robot is disconnected if the backlog has not cleared within a time limit. How soon the kernel refuses depends on the socket buffers of both ends, which are sized for commands.
- **Half-open connections.** A peer that vanishes without closing is found by the idle timeout. The frame stream is the heartbeat. In the other direction, a robot whose frames stay unsent for a stall timeout ends the session and reconnects; commands are not a heartbeat, because a starved model legitimately sends none.
- **Disconnect race.** A result that returns after its robot has gone is dropped and counted.
- **Reconnect.** The robot backs off exponentially with jitter. A second connection with the same robot id replaces the first session, which is closed with 4001; a robot closed that way does not reconnect, so two processes with one id cannot evict each other for ever. A second hello that admission refuses leaves the first session untouched. Frame numbers may restart, so ordering is checked per session, and messages carry the session number.
- **Startup and shutdown.** Connections are accepted only after both models are loaded and warmed (`/healthz`). The models warm up one at a time. On SIGTERM the gateway stops accepting, ends every session and lets the batches in flight finish within a grace period, then aborts what is left and stops the workers, all within the container's stop timeout. A model that fails to load fails fast with a clear error. Workers exit when their gateway connection closes.
- **Idle.** With no robots connected, nothing spins.

### 1.5 Soft state: availability in the cloud, freshness on the robot

- The cloud chooses availability. Slots, the connection table and metrics are all soft state, and the next frame arrives one frame period later anyway. So the service is crash-only: no retries, no persistence, no promise to process every frame. Frames and commands are delivered at most once.
- The robot chooses freshness. A stale command is rejected, not applied late. During a partition no new command arrives, the watchdog expires, and the robot falls back to its own safe behaviour.
- Partition tolerance comes for free. Sessions are independent. A partitioned robot stops sending, so its share is released at once. When the link returns it reconnects, and there is no state to reconcile.
- One ordering guarantee is kept: per robot and per model, commands are applied in increasing frame number.
- Cross-model consistency is given up. The classification and depth commands a robot holds may come from different frames. Waiting to pair them would tie the fast model to the slow one. Every command carries its frame number, so a robot can align them if it needs to.
- Between gateway and workers the same reasoning applies. A dead worker is an internal partition: the other model keeps serving and the in-flight batch is not retried.

## 2. Evidence plan

One row per design question of the brief (Q), per demo (D), for the topology comparison (T) and for three non-functional requirements (R). Scenarios are defined in section 11, metric fields in section 10. Every status is "planned" until its runs exist.

| ID | Claim | Strongest objection | Experiment | Metric fields | Plot or table | Status |
|---|---|---|---|---|---|---|
| Q1a Language and framework | Python is enough here. Inference runs in native code in separate processes. The Python share (frame I/O and scheduling in one event loop) at 50 robots is `{{massive.gateway.cpu_cores}}` cores, with event-loop lag p99 of `{{massive.gateway.loop_lag_ms.p99}}` ms. | One Python event loop cannot move 1500 frames per second without adding latency. A compiled gateway is the obvious choice. | `massive`; `massive` with the null model to isolate transport and scheduling | `processes.gateway.*`, `loop.lag_ms`, `connections.bytes_in`, `stages.<m>.transport_up_ms` | `massive.md`, `sweep_cpu.png` | planned |
| Q1b Transport | WebSocket over TCP fits this container: message framing, close codes and an HTTP endpoint on one port, at an uplink cost of `{{massive.stage.transport_up_ms.p50}}` ms per frame. Its weakness, head-of-line blocking, needs packet loss and so does not appear at 50 robots on loopback. | TCP delivers every frame in order, including frames that are already stale. Datagrams would let stale data die in the network. | `massive`; delay sweep with the link model; a loss run with `tc netem` in the `split` topology if a root container with NET_ADMIN is added for it | `stages.<m>.transport_up_ms`, `robots.<id>.models.<m>.frames_overwritten`, telemetry `freshness_ms` | `delay_sweep.png`, `massive.md` | planned |
| Q1c Batcher | A pull-based scheduler with a measured batch cap fits one inference thread per model. The cap is the largest calibrated batch size whose batch time fits the model's freshness budget, and batch size 1 is measured against it: `{{model_bench.cls.batch_cap}}` for classification and `{{model_bench.depth.batch_cap}}` for depth. | On one inference thread a larger batch may add latency without adding throughput. | `model_bench`; the comparison of batch cap 1 against the default cap; batch-size distributions from `sweep` | `workers.<m>.t_b_ms`, `workers.<m>.t_b_cpu_ms`, `models.<m>.batch_size`, `models.<m>.batch_ms` | `model_tb.png`, `sweep_batch_size.png` | planned |
| Q2 Frame management | One newest-frame slot per robot with late binding keeps freshness bounded at any load: at most one frame period of slot wait, plus one service time, plus overhead. Thresholds are 150 ms for depth and 500 ms for classification. Stale ratio in the steady demo: `{{steady.depth.stale_ratio}}` for depth and `{{steady.cls.stale_ratio}}` for classification. | A short FIFO queue is simpler and loses fewer frames. And 150 ms may be out of reach for depth on slower hardware. | `steady` under latest-wins and under FIFO; stale ratio against threshold from per-command samples; the depth margin reported at startup | telemetry `freshness_ms`, `commands_stale`; `frames_overwritten`; `workers.depth.margin_ms` | `fifo_vs_latest.png`, `stale_ratio_vs_threshold.png` | planned |
| Q3 Batch, scheduling, scaling | Per-robot throughput of a model stays at 30 fps while its utilization (demand over measured capacity) is below 1, and falls as capacity over N above that. The inflection is at N* = capacity / 30: `{{sweep.cls.inflection_n}}` robots (`{{sweep.cls.inflection_between}}`) for classification and `{{sweep.depth.inflection_n}}` (`{{sweep.depth.inflection_between}}`) for depth. The CPU share per model, the CPU cost per frame and the pacing share move it. | The inflection point is a property of the test machine, not of the design. | `sweep` and `massive`; the capacity model run before the sweep; the candidates in section 12 that change share or cost | `robots.<id>.models.<m>.commands_sent`, telemetry `freshness_ms`, `models.<m>.batch_size`, `models.<m>.busy_ms`, `cgroup.cpu_usage_us`, `cgroup.throttled_us` | `sweep_throughput.png`, `sweep_freshness.png`, `sweep_batch_size.png`, `sweep_cpu.png`, `predicted_vs_measured.md` | planned |
| Q4 Latency breakdown | At 5 robots the stage means add up to the end-to-end mean, with a residual of `{{sweep.n5.depth.residual_ms.mean}}` ms for depth and `{{sweep.n5.cls.residual_ms.mean}}` ms for classification. The largest stage and the biggest optimization opportunity are read from this table. | Timers in different processes do not add up, and percentiles of stages cannot be summed. | `sweep` at 5 robots; every command carries its stage durations; sums use means and tails are reported per stage | `stages.<m>.*`, telemetry `fire_lag_ms`, `deliver_ms`, per-command samples | `breakdown_n5.png`, `breakdown_n5.md` | planned |
| D1 Steady load | With 1 robot at 30 fps for 30 s, freshness stays bounded: no upward trend, and p50, p95, p99 and max are reported per model. | 30 s is too short to show drift, and a maximum can hide a slowly growing queue. | `steady`, repeated; a trend test of freshness against time; `soak` covers the long run | telemetry `freshness_ms`; per-command samples | `steady_freshness.png`, `steady.md` | planned |
| D2 Late joiner | When B joins, capacity is split evenly at once: A and B end with the same per-robot throughput on each model, and A's p95 freshness does not change. | The newcomer gets a burst, or the incumbent keeps an advantage. | `late_joiner`; windows 0 to 10 s and 10 to 30 s | per-robot `commands_sent` and telemetry `freshness_ms`, per window | `late_joiner_throughput.png`, `late_joiner.md` | planned |
| D3 Scaling sweep | The sweep shows per-robot throughput, p50 and p95 freshness, the batch-size distribution per model and CPU utilization at 1, 3, 5 and 10 robots, against the capacity model's predictions, which no demo run feeds (section 6.3). | The results depend on how robot phases line up and on the simulator sharing the CPU. | `sweep` with random phases; `single` and `split`; aligned phases are a planned experiment (section 12) | as Q3, plus telemetry `fire_lag_ms` | the four sweep plots, `sweep.md` | planned |
| D4 Disconnect | After one of 3 robots is killed with SIGKILL at 20 s, the survivors' depth throughput rises as the freed share is redistributed, nothing errors, and the dead robot's state is released. | A dead peer that never closes its socket keeps its share or leaks state. | `disconnect`; windows 0 to 20 s and 20 to 40 s; a SIGSTOP integration test covers the half-open case | per-robot `commands_sent` per window, `connections.closed`, `frames_dropped_disconnect`, `cgroup.memory_*` | `disconnect_throughput.png`, `disconnect.md` | planned |
| D5 Massive connections | 50 robots for 30 s: no crash, flat memory, and no robot starved. Per-robot command rates stay close together on each model, and every command gap stays within the scheduler's bound. | Equal shares of too little is still starvation: a command that arrives too rarely cannot control a robot. | `massive`, repeated; `soak` for memory; watchdog coverage reported per model | `robots.<id>.models.<m>.commands_sent`, `max_service_gap_ms`; telemetry `watchdog_expired_ms`; `cgroup.memory_anon_bytes`, `cgroup.memory_sock_bytes`; `connections.*` | `massive_fairness.png`, `massive_memory.png`, `coverage_vs_watchdog.png`, `massive.md` | planned |
| T1 Topology | Running the simulator inside the limited container costs the system `{{massive.fleet.cpu_cores}}` cores at 50 robots. The `split` topology runs the cloud alone under the quota with the same pacing reserve, and shows what the simulator does to timing and throttling. | In production the robots are elsewhere, so a co-located simulator distorts both capacity and timing. | every scenario in `single` and in `split`, in alternating order | fleet `processes`, telemetry `fire_lag_ms`, all freshness and throughput fields | `topology.md` | planned |
| R1 Bounded resources | Memory does not grow: the slope is `{{soak.memory.slope_mb_per_min}}` MB per minute over 5 minutes at 50 robots, and fd and task counts stay flat. | Slow leaks (per-robot metrics, reconnect churn) do not show in 60 s. | `soak`; the reconnect-churn integration test | `cgroup.memory_anon_bytes`, `cgroup.memory_sock_bytes`, `processes.<p>.fds`, `loop.tasks` | `soak_memory.png` | planned |
| R2 Disconnect and reconnect | A robot that disconnects or reconnects does not affect the others: a frozen robot is closed by the idle timeout and comes back under a new session, and the other robots' classification rate stays close to its rate before. | A reconnect storm resets fairness or leaks sessions. | the frozen-robot reconnect integration test (SIGSTOP); the churn integration test | `connections.*`, per-robot rates, `robots.<id>.session` | `faults.md` | planned |
| R3 Worker failure | A worker that dies is restarted, and the other model keeps serving: after the depth worker is killed with SIGKILL, depth commands arrive again `{{worker_kill.depth.outage_ms}}` ms later, and classification serves `{{worker_kill.before.cls.commands_per_s_per_robot.mean}}` commands/s per robot before the kill and `{{worker_kill.after.cls.commands_per_s_per_robot.mean}}` after it. | A restart that loads and calibrates the model again leaves the robots without depth commands for seconds, and its warm-up competes with the classifier for the quota. | `worker_kill`; the killed-worker, inference-timeout and gateway-death integration tests | `workers.<m>.state`, `workers.<m>.restarts`, `workers.<m>.pid`, `robots.<id>.models.<m>.frames_failed`, per-robot rates | `faults.md` | planned |

## 3. Assumptions

The brief leaves these open. Each assumption is also written in the code at the point where it applies, as a comment that starts with `ASSUMPTION:`.

| ID | Assumption | Reason |
|---|---|---|
| A1 | The robot simulator runs in the same container as the cloud by default (`single` topology). A `split` topology with robots in their own container is also supported and measured. | The brief says the entire system must run in one container. |
| A2 | Every frame is offered to both models. A robot can subscribe to one model in the handshake. The two command streams are independent and have separate statistics. | The brief says both models are deployed and frames are routed to models, but not how. |
| A3 | A command is a small structure derived from the model output. Classification: top-1 class and confidence. Depth: proximity of the left, center and right zones, mapped to steering and speed. The depth map is not sent back. The robot "acts" by making an accepted command the one in effect, which its watchdog expires when no newer one arrives in time; it counts accepted, stale and out-of-order commands and records each in its samples, and logs none of them one by one. | The brief says results function as control commands. |
| A4 | The frame number and capture time travel in the message header, not inside the JPEG. | "Each frame carrying its frame number and capture time." |
| A5 | Frames are encoded once at startup and cycled. | A real robot encodes on its own hardware. Encoding 30 fps per robot inside the 2 CPUs would measure the simulator. |
| A6 | Capture time is the scheduled tick time, t0 + k/30 s, not the actual send time. | A late simulator must not hide latency (no coordinated omission). The difference is reported as `fire_lag`. |
| A7 | Freshness is robot receive time minus capture time, both on the robot's clock. | Capture to receive, as the brief defines it, with no clock synchronization needed. |
| A8 | Throughput is commands received per robot, per model, per second. The non-stale part is reported separately. | The brief does not define throughput. |
| A9 | The stale thresholds are ours to choose. Section 5 gives them and why. | The brief asks for the threshold and the reason. |
| A10 | The stale count is reported in the robot's log and through 1 Hz robot telemetry that the gateway includes in `/metrics`. | "Stale count is reported." |
| A11 | "Kill one" means SIGKILL of that robot's process, with no clean close. | The harder case. |
| A12 | "Starved" is defined by the longest command gap per robot and the ratio of the lowest to the mean per-robot command rate. The Jain index is reported as well. | The brief does not define starvation. |
| A13 | "Memory leak" is judged by the slope of anonymous plus socket memory after warm-up, over a 5-minute soak, not only the 60 s the brief names. | Page cache from writing result files is not a leak, and slow leaks need more than 60 s. |
| A14 | Model warm-up finishes before robots connect. Statistics cover the whole run window. Any exclusion is stated next to the number. | The brief does not say whether windows include warm-up. |
| A15 | CPU utilization is the container's cgroup CPU usage as a fraction of 2 CPUs, with a per-process breakdown. | The brief does not define it. |
| A16 | All demos send 30 fps on the wire. A server-advised lower rate at 50 robots is a planned experiment and is not built. | The brief lists backpressure as an allowed mechanism at 50 robots. |
| A17 | Workers are separate processes by choice. The brief does not require it. | Isolation between models and between I/O and compute. |
| A18 | Five demos are done. The brief's evaluation text says four. | The brief lists five. |
| A19 | The CPU limit is a quota (2 CPUs of time per period), not two pinned cores. Behaviour under a cpuset limit is measured as a control. | "Limited to 2 CPU cores" can be applied either way. |
| A20 | All processes on one host share a monotonic clock, so stage timestamps can be compared in both topologies. Protocol logic never compares clocks across machines (section 5.4). | Needed for the latency breakdown. |
| A21 | TLS and real authentication are out of scope. Plain `ws` runs inside the container. | The brief is about scheduling under load. |
| A22 | Robot ids are unique. A second connection with the same id replaces the first session. | The brief does not cover identity. |
| A23 | Frames are synthetic. Their content sets the JPEG size, which affects transport and decode cost. The frame set is fixed, and its size statistics are recorded with every run. | The brief asks for synthetic frames. |
| A24 | The memory limit of 4 GB is 4 GiB of memory with no swap: the container's memory limit and its memory-plus-swap limit are equal. The run's gate compares the limits read inside the container with these values exactly, as it compares the CPU quota of A19; where the kernel does not report the swap limit, the run is not gated. | The brief states the memory limit without its unit base or a swap setting. |

## 4. Tool choice

Each record states the context, the options, the decision, the consequence, and what would break at 50 robots.

### 4.1 Language

- **Context.** The CPU budget is 2 CPUs for everything. Most of it should go to inference, which runs in native code whatever the language around it. The brief's required layout lists `requirements.txt`.
- **Options.** (a) Python with one process per component. (b) A Rust, Go or C++ gateway with the ONNX Runtime C API. (c) Python in a single process with threads.
- **Decision.** (a), on Python 3.14.
- **Consequence.** Python code in one process runs on one core at a time, so each process is kept to one job: the gateway moves bytes and schedules, the workers decode and infer. (b) would lower the I/O cost, but the most it could win back is the gateway's share, `{{massive.gateway.cpu_cores}}` of 2 cores. (c) would put decoding and the event loop in contention for the interpreter lock and remove fault isolation between models.
- **At 50 robots.** The gateway handles 1500 frames per second in one event loop at `{{massive.gateway.cpu_ms_per_frame}}` CPU-ms per frame, with loop lag p99 of `{{massive.gateway.loop_lag_ms.p99}}` ms. A single loop saturates at 1000 / (30 x that cost) robots. What is expected to hurt first is not moving bytes but sharing the quota with two saturated inference threads (section 6.4).

### 4.2 Event loop and framework

- **Context.** The gateway needs many concurrent connections, timers and one HTTP endpoint. It does not need routing, templating or a request model.
- **Options.** (a) asyncio with uvloop and the `websockets` library. (b) asyncio's default loop. (c) A web framework (FastAPI or aiohttp) with WebSocket support. (d) A serving framework (Triton, Ray Serve) for batching, behind a custom gateway.
- **Decision.** (a). The same server answers plain HTTP requests for `/metrics` and `/healthz`, so there is one port and no second framework.
- **Consequence.** The scheduler is ours to write and to test. (d) is rejected because their built-in dynamic batchers queue requests and release them by batch size and timeout. Keeping only the newest frame per robot and serving the least recently served robot would still have to be built around them, inside a heavier process with less visible timing. (c) adds a dependency and a layer for one endpoint.
- **At 50 robots.** One loop serves all connections. Its health is visible as `loop.lag_ms`.

### 4.3 Transport

- **Context.** In production a robot sits behind NAT and firewalls and reaches the cloud through load balancers. So the robot dials out, and the transport must pass through that path. A frame is much larger than a network packet, frames arrive at 30 fps, and old frames have no value.
- **Options.**

| Option | For | Against |
|---|---|---|
| WebSocket binary frames over TCP | Message boundaries, ping/pong, close codes. Starts as HTTP, so it passes proxies and load balancers. One port also serves `/metrics`. | TCP head-of-line blocking under loss. A small framing cost over raw TCP (not measured here). |
| Raw TCP with a length prefix | Lowest overhead, no dependency. | Keepalive, close semantics and the HTTP endpoint must be built. Harder to route through HTTP load balancers. |
| gRPC bidirectional streaming | Schema and flow control. | More per-message work in Python. HTTP/2 flow-control windows buffer frames inside the stack, which is the kind of hidden queue this design removes. |
| HTTP POST per frame | Simple. | Request and response are tied one to one, so a dropped frame needs an explicit reply. A poor fit for streaming results. |
| UDP, QUIC datagrams, WebRTC | No retransmission of stale data and no head-of-line blocking. The best fit for lossy links. | A frame exceeds the MTU, so fragmentation and reassembly are needed. No advantage on loopback. |

- **Decision.** WebSocket binary frames, with compression off (JPEG does not compress further).
- **Production view.**
  - The robot initiates the connection, outbound, so it crosses NAT, firewalls and load balancers.
  - TLS ends at the load balancer. Inside the container the connection is plain `ws`. TLS cost is not measured.
  - The handshake carries a protocol version and an auth token. An unknown version is rejected. The token is checked against an optional shared token, off by default. Real authentication is not built.
  - The frame protocol (`common/protocol.py`) does not depend on the transport. WebSocket sits behind a small adapter, and tests use an in-memory adapter. A QUIC adapter is the migration path for lossy links. It is not built.
- **Consequence.** TCP delivers in order. On a lossy link one lost segment delays every frame behind it (head-of-line blocking). Late frames still cost little compute, because each one overwrites the slot on arrival and only the newest is decoded, but they arrive late. This is a weakness on production networks. It is not a bottleneck at 50 robots on loopback, where nothing is lost. To keep TCP from hiding a queue, the buffers that carry frames (the robot's send buffer and the gateway's receive buffer) are sized so that a stalled path holds about two frames of the default frame set, the buffers that carry commands (the gateway's send buffer and the robot's receive buffer) to a few kilobytes, and the robot skips a frame when the previous one is still unsent. The robot also sets `TCP_NOTSENT_LOWAT`, but on loopback one segment can carry a whole frame past it, so the buffer sizes are what bound the path.
- **At 50 robots.** The cost is bytes: `{{massive.bytes_in_per_s}}` bytes per second into the gateway, and an uplink stage of `{{massive.stage.transport_up_ms.p99}}` ms at p99.

Wire protocol, version 1:

| Message | Direction | Encoding | Fields |
|---|---|---|---|
| `hello` | robot to cloud | JSON | protocol version, robot id, auth token (optional), subscribed models, stale threshold per model, watchdog limit per model, frame rate, frame size |
| `welcome` | cloud to robot | JSON | protocol version, session number, whether the clock is shared |
| `frame` | robot to cloud | binary header plus JPEG | session number, frame number, scheduled capture time, send time (both on the robot's clock) |
| `command` | cloud to robot | binary header plus body | model, session number, frame number, capture time (echoed), batch size, server stage durations, gateway result time, command body |
| `telemetry` | robot to cloud | JSON, 1 Hz | the robot-reported fields of section 10.4 |

Close codes: 1000 normal, 1001 server shutdown, 1002 protocol error (a first message that is not a hello, a malformed text message), 1008 policy (bad version, bad token, unknown model, ingress limit), 1009 message too big, 1013 over the connection cap, 4001 session replaced, 4003 idle timeout (also no hello within the handshake timeout). Two kinds of end are aborted without a close frame: a slow consumer, because it is found only while bytes to it are unsent and a close frame would wait behind them (code 4002 stays reserved for it in the protocol and is not sent), and a connection ended by a fault in the gateway's own code. A robot does not reconnect after 1002, 1007, 1008, 1009 or 4001, because retrying would repeat the refusal or start a replacement loop; it reconnects, with backoff, after any other close, an abort, a refused connection or an HTTP 503. Robots connect to the path `/ws`.

### 4.4 Batcher and scheduler

- **Context.** Demand exceeds capacity as a normal state. The question is which frame to run, for whom, and when to drop.
- **Options.** (a) A FIFO request queue with a maximum batch size and a timeout, the usual dynamic batcher. (b) A FIFO queue per robot. (c) One newest-frame slot per robot, pulled by an idle worker, robots served least recently served first. (d) Credits: the robot sends only when the server grants a frame. (e) A server-advised frame rate, as an addition to (c).
- **Decision.** (c). The batch cap comes from measurement (section 6.2). The scheduler never waits to fill a batch: a batch is whatever is eligible when the worker asks, up to the cap. (b), a bounded FIFO per robot that drops new frames when full, is kept as a switch for a control experiment; with one robot, as in that control, it behaves as (a) without the batch-fill timeout. (e) is a planned experiment (section 12) and is not built.
- **Consequence.** Memory is bounded at one frame per robot. A robot that sends faster gains nothing. A joining robot is served at the next dispatch and then rotates like the others. A departed robot offers no frames, so its share is free before the server even detects the disconnect. (d) saves bandwidth but adds a round trip per grant, so the worker idles when the network has delay.
- **At 50 robots.** Nothing changes structurally. The per-robot command interval grows linearly with N: `{{massive.depth.command_gap_ms.max}}` ms at worst for depth, against a bound of `{{massive.depth.command_gap_bound_ms}}` ms.

### 4.5 Gateway-to-worker IPC

- **Context.** Slots live in the gateway and workers pull, so every dispatch is a round trip through the gateway's event loop. The classifier's service time is short, so the round trip may be a visible share of it.
- **Options.** (a) A Unix socket with length-prefixed messages carrying raw JPEG bytes. (b) `multiprocessing` queues with pickled arrays. (c) A shared-memory slot table that workers read directly. (d) A message library or RPC layer.
- **Decision.** (a). One message each way per cycle: the worker's message carries the results of the previous batch and the request for the next one.
- **Consequence.** The round trip is reported as its own stage, `dispatch_rt`, not folded into batch wait. The gateway stamps a result when its last byte arrives, in the socket callback, before its dispatcher resumes, so `dispatch_rt` holds no wake-up of the gateway's own tasks. On a result the dispatcher queues the commands, selects and writes the next batch, and only then writes the commands, so between two batches the worker waits for the scheduler's work and one wake-up of the dispatcher. Per-process CPU accounting gives per-model CPU attribution directly. (c) removes the gateway from the dispatch path but spreads scheduler state across processes, which costs testability. It is reconsidered only if the round trip exceeds about 10% of the classifier's service time.
- **At 50 robots.** The round trip costs `{{massive.stage.dispatch_rt_ms.p50}}` ms.

### 4.6 Model runtime

- **Context.** Two ONNX models, CPU only, under a quota. Inside a container limited by quota, the host's core count is still visible, and ONNX Runtime sizes its thread pool from it by default.
- **Options.** (a) One session per worker, one intra-op thread each. (b) One executor with 2 threads running both models' batches in turn. (c) As (a), with a second depth thread when the classifier leaves CPU idle. (d) ONNX Runtime's default thread count.
- **Decision.** (a) as the baseline. Thread counts are always set explicitly, for ONNX Runtime and OpenCV, and thread spinning is off, because spinning burns quota. The image sets `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`, because importing numpy otherwise starts an OpenBLAS pool with one thread per visible host core, and since no code here calls a BLAS routine, those threads would only compete for the quota. (c) is the first optimization candidate. (d) is run once as a control.
- **Consequence.** The slow model cannot delay the fast one. With few robots the classifier is expected to leave part of its core idle while depth drops frames, which is what (c) addresses. The published MiDaS file has a batch dimension fixed at 1 and normalizes its input inside the graph. It is served as published when depth's batch cap is 1. A copy patched to a dynamic batch dimension is used when the cap is larger, and in the batch-cap comparison. MobileNetV3 is preprocessed outside the graph: resize the shorter side to 256, center crop to 224, normalize with the ImageNet mean and standard deviation.
- **At 50 robots.** Both workers are saturated, and the limit is the quota. Section 6.4 budgets it.

## 5. Frame management

### 5.1 How 30 fps arrival meets slower processing

| Policy | Result |
|---|---|
| FIFO queue, drop new frames when full | Old frames are processed first, so a served frame is as old as the queue is long: freshness grows with the queue length and with the robot count, and for the whole run when the queue never fills. Kept as a switch for the control experiment. |
| FIFO queue, drop the oldest when full | Better, but unfair between robots. |
| **One slot per robot, newest frame wins** | Memory bound of one frame per robot. Sending faster gains no share. Chosen. |
| Credits granted by the server | Saves bandwidth. Each grant costs a round trip, so a worker idles when the network has delay. |
| Server-advised frame rate | Saves transport. Suits the 50-robot case as backpressure on top of the slot. A planned experiment (section 12), not built. |

The mechanism:

- The gateway keeps one slot per robot holding the newest frame's header and JPEG bytes, and each model marks whether the slot holds a frame it has not finished with. The gateway accepts a frame only when its number is above the last frame number accepted in that session, so a model can serve a robot only when the slot's frame number is above every frame number that model has finished with.
- A frame replaced in the slot before a model served it counts as `overwritten` for that model. A frame passed over by the skip rule (section 5.5) counts as `skipped`, and the model is finished with it.
- Each model selects on its own. One frame may be processed by both models, by one, or by neither.
- A dropped frame is never decoded. Decoding happens in the worker, after selection.

Places where a queue could hide, and what closes each one:

| Place | Risk | What closes it |
|---|---|---|
| WebSocket library receive queue | Frames wait behind a slow reader. | A high-water mark of a few frames, above which the library stops reading. The reader awaits nothing else and only overwrites the slot, so it drains the queue in one loop pass. |
| asyncio write buffer | Grows without bound if unchecked. | A write limit of 0: each send waits until nothing is unsent, so at most one command's remainder and one whole command sit in user space. The bounded downlink queue holds the rest; a backlog that does not clear within the slow-consumer timeout closes the connection. |
| Kernel socket buffers | Loopback buffers auto-tune to many frames. | Fixed sizes, set before the connection exists: where frames travel (the gateway's receive buffer, the robot's send buffer), sized so that a path whose reader has stopped holds about two frames of the default frame set; a few kilobytes where commands travel. Linux doubles the size requested; the sizes the kernel grants are read back and logged, and the transport benchmark reports what a stalled path holds. |
| Robot send path | The kernel accepts frames it cannot send yet. | The robot skips a frame while the previous one is unsent, and a send is complete only when the kernel holds the whole frame. `TCP_NOTSENT_LOWAT` is set too, but a loopback segment can carry a whole frame past it. |
| Gateway to worker | A pushed batch waits at a busy worker. | Workers pull. At most one batch per worker is in flight. |
| Link-model delay lines | Injected delay buffers messages in the simulator. | Each line holds what its delay covers at the message rate. A full uplink line skips the tick; a full downlink line stops the robot reading, which pushes back on the gateway. |
| Connections before their hello | Many open connections that never say hello. | Open connections are capped at twice the connection cap; beyond it the upgrade is answered with 503. |
| `/metrics` responses | Many concurrent scrapes hold snapshots. | At most four render at once; beyond that the request is answered with 503. |

### 5.2 Freshness thresholds

Thresholds come from what the control loop can tolerate, not from what the system achieves. The robot declares them per model in the handshake. All are configurable; a watchdog limit below the stale threshold is refused, because a command accepted at the threshold would then be applied already expired.

| Model | Stale threshold (age at receipt) | Watchdog limit (age of the command in effect) | Reason |
|---|---|---|---|
| `depth` | 150 ms, hard reject | 300 ms | A displacement budget: at 1 m/s a robot moves 15 cm in 150 ms. Steering and speed must not act on older geometry. |
| `cls` | 500 ms | 1 s | Semantics change slowly: what is in view rarely changes within half a second. |

The watchdog limit is twice the stale threshold: a command that was fresh enough to accept may age by one more threshold before the robot stops acting on it.

Whether depth can meet 150 ms on a given machine is a measurement, not a choice: `{{model_bench.depth.margin_ms}}` ms of margin on the test machine (section 5.5).

### 5.3 Robot-side rules

- **Stale.** A command whose age at receipt (receive time minus capture time) exceeds the model's stale threshold is rejected and counted in `commands_stale`.
- **Out of order.** A command whose frame number is not above the last applied frame number for that model and session is rejected and counted in `commands_out_of_order`.
- **Watchdog.** For each model the robot tracks the age of the command in effect: now minus that command's capture time. When it exceeds the watchdog limit, that model's state becomes `expired` until a fresh command arrives. Trips and the total time without an unexpired command are counted. Coverage is the fraction of connected time during which the robot holds an unexpired command; the time before the first applied command is not covered.

Two curves show how sensitive the results are to these choices. Both are computed offline from per-command samples, so trying other values needs no rerun:

- **Stale ratio against threshold** (`stale_ratio_vs_threshold.png`): for each candidate threshold, the fraction of commands older than it at receipt, per model and scenario.
- **Coverage against watchdog limit** (`coverage_vs_watchdog.png`): for each candidate limit, the fraction of time a robot holds an unexpired command, per model and robot count.

### 5.4 Age without synchronized clocks

The server needs a frame's age for the skip rule. Unless robot and gateway share a clock (`config.shared_clock`), it never subtracts a robot timestamp from its own clock. The frame header carries capture time and send time, both on the robot's clock, so their difference is the age at send. The server adds the uplink, measured exactly on a shared clock and otherwise estimated as half the WebSocket ping round trip, and its own dwell time, measured on its own clock. Thresholds can differ per robot, because each robot declares its own.

On one host all processes share a monotonic clock. The latency breakdown uses that to measure the uplink exactly (assumption A20). On a shared clock the gateway also refuses a frame whose send time is after its arrival or whose capture time is implausibly old; without one, the protocol logic uses only differences of robot timestamps and the ping round trip.

### 5.5 When the hardware cannot meet the threshold

Freshness has a floor that does not depend on load (section 6.3): slot wait, plus one service time, plus overhead, plus twice the one-way link delay. If the floor is above the threshold, no scheduling can help.

- At startup each worker measures its service time and the gateway reports the margin per model: threshold minus one frame period, minus the service time at the batch cap, minus an overhead allowance. It appears as `workers.<m>.margin_ms` and in the log.
- **Skip rule.** When only one robot has a pending frame, that frame would be stale by the time its result arrives, and a fresher frame from that robot could still make it, the scheduler drops the frame and waits for the next frame to arrive. When two or more robots have a pending frame it skips nothing, so the rule never changes whose turn it is. On a host with a negative margin, robots that share a saturated model then get equal shares and the stale ratio that the margin implies.
- **The skip rule can never starve.** It is off for a model whose service time alone exceeds the threshold, and it never leaves a robot unserved for longer than a configured interval.
- **On slow hardware.** When a model's service time alone exceeds its threshold, the system keeps serving and reports the threshold as infeasible on this host (`workers.<m>.feasible` is false). It does not silently produce a stream that is entirely stale. The system never changes a thread count by itself. A second depth thread (`models.depth.threads`) is a setting an operator changes: it can shorten the service time while CPU is idle, at a cost in CPU per frame (section 6.2).

### 5.6 Results

| Scenario | Model | Commands received | Stale | Stale ratio | Out of order |
|---|---|---|---|---|---|
| `steady` | `cls` | `{{steady.cls.commands_received}}` | `{{steady.cls.stale_count}}` | `{{steady.cls.stale_ratio}}` | `{{steady.cls.out_of_order_count}}` |
| `steady` | `depth` | `{{steady.depth.commands_received}}` | `{{steady.depth.stale_count}}` | `{{steady.depth.stale_ratio}}` | `{{steady.depth.out_of_order_count}}` |
| `massive` | `cls` | `{{massive.cls.commands_received}}` | `{{massive.cls.stale_count}}` | `{{massive.cls.stale_ratio}}` | `{{massive.cls.out_of_order_count}}` |
| `massive` | `depth` | `{{massive.depth.commands_received}}` | `{{massive.depth.stale_count}}` | `{{massive.depth.stale_ratio}}` | `{{massive.depth.out_of_order_count}}` |

## 6. Batching, scheduling and scaling

### 6.1 Scheduler

One scheduler per model, driven by the worker's "ready" message:

1. Collect the eligible robots: connected, subscribed to this model, with a slot frame this model has not finished with.
2. If pacing holds this model back (section 6.4), wait until it is released, then collect again. The frames chosen are the ones that are newest at dispatch time.
3. Order eligible robots by the time they were last served by this model, oldest first. A robot id that has never been served goes first, once.
4. If exactly one robot is eligible, apply the skip rule (section 5.5).
5. Take up to the batch cap and send them. If nobody is eligible, hold the worker's request until a frame arrives. Nothing polls.

Properties that the tests assert: frames are conserved (section 10.3); the frame served is always that robot's newest; frame numbers increase per robot, model and session; with the skip rule off, among robots with a pending frame, service counts differ by at most one, a faster sender gets no extra share, and a robot that reconnects gains no share, because its last-served time is kept per robot id for a while; with the skip rule on, every robot is still served within the bound of section 6.3. The skip rule acts only when one robot has a pending frame, so it does not change the order among robots that are waiting.

Two options are planned experiments (section 12) and are not built: waiting for the next frame when the only pending frame is older than half a frame period and no other robot is eligible, and a per-robot rate cap per model.

### 6.2 Batch cap

Write T(b) for the wall time of one batch of size b in the worker, and c(b) for the CPU time per frame at that size.

- **Rule.** The cap is the largest calibrated batch size whose T(b) fits the model's freshness budget: the model's configured stale threshold minus one frame period and the overhead allowance. When none fits, the cap is the smallest size and the margin is negative. The batch-cap comparison of section 12 sets the cap by configuration instead.
- **Comparison with 1.** Batch size 1 is always measured against the default cap, on per-robot throughput and freshness (section 12). When c(b) is flat, a larger batch gives the same throughput and makes every frame in it wait for the whole batch. When a dispatch has a fixed cost, a larger batch spreads it over more frames. The comparison shows which effect wins on the test machine.
- **Calibration.** At warm-up each worker runs every allowed batch size (1, 2, 4, 8): one untimed run, which also grows the runtime's memory arena to its final size, then a configured number of timed runs, whose medians it reports to the gateway as T(b) in wall and CPU time. The workers warm up one at a time, so neither calibration competes with the other for the quota. Depth calibrates on the patched copy of the MiDaS graph, which runs every batch size; when its cap comes out at 1, the worker loads the published file and measures batch size 1 again on it (section 4.6). The cap is computed once and fixed for the run. A restarted worker serves the same cap with the same file and calibrates only the sizes up to the cap; the gateway keeps the first calibration. This adapts to unknown hardware without a feedback loop.
- **Extension.** Where T(b) grows more slowly than b, for example on a GPU or with a multi-thread session, the same rule yields a larger cap with no change.

Measured on the test machine (`model_tb.png`):

| Model | Threads | T(1) | T(2) | T(4) | T(8) | CPU-ms per frame at b=1 | Cap |
|---|---|---|---|---|---|---|---|
| `cls` | 1 | `{{model_bench.cls.t1.b1.wall_ms}}` | `{{model_bench.cls.t1.b2.wall_ms}}` | `{{model_bench.cls.t1.b4.wall_ms}}` | `{{model_bench.cls.t1.b8.wall_ms}}` | `{{model_bench.cls.t1.b1.cpu_ms}}` | `{{model_bench.cls.batch_cap}}` |
| `cls` | 2 | `{{model_bench.cls.t2.b1.wall_ms}}` | `{{model_bench.cls.t2.b2.wall_ms}}` | `{{model_bench.cls.t2.b4.wall_ms}}` | `{{model_bench.cls.t2.b8.wall_ms}}` | `{{model_bench.cls.t2.b1.cpu_ms}}` | |
| `depth` | 1 | `{{model_bench.depth.t1.b1.wall_ms}}` | `{{model_bench.depth.t1.b2.wall_ms}}` | `{{model_bench.depth.t1.b4.wall_ms}}` | `{{model_bench.depth.t1.b8.wall_ms}}` | `{{model_bench.depth.t1.b1.cpu_ms}}` | `{{model_bench.depth.batch_cap}}` |
| `depth` | 2 | `{{model_bench.depth.t2.b1.wall_ms}}` | `{{model_bench.depth.t2.b2.wall_ms}}` | `{{model_bench.depth.t2.b4.wall_ms}}` | `{{model_bench.depth.t2.b8.wall_ms}}` | `{{model_bench.depth.t2.b1.cpu_ms}}` | |

Times in ms. Depth batches above 1 use the patched graph.

The batch-size distribution depends on how robot phases line up. With random phases (the default) an unsaturated classifier mostly sees one frame at a time. With all robots in phase it would see N at once; aligned phases are a planned experiment (section 12), not part of the default evidence.

### 6.3 Capacity model

Capacity is counted in CPU-ms per frame, because under a quota the scarce resource is CPU time, not wall time. A second thread shortens the service time but can raise the CPU cost per frame.

Symbols:

| Symbol | Meaning | Unit |
|---|---|---|
| f | frame rate per robot, 30 | frames/s |
| tau | frame period, 1000 / f | ms |
| N | number of robots | |
| c_m | CPU time per frame of model m: decode + preprocess + inference + postprocess + the dispatch cost shared by the batch | CPU-ms |
| T_m | service time of model m: wall time of one batch at the cap | ms |
| b_m | batch cap of model m | frames |
| s_m | CPU share granted to model m's worker; for the paced worker it depends on N (section 6.4) | cores |
| e | overhead outside the worker: uplink, dispatch round trip, delivery | ms |
| L | one-way link delay | ms |
| S_m, W_m | stale threshold and watchdog limit of model m | ms |
| Q | CPU quota, 2 | cores |
| P | quota period, 100 by default | ms |
| D | total CPU demand of all processes | cores |

Formulas:

| Quantity | Formula |
|---|---|
| Capacity of model m | K_m = 1000 x s_m / c_m frames/s |
| Utilization | rho_m = N x f / K_m |
| Inflection point | N*_m = K_m / f |
| Per-robot throughput | x_m(N) = min(f, K_m / N) |
| Command interval per robot | G_m(N) = 1000 / x_m(N) |
| Freshness floor, saturated model | F_m = w + T_m + e + 2L, with slot wait w between 0 and tau |
| Freshness floor, unsaturated model | F_m = T_m + e + 2L, plus the wait for a busy worker |
| Peak command age | A_m(N) = F_m + G_m(N) |
| Watchdog coverage | 1 - min(1, max(0, A_m(N) - W_m) / G_m(N)) |
| Robots served within the watchdog limit | the largest N with A_m(N) <= W_m |
| Longest command gap (bound) | tau + (ceil(N / b_m) + 1) x C_m, plus the skip rule's interval when the skip rule is on. C_m is the longest cycle from one dispatch to the next while frames are pending: the batch time plus the dispatch round trip, or the batch's CPU time over the pacing share when that is longer. N counts every robot connected at some time during the run, because a robot that joins goes ahead of the others once. It holds when every robot sends every frame period, no worker fails and no command is dropped from a downlink queue, and it also covers the wait from connect to the first command when the robot's first frame reaches the gateway within one frame period of the connect. The scheduler's property tests assert it. A run is checked with its own dispatch times: C_m is then the mean of that run's longest ceil(N / b_m) + 1 consecutive cycles, and the skip rule's interval is added only when a frame was skipped |
| Quota freeze per period | P x (1 - Q / D) when D > Q, else 0 |

Under late binding the freshness floor of a saturated model does not depend on N. Only the command interval does. That is the design's central claim, and the sweep and the massive demo test it.

Peak command age is what a control loop feels: how old the information it is acting on can get. Scheduling options are judged by it. Freshness is still reported as the brief defines it.

Hypotheses that the model and the design rest on. Each is confirmed or refuted by a measurement, and none is a result yet:

| ID | Hypothesis | Settled by |
|---|---|---|
| H1 | On one thread, depth capacity is below 30 frames/s, so depth is saturated from the first robot. | `model_bench`, `steady` |
| H2 | On one thread, classification capacity covers several robots at 30 fps, so its inflection point falls inside or near the sweep range. | `model_bench`, `sweep`, `massive` |
| H3 | On one thread, the per-frame time of neither model falls with batch size, so a larger cap costs freshness without adding throughput unless the dispatch cost is large. | `model_bench`, batch-caps candidate |
| H4 | For a saturated model, freshness does not depend on N. | `sweep`, `massive` |
| H5 | Two saturated inference threads plus I/O exceed the quota at high N, and without pacing the kernel freezes the whole container for part of each period. | `massive` with pacing on and off |
| H6 | The patched MiDaS graph is no faster than the published one at batch size 1. | `model_bench` |
| H7 | Python frame I/O at 50 robots uses a small share of one core. | `massive` |
| H8 | A second depth thread shortens the service time but raises CPU-ms per frame. | `model_bench`, 2-thread candidate |

Predictions for the five demos are produced by running the real scheduler code under a simulated clock (`bench/capacity_model.py`). Every measured input comes from the model benchmark alone: the batch times T(b) in wall and CPU time, the dispatch round trip and its CPU cost, and the WebSocket round trip and its CPU cost per frame. The simulated gateway computes the pacing share from the configured I/O cost per frame, as the gateway does (section 6.4); the measured cost enters only the predicted CPU demand. No demo run feeds a prediction, so no prediction is fitted to the measurement it is compared with. The tables of section 9 show the predictions computed from the gated model benchmark. Preliminary predictions, computed from the batch times of a container spike, were committed before the first end-to-end run, and the git history keeps them. `predicted_vs_measured.md` sets each prediction beside its measurement, with the gap and the run-to-run spread, and marks every gap larger than the spread for explanation. The explanation is written in the text of this document, at the claim the prediction concerns.

The model is an idealization, and its limits are stated before the first end-to-end run. Each batch takes the mean wall and CPU time that the model benchmark measured for its size with the worker alone in the container. Each transport and dispatch leg is a constant, set to the whole measured round trip that contains it, which is an upper bound. Robots send exactly on their ticks. The model has no spread in service time, no fire lag, no slowdown of a worker beside the other busy processes, no quota freeze, and no gateway work beyond the measured I/O cost. Its percentiles therefore spread only with frame phase and scheduling. The predicted p95, p99 and max carry no spread from service time or fire lag, so the measured tails are expected at or above them, to within the run-to-run spread of the batch time, and the measured maximum well above. The predicted throughput of a saturated model is a ceiling, to within the same spread. The predicted CPU demand counts the workers and the frame I/O only. Below saturation the classifier's freshness and batch size depend on the phase offsets drawn for a run, so those predictions hold for one draw. After the first end-to-end run a prediction changes only through its model-benchmark inputs; a change to the model's code, or to the scheduler code and defaults it runs, that moves a prediction is listed in section 13 with the predictions before and after it.

### 6.4 CPU budget

- The limit is a quota: the container may use Q x P of CPU time per period P. When total demand D exceeds Q, the kernel freezes every process in the container for the rest of the period. The gateway stops reading, the classifier stalls and the simulator's timers fire late. So the freeze inflates the freshness tail and corrupts the measurement at the same moment.
- **Budget.** The gateway, the simulator and the classifier get what they need. The depth worker is saturated by design and is the one elastic consumer, so it is paced: after each batch it becomes eligible again only once its duty cycle is back under its share. The idle time is then taken by depth alone, not by the whole container.
- The pacing share is feed-forward, not a feedback loop. When a robot joins or leaves, the gateway recomputes it from the frame rates the connected robots declared, the workers' calibrated CPU time per frame and a configured CPU cost per frame of the frame I/O: the quota, minus a headroom, minus the predicted demand of the other consumers (the frame I/O at that configured cost, which covers the gateway and the co-located simulator and is reserved in both topologies, plus each unpaced worker at its calibrated CPU time per frame, capped at its threads, or at the quota when ONNX Runtime chooses its thread count), and never below a minimum share. When that leaves the paced worker at least as many cores as it can use (its threads, or the quota when ONNX Runtime chooses its thread count), it is not paced, and without a CPU quota nothing is paced. The I/O cost is one setting, `pacing.io_cpu_ms_per_frame`, whose default is set for the gateway and the simulator together. The evidence runs keep that default in the `split` topology, where the simulator runs outside the quota, so the share still reserves the simulator's part there and the depth worker is paced to the same share as in `single`. Throttled time, classifier p99 freshness and fire lag are compared with pacing on and off.
- The share holds over each dispatch-to-dispatch span, not within one quota period, because a batch cannot be interrupted. A period that holds a whole depth batch must fit the classifier's CPU, the frame I/O and that batch in Q x P; where it does not, that period ends in a freeze that no share removes. A depth batch longer than the period (caps above 1) always overruns while the classifier is saturated.
- Under a cpuset limit or on a real 2-core host there is no freeze. The same contention shows up as time slicing. Workers can run at lower priority (`worker_nice`), so that the gateway and the simulator preempt them, but the default niceness is 0: under the CPU quota niceness has no effect, and on a two-CPU cpuset a niced depth batch that shares its CPU with a busy process stretches about tenfold, which is why it is off by default.
- `cgroup.nr_throttled` and `cgroup.throttled_us` are in `/metrics`, so every run shows whether it was throttled.

### 6.5 The inflection point

From the sweep and the 50-robot run:

| Model | Capacity at the first saturated point (frames/s) | Inflection N* | Utilization crosses 1 | Utilization at 1 / 3 / 5 / 10 robots |
|---|---|---|---|---|
| `cls` | `{{sweep.cls.capacity_fps}}` | `{{sweep.cls.inflection_n}}` | `{{sweep.cls.inflection_between}}` | `{{sweep.n1.cls.utilization}}` / `{{sweep.n3.cls.utilization}}` / `{{sweep.n5.cls.utilization}}` / `{{sweep.n10.cls.utilization}}` |
| `depth` | `{{sweep.depth.capacity_fps}}` | `{{sweep.depth.inflection_n}}` | `{{sweep.depth.inflection_between}}` | `{{sweep.n1.depth.utilization}}` / `{{sweep.n3.depth.utilization}}` / `{{sweep.n5.depth.utilization}}` / `{{sweep.n10.depth.utilization}}` |

Why there: a model's per-robot throughput leaves 30 fps when its worker's utilization reaches 1. The absolute N depends on the hardware, so the explanation is given as utilization, which carries across machines. For the paced depth worker the share, and with it capacity and utilization, depends on N; each is computed with the share in effect at that robot count. Capacity is computed per run from that run's CPU cost per frame and share, so it differs between robot counts: the classifier's cost per frame falls as its batches fill. N* uses the capacity of the first scale point that is saturated; where the sweep does not reach saturation, the 50-robot run supplies that point and the sweep gives only the lower end of the bracket.

What moves it, with the experiment that measures each:

| Lever | Effect | Experiment |
|---|---|---|
| CPU share per model (threads, pacing share) | Changes s_m | 2-thread depth; pacing on and off |
| Decode cost | Part of c_m; a larger part for the classifier | half-size JPEG decode |
| Batch cap | Spreads the dispatch cost | batch caps |
| Per-robot rate cap on the classifier | Frees CPU for depth | rate cap |
| Frame rate on the wire | Lowers I/O demand | server-advised frame rate |
| Model cost | Changes c_m | INT8 MiDaS |

## 7. Latency breakdown at 5 robots

Stages are consecutive intervals on one clock, so they add up to the end-to-end time by construction. The check is whether they do in practice.

| Stage (metrics name) | From | To | Measured by | Brief's term |
|---|---|---|---|---|
| `fire_lag_ms` | scheduled capture time | robot starts sending | robot | (simulator lateness) |
| `transport_up_ms` | robot starts sending | gateway has the whole frame | gateway | transport |
| `slot_wait_ms` | gateway has the frame | frame selected for a batch | gateway | batch wait |
| `dispatch_rt_ms` | selected, and results sent | worker has the batch, and gateway has the results | gateway and worker | batch wait |
| `decode_ms` | | | worker, per frame | JPEG decode |
| `preprocess_ms` | | | worker, per frame | preprocessing |
| `batch_peers_ms` | | | worker: time spent on the other frames of the same batch | batch wait |
| `inference_ms` | | | worker: assembling the batch tensor and the model run, for the batch | inference |
| `postprocess_ms` | | | worker, per frame | postprocessing |
| `deliver_ms` | gateway has the results | robot has the command | robot | result delivery |

`dispatch_rt_ms` is the sum of the two IPC legs. `batch_peers_ms` is defined as the worker's time for the batch minus this frame's own decode, preprocess, inference and postprocess, so the worker stages add up exactly. At batch size 1 it is only the worker's own untimed overhead, close to zero.

Each boundary is one clock reading, taken once: the robot stamps the send at the tick, the gateway stamps a frame when its reader gets it from the WebSocket library, a dispatch when the scheduler selects it, and a result in the socket callback that completes it; the robot stamps the command when it gets it. The time between the result and the write of the command falls in `deliver_ms`. The eight server stages travel as whole microseconds, computed as differences of cumulative offsets rounded down, so they add up exactly to the whole interval.

How the sum is checked:

- Every command carries its server-side stage durations back to the robot, so one robot-side record holds the whole breakdown.
- Sums use means, because means add and percentiles do not. The residual is the mean end-to-end time minus the sum of the stage means, and it is shown as its own row. On a shared clock the residual of each command is below one microsecond by construction, so a larger one means a boundary was stamped twice or not at all. Without a shared clock `transport_up` is an estimate and the residual is real.
- Tails are reported per stage as p50 and p95.
- Each model's stage rows, sum, end-to-end mean and residual come from one run, the run whose end-to-end mean is the median over the runs, because medians taken stage by stage come from different runs and do not add up. The end-to-end p50 and p95 are medians over the runs, as in section 9. Cells are rounded for display; the residual is computed before rounding.

| Stage | `cls` mean | `cls` p50 | `cls` p95 | `depth` mean | `depth` p50 | `depth` p95 |
|---|---|---|---|---|---|---|
| `fire_lag` | `{{sweep.n5.cls.stage.fire_lag_ms.mean}}` | `{{sweep.n5.cls.stage.fire_lag_ms.p50}}` | `{{sweep.n5.cls.stage.fire_lag_ms.p95}}` | `{{sweep.n5.depth.stage.fire_lag_ms.mean}}` | `{{sweep.n5.depth.stage.fire_lag_ms.p50}}` | `{{sweep.n5.depth.stage.fire_lag_ms.p95}}` |
| `transport_up` | `{{sweep.n5.cls.stage.transport_up_ms.mean}}` | `{{sweep.n5.cls.stage.transport_up_ms.p50}}` | `{{sweep.n5.cls.stage.transport_up_ms.p95}}` | `{{sweep.n5.depth.stage.transport_up_ms.mean}}` | `{{sweep.n5.depth.stage.transport_up_ms.p50}}` | `{{sweep.n5.depth.stage.transport_up_ms.p95}}` |
| `slot_wait` | `{{sweep.n5.cls.stage.slot_wait_ms.mean}}` | `{{sweep.n5.cls.stage.slot_wait_ms.p50}}` | `{{sweep.n5.cls.stage.slot_wait_ms.p95}}` | `{{sweep.n5.depth.stage.slot_wait_ms.mean}}` | `{{sweep.n5.depth.stage.slot_wait_ms.p50}}` | `{{sweep.n5.depth.stage.slot_wait_ms.p95}}` |
| `dispatch_rt` | `{{sweep.n5.cls.stage.dispatch_rt_ms.mean}}` | `{{sweep.n5.cls.stage.dispatch_rt_ms.p50}}` | `{{sweep.n5.cls.stage.dispatch_rt_ms.p95}}` | `{{sweep.n5.depth.stage.dispatch_rt_ms.mean}}` | `{{sweep.n5.depth.stage.dispatch_rt_ms.p50}}` | `{{sweep.n5.depth.stage.dispatch_rt_ms.p95}}` |
| `decode` | `{{sweep.n5.cls.stage.decode_ms.mean}}` | `{{sweep.n5.cls.stage.decode_ms.p50}}` | `{{sweep.n5.cls.stage.decode_ms.p95}}` | `{{sweep.n5.depth.stage.decode_ms.mean}}` | `{{sweep.n5.depth.stage.decode_ms.p50}}` | `{{sweep.n5.depth.stage.decode_ms.p95}}` |
| `preprocess` | `{{sweep.n5.cls.stage.preprocess_ms.mean}}` | `{{sweep.n5.cls.stage.preprocess_ms.p50}}` | `{{sweep.n5.cls.stage.preprocess_ms.p95}}` | `{{sweep.n5.depth.stage.preprocess_ms.mean}}` | `{{sweep.n5.depth.stage.preprocess_ms.p50}}` | `{{sweep.n5.depth.stage.preprocess_ms.p95}}` |
| `batch_peers` | `{{sweep.n5.cls.stage.batch_peers_ms.mean}}` | `{{sweep.n5.cls.stage.batch_peers_ms.p50}}` | `{{sweep.n5.cls.stage.batch_peers_ms.p95}}` | `{{sweep.n5.depth.stage.batch_peers_ms.mean}}` | `{{sweep.n5.depth.stage.batch_peers_ms.p50}}` | `{{sweep.n5.depth.stage.batch_peers_ms.p95}}` |
| `inference` | `{{sweep.n5.cls.stage.inference_ms.mean}}` | `{{sweep.n5.cls.stage.inference_ms.p50}}` | `{{sweep.n5.cls.stage.inference_ms.p95}}` | `{{sweep.n5.depth.stage.inference_ms.mean}}` | `{{sweep.n5.depth.stage.inference_ms.p50}}` | `{{sweep.n5.depth.stage.inference_ms.p95}}` |
| `postprocess` | `{{sweep.n5.cls.stage.postprocess_ms.mean}}` | `{{sweep.n5.cls.stage.postprocess_ms.p50}}` | `{{sweep.n5.cls.stage.postprocess_ms.p95}}` | `{{sweep.n5.depth.stage.postprocess_ms.mean}}` | `{{sweep.n5.depth.stage.postprocess_ms.p50}}` | `{{sweep.n5.depth.stage.postprocess_ms.p95}}` |
| `deliver` | `{{sweep.n5.cls.stage.deliver_ms.mean}}` | `{{sweep.n5.cls.stage.deliver_ms.p50}}` | `{{sweep.n5.cls.stage.deliver_ms.p95}}` | `{{sweep.n5.depth.stage.deliver_ms.mean}}` | `{{sweep.n5.depth.stage.deliver_ms.p50}}` | `{{sweep.n5.depth.stage.deliver_ms.p95}}` |
| Sum of stage means | `{{sweep.n5.cls.stage_sum_ms.mean}}` | | | `{{sweep.n5.depth.stage_sum_ms.mean}}` | | |
| End to end | `{{sweep.n5.cls.e2e_ms.mean}}` | `{{sweep.n5.cls.freshness_ms.p50}}` | `{{sweep.n5.cls.freshness_ms.p95}}` | `{{sweep.n5.depth.e2e_ms.mean}}` | `{{sweep.n5.depth.freshness_ms.p50}}` | `{{sweep.n5.depth.freshness_ms.p95}}` |
| Residual | `{{sweep.n5.cls.residual_ms.mean}}` | | | `{{sweep.n5.depth.residual_ms.mean}}` | | |

All values in ms.

**Biggest opportunity.** To be named from this table and from section 12. The working hypothesis is that depth's service time dominates its end-to-end time, so a second depth thread at low robot counts is the largest single lever. It is not a result yet.

## 8. The 50-robot mechanism

### 8.1 What is implemented

The brief names four kinds of mechanism. The system uses all four, each for a different job:

| Mechanism | Implementation | Job |
|---|---|---|
| Connection admission | Every robot is accepted up to a configurable cap (default 64). Beyond it the handshake is refused with close code 1013. | A safety valve, not the main control. |
| Per-robot quota | Fair rotation: each model serves robots least recently served first, so each robot gets one N-th of each model's capacity. | No robot can take another's share. |
| Adaptive drops | The slot overwrite: each model serves a robot's newest frame, and older frames are dropped unprocessed. The skip rule of section 5.5 acts only while one robot is waiting, so it plays no part at 50 robots. | Compute goes only to each robot's newest frame. |
| Backpressure | Bounded downlink queues, disconnection of slow consumers, and an ingress limit per connection (messages and bytes per second, every message counted before it is decoded). A server-advised frame rate is a planned experiment (section 12), not built. | One robot cannot cost the others event-loop time or memory. |

In addition, the CPU budget of section 6.4 paces the depth worker at this load; the container was throttled for `{{massive.cpu.throttled_ms_per_s}}` ms per second.

**Why refusal is not the main mechanism.** A refused robot is a starved robot, which is what the brief asks to avoid. Refusal would be justified only against a stated minimum useful rate per robot. No such rule is built; the useful capacity of section 8.2 is the number it would be set from.

### 8.2 What "not starved" means here

- **Fairness.** On each model, the lowest per-robot command rate as a fraction of the mean is `{{massive.cls.fairness.min_over_mean}}` for classification and `{{massive.depth.fairness.min_over_mean}}` for depth. Jain index: `{{massive.cls.fairness.jain}}` and `{{massive.depth.fairness.jain}}`.
- **Bounded gap.** Each robot's longest command gap stays within the scheduler's bound (section 6.3): `{{massive.depth.command_gap_ms.max}}` ms against `{{massive.depth.command_gap_bound_ms}}` ms for depth.
- **Usefulness.** Fair shares of too little are still too little. Watchdog coverage states this as a number: the fraction of time a robot holds an unexpired command is `{{massive.cls.watchdog_coverage.mean}}` for classification and `{{massive.depth.watchdog_coverage.mean}}` for depth at 50 robots. Within the watchdog limits the system serves `{{massive.cls.useful_robots}}` robots for classification and `{{massive.depth.useful_robots}}` for depth. That is the system's useful capacity per model, and the number a minimum-rate admission rule would be set from.

### 8.3 Memory budget

| Component | Predicted bound | Measured at 50 robots |
|---|---|---|
| Gateway process | base + N x (one frame in the slot + connection buffers) | `{{massive.gateway.rss_mb}}` MB |
| Worker `cls` | model + arena for the largest batch | `{{massive.worker_cls.rss_mb}}` MB |
| Worker `depth` | model + arena for the largest batch | `{{massive.worker_depth.rss_mb}}` MB |
| Robot fleet | base + frame pool + N x robot state + samples | `{{massive.fleet.rss_mb}}` MB |
| Kernel socket memory | N x (receive buffer + send buffer), both ends | `{{massive.memory.sock_mb}}` MB |
| Container total (anonymous + socket) | sum of the above | `{{massive.memory.anon_sock_mb_end}}` MB of 4 GB |

The leak signal is anonymous plus socket memory. `memory.current` also counts page cache, which grows when result files are written inside the container.

## 9. The five demos

Each demo is a scenario of section 11.2, run several times as gated runs, in both topologies. The tables show the `single` topology. The predicted values come from the capacity model (`bench/capacity_model.py`), fed by the gated model benchmark alone; no demo run feeds it (section 6.3). Preliminary predictions, computed from the batch times of a container spike on the test machine, were committed before the first end-to-end run; the git history keeps them.

### 9.1 Freshness under steady load

- **Procedure.** 1 robot, 30 fps, 30 s.
- **Assertions.** Freshness shows no upward trend. The frame conservation identities hold.
- **Evidence.** `steady_freshness.png` (freshness of every command against time, per model, with the thresholds marked), `steady.md`.

| Quantity | `cls` predicted | `cls` measured | `depth` predicted | `depth` measured |
|---|---|---|---|---|
| Freshness p50 (ms) | `{{predict.steady.cls.freshness_ms.p50}}` | `{{steady.cls.freshness_ms.p50}}` | `{{predict.steady.depth.freshness_ms.p50}}` | `{{steady.depth.freshness_ms.p50}}` |
| Freshness p95 (ms) | `{{predict.steady.cls.freshness_ms.p95}}` | `{{steady.cls.freshness_ms.p95}}` | `{{predict.steady.depth.freshness_ms.p95}}` | `{{steady.depth.freshness_ms.p95}}` |
| Freshness p99 (ms) | `{{predict.steady.cls.freshness_ms.p99}}` | `{{steady.cls.freshness_ms.p99}}` | `{{predict.steady.depth.freshness_ms.p99}}` | `{{steady.depth.freshness_ms.p99}}` |
| Freshness max (ms) | `{{predict.steady.cls.freshness_ms.max}}` | `{{steady.cls.freshness_ms.max}}` | `{{predict.steady.depth.freshness_ms.max}}` | `{{steady.depth.freshness_ms.max}}` |
| Commands per second | `{{predict.steady.cls.commands_per_s}}` | `{{steady.cls.commands_per_s}}` | `{{predict.steady.depth.commands_per_s}}` | `{{steady.depth.commands_per_s}}` |
| Freshness trend (ms per s) | 0 | `{{steady.cls.freshness_trend_ms_per_s}}` | 0 | `{{steady.depth.freshness_trend_ms_per_s}}` |
| Peak command age p95 (ms) | `{{predict.steady.cls.command_age_peak_ms.p95}}` | `{{steady.cls.command_age_peak_ms.p95}}` | `{{predict.steady.depth.command_age_peak_ms.p95}}` | `{{steady.depth.command_age_peak_ms.p95}}` |

The predicted p95, p99 and max carry no service-time spread and no fire lag (section 6.3). `steady_from_metrics.md` rebuilds every measured cell of this table but the freshness trend from each counted run's `/metrics` scrapes alone, and shows whether each agrees with the value from the samples within what `/metrics` resolves (section 10.1).

### 9.2 Late joiner

- **Procedure.** Robot A streams alone for 10 s. Robot B joins. Both run for 20 s more.
- **Assertions.** After B joins, A and B get equal shares of each model, within run-to-run spread. A's p95 freshness after the join stays within run-to-run spread of its value before.
- **Evidence.** `late_joiner_throughput.png` (per-robot commands per second over time, with the join marked), `late_joiner.md`.

| Robot | Window | `cls` commands/s | `cls` p95 freshness (ms) | `depth` commands/s | `depth` p95 freshness (ms) |
|---|---|---|---|---|---|
| A | before, 0 to 10 s | `{{late_joiner.before.a.cls.commands_per_s}}` | `{{late_joiner.before.a.cls.freshness_ms.p95}}` | `{{late_joiner.before.a.depth.commands_per_s}}` | `{{late_joiner.before.a.depth.freshness_ms.p95}}` |
| A | after, 10 to 30 s | `{{late_joiner.after.a.cls.commands_per_s}}` | `{{late_joiner.after.a.cls.freshness_ms.p95}}` | `{{late_joiner.after.a.depth.commands_per_s}}` | `{{late_joiner.after.a.depth.freshness_ms.p95}}` |
| B | after, 10 to 30 s | `{{late_joiner.after.b.cls.commands_per_s}}` | `{{late_joiner.after.b.cls.freshness_ms.p95}}` | `{{late_joiner.after.b.depth.commands_per_s}}` | `{{late_joiner.after.b.depth.freshness_ms.p95}}` |
| A, predicted | after | `{{predict.late_joiner.after.a.cls.commands_per_s}}` | | `{{predict.late_joiner.after.a.depth.commands_per_s}}` | |

B's first depth command arrives `{{late_joiner.b.depth.first_command_ms}}` ms after it joins. Time for A and B to settle at equal depth rates, counted in 1 s bins: `{{late_joiner.depth.settle_s}}` s.

### 9.3 Scaling sweep

- **Procedure.** 1, 3, 5 and 10 robots, 30 s each, as separate runs.
- **Assertions.** The conservation identities hold. Per-robot rates within one run are equal on each model, within run-to-run spread.
- **Evidence.** `sweep_throughput.png`, `sweep_freshness.png`, `sweep_batch_size.png`, `sweep_cpu.png`, each with the predicted curve drawn in; `sweep.md`. The inflection point is discussed in section 6.5.

| Robots | `cls` commands/s per robot (predicted / measured) | `cls` freshness p50 / p95 (ms) | `depth` commands/s per robot (predicted / measured) | `depth` freshness p50 / p95 (ms) |
|---|---|---|---|---|
| 1 | `{{predict.sweep.n1.cls.commands_per_s_per_robot.mean}}` / `{{sweep.n1.cls.commands_per_s_per_robot.mean}}` | `{{sweep.n1.cls.freshness_ms.p50}}` / `{{sweep.n1.cls.freshness_ms.p95}}` | `{{predict.sweep.n1.depth.commands_per_s_per_robot.mean}}` / `{{sweep.n1.depth.commands_per_s_per_robot.mean}}` | `{{sweep.n1.depth.freshness_ms.p50}}` / `{{sweep.n1.depth.freshness_ms.p95}}` |
| 3 | `{{predict.sweep.n3.cls.commands_per_s_per_robot.mean}}` / `{{sweep.n3.cls.commands_per_s_per_robot.mean}}` | `{{sweep.n3.cls.freshness_ms.p50}}` / `{{sweep.n3.cls.freshness_ms.p95}}` | `{{predict.sweep.n3.depth.commands_per_s_per_robot.mean}}` / `{{sweep.n3.depth.commands_per_s_per_robot.mean}}` | `{{sweep.n3.depth.freshness_ms.p50}}` / `{{sweep.n3.depth.freshness_ms.p95}}` |
| 5 | `{{predict.sweep.n5.cls.commands_per_s_per_robot.mean}}` / `{{sweep.n5.cls.commands_per_s_per_robot.mean}}` | `{{sweep.n5.cls.freshness_ms.p50}}` / `{{sweep.n5.cls.freshness_ms.p95}}` | `{{predict.sweep.n5.depth.commands_per_s_per_robot.mean}}` / `{{sweep.n5.depth.commands_per_s_per_robot.mean}}` | `{{sweep.n5.depth.freshness_ms.p50}}` / `{{sweep.n5.depth.freshness_ms.p95}}` |
| 10 | `{{predict.sweep.n10.cls.commands_per_s_per_robot.mean}}` / `{{sweep.n10.cls.commands_per_s_per_robot.mean}}` | `{{sweep.n10.cls.freshness_ms.p50}}` / `{{sweep.n10.cls.freshness_ms.p95}}` | `{{predict.sweep.n10.depth.commands_per_s_per_robot.mean}}` / `{{sweep.n10.depth.commands_per_s_per_robot.mean}}` | `{{sweep.n10.depth.freshness_ms.p50}}` / `{{sweep.n10.depth.freshness_ms.p95}}` |

| Robots | `cls` mean batch size | `cls` share of batches of 1 | `depth` mean batch size | CPU utilization (of 2 CPUs) | Throttled (ms per s) |
|---|---|---|---|---|---|
| 1 | `{{sweep.n1.cls.batch_size.mean}}` | `{{sweep.n1.cls.batch_size.share_1}}` | `{{sweep.n1.depth.batch_size.mean}}` | `{{sweep.n1.cpu.utilization}}` | `{{sweep.n1.cpu.throttled_ms_per_s}}` |
| 3 | `{{sweep.n3.cls.batch_size.mean}}` | `{{sweep.n3.cls.batch_size.share_1}}` | `{{sweep.n3.depth.batch_size.mean}}` | `{{sweep.n3.cpu.utilization}}` | `{{sweep.n3.cpu.throttled_ms_per_s}}` |
| 5 | `{{sweep.n5.cls.batch_size.mean}}` | `{{sweep.n5.cls.batch_size.share_1}}` | `{{sweep.n5.depth.batch_size.mean}}` | `{{sweep.n5.cpu.utilization}}` | `{{sweep.n5.cpu.throttled_ms_per_s}}` |
| 10 | `{{sweep.n10.cls.batch_size.mean}}` | `{{sweep.n10.cls.batch_size.share_1}}` | `{{sweep.n10.depth.batch_size.mean}}` | `{{sweep.n10.cpu.utilization}}` | `{{sweep.n10.cpu.throttled_ms_per_s}}` |

### 9.4 Disconnect

- **Procedure.** 3 robots for 20 s. One robot's process is killed with SIGKILL. The other two run for 20 s more.
- **Assertions.** The survivors' depth throughput rises to the predicted value. No errors. The dead robot's slot is released and its connection is counted as closed.
- **Evidence.** `disconnect_throughput.png` (per-robot commands per second over time, with the kill marked), `disconnect.md`.

| Quantity | Before, 0 to 20 s | After, 20 to 40 s |
|---|---|---|
| Survivors' `depth` commands/s per robot | `{{disconnect.before.survivors.depth.commands_per_s_per_robot}}` | `{{disconnect.after.survivors.depth.commands_per_s_per_robot}}` |
| Survivors' `cls` commands/s per robot | `{{disconnect.before.survivors.cls.commands_per_s_per_robot}}` | `{{disconnect.after.survivors.cls.commands_per_s_per_robot}}` |
| Survivors' `depth` p95 freshness (ms) | `{{disconnect.before.survivors.depth.freshness_ms.p95}}` | `{{disconnect.after.survivors.depth.freshness_ms.p95}}` |
| Survivors' `cls` p95 freshness (ms) | `{{disconnect.before.survivors.cls.freshness_ms.p95}}` | `{{disconnect.after.survivors.cls.freshness_ms.p95}}` |
| Predicted survivors' `depth` commands/s per robot | `{{predict.disconnect.before.survivors.depth.commands_per_s_per_robot}}` | `{{predict.disconnect.after.survivors.depth.commands_per_s_per_robot}}` |

Time from the kill until the server closes the connection: `{{disconnect.detect_ms}}` ms. Time until the survivors' depth rate settles, counted in 1 s bins: `{{disconnect.depth.recovery_s}}` s.

### 9.5 Massive connections

- **Procedure.** 50 robots at 30 fps for 30 s. The `soak` scenario repeats this for 5 minutes for the memory check.
- **Assertions.** No crash and no worker restart. Memory is flat. On each model no robot's command rate falls away from the others. Every robot's longest command gap is within the scheduler's bound. All 50 connections are accepted.
- **Evidence.** `massive_fairness.png` (distribution of per-robot command rates per model), `massive_memory.png`, `soak_memory.png`, `coverage_vs_watchdog.png`, `massive.md`.

| Quantity | `cls` | `depth` |
|---|---|---|
| Commands/s per robot, min / mean / max | `{{massive.cls.commands_per_s_per_robot.min}}` / `{{massive.cls.commands_per_s_per_robot.mean}}` / `{{massive.cls.commands_per_s_per_robot.max}}` | `{{massive.depth.commands_per_s_per_robot.min}}` / `{{massive.depth.commands_per_s_per_robot.mean}}` / `{{massive.depth.commands_per_s_per_robot.max}}` |
| Predicted commands/s per robot | `{{predict.massive.cls.commands_per_s_per_robot.mean}}` | `{{predict.massive.depth.commands_per_s_per_robot.mean}}` |
| Lowest over mean rate | `{{massive.cls.fairness.min_over_mean}}` | `{{massive.depth.fairness.min_over_mean}}` |
| Jain index | `{{massive.cls.fairness.jain}}` | `{{massive.depth.fairness.jain}}` |
| Longest command gap (ms), measured / bound | `{{massive.cls.command_gap_ms.max}}` / `{{massive.cls.command_gap_bound_ms}}` | `{{massive.depth.command_gap_ms.max}}` / `{{massive.depth.command_gap_bound_ms}}` |
| Freshness p50 / p95 / p99 / max (ms) | `{{massive.cls.freshness_ms.p50}}` / `{{massive.cls.freshness_ms.p95}}` / `{{massive.cls.freshness_ms.p99}}` / `{{massive.cls.freshness_ms.max}}` | `{{massive.depth.freshness_ms.p50}}` / `{{massive.depth.freshness_ms.p95}}` / `{{massive.depth.freshness_ms.p99}}` / `{{massive.depth.freshness_ms.max}}` |
| Stale ratio | `{{massive.cls.stale_ratio}}` | `{{massive.depth.stale_ratio}}` |
| Watchdog coverage, min / mean | `{{massive.cls.watchdog_coverage.min}}` / `{{massive.cls.watchdog_coverage.mean}}` | `{{massive.depth.watchdog_coverage.min}}` / `{{massive.depth.watchdog_coverage.mean}}` |

| Quantity | Value |
|---|---|
| Connections accepted / refused | `{{massive.connections.accepted}}` / `{{massive.connections.rejected}}` |
| CPU utilization (of 2 CPUs) | `{{massive.cpu.utilization}}` |
| Throttled (ms per s) | `{{massive.cpu.throttled_ms_per_s}}` |
| Simulator fire lag p99 (ms) | `{{massive.fleet.fire_lag_ms.p99}}` |
| Memory, anonymous + socket, after the connections settle / at the end (MB) | `{{massive.memory.anon_sock_mb_start}}` / `{{massive.memory.anon_sock_mb_end}}` |
| Memory slope over the 5-minute soak (MB per minute) | `{{soak.memory.slope_mb_per_min}}` |
| Change in fd count / task count over the soak | `{{soak.fds.delta}}` / `{{soak.tasks.delta}}` |

### 9.6 Topology comparison

Every scenario also runs in the `split` topology, where the robots have their own container and the cloud has the 2 CPUs to itself. The robots' container has a memory limit of 2 GB and no CPU limit, so the simulator cannot be the bottleneck of that run; its CPU is recorded from its own cgroup. The pacing reserve of section 6.4 is one configured value in both topologies, so the depth worker's share is the same in both, and the comparison shows timing and throttling, not extra depth capacity. In `split`, frames also cross a veth pair and a bridge whose MTU is smaller than a frame, where loopback carries a whole frame in one segment, so the transport path differs as well. Runs alternate between the two topologies.

| Quantity at 50 robots | `single` | `split` |
|---|---|---|
| `cls` commands/s per robot | `{{massive.cls.commands_per_s_per_robot.mean}}` | `{{split.massive.cls.commands_per_s_per_robot.mean}}` |
| `depth` commands/s per robot | `{{massive.depth.commands_per_s_per_robot.mean}}` | `{{split.massive.depth.commands_per_s_per_robot.mean}}` |
| `cls` freshness p95 (ms) | `{{massive.cls.freshness_ms.p95}}` | `{{split.massive.cls.freshness_ms.p95}}` |
| `depth` freshness p95 (ms) | `{{massive.depth.freshness_ms.p95}}` | `{{split.massive.depth.freshness_ms.p95}}` |
| Simulator CPU (cores) | `{{massive.fleet.cpu_cores}}` | `{{split.massive.fleet.cpu_cores}}` |
| Simulator fire lag p99 (ms) | `{{massive.fleet.fire_lag_ms.p99}}` | `{{split.massive.fleet.fire_lag_ms.p99}}` |
| Throttled (ms per s) | `{{massive.cpu.throttled_ms_per_s}}` | `{{split.massive.cpu.throttled_ms_per_s}}` |

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

## 11. Run-output contract

### 11.1 Layout

```
results/
  raw/<scenario_key>/<run_id>/              one directory per evidence run (section 11.3)
  raw/<scenario_key>/<run_id>.attempt<k>/   an interrupted or failed attempt, kept
  raw/compare/<candidate>/<arm>/<run_id>/   comparison runs (section 12), never used for a demo fact
  gates/                                    the test gate's output and its parsed result, per image and code digest
  summary/facts.json                        every number this document uses (section 11.4)
  summary/t_b.json                          the capacity model's input, from the model benchmark
  summary/predictions.json                  the capacity model's predictions
  summary/excluded.json                     every run directory that is not counted, with its reasons
  summary/failed_checks.json                counted runs with a failed check
  tables/*.md                               generated tables
  plots/*.png                               generated plots
  ledger.jsonl                              one line per candidate (section 12)
```

- The results root is `results/` in the repository. The environment variable `RRI_RESULTS` moves it; the make targets and the compose files follow it.
- A run is identified by its run spec: scenario key, topology, phase mode, model set, the resolved server config, the link model, the hash of any extra compose file, the candidate and arm of a comparison, the code digest and the repeat index. The run id joins the topology, a hash of the spec without the repeat index and the code digest, a prefix of the code digest and the repeat index. The seed follows from the repeat index. A change to the code or to any setting gives new run ids.
- The code digest is a hash over the files that decide what a run does: the source of the gateway, the workers, the robots, the shared code and the scripts, the modules of the runner and the scenario driver, the Dockerfile, the compose files and the list of packages the image installs. Each run also records its commit and whether those files differ from it.
- Resume. A run directory with a valid `DONE` is finished and is skipped, whatever the state of its summary. A directory without one is renamed to the next free attempt and its spec runs again. Attempts are never deleted, and aggregation counts them. After a code change, `make evidence` produces a complete new set.
- Each run happens in fresh containers: before it starts, the runner removes every container of the project.
- Aggregation, tables and plots read only the summaries, `summary/` and the ledger, never samples or scrapes; the one exception is `steady_from_metrics.md` (section 11.5), which reads the scrapes of the counted `steady` runs to show that `/metrics` alone gives the table of section 9.1. A run's summary is written once the run is finished, and can be written again without redoing the run: when the summarizer changes, `make report` first writes the summary of each finished run again from the run's files and samples, and aggregation stops when the summary of a counted run was written by another version of the summarizer.
- Per-command samples (`samples/`) are large and are not committed. Everything else is, including the scrapes, which are compressed.

### 11.2 Scenarios

| Scenario | Robots | Timeline | Used for |
|---|---|---|---|
| `steady` | 1 | 30 s | D1, Q2 |
| `late_joiner` | 2 | A from 0 s, B joins at 10 s, end at 30 s | D2 |
| `sweep_n1`, `sweep_n3`, `sweep_n5`, `sweep_n10` | 1, 3, 5, 10 (one scenario key each) | 30 s each | D3, Q3, Q4 (the 5-robot runs) |
| `disconnect` | 3 | SIGKILL of one robot process at 20 s, end at 40 s | D4 |
| `worker_kill` | 2 | SIGKILL of the depth worker at 20 s, end at 40 s | R3 |
| `massive` | 50 | 30 s | D5, Q1 |
| `soak` | 50 | 5 minutes | R1 |
| `model_bench` | none; always on the real models, first in each repeat | Batch time per model, batch size and thread count; decode, preprocess and postprocess; the dispatch round trip; a WebSocket run of 50 robots at 30 fps with no model loaded, for the I/O cost per frame and the round trip. Each command runs in its own container with the image's limits | Q1c, Q3, section 6.2; the only measured input of the predictions (section 6.3) |

Every scenario takes a topology (`single` or `split`), a phase mode, a model set and overrides of the server config. The phase mode is random: each robot's phase is drawn from the run's seed. Aligned phases are a planned experiment (section 12), not part of the default evidence. The model sets are `real` (the published models), `fake` (a fixed service time per frame and no model file) and `null` (a result at once, which isolates transport and scheduling). The experiments of section 12 are these scenarios with one override each, except the FIFO control, which also turns the skip rule off: the configuration refuses the skip rule on a queue. Overrides reach the gateway and its workers through the environment variable `RRI_OVERRIDES`, a JSON object over the defaults; the topology comes from `RRI_TOPOLOGY`, which each compose file sets. A run with changed settings starts a fresh container with them, and `/metrics` echoes the effective configuration.

Order. The evidence runs `REPEATS` repeats, three by default. Each repeat runs `model_bench` first, then `steady`, `late_joiner`, the sweep, `disconnect`, `worker_kill`, `massive` and `soak`. The sweep's robot counts run in increasing order in odd repeats and in decreasing order in even ones. Each serving scenario runs in both topologies, `single` first in odd repeats and `split` first in even ones, so two repeats give the order A B B A (section 9.6).

### 11.3 Files written by one run

| File | Committed | Content |
|---|---|---|
| `config.json` | yes | The run spec: scenario key, topology, phase mode, model set, the resolved server config, the link model (delay and jitter per direction), the extra compose file and its hash, candidate and arm (null outside a comparison), the code digest and the repeat index. The run id and the seed. Robots: count, frame rate, subscribed models, stale thresholds, watchdog limits, planned start and stop. The windows, the planned event times relative to t0, and the worker that the scenario kills with its planned time, null in every scenario but `worker_kill`. `hand_run`, true when the scenario driver was started by hand and wrote this file itself; such a run has no run spec and is never counted. |
| `env.json` | yes | CPU model and visible CPU count, the cpuset, machine architecture, kernel, the host's total memory, container engine and version, cgroup version, the limits as read inside the container. Python and package versions. Model file hashes. Frame set: count, JPEG quality, smallest, mean and largest size in bytes. Git commit, dirty flag, code digest, image digest. Start time in UTC. In the `split` topology, the limits of the robots' container. |
| `env_inside.json` | yes | What the scenario driver read inside the container at the start: cgroup limits, CPU model, kernel, packages, model hashes, frame-set statistics. `env.json` takes its inside fields from it. |
| `gate.json` | yes | Five gates and `gated`, which is true only when all five pass. `tests_passed`: the test suite in this image for this code digest, with the commit, the dirty flag, the counts, and the gate file under `results/gates/` with the hash of its output; it passes when nothing failed and nothing was skipped (the tests that need the host's docker CLI are not selected). `image_built`: the image built in this invocation, its digest, and the image of every container of the run. `limits_ok`: the CPU, memory and swap limits, expected and read. `fresh`: every container of the run is new, and no connection was accepted before t0. `config_ok`: the config echo of the first successful scrape (section 10.8), apart from `version`, equals the resolved server config in `config.json`, with the paths that differ; null for `model_bench`, which starts no gateway. A run whose gate fails still writes its files and `DONE`. |
| `plans/` | yes | The plan file of each fleet process: t0, the duration, and each robot's start and stop. A robot that the scenario kills runs alone in its own process, with its own seed, and only the kill ends it. |
| `events.jsonl` | yes | One line per event, with its time in seconds relative to t0. The first line is `t0`, which also holds t0 on the shared monotonic clock. Then `server_ready` (from the gateway's own clock, so before t0), `robot_start`, `robot_join`, `robot_kill`, `robot_stop`, `scenario_end`, `final_poll` (whether the final poll settled and how long it took), and, in `worker_kill` runs, `fault`: the model and the pid of the worker killed, at the time read in the gateway's container right after the kill; the runner adds this line once the driver has finished. |
| `scrapes.jsonl.gz` | yes | Gzip, one line per scrape: at t0 plus every whole second, from one second before t0 to the end, then a final scrape after a bounded wait for every robot to disconnect and send its last telemetry. Each line: the planned second, the driver's clock just before the request and just after the response, whether the scrape succeeded, the full `/metrics` object, and a `fleet` object with the `processes` fields of every fleet process and of the scenario driver and, in the `split` topology, the `cgroup` fields of the robots' container. |
| `summary.json` | yes | Everything aggregation needs, below. |
| `reports/*.json` | yes | `model_bench` only: one report per benchmark command, each from its own container with the image's limits. That run writes `config.json`, `env.json` (built from the reports' environment blocks), `gate.json`, these reports, `DONE` and `summary.json`; its `fresh` gate checks only that every container was new. |
| `samples/robot_<id>.npz` | no | Per command: session, model, frame number, capture time, send time, receive time, gateway result time, applied or the reject reason, batch size, every stage duration. Per tick: frame number, scheduled time, actual send time, and the outcome: sent, skipped because the previous frame was unsent, or passed because the simulator woke late. A robot killed with SIGKILL leaves no file. |
| `DONE` | yes | Written by the runner once the other files of the run are complete: the hash of every file of the run except `samples/` and `summary.json`, and, apart, the hash of `summary.json`, which is updated when the summary is written again. A directory without a valid `DONE` is an attempt (section 11.1). |

`summary.json`:

| Key | Content |
|---|---|
| `summarizer_digest` | Hash of `bench/summarize.py` and every module of this repository it imports; a summary with another digest is written again by `make report`, from the run's files and samples, before the run can count |
| `run` | Scenario, run id, repeat, topology, duration, whether gated |
| `validity` | Fire lag p99 and whether the run is simulator-limited, the count of each tick outcome, throttled time per second, whether the conservation identities held, whether the gateway's clock lay between the driver's readings around the first scrape, failed scrapes, samples dropped, whether the final scrape settled, notes |
| `windows` | Named time windows: `all` for every scenario, `before` and `after` for `late_joiner` and `disconnect`; for each, `span_s`, the time between the two scrapes its counter rates are taken over |
| `models.<m>.<window>` | Robots; commands per second in total and per robot (min, mean, max); applied commands per second per robot; freshness p50, p95, p99, max and mean, exact from samples, and `freshness_trend_ms_per_s`, its least-squares slope against receive time; stale count and ratio; out-of-order count; command gap p50, p95 and max, and the open interval before each robot's first command (`first_gap`); peak command age p50, p95 and max; watchdog coverage (min, mean); fairness (Jain index, lowest over mean); batch-size distribution and mean; batch time p50 and p95; measured capacity and utilization; per stage: mean, p50, p95, p99; end-to-end mean, stage sum, residual |
| `models.<m>.curves` | Freshness counts in 1 ms bins, and peak command age in 1 ms bins weighted by time, for the two curves of section 5.3 |
| `stage_all.<stage>` | Each stage over both models' commands pooled, with `transport_up` and `fire_lag` counted once per frame |
| `gateway.<window>` | Gateway CPU in cores, CPU-ms per frame received, event-loop lag p99, bytes received per second |
| `robots.<id>.<m>.<window>` | Commands per second; freshness p50, p95, p99, max; stale count; longest command gap; the open interval before its first command; watchdog coverage |
| `survivors.<m>.<window>` | Scenarios that kill a robot: commands per second per surviving robot, and freshness p95 over the survivors' commands |
| `fault` | `worker_kill` runs, null in the others: the model and the pid killed, the planned and the actual kill time and how late the kill was, the outage (from the kill to the first command of that model, over every robot, whose batch was dispatched after the kill), and the change in each worker's `restarts` between the scrape before t0 and the final scrape |
| `cpu.<window>` | Utilization (cgroup usage over the limit), cores per process (gateway, both workers, every fleet process, the scenario driver), throttled time per second, share of periods throttled; in the `split` topology, the cores of the robots' container |
| `memory` | Anonymous plus socket memory once the connections have settled and at the end, its slope between the two, the peak, socket memory and RSS per process at the end, the change in open file descriptors and in event-loop tasks between the same two scrapes |
| `connections` | Accepted, closed by reason, rejected by reason, from the scrape before t0 to the final scrape |
| `series` | For the plots: in `steady`, the freshness of every command against time; per robot and model, commands in each second from t0; anonymous plus socket memory at each scrape |
| `checks` | Name, passed and observed value, for each demo check |

Every quantity in this table is defined in `bench/summarize.py`, where it is computed. Five definitions change what a number means. A command belongs to a window when its receive time does, and a tick when its capture time does. A rate from a counter is its difference over the actual time between the two scrapes that bound the window (`span_s`), while a command rate is the count of samples over the window's length. The longest command gap and the peak command age include the open intervals at the edges: from the later of the window's start and the robot's start to its first command, and from its last command to the earlier of the window's end and the robot's stop. Watchdog coverage counts the time before the first applied command as uncovered. Batch statistics come from differences of scrapes, so their time resolution is one second.

### 11.4 Fact keys

`results/summary/facts.json` holds the model set, the code digest, the ids of the counted runs, and one fact per key. A fact is the median over its counted runs, with its unit, the number of runs, the lowest and highest value, the run ids, how many of them come from a dirty tree, the failed checks of each run, and the number of attempts and failed attempts behind it. Some keys are not medians. A statistic that is itself a maximum or a minimum (a key that ends in `max` or `min`, and `fairness.min_over_mean`) takes the largest or the smallest run value, so a reported extreme is the extreme of every counted run. `command_gap_bound_ms` takes the smallest of the runs' bounds, so the longest gap shown beside it is compared with the tightest bound of any run. The Breakdown keys of a model all come from one run, the run whose end-to-end mean is the median, so that the stage means add up (section 7). A fact that is a string takes the latest run's value. A run with a failed check is counted, not dropped: this document reports what was measured, and `summary/failed_checks.json` lists the failures. Every key of the registry in `bench/facts.py` is present; a key with no counted run is null.

Which runs count. For a demo fact, the runs whose spec, apart from the repeat index and the seed, is the evidence plan's spec for that scenario key and topology on the model set and the code digest, with a valid `DONE`, all five gates passed, and not started by hand. Every other run directory is listed in `summary/excluded.json` with each reason that applies. Comparison runs feed only the ledger (section 12).

Two digest modes. In `current` mode, used by `make report` and `make numbers`, the code digest is computed from the working tree, so evidence from other code is not counted and the check says whether the evidence matches the code. In `recorded` mode, used by `make check`, the runs that `facts.json` recorded as counted are the runs counted, with their summaries as they are, so a code change alone does not fail the check of committed evidence. In `current` mode aggregation refuses to replace facts recorded for other code when no run of the working tree's code counts; `make report DIGEST=recorded` renders the recorded evidence again.

This document is rendered from a template, `docs/DESIGN.tmpl.md`. In it every measured value is a fact key in double braces, alone or as the whole of a code span, and every number that is not measured is a design constant, written as its literal. `docs/constants.json` declares each constant: its text, which is a single numeric literal; its category, one of brief, protocol, design, assumption or derived; and its source, the sentence of the brief, the protocol constant, the section of this document, the assumption or the formula over other constants that decides it. A value measured anywhere is never a constant. The check compares literals, not meanings: a number typed by hand passes when it equals a declared literal. The same file lists the identifiers with a digit that may stand in the text: assumption, question and demo ids, statistics, segments of fact keys, and names.

The number check fails, and prints where, when any of these holds:

1. A rendered file differs from rendering its template now.
2. `facts.json`, the tables or the ledger differ from what aggregation and table generation produce now from the same results.
3. The template holds a character with a numeric value outside a placeholder, other than a heading's section number when the headings are numbered in order, a number in a reference to a section that exists, an ordered-list number that is its position in the list, a declared identifier, or a numeric literal equal to a declared constant's text; or it holds a character reference written with an ampersand and a hash sign.
4. A constant's text is not one numeric literal, or its category is not one of the five.

A null fact renders as n/a: it means no counted run, or a key computed across runs that has no answer. On the real model set `make report` exits with an error while a key that this document uses is null; on the other model sets it only lists those keys.

Key form: `<scenario>.<subject>.<metric>`, with an optional window or robot-count segment after the scenario and an optional statistic at the end. A `split.` prefix selects the `split` topology; without it the `single` topology is meant. A `predict.` prefix selects the capacity model's prediction for the same key. `<m>` is `cls` or `depth`.

| Family | Keys |
|---|---|
| Steady | `steady.<m>.freshness_ms.{p50,p95,p99,max}`, `steady.<m>.commands_per_s`, `steady.<m>.commands_received`, `steady.<m>.stale_count`, `steady.<m>.stale_ratio`, `steady.<m>.out_of_order_count`, `steady.<m>.freshness_trend_ms_per_s`, `steady.<m>.command_age_peak_ms.p95` |
| Late joiner | `late_joiner.{before.a,after.a,after.b}.<m>.commands_per_s`, `late_joiner.{before.a,after.a,after.b}.<m>.freshness_ms.p95`, `late_joiner.<m>.settle_s`, `late_joiner.b.<m>.first_command_ms` |
| Sweep | `sweep.n{1,3,5,10}.<m>.commands_per_s_per_robot.{min,mean,max}`, `sweep.n{1,3,5,10}.<m>.freshness_ms.{p50,p95}`, `sweep.n{1,3,5,10}.<m>.batch_size.{mean,share_1}`, `sweep.n{1,3,5,10}.<m>.utilization`, `sweep.n{1,3,5,10}.cpu.{utilization,throttled_ms_per_s}`, `sweep.<m>.capacity_fps`, `sweep.<m>.inflection_n`, `sweep.<m>.inflection_between` |
| Breakdown | `sweep.n5.<m>.stage.<stage>_ms.{mean,p50,p95}` for the ten stages of section 7, `sweep.n5.<m>.stage_sum_ms.mean`, `sweep.n5.<m>.e2e_ms.mean`, `sweep.n5.<m>.residual_ms.mean` |
| Disconnect | `disconnect.{before,after}.survivors.<m>.commands_per_s_per_robot`, `disconnect.{before,after}.survivors.<m>.freshness_ms.p95`, `disconnect.<m>.recovery_s`, `disconnect.detect_ms` |
| Worker kill | `worker_kill.depth.outage_ms`, `worker_kill.{before,after}.<m>.commands_per_s_per_robot.mean` |
| Massive | `massive.<m>.commands_per_s_per_robot.{min,mean,max}`, `massive.<m>.commands_received`, `massive.<m>.fairness.{jain,min_over_mean}`, `massive.<m>.command_gap_ms.max`, `massive.<m>.command_gap_bound_ms`, `massive.<m>.freshness_ms.{p50,p95,p99,max}`, `massive.<m>.stale_count`, `massive.<m>.stale_ratio`, `massive.<m>.out_of_order_count`, `massive.<m>.watchdog_coverage.{min,mean}`, `massive.<m>.useful_robots`, `massive.stage.transport_up_ms.{p50,p99}`, `massive.stage.dispatch_rt_ms.p50`, `massive.cpu.{utilization,throttled_ms_per_s}`, `massive.bytes_in_per_s`, `massive.connections.{accepted,rejected}` |
| Massive, per process | `massive.gateway.{cpu_cores,cpu_ms_per_frame,rss_mb}`, `massive.gateway.loop_lag_ms.p99`, `massive.fleet.{cpu_cores,rss_mb}`, `massive.fleet.fire_lag_ms.p99`, `massive.worker_cls.rss_mb`, `massive.worker_depth.rss_mb`, `massive.memory.{anon_sock_mb_start,anon_sock_mb_end,sock_mb}` |
| Soak | `soak.memory.slope_mb_per_min`, `soak.fds.delta`, `soak.tasks.delta`, `soak.<m>.fairness.{jain,min_over_mean}`, `soak.<m>.commands_per_s_per_robot.{min,mean,max}` |
| Model bench | `model_bench.<m>.t{1,2}.b{1,2,4,8}.{wall_ms,cpu_ms}`, `model_bench.<m>.batch_cap`, `model_bench.depth.margin_ms`, `model_bench.decode.{full_ms,half_ms}`, `model_bench.<m>.preprocess_ms`, `model_bench.<m>.postprocess_ms`, `model_bench.dispatch_rt_ms` |
| Experiments | `ledger.<candidate>.outcome`, `ledger.<candidate>.main_delta`, `ledger.<candidate>.baseline_spread` |
| Environment | `env.cpu_model`, `env.kernel`, `env.engine`, `env.repeats` |

### 11.5 Plots and tables

| File | Shows |
|---|---|
| `steady_freshness.png` | Freshness of every command against time, per model, with rolling p95 and the thresholds |
| `late_joiner_throughput.png` | Per-robot commands per second over time, per model, with the join marked |
| `sweep_throughput.png` | Per-robot throughput against robot count, per model, measured points (the sweep's four counts and the 50-robot run) and predicted curve |
| `sweep_freshness.png` | Freshness p50 and p95 against robot count, per model, measured and predicted |
| `sweep_batch_size.png` | Batch-size distribution per model at each robot count |
| `sweep_cpu.png` | CPU utilization against robot count: total, per process, and throttled time |
| `disconnect_throughput.png` | Per-robot commands per second over time, with the kill marked |
| `massive_fairness.png` | Distribution of per-robot command rates, per model |
| `massive_memory.png`, `soak_memory.png` | Anonymous plus socket memory over time |
| `breakdown_n5.png` | Stage means per model as a stacked bar, with the residual |
| `stale_ratio_vs_threshold.png` | Stale ratio against candidate threshold, per model and scenario |
| `coverage_vs_watchdog.png` | Watchdog coverage against candidate limit, per model and robot count |
| `model_tb.png` | Per-frame wall and CPU time against batch size, per model and thread count |
| `fifo_vs_latest.png` | Freshness over time under the FIFO control and under latest-wins |
| `delay_sweep.png` | Freshness and stale ratio against injected one-way delay |
| `steady.md`, `late_joiner.md`, `sweep.md`, `disconnect.md`, `massive.md` | The demo tables of section 9 |
| `steady_from_metrics.md` | The table of section 9.1 rebuilt for each counted `steady` run from its `/metrics` scrapes alone, each cell beside the value from the samples, with whether the two agree within what `/metrics` resolves |
| `breakdown_n5.md` | The table of section 7 |
| `predicted_vs_measured.md` | Every prediction next to its measurement, with the gap and the run-to-run spread; a gap larger than the spread is marked for explanation, and the explanation is written in the text of this document |
| `topology.md` | The table of section 9.6 for every scenario |
| `memory_budget.md` | The table of section 8.3 |
| `faults.md` | The fault runs, a robot killed in `disconnect` and the depth worker killed in `worker_kill`: case, checks, recovery time |
| `ledger.md` | Every candidate of section 12 with its outcome, those not run included |
| `env.md` | The test machines |

Tables and plots are deterministic: the same inputs give the same bytes. An item whose runs do not exist (the FIFO control, the delay sweep, the fault runs, an empty ledger) is still written: a table with its header and one row that says "not run", or a figure that says "not run". Tables and plots use only the runs that `facts.json` and the ledger name (section 11.4). A plot of one run's series shows the counted run of the `single` topology with the lowest repeat index.

## 12. Tried and rejected

Each optimization or control is a candidate with a stated hypothesis, one main metric and guard metrics, each with the direction that counts as better, all written down in `bench/candidates.py` before it runs. A candidate runs one scenario on the real models in arm A, the defaults (for the depth caps, the default cap forced, so that both arms run the patched graph), and arm B, one change. The arms run in the order A, B, B, A, each run in fresh containers, to cancel drift. Comparison runs live under `results/raw/compare/` and never feed a demo fact.

The outcome compares the median of the B runs with the median of the A runs, metric by metric. Each metric has a threshold, the largest of four values: the spread of its A runs and the spread of its B runs (each highest minus lowest); 4% of the absolute value of its A median; and its absolute floor. The 4% is a design choice whose size the run-to-run variation of the container spike informed. The absolute floor is one frame period divided by the `steady` duration for the freshness trend (the scale of the `steady` check that freshness has no upward trend), 1 ms per s for throttled time (one hundredth of a quota period in each second, below any effect section 6.4 discusses), and zero for every other metric. The outcome is the first of these that holds: `rejected_guard`, when a guard's median moves in its worse direction by more than its threshold; `accepted`, when the main median moves in its better direction by more than its threshold; `rejected_regression`, when the main median moves in its worse direction by more than its threshold; `inconclusive`, when it moves in its better direction by more than half its threshold; otherwise `rejected_no_gain`.

Every candidate is logged in `results/ledger.jsonl` with its fields, the run ids of each arm, each metric's median per arm, the change, the spread of each arm and the outcome; a candidate with no runs has the outcome "not run". `ledger.md` is generated from that file, and the Outcome column below is filled from it.

| Candidate | Scenario | Change in B | Hypothesis | Main metric | Guard metrics | Outcome |
|---|---|---|---|---|---|---|
| `fifo_vs_latest` (control) | `steady`, `single` | FIFO queue, skip rule off | Under FIFO, freshness grows for the whole run; under latest-wins it is flat. | Depth freshness trend, lower | Depth commands per second, higher | `{{ledger.fifo_vs_latest.outcome}}` |
| `cls_cap_1` | `massive`, `single` | Classifier batch cap 1 | A cap of 1 lowers the classifier's freshness tail at 50 robots without lowering its throughput (H3). | Classifier freshness p95, lower | Classifier commands per second per robot (mean), higher | `{{ledger.cls_cap_1.outcome}}` |
| `depth_cap_2` | `massive`, `single` | Depth batch cap 2 | A larger depth cap spreads the dispatch cost, and lowers depth's freshness tail at 50 robots only if that cost is large (H3). | Depth freshness p95, lower | Depth commands per second per robot (mean), higher | `{{ledger.depth_cap_2.outcome}}` |
| `depth_cap_4` | `massive`, `single` | Depth batch cap 4 | As `depth_cap_2`, with a larger cap. | Depth freshness p95, lower | Depth commands per second per robot (mean), higher | `{{ledger.depth_cap_4.outcome}}` |
| `ort_default_threads` (control) | `sweep_n5`, `single` | Both thread counts left to ONNX Runtime | Default thread counts oversubscribe the quota and add throttling. | Throttled time, lower | Classifier freshness p99, lower; classifier commands per second per robot (mean), higher | `{{ledger.ort_default_threads.outcome}}` |
| `depth_unpaced` (control) | `massive`, `single` | Depth not paced | Without pacing, two saturated inference threads plus the frame I/O exceed the quota at 50 robots, and the kernel freezes the whole container for part of each period (H5). | Throttled time, lower | Classifier freshness p99, lower; simulator fire lag p99, lower; depth commands per second per robot (mean), higher | `{{ledger.depth_unpaced.outcome}}` |

Planned experiments, defined and not run in this evidence:

- **2-thread depth at 1, 3 and 5 robots.** A second thread shortens depth's service time while CPU is idle. Main metric: p95 peak command age for depth. Guards: classifier p99 freshness; throttled time.
- **Take the pending frame now against waiting for the next.** Waiting lowers peak command age with one robot, at a small cost in commands. Main metric: mean peak command age at 1 robot. Guard: depth commands per second.
- **Half-size JPEG decode.** Decoding at half size lowers the classifier's CPU per frame and moves its inflection point. Main metric: classifier capacity. Guard: top-1 agreement with full-size decode.
- **Server-advised frame rate at 50 robots.** Dropping frames at the source lowers gateway CPU and bytes on the wire. Main metrics: gateway CPU; bytes per second. Guards: freshness; fairness.
- **Quota against cpuset limit (control).** Under cpuset the same contention shows as time slicing, not freezes. Main metrics: throttled time; p99 freshness. Guard: throughput.
- **Per-robot classifier rate cap.** Capping the classifier frees CPU that a 2-thread depth worker can use. Main metric: p95 peak command age for depth. Guard: classifier coverage.
- **INT8 MiDaS.** Quantization lowers depth's service time. Main metric: depth service time. Guard: agreement of depth commands with the float model.
- **Aligned phases.** With every robot in phase, an unsaturated classifier sees all pending frames at once, so batches grow. Main metrics: the batch-size distribution and freshness p95 across the sweep, against random phases.

## 13. Limitations and what is not measured

- **Nothing in this skeleton is measured.** Every placeholder is open.
- **TLS.** Not built and its cost is not measured. The design assumes it ends at a load balancer.
- **Authentication.** A shared-token check exists as a hook. Real authentication is not built.
- **Fresh robot ids.** A robot id that has never been served goes first once (section 6.1), and ids are not authenticated (A21, A22). A client that connects under a new id each time is therefore served ahead of the others each time. The fairness properties of section 6.1 hold per robot id.
- **Wide-area networks.** All runs are on one host. Delay and jitter are injected by a link model in the simulator. Real WAN behaviour is not measured.
- **Packet loss.** Measured only if a loss run is made: it needs `tc netem` in the kernel, iproute2 in the image, and a robots container that runs as root with NET_ADMIN, which the default image and compose files do not give. Otherwise head-of-line blocking is argued, not measured.
- **Bandwidth saved by dropping frames at the source.** On a real uplink this is derived from frame size and rate, not measured. Bytes per second on loopback are measured if that experiment runs.
- **QUIC or datagram transport.** Not built.
- **Real cameras and real robots.** Frames are synthetic and pre-encoded. Commands drive a simulated state.
- **Model accuracy.** Commands are derived from model outputs on synthetic frames. Whether they are good commands is not evaluated.
- **Hardware dependence.** Every number belongs to the machine it was measured on. The inflection point is explained as utilization for that reason. The development machine mixes fast and slow cores and does not pin virtual CPUs, so results are repeated and reported with their spread. The I/O cost per frame that the pacing share reserves is a configured value measured on the test machine.
- **Co-located simulator.** In the `single` topology the simulator competes with the system under test. Fire lag flags runs where the simulator fell behind, and the `split` topology shows the difference.
- **Shared clock.** The latency breakdown relies on all processes sharing one monotonic clock. Across machines only robot-clock freshness and server dwell time are valid.
- **Timer resolution.** The event loop's timers have a resolution of 1 ms. A pacing release, a robot's tick and the loop-lag probe can be late by up to a millisecond, and the ping round trip is known to the millisecond, so on one host it is usually reported as null.
- **Connections before HTTP.** A connection that never sends its HTTP request is bounded in time by the handshake timeout but not in number (section 4.3: in production a load balancer sits in front of the gateway and bounds them).
- **Slow-consumer detection.** A robot that stops reading is found only after the kernel buffers of both ends have filled with commands, so the time to find it depends on those buffer sizes and on the command rate.
- **Declared frame rates.** The pacing share reserves the frame I/O of the frame rates that robots declare in the handshake, which refuses a rate above 60 frames per second. A robot that sends faster than it declares gets no more of either model, but its frames cost the gateway CPU that the reserve does not hold, up to the ingress limit of its connection.
- **Fake models.** The fake and null models use almost no CPU, so the feed-forward pacing share does not engage in runs with them.
- **Skip rule.** On the test machine depth's margin is positive, so the skip rule's condition is not met at 30 fps and the runs do not exercise it; `frames_skipped` in each run's `/metrics` shows whether it fired. It is covered by unit and property tests only.
- **Server-side percentiles** come from log-bucket histograms with about 9% resolution. Exact percentiles come from robot-side samples.
- **Per-process memory.** `processes.<p>.rss_bytes` is the kernel's approximate per-process counter from `/proc/<pid>/stat`. The leak check uses the cgroup's anonymous plus socket memory instead.
- **Damaged frames.** A frame whose image data is damaged but still decodes is served, and its command is computed from the pixels the decoder recovered. For such a frame the JPEG decoder writes one line to the worker's standard error, which cannot be switched off through OpenCV, so the container's log size is capped.
- **Scale beyond one container.** Section 14 is design reasoning, not measurement.
- **GPU.** Not used and not measured.

## 14. Beyond one container

- **More gateways.** One event loop saturates at 1000 / (30 x `{{massive.gateway.cpu_ms_per_frame}}`) robots. Past that, several gateway processes share the robots.
- **Sharding by robot.** Fairness then holds within a shard, not globally. Shards need similar load, or a robot's share depends on where it landed.
- **Scaling signal.** Command interval or utilization (demand over measured capacity), not CPU percentage. Depth runs at full utilization by design, so CPU percentage says nothing.
- **Workers on other hosts.** The IPC becomes a network hop. Pull and late binding still apply: the worker asks, the gateway picks the newest frames at that moment.
- **GPU workers.** Batch time grows more slowly than batch size, and the batch rule of section 6.2 then yields a larger cap with no change.
- **Lossy links.** A QUIC adapter behind the transport interface removes head-of-line blocking. The frame protocol does not change.
- **Edge.** TLS and authentication end at the load balancer, in front of the gateways.
- **Non-goals.** Authentication and TLS, model hot swap, persistence, exactly-once delivery.

## 15. Reproduction

The interface below is planned. The README will carry the final commands.

```
docker compose build                                       # downloads both models and verifies their sha256
docker compose up                                          # gateway and workers, limited to 2 CPUs and 4 GB; /metrics on port 8080
# one scenario by hand, against the running service; prints each robot's freshness; ungated and never counted
docker compose exec system python -m bench.run --run-dir /tmp/steady --scenario steady
docker compose exec system python -m robot.fleet --robots 3 --duration-s 30    # robots by hand, without the runner
docker compose run --rm system pytest -q                   # the tests, inside the image
make evidence                                              # on the host: every scenario, repeated, gated, fresh containers per run; skips finished runs
make compare                                               # on the host: the comparisons of section 12, each A, B, B, A in fresh containers
make report                                                # summaries where needed, aggregate, tables, plots, render this document, check its numbers, list failed checks
make numbers                                               # the number check alone, against the current code
make report DIGEST=recorded                                # render again from the runs facts.json recorded, when the code has changed since they ran
RRI_RESULTS=/absolute/path make evidence                   # a results root elsewhere; the make targets and the compose files follow it
docker compose -f docker-compose.split.yml up              # the split topology: the service alone
docker compose -f docker-compose.split.yml run --rm robots python -m robot.fleet --url ws://system:8080/ws --robots 3 --duration-s 30
```

How a number reaches this document: a gated run writes `results/raw/<scenario_key>/<run_id>/summary.json`; aggregation writes `results/summary/facts.json` from the counted runs; rendering fills the placeholders of the template; the number check fails if a number here is neither a rendered fact nor a declared design constant, or if this document, the facts, the tables or the ledger are out of date. `make report` and `make numbers` check against the code digest of the working tree, so they say whether the evidence matches the current code; `make check` checks against the runs that `facts.json` recorded, so a code change alone does not fail it (section 11.4).

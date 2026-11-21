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

One row per design question of the brief (Q), per demo (D), for the topology comparison (T) and for three non-functional requirements (R). Scenarios are defined in section 11, metric fields in section 10. The measured values in the Claim column are from the `single` topology. Every status is "planned" until its runs exist.

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
| A19 | The CPU limit is a quota (2 CPUs of time per period), not two pinned cores. Behaviour under a cpuset limit is a planned control (section 12) and is not run. | "Limited to 2 CPU cores" can be applied either way. |
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
| `telemetry` | robot to cloud | JSON, 1 Hz | the robot-reported fields of the [metrics contract](docs/metrics.md) |

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

Properties that the tests assert: frames are conserved (the identities of the [metrics contract](docs/metrics.md)); the frame served is always that robot's newest; frame numbers increase per robot, model and session; with the skip rule off, among robots with a pending frame, service counts differ by at most one, a faster sender gets no extra share, and a robot that reconnects gains no share, because its last-served time is kept per robot id for a while; with the skip rule on, every robot is still served within the bound of section 6.3. The skip rule acts only when one robot has a pending frame, so it does not change the order among robots that are waiting.

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
| H5 | Two saturated inference threads plus I/O exceed the quota at high N, and without pacing the kernel freezes the whole container for part of each period. | `massive` in `single`, with pacing on and off |
| H6 | The patched MiDaS graph is no faster than the published one at batch size 1. | `model_bench` |
| H7 | Python frame I/O at 50 robots uses a small share of one core. | `massive` |
| H8 | A second depth thread shortens the service time but raises CPU-ms per frame. | `model_bench` |

Predictions for the five demos are produced by running the real scheduler code under a simulated clock (`bench/capacity_model.py`). Every measured input comes from the model benchmark alone: the batch times T(b) in wall and CPU time, the dispatch round trip and its CPU cost, and the WebSocket round trip and its CPU cost per frame. The simulated gateway computes the pacing share from the configured I/O cost per frame, as the gateway does (section 6.4); the measured cost enters only the predicted CPU demand. No demo run feeds a prediction, so no prediction is fitted to the measurement it is compared with. The tables of section 9 show the predictions computed from the gated model benchmark. Preliminary predictions, computed from the batch times of a container spike, were committed before the first end-to-end run, and the git history keeps them. `predicted_vs_measured.md` sets each prediction beside its measurement in the `single` topology, with the gap and the run-to-run spread, and marks every gap larger than the spread for explanation. The explanation is written in the text of this document, at the claim the prediction concerns.

The model is an idealization, and its limits are stated before the first end-to-end run. Each batch takes the mean wall and CPU time that the model benchmark measured for its size with the worker alone in the container. Each transport and dispatch leg is a constant, set to the whole measured round trip that contains it, which is an upper bound. Robots send exactly on their ticks. The model has no spread in service time, no fire lag, no slowdown of a worker beside the other busy processes, no quota freeze, and no gateway work beyond the measured I/O cost. Its percentiles therefore spread only with frame phase and scheduling. The predicted p95, p99 and max carry no spread from service time or fire lag, so the measured tails are expected at or above them, to within the run-to-run spread of the batch time, and the measured maximum well above. The predicted throughput of a saturated model is a ceiling, to within the same spread. The predicted CPU demand counts the workers and the frame I/O only. Below saturation the classifier's freshness and batch size depend on the phase offsets drawn for a run, so those predictions hold for one draw. After the first end-to-end run a prediction changes only through its model-benchmark inputs; a change to the model's code, or to the scheduler code and defaults it runs, that moves a prediction is listed in section 13 with the predictions before and after it.

### 6.4 CPU budget

- The limit is a quota: the container may use Q x P of CPU time per period P. When total demand D exceeds Q, the kernel freezes every process in the container for the rest of the period. The gateway stops reading, the classifier stalls and, in the `single` topology, the simulator's timers fire late. So the freeze inflates the freshness tail and, in `single`, corrupts the measurement at the same moment.
- **Budget.** The gateway, the simulator and the classifier get what they need. The depth worker is saturated by design and is the one elastic consumer, so it is paced: after each batch it becomes eligible again only once its duty cycle is back under its share. The idle time is then taken by depth alone, not by the whole container.
- The pacing share is feed-forward, not a feedback loop. When a robot joins or leaves, the gateway recomputes it from the frame rates the connected robots declared, the workers' calibrated CPU time per frame and a configured CPU cost per frame of the frame I/O: the quota, minus a headroom, minus the predicted demand of the other consumers (the frame I/O at that configured cost, which covers the gateway and the co-located simulator and is reserved in both topologies, plus each unpaced worker at its calibrated CPU time per frame, capped at its threads, or at the quota when ONNX Runtime chooses its thread count), and never below a minimum share. When that leaves the paced worker at least as many cores as it can use (its threads, or the quota when ONNX Runtime chooses its thread count), it is not paced, and without a CPU quota nothing is paced. The I/O cost is one setting, `pacing.io_cpu_ms_per_frame`, whose default is set for the gateway and the simulator together. The evidence runs keep that default in the `split` topology, where the simulator runs outside the quota, so the share still reserves the simulator's part there and the depth worker is paced to the same share as in `single`. Throttled time, classifier p99 freshness and fire lag are compared with pacing on and off in the `single` topology (section 12).
- The share holds over each dispatch-to-dispatch span, not within one quota period, because a batch cannot be interrupted. A period that holds a whole depth batch must fit the classifier's CPU, the frame I/O and that batch in Q x P; where it does not, that period ends in a freeze that no share removes. This holds in both topologies; only the frame I/O under the quota differs, which is the gateway's in `split` and also the simulator's in `single`. A depth batch longer than the period (caps above 1) always overruns while the classifier is saturated.
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

The measured values of this section are from the `single` topology, where the simulator shares the quota with the service and is frozen with it when the quota runs out (section 6.4). The `simulator_valid` check of each run shows whether the simulator limited that run, and the `split.massive.*` keys give the service measured alone (sections 9.6 and 13).

### 8.1 What is implemented

The brief names four kinds of mechanism. The system uses all four, each for a different job:

| Mechanism | Implementation | Job |
|---|---|---|
| Connection admission | Every robot is accepted up to a configurable cap (default 64). Beyond it the handshake is refused with close code 1013. | A safety valve, not the main control. |
| Per-robot quota | Fair rotation: each model serves robots least recently served first, so each robot gets one N-th of each model's capacity. | No robot can take another's share. |
| Adaptive drops | The slot overwrite: each model serves a robot's newest frame, and older frames are dropped unprocessed. The skip rule of section 5.5 acts only while one robot is waiting, so it plays no part at 50 robots. | Compute goes only to each robot's newest frame. |
| Backpressure | Bounded downlink queues, disconnection of slow consumers, and an ingress limit per connection (messages and bytes per second, every message counted before it is decoded). A server-advised frame rate is a planned experiment (section 12), not built. | One robot cannot cost the others event-loop time or memory. |

In addition, the CPU budget of section 6.4 paces the depth worker at this load, though no share removes the freeze of a period that holds a whole depth batch. The container was throttled for `{{massive.cpu.throttled_ms_per_s}}` ms per second.

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

Each demo is a scenario of the [run-output contract](docs/run-output.md), run several times as gated runs, in both topologies. The tables show the `single` topology. The predicted values come from the capacity model (`bench/capacity_model.py`), fed by the gated model benchmark alone; no demo run feeds it (section 6.3). Preliminary predictions, computed from the batch times of a container spike on the test machine, were committed before the first end-to-end run; the git history keeps them.

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

The predicted p95, p99 and max carry no service-time spread and no fire lag (section 6.3). `steady_from_metrics.md` rebuilds every measured cell of this table but the freshness trend from each counted run's `/metrics` scrapes alone, and shows whether each agrees with the value from the samples within what `/metrics` resolves (the conventions of the [metrics contract](docs/metrics.md)).

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

`GET /metrics` returns one JSON object, and `GET /healthz` says whether the gateway accepts connections and every worker is ready. Their contract is in [docs/metrics.md](docs/metrics.md): the conventions for counters, gauges and time histograms, every field with its type, kind and meaning, the conservation identities that tests assert and every run checks at its end, and the config echo. A change to a field changes that file in the same commit, and a test compares a rendered snapshot with its tables.

## 11. Run-output contract

Every run writes its files under the results root, and aggregation turns the counted runs into the facts this document is rendered from. The contract for both is in [docs/run-output.md](docs/run-output.md): the layout of the results root, the scenarios and the order they run in, the files of one run and its summary, the fact keys with which runs count and what the number check checks, and the generated plots and tables.

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

Planned experiments. None of them was run, and this document reports no outcome for any of them. They are not candidates in `bench/candidates.py`, so the ledger has no line for them. Each one below gives the hypothesis, the main metric and guards, with the direction that counts as better, and the scenario, all in the `single` topology. An experiment that is run becomes a candidate first, with these fields.

- **2-thread depth at 1, 3 and 5 robots.** Depth is saturated from the first robot (H1) while the classifier leaves part of its core idle (section 4.6), so a second depth thread shortens depth's service time, and with it depth's peak command age, at a higher CPU cost per frame (H8). Main metric: depth `command_age_peak_ms` p95, lower. Guards: classifier `freshness_ms` p99, lower; `throttled_ms_per_s`, lower. Scenarios: `sweep_n1`, `sweep_n3` and `sweep_n5`, with `models.depth.threads` set to 2 in arm B. The model benchmark already measures depth's batch time and CPU time per frame on 2 threads with the worker alone (section 6.2); this experiment would measure the change beside the classifier, under the quota.
- **Take the pending frame now against waiting for the next one, at 1 robot.** With one robot, depth takes a frame that has waited in the slot for up to one frame period (the slot wait w of section 6.3). Waiting for the next frame when the pending one is older than half a frame period (section 6.1) shortens that wait, and so lowers depth's peak command age, at a cost in idle worker time and in commands. Main metric: depth `command_age_peak_ms` p95, lower. Guard: depth `commands_per_s`, higher. Scenario: `steady`, with the wait turned on in arm B. The wait is not built.
- **Half-size JPEG decode.** Decode is part of each model's CPU cost per frame, a larger part for the classifier (section 6.5), so decoding the classifier's frames at half size lowers that cost and raises its capacity. Main metric: classifier `commands_per_s_per_robot` mean, higher; at 50 robots both workers are expected to be saturated (section 4.6), so this rate is the classifier's capacity over N (section 6.3). Guard: top-1 agreement of the classifier with full-size decode on the frame set, computed offline, higher. Scenario: `massive`, with half-size decode for the classifier in arm B. The model benchmark already times both decode sizes with the worker alone (`model_bench.decode.full_ms`, `model_bench.decode.half_ms`).
- **Server-advised frame rate at 50 robots.** At 50 robots each model is expected to serve a robot well below the rate it sends (section 6.3), so most frames are overwritten in the slot after the gateway has received them (section 5.1). A rate that the server advises, close to the rate each robot is served, is backpressure at the source (section 8.1) and lowers gateway CPU and the bytes into the gateway. Main metrics: `massive.gateway.cpu_cores`, lower; `massive.bytes_in_per_s`, lower. Guards: classifier and depth `freshness_ms` p95, lower, because a robot that sends less often leaves an older frame in the slot; `fairness.min_over_mean` per model, higher. Scenario: `massive`, with the advised rate in arm B. The advised rate is not built (section 4.4).
- **CPU quota against a cpuset limit (control).** Under the quota, demand above two CPUs freezes the whole container for the rest of the period; under a cpuset of two CPUs the same contention shows as time slicing instead (section 6.4, A19). Without a quota nothing is paced, so depth also runs unpaced in arm B. Main metric: classifier `freshness_ms` p99, lower. Guards: simulator `fire_lag_ms` p99, lower; depth `commands_per_s_per_robot` mean, higher. Throttled time is not a metric here, because without a quota nothing is throttled. Scenario: `massive`, with a cpuset of two CPUs in place of the quota in arm B. The `limits_ok` gate expects the quota, so these runs would need a gate that accepts the cpuset instead.
- **Per-robot classifier rate cap.** A cap on how often the classifier serves each robot (section 6.1) frees the CPU it would spend on frames above the cap (section 6.5). At 1, 3 and 5 robots one depth thread is already busy all the time (H1) and cannot use that CPU; a 2-thread depth worker can. So this experiment is useful only on top of the 2-thread one: both arms run 2 depth threads, and arm B adds the cap. Main metric: depth `command_age_peak_ms` p95, lower. Guards: classifier `watchdog_coverage` min, higher; classifier `command_age_peak_ms` p95, lower. Scenarios: those of the 2-thread experiment. The cap is not built.
- **INT8 MiDaS.** The model's cost is part of depth's CPU cost per frame and sets its service time (section 6.5), so an INT8 copy of MiDaS-small lowers both. Main metric: depth's calibrated batch time at the cap, `workers.depth.t_b_ms`, lower. Guard: agreement of depth's commands (steering and speed, A3) with those of the float model on the frame set, computed offline, higher. Scenario: `steady`, with the INT8 graph in arm B; the model benchmark would time the INT8 graph first.
- **Aligned phases.** With every robot in phase, an unsaturated classifier sees all pending frames at once, so batches grow (section 6.2). Main metric: the classifier's mean batch size, from `models.cls.batch_size`, higher. Guard: classifier `freshness_ms` p95, lower. Scenarios: the four sweep scenarios, with random phases in arm A and aligned phases in arm B.

## 13. Limitations and what is not measured

- **Nothing in this skeleton is measured.** Every placeholder is open.
- **TLS.** Not built and its cost is not measured. The design assumes it ends at a load balancer.
- **Authentication.** A shared-token check exists as a hook. Real authentication is not built.
- **Fresh robot ids.** A robot id that has never been served goes first once (section 6.1), and ids are not authenticated (A21, A22). A client that connects under a new id each time is therefore served ahead of the others each time. The fairness properties of section 6.1 hold per robot id.
- **Wide-area networks.** All runs are on one host. Delay and jitter are injected by a link model in the simulator. Real WAN behaviour is not measured.
- **Packet loss.** Measured only if a loss run is made: it needs `tc netem` in the kernel, iproute2 in the image, and a robots container that runs as root with NET_ADMIN, which the default image and compose files do not give. Otherwise head-of-line blocking is argued, not measured.
- **Bandwidth saved by dropping frames at the source.** On a real uplink this is derived from frame size and rate, not measured. The server-advised frame rate is a planned experiment that is not run (section 12), so the bytes it would save on loopback are not measured either.
- **QUIC or datagram transport.** Not built.
- **Real cameras and real robots.** Frames are synthetic and pre-encoded. Commands drive a simulated state.
- **Model accuracy.** Commands are derived from model outputs on synthetic frames. Whether they are good commands is not evaluated.
- **Hardware dependence.** Every number belongs to the machine it was measured on. The inflection point is explained as utilization for that reason. The development machine mixes fast and slow cores and does not pin virtual CPUs, so results are repeated and reported with their spread. The I/O cost per frame that the pacing share reserves is a configured value measured on the test machine.
- **Co-located simulator.** In the `single` topology the simulator competes with the system under test, and at high load the quota freezes it with the service (section 6.4): its frames leave late, it reads its commands late, and both delays enter the freshness it reports. Each run's `simulator_valid` check fails when the run's fire lag p99 is above a tenth of a frame period, or when the simulator woke so late that it passed a tick. A run that fails the check is still counted (the fact keys of the [run-output contract](docs/run-output.md)), and `facts.json` lists the failure beside every fact the run feeds, so the 50-robot facts of `single` carry the check. In the `split` topology the simulator has its own container with no CPU limit, so the `split.massive.*` keys measure the service alone at 50 robots. Fire lag p99 at 50 robots is `{{massive.fleet.fire_lag_ms.p99}}` ms in `single` and `{{split.massive.fleet.fire_lag_ms.p99}}` ms in `split`.
- **Freezes that pacing cannot remove.** The pacing share holds over each span from one dispatch to the next, not within one quota period, because a depth batch cannot be interrupted. A period that holds a whole depth batch ends in a freeze when that batch, the classifier's CPU and the frame I/O need more than Q x P, whatever the share (section 6.4). Throttled time at 50 robots, `{{massive.cpu.throttled_ms_per_s}}` ms per second in `single` and `{{split.massive.cpu.throttled_ms_per_s}}` in `split`, counts such freezes together with any others. It is a total from the cgroup's counter, so it cannot tell them from freezes caused by a configured I/O cost (`pacing.io_cpu_ms_per_frame`) below the real one.
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

How a number reaches this document: a gated run writes `results/raw/<scenario_key>/<run_id>/summary.json`; aggregation writes `results/summary/facts.json` from the counted runs; rendering fills the placeholders of the template; the number check fails if a number here is neither a rendered fact nor a declared design constant, or if this document, the facts, the tables or the ledger are out of date. `make report` and `make numbers` check against the code digest of the working tree, so they say whether the evidence matches the current code; `make check` checks against the runs that `facts.json` recorded, so a code change alone does not fail it (the fact keys of the [run-output contract](docs/run-output.md)).

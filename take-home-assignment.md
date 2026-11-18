# Systems Engineer, ML: Take-Home Assignment

## Overview

Build a real-time inference service where multiple robots stream camera frames to a cloud, receive inference results that function as control commands, and act on them. The cloud runs real (small, CPU-only) ML models. Robots produce frames faster than models can process them. Your system decides which frames to process, keeps command freshness bound, and must behave well under load.

## Problem Statement

You are building a system with three components:

- **Robot** — A camera that streams synthetic JPEG frames (640x480, each frame carrying its frame number and capture time) at 30fps. It receives inference results as control commands and acts on them.
- **Cloud Service** — Accepts connections from multiple robots simultaneously, routes frames to models, streams results back, exposes a metrics endpoint.
- **Model Workers** — Deploy both of the following ONNX models. GPU is not needed; CPU inference is expected.

| Model | Task | Input size |
|---|---|---|
| MobileNetV3-small | Classification | 224×224 RGB |
| MiDaS-small | Depth estimation | 256×256 RGB |

- **MobileNetV3-small ONNX:** <https://huggingface.co/onnx-community/mobilenetv3_small_100.lamb_in1k/resolve/main/onnx/model.onnx>
- **MiDaS-small ONNX:** <https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx>

### The Goal

When multiple robots stream at 30fps and models process slower than that, the cloud must keep the control loop tight: freshness bounded, capacity shared fairly across robots, graceful behavior as robots connect and disconnect. Frames will be dropped. Commands will sometimes go stale. The question is how you handle those.

## Requirements

### Functional Requirements

- At least 3 concurrent robots, both models running
- Server-side request batching/scheduling Robot receives commands and rejects stale ones; stale count is reported
- Metrics endpoint sufficient to verify the demos below (you decide the fields)

### Runtime Constraints

Your entire system must run inside a single Docker container limited to 2 CPU cores and 4GB RAM. A `docker-compose.yml` or `Dockerfile` is part of your submission. We will run it with these limits applied.

### Demonstration

Run each scenario and capture evidence (logs, metrics JSON, plots, traces — your choice).

- **Freshness under steady load:** 1 robot, 30fps, 30 seconds. Show that command freshness (capture to receive) stays bounded throughout. Report p50, p95, p99, and max.
- **Late joiner:** Robot A streams alone for 10 seconds. Robot B joins. Run for 20 more. Show per-robot throughput and per-robot p95 freshness before and after B joins.
- **Scaling sweep:** Run with 1, 3, 5, and 10 concurrent robots for 30 seconds each. Plot, for each run: per-robot throughput, per-robot p50/p95 freshness, actual batch size distribution per model, CPU utilization. Identify the inflection point and explain where it comes from.
- **Disconnect:** 3 robots for 20 seconds. Kill one. Run for 20 more. Show that the remaining robots' throughput improves and the system recovers cleanly.
- **Massive connections:** Handle 50 concurrent robots at 30fps for 30 seconds without crashing or leaking memory, and without letting any single robot be starved of commands. You decide the mechanism — connection admission, per-robot quotas, adaptive drops, backpressure — and implement it. Report what you chose and why.

## Design Document

Include a design document at the root of your submission. Most of these require numbers from actually running your system.

- **Tool choice:** Language, framework, transport, batcher. What alternatives did you consider and reject? What would break about your choice at 50 robots?
- **Frame management:** How did you handle 30fps arrival vs. slower processing? What freshness threshold did the robot use, and why?
- **Batch, scheduling, and scaling:** From the scaling sweep, what was the inflection point? Why there? What in your implementation makes it move?
- **Latency breakdown** for a single frame at 5 robots: transport, JPEG decode, preprocessing, batch wait, inference, postprocessing, result delivery. Do the components sum to your observed end-to-end? Where is the biggest opportunity for optimization?

### Non-Functional Requirements

- Clean, readable code with clear separation of concerns
- Bounded resource usage (no memory leaks over 60 seconds of continuous streaming)
- Handles robot disconnect/reconnect without affecting other robots
- Testing and benchmarking

## Technical Constraints

- **Language and framework**: any of your choice

## Deliverables

### Source Code (GitHub repo)

```
repo/
├── cloud/              # cloud service
├── robot/              # robot simulator
├── workers/            # model worker(s)
├── DESIGN.md
├── README.md
├── docker-compose.yml  # or Dockerfile, with 2 CPU / 4GB limits
└── requirements.txt
```

## Evaluation Criteria

We look at system correctness across the four demos, architecture and code quality, your tool choices and the reasoning behind them, and the depth of reasoning in your design document.

—---

If anything is ambiguous, please feel free to make assumptions, just clearly state in the code what assumptions are being made.

# realtime-robot-inference

A real-time inference service for a fleet of robots. Simulated robots stream 640x480 JPEG frames at 30 fps over WebSockets to a gateway. The gateway keeps the newest frame of each robot, schedules frames onto two CPU models that run in worker processes on ONNX Runtime (MobileNetV3-small for classification, MiDaS-small for depth), and sends each result back to its robot as a command. Gateway, workers and robots run in one container limited to 2 CPUs and 4 GB.

[DESIGN.md](DESIGN.md) holds the design and the evidence: the frame-management and scheduling decisions, the capacity model, the metrics contract, the demos and their measured results, and the limitations. This README says how to run the service, the demos, the tests and the evidence pipeline.

## Requirements

- Docker with Compose (the `docker compose` command). The service, the robots and the tests run in the image.
- For the tests on the host and the evidence pipeline: [uv](https://docs.astral.sh/uv/), make and git. uv installs the Python that `pyproject.toml` pins (Python 3.14) and the locked dependencies.
- Linux or macOS. On Windows, run everything inside WSL.

## Quick start

```sh
docker compose build            # the image; downloads both models and verifies their sha256
docker compose up -d            # gateway and workers, limited to 2 CPUs and 4 GB, on port 8080
until curl -sf http://localhost:8080/healthz >/dev/null; do sleep 1; done   # wait until the service is ready
```

The service is ready a few seconds after `up`, once both workers have warmed up and calibrated their models. The wait above returns once `/healthz` answers 200. Do not wait by searching the logs for "ready:": each worker's log line also contains it, so a plain grep fires before the gateway is ready.

The gateway loads and warms both models before it accepts robots. It prints `listening on HOST:PORT` once its socket is bound, then a banner whose first line starts with `ready:` and gives the port, each model's batch cap and the limits it read from the container's cgroup. `docker compose logs system` shows it. Until then `/healthz` answers 503 and robot connections are refused.

Stream robots at it from inside the container:

```sh
docker compose exec system python -m robot.fleet --robots 3 --duration-s 30
```

Each robot sends frames at 30 fps, rejects stale or out-of-order commands, and reports its telemetry to the gateway. The fleet prints counts only: when a session ends, each robot logs, per model, how many commands it received, applied, rejected as stale and rejected as out of order. Per-robot freshness comes from a hand run with `bench.run`, described in [Running a demo by hand](#running-a-demo-by-hand), or from `/metrics`. `python -m robot.fleet --help` lists the other options (frame rate, models, stale thresholds, link delay, `--samples-dir` for per-command samples).

Read the service's state, during a run or after it:

```sh
curl -s http://localhost:8080/metrics   # one JSON object; docs/metrics.md defines every field
curl -i http://localhost:8080/healthz   # 200 while robots are accepted and both workers are ready, 503 otherwise
```

Under a histogram, the keys of `buckets` are bucket indexes, not milliseconds: each index stands for a fixed range of durations on a log scale, and the conventions of [docs/metrics.md](docs/metrics.md) give the formula that turns an index into its range.

`config.version` in `/metrics` is the commit the image was built from. It reads "unknown" unless the image is built with the `GIT_COMMIT` build argument, which the compose files take from the environment: for a hand build, run `GIT_COMMIT=$(git rev-parse HEAD) docker compose build`. The evidence runner sets it itself.

Stop it:

```sh
docker compose down             # SIGTERM: stop accepting, end every session, drain, stop the workers
```

The gateway takes its settings from the environment. `RRI_OVERRIDES` is a JSON object of settings merged over the defaults in `cloud/config.py`, and `/metrics` echoes the effective configuration under `config`. For example, the FIFO control:

```sh
RRI_OVERRIDES='{"queue_policy": "fifo", "skip_rule": false}' docker compose up -d
```

## Split topology

`docker-compose.split.yml` runs the cloud alone in the container limited to 2 CPUs and 4 GB, and the robots in a container of their own, with a 2 GB memory limit and no CPU limit. The `robots` service is in the `robots` profile, so `up` starts only the service:

```sh
docker compose -f docker-compose.split.yml up -d
until curl -sf http://localhost:8080/healthz >/dev/null; do sleep 1; done
docker compose -f docker-compose.split.yml run --rm robots python -m robot.fleet --url ws://system:8080/ws --robots 3 --duration-s 30
docker compose -f docker-compose.split.yml down
```

From the robots' container the gateway is `system:8080`, not localhost. Both containers run on one host and read its one monotonic clock, which the latency breakdown relies on.

## Running a demo by hand

The scenario driver, `bench.run`, plays one scenario against a running service, scrapes `/metrics` every second, and at the end prints each robot's results. Start the service as in the quick start, wait until `/healthz` answers 200, then:

```sh
docker compose exec system python -m bench.run --run-dir /tmp/steady --scenario steady
```

| Demo in DESIGN.md | Scenario keys |
|---|---|
| Freshness under steady load | `steady` |
| Late joiner | `late_joiner` |
| Scaling sweep | `sweep_n1`, `sweep_n3`, `sweep_n5`, `sweep_n10` |
| Disconnect | `disconnect` |
| Massive connections | `massive`, and `soak` for its long memory run |

`worker_kill` is accepted too, but a hand run only streams its robots: the evidence runner kills the depth worker.

The run directory must not hold a run already, so give each run a new one. The driver writes the run's config, the robots' plans, `events.jsonl`, `scrapes.jsonl.gz` and the per-command samples of every robot that ran to the end into it, then prints one line per robot and model:

```
robot <id> <model>: <n> commands, freshness ms p50 <ms> p95 <ms> p99 <ms> max <ms>, <k> stale
```

Freshness is the receive time of a command minus the capture time of its frame, on the robot's clock. A robot that the scenario kills writes no samples and gets no line. A hand run is ungated and never counted: its `config.json` marks it as a hand run, and no number in DESIGN.md comes from one. The directory stays in the container; `docker compose cp system:/tmp/steady ./steady` copies it out.

In the split topology the driver runs in the robots' container and reaches the gateway by name. The run directory is in that container, so give it a name and leave out `--rm`, then copy the directory out and remove the container:

```sh
docker compose -f docker-compose.split.yml run --name rri-steady robots python -m bench.run --run-dir /tmp/steady --scenario steady --url ws://system:8080/ws --metrics-url http://system:8080/metrics
docker cp rri-steady:/tmp/steady ./steady && docker rm rri-steady
```

## Tests

```sh
make check                                                 # on the host: ruff, mypy --strict, the tests not marked docker, requirements drift, the number check
docker compose run --rm system pytest -q -m "not docker"   # the suite inside the image, where the model files are
make test-docker                                           # on the host: the evidence runner end to end; builds the image and starts containers
```

Tests that need the model files skip outside the image. Tests that need the host's docker CLI are marked `docker`: `make check` leaves them out and `make test-docker` runs them.

The service's startup banner suggests `docker compose run --rm system pytest -q tests/integration`. That command runs only the integration tests in the image: the service on real sockets, with the real models. The second command above runs the whole suite in the image, the unit, property and model tests as well, so it includes the banner's tests. Neither runs the tests marked `docker`: the second command leaves them out, and the banner's command skips them because the image has no docker CLI.

## How the numbers are produced

Every measured value in DESIGN.md comes from a file under `results/` written by a gated run. The design document is rendered from `docs/DESIGN.tmpl.md`, where each measured value is a placeholder filled from `results/summary/facts.json`, and every other number is a literal declared in `docs/constants.json` with its category and source. This README is rendered from `docs/README.tmpl.md` the same way, so edit the templates, not the rendered files. To go from a number in DESIGN.md to its table: each demo's table, with the lowest and highest value of its counted runs, is in `results/tables/`, in a file named after its scenario, such as `steady.md` or `sweep.md`, and the plots are in `results/plots/`.

The pipeline runs on the host:

```sh
make evidence                  # every scenario of the evidence plan, repeated, in both topologies, each run in fresh containers; finished runs are skipped
make compare                   # the comparisons of the Tried and rejected section of DESIGN.md, each in the order A, B, B, A, on the real models
make report                    # summaries, facts.json, tables, plots, renders DESIGN.md and README.md, the number check, failed checks
make numbers                   # the number check alone, against the code as it is now
make report DIGEST=recorded    # render again from the runs facts.json recorded, after the code has changed
RRI_RESULTS=/absolute/path make evidence   # a results root elsewhere; the make targets and the compose files follow it
```

`REPEATS` sets how many times the plan runs, and `MODELS` picks the model set: `real`, or `fake` and `null`, which exercise the pipeline without the real models. A run counts only when it passes five gates: the test suite passed in the run's image with nothing skipped, the image was built in the same invocation, the container reports the 2-CPU and 4 GB limits and no swap, the run started in fresh containers, and the configuration the gateway reports is the one planned. An interrupted or failed run is kept as an attempt and never used.

The number check fails when a rendered document is out of date with its template and the facts, when the facts, tables or ledger differ from what the results produce now, or when a template holds a number that is neither a placeholder nor a declared literal. [docs/run-output.md](docs/run-output.md) describes the results layout, the scenarios, the files of a run and the fact keys. The Reproduction section of DESIGN.md lists the commands.

## Model files

The image gets the models at build time. `scripts/fetch_models.py` downloads MobileNetV3-small from Hugging Face and MiDaS-small from the MiDaS release on GitHub into `/opt/models` (the image's `MODEL_DIR`), and checks each file's sha256: a failed download or a mismatch fails the build. `scripts/patch_midas.py` then writes a copy of the MiDaS graph whose batch dimension is symbolic, so that a depth batch can hold more than one frame; the published graph takes one frame per run. The model files are not in the repository.

## Repository layout

```
cloud/     the gateway: connections, per-robot slots, schedulers, worker supervision, /metrics and /healthz
workers/   the model worker process, and each model's preprocessing, inference and command
robot/     the robot simulator and the fleet command line
common/    code the processes share: wire protocol, transport, clock, histograms, cgroup and /proc readers
bench/     the evidence: scenarios, scenario driver, evidence runner, gates, model benchmark, capacity model, summaries, aggregation, tables, plots, rendering, number check
tests/     unit, property, integration and model tests, and the fakes they use
scripts/   the build-time scripts that fetch and patch the models
docs/      the templates of DESIGN.md and README.md, and the declared constants
results/   the results root: run outputs, facts, tables, plots, the ledger
.github/   CI: make check, and the image build with the tests run inside it
```

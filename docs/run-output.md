# Run-output contract

This file is section 11 of the [design document](../DESIGN.md), kept in a file of its own for its length. Its headings keep the design document's numbering: a reference to a subsection of section 11 points into this file, a reference to section 10 points to the [metrics contract](metrics.md), and every other section reference, like "this document", means the design document.

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

`results/summary/facts.json` holds the model set, the code digest, the ids of the counted runs, and one fact per key. A fact is the median over its counted runs, with its unit, the number of runs, the lowest and highest value, the run ids, how many of them come from a dirty tree, the failed checks of each run, and the number of attempts and failed attempts behind it. Some keys are not medians. A statistic that is itself a maximum or a minimum (a key that ends in `max` or `min`, and `fairness.min_over_mean`) takes the largest or the smallest run value, so a reported extreme is the extreme of every counted run. `command_gap_bound_ms` is the bound of the run with the longest gap, so the longest gap shown beside it is set against the bound its own run checked it with. The Breakdown keys of a model all come from one run, the run whose end-to-end mean is the median, so that the stage means add up (section 7). A fact that is a string takes the latest run's value. A run with a failed check is counted, not dropped: this document reports what was measured, and `summary/failed_checks.json` lists the failures. Every key of the registry in `bench/facts.py` is present; a key with no counted run is null.

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

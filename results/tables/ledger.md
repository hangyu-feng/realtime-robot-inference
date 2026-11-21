# Tried and rejected

DESIGN.md 12: every candidate, with its outcome from the ledger. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## Candidates

| Candidate | Scenario | Arm A | Arm B | Hypothesis | Main metric | Guard metrics | Main change (B minus A) | Baseline spread | Runs | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fifo_vs_latest` | `steady`, `single` | defaults | `{"queue_policy": "fifo", "skip_rule": false}` | Under FIFO, freshness grows for the whole run; under latest-wins it is flat. | `models.depth.all.freshness_trend_ms_per_s`, lower | `models.depth.all.commands_per_s`, higher | 607 | 0.0537 | 4 | rejected_regression |
| `cls_cap_1` | `massive`, `single` | defaults | `{"models": {"cls": {"forced_batch_cap": 1}}}` | A cap of 1 lowers the classifier's freshness tail at 50 robots without lowering its throughput (H3). | `models.cls.all.freshness_ms.p95`, lower | `models.cls.all.commands_per_s_per_robot.mean`, higher | -21.5 | 0.490 | 4 | rejected_guard |
| `depth_cap_2` | `massive`, `single` | `{"models": {"depth": {"forced_batch_cap": 1}}}` | `{"models": {"depth": {"forced_batch_cap": 2}}}` | A larger depth cap spreads the dispatch cost, and lowers depth's freshness tail at 50 robots only if that cost is large (H3). | `models.depth.all.freshness_ms.p95`, lower | `models.depth.all.commands_per_s_per_robot.mean`, higher | 108 | 2.47 | 4 | rejected_regression |
| `depth_cap_4` | `massive`, `single` | `{"models": {"depth": {"forced_batch_cap": 1}}}` | `{"models": {"depth": {"forced_batch_cap": 4}}}` | As `depth_cap_2`, with a larger cap. | `models.depth.all.freshness_ms.p95`, lower | `models.depth.all.commands_per_s_per_robot.mean`, higher | 350 | 0.629 | 4 | rejected_regression |
| `ort_default_threads` | `sweep_n5`, `single` | defaults | `{"models": {"cls": {"threads": 0}, "depth": {"threads": 0}}}` | Default thread counts oversubscribe the quota and add throttling. | `cpu.all.throttled_ms_per_s`, lower | `models.cls.all.freshness_ms.p99`, lower; `models.cls.all.commands_per_s_per_robot.mean`, higher | 765 | 1.20 | 4 | rejected_guard |
| `depth_unpaced` | `massive`, `single` | defaults | `{"models": {"depth": {"paced": false}}}` | Without pacing, two saturated inference threads plus the frame I/O exceed the quota at 50 robots, and the kernel freezes the whole container for part of each period (H5). | `cpu.all.throttled_ms_per_s`, lower | `models.cls.all.freshness_ms.p99`, lower; `validity.fire_lag_ms.p99`, lower; `models.depth.all.commands_per_s_per_robot.mean`, higher | 426 | 25.1 | 4 | rejected_guard |

Counted runs: 24.

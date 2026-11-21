# Fault runs

DESIGN.md 11.2, `single` topology: a robot killed in `disconnect` and the `depth` worker killed in `worker_kill`, each with the checks of its runs and the time the service takes to recover. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## Cases

| Case | Scenario | Checks | Recovery |
| --- | --- | --- | --- |
| A robot's process killed with SIGKILL | `disconnect` | `closed_abnormal`, `killed_robot_released` | session closed after 2.45 (2.45 to 2.62) ms; survivors' `depth` rate risen after 1 (1 to 1) s |
| The `depth` worker killed with SIGKILL | `worker_kill` | `worker_restarted`; `cls` commands/s per robot before / after: 30.0 (30.0 to 30.0) / 30.0 (30.0 to 30.0) | `depth` commands again after 1611 (1602 to 1733) ms |

Counted runs: 3 of `disconnect`, 3 of `worker_kill`.
Failed checks: none.

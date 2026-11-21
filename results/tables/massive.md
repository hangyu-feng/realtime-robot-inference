# Massive connections

DESIGN.md 9.5, `single` topology. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## Per model

| Quantity | `cls` | `depth` |
| --- | --- | --- |
| Commands/s per robot, min | 6.27 (6.27 to 6.40) | 0.100 (0.100 to 0.100) |
| Commands/s per robot, mean | 6.34 (6.30 to 6.40) | 0.116 (0.116 to 0.117) |
| Commands/s per robot, max | 6.43 (6.30 to 6.43) | 0.133 (0.133 to 0.133) |
| Commands received | 9506 (9449 to 9601) | 174 (174 to 176) |
| Lowest over mean rate | 0.995 (0.995 to 1.000) | 0.852 (0.852 to 0.862) |
| Jain index | 1.000 (1.000 to 1.000) | 0.980 (0.980 to 0.980) |
| Longest command gap (ms) | 251 (232 to 251) | 8748 (8566 to 8748) |
| Command gap bound (ms) | 275 (275 to 294) | 8951 (8768 to 8951) |
| Freshness p50 (ms) | 43.6 (43.3 to 43.7) | 107 (106 to 107) |
| Freshness p95 (ms) | 59.0 (58.4 to 59.1) | 124 (122 to 127) |
| Freshness p99 (ms) | 64.5 (62.7 to 65.0) | 133 (128 to 134) |
| Freshness max (ms) | 81.1 (79.7 to 81.1) | 140 (134 to 140) |
| Stale commands | 0 (0 to 0) | 0 (0 to 0) |
| Stale ratio | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |
| Out-of-order commands | 0 (0 to 0) | 0 (0 to 0) |
| Watchdog coverage, min | 0.992 (0.992 to 0.992) | 0.018 (0.018 to 0.018) |
| Watchdog coverage, mean | 0.995 (0.995 to 0.995) | 0.022 (0.022 to 0.023) |

Counted runs: 3.

## Predicted

| Quantity | `cls` | `depth` |
| --- | --- | --- |
| Commands/s per robot, mean | 6.72 | 0.123 |

Counted runs: 0.
Counted runs behind the predictions: 3.

## Connections, CPU and memory

| Quantity | Value |
| --- | --- |
| Connections accepted | 50 (50 to 50) |
| Connections refused | 0 (0 to 0) |
| CPU utilization | 0.911 (0.907 to 0.917) |
| Throttled (ms per s) | 41.3 (34.7 to 42.3) |
| Gateway CPU (cores) | 0.173 (0.165 to 0.179) |
| Gateway CPU per frame (ms) | 0.116 (0.110 to 0.120) |
| Gateway event-loop lag p99 (ms) | 2.15 (1.97 to 5.12) |
| Bytes received per second | 61029438 (61028106 to 61073451) |
| Stage `transport_up` p50 (ms) | 0.186 (0.180 to 0.190) |
| Stage `transport_up` p99 (ms) | 1.24 (0.988 to 1.25) |
| Stage `dispatch_rt` p50 (ms) | 0.444 (0.434 to 0.448) |
| Simulator CPU (cores) | 0.168 (0.165 to 0.176) |
| Simulator fire lag p99 (ms) | 2.25 (2.01 to 4.06) |
| Memory, anonymous + socket, after the connections settle (MB) | 524 (523 to 525) |
| Memory, anonymous + socket, at the end (MB) | 524 (524 to 525) |
| Socket memory at the end (MB) | 0.0352 (0.0234 to 0.0469) |

Counted runs: 3.

## Within the watchdog limits, from the sweep and the 50-robot run

| Quantity | `cls` | `depth` |
| --- | --- | --- |
| Robots served within the watchdog limit, of the counts measured | 50 | 1 |

Counted runs: 15.

## Soak, 5 minutes

| Quantity | Value |
| --- | --- |
| Memory slope (MB per minute) | 0.160 (0.141 to 0.198) |
| Change in open file descriptors | 0 (0 to 0) |
| Change in event-loop tasks | 0 (0 to 0) |

Counted runs: 3.

## Soak, per model

| Quantity | `cls` | `depth` |
| --- | --- | --- |
| Commands/s per robot, min | 6.19 (6.19 to 6.36) | 0.113 (0.113 to 0.117) |
| Commands/s per robot, mean | 6.26 (6.20 to 6.36) | 0.116 (0.116 to 0.118) |
| Commands/s per robot, max | 6.37 (6.20 to 6.37) | 0.120 (0.117 to 0.120) |
| Lowest over mean rate | 1.000 (1.000 to 1.000) | 0.975 (0.975 to 0.992) |
| Jain index | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) |

Counted runs: 3.

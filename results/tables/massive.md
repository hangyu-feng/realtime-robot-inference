# Massive connections

DESIGN.md 9.5, `single` topology. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## Per model

| Quantity | `cls` | `depth` |
| --- | --- | --- |
| Commands/s per robot, min | 4.90 (4.90 to 5.90) | 0.0667 (0.0667 to 0.100) |
| Commands/s per robot, mean | 5.66 (4.91 to 5.92) | 0.107 (0.0960 to 0.111) |
| Commands/s per robot, max | 5.93 (4.93 to 5.93) | 0.133 (0.100 to 0.133) |
| Commands received | 8484 (7358 to 8883) | 161 (144 to 166) |
| Lowest over mean rate | 0.996 (0.996 to 0.999) | 0.694 (0.694 to 0.932) |
| Jain index | 1.000 (1.000 to 1.000) | 0.984 (0.981 to 0.987) |
| Longest command gap (ms) | 310 (262 to 310) | 10686 (9396 to 10686) |
| Command gap bound (ms) | 307 (307 to 380) | 9599 (9599 to 10914) |
| Freshness p50 (ms) | 46.4 (45.4 to 51.1) | 118 (112 to 132) |
| Freshness p95 (ms) | 62.6 (60.9 to 69.2) | 138 (136 to 156) |
| Freshness p99 (ms) | 72.0 (66.6 to 79.1) | 146 (145 to 171) |
| Freshness max (ms) | 102 (93.1 to 102) | 191 (157 to 191) |
| Stale commands | 0 (0 to 0) | 1 (1 to 15) |
| Stale ratio | 0.000 (0.000 to 0.000) | 0.006 (0.006 to 0.104) |
| Out-of-order commands | 0 (0 to 0) | 0 (0 to 0) |
| Watchdog coverage, min | 0.991 (0.991 to 0.993) | 0.006 (0.006 to 0.013) |
| Watchdog coverage, mean | 0.994 (0.994 to 0.995) | 0.019 (0.015 to 0.021) |

Counted runs: 3.

## Predicted

| Quantity | `cls` | `depth` |
| --- | --- | --- |
| Commands/s per robot, mean | 6.04 | 0.113 |

Counted runs: 0.
Counted runs behind the predictions: 3.

## Connections, CPU and memory

| Quantity | Value |
| --- | --- |
| Connections accepted | 50 (50 to 50) |
| Connections refused | 0 (0 to 0) |
| CPU utilization | 0.921 (0.920 to 0.943) |
| Throttled (ms per s) | 85.8 (60.0 to 155) |
| Gateway CPU (cores) | 0.185 (0.180 to 0.216) |
| Gateway CPU per frame (ms) | 0.124 (0.120 to 0.144) |
| Gateway event-loop lag p99 (ms) | 1.97 (1.97 to 11.2) |
| Bytes received per second | 61034510 (61009079 to 61054491) |
| Stage `transport_up` p50 (ms) | 0.202 (0.192 to 0.230) |
| Stage `transport_up` p99 (ms) | 1.20 (0.997 to 1.72) |
| Stage `dispatch_rt` p50 (ms) | 0.486 (0.471 to 0.506) |
| Simulator CPU (cores) | 0.189 (0.186 to 0.220) |
| Simulator fire lag p99 (ms) | 8.56 (4.68 to 12.3) |
| Memory, anonymous + socket, after the connections settle (MB) | 525 (523 to 526) |
| Memory, anonymous + socket, at the end (MB) | 525 (523 to 527) |
| Socket memory at the end (MB) | 0.0195 (0.0117 to 0.109) |

Counted runs: 3.

## Within the watchdog limits, from the sweep and the 50-robot run

| Quantity | `cls` | `depth` |
| --- | --- | --- |
| Robots served within the watchdog limit, of the counts measured | 50 | 1 |

Counted runs: 15.

## Soak, 5 minutes

| Quantity | Value |
| --- | --- |
| Memory slope (MB per minute) | 0.240 (0.204 to 0.465) |
| Change in open file descriptors | 0 (0 to 0) |
| Change in event-loop tasks | 0 (0 to 0) |

Counted runs: 3.

## Soak, per model

| Quantity | `cls` | `depth` |
| --- | --- | --- |
| Commands/s per robot, min | 5.48 (5.48 to 5.61) | 0.103 (0.103 to 0.107) |
| Commands/s per robot, mean | 5.52 (5.48 to 5.61) | 0.105 (0.105 to 0.107) |
| Commands/s per robot, max | 5.62 (5.48 to 5.62) | 0.110 (0.107 to 0.110) |
| Lowest over mean rate | 0.999 (0.999 to 1.000) | 0.984 (0.984 to 0.993) |
| Jain index | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) |

Counted runs: 3.

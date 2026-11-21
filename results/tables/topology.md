# Topology comparison

DESIGN.md 9.6 for every serving scenario, over the whole run unless the row names another span. A cell shows the median over the counted runs, with the lowest and highest run value in parentheses.

## `steady`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 11.9 (11.9 to 12.9) | 12.1 (11.9 to 12.7) |
| `cls` freshness p95 (ms) | 6.55 (5.88 to 6.80) | 6.70 (6.18 to 6.77) |
| `depth` freshness p95 (ms) | 118 (111 to 118) | 116 (112 to 117) |
| Simulator CPU (cores) | 0.0169 (0.0152 to 0.0172) | 0.0172 (0.0155 to 0.0183) |
| Simulator fire lag p99 (ms) | 1.85 (1.80 to 1.88) | 1.81 (1.78 to 1.81) |
| Throttled (ms per s) | 0 (0 to 0) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `late_joiner`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 8.07 (7.93 to 8.60) | 8.00 (7.97 to 8.57) |
| `cls` freshness p95 (ms) | 6.60 (5.91 to 7.23) | 6.74 (6.03 to 7.34) |
| `depth` freshness p95 (ms) | 117 (112 to 118) | 116 (112 to 116) |
| Simulator CPU (cores) | 0.0238 (0.0228 to 0.0259) | 0.0252 (0.0248 to 0.0276) |
| Simulator fire lag p99 (ms) | 1.84 (1.80 to 1.94) | 1.81 (1.81 to 1.88) |
| Throttled (ms per s) | 0 (0 to 0) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `sweep_n1`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 11.9 (11.4 to 12.9) | 11.9 (11.9 to 12.9) |
| `cls` freshness p95 (ms) | 6.54 (5.95 to 8.64) | 6.51 (6.00 to 6.81) |
| `depth` freshness p95 (ms) | 117 (110 to 124) | 118 (111 to 118) |
| Simulator CPU (cores) | 0.0159 (0.0152 to 0.0166) | 0.0172 (0.0155 to 0.0176) |
| Simulator fire lag p99 (ms) | 1.80 (1.79 to 3.78) | 1.88 (1.88 to 1.92) |
| Throttled (ms per s) | 0 (0 to 0) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `sweep_n3`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 3.99 (3.94 to 4.22) | 3.94 (3.92 to 4.21) |
| `cls` freshness p95 (ms) | 8.70 (6.08 to 11.1) | 8.93 (6.07 to 11.0) |
| `depth` freshness p95 (ms) | 117 (112 to 117) | 117 (112 to 118) |
| Simulator CPU (cores) | 0.0393 (0.0324 to 0.0407) | 0.0421 (0.0345 to 0.0424) |
| Simulator fire lag p99 (ms) | 1.83 (1.81 to 1.85) | 1.86 (1.82 to 1.93) |
| Throttled (ms per s) | 0 (0 to 0) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `sweep_n5`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 2.37 (2.33 to 2.50) | 2.33 (2.31 to 2.48) |
| `cls` freshness p95 (ms) | 10.7 (8.51 to 16.3) | 10.7 (8.52 to 16.2) |
| `depth` freshness p95 (ms) | 117 (114 to 120) | 119 (114 to 120) |
| Simulator CPU (cores) | 0.0500 (0.0472 to 0.0603) | 0.0517 (0.0493 to 0.0614) |
| Simulator fire lag p99 (ms) | 1.85 (1.85 to 1.86) | 1.77 (1.76 to 1.83) |
| Throttled (ms per s) | 0 (0 to 0) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `sweep_n10`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (29.9 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 0.993 (0.937 to 1.11) | 0.960 (0.947 to 1.01) |
| `cls` freshness p95 (ms) | 32.0 (29.6 to 40.1) | 25.3 (17.4 to 31.4) |
| `depth` freshness p95 (ms) | 121 (120 to 122) | 119 (113 to 120) |
| Simulator CPU (cores) | 0.0772 (0.0728 to 0.0803) | 0.0821 (0.0755 to 0.0879) |
| Simulator fire lag p99 (ms) | 1.84 (1.84 to 2.01) | 1.80 (1.80 to 1.82) |
| Throttled (ms per s) | 15.3 (6.03 to 97.5) | 0 (0 to 0.0825) |

Counted runs: 3 in `single`, 3 in `split`.

## `disconnect`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 4.91 (4.88 to 5.33) | 4.99 (4.91 to 5.91) |
| `cls` freshness p95 (ms) | 6.55 (6.08 to 7.67) | 6.61 (5.52 to 7.51) |
| `depth` freshness p95 (ms) | 118 (112 to 118) | 117 (104 to 119) |
| Simulator CPU before the kill (cores) | 0.0421 (0.0405 to 0.0442) | 0.0453 (0.0368 to 0.0453) |
| Simulator fire lag p99 (ms) | 1.87 (1.80 to 1.89) | 1.86 (1.84 to 1.86) |
| Throttled (ms per s) | 0 (0 to 0) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `worker_kill`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 5.79 (5.75 to 6.09) | 5.71 (5.68 to 5.74) |
| `cls` freshness p95 (ms) | 6.56 (6.18 to 7.31) | 6.73 (6.62 to 7.32) |
| `depth` freshness p95 (ms) | 117 (115 to 118) | 118 (117 to 119) |
| Simulator CPU (cores) | 0.0277 (0.0274 to 0.0303) | 0.0313 (0.0282 to 0.0331) |
| Simulator fire lag p99 (ms) | 1.87 (1.83 to 1.97) | 1.86 (1.85 to 1.89) |
| Throttled (ms per s) | 0 (0 to 0) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `massive`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 6.34 (6.30 to 6.40) | 6.40 (6.26 to 6.92) |
| `depth` commands/s per robot | 0.116 (0.116 to 0.117) | 0.119 (0.117 to 0.122) |
| `cls` freshness p95 (ms) | 59.0 (58.4 to 59.1) | 58.6 (56.4 to 59.0) |
| `depth` freshness p95 (ms) | 124 (122 to 127) | 120 (120 to 123) |
| Simulator CPU (cores) | 0.168 (0.165 to 0.176) | 0.173 (0.166 to 0.177) |
| Simulator fire lag p99 (ms) | 2.25 (2.01 to 4.06) | 1.83 (1.83 to 1.84) |
| Throttled (ms per s) | 41.3 (34.7 to 42.3) | 1.81 (1.54 to 2.01) |

Counted runs: 3 in `single`, 3 in `split`.

## `soak`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 6.26 (6.20 to 6.36) | 6.29 (6.21 to 6.50) |
| `depth` commands/s per robot | 0.116 (0.116 to 0.118) | 0.117 (0.115 to 0.119) |
| `cls` freshness p95 (ms) | 59.6 (58.8 to 59.8) | 58.9 (58.0 to 59.6) |
| `depth` freshness p95 (ms) | 129 (125 to 130) | 121 (120 to 125) |
| Simulator CPU (cores) | 0.173 (0.167 to 0.174) | 0.178 (0.178 to 0.189) |
| Simulator fire lag p99 (ms) | 5.64 (5.53 to 5.93) | 1.83 (1.83 to 1.84) |
| Throttled (ms per s) | 58.3 (53.3 to 62.3) | 4.64 (3.80 to 6.54) |

Counted runs: 3 in `single`, 3 in `split`.

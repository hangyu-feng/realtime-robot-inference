# Topology comparison

DESIGN.md 9.6 for every serving scenario, over the whole run unless the row names another span. A cell shows the median over the counted runs, with the lowest and highest run value in parentheses.

## `steady`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 10.5 (9.53 to 11.5) | 10.4 (9.67 to 11.6) |
| `cls` freshness p95 (ms) | 7.25 (6.40 to 8.04) | 7.45 (6.41 to 8.04) |
| `depth` freshness p95 (ms) | 129 (121 to 138) | 131 (119 to 138) |
| Simulator CPU (cores) | 0.0190 (0.0176 to 0.0203) | 0.0200 (0.0186 to 0.0214) |
| Simulator fire lag p99 (ms) | 1.89 (1.81 to 1.99) | 1.94 (1.78 to 1.94) |
| Throttled (ms per s) | 0 (0 to 0) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `late_joiner`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 7.17 (6.37 to 7.53) | 6.93 (6.57 to 7.60) |
| `cls` freshness p95 (ms) | 7.54 (7.42 to 8.03) | 7.82 (7.70 to 7.96) |
| `depth` freshness p95 (ms) | 128 (123 to 140) | 133 (122 to 136) |
| Simulator CPU (cores) | 0.0300 (0.0262 to 0.0324) | 0.0328 (0.0269 to 0.0331) |
| Simulator fire lag p99 (ms) | 1.96 (1.82 to 2.05) | 1.91 (1.85 to 2.09) |
| Throttled (ms per s) | 0 (0 to 0) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `sweep_n1`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 10.2 (10.1 to 11.3) | 10.5 (10.4 to 11.4) |
| `cls` freshness p95 (ms) | 7.76 (6.91 to 7.94) | 7.54 (6.77 to 7.62) |
| `depth` freshness p95 (ms) | 133 (123 to 134) | 128 (121 to 132) |
| Simulator CPU (cores) | 0.0197 (0.0183 to 0.0200) | 0.0200 (0.0193 to 0.0214) |
| Simulator fire lag p99 (ms) | 1.95 (1.85 to 2.04) | 1.95 (1.95 to 1.96) |
| Throttled (ms per s) | 0 (0 to 0) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `sweep_n3`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 3.53 (3.48 to 3.76) | 3.60 (3.22 to 3.69) |
| `cls` freshness p95 (ms) | 10.3 (7.38 to 11.1) | 9.73 (8.26 to 11.6) |
| `depth` freshness p95 (ms) | 127 (123 to 129) | 127 (124 to 141) |
| Simulator CPU (cores) | 0.0486 (0.0348 to 0.0514) | 0.0486 (0.0372 to 0.0566) |
| Simulator fire lag p99 (ms) | 1.91 (1.87 to 1.93) | 1.89 (1.86 to 1.95) |
| Throttled (ms per s) | 0 (0 to 0) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `sweep_n5`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 2.05 (1.85 to 2.23) | 2.09 (1.98 to 2.21) |
| `cls` freshness p95 (ms) | 16.8 (11.2 to 17.4) | 14.8 (11.3 to 16.8) |
| `depth` freshness p95 (ms) | 133 (122 to 150) | 131 (123 to 137) |
| Simulator CPU (cores) | 0.0738 (0.0503 to 0.0762) | 0.0748 (0.0524 to 0.0762) |
| Simulator fire lag p99 (ms) | 1.90 (1.85 to 1.95) | 1.89 (1.83 to 1.93) |
| Throttled (ms per s) | 0.138 (0 to 6.25) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `sweep_n10`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 28.5 (28.3 to 29.6) | 28.4 (27.5 to 29.6) |
| `depth` commands/s per robot | 0.867 (0.860 to 0.893) | 0.890 (0.830 to 0.900) |
| `cls` freshness p95 (ms) | 57.4 (50.7 to 59.8) | 59.4 (48.8 to 60.2) |
| `depth` freshness p95 (ms) | 129 (125 to 130) | 132 (126 to 133) |
| Simulator CPU (cores) | 0.0876 (0.0776 to 0.0914) | 0.0928 (0.0821 to 0.0993) |
| Simulator fire lag p99 (ms) | 1.91 (1.84 to 2.16) | 1.87 (1.85 to 1.88) |
| Throttled (ms per s) | 12.3 (5.55 to 22.0) | 1.38 (0.0785 to 2.91) |

Counted runs: 3 in `single`, 3 in `split`.

## `disconnect`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 4.53 (4.46 to 4.73) | 4.54 (4.35 to 4.68) |
| `cls` freshness p95 (ms) | 7.53 (7.30 to 7.67) | 7.68 (7.43 to 8.68) |
| `depth` freshness p95 (ms) | 127 (122 to 129) | 124 (122 to 131) |
| Simulator CPU before the kill (cores) | 0.0463 (0.0463 to 0.0484) | 0.0495 (0.0484 to 0.0574) |
| Simulator fire lag p99 (ms) | 1.85 (1.84 to 1.94) | 1.96 (1.85 to 2.01) |
| Throttled (ms per s) | 0 (0 to 0) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `worker_kill`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| `depth` commands/s per robot | 5.06 (4.59 to 5.14) | 5.11 (4.49 to 5.40) |
| `cls` freshness p95 (ms) | 8.04 (7.53 to 9.11) | 7.67 (7.60 to 8.53) |
| `depth` freshness p95 (ms) | 132 (128 to 142) | 128 (122 to 141) |
| Simulator CPU (cores) | 0.0338 (0.0336 to 0.0379) | 0.0362 (0.0313 to 0.0418) |
| Simulator fire lag p99 (ms) | 1.91 (1.80 to 1.93) | 1.96 (1.84 to 2.04) |
| Throttled (ms per s) | 0 (0 to 0) | 0 (0 to 0) |

Counted runs: 3 in `single`, 3 in `split`.

## `massive`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 5.66 (4.91 to 5.92) | 5.77 (5.45 to 6.02) |
| `depth` commands/s per robot | 0.107 (0.0960 to 0.111) | 0.107 (0.101 to 0.111) |
| `cls` freshness p95 (ms) | 62.6 (60.9 to 69.2) | 61.2 (60.0 to 66.2) |
| `depth` freshness p95 (ms) | 138 (136 to 156) | 129 (127 to 148) |
| Simulator CPU (cores) | 0.189 (0.186 to 0.220) | 0.209 (0.201 to 0.248) |
| Simulator fire lag p99 (ms) | 8.56 (4.68 to 12.3) | 1.90 (1.84 to 2.31) |
| Throttled (ms per s) | 85.8 (60.0 to 155) | 11.5 (7.23 to 25.5) |

Counted runs: 3 in `single`, 3 in `split`.

## `soak`

| Quantity | `single` | `split` |
| --- | --- | --- |
| `cls` commands/s per robot | 5.52 (5.48 to 5.61) | 5.65 (5.57 to 6.02) |
| `depth` commands/s per robot | 0.105 (0.105 to 0.107) | 0.105 (0.104 to 0.110) |
| `cls` freshness p95 (ms) | 64.0 (63.5 to 64.2) | 62.4 (60.2 to 62.9) |
| `depth` freshness p95 (ms) | 143 (140 to 144) | 136 (128 to 138) |
| Simulator CPU (cores) | 0.201 (0.199 to 0.202) | 0.222 (0.201 to 0.228) |
| Simulator fire lag p99 (ms) | 10.8 (9.39 to 13.6) | 1.87 (1.85 to 1.95) |
| Throttled (ms per s) | 105 (96.6 to 112) | 14.2 (9.03 to 17.6) |

Counted runs: 3 in `single`, 3 in `split`.

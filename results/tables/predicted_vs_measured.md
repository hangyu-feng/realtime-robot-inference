# Predicted against measured

Every prediction of the capacity model beside its measurement. The gap is the measured value minus the prediction and the spread is the highest minus the lowest run value; a gap larger than the spread is marked, and DESIGN.md explains it in its text. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## Every prediction

| Key | Predicted | Measured | Gap | Spread | Runs | To explain |
| --- | --- | --- | --- | --- | --- | --- |
| `disconnect.after.survivors.depth.commands_per_s_per_robot` | 5.50 | 5.50 (5.18 to 5.65) | 0 | 0.475 | 3 |  |
| `disconnect.before.survivors.depth.commands_per_s_per_robot` | 3.70 | 3.75 (3.55 to 3.80) | 0.0500 | 0.250 | 3 |  |
| `late_joiner.after.a.cls.commands_per_s` | 30.0 | 30.0 (30.0 to 30.0) | 0 | 0 | 3 |  |
| `late_joiner.after.a.depth.commands_per_s` | 5.55 | 5.40 (5.00 to 5.70) | -0.150 | 0.700 | 3 |  |
| `massive.cls.commands_per_s_per_robot.mean` | 6.04 | 5.66 (4.91 to 5.92) | -0.381 | 1.02 | 3 |  |
| `massive.cls.freshness_ms.p50` | 43.8 | 46.4 (45.4 to 51.1) | 2.62 | 5.71 | 3 |  |
| `massive.cls.freshness_ms.p95` | 58.7 | 62.6 (60.9 to 69.2) | 3.86 | 8.25 | 3 |  |
| `massive.depth.command_gap_bound_ms` | 10034 | 10914 (9599 to 10914) | 880 | 1315 | 3 |  |
| `massive.depth.commands_per_s_per_robot.mean` | 0.113 | 0.107 (0.0960 to 0.111) | -0.00600 | 0.0147 | 3 |  |
| `massive.depth.freshness_ms.p50` | 107 | 118 (112 to 132) | 11.2 | 20.4 | 3 |  |
| `massive.depth.freshness_ms.p95` | 122 | 138 (136 to 156) | 15.5 | 19.4 | 3 |  |
| `steady.cls.command_age_peak_ms.p95` | 38.1 | 40.6 (39.7 to 41.4) | 2.50 | 1.63 | 3 | yes |
| `steady.cls.commands_per_s` | 30.0 | 30.0 (30.0 to 30.0) | -0.0333 | 0 | 3 | yes |
| `steady.cls.freshness_ms.max` | 4.74 | 11.0 (8.07 to 11.0) | 6.27 | 2.94 | 3 | yes |
| `steady.cls.freshness_ms.p50` | 4.74 | 6.11 (5.52 to 6.56) | 1.36 | 1.04 | 3 | yes |
| `steady.cls.freshness_ms.p95` | 4.74 | 7.25 (6.40 to 8.04) | 2.50 | 1.64 | 3 | yes |
| `steady.cls.freshness_ms.p99` | 4.74 | 7.81 (7.18 to 9.30) | 3.07 | 2.11 | 3 | yes |
| `steady.depth.command_age_peak_ms.p95` | 213 | 224 (211 to 247) | 10.9 | 36.2 | 3 |  |
| `steady.depth.commands_per_s` | 11.1 | 10.5 (9.53 to 11.5) | -0.533 | 2.00 | 3 |  |
| `steady.depth.freshness_ms.max` | 124 | 188 (130 to 188) | 64.0 | 58.2 | 3 | yes |
| `steady.depth.freshness_ms.p50` | 108 | 112 (104 to 120) | 4.36 | 16.4 | 3 |  |
| `steady.depth.freshness_ms.p95` | 123 | 129 (121 to 138) | 6.20 | 17.6 | 3 |  |
| `steady.depth.freshness_ms.p99` | 124 | 133 (126 to 143) | 9.48 | 16.3 | 3 |  |
| `sweep.n1.cls.batch_size.mean` | 1.00 | 1.00 (1.00 to 1.00) | 0 | 0 | 3 |  |
| `sweep.n1.cls.commands_per_s_per_robot.mean` | 30.0 | 30.0 (30.0 to 30.0) | -0.0333 | 0 | 3 | yes |
| `sweep.n1.cls.freshness_ms.p50` | 4.74 | 6.31 (5.77 to 6.47) | 1.56 | 0.700 | 3 | yes |
| `sweep.n1.cls.freshness_ms.p95` | 4.74 | 7.76 (6.91 to 7.94) | 3.02 | 1.03 | 3 | yes |
| `sweep.n1.depth.batch_size.mean` | 1.00 | 1.00 (1.00 to 1.00) | 0 | 0 | 3 |  |
| `sweep.n1.depth.commands_per_s_per_robot.mean` | 11.1 | 10.2 (10.1 to 11.3) | -0.867 | 1.17 | 3 |  |
| `sweep.n1.depth.freshness_ms.p50` | 108 | 114 (106 to 116) | 6.95 | 9.71 | 3 |  |
| `sweep.n1.depth.freshness_ms.p95` | 123 | 133 (123 to 134) | 10.3 | 11.4 | 3 |  |
| `sweep.n10.cls.batch_size.mean` | 6.16 | 7.26 (5.51 to 7.82) | 1.10 | 2.32 | 3 |  |
| `sweep.n10.cls.commands_per_s_per_robot.mean` | 30.0 | 28.5 (28.3 to 29.6) | -1.44 | 1.26 | 3 | yes |
| `sweep.n10.cls.freshness_ms.p50` | 31.7 | 40.9 (32.3 to 45.6) | 9.26 | 13.3 | 3 |  |
| `sweep.n10.cls.freshness_ms.p95` | 43.0 | 57.4 (50.7 to 59.8) | 14.4 | 9.09 | 3 | yes |
| `sweep.n10.depth.batch_size.mean` | 1.00 | 1.00 (1.00 to 1.00) | 0 | 0 | 3 |  |
| `sweep.n10.depth.commands_per_s_per_robot.mean` | 0.913 | 0.867 (0.860 to 0.893) | -0.0467 | 0.0333 | 3 | yes |
| `sweep.n10.depth.freshness_ms.p50` | 107 | 114 (110 to 115) | 6.48 | 4.66 | 3 | yes |
| `sweep.n10.depth.freshness_ms.p95` | 123 | 129 (125 to 130) | 6.05 | 4.83 | 3 | yes |
| `sweep.n3.cls.batch_size.mean` | 1.00 | 1.00 (1.00 to 1.47) | 0.000383 | 0.474 | 3 |  |
| `sweep.n3.cls.commands_per_s_per_robot.mean` | 30.0 | 30.0 (30.0 to 30.0) | -0.0111 | 0.0222 | 3 |  |
| `sweep.n3.cls.freshness_ms.p50` | 4.74 | 6.70 (6.11 to 9.20) | 1.96 | 3.09 | 3 |  |
| `sweep.n3.cls.freshness_ms.p95` | 5.95 | 10.3 (7.38 to 11.1) | 4.31 | 3.72 | 3 | yes |
| `sweep.n3.depth.batch_size.mean` | 1.00 | 1.00 (1.00 to 1.00) | 0 | 0 | 3 |  |
| `sweep.n3.depth.commands_per_s_per_robot.mean` | 3.69 | 3.53 (3.48 to 3.76) | -0.156 | 0.278 | 3 |  |
| `sweep.n3.depth.freshness_ms.p50` | 107 | 112 (107 to 115) | 4.45 | 7.84 | 3 |  |
| `sweep.n3.depth.freshness_ms.p95` | 123 | 127 (123 to 129) | 3.63 | 5.77 | 3 |  |
| `sweep.n5.cls.batch_size.mean` | 1.00 | 1.26 (1.03 to 1.67) | 0.261 | 0.641 | 3 |  |
| `sweep.n5.cls.commands_per_s_per_robot.mean` | 30.0 | 30.0 (30.0 to 30.0) | -0.00667 | 0.0133 | 3 |  |
| `sweep.n5.cls.freshness_ms.p50` | 5.95 | 9.62 (7.24 to 13.0) | 3.67 | 5.72 | 3 |  |
| `sweep.n5.cls.freshness_ms.p95` | 8.52 | 16.8 (11.2 to 17.4) | 8.24 | 6.23 | 3 | yes |
| `sweep.n5.depth.batch_size.mean` | 1.00 | 1.00 (1.00 to 1.00) | 0 | 0 | 3 |  |
| `sweep.n5.depth.commands_per_s_per_robot.mean` | 2.21 | 2.05 (1.85 to 2.23) | -0.160 | 0.387 | 3 |  |
| `sweep.n5.depth.freshness_ms.p50` | 108 | 115 (107 to 126) | 7.76 | 18.7 | 3 |  |
| `sweep.n5.depth.freshness_ms.p95` | 123 | 133 (122 to 150) | 10.9 | 27.4 | 3 |  |

Counted runs: 24.
Counted runs behind the predictions: 3.

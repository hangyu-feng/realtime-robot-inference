# Predicted against measured

Every prediction of the capacity model beside its measurement. The gap is the measured value minus the prediction and the spread is the highest minus the lowest run value; a gap larger than the spread is marked, and DESIGN.md explains it in its text. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## Every prediction

| Key | Predicted | Measured | Gap | Spread | Runs | To explain |
| --- | --- | --- | --- | --- | --- | --- |
| `disconnect.after.survivors.depth.commands_per_s_per_robot` | 6.05 | 5.93 (5.85 to 6.48) | -0.125 | 0.625 | 3 |  |
| `disconnect.before.survivors.depth.commands_per_s_per_robot` | 4.03 | 3.90 (3.90 to 4.18) | -0.125 | 0.275 | 3 |  |
| `late_joiner.after.a.cls.commands_per_s` | 30.0 | 30.0 (30.0 to 30.0) | 0 | 0 | 3 |  |
| `late_joiner.after.a.depth.commands_per_s` | 6.05 | 6.05 (5.95 to 6.45) | 0 | 0.500 | 3 |  |
| `massive.cls.commands_per_s_per_robot.mean` | 6.72 | 6.34 (6.30 to 6.40) | -0.382 | 0.101 | 3 | yes |
| `massive.cls.freshness_ms.p50` | 40.9 | 43.6 (43.3 to 43.7) | 2.67 | 0.323 | 3 | yes |
| `massive.cls.freshness_ms.p95` | 55.9 | 59.0 (58.4 to 59.1) | 3.08 | 0.744 | 3 | yes |
| `massive.depth.command_gap_bound_ms` | 9285 | 8951 (8768 to 8951) | -334 | 182 | 3 | yes |
| `massive.depth.commands_per_s_per_robot.mean` | 0.123 | 0.116 (0.116 to 0.117) | -0.00733 | 0.00133 | 3 | yes |
| `massive.depth.freshness_ms.p50` | 100 | 107 (106 to 107) | 6.22 | 0.966 | 3 | yes |
| `massive.depth.freshness_ms.p95` | 115 | 124 (122 to 127) | 9.64 | 4.41 | 3 | yes |
| `steady.cls.command_age_peak_ms.p95` | 37.4 | 39.9 (39.2 to 40.1) | 2.47 | 0.930 | 3 | yes |
| `steady.cls.commands_per_s` | 30.0 | 30.0 (30.0 to 30.0) | -0.0333 | 0 | 3 | yes |
| `steady.cls.freshness_ms.max` | 4.08 | 8.16 (7.83 to 8.16) | 4.08 | 0.332 | 3 | yes |
| `steady.cls.freshness_ms.p50` | 4.08 | 5.50 (4.92 to 5.65) | 1.43 | 0.730 | 3 | yes |
| `steady.cls.freshness_ms.p95` | 4.08 | 6.55 (5.88 to 6.80) | 2.47 | 0.927 | 3 | yes |
| `steady.cls.freshness_ms.p99` | 4.08 | 7.15 (6.29 to 7.25) | 3.07 | 0.960 | 3 | yes |
| `steady.depth.command_age_peak_ms.p95` | 197 | 203 (190 to 203) | 5.88 | 12.7 | 3 |  |
| `steady.depth.commands_per_s` | 12.1 | 11.9 (11.9 to 12.9) | -0.133 | 1.00 | 3 |  |
| `steady.depth.freshness_ms.max` | 115 | 127 (116 to 127) | 11.6 | 11.0 | 3 | yes |
| `steady.depth.freshness_ms.p50` | 99.3 | 101 (95.0 to 101) | 2.19 | 6.51 | 3 |  |
| `steady.depth.freshness_ms.p95` | 114 | 118 (111 to 118) | 4.08 | 7.67 | 3 |  |
| `steady.depth.freshness_ms.p99` | 115 | 122 (114 to 123) | 6.66 | 8.98 | 3 |  |
| `sweep.n1.cls.batch_size.mean` | 1.00 | 1.00 (1.00 to 1.00) | 0 | 0 | 3 |  |
| `sweep.n1.cls.commands_per_s_per_robot.mean` | 30.0 | 30.0 (30.0 to 30.0) | -0.0333 | 0 | 3 | yes |
| `sweep.n1.cls.freshness_ms.p50` | 4.08 | 5.55 (5.05 to 6.39) | 1.47 | 1.34 | 3 | yes |
| `sweep.n1.cls.freshness_ms.p95` | 4.08 | 6.54 (5.95 to 8.64) | 2.46 | 2.69 | 3 |  |
| `sweep.n1.depth.batch_size.mean` | 1.00 | 1.00 (1.00 to 1.00) | 0 | 0 | 3 |  |
| `sweep.n1.depth.commands_per_s_per_robot.mean` | 12.1 | 11.9 (11.4 to 12.9) | -0.133 | 1.50 | 3 |  |
| `sweep.n1.depth.freshness_ms.p50` | 99.3 | 101 (94.7 to 105) | 2.16 | 10.7 | 3 |  |
| `sweep.n1.depth.freshness_ms.p95` | 114 | 117 (110 to 124) | 3.26 | 14.3 | 3 |  |
| `sweep.n10.cls.batch_size.mean` | 1.56 | 2.81 (2.48 to 3.91) | 1.25 | 1.43 | 3 |  |
| `sweep.n10.cls.commands_per_s_per_robot.mean` | 30.0 | 30.0 (29.9 to 30.0) | -0.0233 | 0.0800 | 3 |  |
| `sweep.n10.cls.freshness_ms.p50` | 8.96 | 19.0 (14.5 to 24.3) | 10.0 | 9.84 | 3 | yes |
| `sweep.n10.cls.freshness_ms.p95` | 12.9 | 32.0 (29.6 to 40.1) | 19.1 | 10.5 | 3 | yes |
| `sweep.n10.depth.batch_size.mean` | 1.00 | 1.00 (1.00 to 1.00) | 0 | 0 | 3 |  |
| `sweep.n10.depth.commands_per_s_per_robot.mean` | 0.997 | 0.993 (0.937 to 1.11) | -0.00333 | 0.177 | 3 |  |
| `sweep.n10.depth.freshness_ms.p50` | 99.8 | 106 (101 to 107) | 5.84 | 5.93 | 3 |  |
| `sweep.n10.depth.freshness_ms.p95` | 115 | 121 (120 to 122) | 6.41 | 2.38 | 3 | yes |
| `sweep.n3.cls.batch_size.mean` | 1.00 | 1.00 (1.00 to 1.47) | 0.000383 | 0.470 | 3 |  |
| `sweep.n3.cls.commands_per_s_per_robot.mean` | 30.0 | 30.0 (30.0 to 30.0) | -0.0222 | 0.0222 | 3 |  |
| `sweep.n3.cls.freshness_ms.p50` | 4.08 | 5.94 (5.11 to 9.15) | 1.86 | 4.04 | 3 |  |
| `sweep.n3.cls.freshness_ms.p95` | 4.77 | 8.70 (6.08 to 11.1) | 3.93 | 5.06 | 3 |  |
| `sweep.n3.depth.batch_size.mean` | 1.00 | 1.00 (1.00 to 1.00) | 0 | 0 | 3 |  |
| `sweep.n3.depth.commands_per_s_per_robot.mean` | 4.02 | 3.99 (3.94 to 4.22) | -0.0333 | 0.278 | 3 |  |
| `sweep.n3.depth.freshness_ms.p50` | 99.9 | 102 (96.4 to 102) | 2.14 | 5.73 | 3 |  |
| `sweep.n3.depth.freshness_ms.p95` | 115 | 117 (112 to 117) | 1.96 | 5.59 | 3 |  |
| `sweep.n5.cls.batch_size.mean` | 1.00 | 1.22 (1.00 to 1.68) | 0.219 | 0.674 | 3 |  |
| `sweep.n5.cls.commands_per_s_per_robot.mean` | 30.0 | 30.0 (30.0 to 30.0) | -0.0133 | 0.0133 | 3 |  |
| `sweep.n5.cls.freshness_ms.p50` | 4.77 | 5.96 (5.60 to 12.5) | 1.18 | 6.92 | 3 |  |
| `sweep.n5.cls.freshness_ms.p95` | 6.83 | 10.7 (8.51 to 16.3) | 3.83 | 7.78 | 3 |  |
| `sweep.n5.depth.batch_size.mean` | 1.00 | 1.00 (1.00 to 1.00) | 0 | 0 | 3 |  |
| `sweep.n5.depth.commands_per_s_per_robot.mean` | 2.41 | 2.37 (2.33 to 2.50) | -0.0400 | 0.173 | 3 |  |
| `sweep.n5.depth.freshness_ms.p50` | 99.4 | 102 (98.5 to 104) | 2.19 | 5.50 | 3 |  |
| `sweep.n5.depth.freshness_ms.p95` | 114 | 117 (114 to 120) | 2.63 | 5.51 | 3 |  |

Counted runs: 24.
Counted runs behind the predictions: 3.

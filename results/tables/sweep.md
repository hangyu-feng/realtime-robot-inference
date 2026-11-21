# Scaling sweep

DESIGN.md 9.3 and 6.5, `single` topology. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## `cls`

| Robots | Runs | Commands/s per robot, predicted | Commands/s per robot, min | Commands/s per robot, mean | Commands/s per robot, max | Freshness p50 (ms) | Freshness p95 (ms) | Mean batch size | Share of batches of 1 | Utilization |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 30.0 | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) | 5.55 (5.05 to 6.39) | 6.54 (5.95 to 8.64) | 1.00 (1.00 to 1.00) | 1.000 (1.000 to 1.000) | 0.115 (0.104 to 0.116) |
| 3 | 3 | 30.0 | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) | 5.94 (5.11 to 9.15) | 8.70 (6.08 to 11.1) | 1.00 (1.00 to 1.47) | 1.000 (0.530 to 1.000) | 0.322 (0.318 to 0.345) |
| 5 | 3 | 30.0 | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) | 5.96 (5.60 to 12.5) | 10.7 (8.51 to 16.3) | 1.22 (1.00 to 1.68) | 0.781 (0.641 to 0.998) | 0.526 (0.502 to 0.574) |
| 10 | 3 | 30.0 | 29.8 (29.8 to 30.0) | 30.0 (29.9 to 30.0) | 30.0 (30.0 to 30.0) | 19.0 (14.5 to 24.3) | 32.0 (29.6 to 40.1) | 2.81 (2.48 to 3.91) | 0.235 (0.194 to 0.306) | 0.961 (0.915 to 0.970) |

Counted runs: 12.
Counted runs behind the predictions: 3.

## `depth`

| Robots | Runs | Commands/s per robot, predicted | Commands/s per robot, min | Commands/s per robot, mean | Commands/s per robot, max | Freshness p50 (ms) | Freshness p95 (ms) | Mean batch size | Share of batches of 1 | Utilization |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 12.1 | 11.4 (11.4 to 12.9) | 11.9 (11.4 to 12.9) | 12.9 (11.4 to 12.9) | 101 (94.7 to 105) | 117 (110 to 124) | 1.00 (1.00 to 1.00) | 1.000 (1.000 to 1.000) | 2.493 (2.295 to 2.498) |
| 3 | 3 | 4.02 | 3.93 (3.93 to 4.20) | 3.99 (3.94 to 4.22) | 4.23 (3.97 to 4.23) | 102 (96.4 to 102) | 117 (112 to 117) | 1.00 (1.00 to 1.00) | 1.000 (1.000 to 1.000) | 7.466 (7.046 to 7.537) |
| 5 | 3 | 2.41 | 2.30 (2.30 to 2.50) | 2.37 (2.33 to 2.50) | 2.50 (2.33 to 2.50) | 102 (98.5 to 104) | 117 (114 to 120) | 1.00 (1.00 to 1.00) | 1.000 (1.000 to 1.000) | 12.561 (11.909 to 12.779) |
| 10 | 3 | 0.997 | 0.933 (0.933 to 1.10) | 0.993 (0.937 to 1.11) | 1.13 (0.967 to 1.13) | 106 (101 to 107) | 121 (120 to 122) | 1.00 (1.00 to 1.00) | 1.000 (1.000 to 1.000) | 29.899 (26.608 to 31.672) |

Counted runs: 12.
Counted runs behind the predictions: 3.

## CPU

| Robots | Runs | CPU utilization | Throttled (ms per s) |
| --- | --- | --- | --- |
| 1 | 3 | 0.572 (0.559 to 0.579) | 0 (0 to 0) |
| 3 | 3 | 0.707 (0.701 to 0.723) | 0 (0 to 0) |
| 5 | 3 | 0.821 (0.813 to 0.861) | 0 (0 to 0) |
| 10 | 3 | 0.995 (0.978 to 1.000) | 15.3 (6.03 to 97.5) |

Counted runs: 12.

## Inflection point, from the sweep and the 50-robot run

| Model | Capacity at the first saturated point (frames/s) | Inflection N* (robots) | Utilization crosses 1 |
| --- | --- | --- | --- |
| `cls` | 330 | 11.0 | between 10 and 50 robots |
| `depth` | 12.0 | 0.401 | below 1 robot |

Counted runs: 15.

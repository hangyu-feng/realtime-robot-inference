# Scaling sweep

DESIGN.md 9.3 and 6.5, `single` topology. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## `cls`

| Robots | Runs | Commands/s per robot, predicted | Commands/s per robot, min | Commands/s per robot, mean | Commands/s per robot, max | Freshness p50 (ms) | Freshness p95 (ms) | Mean batch size | Share of batches of 1 | Utilization |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 30.0 | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) | 6.31 (5.77 to 6.47) | 7.76 (6.91 to 7.94) | 1.00 (1.00 to 1.00) | 1.000 (1.000 to 1.000) | 0.133 (0.120 to 0.136) |
| 3 | 3 | 30.0 | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) | 6.70 (6.11 to 9.20) | 10.3 (7.38 to 11.1) | 1.00 (1.00 to 1.47) | 1.000 (0.526 to 1.000) | 0.389 (0.322 to 0.390) |
| 5 | 3 | 30.0 | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) | 9.62 (7.24 to 13.0) | 16.8 (11.2 to 17.4) | 1.26 (1.03 to 1.67) | 0.745 (0.655 to 0.968) | 0.653 (0.543 to 0.716) |
| 10 | 3 | 30.0 | 27.5 (27.5 to 28.9) | 28.5 (28.3 to 29.6) | 29.9 (29.4 to 29.9) | 40.9 (32.3 to 45.6) | 57.4 (50.7 to 59.8) | 7.26 (5.51 to 7.82) | 0.002 (0.000878 to 0.067) | 1.019 (0.988 to 1.029) |

Counted runs: 12.
Counted runs behind the predictions: 3.

## `depth`

| Robots | Runs | Commands/s per robot, predicted | Commands/s per robot, min | Commands/s per robot, mean | Commands/s per robot, max | Freshness p50 (ms) | Freshness p95 (ms) | Mean batch size | Share of batches of 1 | Utilization |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 11.1 | 10.1 (10.1 to 11.3) | 10.2 (10.1 to 11.3) | 11.3 (10.1 to 11.3) | 114 (106 to 116) | 133 (123 to 134) | 1.00 (1.00 to 1.00) | 1.000 (1.000 to 1.000) | 2.914 (2.633 to 2.923) |
| 3 | 3 | 3.69 | 3.47 (3.47 to 3.73) | 3.53 (3.48 to 3.76) | 3.77 (3.50 to 3.77) | 112 (107 to 115) | 127 (123 to 129) | 1.00 (1.00 to 1.00) | 1.000 (1.000 to 1.000) | 8.442 (7.922 to 8.547) |
| 5 | 3 | 2.21 | 1.83 (1.83 to 2.23) | 2.05 (1.85 to 2.23) | 2.23 (1.87 to 2.23) | 115 (107 to 126) | 133 (122 to 150) | 1.00 (1.00 to 1.00) | 1.000 (1.000 to 1.000) | 14.488 (13.329 to 16.065) |
| 10 | 3 | 0.913 | 0.833 (0.833 to 0.867) | 0.867 (0.860 to 0.893) | 0.900 (0.867 to 0.900) | 114 (110 to 115) | 129 (125 to 130) | 1.00 (1.00 to 1.00) | 1.000 (1.000 to 1.000) | 34.309 (33.268 to 34.613) |

Counted runs: 12.
Counted runs behind the predictions: 3.

## CPU

| Robots | Runs | CPU utilization | Throttled (ms per s) |
| --- | --- | --- | --- |
| 1 | 3 | 0.593 (0.584 to 0.595) | 0 (0 to 0) |
| 3 | 3 | 0.756 (0.703 to 0.757) | 0 (0 to 0) |
| 5 | 3 | 0.916 (0.833 to 0.950) | 0.138 (0 to 6.25) |
| 10 | 3 | 0.985 (0.980 to 0.989) | 12.3 (5.55 to 22.0) |

Counted runs: 12.

## Inflection point, from the sweep and the 50-robot run

| Model | Capacity at the first saturated point (frames/s) | Inflection N* (robots) | Utilization crosses 1 |
| --- | --- | --- | --- |
| `cls` | 294 | 9.81 | between 5 and 10 robots |
| `depth` | 10.3 | 0.343 | below 1 robot |

Counted runs: 12.

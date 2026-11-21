# Freshness under steady load

DESIGN.md 9.1, `single` topology. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## One robot

| Quantity | `cls` predicted | `cls` measured | `depth` predicted | `depth` measured |
| --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 4.08 | 5.50 (4.92 to 5.65) | 99.3 | 101 (95.0 to 101) |
| Freshness p95 (ms) | 4.08 | 6.55 (5.88 to 6.80) | 114 | 118 (111 to 118) |
| Freshness p99 (ms) | 4.08 | 7.15 (6.29 to 7.25) | 115 | 122 (114 to 123) |
| Freshness max (ms) | 4.08 | 8.16 (7.83 to 8.16) | 115 | 127 (116 to 127) |
| Commands per second | 30.0 | 30.0 (30.0 to 30.0) | 12.1 | 11.9 (11.9 to 12.9) |
| Freshness trend (ms per s) |  | 0.00143 (-0.00288 to 0.00354) |  | 0.0267 (0.0200 to 0.0880) |
| Peak command age p95 (ms) | 37.4 | 39.9 (39.2 to 40.1) | 197 | 203 (190 to 203) |
| Commands received |  | 899 (899 to 899) |  | 358 (358 to 388) |
| Stale commands |  | 0 (0 to 0) |  | 0 (0 to 0) |
| Stale ratio |  | 0.000 (0.000 to 0.000) |  | 0.000 (0.000 to 0.000) |
| Out-of-order commands |  | 0 (0 to 0) |  | 0 (0 to 0) |

Counted runs: 3.
Counted runs behind the predictions: 3.

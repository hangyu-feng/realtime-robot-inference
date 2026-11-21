# Freshness under steady load

DESIGN.md 9.1, `single` topology. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## One robot

| Quantity | `cls` predicted | `cls` measured | `depth` predicted | `depth` measured |
| --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 4.74 | 6.11 (5.52 to 6.56) | 108 | 112 (104 to 120) |
| Freshness p95 (ms) | 4.74 | 7.25 (6.40 to 8.04) | 123 | 129 (121 to 138) |
| Freshness p99 (ms) | 4.74 | 7.81 (7.18 to 9.30) | 124 | 133 (126 to 143) |
| Freshness max (ms) | 4.74 | 11.0 (8.07 to 11.0) | 124 | 188 (130 to 188) |
| Commands per second | 30.0 | 30.0 (30.0 to 30.0) | 11.1 | 10.5 (9.53 to 11.5) |
| Freshness trend (ms per s) |  | -0.00977 (-0.0108 to 0.0000167) |  | -0.0594 (-0.129 to 0.0140) |
| Peak command age p95 (ms) | 38.1 | 40.6 (39.7 to 41.4) | 213 | 224 (211 to 247) |
| Commands received |  | 899 (899 to 899) |  | 316 (286 to 346) |
| Stale commands |  | 0 (0 to 0) |  | 0 (0 to 1) |
| Stale ratio |  | 0.000 (0.000 to 0.000) |  | 0.000 (0.000 to 0.003) |
| Out-of-order commands |  | 0 (0 to 0) |  | 0 (0 to 0) |

Counted runs: 3.
Counted runs behind the predictions: 3.

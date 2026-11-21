# Late joiner

DESIGN.md 9.2, `single` topology. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## Per robot and window

| Robot | Window | `cls` commands/s | `cls` p95 freshness (ms) | `depth` commands/s | `depth` p95 freshness (ms) |
| --- | --- | --- | --- | --- | --- |
| A | before | 29.9 (29.9 to 29.9) | 7.26 (6.82 to 8.44) | 10.7 (9.10 to 11.2) | 126 (120 to 147) |
| A | after | 30.0 (30.0 to 30.0) | 7.89 (7.52 to 8.06) | 5.40 (5.00 to 5.70) | 128 (123 to 134) |
| B | after | 30.0 (30.0 to 30.0) | 7.44 (6.77 to 7.86) | 5.35 (4.95 to 5.70) | 127 (123 to 136) |
| A, predicted | after | 30.0 |  | 5.55 |  |

Counted runs: 3.
Counted runs behind the predictions: 3.

## After the join

| Quantity | `cls` | `depth` |
| --- | --- | --- |
| B's first command after its join (ms) | 25.9 (25.6 to 34.1) | 153 (125 to 188) |
| A and B settle at equal rates, counted in 1 s bins (s) | 1 (1 to 1) | 1 (1 to 1) |

Counted runs: 3.

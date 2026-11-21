# Late joiner

DESIGN.md 9.2, `single` topology. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## Per robot and window

| Robot | Window | `cls` commands/s | `cls` p95 freshness (ms) | `depth` commands/s | `depth` p95 freshness (ms) |
| --- | --- | --- | --- | --- | --- |
| A | before | 29.9 (29.9 to 29.9) | 6.40 (6.05 to 6.61) | 12.1 (11.9 to 12.9) | 115 (112 to 116) |
| A | after | 30.0 (30.0 to 30.0) | 6.63 (5.84 to 7.62) | 6.05 (5.95 to 6.45) | 117 (112 to 118) |
| B | after | 30.0 (30.0 to 30.0) | 6.59 (5.88 to 6.60) | 6.00 (5.95 to 6.40) | 116 (112 to 118) |
| A, predicted | after | 30.0 |  | 6.05 |  |

Counted runs: 3.
Counted runs behind the predictions: 3.

## After the join

| Quantity | `cls` | `depth` |
| --- | --- | --- |
| B's first command after its join (ms) | 25.5 (23.9 to 34.1) | 136 (125 to 151) |
| A and B settle at equal rates, counted in 1 s bins (s) | 1 (1 to 1) | 1 (1 to 1) |

Counted runs: 3.

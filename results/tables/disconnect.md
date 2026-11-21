# Disconnect

DESIGN.md 9.4, `single` topology. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## Survivors

| Quantity | Before the kill | After the kill |
| --- | --- | --- |
| Survivors' `depth` commands/s per robot | 3.75 (3.55 to 3.80) | 5.50 (5.18 to 5.65) |
| Survivors' `cls` commands/s per robot | 30.0 (30.0 to 30.0) | 30.0 (30.0 to 30.0) |
| Survivors' `depth` p95 freshness (ms) | 125 (123 to 129) | 125 (122 to 133) |
| Survivors' `cls` p95 freshness (ms) | 7.52 (7.40 to 7.75) | 7.56 (7.07 to 7.59) |
| Predicted survivors' `depth` commands/s per robot | 3.70 | 5.50 |

Counted runs: 3.
Counted runs behind the predictions: 3.

## After the kill

| Quantity | Value |
| --- | --- |
| Time from the kill until the server closes the connection (ms) | 2.81 (2.36 to 2.82) |
| Time until the survivors' `cls` rate settles, counted in 1 s bins (s) | n/a |
| Time until the survivors' `depth` rate settles, counted in 1 s bins (s) | 1 (1 to 2) |

Counted runs: 3.

# Memory budget

DESIGN.md 8.3, `single` topology. A process row is the process's resident set, which counts shared library pages too, so the rows add up to more than the container total. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## At 50 robots

| Component | Predicted bound | Measured (MB) |
| --- | --- | --- |
| Gateway process | base + N x (one frame in the slot + connection buffers) | 38.0 (37.9 to 38.5) |
| Worker `cls` | model + arena for the largest batch | 144 (143 to 144) |
| Worker `depth` | model + arena for the largest batch | 382 (380 to 384) |
| Robot fleet | base + frame pool + N x robot state + samples | 85.4 (85.2 to 85.4) |
| Kernel socket memory | N x (receive buffer + send buffer), both ends | 0.0195 (0.0117 to 0.109) |
| Container total (anonymous + socket) | anonymous memory of the container's processes plus socket memory | 525 (523 to 527) |

Counted runs: 3.

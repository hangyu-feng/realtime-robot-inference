# Memory budget

DESIGN.md 8.3, `single` topology. A process row is the process's resident set, which counts shared library pages too, so the rows add up to more than the container total. A measured cell shows the fact's value, with the lowest and highest value of its counted runs in parentheses.

## At 50 robots

| Component | Predicted bound | Measured (MB) |
| --- | --- | --- |
| Gateway process | base + N x (one frame in the slot + connection buffers) | 38.2 (37.7 to 38.4) |
| Worker `cls` | model + arena for the largest batch | 144 (144 to 144) |
| Worker `depth` | model + arena for the largest batch | 381 (381 to 383) |
| Robot fleet | base + frame pool + N x robot state + samples | 85.6 (85.6 to 85.7) |
| Kernel socket memory | N x (receive buffer + send buffer), both ends | 0.0352 (0.0234 to 0.0469) |
| Container total (anonymous + socket) | anonymous memory of the container's processes plus socket memory | 524 (524 to 525) |

Counted runs: 3.

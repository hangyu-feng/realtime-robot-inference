# Freshness under steady load, from `/metrics`

DESIGN.md 9.1 for every counted `steady` run, in both topologies, rebuilt from the run's `/metrics` scrapes alone and set beside the value its summary computes from the robot's samples.

Freshness p50, p95, p99 and max come from the robot's `freshness_ms` histogram, and the peak command age from its `command_age_peak_ms` histogram, both in the report the robot sent when it stopped, which the final scrape holds. A quantile read from a histogram is the upper edge of the bucket that holds the sample of rank ceil(q x count): it agrees when it is at or above the sample value and at most 2^(1/8) times it, one bucket width. A histogram's maximum is the recorded value itself: it agrees within 1 µs. Commands per second is the rise of `commands_sent`, the gateway's count of the commands it handed to the robot, between the first and the last successful scrape planned within the window, over the time S between the two on the gateway's clock, where the samples count over the window's length L. The robot's own `commands_received` is not used: it stands as of the robot's last report, up to a report period before the scrape. The two agree within (rate x |S - L| + 2) / S: the commands that the difference between S and L holds at the samples' rate, and one command at each end of the span, which the gateway counts when it hands the command over and the robot when it receives it.

The samples' percentiles interpolate between neighbouring samples, so where those lie far apart, as in a sparse tail, a cell can fall outside its rule while the histogram is right (DESIGN.md 10.1). The report covers the robot's whole session, which can hold commands received after the window's end and before the robot stopped. The peak-age histogram holds the peak of every command that a later one replaced, where the samples also count the last command's age at the window's end. The freshness trend is not rebuilt: its slope is fitted against each command's receive time, and no histogram of `/metrics` holds that time.

## `single-777edba2cc15-60d80142-r1`

| Quantity | `cls` `/metrics` | `cls` samples | `cls` agrees | `depth` `/metrics` | `depth` samples | `depth` agrees |
| --- | --- | --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 6.640 | 6.555 | yes | 126.338 | 120.383 | yes |
| Freshness p95 (ms) | 8.611 | 8.039 | yes | 150.242 | 138.402 | yes |
| Freshness p99 (ms) | 10.240 | 9.297 | no | 150.242 | 142.715 | yes |
| Freshness max (ms) | 11.017 | 11.017 | yes | 188.090 | 188.090 | yes |
| Commands per second | 29.967 | 29.967 | yes | 9.533 | 9.533 | yes |
| Peak command age p95 (ms) | 44.667 | 41.370 | yes | 252.676 | 247.294 | yes |

Rates between the scrapes planned at 0 s and 30 s, 30.000 s apart on the gateway's clock; the samples' window is 30 s.

## `single-777edba2cc15-60d80142-r2`

| Quantity | `cls` `/metrics` | `cls` samples | `cls` agrees | `depth` `/metrics` | `depth` samples | `depth` agrees |
| --- | --- | --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 5.583 | 5.517 | yes | 106.237 | 103.954 | yes |
| Freshness p95 (ms) | 6.640 | 6.404 | yes | 126.338 | 120.835 | yes |
| Freshness p99 (ms) | 7.241 | 7.184 | yes | 137.772 | 126.448 | yes |
| Freshness max (ms) | 8.074 | 8.074 | yes | 129.862 | 129.862 | yes |
| Commands per second | 29.966 | 29.967 | yes | 11.533 | 11.533 | yes |
| Peak command age p95 (ms) | 40.960 | 39.738 | yes | 212.474 | 211.083 | yes |

Rates between the scrapes planned at 0 s and 30 s, 30.000 s apart on the gateway's clock; the samples' window is 30 s.

## `single-777edba2cc15-60d80142-r3`

| Quantity | `cls` `/metrics` | `cls` samples | `cls` agrees | `depth` `/metrics` | `depth` samples | `depth` agrees |
| --- | --- | --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 6.640 | 6.108 | yes | 115.852 | 111.881 | yes |
| Freshness p95 (ms) | 7.896 | 7.247 | yes | 137.772 | 128.796 | yes |
| Freshness p99 (ms) | 7.896 | 7.810 | yes | 137.772 | 133.384 | yes |
| Freshness max (ms) | 8.355 | 8.355 | yes | 136.944 | 136.944 | yes |
| Commands per second | 29.966 | 29.967 | yes | 10.533 | 10.533 | yes |
| Peak command age p95 (ms) | 40.960 | 40.581 | yes | 231.705 | 223.819 | yes |

Rates between the scrapes planned at 0 s and 30 s, 30.000 s apart on the gateway's clock; the samples' window is 30 s.

## `split-dde786360952-60d80142-r1`

| Quantity | `cls` `/metrics` | `cls` samples | `cls` agrees | `depth` `/metrics` | `depth` samples | `depth` agrees |
| --- | --- | --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 6.640 | 6.560 | yes | 126.338 | 123.234 | yes |
| Freshness p95 (ms) | 8.611 | 8.044 | yes | 150.242 | 138.180 | yes |
| Freshness p99 (ms) | 9.390 | 8.752 | yes | 150.242 | 144.371 | yes |
| Freshness max (ms) | 10.811 | 10.811 | yes | 147.082 | 147.082 | yes |
| Commands per second | 29.966 | 29.967 | yes | 9.667 | 9.667 | yes |
| Peak command age p95 (ms) | 44.667 | 41.377 | yes | 252.676 | 244.739 | yes |

Rates between the scrapes planned at 0 s and 30 s, 30.000 s apart on the gateway's clock; the samples' window is 30 s.

## `split-dde786360952-60d80142-r2`

| Quantity | `cls` `/metrics` | `cls` samples | `cls` agrees | `depth` `/metrics` | `depth` samples | `depth` agrees |
| --- | --- | --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 5.583 | 5.523 | yes | 106.237 | 104.666 | yes |
| Freshness p95 (ms) | 6.640 | 6.411 | yes | 126.338 | 118.647 | yes |
| Freshness p99 (ms) | 7.241 | 6.925 | yes | 126.338 | 121.173 | yes |
| Freshness max (ms) | 8.502 | 8.502 | yes | 127.426 | 127.426 | yes |
| Commands per second | 29.967 | 29.967 | yes | 11.600 | 11.600 | yes |
| Peak command age p95 (ms) | 40.960 | 39.745 | yes | 212.474 | 205.547 | yes |

Rates between the scrapes planned at 0 s and 30 s, 30.000 s apart on the gateway's clock; the samples' window is 30 s.

## `split-dde786360952-60d80142-r3`

| Quantity | `cls` `/metrics` | `cls` samples | `cls` agrees | `depth` `/metrics` | `depth` samples | `depth` agrees |
| --- | --- | --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 6.640 | 6.249 | yes | 115.852 | 113.330 | yes |
| Freshness p95 (ms) | 7.896 | 7.448 | yes | 137.772 | 131.074 | yes |
| Freshness p99 (ms) | 8.611 | 8.304 | yes | 137.772 | 134.522 | yes |
| Freshness max (ms) | 8.958 | 8.958 | yes | 139.390 | 139.390 | yes |
| Commands per second | 29.967 | 29.967 | yes | 10.400 | 10.400 | yes |
| Peak command age p95 (ms) | 40.960 | 40.782 | yes | 231.705 | 230.355 | yes |

Rates between the scrapes planned at 0 s and 30 s, 29.999 s apart on the gateway's clock; the samples' window is 30 s.

Not every cell agrees: 1 of 72 cells do not (counted runs: 6).

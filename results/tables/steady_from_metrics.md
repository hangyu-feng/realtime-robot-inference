# Freshness under steady load, from `/metrics`

DESIGN.md 9.1 for every counted `steady` run, in both topologies, rebuilt from the run's `/metrics` scrapes alone and set beside the value its summary computes from the robot's samples.

Freshness p50, p95, p99 and max come from the robot's `freshness_ms` histogram, and the peak command age from its `command_age_peak_ms` histogram, both in the report the robot sent when it stopped, which the final scrape holds. A quantile read from a histogram is the upper edge of the bucket that holds the sample of rank ceil(q x count): it agrees when it is at or above the sample value and at most 2^(1/8) times it, one bucket width. A histogram's maximum is the recorded value itself: it agrees within 1 µs. Commands per second is the rise of `commands_sent`, the gateway's count of the commands it handed to the robot, between the first and the last successful scrape planned within the window, over the time S between the two on the gateway's clock, where the samples count over the window's length L. The robot's own `commands_received` is not used: it stands as of the robot's last report, up to a report period before the scrape. The two agree within (rate x |S - L| + 2) / S: the commands that the difference between S and L holds at the samples' rate, and one command at each end of the span, which the gateway counts when it hands the command over and the robot when it receives it.

The samples' percentiles interpolate between neighbouring samples, so where those lie far apart, as in a sparse tail, a cell can fall outside its rule while the histogram is right (DESIGN.md 10.1). The report covers the robot's whole session, which can hold commands received after the window's end and before the robot stopped. The peak-age histogram holds the peak of every command that a later one replaced, where the samples also count the last command's age at the window's end. The freshness trend is not rebuilt: its slope is fitted against each command's receive time, and no histogram of `/metrics` holds that time.

## `single-777edba2cc15-09dba712-r1`

| Quantity | `cls` `/metrics` | `cls` samples | `cls` agrees | `depth` `/metrics` | `depth` samples | `depth` agrees |
| --- | --- | --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 5.120 | 4.924 | yes | 97.420 | 94.955 | yes |
| Freshness p95 (ms) | 6.089 | 5.876 | yes | 115.852 | 110.598 | yes |
| Freshness p99 (ms) | 6.640 | 6.286 | yes | 115.852 | 113.692 | yes |
| Freshness max (ms) | 8.162 | 8.162 | yes | 115.948 | 115.948 | yes |
| Commands per second | 29.967 | 29.967 | yes | 12.933 | 12.933 | yes |
| Peak command age p95 (ms) | 40.960 | 39.206 | yes | 194.840 | 190.282 | yes |

Rates between the scrapes planned at 0 s and 30 s, 30.000 s apart on the gateway's clock; the samples' window is 30 s.

## `single-777edba2cc15-09dba712-r2`

| Quantity | `cls` `/metrics` | `cls` samples | `cls` agrees | `depth` `/metrics` | `depth` samples | `depth` agrees |
| --- | --- | --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 6.089 | 5.654 | yes | 106.237 | 101.464 | yes |
| Freshness p95 (ms) | 7.241 | 6.803 | yes | 126.338 | 118.270 | yes |
| Freshness p99 (ms) | 7.896 | 7.247 | yes | 126.338 | 121.989 | yes |
| Freshness max (ms) | 7.830 | 7.830 | yes | 126.975 | 126.975 | yes |
| Commands per second | 29.967 | 29.967 | yes | 11.933 | 11.933 | yes |
| Peak command age p95 (ms) | 40.960 | 40.136 | yes | 212.474 | 202.959 | yes |

Rates between the scrapes planned at 0 s and 30 s, 30.000 s apart on the gateway's clock; the samples' window is 30 s.

## `single-777edba2cc15-09dba712-r3`

| Quantity | `cls` `/metrics` | `cls` samples | `cls` agrees | `depth` `/metrics` | `depth` samples | `depth` agrees |
| --- | --- | --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 5.583 | 5.503 | yes | 106.237 | 101.455 | yes |
| Freshness p95 (ms) | 6.640 | 6.551 | yes | 126.338 | 118.093 | yes |
| Freshness p99 (ms) | 7.241 | 7.147 | yes | 126.338 | 122.677 | yes |
| Freshness max (ms) | 7.942 | 7.942 | yes | 126.695 | 126.695 | yes |
| Commands per second | 29.967 | 29.967 | yes | 11.933 | 11.933 | yes |
| Peak command age p95 (ms) | 40.960 | 39.884 | yes | 212.474 | 202.561 | yes |

Rates between the scrapes planned at 0 s and 30 s, 30.000 s apart on the gateway's clock; the samples' window is 30 s.

## `split-dde786360952-09dba712-r1`

| Quantity | `cls` `/metrics` | `cls` samples | `cls` agrees | `depth` `/metrics` | `depth` samples | `depth` agrees |
| --- | --- | --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 5.583 | 5.184 | yes | 97.420 | 96.619 | yes |
| Freshness p95 (ms) | 6.640 | 6.177 | yes | 115.852 | 112.344 | yes |
| Freshness p99 (ms) | 6.640 | 6.551 | yes | 126.338 | 116.304 | yes |
| Freshness max (ms) | 7.690 | 7.690 | yes | 119.956 | 119.956 | yes |
| Commands per second | 29.966 | 29.967 | yes | 12.733 | 12.733 | yes |
| Peak command age p95 (ms) | 40.960 | 39.511 | yes | 194.840 | 193.787 | yes |

Rates between the scrapes planned at 0 s and 30 s, 30.001 s apart on the gateway's clock; the samples' window is 30 s.

## `split-dde786360952-09dba712-r2`

| Quantity | `cls` `/metrics` | `cls` samples | `cls` agrees | `depth` `/metrics` | `depth` samples | `depth` agrees |
| --- | --- | --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 6.089 | 5.647 | yes | 106.237 | 100.316 | yes |
| Freshness p95 (ms) | 7.241 | 6.703 | yes | 126.338 | 116.273 | yes |
| Freshness p99 (ms) | 7.896 | 7.447 | yes | 126.338 | 119.673 | yes |
| Freshness max (ms) | 8.318 | 8.318 | yes | 123.311 | 123.311 | yes |
| Commands per second | 29.966 | 29.967 | yes | 12.100 | 12.100 | yes |
| Peak command age p95 (ms) | 40.960 | 40.036 | yes | 212.474 | 199.506 | yes |

Rates between the scrapes planned at 0 s and 30 s, 30.000 s apart on the gateway's clock; the samples' window is 30 s.

## `split-dde786360952-09dba712-r3`

| Quantity | `cls` `/metrics` | `cls` samples | `cls` agrees | `depth` `/metrics` | `depth` samples | `depth` agrees |
| --- | --- | --- | --- | --- | --- | --- |
| Freshness p50 (ms) | 6.089 | 5.646 | yes | 106.237 | 101.672 | yes |
| Freshness p95 (ms) | 7.241 | 6.766 | yes | 126.338 | 117.027 | yes |
| Freshness p99 (ms) | 7.241 | 7.131 | yes | 126.338 | 121.318 | yes |
| Freshness max (ms) | 8.264 | 8.264 | yes | 128.628 | 128.628 | yes |
| Commands per second | 29.967 | 29.967 | yes | 11.900 | 11.900 | yes |
| Peak command age p95 (ms) | 40.960 | 40.099 | yes | 212.474 | 201.952 | yes |

Rates between the scrapes planned at 0 s and 30 s, 30.000 s apart on the gateway's clock; the samples' window is 30 s.

Every cell of every counted run agrees (72 cells, counted runs: 6).

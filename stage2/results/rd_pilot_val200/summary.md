# rd_pilot_val200

200 blocks (val200), input `oracle`, 0 without prediction.
Buildings: 8263 predicted, 6868 real.

| Metric | Value | 95% CI (blocks) |
|---|---|---|
| Building count error (relative, per block) | 0.405 | 0.330 - 0.488 |
| Building count (WD) | 0.281 | 0.238 - 0.334 |
| Footprint area (WD) | 0.226 | 0.184 - 0.274 |
| Width (WD) | 0.176 | 0.159 - 0.202 |
| Length (WD) | 0.188 | 0.148 - 0.250 |
| Height (WD) | 0.068 | 0.048 - 0.100 |
| Height step between touching buildings (WD) | 0.236 | 0.211 - 0.270 |
| Share of touching buildings (abs. error) | 0.137 | 0.108 - 0.167 |
| Overlapping area | 0.000 | 0.000 - 0.000 |
| Area outside the block | 0.001 | 0.001 - 0.002 |
| Coverage error (relative) | 0.021 | 0.018 - 0.024 |
| Volume error (relative) | 0.153 | 0.135 - 0.170 |
| Agreement with the input map, IoU (higher better) | 0.913 | 0.908 - 0.918 |
| Height class agrees with the input map (higher better) | 0.851 | 0.835 - 0.870 |
| Mean score used to pick checkpoints | 0.167 | 0.154 - 0.186 |

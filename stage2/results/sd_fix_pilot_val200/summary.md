# sd_fix_pilot_val200

200 blocks (val200), input `oracle`, 0 without prediction.
Buildings: 7380 predicted, 6868 real.

| Metric | Value | 95% CI (blocks) |
|---|---|---|
| Building count error (relative, per block) | 0.302 | 0.217 - 0.387 |
| Building count (WD) | 0.109 | 0.083 - 0.147 |
| Footprint area (WD) | 0.077 | 0.056 - 0.111 |
| Width (WD) | 0.128 | 0.102 - 0.162 |
| Length (WD) | 0.084 | 0.073 - 0.107 |
| Height (WD) | 0.078 | 0.060 - 0.101 |
| Height step between touching buildings (WD) | 0.175 | 0.133 - 0.231 |
| Share of touching buildings (abs. error) | 0.316 | 0.292 - 0.345 |
| Overlapping area | 0.000 | 0.000 - 0.000 |
| Area outside the block | 0.018 | 0.015 - 0.022 |
| Coverage error (relative) | 0.079 | 0.070 - 0.088 |
| Volume error (relative) | 0.206 | 0.181 - 0.228 |
| Agreement with the input map, IoU (higher better) | 0.675 | 0.656 - 0.690 |
| Height class agrees with the input map (higher better) | 0.670 | 0.647 - 0.696 |
| Mean score used to pick checkpoints | 0.175 | 0.162 - 0.192 |

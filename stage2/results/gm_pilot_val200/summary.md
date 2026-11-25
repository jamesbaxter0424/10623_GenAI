# gm_pilot_val200

200 blocks (val200), input `oracle`, 0 without prediction.
Buildings: 7411 predicted, 6868 real.

| Metric | Value | 95% CI (blocks) |
|---|---|---|
| Building count error (relative, per block) | 0.377 | 0.310 - 0.454 |
| Building count (WD) | 0.156 | 0.121 - 0.208 |
| Footprint area (WD) | 0.362 | 0.284 - 0.432 |
| Width (WD) | 0.235 | 0.190 - 0.277 |
| Length (WD) | 0.297 | 0.250 - 0.358 |
| Height (WD) | 0.307 | 0.245 - 0.379 |
| Height step between touching buildings (WD) | 0.794 | 0.703 - 0.899 |
| Share of touching buildings (abs. error) | 0.221 | 0.188 - 0.252 |
| Overlapping area | 0.000 | 0.000 - 0.000 |
| Area outside the block | 0.039 | 0.034 - 0.045 |
| Coverage error (relative) | 0.291 | 0.256 - 0.319 |
| Volume error (relative) | 0.397 | 0.364 - 0.431 |
| Agreement with the input map, IoU (higher better) | 0.388 | 0.369 - 0.413 |
| Height class agrees with the input map (higher better) | 0.673 | 0.640 - 0.705 |
| Mean score used to pick checkpoints | 0.363 | 0.338 - 0.387 |

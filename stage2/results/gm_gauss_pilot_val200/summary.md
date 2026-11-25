# gm_gauss_pilot_val200

200 blocks (val200), input `oracle`, 0 without prediction.
Buildings: 7701 predicted, 6868 real.

| Metric | Value | 95% CI (blocks) |
|---|---|---|
| Building count error (relative, per block) | 0.409 | 0.328 - 0.507 |
| Building count (WD) | 0.181 | 0.138 - 0.231 |
| Footprint area (WD) | 0.366 | 0.295 - 0.443 |
| Width (WD) | 0.239 | 0.195 - 0.283 |
| Length (WD) | 0.311 | 0.257 - 0.373 |
| Height (WD) | 0.299 | 0.239 - 0.385 |
| Height step between touching buildings (WD) | 0.789 | 0.686 - 0.890 |
| Share of touching buildings (abs. error) | 0.227 | 0.199 - 0.261 |
| Overlapping area | 0.000 | 0.000 - 0.000 |
| Area outside the block | 0.052 | 0.047 - 0.058 |
| Coverage error (relative) | 0.248 | 0.217 - 0.270 |
| Volume error (relative) | 0.366 | 0.335 - 0.400 |
| Agreement with the input map, IoU (higher better) | 0.401 | 0.382 - 0.423 |
| Height class agrees with the input map (higher better) | 0.664 | 0.635 - 0.696 |
| Mean score used to pick checkpoints | 0.366 | 0.343 - 0.394 |

# sd_pilot_val200

200 blocks (val200), input `oracle`, 0 without prediction.
Buildings: 7909 predicted, 6868 real.

| Metric | Value | 95% CI (blocks) |
|---|---|---|
| Building count error (relative, per block) | 0.409 | 0.310 - 0.522 |
| Building count (WD) | 0.212 | 0.175 - 0.262 |
| Footprint area (WD) | 0.190 | 0.155 - 0.243 |
| Width (WD) | 0.205 | 0.165 - 0.254 |
| Length (WD) | 0.100 | 0.082 - 0.146 |
| Height (WD) | 0.076 | 0.058 - 0.103 |
| Height step between touching buildings (WD) | 0.209 | 0.152 - 0.260 |
| Share of touching buildings (abs. error) | 0.268 | 0.243 - 0.296 |
| Overlapping area | 0.000 | 0.000 - 0.000 |
| Area outside the block | 0.032 | 0.026 - 0.037 |
| Coverage error (relative) | 0.108 | 0.095 - 0.126 |
| Volume error (relative) | 0.239 | 0.203 - 0.272 |
| Agreement with the input map, IoU (higher better) | 0.625 | 0.608 - 0.640 |
| Height class agrees with the input map (higher better) | 0.660 | 0.630 - 0.689 |
| Mean score used to pick checkpoints | 0.216 | 0.203 - 0.241 |

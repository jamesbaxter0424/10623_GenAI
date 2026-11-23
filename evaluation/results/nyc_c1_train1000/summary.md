# nyc_c1_train1000

1000 windows, 2028 target blocks, 4 sample(s) per window, 29208 block-neighbour pairs. 95% block-cluster bootstrap intervals (2000 draws).

| Metric | Value | 95% CI | Better |
|---|---|---|---|
| FAR err. | 0.755 | [0.670, 0.851] | lower |
| FAR rel. err. (median) | 0.273 | [0.264, 0.283] | lower |
| Cov. err. | 0.105 | [0.102, 0.109] | lower |
| Height acc. | 0.819 | [0.806, 0.832] | higher |
| Res. err. | 0.162 | [0.154, 0.170] | lower |
| Use acc. (text only) | 0.898 | [0.888, 0.909] | higher |
| Hit rate (FAR and cov. within 10%) | 0.141 | [0.131, 0.151] | higher |
| Boundary (WD of height difference) | 0.256 | [0.181, 0.350] | lower |
| Street wall (WD) | 0.086 | [0.081, 0.092] | lower |
| Street wall vs neighbour (WD) | 0.046 | [0.042, 0.050] | lower |
| Footprint (mean normalised WD) | 0.144 | [0.133, 0.155] | lower |
|   region area | 0.076 | [0.070, 0.083] | lower |
|   region convexity | 0.062 | [0.048, 0.080] | lower |
|   nearest-region distance | 0.186 | [0.168, 0.204] | lower |
|   regions per ha | 0.250 | [0.234, 0.267] | lower |
| Diversity | 0.506 | [0.501, 0.511] | - |

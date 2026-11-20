# nyc_baseline_test1000_gtpos

1000 windows, 1817 target blocks, 4 sample(s) per window, 28908 block-neighbour pairs. 95% block-cluster bootstrap intervals (2000 draws).

| Metric | Value | 95% CI | Better |
|---|---|---|---|
| FAR err. | 0.427 | [0.401, 0.458] | lower |
| FAR rel. err. (median) | 0.321 | [0.312, 0.330] | lower |
| Cov. err. | 0.123 | [0.119, 0.128] | lower |
| Height acc. | 0.845 | [0.832, 0.857] | higher |
| Res. err. | 0.186 | [0.177, 0.195] | lower |
| Use acc. (text only) | 0.892 | [0.879, 0.904] | higher |
| Hit rate (FAR and cov. within 10%) | 0.107 | [0.098, 0.115] | higher |
| Boundary (WD of height difference) | 0.133 | [0.096, 0.189] | lower |
| Street wall (WD) | 0.073 | [0.068, 0.080] | lower |
| Street wall vs neighbour (WD) | 0.070 | [0.065, 0.075] | lower |
| Footprint (mean normalised WD) | 0.159 | [0.140, 0.182] | lower |
|   region area | 0.092 | [0.079, 0.106] | lower |
|   region convexity | 0.086 | [0.073, 0.101] | lower |
|   nearest-region distance | 0.171 | [0.147, 0.196] | lower |
|   regions per ha | 0.288 | [0.241, 0.340] | lower |
| Diversity | 0.529 | [0.524, 0.534] | - |
| CNN acc. (two-sample test) | 0.869 | [0.748, 0.985] over 5 splits | -> 0.5 |

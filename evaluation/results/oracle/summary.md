# oracle

1000 windows, 1817 target blocks, 1 sample(s) per window, 7227 block-neighbour pairs. 95% block-cluster bootstrap intervals (2000 draws).

| Metric | Value | 95% CI | Better |
|---|---|---|---|
| FAR err. | 0.131 | [0.116, 0.148] | lower |
| FAR rel. err. (median) | 0.092 | [0.086, 0.097] | lower |
| Cov. err. | 0.012 | [0.011, 0.013] | lower |
| Height acc. | 0.996 | [0.992, 0.999] | higher |
| Res. err. | 0.070 | [0.067, 0.074] | lower |
| Use acc. (text only) | 0.969 | [0.959, 0.977] | higher |
| Hit rate (FAR and cov. within 10%) | 0.560 | [0.537, 0.584] | higher |
| Boundary (WD of height difference) | 0.000 | [0.000, 0.000] | lower |
| Street wall (WD) | 0.000 | [0.000, 0.000] | lower |
| Street wall vs neighbour (WD) | 0.000 | [0.000, 0.000] | lower |
| Footprint (mean normalised WD) | 0.000 | [0.000, 0.000] | lower |
|   region area | 0.000 | [0.000, 0.000] | lower |
|   region convexity | 0.000 | [0.000, 0.000] | lower |
|   nearest-region distance | 0.000 | [0.000, 0.000] | lower |
|   regions per ha | 0.000 | [0.000, 0.000] | lower |
| Diversity | nan | [nan, nan] | - |
| CNN acc. (two-sample test) | 0.500 | [0.500, 0.500] over 5 splits | -> 0.5 |

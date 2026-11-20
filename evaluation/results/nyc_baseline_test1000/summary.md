# nyc_baseline_test1000

1000 windows, 1817 target blocks, 4 sample(s) per window, 28908 block-neighbour pairs. 95% block-cluster bootstrap intervals (2000 draws).

| Metric | Value | 95% CI | Better |
|---|---|---|---|
| FAR err. | 0.733 | [0.697, 0.775] | lower |
| FAR rel. err. (median) | 0.685 | [0.675, 0.693] | lower |
| Cov. err. | 0.273 | [0.267, 0.280] | lower |
| Height acc. | 0.725 | [0.710, 0.739] | higher |
| Res. err. | 0.245 | [0.235, 0.255] | lower |
| Use acc. (text only) | 0.798 | [0.783, 0.813] | higher |
| Hit rate (FAR and cov. within 10%) | 0.028 | [0.023, 0.033] | higher |
| Boundary (WD of height difference) | 0.210 | [0.177, 0.282] | lower |
| Street wall (WD) | 0.106 | [0.100, 0.114] | lower |
| Street wall vs neighbour (WD) | 0.057 | [0.052, 0.062] | lower |
| Footprint (mean normalised WD) | 0.223 | [0.209, 0.239] | lower |
|   region area | 0.203 | [0.189, 0.218] | lower |
|   region convexity | 0.159 | [0.133, 0.185] | lower |
|   nearest-region distance | 0.153 | [0.124, 0.185] | lower |
|   regions per ha | 0.378 | [0.335, 0.421] | lower |
| Diversity | 0.596 | [0.591, 0.600] | - |
| CNN acc. (two-sample test) | 0.964 | [0.909, 0.984] over 5 splits | -> 0.5 |

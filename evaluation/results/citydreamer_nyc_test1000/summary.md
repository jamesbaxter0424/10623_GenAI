# citydreamer_nyc_test1000

1000 windows, 1817 target blocks, 4 sample(s) per window, 28908 block-neighbour pairs. 95% block-cluster bootstrap intervals (2000 draws).

| Metric | Value | 95% CI | Better |
|---|---|---|---|
| FAR err. | 0.510 | [0.478, 0.548] | lower |
| FAR rel. err. (median) | 0.402 | [0.390, 0.414] | lower |
| Cov. err. | 0.151 | [0.146, 0.156] | lower |
| Height acc. | 0.738 | [0.723, 0.754] | higher |
| Res. err. | 0.215 | [0.206, 0.225] | lower |
| Use acc. (text only) | 0.812 | [0.798, 0.827] | higher |
| Hit rate (FAR and cov. within 10%) | 0.068 | [0.060, 0.075] | higher |
| Boundary (WD of height difference) | 0.092 | [0.076, 0.148] | lower |
| Street wall (WD) | 0.081 | [0.075, 0.087] | lower |
| Street wall vs neighbour (WD) | 0.029 | [0.025, 0.033] | lower |
| Footprint (mean normalised WD) | 0.126 | [0.119, 0.135] | lower |
|   region area | 0.102 | [0.090, 0.113] | lower |
|   region convexity | 0.133 | [0.112, 0.156] | lower |
|   nearest-region distance | 0.085 | [0.070, 0.106] | lower |
|   regions per ha | 0.185 | [0.163, 0.208] | lower |
| Diversity | 0.570 | [0.564, 0.576] | - |

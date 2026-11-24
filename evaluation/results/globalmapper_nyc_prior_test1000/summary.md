# globalmapper_nyc_prior_test1000

1000 windows, 1817 target blocks, 4 sample(s) per window, 28908 block-neighbour pairs. 95% block-cluster bootstrap intervals (2000 draws).

| Metric | Value | 95% CI | Better |
|---|---|---|---|
| FAR err. | 0.630 | [0.582, 0.683] | lower |
| FAR rel. err. (median) | 0.434 | [0.423, 0.444] | lower |
| Cov. err. | 0.133 | [0.127, 0.140] | lower |
| Height acc. | 0.786 | [0.766, 0.805] | higher |
| Res. err. | 0.268 | [0.253, 0.284] | lower |
| Use acc. (text only) | 0.812 | [0.792, 0.831] | higher |
| Hit rate (FAR and cov. within 10%) | 0.045 | [0.037, 0.053] | higher |
| Boundary (WD of height difference) | 0.836 | [0.736, 0.949] | lower |
| Street wall (WD) | 0.131 | [0.122, 0.141] | lower |
| Street wall vs neighbour (WD) | 0.059 | [0.053, 0.065] | lower |
| Footprint (mean normalised WD) | 0.214 | [0.199, 0.234] | lower |
|   region area | 0.145 | [0.133, 0.159] | lower |
|   region convexity | 0.126 | [0.111, 0.143] | lower |
|   nearest-region distance | 0.124 | [0.115, 0.135] | lower |
|   regions per ha | 0.463 | [0.403, 0.532] | lower |
| Diversity | 0.267 | [0.264, 0.271] | - |

# Change (3): real map outside the mask as input (Nov 25)

Cumulative: (3) = baseline + change (1) numeric nodes + context input (change (2) is not trained). 55 epochs on an A10G (12.2 h), 1,000 test
windows, true centres, 4 samples. Lower is better except where noted. Intervals: 95% block-cluster bootstrap.

| Metric | Liu, true centres | (1) | **(3)** | CityDreamer | GlobalMapper (random) |
|---|---|---|---|---|---|
| Coverage error | 0.123 | 0.108 | **0.083** | 0.151 | 0.147 |
| FAR error | 0.427 | 0.393 | **0.359** | 0.510 | 0.617 |
| FAR relative error (median) | 0.321 | 0.282 | **0.224** | 0.402 | 0.397 |
| Hit rate, higher better | 0.107 | 0.114 | **0.149** | 0.068 | 0.045 |
| Street wall vs neighbour (WD) | 0.070 | 0.058 | **0.005** | 0.029 | 0.084 |
| Street wall (WD) | 0.073 | 0.081 | **0.037** | 0.081 | 0.117 |
| C2ST accuracy, 0.5 = real-like | 0.869 | 0.951 | **0.608** | 0.942 | not run |
| Footprint | 0.159 | **0.092** | 0.112 | 0.126 | 0.231 |
| Height accuracy, higher better | 0.845 | **0.852** | 0.825 | 0.738 | 0.789 |
| Use accuracy, higher better | 0.892 | **0.901** | 0.868 | 0.812 | 0.810 |
| Residential error | 0.186 | **0.178** | 0.190 | 0.215 | 0.269 |
| Boundary height step (WD) | **0.133** | 0.140 | 0.370 | 0.092 | 0.824 |

Paired differences (3) - (1), all significant (95% interval excludes 0): FAR error -0.034, coverage error -0.025, hit rate +0.035 (better);
height accuracy -0.027, residential error +0.013 (worse). Against the baseline: FAR error -0.068, coverage -0.040, hit +0.042 (better), height -0.020 (worse),
residential error not significant.

**Main weakness: tall buildings.** Mean height class by requested height class (target H, generated H, per-block share correct):

| Requested class | blocks | (1): H, share correct | (3): H, share correct | real H |
|---|---|---|---|---|
| 1-2 floors | 1,636 | 1.94, 0.92 | 1.73, 0.97 | 1.84 |
| 3-5 floors | 338 | 2.99, 0.62 | 2.71, **0.25** | 3.49 |
| 6-12 floors | 78 | 4.88, 0.39 | 4.25, 0.35 | 6.80 |
| 13+ floors | 9 | 11.05, 0.31 | 4.80, **0.00** | 23.55 |

The mean height (generated 1.98, real 2.32) and the boundary height step (-0.42 floors on average, real -0.05) are both too low, mostly in
the less common tall classes. Possible causes, not tested: few tall blocks in the data, and the context being a stronger signal than the
requested height.

Caveats: the (3) diffusion loss is inside-mask only, so it is not comparable with (1). Street-wall continuity is measured against the real
neighbours, which the model sees as input, so a large gain there is expected by construction. C2ST empties everything outside the block before
the test, so it does not reward copying the surroundings. The class table has one row per window and target block (2,126 rows, 1,817 distinct blocks), averaged over the 4 samples.

Files: `aws/c3_pipeline.log`, `evaluation/results/nyc_c3_test1000/`. Stage-1 maps for stage 2: `stage2/out/stage1/nyc_c3.pkl` (local).

# GlobalMapper + Map: Final Baseline Results

## Run configuration

- Method: GlobalMapper + map (baseline)
- Full training: 22,237 graph-matched NYC blocks
- Trained map-to-latent network: 7,991,328 parameters
- GlobalMapper decoder: frozen
- Stopping: no improvement
- Training steps: 97,500
- Training time: 2.216 hours
- Training GPU: NVIDIA A100-SXM4-80GB
- Best validation score: 0.295853
- Selected temperature: 1.0
- Test split: 1,000 blocks; each configuration scored once

## Final scores

| Test condition | Mean score (lower better) |
|---|---:|
| Oracle (real map, trimmed) | 0.338 |
| Oracle raw (real map, untrimmed) | 0.277 |
| Stage 1 NYC C3 (generated map) | 0.535 |

## Findings and limitations

All three test configurations produced predictions for all
1,000 test blocks. The raw oracle run has a lower mean score,
but it permits overlaps (overlapping-area metric 0.096).
The trimmed oracle output has an overlapping-area metric
rounded to 0.000.

The end-to-end Stage 1 NYC C3 result is worse than the
real-map oracle result. Its count error is 1.403, coverage
error is 17.275, and relative volume error is 141.857.
These results indicate substantial sensitivity to the
generated map. The large relative errors warrant
per-block investigation; their cause has not been verified.

## Method summary (for the final report)

The method borrows COHO's idea of quantizing the
GlobalMapper latent into discrete levels and learning a
masked prediction over those levels. In this implementation,
the conditioning signal is the block's own 4 m map,
rather than COHO's neighboring-block context. Each
block is generated separately. A small CNN extracts
map features, and a Transformer predicts 64 groups
of latent codes (eight latent dimensions per group,
20 levels per dimension). Predicted levels become
bin centers and are passed into a frozen GlobalMapper
decoder to generate building footprints. A separately
fitted height network assigns building heights.

The frozen decoder creates a representation ceiling.
On validation blocks, quantizing real latents to 20
levels scores approximately 0.176, whereas the trained
map-conditioned predictor's best validation score
is 0.29585. This difference suggests that predicting
plausible latent codes from the map remains a central
challenge. The method is a baseline for comparison,
not an attempt to tune against the test set.

## Validation figure

`fig_gm_selected.png` compares input maps, real
buildings, the rule baseline and GlobalMapper + map.
Selected validation row IDs: 16970, 4285, 7846, 8181, 8126, 16936.

## Reproducibility

Full raw outputs, model weights, bin definitions,
height network, validation summaries, and test predictions
are in `globalmapper_final_handoff.zip` on the team
shared Google Drive. Large artifacts are not committed
to GitHub. The test was not rerun during packaging.

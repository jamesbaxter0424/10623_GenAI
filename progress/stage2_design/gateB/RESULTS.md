# Stage 2, gate B (Nov 25)

Each method was trained on 2,000 blocks (`pilot2000`) on one A10G and scored on the 200 validation blocks (`val200`) with the
real 4 m map as input. `test1000` was not touched. Gate B: better than the rule baseline in at least one of five metrics
(marked *). Lower is better except where marked.

| Metric | Rule | Raster diffusion | GlobalMapper + map (codes) | GlobalMapper + map (gaussian) | Set diffusion |
|---|---|---|---|---|---|
| Building count error* | 0.289 | 0.405 | 0.377 | 0.409 | 0.409 |
| Footprint area* (WD) | 0.161 | 0.226 | 0.362 | 0.366 | 0.190 |
| Width* (WD) | 0.429 | **0.176** | 0.235 | 0.239 | 0.205 |
| Height* (WD) | 0.133 | **0.068** | 0.307 | 0.299 | 0.076 |
| Height step between touching buildings* (WD) | 0.317 | 0.236 | 0.794 | 0.789 | **0.209** |
| Length (WD) | 0.270 | 0.188 | 0.297 | 0.311 | **0.100** |
| Share of touching buildings | 0.150 | **0.137** | 0.221 | 0.227 | 0.268 |
| Coverage error | 0.025 | **0.021** | 0.291 | 0.248 | 0.108 |
| Volume error | 0.195 | **0.153** | 0.397 | 0.366 | 0.239 |
| Agreement with the map, IoU (higher better) | 0.867 | **0.913** | 0.388 | 0.401 | 0.625 |
| Mean validation score | 0.202 | **0.167** | 0.363 | 0.366 | 0.216 |
| 95% interval of the score (blocks) | .192 - .221 | .154 - .186 | .338 - .387 | .343 - .394 | .203 - .241 |
| Wins out of 5 | - | 3 | 1 | 1 | 3 |
| Buildings (real: 6,868) | 6,259 | 8,263 | 7,411 | 7,701 | 7,909 |
| Best step / steps run / hours | - | 15,000 / 20,000 / 2.31 | 12,000 / 15,000 / 0.42 | 3,000 / 11,000 / 0.39 | 20,000 / 30,000 / 1.01 |

Values are from `stage2/results/<name>_val200/`; they differ slightly from the gate output because the predictions were
generated again (the raster model with 50 sampling steps instead of 25).

## What this says

- **Raster diffusion** is the only method whose mean score beats the rule baseline. It draws too many buildings (8,263 for
  6,868) and the footprint area and count are worse than the rule's.
- **Set diffusion** passes on three metrics but its mean score is not better than the rule's (the intervals overlap). It is
  the best in length and height step and the worst in the share of touching buildings: it leaves gaps between row houses.
- **GlobalMapper + map** passes on one metric only and follows the map poorly (IoU 0.39). The gaussian head is not better. The
  score is flat from about step 6,000, so more steps on these data will not fix it. The decoder ceiling is 0.170.
- All four models predict too many buildings.

## Problem: the set diffusion training diverges

The training loss falls to 0.41 until about step 21,500 (the existence loss is already noisy from 18,000), then rises to 2.7
at step 30,000, and the validation score goes from 0.217 to 0.598. `best.pt` (step 20,000) is from before. The other three
runs do not show this. Gate C would run up to 150,000 steps (early stopping would end it after eight validations without
improvement), so this has to be fixed first. The cause is not known. Candidates, not tested: the constant learning rate (no
decay), bfloat16 attention without query/key normalisation, unbounded modulation scales in the transformer blocks.

## Diagnosis (Nov 25, from the logs, not yet tested by a new run)

- **GlobalMapper + map overfits the 2,000 blocks.** Gaussian head: training MSE of the latent falls from 1.12 (step 900) to 0.0096
  (step 10,900) while the validation score is best at step 3,000 (0.366) and then worsens to 0.38. More training data (gate C has
  22,237 blocks with a graph) may therefore help; more steps on the pilot do not.
- **Its ceiling is low.** Encoding and decoding the real val200 blocks gives score 0.170, map IoU 0.596, touching error 0.219,
  coverage error 0.133. So even perfect latents would only reach about the raster pilot (0.167).
- **Set diffusion:** the weights of `best.pt` (EMA, step 20,000) have normal sizes, so there is no slow weight blow-up before
  that; the cause of the divergence remains open.
- **Small blocks (at most 10 buildings, 41 of 200):** the real 216 buildings become 382 (set) and 368 (raster); the count error
  is about 1.0 for all three methods. It falls to about 0.13-0.17 for 50-100 buildings.

## Set diffusion after the fix (Nov 25, a4-gpu, A10G)

Same 2,000 blocks and val200. Changes together: QK-norm, learning rate 1e-4 (was 2e-4), steps with an exploding gradient skipped
(117 of 60,000), 60,000 steps (was 30,000). It is not known which change stopped the divergence.

| | first version | **after the fix** | rule | raster pilot |
|---|---|---|---|---|
| Mean score (95% interval) | 0.216 (.203 - .241) | **0.175** (.162 - .192) | 0.202 (.192 - .221) | 0.167 (.154 - .186) |
| Better than the rule in | 3 of 5 | 4 of 5 | - | 3 of 5 |
| Building count error | 0.409 | 0.302 | 0.289 | 0.405 |
| Footprint area (WD) | 0.190 | **0.077** | 0.161 | 0.226 |
| Width (WD) | 0.205 | **0.128** | 0.429 | 0.176 |
| Length (WD) | 0.100 | **0.084** | 0.270 | 0.188 |
| Height (WD) | 0.076 | 0.078 | 0.133 | **0.068** |
| Height step (WD) | 0.209 | **0.175** | 0.317 | 0.236 |
| Share of touching buildings | 0.268 | 0.316 (worse) | 0.150 | **0.137** |
| Coverage error | 0.108 | 0.079 | 0.025 | **0.021** |
| Map IoU (higher better) | 0.625 | 0.675 | 0.867 | **0.913** |
| Buildings (real: 6,868) | 7,909 | 7,380 | 6,259 | 8,263 |
| Best step / steps run / hours | 20,000 / 30,000 / 1.01 | 57,500 / 60,000 / 2.75 | - | 15,000 / 20,000 / 2.31 |

Gate A also passed again (0.070). The training loss falls without interruption to 0.30, and the validation score was still going
down slowly at the end (0.181, 0.175, 0.176 in the last three validations). The set pilot had 60,000 steps and the raster pilot 20,000, so the two scores are not a fair
ranking of the methods. 60,000 steps do not prove that 300,000 steps are stable.

## Raster post-processing (`tune_instances`, val200)

| core | 0.75 | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 |
|---|---|---|---|---|---|---|
| score | 0.162 | 0.164 | 0.167 | 0.193 | 0.164 | 0.163 |

The differences other than 1.50 are smaller than the interval of the score; `core` has to be chosen again on the final model.

## Limits

- val200 picked the best checkpoint and is reported here, so the trained models look slightly too good.
- The rule table is fitted on all training blocks, the models saw 2,000.
- One seed, one sample per block.
- Not run: the regression control of the raster model, gate C, the stage-1 maps.

Files: `<run>/` (val.csv, config.yaml, DONE.txt), figures `fig*.png` and `gateB_all_val200.png`. The raw run logs and the
per-100-step training logs (`log.csv`) stay on James' laptop. Checkpoints (best.pt) and
predictions are in `stage2/out/gateB/` (not in git).

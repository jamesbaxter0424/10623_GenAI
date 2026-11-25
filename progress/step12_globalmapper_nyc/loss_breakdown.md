# GlobalMapper NYC: loss breakdown on the validation split

Checkpoint `val_least_loss_all.pth` (epoch 243), 1,440 validation blocks, batch 200 (8 batches), release loss code and weights (`graph_trainer.validation`, `train_gnn.yaml`), deterministic transform (`test_graph_transform`, no flip). Script `baselines/globalmapper_nyc/loss_breakdown.py`, numbers in `loss_breakdown.json`. The total (13,397 per batch) is close to the logged validation loss at epoch 243 (12,414, with random flips and mask noise).

| Term | Weight | Raw (per batch) | Weighted (per batch) | Share of total |
|---|---|---|---|---|
| Size (MSE, summed) | 4.0 | 1,302 | 5,208 | 38.9% |
| Existence (BCE, summed) | 3.0 | 1,412 | 4,236 | 31.6% |
| Position (MSE, summed) | 4.0 | 650 | 2,601 | 19.4% |
| KL (mean over blocks) | 0.5 | 1,324 | 662 | 4.9% |
| Occupancy (MSE, summed) | 1.0 | 264 | 264 | 2.0% |
| Building count (extsum) | 2.0 | 119 | 238 | 1.8% |
| Shape type (CE, summed) | 0.05 | 3,774 | 189 | 1.4% |
| Total | | | 13,397 | |

**KL is weak by construction.** All other terms are summed over the 120 slots of every block in the batch; KL is averaged over blocks. Per block, the terms other than KL add up to about 71 (geometry 43), while the KL of one block enters the total with weight 0.5 / 180 (blocks per batch), about 3.6. So the effective KL weight is about 0.003 per block, and KL is about 12x smaller than geometry and 19x smaller than all other terms. Summed like the others, the same KL (654 per block after the 0.5 weight) would be 9x larger than everything else.

**Building count term has no gradient**: `torch.ge` before the sum; checked on a real batch, `requires_grad` = False (existence BCE: True). It only adds to the reported number.

**Latent statistics** (512 dims, 1,440 validation blocks):

| | Value | Standard normal |
|---|---|---|
| Posterior std of one block (exp(0.5 log_var)) | 0.13 (mean log_var -4.35) | 1 |
| Spread of mu across blocks (std per dim, mean over dims) | 1.21 | 1 |
| KL per dim | 2.56 nats | 0 |

Each block's latent is a narrow point (std 0.13) and the points are spread wider than N(0, I). Most of the space between encoded blocks was never decoded in training, which is consistent with good reconstruction and poor sampling.

**Buildings per block** (existence logit >= 0.5, as test.py):

| Latent | Mean | Median |
|---|---|---|
| Real | 34.3 | 29 |
| Recon (own encoding) | 34.4 | 29 |
| Random (Gaussian fitted to validation encodings, as in our evaluation) | 35.1 | 34 |
| N(0, I) (the paper's prior) | 18.3 | 18 |

Counts are right on average for recon and random; random has less spread (median 34 for every block). Sampling from N(0, I) instead halves the buildings, another sign that the latents are not standard normal.

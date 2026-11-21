# CityDreamer retrained on NYC

CityDreamer's layout generator (Xie et al., CVPR 2024; https://github.com/hzxie/CityDreamer, NTU S-Lab
License 1.0), trained from scratch on our NYC training windows. It is the "context, no targets"
baseline: it sees the real map around the target blocks and the mask, but no graph, no table and no text.
The zero-shot version with the released weights is in `../citydreamer/`.

## Same as our models

- Training windows: the Liu baseline's (`split == "train"` and not `ctx_other_split`, 55,813 windows);
  validation windows for checks only.
- Representation: our 128 x 128 maps of 10 classes at 4 m (no conversion, land use kept).
- Test: the first 1,000 windows of the fixed test ordering, 4 samples each, scored inside the target
  blocks only with `evaluation/`.
- Budget: 12 GPU-hours of training (VQGAN 4 h + sampler 8 h), as for the Liu baseline.

## CityDreamer's method, kept

- Model code unmodified (`third_party/CityDreamer/models/{vqgan,sampler}.py`, commit b3acc0a).
- VQGAN with a 512-word codebook; absorbing-diffusion transformer (24 layers, width 512, 256 steps).
- Losses and optimisers of `core/vqgan/train.py` and `core/sampler/train.py`: Adam (0.5, 0.9) with
  lr = 4.5e-6 x batch for the VQGAN; Adam (0.9, 0.999), lr 2e-4, 7,500-step warm-up, reweighted ELBO,
  batch 10 for the sampler; random flips.
- Inpainting as CityDreamer extends cities: known tokens fixed, masked tokens filled, 32 sampling steps.

## Changes (to be stated in the report)

| Setting | CityDreamer | Here | Why |
|---|---|---|---|
| VQGAN input/output channels | 7 (height + 6 classes) | 10 (our classes) | our representation |
| Resolution | 512 | 128 | our window |
| Channel factors | [1,1,2,2,4] (x16 down) | [2,2,4] (x4 down) | 128 px -> the same 32 x 32 tokens; the original's last three levels |
| VQGAN loss | height L1 + smoothness + class CE + quantisation | class CE + quantisation | no height channel |
| VQGAN batch | 2 | 16 | 32 did not fit in 24 GB; lr follows the original rule |
| Precision | float32 | bf16 autocast, codebook lookup float32 | memory and speed on one A10G |
| Training length | 1,000 epochs | 4 h + 8 h | same budget as our models |

## Checks (gates in `run_pipeline.sh`)

1. VQGAN memorises 8 windows: reconstruction accuracy >= 0.95.
2. VQGAN reconstruction on 512 validation windows >= 0.85 (else use x2 downsampling).
3. Sampler memorises 8 windows (loss falls).
4. Figures on validation windows only: unconditional samples and filled target blocks.

## Running (AWS, conda env `citydreamer`, `NYC_DATA=~/genai-final/data/city`)

```bash
tmux new -s cdnyc 'bash run_pipeline.sh 2>&1 | tee ~/genai-final/logs/cdnyc_pipeline.log'
python generate.py --ckpt ~/genai-final/checkpoints/citydreamer_nyc/sampler/latest.pth --n 1000 --viz 10 \
    --out ~/genai-final/generated/citydreamer_nyc_test1000
cd ../../evaluation
python evaluate.py --pred_dir ~/genai-final/generated/citydreamer_nyc_test1000 --name citydreamer_nyc_test1000
python c2st.py --pred_dir ~/genai-final/generated/citydreamer_nyc_test1000 --name citydreamer_nyc_test1000
```

Measured on the A10G: VQGAN 0.76 s/step (batch 16, bf16, 19 GB); sampler 0.58 s/step (batch 10);
generation 4.5 s per window (4 samples).

| file | content |
|---|---|
| `common.py` | configuration changes, data (same windows as Liu), model building and loading, figures |
| `train_vqgan.py` | stage 1 |
| `train_sampler.py` | stage 2 |
| `generate.py` | target-block filling, unconditional samples, figures |
| `run_pipeline.sh` | stages 1-2 with gates and first figures |

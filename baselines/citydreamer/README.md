# CityDreamer baseline (zero-shot)

The layout generator of CityDreamer (Xie et al., CVPR 2024; https://github.com/hzxie/CityDreamer,
NTU S-Lab License 1.0) used as a context-aware baseline: it sees the real surroundings but no targets,
no graph and no land use. Released weights, no training or fine-tuning.

## What it receives

The real map of the test window with the target blocks masked, nothing else.

1. Our 4 m class map is converted to CityDreamer's layout: a height field (metres / 640) and a one-hot
   of its six classes (road, building, green, construction, coast, other), at 9 px per cell
   (0.444 m/px; CityDreamer's OSM data uses zoom 18, about 0.45 m/px in New York).
   Street -> road; open ground -> null (CityDreamer trains null pixels as road); buildings -> building
   with height = fitted floors of the height class x 3.9 m; water -> coast.
2. The VQGAN encodes the map into 16-px tokens (a 512-px window = 32 x 32 tokens, about 230 m).
3. Tokens touching a target block are masked. The absorbing-diffusion sampler fills them window by
   window (32 steps, as in CityDreamer's inference) while all other tokens stay fixed.
4. The result is decoded and converted back inside the target blocks only: a cell is a building if at
   least half of its pixels are, with the height class of the median building height; other cells are
   open ground. Outside the targets the real map is kept.

## Reporting

- CityDreamer has no land use: generated buildings are written as residential, so **Res. err. and Use
  acc. are n/a**.
- Run the CNN test with `--collapse_use` (uses merged and street/open merged in real and generated crops).
- The released training data appear to include New York (the authors' demo ships NYC height and
  segmentation maps), so the model may have seen our test districts. This can only flatter the baseline.

## Running (AWS, conda env `citydreamer`)

```bash
bash baselines/citydreamer/setup_citydreamer.sh   # clone repo, env, check checkpoint
cd baselines/citydreamer
python run_citydreamer.py --selftest                          # conversion round trip, no model
python run_citydreamer.py --check_model --sids 43146 --device cpu   # load weights, encode/decode only
python run_citydreamer.py --sids 43146,5626 --viz 2 --out ~/genai-final/generated/citydreamer_try
python run_citydreamer.py --n 1000 --viz 20                   # the test set; resumable
cd ../../evaluation
python evaluate.py --pred_dir ~/genai-final/generated/citydreamer_test1000 --name citydreamer_test1000
python c2st.py --pred_dir ~/genai-final/generated/citydreamer_test1000 --name citydreamer_test1000 --collapse_use
```

The checkpoint `LayoutGen.pth` must be downloaded by hand from https://gateway.infinitescript.com/?f=LayoutGen.pth
(the page needs a verification step) and placed at `~/genai-final/checkpoints/citydreamer/LayoutGen.pth`.
Weights are loaded with `strict=True` after removing the `module.` prefix, so a mismatch stops the run
instead of silently skipping weights.

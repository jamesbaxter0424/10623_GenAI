# Evaluation

One evaluation for every model we compare (original Liu model, Liu + true centers, changes 1–3,
no-GNN, GlobalMapper). It reads a model's generated maps, scores only the target blocks,
and writes a per-block table and a summary with 95% confidence intervals.

## Quick start

```bash
# environment: numpy, pandas, scipy (conda env nycdata on the laptop, genai-final on AWS)
cd evaluation
python evaluate.py --oracle --name oracle                       # real map scored as a model
python evaluate.py --pred_dir PATH/Generated --name liu_baseline # one model
python c2st.py --pred_dir PATH/Generated --name liu_baseline     # CNN realism test (needs torch)
python evaluate.py --oracle --name oracle                        # rerun to merge c2st.json into summary.md
python evaluate.py --compare results/model_a results/model_b     # paired differences
python tests/test_metrics.py                                     # sanity checks (16 checks)
```

The data path defaults to `../nyc-massing-data/data/processed/city`; set `NYC_DATA` to change it.
The first run caches per-block class counts and fitted floors in `cache/`.

## Input format (for every model)

A folder with one file per test window, `<sample_id>.npz`:

| array | shape | content |
|---|---|---|
| `gen` | `[S, 128, 128]` uint8 | S generated maps (we use S = 4), classes 0–9 |
| `gt` | `[128, 128]` | the real map (checked against the dataset) |
| `mask` | `[128, 128]` | target blocks (optional; the evaluation uses the dataset's block map) |

Only cells of the target blocks are read, so a model may generate the whole tile (Liu) or only the
mask (change 3). Test windows are the first `N_TEST` (1,000) of the fixed seed-0 ordering of the clean
test windows, the same ordering as `control-3d-scene/tools/nyc_tools.py subset`.

## Metrics

All metrics are per target block, averaged over the S samples. Floors per height class are fitted by
least squares of block FAR on training blocks (1.53 / 3.42 / 7.13 / 25.08 floors).

| Group | Metric | Definition | Better | Reference |
|---|---|---|---|---|
| Condition | Cov. err. | \|building cells / block cells − target coverage\| | lower | L1 density error of Text-to-City [1] |
| | FAR err. | \|Σ floors of building cells / block cells − target FAR\|; also the median relative error | lower | [1] extended to FAR; GenFusion [2] conditions on FAR but evaluates it only visually |
| | Height acc. | area-weighted median height class of the building cells = target class; blocks without buildings are excluded | higher | same definition as `median_floors` in `nyc-massing-data/03_blocks.py` |
| | Res. err. | \|residential floor area / total floor area − target residential share\| (floor area = cells × floors) | lower | ResArea / BldgArea in PLUTO [13] |
| | Use acc. | residential share ≥ 0.5 agrees with the target; reported in the text only | higher | |
| | Hit rate | FAR and coverage both within ±10% of the target (at least ±0.05) | higher | ours |
| Surroundings | Boundary | Wasserstein distance between generated and real distributions of (mean floors of target − mean floors of a real block facing it across a street) | lower | WD as in GlobalMapper [3] and COHO [4]; height transitions in NYC ZR 23-443 [10] |
| | Street wall | WD between generated and real distributions of the built share of the block's street-edge cells | lower | Built Front within 4 m of the street [12]; NYC ZR 23-431 [10]; enclosure [8][9] |
| | Street wall vs neighbour | WD of \|street wall − neighbour's street wall\|, generated vs real | lower | NYC ZR 23-431 (new street walls align with neighbours) |
| Realism | CNN acc. | accuracy of a small CNN separating real from generated block crops (outside the block blanked), half of the blocks for training, half for testing, 5 random splits | → 0.5 | classifier two-sample test [5] |
| | Footprint | mean over four measures of WD(generated, real) / std(real): region area, convexity (area / convex hull), nearest-region distance, regions per hectare | lower | measures as in momepy [7]; WD as in [3][4] |
| Diversity | Diversity | share of block cells that differ between two samples, averaged over pairs | — | multi-sample pairwise distance as in [6] (LPIPS [14] replaced by cell disagreement) |

Notes
- Regions are 4-connected building cells; at 4 m adjacent row houses merge, so they are regions, not buildings.
- Out-Block (GlobalMapper) is not reported: when only target blocks are scored it is zero by construction.
- A paired Boundary error \|d_gen − d_real\| is not reported because the neighbour term cancels (it is the target's height error).
- Water cells are excluded from street edges.

## Statistics

Test windows overlap and share blocks, so intervals use a block-cluster bootstrap (2,000 draws): target
blocks are resampled with replacement and all rows of a sampled block are kept together [15].
`--compare` reports the mean per-block difference between two models on the blocks both were scored on,
with the same bootstrap; a difference is significant when its interval excludes 0.

## Oracle row

`--oracle` scores the real map with the same code. Distribution metrics are 0 and the CNN test is 0.5 by
definition; condition errors are not 0 because the 10-class map cannot encode exact floors, mixed-use
buildings or sub-cell footprints. These values are the floor any model can reach.

## Sanity checks (`tests/test_metrics.py`)

Real map: Boundary, street wall and footprint = 0, height accuracy = 1. Heights moved one class up: FAR
error, Boundary rise, coverage unchanged. Buildings removed: coverage error and street wall rise.
Cells shuffled inside blocks: coverage unchanged, footprint and diversity rise. Identical samples:
diversity = 0. CNN test: 0.50 on real vs real, 0.88 on real vs shuffled blocks (300 windows).

## Files

| file | content |
|---|---|
| `config.py` | paths and settings |
| `data.py` | data loading, per-block cache, fitted floors, street edges, test ordering |
| `condition.py` | coverage, FAR, height class, residential share of one block |
| `context.py` | Boundary and street wall inputs |
| `footprint.py` | building regions and their measures |
| `diversity.py` | pairwise sample disagreement |
| `stats.py` | block-cluster bootstrap and paired differences |
| `evaluate.py` | main script |
| `c2st.py` | CNN two-sample test |
| `results/<name>/` | `per_block.csv`, `pairs.csv`, `footprint.csv`, `summary.json`, `summary.md`, `c2st.json` |

## References

[1] Zhuang et al., Text-to-City, CAADRIA 2024. [2] Gao et al., GenFusion, Information Geography 2025.
[3] He & Aliaga, GlobalMapper, ICCV 2023. [4] He & Aliaga, COHO, ECCV 2024.
[5] Lopez-Paz & Oquab, Revisiting Classifier Two-Sample Tests, ICLR 2017.
[6] Zhu et al., Toward Multimodal Image-to-Image Translation, NeurIPS 2017.
[7] Fleischmann, momepy, JOSS 2019. [8] Ewing & Handy, Journal of Urban Design 2009.
[9] Purciel et al., Journal of Environmental Psychology 2009. [10] NYC Zoning Resolution §23-431, §23-443.
[12] University of Strathclyde urban morphometrics, Built Front (source to be confirmed).
[13] NYC DCP, PLUTO data dictionary. [14] Zhang et al., LPIPS, CVPR 2018.
[15] Efron & Tibshirani, An Introduction to the Bootstrap, 1993.

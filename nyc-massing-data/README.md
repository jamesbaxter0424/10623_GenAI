# NYC block-massing dataset

Dataset for "Generating Urban Massing in Context from Text and Block Graphs".

## Build

```
conda activate nycdata
# pilot (Park Slope, Midtown, LIC) - fast sanity run
python 00_download.py && python 01_lots.py && python 02_buildings.py && python 03_blocks.py && python 04_raster.py && python qa_report.py
# full city (~30 min, ~300 MB download)
bash run_city.sh            # SKIP_DOWNLOAD=1 to reuse data/raw/nyc
```

Stages 1-3 run on `NYC_SET=pilot` (default), `NYC_SET=city` (five boroughs) or one area (`NYC_SET=BK`).
All open cleaning decisions are constants in `config.py` / top of `05_samples.py` (marked `[TEAM DECISION]`).
Every step logs row counts to `data/interim/funnel.csv`.

| stage | script | output |
|---|---|---|
| 0 | `00b_download_city.py` | PLUTO, tax lots, footprints for NYC, split per borough |
| 1 | `01_lots.py` | cleaned lots + PLUTO join, land mask |
| 2 | `02_buildings.py` | footprints linked to PLUTO, floors, height class, use |
| 3 | `03_blocks.py` | block polygons + targets (coverage, FAR, median floors, res share) + validity flags |
| 4 | `04b_mosaic_city.py` | one 4 m class map of NYC (`cls.npy`, `block.npy`, `water.npy`) |
| 5 | `05_samples.py` | CD split, block graph, 1-4 block target groups, 512 m windows |
| QA | `qa_city.py` | `qa/city/summary.md` + figures |

## Use

```python
from dataset import NYCMassing            # numpy; NYCMassingTorch for a torch Dataset
ds = NYCMassing("train")                  # "val" / "test"
x = ds[0]   # cls, mask, node_map, node_attr, node_target, edge_index, edge_attr, ...
```

See the docstring of `dataset.py` for every field. `data/processed/city/quantiles.json`
holds NYC quantiles of the targets for vague words ("dense" = above q66 of FAR).

## Numbers (2026-11-16 build, PLUTO 26v2)

33,337 blocks, 30,866 valid targets; 1.08 M buildings; 84,981 samples
(train 61,621 / val 3,980 / test 19,380; 5,808 train samples show val/test blocks in context and are
excluded by default). Block FAR from footprints vs PLUTO: r = 0.978 on valid blocks.

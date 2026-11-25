# Shared part: data, input, output, metrics

Frozen: all three experiments depend on it. If you find a bug, tell the others before changing anything.

| File | Content |
|---|---|
| `config.py` | paths and constants. Paths can be set with `NYC_DATA`, `NYC_BLDG`, `STAGE2_DATA`, `STAGE2_OUT` |
| `make_data.py`, `splits.py` | build `data/` from the NYC pipeline; the named block lists |
| `records.py` | read records; write and read predictions |
| `geometry.py`, `dataset.py` | polygons <-> pixels; condition images; block-aligned frames |
| `noise.py` | corrupts the input map during training |
| `diffusion.py`, `train_utils.py` | diffusion process; the training loop (EMA, resume, validation, early stopping) |
| `metrics.py`, `evaluate.py` | scores |
| `gate.py` | gates A and B |
| `rule_split.py` | rule baseline |
| `stage1_input.py` | stage-1 samples -> stage-2 input |
| `visualize.py` | side-by-side figures |

## Data: one record per block

`data/{train,val,test}.pkl` hold 22,519 / 1,445 / 6,902 valid blocks with 753,209 / 50,052 / 241,889 buildings. The splits
are the stage-1 splits (by community district). Fields: see the top of `make_data.py`. The two to remember:

- `cls4` is the 4 m map of the block **and its surroundings**, `inside` marks the target block;
- polygons are in **local coordinates**: metres, x to the right, **y down**, origin at the top-left of the window.

| List (`splits.json`) | Blocks | Use |
|---|---|---|
| `overfit8` | 8 | gate A |
| `pilot2000` | 2,000 | gate B |
| `train` | 22,519 | gate C |
| `val200` | 200 | pick checkpoints and settings |
| `test1000` | 1,000 | final scores, once |

## Input noise

The stage-1 maps contain errors, but a generated map has no matching real buildings, so it cannot be a training pair.
Instead the real map of the target block is corrupted at random (boundary cells flipped, region class changed, small
blobs added or removed); 30% of the time it stays clean. The surroundings are never changed.

## Output: the same for every method

JSON lines, one line per block and sample, polygons as WKT in local coordinates, height in metres:

```json
{"row": 123, "sample": 0, "buildings": [{"wkt": "POLYGON ((...))", "h": 9.5}, ...]}
```

Use `records.Writer`. Blocks you cannot generate must still be written, with an empty list: they count as empty.

## Metrics

The stage-1 footprint metric works on 4 m cells, where touching houses merge again, so it **cannot see** whether houses
were separated. These metrics use polygons:

| Metric | Meaning | Better |
|---|---|---|
| `count_err` | relative error of the number of buildings, per block | 0 |
| `wd_count`, `wd_area`, `wd_width`, `wd_length`, `wd_height` | distance between the distributions of real and generated values (Wasserstein, on log values except the count, divided by the real std) | 0 |
| `wd_dz` | the same for the height step between touching buildings | 0 |
| `touch_err` | error of the share of buildings that touch another one (shared wall, gap < 0.3 m) | 0 |
| `overlap`, `outside` | share of the area that overlaps / lies outside the block | 0 |
| `coverage_err`, `volume_err` | relative error of built area and of volume, per block | 0 |
| `map_iou` | do the buildings, drawn at 4 m, agree with the input map? | 1 |
| `hclass_acc` | does the height agree with the height class in the input map? Real buildings only reach 0.80 - 0.87, because classes come from floors, not metres | high |
| `score` | mean of count, area, width, height, dz, touch, overlap and 1 - map_iou; picks checkpoints, not a headline result | 0 |

`python -m common.evaluate --pred <file> --name <name> [--subset val200] [--input <stage-1 file>]` writes
`results/<name>/summary.md` with 95% intervals (bootstrap over blocks; it does not include training randomness).

Reference, val200 with the real map: rule baseline count 0.289, area 0.16, width 0.43, height 0.13, dz 0.32, touch 0.15,
map IoU 0.87, score 0.202.

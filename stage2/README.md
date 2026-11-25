# Stage 2: from the 4 m map to individual buildings

Stage 1 generates a 4 m map of each target block (street, open space, and eight building classes = use x height class).
Stage 2 turns that map into **buildings: one polygon and one height each**.

The question we answer: *in the 4 m map the walls between attached houses are invisible. Which representation fills
them in best?* Measured on a sample of training blocks from all five boroughs: the median building is 6.3 m wide (1.6
cells), 59% of the buildings touch a neighbour, and 81% of the touching pairs have the same class, so the map shows no
boundary between them.

| | Folder | Owner | Idea | Can buildings overlap? |
|---|---|---|---|---|
| Rule baseline | `common/rule_split.py` | shared | cut regions with training statistics | no |
| Raster diffusion | `raster_diffusion/` | James | draw a 1 m image, then split it | no |
| GlobalMapper + map | `globalmapper_map/` | Chenyue | predict GlobalMapper's latent from the map | yes, trimmed |
| Set diffusion | `set_diffusion/` | Daehong | denoise a list of boxes | yes, trimmed |

Chenyue's and Daehong's methods are **baselines** for the comparison with raster diffusion: train them properly, do not tune
them endlessly. Each folder has a README with the commands, the expected numbers and what to send back.

**Chenyue:** start with [globalmapper_map/README.md](globalmapper_map/README.md).
**Daehong:** start with [set_diffusion/README.md](set_diffusion/README.md).
Everything shared (data, metrics, output format) is described in [common/README.md](common/README.md).

## Setup (everybody, once)

1. Install torch >= 2.5 for your CUDA, then `pip install -r stage2/requirements.txt` (shapely must be >= 2.1).
2. Get the data (check with `sha256sum -c SHA256SUMS.txt`; GlobalMapper files: `SHA256SUMS_globalmapper.txt`). Either copy the folder `both/data/` from James' shared folder to `stage2/data/` (353 MB; each README says where every
   file goes), or rebuild it from the NYC pipeline
   (about 3 minutes): `cd stage2 && python -m common.make_data`. Both give the same blocks.
3. Run everything **from the folder `stage2/`**. Check the installation (CPU, a few seconds):

```bash
cd stage2
python -m pytest -q                                          # 22 passed
python -m common.evaluate --real --name check --boot 0       # real buildings: every error 0, map IoU 0.999
```

## Order of work: three gates

| Gate | Data | Question | If it fails |
|---|---|---|---|
| A | `overfit8`, 8 blocks | is the code able to learn? | fix the code |
| B | `pilot2000`, 2,000 blocks | better than the rule baseline on val200 in at least one distribution metric? | stop, report it |
| C | `train`, 22,519 blocks | final model | stops when val200 no longer improves, at most 12 h |

`bash <folder>/run.sh A`, `... B`, `... C` run one stage. Without an argument the script only prints the list of stages; `ALL` runs
the training stages but never the test. James has run A and B for all methods, so Chenyue and Daehong start at C.

**Fairness rules**

- Same blocks, same input maps, same scoring for everybody: never edit `common/`, `data/` or the lists in `splits.json`.
- Choose checkpoints and settings on `val200` only. Score `test1000` **once**, at the end.
- Methods are trained until their own validation score stops improving (20,000 steps without improvement; raster 12,500), with
  the same 12 h cap. The step caps (GlobalMapper 200,000, set 300,000, raster 100,000) should not be reached first. Report steps, hours and
  GPU (all in `DONE.txt`). Do not compare methods at equal epochs: their epochs cost different amounts.
- Every number is reported twice: with the **real** map as input (the method itself) and with the **stage-1** maps
  (the whole pipeline).

## Stage-1 maps as input

```bash
python -m common.stage1_input --pred_dir <generated>/nyc_c3_test1000 --out out/stage1/nyc_c3.pkl
STAGE1="nyc_c3" bash raster_diffusion/run.sh TEST            # same for the other two folders
```

## Final table

```bash
python -m common.rule_split --subset test1000 --out out/rule/test1000_oracle.jsonl
python -m common.evaluate --pred out/rule/test1000_oracle.jsonl --name rule_oracle
python -m common.evaluate --compare rule_oracle diffusion_oracle globalmapper_oracle set_oracle
```

## What has been checked (Nov 24)

On an AWS g5.xlarge (A10G); figures and results in `progress/stage2_design/` (raw logs stay local). Scores are the mean validation score, lower is better.

| Check | Result |
|---|---|
| Rule baseline on val200 | 0.202 |
| Raster diffusion, gate A | **0.055** (need 0.10): pass. Ceiling of the representation on val200: 0.027 |
| GlobalMapper + map, gate A | **0.177** (need 0.22): pass. Ceiling (encode and decode real val200 blocks): 0.174 |
| Set diffusion, gate A | **0.072** (need 0.15): pass. Ceiling (real buildings as rectangles, val200): 0.065 |
| Speed, training steps on `train` | raster 0.37 s/step (batch 16), set 0.11 s/step (batch 64) |

Gate A only shows that 8 blocks can be memorised; it says nothing about new blocks.

Gate B (Nov 25, 2,000 blocks, val200, real map; full table in `../progress/stage2_design/gateB/RESULTS.md`):

| Method | Mean score | Better than the rule in | Remark |
|---|---|---|---|
| Rule baseline | 0.202 | | |
| Raster diffusion | **0.167** | 3 of 5 | too many buildings |
| Set diffusion | 0.175 (first version 0.216) | 4 of 5 | the first version diverged after 23,000 steps; the fix ran 60,000 steps without diverging |
| GlobalMapper + map | 0.363 (gaussian head 0.366) | 1 of 5 | weak: follows the map poorly (IoU 0.39), score flat from step 6,000 |

Gate C has **not** been run.

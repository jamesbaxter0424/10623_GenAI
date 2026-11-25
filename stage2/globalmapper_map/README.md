# GlobalMapper driven by the map (Chenyue)

**Your goal:** train GlobalMapper + map as a **baseline** and score it on the same test blocks as the other methods. The code is
finished; gates A and B are done (James, Nov 24-25). You run the preparation, the final training and the test, and send the
results back. About 1-6 hours of GPU time (A10G), about one hour of your own time. You can start as soon as you have the files.

**You do not need to improve the method.** A baseline must be trained properly and treated like the others, not tuned until it
wins. The only choice you make is one of two sampling temperatures.

## 1. The idea in one minute

Our GlobalMapper is already trained on NYC. It encodes a block into 512 numbers (the latent) and decodes them back into
buildings. It **reconstructs** well but **generates** badly: sampling the 512 numbers from a normal distribution hits
regions the decoder has never seen ("holes"). On 200 validation blocks:

| Latent given to the decoder | score | building count error |
|---|---|---|
| the block's own encoding | 0.170 | 1.3% |
| random normal, as in the official code | 0.621 | 238% |

So the decoder is fine; **choosing the latent is the problem**. COHO (He & Aliaga, ECCV 2024, same group, no code released)
solves it by cutting every latent number into 20 levels and *predicting* the levels from the context. We borrow that, with
a different context: COHO looks at neighbouring blocks, **we look at the stage-1 map of the block itself**.

```
4 m map, resampled along GlobalMapper's own main axis (12 x 32 x 128)
   -> small CNN -> Transformer fills in 64 masked tokens (8 latent numbers each, 20 levels)
   -> 512 levels -> bin centres -> FROZEN GlobalMapper decoder -> building polygons
   -> height network -> one height per building
```

GlobalMapper itself is **not** retrained. Only two small networks are: map -> levels (8 M parameters) and the heights.

| File | Content |
|---|---|
| `gm.py` | wraps the official model; the map in GlobalMapper's canonical space |
| `extract_latents.py` | step 1: encode all blocks |
| `heights.py` | step 2: height network (GlobalMapper has no heights) |
| `quantize.py` | step 3: 20 levels, and the check that decoding still works |
| `model.py`, `train.py` | step 4: map -> levels (`head: codes`), or a normal distribution (`head: gaussian`) |
| `sampling.py`, `generate.py` | generation |
| `config.yaml` | all settings; change one with `--set key=value` |

## 2. What you need

| What | Where from | Size |
|---|---|---|
| this repository, branch `main` | `git clone https://github.com/jamesbaxter0424/10623_GenAI.git` | |
| `stage2/data/` | James (shared folder) | 353 MB |
| `stage2/out/stage1/nyc_c3.pkl` | James | 8 MB |
| `val_least_loss_all.pth`, our NYC GlobalMapper weights (epoch 243 of 250) | James | 288 MB |
| `gm_nyc.tar`: `norm.json` and `train/`, `val/`, `test/` with `processed/`, `index.csv`, `geom.pkl` | James | 655 MB |
| the official code | `git clone https://github.com/Arking1995/GlobalMapper` | small |
| Linux (or WSL) with an NVIDIA GPU | tested on an A10G (24 GB); smaller cards are not tested | |

No `skgeom` is needed: the straight skeletons are already stored in `gm_nyc`.

## 3. Setup (once)

**Where the files from the shared folder go.** James' shared folder `stage2_handover/` has a folder `both/` for both of you and a folder `globalmapper/` only for you.
Download it. After `git clone` (first command block below), copy into your clone of the repository (the folder
`out/stage1/` does not exist yet):

```
stage2_handover/both/data/          ->  10623_GenAI/stage2/data/            (train.pkl, val.pkl, test.pkl, splits.json, info.json)
stage2_handover/both/nyc_c3.pkl     ->  10623_GenAI/stage2/out/stage1/nyc_c3.pkl
```

```bash
cd 10623_GenAI/stage2
mkdir -p out/stage1
cp -r /path/stage2_handover/both/data .
cp /path/stage2_handover/both/nyc_c3.pkl out/stage1/
```

The folder `globalmapper/` (the weights and `gm_nyc.tar`) can stay anywhere outside the repository, for example
`~/gm_files/`; the three `export` variables below point to it.


```bash
git clone https://github.com/jamesbaxter0424/10623_GenAI.git
git clone https://github.com/Arking1995/GlobalMapper
conda create -n stage2 python=3.11 -y && conda activate stage2
pip install torch                          # 2.5 or newer, matching your CUDA
cd 10623_GenAI/stage2
pip install -r requirements.txt torch_geometric networkx scikit-learn torchvision pillow
# copy data/ and nyc_c3.pkl as shown above, then:
sha256sum -c SHA256SUMS.txt                # six times "OK"
python -m pytest -q                        # 22 passed
python -m common.evaluate --real --name check --boot 0     # every error 0, map IoU 0.999

# put the content of globalmapper/ from the shared folder in ~/gm_files (any folder works):
mkdir -p ~/gm_files && cp /path/stage2_handover/globalmapper/* ~/gm_files/ && cd ~/gm_files
sha256sum -c /path/10623_GenAI/stage2/SHA256SUMS_globalmapper.txt     # two times "OK"
tar xf gm_nyc.tar                                                # gives the folder gm_nyc/

export GM_REPO=/path/GlobalMapper GM_CKPT=~/gm_files/val_least_loss_all.pth GM_DATA=~/gm_files/gm_nyc
cd /path/10623_GenAI/stage2
```

Set the three variables in every new terminal (or put the line in `~/.bashrc`). Run everything from the folder `stage2/`.
Start long jobs inside `tmux`, so they survive a closed laptop: `tmux new -s gm`, run the command, leave with Ctrl-b d, come
back with `tmux attach -t gm`.

`bash globalmapper_map/run.sh` without an argument only prints the list of stages. Nothing starts by accident.
The first stage you run also builds the rule baseline on `val200` (a few CPU minutes, once). Every stage prints a table at the end
or `TRAIN_DONE`; the last line of a successful stage is `GM_RUN_DONE`.

## 4. What to do, step by step

**Step 1. Preparation** (about 5 minutes): `bash globalmapper_map/run.sh PREP`

It fits the heights, encodes all blocks, and prints a table. You should see about: heights with a median error of 0.8 m;
"latent 512 dims, spread 1.15, posterior std 0.14"; 22,237 of 22,519 training blocks with a graph; and this table (James, A10G).
If one of your scores differs by more than 0.01 from the rows below, stop and write to James.

| | score | count | area | width | overlap | map IoU |
|---|---|---|---|---|---|---|
| posterior mean | 0.170 | 0.013 | 0.059 | 0.104 | 0.082 | 0.596 |
| 10 levels | 0.179 | 0.014 | 0.067 | 0.126 | 0.082 | 0.579 |
| **20 levels** (used) | **0.176** | 0.014 | 0.064 | 0.114 | 0.082 | 0.589 |
| 30 levels | 0.173 | 0.014 | 0.061 | 0.109 | 0.081 | 0.592 |
| random normal (one random draw, may differ more) | 0.621 | 2.381 | 0.220 | 0.405 | 0.103 | 0.270 |

The 20-level row is the **ceiling** of this method: even a perfect network scores about 0.17, because the decoder is frozen. Say so
when you write it up.

**Step 2. Final training, gate C**: `bash globalmapper_map/run.sh C`

- All 22,237 blocks with a graph. It stops by itself when the validation score has not improved for 8 validations (20,000 steps),
  at 200,000 steps, or at 12 h. Expected 1-6 hours on an A10G (the 15,000 steps of the pilot took 0.42 h).
- Watch it: `tail -f out/gm_full/log.csv` (column `total`, must fall) and `cat out/gm_full/val.csv` (column `score`, lower is
  better; the 2,000-block pilot reached 0.363). The pilot overfitted on 2,000 blocks, so more blocks should help, but nobody has
  run this yet.
- Interrupted? `RESUME=1 bash globalmapper_map/run.sh C` continues from the last validation.
- It is finished when the terminal shows `TRAIN_DONE` and `out/gm_full/DONE.txt` exists. If `DONE.txt` says `max_steps` or
  `max_hours`, the model may still have been improving: tell James before the next step.

**Step 3. Choose the temperature** on `val200` (a few minutes): `bash globalmapper_map/run.sh TUNE`

It prints two tables (temperatures 0.8 and 1.0; folders `results/gm_full_val200_T0.8` and `_T1.0`). Take the lower "Mean score".
If they differ by less than 0.005, take 1.0. Do not try other values: the other methods get a choice of about the same size.

**Step 4. Look at pictures.** They tell you if something is broken.

```bash
python -m common.visualize --preds rule=out/rule/val200.jsonl globalmapper=out/gm_full/val200_T1.0.jsonl --out out/fig_gm.png
```

Use `val200_T0.8.jsonl` instead if you chose 0.8.

For the figure to send back, choose blocks with `--rows <row> <row> ...` (the `row` column of `results/<name>/per_block.csv`; sort
it by `count_err` to find failures) and `--n 6` for six random ones.

Do buildings follow the map? Do they line up along the street? Do neighbours touch? Expect gaps and overlaps; the 2,000-block
pilot is shown in `../progress/stage2_design/gateB/fig1_random_blocks_a.png`.

**Step 5. Test, once** (about 15 minutes), with your temperature:

```bash
TEMP=1.0 STAGE1="nyc_c3" bash globalmapper_map/run.sh TEST
```

Two inputs (the real map, and the stage-1 maps `nyc_c3`) and one extra run without trimming (`globalmapper_oracle_raw`), which
shows how often buildings overlap. The script refuses to run a second time. `FORCE=1` is only for a run that was interrupted.

## 5. What to send back

Pack these and send them to James (shared folder):

- [ ] the table of step 1 (`out/gm_latents/quantize.txt`)
- [ ] `out/gm_full/`: `DONE.txt`, `val.csv`, `log.csv`, `config.yaml`, `best.pt`, `test1000_*.jsonl`
- [ ] `out/gm_latents/bins_20.npz` and `out/gm_heights.pt`: without them `best.pt` cannot generate buildings
- [ ] `results/`: `gm_full_val200_T0.8`, `gm_full_val200_T1.0`, `globalmapper_oracle`, `globalmapper_oracle_raw`,
      `globalmapper_nyc_c3`
- [ ] one figure with 4-6 validation blocks, including at least one failure
- [ ] the GPU you used, and anything that looked wrong
- [ ] half a page for the report: what we borrow from COHO (levels, masked prediction), what differs (the map instead of
      neighbours; one block at a time; a frozen decoder), and the ceiling

## 6. Rules (they make the comparison fair)

- Do not change `common/`, `data/` or the lists in `splits.json`, and do not retrain GlobalMapper.
- Choose settings on `val200` only; score `test1000` once.
- Keep the config as it is. Only `batch` may be lowered if memory is short; please tell us.
- Do not fit the bins or the height network on validation or test blocks (the code uses training blocks).
- Report steps, hours and GPU; do not compare methods at equal epochs.

## 7. If something goes wrong

| Symptom | Likely cause |
|---|---|
| `--gm / GM_REPO is not set` | export the three variables in the same shell |
| `run globalmapper_map.quantize first` | step 1 not done, or another `STAGE2_OUT` |
| `no such file gm_heights.pt` | run `python -m globalmapper_map.heights --fit` |
| `missing out/stage1/nyc_c3.pkl` | copy the file James sent |
| `weights_only` or pickle error when loading `GM_CKPT` | a different torch version; the code ran with torch 2.5 and 2.14 |
| out of memory | `python -m globalmapper_map.train --name gm_full --subset train --set batch=64` (add `--resume` to continue) |
| slow first validation | every block is read from its own file once, then cached |
| `score` 9.0 | no buildings were generated at all: wrong checkpoint or bins |
| anything else | send James the last 30 lines of the terminal |

## 8. Background: what is already known

Not needed to run the steps. Gate A (8 blocks): 0.177, pass. Gate B (2,000 blocks, Nov 25): `codes` 0.363, `gaussian` 0.366, rule
baseline 0.202; only the width is better than the rule, map IoU 0.39. The gaussian head clearly overfits (training MSE of the
latent falls to 0.01, validation score best at step 3,000). An earlier short test found temperature 0.5 slightly worse than 1.0.
Details: `../progress/stage2_design/gateB/RESULTS.md`.

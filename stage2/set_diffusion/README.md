# Set diffusion: a list of buildings (Daehong)

**Your goal:** train set diffusion as a **baseline** and score it on the same test blocks as the other methods. The code is
finished; you run the final training and the test, and send the results back. Up to 12 hours of GPU time (A10G), about one hour
of your own time. Nothing here depends on the language-model part of the project: the input is only the 4 m map.

**You do not need to improve the method.** A baseline must be trained properly and treated like the others, not tuned until it
wins. There is no setting to choose.

> **You can start.** The first gate B (Nov 25) diverged after 23,000 steps. The fix (normalised attention, lower learning rate,
> skipping of steps with exploding gradients) passed 60,000 steps on 2,000 blocks without diverging (score 0.175, rule baseline
> 0.202). Longer runs are not proven, so follow the stop rule in step 1.

## 1. The idea in one minute

Think of arranging furniture: start with 128 boxes at random places, and move all of them a little, many times, until they
form a real block. This is how DiffuScene (Tang et al., CVPR 2024) places furniture; a block is our room, buildings are our
furniture. Every building is one row ("token") of seven numbers, in the frame of the block (u along it, v across it):

| exist | u | v | log size along u | log size along v | angle | log height |
|---|---|---|---|---|---|---|
| +1 building, -1 empty | metres / 16 | metres / 16 | standardised | standardised | / 45 degrees | standardised |

```
4 m map, rotated to the block (12 x 64 x 64) -> CNN -> 256 map tokens + fine map features
128 noisy rows -> Transformer, 8 layers. In each layer every row looks at
     the other rows (do not overlap, line up)  and  the whole map (how many, where)
   + the map feature directly under its current position
-> slightly cleaner rows -> repeat 50 times -> rows with exist > 0 are the buildings
```

Training: take the real list, add noise, learn to remove it (v-prediction). Loss = exist + geometry of real buildings +
a penalty on overlapping boxes.

| File | Content |
|---|---|
| `tokens.py` | buildings <-> rows, training dataset, slot order |
| `model.py` | CNN + Transformer |
| `losses.py` | losses, overlap penalty |
| `sampling.py`, `generate.py` | generation, optional overlap guidance |
| `train.py`, `config.yaml` | training and all settings; change one with `--set key=value` |

## 2. What is already known

- **Order matters.** With the rows in random order (a true set, as in DiffuScene), gate A failed: score 0.170 and the number of
  buildings was wrong by 35%. With a fixed order (one side of the block first, then along the axis), a slot embedding and
  positions in units of 16 m instead of 64 m it passes with **0.072** and no count error. Two things changed at once, so we do
  not know which one helped more.
- **The ceiling.** Real val200 buildings turned into rectangles score 0.065; the share of touching buildings is already wrong
  by 0.18, because a rectangle scaled to the true area leaves small gaps. So `touch_err` near 0.18 is the best plain boxes can do.
- **Overlaps.** In the memorised blocks 1.5% of the area overlapped before trimming. Overlap guidance during sampling
  (`--guide 0.1`) lowered it to 1.1% but ruined the rest (score 0.071 -> 0.109), so it is off.
- **Speed.** 0.16 s per step with batch 64 on an A10G (0.11 s before the fix): the 12 h cap is about 270,000 steps.
- **Gate B, first version** (James, Nov 25, 2,000 blocks): best score 0.2165 at step 20,000 (rule baseline 0.202), better than the
  rule in 3 of 5 metrics. The training loss stayed at 0.41 until about step 21,500 and then rose to 2.7.
- **Gate B, after the fix** (James, Nov 25, 60,000 steps, 2.75 h): best score 0.175 at step 57,500, better than the rule in 4 of 5
  metrics, loss still falling (0.30). On the pilot data `total` fell below 0.5 after about 11,000 steps; on all
  blocks nobody has run it yet. Weak point: the share of touching buildings (error 0.316, rule 0.150).
  Details: `../progress/stage2_design/gateB/RESULTS.md`.

## 3. Setup (once)

You need this repository, `stage2/data/` (353 MB) and `stage2/out/stage1/nyc_c3.pkl` (8 MB, both from James), and Linux (or WSL)
with an NVIDIA GPU (tested on an A10G, 24 GB; smaller cards are not tested).

**Where the files from the shared folder go.** James' shared folder `stage2_handover/` has a folder `both/` for both of you.
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

```bash
git clone https://github.com/jamesbaxter0424/10623_GenAI.git
conda create -n stage2 python=3.11 -y && conda activate stage2
pip install torch                          # 2.5 or newer, matching your CUDA
cd 10623_GenAI/stage2
pip install -r requirements.txt
# copy data/ and nyc_c3.pkl as shown above, then:
sha256sum -c SHA256SUMS.txt                # six times "OK"
python -m pytest -q                        # 22 passed
python -m common.evaluate --real --name check --boot 0     # every error 0, map IoU 0.999
```

Run everything from the folder `stage2/`. Start long jobs inside `tmux`, so they survive a closed laptop: `tmux new -s sd`, run the
command, leave with Ctrl-b d, come back with `tmux attach -t sd`.

`bash set_diffusion/run.sh` without an argument only prints the list of stages. Nothing starts by accident.
The first stage you run also builds the rule baseline on `val200` (a few CPU minutes, once). Every stage prints a table at the end
or `TRAIN_DONE`; the last line of a successful stage is `SD_RUN_DONE`.

## 4. What to do, step by step

**Step 1. Final training, gate C**: `bash set_diffusion/run.sh C`

- All 22,519 blocks. It stops by itself when the validation score has not improved for 8 validations (20,000 steps), at 300,000
  steps, or at 12 h. Expected 5-12 hours on an A10G (0.16 s per step with the fix, so the
  12 h cap comes before the step cap).
- Watch it: `tail -f out/sd_full/log.csv` and `cat out/sd_full/val.csv` (column `score`, lower is better). Useful columns of the
  log: `total` (training loss), `grad` (gradient size), `skipped` (share of skipped steps, normally 0 or close to it).
- **If `total` rises to more than twice its lowest value and stays there for 5,000 steps, stop and write to James.** Do not
  change the learning rate on your own.
- Interrupted? `RESUME=1 bash set_diffusion/run.sh C` continues from the last validation.
- It is finished when the terminal shows `TRAIN_DONE` and `out/sd_full/DONE.txt` exists. If `DONE.txt` says `max_steps` or
  `max_hours`, the model may still have been improving: tell James before the next step.

**Step 2. Validation predictions** (a minute): `bash set_diffusion/run.sh VAL`

It writes `out/sd_full/val200.jsonl` and `results/sd_full_val200`. Nothing has to be chosen: the generated "exist" values are almost
all close to -1 or +1 (0.4% lie in between, 40 blocks of the pilot), so the existence threshold changes nothing.

**Step 3. Look at pictures.** They tell you if something is broken.

```bash
python -m common.visualize --preds rule=out/rule/val200.jsonl set=out/sd_full/val200.jsonl --out out/fig_sd.png
```

For the figure to send back, choose blocks with `--rows <row> <row> ...` (the `row` column of `results/<name>/per_block.csv`; sort
it by `count_err` to find failures) and `--n 6` for six random ones.

Expect clean rectangles and small gaps between attached houses; L-shaped buildings cannot be drawn by rectangles. The 2,000-block
pilot is shown in `../progress/stage2_design/gateB/fig1_random_blocks_a.png`.

**Step 4. Test, once** (a few minutes):

```bash
STAGE1="nyc_c3" bash set_diffusion/run.sh TEST
```

Two inputs (the real map, and the stage-1 maps `nyc_c3`) and one extra run without trimming (`set_oracle_raw`), which shows how
often boxes overlap. The script refuses to run a second time. `FORCE=1` is only for a run that was interrupted.

## 5. What to send back

Pack these and send them to James (shared folder):

- [ ] `out/sd_full/`: `DONE.txt`, `val.csv`, `log.csv`, `config.yaml`, `best.pt`, `test1000_*.jsonl`
- [ ] `results/`: `sd_full_val200`, `set_oracle`, `set_oracle_raw`, `set_nyc_c3`
- [ ] one figure with 4-6 validation blocks, including at least one failure with shared walls
- [ ] the GPU you used, and anything that looked wrong
- [ ] half a page for the report: what comes from DiffuScene (a list with empty entries, attention between objects) and what
      differs (map instead of text, fixed slot order, rectangles and height, no shape code)

## 6. Rules (they make the comparison fair)

- Do not change `common/`, `data/` or the lists in `splits.json`.
- Choose settings on `val200` only; score `test1000` once.
- Keep the config as it is. Only `batch` may be lowered if memory is short; please tell us.
- Report steps, hours and GPU; do not compare methods at equal epochs: use the final, early-stopped models.

## 7. If something goes wrong

| Symptom | Likely cause |
|---|---|
| out of memory | `python -m set_diffusion.train --name sd_full --subset train --set batch=32` (add `--resume` to continue) |
| GPU usage low | the data loading is too slow: add `workers=8` after `--set` if you have the cores |
| `loss is not finite`, or `skipped` above 0.05 | stop, send `out/sd_full/log.csv` to James |
| `missing out/stage1/nyc_c3.pkl` | copy the file James sent |
| `unknown setting` | typo in `--set`; the message lists all keys |
| `score` 9.0 | no buildings at all: look at `exist` in `log.csv` |
| a block has more than 128 buildings | only the 128 largest are kept (0.3% of the blocks in our sample) |
| anything else | send James the last 30 lines of the terminal |

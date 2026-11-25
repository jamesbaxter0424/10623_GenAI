# Raster diffusion (James)

A conditional diffusion model draws three 1 m images of the block; a fixed procedure turns them into buildings.

```
4 m map, target mask, distance to street  ->  rotated to the block's long axis, 1 m pixels (12 channels)
noise (3 channels) + condition  ->  U-Net, v-prediction, DDIM
   -> building / not, distance to the wall of the OWN building, height
-> seeds (distance >= core) -> watershed -> merge fragments -> polygons that share walls -> median height
```

- The condition is concatenated with the noise, like Liu's stage 1 and 2 (not a ControlNet: nothing is pretrained).
- Distance to the own wall: high in the middle of a house, low at its walls, so a shared wall is a valley. Every pixel
  belongs to one building, so buildings never overlap.
- The image is rotated to the block first. On the unrotated grid walls become staircases and widths come out 1.1 m too
  large; after rotation the bias is 0.1 m (80 validation blocks, real buildings).
- Control: `mode: regression` uses the same U-Net once, without diffusion, to test whether sampling is needed.

References: distance targets for buildings, Yuan 2016 (arXiv 1602.06564) and Yang et al. 2018 (arXiv 1805.08946); watershed
on a learned map, Bai & Urtasun, CVPR 2017; diffusion that generates distance maps of cell nuclei, NuDiff, MICCAI 2023.

## Run

```bash
bash raster_diffusion/run.sh A        # 8 blocks; passed on Nov 24 with 0.055 (need 0.10), 0.35 h on an A10G
bash raster_diffusion/run.sh B        # 2,000 blocks
bash raster_diffusion/run.sh REG      # regression control, same 2,000 blocks and steps
bash raster_diffusion/run.sh C        # full training (0.37 s/step on an A10G: 100,000 steps = 10.4 h plus validation)
CORE=1.25 STAGE1="nyc_c3" bash raster_diffusion/run.sh TEST
```

`tune_instances` prints the scores for several `core` values on val200; fix the best one before TEST.
A crashed run continues with `RESUME=1 bash raster_diffusion/run.sh C`.

## On NCSA (Delta: x86, DeltaAI: ARM + GH200)

The scripts in `deltaai/` stay on James' laptop and on NCSA (not in git: they contain account paths). `deltaai/setup_env.sh` builds the environment on a login node of either system and runs the checks (both systems, Nov 24: 20
passed, real buildings give error 0). `deltaai/run_next.sh` is what a queued Slurm job executes. Both systems share
`/work/hdd`, so a lock with a heartbeat decides who does what:

- **FULL** (first job): smoke test, 20,000 steps on all training blocks, gate B on val200, then up to 100,000 steps and
  `tune_instances`;
- **PILOT** (a second job while FULL is held or finished): gate B on `pilot2000` and the regression control.

A job stops before its time limit; submit the same job again to continue. `test1000` is not touched. The lock was tested
with a fake trainer; nothing has run on an NCSA GPU yet. If FULL is continued on another GPU type, report both GPUs and
their hours: the 12 h cap is then a mix.

Gate B is checked inside the full training here (22,519 blocks instead of 2,000), because waiting in the queue costs more
than the 20,000 steps.

## Notes

`generate` takes about 1.5 s per block and sample at 50 steps (measured while the GPU was shared), so 1,000 blocks x 4
samples take up to 1.7 h.

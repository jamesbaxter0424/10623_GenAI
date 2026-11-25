"""Train the raster diffusion model (or its one-pass regression control).

  python -m raster_diffusion.train --name rd_pilot --subset pilot2000
  python -m raster_diffusion.train --name rd_reg_pilot --subset pilot2000 --set mode=regression
  add --resume to continue after a crash
"""
import os

import torch
from torch.utils.data import DataLoader

from common import train_utils as T
from common.diffusion import Diffusion
from common.metrics import score_predictions
from common.records import load

from .model import build
from .sampling import block_seed, predict
from .targets import Blocks

HERE = os.path.dirname(os.path.abspath(__file__))


def make_loss(cfg):
    diff = Diffusion()

    def loss_fn(model, b):
        x0, w = b["x0"], b["w"]
        if cfg["mode"] == "regression":
            t = torch.zeros(len(x0), device=x0.device)
            err = (model(torch.zeros_like(x0), t, b["cond"]) - x0) ** 2
        else:
            xt, t, v = diff.training_pair(x0)
            err = (model(xt, t, b["cond"]) - v) ** 2
        e = (err * w).sum((0, 2, 3)) / w.sum((0, 2, 3))
        k = torch.tensor(cfg["channel_weights"], device=e.device, dtype=e.dtype)
        return {"total": (e * k).sum() / k.sum(), "mask": e[0], "dist": e[1], "height": e[2]}
    return loss_fn


def main():
    a, cfg = T.parse(os.path.join(HERE, "config.yaml"))
    T.seed_all(cfg["seed"])
    dev = T.device()
    tr = load(cfg["subset"], cfg["data"])
    overfit = cfg["subset"] == "overfit8"
    va = tr if overfit else load(cfg["val_subset"], cfg["data"])[:cfg["val_blocks"]]
    if overfit:
        cfg.update(noise=False, augment=False)
    ds = Blocks(tr, cfg)
    reps = max(1, cfg["batch"] * 50 // len(ds))              # small sets: avoid restarting workers each step
    ds_long = torch.utils.data.ConcatDataset([ds] * reps)
    loader = DataLoader(ds_long, cfg["batch"], shuffle=True, num_workers=cfg["workers"], drop_last=True,
                        pin_memory=dev.type == "cuda", persistent_workers=cfg["workers"] > 0,
                        worker_init_fn=T.worker_seed)

    def validate(model):
        return score_predictions(va, lambda r, m, s: predict(model, r, m, cfg, cfg["sample_steps"],
                                                             block_seed(r["row"], s), dev))
    T.fit(cfg, build(cfg), loader, make_loss(cfg), validate, a.resume)


if __name__ == "__main__":
    main()

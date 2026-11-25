"""Train the set diffusion model.

  python -m set_diffusion.train --name sd_pilot --subset pilot2000
  python -m set_diffusion.train --name sd_full --subset train
  add --resume to continue after a crash
"""
import os

import torch
from torch.utils.data import ConcatDataset, DataLoader

from common import train_utils as T
from common.metrics import score_predictions
from common.records import load

from .losses import make_loss
from .model import build
from .sampling import predict_batch
from .tokens import Blocks

HERE = os.path.dirname(os.path.abspath(__file__))


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
    reps = max(1, cfg["batch"] * 50 // len(ds))
    loader = DataLoader(ConcatDataset([ds] * reps), cfg["batch"], shuffle=True, num_workers=cfg["workers"],
                        drop_last=True, pin_memory=dev.type == "cuda", persistent_workers=cfg["workers"] > 0,
                        worker_init_fn=T.worker_seed)

    def validate(model):
        got = {}
        for i in range(0, len(va), 50):
            c = va[i:i + 50]
            for r, p in zip(c, predict_batch(model, c, [r["cls4"] for r in c], cfg, i, dev)):
                got[r["row"]] = p
        return score_predictions(va, lambda r, m, s: got[r["row"]])
    T.fit(cfg, build(cfg), loader, make_loss(cfg), validate, a.resume)


if __name__ == "__main__":
    main()

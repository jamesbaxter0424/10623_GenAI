"""Step 4: train the small network map -> latent (GlobalMapper itself is frozen).

  python -m globalmapper_map.train --name gm_pilot --subset pilot2000
  python -m globalmapper_map.train --name gm_full --subset train
  python -m globalmapper_map.train --name gm_gauss --subset pilot2000 --set head=gaussian     # fallback
  add --resume to continue after a crash
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from common import train_utils as T
from common.metrics import score_predictions
from common.noise import corrupt
from common.records import load

from .extract_latents import DIR
from .gm import Geometry, GlobalMapper, axis_points, check_args
from .heights import Heights
from .model import build
from .quantize import encode, load_bins
from .sampling import map_inputs, predict_batch

HERE = os.path.dirname(os.path.abspath(__file__))


class Blocks(Dataset):
    def __init__(self, recs, geo, latents, code, cfg):
        pos = {int(r): i for i, r in enumerate(latents["rows"])}
        self.recs = [r for r in recs if r["row"] in pos and r["row"] in geo]
        self.idx = [pos[r["row"]] for r in self.recs]
        self.mu, self.code, self.cfg = latents["mu"][self.idx], code[self.idx] if code is not None else None, cfg
        self.pts = [axis_points(r, geo.geom[r["row"]]) for r in self.recs]

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        rng = np.random.default_rng(np.random.randint(2 ** 31))
        m = corrupt(r["cls4"], r["inside"], rng) if self.cfg["noise"] else r["cls4"]
        cond, scal = map_inputs(r, m, None, self.pts[i])
        out = {"cond": torch.from_numpy(cond), "scal": torch.from_numpy(scal), "mu": torch.from_numpy(self.mu[i])}
        if self.code is not None:
            out["code"] = torch.from_numpy(self.code[i].astype(np.int64))
        return out


def main():
    extra = [(("--gm",), {"default": os.environ.get("GM_REPO")}), (("--ckpt_gm",), {"default": os.environ.get("GM_CKPT")}),
             (("--gmdata",), {"default": os.environ.get("GM_DATA")}), (("--latents",), {"default": DIR})]
    a, cfg = T.parse(os.path.join(HERE, "config.yaml"), extra)
    check_args(a)
    T.seed_all(cfg["seed"])
    dev = T.device()
    tr = load(cfg["subset"], cfg["data"])
    overfit = cfg["subset"] == "overfit8"
    va, vsplit = (tr, "train") if overfit else (load(cfg["val_subset"], cfg["data"])[:cfg["val_blocks"]], "val")
    if overfit:
        cfg.update(noise=False, dropout=0.0, temperature=0.1)
    lat = np.load(os.path.join(a.latents, "train.npz"))
    code = centre = None
    if cfg["head"] == "codes":
        if not os.path.exists(os.path.join(a.latents, f"bins_{cfg['levels']}.npz")):
            sys.exit("run globalmapper_map.quantize first")
        edges, centre = load_bins(cfg["levels"], a.latents)
        code = encode(lat["mu"], edges)
    ds = Blocks(tr, Geometry(a.gmdata, "train"), lat, code, cfg)
    print(f"{len(ds)} of {len(tr)} blocks have a GlobalMapper graph", flush=True)
    reps = max(1, cfg["batch"] * 50 // len(ds))
    loader = DataLoader(ConcatDataset([ds] * reps), cfg["batch"], shuffle=True, num_workers=cfg["workers"],
                        drop_last=True, pin_memory=dev.type == "cuda", persistent_workers=cfg["workers"] > 0,
                        worker_init_fn=T.worker_seed)
    gm, hts = GlobalMapper(a.gm, a.ckpt_gm, a.gmdata, dev), Heights()

    def validate(model):
        got = {}
        for i in range(0, len(va), 50):
            c = va[i:i + 50]
            for r, p in zip(c, predict_batch(model, gm, hts, centre, vsplit, c, [r["cls4"] for r in c], cfg, i, dev)):
                got[r["row"]] = p
        return score_predictions(va, lambda r, m, s: got[r["row"]])
    T.fit(cfg, build(cfg), loader, lambda m, b: m.loss(b), validate, a.resume)


if __name__ == "__main__":
    main()

"""Pick the seed threshold `core` on val200 (never on test). Maps are generated once and re-split.

  python -m raster_diffusion.tune_instances --ckpt out/rd_full/best.pt
  python -m raster_diffusion.tune_instances --real          # upper bound: real maps, no model
"""
import argparse

from common import config as C
from common import train_utils as T
from common.dataset import RasterFrame
from common.metrics import score_predictions
from common.records import load, polys

from .model import build
from .sampling import block_seed, generate_maps, to_buildings
from .targets import encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--cores", nargs="+", type=float, default=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    a = ap.parse_args()
    recs, dev = load("val200", a.data), T.device()
    cfg = {"dist_clip": 4.0, "height_max": 250.0, "min_area": 10.0, "simplify": 1.0}
    if not a.real:
        model, cfg = T.load_best(a.ckpt, build, dev)
    maps = {}
    for r in recs:
        f = RasterFrame(r["inside"])
        c = f.cond(r["cls4"], r["inside"])
        x = (encode(f.labels(polys(r)), r["height"], cfg) if a.real else
             generate_maps(model, c, cfg, cfg["gen_steps"], block_seed(r["row"], 0), dev))
        maps[r["row"]] = (x, c, f)
    print("core   score  count  area   width  height touch  iou")
    for core in a.cores:
        c = {**cfg, "core": core}
        predict = lambda r, _m, _s: to_buildings(*maps[r["row"]], c)  # noqa: E731
        s = score_predictions(recs, predict)
        print(f"{core:.2f}  " + "  ".join(f"{s[k]:.3f}" for k in ("score", "count_err", "wd_area", "wd_width",
                                                                  "wd_height", "touch_err", "map_iou")), flush=True)


if __name__ == "__main__":
    main()

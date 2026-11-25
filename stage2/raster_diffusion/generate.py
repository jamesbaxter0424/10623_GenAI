"""Generate buildings with a trained model.

  python -m raster_diffusion.generate --ckpt out/rd_full/best.pt --subset test1000 --samples 4 \
      --input oracle --out out/rd_full/test1000_oracle.jsonl
"""
import argparse
import time

from common import config as C
from common import train_utils as T
from common.records import Writer, input_maps, load, stage1_maps

from .model import build
from .sampling import block_seed, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subset", default="test1000")
    ap.add_argument("--input", default="oracle")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--steps", type=int, default=0, help="DDIM steps (default: gen_steps of the config)")
    ap.add_argument("--core", type=float, help="override the seed threshold (tune on val200 only)")
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = T.device()
    model, cfg = T.load_best(a.ckpt, build, dev)
    if a.core:
        cfg["core"] = a.core
    recs, maps, w, t0 = load(a.subset, a.data), stage1_maps(a.input), Writer(a.out), time.time()
    recs = recs[:a.limit] if a.limit else recs
    for i, r in enumerate(recs):
        for s, m in enumerate(input_maps(r, maps, a.samples)):
            w.add(r["row"], s, *predict(model, r, m, cfg, a.steps or cfg["gen_steps"], block_seed(r["row"], s), dev))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(recs)} blocks, {time.time() - t0:.0f} s", flush=True)
    w.close()
    print("written", a.out)


if __name__ == "__main__":
    main()

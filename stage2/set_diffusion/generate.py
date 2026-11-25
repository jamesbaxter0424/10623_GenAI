"""Generate buildings with a trained model.

  python -m set_diffusion.generate --ckpt out/sd_full/best.pt --subset test1000 --samples 4 \
      --input oracle --out out/sd_full/test1000_oracle.jsonl
  python -m set_diffusion.generate --real --subset val200 --out out/sd_real.jsonl   # upper bound: boxes of real buildings
"""
import argparse
import time

from common import config as C
from common import train_utils as T
from common.dataset import Frame
from common.records import Writer, input_maps, load, stage1_maps

from .model import build
from .sampling import predict_batch
from .tokens import boxes, decode, encode, to_polygons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--subset", default="test1000")
    ap.add_argument("--input", default="oracle")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--guide", type=float, help="overrides guide_overlap (tune on val200 only)")
    ap.add_argument("--no_trim", action="store_true", help="keep overlaps, to report them")
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs, w, t0 = load(a.subset, a.data), Writer(a.out), time.time()
    recs = recs[:a.limit] if a.limit else recs
    if a.real:
        for r in recs:
            f = Frame(r["inside"])
            w.add(r["row"], 0, *to_polygons(decode(encode(boxes(r, f), 128)), f, not a.no_trim))
    else:
        dev = T.device()
        model, cfg = T.load_best(a.ckpt, build, dev)
        if a.guide is not None:
            cfg["guide_overlap"] = a.guide
        if a.no_trim:
            cfg["trim"] = False
        maps = stage1_maps(a.input)
        for s in range(a.samples):
            for i in range(0, len(recs), 50):
                c = recs[i:i + 50]
                ms = [input_maps(r, maps, a.samples)[s] for r in c]
                for r, (g, h) in zip(c, predict_batch(model, c, ms, cfg, 1000 * s + i, dev, a.steps or None)):
                    w.add(r["row"], s, g, h)
            print(f"sample {s + 1}/{a.samples}, {time.time() - t0:.0f} s", flush=True)
    w.close()
    print("written", a.out)


if __name__ == "__main__":
    main()

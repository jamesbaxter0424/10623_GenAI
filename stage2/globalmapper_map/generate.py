"""Generate buildings with a trained model.

  python -m globalmapper_map.generate --ckpt out/gm_full/best.pt --subset test1000 --samples 4 \
      --input oracle --out out/gm_full/test1000_oracle.jsonl
"""
import argparse
import time

from common import config as C
from common import train_utils as T
from common.records import Writer, input_maps, load, stage1_maps

from .extract_latents import DIR
from .gm import GlobalMapper, add_args, check_args
from .heights import Heights
from .model import build
from .quantize import load_bins
from .sampling import predict_batch


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--ckpt", required=True, help="best.pt of globalmapper_map.train")
    ap.add_argument("--subset", default="test1000")
    ap.add_argument("--input", default="oracle")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, help="tune on val200 only")
    ap.add_argument("--no_trim", action="store_true", help="keep overlaps, to report them")
    ap.add_argument("--latents", default=DIR)
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    check_args(a)
    dev = T.device()
    model, cfg = T.load_best(a.ckpt, build, dev)
    if a.temperature:
        cfg["temperature"] = a.temperature
    if a.no_trim:
        cfg["trim"] = False
    centre = load_bins(cfg["levels"], a.latents)[1] if cfg["head"] == "codes" else None
    gm, hts = GlobalMapper(a.gm, a.ckpt_gm, a.gmdata, dev), Heights()
    split = "val" if a.subset.startswith("val") else "test" if a.subset.startswith("test") else "train"
    recs, maps, w, t0 = load(a.subset, a.data), stage1_maps(a.input), Writer(a.out), time.time()
    recs = recs[:a.limit] if a.limit else recs
    miss = sum(r["row"] not in gm.geometry(split) for r in recs)
    for s in range(a.samples):
        for i in range(0, len(recs), 50):
            c = recs[i:i + 50]
            ms = [input_maps(r, maps, a.samples)[s] for r in c]
            for r, (g, h) in zip(c, predict_batch(model, gm, hts, centre, split, c, ms, cfg, 1000 * s + i, dev)):
                w.add(r["row"], s, g, h)
        print(f"sample {s + 1}/{a.samples}, {time.time() - t0:.0f} s", flush=True)
    w.close()
    print(f"written {a.out}; {miss} of {len(recs)} blocks have no GlobalMapper graph (written as empty)")


if __name__ == "__main__":
    main()

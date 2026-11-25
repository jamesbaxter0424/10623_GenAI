"""Step 3: cut every latent dimension into equal-count bins (COHO: 20 levels) and check that decoding
still works. Run this BEFORE training: if 20 levels hurt, use more levels or head: gaussian.

  python -m globalmapper_map.quantize --levels 10 20 30

Writes out/gm_latents/bins_<levels>.npz (edges, centres; from TRAINING latents only) and prints the
scores of real val200 blocks that were encoded and decoded:
  posterior mean, no quantisation   = the best this decoder can do ("recon")
  quantised with L levels           = the ceiling of head: codes
  random N(mean, std)               = what the release does (our GlobalMapper "random" baseline)
"""
import argparse
import os

import numpy as np

from common import config as C
from common import train_utils as T
from common.metrics import score_predictions
from common.records import load

from .extract_latents import DIR
from .gm import GlobalMapper, add_args, check_args
from .heights import Heights

SHOW = ("score", "count_err", "wd_area", "wd_width", "wd_height", "touch_err", "overlap", "coverage_err", "map_iou")


def fit_bins(mu, levels):
    """edges [levels - 1, D] and centres [levels, D] (mean of the training values in each bin)."""
    q = np.quantile(mu, np.linspace(0, 1, levels + 1)[1:-1], axis=0)
    code = encode(mu, q)
    centre = np.zeros((levels, mu.shape[1]), np.float32)
    for k in range(levels):
        m = code == k
        n = m.sum(0)
        centre[k] = np.where(n > 0, (mu * m).sum(0) / np.maximum(n, 1), 0)
        empty = n == 0
        if empty.any():                                      # constant dimension
            centre[k, empty] = mu[:, empty].mean(0)
    return q.astype(np.float32), centre


def encode(mu, edges):
    """[n, D] -> integer codes 0..levels-1."""
    return np.stack([np.searchsorted(edges[:, d], mu[:, d], "right") for d in range(mu.shape[1])], 1)


def decode(code, centre):
    return centre[code, np.arange(code.shape[1])]


def load_bins(levels, d=DIR):
    k = np.load(os.path.join(d, f"bins_{levels}.npz"))
    return k["edges"], k["centre"]


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--levels", nargs="+", type=int, default=[10, 20, 30])
    ap.add_argument("--latents", default=DIR)
    a = ap.parse_args()
    check_args(a)
    tr = np.load(os.path.join(a.latents, "train.npz"))
    va = np.load(os.path.join(a.latents, "val.npz"))
    gm, hts = GlobalMapper(a.gm, a.ckpt_gm, a.gmdata, T.device()), Heights()
    pos = {int(r): i for i, r in enumerate(va["rows"])}
    recs = [r for r in load("val200", a.data) if r["row"] in pos]
    mu = va["mu"][[pos[r["row"]] for r in recs]]
    rows = [r["row"] for r in recs]
    rng = np.random.default_rng(0)
    print(f"{len(recs)} of 200 validation blocks have a graph")

    def report(name, z):
        slots = gm.decode("val", rows, z)
        got = {}
        for r, s in zip(recs, slots):
            g = gm.polygons("val", r, s, trim=False)
            got[r["row"]] = (g, hts.sample(g, r["cls4"], r["inside"], np.random.default_rng(r["row"])))
        s = score_predictions(recs, lambda r, m, k: got[r["row"]])
        print(f"{name:28s}" + " ".join(f"{k} {s[k]:.3f}" for k in SHOW), flush=True)

    report("posterior mean", mu)
    for L in a.levels:
        edges, centre = fit_bins(tr["mu"], L)
        np.savez(os.path.join(a.latents, f"bins_{L}.npz"), edges=edges, centre=centre)
        report(f"{L} levels", decode(encode(mu, edges), centre))
    report("random N(mean, std)", rng.normal(tr["mu"].mean(0), tr["mu"].std(0), mu.shape))


if __name__ == "__main__":
    main()

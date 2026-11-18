"""Quick metrics on generated NYC windows (midway check, not the official evaluation).

  python tools/nyc_quick_eval.py DIR/Generated [DIR2/Generated ...]
All metrics are computed inside the target mask (the 1-4 masked blocks) of each window:
  cell_acc        share of cells whose class equals the real map
  bldg_acc        same, counting only building (2-9) vs non-building
  cov_err         |generated - real| building coverage (share of building cells)
  hclass_acc      dominant height class of building cells matches the real map
  use_acc         dominant use (res / non-res) of building cells matches the real map
  diversity       share of cells that differ between two samples of the same input
                  (mean over all pairs of the 4 samples)
  street_in_mask  share of mask cells generated as street (real: 0 by construction)
Means with 95% bootstrap intervals over windows.
"""
import glob
import itertools
import os
import sys

import numpy as np


def dominant(cls):
    b = cls[(cls >= 2)]
    if len(b) == 0:
        return -1, -1
    h = np.bincount((b - 2) % 4, minlength=4).argmax()
    use = int((b <= 5).mean() >= 0.5)          # 1 = residential
    return h, use


def window_metrics(gen, gt, mask):
    g, r = gen[:, mask], gt[mask]                  # [S, n], [n]
    out = {
        "cell_acc": (g == r).mean(),
        "bldg_acc": ((g >= 2) == (r >= 2)).mean(),
        "cov_err": np.abs((g >= 2).mean(1) - (r >= 2).mean()).mean(),
        "street_in_mask": (g == 0).mean(),
    }
    rh, ru = dominant(r)
    if rh >= 0:
        dom = [dominant(s) for s in g]
        out["hclass_acc"] = np.mean([d[0] == rh for d in dom])
        out["use_acc"] = np.mean([d[1] == ru for d in dom])
    pairs = list(itertools.combinations(range(len(g)), 2))
    if pairs:
        out["diversity"] = np.mean([(g[a] != g[b]).mean() for a, b in pairs])
    return out


def ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n_boot, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def evaluate(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        rows.append(window_metrics(z["gen"].astype(np.int64), z["gt"].astype(np.int64), z["mask"]))
    keys = ["cell_acc", "bldg_acc", "cov_err", "hclass_acc", "use_acc", "diversity", "street_in_mask"]
    print(f"\n{d}: {len(rows)} windows")
    for k in keys:
        v = [r[k] for r in rows if k in r]
        if v:
            m, lo, hi = ci(v)
            print(f"  {k:15s} {m:.3f}  [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        evaluate(d)

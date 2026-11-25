"""Step 1: encode every block with the trained GlobalMapper.

  python -m globalmapper_map.extract_latents            # uses GM_REPO, GM_CKPT, GM_DATA
Writes out/gm_latents/{train,val,test}.npz with rows, mu, logvar.
"""
import argparse
import os

import numpy as np

from common import config as C
from common import train_utils as T
from common.records import _split

from .gm import GlobalMapper, add_args, check_args

DIR = os.path.join(C.OUT, "gm_latents")


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--out", default=DIR)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    check_args(a)
    gm = GlobalMapper(a.gm, a.ckpt_gm, a.gmdata, T.device())
    os.makedirs(a.out, exist_ok=True)
    for split in a.splits:
        if not os.path.isdir(os.path.join(a.gmdata, split)):
            print(split, "not in", a.gmdata, "- skipped")
            continue
        geo, recs = gm.geometry(split), _split(a.data, split)
        rows = [r for r in sorted(recs) if r in geo]
        mu, lv = [], []
        for i in range(0, len(rows), 200):
            m, v = gm.encode(split, rows[i:i + 200])
            mu.append(m)
            lv.append(v)
            gm._data.clear()
        mu, lv = np.concatenate(mu), np.concatenate(lv)
        np.savez(os.path.join(a.out, f"{split}.npz"), rows=np.array(rows), mu=mu, logvar=lv)
        print(f"{split}: {len(rows)} of {len(recs)} blocks have a GlobalMapper graph; latent {mu.shape[1]} dims, "
              f"spread {mu.std(0).mean():.2f}, posterior std {np.exp(lv / 2).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()

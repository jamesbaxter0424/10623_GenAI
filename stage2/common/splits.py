"""Named block lists, written once to <data>/splits.json so that everybody uses the same blocks.

  overfit8    gate A: 8 training blocks with 10-60 buildings
  pilot2000   gate B: 2,000 random training blocks
  train       gate C: every valid training block
  val200      200 validation blocks: the ONLY blocks used to pick checkpoints and settings
  test1000    final test: target blocks of the first stage-1 test windows (same window order as
              evaluation/), so stage-1 generated maps exist for them. Score it once, at the end.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd

from . import config as C


def build(data=C.DATA, city=C.CITY):
    recs = {}
    for s in ("train", "val", "test"):
        with open(os.path.join(data, f"{s}.pkl"), "rb") as f:
            recs[s] = pickle.load(f)
    rng = np.random.default_rng(C.SEED)

    def pick(rows, n):
        return sorted(int(x) for x in rng.choice(rows, min(n, len(rows)), replace=False))

    tr = [q["row"] for q in recs["train"]]
    mid = [q["row"] for q in recs["train"] if 10 <= len(q["polys"]) <= 60]
    val = [q["row"] for q in recs["val"] if len(q["polys"])]
    s = pd.read_parquet(os.path.join(city, "samples.parquet"))
    s = s[(s["split"] == "test") & (~s["ctx_other_split"])]
    order = np.random.default_rng(0).permutation(s["sample_id"].to_numpy())   # = evaluation/data.py test_order
    s = s.set_index("sample_id")
    have, test, window = {q["row"] for q in recs["test"]}, [], {}
    for sid in order[:1000]:                                 # the 1,000 windows stage 1 generated
        for t in s.loc[sid, "targets"]:
            t = int(t)
            if t in have and t not in window:
                window[t] = int(sid)
                test.append(t)
    if len(test) < 20:                                       # only with --limit data
        test = sorted(have)
    test = test[:C.N_TEST]
    out = {"overfit8": pick(mid, 8), "pilot2000": pick(tr, C.N_PILOT), "train": tr,
           "val200": pick(val, C.N_VAL), "test1000": test,
           "test_window": {str(k): window[k] for k in test if k in window}}
    json.dump(out, open(os.path.join(data, "splits.json"), "w"))
    print("splits:", {k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    build()

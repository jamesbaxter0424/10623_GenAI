"""Stage-1 generated maps -> stage-2 input file.

  python -m common.stage1_input --pred_dir ~/genai-final/generated/nyc_c3_test1000 --out out/stage1/nyc_c3.pkl

--pred_dir holds (possibly in a sub-folder "Generated") <sample_id>.npz files with gen [S, 128, 128],
as written by control-3d-scene/generate.py. For every block of test1000 the target block is cut out
of the first window that contains it; cells outside that window come from the real map.
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from . import config as C
from .records import load, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--city", default=C.CITY)
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--samples", type=int, default=4)
    a = ap.parse_args()
    d = os.path.join(a.pred_dir, "Generated")
    d = d if os.path.isdir(d) else a.pred_dir
    win = names(a.data)["test_window"]
    s = pd.read_parquet(os.path.join(a.city, "samples.parquet"), columns=["sample_id", "row0", "col0"]).set_index("sample_id")
    out, cut = {}, 0
    for r in load("test1000", a.data):
        sid = win[str(r["row"])]
        with np.load(os.path.join(d, f"{sid}.npz")) as z:
            gen = z["gen"][:a.samples]
        if gen.ndim == 4:                                    # [S, 128, 128, 1]
            gen = gen[..., 0]
        r0, c0 = int(s.loc[sid, "row0"]), int(s.loc[sid, "col0"])
        h, w = r["cls4"].shape
        m = np.repeat(r["cls4"][None], len(gen), 0)
        a0, a1 = max(r["r0"], r0), min(r["r0"] + h, r0 + 128)
        b0, b1 = max(r["c0"], c0), min(r["c0"] + w, c0 + 128)
        sub = m[:, a0 - r["r0"]:a1 - r["r0"], b0 - r["c0"]:b1 - r["c0"]]
        g = gen[:, a0 - r0:a1 - r0, b0 - c0:b1 - c0]
        ins = r["inside"][a0 - r["r0"]:a1 - r["r0"], b0 - r["c0"]:b1 - r["c0"]]
        sub[:, ins] = g[:, ins]                              # only the target block is generated
        cut += int(r["inside"].sum() > ins.sum())
        out[r["row"]] = m.astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"{len(out)} blocks, {cut} of them partly outside their window -> {a.out}")


if __name__ == "__main__":
    main()

"""Real buildings -> one record per city block (the only data the three experiments read).

  cd stage2 && python -m common.make_data                  # all splits
  python -m common.make_data --limit 50 --out data_small   # quick test

Writes <out>/{train,val,test}.pkl (lists of records), info.json and splits.json. A record is a dict:
  row        row of blocks.parquet (the block id used everywhere in this project)
  block_id, boro
  r0, c0     top-left cell of the window in the city grid (4 m cells)
  cls4       uint8 [h, w]  10-class map: the input (real map of the target block AND its surroundings)
  inside     bool  [h, w]  cells of the target block
  polys      list of WKB   building polygons in local coordinates (see geometry.py)
  height     float32 [n]   metres
  cls        uint8 [n]     class 2..9 of each building
  lab        bytes         zlib-compressed uint16 instance labels at 1 m (read with records.labels)
Only valid blocks; train / val / test are the stage-1 splits (by community district).
"""
import argparse
import glob
import json
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import shapely

from . import config as C
from .geometry import clean, rasterize


def load_buildings(root, boro):
    """<root>/<boro>/buildings_final.parquet (laptop) or any *<boro>*.parquet below <root> (AWS)."""
    paths = glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
    hits = [p for p in paths if os.path.basename(p) == "buildings_final.parquet"
            and os.path.basename(os.path.dirname(p)) == boro]
    hits = hits or [p for p in paths if boro in os.path.splitext(os.path.basename(p))[0].split("_")]
    if not hits:
        raise FileNotFoundError(f"no building table for {boro} under {root}")
    return pd.read_parquet(hits[0], columns=["block_id", "geometry", "height_m", "floors", "hclass", "is_res"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=C.CITY)
    ap.add_argument("--bldg", default=C.BLDG)
    ap.add_argument("--out", default=C.DATA)
    ap.add_argument("--limit", type=int, default=0, help="blocks per split, for a quick test")
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    grid = json.load(open(os.path.join(a.city, "grid.json")))
    cls = np.load(os.path.join(a.city, "cls.npy"), mmap_mode="r")
    blk = np.load(os.path.join(a.city, "block.npy"), mmap_mode="r")
    bl = pd.read_parquet(os.path.join(a.city, "blocks.parquet"),
                         columns=["block_id", "geometry", "boro", "split", "valid"])
    bl["row"] = np.arange(len(bl))
    bl = bl[bl["valid"]]
    if a.limit:
        bl = pd.concat([g.sample(min(a.limit, len(g)), random_state=a.seed) for _, g in bl.groupby("split")])

    recs, dropped = {s: [] for s in ("train", "val", "test")}, 0
    for boro, bb in bl.groupby("boro", observed=True):
        bd = load_buildings(a.bldg, str(boro))
        bd = bd[bd["block_id"].isin(bb["block_id"])]
        groups = {k: g for k, g in bd.groupby("block_id")}
        for r in bb.itertuples():
            g = groups.get(r.block_id)
            geoms, hs, cs = [], [], []
            if g is not None:
                for wkb, h, fl, hc, res in zip(g["geometry"], g["height_m"], g["floors"], g["hclass"], g["is_res"]):
                    p = clean(shapely.from_wkb(wkb))
                    if p is None or p.area < C.MIN_BLDG_M2:
                        dropped += 1
                        continue
                    if not np.isfinite(h) or h < C.H_MIN or h > C.H_MAX:
                        h = max(float(fl), 1.0) * 3.5
                    geoms.append(p)
                    hs.append(float(np.clip(h, C.H_MIN, C.H_MAX)))
                    cs.append((2 if (pd.notna(res) and bool(res)) else 6) + int(hc))
            b = shapely.from_wkb(r.geometry)
            x0, y0, x1, y1 = shapely.total_bounds(np.array([b] + geoms, dtype=object))
            c0 = int(np.floor((x0 - grid["x0"]) / C.CELL)) - C.MARGIN
            c1 = int(np.ceil((x1 - grid["x0"]) / C.CELL)) + C.MARGIN
            r0 = int(np.floor((grid["y_top"] - y1) / C.CELL)) - C.MARGIN
            r1 = int(np.ceil((grid["y_top"] - y0) / C.CELL)) + C.MARGIN
            r0, c0, r1, c1 = max(r0, 0), max(c0, 0), min(r1, cls.shape[0]), min(c1, cls.shape[1])
            ox, oy = grid["x0"] + c0 * C.CELL, grid["y_top"] - r0 * C.CELL
            loc = [shapely.transform(p, lambda q: np.stack([q[:, 0] - ox, oy - q[:, 1]], 1)) for p in geoms]
            lab = rasterize(loc, ((r1 - r0) * C.F, (c1 - c0) * C.F)).astype(np.uint16)
            inside = np.asarray(blk[r0:r1, c0:c1]) == r.row
            if not inside.any():
                continue
            recs[r.split].append({
                "row": int(r.row), "block_id": r.block_id, "boro": str(boro), "r0": r0, "c0": c0,
                "cls4": np.asarray(cls[r0:r1, c0:c1]).astype(np.uint8), "inside": inside,
                "polys": [shapely.to_wkb(p) for p in loc], "height": np.array(hs, np.float32),
                "cls": np.array(cs, np.uint8), "lab": zlib.compress(lab.tobytes(), 6)})
        print(boro, {s: len(v) for s, v in recs.items()}, flush=True)
    for s, v in recs.items():
        v.sort(key=lambda q: q["row"])
        with open(os.path.join(a.out, f"{s}.pkl"), "wb") as f:
            pickle.dump(v, f, protocol=4)
    info = {"grid": grid, "px": C.PX, "cell": C.CELL, "limit": a.limit, "dropped_small": dropped,
            "blocks": {s: len(v) for s, v in recs.items()},
            "buildings": {s: int(sum(len(q["polys"]) for q in v)) for s, v in recs.items()}}
    json.dump(info, open(os.path.join(a.out, "info.json"), "w"), indent=1)
    print(info["blocks"], info["buildings"], "dropped (< 10 m2 or invalid)", dropped)
    from .splits import build
    build(a.out, a.city)


if __name__ == "__main__":
    main()

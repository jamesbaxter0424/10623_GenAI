"""Rule baseline: split the 4 m map into buildings with statistics of the TRAINING buildings. No learning.

  python -m common.rule_split --fit                                         # once: out/rule_table.pkl
  python -m common.rule_split --subset test1000 --samples 4 --out out/rule/test1000_oracle.jsonl
  python -m common.rule_split --subset test1000 --samples 4 --input out/stage1/nyc_c3.pkl --out ...

Steps for each region (connected cells of one class inside the block):
  1. draw how many buildings it holds, from training regions of the same class and similar area;
  2. cut it into 1-3 rows by depth, and each row into strips along its long side; strip widths are
     drawn from the widths of training buildings of that class and scaled to fill the row;
  3. each strip becomes a rectangle with the strip's width and the area of the covered cells
     (a single large building keeps its outline);
  4. heights: training heights of the class; touching houses of one region share the same height
     with the probability measured in training.
"""
import argparse
import os
import pickle

import numpy as np
import shapely
import shapely.affinity
from scipy import ndimage as ndi

from . import config as C
from .geometry import block_frame, cells_polygon, clean, obb, rect, trim_overlaps, up
from .metrics import touching
from .records import Writer, input_maps, labels, load, polys, stage1_maps

TABLE = os.path.join(C.OUT, "rule_table.pkl")
SQUARE = 1.25                                                # area bins: powers of this many cells


def regions(cls4, inside):
    """[(class, mask)] connected cells of one class inside the block."""
    out = []
    for v in np.unique(cls4[inside & (cls4 >= 2)]):
        lab, n = ndi.label(inside & (cls4 == v))
        out += [(int(v), lab == k) for k in range(1, n + 1)]
    return out


def abin(cells):
    return int(np.floor(np.log(max(cells, 1)) / np.log(SQUARE)))


def fit(recs, path=TABLE, seed=0):
    rng = np.random.default_rng(seed)
    counts, width, length, height, same = {}, {}, {}, {}, {}
    for n_rec, r in enumerate(recs):
        regs = regions(r["cls4"], r["inside"])
        if not regs:
            continue
        ids = np.zeros(r["cls4"].shape, np.int32)
        for i, (_, m) in enumerate(regs, 1):
            ids[m] = i
        lab = labels(r)
        px = up(ids)
        num = np.zeros(len(regs) + 1, int)
        k = lab > 0
        pair = np.zeros((lab.max() + 1, len(regs) + 1), np.int32)
        np.add.at(pair, (lab[k], px[k]), 1)
        for b in range(1, len(pair)):
            if pair[b, 1:].any():
                num[pair[b, 1:].argmax() + 1] += 1
        for (v, m), n in zip(regs, num[1:]):
            counts.setdefault((v, abin(m.sum())), []).append(int(n))
        g = polys(r)
        for p, v, h in zip(g, r["cls"], r["height"]):
            o = obb(p)
            width.setdefault(int(v), []).append(o[3])
            length.setdefault(int(v), []).append(o[2])
            height.setdefault(int(v), []).append(float(h))
        if n_rec % 4 == 0:
            for i, j in touching(g):
                if r["cls"][i] == r["cls"][j]:
                    same.setdefault(int(r["cls"][i]), []).append(abs(r["height"][i] - r["height"][j]) < 0.5)
        if n_rec % 4000 == 0:
            print(n_rec, len(recs), flush=True)
    cut = lambda x: np.array(x, np.float32) if len(x) <= 20000 else rng.choice(np.array(x, np.float32), 20000)
    tab = {"counts": {k: np.array(v) for k, v in counts.items()},
           "width": {k: cut(v) for k, v in width.items()}, "height": {k: cut(v) for k, v in height.items()},
           "depth": {k: float(np.median(v)) for k, v in length.items()},
           "same": {k: float(np.mean(v)) for k, v in same.items()}, "blocks": len(recs)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tab, f)
    print("depth", {k: round(v, 1) for k, v in tab["depth"].items()})
    print("share of touching same-class pairs with equal height", {k: round(v, 2) for k, v in tab["same"].items()})


def load_table(path=TABLE):
    with open(path, "rb") as f:
        return pickle.load(f)


def draw_count(tab, v, cells, rng):
    b = abin(cells)
    for d in (0, 1, -1, 2, -2, 3, -3):                       # nearest filled bin of the same class
        x = tab["counts"].get((v, b + d))
        if x is not None and len(x) >= 5:
            return max(int(rng.choice(x)), 1)
    return max(int(round(cells * C.CELL ** 2 / 300)), 1)


def split_region(tab, v, mask, theta, rng):
    """One region -> list of polygons."""
    n = draw_count(tab, v, mask.sum(), rng)
    geo = cells_polygon(mask)
    ctr = geo.centroid
    g = shapely.affinity.rotate(geo, -theta, origin=ctr, use_radians=True)
    x0, y0, x1, y1 = g.bounds
    t = theta
    if (y1 - y0) > (x1 - x0):                                # rows run along the longer side
        g = shapely.affinity.rotate(g, -90, origin=ctr)
        t = theta + np.pi / 2
        x0, y0, x1, y1 = g.bounds
    if n == 1:
        big = clean(geo)
        if big.area / max(obb(big)[2] * obb(big)[3], 1) < 0.75:
            return [clean(shapely.simplify(big, 3.0)) or big]
    depth = tab["depth"].get(v, 15.0)
    rows = int(np.clip(round((y1 - y0) / (1.2 * depth)), 1, min(3, n))) if n > 1 else 1
    out = []
    edges = np.linspace(y0, y1, rows + 1)
    parts = [g.intersection(shapely.box(x0, edges[k], x1, edges[k + 1])) for k in range(rows)]
    area = np.array([p.area for p in parts])
    num = np.maximum(np.round(n * area / area.sum()).astype(int), 1)
    for p, k in zip(parts, num):
        if p.area < C.MIN_BLDG_M2:
            continue
        a0, _, a1, _ = p.bounds
        w = rng.choice(tab["width"].get(v, np.array([6.3])), k)
        xs = a0 + np.concatenate([[0], np.cumsum(w / w.sum())]) * (a1 - a0)
        for a, b in zip(xs[:-1], xs[1:]):
            q = p.intersection(shapely.box(a, y0, b, y1))
            if q.area < C.MIN_BLDG_M2:
                continue
            box = rect((a + b) / 2, q.centroid.y, b - a, q.area / (b - a), 0.0)
            out.append(shapely.affinity.rotate(box, t, origin=ctr, use_radians=True))
    return out


def predict(tab, rec, cls4, seed):
    rng = np.random.default_rng([seed, rec["row"]])
    theta = block_frame(rec["inside"])[4]
    g, h = [], []
    for v, m in regions(cls4, rec["inside"]):
        ps = split_region(tab, v, m, theta, rng)
        pool = tab["height"].get(v, np.array([7.0]))
        p_same = tab["same"].get(v, 0.5)
        base = float(rng.choice(pool))
        for p in ps:
            if rng.random() > p_same:
                base = float(rng.choice(pool))
            g.append(p)
            h.append(base)
    keep = [(a, b) for a, b in zip(trim_overlaps(g), h) if a is not None]
    return [a for a, _ in keep], np.array([b for _, b in keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--subset", default="test1000")
    ap.add_argument("--input", default="oracle")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.fit:
        fit(load("train", a.data), a.table)
    if a.out:
        tab, maps, w = load_table(a.table), stage1_maps(a.input), Writer(a.out)
        for r in load(a.subset, a.data):
            for s, m in enumerate(input_maps(r, maps, a.samples)):
                w.add(r["row"], s, *predict(tab, r, m, s))
        w.close()
        print("written", a.out)


if __name__ == "__main__":
    main()

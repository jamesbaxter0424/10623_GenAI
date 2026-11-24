"""Convert NYC blocks and buildings to GlobalMapper's processed graph format.

  python prep_nyc.py --city ~/genai-final/data/city --bldg ~/genai-final/data/nyc_bldg \
      --gm ~/genai-final/third_party/GlobalMapper --out ~/genai-final/data/gm_nyc --workers 4

Output, one folder per split (train / val / test):
  <out>/<split>/processed/<k>.gpickle   graphs in the official format (k = 0..n-1)
  <out>/<split>/index.csv               k -> block row (the id used in block.npy / samples.parquet)
  <out>/<split>/geom.pkl                per block: canonical-transform data needed to map
                                        generated graphs back to the map (midaxis, aspect ratio)
  <out>/norm.json                       mean / std used to standardise pos and size (train only)

What comes from the release, and what had to be re-implemented (the release has no script
between raw_geo/ and processed/; see reverse_format.py and the README):
  - canonical transform: example_canonical_transform.py (straight skeleton -> extended main axis
    -> warp_bldg_by_midaxis), with the missing get_k_angle supplied. Block outlines are
    simplified by 0.5 m first (the unsimplified skeleton takes ~40 s per block).
  - slots (re-implemented): row 0 = buildings on the negative side of the main axis, row 1 = the
    positive side, rows 2 and 3 = the inner half of a side that has more than 30 buildings.
    Within a row, buildings are ordered along the axis and put in the column nearest their
    position, shifted right on collisions, which keeps columns aligned across rows as in the
    released sample.
  - standardisation (re-implemented as described in the release's comment): pos and size of
    existing buildings minus the training mean, divided by the training std, clipped to +-5.
  - shape class and occupancy (re-implemented; the fitting code is not released):
    iou = building area / area of its minimum rotated rectangle; shape = 0 if iou >= 0.9 or no
    reflex corner, else 1 / 2 / 3 / 4 / 5 for 1 / 2 / 4 / 3 / >= 5 reflex corners of the 1 m
    simplified outline (6 classes as in the model).
  - binary_mask: block rotated so its long side is horizontal, scaled to fit 64 x 64, filled 255;
    block_scale = long side of the block's minimum rotated rectangle in metres.
Blocks with 0 buildings are kept; blocks with more than 120 buildings are dropped.
"""
import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
import shapely
import shapely.affinity as sa
from shapely import wkb
from shapely.geometry import Polygon

W, H = 30, 4
N = W * H
ROWS_PER_SIDE = 2
GM = None


def init(gm_path):
    global GM
    sys.path.insert(0, gm_path)
    import example_canonical_transform as gm
    gm.get_k_angle = lambda p1, p2: np.arctan2(p2.y - p1.y, p2.x - p1.x)
    gm.print = lambda *a, **k: None
    GM = gm


def largest(poly):
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    return Polygon(poly.exterior)          # GlobalMapper cannot represent interior boundaries


def straight_axis(blk):
    """Centre line of the block's minimum rotated rectangle along its long side."""
    from shapely.geometry import LineString
    xs, ys = blk.minimum_rotated_rectangle.exterior.coords.xy
    p = np.array(list(zip(xs, ys)))[:4]
    if np.linalg.norm(p[1] - p[0]) >= np.linalg.norm(p[2] - p[1]):
        a, b = (p[0] + p[3]) / 2, (p[1] + p[2]) / 2
    else:
        a, b = (p[0] + p[1]) / 2, (p[3] + p[2]) / 2
    return LineString([tuple(a), tuple(b)])


def canonical(block, bldgs):
    import skgeom
    c = block.centroid
    blk = sa.translate(block, -c.x, -c.y)
    bl = [sa.translate(b, -c.x, -c.y) for b in bldgs]
    ext = list(blk.exterior.coords)[:-1]
    if not Polygon(ext).exterior.is_ccw:   # the skeleton needs a counter-clockwise ring
        ext.reverse()
    sk = skgeom.Polygon(ext)
    skel = skgeom.skeleton.create_interior_straight_skeleton(sk)
    _, longest = GM.get_polyskeleton_longest_path(skel, sk)
    med = GM.modified_skel_to_medaxis(longest, blk)
    if med.geom_type != "LineString":       # the extension did not merge into one line (~1% of
        med = straight_axis(blk)             # blocks): fall back to the long axis of the block
    pos, size, order, asp = GM.warp_bldg_by_midaxis(bl, blk, med)
    pos = np.asarray(pos, float).reshape(-1, 2)
    size = np.asarray(size, float).reshape(-1, 2)
    return pos, size, np.asarray(order, int), float(asp), med, (c.x, c.y)


def reflex_corners(poly):
    p = poly.simplify(1.0)
    if p.is_empty or p.geom_type != "Polygon":
        return 0
    pts = np.array(p.exterior.coords)[:-1]
    if len(pts) < 4:
        return 0
    ccw = Polygon(pts).exterior.is_ccw
    n = 0
    for i in range(len(pts)):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if (cross < 0) == ccw and abs(cross) > 1e-6:
            n += 1
    return n


def shape_iou(poly):
    mrr = poly.minimum_rotated_rectangle
    iou = float(poly.area / mrr.area) if mrr.area > 0 else 1.0
    r = reflex_corners(poly)
    if iou >= 0.9 or r == 0:
        s = 0
    else:
        s = {1: 1, 2: 2, 4: 3, 3: 4}.get(r, 5)
    return s, min(iou, 1.0)


def assign_slots(pos):
    """pos: [n,2] canonical positions. Returns slot index per building, or None if it does not fit."""
    n = len(pos)
    slot = np.full(n, -1)
    for side, row0 in ((pos[:, 1] < 0, 0), (pos[:, 1] >= 0, 1)):
        idx = np.nonzero(side)[0]
        if len(idx) == 0:
            continue
        groups = [idx]
        if len(idx) > W:                       # split the side into an outer and an inner row
            far = np.abs(pos[idx, 1])
            cut = np.sort(far)[::-1][W - 1] if len(idx) > 2 * W else np.median(far)
            outer = idx[far >= cut][:W] if len(idx) > 2 * W else idx[far >= cut]
            inner = np.setdiff1d(idx, outer)
            groups = [outer, inner]
        for gi, g in enumerate(groups):
            if len(g) > W:
                return None
            row = row0 + 2 * gi
            g = g[np.argsort(pos[g, 0], kind="stable")]
            target = np.clip(np.round((pos[g, 0] + 1) / 2 * (W - 1)), 0, W - 1).astype(int)
            cols = np.zeros(len(g), int)
            prev = -1
            for k, t in enumerate(target):
                cols[k] = max(t, prev + 1)
                prev = cols[k]
            over = cols[-1] - (W - 1)
            if over > 0:                        # push back from the right end
                for k in range(len(g) - 1, -1, -1):
                    lim = (W - 1) - (len(g) - 1 - k)
                    cols[k] = min(cols[k], lim)
            slot[g] = row * W + cols
    if len(set(slot.tolist())) != n:
        return None
    return slot


def block_mask(block):
    mrr = block.minimum_rotated_rectangle
    xs, ys = mrr.exterior.coords.xy
    e = [(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(2)]
    lng = max(e, key=lambda v: np.hypot(*v))
    ang = np.degrees(np.arctan2(lng[1], lng[0]))
    c = block.centroid
    b = sa.rotate(sa.translate(block, -c.x, -c.y), -ang, origin=(0, 0))
    minx, miny, maxx, maxy = b.bounds
    s = 62.0 / max(maxx - minx, maxy - miny)
    b = sa.scale(sa.translate(b, -(minx + maxx) / 2, -(miny + maxy) / 2), s, -s, origin=(0, 0))
    b = sa.translate(b, 32, 32)
    import cv2
    m = np.zeros((64, 64), np.uint8)
    polys = b.geoms if b.geom_type == "MultiPolygon" else [b]
    for p in polys:
        cv2.fillPoly(m, [np.round(np.array(p.exterior.coords)).astype(np.int32)], 255)
    return m, float(max(np.hypot(*e[0]), np.hypot(*e[1])))


def convert(args):
    row, block_wkb, bldg_wkbs = args
    try:
        block = largest(wkb.loads(block_wkb)).simplify(0.5)
        bl = [largest(wkb.loads(x)) for x in bldg_wkbs]
        bl = [b for b in bl if b.area > 1.0]
        if len(bl) > N:
            return row, None, "too_many"
        pos, size, order, asp, med, center = canonical(block, bl)
        slots = assign_slots(pos) if len(bl) else np.zeros(0, int)
        if slots is None:
            return row, None, "slots"
        bl_sorted = [bl[i] for i in order]
        shp = [shape_iou(b) for b in bl_sorted]
        mask, scale = block_mask(block)
        return row, {"pos": pos, "size": size, "slots": slots, "shape": [s for s, _ in shp],
                     "iou": [i for _, i in shp], "asp": asp, "midaxis": med.wkb, "center": center,
                     "mask": mask, "scale": scale, "n": len(bl)}, "ok"
    except Exception as e:  # noqa: BLE001
        return row, None, f"error: {type(e).__name__}"


def to_graph(rec, norm):
    g = nx.grid_2d_graph(H, W)
    g = nx.convert_node_labels_to_integers(g, first_label=0, ordering="default", label_attribute="old_label")
    g.graph["aspect_ratio"] = rec["asp"]
    g.graph["long_side"] = rec["scale"] / 400.0
    g.graph["binary_mask"] = rec["mask"]
    g.graph["block_scale"] = np.float32(rec["scale"])
    for k in range(N):
        g.nodes[k].update(posx=0.0, posy=0.0, exist=0.0, merge=0, size_x=0.0, size_y=0.0,
                          shape=0.0, iou=0.0, height=0.0)
    for j, s in enumerate(rec["slots"]):
        z = lambda v, m, sd: float(np.clip((v - m) / sd, -5, 5))
        g.nodes[int(s)].update(posx=z(rec["pos"][j, 0], *norm["posx"]), posy=z(rec["pos"][j, 1], *norm["posy"]),
                               size_x=z(rec["size"][j, 0], *norm["size_x"]),
                               size_y=z(rec["size"][j, 1], *norm["size_y"]),
                               exist=1.0, shape=float(rec["shape"][j]), iou=float(rec["iou"][j]),
                               height=float(j + 1))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--bldg", required=True)
    ap.add_argument("--gm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="blocks per split, for a quick test")
    a = ap.parse_args()

    blocks = pd.read_parquet(os.path.join(a.city, "blocks.parquet"), columns=["block_id", "geometry", "split", "valid"])
    blocks["row"] = np.arange(len(blocks))
    bld = pd.concat([pd.read_parquet(os.path.join(a.bldg, f), columns=["block_id", "geometry"])
                     for f in sorted(os.listdir(a.bldg)) if f.endswith(".parquet")])
    by_block = bld.groupby("block_id")["geometry"].apply(list).to_dict()
    os.makedirs(a.out, exist_ok=True)

    recs, status = {}, {}
    for split in ["train", "val", "test"]:
        s = blocks[blocks["split"] == split]
        if split != "test":
            s = s[s["valid"]]
        if a.limit:
            s = s.head(a.limit)
        jobs = [(int(r.row), r.geometry, by_block.get(r.block_id, [])) for r in s.itertuples()]
        with Pool(a.workers, initializer=init, initargs=(a.gm,)) as pool:
            out = pool.map(convert, jobs, chunksize=16)
        recs[split] = {r: rec for r, rec, st in out if rec is not None}
        status[split] = pd.Series([st for _, _, st in out]).value_counts().to_dict()
        print(split, len(jobs), "blocks ->", status[split], flush=True)

    tr = [rec for rec in recs["train"].values() if rec["n"]]
    allp = np.concatenate([r["pos"] for r in tr]); alls = np.concatenate([r["size"] for r in tr])
    norm = {"posx": [allp[:, 0].mean(), allp[:, 0].std()], "posy": [allp[:, 1].mean(), allp[:, 1].std()],
            "size_x": [alls[:, 0].mean(), alls[:, 0].std()], "size_y": [alls[:, 1].mean(), alls[:, 1].std()]}
    norm = {k: [float(v[0]), float(v[1])] for k, v in norm.items()}
    json.dump({"norm": norm, "status": status, "W": W, "H": H}, open(os.path.join(a.out, "norm.json"), "w"), indent=1)

    for split, rs in recs.items():
        d = os.path.join(a.out, split, "processed")
        os.makedirs(d, exist_ok=True)
        idx, geom = [], {}
        for k, (row, rec) in enumerate(sorted(rs.items())):
            pickle.dump(to_graph(rec, norm), open(os.path.join(d, f"{k}.gpickle"), "wb"))
            idx.append((k, row, rec["n"]))
            geom[row] = {"midaxis": rec["midaxis"], "asp": rec["asp"], "center": rec["center"],
                         "scale": rec["scale"]}
        pd.DataFrame(idx, columns=["k", "block_row", "n_bldg"]).to_csv(os.path.join(a.out, split, "index.csv"), index=False)
        pickle.dump(geom, open(os.path.join(a.out, split, "geom.pkl"), "wb"))
        print(split, "written", len(idx), flush=True)


if __name__ == "__main__":
    main()

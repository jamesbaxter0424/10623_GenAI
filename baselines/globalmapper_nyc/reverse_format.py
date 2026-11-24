"""Work out how the official GlobalMapper sample was built from its raw geometry.

The release ships raw_geo/<i> = [block polygon, [building polygons]] and processed/<i>.gpickle,
but not the script between them. This runs the released canonical transform on raw_geo and
compares it with processed/ to find the slot assignment and the normalisation.

  python reverse_format.py --data ~/genai-final/data/gm10k/dataset --gm ~/genai-final/third_party/GlobalMapper
"""
import argparse
import pickle
import sys

import numpy as np


def patch(gm):
    """example_canonical_transform.py calls get_k_angle but never defines it: the direction
    angle of the segment p1 -> p2 (used to find where the skeleton bends by > 5 degrees)."""
    if not hasattr(gm, "get_k_angle"):
        gm.get_k_angle = lambda p1, p2: np.arctan2(p2.y - p1.y, p2.x - p1.x)
    gm.print = lambda *a, **k: None  # silence its per-block prints
    return gm


def canonical(gm, block, bldgs):
    import skgeom
    from shapely.geometry import Polygon
    c = block.centroid
    import shapely.affinity as sa
    blk = sa.translate(block, -c.x, -c.y)
    bl = [sa.translate(b, -c.x, -c.y) for b in bldgs]
    ext = list(blk.exterior.coords)[:-1]
    ext.reverse()
    sk = skgeom.Polygon(ext)
    skel = skgeom.skeleton.create_interior_straight_skeleton(sk)
    _, longest = gm.get_polyskeleton_longest_path(skel, sk)
    med = gm.modified_skel_to_medaxis(longest, blk)
    pos, size, order, asp = gm.warp_bldg_by_midaxis(bl, blk, med)
    return pos, size, order, asp, med, blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gm", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    sys.path.insert(0, a.gm)
    import example_canonical_transform as gm
    patch(gm)

    P, R, F = [], [], []
    rows = []
    for i in range(a.n):
        block, bldgs = pickle.load(open(f"{a.data}/raw_geo/{i}", "rb"))
        g = pickle.load(open(f"{a.data}/processed/{i}.gpickle", "rb"))
        ex = np.array([g.nodes[k]["exist"] for k in range(120)]) > 0
        try:
            pos, size, order, asp, med, blk = canonical(gm, block, bldgs)
        except Exception as e:  # noqa: BLE001
            print(i, "transform failed:", e)
            continue
        n = len(bldgs)
        slots = np.nonzero(ex)[0]
        rows.append((i, n, int(ex.sum()), round(asp, 4), round(float(g.graph["aspect_ratio"]), 4),
                     round(med.length, 2), float(g.graph["long_side"]), float(g.graph["block_scale"])))
        if n == ex.sum() and n > 0:
            px = np.array([g.nodes[k]["posx"] for k in slots]); py = np.array([g.nodes[k]["posy"] for k in slots])
            sx = np.array([g.nodes[k]["size_x"] for k in slots]); sy = np.array([g.nodes[k]["size_y"] for k in slots])
            # slots are visited row by row; try to match each slot to a building by order
            P.append((i, pos, size, slots, px, py, sx, sy))
    print("block, n_raw, n_exist, asp(ours), asp(file), midaxis_len, long_side, block_scale")
    for r in rows[:25]:
        print(r)
    same = sum(r[1] == r[2] for r in rows)
    print(f"blocks with n_raw == n_exist: {same} / {len(rows)}")
    d = np.array([[r[3], r[4]] for r in rows])
    print("aspect ratio ours vs file: max abs diff", np.abs(d[:, 0] - d[:, 1]).max().round(4))
    ml = np.array([[r[5], r[6], r[7]] for r in rows])
    print("corr(midaxis_len, long_side)", np.corrcoef(ml[:, 0], ml[:, 1])[0, 1].round(3),
          "corr(midaxis_len, block_scale)", np.corrcoef(ml[:, 0], ml[:, 2])[0, 1].round(3))
    print("ratio long_side/midaxis", np.median(ml[:, 1] / ml[:, 0]).round(5),
          "block_scale/midaxis", np.median(ml[:, 2] / ml[:, 0]).round(5))
    pickle.dump({"P": P, "rows": rows}, open("/tmp/gm_reverse.pkl", "wb"))


if __name__ == "__main__":
    main()

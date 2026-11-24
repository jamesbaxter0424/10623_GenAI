"""Map GlobalMapper graphs (standardised slots) back to building polygons in map coordinates,
and draw them on our 4 m class grid."""
import json
import os
import sys

import numpy as np
import shapely.affinity as sa
from shapely import wkb

W, H = 30, 4


def load_norm(out_dir):
    return json.load(open(os.path.join(out_dir, "norm.json")))["norm"]


def unstandardise(px, py, sx, sy, norm):
    f = lambda v, k: np.asarray(v, float) * norm[k][1] + norm[k][0]
    return f(px, "posx"), f(py, "posy"), f(sx, "size_x"), f(sy, "size_y")


def graph_to_polygons(exist, px, py, sx, sy, iou, geom, norm, gm_path):
    """exist, px, ... : arrays over the 120 slots (standardised, as the model outputs them).
    geom: {"midaxis": wkb, "asp", "center"}. Returns building polygons in map coordinates.
    Uses the release's inverse_warp_bldg_by_midaxis (rectangles), then shrinks each rectangle
    about its centre by sqrt(iou) so its area matches the predicted occupancy."""
    sys.path.insert(0, gm_path)
    from geo_utils import inverse_warp_bldg_by_midaxis
    keep = np.asarray(exist) > 0
    if not keep.any():
        return []
    x, y, a, b = unstandardise(np.asarray(px)[keep], np.asarray(py)[keep], np.asarray(sx)[keep],
                               np.asarray(sy)[keep], norm)
    pos = np.stack([np.clip(x, -1, 1), y], 1)
    size = np.stack([np.clip(a, 1e-3, None), np.clip(b, 1e-3, None)], 1)
    med = wkb.loads(geom["midaxis"])
    polys, _, _ = inverse_warp_bldg_by_midaxis(pos.copy(), size.copy(), med, geom["asp"])
    cx, cy = geom["center"]
    out = []
    for p, q in zip(polys, np.asarray(iou)[keep]):
        s = float(np.sqrt(np.clip(q, 0.05, 1.0)))
        p = sa.scale(p, s, s, origin="centroid")
        out.append(sa.translate(p, cx, cy))
    return out


def rasterise(polys, r0, c0, x0, y_top, cell=4.0, n=128):
    """Cells of the n x n window (top-left grid index r0, c0) whose centre lies in any polygon."""
    from skimage.draw import polygon as fill
    m = np.zeros((n, n), bool)
    for p in polys:
        if p.is_empty:
            continue
        geoms = p.geoms if p.geom_type == "MultiPolygon" else [p]
        for g in geoms:
            xy = np.array(g.exterior.coords)
            col = (xy[:, 0] - x0) / cell - c0 - 0.5      # cell (i, j) has its centre at integer (i, j)
            row = (y_top - xy[:, 1]) / cell - r0 - 0.5
            rr, cc = fill(row, col, shape=(n, n))
            m[rr, cc] = True
    return m

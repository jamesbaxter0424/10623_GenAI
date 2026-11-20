"""Footprint measures of building regions inside one block (momepy-style measures on a grid).

A building region is a 4-connected group of building cells; at 4 m adjacent row houses merge,
so these are regions, not buildings (dataset audit D4). Measures:
  area       region area (m2)
  convexity  region area / convex hull area (1 = convex)
  nn_dist    distance between region centroids and the nearest other region in the block (m)
  density    regions per hectare of block (one value per block)
Generated and real distributions are compared with the Wasserstein distance, divided by the
standard deviation of the real distribution, and averaged over the four measures.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial import cKDTree

import config as C

MEASURES = ("area", "convexity", "nn_dist", "density")


def _hull_area(rr, cc):
    pts = np.unique(np.concatenate([np.stack([rr + dr, cc + dc], 1) for dr in (0, 1) for dc in (0, 1)]), axis=0)
    try:
        return ConvexHull(pts).volume
    except QhullError:
        return float(len(rr))


def block_regions(cls_window, inside):
    """Region measures of the building cells of one block in one map."""
    bld = inside & (cls_window >= 2)
    lab, n = ndimage.label(bld)
    out = {m: [] for m in MEASURES}
    out["density"] = [n / (inside.sum() * C.CELL_M2 / 1e4)]
    if n == 0:
        return out
    cents = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        rr, cc = np.nonzero(lab[sl] == k)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        out["area"].append(len(rr) * C.CELL_M2)
        out["convexity"].append(len(rr) / _hull_area(rr, cc))
        cents.append((rr.mean(), cc.mean()))
    if n > 1:
        d, _ = cKDTree(np.array(cents)).query(cents, k=2)
        out["nn_dist"] = list(d[:, 1] * C.CELL_M)
    return out
